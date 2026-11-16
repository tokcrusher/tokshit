#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache_block.hpp"  // CacheBlock (AHB-backed)

namespace tokshit {

class ResidentCache;

// фоновые читатели экспертов. главная идея: диск это самый медленный ярус,
// поэтому он не должен блокировать счётные потоки. спрос (demand) и
// предсказание (prefetch) идут в одну очередь, но предсказание уходит в конец,
// а спрос лезет вперёд. если блок уже читается по предсказанию, demand просто
// ждёт его, а не читает второй раз
//
// background expert readers. the point: the disk is the slowest tier, it must
// not block the compute threads. demand and prediction share one queue, but a
// prediction goes to the back while demand jumps to the front. if a block is
// already being read by a prediction, demand just waits for it instead of
// reading it twice
class ExpertIo {
 public:
  struct Loaded {
    CacheBlock block;
    bool hit = false;
  };

  // path нужен для o_direct: mmap держится на буферизованном fd, а эксперты
  // читаем мимо page cache. пустой path = обычный pread
  // path is needed for o_direct: the mmap stays on the buffered fd while
  // experts bypass the page cache. empty path = plain pread
  ExpertIo(int fd, uint64_t file_len, ResidentCache& cache, uint64_t budget,
           std::size_t threads, bool fadvise, std::string path = {});
  ~ExpertIo();

  ExpertIo(const ExpertIo&) = delete;
  ExpertIo& operator=(const ExpertIo&) = delete;

  // блок из кэша, иначе синхронное чтение. вернёт пустой block на ошибке,
  // вызывающий тогда падает обратно на mmap
  // cached block, otherwise a synchronous read. empty block on error, the
  // caller then falls back to the mmap
  [[nodiscard]] Loaded load(std::size_t layer, uint64_t key, uint64_t offset,
                            std::size_t size);

  // заказать чтение заранее, не дожидаясь его. вернёт управление сразу.
  // используется чтобы параллелить диск с счётом: заказали все эксперты
  // слоя, посчитали gate/up, за это время down уже приехал
  // order a read ahead of time, without waiting for it. returns immediately.
  // used to overlap the drive with compute: order all of the layer's experts,
  // compute gate/up, by then down has already arrived
  // true если блок уже был в кэше и читать не пришлось
  // true if the block was already cached and no read was ordered
  [[nodiscard]] bool demand_async(std::size_t layer, uint64_t key,
                                  uint64_t offset, std::size_t size);

  // то же самое, но группа соседних срезов (эксперты одного proj, подряд в
  // файле) уезжает одним pread: воркер читает весь диапазон в bounce и режет
  // его по слотам. entries — пары (key, offset) по возрастанию offset, у всех
  // срезов одинаковый size. разрывы в цепочке дают отдельные задачи, одиночные
  // срезы ведут себя как demand_async
  // same, but a group of adjacent slices (one proj, contiguous in the file)
  // goes out as one pread: a worker reads the whole range into a bounce and
  // splits it into slots. entries are (key, offset) pairs in ascending offset
  // order, all slices share `size`. gaps split the chain into separate jobs,
  // single slices behave like demand_async
  void demand_merged(std::size_t layer,
                     std::span<const std::pair<uint64_t, uint64_t>> entries,
                     std::size_t size);

  // проверка попадания без заказа чтения: для учёта статистики в merge-пути
  // hit probe without ordering a read: for merge-path accounting
  [[nodiscard]] bool is_cached(std::size_t layer, uint64_t key) const;

  // дождаться блока, заказанного demand_async или prefetch. nullptr если
  // чтение не удалось или блок уже вытеснили
  // wait for a block ordered by demand_async or prefetch. nullptr if the read
  // failed or the block was already evicted
  [[nodiscard]] CacheBlock wait_for(std::size_t layer, uint64_t key);

  // прогреть кэш в фоне. не блокирует
  // warm the cache in the background. does not block
  void prefetch(std::size_t layer, uint64_t key, uint64_t offset,
                std::size_t size);

  // новый токен: всё что осталось в очереди от старого предсказания уже не
  // нужно, выбрасываем
  // new token: whatever is left in the queue from the old prediction is
  // useless, drop it
  void new_epoch();

  [[nodiscard]] int fd() const noexcept { return fd_; }
  [[nodiscard]] uint64_t reads() const noexcept {
    return reads_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t bytes() const noexcept {
    return bytes_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t queued() const;

  // дисковый путь свободен: спрос пуст и ни один воркер не читает. для
  // TOKSHIT_PREFETCH_IDLE
  // the drive path is idle: demand empty and no worker reading. for
  // TOKSHIT_PREFETCH_IDLE
  [[nodiscard]] bool drive_idle() const {
    if (busy_.load(std::memory_order_relaxed) != 0) return false;
    std::lock_guard<std::mutex> lock(mu_);
    return demand_q_.empty();
  }

  // Временно поднять бюджет per-layer кэша на время батч-слоя: union
  // экспертов чанка не влезает в штатные 10 МБ на слой, и без этого блоки
  // вытесняются до take, вызывая повторные чтения с диска.
  // temporarily raise the per-layer cache budget for a batched layer: the
  // chunk's expert union does not fit the default per-layer budget and would
  // be evicted before take, forcing duplicate reads.
  void set_budget(uint64_t bytes) noexcept {
    budget_.store(bytes, std::memory_order_relaxed);
  }

 private:
  struct Slot {
    std::atomic<int> state{0};
    CacheBlock data;
  };

  struct Job {
    std::size_t layer = 0;
    uint64_t key = 0;
    uint64_t gkey = 0;
    uint64_t offset = 0;
    std::size_t size = 0;
    bool demand = false;
    uint64_t epoch = 0;
    std::shared_ptr<Slot> slot;
    // merge-задача: весь диапазон [offset, offset+size) одним pread, дальше
    // байты режутся по этим слотам. пусто у обычных задач
    // merged job: the whole [offset, offset+size) range in one pread, bytes
    // are then split across these slots. empty for plain jobs
    std::vector<std::pair<uint64_t, std::shared_ptr<Slot>>> group;
  };

  struct Bounce {
    void* p = nullptr;
    std::size_t cap = 0;
  };

  void worker(std::size_t lane);
  [[nodiscard]] Bounce& bounce(std::size_t lane, std::size_t need);
  [[nodiscard]] CacheBlock fetch(const Job& job, std::size_t lane);
  // merge-чтение: один pread на весь диапазон задачи, сплит по слотам.
  // возвращает по блоку на элемент group (nullptr на ошибке)
  // merged read: one pread over the job range, split across slots. returns
  // one block per group element (nullptr on error)
  [[nodiscard]] std::vector<CacheBlock> fetch_group(const Job& job,
                                                    std::size_t lane);
  // [offset, offset + size) внутри файла, без переполнения offset + size
  // in-file range check that cannot be fooled by offset + size overflow
  [[nodiscard]] bool in_file(uint64_t offset, std::size_t size) const noexcept;

  int fd_ = -1;
  int dfd_ = -1;  // o_direct fd, -1 = читаем буферизованно
  std::size_t align_ = 4096;
  uint64_t file_len_ = 0;
  ResidentCache& cache_;
  std::atomic<uint64_t> budget_{0};
  bool fadvise_ = false;
  std::vector<std::thread> threads_;
  std::vector<Bounce> bounces_;  // по одному на воркера
  std::mutex inline_mu_;         // защищает bounce для inline-чтений
  Bounce inline_bounce_;

  mutable std::mutex mu_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  // два фронта: спрос и предсказание. воркер всегда берёт спрос первым, как
  // только он появился — реальная работа не должна стоять за спекуляцией
  // two fronts: demand and speculation. a worker always takes demand first
  // as soon as it appears — real work must never wait behind speculation
  std::deque<Job> demand_q_;
  std::deque<Job> spec_q_;
  std::unordered_map<uint64_t, std::shared_ptr<Slot>> inflight_;
  bool stop_ = false;
  uint64_t epoch_ = 0;
  // сколько воркеров прямо сейчас держит pread. TOKSHIT_PREFETCH_IDLE
  // смотрит на это, чтобы не душить demand спекуляцией
  // how many workers hold a pread right now. TOKSHIT_PREFETCH_IDLE watches
  // this so speculation never chokes demand
  std::atomic<int> busy_{0};

  std::atomic<uint64_t> reads_{0};
  std::atomic<uint64_t> bytes_{0};

  static constexpr std::size_t kMaxQueued = 512;
  // lane для inline-чтений (не воркер): свой bounce
  // lane for inline reads (not a worker): its own bounce
  static constexpr std::size_t kInlineLane = static_cast<std::size_t>(-1);
};

}  // namespace tokshit
