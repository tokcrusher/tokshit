#include "tokshit/expert_io.hpp"
#include "tokshit/ahb.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

#include "tokshit/gguf.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"

namespace tokshit {
namespace {

// глобальный ключ: слой в старших битах, внутри слоя уже (proj << 32) | expert
uint64_t global_key(std::size_t layer, uint64_t key) {
  return (static_cast<uint64_t>(layer) << 40) | (key & 0xFF'FFFF'FFFFULL);
}

bool add_overflow(uint64_t a, uint64_t b, uint64_t* out) noexcept {
  if (b > std::numeric_limits<uint64_t>::max() - a) return false;
  *out = a + b;
  return true;
}

}  // namespace

// True iff [offset, offset + size) lies inside the file. Written without
// `offset + size` so a call with offset near UINT64_MAX cannot wrap the range
// check and reach pread() with a bogus window (the O_DIRECT path then computed
// `end - start` from the wrapped value as well).
bool ExpertIo::in_file(uint64_t offset, std::size_t size) const noexcept {
  if (size == 0 || offset > file_len_) return false;
  return size <= file_len_ - offset;
}

bool ExpertIo::is_cached(std::size_t layer, uint64_t key) const {
  return cache_.get(layer, key) != nullptr;
}

ExpertIo::ExpertIo(int fd, uint64_t file_len, ResidentCache& cache,
                   uint64_t budget, std::size_t threads, bool fadvise,
                   std::string path)
    : fd_(fd),
      file_len_(file_len),
      cache_(cache),
      budget_(budget),
      fadvise_(fadvise) {
  // просим o_direct: экспертные чтения не должны засорять page cache и
  // вытеснять оттуда плотные веса. на андроиде это ещё и меньше давление
  // на reclaim: грязных анонимных страниц не появляется
  if (!path.empty() && std::getenv("TOKSHIT_O_DIRECT") != nullptr &&
      std::string(std::getenv("TOKSHIT_O_DIRECT")) == "0") {
    path.clear();
  }
  if (!path.empty()) {
    const int dfd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (dfd < 0) {
      std::fprintf(stderr,
                   "[tokshit] expert io: o_direct open failed (errno=%d), using "
                   "buffered reads\n",
                   errno);
    }
    if (dfd >= 0) {
      struct stat st {};
      align_ = (::fstat(dfd, &st) == 0 && st.st_blksize > 0)
                   ? static_cast<std::size_t>(st.st_blksize)
                   : 4096;
      // align_ is used both as a mask (align_ - 1) and as a posix_memalign
      // argument, which requires a power of two >= sizeof(void*). A filesystem
      // reporting anything else would make every O_DIRECT read fail (or
      // misalign the read window).
      if (align_ < sizeof(void*) || (align_ & (align_ - 1)) != 0) align_ = 4096;
      // o_direct бывает врёт на эмулированных и fuse-разделах. сверяем один
      // выровненный блок с буферизованным чтением и молча откатываемся
      bool good = true;
      if (file_len_ >= align_) {
        void* d = nullptr;
        if (::posix_memalign(&d, align_, align_) != 0) d = nullptr;
        std::vector<uint8_t> b(align_);
        const uint64_t off = (file_len_ / 2) & ~(static_cast<uint64_t>(align_) - 1);
        const auto rd = pread_at(dfd, off, std::span<uint8_t>(
                                             static_cast<uint8_t*>(d), align_));
        const auto rb = pread_at(fd_, off, std::span<uint8_t>(b.data(), b.size()));
        good = d != nullptr && rd.ok() && rb.ok() &&
               std::memcmp(d, b.data(), align_) == 0;
        std::free(d);
      }
      if (good) {
        dfd_ = dfd;
      } else {
        std::fprintf(stderr,
                     "[tokshit] expert io: o_direct unavailable (verify failed, "
                     "align=%zu), using buffered reads\n",
                     align_);
        ::close(dfd);
      }
    }
  }
  const std::size_t n = threads == 0 ? 1 : threads;
  bounces_.resize(n);
  threads_.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    threads_.emplace_back([this, i] { worker(i); });
  }
  if (dfd_ >= 0) {
    std::fprintf(stderr, "[tokshit] expert io: o_direct, align=%zu, %zu lanes\n",
                 align_, n);
  } else {
    std::fprintf(stderr, "[tokshit] expert io: buffered pread, %zu lanes\n", n);
  }
}

ExpertIo::~ExpertIo() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
    // будим всех, воркеры увидят stop и выйдут
    for (auto& slot : inflight_) slot.second->state.store(1);
    demand_q_.clear();
    spec_q_.clear();
  }
  cv_work_.notify_all();
  cv_done_.notify_all();
  for (auto& t : threads_) {
    if (t.joinable()) t.join();
  }
  for (auto& b : bounces_) std::free(b.p);
  std::free(inline_bounce_.p);
  if (dfd_ >= 0) ::close(dfd_);
}

uint64_t ExpertIo::queued() const {
  std::lock_guard<std::mutex> lock(mu_);
  return demand_q_.size() + spec_q_.size();
}

// само чтение. pread прямо в буфер, потом put в кэш (он же решает выкинуть
// кого-то). без кэша не бывает: budget 0 обрабатывается выше в Model
ExpertIo::Bounce& ExpertIo::bounce(std::size_t lane, std::size_t need) {
  Bounce& b = (lane == kInlineLane) ? inline_bounce_ : bounces_[lane];
  if (b.cap >= need) return b;
  std::free(b.p);
  b.p = nullptr;
  b.cap = 0;
  void* p = nullptr;
  if (::posix_memalign(&p, align_, need) != 0) return b;
  b.p = p;
  b.cap = need;
  return b;
}

// чтение эксперта: через o_direct мимо page cache. если окно не выровнено,
// читаем выровненный диапазон в bounce и копируем нужный кусок
CacheBlock ExpertIo::fetch(const Job& job, std::size_t lane) {
  auto buf = std::make_shared<CacheBlockData>(job.size);
  if (!buf->valid()) {
    metrics().c_read_fail.fetch_add(1, std::memory_order_relaxed);
    if (metrics().c_read_fail.load() < 4) {
      std::fprintf(stderr, "[io] out of memory for %zu byte expert slice\n",
                   job.size);
    }
    return {};
  }
  bool ok = false;
  std::string err;
  if (dfd_ >= 0) {
    const uint64_t start =
        job.offset & ~static_cast<uint64_t>(align_ - 1);
    const uint64_t delta = job.offset - start;
    uint64_t raw_end = 0;
    const bool window_ok =
        add_overflow(job.offset, job.size, &raw_end) &&
        add_overflow(raw_end, align_ - 1, &raw_end);
    const uint64_t end =
        window_ok ? (raw_end & ~static_cast<uint64_t>(align_ - 1)) : 0;
    if (!window_ok) {
      err = "read window overflows";
    } else if (end <= file_len_) {
      const std::size_t len = static_cast<std::size_t>(end - start);
      Bounce& b = bounce(lane, len);
      if (b.p != nullptr) {
        const auto rd = pread_at(dfd_, start,
                                 std::span<uint8_t>(
                                     static_cast<uint8_t*>(b.p), len));
        ok = rd.ok();
        if (ok) {
          std::memcpy(buf->data(), static_cast<uint8_t*>(b.p) + delta,
                      job.size);
        } else {
          err = rd.error().message;
        }
      } else {
        err = "bounce alloc failed";
      }
    } else {
      err = "read window runs past eof";
    }
  }
  if (!ok && err.empty()) {
    const auto rd = pread_at(
        fd_, job.offset, std::span<uint8_t>(buf->data(), buf->size()));
    ok = rd.ok();
    if (!ok) err = rd.error().message;
  }
  if (!ok) {
    metrics().c_read_fail.fetch_add(1, std::memory_order_relaxed);
    if (metrics().c_read_fail.load() < 4) {
      std::fprintf(stderr, "[io] fetch fail: off=%llu size=%zu : %s\n",
                   static_cast<unsigned long long>(job.offset), job.size,
                   err.c_str());
    }
    return {};
  }
  reads_.fetch_add(1, std::memory_order_relaxed);
  bytes_.fetch_add(job.size, std::memory_order_relaxed);
  if (fadvise_) fadvise_dontneed(fd_, job.offset, job.size);
  if (std::getenv("TOKSHIT_AHB_DEBUG") != nullptr) {
    static std::atomic<uint64_t> dbg{0};
    const uint64_t k = dbg.fetch_add(1, std::memory_order_relaxed);
    if (k < 5) {
      const bool on = ahb::Pool::instance().is_ahb(const_cast<uint8_t*>(buf->data()));
      std::fprintf(stderr, "[ahb-dbg] fetch #%llu size=%zu on_ahb=%d\n",
                   static_cast<unsigned long long>(k), job.size, on ? 1 : 0);
    }
  }
  CacheBlock block = std::move(buf);
  return cache_.put(job.layer, job.key, std::move(block), budget_.load(std::memory_order_relaxed));
}

// merge-чтение: весь диапазон задачи одним pread в bounce (o_direct требует
// выровненного окна), потом memcpy-сплит по слотам. мимо page cache, как и
// одиночный fetch; o_direct не поедет — падаем на буферизованный pread по
// срезам. возвращает ровно group.size() блоков: при ошибке элементы nullptr,
// воркер всё равно закрывает слоты, и ожидающие уходят в fallback
std::vector<CacheBlock> ExpertIo::fetch_group(const Job& job,
                                              std::size_t lane) {
  const std::size_t n = job.group.size();
  if (n == 0) return {};
  // у merge-задачи все срезы одинаковые, job.size = n * slice — это контракт
  // demand_merged. аномалию трактуем как ошибку чтения: воркер закроет слоты
  // пустыми блоками
  if (job.size % n != 0) {
    metrics().c_read_fail.fetch_add(1, std::memory_order_relaxed);
    return std::vector<CacheBlock>(n);
  }
  const std::size_t slice = job.size / n;
  // свежие блоки ещё не опубликованы в кэше: писать в них можно, CacheBlock
  // (shared_ptr<const ...>) отдаёт их только после put
  std::vector<std::shared_ptr<CacheBlockData>> bufs(n);
  for (std::size_t i = 0; i < n; ++i) {
    bufs[i] = std::make_shared<CacheBlockData>(slice);
    if (!bufs[i]->valid()) {
      metrics().c_read_fail.fetch_add(1, std::memory_order_relaxed);
      if (metrics().c_read_fail.load() < 4) {
        std::fprintf(stderr, "[io] out of memory for %zu byte expert slice\n",
                     slice);
      }
      return std::vector<CacheBlock>(n);
    }
  }

  bool ok = false;
  std::string err;
  if (dfd_ >= 0) {
    const uint64_t start = job.offset & ~static_cast<uint64_t>(align_ - 1);
    uint64_t raw_end = 0;
    const bool window_ok =
        add_overflow(job.offset, job.size, &raw_end) &&
        add_overflow(raw_end, align_ - 1, &raw_end);
    const uint64_t end =
        window_ok ? (raw_end & ~static_cast<uint64_t>(align_ - 1)) : 0;
    if (!window_ok) {
      err = "read window overflows";
    } else if (end > file_len_) {
      err = "read window runs past eof";
    } else {
      const std::size_t len = static_cast<std::size_t>(end - start);
      Bounce& b = bounce(lane, len);
      if (b.p != nullptr) {
        const auto rd = pread_at(dfd_, start,
                                 std::span<uint8_t>(
                                     static_cast<uint8_t*>(b.p), len));
        if (rd.ok()) {
          ok = true;
          for (std::size_t i = 0; i < n; ++i) {
            const std::size_t delta =
                static_cast<std::size_t>(job.offset + i * slice - start);
            std::memcpy(bufs[i]->data(),
                        static_cast<const uint8_t*>(b.p) + delta, slice);
          }
        } else {
          err = rd.error().message;
        }
      } else {
        err = "bounce alloc failed";
      }
    }
  }
  if (!ok) {
    // буферизованный fallback: читаем каждый срез сразу в его блок
    ok = true;
    for (std::size_t i = 0; i < n; ++i) {
      const auto rd = pread_at(fd_, job.offset + i * slice,
                               std::span<uint8_t>(bufs[i]->data(), slice));
      if (!rd.ok()) {
        err = rd.error().message;
        ok = false;
        break;
      }
    }
  }
  if (!ok) {
    metrics().c_read_fail.fetch_add(1, std::memory_order_relaxed);
    if (metrics().c_read_fail.load() < 4) {
      std::fprintf(stderr,
                   "[io] fetch_group fail: off=%llu size=%zu n=%zu : %s\n",
                   static_cast<unsigned long long>(job.offset), job.size, n,
                   err.c_str());
    }
    return std::vector<CacheBlock>(n);
  }
  reads_.fetch_add(1, std::memory_order_relaxed);  // один pread на всю группу
  bytes_.fetch_add(job.size, std::memory_order_relaxed);
  if (fadvise_) fadvise_dontneed(fd_, job.offset, job.size);
  std::vector<CacheBlock> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const uint64_t key = job.group[i].first & 0xFF'FFFF'FFFFULL;
    out[i] = cache_.put(job.layer, key, std::move(bufs[i]), budget_);
  }
  return out;
}

void ExpertIo::demand_merged(
    std::size_t layer,
    std::span<const std::pair<uint64_t, uint64_t>> entries,
    std::size_t size) {
  if (entries.empty() || size == 0 || budget_ == 0) return;
  // сначала кэш (как в demand_async: попадание это попадание), потом под
  // мьютексом inflight и сборка цепочек
  std::vector<std::pair<uint64_t, uint64_t>> live;  // (gkey, offset)
  live.reserve(entries.size());
  for (const auto& en : entries) {
    const uint64_t off = en.second;
    if (!in_file(off, size)) {
      metrics().c_dem_range.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (cache_.get(layer, en.first) != nullptr) {
      metrics().c_dem_cached.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    live.emplace_back(global_key(layer, en.first), off);
  }
  if (live.empty()) return;
  std::sort(live.begin(), live.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // ограничение на цепочку: bounce воркера растёт до размера самой длинной,
  // на телефоне лишние десятки мегабайт никто не дарил
  constexpr std::size_t kMaxMergeSlices = 8;

  std::lock_guard<std::mutex> lock(mu_);
  if (stop_) return;
  std::vector<std::pair<uint64_t, uint64_t>> pending;  // (gkey, offset)
  pending.reserve(live.size());
  for (const auto& en : live) {
    if (inflight_.find(en.first) != inflight_.end()) {
      metrics().c_dem_dup.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    pending.push_back(en);
  }
  std::size_t i = 0;
  while (i < pending.size()) {
    std::size_t j = i;
    while (j + 1 < pending.size() &&
           pending[j + 1].second == pending[j].second + size &&
           (j + 1 - i) < kMaxMergeSlices) {
      ++j;
    }
    const std::size_t run = j - i + 1;
    Job job;
    job.layer = layer;
    job.offset = pending[i].second;
    job.size = run * size;
    job.demand = true;
    job.epoch = epoch_;
    job.group.reserve(run);
    for (std::size_t k = i; k <= j; ++k) {
      auto slot = std::make_shared<Slot>();
      inflight_.emplace(pending[k].first, slot);
      job.group.emplace_back(pending[k].first, std::move(slot));
    }
    demand_q_.push_back(std::move(job));
    metrics().c_dem_queued.fetch_add(run, std::memory_order_relaxed);
    metrics().c_dem_merged_jobs.fetch_add(1, std::memory_order_relaxed);
    metrics().c_dem_merged_slices.fetch_add(run, std::memory_order_relaxed);
    i = j + 1;
  }
  cv_work_.notify_all();
}

void ExpertIo::worker(std::size_t lane) {
  for (;;) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_work_.wait(lock, [this] {
        return stop_ || !demand_q_.empty() || !spec_q_.empty();
      });
      if (demand_q_.empty() && spec_q_.empty()) {
        if (stop_) return;
        continue;
      }
      std::deque<Job>& q = demand_q_.empty() ? spec_q_ : demand_q_;
      job = std::move(q.front());
      q.pop_front();
    }
    busy_.fetch_add(1, std::memory_order_relaxed);

    if (job.group.empty()) {
      CacheBlock block = fetch(job, lane);
      std::lock_guard<std::mutex> lock(mu_);
      job.slot->data = std::move(block);
      job.slot->state.store(1, std::memory_order_release);
      const auto it = inflight_.find(job.gkey);
      if (it != inflight_.end() && it->second == job.slot) {
        inflight_.erase(it);
      }
    } else {
      auto blocks = fetch_group(job, lane);
      std::lock_guard<std::mutex> lock(mu_);
      for (std::size_t i = 0; i < job.group.size() && i < blocks.size(); ++i) {
        auto& [gkey, slot] = job.group[i];
        slot->data = std::move(blocks[i]);
        slot->state.store(1, std::memory_order_release);
        const auto it = inflight_.find(gkey);
        if (it != inflight_.end() && it->second == slot) inflight_.erase(it);
      }
    }
    busy_.fetch_sub(1, std::memory_order_relaxed);
    cv_done_.notify_all();
  }
}

// спрос: сначала кэш (это же считает попадания). если этот блок уже читает
// префетч — просто ждём его, второй раз с диска не тянем. иначе читаем сами
// прямо в вызывающем потоке: у счётных воркеров параллелизма больше, чем у
// пары фоновых читателей, и диск это любит
ExpertIo::Loaded ExpertIo::load(std::size_t layer, uint64_t key,
                                uint64_t offset, std::size_t size) {
  if (const auto hit = cache_.get(layer, key); hit != nullptr) {
    return Loaded{std::move(hit), true};
  }
  if (budget_.load(std::memory_order_relaxed) == 0 || !in_file(offset, size)) {
    return Loaded{};
  }

  const uint64_t gkey = global_key(layer, key);
  bool ours = false;
  uint64_t epoch = 0;
  std::shared_ptr<Slot> slot;
  {
    std::lock_guard<std::mutex> lock(mu_);
    epoch = epoch_;
    const auto it = inflight_.find(gkey);
    if (it != inflight_.end()) {
      slot = it->second;
    } else {
      slot = std::make_shared<Slot>();
      inflight_.emplace(gkey, slot);
      ours = true;
    }
  }

  if (!ours && slot != nullptr) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [&slot] {
      return slot->state.load(std::memory_order_acquire) != 0;
    });
    // копия, не move: слот могут ждать несколько потоков, и данные остаются
    // на месте для следующего ожидающего. чтение под mu_ — тот же мьютекс,
    // под которым данные положил читатель
    return Loaded{slot->data, false};
  }

  CacheBlock block;
  {
    std::lock_guard<std::mutex> lock(inline_mu_);
    block = fetch(Job{layer, key, gkey, offset, size, true, epoch, slot, {}},
                  kInlineLane);
  }
  CacheBlock out;
  {
    std::lock_guard<std::mutex> lock(mu_);
    slot->data = std::move(block);
    slot->state.store(1, std::memory_order_release);
    const auto it = inflight_.find(gkey);
    if (it != inflight_.end() && it->second == slot) inflight_.erase(it);
    // забираем копию под мьютексом: снаружи слот->data двигать нельзя, ждущие
    // потоки просыпаются параллельно и читают его же
    // take a copy under the mutex: slot->data must not be moved outside it,
    out = slot->data;
  }
  cv_done_.notify_all();
  return Loaded{std::move(out), false};
}

// заказ чтения без ожидания. если блок уже в кэше или уже читается — ничего
bool ExpertIo::demand_async(std::size_t layer, uint64_t key, uint64_t offset,
                            std::size_t size) {
  if (budget_.load(std::memory_order_relaxed) == 0 || !in_file(offset, size)) {
    metrics().c_dem_range.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  if (cache_.get(layer, key) != nullptr) {
    metrics().c_dem_cached.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  const uint64_t gkey = global_key(layer, key);
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_ || inflight_.find(gkey) != inflight_.end()) {
      metrics().c_dem_dup.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    auto slot = std::make_shared<Slot>();
    inflight_.emplace(gkey, slot);
    demand_q_.push_back(
        Job{layer, key, gkey, offset, size, true, epoch_, std::move(slot),
            {}});
  }
  metrics().c_dem_queued.fetch_add(1, std::memory_order_relaxed);
  cv_work_.notify_all();
  return false;
}

// ждём пока ключ перестанет быть в полёте, потом берём из кэша
CacheBlock ExpertIo::wait_for(std::size_t layer, uint64_t key) {
  const uint64_t gkey = global_key(layer, key);
  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [this, gkey] {
      return stop_ || inflight_.find(gkey) == inflight_.end();
    });
  }
  auto block = cache_.get(layer, key);
  if (block == nullptr) {
    metrics().c_wait_null.fetch_add(1, std::memory_order_relaxed);
  } else {
    metrics().c_wait_ok.fetch_add(1, std::memory_order_relaxed);
  }
  return block;
}

// предсказание: если блок уже в кэше или уже читается, ничего не делаем.
// TOKSHIT_PREFETCH_IDLE: спек-заказы принимаются только когда дисковый путь
// свободен — иначе спекуляция встаёт в очередь перед спросом и душит его
// TOKSHIT_PREFETCH_IDLE: spec orders are accepted only while the drive path
void ExpertIo::prefetch(std::size_t layer, uint64_t key, uint64_t offset,
                        std::size_t size) {
  if (budget_.load(std::memory_order_relaxed) == 0 || !in_file(offset, size))
    return;
  static const bool idle_only = [] {
    const char* v = std::getenv("TOKSHIT_PREFETCH_IDLE");
    return v != nullptr && *v != '\0' && v[0] != '0';
  }();
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_) return;
    if (idle_only &&
        (busy_.load(std::memory_order_relaxed) != 0 || !demand_q_.empty())) {
      metrics().c_spec_idle_drop.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  if (cache_.get(layer, key) != nullptr) return;

  const uint64_t gkey = global_key(layer, key);
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_ || inflight_.find(gkey) != inflight_.end()) return;
    if (spec_q_.size() >= kMaxQueued) return;
    auto slot = std::make_shared<Slot>();
    inflight_.emplace(gkey, slot);
    spec_q_.push_back(
        Job{layer, key, gkey, offset, size, false, epoch_, std::move(slot),
            {}});
  }
  cv_work_.notify_one();
}

// новый токен. старые предсказания выбрасываем, но начатые чтения не трогаем
void ExpertIo::new_epoch() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++epoch_;
    std::deque<Job> keep;
    for (auto& job : spec_q_) {
      if (job.epoch >= epoch_) {
        keep.push_back(std::move(job));
      } else {
        const auto it = inflight_.find(job.gkey);
        if (it != inflight_.end() && it->second == job.slot) {
          inflight_.erase(it);
        }
        job.slot->state.store(1, std::memory_order_release);
      }
    }
    spec_q_ = std::move(keep);
  }
  cv_done_.notify_all();
}

}  // namespace tokshit
