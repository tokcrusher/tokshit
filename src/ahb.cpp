#include "tokshit/ahb.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__ANDROID__)
#  define TOKSHIT_HAVE_AHB 1
#  include <dlfcn.h>
#else
#  define TOKSHIT_HAVE_AHB 0
#endif

namespace tokshit::ahb {

// AHB грузим через dlopen, а не через -landroid. Termux не даёт системную
// libandroid.so: его stub собран без version-узлов, а бинарь, слинкованный
// с -landroid, требует LIBANDROID vN и падает ещё до main
// ("CANNOT LINK EXECUTABLE ... cannot find libandroid.so from verneed").
// dlopen обходит и линковку, и version-проверку, и работает одинаково
// и в Termux, и в APK.
// AHB is reached through dlopen instead of -landroid, see the long note in the
// git history: Termux's unversioned libandroid.so stub cannot satisfy the
// LIBANDROID verneed of a -landroid binary.
struct AHardwareBuffer;

struct AhbApi {
  int (*allocate)(const void* desc, AHardwareBuffer** out) = nullptr;
  void (*release)(AHardwareBuffer* buf) = nullptr;
  int (*lock)(AHardwareBuffer* buf, uint64_t usage, int32_t fence,
              const void* rect, void** out) = nullptr;
  int (*unlock)(AHardwareBuffer* buf, int32_t* fence) = nullptr;
  bool ok = false;
};

// AHardwareBuffer_Desc из <android/hardware_buffer.h>. объявляем сами чтобы
// не зависеть от уровня __ANDROID_API__ (в Termux это 24, а структура есть
// только с 26).
// AHardwareBuffer_Desc from <android/hardware_buffer.h>, declared by hand so we
// dont depend on __ANDROID_API__ (Termux pins it to 24 while the struct only
// exists from 26).
struct AhbDesc {
  uint32_t width;
  uint32_t height;
  uint32_t layers;
  uint32_t format;
  uint64_t usage;
  uint32_t stride;
  uint32_t rfu0;
  uint64_t rfu1;
};

constexpr uint32_t kFormatBlob = 0x21;  // AHARDWAREBUFFER_FORMAT_BLOB

// Значения строго из NDK-хедера (android/hardware_buffer.h):
//   CPU_READ_NEVER=0, CPU_READ_RARELY=2, CPU_READ_OFTEN=3
//   CPU_WRITE_NEVER=0, CPU_WRITE_RARELY=0x20, CPU_WRITE_OFTEN=0x30
// Старый код использовал 0x40 как "write never": это не флаг CPU, а мусор,
// из-за чего gralloc выдавал read-only mapping, а мы в него писали memcpy.
// Values are taken verbatim from the NDK header. The old code used 0x40 for
// CPU_WRITE_NEVER; that is not a CPU flag at all, so the mapping could come
// back read-only while the loader memcpy'd into it.
constexpr uint64_t kCpuReadRarely = 0x2;
constexpr uint64_t kCpuReadOften = 0x3;
constexpr uint64_t kCpuWriteRarely = 0x20;

const AhbApi& ahb_api() {
  static const AhbApi api = [] {
    AhbApi a;
#if TOKSHIT_HAVE_AHB
    for (const char* name : {"libandroid.so",
                             "/system/lib64/libandroid.so",
                             "/apex/com.android.runtime/lib64/libandroid.so"}) {
      void* h = ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (h == nullptr) continue;
      a.allocate = reinterpret_cast<decltype(a.allocate)>(
          ::dlsym(h, "AHardwareBuffer_allocate"));
      a.release = reinterpret_cast<decltype(a.release)>(
          ::dlsym(h, "AHardwareBuffer_release"));
      a.lock = reinterpret_cast<decltype(a.lock)>(
          ::dlsym(h, "AHardwareBuffer_lock"));
      a.unlock = reinterpret_cast<decltype(a.unlock)>(
          ::dlsym(h, "AHardwareBuffer_unlock"));
      if (a.allocate != nullptr && a.release != nullptr &&
          a.lock != nullptr && a.unlock != nullptr) {
        a.ok = true;
        break;
      }
      ::dlclose(h);
    }
#endif
    return a;
  }();
  return api;
}

// Пробуем несколько комбинаций usage: у разных вендоров gralloc по-разному
// относится к read-only/read-write и к частоте доступа. Нам нужен один
// долгоживущий mapping, в который можно писать при загрузке и читать на
// каждом matvec.
// Try a few usage combinations: gralloc implementations disagree about which
// CPU flags produce a writable mapping. We need one long-lived mapping that
// can be filled once and read on every matvec.
bool lock_ahb_buffer(AHardwareBuffer* buf, void** out, uint64_t* usage_used) {
  const AhbApi& api = ahb_api();
  struct Combo {
    uint64_t usage;
    const char* name;
  };
  // read-rarely по умолчанию: gralloc на этом устройстве отдаёт для
  // read-often некэшируемый mapping, и streaming 1.6 ГБ hot-весов падает с
  // 5.2 до 3.1 ГБ/с. read-rarely кэшируется, а lock мы держим один раз.
  // read-rarely is the default: on this device gralloc returns an uncached
  // mapping for read-often, and streaming 1.6 GB of hot weights drops from 5.2
  // to 3.1 GB/s. read-rarely is cacheable and the lock is taken once.
  static const Combo combos[] = {
      {kCpuReadRarely | kCpuWriteRarely, "read-rarely+write-rarely"},
      {kCpuReadOften | kCpuWriteRarely, "read-often+write-rarely"},
  };
  const char* want = std::getenv("TOKSHIT_AHB_USAGE");
  for (const Combo& c : combos) {
    if (want != nullptr && *want != '\0') {
      // "often" / "rare" сокращения для быстрых экспериментов
      if (std::strcmp(want, "often") == 0 &&
          std::strstr(c.name, "read-often") == nullptr)
        continue;
      if (std::strcmp(want, "rare") == 0 &&
          std::strstr(c.name, "read-rarely") == nullptr)
        continue;
    }
    void* cpu = nullptr;
    if (api.lock(buf, c.usage, -1, nullptr, &cpu) == 0 && cpu != nullptr) {
      *out = cpu;
      if (usage_used != nullptr) *usage_used = c.usage;
      return true;
    }
  }
  return false;
}

namespace {

bool ahb_disabled() {
  const char* v = std::getenv("TOKSHIT_AHB");
  return v != nullptr && *v != '\0' && v[0] == '0';
}

// Общий бюджет AHB на процесс: hot-веса и expert-кэш делят его, иначе
// «5 ГиБ на hot + 5 ГиБ на экспертов» снова заканчиваются OOM.
// One process-wide AHB budget shared by hot weights and the expert cache.
std::size_t ahb_limit_bytes() {
  static const std::size_t limit = [] {
    std::size_t mb = 5120;  // 5 GiB по умолчанию
    const char* v = std::getenv("TOKSHIT_AHB_MAX_MB");
    if (v != nullptr && *v != '\0') {
      const unsigned long long parsed = std::strtoull(v, nullptr, 10);
      // clamp before multiplying: TOKSHIT_AHB_MAX_MB=18446744073709551615
      // used to wrap the limit around to a tiny value (or zero).
      constexpr unsigned long long kMaxMb =
          std::numeric_limits<std::size_t>::max() / (1024ULL * 1024ULL);
      if (parsed != 0) {
        mb = static_cast<std::size_t>(std::min(parsed, kMaxMb));
      }
    }
    return mb * 1024 * 1024;
  }();
  return limit;
}

std::atomic<std::size_t> g_ahb_used{0};

std::size_t mem_available_bytes() {
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (f == nullptr) return 0;
  char line[256];
  std::size_t kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
  }
  std::fclose(f);
  return kb * 1024;
}

bool reserve_ahb(std::size_t bytes) {
  // Не занимаем последний гигабайт даже в рамках формального лимита:
  // MemAvailable важнее, чем «5 ГиБ разрешено».
  // Never eat the last gigabyte even inside the formal limit.
  constexpr std::size_t kMemReserve = 1ULL * 1024 * 1024 * 1024;
  const std::size_t limit = ahb_limit_bytes();
  // All arithmetic below is guarded: `bytes` comes from configuration, and
  // `bytes + kMemReserve` / `cur + bytes` used to wrap around for huge values,
  // letting an absurd reservation through to AHardwareBuffer_allocate.
  if (bytes == 0 || bytes > limit) return false;
  const std::size_t avail = mem_available_bytes();
  if (avail != 0 && (bytes > avail || avail - bytes < kMemReserve)) return false;
  std::size_t cur = g_ahb_used.load(std::memory_order_relaxed);
  while (cur <= limit - bytes) {
    if (g_ahb_used.compare_exchange_weak(cur, cur + bytes,
                                         std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

void release_ahb(std::size_t bytes) {
  // Saturating: a double release must not wrap the process-wide counter to
  // ~SIZE_MAX and permanently disable AHB for the rest of the run.
  std::size_t cur = g_ahb_used.load(std::memory_order_relaxed);
  for (;;) {
    const std::size_t next = bytes >= cur ? 0 : cur - bytes;
    if (g_ahb_used.compare_exchange_weak(cur, next,
                                         std::memory_order_relaxed)) {
      return;
    }
  }
}

// Размер сегмента hot-весов. 0 — старый режим: отдельный буфер на тензор.
// Anything above 16 GiB is clamped: the phone has 15.2 GiB of RAM, and a huge
// value must not overflow `mb * 1 MiB`.
std::size_t hot_seg_target_bytes() {
  std::size_t mb = 32;
  if (const char* v = std::getenv("TOKSHIT_HOT_SEG_MB");
      v != nullptr && *v != '\0') {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end != v) {
      mb = static_cast<std::size_t>(
          std::min<unsigned long long>(parsed, 16ULL * 1024ULL));
    }
  }
  return mb * 1024 * 1024;
}

}  // namespace

struct Slab {
  AHardwareBuffer* buf = nullptr;
  uint8_t* base = nullptr;
  std::size_t bytes = 0;
  std::size_t n_slots = 0;
  bool is_ahb = false;
};

struct SizeClass {
  std::size_t slot_bytes = 0;
  std::vector<Slab> slabs;
  std::vector<uint8_t*> free_slots;
  std::size_t total_slots = 0;
};

struct Pool::Impl {
  Config cfg;
  std::vector<SizeClass> classes;
  // ptr -> size actually handed out. Keyed by the block's own record so
  // deallocate() never trusts the caller's `size` for accounting.
  std::unordered_map<uint8_t*, std::size_t> malloc_fallback;
  // Slots currently handed out. Lets deallocate() ignore a double free
  // instead of pushing the same pointer into free_slots twice (which would
  // hand one buffer to two callers).
  std::unordered_set<uint8_t*> live_slots;
  mutable std::mutex mu;
  bool active = false;
  bool initialised = false;
  std::size_t slab_bytes = 0;
  std::size_t budget_left = 0;
  std::size_t bytes_used = 0;
  std::size_t fallback_bytes = 0;

  ~Impl() {
    const AhbApi& api = ahb_api();
    for (auto& c : classes) {
      for (auto& s : c.slabs) {
        if (s.buf == nullptr) continue;
        if (s.base != nullptr && api.unlock != nullptr) api.unlock(s.buf, nullptr);
        if (api.release != nullptr) api.release(s.buf);
        // Give this slab's share of the process-wide budget back. If the pool
        // is ever destroyed while hot weights are still alive, the next
        // allocation must not think those bytes are still pinned.
        release_ahb(s.bytes);
      }
    }
    for (auto& kv : malloc_fallback) std::free(kv.first);
  }

  bool ensure_slab(SizeClass& c) {
    const AhbApi& api = ahb_api();
    if (!api.ok) return false;
    // AhbDesc::width is a uint32_t. A larger class would silently allocate a
    // smaller buffer than `want` while accounting `want` bytes and handing out
    // slots past the end of the mapping.
    constexpr std::size_t kMaxSlab = 0xFFFF'FFFFULL;
    if (c.slot_bytes == 0 || c.slot_bytes > kMaxSlab) return false;
    if (budget_left < c.slot_bytes) return false;

    std::size_t want = slab_bytes != 0 ? slab_bytes : cfg.slab_bytes;
    if (want > kMaxSlab) want = kMaxSlab;
    // round up to whole slots without overflowing; at least one slot
    want = std::max<std::size_t>(want / c.slot_bytes, 1) * c.slot_bytes;
    want = std::min(want, kMaxSlab / c.slot_bytes * c.slot_bytes);
    if (want > budget_left) want = budget_left / c.slot_bytes * c.slot_bytes;
    if (want < c.slot_bytes) want = c.slot_bytes;
    if (want > budget_left) return false;

    if (!reserve_ahb(want)) {
      if (std::getenv("TOKSHIT_AHB_DEBUG") != nullptr) {
        std::fprintf(stderr,
                     "[ahb] slab %zu KB does not fit global budget (%zu MB "
                     "used of %zu MB)\n",
                     want / 1024,
                     g_ahb_used.load(std::memory_order_relaxed) /
                         (1024 * 1024),
                     ahb_limit_bytes() / (1024 * 1024));
      }
      return false;
    }

    AhbDesc desc{};
    desc.width  = static_cast<uint32_t>(want);
    desc.height = 1;
    desc.layers = 1;
    desc.format = kFormatBlob;
    // Allocation flags must include both read and write: the buffer is filled
    // through the same mapping it is later read from.
    desc.usage  = kCpuReadRarely | kCpuWriteRarely;
    desc.stride = 0;

    AHardwareBuffer* buf = nullptr;
    const int rc = api.allocate(&desc, &buf);
    if (rc != 0 || buf == nullptr) {
      if (std::getenv("TOKSHIT_AHB_DEBUG") != nullptr) {
        std::fprintf(stderr, "[ahb] allocate(%zu KB) failed rc=%d\n",
                     want / 1024, rc);
      }
      release_ahb(want);
      return false;
    }
    void* cpu = nullptr;
    uint64_t used = 0;
    if (!lock_ahb_buffer(buf, &cpu, &used)) {
      std::fprintf(stderr, "[ahb] lock failed, falling back to malloc\n");
      api.release(buf);
      release_ahb(want);
      return false;
    }
    Slab s;
    s.buf = buf;
    s.base = static_cast<uint8_t*>(cpu);
    s.bytes = want;
    s.n_slots = want / c.slot_bytes;
    s.is_ahb = true;
    c.total_slots += s.n_slots;
    c.slabs.push_back(s);
    for (std::size_t k = 0; k < s.n_slots; ++k) {
      c.free_slots.push_back(s.base + k * c.slot_bytes);
    }
    budget_left -= want;
    bytes_used += want;
    active = true;
    if (std::getenv("TOKSHIT_AHB_DEBUG") != nullptr) {
      std::fprintf(stderr,
                   "[ahb] slab %zu KB, slot %zu KB x %zu, usage=0x%llx, "
                   "budget left %zu MB\n",
                   want / 1024, c.slot_bytes / 1024, s.n_slots,
                   static_cast<unsigned long long>(used),
                   budget_left / (1024 * 1024));
    }
    return true;
  }
};

Pool& Pool::instance() {
  // намеренная утечка: пул держит мьютекс и AHB-слябы, а CacheBlock'и из
  // статических/thread_local кэшей могут умирать уже после статической
  // деструкции. разрушение пула раньше них даёт
  // "pthread_mutex_lock called on a destroyed mutex".
  // deliberate leak: the pool owns a mutex and AHB slabs, while CacheBlocks in
  // static/thread_local caches can die after static destruction starts.
  static Pool* p = new Pool();
  return *p;
}

Pool::Pool() : impl_(std::make_unique<Impl>()) {}
Pool::~Pool() = default;

Buffer::Buffer(std::size_t bytes, BufferMode mode) {
  if (bytes == 0) return;
  const bool ahb_only = mode == BufferMode::AhbOnly;
  if (mode != BufferMode::MallocOnly && !ahb_disabled()) {
    const AhbApi& api = ahb_api();
    // width у BLOB-дескриптора 32-битный: больше 4 ГиБ одним буфером не выйдет
    if (api.ok && bytes <= 0xFFFF'FFFFULL) {
      if (!reserve_ahb(bytes)) {
        // AhbOnly — это пробный запрос: вызывающий сам решит, дробить ли
        // сегмент. Не шумим на каждый неудачный размер.
        if (!ahb_only) {
          std::fprintf(stderr,
                       "[ahb] hot: %zu MB does not fit global AHB budget (%zu MB "
                       "used of %zu MB), using malloc\n",
                       bytes / (1024 * 1024),
                       g_ahb_used.load(std::memory_order_relaxed) /
                           (1024 * 1024),
                       ahb_limit_bytes() / (1024 * 1024));
        }
      } else {
        AhbDesc desc{};
        desc.width  = static_cast<uint32_t>(bytes);
        desc.height = 1;
        desc.layers = 1;
        desc.format = kFormatBlob;
        desc.usage  = kCpuReadRarely | kCpuWriteRarely;
        desc.stride = 0;

        AHardwareBuffer* buf = nullptr;
        const int rc = api.allocate(&desc, &buf);
        if (rc == 0 && buf != nullptr) {
          void* cpu = nullptr;
          uint64_t used = 0;
          if (lock_ahb_buffer(buf, &cpu, &used)) {
            data_ = static_cast<uint8_t*>(cpu);
            size_ = bytes;
            handle_ = buf;
            on_ahb_ = true;
            if (std::getenv("TOKSHIT_AHB_DEBUG") != nullptr) {
              std::fprintf(stderr,
                           "[ahb] hot buffer: %zu MB pinned (usage=0x%llx)\n",
                           bytes / (1024 * 1024),
                           static_cast<unsigned long long>(used));
            }
            return;
          }
          api.release(buf);
          if (!ahb_only) {
            std::fprintf(stderr, "[ahb] hot lock failed, using malloc\n");
          }
        } else if (!ahb_only) {
          std::fprintf(stderr,
                       "[ahb] hot allocate %zu MB failed rc=%d, using malloc\n",
                       bytes / (1024 * 1024), rc);
        }
        release_ahb(bytes);
      }
    }
  }
  if (ahb_only) return;  // вызывающий сам раздробит сегмент
  void* p = std::malloc(bytes);
  if (p != nullptr) {
    data_ = static_cast<uint8_t*>(p);
    size_ = bytes;
    on_ahb_ = false;
  }
}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
      handle_(other.handle_),
      on_ahb_(other.on_ahb_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.handle_ = nullptr;
  other.on_ahb_ = false;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this == &other) return *this;
  release();
  data_ = other.data_;
  size_ = other.size_;
  handle_ = other.handle_;
  on_ahb_ = other.on_ahb_;
  other.data_ = nullptr;
  other.size_ = 0;
  other.handle_ = nullptr;
  other.on_ahb_ = false;
  return *this;
}

void Buffer::release() noexcept {
  if (handle_ != nullptr) {
    const AhbApi& api = ahb_api();
    auto* buf = static_cast<AHardwareBuffer*>(handle_);
    if (api.unlock != nullptr) api.unlock(buf, nullptr);
    if (api.release != nullptr) api.release(buf);
    release_ahb(size_);
  } else if (data_ != nullptr) {
    std::free(data_);
  }
  data_ = nullptr;
  size_ = 0;
  handle_ = nullptr;
  on_ahb_ = false;
}

bool HotWeights::allocate(const std::vector<std::size_t>& sizes) {
  clear();
  // offsets[i] — логическое смещение тензора i; offsets[n] — полный размер.
  // Сегменты режутся только по этим границам, поэтому тензор никогда не
  // пересекает mapping.
  // offsets[i] is tensor i's logical offset; offsets[n] is the total. Segments
  // are cut only at these boundaries, so a tensor never crosses a mapping.
  std::vector<std::size_t> offsets(sizes.size() + 1, 0);
  std::size_t total = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    // A wrapped total used to make the MemAvailable check nonsensical and let
    // the loader start allocating an impossible layout.
    if (sizes[i] > std::numeric_limits<std::size_t>::max() - total) {
      return false;
    }
    offsets[i] = total;
    total += sizes[i];
  }
  offsets[sizes.size()] = total;
  if (total == 0) return false;

  // Жёсткий предохранитель: не занимаем последний 1 ГиБ. Лучше остаться на
  // mmap и потерять скорость, чем получить LMK/OOM на телефоне.
  // Hard guard: never eat the last 1 GiB. Falling back to mmap is slower, an
  // OOM on a phone is worse.
  const std::size_t avail = mem_available_bytes();
  constexpr std::size_t kReserve = 1ULL * 1024 * 1024 * 1024;
  if (avail != 0 && (total > avail || avail - total < kReserve)) {
    std::fprintf(stderr,
                 "[ahb] hot: %zu MB requested, MemAvailable %zu MB, reserve "
                 "%zu MB -> keeping mmap\n",
                 total / (1024 * 1024), avail / (1024 * 1024),
                 kReserve / (1024 * 1024));
    return false;
  }

  const std::size_t seg_target = hot_seg_target_bytes();

  // Группируем подряд идущие тензоры в сегмент не больше seg_target. Если
  // следующий тензор не влезает — начинаем новый сегмент. Тензор больше
  // seg_target получает собственный сегмент.
  // Pack consecutive tensors into a segment of at most seg_target bytes. The
  // next tensor starts a new segment when it does not fit; a tensor larger than
  // seg_target gets a segment of its own.
  struct Run {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t bytes = 0;
  };
  std::vector<Run> runs;
  if (seg_target == 0) {
    runs.reserve(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      if (sizes[i] != 0) runs.push_back(Run{i, i + 1, sizes[i]});
    }
  } else {
    std::size_t begin = 0;
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      const std::size_t s = sizes[i];
      if (s == 0) {
        if (bytes == 0) begin = i + 1;
        continue;
      }
      // `bytes <= seg_target`, so the subtraction cannot underflow.
      if (bytes != 0 && s > seg_target - bytes) {
        runs.push_back(Run{begin, i, bytes});
        begin = i;
        bytes = 0;
      }
      if (bytes == 0) begin = i;
      bytes += s;
    }
    if (bytes != 0) runs.push_back(Run{begin, sizes.size(), bytes});
  }

  const bool ahb_ok = !ahb_disabled() && ahb_api().ok;
  // Не дробим меньше мегабайта: на такой гранулярности per-buffer overhead
  // уже не окупается, а бюджет AHB всё равно не спасёт.
  // Do not split below 1 MiB: at that granularity per-buffer overhead wins
  // nothing and an exhausted AHB budget will not recover.
  constexpr std::size_t kMinSplit = 1ULL << 20;
  // Если свободного AHB-бюджета меньше этого, дробить бессмысленно: получим
  // сотни malloc-сегментов. Лучше отдать остаток сегментов в malloc целиком.
  // Below this much free AHB budget splitting is pointless: it only produces
  // hundreds of malloc segments, so the remaining runs go straight to malloc.
  constexpr std::size_t kMinUsefulAhb = 8ULL << 20;
  const std::size_t ahb_limit = ahb_limit_bytes();
  auto ahb_room = [&]() -> std::size_t {
    const std::size_t used = g_ahb_used.load(std::memory_order_relaxed);
    return used >= ahb_limit ? 0 : ahb_limit - used;
  };

  bool ok = true;
  auto alloc_run = [&](auto&& self, std::size_t begin, std::size_t end,
                       std::size_t bytes) -> bool {
    Buffer buf;
    bool budget_blocked = false;
    if (ahb_ok) {
      const std::size_t room = ahb_room();
      if (bytes > room && room < kMinUsefulAhb) {
        budget_blocked = true;
        buf = Buffer(bytes, BufferMode::MallocOnly);
      } else {
        buf = Buffer(bytes, BufferMode::AhbOnly);
      }
    } else {
      buf = Buffer(bytes);
    }
    if (!buf.valid() && ahb_ok && !budget_blocked) {
      // gralloc не дал сегмент целиком — режем по границе тензора и пробуем
      // меньшие буферы. Это лучше, чем молча уйти в malloc на весь сегмент.
      // gralloc refused the whole segment: split at a tensor boundary and try
      // smaller buffers instead of silently dropping the whole run to malloc.
      if (end - begin > 1 && bytes > kMinSplit) {
        const std::size_t half = bytes / 2;
        std::size_t mid = begin + 1;
        std::size_t left = sizes[begin];
        std::size_t best_mid = mid;
        std::size_t best_diff =
            left > half ? left - half : half - left;
        while (mid < end) {
          left += sizes[mid];
          ++mid;
          if (mid >= end) break;
          const std::size_t diff = left > half ? left - half : half - left;
          if (diff < best_diff) {
            best_diff = diff;
            best_mid = mid;
          }
        }
        if (best_mid > begin && best_mid < end) {
          return self(self, begin, best_mid, offsets[best_mid] - offsets[begin]) &&
                 self(self, best_mid, end, offsets[end] - offsets[best_mid]);
        }
      }
      // Один тензор или уже слишком мелко: последняя попытка через Auto
      // (AHB ещё раз, затем malloc).
      // Single tensor or already tiny: one Auto attempt (AHB again, then
      // malloc).
      buf = Buffer(bytes);
    }
    if (!buf.valid()) return false;
    if (buf.on_ahb()) ahb_bytes_ += bytes;
    segs_.push_back(Seg{offsets[begin], bytes, std::move(buf)});
    return true;
  };

  segs_.reserve(runs.size());
  for (const Run& run : runs) {
    if (!alloc_run(alloc_run, run.begin, run.end, run.bytes)) {
      ok = false;
      break;
    }
  }
  if (!ok) {
    clear();
    return false;
  }
  size_ = total;

  std::size_t ahb_segs = 0;
  std::size_t malloc_segs = 0;
  std::size_t min_seg = 0;
  std::size_t max_seg = 0;
  for (const Seg& seg : segs_) {
    if (seg.buf.on_ahb()) {
      ++ahb_segs;
    } else {
      ++malloc_segs;
    }
    min_seg = min_seg == 0 ? seg.len : std::min(min_seg, seg.len);
    max_seg = std::max(max_seg, seg.len);
  }
  std::fprintf(stderr,
               "[ahb] hot: %zu tensor(s) -> %zu segment(s) (target %zu MB, "
               "min %.1f MB, max %.1f MB): %zu AHB, %zu malloc, %zu MB total, "
               "%zu MB pinned\n",
               sizes.size(), segs_.size(), seg_target / (1024 * 1024),
               static_cast<double>(min_seg) / (1024.0 * 1024.0),
               static_cast<double>(max_seg) / (1024.0 * 1024.0), ahb_segs,
               malloc_segs, size_ / (1024 * 1024),
               ahb_bytes_ / (1024 * 1024));
  return true;
}

void HotWeights::clear() {
  segs_.clear();
  size_ = 0;
  ahb_bytes_ = 0;
}

uint8_t* HotWeights::data_at(std::size_t offset) {
  auto it = std::upper_bound(
      segs_.begin(), segs_.end(), offset,
      [](std::size_t off, const Seg& s) { return off < s.start; });
  if (it == segs_.begin()) return nullptr;
  --it;
  // `offset - it->start` cannot overflow (offset >= start), and comparing it
  // to len avoids `start + len` wrapping around for a pathological segment.
  if (offset - it->start >= it->len) return nullptr;
  return it->buf.data() + (offset - it->start);
}

const uint8_t* HotWeights::data_at(std::size_t offset) const {
  auto it = std::upper_bound(
      segs_.begin(), segs_.end(), offset,
      [](std::size_t off, const Seg& s) { return off < s.start; });
  if (it == segs_.begin()) return nullptr;
  --it;
  if (offset - it->start >= it->len) return nullptr;
  return it->buf.data() + (offset - it->start);
}

void Pool::init(const Config& cfg) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (impl_->initialised) return;
  impl_->initialised = true;
  impl_->cfg = cfg;

  const char* disabled = std::getenv("TOKSHIT_AHB");
  if (!cfg.enabled || (disabled != nullptr && *disabled != '\0' &&
                       disabled[0] == '0')) {
    std::fprintf(stderr, "[ahb] disabled, expert cache uses malloc\n");
    return;
  }
  if (cfg.total_bytes == 0) return;

  std::vector<std::size_t> sizes = cfg.slot_sizes;
  if (sizes.empty()) sizes.push_back(cfg.slot_bytes);
  for (std::size_t& s : sizes) {
    if (s == 0) continue;
    s = (s + 4095) & ~static_cast<std::size_t>(4095);
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  sizes.erase(std::remove(sizes.begin(), sizes.end(), 0), sizes.end());
  if (sizes.empty()) return;

  impl_->slab_bytes = cfg.slab_bytes;
  // Не даём expert-кэшу съесть последние гигабайты: на телефоне это
  // заканчивается LMK/OOM, а не ускорением.
  // Never let the expert cache eat the last few gigabytes: on a phone that ends
  // in LMK/OOM, not in speed.
  std::size_t budget = std::min(cfg.total_bytes, ahb_limit_bytes());
  const std::size_t avail = mem_available_bytes();
  constexpr std::size_t kReserve = 1ULL * 1024 * 1024 * 1024;
  if (avail != 0 && (budget > avail || avail - budget < kReserve)) {
    budget = avail > kReserve ? avail - kReserve : 0;
    budget = budget / (64ULL * 1024 * 1024) * (64ULL * 1024 * 1024);
    std::fprintf(stderr,
                 "[ahb] expert budget capped: %zu -> %zu MB (MemAvailable %zu "
                 "MB, global AHB limit %zu MB)\n",
                 cfg.total_bytes / (1024 * 1024), budget / (1024 * 1024),
                 avail / (1024 * 1024), ahb_limit_bytes() / (1024 * 1024));
  }
  impl_->budget_left = budget;
  for (std::size_t s : sizes) {
    SizeClass c;
    c.slot_bytes = s;
    impl_->classes.push_back(std::move(c));
  }

  // Не аллоцируем слябы заранее: эксперт-кэш заполняется только после того,
  // как роутер реально начнёт выбирать экспертов. Так 6 ГБ конфига не
  // превращаются в 6 ГБ занятой памяти на старте.
  // Slabs are allocated lazily: an idle cache must not pin the whole budget at
  // startup.
  std::fprintf(stderr, "[ahb] ready: %zu size class(es), %zu MB budget\n",
               impl_->classes.size(), impl_->budget_left / (1024 * 1024));
}

uint8_t* Pool::allocate(std::size_t size) {
  if (size == 0) return nullptr;
  std::lock_guard<std::mutex> lock(impl_->mu);
  SizeClass* cls = nullptr;
  for (auto& c : impl_->classes) {
    if (c.slot_bytes >= size) {
      cls = &c;
      break;
    }
  }
  if (cls != nullptr) {
    if (cls->free_slots.empty()) (void)impl_->ensure_slab(*cls);
    if (!cls->free_slots.empty()) {
      uint8_t* p = cls->free_slots.back();
      cls->free_slots.pop_back();
      impl_->live_slots.insert(p);
      return p;
    }
  }
  // Когда AHB-бюджет исчерпан, не уходим в бесконечный malloc: лучше
  // вернуть nullptr, и вызывающий прочитает эксперта из mmap. Иначе кэш
  // «5 ГиБ AHB + сколько получится malloc» снова приводит к OOM.
  // When the AHB budget is exhausted, avoid unbounded malloc: return nullptr
  // and let the caller read the expert from mmap instead of risking OOM.
  constexpr std::size_t kFallbackCap = 256ULL * 1024 * 1024;
  // `fallback_bytes + size` used to overflow for a huge request, sneaking past
  // the cap and reaching malloc(SIZE_MAX).
  if (size > kFallbackCap || impl_->fallback_bytes > kFallbackCap - size) {
    return nullptr;
  }
  void* p = std::malloc(size);
  if (p != nullptr) {
    impl_->malloc_fallback.emplace(static_cast<uint8_t*>(p), size);
    impl_->fallback_bytes += size;
  }
  return static_cast<uint8_t*>(p);
}

void Pool::deallocate(uint8_t* ptr, std::size_t size) {
  if (ptr == nullptr) return;
  (void)size;  // hint only, accounting comes from the pool's own records
  std::lock_guard<std::mutex> lock(impl_->mu);
  // `size` is only a hint. Both the slab pool and the malloc fallback know
  // what they handed out, so accounting must never be driven by the caller:
  // a mismatched size used to underflow fallback_bytes (permanently disabling
  // the fallback) and the old loop `break` after the first class whose slot
  // size was >= size, stranding a slab slot forever when the hint was small.
  for (auto& c : impl_->classes) {
    for (auto& s : c.slabs) {
      if (ptr >= s.base && ptr < s.base + s.bytes) {
        // A second deallocate of the same slot must not put it on the free
        // list twice: two callers would then get the same buffer.
        if (impl_->live_slots.erase(ptr) != 0) c.free_slots.push_back(ptr);
        return;
      }
    }
  }
  auto it = impl_->malloc_fallback.find(ptr);
  if (it != impl_->malloc_fallback.end()) {
    const std::size_t owned = it->second;
    impl_->malloc_fallback.erase(it);
    impl_->fallback_bytes =
        owned <= impl_->fallback_bytes ? impl_->fallback_bytes - owned : 0;
    std::free(ptr);
  }
}

bool Pool::is_ahb(uint8_t* ptr) const {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->mu);
  for (const auto& c : impl_->classes) {
    for (const auto& s : c.slabs) {
      if (ptr >= s.base && ptr < s.base + s.bytes) return true;
    }
  }
  return false;
}

std::size_t Pool::slots_total() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::size_t n = 0;
  for (const auto& c : impl_->classes) n += c.total_slots;
  return n;
}

std::size_t Pool::slots_free() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::size_t n = 0;
  for (const auto& c : impl_->classes) n += c.free_slots.size();
  return n;
}

std::size_t Pool::slot_bytes() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::size_t m = 0;
  for (const auto& c : impl_->classes) m = std::max(m, c.slot_bytes);
  return m;
}

std::size_t Pool::class_count() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->classes.size();
}

std::size_t Pool::bytes_used() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->bytes_used;
}

std::size_t Pool::fallback_bytes() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->fallback_bytes;
}

bool Pool::ahb_active() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->active;
}

}  // namespace tokshit::ahb
