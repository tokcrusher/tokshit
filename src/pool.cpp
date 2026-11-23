#include "tokshit/pool.hpp"

#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <time.h>
#include <vector>

namespace tokshit::pool {
namespace {

constexpr std::size_t kCacheLine = 64;

// читаем env как размер, мусор -> fallback
// parse an env var as size, garbage -> fallback
std::size_t env_usize(const char* name, std::size_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(v, &end, 10);
  if (end == v) return fallback;
  return static_cast<std::size_t>(parsed);
}

// прибить текущий поток к ядру cpu. best effort, на андроиде может не дать
// pin the current thread to a cpu. best effort, android may refuse
void pin_to_cpu(std::size_t cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
  CPU_SET(static_cast<int>(cpu), &set);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  (void)::sched_setaffinity(0, sizeof(set), &set);
}

// cpu max frequency in kHz, 0 if sysfs is unavailable
unsigned long cpu_max_freq(std::size_t cpu) {
  char path[128];
  std::snprintf(path, sizeof(path),
                "/sys/devices/system/cpu/cpu%zu/cpufreq/cpuinfo_max_freq",
                cpu);
  FILE* f = std::fopen(path, "r");
  if (f == nullptr) return 0;
  unsigned long v = 0;
  if (std::fscanf(f, "%lu", &v) != 1) v = 0;
  std::fclose(f);
  return v;
}

// CPUs sorted by max frequency, fastest first. Falls back to the natural
// order when sysfs is missing. This keeps 7 spinning workers off the two
// 2.0 GHz little cores on a 3.2+3.0+2.8 GHz phone.
std::vector<std::size_t> cpus_by_freq(std::size_t total) {
  std::vector<std::size_t> order(total);
  for (std::size_t i = 0; i < total; ++i) order[i] = i;
  std::vector<unsigned long> freq(total, 0);
  bool any = false;
  for (std::size_t i = 0; i < total; ++i) {
    freq[i] = cpu_max_freq(i);
    any = any || freq[i] != 0;
  }
  if (!any) return order;
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) {
                     return freq[a] > freq[b];
                   });
  return order;
}

// explicit CPU mask for experiments: TOKSHIT_POOL_CPUS=2,3,4
std::vector<std::size_t> parse_cpu_list(const char* s) {
  std::vector<std::size_t> out;
  if (s == nullptr) return out;
  while (*s != '\0') {
    while (*s == ',' || *s == ' ' || *s == '\t') ++s;
    if (*s == '\0') break;
    char* end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (end == s) break;
    out.push_back(static_cast<std::size_t>(v));
    s = end;
  }
  return out;
}

// просим sched_fifo и отрицательный nice, обычно в отказ но попытка не пытка
// ask for sched_fifo and a negative nice, usually denied but whatever
void try_realtime() {
  struct sched_param param {};
  param.sched_priority = 1;
  (void)::sched_setscheduler(0, SCHED_FIFO, &param);
  (void)::setpriority(PRIO_PROCESS, 0, -10);
}

#if defined(__aarch64__)
constexpr long kSchedSetattrNr = 274;
#elif defined(__x86_64__)
constexpr long kSchedSetattrNr = 314;
#else
constexpr long kSchedSetattrNr = -1;
#endif

int boost_uclamp() {
  if (kSchedSetattrNr < 0) return -1;
  struct SchedAttr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
    uint32_t sched_util_min;
    uint32_t sched_util_max;
  };
  SchedAttr attr {};
  attr.size = static_cast<uint32_t>(sizeof(SchedAttr));
  attr.sched_policy = 0;
  attr.sched_flags = 0x20;  // SCHED_FLAG_UTIL_CLAMP_MIN
  attr.sched_util_min = 1024;
  attr.sched_util_max = 1024;
  return static_cast<int>(::syscall(kSchedSetattrNr, 0, &attr, 0U));
}

}  // namespace

struct TaskPool::Impl {
  struct alignas(kCacheLine) State {
    std::atomic<uint64_t> generation{0};
    alignas(kCacheLine) std::atomic<std::size_t> next{0};
    alignas(kCacheLine) std::atomic<std::size_t> done{0};
    alignas(kCacheLine) TaskFn run = nullptr;
    void* ctx = nullptr;
    std::size_t n = 0;
    std::atomic<bool> stop{false};
    std::atomic<bool> active{false};
    std::atomic<uint32_t> idle{0};
    std::atomic<uint64_t> exec_main{0};
    std::atomic<uint64_t> exec_workers{0};
    // диагностика барьера, включается TOKSHIT_POOL_WAIT_STATS
    // barrier diagnostics, enabled by TOKSHIT_POOL_WAIT_STATS
    std::atomic<uint64_t> stat_calls{0};
    std::atomic<uint64_t> stat_work_ns{0};
    std::atomic<uint64_t> stat_wait_ns{0};
    std::atomic<uint64_t> stat_wait_max{0};
    std::atomic<uint64_t> stat_worker_ns{0};
    std::atomic<uint64_t> stat_worker_max{0};
  };

  State state;
  std::vector<std::thread> threads;
  std::size_t workers = 0;

  void worker_loop() {
    // воркер крутится и ждёт новую generation. если active то yield, иначе
    // через какое-то время уходит в sleep чтобы не жечь батарею
    // worker spins waiting for a new generation. when active it yields,
    // otherwise after a bit it sleeps so it doesnt eat the battery
    uint64_t seen = 0;
    for (;;) {
      const uint64_t g = state.generation.load(std::memory_order_acquire);
      if (g == seen) {
        if (state.stop.load(std::memory_order_relaxed)) return;
        if (state.active.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        } else {
          const uint32_t idle = state.idle.fetch_add(1, std::memory_order_relaxed);
          if (idle < 4096) {
            std::this_thread::yield();
          } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
        continue;
      }
      seen = g;
      state.idle.store(0, std::memory_order_relaxed);
      if (state.stop.load(std::memory_order_relaxed)) return;

      TaskFn run = state.run;
      void* ctx = state.ctx;
      const std::size_t n = state.n;
      const auto t_begin = std::chrono::steady_clock::now();
      uint64_t mine = 0;
      for (;;) {
        const std::size_t i = state.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) break;
        run(ctx, i);
        ++mine;
      }
      {
        const auto dt = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_begin)
                .count());
        state.stat_worker_ns.fetch_add(dt, std::memory_order_relaxed);
        uint64_t prev = state.stat_worker_max.load(std::memory_order_relaxed);
        while (dt > prev &&
               !state.stat_worker_max.compare_exchange_weak(
                   prev, dt, std::memory_order_relaxed)) {
        }
      }
      state.exec_workers.fetch_add(mine, std::memory_order_relaxed);
      state.done.fetch_add(1, std::memory_order_release);
    }
  }
};

namespace {
// snap-to-4 по умолчанию на арме; определение ниже (после конструктора),
// но вызывается из него
std::size_t default_workers();
}  // namespace

TaskPool::TaskPool() {
  impl_ = new Impl();
  const std::size_t total =
      std::max<std::size_t>(1, std::thread::hardware_concurrency());
  impl_->workers = env_usize("TOKSHIT_POOL_WORKERS", default_workers());
  if (impl_->workers > total) impl_->workers = total;
  if (impl_->workers > 1024) impl_->workers = 1024;

  const bool pin = std::getenv("TOKSHIT_POOL_PIN") != nullptr;
  const bool rt = std::getenv("TOKSHIT_POOL_RT") != nullptr;
  const bool stats = std::getenv("TOKSHIT_POOL_STATS") != nullptr;
  std::vector<std::size_t> order = cpus_by_freq(total);
  bool explicit_mask = false;
  if (const char* cpus = std::getenv("TOKSHIT_POOL_CPUS")) {
    std::vector<std::size_t> explicit_cpus = parse_cpu_list(cpus);
    if (!explicit_cpus.empty()) {
      order = std::move(explicit_cpus);
      explicit_mask = true;
    }
  }
#if defined(__aarch64__)
  // The prime X4 is faster per thread but finishes its share early and then
  // waits on the A720s. Keep the main thread on the fastest core and pin the
  // workers to the next (more homogeneous) cores.
  const bool skip_fastest = !explicit_mask && order.size() > 1;
#else
  const bool skip_fastest = false;
#endif

  impl_->threads.reserve(impl_->workers);
  for (std::size_t i = 0; i < impl_->workers; ++i) {
    impl_->threads.emplace_back(
        [this, i, pin, rt, stats, order, skip_fastest]() {
          if (rt) try_realtime();
          const int rc = boost_uclamp();
          if (i == 0 && stats) {
            std::fprintf(stderr, "[pool] uclamp.min=1024 -> %d\n", rc);
          }
          if (pin && !order.empty()) {
            const std::size_t slot = skip_fastest ? i + 1 : i;
            pin_to_cpu(order[slot % order.size()]);
          }
          impl_->worker_loop();
        });
  }
}

TaskPool::~TaskPool() {
  if (impl_ == nullptr) return;
  impl_->state.stop.store(true, std::memory_order_relaxed);
  for (std::thread& t : impl_->threads) {
    if (t.joinable()) t.join();
  }
  delete impl_;
  impl_ = nullptr;
}

TaskPool& TaskPool::instance() {
  static TaskPool pool;
  return pool;
}

// Snap-to-4 threads: decode is GEMV (memory-bound), prefill is GEMM. Four
// big cores is the sweet spot on a modern phone; more threads mostly spin on
// the heterogeneous little cores and heat the fast ones.
namespace {
std::size_t default_workers() {
#if defined(__aarch64__)
  return 4;
#else
  return std::max<std::size_t>(1, std::thread::hardware_concurrency() - 1);
#endif
}
}  // namespace

void TaskPool::run(std::size_t n, TaskFn fn, void* ctx) {
  if (n == 0 || fn == nullptr) return;
  if (impl_->workers == 0) {
    for (std::size_t i = 0; i < n; ++i) fn(ctx, i);
    return;
  }

  // публикуем задачу, будим воркеров через generation, сами тоже берём задачи
  // и в конце ждём пока все воркеры не отметятся done
  // publish the job, wake workers via generation, take tasks ourselves too,
  // and at the end wait until every worker bumps done
  Impl::State& st = impl_->state;
  static const bool wait_stats =
      std::getenv("TOKSHIT_POOL_WAIT_STATS") != nullptr;
  const auto t_pub = std::chrono::steady_clock::now();
  st.run = fn;
  st.ctx = ctx;
  st.n = n;
  st.next.store(0, std::memory_order_relaxed);
  st.done.store(0, std::memory_order_relaxed);
  st.generation.fetch_add(1, std::memory_order_acq_rel);

  uint64_t mine = 0;
  for (;;) {
    const std::size_t i = st.next.fetch_add(1, std::memory_order_relaxed);
    if (i >= n) break;
    fn(ctx, i);
    ++mine;
  }
const auto t_work = std::chrono::steady_clock::now();
  // Гетерогенные ядра: быстрые чистить спином, но периодически спать —
  // на 2.8 ГГц-ядрах ждать на YIELD забивает опропорциональную тепловую разницу.
  {
    const uint64_t target = impl_->workers;
    // decode calls this barrier thousands of times per token; sleeping here
    // adds ~1 ms per short task and dominates the token time. Spin-yield is
    // cheaper for the common <100 us task; the worker loop still sleeps when
    // the pool is idle.
    while (st.done.load(std::memory_order_acquire) < target) {
      std::this_thread::yield();
    }
  }
  if (wait_stats) {
    const auto t_end = std::chrono::steady_clock::now();
    const auto work_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_work - t_pub)
            .count());
    const auto wait_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_work)
            .count());
    st.stat_calls.fetch_add(1, std::memory_order_relaxed);
    st.stat_work_ns.fetch_add(work_ns, std::memory_order_relaxed);
    st.stat_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    uint64_t prev = st.stat_wait_max.load(std::memory_order_relaxed);
    while (wait_ns > prev &&
           !st.stat_wait_max.compare_exchange_weak(prev, wait_ns,
                                                   std::memory_order_relaxed)) {
    }
  }

  st.exec_main.fetch_add(mine, std::memory_order_relaxed);
}

void TaskPool::set_active(bool active) {
  // на время декода поднимаем uclamp и пинним главный поток, чтобы телефон не
  // решил что мы фоновый процесс
  // during decode bump uclamp and pin the main thread so the phone doesnt
  // decide we are a background app
  if (active) {
    const int rc = boost_uclamp();
    if (std::getenv("TOKSHIT_POOL_STATS") != nullptr) {
      std::fprintf(stderr, "[pool] main uclamp.min=1024 -> %d\n", rc);
    }
    if (std::getenv("TOKSHIT_POOL_PIN") != nullptr) {
      static const std::size_t fastest = [] {
        const std::size_t total =
            std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::vector<std::size_t> order = cpus_by_freq(total);
        return order.empty() ? std::size_t{0} : order.front();
      }();
      const std::size_t main_cpu = env_usize("TOKSHIT_MAIN_CPU", fastest);
      pin_to_cpu(main_cpu);
    }
  }
  impl_->state.active.store(active, std::memory_order_relaxed);
  if (!active && std::getenv("TOKSHIT_POOL_WAIT_STATS") != nullptr) {
    const auto& s = impl_->state;
    const uint64_t calls = s.stat_calls.load(std::memory_order_relaxed);
    const uint64_t work = s.stat_work_ns.load(std::memory_order_relaxed);
    const uint64_t wait = s.stat_wait_ns.load(std::memory_order_relaxed);
    const uint64_t wmax = s.stat_wait_max.load(std::memory_order_relaxed);
    const uint64_t worker_max =
        s.stat_worker_max.load(std::memory_order_relaxed);
    std::fprintf(
        stderr,
        "[pool-wait] calls=%llu work=%.3fs wait=%.3fs wait_max=%.1fms "
        "worker_max=%.1fms\n",
        static_cast<unsigned long long>(calls),
        static_cast<double>(work) / 1e9, static_cast<double>(wait) / 1e9,
        static_cast<double>(wmax) / 1e6,
        static_cast<double>(worker_max) / 1e6);
  }
  if (!active && std::getenv("TOKSHIT_POOL_STATS") != nullptr) {
    std::fprintf(stderr, "[pool] workers=%zu main_tasks=%llu worker_tasks=%llu\n",
                 impl_->workers,
                 static_cast<unsigned long long>(
                     impl_->state.exec_main.load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(
                     impl_->state.exec_workers.load(std::memory_order_relaxed)));
  }
}

bool TaskPool::active() const noexcept {
  return impl_->state.active.load(std::memory_order_relaxed);
}

std::size_t TaskPool::workers() const noexcept { return impl_->workers; }

uint64_t TaskPool::main_tasks() const noexcept {
  return impl_->state.exec_main.load(std::memory_order_relaxed);
}

uint64_t TaskPool::worker_tasks() const noexcept {
  return impl_->state.exec_workers.load(std::memory_order_relaxed);
}

bool is_active() noexcept { return TaskPool::instance().active(); }

void set_active(bool active) { TaskPool::instance().set_active(active); }

}  // namespace tokshit::pool
