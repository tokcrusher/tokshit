// Compare matvec parallel paths: std::thread chunks vs the persistent pool.
// usage: pbench <gguf> [iters] <tensor>...
// Env: PB_ROWS=4,8,16,... (rows_per_task list for the pool)
//      PB_RAW=1 (read-only walk), PB_WAKE=1 (barrier cost), PB_BURN=secs,
//      PB_LIMIT_ROWS=n, PB_SKIP_THREADS=1, PB_TAIL=1.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "tokshit/gguf.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/pool.hpp"
#include "tokshit/quant.hpp"

namespace {

struct Seg {
  std::span<const uint8_t> bytes;
  std::size_t rb = 0;
  tokshit::QuantType quant = tokshit::QuantType::F32;
  std::size_t in_dim = 0;
  const float* x = nullptr;
  const tokshit::kernels::ActQ8* act = nullptr;
  float* out = nullptr;
  std::size_t rows = 0;
};

struct Job {
  const Seg* segs = nullptr;
  const std::size_t* task_starts = nullptr;
  std::size_t task_starts_len = 0;
  std::size_t rows_per_task = 0;
};

void row_task(void* ctx, std::size_t task) {
  const Job& j = *static_cast<const Job*>(ctx);
  std::size_t s = 0;
  while (s + 1 < j.task_starts_len && task >= j.task_starts[s + 1]) ++s;
  const std::size_t local = task - j.task_starts[s];
  const Seg& seg = j.segs[s];
  const std::size_t r0 = local * j.rows_per_task;
  const std::size_t r1 = std::min(r0 + j.rows_per_task, seg.rows);
  std::vector<float> scratch;
  for (std::size_t r = r0; r < r1; ++r) {
    const auto row = seg.bytes.subspan(r * seg.rb, seg.rb);
    seg.out[r] = tokshit::kernels::dot_row(
        seg.quant, row, seg.rb, std::span<const float>(seg.x, seg.in_dim),
        *seg.act, seg.in_dim, scratch);
  }
}

void raw_task(void* ctx, std::size_t task) {
  const Job& j = *static_cast<const Job*>(ctx);
  const Seg& seg = j.segs[0];
  const std::size_t r0 = task * j.rows_per_task;
  const std::size_t r1 = std::min(r0 + j.rows_per_task, seg.rows);
  uint64_t acc = 0;
  for (std::size_t r = r0; r < r1; ++r) {
    const auto row = seg.bytes.subspan(r * seg.rb, seg.rb);
    std::size_t c = 0;
    while (c + 8 <= row.size()) {
      uint64_t word = 0;
      std::memcpy(&word, row.data() + c, sizeof(word));
      acc += word;
      c += 8;
    }
  }
  seg.out[r0] = static_cast<float>(acc & 0xFFFFFFU);
}

void nop_task(void*, std::size_t) {}

void run_pool(const Seg& seg, std::size_t rows_per_task) {
  const std::size_t total = (seg.rows + rows_per_task - 1) / rows_per_task;
  const std::size_t starts[2] = {0, total};
  const Job job{&seg, starts, 2, rows_per_task};
  if (std::getenv("PB_RAW") != nullptr) {
    tokshit::pool::run_tasks(total, &raw_task, const_cast<Job*>(&job));
  } else {
    tokshit::pool::run_tasks(total, &row_task, const_cast<Job*>(&job));
  }
}

void bench_wake() {
  const std::size_t iters = 2000;
  auto t0 = tokshit::Clock::now();
  for (std::size_t i = 0; i < iters; ++i) {
    tokshit::pool::run_tasks(1, &nop_task, nullptr);
  }
  double dt = std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
  std::fprintf(stderr, "[wake] %zu empty tasks in %.3fs = %.1f us/call\n", iters,
               dt, dt * 1e6 / static_cast<double>(iters));
  t0 = tokshit::Clock::now();
  for (std::size_t i = 0; i < iters; ++i) {
    tokshit::pool::run_tasks(256, &nop_task, nullptr);
  }
  dt = std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
  std::fprintf(stderr, "[wake] 256 tasks x %zu = %.1f us/task\n", iters,
               dt * 1e6 / (static_cast<double>(iters) * 256.0));
}

std::vector<std::size_t> parse_rows() {
  std::vector<std::size_t> out;
  const char* v = std::getenv("PB_ROWS");
  const std::string text = v != nullptr ? v : "4,8,16,32,64,128";
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const std::size_t comma = text.find(',', pos);
    const std::string item =
        text.substr(pos, comma == std::string::npos ? std::string::npos
                                                    : comma - pos);
    if (!item.empty()) out.push_back(static_cast<std::size_t>(std::strtoull(item.c_str(), nullptr, 10)));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (std::getenv("PB_WAKE") != nullptr) {
    tokshit::pool::set_active(true);
    bench_wake();
    tokshit::pool::set_active(false);
    return 0;
  }

  std::string path = argc > 1
                         ? argv[1]
                         : "/data/data/com.termux/files/home/models/"
                           "flint-UD-Q3_K_XL.gguf";
  std::size_t iters = 3;
  if (argc > 2) iters = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));

  auto opened = tokshit::Gguf::open(path);
  if (!opened.ok()) {
    std::fprintf(stderr, "gguf: %s: %s\n",
                 tokshit::to_string(opened.error().code).data(),
                 opened.error().message.c_str());
    return 1;
  }
  tokshit::Gguf gguf = std::move(opened.value());
  auto mapped = gguf.mmap_file();
  if (!mapped.ok()) {
    std::fprintf(stderr, "mmap: %s\n", mapped.error().message.c_str());
    return 1;
  }
  tokshit::Mmap mm = std::move(mapped.value());
  const auto bytes = mm.bytes();
  const uint8_t* data = reinterpret_cast<const uint8_t*>(bytes.data());
  const std::size_t base_off = static_cast<std::size_t>(gguf.data_offset());
  const std::vector<std::size_t> rows_list = parse_rows();

  std::vector<std::string> names;
  for (int i = 3; i < argc; ++i) names.emplace_back(argv[i]);

  struct Total {
    double time = 0;
    double bytes = 0;
    uint64_t minflt = 0;
    uint64_t majflt = 0;
    uint64_t io = 0;
  };
  std::vector<Total> totals(iters);

  tokshit::pool::set_active(true);

  if (const char* burn = std::getenv("PB_BURN"); burn != nullptr) {
    const double secs = std::strtod(burn, nullptr);
    const auto deadline =
        tokshit::Clock::now() + std::chrono::duration_cast<tokshit::Clock::duration>(
                                    std::chrono::duration<double>(secs));
    std::vector<std::thread> burners;
    burners.reserve(8);
    for (std::size_t i = 0; i < 8; ++i) {
      burners.emplace_back([deadline]() {
        while (tokshit::Clock::now() < deadline) {
#if defined(__aarch64__) || defined(__x86_64__)
          asm volatile("" ::: "memory");
#endif
        }
      });
    }
    for (std::thread& b : burners) b.join();
    const auto freqs = tokshit::cpu_freqs();
    std::fprintf(stderr, "[burn] %.0fs ready, freqs:", secs);
    for (uint32_t f : freqs) std::fprintf(stderr, " %u", f);
    std::fputc('\n', stderr);
  }

  const std::size_t workers = tokshit::pool::TaskPool::instance().workers();

  for (const std::string& name : names) {
    auto tensor = gguf.tensor(name);
    if (!tensor.ok()) {
      std::fprintf(stderr, "skip %s: %s\n", name.c_str(),
                   tensor.error().message.c_str());
      continue;
    }
    const tokshit::GgufTensor& t = *tensor.value();
    const std::size_t in_dim = static_cast<std::size_t>(t.dims.empty() ? 0 : t.dims[0]);
    std::size_t out_dim = 1;
    for (std::size_t i = 1; i < t.dims.size(); ++i) {
      out_dim *= static_cast<std::size_t>(t.dims[i]);
    }
    if (const char* lim = std::getenv("PB_LIMIT_ROWS"); lim != nullptr) {
      out_dim = std::min(out_dim, static_cast<std::size_t>(std::strtoull(lim, nullptr, 10)));
    }
    auto rb_result = tokshit::row_bytes(t.quant, in_dim);
    if (!rb_result.ok()) {
      std::fprintf(stderr, "skip %s: %s\n", name.c_str(),
                   rb_result.error().message.c_str());
      continue;
    }
    const std::size_t rb = static_cast<std::size_t>(rb_result.value());
    const std::size_t base = base_off + static_cast<std::size_t>(t.offset);

    std::vector<float> x(in_dim);
    for (std::size_t i = 0; i < in_dim; ++i) {
      x[i] = (static_cast<float>((i * 37 + 11) % 101) - 50.0F) / 50.0F;
    }
    const auto act = tokshit::kernels::quantize(x);
    std::vector<float> y(out_dim, 0.0F);
    const double nbytes = static_cast<double>(out_dim * rb);
    std::fprintf(stderr, "== %s %s %zux%zu %.1f MB\n", name.c_str(),
                 std::string(tokshit::quant_name(t.quant)).c_str(), out_dim,
                 in_dim, nbytes / 1e6);

    // std::thread baseline: contiguous chunks, one per worker.
    if (std::getenv("PB_SKIP_THREADS") == nullptr) {
      for (std::size_t it = 0; it < iters; ++it) {
        const auto t0 = tokshit::Clock::now();
        const std::size_t nthreads = std::max<std::size_t>(workers + 1, 1);
        const std::size_t chunk = std::max<std::size_t>(out_dim / nthreads, 64);
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> threads;
        threads.reserve(nthreads);
        for (std::size_t tno = 0; tno < nthreads; ++tno) {
          threads.emplace_back([&]() {
            std::vector<float> scratch;
            for (;;) {
              const std::size_t r0 = next.fetch_add(chunk);
              if (r0 >= out_dim) break;
              const std::size_t r1 = std::min(r0 + chunk, out_dim);
              for (std::size_t r = r0; r < r1; ++r) {
                const auto row = std::span<const uint8_t>(data + base + r * rb, rb);
                y[r] = tokshit::kernels::dot_row(t.quant, row, rb, x, act, in_dim,
                                                 scratch);
              }
            }
          });
        }
        for (std::thread& th : threads) th.join();
        const double dt =
            std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
        std::printf("  threads it%zu: %.2f GB/s\n", it, nbytes / dt / 1e9);
      }
    }

    for (std::size_t rpt : rows_list) {
      for (std::size_t it = 0; it < iters; ++it) {
        const bool tail = std::getenv("PB_TAIL") != nullptr;
        Seg seg;
        seg.bytes = tail ? std::span<const uint8_t>(data + base, bytes.size() - base)
                         : std::span<const uint8_t>(data + base, out_dim * rb);
        seg.rb = rb;
        seg.quant = t.quant;
        seg.in_dim = in_dim;
        seg.x = x.data();
        seg.act = &act;
        seg.out = y.data();
        seg.rows = out_dim;
        const auto f0 = tokshit::cpu_freqs();
        const auto pf0 = tokshit::page_faults();
        const uint64_t io0 = tokshit::io_read_bytes();
        const auto t0 = tokshit::Clock::now();
        run_pool(seg, rpt);
        const double dt =
            std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
        const auto f1 = tokshit::cpu_freqs();
        const auto pf1 = tokshit::page_faults();
        const uint64_t io1 = tokshit::io_read_bytes();
        totals[it].time += dt;
        totals[it].bytes += nbytes;
        totals[it].minflt += pf1.first - pf0.first;
        totals[it].majflt += pf1.second - pf0.second;
        totals[it].io += io1 - io0;
        std::printf("  pool rpt=%zu it%zu: %.2f GB/s before=", rpt, it,
                    nbytes / dt / 1e9);
        for (uint32_t f : f0) std::printf("%u ", f);
        std::printf("after=");
        for (uint32_t f : f1) std::printf("%u ", f);
        std::fputc('\n', stdout);
      }
    }
    float chk = 0.0F;
    for (std::size_t i = 0; i < out_dim; i += 97) chk += y[i];
    std::fprintf(stderr, "  chk=%.1f\n", static_cast<double>(chk));
  }

  for (std::size_t it = 0; it < totals.size(); ++it) {
    const Total& t = totals[it];
    std::fprintf(stderr,
                 "TOTAL it%zu: %.0f MB in %.3fs = %.2f GB/s minflt=%llu "
                 "majflt=%llu io=%.2f GB rss=%llu MB\n",
                 it, t.bytes / 1e6, t.time,
                 t.bytes / std::max(t.time, 1e-9) / 1e9,
                 static_cast<unsigned long long>(t.minflt),
                 static_cast<unsigned long long>(t.majflt),
                 static_cast<double>(t.io) / 1e9,
                 static_cast<unsigned long long>(tokshit::rss_mb()));
  }
  tokshit::pool::set_active(false);
  return 0;
}
