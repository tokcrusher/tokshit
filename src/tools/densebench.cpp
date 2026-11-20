// Диагностика dense-пути: чистый поток hot-байтов, полный matvec и
// гистограмма времени по размеру тензора.
// usage: densebench <model.gguf> [passes]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"
#include "tokshit/pool.hpp"

namespace {

struct Entry {
  const char* role = "";
  std::size_t layer = 0;
  const tokshit::Meta* meta = nullptr;
};

struct Bucket {
  const char* name = "";
  std::size_t count = 0;
  uint64_t bytes = 0;
  double ns = 0.0;
};

const char* bucket_for(uint64_t bytes) {
  if (bytes < 1024ULL * 1024ULL) return "<1 MB";
  if (bytes < 4ULL * 1024ULL * 1024ULL) return "1-4 MB";
  if (bytes < 16ULL * 1024ULL * 1024ULL) return "4-16 MB";
  return ">=16 MB";
}

void collect(const tokshit::Model& model, std::vector<Entry>& out) {
  const auto& layers = model.layers();
  for (std::size_t l = 0; l < layers.size(); ++l) {
    const auto& lt = layers[l];
    out.push_back({"q", l, &lt.attn_q});
    out.push_back({"k", l, &lt.attn_k});
    out.push_back({"v", l, &lt.attn_v});
    out.push_back({"o", l, &lt.attn_o});
    if (lt.ffn_gate) out.push_back({"dense_gate", l, &*lt.ffn_gate});
    if (lt.ffn_up) out.push_back({"dense_up", l, &*lt.ffn_up});
    if (lt.ffn_down) out.push_back({"dense_down", l, &*lt.ffn_down});
    if (lt.gate_inp) out.push_back({"router", l, &*lt.gate_inp});
    if (lt.gate_shexp) out.push_back({"shexp_gate", l, &*lt.gate_shexp});
    if (lt.up_shexp) out.push_back({"shexp_up", l, &*lt.up_shexp});
    if (lt.down_shexp) out.push_back({"shexp_down", l, &*lt.down_shexp});
  }
  out.push_back({"out", layers.size(), &model.out_meta()});
}

void histogram(tokshit::Model& model, std::size_t passes) {
  std::vector<Entry> entries;
  collect(model, entries);
  std::size_t max_in = 1;
  std::size_t max_out = 1;
  for (const Entry& e : entries) {
    max_in = std::max(max_in, e.meta->in_dim);
    max_out = std::max(max_out, e.meta->out_dim);
  }
  std::vector<float> x(max_in, 0.01F);
  std::vector<float> y(max_out, 0.0F);

  std::vector<Bucket> buckets = {
      {"<1 MB", 0, 0, 0.0},
      {"1-4 MB", 0, 0, 0.0},
      {"4-16 MB", 0, 0, 0.0},
      {">=16 MB", 0, 0, 0.0},
  };
  struct RoleBucket {
    const char* role = "";
    std::size_t count = 0;
    uint64_t bytes = 0;
    double ns = 0.0;
  };
  std::vector<RoleBucket> roles;
  struct Slow {
    double ns = 0.0;
    const char* role = "";
    std::size_t layer = 0;
    uint64_t bytes = 0;
  };
  std::vector<Slow> slow;
  slow.reserve(entries.size());

  // Два полных прогрева: первый проход после загрузки идёт на низких частотах.
  for (int w = 0; w < 2; ++w) {
    for (const Entry& e : entries) {
      const auto xi = std::span<const float>(x.data(), e.meta->in_dim);
      auto yi = std::span<float>(y.data(), e.meta->out_dim);
      model.matvec(*e.meta, xi, yi);
    }
  }

  for (const Entry& e : entries) {
    const uint64_t bytes =
        static_cast<uint64_t>(e.meta->out_dim) * e.meta->rb;
    const auto xi = std::span<const float>(x.data(), e.meta->in_dim);
    auto yi = std::span<float>(y.data(), e.meta->out_dim);
    std::vector<double> samples;
    samples.reserve(passes);
    for (std::size_t p = 0; p < passes; ++p) {
      const auto t0 = std::chrono::steady_clock::now();
      model.matvec(*e.meta, xi, yi);
      samples.push_back(std::chrono::duration<double, std::nano>(
                            std::chrono::steady_clock::now() - t0)
                            .count());
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];

    const char* size_name = bucket_for(bytes);
    for (Bucket& b : buckets) {
      if (std::string_view(b.name) == size_name) {
        ++b.count;
        b.bytes += bytes;
        b.ns += median;
        break;
      }
    }
    const char* role = e.role;
    auto it = std::find_if(roles.begin(), roles.end(),
                           [&](const RoleBucket& r) {
                             return std::string_view(r.role) == role;
                           });
    if (it == roles.end()) {
      roles.push_back({role, 0, 0, 0.0});
      it = roles.end() - 1;
    }
    ++it->count;
    it->bytes += bytes;
    it->ns += median;
    slow.push_back({median, role, e.layer, bytes});
  }

  double total_ns = 0.0;
  for (const Bucket& b : buckets) total_ns += b.ns;
  std::printf("\nhistogram by tensor size (passes=%zu):\n", passes);
  std::printf("%-8s %5s %10s %8s %10s %7s\n", "bucket", "count", "MB",
              "GB/s", "ms/pass", "time%");
  for (const Bucket& b : buckets) {
    const double gb = static_cast<double>(b.bytes) / 1e9;
    const double sec = b.ns / 1e9;
    std::printf("%-8s %5zu %10.1f %8.2f %10.2f %6.1f%%\n", b.name, b.count,
                static_cast<double>(b.bytes) / 1e6,
                sec > 1e-12 ? gb / sec : 0.0, b.ns / 1e6,
                total_ns > 0.0 ? 100.0 * b.ns / total_ns : 0.0);
  }

  std::printf("\nhistogram by role:\n");
  std::printf("%-12s %5s %10s %8s %10s %7s\n", "role", "count", "MB", "GB/s",
              "ms/pass", "time%");
  for (const RoleBucket& r : roles) {
    const double gb = static_cast<double>(r.bytes) / 1e9;
    const double sec = r.ns / 1e9;
    std::printf("%-12s %5zu %10.1f %8.2f %10.2f %6.1f%%\n", r.role, r.count,
                static_cast<double>(r.bytes) / 1e6,
                sec > 1e-12 ? gb / sec : 0.0, r.ns / 1e6,
                total_ns > 0.0 ? 100.0 * r.ns / total_ns : 0.0);
  }
  std::sort(slow.begin(), slow.end(),
            [](const Slow& a, const Slow& b) { return a.ns > b.ns; });
  std::printf("\nslowest tensors (median single call):\n");
  for (std::size_t i = 0; i < std::min<std::size_t>(slow.size(), 12); ++i) {
    const Slow& s = slow[i];
    std::printf("  layer %-3zu %-12s %7.1f us %8.2f MB %6.2f GB/s\n",
                s.layer, s.role, s.ns / 1e3,
                static_cast<double>(s.bytes) / 1e6,
                s.ns > 0.0 ? static_cast<double>(s.bytes) / s.ns : 0.0);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path =
      argc > 1 ? argv[1]
               : "/data/data/com.termux/files/home/models/"
                 "flint-UD-Q3_K_XL.gguf";
  const std::size_t passes =
      argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10))
               : 3;

  tokshit::ModelOptions options;
  options.expert_cache_mb = 0;
  options.hot = true;
  auto loaded = tokshit::Model::load(path, options);
  if (!loaded.ok()) {
    std::fprintf(stderr, "load: %s: %s\n",
                 tokshit::to_string(loaded.error().code).data(),
                 loaded.error().message.c_str());
    return 1;
  }
  tokshit::Model model = std::move(loaded.value());
  model.collect_hot();

  tokshit::pool::set_active(true);
  std::vector<double> reads;
  std::vector<double> denses;
  reads.reserve(10);
  denses.reserve(10);
  for (int i = 0; i < 10; ++i) {
    reads.push_back(model.bench_hot_read(1));
    denses.push_back(model.bench_dense_matvec(1));
  }
  std::sort(reads.begin(), reads.end());
  std::sort(denses.begin(), denses.end());
  const double read = reads[reads.size() / 2];
  const double dense = denses[denses.size() / 2];

  const double gb = static_cast<double>(model.hot_bytes()) / 1e9;
  const double read_ms = read > 1e-9 ? gb / read * 1e3 : 0.0;
  const double dense_ms = dense > 1e-9 ? gb / dense * 1e3 : 0.0;

  std::printf("hot_bytes_gb=%.3f hot_on_ahb=%d passes=%zu rss_mb=%llu\n", gb,
              model.hot_on_ahb() ? 1 : 0, passes,
              static_cast<unsigned long long>(tokshit::rss_mb()));
  std::printf("pure_read_gbps=%.2f (min %.2f max %.2f) pass_ms=%.1f\n", read,
              reads.front(), reads.back(), read_ms);
  std::printf("dense_matvec_gbps=%.2f (min %.2f max %.2f) pass_ms=%.1f\n",
              dense, denses.front(), denses.back(), dense_ms);

  histogram(model, passes);
  tokshit::pool::set_active(false);
  return 0;
}
