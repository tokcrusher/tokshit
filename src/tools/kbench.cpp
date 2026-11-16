// Microbenchmark of the matvec kernels on real tensors.
// usage: kbench <gguf> [iters] <tensor>...

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tokshit/gguf.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/pool.hpp"
#include "tokshit/quant.hpp"

namespace {

struct Job {
  const uint8_t* data = nullptr;
  std::size_t rb = 0;
  tokshit::QuantType quant = tokshit::QuantType::F32;
  std::size_t in_dim = 0;
  const float* x = nullptr;
  const tokshit::kernels::ActQ8* act = nullptr;
  float* out = nullptr;
  std::size_t rows = 0;
  std::size_t chunk = 0;
};

void run_task(void* ctx, std::size_t task) {
  const Job& job = *static_cast<const Job*>(ctx);
  const std::size_t r0 = task * job.chunk;
  const std::size_t r1 = std::min(r0 + job.chunk, job.rows);
  std::vector<float> scratch;
  const std::size_t nrows = r1 - r0;
  if (nrows == 0) return;
  tokshit::kernels::dot_rows(
      job.quant, std::span<const uint8_t>(job.data + r0 * job.rb,
                                          nrows * job.rb),
      job.rb, nrows, std::span<const float>(job.x, job.in_dim), *job.act,
      job.in_dim, job.out + r0, scratch);
}

}  // namespace

int main(int argc, char** argv) {
  std::string path = argc > 1
                         ? argv[1]
                         : "/data/data/com.termux/files/home/models/"
                           "flint-UD-Q3_K_XL.gguf";
  std::size_t iters = 3;
  if (argc > 2) iters = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));

  std::vector<std::string> names;
  for (int i = 3; i < argc; ++i) names.emplace_back(argv[i]);
  if (names.empty()) {
    names = {"output.weight", "blk.0.attn_q.weight",
             "blk.2.ffn_gate_exps.weight", "blk.14.ffn_down_exps.weight"};
  }

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
  const uint8_t* base_data = reinterpret_cast<const uint8_t*>(bytes.data());
  const std::size_t base_off = static_cast<std::size_t>(gguf.data_offset());

  const std::size_t workers = tokshit::pool::TaskPool::instance().workers();
  std::fprintf(stderr, "pool workers = %zu\n", workers);
  tokshit::pool::set_active(true);

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

    for (std::size_t it = 0; it < iters; ++it) {
      const auto t0 = tokshit::Clock::now();
      const std::size_t chunk = std::max<std::size_t>(out_dim / std::max<std::size_t>(workers, 1), 64);
      Job job;
      job.data = base_data + base;
      job.rb = rb;
      job.quant = t.quant;
      job.in_dim = in_dim;
      job.x = x.data();
      job.act = &act;
      job.out = y.data();
      job.rows = out_dim;
      job.chunk = chunk;
      const std::size_t tasks = (out_dim + chunk - 1) / chunk;
      tokshit::pool::run_tasks(tasks, &run_task, &job);
      const double dt =
          std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
      const double nbytes = static_cast<double>(out_dim * rb);
      float chk = 0.0F;
      for (std::size_t i = 0; i < out_dim; i += 97) chk += y[i];
      std::printf("%s %s %zux%zu %.1f MB  it%zu: %.2f GB/s  %.1f Gweight/s  "
                  "chk=%.1f\n",
                  name.c_str(), std::string(tokshit::quant_name(t.quant)).c_str(),
                  out_dim, in_dim, nbytes / 1e6, it, nbytes / dt / 1e9,
                  static_cast<double>(out_dim * in_dim) / dt / 1e9,
                  static_cast<double>(chk));
    }
  }
  tokshit::pool::set_active(false);
  return 0;
}
