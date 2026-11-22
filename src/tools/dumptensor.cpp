// Dump a tensor from a GGUF as little-endian float32 for reference checks.
// usage: dumptensor <model.gguf> <tensor> <out.bin> [--expert N | --row N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/gguf.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s model.gguf tensor out.bin [expert]\n",
                 argv[0]);
    return 2;
  }
  auto g = tokshit::Gguf::open(argv[1]);
  if (!g.ok()) {
    std::fprintf(stderr, "open: %s\n", g.error().message.c_str());
    return 1;
  }
  const auto* t = g.value().tensor(argv[2]).value();
  if (t == nullptr) {
    std::fprintf(stderr, "no tensor %s\n", argv[2]);
    return 1;
  }
  std::vector<uint64_t> dims = t->dims;
  uint64_t numel = 1;
  for (uint64_t d : dims) numel *= d;
  auto data = g.value().read_tensor(argv[2]);
  if (!data.ok()) {
    std::fprintf(stderr, "read: %s\n", data.error().message.c_str());
    return 1;
  }
  const auto spec = *tokshit::block_spec(t->quant);
  auto slice = [&](std::size_t first, std::size_t count) {
    const std::size_t bytes = count / spec.weights * spec.bytes;
    const std::size_t off = first / spec.weights * spec.bytes;
    std::vector<uint8_t> cut(data.value().begin() + off,
                             data.value().begin() + off + bytes);
    data.value() = std::move(cut);
    numel = count;
  };
  if (argc > 5 && dims.size() == 3 && std::strcmp(argv[4], "--expert") == 0) {
    const std::size_t e = std::strtoull(argv[5], nullptr, 10);
    slice(e * dims[0] * dims[1], dims[0] * dims[1]);
  } else if (argc > 5 && dims.size() == 2 &&
             std::strcmp(argv[4], "--row") == 0) {
    const std::size_t r = std::strtoull(argv[5], nullptr, 10);
    slice(r * dims[0], dims[0]);
  }
  std::vector<float> out(numel);
  const auto deq = tokshit::dequant_into(
      t->quant, data.value(), numel,
      std::span<float>(out.data(), out.size()));
  if (!deq.ok()) {
    std::fprintf(stderr, "dequant: %s\n", deq.error().message.c_str());
    return 1;
  }
  FILE* f = std::fopen(argv[3], "wb");
  if (f == nullptr) {
    std::perror("fopen");
    return 1;
  }
  std::fwrite(out.data(), sizeof(float), out.size(), f);
  std::fclose(f);
  std::fprintf(stderr, "%s dims=[", argv[2]);
  for (uint64_t d : dims) std::fprintf(stderr, "%llu ", (unsigned long long)d);
  std::fprintf(stderr, "] quant=%u numel=%llu -> %s\n", (unsigned)t->quant,
               (unsigned long long)numel, argv[3]);
  return 0;
}
