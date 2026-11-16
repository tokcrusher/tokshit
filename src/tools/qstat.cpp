// GGUF quantization statistics: bytes and tensor counts per quant type and
// per model role.

#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokshit/gguf.hpp"

namespace {

struct Stat {
  uint64_t bytes = 0;
  uint64_t count = 0;
};

const char* role_of(std::string_view name) {
  if (name.starts_with("token_embd")) return "embd";
  if (name.starts_with("output")) return "output";
  if (name.find("ffn_") != std::string_view::npos) return "expert";
  if (name.find("attn_") != std::string_view::npos) return "attn";
  return "other";
}

template <class Map>
std::vector<std::pair<std::string, Stat>> sorted(const Map& map) {
  std::vector<std::pair<std::string, Stat>> out;
  out.reserve(map.size());
  for (const auto& entry : map) out.emplace_back(entry.first, entry.second);
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    if (a.second.bytes != b.second.bytes) return a.second.bytes > b.second.bytes;
    return a.first < b.first;
  });
  return out;
}

void print_group(const std::vector<std::pair<std::string, Stat>>& rows,
                 uint64_t total) {
  for (const auto& [name, stat] : rows) {
    const double gb = static_cast<double>(stat.bytes) / 1e9;
    const double pct =
        total == 0 ? 0.0
                   : static_cast<double>(stat.bytes) * 100.0 /
                         static_cast<double>(total);
    std::printf("%-12s %8.3f GB %5.1f%%  n=%llu\n", name.c_str(), gb, pct,
                static_cast<unsigned long long>(stat.count));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: qstat <gguf>\n");
    return 2;
  }

  auto opened = tokshit::Gguf::open(argv[1]);
  if (!opened.ok()) {
    std::fprintf(stderr, "gguf: %s: %s\n",
                 std::string(tokshit::to_string(opened.error().code)).c_str(),
                 opened.error().message.c_str());
    return 1;
  }
  const tokshit::Gguf& g = opened.value();

  if (argc > 2 && std::string_view(argv[2]) == "--list") {
    const std::string_view filter = argc > 3 ? argv[3] : std::string_view();
    for (const tokshit::GgufTensor& t : g.tensors()) {
      if (!filter.empty() && t.name.find(filter) == std::string::npos) {
        continue;
      }
      std::printf("%-40s %-6s", t.name.c_str(),
                  std::string(tokshit::quant_name(t.quant)).c_str());
      for (std::size_t i = 0; i < t.dims.size(); ++i) {
        std::printf(" %llu", static_cast<unsigned long long>(t.dims[i]));
      }
      std::printf("  %.1f KB\n", static_cast<double>(t.byte_len) / 1024.0);
    }
    return 0;
  }

  std::unordered_map<std::string, Stat> by_quant;
  std::unordered_map<std::string, Stat> by_role;
  uint64_t total = 0;

  for (const tokshit::GgufTensor& t : g.tensors()) {
    Stat& q = by_quant[std::string(tokshit::quant_name(t.quant))];
    q.bytes += t.byte_len;
    q.count += 1;

    Stat& r = by_role[role_of(t.name)];
    r.bytes += t.byte_len;
    r.count += 1;

    total += t.byte_len;
  }

  std::printf("total %.2f GB\n", static_cast<double>(total) / 1e9);
  print_group(sorted(by_quant), total);
  print_group(sorted(by_role), total);
  return 0;
}
