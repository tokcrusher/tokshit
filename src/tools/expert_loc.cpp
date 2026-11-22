// Standalone locality probe: load the model, run N decode steps, then print
// the per-layer router top-k pass overlaps (see expert_loc_report()).
// usage: expert_loc <gguf> [n_tokens] [token_id]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tokshit/kv.hpp"
#include "tokshit/model.hpp"
#include "tokshit/pool.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <gguf> [n_tokens=4] [token_id=1]\n",
                 argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const std::size_t n_tok = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 4;
  const std::size_t tok0 = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1;
  if (n_tok == 0) return 2;

  // собирать локальность по умолчанию в этом инструменте
  // collect locality by default in this tool
  if (std::getenv("TOKSHIT_EXPERT_LOCALITY") == nullptr) {
    setenv("TOKSHIT_EXPERT_LOCALITY", "1", 0);
  }

  auto loaded = tokshit::Model::load(path);
  if (!loaded.ok()) {
    std::fprintf(stderr, "model: %s\n", loaded.error().message.c_str());
    return 1;
  }
  tokshit::Model& m = loaded.value();
  tokshit::KVCache kv(m.n_layers());
  std::vector<float> logits(m.vocab(), 0.0F);
  tokshit::pool::set_active(true);
  for (std::size_t p = 0; p < n_tok; ++p) {
    m.token_logits(static_cast<uint32_t>(tok0), kv, p, logits);
  }
  tokshit::pool::set_active(false);
  tokshit::expert_loc_report();
  return 0;
}
