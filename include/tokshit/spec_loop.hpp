#pragma once

#include <cstddef>
#include <string>

#include "tokshit/model.hpp"
#include "tokshit/tokenizer.hpp"

namespace tokshit::spec {

// спекулятивный проход одной моделью (SS-MoE-стиль): батч-префилл, потом
// раунды draft → verify → accept → откат kv. только жадный семплинг —
// lossless гарантию держим: выход при жадном семплинге обязан совпасть
// linear декодом байт в байт (проверяется тестами)
//
// env knobs, читаются внутри:
//   TOKSHIT_SPEC_K     — сколько черновиков за раунд, 0/1 = выключено (2)
//   TOKSHIT_DRAFT_TOPK — экспертов у черновика (2)
//   TOKSHIT_SPEC_CTX   — ёмкость kv, токенов (2048)
//   TOKSHIT_SPEC_MAXB  — максимум строк verify-батча, тоже префилл-чанк (8)
void run(Model& model, const BpeTokenizer& tok, const std::string& prompt,
         std::size_t max_new);

// DSpark drafter (Inkling2 target): greedy, lossless. `draft_path` points at
// the dflash GGUF. K is read from TOKSHIT_DSPARK_K (default 3, max 7).
// Returns false when DSpark cannot run (wrong passes, missing draft, ...);
// the caller should then fall back to the normal decode path.
bool run_dspark(Model& model, const BpeTokenizer& tok,
                const std::string& prompt, std::size_t max_new,
                const std::string& draft_path);

}  // namespace tokshit::spec
