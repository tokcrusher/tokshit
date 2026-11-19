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

}  // namespace tokshit::spec
