#include "cli.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"
#include "tokshit/quant.hpp"
#include "tokshit/reasoning.hpp"
#include "tokshit/sampling.hpp"
#include "tokshit/special_tokens.hpp"
#include "tokshit/tokenizer.hpp"

namespace tokshit::cli {

DecodeResult decode(Model& model, const BpeTokenizer& tok, KVCache& kv,
                    std::size_t& pos, std::vector<float>& logits,
                    const SampleParams& params, Rng& rng, int reasoning_budget,
                    FILE* dump, const DecodeOptions& options) {
  DecodeResult result;
  Utf8Buffer utf8;
  ReasoningBudget reasoning(reasoning_budget);
  const auto t0 = Clock::now();

  // Стриминговый стоп-фильтр: TOKSHIT_STOP_AT / _AT2 / _AT3 (например "\n",
  // "Q:") обрывают ход, не выводя сам маркер. Хвост длиной maxlen-1 держим,
  // чтобы маркер не протёк в stdout по частям. Ведущий маркер игнорируется,
  // иначе ответ был бы пустым.
  //
  // TOKSHIT_STOP_SENTENCE=1 — стоп на первом конце предложения ('.', '!', '?'),
  // когда к этому моменту готово к выводу >= 8 непустых символов; терминатор
  // не выводим. Упрощения, чтобы не тащить полноценный sentence splitter:
  //   * '.' после цифры или перед цифрой не считается концом (1., 3.14, .5);
  //   * '.' в сокращении/инициале не считается: одиночная буква (J.) или
  //     слово из короткого списка ниже (Mr., Dr., e.g., i.e., etc.);
  //   * '!' и '?' считаются концом без доп. проверок.
  // Хвост в 1 байт держим, чтобы видеть символ после терминатора (для цифр).
  // Порядок не меняем: сначала стоп-маркеры, потом конец предложения, так что
  // уже целиком лежащий в буфере маркер по-прежнему срабатывает первым.
  // ВАЖНО: это строго opt-in. Переменная не задана (или =0) — фильтр выключен,
  // ответ идёт до EOG / max_new / стоп-маркеров, как и раньше. Включает его
  // только явный TOKSHIT_STOP_SENTENCE=1 (voice script выставляет сам).
  std::vector<std::string> stops;
  auto add_stop = [&](const char* sv) {
    if (sv == nullptr || *sv == '\0') return;
    std::string s;
    for (std::size_t i = 0; sv[i] != '\0'; ++i) {
      if (sv[i] == '\\' && sv[i + 1] == 'n') {
        s.push_back('\n');
        ++i;
      } else {
        s.push_back(sv[i]);
      }
    }
    if (!s.empty()) stops.push_back(std::move(s));
  };
  add_stop(std::getenv("TOKSHIT_STOP_AT"));
  add_stop(std::getenv("TOKSHIT_STOP_AT2"));
  add_stop(std::getenv("TOKSHIT_STOP_AT3"));
  const bool stop_sentence = [] {
    const char* v = std::getenv("TOKSHIT_STOP_SENTENCE");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
  }();
  std::size_t max_stop = 0;
  for (const auto& s : stops) max_stop = std::max(max_stop, s.size());
  std::size_t keep = max_stop > 0 ? max_stop - 1 : 0;
  if (stop_sentence) keep = std::max<std::size_t>(keep, 1);
  std::string held;
  std::size_t non_space = 0;  // непустых символов уже выведено
  bool stopped = false;
  auto emit = [&](std::string_view s) {
    if (s.empty()) return;
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
    result.text.append(s);
    for (char c : s) {
      if (!std::isspace(static_cast<unsigned char>(c))) ++non_space;
    }
  };

  for (;;) {
    if (options.check_interrupt && g_interrupt != 0) {
      result.interrupted = true;
      break;
    }
    const uint32_t sampled = sample_utf8_safe(logits, params, tok, utf8, rng);
    // Бюджет проверяется до печати и forward: принудительный </think/>
    // проходит тот же путь, что и обычный токен.
    const uint32_t next = reasoning.constrain(sampled);
    if (tok.is_eog(next)) break;
    result.ids.push_back(next);

    // <think/> и </think/> не текст ответа; thinking-блок скрывается целиком.
    // У inkling2 все спецтокены (id >= 199998) — разметка, кроме end_message,
    // который печатаем как перевод строки.
    const bool ink_special = model.is_inkling2() && next >= 199998;
    const bool control =
        next == kThinkOpen || next == kThinkClose || ink_special;
    std::string chunk;
    if (ink_special) {
      utf8 = Utf8Buffer{};
      if (next == 200010) chunk = "\n";
    } else if (control) {
      utf8 = Utf8Buffer{};
    } else {
      chunk = utf8.push(tok.token_bytes(next));
    }
    if (!chunk.empty()) {
      if (stops.empty() && !stop_sentence) {
        emit(chunk);
      } else {
        held.append(chunk);
        bool stop_now = false;
        for (;;) {
          std::size_t best = std::string::npos;
          std::size_t best_len = 0;
          for (const auto& s : stops) {
            const std::size_t p = held.find(s);
            if (p != std::string::npos && p < best) {
              best = p;
              best_len = s.size();
            }
          }
          if (best == std::string::npos) break;
          if (best == 0 && result.text.empty()) {
            // ведущий маркер (обычно \n) не должен давать пустой ответ
            held.erase(0, best_len);
            continue;
          }
          emit(std::string_view(held).substr(0, best));
          held.clear();
          stop_now = true;
          break;
        }
        // Конец предложения ищем только в той части held, что уже готова к
        // выводу (p < n); символ после терминатора ещё лежит в хвосте keep.
        if (!stop_now && stop_sentence) {
          const std::size_t n = held.size() > keep ? held.size() - keep : 0;
          for (std::size_t p = 0; p < n; ++p) {
            const char c = held[p];
            if (c != '.' && c != '!' && c != '?') continue;
            std::size_t ns = non_space;
            for (std::size_t k = 0; k < p; ++k) {
              if (!std::isspace(static_cast<unsigned char>(held[k]))) ++ns;
            }
            if (ns < 8) continue;  // слишком коротко: ждём следующий термин.
            if (c == '.') {
              // Контекст для точки: последние 64 байта выведенного + held[:p+2].
              std::string ctx;
              if (result.text.size() > 64) {
                ctx.assign(result.text, result.text.size() - 64, 64);
              } else {
                ctx = result.text;
              }
              const std::size_t q = ctx.size() + p;
              ctx.append(held, 0, std::min(p + 2, held.size()));
              const char prev = q > 0 ? ctx[q - 1] : '\0';
              const char next = q + 1 < ctx.size() ? ctx[q + 1] : '\0';
              const auto digit = [](char ch) {
                return ch >= '0' && ch <= '9';
              };
              if (digit(prev) || digit(next)) continue;  // 1. / 3.14 / .5
              std::size_t b = q;
              while (b > 0 &&
                     std::isalpha(static_cast<unsigned char>(ctx[b - 1]))) {
                --b;
              }
              std::string word = ctx.substr(b, q - b);
              std::transform(word.begin(), word.end(), word.begin(),
                             [](unsigned char ch) {
                               return static_cast<char>(std::tolower(ch));
                             });
              static constexpr const char* kAbbrev[] = {
                  "mr",  "mrs", "ms",   "dr", "prof", "sr", "jr", "st",
                  "vs",  "etc", "fig",  "no", "vol",  "inc", "ltd", "dept"};
              bool abbrev = word.size() == 1;  // инициал: J.
              for (const char* a : kAbbrev) {
                if (word == a) {
                  abbrev = true;
                  break;
                }
              }
              if (abbrev) continue;  // Mr. / e.g. / i.e. / U.S.
            }
            emit(std::string_view(held).substr(0, p));
            held.clear();
            stop_now = true;
            break;
          }
        }
        if (stop_now) {
          stopped = true;
        } else if (held.size() > keep) {
          const std::size_t n = held.size() - keep;
          emit(std::string_view(held).substr(0, n));
          held.erase(0, n);
        }
      }
    }
    ++result.generated;
    if (stopped) break;
    if (result.generated >= options.max_new) break;

    if (options.pin_every) {
      if (const char* pv = std::getenv("TOKSHIT_PIN_EVERY");
          pv != nullptr && *pv != '\0') {
        const std::size_t pin_every =
            static_cast<std::size_t>(std::strtoull(pv, nullptr, 10));
        if (pin_every != 0 && result.generated % pin_every == 0) {
          const char* kv_pin = std::getenv("TOKSHIT_PIN_EXPERTS");
          const std::size_t pin_k =
              kv_pin != nullptr
                  ? static_cast<std::size_t>(std::strtoull(kv_pin, nullptr, 10))
                  : 200;
          model.pin_hot_experts(pin_k);
        }
      }
    }

    const auto t_tok = Clock::now();
    const uint64_t io_t0 = options.token_debug ? io_read_bytes() : 0;
    const auto pf_t0 = options.token_debug ? page_faults()
                                           : std::pair<uint64_t, uint64_t>{0, 0};
    const uint64_t ld_t0 =
        options.token_debug ? metrics().t_load.load() : 0;
    const uint64_t mm_t0 =
        options.token_debug ? metrics().t_expert_mm.load() : 0;
    const uint64_t mb_t0 =
        options.token_debug ? metrics().c_mm_bytes.load() : 0;
    const uint64_t fb_t0 =
        options.token_debug ? metrics().c_dot_fallback.load() : 0;
    std::array<uint64_t, 64> fbq_t0{};
    if (options.token_debug) {
      for (std::size_t i = 0; i < fbq_t0.size(); ++i) {
        fbq_t0[i] = metrics().c_fallback_q[i].load(std::memory_order_relaxed);
      }
    }
    const uint64_t dw_t0 = metrics().t_down_wait.load();
    const uint64_t hit_t0 = metrics().c_expert_hit.load();
    const uint64_t miss_t0 = metrics().c_expert_miss.load();

    model.token_logits(next, kv, pos, logits);
    dump_logits(dump, logits);
    ++pos;
    reasoning.observe(next);

    if (options.token_debug) {
      const auto& m = metrics();
      const auto pf = page_faults();
      const double dt =
          std::chrono::duration<double>(Clock::now() - t_tok).count();
      std::fprintf(
          stderr,
          "[tok %zu] %.0fms io=%.1fMB w=%.1fMB fb=%llu load=%.0fms exmm=%.0fms "
          "dw=%.0fms minf=%llu majf=%llu rss=%lluMB hit=%llu miss=%llu\n",
          result.generated, dt * 1e3,
          static_cast<double>(io_read_bytes() - io_t0) / 1e6,
          static_cast<double>(m.c_mm_bytes.load() - mb_t0) / 1e6,
          static_cast<unsigned long long>(m.c_dot_fallback.load() - fb_t0),
          static_cast<double>(m.t_load.load() - ld_t0) / 1e6,
          static_cast<double>(m.t_expert_mm.load() - mm_t0) / 1e6,
          static_cast<double>(m.t_down_wait.load() - dw_t0) / 1e6,
          static_cast<unsigned long long>(pf.first - pf_t0.first),
          static_cast<unsigned long long>(pf.second - pf_t0.second),
          static_cast<unsigned long long>(rss_mb()),
          static_cast<unsigned long long>(m.c_expert_hit.load() - hit_t0),
          static_cast<unsigned long long>(m.c_expert_miss.load() - miss_t0));
      std::string fbs;
      for (std::size_t qi = 0; qi < fbq_t0.size(); ++qi) {
        const uint64_t d =
            m.c_fallback_q[qi].load(std::memory_order_relaxed) - fbq_t0[qi];
        if (d == 0) continue;
        fbs += ' ';
        fbs += quant_name(static_cast<QuantType>(qi));
        fbs += '=';
        fbs += std::to_string(d);
      }
      if (!fbs.empty()) std::fprintf(stderr, "     fbq:%s\n", fbs.c_str());
    }
  }

  std::string tail = reasoning.in_think() ? std::string{} : utf8.flush();
  if (!stopped) tail += held;
  if (!tail.empty()) {
    std::fwrite(tail.data(), 1, tail.size(), stdout);
    result.text.append(tail);
  }
  result.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  return result;
}

}  // namespace tokshit::cli
