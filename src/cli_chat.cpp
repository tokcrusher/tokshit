#include "cli.hpp"

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include <unistd.h>

#include "env.hpp"
#include "tokshit/inkling2.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"
#include "tokshit/pool.hpp"
#include "tokshit/reasoning.hpp"
#include "tokshit/sampling.hpp"
#include "tokshit/special_tokens.hpp"
#include "tokshit/tokenizer.hpp"
#include "tokshit/tui.hpp"

namespace tokshit::cli {

// Inline REPL: без ncurses и без полного экрана. В TTY — цветные роли,
// стриминг по токену, строка статуса; в пайпе — обычный построчный читатель
// без ANSI. Управление: /help, /clear, /stats, /temp, /topk, /system, /save.
struct ChatSession {
  tokshit::KVCache kv;
  std::size_t pos = 0;
  tokshit::SampleParams params;
  std::string system_prompt;
  std::string transcript;
  std::uint64_t total_tokens = 0;
  std::uint64_t last_tokens = 0;
  std::uint64_t turns = 0;
  double total_seconds = 0.0;
  double last_prefill_s = 0.0;
  double last_tok_per_s = 0.0;
  std::uint64_t cache_budget_mb = 1024;
  int reasoning_budget = -1;
};

void print_chat_status(const ChatSession& s, bool color) {
  tokshit::tui::Status st;
  st.tok_per_s = s.last_tok_per_s;
  st.gen_tokens = s.last_tokens;
  st.cache_hit = tokshit::metrics().c_expert_hit.load();
  st.cache_miss = tokshit::metrics().c_expert_miss.load();
  st.rss_mb = tokshit::rss_mb();
  st.mem_avail_mb = tokshit::tui::mem_available_mb();
  st.ctx_pos = s.pos;
  st.cache_budget_mb = s.cache_budget_mb;
  st.spec = false;  // чат идёт обычным декодом
  const std::string text = tokshit::tui::format_status(st, color);
  std::fprintf(stderr, "%s\n", text.c_str());
}

void print_chat_stats(const ChatSession& s, bool color) {
  tokshit::tui::Stats stats;
  const auto& m = tokshit::metrics();
  const uint64_t passes = m.c_expert_passes.load();
  stats.status.tok_per_s = s.last_tok_per_s;
  stats.status.gen_tokens = s.last_tokens;
  stats.status.cache_hit = m.c_expert_hit.load();
  stats.status.cache_miss = m.c_expert_miss.load();
  stats.status.rss_mb = tokshit::rss_mb();
  stats.status.mem_avail_mb = tokshit::tui::mem_available_mb();
  stats.status.ctx_pos = s.pos;
  stats.status.cache_budget_mb = s.cache_budget_mb;
  stats.turns = s.turns;
  stats.total_tokens = s.total_tokens;
  stats.avg_tok_per_s =
      s.total_seconds > 1e-9 ? static_cast<double>(s.total_tokens) / s.total_seconds
                             : 0.0;
  stats.last_prefill_s = s.last_prefill_s;
  stats.union_per_pass = passes != 0
                             ? static_cast<double>(m.c_expert_union.load()) /
                                   static_cast<double>(passes)
                             : 0.0;
  stats.temperature = s.params.temperature;
  stats.top_k = s.params.top_k;
  stats.top_p = s.params.top_p;
  stats.reasoning_budget = s.reasoning_budget;
  stats.has_system = !s.system_prompt.empty();
  stats.system_prompt = s.system_prompt;
  const std::string text = tokshit::tui::format_stats(stats, color);
  std::fwrite(text.data(), 1, text.size(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

void clear_chat_context(ChatSession& s) {
  for (auto& layer : s.kv) layer.clear();
  s.pos = 0;
}

// Литеральные \n -> реальные переводы строк. Нужно и для строк stdin
// (TOKSHIT_CHAT_UNESCAPE), и для стартового префикса.
std::string unescape_newlines(std::string_view in) {
  std::string out;
  out.reserve(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == 'n') {
      out.push_back('\n');
      ++i;
    } else {
      out.push_back(in[i]);
    }
  }
  return out;
}

void chat_loop(tokshit::Model& model, const tokshit::BpeTokenizer& tok,
               const FreqProfileCfg& freq_cfg) {
  const bool tty = ::isatty(STDIN_FILENO) != 0 && ::isatty(STDOUT_FILENO) != 0;
  const bool err_tty = ::isatty(STDERR_FILENO) != 0;
  // ширина терминала для живых строк: на узких экранах хвост индикатора
  // отрезается, чтобы \r-перерисовка не ломалась
  const std::size_t width = tokshit::tui::terminal_width(STDERR_FILENO);

  install_sigint();

  ChatSession s;
  s.kv.assign(model.n_layers(), std::vector<float>{});
  s.params = sample_params_from_env();
  if (const char* sp = std::getenv("TOKSHIT_SYSTEM_PROMPT");
      sp != nullptr) {
    s.system_prompt = sp;
  }
  s.cache_budget_mb = tokshit::env_u64("TOKSHIT_EXPERT_CACHE_MB", 1024);
  s.reasoning_budget = reasoning_budget_from_env();
  tokshit::Rng rng;

  std::vector<float> logits(model.vocab(), 0.0F);
  FILE* dump = nullptr;
  if (const char* v = std::getenv("TOKSHIT_DUMP_LOGITS");
      v != nullptr && *v != '\0') {
    dump = std::fopen(v, "wb");
  }
  // Машинный протокол для внешних клиентов (voice_chat): каждый ответ
  // завершается байтом 0x1e (RS) без перевода строки, а логи остаются в
  // stderr. Позволяет читать stdout потоком и точно знать конец хода.
  const bool chat_protocol = std::getenv("TOKSHIT_CHAT_PROTOCOL") != nullptr;
  std::size_t chat_max_new = static_cast<std::size_t>(-1);
  if (const char* v = std::getenv("TOKSHIT_CHAT_MAX_NEW");
      v != nullptr && *v != '\0') {
    const unsigned long long n = std::strtoull(v, nullptr, 10);
    if (n != 0) chat_max_new = static_cast<std::size_t>(n);
  }

  // Стартовый префикс: few-shot/инструкции, которые можно прогреть до
  // первого вопроса. Для голоса это убирает 15–20 токенов из prefill
  // первого хода; прогрев идёт, пока пользователь говорит/идёт ASR.
  if (const char* pv = std::getenv("TOKSHIT_CHAT_PREFIX");
      pv != nullptr && *pv != '\0' && model.is_inkling2()) {
    const std::string prefix = unescape_newlines(pv);
    const std::vector<uint32_t> ids = tok.encode(prefix);
    if (!ids.empty() && s.pos + ids.size() < model.inkling2()->kv_ctx()) {
      tokshit::pool::set_active(true);
      model.set_prefilling(true);
      const auto t0 = tokshit::Clock::now();
      for (uint32_t id : ids) {
        model.token_logits(id, s.kv, s.pos, logits);
        ++s.pos;
      }
      model.set_prefilling(false);
      tokshit::pool::set_active(false);
      const double dt =
          std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
      std::fprintf(stderr, "[prefix] %zu tok за %.1fs, pos=%zu\n", ids.size(),
                   dt, s.pos);
    }
  }

  if (tty) {
    std::fputs("токшит · интерактивный чат · /help — команды, /exit — выход\n",
               stderr);
  }

  std::string line;
  for (;;) {
    if (tty) {
      std::fputs("\nвы › ", stdout);
      std::fflush(stdout);
    }
    g_interrupt = 0;
    // замечание: ядро собрано с -fno-exceptions. если поток упадёт то
    // libstdc++ позовёт std::terminate, для cli это ок (нормальный выход это
    // eof), просто держим это внутри цикла
    if (!std::getline(std::cin, line)) {
      if (g_interrupt != 0) {
        g_interrupt = 0;
        std::cin.clear();
        if (tty) {
          std::fputs(err_tty ? "^C — /exit или Ctrl-D для выхода\n"
                             : "^C — /exit или Ctrl-D для выхода\n",
                     stderr);
        }
        continue;
      }
      // Ctrl-D/EOF: аккуратный выход, не падаем и не зависаем
      if (tty) {
        std::fputc('\n', stdout);
        std::fputs("пока\n", stderr);
      }
      break;
    }

    const tokshit::tui::Command cmd = tokshit::tui::parse_command(line);
    using Kind = tokshit::tui::CommandKind;
    switch (cmd.kind) {
      case Kind::None:
        continue;
      case Kind::Quit:
        goto done;
      case Kind::Help:
        std::fputs(tokshit::tui::command_help().c_str(), stdout);
        std::fflush(stdout);
        continue;
      case Kind::Clear:
        clear_chat_context(s);
        // utf8-буфер живёт один ход, так что после очистки контекста он
        // уже пуст; в TTY дополнительно чистим экран целиком
        if (tty) {
          std::fputs("\x1b[H\x1b[2J\x1b[3J", stdout);
        }
        std::fputs("контекст очищен, позиция 0\n", stdout);
        std::fflush(stdout);
        continue;
      case Kind::Stats:
        print_chat_stats(s, ::isatty(STDOUT_FILENO) != 0);
        continue;
      case Kind::Reset:
        s.params = tokshit::SampleParams{};
        std::fputs("сэмплинг сброшен: temp 0.80 · top-k 0 · top-p 0.90\n",
                   stdout);
        std::fflush(stdout);
        continue;
      case Kind::Temp:
        if (!cmd.valid || cmd.number < 0.0F || cmd.number > 10.0F) {
          if (cmd.arg.empty()) {
            std::fputs("использование: /temp <0..10> (0 = жадный)\n", stdout);
          } else {
            std::fprintf(stdout,
                         "не понял температуру «%s»: нужно число 0..10 "
                         "(0 = жадный), сейчас %.2f\n",
                         cmd.arg.c_str(),
                         static_cast<double>(s.params.temperature));
          }
        } else {
          s.params.temperature = cmd.number;
          std::fprintf(stdout, "температура: %.2f\n",
                       static_cast<double>(s.params.temperature));
        }
        std::fflush(stdout);
        continue;
      case Kind::TopK:
        if (!cmd.valid || cmd.count > model.vocab()) {
          if (cmd.arg.empty()) {
            std::fputs("использование: /topk <n>, 0 = без отсечения\n", stdout);
          } else {
            std::fprintf(stdout,
                         "не понял top-k «%s»: нужно целое 0..%zu "
                         "(0 = без отсечения)\n",
                         cmd.arg.c_str(), model.vocab());
          }
        } else {
          s.params.top_k = cmd.count;
          std::fprintf(stdout, "top-k: %zu\n", s.params.top_k);
        }
        std::fflush(stdout);
        continue;
      case Kind::TopP:
        if (!cmd.valid || cmd.number < 0.0F || cmd.number > 1.0F) {
          if (cmd.arg.empty()) {
            std::fputs("использование: /topp <0..1>, 1 = выключено\n", stdout);
          } else {
            std::fprintf(stdout,
                         "не понял top-p «%s»: нужно число 0..1 "
                         "(1 = выключено), сейчас %.2f\n",
                         cmd.arg.c_str(),
                         static_cast<double>(s.params.top_p));
          }
        } else {
          s.params.top_p = cmd.number;
          std::fprintf(stdout, "top-p: %.2f\n",
                       static_cast<double>(s.params.top_p));
        }
        std::fflush(stdout);
        continue;
      case Kind::ReasoningBudget:
        if (!cmd.valid) {
          if (cmd.arg.empty()) {
            std::fputs(
                "использование: /reasoning-budget <-1|0|N>, -1 = без лимита\n",
                stdout);
          } else {
            std::fprintf(stdout,
                         "не понял бюджет «%s»: нужно -1, 0 или N>0, "
                         "сейчас %s\n",
                         cmd.arg.c_str(),
                         s.reasoning_budget < 0
                             ? "без лимита"
                             : std::to_string(s.reasoning_budget).c_str());
          }
        } else {
          s.reasoning_budget = cmd.budget;
          if (cmd.budget < 0) {
            std::fputs("reasoning-budget: без лимита\n", stdout);
          } else {
            std::fprintf(stdout, "reasoning-budget: %d\n", cmd.budget);
          }
        }
        std::fflush(stdout);
        continue;
      case Kind::System:
        s.system_prompt = cmd.arg;
        if (s.system_prompt.empty()) {
          std::fputs("системная инструкция убрана\n", stdout);
        } else {
          std::fputs(
              "системная инструкция сохранена; она подставляется в начало "
              "следующих сообщений (для чистого контекста — /clear)\n",
              stdout);
        }
        std::fflush(stdout);
        continue;
      case Kind::Save:
        if (!cmd.valid) {
          std::fputs("использование: /save <файл>\n", stdout);
        } else if (s.transcript.empty()) {
          std::fputs("нечего сохранять: диалог пока пуст\n", stdout);
        } else if (FILE* f = std::fopen(cmd.arg.c_str(), "ab"); f == nullptr) {
          std::fprintf(stdout, "не удалось открыть %s на запись\n",
                       cmd.arg.c_str());
        } else {
          std::fwrite(s.transcript.data(), 1, s.transcript.size(), f);
          std::fclose(f);
          std::fprintf(stdout, "диалог дописан в %s (%zu байт)\n",
                       cmd.arg.c_str(), s.transcript.size());
        }
        std::fflush(stdout);
        continue;
      case Kind::Unknown:
        std::fprintf(stdout,
                     "неизвестная команда /%s; список команд — /help\n",
                     cmd.arg.empty() ? "" : cmd.arg.c_str());
        std::fflush(stdout);
        continue;
      case Kind::Chat:
        break;
    }

    // Для внешних клиентов: одна строка = один ход, переводы строк можно
    // передавать как литеральные \n, чтобы few-shot промпт влезал в строку.
    std::string text = cmd.arg;
    if (const char* ue = std::getenv("TOKSHIT_CHAT_UNESCAPE");
        ue != nullptr && *ue != '\0' && *ue != '0') {
      text = unescape_newlines(text);
    }
    if (text.empty()) continue;

    tokshit::pool::set_active(true);
    std::vector<uint32_t> turn =
        tokshit::cli::build_turn(model, tok, text, s.system_prompt);
    // Многоходовой чат: предыдущий ответ сгенерирован, но его EOG
    // (<|end_message|>) не прогонялся через модель — decode обрывается до
    // форварда. Закрываем сообщение при следующем префилле, иначе шаблон
    // ломается: user идёт сразу после текста ассистента. В raw-режиме
    // разметки нет, вставлять спецтокен нельзя.
    const bool raw_prompt = std::getenv("TOKSHIT_RAW_PROMPT") != nullptr;
    if (!raw_prompt && model.is_inkling2() && s.pos > 0) {
      if (const auto eom = tok.token_id("<|end_message|>");
          eom.has_value()) {
        turn.insert(turn.begin(), *eom);
      }
    }

    const auto tp = tokshit::Clock::now();
    model.set_prefilling(true);
    {
      PrefillBar bar;
      bar.start();
      std::size_t done = 0;
      std::size_t chunk = 0;
      if (model.is_inkling2() && dump == nullptr) {
        const char* v = std::getenv("TOKSHIT_INK_PREFILL_BATCH");
        chunk = (v != nullptr && *v != '\0')
                    ? static_cast<std::size_t>(std::strtoull(v, nullptr, 10))
                    : 16;
        if (chunk < 2) chunk = 0;
      }
      if (chunk != 0 && turn.size() > 1) {
        ::setenv("TOKSHIT_INK_PREFILL_ONLY_LAST", "1", 0);
        auto* ink = model.inkling2();
        const std::size_t v = ink->vocab();
        std::vector<float> outs(chunk * v, 0.0F);
        for (std::size_t i = 0; i < turn.size(); i += chunk) {
          const std::size_t nb = std::min(chunk, turn.size() - i);
          ink->token_logits_batch(
              std::span<const uint32_t>(turn.data() + i, nb), s.pos + i,
              std::span<float>(outs.data(), nb * v));
          std::copy(outs.begin() +
                        static_cast<std::ptrdiff_t>((nb - 1) * v),
                    outs.begin() + static_cast<std::ptrdiff_t>(nb * v),
                    logits.begin());
          done += nb;
          bar.tick(done, turn.size(), width, err_tty);
        }
        s.pos += turn.size();
      } else {
        for (uint32_t id : turn) {
          model.token_logits(id, s.kv, s.pos, logits);
          dump_logits(dump, logits);
          ++s.pos;
          ++done;
          bar.tick(done, turn.size(), width, err_tty);
        }
      }
      bar.finish(err_tty);
    }
    model.set_prefilling(false);
    const double prefill =
        std::chrono::duration<double>(tokshit::Clock::now() - tp).count();
    if (tokshit::prefetch_enabled()) model.pin_hot_experts(64);
    if (const char* v = std::getenv("TOKSHIT_PIN_EXPERTS");
        v != nullptr && *v != '\0') {
      const std::size_t k =
          static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
      if (k != 0) model.pin_hot_experts(k);
    }

    if (tty) {
      std::fputs("токшит: ", stdout);
      std::fflush(stdout);
    }
    DecodeOptions dopt;
    dopt.check_interrupt = true;
    dopt.max_new = chat_max_new;
    const DecodeResult dec =
        decode(model, tok, s.kv, s.pos, logits, s.params, rng,
               s.reasoning_budget, dump, dopt);
    const std::string answer = dec.text;
    const std::size_t n_gen = dec.generated;
    const double dt = dec.seconds;
    if (err_tty) std::fputs("\r\x1b[K", stderr);
    if (dec.interrupted) {
      g_interrupt = 0;
      std::fputs(" [прервано]", stdout);
    }
    std::fputc('\n', stdout);
    std::fflush(stdout);
    if (chat_protocol) {
      std::fputc('\x1e', stdout);
      std::fflush(stdout);
    }

    s.last_tokens = n_gen;
    s.total_tokens += n_gen;
    s.turns += 1;
    s.total_seconds += dt;
    s.last_prefill_s = prefill;
    s.last_tok_per_s =
        dt > 1e-9 ? static_cast<double>(n_gen) / dt : 0.0;
    s.transcript.append("вы: ");
    s.transcript.append(text);
    s.transcript.append("\nтокшит: ");
    s.transcript.append(answer);
    s.transcript.append("\n\n");

    std::fprintf(stderr,
                 "[prefill: %zu tok за %.1fs | генерация: %zu tok за %.1fs = "
                 "%.2f tok/s]\n",
                 turn.size(), prefill, n_gen, dt, s.last_tok_per_s);
    if (freq_cfg.persist && n_gen > 0) {
      if (model.save_freq_profile(freq_cfg.path)) {
        std::fprintf(stderr, "[freq] профиль сохранён: %s\n",
                     freq_cfg.path.c_str());
      }
    }
    if (tokshit::timing_enabled()) print_timing_short(tokshit::metrics());
    if (tty) print_chat_status(s, err_tty);
    tokshit::pool::set_active(false);
  }

done:
  if (dump != nullptr) std::fclose(dump);
}

}  // namespace tokshit::cli
