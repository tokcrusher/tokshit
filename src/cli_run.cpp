#include "cli.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <unistd.h>

#include "tokshit/dequant.hpp"
#include "tokshit/kv.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"
#include "tokshit/pool.hpp"
#include "tokshit/reasoning.hpp"
#include "tokshit/sampling.hpp"
#include "tokshit/special_tokens.hpp"
#include "tokshit/tokenizer.hpp"
#include "tokshit/tui.hpp"

namespace tokshit::cli {

std::size_t prefill_batch_size() {
  const char* v = std::getenv("TOKSHIT_PREFILL_BATCH");
  if (v == nullptr || *v == '\0') return 32;
  const unsigned long long n = std::strtoull(v, nullptr, 10);
  return static_cast<std::size_t>(std::min<unsigned long long>(n, 512));
}

// Переносим готовый батч-kv в старый KVCache, чтобы декод дальше шёл
// неизменным путём. Слой хранится как [k0 v0 k1 v1 ...], по kv_dim float.
void seed_kvcache(const tokshit::Kv& src, std::size_t n,
                  tokshit::KVCache& dst) {
  const std::size_t kvd = src.kv_dim();
  const std::size_t sk = src.k_stride_bytes();
  const std::size_t sv = src.v_stride_bytes();
  for (std::size_t l = 0; l < dst.size(); ++l) {
    auto& row = dst[l];
    row.resize(n * 2 * kvd);
    const uint8_t* kb = src.k_slab(l);
    const uint8_t* vb = src.v_slab(l);
    for (std::size_t p = 0; p < n; ++p) {
      float* out = row.data() + p * 2 * kvd;
      if (src.f16()) {
        const auto* k = reinterpret_cast<const uint16_t*>(kb + p * sk);
        const auto* v = reinterpret_cast<const uint16_t*>(vb + p * sv);
        for (std::size_t i = 0; i < kvd; ++i) {
          out[i] = tokshit::f16_to_f32(k[i]);
          out[kvd + i] = tokshit::f16_to_f32(v[i]);
        }
      } else {
        std::memcpy(out, kb + p * sk, kvd * sizeof(float));
        std::memcpy(out + kvd, vb + p * sv, kvd * sizeof(float));
      }
    }
  }
}

// Батч-префилл чанками: веса читаются один раз на чанк, эксперты — union
// роутинга чанка. Логиты последнего токена каждого чанка пишем в `logits`.
void prefill_batched(tokshit::Model& model,
                     std::span<const uint32_t> tokens, std::size_t pos,
                     std::size_t chunk, tokshit::KVCache& kv,
                     std::vector<float>& logits) {
  const std::size_t vocab = model.vocab();
  const std::size_t cap = pos + tokens.size() + 1;
  tokshit::Kv bk = tokshit::Kv::alloc(
      {model.n_layers(), model.kv_dim(), cap, false});
  std::vector<float> outs(chunk * vocab, 0.0F);
  std::size_t i = 0;
  while (i < tokens.size()) {
    const std::size_t nb = std::min(chunk, tokens.size() - i);
    model.token_logits_batch(tokens.subspan(i, nb), bk, pos + i,
                             std::span<float>(outs.data(), nb * vocab));
    std::copy(outs.begin() + static_cast<std::ptrdiff_t>((nb - 1) * vocab),
              outs.begin() + static_cast<std::ptrdiff_t>(nb * vocab),
              logits.begin());
    i += nb;
  }
  seed_kvcache(bk, pos + tokens.size(), kv);
}

// один прогон: промпт -> генерация -> метрики

void single_shot(tokshit::Model& model, const tokshit::BpeTokenizer& tok,
                 const std::string& prompt, std::size_t max_new,
                 const FreqProfileCfg& freq_cfg) {
  // индикаторы живут только на tty-stderr; в пайпе stdout/stderr чистые
  const bool err_tty = ::isatty(STDERR_FILENO) != 0;
  const std::size_t width = tokshit::tui::terminal_width(STDERR_FILENO);
  const uint64_t io0 = tokshit::io_read_bytes();
  const auto pf0 = tokshit::page_faults();
  tokshit::pool::set_active(true);
  tokshit::KVCache kv(model.n_layers());
  std::size_t pos = 0;
  std::vector<float> logits(model.vocab(), 0.0F);
  FILE* dump = nullptr;
  if (const char* v = std::getenv("TOKSHIT_DUMP_LOGITS");
      v != nullptr && *v != '\0') {
    dump = std::fopen(v, "wb");
  }
  const tokshit::SampleParams params = sample_params_from_env();
  const auto t0 = tokshit::Clock::now();

  std::vector<uint32_t> turn;
  turn.push_back(kUser);
  for (uint32_t id : tok.encode(prompt)) turn.push_back(id);
  turn.push_back(kAssistant);

  model.set_prefilling(true);
  {
    PrefillBar bar;
    bar.start();
    std::size_t done = 0;
    const std::size_t pchunk = dump == nullptr ? prefill_batch_size() : 0;
    if (pchunk != 0 && !turn.empty()) {
      prefill_batched(model, std::span<const uint32_t>(turn), pos, pchunk, kv,
                      logits);
      pos += turn.size();
    } else {
      for (uint32_t id : turn) {
        model.token_logits(id, kv, pos, logits);
        dump_logits(dump, logits);
        ++pos;
        ++done;
        bar.tick(done, turn.size(), width, err_tty);
      }
    }
    bar.finish(err_tty);
  }
  model.set_prefilling(false);
  const double prefill =
      std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
  if (tokshit::prefetch_enabled()) model.pin_hot_experts(64);
  if (const char* v = std::getenv("TOKSHIT_PIN_EXPERTS");
      v != nullptr && *v != '\0') {
    const std::size_t k = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
    if (k != 0) model.pin_hot_experts(k);
  }
  // TOKSHIT_FREQ_PERSIST: профиль накопился за prefill+генерацию — пишем
  // TOKSHIT_FREQ_PERSIST: the profile accumulated over prefill+generation —
  if (freq_cfg.persist) {
    if (model.save_freq_profile(freq_cfg.path)) {
      std::fprintf(stderr, "[freq] профиль сохранён: %s\n",
                   freq_cfg.path.c_str());
    }
  }

  // снимок метрик перед генерацией: увидим prefill отдельно от генерации
  const uint64_t prefill_hit = tokshit::metrics().c_expert_hit.load();
  const uint64_t prefill_miss = tokshit::metrics().c_expert_miss.load();
  // TOKSHIT_PROF=1: обнуляем счётчики после прогрева, чтобы в конце увидеть
  // чистый профиль генерации, а не prefill+генерацию вместе
  // TOKSHIT_PROF=1: zero the counters after warm-up so the report at the end
  const bool prof_gen_only = std::getenv("TOKSHIT_PROF") != nullptr;
  if (prof_gen_only) tokshit::metrics_reset();

  const bool print_ids = std::getenv("TOKSHIT_PRINT_IDS") != nullptr;
  Rng rng;
  DecodeOptions dopt;
  dopt.max_new = max_new;
  dopt.token_debug = std::getenv("TOKSHIT_TOKEN_DEBUG") != nullptr;
  dopt.pin_every = true;
  const DecodeResult dec = decode(model, tok, kv, pos, logits, params, rng,
                                  reasoning_budget_from_env(), dump, dopt);
  if (dump != nullptr) std::fclose(dump);
  const std::size_t n_gen = dec.generated;
  const double dt = dec.seconds;
  // стереть живую строку декода и закончить ответ на чистой строке
  if (err_tty) {
    std::fputs("\r\x1b[K", stderr);
    std::fflush(stderr);
  }
  std::fputc('\n', stdout);
  if (print_ids && !dec.ids.empty()) {
    std::fprintf(stderr, "[ids:");
    for (uint32_t id : dec.ids) {
      std::fprintf(stderr, " %u", id);
    }
    std::fprintf(stderr, "]\n");
  }
  std::fprintf(stderr,
               "[prefill: %zu tok за %.1fs | генерация: %zu tok за %.1fs = "
               "%.2f tok/s]\n",
               turn.size(), prefill, n_gen, dt,
               static_cast<double>(n_gen) / (dt > 1e-9 ? dt : 1e-9));

  const auto& m = tokshit::metrics();
  {
    // TOKSHIT_PROF обнуляет счётчики после prefill, поэтому вычитать
    // prefill_hit/miss здесь нельзя: получится underflow и hit-rate > 100%.
    // TOKSHIT_PROF zeroes the counters after prefill, so subtracting
    const uint64_t gh = prof_gen_only
                            ? m.c_expert_hit.load()
                            : m.c_expert_hit.load() - prefill_hit;
    const uint64_t gm = prof_gen_only
                            ? m.c_expert_miss.load()
                            : m.c_expert_miss.load() - prefill_miss;
    std::fprintf(stderr,
                 "[gen-only] hit=%llu miss=%llu rate=%.1f%% (prefill was %llu/%llu)\n",
                 static_cast<unsigned long long>(gh),
                 static_cast<unsigned long long>(gm),
                 (gh + gm) > 0 ? 100.0 * static_cast<double>(gh) / static_cast<double>(gh + gm) : 0.0,
                 static_cast<unsigned long long>(prefill_hit),
                 static_cast<unsigned long long>(prefill_miss));
  }
  if (tokshit::timing_enabled()) {
    print_timing_full(m);
    const uint64_t io1 = tokshit::io_read_bytes();
    const auto pf1 = tokshit::page_faults();
    const double t_mm = static_cast<double>(m.t_mm.load()) / 1e9;
    const double mm_bytes = static_cast<double>(m.c_mm_bytes.load()) / 1e9;
    std::fprintf(stderr,
                 "[io: read_bytes=%.2f GB, rss=%llu MB, minflt=%llu "
                 "majflt=%llu]\n",
                 static_cast<double>(io1 - io0) / 1e9,
                 static_cast<unsigned long long>(tokshit::rss_mb()),
                 static_cast<unsigned long long>(pf1.first - pf0.first),
                 static_cast<unsigned long long>(pf1.second - pf0.second));
    std::fprintf(stderr, "[matvec: %.2f GB за %.2fs = %.2f GB/s]\n", mm_bytes,
                 t_mm, mm_bytes / (t_mm > 1e-9 ? t_mm : 1e-9));
  }
  if (std::getenv("TOKSHIT_EXPERT_STATS") != nullptr) {
    print_expert_stats(m);
  }
  tokshit::pool::set_active(false);
}

// teacher-forced verify: гоним draft по референсной последовательности идов
// (например с полного жадного прогона) и на каждом шаге сравниваем argmax
// черновика с референсом. это и есть acceptance rate, который решает судьбу
// спекции: установленного контекста больше не менять

void verify_ids(tokshit::Model& model, const tokshit::BpeTokenizer& tok,
                const std::string& prompt, const std::string& path,
                std::size_t cap) {
  std::vector<uint32_t> ref;
  {
    // парсер живучий: ест и сырую строку "[ids: 1 2 ...]" от TOKSHIT_PRINT_IDS,
    // и просто числа в столбик. нечисловой мусор перепрыгиваем
    // TOKSHIT_PRINT_IDS and bare numbers alike, skips non-numeric junk
    if (FILE* f = std::fopen(path.c_str(), "rb"); f != nullptr) {
      std::string buf;
      char chunk[4096];
      std::size_t got = 0;
      while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0) {
        buf.append(chunk, got);
      }
      std::fclose(f);
      const char* p = buf.data();
      while (*p != '\0') {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) {
          ++p;
        } else {
          if (v > 0 && v < 1000000000L) {
            ref.push_back(static_cast<uint32_t>(v));
          }
          p = end;
        }
      }
    }
  }
  if (cap != 0 && cap < ref.size()) ref.resize(cap);
  if (ref.empty()) {
    std::fprintf(stderr, "[accept] нет идов в файле\n");
    return;
  }

  tokshit::KVCache kv(model.n_layers());
  std::vector<float> logits(model.vocab(), 0.0F);
  std::size_t pos = 0;
  std::vector<uint32_t> turn;
  turn.push_back(kUser);
  for (uint32_t id : tok.encode(prompt)) turn.push_back(id);
  turn.push_back(kAssistant);

  tokshit::pool::set_active(true);
  const auto g0 = tokshit::Clock::now();
  for (uint32_t id : turn) {
    model.token_logits(id, kv, pos, logits);
    ++pos;
  }
  if (tokshit::prefetch_enabled()) model.pin_hot_experts(64);
  if (const char* v = std::getenv("TOKSHIT_PIN_EXPERTS");
      v != nullptr && *v != '\0') {
    const std::size_t k = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
    if (k != 0) model.pin_hot_experts(k);
  }

  std::size_t agree = 0;
  std::size_t runs = 0;
  std::size_t current = 0;
  std::size_t longest = 0;
  std::size_t first_miss = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    std::size_t next = 0;
    for (std::size_t k = 1; k < logits.size(); ++k) {
      if (logits[k] > logits[next]) next = k;
    }
    if (next == ref[i]) {
      if (current == 0) ++runs;
      ++current;
      longest = std::max(longest, current);
      ++agree;
    } else {
      if (first_miss == 0) first_miss = i + 1;
      current = 0;
    }
    model.token_logits(ref[i], kv, pos, logits);
    ++pos;
  }
  const double dt =
      std::chrono::duration<double>(tokshit::Clock::now() - g0).count();
  tokshit::pool::set_active(false);

  // средняя серия = длина принятого куска на раунд спин-вери
  const double avg_run =
      runs == 0 ? 0.0 : static_cast<double>(agree) / static_cast<double>(runs);
  std::fprintf(stderr,
               "[accept: %zu/%zu ok = %.1f%%, позиция первой мизерии i=%zu, "
               "серий=%zu, средняя серия=%.2f, самой длинной=%zu, токен/с=%.2f]\n",
               agree, ref.size(),
               100.0 * static_cast<double>(agree) /
                   static_cast<double>(ref.size()),
               first_miss, runs, avg_run, longest,
               static_cast<double>(ref.size()) / (dt > 1e-9 ? dt : 1e-9));
}

// ── интерактивный чат ──────────────────────────────────────────────────────

}  // namespace tokshit::cli
