// спекулятивный цикл одной моделью: черновик с урезанным роутером,
// verify полным роутингом батчем, приём префикса, откат kv. жадный семплинг
#include "tokshit/spec_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tokshit/dspark.hpp"
#include "tokshit/gguf.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/mmap.hpp"
#include "tokshit/pool.hpp"
#include "tokshit/sampling.hpp"
#include "tokshit/special_tokens.hpp"

namespace tokshit::spec {

namespace {

std::size_t env_u(const char* name, std::size_t def) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return def;
  return static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
}

std::size_t argmax(std::span<const float> x) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < x.size(); ++i) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

// Same Inkling chat template as cli::build_turn, kept here so the core spec
// loop does not depend on the executable's cli translation unit.
std::vector<uint32_t> ink_turn(const BpeTokenizer& tok,
                               std::string_view user) {
  std::vector<uint32_t> turn;
  if (std::getenv("TOKSHIT_RAW_PROMPT") != nullptr) return tok.encode(user);
  auto push_special = [&](std::string_view name) {
    const auto id = tok.token_id(name);
    if (id.has_value()) turn.push_back(*id);
  };
  push_special("<|message_system|>");
  push_special("<|content_text|>");
  for (uint32_t id : tok.encode("Thinking effort level: 0.9")) {
    turn.push_back(id);
  }
  push_special("<|end_message|>");
  push_special("<|message_user|>");
  push_special("<|content_text|>");
  for (uint32_t id : tok.encode(user)) turn.push_back(id);
  push_special("<|end_message|>");
  push_special("<|message_model|>");
  return turn;
}

}  // namespace

void run(Model& model, const BpeTokenizer& tok, const std::string& prompt,
         std::size_t max_new) {
  const std::size_t dtopk = env_u("TOKSHIT_DRAFT_TOPK", 2);
  // verify можно тоже урезать: полный top-6 на батче b раздувает union
  // экспертов в b раз, а именно на нём спекуляция и теряет
  const std::size_t vtopk = env_u("TOKSHIT_SPEC_VERIFY_TOPK", 0);
  const std::size_t ctx = env_u("TOKSHIT_SPEC_CTX", 2048);
  const std::size_t maxb = std::max<std::size_t>(env_u("TOKSHIT_SPEC_MAXB", 8), 2);
  // K не может быть больше батча: cand и outs размечены под maxb
  const std::size_t K = std::min(env_u("TOKSHIT_SPEC_K", 4), maxb);
  const std::size_t n_used = model.n_used();

  Kv kv = Kv::alloc({model.n_layers(), model.kv_dim(), ctx, true});
  std::fprintf(stderr,
               "[spec] kv cap=%zu, f16, %.1f MB; K=%zu, draft topk=%zu, "
               "verify topk=%zu\n",
               ctx, static_cast<double>(kv.bytes()) / 1e6, K, dtopk,
               vtopk != 0 ? std::min(vtopk, n_used) : n_used);

  std::vector<uint32_t> turn;
  turn.push_back(kUser);
  for (uint32_t id : tok.encode(prompt)) turn.push_back(id);
  turn.push_back(kAssistant);
  // промпт длиннее кеша не провернём: честно режем хвост, оставляя место
  // под генерацию, и говорим об этом в stderr
  const std::size_t reserve = std::min<std::size_t>(maxb, ctx / 4);
  if (turn.size() + reserve > ctx) {
    const std::size_t keep = ctx > reserve ? ctx - reserve : ctx;
    std::fprintf(stderr, "[spec] промпт %zu > ctx %zu, оставляю %zu хвостовых токенов\n",
                 turn.size(), ctx, keep);
    turn.erase(turn.begin(),
               turn.end() - static_cast<std::ptrdiff_t>(keep));
  }

  const std::size_t vocab = model.vocab();
  std::vector<float> outs(vocab * maxb);
  std::vector<float> head(vocab, 0.0F);

  const auto t0 = Clock::now();
  pool::set_active(true);

  // префилл чанками: один проход весов на чанк
  std::size_t pos = 0;
  std::size_t i = 0;
  while (i < turn.size()) {
    const std::size_t nb = std::min(maxb, turn.size() - i);
    model.token_logits_batch(std::span<const uint32_t>(turn.data() + i, nb),
                             kv, pos, std::span<float>(outs.data(), nb * vocab));
    std::copy(outs.begin() + static_cast<std::ptrdiff_t>((nb - 1) * vocab),
              outs.begin() + static_cast<std::ptrdiff_t>(nb * vocab),
              head.begin());
    pos += nb;
    i += nb;
  }
  const double prefill_s =
      std::chrono::duration<double>(Clock::now() - t0).count();

  // после префилла частоты роутера уже осмысленные: прогреваем/пиним топ
  // экспертов, чтобы генерация не начиналась с холодных чтений
  if (const char* v = std::getenv("TOKSHIT_PIN_EXPERTS");
      v != nullptr && *v != '\0') {
    const std::size_t k = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
    if (k != 0) model.pin_hot_experts(k);
  }

  std::size_t gen = 0;
  std::size_t rounds = 0;
  std::size_t accepted_draft = 0;
  std::size_t verify_passes = 0;
  std::size_t draft_steps = 0;
  std::vector<uint32_t> ids;
  std::vector<uint32_t> cand(maxb);
  const auto g0 = Clock::now();

  while (gen < max_new) {
    if (pos >= kv.cap()) break;  // кеш полон
    const uint32_t first = static_cast<uint32_t>(argmax(head));
    if (tok.is_eog(first)) break;
    ++rounds;

    // черновик: топ-1 по полным логитам уже есть, дальше — урезанный роутер
    std::size_t nd = 1;
    cand[0] = first;
    if (K > 1 && pos + 1 < kv.cap()) {
      model.set_router_topk(std::min(dtopk, n_used));
      while (nd < K && pos + nd < kv.cap()) {
        model.token_logits_batch(
            std::span<const uint32_t>(cand.data() + nd - 1, 1), kv, pos + nd - 1,
            std::span<float>(outs.data(), vocab));
        cand[nd] = static_cast<uint32_t>(argmax(
            std::span<const float>(outs.data(), vocab)));
        ++draft_steps;
        ++nd;
        if (tok.is_eog(cand[nd - 1])) break;
      }
    }

    // verify: роутер можно урезать (TOKSHIT_SPEC_VERIFY_TOPK), батч по
    // кандидатам, kv перезаписывает их строки
    model.set_router_topk(vtopk < n_used ? vtopk : 0);
    model.token_logits_batch(std::span<const uint32_t>(cand.data(), nd), kv,
                             pos, std::span<float>(outs.data(), nd * vocab));
    ++verify_passes;

    // цепочка: r_j предсказывает cand[j+1]. принято >= 1
    std::size_t m = 1;
    // диагностика acceptance: что верификация сказала на каждую позицию
    // цепочки и насколько близко был черновой токен
    const bool spec_dbg = std::getenv("TOKSHIT_SPEC_DEBUG") != nullptr;
    if (spec_dbg) {
      std::fprintf(stderr, "[spec r%zu] pos=%zu nd=%zu cand:", rounds, pos, nd);
      for (std::size_t j = 0; j < nd; ++j) {
        std::fprintf(stderr, " %u", cand[j]);
      }
      std::fputs(" | vrow:", stderr);
      for (std::size_t j = 0; j + 1 < nd; ++j) {
        std::fprintf(stderr, " %zu", argmax(std::span<const float>(
                                          outs.data() + j * vocab, vocab)));
      }
      std::fputs("\n", stderr);
    }
    for (std::size_t j = 0; j + 1 < nd; ++j) {
      const std::size_t nxt = argmax(
          std::span<const float>(outs.data() + j * vocab, vocab));
      if (static_cast<uint32_t>(nxt) == cand[j + 1]) {
        ++m;
      } else {
        if (spec_dbg) {
          // два лучших токена verify-строки с логитами: видно, был ли
          // черновой кандидат вторым или вообще не в игре
          const float* row = outs.data() + j * vocab;
          std::size_t b1 = 0, b2 = 0;
          for (std::size_t i = 1; i < vocab; ++i) {
            if (row[i] > row[b1]) {
              b2 = b1;
              b1 = i;
            } else if (row[i] > row[b2]) {
              b2 = i;
            }
          }
          std::fprintf(stderr,
                       "[spec r%zu] j=%zu отказ: cand=%u (%.3f), verify1=%zu "
                       "(%.3f), verify2=%zu (%.3f)\n",
                       rounds, j, cand[j + 1], row[cand[j + 1]], b1, row[b1],
                       b2, row[b2]);
        }
        break;
      }
    }
    // логиты последнего принятого токена — голова следующего раунда
    std::copy(outs.begin() + static_cast<std::ptrdiff_t>((m - 1) * vocab),
              outs.begin() + static_cast<std::ptrdiff_t>(m * vocab),
              head.begin());

    std::size_t emitted = 0;
    for (std::size_t j = 0; j < m && gen < max_new; ++j) {
      const uint32_t id = cand[j];
      if (tok.is_eog(id)) {
        gen = max_new;
        break;
      }
      ids.push_back(id);
      const auto& bytes = tok.token_bytes(id);
      if (!bytes.empty()) {
        std::fwrite(bytes.data(), 1, bytes.size(), stdout);
      }
      ++gen;
      ++emitted;
    }
    if (gen >= max_new) break;
    if (m > 1) accepted_draft += m - 1;
    pos += m;
    // откат непринятого хвоста: черновые строки за префиксом больше не нужны
    kv.truncate_all(pos);
    if (emitted == 0) break;
  }
  const double gen_s =
      std::chrono::duration<double>(Clock::now() - g0).count();
  pool::set_active(false);
  std::fputc('\n', stdout);

  // трейс идов для паритетного сравнения с plain greedy
  if (std::getenv("TOKSHIT_PRINT_IDS") != nullptr && !ids.empty()) {
    std::fputs("[ids:", stderr);
    for (uint32_t id : ids) std::fprintf(stderr, " %u", id);
    std::fputs("]\n", stderr);
  }

  const auto pf = page_faults();
  const auto& m = metrics();
  const uint64_t passes = m.c_expert_passes.load();
  std::fprintf(
      stderr,
      "[spec: prefill %zu tok за %.1fs | генерация %zu tok за %.1fs = "
      "%.2f tok/s | раундов %zu, verify %zu, черновых шагов %zu, принято "
      "черновиков %zu (%.1f%%), kv %.1f MB]\n",
      turn.size(), prefill_s, gen, gen_s,
      static_cast<double>(gen) / (gen_s > 1e-9 ? gen_s : 1e-9), rounds,
      verify_passes, draft_steps, accepted_draft,
      draft_steps > 0
          ? 100.0 * static_cast<double>(accepted_draft) /
                static_cast<double>(draft_steps)
          : 0.0,
      static_cast<double>(kv.bytes()) / 1e6);
  std::fprintf(
      stderr,
      "[spec: expert_hit=%llu miss=%llu union=%.2f/pass passes=%llu]\n",
      static_cast<unsigned long long>(m.c_expert_hit.load()),
      static_cast<unsigned long long>(m.c_expert_miss.load()),
      passes != 0 ? static_cast<double>(m.c_expert_union.load()) /
                        static_cast<double>(passes)
                  : 0.0,
      static_cast<unsigned long long>(passes));
  std::fprintf(stderr, "[io: read_bytes=%.2f GB, rss=%llu MB, majflt=%llu]\n",
               static_cast<double>(io_read_bytes()) / 1e9,
               static_cast<unsigned long long>(rss_mb()),
               static_cast<unsigned long long>(pf.second));
}

bool run_dspark(Model& model, const BpeTokenizer& tok,
                const std::string& prompt, std::size_t max_new,
                const std::string& draft_path) {
  Inkling2* ink = model.inkling2();
  if (ink == nullptr) {
    std::fprintf(stderr, "[dspark] not an Inkling2 model\n");
    return false;
  }
  // Tap 41 живёт только на третьем core-проходе. С TOKSHIT_INK_PASSES=1..2
  // target-тэпы неполные, и черновик получил бы мёртвый контекст.
  if (const char* pv = std::getenv("TOKSHIT_INK_PASSES");
      pv != nullptr && *pv != '\0') {
    const unsigned long n = std::strtoul(pv, nullptr, 10);
    if (n != 3) {
      std::fprintf(stderr,
                   "[dspark] TOKSHIT_INK_PASSES=%lu: DSpark требует все 3 "
                   "core-прохода (taps 5/23/41/48); обычный decode\n",
                   n);
      return false;
    }
  }
  float temperature = 0.0F;
  if (const char* tv = std::getenv("TOKSHIT_TEMPERATURE");
      tv != nullptr && *tv != '\0') {
    temperature = std::strtof(tv, nullptr);
  }
  if (temperature > 1e-6F) {
    std::fprintf(stderr,
                 "[dspark] только greedy: выставьте --temperature 0; "
                 "спекуляция выключена\n");
    return false;
  }

  auto g = Gguf::open(draft_path);
  if (!g.ok()) {
    std::fprintf(stderr, "[dspark] open: %s\n", g.error().message.c_str());
    return false;
  }
  Gguf draft_gguf = std::move(g.value());
  auto mm = draft_gguf.mmap_file();
  if (!mm.ok()) {
    std::fprintf(stderr, "[dspark] mmap: %s\n", mm.error().message.c_str());
    return false;
  }
  Mmap draft_mm = std::move(mm.value());
  auto d = DSpark::load(draft_gguf, draft_mm, *ink, ink->kv_ctx());
  if (!d.ok()) {
    std::fprintf(stderr, "[dspark] load: %s\n", d.error().message.c_str());
    return false;
  }
  auto& draft = *d.value();

  std::size_t K = env_u("TOKSHIT_DSPARK_K", 3);
  K = std::min(K, draft.max_draft());
  if (K == 0) {
    std::fprintf(stderr, "[dspark] draft max is 0\n");
    return false;
  }
  const bool batch_cfg = env_u("TOKSHIT_DSPARK_BATCH", 1) != 0;
  const std::size_t vocab = ink->vocab();

  std::vector<uint32_t> turn = ink_turn(tok, prompt);
  std::vector<float> head(ink->vocab(), 0.0F);
  std::size_t pos = 0;
  pool::set_active(true);
  const auto t0 = Clock::now();
  for (uint32_t id : turn) {
    ink->token_logits(id, pos, head);
    draft.inject(pos, ink->taps());
    ++pos;
  }
  const double prefill_s =
      std::chrono::duration<double>(Clock::now() - t0).count();
  uint32_t last = turn.empty() ? 0 : turn.back();

  std::size_t gen = 0;
  std::size_t rounds = 0;
  std::size_t proposed = 0;
  std::size_t accepted = 0;
  std::size_t verifies = 0;
  std::size_t batch_rounds = 0;
  std::vector<uint32_t> ids;
  std::vector<uint32_t> cand;
  std::vector<float> batch_logits(vocab * K, 0.0F);
  Utf8Buffer utf8;

  const auto amax = [](std::span<const float> x) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < x.size(); ++i) {
      if (x[i] > x[best]) best = i;
    }
    return best;
  };

  const auto emit = [&](uint32_t id) -> bool {
    if (tok.is_eog(id)) return false;
    ids.push_back(id);
    if (id >= 199998) {
      utf8 = Utf8Buffer{};
      if (id == 200010) std::fputc('\n', stdout);
    } else {
      const std::string chunk = utf8.push(tok.token_bytes(id));
      if (!chunk.empty()) {
        std::fwrite(chunk.data(), 1, chunk.size(), stdout);
        std::fflush(stdout);
      }
    }
    ++gen;
    return gen < max_new;
  };

  const auto g0 = Clock::now();
  while (gen < max_new && pos < ink->kv_ctx()) {
    const std::size_t first = amax(head);
    if (tok.is_eog(static_cast<uint32_t>(first))) break;

    draft.draft(last, pos, K, cand);
    ++rounds;
    proposed += cand.size();
    std::size_t m = 0;
    bool stop = false;
    if (cand.empty()) {
      ink->token_logits(static_cast<uint32_t>(first), pos, head);
      draft.inject(pos, ink->taps());
      last = static_cast<uint32_t>(first);
      ++m;
      ++verifies;
      stop = !emit(last);
    } else if (cand[0] != first) {
      // Черновик промахнулся уже на первом токене: батч всё равно был бы
      // отвергнут, платим только за одну коррекцию.
      const uint32_t id = static_cast<uint32_t>(first);
      ink->token_logits(id, pos, head);
      draft.inject(pos, ink->taps());
      last = id;
      ++m;
      ++verifies;
      stop = !emit(id);
    } else if (tok.is_eog(cand[0])) {
      // target greedy сам хочет EOG: не кормим его обратно
      stop = true;
    } else {
      // Кандидаты до первого EOG; сам EOG в target не прогоняем.
      std::size_t n = cand.size();
      for (std::size_t j = 1; j < cand.size(); ++j) {
        if (tok.is_eog(cand[j])) {
          n = j;
          break;
        }
      }
      const std::size_t npos = pos + n;
      if (batch_cfg && npos <= ink->kv_ctx()) {
        static const bool check =
            std::getenv("TOKSHIT_DSPARK_CHECK") != nullptr;
        std::vector<float> check_logits;
        if (check) {
          check_logits.assign(n * vocab, 0.0F);
          Inkling2::Snapshot pre;
          ink->snapshot(pre);
          std::vector<float> row(vocab, 0.0F);
          for (std::size_t j = 0; j < n; ++j) {
            ink->token_logits(cand[j], pos + j, row);
            std::copy(row.begin(), row.end(),
                      check_logits.begin() +
                          static_cast<std::ptrdiff_t>(j * vocab));
          }
          ink->restore(pre);
        }
        ink->token_logits_batch(std::span<const uint32_t>(cand.data(), n),
                                pos,
                                std::span<float>(batch_logits.data(),
                                                 n * vocab));
        if (check) {
          for (std::size_t j = 0; j < n; ++j) {
            const float* a = check_logits.data() + j * vocab;
            const float* b = batch_logits.data() + j * vocab;
            float mx = 0.0F;
            std::size_t ai = 0;
            std::size_t bi = 0;
            for (std::size_t i = 0; i < vocab; ++i) {
              mx = std::max(mx, std::fabs(a[i] - b[i]));
              if (a[i] > a[ai]) ai = i;
              if (b[i] > b[bi]) bi = i;
            }
            std::fprintf(stderr,
                         "[dspark check] pos=%zu j=%zu max|d|=%.6g "
                         "seq=%zu bat=%zu %s\n",
                         pos, j, mx, ai, bi, ai == bi ? "ok" : "MISS");
          }
        }
        ++batch_rounds;
        verifies += n;
        std::size_t acc = 1;
        while (acc < n) {
          const std::size_t want = amax(std::span<const float>(
              batch_logits.data() + (acc - 1) * vocab, vocab));
          if (cand[acc] == want) {
            ++acc;
          } else {
            break;
          }
        }
        const auto taps = ink->taps_batch();
        const std::size_t nt = ink->n_taps();
        if (acc == n) {
          for (std::size_t j = 0; j < n; ++j) {
            draft.inject(pos + j, taps.subspan(j * nt * ink->n_embd(),
                                               nt * ink->n_embd()));
          }
          accepted += n;
          std::copy(batch_logits.begin() +
                        static_cast<std::ptrdiff_t>((n - 1) * vocab),
                    batch_logits.begin() +
                        static_cast<std::ptrdiff_t>(n * vocab),
                    head.begin());
          last = cand[n - 1];
          m = n;
          for (std::size_t j = 0; j < n; ++j) {
            if (!emit(cand[j])) {
              stop = true;
              break;
            }
          }
        } else {
          ink->restore_batch(acc);
          for (std::size_t j = 0; j < acc; ++j) {
            draft.inject(pos + j, taps.subspan(j * nt * ink->n_embd(),
                                               nt * ink->n_embd()));
          }
          const uint32_t id = static_cast<uint32_t>(amax(
              std::span<const float>(batch_logits.data() + (acc - 1) * vocab,
                                     vocab)));
          accepted += acc;
          ink->token_logits(id, pos + acc, head);
          draft.inject(pos + acc, ink->taps());
          last = id;
          ++verifies;
          m = acc + 1;
          for (std::size_t j = 0; j < acc; ++j) {
            if (!emit(cand[j])) {
              stop = true;
              break;
            }
          }
          if (!stop) stop = !emit(id);
        }
      } else {
        // Последовательный fallback: тот же код, что был до батча.
        for (std::size_t j = 0; !stop && j < cand.size() && gen < max_new;
             ++j) {
          const std::size_t want = amax(head);
          if (cand[j] != want) {
            const uint32_t id = static_cast<uint32_t>(want);
            ink->token_logits(id, pos + m, head);
            draft.inject(pos + m, ink->taps());
            last = id;
            ++m;
            ++verifies;
            stop = !emit(id);
            break;
          }
          const uint32_t id = cand[j];
          if (tok.is_eog(id)) {
            stop = true;
            break;
          }
          ink->token_logits(id, pos + m, head);
          draft.inject(pos + m, ink->taps());
          last = id;
          ++m;
          ++accepted;
          ++verifies;
          stop = !emit(id);
        }
      }
    }
    pos += m;
    if (m == 0 || stop) break;
  }
  const double gen_s =
      std::chrono::duration<double>(Clock::now() - g0).count();
  pool::set_active(false);
  std::fputc('\n', stdout);

  if (std::getenv("TOKSHIT_PRINT_IDS") != nullptr && !ids.empty()) {
    std::fputs("[ids:", stderr);
    for (uint32_t id : ids) std::fprintf(stderr, " %u", id);
    std::fputs("]\n", stderr);
  }
  std::fprintf(
      stderr,
      "[dspark: prefill %zu tok за %.1fs | генерация %zu tok за %.1fs = "
      "%.2f tok/s | раундов %zu, verify %zu, предложено %zu, принято %zu "
      "(%.1f%%), средняя длина %.2f, батч-раундов %zu]\n",
      turn.size(), prefill_s, gen, gen_s,
      static_cast<double>(gen) / (gen_s > 1e-9 ? gen_s : 1e-9), rounds,
      verifies, proposed, accepted,
      proposed != 0 ? 100.0 * static_cast<double>(accepted) /
                          static_cast<double>(proposed)
                    : 0.0,
      rounds != 0 ? static_cast<double>(gen) / static_cast<double>(rounds)
                  : 0.0,
      batch_rounds);
  return true;
}

}  // namespace tokshit::spec
