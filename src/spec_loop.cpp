// спекулятивный цикл одной моделью: черновик с урезанным роутером,
// verify полным роутингом батчем, приём префикса, откат kv. жадный семплинг
// single-model speculative loop: reduced-router draft, full-router batched
// verify, prefix accept, kv rollback. greedy sampling only
#include "tokshit/spec_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tokshit/metrics.hpp"
#include "tokshit/pool.hpp"
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

}  // namespace

void run(Model& model, const BpeTokenizer& tok, const std::string& prompt,
         std::size_t max_new) {
  const std::size_t dtopk = env_u("TOKSHIT_DRAFT_TOPK", 2);
  // verify можно тоже урезать: полный top-6 на батче b раздувает union
  // экспертов в b раз, а именно на нём спекуляция и теряет
  // verify can be cut too: a full top-6 on a batch of b blows the expert
  // union up by b, which is exactly where speculation loses
  const std::size_t vtopk = env_u("TOKSHIT_SPEC_VERIFY_TOPK", 0);
  const std::size_t ctx = env_u("TOKSHIT_SPEC_CTX", 2048);
  const std::size_t maxb = std::max<std::size_t>(env_u("TOKSHIT_SPEC_MAXB", 8), 2);
  // K не может быть больше батча: cand и outs размечены под maxb
  // K cannot exceed the batch: cand and outs are sized for maxb
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
  // a prompt longer than the cache cannot run: drop the tail honestly,
  // leaving room to generate, and say so on stderr
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
  // prefill in chunks: one weight pass per chunk
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
  // after prefill the router frequencies mean something: pin the top experts
  // so generation does not start on cold reads
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
    if (pos >= kv.cap()) break;  // кеш полон / cache is full
    const uint32_t first = static_cast<uint32_t>(argmax(head));
    if (tok.is_eog(first)) break;
    ++rounds;

    // черновик: топ-1 по полным логитам уже есть, дальше — урезанный роутер
    // draft: the top-1 under full logits is there, the rest uses the cut router
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
    // verify: the router can be cut too (TOKSHIT_SPEC_VERIFY_TOPK), batch
    // over the candidates, kv rewrites those rows
    model.set_router_topk(vtopk < n_used ? vtopk : 0);
    model.token_logits_batch(std::span<const uint32_t>(cand.data(), nd), kv,
                             pos, std::span<float>(outs.data(), nd * vocab));
    ++verify_passes;

    // цепочка: r_j предсказывает cand[j+1]. принято >= 1
    // chain: r_j predicts cand[j+1]. at least one token is accepted
    std::size_t m = 1;
    // диагностика acceptance: что верификация сказала на каждую позицию
    // цепочки и насколько близко был черновой токен
    // acceptance diagnostics: what verification said at every chain position
    // and how close the draft token was
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
          // top-2 tokens of the verify row with logits: shows whether the
          // draft candidate was the runner-up or out of the race entirely
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
    // logits of the last accepted token seed the next round
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
    // rollback the unaccepted tail: draft rows past the prefix are dropped
    kv.truncate_all(pos);
    if (emitted == 0) break;
  }
  const double gen_s =
      std::chrono::duration<double>(Clock::now() - g0).count();
  pool::set_active(false);
  std::fputc('\n', stdout);

  // трейс идов для паритетного сравнения с plain greedy
  // id trace for the parity check against plain greedy
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

}  // namespace tokshit::spec
