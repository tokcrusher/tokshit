#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "tokshit/dequant.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"
#include "tokshit/pool.hpp"
#include "tokshit/reasoning.hpp"
#include "tokshit/sampling.hpp"
#include "tokshit/spec_loop.hpp"
#include "tokshit/tokenizer.hpp"
#include "tokshit/tui.hpp"

namespace {

constexpr uint32_t kUser = 150003;
constexpr uint32_t kAssistant = 150004;

// Ctrl-C не выбивает из чата: он прерывает текущий ответ или строку ввода
// Ctrl-C does not kill the chat: it aborts the current reply or input line
volatile std::sig_atomic_t g_interrupt = 0;

extern "C" void on_sigint(int) { g_interrupt = 1; }

uint64_t env_u64(const char* name, uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  return std::strtoull(v, nullptr, 10);
}

// TOKSHIT_REASONING_BUDGET: -1 без лимита, 0 закрыть сразу, N > 0 кап.
// мусор и значения меньше -1 трактуем как «без лимита»
// TOKSHIT_REASONING_BUDGET: -1 unlimited, 0 close at once, N > 0 caps.
// junk and values below -1 mean unlimited
int reasoning_budget_from_env() {
  const char* v = std::getenv("TOKSHIT_REASONING_BUDGET");
  if (v == nullptr || *v == '\0') return -1;
  char* end = nullptr;
  errno = 0;
  const long n = std::strtol(v, &end, 10);
  if (end == nullptr || *end != '\0' || errno == ERANGE) {
    std::fprintf(stderr,
                 "[reasoning-budget: '%s' не число, беру -1 (без лимита)]\n",
                 v);
    return -1;
  }
  if (n < -1) return -1;
  if (n > INT_MAX) return INT_MAX;
  return static_cast<int>(n);
}

// путь к модели по умолчанию, из TOKSHIT_MODEL или вшитый
// default model path, from TOKSHIT_MODEL or the builtin one
std::string default_model_path() {
  if (const char* v = std::getenv("TOKSHIT_MODEL");
      v != nullptr && *v != '\0') {
    return v;
  }
  if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
    return std::string(home) + "/models/flint-UD-Q3_K_XL.gguf";
  }
  return "/data/data/com.termux/files/home/models/flint-UD-Q3_K_XL.gguf";
}

// параметры сэмплинга из env: температура, top-k, top-p
// sampling params from env: temperature, top-k, top-p
tokshit::SampleParams sample_params_from_env() {
  tokshit::SampleParams p;
  if (const char* v = std::getenv("TOKSHIT_TEMPERATURE");
      v != nullptr && *v != '\0') {
    p.temperature = std::strtof(v, nullptr);
  }
  if (const char* v = std::getenv("TOKSHIT_TOP_K"); v != nullptr && *v != '\0') {
    p.top_k = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
  }
  if (const char* v = std::getenv("TOKSHIT_TOP_P"); v != nullptr && *v != '\0') {
    p.top_p = std::strtof(v, nullptr);
  }
  return p;
}

// TOKSHIT_DUMP_LOGITS=<path>: пишем uint32 count + little-endian f32 логиты на
// каждый посчитанный токен. это для паритетного гейта (сравнить с эталоном)
// TOKSHIT_DUMP_LOGITS=<path>: append uint32 count + little-endian f32 logits per
// evaluated token. used by the parity gate (compare against the reference)
void dump_logits(FILE* f, const std::vector<float>& logits) {
  if (f == nullptr) return;
  const auto count = static_cast<uint32_t>(logits.size());
  std::fwrite(&count, sizeof(count), 1, f);
  std::fwrite(logits.data(), sizeof(float), logits.size(), f);
}

// короткая разбивка по времени
// short timing breakdown
void print_timing_short(const tokshit::Metrics& m) {
  std::fprintf(
      stderr,
      "[время: attn=%.2fs ffn=%.2fs expert=%.2fs router=%.2fs head=%.2fs "
      "embd=%.2fs shexp=%.2fs]\n",
      static_cast<double>(m.t_attn.load()) / 1e9,
      static_cast<double>(m.t_ffn.load()) / 1e9,
      static_cast<double>(m.t_expert.load()) / 1e9,
      static_cast<double>(m.t_router.load()) / 1e9,
      static_cast<double>(m.t_head.load()) / 1e9,
      static_cast<double>(m.t_embd.load()) / 1e9,
      static_cast<double>(m.t_shexp.load()) / 1e9);
}

// полная разбивка + счётчики кэша
// full breakdown + cache counters
void print_timing_full(const tokshit::Metrics& m) {
  const uint64_t passes = m.c_expert_passes.load();
  const uint64_t union_sum = m.c_expert_union.load();
  std::fprintf(
      stderr,
      "[метрики: expert_hit=%llu expert_miss=%llu expert_union=%.2f/pass "
      "passes=%llu]\n",
      static_cast<unsigned long long>(m.c_expert_hit.load()),
      static_cast<unsigned long long>(m.c_expert_miss.load()),
      passes != 0 ? static_cast<double>(union_sum) /
                        static_cast<double>(passes)
                  : 0.0,
      static_cast<unsigned long long>(passes));
  std::fprintf(
      stderr,
      "[время: attn=%.2fs (mm=%.2fs core=%.2fs) ffn=%.2fs expert=%.2fs "
      "load=%.2fs downwait=%.2fs exmm=%.2fs inline=%llu null=%llu fail=%llu "
      "ok=%llu dq=%llu dc=%llu dd=%llu dr=%llu "
      "dmj=%llu dms=%llu sidle=%llu "
      "router=%.2fs "
      "head=%.2fs embd=%.2fs shexp=%.2fs]\n",
      static_cast<double>(m.t_attn.load()) / 1e9,
      static_cast<double>(m.t_attn_mm.load()) / 1e9,
      static_cast<double>(m.t_attn_core.load()) / 1e9,
      static_cast<double>(m.t_ffn.load()) / 1e9,
      static_cast<double>(m.t_expert.load()) / 1e9,
      static_cast<double>(m.t_load.load()) / 1e9,
      static_cast<double>(m.t_down_wait.load()) / 1e9,
      static_cast<double>(m.t_expert_mm.load()) / 1e9,
      static_cast<unsigned long long>(m.c_inline.load()),
      static_cast<unsigned long long>(m.c_wait_null.load()),
      static_cast<unsigned long long>(m.c_read_fail.load()),
      static_cast<unsigned long long>(m.c_wait_ok.load()),
      static_cast<unsigned long long>(m.c_dem_queued.load()),
      static_cast<unsigned long long>(m.c_dem_cached.load()),
      static_cast<unsigned long long>(m.c_dem_dup.load()),
      static_cast<unsigned long long>(m.c_dem_range.load()),
      static_cast<unsigned long long>(m.c_dem_merged_jobs.load()),
      static_cast<unsigned long long>(m.c_dem_merged_slices.load()),
      static_cast<unsigned long long>(m.c_spec_idle_drop.load()),
      static_cast<double>(m.t_router.load()) / 1e9,
      static_cast<double>(m.t_head.load()) / 1e9,
      static_cast<double>(m.t_embd.load()) / 1e9,
      static_cast<double>(m.t_shexp.load()) / 1e9);
}

// компактная строка про экспертов: сколько уникальных на слой и попадания
// compact expert line: unique experts per layer pass and cache hits
void print_expert_stats(const tokshit::Metrics& m) {
  const uint64_t passes = m.c_expert_passes.load();
  const uint64_t union_sum = m.c_expert_union.load();
  const uint64_t hit = m.c_expert_hit.load();
  const uint64_t miss = m.c_expert_miss.load();
  std::fprintf(
      stderr,
      "[expert-stats: union=%.2f/pass passes=%llu hit=%llu miss=%llu "
      "hit_rate=%.1f%%]\n",
      passes != 0 ? static_cast<double>(union_sum) /
                        static_cast<double>(passes)
                  : 0.0,
      static_cast<unsigned long long>(passes),
      static_cast<unsigned long long>(hit),
      static_cast<unsigned long long>(miss),
      hit + miss != 0
          ? 100.0 * static_cast<double>(hit) /
                static_cast<double>(hit + miss)
          : 0.0);
}

// бенч matvec, читаем те же байты что и в бою
// matvec bench, same bytes as in the real run
void bench_mm(const tokshit::Model& model) {
  const std::size_t hidden = model.hidden();
  const std::size_t q_dim = model.n_heads() * model.head_dim();
  const std::size_t kv_dim = model.n_kv() * model.head_dim();

  std::vector<float> x(hidden);
  for (std::size_t i = 0; i < hidden; ++i) {
    x[i] = static_cast<float>((i * 37 + 11) % 101) / 50.0F - 1.0F;
  }
  std::vector<float> q_in(q_dim);
  for (std::size_t i = 0; i < q_dim; ++i) {
    q_in[i] = static_cast<float>((i * 17 + 3) % 97) / 48.0F - 1.0F;
  }
  std::vector<float> yq(q_dim, 0.0F);
  std::vector<float> yk(kv_dim, 0.0F);
  std::vector<float> yv(kv_dim, 0.0F);
  std::vector<float> yo(hidden, 0.0F);

  tokshit::pool::set_active(true);
  std::size_t passes = 5;
  if (const char* v = std::getenv("TOKSHIT_BENCH_PASSES"); v != nullptr && *v != '\0') {
    passes = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
  }
  for (std::size_t pass = 0; pass < passes; ++pass) {
    const auto t0 = tokshit::Clock::now();
    uint64_t bytes = 0;
    double tb[4] = {0, 0, 0, 0};
    double bb[4] = {0, 0, 0, 0};
    for (const tokshit::Layer& lt : model.layers()) {
      struct Item {
        const tokshit::Meta* meta;
        const std::vector<float>* input;
        std::vector<float>* output;
      };
      const Item items[4] = {
          {&lt.attn_q, &x, &yq},
          {&lt.attn_k, &x, &yk},
          {&lt.attn_v, &x, &yv},
          {&lt.attn_o, &q_in, &yo},
      };
      for (std::size_t ci = 0; ci < 4; ++ci) {
        const auto ti = tokshit::Clock::now();
        model.matvec(*items[ci].meta, *items[ci].input, *items[ci].output);
        tb[ci] += std::chrono::duration<double>(tokshit::Clock::now() - ti).count();
        bb[ci] += static_cast<double>(items[ci].meta->out_dim *
                                      items[ci].meta->rb);
      }
      bytes += static_cast<uint64_t>(lt.attn_q.out_dim * lt.attn_q.rb +
                                     lt.attn_k.out_dim * lt.attn_k.rb +
                                     lt.attn_v.out_dim * lt.attn_v.rb +
                                     lt.attn_o.out_dim * lt.attn_o.rb);
    }
    const double dt =
        std::chrono::duration<double>(tokshit::Clock::now() - t0).count();
    std::fprintf(stderr,
                 "[mmbench pass %zu] %.2f GB за %.3fs = %.2f GB/s rss=%llu MB\n",
                 pass, static_cast<double>(bytes) / 1e9, dt,
                 static_cast<double>(bytes) / dt / 1e9,
                 static_cast<unsigned long long>(tokshit::rss_mb()));
    if (std::getenv("TOKSHIT_BENCH_BREAKDOWN") != nullptr) {
      const char* names[4] = {"q", "k", "v", "o"};
      for (std::size_t i = 0; i < 4; ++i) {
        std::fprintf(stderr, "  %s: %.2f GB за %.3fs = %.2f GB/s\n", names[i],
                     bb[i] / 1e9, tb[i], bb[i] / (tb[i] > 1e-9 ? tb[i] : 1e-9) / 1e9);
      }
    }
  }
  tokshit::pool::set_active(false);
}

// путь по умолчанию частотного профиля: $TMPDIR/tokshit-freq-<ключ модели>.bin
// default frequency profile path: $TMPDIR/tokshit-freq-<model key>.bin
std::string freq_profile_path(const tokshit::Model& model) {
  const char* tmp = std::getenv("TMPDIR");
  const std::string base = tmp != nullptr && *tmp != '\0'
                               ? tmp
                               : "/data/data/com.termux/files/usr/tmp";
  return base + "/tokshit-freq-" + model.freq_key() + ".bin";
}

// TOKSHIT_FREQ_PERSIST=1: писать профиль частот роутинга после генерации.
// TOKSHIT_FREQ_FILE=<путь>: какой профиль грузить/писать; не задан — дефолтный
// TOKSHIT_FREQ_PERSIST=1: write the routing frequency profile after
// generation. TOKSHIT_FREQ_FILE=<path>: which profile to load/write; unset
// means the default one
struct FreqProfileCfg {
  bool persist = false;
  std::string path;
};

FreqProfileCfg freq_profile_cfg(const tokshit::Model& model) {
  FreqProfileCfg cfg;
  const char* v = std::getenv("TOKSHIT_FREQ_PERSIST");
  cfg.persist = v != nullptr && *v != '\0' && v[0] != '0';
  const char* p = std::getenv("TOKSHIT_FREQ_FILE");
  cfg.path = (p != nullptr && *p != '\0') ? p : freq_profile_path(model);
  return cfg;
}

// ── живые индикаторы (только tty-stderr, в пайпе тишина) ──────────────────
// Live indicators (tty-stderr only, silent in a pipe)

// «предобработка N/M ток»: одна строка, обновление по \r, перерисовка не
// чаще 4 раз в секунду
// "prefill N/M tok": one line, \r updates, redraw at most 4 times a second
struct PrefillBar {
  tokshit::Clock::time_point t0{};
  tokshit::Clock::time_point drawn{};

  void start() {
    t0 = tokshit::Clock::now();
    drawn = tokshit::Clock::time_point{};
  }

  void tick(std::size_t done, std::size_t total, std::size_t width, bool on) {
    if (!on || total == 0) return;
    const auto now = tokshit::Clock::now();
    if (drawn != tokshit::Clock::time_point{} &&
        now - drawn < std::chrono::milliseconds(250)) {
      return;
    }
    drawn = now;
    tokshit::tui::PrefillInfo info;
    info.done = done;
    info.total = total;
    info.seconds = std::chrono::duration<double>(now - t0).count();
    std::fprintf(stderr, "\r%s\x1b[K",
                 tokshit::tui::format_prefill(info, width).c_str());
    std::fflush(stderr);
  }

  // чистый erase строки перед ответом
  // clean erase of the line before the answer
  void finish(bool on) {
    if (!on) return;
    std::fputs("\r\x1b[K", stderr);
    std::fflush(stderr);
  }
};

// «◜ генерирую · N ток · X.X tok/s · кэш NN% · свб N MB».
// спиннер тикает по времени (120 мс), а не по токену: на декоде 2 tok/s
// покадровое обновление выглядит замершим; hit-rate берётся дельтой с
// начала генерации, MemAvailable — не чаще ~2 раз в секунду
// "◜ generating · N tok · X.X tok/s · cache NN% · free N MB". the spinner
// ticks on time (120 ms), not per token: at 2 tok/s a per-token frame looks
// frozen; the hit-rate is the delta since decode start, MemAvailable is
// refreshed at most twice a second
struct LiveDecode {
  tokshit::Clock::time_point t0{};
  std::uint64_t hit0 = 0;
  std::uint64_t miss0 = 0;
  std::uint64_t mem_mb = 0;
  tokshit::Clock::time_point mem_at{};

  void start() {
    t0 = tokshit::Clock::now();
    hit0 = tokshit::metrics().c_expert_hit.load();
    miss0 = tokshit::metrics().c_expert_miss.load();
  }

  void draw(std::size_t width, std::uint64_t tokens, bool on) {
    if (!on) return;
    const auto now = tokshit::Clock::now();
    if (mem_at == tokshit::Clock::time_point{} ||
        now - mem_at > std::chrono::milliseconds(500)) {
      mem_mb = tokshit::tui::mem_available_mb();
      mem_at = now;
    }
    tokshit::tui::DecodeInfo info;
    info.frame =
        static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     now - t0)
                                     .count() /
                                 120);
    info.tokens = tokens;
    info.seconds = std::chrono::duration<double>(now - t0).count();
    info.cache_hit = tokshit::metrics().c_expert_hit.load() - hit0;
    info.cache_miss = tokshit::metrics().c_expert_miss.load() - miss0;
    info.mem_avail_mb = mem_mb;
    std::fprintf(stderr, "\r\x1b[2m%s\x1b[0m\x1b[K",
                 tokshit::tui::format_decode(info, width).c_str());
    std::fflush(stderr);
  }
};

// Размер чанка батч-префилла. 0 отключает батч-путь и оставляет старый
// токен-за-токеном (для A/B). TOKSHIT_PREFILL_BATCH задаёт вручную.
// batched prefill chunk size. 0 keeps the old token-at-a-time path for A/B.
std::size_t prefill_batch_size() {
  const char* v = std::getenv("TOKSHIT_PREFILL_BATCH");
  if (v == nullptr || *v == '\0') return 32;
  const unsigned long long n = std::strtoull(v, nullptr, 10);
  return static_cast<std::size_t>(std::min<unsigned long long>(n, 512));
}

// Переносим готовый батч-kv в старый KVCache, чтобы декод дальше шёл
// неизменным путём. Слой хранится как [k0 v0 k1 v1 ...], по kv_dim float.
// copy the batched kv into the legacy KVCache so decode keeps its old path.
// a layer is [k0 v0 k1 v1 ...] with kv_dim floats per half-row
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
// batched prefill in chunks: weights once per chunk, experts by routing union;
// the last row of each chunk lands in `logits`
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
// single shot run: prompt -> generation -> metrics
void single_shot(tokshit::Model& model, const tokshit::BpeTokenizer& tok,
                 const std::string& prompt, std::size_t max_new,
                 const FreqProfileCfg& freq_cfg) {
  // индикаторы живут только на tty-stderr; в пайпе stdout/stderr чистые
  // indicators live on tty-stderr only; a pipe gets clean stdout/stderr
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
  // write it out
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
  // is the decode profile alone rather than prefill+decode mixed together
  const bool prof_gen_only = std::getenv("TOKSHIT_PROF") != nullptr;
  if (prof_gen_only) tokshit::metrics_reset();

  std::vector<uint32_t> gen_ids;
  const bool print_ids = std::getenv("TOKSHIT_PRINT_IDS") != nullptr;
  tokshit::Utf8Buffer utf8;
  tokshit::Rng rng;
  tokshit::ReasoningBudget reasoning(reasoning_budget_from_env());
  std::size_t n_gen = 0;
  const auto g0 = tokshit::Clock::now();
  for (;;) {
    const uint32_t sampled =
        tokshit::sample_utf8_safe(logits, params, tok, utf8, rng);
    // бюджет проверяем до печати и до forward: принудительный </think/>
    // проходит ровно тот же путь, что и обычный токен
    // check the budget before printing and forward: a forced </think/>
    // goes through exactly the same path as a regular token
    const uint32_t next = reasoning.constrain(sampled);
    if (tok.is_eog(next)) break;
    gen_ids.push_back(next);
    // <think/></think/> — служебные токены, их байты не текст ответа.
    // Thinking-блок скрываем целиком: пользователю нужен финальный ответ.
    // <think/></think/> are control tokens, not answer text. Hide the whole
    // thinking block; the user only wants the final answer.
    const bool control = next == tokshit::kThinkOpen ||
                         next == tokshit::kThinkClose;
    // thinking печатаем обычным текстом: иначе на медленном декоде
    // пользователь видит пустой экран, пока модель «думает»
    // print thinking as plain text: otherwise, at slow decode speed, the
    // user stares at an empty screen while the model is thinking
    const bool hidden = control;
    std::string text;
    if (control) {
      utf8 = tokshit::Utf8Buffer{};
    } else {
      text = utf8.push(tok.token_bytes(next));
    }
    if (!hidden && !text.empty()) {
      std::fwrite(text.data(), 1, text.size(), stdout);
      std::fflush(stdout);
    }
    ++n_gen;
    if (n_gen >= max_new) break;

    // периодически донашивать топ-эксперты в кэш: freq_ накапливается от
    // генерации, и каждый такой "pin" добавляет новые горячие эксперты
    // (без clear, лру сам выкинет холодные)
    if (const char* pv = std::getenv("TOKSHIT_PIN_EVERY");
        pv != nullptr && *pv != '\0') {
      const std::size_t pin_every =
          static_cast<std::size_t>(std::strtoull(pv, nullptr, 10));
      if (pin_every != 0 && n_gen % pin_every == 0) {
        const char* kv_pin = std::getenv("TOKSHIT_PIN_EXPERTS");
        const std::size_t pin_k =
            kv_pin != nullptr
                ? static_cast<std::size_t>(std::strtoull(kv_pin, nullptr, 10))
                : 200;
        model.pin_hot_experts(pin_k);
      }
    }

    const bool tok_dbg = std::getenv("TOKSHIT_TOKEN_DEBUG") != nullptr;
    const auto t_tok = tokshit::Clock::now();
    const uint64_t io_t0 = tok_dbg ? tokshit::io_read_bytes() : 0;
    const uint64_t ld_t0 = tok_dbg ? tokshit::metrics().t_load.load() : 0;
    const uint64_t mm_t0 = tok_dbg ? tokshit::metrics().t_expert_mm.load() : 0;
    const uint64_t dw_t0 = tokshit::metrics().t_down_wait.load();
    const uint64_t hit_t0 = tokshit::metrics().c_expert_hit.load();
    const uint64_t miss_t0 = tokshit::metrics().c_expert_miss.load();
    model.token_logits(next, kv, pos, logits);
    dump_logits(dump, logits);
    ++pos;
    // токен уже напечатан и попал в kv: обновляем состояние блока
    // the token is printed and in the kv already: update the block state
    reasoning.observe(next);
    if (tok_dbg) {
      const auto& m = tokshit::metrics();
      const double dt = std::chrono::duration<double>(
                            tokshit::Clock::now() - t_tok)
                            .count();
      std::fprintf(
          stderr,
          "[tok %zu] %.0fms io=%.1fMB load=%.0fms exmm=%.0fms dw=%.0fms "
          "hit=%llu miss=%llu\n",
          n_gen, dt * 1e3,
          static_cast<double>(tokshit::io_read_bytes() - io_t0) / 1e6,
          static_cast<double>(m.t_load.load() - ld_t0) / 1e6,
          static_cast<double>(m.t_expert_mm.load() - mm_t0) / 1e6,
          static_cast<double>(m.t_down_wait.load() - dw_t0) / 1e6,
          static_cast<unsigned long long>(m.c_expert_hit.load() - hit_t0),
          static_cast<unsigned long long>(m.c_expert_miss.load() - miss_t0));
    }
  }
  if (dump != nullptr) std::fclose(dump);
  const std::string tail = reasoning.in_think() ? std::string{} : utf8.flush();
  if (!tail.empty()) std::fwrite(tail.data(), 1, tail.size(), stdout);
  const double dt =
      std::chrono::duration<double>(tokshit::Clock::now() - g0).count();
  // стереть живую строку декода и закончить ответ на чистой строке
  // erase the live decode line and end the answer on a clean line
  if (err_tty) {
    std::fputs("\r\x1b[K", stderr);
    std::fflush(stderr);
  }
  std::fputc('\n', stdout);
  if (print_ids && !gen_ids.empty()) {
    std::fprintf(stderr, "[ids:");
    for (uint32_t id : gen_ids) {
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
    // prefill_hit/miss here underflows and reports a hit-rate above 100%.
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
// teacher-forced verify: run the draft over the reference token sequence
// produced by a full greedy run, and at every step compare the greedy draft
// token with the reference. this is the acceptance rate that decides whether
// speculation pays off; the established context is never repaired
void verify_ids(tokshit::Model& model, const tokshit::BpeTokenizer& tok,
                const std::string& prompt, const std::string& path,
                std::size_t cap) {
  std::vector<uint32_t> ref;
  {
    // парсер живучий: ест и сырую строку "[ids: 1 2 ...]" от TOKSHIT_PRINT_IDS,
    // и просто числа в столбик. нечисловой мусор перепрыгиваем
    // forgiving parser: eats the raw "[ids: 1 2 ...]" line from
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
  // avg run = accepted chunks per draft/verify round
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
// Inline REPL: без ncurses и без полного экрана. В TTY — цветные роли,
// стриминг по токену, строка статуса; в пайпе — обычный построчный читатель
// без ANSI. Управление: /help, /clear, /stats, /temp, /topk, /system, /save.
// Inline REPL, no ncurses, no full screen. In a TTY: colored roles, per-token
// streaming, a status line; in a pipe: a plain line reader without ANSI.
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
  const std::string text = tokshit::tui::format_status(st, false);
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
  const std::string text = tokshit::tui::format_stats(stats, false);
  std::fwrite(text.data(), 1, text.size(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

void clear_chat_context(ChatSession& s) {
  for (auto& layer : s.kv) layer.clear();
  s.pos = 0;
}

void chat_loop(tokshit::Model& model, const tokshit::BpeTokenizer& tok,
               const FreqProfileCfg& freq_cfg) {
  const bool tty = ::isatty(STDIN_FILENO) != 0 && ::isatty(STDOUT_FILENO) != 0;
  const bool err_tty = ::isatty(STDERR_FILENO) != 0;
  // ширина терминала для живых строк: на узких экранах хвост индикатора
  // отрезается, чтобы \r-перерисовка не ломалась
  // terminal width for live lines: on narrow screens the indicator tail is
  // trimmed so the \r redraw stays intact
  const std::size_t width = tokshit::tui::terminal_width(STDERR_FILENO);

  // SIGINT без SA_RESTART: getline и цикл генерации сами увидят флаг
  // SIGINT without SA_RESTART: getline and the decode loop both check the flag
  struct sigaction sa {};
  sa.sa_handler = on_sigint;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr);

  ChatSession s;
  s.kv.assign(model.n_layers(), std::vector<float>{});
  s.params = sample_params_from_env();
  s.cache_budget_mb = env_u64("TOKSHIT_EXPERT_CACHE_MB", 1024);
  s.reasoning_budget = reasoning_budget_from_env();
  tokshit::Rng rng;

  std::vector<float> logits(model.vocab(), 0.0F);
  FILE* dump = nullptr;
  if (const char* v = std::getenv("TOKSHIT_DUMP_LOGITS");
      v != nullptr && *v != '\0') {
    dump = std::fopen(v, "wb");
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
    // NOTE: the core is built with -fno-exceptions. a stream failure here
    // makes libstdc++ call std::terminate, acceptable for the cli (eof is the
    // normal exit path), so keep it contained to this loop
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
      // Ctrl-D/EOF: a clean exit, no crash and no hang
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
        // the utf8 buffer lives for one turn, so it is already empty once
        // the context is cleared; in a TTY the screen is cleared as well
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

    const std::string text = cmd.arg;
    if (text.empty()) continue;

    tokshit::pool::set_active(true);
    std::vector<uint32_t> turn;
    turn.push_back(kUser);
    if (!s.system_prompt.empty()) {
      for (uint32_t id : tok.encode(s.system_prompt + "\n\n")) {
        turn.push_back(id);
      }
    }
    for (uint32_t id : tok.encode(text)) turn.push_back(id);
    turn.push_back(kAssistant);

    const auto tp = tokshit::Clock::now();
    {
      PrefillBar bar;
      bar.start();
      std::size_t done = 0;
      for (uint32_t id : turn) {
        model.token_logits(id, s.kv, s.pos, logits);
        dump_logits(dump, logits);
        ++s.pos;
        ++done;
        bar.tick(done, turn.size(), width, err_tty);
      }
      bar.finish(err_tty);
    }
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
    tokshit::Utf8Buffer utf8;
    std::string answer;
    tokshit::ReasoningBudget reasoning(s.reasoning_budget);
    std::size_t n_gen = 0;
    const auto g0 = tokshit::Clock::now();
    for (;;) {
      if (g_interrupt != 0) {
        break;
      }
      const uint32_t sampled =
          tokshit::sample_utf8_safe(logits, s.params, tok, utf8, rng);
      const uint32_t next = reasoning.constrain(sampled);
      if (tok.is_eog(next)) break;
      // Не печатаем служебные <think/></think/> и скрываем thinking-блок.
      // Do not print the <think/></think/> control tokens and hide thinking.
      const bool control = next == tokshit::kThinkOpen ||
                           next == tokshit::kThinkClose;
      const bool hidden = control;
      std::string chunk;
      if (control) {
        utf8 = tokshit::Utf8Buffer{};
      } else {
        chunk = utf8.push(tok.token_bytes(next));
      }
      if (!hidden && !chunk.empty()) {
        std::fwrite(chunk.data(), 1, chunk.size(), stdout);
        answer.append(chunk);
        std::fflush(stdout);
      }
      ++n_gen;
      model.token_logits(next, s.kv, s.pos, logits);
      dump_logits(dump, logits);
      ++s.pos;
      reasoning.observe(next);
    }
    const std::string tail = reasoning.in_think() ? std::string{} : utf8.flush();
    if (!tail.empty()) {
      std::fwrite(tail.data(), 1, tail.size(), stdout);
      answer.append(tail);
    }
    const double dt =
        std::chrono::duration<double>(tokshit::Clock::now() - g0).count();
    if (err_tty) std::fputs("\r\x1b[K", stderr);
    if (g_interrupt != 0) {
      g_interrupt = 0;
      if (tty) {
        std::fputs(" [прервано]", stdout);
      } else {
        std::fputs(" [прервано]", stdout);
      }
    }
    std::fputc('\n', stdout);
    std::fflush(stdout);

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

}  // namespace

namespace {

// единственный источник правды о флагах — tokshit::tui::flag_docs(): по нему
// печатается --help и разбираются аргументы, так что новый флаг нельзя
// добавить без описания
// the single source of truth for flags is tokshit::tui::flag_docs(): --help
// and the parser both read it, so a flag cannot be added without docs
const char* env_for_flag(const std::string& name) {
  for (const tokshit::tui::FlagDoc& f : tokshit::tui::flag_docs()) {
    if (name == f.flag) return f.env;
  }
  return nullptr;
}

// булевы флаги не съедают следующий аргумент: --spec "model.gguf" должен
// трактовать model.gguf как позиционный, а не как значение
// boolean flags do not consume the next arg: --spec "model.gguf" must treat
// model.gguf as positional, not as a value
bool flag_is_bool(const std::string& name) {
  static const char* const kBool[] = {
      "hot",           "hot-conv",       "o-direct",       "drop-cache",
      "prefetch",      "prefetch-idle",  "io-merge",       "freq-persist",
      "mmap-willneed", "fadv-dontneed",  "fadv-random",    "pool-pin",
      "pool-rt",       "pool-stats",     "print-ids",
      "spec",          "bench-mm",       "bench-breakdown", "debug-tok",
      "seg-debug",     "mlock",          "expert-stats",   "fast",
      "spec-fast",
  };
  for (const char* b : kBool) {
    if (name == b) return true;
  }
  return false;
}

void print_help() {
  const std::string text = tokshit::tui::help_text(default_model_path());
  std::fwrite(text.data(), 1, text.size(), stdout);
  std::fflush(stdout);
}

// пресеты для быстрых замеров. это просто набор флагов, ничего скрытого:
// fast — обычный декод top-2 через page cache, spec-fast — то же плюс
// спекуляция с урезанным verify-роутером
// presets for quick measurements. they are just flag bundles, nothing
// hidden: fast is plain top-2 decode through the page cache, spec-fast adds
// speculation with a cut verify router
bool apply_preset(const std::string& name) {
  if (name != "fast" && name != "spec-fast") return false;
  setenv("TOKSHIT_O_DIRECT", "0", 1);
  setenv("TOKSHIT_EXPERT_CACHE_MB", "0", 1);
  setenv("TOKSHIT_DRAFT_TOPK", "2", 1);
  setenv("TOKSHIT_DROP_CACHE", "0", 1);
  unsetenv("TOKSHIT_PREFETCH");
  // --fast — это именно обычный декод, поэтому спекуляцию гасим, даже если
  // она пришла из окружения. --spec-fast включает её обратно
  // --fast means plain decode, so turn speculation off even if the
  // environment had it. --spec-fast turns it back on
  unsetenv("TOKSHIT_SPEC");
  if (name == "spec-fast") {
    setenv("TOKSHIT_SPEC", "1", 1);
    setenv("TOKSHIT_SPEC_K", "4", 1);
    setenv("TOKSHIT_SPEC_MAXB", "8", 1);
    setenv("TOKSHIT_SPEC_VERIFY_TOPK", "2", 1);
  }
  return true;
}

// пресеты из окружения: справка обещает TOKSHIT_FAST/TOKSHIT_SPEC_FAST, так
// что они должны работать и без флага. вызываем до разбора argv, чтобы явные
// флаги могли переопределить отдельные значения пресета
// env presets: --help advertises TOKSHIT_FAST/TOKSHIT_SPEC_FAST, so they must
// work without flags. called before argv parsing so explicit flags can override
// individual preset values
void apply_env_presets() {
  const auto enabled = [](const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
  };
  if (enabled("TOKSHIT_SPEC_FAST")) {
    apply_preset("spec-fast");
  } else if (enabled("TOKSHIT_FAST")) {
    apply_preset("fast");
  }
}

// разбираем флаги, выставляем TOKSHIT_* и оставляем только позиционные
// аргументы. так все существующие getenv-чтения продолжают работать, а
// пользователь получает нормальный --cli
// parse flags, export TOKSHIT_* and keep only positional args. existing
// getenv reads keep working while users get a normal --cli
void apply_flags(int& argc, char** argv) {
  // static: argv указывает на эти строки и после возврата из функции
  // static: argv points here and must outlive this function
  static std::vector<std::string> pos;
  pos.clear();
  pos.reserve(static_cast<std::size_t>(argc));
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      print_help();
      std::exit(0);
    }
    // "--" дальше не флаги: промпт может начинаться с дефиса
    // "--" ends flag parsing: a prompt may start with a dash
    if (a == "--") {
      for (++i; i < argc; ++i) pos.emplace_back(argv[i]);
      break;
    }
    if (a.rfind("--", 0) != 0) {
      pos.push_back(std::move(a));
      continue;
    }
    std::string name;
    std::string val;
    bool has_val = false;
    const std::size_t eq = a.find('=');
    if (eq != std::string::npos) {
      name = a.substr(2, eq - 2);
      val = a.substr(eq + 1);
      has_val = true;
    } else {
      name = a.substr(2);
    }
    if (!has_val && !flag_is_bool(name) && i + 1 < argc) {
      std::string nx = argv[i + 1];
      if (nx.rfind("--", 0) != 0) {
        val = std::move(nx);
        has_val = true;
        ++i;
      }
    }
    if (!has_val) val = "1";
    if (apply_preset(name)) continue;
    const char* env = env_for_flag(name);
    if (env == nullptr) {
      std::fprintf(stderr,
                   "неизвестный флаг --%s\n"
                   "список флагов и примеры: tokshit --help\n",
                   name.c_str());
      std::exit(2);
    }
    setenv(env, val.c_str(), 1);
  }
  argc = 1 + static_cast<int>(pos.size());
  for (std::size_t i = 0; i < pos.size(); ++i) argv[i + 1] = pos[i].data();
  argv[argc] = nullptr;
}

}  // namespace

// точка входа cli. путь к модели, промпт, сколько генерить
// cli entry point. model path, prompt, how many tokens to generate
int main(int argc, char** argv) {
  apply_env_presets();
  apply_flags(argc, argv);
  std::string path = argc > 1 ? argv[1] : default_model_path();

  const auto t0 = tokshit::Clock::now();
  tokshit::ModelOptions options;
  if (const char* v = std::getenv("TOKSHIT_EXPERT_CACHE_MB");
      v != nullptr && *v != '\0') {
    options.expert_cache_mb = std::strtoull(v, nullptr, 10);
  }
  // плотные веса по умолчанию копируем в ram: иначе каждый токен фолтит их
  // из mmap, а на андроиде это ещё и война с reclaim. TOKSHIT_HOT=0 выключает
  // dense weights are copied to ram by default: otherwise every token faults
  // them from the mmap, and on android that is a fight with reclaim.
  // TOKSHIT_HOT=0 turns it off. ModelOptions::hot defaults to true, so the
  // explicit "0" must clear it instead of simply leaving the default alone.
  const char* hot_env = std::getenv("TOKSHIT_HOT");
  options.hot = !(hot_env != nullptr && *hot_env == '0');
  options.mmap_willneed = std::getenv("TOKSHIT_MMAP_WILLNEED") != nullptr;

  auto loaded = tokshit::Model::load(path, options);
  if (!loaded.ok()) {
    std::fprintf(stderr,
                 "не удалось загрузить модель (%s)\n"
                 "  путь: %s\n"
                 "  ошибка: %s\n"
                 "  подсказка: укажите путь к .gguf или задайте TOKSHIT_MODEL\n",
                 std::string(tokshit::to_string(loaded.error().code)).c_str(),
                 path.c_str(), loaded.error().message.c_str());
    return 1;
  }
  tokshit::Model model = std::move(loaded.value());
  if (options.hot) {
    model.collect_hot();
    // после копии плотных весов page cache от mmap бесполезен: эксперты
    // читаются мимо него. сбрасываем, чтобы отдать память кэшу экспертов
    // once the dense weights are copied the mmap page cache is useless: experts
    // bypass it. drop it so the memory goes to the expert cache instead
    if (const char* v = std::getenv("TOKSHIT_DROP_CACHE");
        v == nullptr || *v != '0') {
      model.gguf().fadvise_dontneed(0, model.gguf().file_len());
    }
  }

  // частотный профиль: грузим до первого токена, чтобы pin_hot_experts
  // ставил горячих экспертов в кэш сразу (кэш тёплый с первого же токена).
  // TOKSHIT_FREQ_FILE задаёт путь; не задан — дефолтный, если он существует
  // frequency profile: loaded before the first token so pin_hot_experts pins
  // the hot experts right away (warm cache from the very first token).
  // TOKSHIT_FREQ_FILE sets the path; unset means the default one, when it
  // exists
  const FreqProfileCfg freq_cfg = freq_profile_cfg(model);
  {
    struct stat st {};
    if (::stat(freq_cfg.path.c_str(), &st) == 0 &&
        model.load_freq_profile(freq_cfg.path)) {
      std::fprintf(stderr, "[freq] профиль загружен: %s\n",
                   freq_cfg.path.c_str());
      // ранний pin по загруженному профилю: только когда профиль реально
      // есть, иначе freq пуст и пинить нечего
      // early pin from the loaded profile: only when the profile really
      // exists, otherwise freq is empty and there is nothing to pin
      if (const char* v = std::getenv("TOKSHIT_PIN_EXPERTS");
          v != nullptr && *v != '\0') {
        const std::size_t k =
            static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        if (k != 0) model.pin_hot_experts(k);
      }
    }
  }
  // draft-режим: роутер отбирает меньше экспертов сверху. замер acceptance:
  // один и тот же промпт с TOKSHIT_DRAFT_TOPK=1 и без, жадный семплинг, общий
  // префикс идов = сколько токенов черновик живёт в согласии с полным
  // top values. TOKSHIT_TOPK=0 leaves the model config alone. acceptance
  // measurement: same prompt with TOKSHIT_DRAFT_TOPK=1 and without it, greedy,
  // the common id prefix = how far the draft agrees with the full model
  if (const char* v = std::getenv("TOKSHIT_DRAFT_TOPK");
      v != nullptr && *v != '\0') {
    const uint64_t k = std::strtoull(v, nullptr, 10);
    if (k != 0 && k < model.n_used()) {
      model.set_router_topk(static_cast<std::size_t>(k));
      std::fprintf(stderr, "[tokshit] draft top-k = %llu из %llu\n",
                   static_cast<unsigned long long>(k),
                   static_cast<unsigned long long>(model.n_used()));
    }
  }

  auto tok_result = tokshit::build_tokenizer(model.gguf());
  if (!tok_result.ok()) {
    std::fprintf(stderr, "tokenizer: %s: %s\n",
                 tokshit::to_string(tok_result.error().code).data(),
                 tok_result.error().message.c_str());
    return 1;
  }
  tokshit::BpeTokenizer tok = std::move(tok_result.value());
  std::fprintf(stderr, "loaded in %.3fs\n",
               std::chrono::duration<double>(tokshit::Clock::now() - t0).count());

  if (std::getenv("TOKSHIT_BENCH_MM") != nullptr) {
    bench_mm(model);
    return 0;
  }

  // verify-режим: draft teacher-forced по референсным идам, замер acceptance.
  // ид-файл — это строка "[ids: ...]" от прогона с TOKSHIT_PRINT_IDS=1, числа
  // читаются как whitespace-separated
  // the reference file is the "[ids: ...]" line from an TOKSHIT_PRINT_IDS=1 run;
  // the ids parse as whitespace separated numbers
  if (const char* v = std::getenv("TOKSHIT_VERIFY_IDS");
      v != nullptr && *v != '\0') {
    verify_ids(model, tok, argc > 2 ? argv[2] : "hi", v,
               argc > 3
                   ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10))
                   : 0);
    return 0;
  }

  const char* prompt_file = std::getenv("TOKSHIT_PROMPT_FILE");
  const bool has_prompt_file = prompt_file != nullptr && *prompt_file != '\0';
  bool have_prompt = argc > 2;
  if (have_prompt || has_prompt_file) {
    const std::size_t max_new =
        argc > 3 ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10))
                 : 12;
    std::string prompt = have_prompt ? argv[2] : "";
    // длинный промпт (2К контекста) в argv не влезет — читаем из файла
    // a long prompt (2k context) does not fit argv, read it from a file
    if (has_prompt_file) {
      if (FILE* f = std::fopen(prompt_file, "rb"); f != nullptr) {
        prompt.clear();
        char buf[8192];
        std::size_t got = 0;
        while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) {
          prompt.append(buf, got);
        }
        std::fclose(f);
        have_prompt = true;
        std::fprintf(stderr, "[prompt: %zu bytes из %s]\n", prompt.size(),
                     prompt_file);
      } else {
        std::fprintf(stderr,
                     "не удалось прочитать --prompt-file %s\n",
                     prompt_file);
        if (!have_prompt) return 1;
      }
    }
    if (have_prompt) {
      // спекулятивный/батч-путь с новым kv-кешем, если просили
      // speculative/batched path with the new kv cache, when asked
      if (const char* sv = std::getenv("TOKSHIT_SPEC");
          sv != nullptr && *sv != '\0' && std::strcmp(sv, "0") != 0) {
        tokshit::spec::run(model, tok, prompt, max_new);
        return 0;
      }
      single_shot(model, tok, prompt, max_new, freq_cfg);
      return 0;
    }
  }
  chat_loop(model, tok, freq_cfg);
  return 0;
}
