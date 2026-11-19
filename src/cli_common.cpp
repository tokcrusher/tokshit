#include "cli.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "env.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"
#include "tokshit/pool.hpp"
#include "tokshit/sampling.hpp"
#include "tokshit/tui.hpp"

namespace tokshit::cli {

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
void dump_logits(FILE* f, const std::vector<float>& logits) {
  if (f == nullptr) return;
  const auto count = static_cast<uint32_t>(logits.size());
  std::fwrite(&count, sizeof(count), 1, f);
  std::fwrite(logits.data(), sizeof(float), logits.size(), f);
}

// короткая разбивка по времени

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

std::string freq_profile_path(const tokshit::Model& model) {
  const char* tmp = std::getenv("TMPDIR");
  const std::string base = tmp != nullptr && *tmp != '\0'
                               ? tmp
                               : "/data/data/com.termux/files/usr/tmp";
  return base + "/tokshit-freq-" + model.freq_key() + ".bin";
}

// TOKSHIT_FREQ_PERSIST=1: писать профиль частот роутинга после генерации.
// TOKSHIT_FREQ_FILE=<путь>: какой профиль грузить/писать; не задан — дефолтный.
FreqProfileCfg freq_profile_cfg(const tokshit::Model& model) {
  FreqProfileCfg cfg;
  const char* v = std::getenv("TOKSHIT_FREQ_PERSIST");
  cfg.persist = v != nullptr && *v != '\0' && v[0] != '0';
  const char* p = std::getenv("TOKSHIT_FREQ_FILE");
  cfg.path = (p != nullptr && *p != '\0') ? p : freq_profile_path(model);
  return cfg;
}

}  // namespace tokshit::cli
