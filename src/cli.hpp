#pragma once

#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tokshit/metrics.hpp"
#include "tokshit/model.hpp"
#include "tokshit/sampling.hpp"
#include "tokshit/tokenizer.hpp"
#include "tokshit/tui.hpp"

namespace tokshit::cli {

struct FreqProfileCfg {
  bool persist = false;
  std::string path;
};

// Опции одного прохода генерации. max_new = SIZE_MAX означает «до EOG».
struct DecodeOptions {
  std::size_t max_new = static_cast<std::size_t>(-1);
  bool token_debug = false;
  bool pin_every = false;
  bool check_interrupt = false;
};

struct DecodeResult {
  std::size_t generated = 0;
  double seconds = 0.0;
  std::vector<uint32_t> ids;
  std::string text;
  bool interrupted = false;
};

extern volatile std::sig_atomic_t g_interrupt;
void install_sigint();

// Собирает ids хода. Для Flint — старый формат USER/text/ASSISTANT, для
// inkling2 — шаблон Thinking Machines с effort-заголовком.
[[nodiscard]] std::vector<uint32_t> build_turn(const Model& model,
                                               const BpeTokenizer& tok,
                                               std::string_view user,
                                               std::string_view system);

DecodeResult decode(Model& model, const BpeTokenizer& tok, KVCache& kv,
                    std::size_t& pos, std::vector<float>& logits,
                    const SampleParams& params, Rng& rng, int reasoning_budget,
                    FILE* dump, const DecodeOptions& options);

// Живой прогресс префилла. Рисуется только на tty, обновляется не чаще 4 Гц.
struct PrefillBar {
  Clock::time_point t0{};
  Clock::time_point drawn{};

  void start() {
    t0 = Clock::now();
    drawn = Clock::time_point{};
  }

  void tick(std::size_t done, std::size_t total, std::size_t width, bool on) {
    if (!on || total == 0) return;
    const auto now = Clock::now();
    if (drawn != Clock::time_point{} &&
        now - drawn < std::chrono::milliseconds(250)) {
      return;
    }
    drawn = now;
    tui::PrefillInfo info;
    info.done = done;
    info.total = total;
    info.seconds = std::chrono::duration<double>(now - t0).count();
    std::fprintf(stderr, "\r%s\x1b[K",
                 tui::format_prefill(info, width).c_str());
    std::fflush(stderr);
  }

  void finish(bool on) {
    if (!on) return;
    std::fputs("\r\x1b[K", stderr);
    std::fflush(stderr);
  }
};

std::string default_model_path();
int reasoning_budget_from_env();
SampleParams sample_params_from_env();

std::string freq_profile_path(const Model& model);
FreqProfileCfg freq_profile_cfg(const Model& model);

std::size_t prefill_batch_size();
void dump_logits(FILE* f, const std::vector<float>& logits);
void print_timing_short(const Metrics& m);
void print_timing_full(const Metrics& m);
void print_expert_stats(const Metrics& m);
void bench_mm(const Model& model);

void single_shot(Model& model, const BpeTokenizer& tok, const std::string& prompt,
                 std::size_t max_new, const FreqProfileCfg& freq_cfg);
void chat_loop(Model& model, const BpeTokenizer& tok,
               const FreqProfileCfg& freq_cfg);
void verify_ids(Model& model, const BpeTokenizer& tok, const std::string& prompt,
                const std::string& path, std::size_t cap);
void dspark_debug(Model& model, const BpeTokenizer& tok,
                  const std::string& prompt, std::size_t steps,
                  const std::string& draft_path);

// Debug parity check: batched Inkling2 logits/taps против последовательных
// на одном и том же префиксе.
void ink_batch_debug(Model& model, const BpeTokenizer& tok,
                     const std::string& prompt, std::size_t group);

void print_help();
void apply_env_presets();
void apply_flags(int& argc, char** argv);

}  // namespace tokshit::cli
