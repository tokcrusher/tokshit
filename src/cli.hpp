#pragma once

#include <chrono>
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

void print_help();
void apply_env_presets();
void apply_flags(int& argc, char** argv);

}  // namespace tokshit::cli
