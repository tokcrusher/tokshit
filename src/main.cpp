#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

#include "tokshit/model.hpp"
#include "tokshit/spec_loop.hpp"
#include "tokshit/tokenizer.hpp"

#include "cli.hpp"

namespace cli = tokshit::cli;

int main(int argc, char** argv) {
  cli::apply_env_presets();
  cli::apply_flags(argc, argv);
  std::string path = argc > 1 ? argv[1] : cli::default_model_path();

  const auto t0 = tokshit::Clock::now();
  tokshit::ModelOptions options;
  if (const char* v = std::getenv("TOKSHIT_EXPERT_CACHE_MB");
      v != nullptr && *v != '\0') {
    options.expert_cache_mb = std::strtoull(v, nullptr, 10);
  }
  // Плотные веса копируются в RAM по умолчанию: mmap на каждом токене
  // фолтит их заново и воюет с reclaim. TOKSHIT_HOT=0 оставляет mmap.
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
    // После копии плотных весов page cache от mmap не нужен: эксперты
    // читаются мимо него. Сбрасываем, чтобы отдать память кэшу экспертов.
    // Для inkling2 плотные веса остаются в mmap — cache не трогаем.
    if (!model.is_inkling2()) {
      if (const char* v = std::getenv("TOKSHIT_DROP_CACHE");
          v == nullptr || *v != '0') {
        model.gguf().fadvise_dontneed(0, model.gguf().file_len());
      }
    }
  }

  // Профиль частот грузится до первого токена, чтобы pin_hot_experts сразу
  // поставил горячих экспертов в кэш.
  const cli::FreqProfileCfg freq_cfg = cli::freq_profile_cfg(model);
  {
    struct stat st {};
    if (::stat(freq_cfg.path.c_str(), &st) == 0 &&
        model.load_freq_profile(freq_cfg.path)) {
      std::fprintf(stderr, "[freq] профиль загружен: %s\n",
                   freq_cfg.path.c_str());
      if (const char* v = std::getenv("TOKSHIT_PIN_EXPERTS");
          v != nullptr && *v != '\0') {
        const std::size_t k =
            static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        if (k != 0) model.pin_hot_experts(k);
      }
    }
  }

  // Draft-режим: роутер отбирает меньше экспертов. TOKSHIT_DRAFT_TOPK=0
  // оставляет конфиг модели.
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
    cli::bench_mm(model);
    return 0;
  }

  // Teacher-forced verify: сравнить draft с эталонными id и посчитать acceptance.
  if (const char* v = std::getenv("TOKSHIT_VERIFY_IDS");
      v != nullptr && *v != '\0') {
    cli::verify_ids(model, tok, argc > 2 ? argv[2] : "hi", v,
                    argc > 3 ? static_cast<std::size_t>(
                                   std::strtoull(argv[3], nullptr, 10))
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
        std::fprintf(stderr, "не удалось прочитать --prompt-file %s\n",
                     prompt_file);
        if (!have_prompt) return 1;
      }
    }
    if (have_prompt) {
      if (const char* sv = std::getenv("TOKSHIT_SPEC");
          sv != nullptr && *sv != '\0' && std::strcmp(sv, "0") != 0) {
        tokshit::spec::run(model, tok, prompt, max_new);
        return 0;
      }
      cli::single_shot(model, tok, prompt, max_new, freq_cfg);
      return 0;
    }
  }
  cli::chat_loop(model, tok, freq_cfg);
  return 0;
}
