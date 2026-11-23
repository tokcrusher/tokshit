#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tokshit/model.hpp"
#include "tokshit/spec_loop.hpp"
#include "tokshit/tokenizer.hpp"

#include "cli.hpp"

namespace cli = tokshit::cli;

namespace {

// Drop clean page-cache pages of sibling .gguf files (mmproj, other quants).
// The text runtime does not need them, and on a 15 GB phone every cached
// byte competes with the 9.14 GB target model.
void drop_sibling_gguf(const std::string& model_path) {
  const std::size_t slash = model_path.find_last_of('/');
  const std::string dir = slash == std::string::npos
                              ? std::string(".")
                              : model_path.substr(0, slash);
  const std::string self = slash == std::string::npos
                               ? model_path
                               : model_path.substr(slash + 1);
  DIR* d = ::opendir(dir.c_str());
  if (d == nullptr) return;
  std::size_t dropped = 0;
  while (dirent* e = ::readdir(d)) {
    const std::string name = e->d_name;
    if (name.size() <= 5 ||
        name.compare(name.size() - 5, 5, ".gguf") != 0 || name == self) {
      continue;
    }
    const std::string full = dir + "/" + name;
    const int fd = ::open(full.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) continue;
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    ++dropped;
  }
  ::closedir(d);
  if (dropped != 0) {
    std::fprintf(stderr, "[tokshit] dropped page cache of %zu sibling .gguf\n",
                 dropped);
  }
}

}  // namespace

int main(int argc, char** argv) {
  cli::apply_env_presets();
  cli::apply_flags(argc, argv);
  // --model / TOKSHIT_MODEL: the positional argv[1] is the prompt, not the
  // model. Without it, argv[1] is the model (legacy form). This keeps
  // `tokshit --model inkling "hi" 16` and `tokshit model.gguf "hi" 16` both
  // correct instead of silently treating the prompt as a model path.
  const char* model_env = std::getenv("TOKSHIT_MODEL");
  const bool model_from_flag = model_env != nullptr && *model_env != '\0';
  std::string path = model_from_flag
                         ? std::string(model_env)
                         : (argc > 1 ? argv[1] : cli::default_model_path());
  const int pbase = model_from_flag ? 1 : 2;

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

  if (const char* v = std::getenv("TOKSHIT_DROP_SIBLINGS");
      v != nullptr && *v != '\0' && v[0] != '0') {
    drop_sibling_gguf(path);
  }

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

  if (const char* bt = std::getenv("TOKSHIT_INK_BATCH_TEST");
      bt != nullptr && *bt != '\0') {
    const std::size_t group =
        argc > pbase + 1
            ? static_cast<std::size_t>(std::strtoull(argv[pbase + 1], nullptr, 10))
            : 3;
    cli::ink_batch_debug(model, tok, argc > pbase ? argv[pbase] : "hi", group);
    return 0;
  }

  if (const char* draft_path = std::getenv("TOKSHIT_DSPARK_DEBUG");
      draft_path != nullptr && *draft_path != '\0') {
    const std::size_t steps =
        argc > pbase + 1
            ? static_cast<std::size_t>(std::strtoull(argv[pbase + 1], nullptr, 10))
            : 16;
    cli::dspark_debug(model, tok, argc > pbase ? argv[pbase] : "hi", steps,
                      draft_path);
    return 0;
  }

  // Teacher-forced verify: сравнить draft с эталонными id и посчитать acceptance.
  if (const char* v = std::getenv("TOKSHIT_VERIFY_IDS");
      v != nullptr && *v != '\0') {
    cli::verify_ids(model, tok, argc > pbase ? argv[pbase] : "hi", v,
                    argc > pbase + 1
                        ? static_cast<std::size_t>(
                              std::strtoull(argv[pbase + 1], nullptr, 10))
                        : 0);
    return 0;
  }

  const char* prompt_file = std::getenv("TOKSHIT_PROMPT_FILE");
  const bool has_prompt_file = prompt_file != nullptr && *prompt_file != '\0';
  bool have_prompt = argc > pbase;
  if (have_prompt || has_prompt_file) {
    const std::size_t max_new =
        argc > pbase + 1
            ? static_cast<std::size_t>(std::strtoull(argv[pbase + 1], nullptr, 10))
            : 12;
    std::string prompt = have_prompt ? argv[pbase] : "";
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
      if (const char* dp = std::getenv("TOKSHIT_DSPARK");
          dp != nullptr && *dp != '\0') {
        float temperature = 0.0F;
        if (const char* tv = std::getenv("TOKSHIT_TEMPERATURE");
            tv != nullptr && *tv != '\0') {
          temperature = std::strtof(tv, nullptr);
        }
        if (temperature > 1e-6F) {
          std::fprintf(stderr,
                       "[dspark] только greedy; выставьте --temperature 0\n");
        } else if (tokshit::spec::run_dspark(model, tok, prompt, max_new, dp)) {
          return 0;
        } else {
          std::fprintf(stderr,
                       "[dspark] не запустился; обычный decode\n");
        }
      }
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
