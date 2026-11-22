#include "cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tokshit/tui.hpp"

namespace tokshit::cli {

// единственный источник правды о флагах — tokshit::tui::flag_docs(): по нему
// печатается --help и разбираются аргументы, так что новый флаг нельзя
// добавить без описания
const char* env_for_flag(const std::string& name) {
  for (const tokshit::tui::FlagDoc& f : tokshit::tui::flag_docs()) {
    if (name == f.flag) return f.env;
  }
  return nullptr;
}

// булевы флаги не съедают следующий аргумент: --spec "model.gguf" должен
// трактовать model.gguf как позиционный, а не как значение
bool flag_is_bool(const std::string& name) {
  static const char* const kBool[] = {
      "hot",           "hot-conv",       "o-direct",       "drop-cache",
      "drop-siblings",  "prefault",
      "prefetch",      "prefetch-idle",  "io-merge",       "freq-persist",
      "mmap-willneed", "fadv-dontneed",  "fadv-random",    "pool-pin",
      "pool-rt",       "pool-stats",     "print-ids",
      "spec",          "bench-mm",       "bench-breakdown", "debug-tok",
      "seg-debug",     "mlock",          "expert-stats",
      "spec-fast",     "turbo",
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

// Пресеты — просто наборы флагов. Оба LOSSY: turbo меняет квант dense и
// число экспертов, spec-fast урезает черновик и verify.
bool apply_preset(const std::string& name) {
  if (name == "turbo") {
    setenv("TOKSHIT_DRAFT_TOPK", "2", 1);
    setenv("TOKSHIT_HOT_Q4", "1", 1);
    setenv("TOKSHIT_O_DIRECT", "1", 1);
    setenv("TOKSHIT_EXPERT_CACHE_MB", "1024", 1);
    setenv("TOKSHIT_IO_THREADS", "8", 1);
    setenv("TOKSHIT_DROP_CACHE", "0", 1);
    unsetenv("TOKSHIT_SPEC");
    unsetenv("TOKSHIT_PREFETCH");
    return true;
  }
  if (name != "spec-fast") return false;
  setenv("TOKSHIT_O_DIRECT", "0", 1);
  setenv("TOKSHIT_EXPERT_CACHE_MB", "0", 1);
  setenv("TOKSHIT_DRAFT_TOPK", "2", 1);
  setenv("TOKSHIT_DROP_CACHE", "0", 1);
  unsetenv("TOKSHIT_PREFETCH");
  setenv("TOKSHIT_SPEC", "1", 1);
  setenv("TOKSHIT_SPEC_K", "4", 1);
  setenv("TOKSHIT_SPEC_MAXB", "8", 1);
  setenv("TOKSHIT_SPEC_VERIFY_TOPK", "2", 1);
  return true;
}

// Пресеты из окружения работают и без флага. Вызываются до разбора argv,
// чтобы явные флаги могли переопределить отдельные значения.
void apply_env_presets() {
  const auto enabled = [](const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
  };
  if (enabled("TOKSHIT_TURBO")) {
    apply_preset("turbo");
  } else if (enabled("TOKSHIT_SPEC_FAST")) {
    apply_preset("spec-fast");
  }
}

// разбираем флаги, выставляем TOKSHIT_* и оставляем только позиционные
// аргументы. так все существующие getenv-чтения продолжают работать, а
// пользователь получает нормальный --cli
void apply_flags(int& argc, char** argv) {
  // static: argv указывает на эти строки и после возврата из функции
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

}  // namespace tokshit::cli
