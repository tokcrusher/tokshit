#include "cli.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "tokshit/tui.hpp"

namespace tokshit::cli {

namespace {

std::string home_dir() {
  const char* home = std::getenv("HOME");
  return (home != nullptr && *home != '\0')
             ? std::string(home)
             : std::string("/data/data/com.termux/files/home");
}

bool file_exists(const std::string& p) {
  struct stat st {};
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string resolve_model_arg(const std::string& v) {
  if (v.find('/') != std::string::npos) return v;
  const std::string h = home_dir();
  if (v == "inkling" || v == "inkling2" || v == "inkling-mini") {
    return h + "/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf";
  }
  if (v == "flint") return h + "/models/flint-UD-Q3_K_XL.gguf";
  if (v == "dspark") {
    return h + "/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf";
  }
  if (v == "mmproj") return h + "/Inkling-2-Mini-GGUF/mmproj-F16.gguf";
  const std::string dirs[] = {h + "/Inkling-2-Mini-GGUF", h + "/models", h};
  for (const std::string& d : dirs) {
    const std::string direct = d + "/" + v;
    if (file_exists(direct)) return direct;
    const std::string gguf = direct + ".gguf";
    if (file_exists(gguf)) return gguf;
  }
  return v;
}

void print_models() {
  const std::string h = home_dir();
  const std::string dirs[] = {h + "/Inkling-2-Mini-GGUF", h + "/models", h};
  std::printf("available .gguf models:\n");
  std::size_t n = 0;
  for (const std::string& d : dirs) {
    DIR* dir = ::opendir(d.c_str());
    if (dir == nullptr) continue;
    while (dirent* e = ::readdir(dir)) {
      const std::string name = e->d_name;
      if (name.size() < 6 ||
          name.compare(name.size() - 5, 5, ".gguf") != 0) {
        continue;
      }
      const std::string full = d + "/" + name;
      struct stat st {};
      if (::stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
      std::printf("  %-46s %7.2f GB  %s\n", name.c_str(),
                  static_cast<double>(st.st_size) / 1e9, full.c_str());
      ++n;
    }
    ::closedir(dir);
  }
  if (n == 0) {
    std::printf("  (none found in %s/Inkling-2-Mini-GGUF, %s/models)\n",
                h.c_str(), h.c_str());
  }
  std::printf(
      "\nuse:  tokshit --model <name|path> [prompt] [max_new]\n"
      "      names: inkling, flint, dspark, mmproj, or a .gguf basename\n");
}

}  // namespace

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
    if (a == "--models") {
      print_models();
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
    if (name == "model") val = resolve_model_arg(val);
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
