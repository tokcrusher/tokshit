#include "tokshit/tui.hpp"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>
#include <string_view>

#include "tui_internal.hpp"

namespace tokshit::tui {

using detail::append_line;
using detail::append_wrapped;
using detail::columns;
using detail::pad_to;
using detail::to_lower;
using detail::trim;
using detail::wrap;

Command parse_command(std::string_view line) {
  Command cmd;
  std::string_view raw = line;
  while (!raw.empty() && (raw.back() == '\r' || raw.back() == '\n')) {
    raw.remove_suffix(1);
  }
  const std::string_view t = trim(raw);
  if (t.empty()) return cmd;
  if (t.front() != '/') {
    const std::string lower = to_lower(t);
    if (lower == "exit" || lower == "quit") {
      cmd.kind = CommandKind::Quit;
      return cmd;
    }
    cmd.kind = CommandKind::Chat;
    cmd.arg.assign(raw);
    return cmd;
  }

  std::size_t sp = 1;
  while (sp < t.size() && t[sp] != ' ' && t[sp] != '\t') ++sp;
  const std::string name = to_lower(t.substr(1, sp - 1));
  const std::string_view rest = trim(t.substr(sp));
  cmd.arg.assign(rest);

  if (name.empty()) {
    cmd.kind = CommandKind::Unknown;
    return cmd;
  }
  if (name == "help" || name == "?") {
    cmd.kind = CommandKind::Help;
    return cmd;
  }
  if (name == "exit" || name == "quit" || name == "q") {
    cmd.kind = CommandKind::Quit;
    return cmd;
  }
  if (name == "clear" || name == "cls") {
    cmd.kind = CommandKind::Clear;
    return cmd;
  }
  if (name == "stats") {
    cmd.kind = CommandKind::Stats;
    return cmd;
  }
  if (name == "reset") {
    cmd.kind = CommandKind::Reset;
    return cmd;
  }
  if (name == "system" || name == "sys") {
    cmd.kind = CommandKind::System;
    return cmd;
  }
  if (name == "save") {
    cmd.kind = CommandKind::Save;
    if (rest.empty()) cmd.valid = false;
    return cmd;
  }
  if (name == "temp" || name == "temperature") {
    cmd.kind = CommandKind::Temp;
    if (!rest.empty()) {
      char* end = nullptr;
      const std::string tmp(rest);
      const float v = std::strtof(tmp.c_str(), &end);
      if (end != nullptr && static_cast<std::size_t>(end - tmp.c_str()) == tmp.size()) {
        cmd.number = v;
      } else {
        cmd.valid = false;
      }
    } else {
      cmd.valid = false;
    }
    return cmd;
  }
  if (name == "topk" || name == "top-k") {
    cmd.kind = CommandKind::TopK;
    if (!rest.empty() && rest.front() != '-') {
      char* end = nullptr;
      const std::string tmp(rest);
      const unsigned long long v = std::strtoull(tmp.c_str(), &end, 10);
      if (end != nullptr && static_cast<std::size_t>(end - tmp.c_str()) == tmp.size()) {
        cmd.count = static_cast<std::size_t>(v);
      } else {
        cmd.valid = false;
      }
    } else {
      cmd.valid = false;
    }
    return cmd;
  }
  if (name == "topp" || name == "top-p") {
    cmd.kind = CommandKind::TopP;
    if (!rest.empty()) {
      char* end = nullptr;
      const std::string tmp(rest);
      const float v = std::strtof(tmp.c_str(), &end);
      if (end != nullptr && static_cast<std::size_t>(end - tmp.c_str()) == tmp.size()) {
        cmd.number = v;
      } else {
        cmd.valid = false;
      }
    } else {
      cmd.valid = false;
    }
    return cmd;
  }
  if (name == "reasoning-budget") {
    cmd.kind = CommandKind::ReasoningBudget;
    if (!rest.empty()) {
      char* end = nullptr;
      errno = 0;
      const std::string tmp(rest);
      const long v = std::strtol(tmp.c_str(), &end, 10);
      const bool whole =
          end != nullptr &&
          static_cast<std::size_t>(end - tmp.c_str()) == tmp.size();
      // допустимы только -1, 0 и положительные значения
      // only -1, 0 and positive values are valid
      if (whole && errno != ERANGE && v >= -1 && v <= INT_MAX) {
        cmd.budget = static_cast<int>(v);
      } else {
        cmd.valid = false;
      }
    } else {
      cmd.valid = false;
    }
    return cmd;
  }
  cmd.kind = CommandKind::Unknown;
  cmd.arg = name;
  return cmd;
}

std::string command_help() {
  std::string out;
  append_line(out, "  /help              эта справка по командам");
  append_line(out, "  /exit, /quit       выйти (или Ctrl-D)");
  append_line(out, "  /clear             очистить контекст (KV), в TTY — и экран");
  append_line(out, "  /stats             скорость, кэш, память, сессия, параметры");
  append_line(out, "  /reset             вернуть сэмплинг к умолчаниям");
  append_line(out, "  /temp <x>          температура; 0 = жадный argmax");
  append_line(out, "  /topk <n>          top-k; 0 = без отсечения");
  append_line(out, "  /topp <x>          top-p; 1 = выключено");
  append_line(out,
              "  /reasoning-budget N  thinking: N>0 кап, 0 закрыть сразу, -1 без лимита");
  append_line(out, "  /system <текст>    системная инструкция: текст подставляется");
  append_line(out, "                     в начало следующего сообщения (пусто = убрать)");
  append_line(out, "  /save <файл>       дописать диалог в файл");
  append_line(out, "  пустая строка      ничего не делает");
  append_line(out, "  Ctrl-C             прервать ответ, не выходить из чата");
  append_line(out, "  Ctrl-D             аккуратный выход");
  out.push_back('\n');
  append_line(out, "  примеры:");
  append_line(out, "    /temp 0.2              мягкий сэмплинг вместо жадного");
  append_line(out, "    /topk 40               выбор из 40 кандидатов");
  append_line(out, "    /reasoning-budget 0    отвечать без длинных размышлений");
  append_line(out, "    /system отвечай кратко  задать системную инструкцию");
  append_line(out, "    /stats                  проверить скорость и hit-rate кэша");
  return out;
}
}  // namespace tokshit::tui
