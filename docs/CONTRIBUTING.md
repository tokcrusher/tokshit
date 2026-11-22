# Как участвовать

`tokshit` — специализированный ARM64-рантайм. Изменения должны сохранять
горячий путь быстрым, тесты зелёными, а файлы моделей — нетронутыми.

## 1. Правила безопасности

### Модель только для чтения

- `.gguf` никогда не изменяется, не перезаписывается, не обрезается, не
  переименовывается и не перемещается.
- Разрешено только чтение: `O_RDONLY`, `PROT_READ`, `qstat`, `kbench`,
  `safe_run.sh`.
- Запрещены `>`, `>>`, `tee`, `truncate`, `dd of=`, `cp` поверх, `mv`
  поверх, `rm`, `sed -i`, `chmod +w` по отношению к модели.
- Для экспериментов создаётся копия в отдельном каталоге.
- Проверка: `scripts/model_guard.sh <model.gguf>`; `--hash` считает sha256
  только на чтение.

### Память и запуск

- На целевом телефоне ~15.2 ГБ RAM, свободно обычно 5–7 ГБ. OOM приводил к
  перезагрузке.
- Один процесс модели за раз: `pgrep -x tokshit` должен быть пуст.
- Всегда через watchdog:

  ```sh
  TOKSHIT_MIN_AVAIL_MB=1200 scripts/safe_run.sh build/tokshit ...
  ```

- Общий бюджет AHB — 5 ГиБ (`TOKSHIT_AHB_MAX_MB`, по умолчанию 5120). Его
  делят hot-веса и кэш экспертов. `--cache-mb` не больше 4096.
- `mlock` бесполезен: `ulimit -l` = 64 КБ.
- Перед замером `MemAvailable` > 4000 МБ.

## 2. Сборка и тесты

```sh
cmake --preset release
cmake --build build --parallel 2
build/tokshit_tests
```

`./install.sh` проверяет тулчейн, собирает Release/Ninja и ставит бинарь в
`$PREFIX/bin` на Termux или `/usr/local/bin` на Linux. Текущая линия —
186 кейсов / 9951 проверка.

Полезные опции CMake:

| опция | что делает |
|---|---|
| `TOKSHIT_BUILD_TOOLS` | собрать `qstat`, `kbench`, `vcheck`, `bwbench`, `pbench`, `ahbtest`, `densebench` |
| `TOKSHIT_NEON` | NEON-ядра на aarch64 |
| `TOKSHIT_DOTPROD` | dotprod-варианты |
| `TOKSHIT_NO_EXCEPTIONS` | `-fno-exceptions -fno-rtti` |

## 3. Git-hook и коммиты

В репозитории включён `core.hooksPath=.githooks`. Pre-commit блокирует:

- `.gguf` и файлы моделей;
- файлы больше 5 МБ;
- приватные ключи в staged-диффе.

Коммит:

```sh
git diff --check
git add -A
git commit -m "docs: ..."
```

Стиль:

- один коммит — одна тема;
- сообщение в императиве, с префиксом (`docs:`, `perf:`, `fix:`, `test:`);
- не коммитить `build/`, логи, `freq.log`, скриншоты, модели;
- перед коммитом `git diff --check` (пробелы в конце строк).

## 4. Worktree

Для параллельных тем используйте отдельные worktree и ветки:

```sh
git worktree add ../tokshit-wt-docs -b docs/topic main
cd ../tokshit-wt-docs
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

Правила:

- одна тема — один worktree — одна ветка;
- не переиспользовать `build/` между ветками;
- не делать `git checkout` в чужом worktree;
- merge — из основного чекаута.

## 5. Стиль кода

- C++20, пространство имён `tokshit`.
- Без исключений и RTTI в ядре: ошибки через `Result<T>`.
- Горячий путь не аллоцирует: буферы в `Workspace`.
- Байты — `std::span<const std::byte>` или `const uint8_t*` + размер.
- Комментарии — по делу, на русском или английском; не пересказывать код.
- Новое NEON-ядро принимается только вместе со скалярным тестом.
- `-ffast-math` не включается: меняет `exp`/`softmax`.
- Не добавлять внешние зависимости, кроме libc/pthreads и doctest.

## 6. Стиль документации

- README — карта и быстрый старт, не простыня.
- Одна тема — один документ в `docs/`.
- Короткие абзацы, таблицы, блоки кода; ширина строки ~80–100 колонок.
- Ссылки только относительные и на существующие файлы.
- Числа сопровождаются условиями: устройство, кэш, I/O, температура.
- Идеи помечаются как идеи, а не как результат.
- `--turbo`, `--spec-fast`, `--topk-experts` и `--passes 1/2` всегда
  помечаются **LOSSY**.
- `--dspark` — экспериментальный greedy-путь; без замера не выдавать его
  за ускорение. DSpark требует `--passes 3`.

Проверка ссылок и shell-скриптов перед коммитом:

```sh
scripts/check_links.sh
bash -n scripts/*.sh
```

Внешние ссылки не используются: все ссылки в README и `docs/` —
относительные и ведут на существующие файлы.

## 7. Что проверить перед PR

- [ ] `cmake --preset release && cmake --build build --parallel 2`
- [ ] `git diff --check`
- [ ] нет битых ссылок и устаревших названий
- [ ] нет `.gguf`, `build/`, секретов в диффе
- [ ] числа tok/s с условиями замера
- [ ] LOSSY-режимы помечены
