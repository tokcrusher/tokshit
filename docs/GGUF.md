# GGUF: чтение модели

`tokshit` читает GGUF v2 и v3. Заголовок разбирается один раз, таблица
тензоров превращается в смещения, а веса остаются в файле: либо в `mmap`,
либо читаются через `pread` в собственный буфер. Запись в файл модели не
предусмотрена.

Смежные темы:

- раскладка блоков и деквант — [QUANTIZATION.md](QUANTIZATION.md);
- загрузка модели и горячие веса — [RUNTIME.md](RUNTIME.md);
- общая карта модулей — [ARCHITECTURE.ru.md](ARCHITECTURE.ru.md).

## 1. Раскладка файла

```
┌──────────────────────────────────────────────────────────────┐
│ "GGUF"            4 байта                                    │
│ version           u32, поддерживаются 2 и 3                  │
│ tensor_count      u64                                        │
│ kv_count          u64                                        │
├──────────────────────────────────────────────────────────────┤
│ metadata          kv_count пар: ключ, тип, значение          │
├──────────────────────────────────────────────────────────────┤
│ tensor infos      tensor_count записей                       │
│   name            gguf-строка                                │
│   n_dims          u32, 1..4                                  │
│   dims[n_dims]    u64                                        │
│   type            u32, код GGML                              │
│   offset          u64, относительно начала данных            │
├──────────────────────────────────────────────────────────────┤
│ padding           до general.alignment, по умолчанию 32      │
├──────────────────────────────────────────────────────────────┤
│ tensor data       блоки весов                                │
└──────────────────────────────────────────────────────────────┘
```

Все целые — little-endian. Отдельного разбора big-endian нет: целевые
платформы (aarch64, x86_64) little-endian, и GGUF-файлы в проекте такие же.
Чтение идёт через `load_le16/32/64`, поэтому код не зависит от выравнивания
полей в памяти.

## 2. Типы метаданных

Поддерживаются все типы GGUF:

| код | тип | код | тип |
|---:|---|---:|---|
| 0 | `uint8` | 7 | `bool` |
| 1 | `int8` | 8 | `string` |
| 2 | `uint16` | 9 | `array` |
| 3 | `int16` | 10 | `uint64` |
| 4 | `uint32` | 11 | `int64` |
| 5 | `int32` | 12 | `float64` |
| 6 | `float32` | | |

`Value` — `std::variant`; массив хранится как
`std::shared_ptr<const std::vector<Value>>`. Это делает значение копируемым
и не ломает рекурсию типа. Строки проверяются на корректный UTF-8, массивы
и строки имеют верхние пределы (`10M` элементов, `64 МиБ` на строку), чтобы
битый файл не вызвал бесконечную аллокацию.

## 3. Таблица тензоров

```cpp
struct GgufTensor {
  std::string name;
  std::vector<uint64_t> dims;  // dims[0] — непрерывная ось
  QuantType quant = QuantType::F32;
  uint64_t offset = 0;         // от data_offset
  uint64_t byte_len = 0;       // вычисляется из numel и блока
};
```

Порядок размерностей GGUF: `dims[0]` (`ne0`) — самая быстрая ось, по ней
идёт вход матвека. Для матрицы `out_dim × in_dim` это `dims[0] = in_dim`,
`dims[1] = out_dim`; третья ось — номер эксперта.

`byte_len` не берётся из файла: он считается как

```
numel   = dims[0] * dims[1] * ... * dims[n-1]
blocks  = numel / block_spec(quant).weights
byte_len = blocks * block_spec(quant).bytes
```

Поэтому файл с числом элементов, не кратным размеру блока, отклоняется
кодом `QuantBlockMismatch`, а не читается «наполовину».

## 4. Смещения и выравнивание

`offset` в записи тензора относителен начала данных. Абсолютный адрес:

```
data_offset = align_up(header_end, general.alignment)
abs_offset  = data_offset + tensor.offset
```

`general.alignment` должен быть степенью двойки; по умолчанию 32. Если
выравнивание выходит за границы файла или тензор не помещается в
`[data_offset, file_len)`, загрузка возвращает `OutOfBounds` ещё до
обращения к весам.

## 5. Два способа доступа

### mmap

`Mmap::map_file(fd, len)` делает `mmap(PROT_READ, MAP_PRIVATE)`. Дальше
`Model` держит `std::span<const std::byte>` и строит `Meta` со смещениями
внутри отображения. Это дефолтный путь для холодных тензоров и для режима
`TOKSHIT_HOT=0`.

`madvise`/`mlock` — best-effort: Android может отклонить вызов, и это не
считается ошибкой.

| метод | вызов | назначение |
|---|---|---|
| `advise_willneed()` | `MADV_WILLNEED` | прогреть всё отображение |
| `advise_range()` | `MADV_WILLNEED` | прогреть диапазон |
| `advise_cold()` | `MADV_COLD` | отдать страницы под reclaim |
| `lock_range()` | `mlock` | попытка закрепить, часто отказ |

### pread

`Gguf::read_at(offset, dst)` и `read_into(name, off, buf)` читают напрямую
в буфер вызывающего. Этим пользуется кэш экспертов: блок попадает сразу в
слот AHB/malloc, без page fault на `mmap` и лишней копии. `pread` повторяется
на `EINTR`, короткое чтение превращается в `Truncated`.

## 6. Проверки при загрузке

Парсер не доверяет ни одному полю:

- magic `GGUF`, иначе `BadMagic`;
- версия только 2 или 3, иначе `UnsupportedVersion`;
- `tensor_count` и `kv_count` не больше 10 млн;
- `n_dims` от 1 до 4;
- код кванта известен, иначе `UnsupportedQuant`;
- `numel` не переполняет `uint64` и кратен размеру блока;
- `byte_len` не переполняет `uint64`;
- `data_offset` не больше длины файла;
- каждый тензор целиком лежит в данных;
- строки — валидный UTF-8;
- выравнивание — степень двойки.

Ошибка содержит код, поле и сообщение. Коды перечислены в
`include/tokshit/error.hpp`; горячий путь ошибок не создаёт.

## 7. Ключи модели

Загрузчик читает архитектурные ключи `flint.*`:

| ключ | смысл |
|---|---|
| `flint.block_count` | число блоков |
| `flint.leading_dense_block_count` | сколько первых блоков плотные |
| `flint.embedding_length` | hidden |
| `flint.attention.head_count` | q-головы |
| `flint.attention.head_count_kv` | kv-головы |
| `flint.attention.key_length` | head_dim |
| `flint.attention.sliding_window` | окно |
| `flint.attention.sliding_window_pattern` | маска слоёв 0/1 |
| `flint.attention.rope_pattern` | где применять RoPE |
| `flint.rope.freq_base` | rope theta |
| `flint.attention.layer_norm_rms_epsilon` | epsilon RMSNorm |
| `flint.expert_count`, `flint.expert_used_count` | эксперты и top-k |
| `flint.feed_forward_length` | ширина плотного FFN |
| `flint.expert_feed_forward_length` | ширина эксперта |
| `flint.expert_shared_feed_forward_length` | ширина shared expert |

Токенизатор читает `tokenizer.ggml.tokens`, `merges`, `token_type`,
`bos_token_id`, `eos_token_id`, `eog_token_ids`; подробности —
в [TOKENIZER.md](TOKENIZER.md).

## 8. Ошибки и диагностика

`Gguf::open` возвращает `Result<Gguf>`. Типичные сообщения:

```text
BadMagic             не GGUF
UnsupportedVersion   версия не 2 и не 3
UnsupportedQuant     неизвестный код кванта
QuantBlockMismatch   numel не кратен блоку
OutOfBounds          тензор или data_offset за файлом
Truncated            короткое чтение
```

Для быстрой проверки файла без запуска модели есть `qstat`:

```sh
build/qstat ~/models/flint-UD-Q3_K_XL.gguf --list attn_q
```

## 9. Правила безопасности

- `.gguf` открывается только на чтение (`O_RDONLY`, `PROT_READ`).
- Код не содержит операций записи в файл модели.
- Проверить, что файл не writable, можно локальным
  `scripts/model_guard.sh <model.gguf>`; хеш считается только на чтение.
- Для экспериментов используется копия, а не оригинал.
