# Ядра и диспетчеризация

Горячий путь `tokshit` — это специализированный ARM64-рантайм: NEON
dot-product, фиксированные раскладки блоков и минимум ветвлений на токен.
Скалярный путь остаётся эталоном для тестов и fallback для платформ без
`FEAT_DotProd`.

Смежное: [QUANTIZATION.md](QUANTIZATION.md) — форматы блоков;
[RUNTIME.md](RUNTIME.md) — как ядра встроены в forward;
[PERFORMANCE.ru.md](PERFORMANCE.ru.md) — замеры.

## 1. Активация в Q8

Все matvec-ядра принимают активацию, заранее разложенную на блоки по 32:

```cpp
struct ActQ8 {
  std::vector<int8_t> q;      // по одному int8 на вес
  std::vector<float> scales;  // один float на блок 32
  size_t len = 0;
};

ActQ8 quantize(std::span<const float> x);
void quantize_into(std::span<const float> x, ActQ8& act);
```

Активация квантуется один раз на вход матвека и переиспользуется всеми
сегментами. Это убирает аллокации из `layer()` и `token_logits()`.

Идея скалярного произведения:

```
dot ≈ Σ_blocks  scale_act[b] * Σ_{i∈block} int8_weight[i] * int8_act[i]
```

Для `Q8_0` вес уже `int8`, поэтому внутренняя сумма — целочисленная. Для
`Q4_0` нибблы распаковываются в `int8` (со сдвигом на 8) и сразу идут в
`vdotq_s32`. Для IQ индексы смотрят в таблицу-решётку, а знаки применяются
отдельным слоем.

## 2. Скалярный эталон

`dot_fallback` — скалярный эталон, с которым сверяются NEON-ядра:

1. деквант строки в `float` через `dequant_into`;
2. скалярное произведение с активацией;
3. при необходимости — повторное использование `scratch`.

Он медленный, но повторяет математику один в один. Любое NEON-ядро
принимается только после сравнения с ним на реальных блоках. Допуск в
тестах — `1e-3`; для Q8_0 на практике расхождение меньше.

## 3. Диспетчеризация

Выбор ядра происходит в `dot_row`:

```cpp
float dot_row(QuantType quant, std::span<const uint8_t> row, size_t rb,
              std::span<const float> x, const ActQ8& act, size_t in_dim,
              std::vector<float>& scratch) {
  if (have_dotprod() && row.size() >= rb) {
    switch (quant) {
      case QuantType::Q8_0:
        if (in_dim % 32 == 0) return neon::dot_q8_0(row.data(), act, in_dim);
        break;
      case QuantType::Iq4Xs:
        if (in_dim % 256 == 0) return neon::dot_iq4xs(row.data(), act, in_dim);
        break;
      // ...
      default: break;
    }
  }
  return dot_fallback(quant, row, rb, x, act, in_dim, scratch);
}
```

Свойства:

- `have_dotprod()` один раз читает `getauxval(AT_HWCAP)` и кэширует
  `HWCAP_ASIMDDP`;
- нет виртуальных вызовов: только `switch` по кванту;
- проверка кратности входа — часть контракта ядра, а не «на всякий случай»;
- если NEON-файл не собран, `dot_row` сразу уходит в fallback.

NEON-код лежит в отдельной единице трансляции
`src/compute/kernels_neon_dotprod.cpp` и компилируется с
`-march=armv8.2-a+dotprod+fp16+lse`. Инструкции `vdotq_s32` не попадают в
базовый бинарь и не требуют `__attribute__((target))` с разным синтаксисом
у GCC и Clang.

## 4. Какие кванты ускоряются

| квант | NEON-ядро | вход | примечание |
|---|---|---:|---|
| `Q8_0` | `dot_q8_0` | % 32 | 4 независимых аккумулятора |
| `Q4_0` | `dot_q4_0` | % 32 | распаковка нибблов |
| `IQ4_XS` | `dot_iq4xs` | % 256 | `vqtbl1q_s8` по `kvalues_iq4nl` |
| `IQ2_S` | `dot_iq2s` | % 256 | решётка + знаки, 4 цепочки |
| `IQ3_S` | `dot_iq3s` | % 256 | `iq3s_grid`, маски знаков |
| `IQ3_XXS` | `dot_iq3xxs` | % 256 | `iq3xxs_grid`, `ksigns_iq2xs` |
| `Q4_K` | `dot_q4k` | % 256 | opt-in `TOKSHIT_KQK=1`, экспериментальный |

`Q5_K`, `Q6_K` и `BF16` идут через fallback. Для плотных весов это
исправляется переквантованием в `Q8_0` при загрузке
(`TOKSHIT_HOT_CONV`, см. [QUANTIZATION.md](QUANTIZATION.md)).

### i8mm/SMMLA

На ARMv9 доступна матричная инструкция `SMMLA` (`vmmlaq_s32`): два MAC за
инструкцию против одного у `SDOT`. Для `Q8_0` есть двухстрочный путь
(`TOKSHIT_I8MM=1`, при наличии `HWCAP_I8MM`): активация грузится один раз,
две независимые цепочки закрывают латентность. На bandwidth-bound декоде
выигрыш в пределах шума. Путь выбирается в `dot_rows`: `TOKSHIT_I8MM=1` и
фича в `getauxval` → SMMLA, иначе dot-product.

## 5. Батчевый путь

`dot_row_multi` применяет одну строку весов к нескольким активациям:

```cpp
void dot_row_multi(QuantType quant, std::span<const uint8_t> row, size_t rb,
                   const ActQ8* acts, size_t n, size_t in_dim, float* out,
                   std::vector<float>& scratch);
```

Если для кванта есть NEON-ядро, строка остаётся в L1 и прогоняется `n` раз.
Если ядра нет, строка один раз деквантится в `float`, после чего считается
`n` скалярных произведений. Это использует спекулятивный verify-батч и
префилл.

## 6. Пул и сегменты

Матвек описывается структурой `Seg`:

```cpp
struct Seg {
  std::span<const uint8_t> bytes;  // веса
  size_t rb;                       // байт на строку
  QuantType quant;
  size_t in_dim;
  const float* x;                  // вход
  const ActQ8* act;                // квантованная активация
  float* out;
  size_t rows;
};
```

`run_segs` режет сегменты на задачи по 16 или 32 строки и отправляет их в
`TaskPool`. Главный поток тоже берёт задачи через `next.fetch_add`, поэтому
барьер не простаивает. Подробности пула — в [RUNTIME.md](RUNTIME.md).

## 7. Замеренная полоса

Микробенчмарки на целевом телефоне (8 ядер, Android/Termux):

| что | ГБ/с |
|---|---:|
| `Q8_0`, отдельные сегменты, `kbench` | 25–35 |
| `IQ4_XS`, отдельные сегменты | 18–26 |
| `IQ3_S` | 8.9 |
| `IQ3_XXS` | 7.5 |
| `IQ2_S` | 7.5 |
| поток hot-весов 1.74 ГБ через AHB, `rare` | 5.21 |
| поток hot-весов, `often` | 3.10 |
| `malloc`, блоки 328 КБ | 21.7 |
| AHB, блоки 328 КБ | 25.7 |

Сегменты и полный поток — разные вещи: декод читает 52 сегмента по 32 МБ,
поэтому видит ~5.2 ГБ/с, а не 25–35. Именно поэтому оптимизация
отдельных ядер уже не двигает tok/s полного декода; дальнейшие шаги —
в [ROADMAP.md](ROADMAP.md).

## 8. Как добавить ядро

1. Реализовать `dequant_*` в `src/dequant.cpp`, если кванта ещё нет.
2. Добавить скалярный путь в `dot_fallback`.
3. Написать NEON-версию в `kernels_neon_dotprod.cpp` с проверкой кратности.
4. Сверить со скалярным эталоном: случайный блок, `d = 1.0`,
   сравнение с `dequant_into` и допуск `1e-3`.
5. Включить квант в `has_neon_dot` и в `switch` диспетчера.
6. Прогнать `vcheck` на реальном GGUF и `kbench` на нужном тензоре.
7. Только после этого пускать ядро в горячий путь.

Правило неизменно: новое ядро без скалярного сравнения в горячий путь не
попадает.

## 9. Диагностика

```sh
# один тензор, 20 итераций
build/kbench ~/models/flint-UD-Q3_K_XL.gguf 20 blk.2.attn_q.weight

# сверка ядра с деквантом на конкретном тензоре
build/vcheck ~/models/flint-UD-Q3_K_XL.gguf blk.2.attn_q.weight

# масштабирование пула
build/pbench ~/models/flint-UD-Q3_K_XL.gguf 20 blk.2.attn_q.weight
```

`--bench-mm` печатает `[matvec: ... GB/s]` по q/k/v/o attention-слоя;
`--bench-breakdown` добавляет разбивку по этим четырём тензорам.
