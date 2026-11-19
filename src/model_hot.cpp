#include "tokshit/model.hpp"
#include "tokshit/ahb.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/kernels.hpp"

namespace tokshit {

// копируем все неэкспертные веса в отдельный vector чтобы их не вытеснял
// поток экспертов из page cache. на андроиде может не спасти (своп)
void Model::collect_hot() {
  std::vector<Meta*> list;
  list.reserve(n_layers_ * 11 + 1);
  for (Layer& lt : layers_) {
    list.push_back(&lt.attn_q);
    list.push_back(&lt.attn_k);
    list.push_back(&lt.attn_v);
    list.push_back(&lt.attn_o);
    if (lt.ffn_gate) list.push_back(&*lt.ffn_gate);
    if (lt.ffn_up) list.push_back(&*lt.ffn_up);
    if (lt.ffn_down) list.push_back(&*lt.ffn_down);
    if (lt.gate_inp) list.push_back(&*lt.gate_inp);
    if (lt.gate_shexp) list.push_back(&*lt.gate_shexp);
    if (lt.up_shexp) list.push_back(&*lt.up_shexp);
    if (lt.down_shexp) list.push_back(&*lt.down_shexp);
  }
  list.push_back(&out_);

  const auto mm_bytes = mm_.bytes();
  const auto* mm_data = reinterpret_cast<const uint8_t*>(mm_bytes.data());

  // Плотные веса кладём в AHB, а не в анонимный vector: под давлением
  // Android отправляет анонимные страницы в zram, и matvec начинает платить
  // за распаковку на каждом токене. dma-buf не свопится.
  //
  // Заодно переквантовываем в Q8_0 всё, для чего нет neon-ядра (Q5K/Q6K/
  // Bf16). Скалярный dequant+dot на этих весах даёт 1-1.6 ГБ/с, а q8_0 -
  // 30+. Это +1.5 бита на вес и ~0.6 ГБ памяти на флинте, но плотные веса
  // читаются на каждом токене, так что обмен того стоит.
  const char* hc = std::getenv("TOKSHIT_HOT_CONV");
  const bool conv_on = hc == nullptr || hc[0] != '0';
  const bool conv_fast = kernels::has_neon_dot(QuantType::Q8_0);
  const char* hq = std::getenv("TOKSHIT_HOT_Q4");
  const bool q4_on = hq != nullptr && hq[0] != '\0' && hq[0] != '0';
  const bool q4_fast = kernels::has_neon_dot(QuantType::Q4_0);
  struct Dest {
    Meta* meta = nullptr;
    std::size_t rb = 0;
    std::size_t src = 0;
    std::size_t bytes = 0;
    QuantType target = QuantType::F32;
    bool conv = false;
  };
  std::vector<Dest> dest;
  dest.reserve(list.size());
  std::size_t total = 0;
  std::size_t src_total = 0;
  std::size_t n_conv_q4 = 0;
  std::size_t n_conv_q8 = 0;
  std::size_t q4_bytes = 0;
  for (Meta* m : list) {
    const std::size_t src = m->out_dim * m->rb;
    Dest d;
    d.meta = m;
    d.rb = m->rb;
    d.src = src;
    d.bytes = src;
    d.target = m->quant;
    QuantType target = m->quant;
    bool want = false;
    // Turbo: everything dense except the BF16 router goes to Q4_0. The router
    // stays on the Q8_0 conversion path because a coarser router would change
    // expert selection, not just the activation precision.
    if (q4_on && q4_fast && m->quant != QuantType::Bf16 &&
        m->quant != QuantType::Q4_0 && m->out_dim != 0 &&
        m->in_dim % 32 == 0) {
      target = QuantType::Q4_0;
      want = true;
    } else if (conv_on && conv_fast && !kernels::has_neon_dot(m->quant) &&
               m->out_dim != 0 && m->in_dim % 32 == 0) {
      target = QuantType::Q8_0;
      want = true;
    }
    if (want) {
      const auto qb = row_bytes(target, m->in_dim);
      if (qb.ok() && qb.value() != 0) {
        d.rb = static_cast<std::size_t>(qb.value());
        // Слот вмещает обе версии: при сбое конвертации туда лягут исходные
        // байты. Для Bf16 исходная строка шире q8_0, поэтому max() оставляет
        // неиспользованный хвост только у Bf16.
        d.bytes = std::max(src, d.rb * m->out_dim);
        d.target = target;
        d.conv = true;
        if (target == QuantType::Q4_0) {
          ++n_conv_q4;
          q4_bytes += d.rb * m->out_dim;
        } else {
          ++n_conv_q8;
        }
      }
    }
    src_total += src;
    total += d.bytes;
    dest.push_back(d);
  }
  if (n_conv_q4 != 0) {
    std::fprintf(stderr,
                 "[tokshit] hot: Q4_0 turbo for %zu tensor(s): %.0f MB -> "
                 "%.0f MB (slot %.0f MB)\n",
                 n_conv_q4, static_cast<double>(src_total) / 1e6,
                 static_cast<double>(q4_bytes) / 1e6,
                 static_cast<double>(total) / 1e6);
  }
  if (n_conv_q8 != 0) {
    std::fprintf(stderr,
                 "[tokshit] hot: requantizing %zu tensor(s) to Q8_0\n",
                 n_conv_q8);
  }

  std::vector<std::size_t> sizes;
  sizes.reserve(dest.size());
  for (const Dest& d : dest) sizes.push_back(d.bytes);

  hot_.clear();
  if (!hot_.allocate(sizes)) {
    std::fprintf(stderr,
                 "[tokshit] hot weights: %.0f MB not pinned, keeping mmap\n",
                 static_cast<double>(total) / 1e6);
    return;
  }

  // Сначала проверяем все адреса, потом копируем: data_at не должен
  // промахнуться, но частично помеченные hot-тензоры хуже понятной ошибки.
  std::vector<uint8_t*> dsts;
  dsts.reserve(dest.size());
  std::size_t off = 0;
  for (std::size_t i = 0; i < dest.size(); ++i) {
    uint8_t* dst = hot_.data_at(off);
    if (dst == nullptr) {
      hot_.clear();
      std::fprintf(stderr, "[tokshit] hot weights: bad offset %zu\n", off);
      return;
    }
    dsts.push_back(dst);
    off += sizes[i];
  }
  // scratch для переквантовки: самая длинная строка in_dim, у флинта 2048
  std::vector<float> row_f;
  std::size_t failed = 0;
  for (std::size_t i = 0; i < dest.size(); ++i) {
    const Dest& d = dest[i];
    const Meta& m = *d.meta;
    if (!d.conv) {
      std::memcpy(dsts[i], mm_data + m.base, sizes[i]);
      continue;
    }
    row_f.resize(m.in_dim);
    bool ok = true;
    for (std::size_t r = 0; r < m.out_dim && ok; ++r) {
      const auto src_row = std::span<const uint8_t>(
          mm_data + m.base + r * m.rb, static_cast<std::size_t>(m.rb));
      const auto dq =
          dequant_into(m.quant, src_row, m.in_dim, std::span<float>(row_f));
      if (!dq.ok()) {
        ok = false;
        break;
      }
      const auto q = d.target == QuantType::Q4_0
                         ? quantize_q4_0_into(
                               row_f,
                               std::span<uint8_t>(dsts[i] + r * d.rb, d.rb))
                         : quantize_q8_0_into(
                               row_f,
                               std::span<uint8_t>(dsts[i] + r * d.rb, d.rb));
      if (!q.ok()) ok = false;
    }
    if (!ok) {
      // не смогли переквантовать: кладём исходные байты, тензор остаётся
      // в своём кванте и считается медленным путём. слот выделен как
      // max(src, q8), поэтому копия всегда помещается
      std::memcpy(dsts[i], mm_data + m.base, d.src);
      ++failed;
      continue;
    }
  }
  off = 0;
  for (std::size_t i = 0; i < dest.size(); ++i) {
    Meta& m = *dest[i].meta;
    m.base = off;
    m.hot = true;
    if (dest[i].conv) {
      m.quant = dest[i].target;
      m.rb = dest[i].rb;
    }
    off += sizes[i];
  }
  if (failed != 0) {
    std::fprintf(stderr, "[tokshit] hot: %zu tensor(s) kept their original quant\n",
                 failed);
  }
  std::fprintf(stderr,
               "[tokshit] hot weights copied: %.0f MB (%s, %zu MB pinned)\n",
               static_cast<double>(hot_.size()) / 1e6,
               hot_.on_ahb() ? "AHB, not swappable" : "partly malloc",
               hot_.ahb_bytes() / (1024 * 1024));
}

}  // namespace tokshit
