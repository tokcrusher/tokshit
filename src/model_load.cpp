#include "tokshit/model.hpp"
#include "tokshit/ahb.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "env.hpp"
#include "tokshit/dequant.hpp"
#include "tokshit/gguf.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/pool.hpp"

namespace tokshit {
namespace {

[[nodiscard]] Result<std::vector<float>> read_f32_tensor(const Gguf& gguf,
                                                         std::string_view name) {
  TOKSHIT_TRY(t, gguf.tensor(name));
  if (t->quant != QuantType::F32) {
    return make_error(ErrCode::InvalidValue, std::string(name),
                      "expected F32 tensor");
  }
  TOKSHIT_TRY(data, gguf.read_tensor(name));
  if (data.size() % 4 != 0) {
    return make_error(ErrCode::TensorSizeMismatch, std::string(name),
                      "F32 tensor length is not a multiple of 4");
  }
  std::vector<float> out(data.size() / 4);
  for (std::size_t i = 0; i < out.size(); ++i) {
    const uint32_t bits = static_cast<uint32_t>(data[i * 4]) |
                          (static_cast<uint32_t>(data[i * 4 + 1]) << 8) |
                          (static_cast<uint32_t>(data[i * 4 + 2]) << 16) |
                          (static_cast<uint32_t>(data[i * 4 + 3]) << 24);
    std::memcpy(&out[i], &bits, sizeof(float));
  }
  return out;
}

// сколько элементов в тензоре начиная с оси start, тупое произведение
// product of the dims from `start` to the end, nothing clever
uint64_t product_dims(const GgufTensor& t, std::size_t start) {
  uint64_t product = 1;
  for (std::size_t i = start; i < t.dims.size(); ++i) product *= t.dims[i];
  return product;
}
}  // namespace

Result<Model> Model::load(const std::string& path,
                          const ModelOptions& options) {
  TOKSHIT_TRY(gguf, Gguf::open(path));
  // запоминаем до перемещения gguf в модель: дальше он moved-from, и его
  // fd/file_len уже нули
  // remember before gguf is moved into the model: after that it is moved-from
  // and its fd/file_len are already zero
  const int file_fd = gguf.fd();
  const uint64_t file_len = gguf.file_len();

  // читаем файл вразнобой: readahead тут только тянет соседние эксперты,
  // которые нам не нужны. включается отдельно, потому что mmap-часть
  // (attention) наоборот любит readahead
  // we read the file all over the place: readahead here only drags in
  // neighbouring experts we dont need. opt-in, because the mmap part
  // (attention) actually likes readahead
  if (env_flag("TOKSHIT_FADV_RANDOM")) {
    (void)::posix_fadvise(gguf.fd(), 0, 0, POSIX_FADV_RANDOM);
  }

  TOKSHIT_TRY(mm, gguf.mmap_file());
  if (options.mmap_willneed) mm.advise_willneed();

  TOKSHIT_TRY(n_layers_v, gguf.u64_kv("flint.block_count"));
  TOKSHIT_TRY(n_dense_v, gguf.u64_kv("flint.leading_dense_block_count"));
  TOKSHIT_TRY(hidden_v, gguf.u64_kv("flint.embedding_length"));
  TOKSHIT_TRY(n_heads_v, gguf.u64_kv("flint.attention.head_count"));
  TOKSHIT_TRY(n_kv_v, gguf.u64_kv("flint.attention.head_count_kv"));
  TOKSHIT_TRY(head_dim_v, gguf.u64_kv("flint.attention.key_length"));
  TOKSHIT_TRY(rope_theta_v, gguf.f32_kv("flint.rope.freq_base"));
  TOKSHIT_TRY(eps_v, gguf.f32_kv("flint.attention.layer_norm_rms_epsilon"));
  TOKSHIT_TRY(window_v, gguf.u64_kv("flint.attention.sliding_window"));
  TOKSHIT_TRY(n_experts_v, gguf.u64_kv("flint.expert_count"));
  TOKSHIT_TRY(n_used_v, gguf.u64_kv("flint.expert_used_count"));
  TOKSHIT_TRY(dense_ffn_v, gguf.u64_kv("flint.feed_forward_length"));
  TOKSHIT_TRY(ffn_dim_v, gguf.u64_kv("flint.expert_feed_forward_length"));
  TOKSHIT_TRY(shared_dim_v,
              gguf.u64_kv("flint.expert_shared_feed_forward_length"));

  const auto n_layers = static_cast<std::size_t>(n_layers_v);
  const auto n_dense = static_cast<std::size_t>(n_dense_v);
  const auto hidden = static_cast<std::size_t>(hidden_v);
  const auto n_heads = static_cast<std::size_t>(n_heads_v);
  const auto n_kv = static_cast<std::size_t>(n_kv_v);
  const auto head_dim = static_cast<std::size_t>(head_dim_v);
  const auto window = static_cast<std::size_t>(window_v);
  const auto n_experts = static_cast<std::size_t>(n_experts_v);
  const auto n_used = static_cast<std::size_t>(n_used_v);
  const auto dense_ffn = static_cast<std::size_t>(dense_ffn_v);
  const auto ffn_dim = static_cast<std::size_t>(ffn_dim_v);
  const auto shared_dim = static_cast<std::size_t>(shared_dim_v);

  auto pattern = [&](std::string_view key) -> Result<std::vector<bool>> {
    TOKSHIT_TRY(value, gguf.get(key));
    const auto* array = value->as_array();
    if (array == nullptr) {
      return make_error(ErrCode::InvalidValue, std::string(key),
                        "expected array");
    }
    std::vector<bool> out;
    out.reserve(array->size());
    for (const Value& item : *array) {
      const auto b = item.as_bool();
      out.push_back(b.has_value() && *b);
    }
    return out;
  };

  TOKSHIT_TRY(swa, pattern("flint.attention.sliding_window_pattern"));
  TOKSHIT_TRY(rope_on, pattern("flint.attention.rope_pattern"));
  if (swa.size() != n_layers || rope_on.size() != n_layers) {
    return make_error(ErrCode::InvalidValue, "flint.pattern",
                      "attention pattern length does not match block_count");
  }
  if (n_dense > n_layers) {
    return make_error(ErrCode::InvalidValue, "flint.leading_dense_block_count",
                      "dense block count exceeds block count");
  }

  const std::size_t n_moe = n_layers > n_dense ? n_layers - n_dense : 0;
  const uint64_t cache_mb =
      env_u64("TOKSHIT_EXPERT_CACHE_MB", options.expert_cache_mb);
  const uint64_t total_budget = cache_mb * 1024ULL * 1024ULL;
  const uint64_t resident_budget =
      n_moe > 0 ? total_budget / static_cast<uint64_t>(n_moe) : total_budget;
  std::fprintf(stderr,
               "[tokshit] expert cache budget: %llu MB total, %llu MB per layer "
               "(%zu MoE layers)\n",
               static_cast<unsigned long long>(cache_mb),
               static_cast<unsigned long long>(resident_budget / (1024ULL * 1024ULL)),
               n_moe);

  const uint64_t data_offset = gguf.data_offset();

  auto mk = [&](std::string_view name) -> Result<Meta> {
    TOKSHIT_TRY(t, gguf.tensor(name));
    Meta meta;
    meta.in_dim = static_cast<std::size_t>(t->dims.empty() ? 0 : t->dims[0]);
    meta.out_dim = static_cast<std::size_t>(product_dims(*t, 1));
    TOKSHIT_TRY(rb, row_bytes(t->quant, meta.in_dim));
    meta.rb = static_cast<std::size_t>(rb);
    meta.base = data_offset + t->offset;
    meta.quant = t->quant;
    meta.hot = false;
    return meta;
  };

  auto mke = [&](std::string_view name) -> Result<ExpertMeta> {
    TOKSHIT_TRY(t, gguf.tensor(name));
    ExpertMeta meta;
    meta.in_dim = static_cast<std::size_t>(t->dims.empty() ? 0 : t->dims[0]);
    meta.rows = static_cast<std::size_t>(t->dims.size() > 1 ? t->dims[1] : 0);
    TOKSHIT_TRY(rb, row_bytes(t->quant, meta.in_dim));
    meta.rb = static_cast<std::size_t>(rb);
    meta.base = data_offset + t->offset;
    meta.quant = t->quant;
    meta.size = meta.rb * meta.rows;
    return meta;
  };

  std::vector<Layer> layers;
  layers.reserve(n_layers);
  for (std::size_t l = 0; l < n_layers; ++l) {
    const auto nm = [l](std::string_view suffix) {
      return "blk." + std::to_string(l) + "." + std::string(suffix);
    };
    const bool is_moe = l >= n_dense;

    Layer lt;
    TOKSHIT_TRY(attn_norm, read_f32_tensor(gguf, nm("attn_norm.weight")));
    lt.attn_norm = std::move(attn_norm);
    TOKSHIT_TRY(attn_q, mk(nm("attn_q.weight")));
    lt.attn_q = attn_q;
    TOKSHIT_TRY(attn_k, mk(nm("attn_k.weight")));
    lt.attn_k = attn_k;
    TOKSHIT_TRY(attn_v, mk(nm("attn_v.weight")));
    lt.attn_v = attn_v;
    TOKSHIT_TRY(attn_q_norm, read_f32_tensor(gguf, nm("attn_q_norm.weight")));
    lt.attn_q_norm = std::move(attn_q_norm);
    TOKSHIT_TRY(attn_k_norm, read_f32_tensor(gguf, nm("attn_k_norm.weight")));
    lt.attn_k_norm = std::move(attn_k_norm);
    TOKSHIT_TRY(attn_sinks, read_f32_tensor(gguf, nm("attn_sinks.weight")));
    lt.attn_sinks = std::move(attn_sinks);
    TOKSHIT_TRY(attn_o, mk(nm("attn_output.weight")));
    lt.attn_o = attn_o;
    TOKSHIT_TRY(ffn_norm, read_f32_tensor(gguf, nm("ffn_norm.weight")));
    lt.ffn_norm = std::move(ffn_norm);

    if (!is_moe) {
      TOKSHIT_TRY(gate, mk(nm("ffn_gate.weight")));
      lt.ffn_gate = gate;
      TOKSHIT_TRY(up, mk(nm("ffn_up.weight")));
      lt.ffn_up = up;
      TOKSHIT_TRY(down, mk(nm("ffn_down.weight")));
      lt.ffn_down = down;
    } else {
      TOKSHIT_TRY(gate_inp, mk(nm("ffn_gate_inp.weight")));
      lt.gate_inp = gate_inp;
      TOKSHIT_TRY(gate_exps, mke(nm("ffn_gate_exps.weight")));
      lt.gate_exps = gate_exps;
      TOKSHIT_TRY(up_exps, mke(nm("ffn_up_exps.weight")));
      lt.up_exps = up_exps;
      TOKSHIT_TRY(down_exps, mke(nm("ffn_down_exps.weight")));
      lt.down_exps = down_exps;
      TOKSHIT_TRY(gate_shexp, mk(nm("ffn_gate_shexp.weight")));
      lt.gate_shexp = gate_shexp;
      TOKSHIT_TRY(up_shexp, mk(nm("ffn_up_shexp.weight")));
      lt.up_shexp = up_shexp;
      TOKSHIT_TRY(down_shexp, mk(nm("ffn_down_shexp.weight")));
      lt.down_shexp = down_shexp;
    }
    layers.push_back(std::move(lt));
  }

  TOKSHIT_TRY(embd, mk("token_embd.weight"));
  TOKSHIT_TRY(out, mk("output.weight"));
  TOKSHIT_TRY(out_norm, read_f32_tensor(gguf, "output_norm.weight"));
  TOKSHIT_TRY(vocab_t, gguf.tensor("output.weight"));

  const std::size_t half = head_dim / 2;
  std::vector<float> inv_freq(half);
  for (std::size_t i = 0; i < half; ++i) {
    inv_freq[i] = std::pow(rope_theta_v,
                           -2.0F * static_cast<float>(i) /
                               static_cast<float>(head_dim));
  }

  Model model;
  model.gguf_ = std::move(gguf);
  model.mm_ = std::move(mm);
  model.n_layers_ = n_layers;
  model.hidden_ = hidden;
  model.n_heads_ = n_heads;
  model.n_kv_ = n_kv;
  model.head_dim_ = head_dim;
  model.n_experts_ = n_experts;
  model.n_used_ = n_used;
  model.dense_ffn_ = dense_ffn;
  model.ffn_dim_ = ffn_dim;
  model.shared_dim_ = shared_dim;
  model.n_dense_ = n_dense;
  std::fprintf(stderr,
               "[tokshit] config: layers=%zu dense=%zu hidden=%zu experts=%zu "
               "used=%zu ffn=%zu shared=%zu heads=%zu kv=%zu head_dim=%zu "
               "vocab=%zu\n",
               n_layers, n_dense, hidden, n_experts, n_used, ffn_dim,
               shared_dim, n_heads, n_kv, head_dim,
               static_cast<std::size_t>(vocab_t->dims.size() > 1
                                            ? vocab_t->dims[1]
                                            : 0));
  model.rope_theta_ = rope_theta_v;
  model.eps_ = eps_v;
  model.window_ = window;
  model.swa_ = std::move(swa);
  model.rope_on_ = std::move(rope_on);
  model.vocab_ = static_cast<std::size_t>(vocab_t->dims.size() > 1
                                             ? vocab_t->dims[1]
                                             : 0);
  model.layers_ = std::move(layers);
  model.embd_ = embd;
  model.out_ = out;
  model.out_norm_ = std::move(out_norm);
  model.inv_freq_ = std::move(inv_freq);
  model.resident_ = std::make_unique<ResidentCache>(n_layers);
  model.freq_ = std::make_unique<FreqState>(n_layers);
  model.resident_budget_ = resident_budget;
  model.total_budget_ = total_budget;
  // ключ частотного профиля: геометрия и кванты, не путь к файлу. один и тот
  // же gguf, скопированный в другое место, делит один профиль
  // frequency profile key: geometry and quants, not the file path. the same
  // gguf copied elsewhere shares one profile
  {
    uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    auto mix = [&h](uint64_t v) {
      for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 0xFFU;
        h *= 1099511628211ULL;
      }
    };
    mix(n_layers);
    mix(n_dense);
    mix(hidden);
    mix(n_experts);
    mix(n_used);
    mix(ffn_dim);
    mix(shared_dim);
    mix(model.vocab_);
    for (const auto& lt : model.layers_) {
      for (const auto* em : {lt.gate_exps ? &*lt.gate_exps : nullptr,
                             lt.up_exps ? &*lt.up_exps : nullptr,
                             lt.down_exps ? &*lt.down_exps : nullptr}) {
        if (em == nullptr) continue;
        mix(em->size);
        mix(static_cast<uint64_t>(em->quant));
      }
    }
    char key[17];
    std::snprintf(key, sizeof(key), "%016llx",
                  static_cast<unsigned long long>(h));
    model.freq_key_ = key;
  }
  if (n_moe != 0) {
    const auto& sl = model.layers_[n_dense].gate_exps;
    std::fprintf(stderr, "[tokshit] expert slice: %zu KB (per layer %zu MB)\n",
                 sl ? sl->size / 1024 : std::size_t{0},
                 n_moe != 0 ? (static_cast<std::size_t>(resident_budget /
                               1048576) + 1)
                            : std::size_t{0});
  }
  if (resident_budget != 0) {
    // AHB pool для expert-кэша: pinned dma-buf, не сжимается ZRAM, не вытесняется LMK.
    // Fallback на malloc если AHB недоступен.
    {
      ahb::Config acfg;
      // The pool budget must cover the whole expert cache. The old 3/4 split
      // left a quarter of the cache to the malloc fallback, whose 256 MB cap
      // silently turned later expert reads into mmap faults. Only the global
      // AHB limit and the MemAvailable reserve may shrink the pool.
      acfg.total_bytes = total_budget;
      // Small slabs keep the per-size-class split from wasting budget: one
      // oversized slab per class used to strand hundreds of megabytes in a
      // class that the current layer mix barely touches.
      acfg.slab_bytes  = env_u64("TOKSHIT_AHB_SLAB_MB", 16) * 1024ULL * 1024ULL;
      std::size_t max_slice = 0;
      for (const auto& lt : model.layers_) {
        for (const auto* em : {lt.gate_exps ? &*lt.gate_exps : nullptr,
                               lt.up_exps ? &*lt.up_exps : nullptr,
                               lt.down_exps ? &*lt.down_exps : nullptr}) {
          if (em == nullptr) continue;
          max_slice = std::max(max_slice, em->size);
          acfg.slot_sizes.push_back(em->size);
        }
      }
      std::sort(acfg.slot_sizes.begin(), acfg.slot_sizes.end());
      acfg.slot_sizes.erase(
          std::unique(acfg.slot_sizes.begin(), acfg.slot_sizes.end()),
          acfg.slot_sizes.end());
      acfg.slot_bytes = max_slice != 0 ? max_slice : std::size_t{576u * 1024u};
      std::fprintf(stderr,
                   "[ahb] %zu expert slice size(s), max %zu KB, cache budget "
                   "%llu MB\n",
                   acfg.slot_sizes.size(), acfg.slot_bytes / 1024,
                   static_cast<unsigned long long>(acfg.total_bytes / 1048576));
      ahb::Pool::instance().init(acfg);
    }
    const auto io_threads = static_cast<std::size_t>(
        env_u64("TOKSHIT_IO_THREADS", options.io_threads));
    // fadvise только по явному запросу: дроп page cache полезен не всегда,
    // на телефоне с малым запасом ram он может выкинуть нужное
    // fadvise only on request: dropping page cache is not always a win, on a
    // phone with little ram it can throw away something useful
    const bool fadv = env_flag("TOKSHIT_FADV_DONTNEED");
    model.io_ = std::make_unique<ExpertIo>(
        file_fd, file_len, *model.resident_, resident_budget,
        io_threads == 0 ? 1 : io_threads, fadv, path);
    std::fprintf(stderr, "[tokshit] expert io: %zu readers, fadvise=%d\n",
                 io_threads == 0 ? std::size_t{1} : io_threads,
                 fadv ? 1 : 0);
  }
  {
    const auto& l0 = model.layers_[n_dense < n_layers ? n_dense : 0];
    std::fprintf(stderr,
                 "[tokshit] quants: attn_q=%u attn_k=%u attn_v=%u attn_o=%u "
                 "gate_inp=%u shexp_g=%u expert_g=%u out=%u\n",
                 static_cast<unsigned>(l0.attn_q.quant),
                 static_cast<unsigned>(l0.attn_k.quant),
                 static_cast<unsigned>(l0.attn_v.quant),
                 static_cast<unsigned>(l0.attn_o.quant),
                 l0.gate_inp ? static_cast<unsigned>(l0.gate_inp->quant) : 999u,
                 l0.gate_shexp ? static_cast<unsigned>(l0.gate_shexp->quant) : 999u,
                 l0.gate_exps ? static_cast<unsigned>(l0.gate_exps->quant) : 999u,
                 static_cast<unsigned>(model.out_.quant));
  }
  model.prepare_workspace();
  return model;
}
}  // namespace tokshit
