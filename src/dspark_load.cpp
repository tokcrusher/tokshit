#include "tokshit/dspark.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/gguf.hpp"
#include "tokshit/mmap.hpp"

namespace tokshit {
namespace {

[[nodiscard]] Result<std::vector<float>> read_f32(const Gguf& g,
                                                  std::string_view name) {
  TOKSHIT_TRY(t, g.tensor(name));
  if (t->quant != QuantType::F32) {
    return make_error(ErrCode::InvalidValue, std::string(name),
                      "expected F32 tensor");
  }
  TOKSHIT_TRY(data, g.read_tensor(name));
  std::vector<float> out(data.size() / 4);
  std::memcpy(out.data(), data.data(), out.size() * sizeof(float));
  return out;
}

uint64_t dim_product(const GgufTensor& t, std::size_t start) {
  uint64_t p = 1;
  for (std::size_t i = start; i < t.dims.size(); ++i) p *= t.dims[i];
  return p;
}

[[nodiscard]] Result<std::vector<bool>> bool_array(const Gguf& g,
                                                   std::string_view key) {
  TOKSHIT_TRY(v, g.get(key));
  const auto* arr = v->as_array();
  if (arr == nullptr) {
    return make_error(ErrCode::InvalidValue, std::string(key),
                      "expected bool array");
  }
  std::vector<bool> out;
  out.reserve(arr->size());
  for (const Value& x : *arr) {
    const auto b = x.as_bool();
    if (!b.has_value()) {
      return make_error(ErrCode::InvalidValue, std::string(key),
                        "expected bool array");
    }
    out.push_back(*b);
  }
  return out;
}

}  // namespace

DSpark::DSpark() = default;
DSpark::~DSpark() = default;

Result<std::unique_ptr<DSpark>> DSpark::load(const Gguf& gguf, const Mmap& mm,
                                             Inkling2& target,
                                             std::size_t kv_ctx) {
  std::unique_ptr<DSpark> d(new DSpark());
  d->target_ = &target;
  d->data_ = reinterpret_cast<const uint8_t*>(mm.bytes().data());
  d->data_len_ = mm.bytes().size();

  TOKSHIT_TRY(arch_v, gguf.get("general.architecture"));
  const auto arch = arch_v->as_str();
  if (!arch.has_value() || *arch != "dflash") {
    return make_error(ErrCode::InvalidValue, "general.architecture",
                      "expected dflash");
  }

  auto u64 = [&](std::string_view key) -> Result<uint64_t> {
    return gguf.u64_kv(std::string("dflash.") + std::string(key));
  };
  auto f32 = [&](std::string_view key) -> Result<float> {
    return gguf.f32_kv(std::string("dflash.") + std::string(key));
  };
  auto f32_def = [&](std::string_view key, float def) {
    auto r = f32(key);
    return r.ok() ? r.value() : def;
  };

  TOKSHIT_TRY(n_layer_v, u64("block_count"));
  TOKSHIT_TRY(hidden_v, u64("embedding_length"));
  TOKSHIT_TRY(ffn_v, u64("feed_forward_length"));
  TOKSHIT_TRY(n_head_v, u64("attention.head_count"));
  TOKSHIT_TRY(n_kv_v, u64("attention.head_count_kv"));
  TOKSHIT_TRY(head_dim_v, u64("attention.key_length"));
  TOKSHIT_TRY(value_dim_v, u64("attention.value_length"));
  TOKSHIT_TRY(vocab_pad_v, u64("vocab_size"));
  TOKSHIT_TRY(vocab_v, u64("vocab_size_unpadded"));
  TOKSHIT_TRY(block_size_v, u64("block_size"));
  TOKSHIT_TRY(rank_v, u64("dspark.markov_rank"));
  TOKSHIT_TRY(swa_v, u64("attention.sliding_window"));
  TOKSHIT_TRY(rope_dim_v, u64("rope.dimension_count"));
  TOKSHIT_TRY(target_layers_v, gguf.u32s("dflash.target_layers"));

  const auto causal_v = gguf.get("dflash.attention.causal");
  if (causal_v.ok() && causal_v.value()->as_bool().value_or(true)) {
    return make_error(ErrCode::InvalidValue, "dflash.attention.causal",
                      "Inkling2 DSpark expects non-causal attention");
  }
  const auto anchor_v = gguf.get("dflash.sample_from_anchor");
  if (!anchor_v.ok() || anchor_v.value()->as_bool().value_or(true)) {
    return make_error(ErrCode::InvalidValue, "dflash.sample_from_anchor",
                      "Inkling2 DSpark expects sample_from_anchor=false");
  }
  const auto conf_v = gguf.get("dflash.has_confidence_head");
  if (conf_v.ok() && conf_v.value()->as_bool().value_or(false)) {
    return make_error(ErrCode::InvalidValue, "dflash.has_confidence_head",
                      "Inkling2 DSpark has no confidence head");
  }

  if (value_dim_v != head_dim_v || rope_dim_v != head_dim_v ||
      n_head_v == 0 || n_kv_v == 0 || n_head_v % n_kv_v != 0) {
    return make_error(ErrCode::InvalidValue, "dflash.attention",
                      "invalid head dimensions");
  }
  if (block_size_v != 8 || rank_v == 0 || rank_v > hidden_v) {
    return make_error(ErrCode::InvalidValue, "dflash.block_size",
                      "Inkling2 DSpark expects block_size=8 and rank<=embd");
  }
  if (target.n_embd() != hidden_v || target.vocab() != vocab_v ||
      target.vocab_padded() != vocab_pad_v) {
    return make_error(ErrCode::InvalidValue, "dflash.target",
                      "draft does not match the target model");
  }
  if (target_layers_v.empty() || target_layers_v.back() > target.n_slots() ||
      !std::is_sorted(target_layers_v.begin(), target_layers_v.end())) {
    return make_error(ErrCode::InvalidValue, "dflash.target_layers",
                      "invalid target layer taps");
  }

  d->n_layer_ = n_layer_v;
  d->hidden_ = hidden_v;
  d->ffn_ = ffn_v;
  d->n_head_ = n_head_v;
  d->n_kv_ = n_kv_v;
  d->head_dim_ = head_dim_v;
  d->rank_ = rank_v;
  d->vocab_padded_ = vocab_pad_v;
  d->vocab_unpadded_ = vocab_v;
  d->block_size_ = block_size_v;
  d->swa_ = swa_v + 1;  // llama.cpp includes the anchor in the draft window
  d->kv_ctx_ = kv_ctx == 0 ? target.kv_ctx() : kv_ctx;
  d->ctx_slots_ = d->block_size_ + 1;
  d->eps_ = f32_def("attention.layer_norm_rms_epsilon", 1e-6F);
  d->rope_base_ = f32_def("rope.freq_base", 1e6F);
  d->softcap_ = f32_def("final_logit_softcapping", 30.0F);
  d->target_slots_ = target_layers_v;

  TOKSHIT_TRY(swa_pattern, bool_array(gguf, "dflash.attention.sliding_window_pattern"));
  if (swa_pattern.size() != d->n_layer_) {
    return make_error(ErrCode::InvalidValue,
                      "dflash.attention.sliding_window_pattern",
                      "wrong pattern length");
  }
  d->swa_layer_.assign(swa_pattern.begin(), swa_pattern.end());

  const uint64_t data_off = gguf.data_offset();
  auto mk = [&](std::string_view name) -> Result<InkMeta> {
    TOKSHIT_TRY(t, gguf.tensor(name));
    InkMeta m;
    m.in_dim = static_cast<uint32_t>(t->dims.empty() ? 0 : t->dims[0]);
    m.rows = static_cast<uint32_t>(dim_product(*t, 1));
    TOKSHIT_TRY(rb, row_bytes(t->quant, m.in_dim));
    m.rb = static_cast<uint32_t>(rb);
    m.base = data_off + t->offset;
    m.quant = t->quant;
    return m;
  };
  auto setm = [&](InkMeta& dst, std::string_view name) -> Result<void> {
    TOKSHIT_TRY(x, mk(name));
    dst = x;
    return {};
  };
  auto name_of = [](std::size_t l, std::string_view suffix) {
    return "blk." + std::to_string(l) + "." + std::string(suffix);
  };
  auto load_bf16 = [&](std::string_view name,
                       std::vector<float>& out) -> Result<void> {
    TOKSHIT_TRY(t, gguf.tensor(name));
    if (t->quant != QuantType::Bf16) {
      return make_error(ErrCode::InvalidValue, std::string(name),
                        "expected BF16 tensor");
    }
    TOKSHIT_TRY(data, gguf.read_tensor(name));
    std::size_t n = 1;
    for (uint64_t x : t->dims) n *= x;
    out.resize(n);
    TOKSHIT_TRY_VOID(dequant_into(t->quant, data, n, out));
    return {};
  };

  TOKSHIT_TRY_VOID(setm(d->fuse_, "dspark_fuse.weight"));
  TOKSHIT_TRY_VOID(setm(d->markov_w1_, "markov_w1.weight"));
  TOKSHIT_TRY_VOID(setm(d->joint_, "joint_proj.weight"));
  TOKSHIT_TRY_VOID(setm(d->hidden_out_, "dspark_hidden_out.weight"));
  TOKSHIT_TRY(joint_bias, read_f32(gguf, "joint_proj.bias"));
  TOKSHIT_TRY(hout_bias, read_f32(gguf, "dspark_hidden_out.bias"));
  TOKSHIT_TRY(mask_embd, read_f32(gguf, "mask_embd.weight"));
  TOKSHIT_TRY(output_norm, read_f32(gguf, "output_norm.weight"));
  TOKSHIT_TRY(dspark_norm, read_f32(gguf, "dspark_norm.weight"));
  d->joint_bias_ = std::move(joint_bias);
  d->hidden_out_bias_ = std::move(hout_bias);
  d->mask_embd_ = std::move(mask_embd);
  d->output_norm_ = std::move(output_norm);
  d->dspark_norm_ = std::move(dspark_norm);

  if (d->fuse_.in_dim != 4 || d->fuse_.rows != 4 ||
      d->markov_w1_.in_dim != d->rank_ ||
      d->markov_w1_.rows != d->vocab_padded_ ||
      d->joint_.in_dim != 2 * d->hidden_ + 2 * d->rank_ ||
      d->joint_.rows != 3 * d->rank_ ||
      d->hidden_out_.in_dim != d->rank_ ||
      d->hidden_out_.rows != d->hidden_ ||
      d->joint_bias_.size() != 3 * d->rank_ ||
      d->hidden_out_bias_.size() != d->hidden_ ||
      d->mask_embd_.size() != d->hidden_ ||
      d->output_norm_.size() != d->hidden_ ||
      d->dspark_norm_.size() != d->hidden_) {
    return make_error(ErrCode::InvalidValue, "dspark",
                      "draft tensor dimensions do not match metadata");
  }

  TOKSHIT_TRY_VOID(load_bf16("joint_proj.weight", d->joint_f32_));
  TOKSHIT_TRY_VOID(load_bf16("dspark_hidden_out.weight", d->hout_f32_));

  d->layers_.resize(d->n_layer_);
  for (std::size_t l = 0; l < d->n_layer_; ++l) {
    Layer& L = d->layers_[l];
    TOKSHIT_TRY_VOID(setm(L.q, name_of(l, "attn_q.weight")));
    TOKSHIT_TRY_VOID(setm(L.k, name_of(l, "attn_k.weight")));
    TOKSHIT_TRY_VOID(setm(L.v, name_of(l, "attn_v.weight")));
    TOKSHIT_TRY_VOID(setm(L.o, name_of(l, "attn_output.weight")));
    TOKSHIT_TRY_VOID(setm(L.gate, name_of(l, "ffn_gate.weight")));
    TOKSHIT_TRY_VOID(setm(L.up, name_of(l, "ffn_up.weight")));
    TOKSHIT_TRY_VOID(setm(L.down, name_of(l, "ffn_down.weight")));
    TOKSHIT_TRY_VOID(setm(L.k_ctx, name_of(l, "attn_k_ctx.weight")));
    TOKSHIT_TRY_VOID(setm(L.v_ctx, name_of(l, "attn_v_ctx.weight")));
    TOKSHIT_TRY(attn_norm, read_f32(gguf, name_of(l, "attn_norm.weight")));
    TOKSHIT_TRY(ffn_norm, read_f32(gguf, name_of(l, "ffn_norm.weight")));
    TOKSHIT_TRY(q_norm, read_f32(gguf, name_of(l, "attn_q_norm.weight")));
    TOKSHIT_TRY(k_norm, read_f32(gguf, name_of(l, "attn_k_norm.weight")));
    TOKSHIT_TRY(k_ctx_norm,
                read_f32(gguf, name_of(l, "attn_k_ctx_norm.weight")));
    TOKSHIT_TRY(dflare_norm,
                read_f32(gguf, name_of(l, "dflare_norm.weight")));
    L.attn_norm = std::move(attn_norm);
    L.ffn_norm = std::move(ffn_norm);
    L.q_norm = std::move(q_norm);
    L.k_norm = std::move(k_norm);
    L.k_ctx_norm = std::move(k_ctx_norm);
    L.dflare_norm = std::move(dflare_norm);
    if (L.q_norm.size() != d->head_dim_ || L.k_norm.size() != d->head_dim_ ||
        L.k_ctx_norm.size() != d->head_dim_ ||
        L.attn_norm.size() != d->hidden_ || L.ffn_norm.size() != d->hidden_ ||
        L.dflare_norm.size() != d->hidden_) {
      return make_error(ErrCode::InvalidValue, name_of(l, "norm"),
                        "wrong norm size");
    }
  }

  d->kv_.resize(d->n_layer_);
  for (std::size_t l = 0; l < d->n_layer_; ++l) {
    const std::size_t kvd = d->n_kv_ * d->head_dim_;
    d->kv_[l].k.assign(d->kv_ctx_ * kvd, 0);
    d->kv_[l].v.assign(d->kv_ctx_ * kvd, 0);
  }

  d->ctx_ring_.assign(d->ctx_slots_ * d->hidden_, 0.0F);
  d->ctx_pos_.assign(d->block_size_ + 1, -1);

  d->x_.assign(d->block_size_ * d->hidden_, 0.0F);
  d->q_.assign(d->block_size_ * d->n_head_ * d->head_dim_, 0.0F);
  d->k_.assign(d->block_size_ * d->n_kv_ * d->head_dim_, 0.0F);
  d->v_.assign(d->block_size_ * d->n_kv_ * d->head_dim_, 0.0F);
  d->attn_.assign(d->block_size_ * d->n_head_ * d->head_dim_, 0.0F);
  d->ffn_in_.assign(d->block_size_ * d->hidden_, 0.0F);
  d->gate_.assign(d->block_size_ * d->ffn_, 0.0F);
  d->up_.assign(d->block_size_ * d->ffn_, 0.0F);
  d->act_.assign(d->block_size_ * d->ffn_, 0.0F);
  d->out_.assign(d->block_size_ * d->hidden_, 0.0F);
  d->fuse_ctx_.assign(d->hidden_, 0.0F);
  d->kctx_.assign(d->n_kv_ * d->head_dim_, 0.0F);
  d->vctx_.assign(d->n_kv_ * d->head_dim_, 0.0F);
  d->state_.assign(d->rank_, 0.0F);
  d->joint_buf_.assign(2 * d->rank_ + 2 * d->hidden_, 0.0F);
  d->joint_out_.assign(3 * d->rank_, 0.0F);
  d->corr_.assign(d->hidden_, 0.0F);
  d->norm_.assign(d->block_size_ * d->hidden_, 0.0F);
  d->base_.assign(d->max_draft() * d->vocab_padded_, 0.0F);
  d->logits_.assign(d->vocab_padded_, 0.0F);
  d->row_.assign(d->rank_, 0.0F);
  d->scores_.assign(d->kv_ctx_ + d->block_size_, 0.0F);
  d->acts_.resize(d->block_size_);
  d->scratch_.reserve(d->hidden_);

  target.set_tap_slots(std::span<const uint32_t>(d->target_slots_));
  return d;
}

}  // namespace tokshit
