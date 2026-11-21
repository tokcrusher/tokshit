#include "tokshit/inkling2.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tokshit/dequant.hpp"
#include "tokshit/gguf.hpp"
#include "tokshit/kernels.hpp"
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

[[nodiscard]] Result<std::vector<float>> read_bf16(const Gguf& g,
                                                   std::string_view name) {
  TOKSHIT_TRY(t, g.tensor(name));
  TOKSHIT_TRY(data, g.read_tensor(name));
  std::size_t n = 1;
  for (uint64_t d : t->dims) n *= d;
  std::vector<float> out(n);
  TOKSHIT_TRY_VOID(dequant_into(t->quant, data, n, out));
  return out;
}

uint64_t dim_product(const GgufTensor& t, std::size_t start) {
  uint64_t p = 1;
  for (std::size_t i = start; i < t.dims.size(); ++i) p *= t.dims[i];
  return p;
}

}  // namespace

Inkling2::Inkling2() = default;
Inkling2::~Inkling2() = default;

const InkLayer& Inkling2::phys(std::size_t slot) const {
  return layers_[slot_phys_[slot]];
}

Result<std::unique_ptr<Inkling2>> Inkling2::load(const Gguf& gguf,
                                                 const Mmap& mm,
                                                 std::size_t kv_ctx) {
  std::unique_ptr<Inkling2> m(new Inkling2());
  m->gguf_ = &gguf;
  m->data_ = reinterpret_cast<const uint8_t*>(mm.bytes().data());
  m->data_len_ = mm.bytes().size();

  TOKSHIT_TRY(arch_v, gguf.get("general.architecture"));
  const auto arch = arch_v->as_str();
  if (!arch.has_value() || *arch != "inkling2") {
    return make_error(ErrCode::InvalidValue, "general.architecture",
                      "expected inkling2");
  }

  auto u64 = [&](std::string_view key) -> Result<uint64_t> {
    return gguf.u64_kv(std::string("inkling2.") + std::string(key));
  };
  auto f32 = [&](std::string_view key) -> Result<float> {
    return gguf.f32_kv(std::string("inkling2.") + std::string(key));
  };
  auto u64_def = [&](std::string_view key, uint64_t def) {
    auto r = u64(key);
    return r.ok() ? r.value() : def;
  };
  // часть ключей — массивы на слот; берём максимум по всем слоям
  auto u64_arr_max = [&](std::string_view key) -> Result<uint64_t> {
    auto arr = gguf.u32s(std::string("inkling2.") + std::string(key));
    if (!arr.ok()) return u64(key);
    uint64_t mx = 0;
    for (uint32_t v : arr.value()) mx = std::max<uint64_t>(mx, v);
    return mx;
  };
  auto f32_def = [&](std::string_view key, float def) {
    auto r = f32(key);
    return r.ok() ? r.value() : def;
  };

  TOKSHIT_TRY(slot_count_v, u64("block_count"));
  TOKSHIT_TRY(n_passes_v, u64("num_loops"));
  TOKSHIT_TRY(core_start_v, u64("loop.core_start"));
  TOKSHIT_TRY(core_end_v, u64("loop.core_end"));
  TOKSHIT_TRY(hidden_v, u64("embedding_length"));
  TOKSHIT_TRY(n_expert_v, u64("expert_count"));
  TOKSHIT_TRY(n_used_v, u64("expert_used_count"));
  TOKSHIT_TRY(n_ff_exp_v, u64("expert_feed_forward_length"));
  TOKSHIT_TRY(n_head_v, u64_arr_max("attention.head_count"));
  TOKSHIT_TRY(n_kv_v, u64_arr_max("attention.head_count_kv"));
  TOKSHIT_TRY(head_dim_v, u64("attention.key_length"));
  TOKSHIT_TRY(value_dim_v, u64("attention.value_length"));
  TOKSHIT_TRY(n_rel_v, u64("attention.relative_length"));
  TOKSHIT_TRY(rel_extent_v, u64("attention.relative_extent"));
  TOKSHIT_TRY(n_swa_v, u64("attention.sliding_window"));
  TOKSHIT_TRY(l_cache_v, u64("shortconv.l_cache"));
  TOKSHIT_TRY(vocab_pad_v, u64("vocab_size"));
  TOKSHIT_TRY(vocab_v, u64("vocab_size_unpadded"));
  TOKSHIT_TRY(block_size_v, u64("attn_res.block_size"));

  m->hidden_ = hidden_v;
  m->n_expert_ = n_expert_v;
  m->n_used_ = n_used_v;
  m->n_ff_exp_ = n_ff_exp_v;
  m->n_head_ = n_head_v;
  m->n_kv_ = n_kv_v;
  m->head_dim_ = head_dim_v;
  m->n_relative_ = n_rel_v;
  m->relative_extent_ = rel_extent_v;
  m->n_swa_ = n_swa_v;
  m->l_cache_ = l_cache_v;
  m->block_size_ = block_size_v;
  m->vocab_padded_ = vocab_pad_v;
  m->vocab_unpadded_ = vocab_v;
  m->slot_count_ = slot_count_v;
  m->n_passes_ = n_passes_v;
  m->core_start_ = core_start_v;
  m->core_end_ = core_end_v;
  m->kv_ctx_ = kv_ctx == 0 ? 2048 : kv_ctx;

  if (value_dim_v != head_dim_v) {
    return make_error(ErrCode::InvalidValue, "inkling2.attention.value_length",
                      "value length must equal key length");
  }
  if (m->core_start_ >= m->core_end_ || m->core_end_ > m->slot_count_) {
    return make_error(ErrCode::InvalidValue, "inkling2.loop",
                      "invalid loop core");
  }
  const std::size_t core = m->core_end_ - m->core_start_;
  if (m->n_passes_ < 2 || m->slot_count_ <= (m->n_passes_ - 1) * core) {
    return make_error(ErrCode::InvalidValue, "inkling2.loop",
                      "invalid loop schedule");
  }
  m->n_phys_ = m->slot_count_ - (m->n_passes_ - 1) * core;
  if (m->n_phys_ < m->core_end_) {
    return make_error(ErrCode::InvalidValue, "inkling2.loop",
                      "loop core exceeds physical blocks");
  }
  if (m->slot_count_ !=
      m->core_start_ + core * m->n_passes_ + (m->n_phys_ - m->core_end_)) {
    return make_error(ErrCode::InvalidValue, "inkling2.loop",
                      "slot count does not match loop schedule");
  }

  m->eps_ = f32_def("attention.layer_norm_rms_epsilon", 1e-6F);
  m->qk_clip_ = f32_def("attention.qk_clip", 0.0F);
  m->rel_clip_ = f32_def("attention.relative_clip", 0.0F);
  m->log_scale_alpha_ = f32_def("attention.log_scale_alpha", 0.0F);
  m->log_scale_floor_ = u64_def("attention.log_scale_floor", 0);
  m->situ_beta_ = f32_def("activation.situ_beta", 1.0F);
  m->situ_linear_beta_ = f32_def("activation.situ_linear_beta", 0.0F);
  m->expert_scale_ = f32_def("expert_weights_scale", 0.0F);
  m->final_softcap_ = f32_def("final_logit_softcapping", 0.0F);

  // скорость shortconv: секции K (n_kv*head_dim), V, attn-out, ffn-out
  m->sconv_channels_ = 2 * m->n_kv_ * m->head_dim_ + 2 * m->hidden_;

  const uint64_t data_off = gguf.data_offset();
  auto mk = [&](std::string_view name, bool required) -> Result<InkMeta> {
    auto tr = gguf.tensor(name);
    if (!tr.ok()) {
      if (!required) return InkMeta{};
      return tr.error();
    }
    const GgufTensor& t = *tr.value();
    InkMeta meta;
    meta.in_dim = static_cast<uint32_t>(t.dims.empty() ? 0 : t.dims[0]);
    meta.rows = static_cast<uint32_t>(dim_product(t, 1));
    TOKSHIT_TRY(rb, row_bytes(t.quant, meta.in_dim));
    meta.rb = static_cast<uint32_t>(rb);
    meta.base = data_off + t.offset;
    meta.quant = t.quant;
    return meta;
  };
  auto mke = [&](std::string_view name) -> Result<InkMeta> {
    TOKSHIT_TRY(t, gguf.tensor(name));
    if (t->dims.size() != 3) {
      return make_error(ErrCode::InvalidValue, std::string(name),
                        "expected 3D expert tensor");
    }
    InkMeta meta;
    meta.in_dim = static_cast<uint32_t>(t->dims[0]);
    meta.rows = static_cast<uint32_t>(t->dims[1]);
    TOKSHIT_TRY(rb, row_bytes(t->quant, meta.in_dim));
    meta.rb = static_cast<uint32_t>(rb);
    meta.base = data_off + t->offset;
    meta.quant = t->quant;
    return meta;
  };
  auto setm = [&](InkMeta& dst, std::string_view name,
                  bool required) -> Result<void> {
    TOKSHIT_TRY(x, mk(name, required));
    dst = x;
    return {};
  };
  auto name_of = [](std::size_t l, std::string_view suffix) {
    return "blk." + std::to_string(l) + "." + std::string(suffix);
  };

  m->layers_.resize(m->n_phys_);
  for (std::size_t l = 0; l < m->n_phys_; ++l) {
    InkLayer& L = m->layers_[l];
    TOKSHIT_TRY_VOID(setm(L.q, name_of(l, "attn_q.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.k, name_of(l, "attn_k.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.v, name_of(l, "attn_v.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.o, name_of(l, "attn_output.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.gate, name_of(l, "attn_gate.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.rel, name_of(l, "attn_rel.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.sink_proj, name_of(l, "attn_sink_proj.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.k_conv, name_of(l, "attn_k_conv.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.v_conv, name_of(l, "attn_v_conv.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.post_conv, name_of(l, "attn_post_conv.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.ffn_conv, name_of(l, "ffn_post_conv.weight"), true));
    TOKSHIT_TRY_VOID(setm(L.k_conv_proj, name_of(l, "attn_k_conv_proj.weight"), false));
    TOKSHIT_TRY_VOID(setm(L.v_conv_proj, name_of(l, "attn_v_conv_proj.weight"), false));

    TOKSHIT_TRY(q_clip,
                read_f32(gguf, name_of(l, "attn_q_clip.weight")));
    L.q_clip = std::move(q_clip);
    TOKSHIT_TRY(rel_bias,
                read_f32(gguf, name_of(l, "attn_rel_bias.weight")));
    L.rel_bias = std::move(rel_bias);
    TOKSHIT_TRY(rel_clip,
                read_f32(gguf, name_of(l, "attn_rel_clip.weight")));
    L.rel_clip = std::move(rel_clip);
    TOKSHIT_TRY(ffn_norm,
                read_f32(gguf, name_of(l, "ffn_norm.weight")));
    L.ffn_norm = std::move(ffn_norm);

    // SWA-блоки имеют attn_norm, global — отдельные q/k/v input norms
    auto an = read_f32(gguf, name_of(l, "attn_norm.weight"));
    if (an.ok()) {
      L.attn_norm = std::move(an.value());
    } else {
      TOKSHIT_TRY(qn, read_f32(gguf, name_of(l, "attn_q_inp_norm.weight")));
      TOKSHIT_TRY(kn, read_f32(gguf, name_of(l, "attn_k_inp_norm.weight")));
      TOKSHIT_TRY(vn, read_f32(gguf, name_of(l, "attn_v_inp_norm.weight")));
      L.q_inp_norm = std::move(qn);
      L.k_inp_norm = std::move(kn);
      L.v_inp_norm = std::move(vn);
    }

    TOKSHIT_TRY_VOID(setm(L.gate_inp, name_of(l, "ffn_gate_inp.weight"), true));
    TOKSHIT_TRY(gate_up_exps, mke(name_of(l, "ffn_gate_up_exps.weight")));
    L.gate_up_exps = gate_up_exps;
    TOKSHIT_TRY(down_exps, mke(name_of(l, "ffn_down_exps.weight")));
    L.down_exps = down_exps;
    TOKSHIT_TRY(exp_bias, read_f32(gguf, name_of(l, "exp_probs_b.bias")));
    L.exp_probs_b = std::move(exp_bias);

    auto gus = read_f32(gguf, name_of(l, "ffn_gate_up_exps.scale"));
    if (gus.ok()) L.gate_up_scale = std::move(gus.value());
    auto ds = read_f32(gguf, name_of(l, "ffn_down_exps.scale"));
    if (ds.ok()) L.down_scale = std::move(ds.value());

    auto rq = read_f32(gguf, name_of(l, "attn_res_query.weight"));
    if (rq.ok()) L.attn_res_query = std::move(rq.value());
    TOKSHIT_TRY(frq, read_f32(gguf, name_of(l, "ffn_res_query.weight")));
    L.ffn_res_query = std::move(frq);

    static const char* kLoopDown[4] = {"loop_attn_down.weight",
                                       "loop_k_down.weight",
                                       "loop_v_down.weight",
                                       "loop_mlp_down.weight"};
    static const char* kLoopGain[4] = {"loop_attn_gain.weight",
                                       "loop_k_gain.weight",
                                       "loop_v_gain.weight",
                                       "loop_mlp_gain.weight"};
    static const char* kLoopForget[4] = {"loop_attn_forget.weight",
                                         "loop_k_forget.weight",
                                         "loop_v_forget.weight",
                                         "loop_mlp_forget.weight"};
    auto down0 = mk(name_of(l, kLoopDown[0]), false);
    if (down0.ok() && down0.value().base != 0) {
      L.has_loop = true;
      for (int c = 0; c < 4; ++c) {
        TOKSHIT_TRY(dn, mk(name_of(l, kLoopDown[c]), true));
        L.loop_down[c] = dn;
        TOKSHIT_TRY(gn, mk(name_of(l, kLoopGain[c]), true));
        L.loop_gain[c] = gn;
        TOKSHIT_TRY(fg, mk(name_of(l, kLoopForget[c]), true));
        L.loop_forget[c] = fg;
      }
    }
  }

  // паттерн SWA задан на 48 слотов
  {
    auto pat = gguf.get("inkling2.attention.sliding_window_pattern");
    if (pat.ok() && pat.value()->as_array() != nullptr) {
      const auto& arr = *pat.value()->as_array();
      m->slot_swa_.reserve(arr.size());
      for (const Value& v : arr) {
        const auto b = v.as_bool();
        m->slot_swa_.push_back(b.has_value() && *b ? 1 : 0);
      }
    }
    if (m->slot_swa_.size() != m->slot_count_) {
      return make_error(ErrCode::InvalidValue,
                        "inkling2.attention.sliding_window_pattern",
                        "pattern length does not match block_count");
    }
  }

  TOKSHIT_TRY(tok_embd, mk("token_embd.weight", true));
  m->tok_embd_ = tok_embd;
  TOKSHIT_TRY(out, mk("output.weight", true));
  m->out_ = out;
  TOKSHIT_TRY(tok_norm, read_f32(gguf, "token_embd_norm.weight"));
  m->tok_norm_ = std::move(tok_norm);
  TOKSHIT_TRY(out_norm, read_f32(gguf, "output_norm.weight"));
  m->out_norm_ = std::move(out_norm);
  TOKSHIT_TRY(out_rq, read_f32(gguf, "output_res_query.weight"));
  m->out_res_query_ = std::move(out_rq);

  auto kvq = read_f32(gguf, "loop_kv_res_query.weight");
  if (kvq.ok()) m->loop_kv_query_ = std::move(kvq.value());
  auto rq = read_f32(gguf, "loop_res_query.weight");
  if (rq.ok()) m->loop_res_query_ = std::move(rq.value());

  // legacy layout: queries core-слоя i — rows [res[2i], kv[2i], kv[2i+1], res[2i+1]]
  for (std::size_t l = m->core_start_; l < m->core_end_; ++l) {
    InkLayer& L = m->layers_[l];
    const std::size_t i = l - m->core_start_;
    const std::size_t h = m->hidden_;
    auto copy_col = [&](const std::vector<float>& src, std::size_t col,
                        std::vector<float>& dst) {
      if (src.size() < (col + 1) * h) return;
      dst.assign(src.begin() + col * h, src.begin() + (col + 1) * h);
    };
    copy_col(m->loop_res_query_, 2 * i, L.msg_query[0]);
    copy_col(m->loop_kv_query_, 2 * i, L.msg_query[1]);
    copy_col(m->loop_kv_query_, 2 * i + 1, L.msg_query[2]);
    copy_col(m->loop_res_query_, 2 * i + 1, L.msg_query[3]);
  }

  // slot schedule: 0..5, core ×passes, хвост
  m->slot_phys_.resize(m->slot_count_);
  for (std::size_t s = 0; s < m->core_start_; ++s) m->slot_phys_[s] = s;
  std::size_t s = m->core_start_;
  for (std::size_t p = 0; p < m->n_passes_; ++p) {
    for (std::size_t l = m->core_start_; l < m->core_end_; ++l) {
      m->slot_phys_[s++] = l;
    }
  }
  for (std::size_t l = m->core_end_; l < m->n_phys_; ++l) {
    m->slot_phys_[s++] = l;
  }
  if (s != m->slot_count_) {
    return make_error(ErrCode::InvalidValue, "inkling2.loop",
                      "slot schedule mismatch");
  }

  // KV и conv-state
  const std::size_t kvd = m->n_kv_ * m->head_dim_;
  m->slots_.resize(m->slot_count_);
  for (auto& st : m->slots_) {
    st.k.assign(m->kv_ctx_ * kvd, 0);
    st.v.assign(m->kv_ctx_ * kvd, 0);
    st.conv.assign((m->l_cache_ - 1) * m->sconv_channels_, 0.0F);
  }

  // рабочие буферы
  auto& w = *m;
  w.x_.assign(m->hidden_, 0.0F);
  w.inp_sa_.assign(m->hidden_, 0.0F);
  w.q_.assign(m->n_head_ * m->head_dim_, 0.0F);
  w.k_.assign(kvd, 0.0F);
  w.v_.assign(kvd, 0.0F);
  w.r_.assign(m->n_relative_ * m->n_head_, 0.0F);
  w.gate_.assign(m->n_head_ * m->head_dim_, 0.0F);
  w.attn_.assign(m->n_head_ * m->head_dim_, 0.0F);
  w.o_.assign(m->hidden_, 0.0F);
  w.ffn_.assign(m->hidden_, 0.0F);
  w.tmp_.assign(m->hidden_, 0.0F);
  w.inp3_.assign(3 * m->hidden_, 0.0F);
  w.nrm3_.assign(3 * m->hidden_, 0.0F);
  w.mix_.assign(3 * m->hidden_, 0.0F);
  w.mix2_.assign(m->hidden_, 0.0F);
  w.ffn_in_.assign(m->hidden_, 0.0F);
  w.down_buf_.assign(m->hidden_, 0.0F);
  w.msg_.assign(4 * m->hidden_, 0.0F);
  w.norm_states_.assign(9 * m->hidden_, 0.0F);
  w.ctrl_z_.assign(64, 0.0F);
  w.ctrl_g_.assign(m->hidden_, 0.0F);
  w.ctrl_f_.assign(m->hidden_, 0.0F);
  w.z_.assign(2 * m->hidden_, 0.0F);
  w.dyn_k_.assign(m->l_cache_ * m->n_kv_, 0.0F);
  w.dyn_v_.assign(m->l_cache_ * m->n_kv_, 0.0F);
  w.sink_.assign(m->n_head_, 0.0F);
  w.scores_.assign(std::max<std::size_t>(m->kv_ctx_ + 1, m->n_swa_ + 1), 0.0F);
  w.table_.assign((m->relative_extent_ + 1) * m->n_head_, 0.0F);
  w.logits_.assign(m->vocab_padded_, -1e30F);
  w.scratch_.assign(std::max<std::size_t>(m->hidden_, 2 * m->n_ff_exp_), 0.0F);
  w.router_.assign(m->n_expert_, 0.0F);
  w.top_.assign(m->n_used_, 0);
  w.wts_.assign(m->n_used_, 0.0F);

  // Горячие dense-веса и голова — в Q8_0: у него есть NEON-ядро, а исходные
  // Q6_K/Q5_K/Q4_K/IQ3_XXS деквантуются скалярно на каждом токене.
  if (kernels::has_neon_dot(QuantType::Q8_0)) {
    auto rb8_of = [](const InkMeta& meta) -> std::size_t {
      const auto rb = row_bytes(QuantType::Q8_0, meta.in_dim);
      return rb.ok() ? static_cast<std::size_t>(rb.value()) : 0;
    };
    std::size_t need = 0;
    auto plan = [&](const InkMeta& meta) {
      if (meta.base == 0 || meta.quant == QuantType::Q8_0 ||
          meta.quant == QuantType::F32 || meta.quant == QuantType::Bf16) {
        return;
      }
      const std::size_t rb = rb8_of(meta);
      if (rb != 0) need += rb * meta.rows;
    };
    for (const auto& L : m->layers_) {
      plan(L.q); plan(L.k); plan(L.v); plan(L.o); plan(L.gate); plan(L.rel);
    }
    plan(m->out_);
    m->hot_.resize(need);
    std::size_t off = 0;
    auto convert = [&](InkMeta& meta) {
      if (meta.base == 0 || meta.quant == QuantType::Q8_0 ||
          meta.quant == QuantType::F32 || meta.quant == QuantType::Bf16) {
        return;
      }
      const std::size_t rb = rb8_of(meta);
      if (rb == 0) return;
      const std::size_t numel =
          static_cast<std::size_t>(meta.in_dim) * meta.rows;
      std::vector<float> tmp(numel);
      const auto deq = dequant_into(
          meta.quant, std::span<const uint8_t>(m->data_ + meta.base,
              static_cast<std::size_t>(meta.rb) * meta.rows), numel,
          std::span<float>(tmp.data(), tmp.size()));
      if (!deq.ok()) return;
      const auto q = quantize_q8_0_into(
          std::span<const float>(tmp.data(), tmp.size()),
          std::span<uint8_t>(m->hot_.data() + off, rb * meta.rows));
      if (!q.ok()) return;
      meta.base = off;
      meta.rb = static_cast<uint32_t>(rb);
      meta.quant = QuantType::Q8_0;
      meta.hot = true;
      off += rb * meta.rows;
    };
    for (auto& L : m->layers_) {
      convert(L.q); convert(L.k); convert(L.v); convert(L.o); convert(L.gate);
      convert(L.rel);
    }
    convert(m->out_);
    m->hot_.resize(off);
    std::fprintf(stderr, "[inkling2] hot Q8_0: %.2f GB\n",
                 static_cast<double>(off) / 1e9);
  }

  std::fprintf(stderr,
               "[inkling2] phys=%zu slots=%zu passes=%zu core=[%zu,%zu) "
               "hidden=%zu heads=%zu/%zu dim=%zu experts=%zu top=%zu ff=%zu "
               "rel=%zu/%zu swa=%zu vocab=%zu/%zu kv_ctx=%zu\n",
               m->n_phys_, m->slot_count_, m->n_passes_, m->core_start_,
               m->core_end_, m->hidden_, m->n_head_, m->n_kv_,
               m->head_dim_, m->n_expert_, m->n_used_, m->n_ff_exp_,
               m->n_relative_, m->relative_extent_, m->n_swa_,
               m->vocab_unpadded_, m->vocab_padded_, m->kv_ctx_);
  std::fprintf(stderr, "[inkling2] softcap=%g situ=%g/%g expert_scale=%g\n",
               m->final_softcap_, m->situ_beta_, m->situ_linear_beta_,
               m->expert_scale_);

  std::fflush(stderr);
  return m;
}

}  // namespace tokshit
