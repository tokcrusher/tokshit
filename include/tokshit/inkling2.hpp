#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "tokshit/ahb.hpp"
#include "tokshit/batch.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/quant.hpp"
#include "tokshit/result.hpp"

namespace tokshit {

class Gguf;
class Mmap;

// Смещение и геометрия одного тензора в mmap. base — абсолютное смещение
// данных в файле, rb — байт на строку, rows — число строк.
struct InkMeta {
  uint64_t base = 0;
  uint32_t in_dim = 0;
  uint32_t rows = 0;
  uint32_t rb = 0;
  QuantType quant = QuantType::F32;
  bool hot = false;  // base — смещение в hot_ (Q8_0), а не в файле
};

// Один физический блок Inkling-2. Луп-поля заполняются только у core-блоков.
struct InkLayer {
  InkMeta q, k, v, o, gate, rel;
  InkMeta sink_proj;  // [n_embd, n_head], BF16
  std::vector<float> q_clip;     // [head_dim, n_head], F32
  std::vector<float> rel_bias;   // [n_relative, n_relative_extent], F32
  std::vector<float> rel_clip;   // [n_relative, n_head], F32

  std::vector<float> attn_norm;      // SWA-блоки
  std::vector<float> q_inp_norm;     // global-блоки
  std::vector<float> k_inp_norm;
  std::vector<float> v_inp_norm;
  std::vector<float> ffn_norm;

  InkMeta k_conv, v_conv, post_conv, ffn_conv;  // F32 [l_cache, channels]
  InkMeta k_conv_proj, v_conv_proj;             // BF16 [n_embd, l_cache*n_kv]

  InkMeta gate_inp;        // F32 [n_embd, n_expert]
  InkMeta gate_up_exps;    // [n_embd, 2*n_ff_exp, n_expert]
  InkMeta down_exps;       // [n_ff_exp, n_embd, n_expert]
  std::vector<float> exp_probs_b;
  std::vector<float> gate_up_scale;  // [n_expert] или пусто
  std::vector<float> down_scale;

  std::vector<float> attn_res_query;  // [n_embd] или [n_embd,3]; пусто у blk.0
  std::vector<float> ffn_res_query;   // [n_embd]
  // legacy layout: msg_query[0..3] = Q, K, V, FFN для core-слоя
  std::vector<float> msg_query[4];

  // loop-контроллеры: down [2*n_embd, rank], gain/forget [rank, n_embd]
  InkMeta loop_down[4];
  InkMeta loop_gain[4];
  InkMeta loop_forget[4];
  bool has_loop = false;
};

class Inkling2 {
 public:
  Inkling2();
  ~Inkling2();
  Inkling2(const Inkling2&) = delete;
  Inkling2& operator=(const Inkling2&) = delete;

  // mm и gguf остаются во владении Model; указатель на данные mmap должен
  // жить всё время работы.
  [[nodiscard]] static Result<std::unique_ptr<Inkling2>> load(
      const Gguf& gguf, const Mmap& mm, std::size_t kv_ctx);

  void token_logits(uint32_t token, std::size_t pos,
                    std::vector<float>& out);

  // Layer-major batched verify/prefill. `tokens` are consecutive positions
  // starting at `pos`; `out` is b*vocab_unpadded. The K/V and conv state end
  // after the last token; use restore_batch() to roll back a rejected tail.
  void token_logits_batch(std::span<const uint32_t> tokens, std::size_t pos,
                          std::span<float> out);
  [[nodiscard]] std::span<const float> taps_batch() const noexcept {
    return batch_taps_;
  }
  [[nodiscard]] std::size_t last_batch() const noexcept { return last_batch_; }
  void restore_batch(std::size_t accepted);

  // DSpark target taps: slot indices whose *input* hidden is captured
  // (slot_count means the final normalized hidden). Ascending order.
  void set_tap_slots(std::span<const uint32_t> slots);
  [[nodiscard]] std::span<const float> taps() const noexcept { return tap_buf_; }
  [[nodiscard]] std::size_t n_taps() const noexcept { return tap_slots_.size(); }

  // Shared target tensors used by the draft: token embedding + tok_norm and
  // the LM head. embed_normed writes `hidden_` floats.
  void embed_normed(uint32_t token, float* out);
  void lm_head(const float* x, float* out);
  // Batched LM head: b rows of `stride` floats, output b*vocab_padded.
  void lm_head_batch(const float* x, std::size_t b, std::size_t stride,
                     float* out);

  // Conv/high-water state snapshot for speculative rollback. K/V rows above
  // the restored high-water mark are ignored by the next forward.
  struct Snapshot {
    std::vector<std::vector<float>> conv;
    std::vector<std::size_t> high;
  };
  void snapshot(Snapshot& s) const;
  void restore(const Snapshot& s);

  [[nodiscard]] std::size_t vocab() const noexcept { return vocab_unpadded_; }
  [[nodiscard]] std::size_t vocab_padded() const noexcept {
    return vocab_padded_;
  }
  [[nodiscard]] std::size_t n_embd() const noexcept { return hidden_; }
  [[nodiscard]] std::size_t n_slots() const noexcept { return slot_count_; }
  [[nodiscard]] std::size_t n_phys() const noexcept { return n_phys_; }
  [[nodiscard]] std::size_t n_experts() const noexcept { return n_expert_; }
  [[nodiscard]] std::size_t n_used() const noexcept { return n_used_; }
  void set_router_topk(std::size_t k) noexcept { router_topk_ = k; }
  [[nodiscard]] std::size_t kv_ctx() const noexcept { return kv_ctx_; }

 private:
  struct SlotState {
    std::vector<uint16_t> k;
    std::vector<uint16_t> v;
    std::vector<float> conv;
    std::size_t high = 0;
  };

  void reset();
  void embed(uint32_t token);
  void capture_tap(std::size_t slot);
  void ensure_batch(std::size_t b);
  void capture_batch_tap(std::size_t slot);
  void run_app_batch(std::size_t slot, std::size_t ip, int pass,
                     const float* msg_b, std::size_t pos0, std::size_t b);
  void moe_batch(const InkLayer& L, float* x, float* out, std::size_t b);
  void apply_layer(std::size_t slot, std::size_t pass, bool has_msg,
                   const std::vector<std::vector<float>>& bank,
                   const std::vector<float>& bank_out);
  void moe(const InkLayer& L, const float* x, float* out);
  void matvec(const InkMeta& m, const float* x, float* y);
  void matvec(const InkMeta& m, const float* x, float* y, float* scratch);
  struct MatvecSeg {
    const InkMeta* meta = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
    const kernels::ActQ8* act = nullptr;
  };
  void matvec_multi(std::span<const MatvecSeg> segs);
  void depth_mix(std::span<const float* const> states, const float* cur,
                 const float* query, std::size_t nq, float* out);
  void shortconv(const InkLayer& L, const InkMeta& w, const float* dyn,
                 std::size_t offset, std::size_t n_channels, float* x);
  void control(float* x, const float* msg, const InkLayer& L, int ctrl);
  void attention(const InkLayer& L, std::size_t slot, std::size_t pos,
                 bool swa, float tau, const float* in_q, const float* in_k,
                 const float* in_v, float* out);
  void rmsnorm(std::span<float> x, std::span<const float> w, float eps);
  [[nodiscard]] const InkLayer& phys(std::size_t slot) const;

  const Gguf* gguf_ = nullptr;
  const uint8_t* data_ = nullptr;
  uint64_t data_len_ = 0;

  std::vector<InkLayer> layers_;
  std::vector<std::size_t> slot_phys_;

  InkMeta tok_embd_, out_;
  std::vector<float> tok_norm_, out_norm_, out_res_query_;
  std::vector<float> loop_kv_query_, loop_res_query_;  // [n_embd * n_phys]

  std::size_t n_phys_ = 0;
  std::size_t slot_count_ = 0;
  std::size_t core_start_ = 0;
  std::size_t core_end_ = 0;
  std::size_t n_passes_ = 0;
  std::size_t hidden_ = 0;
  std::size_t n_head_ = 0;
  std::size_t n_kv_ = 0;
  std::size_t head_dim_ = 0;
  std::size_t n_expert_ = 0;
  std::size_t n_used_ = 0;
  std::size_t router_topk_ = 0;
  std::size_t n_ff_exp_ = 0;
  std::size_t n_relative_ = 0;
  std::size_t relative_extent_ = 0;
  std::size_t n_swa_ = 0;
  std::size_t l_cache_ = 0;
  std::size_t sconv_channels_ = 0;
  std::size_t block_size_ = 0;
  std::size_t vocab_unpadded_ = 0;
  std::size_t vocab_padded_ = 0;
  std::size_t kv_ctx_ = 2048;
  float eps_ = 1e-6F;
  float qk_clip_ = 0.0F;
  float rel_clip_ = 0.0F;
  float log_scale_alpha_ = 0.0F;
  std::size_t log_scale_floor_ = 0;
  float situ_beta_ = 1.0F;
  float situ_linear_beta_ = 0.0F;
  float expert_scale_ = 0.0F;
  float final_softcap_ = 0.0F;

  std::vector<SlotState> slots_;
  std::vector<uint8_t> slot_swa_;
  std::vector<float> x_, inp_sa_, q_, k_, v_, r_, gate_, attn_, o_, ffn_, tmp_;
  std::vector<float> inp3_, nrm3_, mix_, mix2_, msg_, ctrl_z_, ctrl_g_,
      ctrl_f_, z_, norm_states_, ffn_in_, down_buf_;
  std::vector<float> dyn_k_, dyn_v_, sink_, scores_;
  std::vector<float> table_;  // [(extent+1) * n_head]
  std::vector<float> logits_;
  std::vector<float> scratch_;
  std::vector<int> top_;
  std::vector<float> wts_;
  std::vector<float> router_;
  std::vector<float> gu_all_, act_all_, down_all_;
  kernels::ActQ8 act_q8_;
  kernels::ActQ8 act_in_[3];
  std::vector<kernels::ActQ8> act_all_q8_;
  // moe_batch: union экспертов батча на b токенов + flat-пары (токен, вес)
  std::vector<float> router_b_;          // B × n_expert_: открытые роутеры
  std::vector<kernels::ActQ8> router_acts_;  // B: активации gate_inp (in_dim = h)
  std::vector<int> top_b_;               // B × n_used_: топ-эксперты на токен
  std::vector<float> wts_b_;             // B × n_used_: веса (норм. × expert_scale)
  std::vector<std::size_t> pair_at_b_;   // B × n_used_: пара → flat-индекс (MAX = нет)
  std::vector<std::uint8_t> seen_b_;     // n_expert_
  std::vector<int> emap_b_;              // n_expert_: эксперт → индекс в utop_b_
  std::vector<std::size_t> utop_b_;      // ≤ B × n_used_: union экспертов
  std::vector<std::size_t> moff_b_;      // n_expert_: старты групп по индексу utop
  std::vector<std::size_t> mcur_b_;      // n_expert_: курсоры при заполнении
  std::vector<float> gx_b_;              // total × h: входы gate_up пар
  std::vector<kernels::ActQ8> moe_in_b_;  // total: кванты входов gate_up
  std::vector<float> gu_b_;              // total × 2nff: выходы gate_up
  std::vector<float> act_b_;             // total × nff: SiTU (входы down)
  std::vector<kernels::ActQ8> moe_act_b_;  // total: кванты SiTU → down
  std::vector<float> dn_b_;              // total × h: выходы down
  std::vector<BatchSeg> moe_gu_segs_;    // segs gate_up (один проход)
  std::vector<BatchSeg> moe_dn_segs_;    // segs down (овторный проход)
  std::vector<std::vector<uint32_t>> prev_top_;
  std::vector<uint8_t> prev_top_valid_;
  std::vector<uint32_t> tap_slots_;
  std::vector<float> tap_buf_;
  std::vector<kernels::ActQ8> lm_acts_;

  // batched verify buffers
  std::size_t batch_cap_ = 0;
  std::size_t batch_pos0_ = 0;
  std::size_t last_batch_ = 0;
  std::size_t max_blocks_ = 16;
  std::vector<float> xb_, inp_sab_, nrm3b_, inp3b_, mixb_, mix2b_, msg_b_,
      ob_, ffnb_, ffn_in_b_, prev_out_b_, prelude_b_, hb_;
  std::vector<float> qb_, kb_, vb_, rb_, gateb_, attnb_, sinkb_, dyn_kb_,
      dyn_vb_;
  std::vector<float> block_bank_, bank_b_, prelude_blocks_b_;
  std::vector<std::size_t> block_count_, bank_count_, prelude_count_;
  std::vector<float> conv_before_, conv_steps_;
  std::vector<float> batch_taps_;
  std::vector<float> logits_b_;
  std::vector<kernels::ActQ8> acts_b_;
  std::vector<const float*> state_ptrs_;
  std::vector<uint8_t> hot_;  // Q8_0 dense + output, один буфер
  double prof_mv_ns_ = 0.0;
  double prof_moe_ns_ = 0.0;
  std::size_t cur_slot_ = 0;
};

}  // namespace tokshit
