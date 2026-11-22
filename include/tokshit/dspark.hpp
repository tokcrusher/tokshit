#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "tokshit/inkling2.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/quant.hpp"
#include "tokshit/result.hpp"

namespace tokshit {

class Gguf;
class Mmap;

// DSpark drafter for Inkling-2-Mini. The GGUF is an `dflash` architecture:
// three non-causal draft layers plus a semi-autoregressive Markov head. The
// target model owns the token embedding, its norm and the LM head; this class
// only borrows them.
class DSpark {
 public:
  DSpark();
  ~DSpark();
  DSpark(const DSpark&) = delete;
  DSpark& operator=(const DSpark&) = delete;

  [[nodiscard]] static Result<std::unique_ptr<DSpark>> load(
      const Gguf& gguf, const Mmap& mm, Inkling2& target, std::size_t kv_ctx);

  [[nodiscard]] std::size_t block_size() const noexcept { return block_size_; }
  [[nodiscard]] std::size_t max_draft() const noexcept {
    return block_size_ > 0 ? block_size_ - 1 : 0;
  }
  [[nodiscard]] std::span<const uint32_t> target_slots() const noexcept {
    return target_slots_;
  }
  [[nodiscard]] std::size_t kv_ctx() const noexcept { return kv_ctx_; }

  // Fuse the four target taps for one committed position and write the
  // per-layer context K/V plus the head context into the ring.
  void inject(std::size_t pos, std::span<const float> taps, bool store_ctx = true);

  // Run the draft block for `pos0` and return up to `n` greedy tokens.
  // `anchor` is the last committed token (at target position pos0-1).
  void draft(uint32_t anchor, std::size_t pos0, std::size_t n,
             std::vector<uint32_t>& out);

  // The head context vector for the anchor at `pos`, or nullptr if absent.
  [[nodiscard]] const float* context(std::size_t pos) const;

 private:
  struct Layer {
    InkMeta q, k, v, o, gate, up, down;
    InkMeta k_ctx, v_ctx;
    std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, k_ctx_norm,
        dflare_norm;
  };
  struct LayerKv {
    std::vector<uint16_t> k, v;
  };

  [[nodiscard]] const uint8_t* bytes(const InkMeta& m) const noexcept {
    return data_ + m.base;
  }
  void matvec(const InkMeta& m, const float* x, float* y);
  void matvec_f32(const std::vector<float>& w, std::size_t rows,
                  std::size_t cols, const float* x, float* y) const;
  void rmsnorm(std::span<float> x, std::span<const float> w) const;
  void rmsnorm(std::span<float> x) const;
  void rope(std::span<float> x, std::size_t n_heads, std::size_t pos) const;
  void attention(std::size_t layer, std::size_t q_index, std::size_t pos0,
                 const float* q, float* out);
  void fuse(std::size_t row, std::span<const float> taps, float* out,
            std::span<const float> norm) const;
  void head_step(const float* hidden, const float* context, uint32_t prev,
                 std::vector<float>& state, std::vector<float>& logits,
                 bool have_base, const float* base);

  const uint8_t* data_ = nullptr;
  std::size_t data_len_ = 0;
  Inkling2* target_ = nullptr;

  std::vector<Layer> layers_;
  std::vector<LayerKv> kv_;
  std::size_t n_layer_ = 0;
  std::size_t hidden_ = 0;
  std::size_t n_head_ = 0;
  std::size_t n_kv_ = 0;
  std::size_t head_dim_ = 0;
  std::size_t ffn_ = 0;
  std::size_t rank_ = 0;
  std::size_t vocab_padded_ = 0;
  std::size_t vocab_unpadded_ = 0;
  std::size_t block_size_ = 0;
  std::size_t kv_ctx_ = 2048;
  std::size_t swa_ = 0;
  std::size_t ctx_slots_ = 0;
  float eps_ = 1e-6F;
  float rope_base_ = 1e6F;
  float softcap_ = 30.0F;
  std::vector<uint8_t> swa_layer_;
  std::vector<uint32_t> target_slots_;

  InkMeta fuse_;        // [4,4] F32
  InkMeta markov_w1_;   // [rank, vocab_padded] BF16
  InkMeta joint_;       // [2*embd+2*rank, 3*rank] BF16
  std::vector<float> joint_bias_;
  InkMeta hidden_out_;  // [rank, embd] BF16
  std::vector<float> hidden_out_bias_;
  std::vector<float> mask_embd_;
  std::vector<float> output_norm_;
  std::vector<float> dspark_norm_;
  std::vector<float> joint_f32_;   // dequantized BF16
  std::vector<float> hout_f32_;    // dequantized BF16

  std::vector<float> ctx_ring_;    // [ctx_slots * hidden]
  std::vector<int64_t> ctx_pos_;

  // scratch
  std::vector<float> x_, q_, k_, v_, attn_, ffn_in_, gate_, up_, act_, out_;
  std::vector<float> fuse_ctx_, kctx_, vctx_, state_, joint_buf_, joint_out_,
      corr_;
  std::vector<float> base_, logits_, row_, scores_, norm_;
  std::vector<kernels::ActQ8> acts_;
  std::vector<float> scratch_;
  std::size_t injected_high_ = 0;
};

}  // namespace tokshit
