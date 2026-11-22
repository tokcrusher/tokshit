// temp debug: compare Inkling2::moe() vs Inkling2::moe_batch() in detail
#define private public
#include "tokshit/inkling2.hpp"
#undef private

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tokshit/gguf.hpp"
#include "tokshit/mmap.hpp"
#include "tokshit/pool.hpp"

using namespace tokshit;

static double maxdiff(const float* a, const float* b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
  return m;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  auto g = Gguf::open(argv[1]);
  if (!g.ok()) return 1;
  auto mm = g.value().mmap_file();
  if (!mm.ok()) return 1;
  Mmap m = std::move(mm.value());
  auto r = Inkling2::load(g.value(), m, 2048);
  if (!r.ok()) return 1;
  auto& ink = *r.value();
  ink.set_router_topk(0);
  const std::size_t h = ink.n_embd();
  const InkLayer& L = ink.layers_[0];
  const std::size_t nff = ink.n_ff_exp_;

  std::vector<float> x(h);
  for (std::size_t i = 0; i < h; ++i)
    x[i] = std::sin(static_cast<float>(i) * 0.7F) * 1.0F + 0.01F * i;

  pool::set_active(true);
  std::vector<float> out_seq(h, 0.0F);
  ink.moe(L, x.data(), out_seq.data());
  std::vector<float> out_bat(h, 0.0F);
  ink.moe_batch(L, x.data(), out_bat.data(), 1);
  pool::set_active(false);

  std::printf("max|dout|=%.9g\n", maxdiff(out_seq.data(), out_bat.data(), h));
  std::printf("router_seq[0..3]:");
  for (int i = 0; i < 4; ++i) std::printf(" %.6f", ink.router_[i]);
  std::printf("\nrouter_bat[0..3]:");
  for (int i = 0; i < 4; ++i) std::printf(" %.6f", ink.router_b_[i]);
  std::printf("\n");
  std::printf("top_seq:");
  for (std::size_t k = 0; k < ink.n_used_; ++k)
    std::printf(" %d(%.4f)", ink.top_[k], ink.wts_[k]);
  std::printf("\ntop_bat:");
  for (std::size_t k = 0; k < ink.n_used_; ++k)
    std::printf(" %d(%.4f)", ink.top_b_[k], ink.wts_b_[k]);
  std::printf("\n");

  // gu_all_[k*2nff] vs gu_b_[at*2nff] for pair 0
  // compare down outputs: sequential down_all_[k*h] vs batch dn_b_[at*h]
  {
    double md = 0.0;
    std::size_t dbad = 0;
    for (std::size_t k = 0; k < ink.n_used_; ++k) {
      const int e = ink.top_[k];
      if (e < 0) continue;
      // find the pair for k: iterate pair_at_b_
      for (std::size_t kk = 0; kk < ink.n_used_; ++kk) {
        if (ink.pair_at_b_[kk] == std::numeric_limits<std::size_t>::max())
          continue;
        if (ink.top_b_[kk] != e) continue;
        for (std::size_t i = 0; i < h; ++i) {
          const double d = std::fabs(ink.down_all_[k * h + i] -
                                     ink.dn_b_[ink.pair_at_b_[kk] * h + i]);
          md = std::max(md, d);
          if (d != 0.0) ++dbad;
        }
      }
    }
    std::printf("max|ddown|=%.9g dbad=%zu\n", md, dbad);
    std::printf("top_[0..1]: %d %d, top_b_[0..1]: %d %d\n", ink.top_[0],
                ink.top_[1], ink.top_b_[0], ink.top_b_[1]);
    std::printf("utop_b_[0..3]: %zu %zu %zu %zu\n", ink.utop_b_[0],
                ink.utop_b_[1], ink.utop_b_[2], ink.utop_b_[3]);
    std::printf("moff_b_[0..3]: %zu %zu %zu %zu\n", ink.moff_b_[0],
                ink.moff_b_[1], ink.moff_b_[2], ink.moff_b_[3]);
    std::printf("pair_at_[0..7]:");
    for (int k = 0; k < 8; ++k) std::printf(" %zu", ink.pair_at_b_[k]);
    std::printf("\n");
    std::printf("gu_bat[0][0..4]:");
    for (int i = 0; i < 5; ++i) std::printf(" %.6f", ink.gu_b_[i]);
    std::printf("\n");
    std::printf("gu_seq[0..4]:");
    for (int i = 0; i < 5; ++i) std::printf(" %.6f", ink.gu_all_[i]);
    std::printf("\n");
    std::printf("moe_in_b_[0].q[0..3]:");
    for (int i = 0; i < 4; ++i) std::printf(" %d", ink.moe_in_b_[0].q[i]);
    std::printf("\nact_q8_[0..3]:");
    for (int i = 0; i < 4; ++i) std::printf(" %d", ink.act_q8_.q[i]);
    std::printf("\n");
    std::printf("expert109_bytes[0..7]:");
    const uint8_t* eb = ink.data_ + L.gate_up_exps.base +
                        109ULL * L.gate_up_exps.rb * L.gate_up_exps.rows;
    for (int i = 0; i < 8; ++i) std::printf(" %02x", eb[i]);
    std::printf("\n");
  }
  std::printf("gu_seq[0..4]:");
  for (int i = 0; i < 5; ++i) std::printf(" %.6f", ink.gu_all_[i]);
  std::printf("\n");
  // find pair 0's at
  std::printf("pair_at_b_[0..7]:");
  for (int k = 0; k < 8; ++k) {
    if (ink.pair_at_b_[k] == std::numeric_limits<std::size_t>::max())
      std::printf(" ~");
    else
      std::printf(" %zu", ink.pair_at_b_[k]);
  }
  std::printf("\n");
  for (std::size_t at = 0; at < ink.utop_b_.size() && at < 8; ++at) {
    std::printf("gu_bat[%zu][0..4]:", at);
    for (int i = 0; i < 5; ++i)
      std::printf(" %.6f", ink.gu_b_[at * 2 * nff + i]);
    std::printf("\n");
  }
  // find sequential slot 0's expert
  std::printf("seq expert for k=0: %d\n", ink.top_[0]);
  std::size_t e0 = static_cast<std::size_t>(ink.top_[0]);
  std::printf("gu_seq for top0 (k=0) [0..4]:");
  for (int i = 0; i < 5; ++i) std::printf(" %.6f", ink.gu_all_[i]);
  // find which bat pair belongs to seq k=0
  for (std::size_t k = 0; k < 8; ++k) {
    if (ink.pair_at_b_[k] == std::numeric_limits<std::size_t>::max()) continue;
    std::printf("bat pair k=%zu at=%zu ex=%zu w=%.4f vs seq k=0 ex=%zu w=%.4f\n",
                k, ink.pair_at_b_[k], ink.utop_b_[0], ink.wts_b_[k],
                e0, ink.wts_[0]);
    if (ink.utop_b_[0] == e0) {
      std::printf("gu_bat[pair0][0..4]:");
      for (int i = 0; i < 5; ++i) std::printf(" %.6f", ink.gu_b_[ink.pair_at_b_[k] * 2 * nff + i]);
      std::printf("  vs gu_seq[0..4]:");
      for (int i = 0; i < 5; ++i) std::printf(" %.6f", ink.gu_all_[i]);
      std::printf("\n");
      break;
    }
  }
  return 0;
}
