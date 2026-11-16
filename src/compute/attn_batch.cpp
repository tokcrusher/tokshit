// one batch row attention over the kv slabs. NEON: four accumulator chains
// for q*k, vectorised f16->f32 and v-axpy, optional parallelism over kv heads.
// same math as Model::layer: scaled scores, the sink joins the softmax
// denominator only and never enters the weighted v sum.
#include "tokshit/attn_batch.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include "tokshit/dequant.hpp"
#include "tokshit/pool.hpp"
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tokshit::attn {
namespace {

inline float h2f(std::uint16_t h) {
    const std::uint32_t bits = h;
    const std::uint32_t exp = (bits >> 10) & 0x1Fu;
    if (exp != 0 && exp != 31) {
        const std::uint32_t f = ((bits & 0x8000u) << 16) |
                                ((exp + 112u) << 23) | ((bits & 0x3FFu) << 13);
        float r;
        std::memcpy(&r, &f, sizeof(float));
        return r;
    }
    return f16_to_f32(h);
}

inline void load_row(const std::uint8_t* base, bool f16, std::size_t n, float* out) {
    if (!f16) { std::memcpy(out, base, n * sizeof(float)); return; }
    const auto* p = reinterpret_cast<const std::uint16_t*>(base);
    std::size_t i = 0;
#if defined(__aarch64__)
    const auto* q = reinterpret_cast<const float16_t*>(base);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t h = vld1q_f16(q + i);
        vst1q_f32(out + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(out + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vcvt_f32_f16(vld1_f16(q + i)));
#endif
    for (; i < n; ++i) out[i] = h2f(p[i]);
}

inline float dot_f32(const float* a, const float* b, std::size_t n) {
#if defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0.0F), s1 = vdupq_n_f32(0.0F);
    float32x4_t s2 = vdupq_n_f32(0.0F), s3 = vdupq_n_f32(0.0F);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    float r = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
#else
    float r = 0.0F;
    for (std::size_t i = 0; i < n; ++i) r += a[i] * b[i];
    return r;
#endif
}

inline void axpy_f32(float* acc, float w, const float* v, std::size_t n) {
#if defined(__aarch64__)
    const float32x4_t wv = vdupq_n_f32(w);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(acc + i,      vfmaq_f32(vld1q_f32(acc + i),      wv, vld1q_f32(v + i)));
        vst1q_f32(acc + i + 4,  vfmaq_f32(vld1q_f32(acc + i + 4),  wv, vld1q_f32(v + i + 4)));
        vst1q_f32(acc + i + 8,  vfmaq_f32(vld1q_f32(acc + i + 8),  wv, vld1q_f32(v + i + 8)));
        vst1q_f32(acc + i + 12, vfmaq_f32(vld1q_f32(acc + i + 12), wv, vld1q_f32(v + i + 12)));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(acc + i, vfmaq_f32(vld1q_f32(acc + i), wv, vld1q_f32(v + i)));
    for (; i < n; ++i) acc[i] += w * v[i];
#else
    for (std::size_t i = 0; i < n; ++i) acc[i] += w * v[i];
#endif
}

struct KvhJob {
    const AttendArgs* a = nullptr;
    const float* qrow = nullptr;
    float* sc = nullptr;
    float* orow = nullptr;
    std::size_t start = 0, len = 0, group = 0;
};

void kvh_work(const KvhJob& j, std::size_t kvh) {
    const AttendArgs& a = *j.a;
    const std::size_t hd = a.head_dim, group = j.group, len = j.len;
    const std::size_t h0 = kvh * group;
    const std::size_t koff = kvh * hd * (a.f16 ? 2 : 4);

    // слайс скоров этого kvh. раньше все воркеры писали один и тот же диапазон
    // sc[gi*len+col]: индекс не включал kvh, и на батчах с len*group >= 2048
    // (позиции >= ~254 на флинте) головы 8..31 softmax'или и взвешивали скоры
    // чужих голов, а в пуле ещё и гонялись за одними ячейками. на коротком
    // контексте путь сериальный и сломанное место не выполнялось
    // per-kvh score slice. all workers used to write the same sc[gi*len+col]
    // range: the index missed kvh, so on batches with len*group >= 2048
    // (positions >= ~254 on flint) heads 8..31 softmaxed and weighted other
    // heads' scores, and the pool ran the same cells concurrently. short
    // contexts take the serial path and never execute the broken code
    float* sc = j.sc + h0 * len;
    thread_local std::vector<float> kbuf, vbuf, acc;
    kbuf.resize(hd);
    vbuf.resize(hd);
    acc.resize(group * hd);
    std::fill(acc.begin(), acc.end(), 0.0F);

    for (std::size_t p = j.start; p < j.start + len; ++p) {
        load_row(a.k + p * a.k_stride + koff, a.f16, hd, kbuf.data());
        const std::size_t col = p - j.start;
        for (std::size_t gi = 0; gi < group; ++gi) {
            sc[gi * len + col] =
                dot_f32(j.qrow + (h0 + gi) * hd, kbuf.data(), hd) * a.scale;
        }
    }
    for (std::size_t gi = 0; gi < group; ++gi) {
        const std::size_t h = h0 + gi;
        float* s = sc + gi * len;
        float m = a.sinks != nullptr ? a.sinks[h] : -1e30F;
        for (std::size_t c = 0; c < len; ++c) m = std::max(m, s[c]);
        float z = a.sinks != nullptr ? std::exp(a.sinks[h] - m) : 0.0F;
        for (std::size_t c = 0; c < len; ++c) { s[c] = std::exp(s[c] - m); z += s[c]; }
        const float inv = z > 0.0F ? 1.0F / z : 0.0F;
        for (std::size_t c = 0; c < len; ++c) s[c] *= inv;
    }
    for (std::size_t p = j.start; p < j.start + len; ++p) {
        load_row(a.v + p * a.v_stride + koff, a.f16, hd, vbuf.data());
        const std::size_t col = p - j.start;
        for (std::size_t gi = 0; gi < group; ++gi)
            axpy_f32(acc.data() + gi * hd, sc[gi * len + col], vbuf.data(), hd);
    }
    for (std::size_t gi = 0; gi < group; ++gi)
        std::memcpy(j.orow + (h0 + gi) * hd, acc.data() + gi * hd, hd * sizeof(float));
}

void kvh_task(void* ctx, std::size_t i) { kvh_work(*static_cast<const KvhJob*>(ctx), i); }

}  // namespace

void attend_row(const AttendArgs& a, std::size_t row) {
    if (a.head_dim == 0 || a.n_kvh == 0 || a.n_heads == 0) return;
    const std::size_t group = a.n_heads / a.n_kvh;
    if (group == 0) return;
    const std::size_t end = std::min(a.first + row + 1, a.cap);
    const std::size_t start = a.win != nullptr ? std::min(a.win[row], end) : 0;
    if (end <= start) return;
    const std::size_t len = end - start;

    thread_local std::vector<float> sc;
    // n_heads слябов по len: воркер kvh занимает [kvh*group, (kvh+1)*group)
    // n_heads slices of len: worker kvh owns [kvh*group, (kvh+1)*group)
    sc.resize(a.n_heads * len);

    KvhJob job;
    job.a = &a;
    job.qrow = a.q + row * a.sq;
    job.sc = sc.data();
    job.orow = a.out + row * a.sq;
    job.start = start;
    job.len = len;
    job.group = group;

    // a pool barrier on a 3-token context costs more than the attention itself
    if (a.n_kvh > 1 && len * group >= 2048) {
        pool::run_tasks(a.n_kvh, &kvh_task, &job);
    } else {
        for (std::size_t kvh = 0; kvh < a.n_kvh; ++kvh) kvh_work(job, kvh);
    }
}
}  // namespace tokshit::attn
