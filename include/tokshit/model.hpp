#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache_block.hpp"  // CacheBlock

#include "tokshit/ahb.hpp"
#include "tokshit/attn_batch.hpp"
#include "tokshit/expert_io.hpp"
#include "tokshit/gguf.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/kv.hpp"
#include "tokshit/mmap.hpp"
#include "tokshit/quant.hpp"
#include "tokshit/result.hpp"

namespace tokshit {

using KVCache = std::vector<std::vector<float>>;

struct Meta {
  uint64_t base = 0;
  std::size_t in_dim = 0;
  std::size_t out_dim = 0;
  std::size_t rb = 0;
  QuantType quant = QuantType::F32;
  bool hot = true;
};

struct ExpertMeta {
  uint64_t base = 0;
  std::size_t in_dim = 0;
  std::size_t rows = 0;
  std::size_t rb = 0;
  QuantType quant = QuantType::F32;
  std::size_t size = 0;
};

struct Layer {
  std::vector<float> attn_norm;
  Meta attn_q;
  Meta attn_k;
  Meta attn_v;
  std::vector<float> attn_q_norm;
  std::vector<float> attn_k_norm;
  std::vector<float> attn_sinks;
  Meta attn_o;
  std::vector<float> ffn_norm;

  std::optional<Meta> ffn_gate;
  std::optional<Meta> ffn_up;
  std::optional<Meta> ffn_down;

  std::optional<Meta> gate_inp;
  std::optional<ExpertMeta> gate_exps;
  std::optional<ExpertMeta> up_exps;
  std::optional<ExpertMeta> down_exps;
  std::optional<Meta> gate_shexp;
  std::optional<Meta> up_shexp;
  std::optional<Meta> down_shexp;
};

struct ModelOptions {
  // 1 гиг по умолчанию: на телефоне больше легко уходит в своп
  // 1 gig by default: more than that easily pushes a phone into swap
  uint64_t expert_cache_mb = 1024;
  bool hot = true;
  bool mmap_willneed = false;
  std::size_t io_threads = 4;
};

struct CacheEntry {
  CacheBlock data;
  uint64_t last = 0;
  uint32_t size = 0;
  std::list<uint64_t>::iterator lru;
};

// послойный lru. попадание/промах/эвикция за o(1), через intrusive список
// ключей. старая версия на каждую эвикцию сканила весь map целиком, это было
// тупо и медленно
// per-layer lru. o(1) hit/miss/eviction via an intrusive key list, the old
// one scanned the whole map on every eviction, was dumb and slow
class ResidentCache {
 public:
  explicit ResidentCache(std::size_t layers);

  // отдаёт блок из кэша и помечает его самым свежим, или nullptr если нет
  // returns the block and marks it most-recently used, or nullptr
  [[nodiscard]] CacheBlock get(
      std::size_t layer, uint64_t key);

  // кладёт блок, выкидывает lru пока не влезет в бюджет. вернёт указатель на
  // то что реально лежит (может быть тот же `data`)
  // inserts a block, evicts lru until it fits `budget`. returns the stored
  // pointer (may be the same as `data`)
  [[nodiscard]] CacheBlock put(
      std::size_t layer, uint64_t key,
      CacheBlock data, uint64_t budget);

  // полностью очистить кэш (перед batched prefetch)
  void clear();

  [[nodiscard]] uint64_t bytes(std::size_t layer) const;
  [[nodiscard]] std::size_t size(std::size_t layer) const;

 private:
  struct LayerCache {
    mutable std::mutex mu;
    std::unordered_map<uint64_t, CacheEntry> map;
    std::list<uint64_t> order;  // спереди самый свежий / front is mru
    uint64_t bytes = 0;
  };
  std::vector<LayerCache> layers_;
};

struct FreqState {
  explicit FreqState(std::size_t layers)
      : mu(layers), map(layers), last(layers) {}

  // мьютексы — деталь реализации: блокировка не меняет логическое состояние,
  // поэтому mu mutable и профили можно сериализовать из const
  // the mutexes are an implementation detail: locking does not change the
  // logical state, so mu is mutable and profiles can be serialized from const
  mutable std::vector<std::mutex> mu;
  std::vector<std::unordered_map<std::size_t, uint64_t>> map;
  // кто был выбран на прошлом токене. по этому предсказываем следующий
  // who was chosen on the last token. that is our prediction for the next one
  std::vector<std::vector<std::size_t>> last;
};

// частотный профиль роутинга на диске: компактный бинарник TKFQ v1 (заголовок
// + по слою (эксперт, счёт)). запись атомарная (tmp + rename), чтение
// валидирует магию, версию, число слоёв и границы экспертов — битый файл
// игнорируется, а не валит запуск
// routing frequency profile on disk: a compact TKFQ v1 binary (header + per
// layer (expert, count) pairs). writes are atomic (tmp + rename), reads
// validate magic, version, layer count and expert bounds — a corrupt file is
// ignored, it never breaks the run
[[nodiscard]] bool save_freq_profile(const FreqState& freq,
                                     std::size_t n_layers,
                                     const std::string& path);
[[nodiscard]] bool load_freq_profile(FreqState& freq, std::size_t n_layers,
                                     std::size_t n_experts,
                                     const std::string& path);

class Model {
 public:
  Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  // особые члены вынесены в model_batch.cpp: BatchBuf тут неполный тип, и
  // unique_ptr не может сгенерировать деструктор/перемещение в этом заголовке
  // special members live in model_batch.cpp: BatchBuf is opaque here, so
  // unique_ptr cannot generate the dtor/moves in this header
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  ~Model();

  [[nodiscard]] static Result<Model> load(const std::string& path,
                                          const ModelOptions& options = {});

  [[nodiscard]] std::vector<float> token_logits(uint32_t token, KVCache& kv,
                                                 std::size_t pos);
  // вариант без аллокаций: `out` должен быть размера vocab(), внутри
  // переиспользуется workspace модели
  // allocation-free variant, `out` must be vocab() long, reuses the workspace
  void token_logits(uint32_t token, KVCache& kv, std::size_t pos,
                    std::vector<float>& out);

  void collect_hot();

  void set_prefilling(bool v) { prefilling_ = v; }

  // частотный профиль: stable-ключ модели (геометрия+кванты, не путь) и
  // сохранение/загрузка. профиль грузится до первого токена, чтобы
  // pin_hot_experts сразу ставил горячих в кэш
  // frequency profile: stable model key (geometry+quants, not the path) and
  // save/load. the profile loads before the first token so pin_hot_experts
  // pins the hot experts right away
  [[nodiscard]] const std::string& freq_key() const noexcept {
    return freq_key_;
  }
  [[nodiscard]] bool save_freq_profile(const std::string& path) const;
  [[nodiscard]] bool load_freq_profile(const std::string& path);

  // bulk prefill: прочитать весь слой sequential, посчитать, выбросить.
  // overload берёт union экспертов батча, а не top одного токена
  // bulk prefill: read the whole layer sequentially. the overload takes the
  // batch's expert union instead of a single token's top-k
  void prefill_bulk_layer(std::size_t layer);
  void prefill_bulk_layer(std::size_t layer,
                          std::span<const std::size_t> experts);
  void pin_hot_experts(std::size_t k);

  // draft-режим: роутер отбирает меньше экспертов, чем n_used. черновик
  // слабее, но читает в n_used/k раз меньше байтов с диска. 0 = конфиг модели
  // draft mode: the router picks fewer experts than n_used. the draft is
  // weaker but reads n_used/k times less from the drive. 0 = model config
  void set_router_topk(std::size_t k);
  [[nodiscard]] std::size_t router_topk() const noexcept;
  [[nodiscard]] std::size_t n_used() const noexcept { return n_used_; }

  [[nodiscard]] const Gguf& gguf() const noexcept { return gguf_; }
  [[nodiscard]] std::size_t n_layers() const noexcept { return n_layers_; }
  [[nodiscard]] std::size_t hidden() const noexcept { return hidden_; }
  [[nodiscard]] std::size_t n_heads() const noexcept { return n_heads_; }
  [[nodiscard]] std::size_t n_kv() const noexcept { return n_kv_; }
  [[nodiscard]] std::size_t head_dim() const noexcept { return head_dim_; }
  // kv-размер на токен: n_kv × head_dim (у флинта 4 × 128 = 512)
  // kv size per token: n_kv × head_dim (the flint: 4 × 128 = 512)
  [[nodiscard]] std::size_t kv_dim() const noexcept { return n_kv_ * head_dim_; }
  [[nodiscard]] std::size_t vocab() const noexcept { return vocab_; }

  // батч-вход: b токенов за один проход по слоям, позиции pos..pos+b-1
  // (кауза внутри батча, историю платят так же). outs должен быть
  // b × vocab, логиты строки j — outs[j * vocab]
  // batched entry: b tokens per layer pass, positions pos..pos+b-1 causal
  // within the batch, history costs the same. outs must be b × vocab,
  // logits of row j go to outs[j * vocab]
  void token_logits_batch(std::span<const uint32_t> tokens, Kv& kv,
                          std::size_t pos, std::span<float> outs);
  [[nodiscard]] const std::vector<Layer>& layers() const noexcept {
    return layers_;
  }

  struct Seg {
    std::span<const uint8_t> bytes;
    std::size_t rb = 0;
    QuantType quant = QuantType::F32;
    std::size_t in_dim = 0;
    const float* x = nullptr;
    const kernels::ActQ8* act = nullptr;
    float* out = nullptr;
    std::size_t rows = 0;
  };

  struct WRef {
    const uint8_t* data = nullptr;
    std::size_t len = 0;
    CacheBlock owner;
  };

  // буферы горячего пути. всё (пере)выделяется один раз при загрузке, потом
  // токен идёт вообще без обращений к аллокатору (ну почти)
  // reused hot-path storage. sized once at load, a token then runs without
  // touching the allocator (well, almost)
  struct Workspace {
    std::vector<float> x, hn, q, k, v, o, attn, hf;
    std::vector<float> g, u, dd, moe, gu;
    std::vector<float> scores;
    std::vector<float> logits;
    std::vector<float> wts;
    std::vector<std::size_t> idx, top;
    std::vector<WRef> slices, wg, wu, wd;
    std::vector<Seg> segs, down_segs;
    std::vector<kernels::ActQ8> expert_acts;
    kernels::ActQ8 act;
    std::vector<std::pair<float, float>> rope_table;
  };

  void prepare_workspace();

  void matvec(const Meta& meta, std::span<const float> x,
              std::span<float> y) const;
  void matvec(const Meta& meta, std::span<const float> x,
              std::span<float> y, kernels::ActQ8& act) const;
  void matvec_preact(const Meta& meta, std::span<const float> x,
                     std::span<float> y,
                     const kernels::ActQ8& act) const;

 private:
  static void row_task(void* ctx, std::size_t i);
  static void load_task(void* ctx, std::size_t i);

  void layer(std::size_t layer, Workspace& ws, KVCache& kv,
             std::span<const std::pair<float, float>> rope_table);
  // батч-слой: b позиций за проход, kv пишет сам через abi Kv. агрегирован
  // с model_batch.cpp, буферы — через opaque BatchBuf
  // batched layer: b positions per pass, kv written through the Kv abi.
  // implemented in model_batch.cpp, buffers live in the opaque BatchBuf
  struct BatchBuf;
  void ensure_batch_buf(std::size_t b);
  void layer_batch(std::size_t layer, Kv& kv, std::size_t pos, std::size_t b,
                   std::span<const std::pair<float, float>> rope_table);
  void run_segs(std::span<const Seg> segs, std::size_t rows_per_task) const;
  [[nodiscard]] std::span<const uint8_t> meta_bytes(const Meta& meta) const;
  [[nodiscard]] WRef resident_expert(std::size_t layer, uint8_t proj,
                                     const ExpertMeta& em,
                                     std::size_t expert) const;
  // ждём уже заказанного эксперта. если чтение сорвалось — обычный путь
  // wait for an already ordered expert. on a failed read, the normal path
  [[nodiscard]] WRef take_expert(std::size_t layer, uint8_t proj,
                                 const ExpertMeta& em,
                                 std::size_t expert) const;
  void record_freq(std::size_t layer, std::span<const std::size_t> top);
  void prefetch_experts(std::size_t layer, std::span<const std::size_t> top);

  static void rmsnorm(std::span<float> x, std::span<const float> w, float eps);
  static void rope_apply(std::span<float> x,
                         std::span<const std::pair<float, float>> table);
  static void softmax(std::span<float> x);
  static void swiglu(std::span<float> g, std::span<const float> u);

  Gguf gguf_;
  Mmap mm_;

  std::size_t n_layers_ = 0;
  std::size_t hidden_ = 0;
  std::size_t n_heads_ = 0;
  std::size_t n_kv_ = 0;
  std::size_t head_dim_ = 0;
  std::size_t n_experts_ = 0;
  std::size_t n_used_ = 0;
  std::size_t router_topk_ = 0;
  std::size_t dense_ffn_ = 0;
  std::size_t ffn_dim_ = 0;
  std::size_t shared_dim_ = 0;
  std::size_t n_dense_ = 0;
  float rope_theta_ = 0.0F;
  float eps_ = 0.0F;
  std::size_t window_ = 0;
  std::vector<bool> swa_;
  std::vector<bool> rope_on_;
  std::size_t vocab_ = 0;

  std::vector<Layer> layers_;
  Meta embd_;
  Meta out_;
  std::vector<float> out_norm_;
  std::vector<float> inv_freq_;
  std::string freq_key_;

  std::unique_ptr<ResidentCache> resident_;
  std::unique_ptr<ExpertIo> io_;
  uint64_t resident_budget_ = 0;
  uint64_t total_budget_ = 0;
  std::unique_ptr<FreqState> freq_;
  std::unique_ptr<Workspace> ws_;
  std::unique_ptr<BatchBuf> batch_buf_;
  // Плотные веса: AHB-буферы (pinned dma-buf), а не std::vector. Иначе Android
  // кладёт их в zram, и каждый matvec платит за распаковку.
  ahb::HotWeights hot_;
  bool prefilling_ = false;
};

}  // namespace tokshit
