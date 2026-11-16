#include "tokshit/model.hpp"
#include "tokshit/ahb.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "tokshit/gguf.hpp"
#include "tokshit/dequant.hpp"
#include "tokshit/kernels.hpp"
#include "tokshit/metrics.hpp"
#include "tokshit/pool.hpp"

namespace tokshit {
namespace {

struct RowJob {
  const Model::Seg* segs = nullptr;
  std::size_t nsegs = 0;
  const std::size_t* task_starts = nullptr;
  std::size_t task_starts_len = 0;
  std::size_t rows_per_task = 0;
};

struct LoadJob {
  const Model* model = nullptr;
  std::size_t layer = 0;
  const std::size_t* tops = nullptr;
  const ExpertMeta* ems[3] = {nullptr, nullptr, nullptr};
  std::size_t n_used = 0;
  Model::WRef* out = nullptr;
};

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

// читаем env как флаг. любое значение кроме пустого и "0" это true
// read an env var as a flag. anything except empty and "0" is true
bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' && v[0] != '0';
}

// читаем env как число, если пусто или мусор берём fallback
// read an env var as a number, fallback on empty/garbage
uint64_t env_u64(const char* name, uint64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  return std::strtoull(v, nullptr, 10);
}

// ── сериализация частотного профиля (TKFQ v1) ──────────────────────────────
// little-endian: u32 магия "TKFQ", u32 версия=1, u32 слоёв, дальше на каждый
// слой u32 count и count пар (u32 эксперт, u64 счёт)
// frequency profile serialization (TKFQ v1): little-endian, u32 magic "TKFQ",
// u32 version=1, u32 layers, then per layer u32 count and count pairs
// (u32 expert, u64 count)
void put_le32(std::vector<uint8_t>& buf, uint32_t v) {
  buf.push_back(static_cast<uint8_t>(v));
  buf.push_back(static_cast<uint8_t>(v >> 8));
  buf.push_back(static_cast<uint8_t>(v >> 16));
  buf.push_back(static_cast<uint8_t>(v >> 24));
}

void put_le64(std::vector<uint8_t>& buf, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    buf.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
}

uint32_t get_le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t get_le64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

constexpr uint32_t kFreqMagic = 0x51464b54U;  // "TKFQ" little-endian
constexpr uint32_t kFreqVersion = 1;

}  // namespace

bool save_freq_profile(const FreqState& freq, std::size_t n_layers,
                       const std::string& path) {
  std::vector<uint8_t> buf;
  buf.reserve(n_layers * (4 + 256 * 12) + 16);
  put_le32(buf, kFreqMagic);
  put_le32(buf, kFreqVersion);
  put_le32(buf, static_cast<uint32_t>(n_layers));
  for (std::size_t layer = 0; layer < n_layers; ++layer) {
    std::vector<std::pair<uint32_t, uint64_t>> items;
    {
      std::lock_guard<std::mutex> lock(freq.mu[layer]);
      items.reserve(freq.map[layer].size());
      for (const auto& [e, c] : freq.map[layer]) {
        items.emplace_back(static_cast<uint32_t>(e), c);
      }
    }
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    put_le32(buf, static_cast<uint32_t>(items.size()));
    for (const auto& [e, c] : items) {
      put_le32(buf, e);
      put_le64(buf, c);
    }
  }
  // атомарная запись: temp-файл с pid в имени + rename. профили параллельных
  // агентов не рвут друг друга, а сбой записи не оставляет битый файл
  // atomic write: temp file with pid in the name + rename. concurrent agents
  // don't tear each other's profiles and a failed write leaves no trash
  const std::string tmp = path + ".tmp" +
                          std::to_string(static_cast<long>(::getpid()));
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "[freq] не удалось открыть на запись: %s\n",
                 tmp.c_str());
    return false;
  }
  const std::size_t wrote = std::fwrite(buf.data(), 1, buf.size(), f);
  const bool flush_ok = std::fflush(f) == 0;
  const int fd = ::fileno(f);
  const bool fsync_ok = fd >= 0 ? ::fsync(fd) == 0 : false;
  std::fclose(f);
  if (wrote != buf.size() || !flush_ok || !fsync_ok ||
      std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::fprintf(stderr, "[freq] запись не удалась: %s\n", path.c_str());
    (void)::unlink(tmp.c_str());
    return false;
  }
  return true;
}

bool load_freq_profile(FreqState& freq, std::size_t n_layers,
                       std::size_t n_experts, const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  std::vector<uint8_t> buf;
  uint8_t chunk[4096];
  std::size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0) {
    buf.insert(buf.end(), chunk, chunk + got);
    if (buf.size() > n_layers * (4 + 1024 * 12) + 16) break;  // мусор, хватит
  }
  std::fclose(f);
  if (buf.size() < 12) return false;
  if (get_le32(buf.data()) != kFreqMagic) return false;
  if (get_le32(buf.data() + 4) != kFreqVersion) {
    std::fprintf(stderr, "[freq] неизвестная версия профиля: %s\n",
                 path.c_str());
    return false;
  }
  if (get_le32(buf.data() + 8) != n_layers) return false;
  const uint8_t* p = buf.data() + 12;
  const uint8_t* const end = buf.data() + buf.size();
  std::size_t loaded_layers = 0;
  for (std::size_t layer = 0; layer < n_layers; ++layer) {
    if (end - p < 4) return false;
    const uint32_t count = get_le32(p);
    p += 4;
    // count обязан поместиться в файл и не превосходить число экспертов
    // count must fit in the remaining bytes and not exceed the expert count
    if (count > n_experts || end - p < static_cast<std::ptrdiff_t>(count) * 12) {
      return false;
    }
    std::unordered_map<std::size_t, uint64_t> map;
    map.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t e = get_le32(p);
      const uint64_t c = get_le64(p + 4);
      p += 12;
      if (e >= n_experts) return false;
      map.emplace(e, c);
    }
    {
      std::lock_guard<std::mutex> lock(freq.mu[layer]);
      freq.map[layer] = std::move(map);
    }
    ++loaded_layers;
  }
  if (p != end) {
    // хвост разрешён только пустой: расхождение значит битый файл
    // only an empty tail is allowed: a mismatch means a corrupt file
    if (static_cast<std::size_t>(end - p) > 0) return false;
  }
  return loaded_layers == n_layers;
}

ResidentCache::ResidentCache(std::size_t layers) : layers_(layers) {}

// достаём блок и двигаем его в начало списка (mru). список и map живут под
// одним мьютексом слоя
// grab the block and move it to the front (mru). list and map live under the
// same per-layer mutex
CacheBlock ResidentCache::get(std::size_t layer, uint64_t key) {
  LayerCache& lc = layers_[layer];
  std::lock_guard<std::mutex> lock(lc.mu);
  const auto it = lc.map.find(key);
  if (it == lc.map.end()) return nullptr;
  lc.order.erase(it->second.lru);
  lc.order.push_front(key);
  it->second.lru = lc.order.begin();
  return it->second.data;
}

// кладём блок. если бюджет 0 или блок больше бюджета то не кэшируем вообще,
// просто отдаём как есть (direct режим)
// store a block. budget 0 or a block bigger than budget means no caching at
// all, just hand it back (direct mode)
CacheBlock ResidentCache::put(std::size_t layer, uint64_t key,
                            CacheBlock data, uint64_t budget) {
  LayerCache& lc = layers_[layer];
  std::lock_guard<std::mutex> lock(lc.mu);
  const auto existing = lc.map.find(key);
  if (existing != lc.map.end()) {
    lc.order.erase(existing->second.lru);
    lc.order.push_front(key);
    existing->second.lru = lc.order.begin();
    return existing->second.data;
  }
  const uint64_t size = data->size();
  if (budget == 0 || size > budget) return data;
  while (lc.bytes + size > budget && !lc.order.empty()) {
    const uint64_t victim = lc.order.back();
    lc.order.pop_back();
    const auto it = lc.map.find(victim);
    if (it == lc.map.end()) continue;
    lc.bytes -= it->second.size;
    lc.map.erase(it);
  }
  if (lc.bytes + size > budget) return data;
  lc.order.push_front(key);
  lc.map.emplace(key, CacheEntry{data, 0, static_cast<uint32_t>(size),
                                 lc.order.begin()});
  lc.bytes += size;
  return data;
}

void ResidentCache::clear() {
  for (auto& lc : layers_) {
    std::lock_guard<std::mutex> lock(lc.mu);
    lc.map.clear();
    lc.order.clear();
    lc.bytes = 0;
  }
}

uint64_t ResidentCache::bytes(std::size_t layer) const {
  const LayerCache& lc = layers_[layer];
  std::lock_guard<std::mutex> lock(lc.mu);
  return lc.bytes;
}

std::size_t ResidentCache::size(std::size_t layer) const {
  const LayerCache& lc = layers_[layer];
  std::lock_guard<std::mutex> lock(lc.mu);
  return lc.map.size();
}

// одна задача = кусок строк одного сегмента. ищем к какому сегменту относится
// task, потом считаем rows_per_task строк. scratch thread_local чтобы не
// аллоцировать в каждом вызове
// one task = a chunk of rows of one segment. find which segment `task` belongs
// to, then do rows_per_task rows. scratch is thread_local so we dont allocate
// on every call
void Model::row_task(void* ctx, std::size_t task) {
  const RowJob& job = *static_cast<const RowJob*>(ctx);
  std::size_t s = 0;
  while (s + 1 < job.task_starts_len && task >= job.task_starts[s + 1]) ++s;
  const std::size_t local = task - job.task_starts[s];
  const Seg& seg = job.segs[s];
  const std::size_t r0 = local * job.rows_per_task;
  const std::size_t r1 = std::min(r0 + job.rows_per_task, seg.rows);
  thread_local std::vector<float> scratch;
  const std::size_t nrows = r1 - r0;
  if (nrows == 0) return;
  const auto block = seg.bytes.subspan(r0 * seg.rb, nrows * seg.rb);
  kernels::dot_rows(
      seg.quant, block, seg.rb, nrows,
      std::span<const float>(seg.x, seg.in_dim), *seg.act, seg.in_dim,
      seg.out + r0, scratch);
}

// грузим один эксперт одной проекции через кэш. задачи намеренно плоские:
// индекс i кодирует (какой эксперт, какая проекция)
// load one expert of one projection through the cache. tasks are flat on
// purpose: index i encodes (which expert, which projection)
void Model::load_task(void* ctx, std::size_t i) {
  LoadJob& job = *static_cast<LoadJob*>(ctx);
  const std::size_t e = i % job.n_used;
  const auto proj = static_cast<uint8_t>(i / job.n_used);
  job.out[i] = job.model->resident_expert(job.layer, proj, *job.ems[proj],
                                          job.tops[e]);
}

// главный загрузчик. читает конфиг, строит Layer с Meta смещениями, валидирует
// паттерны и создаёт workspace. тяжёлых копий нет (кроме hot и norm векторов)
// the main loader. reads config, builds Layer structs with Meta offsets,
// validates the patterns and makes the workspace. no heavy copies except hot
// and the norm vectors
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

// копируем все неэкспертные веса в отдельный vector чтобы их не вытеснял
// поток экспертов из page cache. на андроиде может не спасти (своп)
// copy all non-expert weights into a separate vector so the expert stream
// doesnt evict them from page cache. on android it may not save you (swap)
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
  // Dense weights go to AHB instead of an anonymous vector: under pressure
  // Android swaps anonymous pages into zram, making every token pay for
  // decompression. dma-buf is not swappable.
  //
  // Заодно переквантовываем в Q8_0 всё, для чего нет neon-ядра (Q5K/Q6K/
  // Bf16). Скалярный dequant+dot на этих весах даёт 1-1.6 ГБ/с, а q8_0 -
  // 30+. Это +1.5 бита на вес и ~0.6 ГБ памяти на флинте, но плотные веса
  // читаются на каждом токене, так что обмен того стоит.
  // While we are here, requantize everything without a neon kernel (Q5K/Q6K/
  // Bf16) to Q8_0. The scalar dequant+dot path runs those at 1-1.6 GB/s while
  // q8_0 does 30+. Costs ~1.5 bits per weight (~0.6 GB on the flint), but the
  // dense weights are read on every token, so the trade is worth it.
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
        // Слот должен вмещать обе версии: если конвертация сорвётся на
        // середине, мы положим туда исходные байты. Для Bf16 исходная
        // строка ШИРЕ q8_0, и старый код оставлял hot-слот неинициализо-
        // ванным, а метаданные — с исходным квантом. max() убирает эту
        // дыру ценой неиспользованного хвоста только у Bf16.
        // The slot must hold both versions: if conversion fails midway we
        // store the original bytes. For Bf16 the source row is WIDER than
        // q8_0, and the old code left the hot slot uninitialized while
        // keeping the original quant in the metadata. max() closes that hole;
        // the only cost is an unused tail for Bf16 tensors.
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
      // requant failed: store the original bytes and keep the slow path.
      // the slot is sized max(src, q8), so the copy always fits
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

// отдаём байты тензора, откуда именно решает флаг hot
// hand out the tensor bytes, the hot flag decides from where
std::span<const uint8_t> Model::meta_bytes(const Meta& meta) const {
  if (meta.hot) {
    const uint8_t* p = hot_.data_at(meta.base);
    if (p != nullptr) {
      return std::span<const uint8_t>(p, hot_.size() - meta.base);
    }
  }
  const auto bytes = mm_.bytes();
  return std::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(bytes.data()) + meta.base,
      bytes.size() - meta.base);
}

// удобная версия: квантуем активацию во временный буфер и считаем. нужна
// только холодным вызовам, горячий путь юзает версию с переданным act
// convenience version, quantizes into a temp buffer and runs. only for cold
// calls, the hot path uses the overload with an external act
void Model::matvec(const Meta& meta, std::span<const float> x,
                   std::span<float> y) const {
  kernels::ActQ8 act;
  matvec(meta, x, y, act);
}

// квантуем x в act и считаем. активация остаётся снаружи чтобы можно было
// переиспользовать её для нескольких матриц с тем же входом
// quantize x into act and run. activation stays outside so several matrices
// with the same input can share it
void Model::matvec(const Meta& meta, std::span<const float> x,
                   std::span<float> y, kernels::ActQ8& act) const {
  kernels::quantize_into(x, act);
  matvec_preact(meta, x, y, act);
}

// версия с уже готовой активацией. именно её зовёт горячий путь, тут нет ни
// одной аллокации
// version with a ready activation. this is what the hot path calls, zero
// allocations in here
void Model::matvec_preact(const Meta& meta, std::span<const float> x,
                          std::span<float> y,
                          const kernels::ActQ8& act) const {
  const auto bytes = meta_bytes(meta);
  const auto t0 = Clock::now();
  const Seg seg{bytes, meta.rb, meta.quant, meta.in_dim, x.data(), &act,
                y.data(), meta.out_dim};
  run_segs(std::span<const Seg>(&seg, 1), 16);
  if (timing_enabled()) {
    metrics().t_mm.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                 t0)
                .count()),
        std::memory_order_relaxed);
    metrics().c_mm_bytes.fetch_add(
        static_cast<uint64_t>(meta.out_dim * meta.rb),
        std::memory_order_relaxed);
  }
}

// раскладываем все сегменты в задачи по строкам и кидаем в пул. вызывающий
// поток тоже работает, так что не простаивает
// flatten segments into row tasks and throw them at the pool. the calling
// thread also works so it doesnt just sit there
void Model::run_segs(std::span<const Seg> segs,
                     std::size_t rows_per_task) const {
  thread_local std::vector<std::size_t> task_starts;
  task_starts.clear();
  task_starts.reserve(segs.size() + 1);
  task_starts.push_back(0);
  std::size_t total = 0;
  for (const Seg& seg : segs) {
    total += (seg.rows + rows_per_task - 1) / rows_per_task;
    task_starts.push_back(total);
  }
  if (total == 0) return;

  const RowJob job{segs.data(), segs.size(), task_starts.data(),
                   task_starts.size(), rows_per_task};
  static const bool dbg = std::getenv("TOKSHIT_SEG_DEBUG") != nullptr;
  const auto pf0 = dbg ? page_faults() : std::pair<uint64_t, uint64_t>{0, 0};
  const auto t0 = Clock::now();
  pool::run_tasks(total, &Model::row_task,
                  const_cast<RowJob*>(&job));

  if (dbg) {
    static std::atomic<uint64_t> n{0};
    static std::atomic<uint64_t> total_bytes{0};
    static std::atomic<uint64_t> total_ns{0};
    const uint64_t k = n.fetch_add(1, std::memory_order_relaxed);
    std::size_t rows = 0;
    std::size_t bytes = 0;
    for (const Seg& seg : segs) {
      rows += seg.rows;
      bytes += seg.rows * seg.rb;
    }
    const double dt =
        std::chrono::duration<double>(Clock::now() - t0).count();
    const uint64_t dt_ns = static_cast<uint64_t>(dt * 1e9);
    const uint64_t acc_b = total_bytes.fetch_add(bytes) + bytes;
    const uint64_t acc_ns = total_ns.fetch_add(dt_ns) + dt_ns;
    const QuantType q0 = segs.empty() ? QuantType::F32 : segs[0].quant;
    const std::size_t rb0 = segs.empty() ? 0 : segs[0].rb;
    const std::size_t in0 = segs.empty() ? 0 : segs[0].in_dim;
    static const bool dbg_all =
        std::getenv("TOKSHIT_SEG_DEBUG_ALL") != nullptr;
    if (k < 12 || k % 500 == 0 ||
        (dbg_all && k < 400 && q0 != QuantType::Q8_0)) {
      const auto pf1 = page_faults();
      std::fprintf(stderr,
                   "[seg %llu] tasks=%zu rows=%zu segs=%zu q=%.*s rb=%zu "
                   "in=%zu %.1f MB %.0f us %.2f GB/s avg=%.2f GB/s active=%d "
                   "minflt=%llu majflt=%llu\n",
                   static_cast<unsigned long long>(k), total, rows, segs.size(),
                   static_cast<int>(quant_name(q0).size()),
                   quant_name(q0).data(), rb0, in0,
                   static_cast<double>(bytes) / 1e6, dt * 1e6,
                   static_cast<double>(bytes) / dt / 1e9,
                   acc_ns == 0 ? 0.0
                               : static_cast<double>(acc_b) / acc_ns,
                   pool::is_active() ? 1 : 0,
                   static_cast<unsigned long long>(pf1.first - pf0.first),
                   static_cast<unsigned long long>(pf1.second - pf0.second));
    }
  }
}

// достаём эксперта. бюджет 0 = отдаём span прямо в mmap. иначе через
// фоновых читателей: они и кэш проверят, и подхватят уже начатое чтение.
// ключ это (proj << 32) | expert
// get an expert. budget 0 = return a span straight into the mmap. otherwise
// through the background readers: they check the cache and pick up a read
// that is already in flight. key is (proj << 32) | expert
Model::WRef Model::resident_expert(std::size_t layer, uint8_t proj,
                                   const ExpertMeta& em,
                                   std::size_t expert) const {
  const uint64_t key = (static_cast<uint64_t>(proj) << 32) |
                       static_cast<uint32_t>(expert);
  const auto mm_bytes = mm_.bytes();
  const auto* mm_data = reinterpret_cast<const uint8_t*>(mm_bytes.data());
  const std::size_t base = static_cast<std::size_t>(em.base) + em.size * expert;

  if (resident_budget_ == 0 || !io_) {
    metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
    return WRef{mm_data + base, em.size, {}};
  }

  auto got = io_->load(layer, key, static_cast<uint64_t>(base), em.size);
  if (got.hit) {
    metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
    return WRef{got.block->data(), got.block->size(), std::move(got.block)};
  }
  metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);

  if (got.block == nullptr) {
    // чтение не удалось. падаем обратно на mmap, лишь бы не сломать токен
    // the read failed. fall back to the mmap so the token still works
    auto owned = std::make_shared<CacheBlockData>(em.size);
    if (!owned->valid()) {
      return WRef{mm_data + base, em.size, {}};
    }
    std::memcpy(owned->data(), mm_data + base, em.size);
    CacheBlock data = owned;
    data = resident_->put(layer, key, std::move(data), resident_budget_);
    return WRef{data->data(), data->size(), std::move(data)};
  }
  return WRef{got.block->data(), got.block->size(), std::move(got.block)};
}

// берём эксперта, который уже заказан через demand_async: ждём его и отдаём
// из кэша. если чтение не удалось, идём обычным путём
// take an expert already ordered through demand_async: wait for it and hand it
// out of the cache. on a failed read, fall back to the normal path
Model::WRef Model::take_expert(std::size_t layer, uint8_t proj,
                               const ExpertMeta& em,
                               std::size_t expert) const {
  const uint64_t key = (static_cast<uint64_t>(proj) << 32) |
                       static_cast<uint32_t>(expert);
  if (io_ != nullptr) {
    if (auto block = io_->wait_for(layer, key); block != nullptr) {
      return WRef{block->data(), block->size(), std::move(block)};
    }
  }
  metrics().c_inline.fetch_add(1, std::memory_order_relaxed);
  return resident_expert(layer, proj, em, expert);
}

// записываем частоту роутинга, потом по ней выбираем что пинить/префетчить.
// мьютекс на слой чтобы не толкаться на каждом токене
// record routing frequency, used later to decide what to pin/prefetch.
// per-layer mutex so we dont fight on every token
void Model::record_freq(std::size_t layer,
                        std::span<const std::size_t> top) {
  std::lock_guard<std::mutex> lock(freq_->mu[layer]);
  auto& map = freq_->map[layer];
  for (std::size_t e : top) ++map[e];
  freq_->last[layer].assign(top.begin(), top.end());
}

// предсказание: кидаем в фоновую очередь ровно тех экспертов, которых выбрал
// прошлый токен на этом же слое. роутинг соседних токенов почти не меняется,
// а диск греется пока cpu считает attention. это замена старому madvise,
// который только мусорил page cache
// prediction: queue exactly the experts the previous token picked on the same
// layer. routing barely changes between neighbours, and the drive warms up
// while the cpu does attention. replaces the old madvise that only polluted
// the page cache
void Model::prefetch_experts(std::size_t layer,
                             std::span<const std::size_t> top) {
  if (!io_ || resident_budget_ == 0 || top.empty()) return;
  if (layer >= n_layers_ || layer < n_dense_) return;
  const Layer& lt = layers_[layer];
  if (!lt.gate_exps || !lt.up_exps || !lt.down_exps) return;
  // TOKSHIT_PREFETCH_IDLE: спекуляция ограничена — только gate/up и не больше
  // kTopP=4 экспертов. на idle-диске это всё, что спрос успеет съесть до
  // своего слоя; down приезжает обычным demand-путём
  // TOKSHIT_PREFETCH_IDLE: speculation is capped — gate/up only, at most
  // kTopP=4 experts. on an idle drive that is all demand can consume before
  // its layer; down arrives through the plain demand path
  static const bool idle = prefetch_idle();
  static constexpr std::size_t kTopP = 4;
  const std::size_t n = idle && top.size() > kTopP ? kTopP : top.size();
  const std::size_t n_proj = idle ? 2 : 3;
  const ExpertMeta* ems[3] = {&*lt.gate_exps, &*lt.up_exps, &*lt.down_exps};
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t e = top[i];
    for (std::size_t pi = 0; pi < n_proj; ++pi) {
      const uint64_t key = (static_cast<uint64_t>(pi) << 32) |
                           static_cast<uint32_t>(e);
      const uint64_t off = ems[pi]->base + ems[pi]->size * e;
      io_->prefetch(layer, key, off, ems[pi]->size);
    }
  }
}

// обёртки частотного профиля: see free functions above
// frequency profile wrappers: see the free functions above
bool Model::save_freq_profile(const std::string& path) const {
  return tokshit::save_freq_profile(*freq_, n_layers_, path);
}

bool Model::load_freq_profile(const std::string& path) {
  return tokshit::load_freq_profile(*freq_, n_layers_, n_experts_, path);
}

// draft-режим: роутер отбирает меньше экспертов, чем n_used. 0 = конфиг
// модели. веса при k_used=1 вырождаются в единицу — черновик читает в n_used
// раз меньше байтов с диска и считается пропорционально быстрее
// draft mode: the router picks fewer experts than n_used. 0 = the model
// config. at k_used=1 weights degenerate to one, so the draft reads n_used
// times fewer expert bytes and runs proportionally faster
void Model::set_router_topk(std::size_t k) {
  router_topk_ = k;
}

std::size_t Model::router_topk() const noexcept {
  return router_topk_ != 0 && router_topk_ < n_used_ ? router_topk_ : n_used_;
}

// прогрев после промпта: топ-k по истории частоты на каждом слое. теперь это
// реальные чтения в наш кэш, а не madvise в page cache
// warm-up after the prompt: top-k by frequency history per layer. these are
// real reads into our cache now, not madvise into page cache
void Model::prefill_bulk_layer(std::size_t layer) {
  const Layer& lt = layers_[layer];
  if (!lt.gate_exps || !lt.up_exps || !lt.down_exps) return;

  // sequential pread на весь регион экспертов слоя (gate+up+down = ~252MB)
  // gguf-конвенция: три тензора идут подряд, gap минимален
  struct Region { uint64_t off; std::size_t size; };
  const ExpertMeta* ems[3] = {&*lt.gate_exps, &*lt.up_exps, &*lt.down_exps};
  Region regions[3];
  for (int pi = 0; pi < 3; ++pi) {
    // n_experts_, не 256: захардкоженное число сделало бы регион неверным
    // для любой модели не из флинта, и memcpy ниже читал бы за границей буфера
    // n_experts_, not 256: the hardcoded count would compute a wrong region
    // for any non-flint model, and the memcpy below would read out of bounds
    regions[pi] = Region{ems[pi]->base,
                         static_cast<std::size_t>(ems[pi]->size * n_experts_)};
  }
  const uint64_t min_off = std::min({regions[0].off, regions[1].off,
                                     regions[2].off});
  const uint64_t max_end =
      std::max({regions[0].off + regions[0].size,
                regions[1].off + regions[1].size,
                regions[2].off + regions[2].size});
  const std::size_t total = static_cast<std::size_t>(max_end - min_off);
  if (total > 512ULL * 1024 * 1024) {
    std::fprintf(stderr, "[prefill] layer %zu region %zu MB too big, skip\n",
                 layer, total / (1024 * 1024));
    return;
  }

  std::vector<uint8_t> big(total);
  const int rfd = io_->fd();
  (void)::posix_fadvise(rfd, static_cast<off_t>(min_off),
                        static_cast<off_t>(total), POSIX_FADV_SEQUENTIAL);
  const auto rd = pread_at(rfd, min_off,
                            std::span<uint8_t>(big.data(), total));
  if (!rd.ok()) {
    std::fprintf(stderr, "[prefill] bulk read fail, fallback\n");
    return;
  }

  // memcpy в AHB и resident_->put. ws.top уже посчитан в layer() ДО этого
  // вызова, значит можем использовать его
  const std::size_t n_used = ws_->top.size();
  for (std::size_t j = 0; j < n_used; ++j) {
    const std::size_t e = ws_->top[j];
    for (int pi = 0; pi < 3; ++pi) {
      const uint8_t* src = big.data() + (regions[pi].off - min_off) +
                           ems[pi]->size * e;
      auto cb = std::make_shared<CacheBlockData>(ems[pi]->size);
      if (!cb->valid()) continue;
      std::memcpy(cb->data(), src, ems[pi]->size);
      const uint64_t key =
          (static_cast<uint64_t>(pi) << 32) | static_cast<uint32_t>(e);
      (void)resident_->put(layer, key, std::move(cb), resident_budget_);
    }
  }
  (void)::posix_fadvise(rfd, static_cast<off_t>(min_off),
                        static_cast<off_t>(total), POSIX_FADV_DONTNEED);
}

void Model::pin_hot_experts(std::size_t k) {
  if (!io_ || resident_budget_ == 0 || k == 0) return;

  std::fprintf(stderr, "[pin] called k=%zu, budget_per_layer=%llu MB\n",
               k, static_cast<unsigned long long>(resident_budget_ / (1024*1024)));

  struct Entry {
    std::size_t layer;
    uint64_t key;
    uint64_t offset;
    std::size_t size;
  };

  // 1. собрать топ-k по freq_ для каждого слоя
  std::vector<Entry> all;
  all.reserve(n_layers_ * k * 3);
  // диагностика: freq реально пуст или места нет (boundary-чек auto-подгонки)
  // diagnostics: is freq really empty or is there just no room (auto-fit
  // boundary check)
  std::size_t layers_with_items = 0, total_items = 0, free_experts = 0;
  for (std::size_t layer = n_dense_; layer < n_layers_; ++layer) {
    const Layer& lt = layers_[layer];
    if (!lt.gate_exps || !lt.up_exps || !lt.down_exps) continue;
    std::vector<std::pair<std::size_t, uint64_t>> items;
    {
      std::lock_guard<std::mutex> lock(freq_->mu[layer]);
      for (const auto& [e, c] : freq_->map[layer]) items.emplace_back(e, c);
    }
    if (items.empty()) continue;
    ++layers_with_items;
    total_items += items.size();
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    // авто-подгонка: не пиним больше, чем влезает в per-layer бюджет,
    // иначе LRU выкинет только что запиненные
    // учитываем уже занятые слоты в слое: не пиним то, что вытеснит prefill-кэш
    const std::size_t slot_b = ahb::Pool::instance().slot_bytes()
                ? ahb::Pool::instance().slot_bytes()
                : static_cast<std::size_t>(576u * 1024u);
            const std::size_t cap_slots  = static_cast<std::size_t>(resident_budget_ / slot_b);
            const std::size_t used_slots = static_cast<std::size_t>(resident_->bytes(layer) / slot_b);
            const std::size_t free_slots = cap_slots > used_slots ? cap_slots - used_slots : 0;
            const std::size_t take = std::min({k, items.size(), free_slots / 3});
    free_experts += free_slots / 3;
    const ExpertMeta* ems[3] = {&*lt.gate_exps, &*lt.up_exps, &*lt.down_exps};
    for (std::size_t i = 0; i < take; ++i) {
      const std::size_t e = items[i].first;
      for (std::size_t pi = 0; pi < 3; ++pi) {
        Entry en;
        en.layer = layer;
        en.key = (static_cast<uint64_t>(pi) << 32) | static_cast<uint32_t>(e);
        en.offset = ems[pi]->base + ems[pi]->size * e;
        en.size = ems[pi]->size;
        all.push_back(en);
      }
    }
  }
  if (all.empty()) {
    std::fprintf(stderr,
                 "[pin] ничего не закреплено: слоёв с freq=%zu/%zu, записей "
                 "freq=%zu, свободных мест (экспертов)=%zu — кэш полон или "
                 "prefill слишком короткий\n",
                 layers_with_items, n_layers_ > n_dense_ ? n_layers_ - n_dense_ : 0,
                 total_items, free_experts);
    return;
  }
  std::fprintf(stderr, "[pin] candidates from freq: %zu entries\n", all.size());

  // 2. сортировать по физическому offset -> sequential pread
  all.erase(std::remove_if(all.begin(), all.end(), [this](const Entry& e) {
        return resident_->get(e.layer, e.key) != nullptr;
    }), all.end());
    if (all.empty()) {
        std::fprintf(stderr, "[pin] all candidates already cached, skip\n");
        return;
    }
    std::fprintf(stderr, "[pin] uncached candidates: %zu\n", all.size());
    std::sort(all.begin(), all.end(),
        [](const Entry& a, const Entry& b) { return a.offset < b.offset; });

  // 3. склеивать соседние эксперты: до 100 MB gap (все три проекции слоя подряд)
  //    и до 500 MB за один pread
  constexpr uint64_t kGapTol   = 1ULL * 1024 * 1024;
  constexpr uint64_t kMaxBatch = 64ULL * 1024 * 1024;

  std::size_t total = 0, pinned = 0, batches = 0;
  std::size_t i = 0;
  std::vector<uint8_t> big;
  while (i < all.size() && total < total_budget_) {
    const std::size_t gs = i;
    uint64_t range_off = all[i].offset;
    uint64_t range_end = all[i].offset + all[i].size;
    while (i + 1 < all.size()) {
      const uint64_t next_off = all[i + 1].offset;
      const uint64_t gap = (next_off > range_end) ? (next_off - range_end) : 0;
      if (gap > kGapTol) break;
      if ((next_off + all[i + 1].size - range_off) > kMaxBatch) break;
      range_end = std::max(range_end, next_off + all[i + 1].size);
      ++i;
    }
    const std::size_t ge = i;
    const std::size_t range_size = static_cast<std::size_t>(range_end - range_off);
    big.resize(range_size);
    const auto rd = pread_at(io_->fd(), range_off,
                              std::span<uint8_t>(big.data(), range_size));
    if (rd.ok()) {
      for (std::size_t j = gs; j <= ge && total < total_budget_; ++j) {
        const auto& en = all[j];
        if (resident_->get(en.layer, en.key) != nullptr) continue;
        auto cb = std::make_shared<CacheBlockData>(en.size);
        if (!cb->valid()) continue;
        std::memcpy(cb->data(), big.data() + (en.offset - range_off), en.size);
        (void)resident_->put(en.layer, en.key, std::move(cb),
                             resident_budget_);
        total += en.size;
        ++pinned;
      }
      ++batches;
    }
    ++i;
  }
  std::fprintf(stderr,
               "[tokshit] batched preload: %zu slots in %zu sequential batches, "
               "%llu MB pinned\n",
               pinned, batches,
               static_cast<unsigned long long>(total / (1024ULL * 1024ULL)));
}
// rmsnorm на месте, как в эталоне
// in-place rmsnorm, same as the reference
void Model::rmsnorm(std::span<float> x, std::span<const float> w, float eps) {
  const std::size_t n = x.size();
  float ss = 0.0F;
  for (float v : x) ss += v * v;
  const float inv = 1.0F / std::sqrt(ss / static_cast<float>(n) + eps);
  for (std::size_t i = 0; i < n; ++i) x[i] *= inv * w[i];
}

// rope применяем половинками: первая половина с cos, вторая с sin
// apply rope in halves, first half with cos, second with sin
void Model::rope_apply(std::span<float> x,
                       std::span<const std::pair<float, float>> table) {
  const std::size_t half = x.size() / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const float s = table[i].first;
    const float c = table[i].second;
    const float a = x[i];
    const float b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = a * s + b * c;
  }
}

// обычный softmax на месте, через вычитание максимума
// plain in-place softmax, subtract the max first
void Model::softmax(std::span<float> x) {
  float max = -std::numeric_limits<float>::infinity();
  for (float v : x) {
    if (v > max) max = v;
  }
  float sum = 0.0F;
  for (float& v : x) {
    v = std::exp(v - max);
    sum += v;
  }
  for (float& v : x) v /= sum;
}

// swiglu: g = silu(g) * u, на месте по g
// swiglu: g = silu(g) * u, in place over g
void Model::swiglu(std::span<float> g, std::span<const float> u) {
  for (std::size_t i = 0; i < g.size(); ++i) {
    g[i] = g[i] / (1.0F + std::exp(-g[i])) * u[i];
  }
}

// один слой. сначала attention, потом ffn (плотный или moe). всё пишется в
// workspace, ни одной аллокации. kv растёт на один токен за вызов
// one layer. attention first, then ffn (dense or moe). everything goes into
// the workspace, zero allocations. kv grows by one token per call
void Model::layer(std::size_t layer, Workspace& ws, KVCache& kv,
                  std::span<const std::pair<float, float>> rope_table) {
  // за один слой вперёд: пока считаем attention текущего слоя, читатели уже
  // тянут предсказанных экспертов следующего. два слоя вперёд тоже пробуем,
  // если они не попали в кэш и очередь не забита
  // TOKSHIT_PREFETCH_IDLE: только следующий слой — спекуляция живёт на
  // окнах простоя диска, читать на два слоя вперёд ей нечем
  // one layer ahead: while we do this layer's attention, the readers already
  // pull the next layer's predicted experts. two layers ahead too, if they
  // are not cached and the queue is not full. TOKSHIT_PREFETCH_IDLE: next
  // layer only — speculation lives on idle windows of the drive
  if (prefetch_enabled() && io_ && layer + 1 >= n_dense_) {
    static const bool idle_pf = prefetch_idle();
    const std::size_t ahead_max = idle_pf ? 1 : 2;
    for (std::size_t ahead = 1; ahead <= ahead_max; ++ahead) {
      const std::size_t nl = layer + ahead;
      if (nl >= n_layers_) break;
      prefetch_experts(nl, freq_->last[nl]);
    }
  }
  const auto t_attn = Clock::now();
  const Layer& lt = layers_[layer];
  const std::size_t kv_dim = n_kv_ * head_dim_;

  std::copy(ws.x.begin(), ws.x.end(), ws.hn.begin());
  rmsnorm(ws.hn, lt.attn_norm, eps_);

  std::fill(ws.q.begin(), ws.q.end(), 0.0F);
  std::fill(ws.k.begin(), ws.k.end(), 0.0F);
  std::fill(ws.v.begin(), ws.v.end(), 0.0F);
  const auto t_mm = Clock::now();
  {
    kernels::quantize_into(ws.hn, ws.act);
    const Seg segs[3] = {
        {meta_bytes(lt.attn_q), lt.attn_q.rb, lt.attn_q.quant, lt.attn_q.in_dim,
         ws.hn.data(), &ws.act, ws.q.data(), lt.attn_q.out_dim},
        {meta_bytes(lt.attn_k), lt.attn_k.rb, lt.attn_k.quant, lt.attn_k.in_dim,
         ws.hn.data(), &ws.act, ws.k.data(), lt.attn_k.out_dim},
        {meta_bytes(lt.attn_v), lt.attn_v.rb, lt.attn_v.quant, lt.attn_v.in_dim,
         ws.hn.data(), &ws.act, ws.v.data(), lt.attn_v.out_dim},
    };
    const uint64_t bytes =
        static_cast<uint64_t>(lt.attn_q.out_dim * lt.attn_q.rb +
                              lt.attn_k.out_dim * lt.attn_k.rb +
                              lt.attn_v.out_dim * lt.attn_v.rb);
    run_segs(std::span<const Seg>(segs, 3), 16);
    if (timing_enabled()) {
      metrics().t_mm.fetch_add(
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                   t_mm)
                  .count()),
          std::memory_order_relaxed);
      metrics().c_mm_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
  }
  tick(metrics().t_attn_mm, t_mm);

  for (std::size_t hd = 0; hd < n_heads_; ++hd) {
    rmsnorm(std::span<float>(ws.q.data() + hd * head_dim_, head_dim_),
            lt.attn_q_norm, eps_);
  }
  for (std::size_t hd = 0; hd < n_kv_; ++hd) {
    rmsnorm(std::span<float>(ws.k.data() + hd * head_dim_, head_dim_),
            lt.attn_k_norm, eps_);
  }
  if (rope_on_[layer]) {
    for (std::size_t hd = 0; hd < n_heads_; ++hd) {
      rope_apply(std::span<float>(ws.q.data() + hd * head_dim_, head_dim_),
                 rope_table);
    }
    for (std::size_t hd = 0; hd < n_kv_; ++hd) {
      rope_apply(std::span<float>(ws.k.data() + hd * head_dim_, head_dim_),
                 rope_table);
    }
  }
  kv[layer].insert(kv[layer].end(), ws.k.begin(), ws.k.end());
  kv[layer].insert(kv[layer].end(), ws.v.begin(), ws.v.end());

  const auto t_core = Clock::now();
  const std::size_t n_hist = kv[layer].size() / (2 * kv_dim);
  const bool swa = swa_[layer];
  const std::size_t start =
      (swa && n_hist > window_) ? n_hist - window_ : 0;
  const std::size_t n_use = (swa && n_hist > window_) ? window_ : n_hist;

  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
  const std::size_t group = n_heads_ / n_kv_;
  std::fill(ws.attn.begin(), ws.attn.end(), 0.0F);
  for (std::size_t hd = 0; hd < n_heads_; ++hd) {
    const std::size_t kv_h = hd / group;
    const std::size_t qs = hd * head_dim_;
    ws.scores.clear();
    for (std::size_t p = start; p < start + n_use; ++p) {
      const std::size_t off = p * 2 * kv_dim + kv_h * head_dim_;
      float dot = 0.0F;
      for (std::size_t i = 0; i < head_dim_; ++i) {
        dot += ws.q[qs + i] * kv[layer][off + i];
      }
      ws.scores.push_back(dot * scale);
    }
    ws.scores.push_back(lt.attn_sinks[hd]);
    softmax(ws.scores);
    ws.scores.pop_back();
    for (std::size_t i = 0; i < head_dim_; ++i) {
      float acc = 0.0F;
      for (std::size_t p = 0; p < n_use; ++p) {
        const std::size_t off =
            (start + p) * 2 * kv_dim + kv_dim + kv_h * head_dim_;
        acc += ws.scores[p] * kv[layer][off + i];
      }
      ws.attn[qs + i] = acc;
    }
  }
  tick(metrics().t_attn_core, t_core);

  std::fill(ws.o.begin(), ws.o.end(), 0.0F);
  const auto t_o = Clock::now();
  kernels::quantize_into(ws.attn, ws.act);
  matvec_preact(lt.attn_o, ws.attn, ws.o, ws.act);
  tick(metrics().t_attn_mm, t_o);
  for (std::size_t i = 0; i < hidden_; ++i) ws.x[i] += ws.o[i];

  tick(metrics().t_attn, t_attn);

  const auto t_ffn = Clock::now();
  // ffn: общая норма и дальше развилка плотный/moe
  // ffn: shared norm and then the dense/moe split
  std::copy(ws.x.begin(), ws.x.end(), ws.hf.begin());
  rmsnorm(ws.hf, lt.ffn_norm, eps_);

  if (layer < n_dense_) {
    std::fill(ws.g.begin(), ws.g.end(), 0.0F);
    std::fill(ws.u.begin(), ws.u.end(), 0.0F);
    std::fill(ws.dd.begin(), ws.dd.end(), 0.0F);
    kernels::quantize_into(ws.hf, ws.act);
    matvec_preact(*lt.ffn_gate, ws.hf,
                  std::span<float>(ws.g.data(), dense_ffn_), ws.act);
    matvec_preact(*lt.ffn_up, ws.hf,
                  std::span<float>(ws.u.data(), dense_ffn_), ws.act);
    swiglu(std::span<float>(ws.g.data(), dense_ffn_),
           std::span<const float>(ws.u.data(), dense_ffn_));
    kernels::quantize_into(std::span<float>(ws.g.data(), dense_ffn_), ws.act);
    matvec_preact(*lt.ffn_down, std::span<float>(ws.g.data(), dense_ffn_),
                  std::span<float>(ws.dd.data(), hidden_), ws.act);
    for (std::size_t i = 0; i < hidden_; ++i) ws.x[i] += ws.dd[i];
  } else {
    const auto t_router = Clock::now();
    // роутер: считаем логиты по экспертам, берём top-k, нормируем веса
    // router: score the experts, take top-k, normalize the weights
    std::fill(ws.logits.begin(), ws.logits.end(), 0.0F);
    std::span<float> router(ws.logits.data(), n_experts_);
    kernels::quantize_into(ws.hf, ws.act);
    matvec_preact(*lt.gate_inp, ws.hf, router, ws.act);

    const std::size_t k_used =
        router_topk_ != 0 && router_topk_ < n_used_ ? router_topk_ : n_used_;
    ws.idx.resize(n_experts_);
    for (std::size_t i = 0; i < n_experts_; ++i) ws.idx[i] = i;
    if (k_used < n_experts_) {
      std::nth_element(ws.idx.begin(),
                       ws.idx.begin() + static_cast<std::ptrdiff_t>(k_used),
                       ws.idx.end(), [&](std::size_t a, std::size_t b) {
                         return router[a] > router[b];
                       });
    }
    ws.top.assign(ws.idx.begin(),
                  ws.idx.begin() + static_cast<std::ptrdiff_t>(k_used));

    ws.wts.resize(ws.top.size());
    for (std::size_t i = 0; i < ws.top.size(); ++i) {
      ws.wts[i] = 1.0F / (1.0F + std::exp(-router[ws.top[i]]));
    }
    float wsum = 0.0F;
    for (float w : ws.wts) wsum += w;
    for (float& w : ws.wts) w /= wsum;
    record_freq(layer, ws.top);

    // Предсказание следующего слоя по текущему роутингу. В top-1/top-2
    // задержка чтения — главная статья, а соседние слои часто выбирают тот же
    // эксперт; заказываем его до того, как дойдём до следующего слоя.
    // Predict the next layer from the current routing. In top-1/top-2 read
    // latency dominates, and neighbouring layers often pick the same expert;
    // order it before the next layer starts.
    if (prefetch_enabled() && io_ != nullptr && resident_budget_ != 0) {
      for (std::size_t ahead = 1; ahead <= 2; ++ahead) {
        const std::size_t nl = layer + ahead;
        if (nl >= n_layers_) break;
        prefetch_experts(nl, ws.top);
      }
    }

    // PREFILL BULK: sequential pread на весь слой, потом take_expert возьмёт hit
    static const bool bulk_mode = false;  // disabled: reads 252MB to use 8MB
    if (bulk_mode && prefilling_ && io_ && resident_budget_ != 0) {
      prefill_bulk_layer(layer);
    }
    tick(metrics().t_router, t_router);

    static const bool layer_dbg = std::getenv("TOKSHIT_LAYER_DEBUG") != nullptr;
    const auto t_expert = Clock::now();
    const uint64_t dbg_miss0 = layer_dbg ? metrics().c_expert_miss.load() : 0;
    const uint64_t dbg_hit0 = layer_dbg ? metrics().c_expert_hit.load() : 0;
    const uint64_t dbg_dem0 = layer_dbg ? metrics().c_dem_dup.load() : 0;
    // грузим выбранных экспертов (параллельно), потом gate/up одним батчем,
    // swiglu, и down. это самый дорогой кусок на реальной модели
    // load the chosen experts (parallel), then gate/up as one batch, swiglu,
    // then down. this is the expensive part on a real model
    const auto t_load = Clock::now();
    Clock::time_point dbg_load_end{}, dbg_gu_end{}, dbg_dw_end{}, dbg_d_end{};
    // в draft-режиме берём ровно выбранное количество, не конфиг модели:
    // ниже не хватает десятков мест, все живут от этой локальной
    // in draft mode use exactly what the router picked, not the model config:
    // everything downstream lives from this one local
    const std::size_t n_used = k_used;
    const std::size_t ffn_dim = ffn_dim_;
    metrics().c_expert_union.fetch_add(n_used, std::memory_order_relaxed);
    metrics().c_expert_passes.fetch_add(1, std::memory_order_relaxed);
    const ExpertMeta em_g = *lt.gate_exps;
    const ExpertMeta em_u = *lt.up_exps;
    const ExpertMeta em_d = *lt.down_exps;
    static std::atomic<bool> expert_logged{false};
    if (!expert_logged.exchange(true)) {
      std::fprintf(
          stderr,
          "[expert] n_experts=%zu used=%zu ffn=%zu | gate in=%zu rb=%zu "
          "size=%zu q=%s | up in=%zu rb=%zu size=%zu q=%s | down in=%zu "
          "rb=%zu size=%zu q=%s\n",
          n_experts_, n_used, ffn_dim, em_g.in_dim, em_g.rb, em_g.size,
          std::string(quant_name(em_g.quant)).c_str(), em_u.in_dim, em_u.rb,
          em_u.size, std::string(quant_name(em_u.quant)).c_str(), em_d.in_dim,
          em_d.rb, em_d.size, std::string(quant_name(em_d.quant)).c_str());
    }

    ws.slices.resize(n_used * 3);
    ws.wg.resize(n_used);
    ws.wu.resize(n_used);
    ws.wd.resize(n_used);
    const std::size_t block = n_used * ffn_dim;
    std::fill(ws.gu.begin(), ws.gu.end(), 0.0F);
    kernels::quantize_into(ws.hf, ws.act);

    // заказываем сразу все чтения слоя. очередь спроса fifo, поэтому первые
    // матвеки не стоят за последними байтами
    // order all of the layer's reads at once. the demand queue is fifo, so
    // the first matmuls never wait behind the last bytes
    // TOKSHIT_IO_MERGE: соседи по proj лежат в файле подряд (эксперт e и
    // e+1 — смежные байты), выбранных экспертов сортируем по индексу и
    // цепочки уезжают одним pread каждый: меньше сисколлов, ближе к
    // sequential. попадания считаем той же бухгалтерией, что и раньше
    // TOKSHIT_IO_MERGE: neighbours within one proj are contiguous bytes
    // (expert e and e+1), so we sort the chosen experts by index and let
    // chains go out as one pread each: fewer syscalls, closer to sequential.
    // hits are accounted exactly like before
    static const bool io_merge = io_merge_enabled();
    if (io_ != nullptr && resident_budget_ != 0) {
      const ExpertMeta* ems[3] = {&em_g, &em_u, &em_d};
      if (io_merge) {
        for (std::size_t pi = 0; pi < 3; ++pi) {
          std::vector<std::pair<uint64_t, uint64_t>> entries;
          entries.reserve(n_used);
          for (std::size_t j = 0; j < n_used; ++j) {
            const uint64_t key = (static_cast<uint64_t>(pi) << 32) |
                                 static_cast<uint32_t>(ws.top[j]);
            if (io_->is_cached(layer, key)) {
              metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
            } else {
              metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);
            }
            entries.emplace_back(key,
                                 ems[pi]->base + ems[pi]->size * ws.top[j]);
          }
          io_->demand_merged(layer, entries, ems[pi]->size);
        }
      } else {
      for (std::size_t j = 0; j < n_used; ++j) {
        for (std::size_t pi = 0; pi < 2; ++pi) {
          const uint64_t key = (static_cast<uint64_t>(pi) << 32) |
                               static_cast<uint32_t>(ws.top[j]);
          const uint64_t off = ems[pi]->base + ems[pi]->size * ws.top[j];
          const bool cached = io_->demand_async(layer, key, off, ems[pi]->size);
          if (cached) {
            metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
          } else {
            metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
      for (std::size_t j = 0; j < n_used; ++j) {
        const uint64_t key = (static_cast<uint64_t>(2) << 32) |
                             static_cast<uint32_t>(ws.top[j]);
        const uint64_t off = em_d.base + em_d.size * ws.top[j];
        const bool cached = io_->demand_async(layer, key, off, em_d.size);
        if (cached) {
          metrics().c_expert_hit.fetch_add(1, std::memory_order_relaxed);
        } else {
          metrics().c_expert_miss.fetch_add(1, std::memory_order_relaxed);
        }
      }
      }

      // ждём gate/up всех экспертов. батч из 16 сегментов параллелится
      // лучше, чем 8 запусков по 2, и это перевешивает выигрыш от раннего
      // старта: замерено, батч быстрее
      // wait for every expert's gate/up. one 16-segment batch parallelises
      // better than 8 runs of 2, and that beats starting earlier: measured,
      // the batch wins
      for (std::size_t j = 0; j < n_used; ++j) {
        ws.wg[j] = take_expert(layer, 0, em_g, ws.top[j]);
        ws.wu[j] = take_expert(layer, 1, em_u, ws.top[j]);
      }
      tick(metrics().t_load, t_load);
      dbg_load_end = Clock::now();

      ws.segs.clear();
      for (std::size_t j = 0; j < n_used; ++j) {
        const std::size_t rb = std::min(
            em_g.rb, ws.wg[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
        ws.segs.push_back(
            Seg{std::span<const uint8_t>(ws.wg[j].data, ws.wg[j].len), rb,
                em_g.quant, em_g.in_dim, ws.hf.data(), &ws.act,
                ws.gu.data() + j * ffn_dim, ffn_dim});
      }
      for (std::size_t j = 0; j < n_used; ++j) {
        const std::size_t rb = std::min(
            em_u.rb, ws.wu[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
        ws.segs.push_back(
            Seg{std::span<const uint8_t>(ws.wu[j].data, ws.wu[j].len), rb,
                em_u.quant, em_u.in_dim, ws.hf.data(), &ws.act,
                ws.gu.data() + block + j * ffn_dim, ffn_dim});
      }
      const auto t_mm2 = Clock::now();
      run_segs(ws.segs, 32);
      tick(metrics().t_expert_mm, t_mm2);
      dbg_gu_end = Clock::now();
    } else {
      LoadJob job;
      job.model = this;
      job.layer = layer;
      job.tops = ws.top.data();
      job.ems[0] = &em_g;
      job.ems[1] = &em_u;
      job.ems[2] = &em_d;
      job.n_used = n_used;
      job.out = ws.slices.data();
      pool::run_tasks(n_used * 3, &Model::load_task, &job);
      for (std::size_t j = 0; j < n_used; ++j) {
        ws.wg[j] = ws.slices[j];
        ws.wu[j] = ws.slices[n_used + j];
      }
      tick(metrics().t_load, t_load);
      dbg_load_end = Clock::now();

      ws.segs.clear();
      for (std::size_t j = 0; j < n_used; ++j) {
        const std::size_t rb = std::min(
            em_g.rb, ws.wg[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
        ws.segs.push_back(
            Seg{std::span<const uint8_t>(ws.wg[j].data, ws.wg[j].len), rb,
                em_g.quant, em_g.in_dim, ws.hf.data(), &ws.act,
                ws.gu.data() + j * ffn_dim, ffn_dim});
      }
      for (std::size_t j = 0; j < n_used; ++j) {
        const std::size_t rb = std::min(
            em_u.rb, ws.wu[j].len / (ffn_dim == 0 ? 1 : ffn_dim));
        ws.segs.push_back(
            Seg{std::span<const uint8_t>(ws.wu[j].data, ws.wu[j].len), rb,
                em_u.quant, em_u.in_dim, ws.hf.data(), &ws.act,
                ws.gu.data() + block + j * ffn_dim, ffn_dim});
      }
      const auto t_mm2 = Clock::now();
      run_segs(ws.segs, 32);
      tick(metrics().t_expert_mm, t_mm2);
      dbg_gu_end = Clock::now();
    }

    std::span<float> g_all(ws.gu.data(), block);
    std::span<float> u_all(ws.gu.data() + block, block);
    for (std::size_t j = 0; j < n_used; ++j) {
      const std::size_t s = j * ffn_dim;
      swiglu(g_all.subspan(s, ffn_dim), u_all.subspan(s, ffn_dim));
    }

    ws.expert_acts.resize(n_used);
    for (std::size_t j = 0; j < n_used; ++j) {
      kernels::quantize_into(g_all.subspan(j * ffn_dim, ffn_dim),
                             ws.expert_acts[j]);
    }
    std::fill(ws.dd.begin(), ws.dd.end(), 0.0F);
    // down ждём только сейчас: всё время пока считались gate/up и swiglu,
    // оно уже читалось в фоне
    // down is awaited only now: while gate/up and swiglu were running it was
    // already being read in the background
    if (io_ != nullptr && resident_budget_ != 0) {
      const auto t_dw = Clock::now();
      for (std::size_t j = 0; j < n_used; ++j) {
        ws.wd[j] = take_expert(layer, 2, em_d, ws.top[j]);
      }
      tick(metrics().t_down_wait, t_dw);
      dbg_dw_end = Clock::now();
    } else {
      for (std::size_t j = 0; j < n_used; ++j) {
        ws.wd[j] = ws.slices[2 * n_used + j];
      }
    }
    ws.down_segs.clear();
    for (std::size_t j = 0; j < n_used; ++j) {
      const std::size_t rb = std::min(
          em_d.rb, ws.wd[j].len / (hidden_ == 0 ? 1 : hidden_));
      ws.down_segs.push_back(Seg{
          std::span<const uint8_t>(ws.wd[j].data, ws.wd[j].len), rb, em_d.quant,
          em_d.in_dim, g_all.data() + j * ffn_dim, &ws.expert_acts[j],
          ws.dd.data() + j * hidden_, hidden_});
    }
    {
      const auto t_mm3 = Clock::now();
      run_segs(ws.down_segs, 32);
      tick(metrics().t_expert_mm, t_mm3);
      dbg_d_end = Clock::now();
    }

    std::fill(ws.moe.begin(), ws.moe.end(), 0.0F);
    for (std::size_t j = 0; j < n_used; ++j) {
      const float w = ws.wts[j];
      const std::size_t base = j * hidden_;
      for (std::size_t i = 0; i < hidden_; ++i) ws.moe[i] += ws.dd[base + i] * w;
    }
    if (layer_dbg && !prefilling_) {
      const auto ms = [](Clock::time_point a, Clock::time_point b) {
        return a == Clock::time_point{}
                   ? 0.0
                   : std::chrono::duration<double>(b - a).count() * 1e3;
      };
      std::fprintf(
          stderr,
          "[layer %zu] load=%.2fms gu=%.2fms dw=%.2fms dmm=%.2fms "
          "miss=%llu hit=%llu dup=%llu q=%s\n",
          layer, ms(t_load, dbg_load_end), ms(dbg_load_end, dbg_gu_end),
          ms(dbg_gu_end, dbg_dw_end), ms(dbg_dw_end, dbg_d_end),
          static_cast<unsigned long long>(metrics().c_expert_miss.load() -
                                          dbg_miss0),
          static_cast<unsigned long long>(metrics().c_expert_hit.load() -
                                          dbg_hit0),
          static_cast<unsigned long long>(metrics().c_dem_dup.load() - dbg_dem0),
          std::string(quant_name(em_g.quant)).c_str());
    }
    tick(metrics().t_expert, t_expert);

    const auto t_shexp = Clock::now();
    // shared expert, обычный плотный mlp, добавляем к moe выходу
    // shared expert, a plain dense mlp, added to the moe output
    std::fill(ws.g.begin(), ws.g.end(), 0.0F);
    std::fill(ws.u.begin(), ws.u.end(), 0.0F);
    kernels::quantize_into(ws.hf, ws.act);
    matvec_preact(*lt.gate_shexp, ws.hf,
                  std::span<float>(ws.g.data(), shared_dim_), ws.act);
    matvec_preact(*lt.up_shexp, ws.hf,
                  std::span<float>(ws.u.data(), shared_dim_), ws.act);
    swiglu(std::span<float>(ws.g.data(), shared_dim_),
           std::span<const float>(ws.u.data(), shared_dim_));
    std::fill(ws.dd.begin(), ws.dd.end(), 0.0F);
    kernels::quantize_into(std::span<float>(ws.g.data(), shared_dim_), ws.act);
    matvec_preact(*lt.down_shexp, std::span<float>(ws.g.data(), shared_dim_),
                  std::span<float>(ws.dd.data(), hidden_), ws.act);
    for (std::size_t i = 0; i < hidden_; ++i) ws.moe[i] += ws.dd[i];
    for (std::size_t i = 0; i < hidden_; ++i) ws.x[i] += ws.moe[i];
    tick(metrics().t_shexp, t_shexp);
  }
  tick(metrics().t_ffn, t_ffn);
}

// выделяем все буферы один раз. дальше токен их только переиспользует.
// размеры берём из конфига модели
// allocate every buffer once. a token just reuses them after that. sizes come
// from the model config
void Model::prepare_workspace() {
  ws_ = std::make_unique<Workspace>();
  Workspace& w = *ws_;
  const std::size_t q_dim = n_heads_ * head_dim_;
  const std::size_t kv_dim = n_kv_ * head_dim_;
  const std::size_t ffn_cap = std::max({dense_ffn_, shared_dim_, std::size_t{1}});
  const std::size_t expert_elem =
      std::max(n_used_ * hidden_, std::max(hidden_, std::size_t{1}));

  w.x.assign(hidden_, 0.0F);
  w.hn.assign(hidden_, 0.0F);
  w.hf.assign(hidden_, 0.0F);
  w.o.assign(hidden_, 0.0F);
  w.dd.assign(expert_elem, 0.0F);
  w.moe.assign(hidden_, 0.0F);
  w.q.assign(q_dim, 0.0F);
  w.k.assign(kv_dim, 0.0F);
  w.v.assign(kv_dim, 0.0F);
  w.attn.assign(q_dim, 0.0F);
  w.g.assign(ffn_cap, 0.0F);
  w.u.assign(ffn_cap, 0.0F);
  w.gu.assign(2 * std::max(n_used_ * ffn_dim_, std::size_t{1}), 0.0F);
  w.scores.clear();
  w.scores.reserve(std::max(window_, std::size_t{4096}));
  w.logits.assign(vocab_, 0.0F);
  w.wts.assign(n_used_, 0.0F);
  w.idx.resize(n_experts_);
  w.top.assign(n_used_, 0);
  w.slices.resize(3 * n_used_);
  w.wg.resize(n_used_);
  w.wu.resize(n_used_);
  w.wd.resize(n_used_);
  w.segs.reserve(2 * n_used_);
  w.down_segs.reserve(n_used_);
  w.expert_acts.resize(n_used_);
  w.rope_table.resize(inv_freq_.size());
}

// главный вход на токен, без аллокаций (при условии что out уже нужного
// размера). делает эмбеддинг, все слои, финальную норму и голову
// the main per-token entry, allocation free (as long as `out` is big enough).
// does the embedding, all layers, the final norm and the head
void Model::token_logits(uint32_t token, KVCache& kv, std::size_t pos,
                         std::vector<float>& out) {
  Workspace& ws = *ws_;
  // prefill: только первые вызовы (до начала генерации). main.cpp
  // выставит prefilling_ = false перед генерацией
  const auto t_embd = Clock::now();
  const auto embd_bytes = meta_bytes(embd_);
  // границы словаря: невалидный id (битый сэмплер/промпт) раньше уходил
  // subspan'ом за пределы буфера весов — UB вместо управляемого нуля
  // vocab bounds: an invalid id (broken sampler/prompt) used to subspan past
  // the weight buffer — UB instead of a controlled zero
  if (static_cast<std::size_t>(token) >= vocab_) {
    std::fill(out.begin(), out.end(), 0.0F);
    return;
  }
  const std::size_t off = embd_.rb * static_cast<std::size_t>(token);
  const auto row = embd_bytes.subspan(off, embd_.rb);
  const auto deq = dequant_into(embd_.quant, row, hidden_, ws.x);
  if (!deq.ok()) {
    std::fill(out.begin(), out.end(), 0.0F);
    return;
  }
  tick(metrics().t_embd, t_embd);

  for (std::size_t i = 0; i < inv_freq_.size(); ++i) {
    const float angle = static_cast<float>(pos) * inv_freq_[i];
    ws.rope_table[i] = {std::sin(angle), std::cos(angle)};
  }

  std::size_t n_run = n_layers_;
  if (const char* v = std::getenv("GGUF_NLAYERS"); v != nullptr && *v != '\0') {
    n_run = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
    n_run = std::min(n_run, n_layers_);
  }
  if (std::getenv("TOKSHIT_DEBUG_NRUN") != nullptr) {
    std::fprintf(stderr, "[nrun] %zu/%zu\n", n_run, n_layers_);
  }

  // новый токен = новые предсказания. старое из очереди выбрасываем, а
  // предсказанное прошлым токеном ставим в очередь. только первый moe слой:
  // читать весь токен вперёд бессмысленно, диск и так в полку, а промахи
  // предсказания тратят полосу
  // new token, new predictions. drop the stale ones and queue what the last
  // token picked. only the first moe layer: reading the whole token ahead is
  // pointless, the drive is saturated anyway and wrong guesses waste
  // bandwidth
  if (prefetch_enabled() && io_ && resident_budget_ != 0) {
    io_->new_epoch();
    static const bool idle_pf = prefetch_idle();
    if (n_dense_ < n_run) {
      prefetch_experts(n_dense_, freq_->last[n_dense_]);
      // в idle-режиме второй слой вперёд не читаем: спекуляция только на
      // окнах простоя, а их хватает на один слой
      // in idle mode skip the second layer ahead: speculation only on idle
      // windows, and one layer is all they cover
      if (!idle_pf && n_dense_ + 1 < n_run) {
        prefetch_experts(n_dense_ + 1, freq_->last[n_dense_ + 1]);
      }
    }
  }
  for (std::size_t layer = 0; layer < n_run; ++layer) {
    this->layer(layer, ws, kv, ws.rope_table);
  }

  const auto t_head = Clock::now();
  rmsnorm(ws.x, out_norm_, eps_);
  out.resize(vocab_);
  std::fill(out.begin(), out.end(), 0.0F);
  kernels::quantize_into(ws.x, ws.act);
  matvec_preact(out_, ws.x, out, ws.act);
  tick(metrics().t_head, t_head);
}

// удобная обёртка для холодных вызовов, возвращает vector по значению
// convenience wrapper for cold calls, returns a vector by value
std::vector<float> Model::token_logits(uint32_t token, KVCache& kv,
                                        std::size_t pos) {
  std::vector<float> out(vocab_, 0.0F);
  token_logits(token, kv, pos, out);
  return out;
}

}  // namespace tokshit
