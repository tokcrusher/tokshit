#include "tokshit/model.hpp"
#include "tokshit/ahb.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "tokshit/gguf.hpp"
#include "tokshit/metrics.hpp"

namespace tokshit {
namespace {

// ── сериализация частотного профиля (TKFQ v1) ──────────────────────────────
// little-endian: u32 магия "TKFQ", u32 версия=1, u32 слоёв, дальше на каждый
// слой u32 count и count пар (u32 эксперт, u64 счёт)
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
    if (static_cast<std::size_t>(end - p) > 0) return false;
  }
  return loaded_layers == n_layers;
}

ResidentCache::ResidentCache(std::size_t layers) : layers_(layers) {}

// достаём блок и двигаем его в начало списка (mru). список и map живут под
// одним мьютексом слоя
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
bool Model::save_freq_profile(const std::string& path) const {
  return tokshit::save_freq_profile(*freq_, n_layers_, path);
}

bool Model::load_freq_profile(const std::string& path) {
  return tokshit::load_freq_profile(*freq_, n_layers_, n_experts_, path);
}

// draft-режим: роутер отбирает меньше экспертов, чем n_used. 0 = конфиг
// модели. веса при k_used=1 вырождаются в единицу — черновик читает в n_used
// раз меньше байтов с диска и считается пропорционально быстрее
void Model::set_router_topk(std::size_t k) {
  router_topk_ = k;
}

std::size_t Model::router_topk() const noexcept {
  return router_topk_ != 0 && router_topk_ < n_used_ ? router_topk_ : n_used_;
}

// прогрев после промпта: топ-k по истории частоты на каждом слое. теперь это
// реальные чтения в наш кэш, а не madvise в page cache
void Model::prefill_bulk_layer(std::size_t layer) {
  if (!ws_) return;
  prefill_bulk_layer(
      layer, std::span<const std::size_t>(ws_->top.data(), ws_->top.size()));
}

void Model::prefill_bulk_layer(std::size_t layer,
                               std::span<const std::size_t> experts) {
  const Layer& lt = layers_[layer];
  if (!lt.gate_exps || !lt.up_exps || !lt.down_exps) return;
  if (experts.empty()) return;

  // sequential pread на весь регион экспертов слоя (gate+up+down = ~252MB)
  // gguf-конвенция: три тензора идут подряд, gap минимален
  struct Region { uint64_t off; std::size_t size; };
  const ExpertMeta* ems[3] = {&*lt.gate_exps, &*lt.up_exps, &*lt.down_exps};
  Region regions[3];
  for (int pi = 0; pi < 3; ++pi) {
    // n_experts_, не 256: захардкоженное число сделало бы регион неверным
    // для любой модели не из флинта, и memcpy ниже читал бы за границей буфера
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

  // memcpy в AHB и resident_->put. при prefill бюджет слоя — общий, иначе
  // union экспертов батча вытесняется до take_expert
  // memcpy into AHB and resident_->put. during prefill the layer budget is
  const std::size_t n_used = experts.size();
  const uint64_t put_budget = prefilling_ ? total_budget_ : resident_budget_;
  std::size_t copied = 0;
  for (std::size_t j = 0; j < n_used; ++j) {
    const std::size_t e = experts[j];
    for (int pi = 0; pi < 3; ++pi) {
      const uint8_t* src = big.data() + (regions[pi].off - min_off) +
                           ems[pi]->size * e;
      auto cb = std::make_shared<CacheBlockData>(ems[pi]->size);
      if (!cb->valid()) continue;
      std::memcpy(cb->data(), src, ems[pi]->size);
      const uint64_t key =
          (static_cast<uint64_t>(pi) << 32) | static_cast<uint32_t>(e);
      (void)resident_->put(layer, key, std::move(cb), put_budget);
      copied += ems[pi]->size;
    }
  }
  static std::atomic<uint64_t> bulk_n{0};
  const uint64_t n = bulk_n.fetch_add(1, std::memory_order_relaxed);
  if (n < 8 || n % 8 == 0) {
    std::fprintf(stderr,
                 "[prefill-bulk] layer=%zu experts=%zu read=%.0f MB copied=%.0f "
                 "MB budget=%.0f MB\n",
                 layer, n_used, static_cast<double>(total) / 1e6,
                 static_cast<double>(copied) / 1e6,
                 static_cast<double>(put_budget) / 1e6);
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
}  // namespace tokshit
