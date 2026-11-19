#pragma once

// Блок памяти expert-кэша из AHB-пула: pinned dma-buf, не сжимается ZRAM и не
// вытесняется LMK. Если AHB недоступен или размер больше slot_bytes, пул
// отдаёт malloc. Интерфейс чтения: .data() и .size(); запись доступна только
// до публикации блока в кэш.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

#include "ahb.hpp"

namespace tokshit {

class CacheBlockData {
public:
    explicit CacheBlockData(std::size_t n) : size_(n) {
        data_ = ahb::Pool::instance().allocate(n);
    }

    ~CacheBlockData() {
        if (data_ != nullptr) {
            ahb::Pool::instance().deallocate(data_, size_);
        }
    }

    CacheBlockData(const CacheBlockData&) = delete;
    CacheBlockData& operator=(const CacheBlockData&) = delete;

    const uint8_t* data() const noexcept { return data_; }
    uint8_t*       data()       noexcept { return data_; }
    // Запись разрешена только до публикации блока в кэш: после этого блок
    // читают несколько потоков одновременно.
    uint8_t*       writable()   noexcept { return data_; }
    std::size_t    size() const noexcept { return size_; }
    bool           valid() const noexcept { return data_ != nullptr; }
    bool           on_ahb() const noexcept {
        return data_ != nullptr && ahb::Pool::instance().is_ahb(data_);
    }

private:
    uint8_t*    data_ = nullptr;
    std::size_t size_ = 0;
};

using CacheBlock = std::shared_ptr<const CacheBlockData>;

} // namespace tokshit
