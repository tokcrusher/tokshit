#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tokshit::ahb {

// Пул под expert-кэш.
//
// Зачем AHardwareBuffer: память dma-buf не является анонимной, поэтому ядро не
// кладёт её в ZRAM и LMK не вытесняет её как обычный anon. Для экспертов это
// важно: сжатая страница эксперта означает распаковку на каждом matvec.
//
// Пул нарезает слябы по классам размеров. Классы задаёт загрузчик модели: у
// gate/up и down экспертов разные размеры, и один общий слот «по максимуму»
// тратил бы до половины бюджета впустую.
//
// Если AHB недоступен (не Android, отказ gralloc, кончилась память) — прозрачный
// fallback на malloc: рантайм продолжает работать, просто без pinning.
struct Config {
    // Уникальные размеры блоков, которые будет выдавать пул. Округляются вверх
    // до 4 КБ внутри init(). Пустой список = один класс на slot_bytes.
    std::vector<std::size_t> slot_sizes;
    std::size_t slot_bytes  = 576 * 1024;                 // fallback-класс
    std::size_t slab_bytes  = 128 * 1024 * 1024;          // сляб на класс
    std::size_t total_bytes = 4ULL * 1024 * 1024 * 1024;  // общий бюджет
    bool        enabled     = true;                       // TOKSHIT_AHB=0 -> malloc
};

// Один большой долгоживущий буфер вне пула: плотные веса модели.
//
// Плотные веса (attention, shared expert, голова) читаются на каждом токене.
// Если это анонимная память, Android под давлением кладёт её в ZRAM, и каждое
// чтение превращается в распаковку. dma-buf не свопится: страница остаётся
// страницей. Поэтому hot-веса живут здесь, а не в std::vector<uint8_t>.
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(std::size_t bytes);  // AHB, при отказе — malloc
    ~Buffer();

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    uint8_t*       data() noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }
    std::size_t    size() const noexcept { return size_; }
    bool           valid() const noexcept { return data_ != nullptr; }
    bool           on_ahb() const noexcept { return on_ahb_; }

    void release() noexcept;

private:
    uint8_t*    data_ = nullptr;
    std::size_t size_ = 0;
    void*       handle_ = nullptr;  // AHardwareBuffer*, непрозрачный
    bool        on_ahb_ = false;
};

// Плотные веса по тензорам. Один общий AHB на 1.5 ГБ gralloc часто
// отвергает, поэтому каждый тензор получает свой буфер, а логические
// смещения (Meta::base) отображаются на буферы через data_at(). Тензор
// никогда не пересекает границу mapping, так что matvec читает непрерывный
// диапазон.
class HotWeights {
public:
    HotWeights() = default;

    // sizes — размеры тензоров в том порядке, в котором их будет копировать
    // загрузчик. false, если памяти заведомо не хватает: тогда вызывающий
    // остаётся на mmap и не рискует OOM.
    bool allocate(const std::vector<std::size_t>& sizes);

    uint8_t*       data_at(std::size_t offset);
    const uint8_t* data_at(std::size_t offset) const;

    std::size_t size() const noexcept { return size_; }
    std::size_t ahb_bytes() const noexcept { return ahb_bytes_; }
    bool        valid() const noexcept { return size_ != 0 && !segs_.empty(); }
    bool        on_ahb() const noexcept { return size_ != 0 && ahb_bytes_ == size_; }
    void        clear();

private:
    struct Seg {
        std::size_t start = 0;
        std::size_t len = 0;
        Buffer buf;
    };
    std::vector<Seg> segs_;
    std::size_t size_ = 0;
    std::size_t ahb_bytes_ = 0;
};

class Pool {
public:
    static Pool& instance();

    // Идемпотентен: повторный вызов с другим конфигом игнорируется.
    void init(const Config& cfg);

    // Возвращает блок не меньше size. nullptr только если и malloc упал.
    uint8_t* allocate(std::size_t size);
    // size обязан совпадать с size, переданным в allocate. Учёт ведётся по
    // собственной записи пула, поэтому неверный size не ломает бюджет.
    void     deallocate(uint8_t* ptr, std::size_t size);

    bool        is_ahb(uint8_t* ptr) const;
    std::size_t slots_total() const;
    std::size_t slots_free()  const;
    std::size_t slot_bytes()  const;  // самый большой класс, для логов
    std::size_t class_count() const;
    std::size_t bytes_used()  const;
    // Сколько байт выдано через malloc fallback (не AHB). Нужен тестам и
    // диагностике: AHB на не-Android не существует.
    std::size_t fallback_bytes() const;
    bool        ahb_active()  const;

    ~Pool();

private:
    Pool();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tokshit::ahb
