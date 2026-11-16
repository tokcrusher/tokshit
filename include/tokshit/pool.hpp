#pragma once

#include <cstddef>
#include <cstdint>

namespace tokshit::pool {

// сигнатура задачи: непрозрачный контекст и индекс. `ctx` общий для всех
// воркеров, `i` уникален в [0, n)
// task signature, opaque ctx + index. ctx is shared, i is unique in [0, n)
using TaskFn = void (*)(void* ctx, std::size_t i);

// постоянный пул воркеров с busy-wait барьером. вызывающий поток тоже считает,
// так что параллелизм это workers + 1
// persistent worker pool with a spin barrier. the calling thread also works,
// so parallelism is workers + 1
class TaskPool {
 public:
  static TaskPool& instance();

  TaskPool(const TaskPool&) = delete;
  TaskPool& operator=(const TaskPool&) = delete;
  ~TaskPool();

  // синхронный барьер: вернётся когда каждый i из [0, n) отработал ровно раз
  // sync barrier, returns after every i in [0, n) ran exactly once
  void run(std::size_t n, TaskFn fn, void* ctx);

  // режим горячего спина на время декода, иначе воркеры ненадолго паркуются
  // hot spin during decode, otherwise workers park for a bit
  void set_active(bool active);
  [[nodiscard]] bool active() const noexcept;

  [[nodiscard]] std::size_t workers() const noexcept;
  [[nodiscard]] uint64_t main_tasks() const noexcept;
  [[nodiscard]] uint64_t worker_tasks() const noexcept;

 private:
  TaskPool();

  struct Impl;
  Impl* impl_ = nullptr;
};

[[nodiscard]] bool is_active() noexcept;
void set_active(bool active);

// Convenience wrapper around the singleton.
inline void run_tasks(std::size_t n, TaskFn fn, void* ctx) {
  TaskPool::instance().run(n, fn, ctx);
}

}  // namespace tokshit::pool
