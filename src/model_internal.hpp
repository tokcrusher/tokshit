#pragma once

#include "tokshit/model.hpp"

namespace tokshit {

// Задача на пачку строк одного сегмента matvec.
struct RowJob {
  const Model::Seg* segs = nullptr;
  std::size_t nsegs = 0;
  const std::size_t* task_starts = nullptr;
  std::size_t task_starts_len = 0;
  std::size_t rows_per_task = 0;
};

// Задача на загрузку одного эксперта одной проекции.
struct LoadJob {
  const Model* model = nullptr;
  std::size_t layer = 0;
  const std::size_t* tops = nullptr;
  const ExpertMeta* ems[3] = {nullptr, nullptr, nullptr};
  std::size_t n_used = 0;
  Model::WRef* out = nullptr;
};

}  // namespace tokshit
