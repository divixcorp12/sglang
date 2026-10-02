// First-touch host allocations: zeroed by the allocating thread, so its NUMA node backs the pages.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace fullstack {

struct FreeDeleter {
  void operator()(uint8_t* p) const {
    std::free(p);
  }
};
using AlignedBuffer = std::unique_ptr<uint8_t[], FreeDeleter>;

constexpr int64_t round_up(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
}

inline AlignedBuffer aligned_zeroed(int64_t bytes, int64_t align = 4096) {
  const auto size = static_cast<size_t>(round_up(bytes, align));
  void* p = std::aligned_alloc(static_cast<size_t>(align), size);
  if (p == nullptr) throw std::bad_alloc();
  std::memset(p, 0, size);
  return AlignedBuffer(static_cast<uint8_t*>(p));
}

}  // namespace fullstack
