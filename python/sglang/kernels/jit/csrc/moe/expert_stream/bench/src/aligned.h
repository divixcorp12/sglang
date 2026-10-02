// Page-aligned, zeroed host allocations for the bench.
//
// The allocating thread zeroes the buffer, which first-touches its pages: that thread's NUMA node backs them. Callers
// therefore allocate on the thread that will use the buffer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace fullstack {

// Releases a buffer from std::aligned_alloc.
struct FreeDeleter {
  void operator()(uint8_t* p) const {
    std::free(p);
  }
};
using AlignedBuffer = std::unique_ptr<uint8_t[], FreeDeleter>;

// Rounds `value` up to a multiple of `align`.
constexpr int64_t round_up(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
}

// Allocates `bytes` (rounded up to `align`) at an `align` boundary, zero-filled by the calling thread.
// Throws std::bad_alloc.
inline AlignedBuffer aligned_zeroed(int64_t bytes, int64_t align = 4096) {
  const auto size = static_cast<size_t>(round_up(bytes, align));
  void* p = std::aligned_alloc(static_cast<size_t>(align), size);
  if (p == nullptr) throw std::bad_alloc();
  std::memset(p, 0, size);
  return AlignedBuffer(static_cast<uint8_t*>(p));
}

}  // namespace fullstack
