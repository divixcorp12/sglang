#pragma once

#include <liburing.h>
#include <sys/uio.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

/// Row-aligned io_uring registered-buffer chunking and a sparse-slot table,
/// shared by UringFileReader and the expert-stream reader (plan
/// 2026-09-28-reader-crtp-uring-registration, Task 6).
namespace sglang::io {

constexpr uint64_t kMaxRegisteredBufferBytes = 1ULL << 30;  // io_buffer_validate refuses more (-EFAULT)
constexpr unsigned kMaxRegisteredBufferSlots = 1U << 14;    // IORING_MAX_REG_BUFFERS

struct ChunkPlan {
  uint64_t base;
  uint64_t length;
};

/// Cut [base, base + bytes) into chunks of at most `cap`. row_bytes == 0: cap-sized chunks from base (the
/// UringFileReader behavior). row_bytes > 0: each chunk is floor(cap / row_bytes) whole rows, the last the
/// remainder, so every row [base + k*row_bytes, +row_bytes) lies in exactly one chunk. Throws
/// std::invalid_argument when row_bytes > cap, or bytes is not a multiple of row_bytes.
inline std::vector<ChunkPlan> plan_chunks(uint64_t base, uint64_t bytes, uint64_t row_bytes, uint64_t cap) {
  if (cap == 0 || cap > kMaxRegisteredBufferBytes) {
    throw std::invalid_argument("registered chunk cap must be in (0, 1 GiB]");
  }
  uint64_t step = cap;
  if (row_bytes > 0) {
    if (row_bytes > cap) {
      throw std::invalid_argument(
          "a " + std::to_string(row_bytes) + " B row does not fit one registered buffer of " + std::to_string(cap) +
          " B");
    }
    if (bytes % row_bytes != 0) {
      throw std::invalid_argument("a registered slab must be whole rows");
    }
    step = (cap / row_bytes) * row_bytes;
  }
  std::vector<ChunkPlan> plan;
  for (uint64_t at = 0; at < bytes; at += step) {
    plan.push_back({base + at, std::min(step, bytes - at)});
  }
  return plan;
}

/// A sparse io_uring registered-buffer table: fixed slots filled and emptied with
/// ``io_uring_register_buffers_update_tag``, tracked by base address so a destination range can be looked up by
/// containment and released by range.
class RegisteredBufferTable {
 public:
  /// Sparse-registers `slots` empty entries on `ring`. False: the ring does not support it (errno in last_error()).
  bool init(io_uring* ring, unsigned slots) {
    ring_ = ring;
    buffers_.clear();
    last_error_ = 0;
    last_error_context_.clear();
    int rc = io_uring_register_buffers_sparse(ring, slots);
    supported_ = rc == 0;
    if (supported_) {
      slot_used_.assign(slots, false);
    } else {
      slot_used_.clear();
      last_error_ = -rc;
      last_error_context_ = "register_buffers_sparse: " + std::string(std::strerror(-rc));
    }
    return supported_;
  }

  bool supported() const {
    return supported_;
  }

  /// Registers plan_chunks(base, bytes, row_bytes, cap) into free slots. Returns the chunk count; 0 on overlap, no
  /// free slot or a kernel refusal, after rolling back every chunk of this call (last_error() says which).
  int64_t add(uint64_t base, uint64_t bytes, uint64_t row_bytes, uint64_t cap = kMaxRegisteredBufferBytes) {
    last_error_ = 0;
    last_error_context_.clear();
    if (!supported_ || bytes == 0) {
      return 0;
    }
    for (const Buffer& buffer : buffers_) {
      if (buffer.base < base + bytes && base < buffer.base + buffer.length) {
        last_error_context_ = "overlap";
        return 0;
      }
    }
    std::vector<unsigned> added;
    for (const ChunkPlan& chunk : plan_chunks(base, bytes, row_bytes, cap)) {
      auto free_slot = std::find(slot_used_.begin(), slot_used_.end(), false);
      if (free_slot == slot_used_.end()) {
        rollback_(added);
        last_error_context_ = "no free slot";
        return 0;
      }
      unsigned slot = static_cast<unsigned>(free_slot - slot_used_.begin());
      struct iovec vector {
        reinterpret_cast<void*>(chunk.base), static_cast<size_t>(chunk.length)
      };
      __u64 tag = 0;
      int rc = io_uring_register_buffers_update_tag(ring_, slot, &vector, &tag, 1);
      if (rc != 1) {
        // update_tag returns the entries it updated: 0 with no errno (nothing registered) is reported as EIO.
        int error = rc < 0 ? -rc : EIO;
        last_error_ = error;
        last_error_context_ = "update slot " + std::to_string(slot) + ": " + std::strerror(error);
        rollback_(added);
        return 0;
      }
      slot_used_[slot] = true;
      buffers_.push_back(Buffer{chunk.base, chunk.length, slot});
      added.push_back(slot);
    }
    sort_buffers_();
    return static_cast<int64_t>(added.size());
  }

  /// Returns the slot holding [destination, destination + length) whole, or -1.
  int find(uint64_t destination, uint64_t length) const {
    auto after =
        std::upper_bound(buffers_.begin(), buffers_.end(), destination, [](uint64_t value, const Buffer& buffer) {
          return value < buffer.base;
        });
    if (after == buffers_.begin()) {
      return -1;
    }
    const Buffer& buffer = *(after - 1);
    if (destination + length <= buffer.base + buffer.length) {
      return static_cast<int>(buffer.slot);
    }
    return -1;
  }

  /// Unregisters every chunk lying inside [low, high). Returns the count.
  int64_t remove_range(uint64_t low, uint64_t high) {
    std::vector<unsigned> removed;
    for (const Buffer& buffer : buffers_) {
      if (buffer.base >= low && buffer.base + buffer.length <= high) {
        removed.push_back(buffer.slot);
      }
    }
    rollback_(removed);
    return static_cast<int64_t>(removed.size());
  }

  /// Forgets every slot, without touching the (already torn-down) ring. Safe to `init` again afterward.
  void clear() {
    buffers_.clear();
    slot_used_.clear();
    supported_ = false;
    ring_ = nullptr;
    last_error_ = 0;
    last_error_context_.clear();
  }

  size_t chunks() const {
    return buffers_.size();
  }

  uint64_t bytes() const {
    uint64_t total = 0;
    for (const Buffer& buffer : buffers_) {
      total += buffer.length;
    }
    return total;
  }

  uint64_t largest() const {
    uint64_t largest = 0;
    for (const Buffer& buffer : buffers_) {
      largest = std::max(largest, buffer.length);
    }
    return largest;
  }

  int last_error() const {
    return last_error_;
  }

  std::string last_error_context() const {
    return last_error_context_;
  }

 private:
  struct Buffer {
    uint64_t base;
    uint64_t length;
    unsigned slot;
  };

  void rollback_(const std::vector<unsigned>& slots) {
    for (unsigned slot : slots) {
      struct iovec empty {
        nullptr, 0
      };
      __u64 tag = 0;
      io_uring_register_buffers_update_tag(ring_, slot, &empty, &tag, 1);
      slot_used_[slot] = false;
    }
    buffers_.erase(
        std::remove_if(
            buffers_.begin(),
            buffers_.end(),
            [&](const Buffer& buffer) { return std::find(slots.begin(), slots.end(), buffer.slot) != slots.end(); }),
        buffers_.end());
  }

  void sort_buffers_() {
    std::sort(buffers_.begin(), buffers_.end(), [](const Buffer& left, const Buffer& right) {
      return left.base < right.base;
    });
  }

  io_uring* ring_ = nullptr;
  bool supported_ = false;
  std::vector<bool> slot_used_;
  std::vector<Buffer> buffers_;
  int last_error_ = 0;
  std::string last_error_context_;
};

}  // namespace sglang::io
