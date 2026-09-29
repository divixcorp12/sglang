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

namespace detail {
// Whether liburing declares io_uring_clone_buffers_offset (2.9+). A dependent call, so a header without it (the
// fake liburing of test_expert_stream_uring_options.py) makes this false instead of failing to compile.
template <class Ring>
constexpr bool kHasCloneBuffers = requires(Ring* ring) { io_uring_clone_buffers_offset(ring, ring, 0u, 0u, 0u, 0u); };
constexpr unsigned kCloneDstReplace = 1u << 1;  // IORING_REGISTER_DST_REPLACE

// Pins `vector` in slot 0 of `scratch`, clones it into `slot` of `ring`, and empties the scratch slot again.
// Returns 1 or a negative errno; `cloned` is false when the failure came before the clone could happen.
template <class Ring>
int clone_into(Ring* ring, Ring* scratch, unsigned slot, const iovec& vector, bool& cloned) {
  cloned = false;
  if constexpr (kHasCloneBuffers<Ring>) {
    __u64 tag = 0;
    int rc = io_uring_register_buffers_update_tag(scratch, 0, &vector, &tag, 1);
    if (rc != 1) return rc < 0 ? rc : -EIO;
    rc = io_uring_clone_buffers_offset(ring, scratch, slot, 0, 1, kCloneDstReplace);
    cloned = rc >= 0;
    struct iovec empty {
      nullptr, 0
    };
    io_uring_register_buffers_update_tag(scratch, 0, &empty, &tag, 1);
    return rc < 0 ? rc : 1;
  } else {
    return -EOPNOTSUPP;
  }
}

template <class Ring>
bool open_scratch(Ring* scratch) {
  if constexpr (kHasCloneBuffers<Ring>) {
    if (io_uring_queue_init(1, scratch, 0) != 0) return false;
    if (io_uring_register_buffers_sparse(scratch, 1) == 0) return true;
    io_uring_queue_exit(scratch);
  }
  return false;
}
}  // namespace detail

/// A sparse io_uring registered-buffer table: fixed slots filled and emptied with
/// ``io_uring_register_buffers_update_tag``, tracked by base address so a destination range can be looked up by
/// containment and released by range.
///
/// Each chunk is registered in slot 0 of a private one-slot scratch ring and cloned into its slot of the table's ring
/// (IORING_REGISTER_CLONE_BUFFERS), not registered there directly. Registering pins the chunk and charges it with
/// io_buffer_account_pin, whose headpage_already_acct walks every page of every buffer already in that ring for each
/// huge page of the new chunk. Over a tier whose transparent huge pages fell back to 4 KiB under fragmentation, that
/// walk made direct registration quadratic: 23-103 s for a 100 GiB tier on divix01. In the scratch ring the walk
/// sees only the chunk itself, and a clone takes references to the pinned pages without accounting them again
/// (analysis/dsv41-drive/thp-fallback/results.md). A liburing or kernel without cloning keeps the direct path.
class RegisteredBufferTable {
 public:
  /// `clone` false always registers directly into the ring (tests of both paths).
  explicit RegisteredBufferTable(bool clone = true) : clone_(clone) {}
  RegisteredBufferTable(const RegisteredBufferTable&) = delete;
  RegisteredBufferTable& operator=(const RegisteredBufferTable&) = delete;
  ~RegisteredBufferTable() {
    close_scratch_();
  }

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
    close_scratch_();
    if (supported_ && clone_) scratch_open_ = detail::open_scratch(&scratch_);
    return supported_;
  }

  bool supported() const {
    return supported_;
  }

  /// Whether the next chunk goes through the scratch ring and a clone. Turns false for good (until init) the first
  /// time a clone is refused, e.g. by a kernel without IORING_REGISTER_CLONE_BUFFERS.
  bool cloning() const {
    return scratch_open_;
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
      int rc = register_slot_(slot, vector);
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
    close_scratch_();
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
  int register_slot_(unsigned slot, const iovec& vector) {
    if (scratch_open_) {
      bool cloned = false;
      const int rc = detail::clone_into(ring_, &scratch_, slot, vector, cloned);
      if (rc == 1 || cloned) return rc;
      // Pinning in the scratch ring or the clone itself was refused: stop cloning and register directly, which
      // reports its own error if the chunk cannot be registered at all.
      close_scratch_();
    }
    __u64 tag = 0;
    return io_uring_register_buffers_update_tag(ring_, slot, &vector, &tag, 1);
  }

  void close_scratch_() {
    if (scratch_open_) io_uring_queue_exit(&scratch_);
    scratch_open_ = false;
  }

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
  bool clone_ = true;
  bool scratch_open_ = false;
  io_uring scratch_{};
  bool supported_ = false;
  std::vector<bool> slot_used_;
  std::vector<Buffer> buffers_;
  int last_error_ = 0;
  std::string last_error_context_;
};

}  // namespace sglang::io
