// The copy engine's CPU-only backend, for the tests and the full-stack bench (device < 0 in enable_copy_engine).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

#include "copy_engine.h"

namespace sglang {
namespace expert_stream {

/// A CPU-only test backend: "copies" between host buffers, completed only when the test releases them.
///
/// A mark's copies land when it is released, in query(), in mark order (one stream), so a CopyDone published before
/// its release shows up as bytes that are not there yet. issue, mark and query run on the copy thread; release, fail
/// and marked on a test thread. The two sides share only atomics and nothing allocates after construction, so the
/// backend cannot hide a race the CUDA one would show.
class HostCopyBackend : public CopyBackend {
 public:
  /// A mark's slot is reused kMarks marks later; a mark is open only while its job is (kCopyRing at most).
  static constexpr int kMarks = 2 * static_cast<int>(kCopyRing);
  /// One mark's copies: every lane's copy of every layout name (row_layout.h caps a layout at 32), plus a ballast copy.
  static constexpr int kEntries = Wire::kLanes * 32 + 1;
  static constexpr int kIssueFailed = -1;
  static constexpr int kQueryFailed = -2;
  static constexpr int kMarkFull = -3;

  std::string init() override {
    return "";
  }

  int issue(uint64_t dst, uint64_t src, int64_t bytes) override {
    if (fail_issue_.load(std::memory_order_acquire)) return kIssueFailed;
    Mark& mark = marks_[open_ % kMarks];
    if (mark.count == kEntries) return kMarkFull;
    mark.entries[mark.count++] = CopyEntry{src, dst, bytes};
    return 0;
  }

  /// Closes the open mark (its token is the number of marks closed before it) and opens the next.
  int mark(int64_t* token) override {
    *token = open_++;
    marks_[open_ % kMarks].count = 0;
    marked_.store(open_, std::memory_order_release);
    return 0;
  }

  int query(int64_t token) override {
    if (fail_query_.load(std::memory_order_acquire)) return kQueryFailed;
    if (token < completed_) return kDone;
    if (token != completed_ || released_.load(std::memory_order_acquire) <= completed_) return kPending;
    const Mark& mark = marks_[token % kMarks];
    for (int i = 0; i < mark.count; ++i) {
      const CopyEntry& copy = mark.entries[i];
      std::memcpy(
          reinterpret_cast<void*>(copy.dst), reinterpret_cast<const void*>(copy.src), static_cast<size_t>(copy.bytes));
    }
    ++completed_;
    return kDone;
  }

  void shutdown(bool) override {}

  /// Lets `marks` more marks complete, or every mark from now on if negative. The budget is standing, so a release
  /// made before the copy thread marks still counts.
  void release(int64_t marks) {
    int64_t seen = released_.load(std::memory_order_relaxed);
    int64_t next = 0;
    do {
      next = marks < 0 || seen > INT64_MAX - marks ? INT64_MAX : seen + marks;
    } while (!released_.compare_exchange_weak(seen, next, std::memory_order_release, std::memory_order_relaxed));
  }

  void fail(bool issue, bool query) {
    fail_issue_.store(issue, std::memory_order_release);
    fail_query_.store(query, std::memory_order_release);
  }

  /// Marks closed so far: one per job issued.
  int64_t marked() const {
    return marked_.load(std::memory_order_acquire);
  }

 private:
  struct Mark {
    std::array<CopyEntry, kEntries> entries{};
    int count = 0;
  };
  std::array<Mark, kMarks> marks_{};  // about 0.5 MB, allocated once with the backend
  int64_t open_ = 0;                  // copy thread
  int64_t completed_ = 0;             // copy thread
  std::atomic<int64_t> marked_{0};    // written by the copy thread
  std::atomic<int64_t> released_{0};  // written by the test thread
  std::atomic<bool> fail_issue_{false};
  std::atomic<bool> fail_query_{false};
};

}  // namespace expert_stream
}  // namespace sglang
