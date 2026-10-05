// Thread-handoff primitives for the service path: a lock-free SPSC ring, a one-thread deque, and futex wait/wake.
//
// Nothing here allocates after construction or takes a lock.
//
//   SpscRing     single-producer single-consumer ring between two threads
//   FixedDeque   circular FIFO owned by one thread (the copy thread's in-flight jobs)
//   futex_wait / futex_wake   sleep and wake on a 32-bit word
//   Doorbell     a futex wake that costs a syscall only when the consumer sleeps
#pragma once

#include <linux/futex.h>
#include <sys/syscall.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <time.h>
#include <unistd.h>

namespace sglang::expert_stream {

// A bounded lock-free queue for exactly one producer thread and one consumer thread; N is a power of two.
//
// Each index lives on its owner's cache line beside that owner's cached copy of the other index, so a push or pop
// that finds room touches no line the other side writes. push() returns false when full and pop() false when empty.
template <class T, size_t N>
class SpscRing {
  static_assert(N >= 2 && (N & (N - 1)) == 0, "SpscRing capacity is a power of two");

 public:
  // Producer: appends `value`, or returns false when the ring is full.
  bool push(const T& value) {
    const uint64_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_seen_ == N) {
      tail_seen_ = tail_.load(std::memory_order_acquire);
      if (head - tail_seen_ == N) return false;
    }
    slots_[head & (N - 1)] = value;
    head_.store(head + 1, std::memory_order_release);
    return true;
  }
  // Consumer: moves the oldest item into `*out`, or returns false when empty.
  bool pop(T* out) {
    const uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_seen_) {
      head_seen_ = head_.load(std::memory_order_acquire);
      if (tail == head_seen_) return false;
    }
    *out = slots_[tail & (N - 1)];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }
  // Consumer: the oldest item without removing it, or null when empty. The pointer is valid until the consumer's next
  // pop(), because the producer never writes a slot the consumer has not released.
  const T* front() {
    const uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_seen_) {
      head_seen_ = head_.load(std::memory_order_acquire);
      if (tail == head_seen_) return nullptr;
    }
    return &slots_[tail & (N - 1)];
  }
  // True when no item is queued. Callable from either side; exact only on the consumer.
  bool empty() const {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

 private:
  alignas(64) std::atomic<uint64_t> head_{0};
  uint64_t tail_seen_ = 0;  // producer's copy of tail_
  alignas(64) std::atomic<uint64_t> tail_{0};
  uint64_t head_seen_ = 0;  // consumer's copy of head_
  alignas(64) std::array<T, N> slots_{};
};

// A circular FIFO owned by a single thread: the copy thread's in-flight, held and acking jobs. Not thread-safe.
// push_back() returns false when full; the caller decides what that means (the copy engine fails stop).
template <class T, size_t N>
class FixedDeque {
 public:
  bool push_back(const T& value) {
    if (n_ == N) return false;
    data_[(head_ + n_++) % N] = value;
    return true;
  }
  T& front() {
    return data_[head_];
  }
  void pop_front() {
    head_ = (head_ + 1) % N;
    --n_;
  }
  bool empty() const {
    return n_ == 0;
  }
  size_t size() const {
    return n_;
  }
  T& operator[](size_t i) {
    return data_[(head_ + i) % N];
  }
  // Keeps order; removes every element `pred` accepts.
  template <class Pred>
  void erase_if(Pred pred) {
    size_t kept = 0;
    for (size_t i = 0; i < n_; ++i) {
      T& value = data_[(head_ + i) % N];
      if (!pred(value)) data_[(head_ + kept++) % N] = value;
    }
    n_ = kept;
  }

 private:
  std::array<T, N> data_{};
  size_t head_ = 0;
  size_t n_ = 0;
};

// Sleeps while *word == expected, for at most timeout_ns. Returns at once if *word already differs: the kernel
// compares under its own lock, so a wake that changed the word before the call is never lost.
inline void futex_wait(std::atomic<uint32_t>* word, uint32_t expected, int64_t timeout_ns) {
  timespec timeout{static_cast<time_t>(timeout_ns / 1000000000), static_cast<long>(timeout_ns % 1000000000)};
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAIT_PRIVATE, expected, &timeout, nullptr, 0);
}

// Wakes at most one thread sleeping in futex_wait on `word`.
inline void futex_wake(std::atomic<uint32_t>* word) {
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

/// Wakes one consumer thread that sleeps when its queue runs dry; ring() makes a syscall only while it sleeps.
///
/// No wake is lost. ring() bumps the word, fences, then reads sleeping_; sleep_unless() sets sleeping_, fences, then
/// re-checks its queue. The two seq_cst fences are totally ordered: if ring()'s comes first, the re-check sees the
/// producer's work and the consumer does not wait; if the consumer's comes first, ring() sees sleeping_ and wakes it,
/// either before futex_wait (the word moved, so the kernel returns at once) or during it. The 1 ms cap is a backstop.
class Doorbell {
 public:
  /// Producer, after publishing the work the consumer's predicate checks.
  void ring() {
    word_.fetch_add(1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_relaxed)) futex_wake(&word_);
  }

  /// Consumer: sleeps until the next ring(), unless `ready()` already holds after the fence.
  template <class Ready>
  void sleep_unless(Ready ready) {
    const uint32_t seen = word_.load(std::memory_order_acquire);
    sleeping_.store(true, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!ready()) futex_wait(&word_, seen, 1'000'000);
    sleeping_.store(false, std::memory_order_relaxed);
  }

  /// Moves on every ring(): a consumer busy elsewhere can watch it for new work.
  const std::atomic<uint32_t>& word() const {
    return word_;
  }

 private:
  std::atomic<uint32_t> word_{0};
  std::atomic<bool> sleeping_{false};
};

}  // namespace sglang::expert_stream
