// Lock-free single-producer single-consumer ring, a fixed-capacity deque for one thread, and futex wait/wake (plan
// 2026-09-29-hotpath-zero-overhead Tasks 12-14). Nothing here allocates after construction or takes a lock.
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

// One producer thread, one consumer thread. Each index lives on its owner's cache line beside that owner's cached
// copy of the other index, so a push or pop that finds room touches no line the other side writes.
template <class T, size_t N>
class SpscRing {
  static_assert(N >= 2 && (N & (N - 1)) == 0, "SpscRing capacity is a power of two");

 public:
  bool push(const T& value) {  // producer
    const uint64_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_seen_ == N) {
      tail_seen_ = tail_.load(std::memory_order_acquire);
      if (head - tail_seen_ == N) return false;
    }
    slots_[head & (N - 1)] = value;
    head_.store(head + 1, std::memory_order_release);
    return true;
  }
  bool pop(T* out) {  // consumer
    const uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_seen_) {
      head_seen_ = head_.load(std::memory_order_acquire);
      if (tail == head_seen_) return false;
    }
    *out = slots_[tail & (N - 1)];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }
  // Consumer: the oldest item without removing it (pop() does), or null when empty. The pointer is valid until the
  // consumer's next pop: the producer never writes a slot the consumer has not released.
  const T* front() {
    const uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_seen_) {
      head_seen_ = head_.load(std::memory_order_acquire);
      if (tail == head_seen_) return nullptr;
    }
    return &slots_[tail & (N - 1)];
  }
  bool empty() const {  // either side; exact only on the consumer
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

 private:
  alignas(64) std::atomic<uint64_t> head_{0};
  uint64_t tail_seen_ = 0;  // producer's copy of tail_
  alignas(64) std::atomic<uint64_t> tail_{0};
  uint64_t head_seen_ = 0;  // consumer's copy of head_
  alignas(64) std::array<T, N> slots_{};
};

// A circular FIFO for one thread (the copy thread's in-flight, held and acking jobs). push_back returns false when
// full; the caller decides what that means (the copy engine fails stop).
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

// Sleeps while *word == expected, for at most timeout_ns; returns at once if *word already differs (the kernel
// compares under its own lock, so a wake that changed the word before the call is never lost).
inline void futex_wait(std::atomic<uint32_t>* word, uint32_t expected, int64_t timeout_ns) {
  timespec timeout{static_cast<time_t>(timeout_ns / 1000000000), static_cast<long>(timeout_ns % 1000000000)};
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAIT_PRIVATE, expected, &timeout, nullptr, 0);
}

inline void futex_wake(std::atomic<uint32_t>* word) {
  syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

}  // namespace sglang::expert_stream
