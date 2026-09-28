// One CPU publisher per mapped mailbox. The GPU's stream waits on ready before it can reuse the mailbox.
#pragma once

#include "../lease_layout.h"
#include "../wait_layout.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace sglang::expert_stream {

class WaitCompletion {
 public:
  WaitCompletion(uint8_t* page, uint8_t* mailbox, const uint8_t* lease)
      : page_(page), mailbox_(mailbox), lease_(lease), thread_([this] { run(); }) {}

  WaitCompletion(const WaitCompletion&) = delete;
  WaitCompletion& operator=(const WaitCompletion&) = delete;

  ~WaitCompletion() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
  }

  const uint8_t* mailbox() const {
    return mailbox_;
  }

  // Keep running until the owner has drained all queued preparation kernels. Those kernels can publish a new
  // pending token after cancellation, and still need this publisher to release their stream waits.
  void cancel() {
    cancelled_.store(true, std::memory_order_release);
  }

 private:
  static uint32_t load32(const uint8_t* address) {
    return __atomic_load_n(reinterpret_cast<const uint32_t*>(address), __ATOMIC_ACQUIRE);
  }

  static uint64_t load64(const uint8_t* address) {
    return __atomic_load_n(reinterpret_cast<const uint64_t*>(address), __ATOMIC_ACQUIRE);
  }

  void publish(uint64_t generation, uint64_t tag) {
    using namespace wire;
    __atomic_store_n(
        reinterpret_cast<uint64_t*>(mailbox_ + kWaitCompletionToken), (tag << 56) | generation, __ATOMIC_RELEASE);
    // LAST publication access. Only this wakes the GPU, which can then prepare the next generation. A second
    // publisher, or any delayed write for this generation after ready, would race that reuse.
    __atomic_store_n(reinterpret_cast<uint32_t*>(mailbox_ + kWaitCompletionReady), 1u, __ATOMIC_RELEASE);
  }

  void run() {
    using namespace wire;
    using Clock = std::chrono::steady_clock;
    constexpr uint64_t generation_mask = (uint64_t{1} << 56) - 1;
    uint64_t observed = 0;
    uint64_t duration = 0;
    Clock::time_point start;
    for (;;) {
      const bool stopping = stop_.load(std::memory_order_acquire);
      const uint64_t token = load64(mailbox_ + kWaitCompletionToken);
      const uint64_t generation = token & generation_mask;
      if ((token >> 56) == kWaitTagPending && generation != 0) {
        if (stopping || cancelled_.load(std::memory_order_acquire) || load32(page_ + kFatal) != 0 ||
            (lease_ != nullptr && load32(lease_ + kLeaseHeaderShutdown) != 0)) {
          publish(generation, kWaitTagAborted);
        } else {
          if (observed != token) {
            observed = token;
            // Acquiring the pending token orders this load after the preparation kernel's duration store.
            // GPU globaltimer and CPU steady_clock are different clocks: only the duration crosses the wire.
            duration = load64(mailbox_ + kWaitCompletionTimeoutNs);
            start = Clock::now();
          }
          const uint32_t seq = static_cast<uint32_t>(generation);
          const uint32_t done = load32(page_ + kDemandDone);
          if (static_cast<int32_t>(done - seq) >= 0) {
            // The service release-publishes demand_done after its payload and record status. This acquire and
            // the release token relay that publication to the GPU validator; ready is only a scheduling flag.
            publish(generation, kWaitTagReady);
          } else if (
              static_cast<uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()) >= duration) {
            publish(generation, kWaitTagTimeout);
          }
        }
      } else {
        observed = 0;
      }
      if (stopping) return;
      std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
  }

  uint8_t* const page_;
  uint8_t* const mailbox_;
  const uint8_t* const lease_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> cancelled_{false};
  std::thread thread_;
};

// Close keeps the registry locked until the publisher joined: another open cannot attach a second publisher to
// the same mailbox during shutdown. The owner retains its buffers, cancels, drains queued GPU work, then closes.
class WaitCompletionRegistry {
 public:
  static int64_t open(uint8_t* page, uint8_t* mailbox, const uint8_t* lease) {
    using namespace wire;
    if (page == nullptr || reinterpret_cast<uintptr_t>(page) % alignof(uint32_t) != 0)
      throw std::invalid_argument("wait completion: page must be nonnull and 4-byte aligned");
    if (mailbox == nullptr || reinterpret_cast<uintptr_t>(mailbox) % alignof(uint64_t) != 0)
      throw std::invalid_argument("wait completion: mailbox must be nonnull and 8-byte aligned");
    if (reinterpret_cast<uintptr_t>(lease) % alignof(uint32_t) != 0)
      throw std::invalid_argument("wait completion: lease address must be 4-byte aligned");
    std::lock_guard<std::mutex> lock(mutex());
    const uintptr_t address = reinterpret_cast<uintptr_t>(mailbox);
    for (const auto& [handle, bridge] : entries()) {
      const uintptr_t existing = reinterpret_cast<uintptr_t>(bridge->mailbox());
      if ((address >= existing ? address - existing : existing - address) < kWaitCompletionBytes)
        throw std::runtime_error("wait completion: mailbox already has a publisher");
    }
    static int64_t next_handle = 1;
    if (next_handle == std::numeric_limits<int64_t>::max())
      throw std::runtime_error("wait completion: handle space exhausted");
    const int64_t handle = next_handle++;
    entries().emplace(handle, std::make_unique<WaitCompletion>(page, mailbox, lease));
    return handle;
  }

  static void close(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex());
    const auto found = entries().find(handle);
    if (found == entries().end()) throw std::invalid_argument("wait completion: unknown handle");
    entries().erase(found);
  }

  static void cancel(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex());
    const auto found = entries().find(handle);
    if (found == entries().end()) throw std::invalid_argument("wait completion: unknown handle");
    found->second->cancel();
  }

 private:
  static std::mutex& mutex() {
    static std::mutex value;
    return value;
  }

  static std::unordered_map<int64_t, std::unique_ptr<WaitCompletion>>& entries() {
    static std::unordered_map<int64_t, std::unique_ptr<WaitCompletion>> value;
    return value;
  }
};

}  // namespace sglang::expert_stream
