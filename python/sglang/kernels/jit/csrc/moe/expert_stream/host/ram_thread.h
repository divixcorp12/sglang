// RamThread: the per-tier service thread.
#pragma once

#include "ram_tier.h"

namespace sglang {
namespace expert_stream {

// Pumps one RamTier on its own thread (plan D19): demands first, then advisories; spins
// with _mm_pause() for spin_ns after the last request, else sleeps 50 us between polls.
// pause() is a handshake: it asks every advisory in flight to give up at its next row,
// skips advisories posted so far (resume() skips those posted during the pause), and returns once the loop has
// acknowledged the pause between two requests. While paused the loop takes no request, so an eager caller owns the
// slots until resume(). The watchdog (plan D15), on its own thread so a stuck read cannot silence it, aborts the
// process when the fatal word stays raised for fatal_wait without stop() (the process did not fail stop), or when one
// demand or advisory stays in service for fatal_wait (a hung read). It outlives the service thread's
// join in stop(), so a stop during a hung read, demand or advisory, still ends in its abort.
// pause()/resume() are not reentrant: their one owner is the slot table's depth counter
// (Task 14), which calls pause at depth 0->1 and resume at 1->0.
template <class Tier>
class RamThread {
 public:
  using Build = typename Tier::Build;

  RamThread(std::shared_ptr<Tier> tier, int cpu_core, int64_t fatal_wait_ns, int64_t spin_ns)
      : tier_(std::move(tier)),
        page_(tier_->page()),
        cpu_core_(cpu_core),
        fatal_wait_ns_(fatal_wait_ns),
        spin_ns_(spin_ns) {}

  ~RamThread() {
    stop();
  }

  // Throws when the thread cannot be pinned to cpu_core (it is then joined, never left floating).
  void start() {
    tier_->set_threaded(true);
    thread_ = std::thread([this] { run(); });
    while (pin_error_.load() == kPinPending)
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    if (const int error = pin_error_.load()) {
      stop_.store(true);
      thread_.join();
      tier_->set_threaded(false);
      throw std::runtime_error(
          error_prefix<typename Tier::Layout>() + "could not pin the service thread to core " +
          std::to_string(cpu_core_) + ": " + std::strerror(error));
    }
    watchdog_ = std::thread([this] { watch(); });
  }

  // The watchdog is stopped only after the service thread has joined: a join that blocks
  // on a hung read is then aborted by its stuck rule instead of hanging the process.
  void stop() {
    stop_.store(true);
    tier_->request_stop(true);  // an advisory in flight gives up at its next row
    if (thread_.joinable()) thread_.join();
    watch_stop_.store(true);
    if (watchdog_.joinable()) watchdog_.join();
    tier_->request_stop(false);
    tier_->set_threaded(false);
  }

  // 1 paused, 0 timed out, 2 refused: a graph lane still holds a lease after one retirement pass (the caller must
  // have synchronized the stream, so the device's acknowledgements are visible), and the slots are not the caller's.
  int pause(int64_t timeout_ns) {
    tier_->request_pause(true);
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(true);
    const int64_t deadline = now_ns() + timeout_ns;
    while (!paused_.load()) {
      if (now_ns() > deadline) {
        resume();
        return 0;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    // The caller synchronized the stream, so every copy wait has seen its CopyDone; the copy thread releases just
    // after.
    tier_->wait_copy_idle(now_ns() + timeout_ns);
    tier_->retire_leases(true);  // a settle pass: the synchronized stream left no signal still to land
    if (tier_->graph_leases_outstanding() > 0) {
      resume();
      return 2;
    }
    return 1;
  }

  // Advisories posted while paused predate the eager use: skip them too.
  void resume() {
    tier_->fill_join();  // a prefill fill uses the reader the service thread is about to use
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(false);
    tier_->request_pause(false);
  }

 private:
  void run() {
    pthread_setname_np(pthread_self(), (std::string(Tier::Layout::kName) + "-ram-miss").substr(0, 15).c_str());
    int error = 0;
    if (cpu_core_ >= 0) {
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      CPU_SET(cpu_core_, &cpus);
      error = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    }
    tier_->set_counter(kSpinCpu, error != 0 ? -error : sched_getcpu());
    pin_error_.store(error);
    if (error != 0) return;
    tier_->set_counter(kRunning, 1);
    int64_t last_active = now_ns();
    uint32_t heartbeat = 0;
    uint32_t iterations = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
      // Not every iteration: the word shares the cache line the device polls.
      if ((++iterations & 1023u) == 1u) store_release(page_ + kHeartbeat, ++heartbeat);
      if (pause_requested_.load()) {
        paused_.store(true);
        while (pause_requested_.load() && !stop_.load())
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        paused_.store(false);
        continue;
      }
      if (tier_->pump_demand() || tier_->pump_prefetch() || tier_->pump_advice()) {
        last_active = now_ns();
        iterations = 0;  // one heartbeat per request served
        continue;
      }
      if (now_ns() - last_active < spin_ns_) {
        _mm_pause();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
    tier_->set_counter(kRunning, 0);
  }

  void watch() {
    int64_t fatal_since = 0;
    bool reported = false;
    while (!watch_stop_.load()) {
      const uint32_t fatal = load_acquire(page_ + kFatal);
      const int64_t now = now_ns();
      if (fatal != 0) {
        if (!reported) {
          reported = true;
          std::fprintf(
              stderr,
              "ERROR %srequest %u timed out or failed; the process must stop\n",
              error_prefix<typename Tier::Layout>().c_str(),
              fatal);
          std::fflush(stderr);
        }
        if (fatal_since == 0) fatal_since = now;
      }
      const int64_t busy_since = tier_->busy_since();
      // Once stop() began, the process is failing stop: only a hung read can still abort.
      const bool fatal_held = !stop_.load() && fatal_since != 0 && now - fatal_since > fatal_wait_ns_;
      const bool stuck = busy_since != 0 && now - busy_since > fatal_wait_ns_;
      if (fatal_held || stuck) {
        std::fprintf(
            stderr,
            "ERROR %s%s for %.1f s (fatal %u, busy %u); aborting instead of hanging decode\n",
            error_prefix<typename Tier::Layout>().c_str(),
            stuck ? "a request stayed in service" : "the fatal word stayed raised without the process stopping",
            static_cast<double>(fatal_wait_ns_) / 1e9,
            fatal,
            load_acquire(page_ + kBusySeq));
        std::fflush(stderr);
        prctl(PR_SET_DUMPABLE, 0);
        std::abort();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  std::shared_ptr<Tier> tier_;
  uint8_t* page_;
  int cpu_core_;
  int64_t fatal_wait_ns_;
  int64_t spin_ns_;
  std::thread thread_;
  std::thread watchdog_;
  static constexpr int kPinPending = -1;
  std::atomic<bool> stop_{false};
  std::atomic<bool> watch_stop_{false};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> paused_{false};
  std::atomic<int> pin_error_{kPinPending};  // 0 pinned (or not asked), else the errno
};

}  // namespace expert_stream
}  // namespace sglang
