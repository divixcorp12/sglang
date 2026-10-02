// The RAM-miss service thread of one RamTier, and its watchdog.
//
// RamThread owns two threads: the service thread, which serves posted demand records through RamTier::pump_demand, and
// a watchdog that aborts the process when a request or a copy wait hangs. It also implements the owner handoff,
// pause() and resume(), by which a Python caller takes the tier from the service thread and gives it back.
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Parties".
#pragma once

#include "ram_tier.h"

namespace sglang {
namespace expert_stream {

// Pumps one RamTier on its own thread.
//
// Idle policy: after the last request the thread spins with _mm_pause() for spin_ns, then sleeps 50 us between polls.
// With busy_poll it runs on a core of its own (checked by start_thread), spins with no PAUSE and never sleeps. The spin
// is an idle-poll budget calibrated once in start() (idle_budget), so the thread reads no clock while it serves.
//
// Ownership: the tier has one owner at a time, this thread or a caller that paused it. pause() and resume() are the
// handoff; the memory-ordering edges are documented at each. They take the tier's caller_mutex(), which the service
// thread never takes.
//
// Teardown: stop() joins the service thread first and the watchdog second, so a join blocked on a hung read is aborted
// by the watchdog's stuck rule instead of hanging the process.
template <class Tier>
class RamThread {
 public:
  using Build = typename Tier::Build;

  RamThread(std::shared_ptr<Tier> tier, int cpu_core, int64_t fatal_wait_ns, int64_t spin_ns, bool busy_poll)
      : tier_(std::move(tier)),
        cpu_core_(cpu_core),
        fatal_wait_ns_(fatal_wait_ns),
        spin_ns_(spin_ns),
        busy_poll_(busy_poll) {}

  ~RamThread() {
    stop();
  }

  // Starts the service thread and the watchdog.
  //
  // Throws when the thread cannot be pinned to cpu_core (it is then joined, never left floating), and refuses a tier
  // whose prefill fill (begun in pump mode) still owes its epilogue: the service would then share the reader with the
  // fill thread, and the epilogue would run off the owner. It refuses rather than joins because the FFI's start_thread
  // holds the registry lock, and a join there would stall every handle's calls behind a slow or hung fill read; the
  // caller must call fill_end() first.
  //
  // Runs under caller_mutex(), which orders the set_parked/set_threaded writes against every Python caller. The
  // service thread never takes it, so holding it across the pin handshake cannot deadlock.
  void start() {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    if (tier_->fill_owed()) {
      throw std::runtime_error(
          error_prefix<typename Tier::Layout>() +
          "start_thread with a prefill fill running (or not yet ended): call fill_end() first");
    }
    spin_iters_ = idle_budget(spin_ns_);  // calibrated here: the service thread never reads the clock to pace
    tier_->set_parked(false);
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

  // Stops the service thread, then the watchdog (see the class comment); idempotent.
  //
  // The service join orders every service write before stop()'s release of threaded_, after which a caller owns the
  // tier. A prefill fill a pausing caller left running is joined by stop_thread's final settle
  // (RamTier::final_settle), under caller_mutex(); the fill thread writes no tier state, so that join is the only edge
  // it needs.
  void stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    watch_stop_.store(true);
    if (watchdog_.joinable()) watchdog_.join();
    tier_->set_threaded(false);  // release: after the join, a caller owns the tier
  }

  // Asks the service thread to park and, once it has, takes ownership of the tier until resume().
  //
  // Returns 1 when paused, 0 on timeout (or a racing stop()), and 2 when refused: the copy thread still has a job after
  // the wait, so the slots are not the caller's to touch. The caller must have synchronized the stream first, so every
  // copy wait has seen its CopyDone and the copy thread has nothing left to do.
  //
  // Edge service -> caller: the loop serves every posted record, then stores parked_epoch_ (release); pause() loads it
  // (acquire), then sets the tier's parked_ (release), which later Python callers acquire in caller_owns(). Each pause
  // has its own epoch (odd while requested), so a pause right after a resume cannot take the previous pause's
  // acknowledgement for its own.
  //
  // Not reentrant: the one owner of the pairing is the slot table's depth counter (pause at depth 0->1, resume at
  // 1->0).
  int pause(int64_t timeout_ns) {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    const uint64_t epoch = (pause_epoch_.load(std::memory_order_relaxed) | 1u) + 2u;  // a new odd epoch
    pause_epoch_.store(epoch);
    const int64_t deadline = now_ns() + timeout_ns;
    while (parked_epoch_.load(std::memory_order_acquire) != epoch) {
      // A stop() racing this pause: the service is leaving and will not park for this epoch, so do not wait out the
      // whole timeout. stop_ is its first store and threaded_ its last; either says so. The deadline test stays on its
      // own line: test_exl3_ram_miss_stage_trace_causal counts clock reads by line.
      if (stop_.load(std::memory_order_acquire) || !tier_->threaded()) {
        resume_locked();
        return 0;
      }
      if (now_ns() > deadline) {
        resume_locked();
        return 0;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    tier_->set_parked(true);  // the service parked for this epoch: this caller owns the tier until resume
    if (!tier_->wait_copy_idle_owned(now_ns() + timeout_ns)) {
      resume_locked();
      return 2;
    }
    return 1;
  }

  // Hands the tier back to the service thread. Safe after a timed-out or refused pause (which already resumed).
  void resume() {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    resume_locked();
  }

 private:
  // The owner hands the tier back, under caller_mutex().
  //
  // The owner's writes happen-before the service's next request through the release of pause_epoch_, which the parked
  // loop acquires. parked_ is cleared first, so a caller that then takes caller_mutex() sees the service as the owner
  // and is refused instead of touching the tier. A timed-out pause never set parked_.
  void resume_locked() {
    // A prefill fill uses the reader the service thread is about to use: join it and run its epilogue here, on the
    // owner, before the release below.
    tier_->fill_join();
    tier_->set_parked(false);
    const uint64_t epoch = pause_epoch_.load(std::memory_order_relaxed);
    if (epoch & 1u) pause_epoch_.store(epoch + 1u, std::memory_order_release);
  }

  // The service thread body: pin, then pump demand records until stopped, parking whenever a pause is requested.
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
    uint64_t idle = 0;  // empty polls since the last request: the spin budget counts these, not elapsed time
    while (!stop_.load(std::memory_order_relaxed)) {
      const uint64_t epoch = pause_epoch_.load(std::memory_order_acquire);
      if (epoch & 1u) {
        // Serve every record posted before the pause first: an all-HIT_SM chain never waits on the service, so its
        // record can still be unread, and checked after the caller moved its expert it would fail-stop a correct
        // device. The caller synchronized the stream, so demand_head is final and this ends.
        while (tier_->pump_demand()) {
        }
        parked_epoch_.store(epoch, std::memory_order_release);  // the handoff: the pausing caller owns the tier
        while (pause_epoch_.load(std::memory_order_acquire) == epoch && !stop_.load())
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        if (pause_epoch_.load(std::memory_order_acquire) == epoch) {
          break;  // stopped while parked: the caller owns the tier, touch nothing
        }
        continue;
      }
      if (tier_->pump_demand()) {
        idle = 0;
        continue;
      }
      if (busy_poll_) continue;  // a core of its own: no PAUSE, no sleep (keeps detection latency minimal)
      if (++idle < spin_iters_) {
        _mm_pause();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));  // idle: stop burning the core
      }
    }
    tier_->set_counter(kRunning, 0);
  }

  // The watchdog body: aborts the process, instead of hanging decode, in two cases.
  //   - One demand or fill stays in service for fatal_wait_ns (a hung read), timed as one busy episode
  //     (RamTier::busy_episode).
  //   - The copy wait's gate stays closed on one value for longer than the copy-wait timeout: the copy thread is stuck
  //     in a driver call, and the device's wait must still end.
  // It runs on its own thread so a stuck read cannot silence it. The clock is read here, every 20 ms, never by the
  // service, so detection is at most 20 ms late against a 30 s deadline.
  void watch() {
    uint64_t episode = 0;       // the busy episode last seen, 0: idle
    int64_t episode_since = 0;  // when the watchdog first saw it
    uint32_t gate = 0;          // the gate word last seen closed, 0: open
    int64_t gate_since = 0;
    while (!watch_stop_.load()) {
      const int64_t now = now_ns();
      const uint64_t busy = tier_->busy_episode();
      if (busy != episode) {
        episode = busy;
        episode_since = now;
      }
      const uint32_t word = tier_->copy_gate();
      const uint32_t closed = (word & 0x80000000u) != 0 ? word : 0;
      if (closed != gate) {
        gate = closed;
        gate_since = now;
      }
      const bool stuck = episode != 0 && now - episode_since > fatal_wait_ns_;
      const bool held = gate != 0 && now - gate_since > tier_->copy_wait_timeout_ns();
      if (stuck || held) {
        std::fprintf(
            stderr,
            "FATAL %s%s for %.1f s; aborting instead of hanging decode\n",
            error_prefix<typename Tier::Layout>().c_str(),
            stuck ? "a request stayed in service" : "a copy wait held the decode stream",
            static_cast<double>(stuck ? fatal_wait_ns_ : tier_->copy_wait_timeout_ns()) / 1e9);
        std::fflush(stderr);
        prctl(PR_SET_DUMPABLE, 0);
        std::abort();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  std::shared_ptr<Tier> tier_;
  int cpu_core_;
  int64_t fatal_wait_ns_;
  int64_t spin_ns_;
  bool busy_poll_;
  uint64_t spin_iters_ = 1;  // idle polls before the idle sleep: idle_budget(spin_ns_), set in start()
  std::thread thread_;
  std::thread watchdog_;
  static constexpr int kPinPending = -1;
  std::atomic<bool> stop_{false};
  std::atomic<bool> watch_stop_{false};
  // The pause handshake. pause_epoch_ is odd while a pause is requested (a new value per pause); the loop stores into
  // parked_epoch_ the epoch it parked for. pause_epoch_ is written under caller_mutex(), parked_epoch_ by the service
  // thread only.
  std::atomic<uint64_t> pause_epoch_{0};
  std::atomic<uint64_t> parked_epoch_{0};
  std::atomic<int> pin_error_{kPinPending};  // 0 pinned (or not asked), else the errno
};

}  // namespace expert_stream
}  // namespace sglang
