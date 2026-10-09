// The RAM-miss service threads of one RamTier, and its watchdog.
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Parties".
#pragma once

#include <future>

#include "ram_tier.h"

namespace sglang {
namespace expert_stream {

/// pause()'s result; the values cross the FFI as ints.
enum PauseResult : int {
  kPauseTimedOut = 0,  // or a racing stop()
  kPaused = 1,
  kPauseRefused = 2,  // the copy thread still has a job: the slots are not the caller's to touch
};

/// Pumps one RamTier with one service thread per NUMA group, each serving its group's posted demand records through
/// RamTier::pump_demand(g), plus a watchdog that aborts the process when a request or a copy wait hangs.
///
/// Idle: after the last request a thread spins for spin_ns, then sleeps 50 us between polls; a parked thread sleeps
/// 20 us between checks. A negative spin_ns never sleeps. The spin is a poll budget calibrated once in start()
/// (idle_budget), so a serving thread reads no clock. It spins with _mm_pause(), or, with busy_poll, on a core of its
/// own (checked by start_thread) with no PAUSE.
///
/// Ownership: the tier has one owner at a time, its service threads or a caller that paused them all; pause() and
/// resume() are the handoff. Both take the tier's caller_mutex(), which the service threads never take.
///
/// Teardown: stop() joins the service threads first and the watchdog second, so a join blocked on a hung read is
/// aborted by the watchdog instead of hanging the process.
template <class Tier>
class RamThread {
 public:
  using Build = typename Tier::Build;

  /// `cpu_cores` has one entry per group; a negative entry leaves that thread unpinned.
  RamThread(
      std::shared_ptr<Tier> tier, std::vector<int> cpu_cores, int64_t fatal_wait_ns, int64_t spin_ns, bool busy_poll)
      : tier_(std::move(tier)),
        cpu_cores_(std::move(cpu_cores)),
        fatal_wait_ns_(fatal_wait_ns),
        spin_ns_(spin_ns),
        busy_poll_(busy_poll),
        threads_(cpu_cores_.size()),
        pinned_(cpu_cores_.size()),
        parked_epoch_(std::make_unique<std::atomic<uint64_t>[]>(cpu_cores_.size())) {
    for (size_t g = 0; g < cpu_cores_.size(); ++g)
      parked_epoch_[g].store(0);
  }

  ~RamThread() {
    stop();
  }

  /// Starts the service threads and the watchdog, under caller_mutex().
  ///
  /// Throws when a thread cannot be pinned to its core, after joining every thread. Refuses a tier whose prefill fill
  /// (begun in pump mode) still owes its epilogue, since the service would share the reader with the fill thread; it
  /// refuses rather than joins because the FFI's start_thread holds the registry lock, and a join there would stall
  /// every handle's calls behind a slow fill read. The caller must call fill_end() first.
  void start() {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    if (tier_->fill_owed()) {
      throw std::runtime_error(
          error_prefix<typename Tier::Layout>() +
          "start_thread with a prefill fill running (or not yet ended): call fill_end() first");
    }
    spin_iters_ = idle_budget(spin_ns_);
    tier_->set_parked(false);
    tier_->set_threaded(true);
    std::vector<std::future<int>> pins;
    for (std::promise<int>& pinned : pinned_)
      pins.push_back(pinned.get_future());
    for (size_t g = 0; g < threads_.size(); ++g)
      threads_[g] = std::thread([this, g] { run(static_cast<int>(g)); });
    int failed = -1;
    int error = 0;
    for (size_t g = 0; g < pins.size(); ++g) {
      const int pin_error = pins[g].get();
      if (failed < 0 && pin_error != 0) {
        failed = static_cast<int>(g);
        error = pin_error;
      }
    }
    if (failed >= 0) {
      stop_.store(true);
      for (std::thread& thread : threads_)
        thread.join();
      tier_->set_threaded(false);
      throw std::runtime_error(
          error_prefix<typename Tier::Layout>() + "could not pin the service thread" +
          (Wire::kNodes > 1 ? " of group " + std::to_string(failed) : std::string()) + " to core " +
          std::to_string(cpu_cores_[failed]) + ": " + std::strerror(error));
    }
    try {
      tier_->start_spec();
    } catch (...) {
      stop_.store(true);
      for (std::thread& thread : threads_)
        thread.join();
      tier_->set_threaded(false);
      throw;
    }
    watchdog_ = std::thread([this] { watch(); });
  }

  /// Stops the service threads, then the watchdog; idempotent.
  ///
  /// The joins order every service write before the release of threaded_, after which a caller owns the tier. A
  /// prefill fill a pausing caller left running is joined by stop_thread's final settle (RamTier::final_settle), under
  /// caller_mutex(); the fill thread writes no tier state, so that join is the only edge it needs.
  void stop() {
    // First: a service thread promoting a pool row waits for its read, which must finish while the services run.
    tier_->stop_spec();
    stop_.store(true);
    for (std::thread& thread : threads_)
      if (thread.joinable()) thread.join();
    watch_stop_.store(true);
    if (watchdog_.joinable()) watchdog_.join();
    tier_->set_threaded(false);
  }

  /// Asks the service threads to park and, once every one has, takes ownership of the tier until resume().
  ///
  /// The caller must have synchronized the stream first, so every copy wait has seen its CopyDone and the copy thread
  /// has nothing left to do. Not reentrant: the slot table's depth counter owns the pairing (pause at depth 0->1,
  /// resume at 1->0).
  ///
  /// Service -> caller edge: each service loop serves every posted record, then stores its parked_epoch_ (release);
  /// pause() loads them all (acquire), then sets the tier's parked_ (release), which later callers acquire in
  /// caller_owns(). Each pause has its own odd epoch, so a pause right after a resume cannot take the previous pause's
  /// acknowledgement for its own.
  ///
  /// The speculative threads are quiesced first, without a bound and ignoring `timeout_ns`: a hung speculative read is
  /// the watchdog's to abort.
  PauseResult pause(int64_t timeout_ns) {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    // The paused caller and a prefill fill use the readers the speculative reads take turns on.
    tier_->quiesce_spec();
    const uint64_t epoch = (pause_epoch_.load(std::memory_order_relaxed) | 1u) + 2u;
    pause_epoch_.store(epoch);
    const int64_t deadline = now_ns() + timeout_ns;
    while (!all_parked(epoch)) {
      // A racing stop() will not park for this epoch: stop_ is its first store and threaded_ its last.
      if (stop_.load(std::memory_order_acquire) || !tier_->threaded()) {
        resume_locked();
        return kPauseTimedOut;
      }
      // On its own line: test_exl3_ram_miss_stage_trace_causal counts clock reads by line.
      if (now_ns() > deadline) {
        resume_locked();
        return kPauseTimedOut;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    tier_->set_parked(true);
    if (!tier_->wait_copy_idle_owned(now_ns() + timeout_ns)) {
      resume_locked();
      return kPauseRefused;
    }
    return kPaused;
  }

  /// Hands the tier back to the service threads. Safe after a timed-out or refused pause (which already resumed).
  void resume() {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    resume_locked();
  }

 private:
  bool all_parked(uint64_t epoch) const {
    for (size_t g = 0; g < threads_.size(); ++g)
      if (parked_epoch_[g].load(std::memory_order_acquire) != epoch) return false;
    return true;
  }

  /// The owner hands the tier back, under caller_mutex().
  ///
  /// A prefill fill uses the reader the service is about to use, so it is joined and its epilogue run here, on the
  /// owner. parked_ is cleared next, so a caller that then takes caller_mutex() sees the service as the owner and is
  /// refused. Last, the release of pause_epoch_ (acquired by the parked loop) orders the owner's writes before the
  /// service's next request. A timed-out pause never set parked_.
  void resume_locked() {
    tier_->fill_join();
    tier_->set_parked(false);
    tier_->resume_spec();
    const uint64_t epoch = pause_epoch_.load(std::memory_order_relaxed);
    if (epoch & 1u) pause_epoch_.store(epoch + 1u, std::memory_order_release);
  }

  /// Names group g's thread and pins it to its core; returns 0 or the errno.
  int pin(int g) {
    std::string name = std::string(Tier::Layout::kName) + "-ram-miss";
    if (Wire::kNodes > 1) name += std::to_string(g);
    pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
    int error = 0;
    if (cpu_cores_[g] >= 0) {
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      CPU_SET(cpu_cores_[g], &cpus);
      error = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    }
    tier_->set_counter(g, kSpinCpu, error != 0 ? -error : sched_getcpu());
    return error;
  }

  /// Group g's service thread: pumps its demand records until stopped, parking whenever a pause is requested.
  void run(int g) {
    const int error = pin(g);
    pinned_[g].set_value(error);
    if (error != 0) return;
    tier_->set_counter(g, kRunning, 1);
    uint64_t idle = 0;  // empty polls since the last request, counted against spin_iters_
    while (!stop_.load(std::memory_order_relaxed)) {
      const uint64_t epoch = pause_epoch_.load(std::memory_order_acquire);
      if (epoch & 1u) {
        if (!park(g, epoch)) break;
        continue;
      }
      if (tier_->pump_demand(g)) {
        idle = 0;
      } else if (++idle < spin_iters_) {
        if (!busy_poll_) _mm_pause();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
    tier_->set_counter(g, kRunning, 0);
  }

  /// Serves every record posted before the pause, acknowledges `epoch` and waits for the resume. False when stopped
  /// while parked: the caller then owns the tier, so the thread must touch nothing.
  ///
  /// The records are served first because an all-HIT_SM chain never waits on the service, so its record can still be
  /// unread; checked after the caller moved its expert, it would fail-stop a correct device. The caller synchronized
  /// the stream, so demand_head is final and this ends.
  bool park(int g, uint64_t epoch) {
    while (tier_->pump_demand(g)) {
    }
    parked_epoch_[g].store(epoch, std::memory_order_release);
    while (pause_epoch_.load(std::memory_order_acquire) == epoch && !stop_.load()) {
      if (spin_ns_ < 0)
        _mm_pause();
      else
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return pause_epoch_.load(std::memory_order_acquire) != epoch;
  }

  /// What the watchdog last saw, per group and for the copy gate.
  struct WatchState {
    std::vector<uint64_t> episode;      // each group's busy episode, then each speculative one; 0: idle
    std::vector<int64_t> episode_since;  // when the watchdog first saw it
    uint32_t gate = 0;                   // the gate word last seen closed, 0: open
    int64_t gate_since = 0;
  };

  /// The watchdog: aborts the process, instead of hanging decode, when one demand or fill stays in service on a group,
  /// or one speculative read stays in flight, for fatal_wait_ns (a hung read, timed as one busy episode), or when the
  /// copy wait's gate stays closed on one value past the copy-wait timeout (the copy thread is stuck in a driver call,
  /// and the device's wait must still end). It reads the clock every 20 ms on its own thread, so a stuck read cannot
  /// silence it.
  void watch() {
    const size_t episodes = 2 * threads_.size();  // each group's service, then each group's speculative read
    WatchState seen{std::vector<uint64_t>(episodes, 0), std::vector<int64_t>(episodes, 0)};
    while (!watch_stop_.load()) {
      const int64_t now = now_ns();
      const int stuck = stuck_group(seen, now);
      if (stuck >= 0 || gate_held(seen, now)) abort_hung(stuck);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  /// A busy episode that has lasted past fatal_wait_ns, or -1: index g is group g's service, groups + g its
  /// speculative read. A speculative one is preferred: a service promoting its pool row waits on it.
  int stuck_group(WatchState& seen, int64_t now) const {
    const size_t groups = threads_.size();
    int stuck = -1;
    for (size_t i = 2 * groups; i-- > 0;) {
      const int g = static_cast<int>(i % groups);
      const uint64_t busy = i < groups ? tier_->busy_episode(g) : tier_->spec_busy_episode(g);
      if (busy != seen.episode[i]) {
        seen.episode[i] = busy;
        seen.episode_since[i] = now;
      }
      if (stuck < 0 && seen.episode[i] != 0 && now - seen.episode_since[i] > fatal_wait_ns_)
        stuck = static_cast<int>(i);
    }
    return stuck;
  }

  /// True once the copy gate has stayed closed on one value past the copy-wait timeout.
  bool gate_held(WatchState& seen, int64_t now) const {
    const uint32_t word = tier_->copy_gate();
    const uint32_t closed = (word & 0x80000000u) != 0 ? word : 0;
    if (closed != seen.gate) {
      seen.gate = closed;
      seen.gate_since = now;
    }
    return seen.gate != 0 && now - seen.gate_since > tier_->copy_wait_timeout_ns();
  }

  /// Reports the hang (`stuck` from stuck_group, or the copy gate when -1) and aborts without a core dump.
  [[noreturn]] void abort_hung(int stuck) const {
    const int groups = static_cast<int>(threads_.size());
    const bool request = stuck >= 0;
    const bool speculative = stuck >= groups;
    const std::string why = request ? "" : tier_->copy_stall();
    const std::string group = request && groups > 1 ? "group " + std::to_string(stuck % groups) + ": " : "";
    std::fprintf(
        stderr,
        "FATAL %s%s%s for %.1f s%s; aborting instead of hanging decode\n",
        error_prefix<typename Tier::Layout>().c_str(),
        group.c_str(),
        !request ? "a copy wait held the decode stream"
                 : (speculative ? "a speculative read stayed in service" : "a request stayed in service"),
        static_cast<double>(request ? fatal_wait_ns_ : tier_->copy_wait_timeout_ns()) / 1e9,
        why.c_str());
    std::fflush(stderr);
    prctl(PR_SET_DUMPABLE, 0);
    std::abort();
  }

  std::shared_ptr<Tier> tier_;
  std::vector<int> cpu_cores_;  // one per group
  int64_t fatal_wait_ns_;
  int64_t spin_ns_;  // < 0: never sleep
  bool busy_poll_;
  uint64_t spin_iters_ = 1;
  std::vector<std::thread> threads_;      // one per group
  std::vector<std::promise<int>> pinned_;  // per group: 0 pinned (or not asked), else the errno
  std::thread watchdog_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> watch_stop_{false};
  // Odd while a pause is requested, a new value per pause; written under caller_mutex().
  std::atomic<uint64_t> pause_epoch_{0};
  // Per group, the epoch its thread parked for; written by that group's thread only.
  std::unique_ptr<std::atomic<uint64_t>[]> parked_epoch_;
};

}  // namespace expert_stream
}  // namespace sglang
