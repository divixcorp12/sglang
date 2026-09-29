// RamThread: the per-tier service thread.
#pragma once

#include "ram_tier.h"

namespace sglang {
namespace expert_stream {

// Pumps one RamTier on its own thread (plan D19): demands first, then advisories; spins
// with _mm_pause() for spin_ns after the last request, else sleeps 50 us between polls. The spin is an idle-poll
// budget calibrated once in start() (idle_budget), so the thread reads no clock while it serves (spec M8).
// pause() is a handshake: it asks every advisory in flight to give up at its next row,
// skips advisories posted so far (resume() skips those posted during the pause), and returns once the loop has
// acknowledged the pause between two requests. While paused the loop takes no request, so an eager caller owns the
// slots until resume().
//
// Ownership (plan 2026-09-29-hotpath-zero-overhead Task 13, the single-owner rule): the tier has one owner at a time,
// and pause()/resume() are the handoff. The happens-before edges:
//   - service -> caller: the loop drains the command ring, then stores parked_epoch_ (release); pause() loads it
//     (acquire), then sets the tier's parked_ (release), which later Python callers acquire in caller_owns().
//   - caller -> service: resume() clears parked_, then stores pause_epoch_ (release); the parked loop loads it
//     (acquire) before it touches the tier again.
//   - stop(): the join orders every service write before stop()'s release of threaded_ (plan F14). A prefill fill a
//     pausing caller left running is joined by stop_thread's final settle (RamTier::final_settle), under
//     caller_mutex(), before it drains and settles: the fill thread writes no tier state, so that join is the only
//     edge it needs.
// Each pause has its own epoch (odd while requested), so a pause that follows a resume at once can never take the
// previous pause's acknowledgement for its own while the service is already running again. pause() and resume() take
// the tier's caller_mutex(); the service thread never does.
//
// The watchdog (plan D15), on its own thread so a stuck read cannot silence it, aborts the
// process when the fatal word stays raised for fatal_wait without stop() (the process did not fail stop), or when one
// demand, advisory or fill stays in service for fatal_wait (a hung read): it times how long one busy episode
// (RamTier::busy_episode) persists, so the clock is read on the watchdog thread only (D6). It outlives the service
// thread's join in stop(), so a stop during a hung read, demand or advisory, still ends in its abort.
// It is also the copy wait's deadline (LEASE_PROTOCOL.md 7.6): a gate armed for longer than the copy-wait timeout is
// opened as a timeout here, and every poll opens one whose CopyDone, fatal or shutdown word no other releaser acted on.
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

  // Throws when the thread cannot be pinned to cpu_core (it is then joined, never left floating), and refuses a tier
  // whose prefill fill (begun in pump mode) still owes its epilogue: the service would then share the reader with the
  // fill thread (the serialized issuers, spec 6.2/D8), and the epilogue would run off the owner. It refuses rather than
  // joins: the FFI's start_thread holds the registry lock, and a join there would stall every handle's calls behind a
  // slow or hung fill read; fill_end() first is the caller's to do. Under caller_mutex(), which also orders the
  // set_parked/set_threaded writes below against every Python caller. The service thread never takes it, so holding
  // it across the thread's start and pin handshake cannot deadlock.
  void start() {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    if (tier_->fill_owed()) {
      throw std::runtime_error(
          error_prefix<typename Tier::Layout>() +
          "start_thread with a prefill fill running (or not yet ended): call fill_end() first");
    }
    if (tier_->copy_waits_aborted()) {
      throw std::runtime_error(
          error_prefix<typename Tier::Layout>() +
          "start_thread after the service was stopped with the copy engine on: stopping raised the lease block's "
          "sticky shutdown word, so every copy wait would fail; build a new host");
    }
    spin_iters_ = idle_budget(spin_ns_);  // on the caller's thread: the service thread never reads the clock to pace
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

  // The watchdog is stopped only after the service thread has joined: a join that blocks
  // on a hung read is then aborted by its stuck rule instead of hanging the process.
  //
  // Plan F14: no caller_mutex() here. A caller waiting for a snapshot holds it, and that caller must see threaded_
  // clear to drain the queue itself; the service's last act is a final drain (run()), and the join orders it first.
  void stop() {
    stop_.store(true);
    tier_->request_stop(true);  // an advisory in flight gives up at its next row
    if (thread_.joinable()) thread_.join();
    watch_stop_.store(true);
    if (watchdog_.joinable()) watchdog_.join();
    tier_->request_stop(false);
    tier_->set_threaded(false);  // release: after the join, a waiting caller owns the tier and drains the ring
  }

  // 1 paused, 0 timed out, 2 refused: a graph lane still holds a lease after one retirement pass (the caller must
  // have synchronized the stream, so the device's acknowledgements are visible), and the slots are not the caller's.
  int pause(int64_t timeout_ns) {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    tier_->request_pause(true);
    tier_->skip_advice_posted_so_far();
    const uint64_t epoch = (pause_epoch_.load(std::memory_order_relaxed) | 1u) + 2u;  // a new odd epoch
    pause_epoch_.store(epoch);
    const int64_t deadline = now_ns() + timeout_ns;
    while (parked_epoch_.load(std::memory_order_acquire) != epoch) {
      // A stop() racing this pause: the service is leaving and will not park for this epoch, so do not wait out the
      // whole timeout. stop_ is its first store, threaded_ its last; either says so. (The deadline test keeps its own
      // line: test_exl3_ram_miss_stage_trace_causal counts the clock reads by line.)
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
    // The service parked for this epoch (it drained the command ring first): this caller owns the tier until resume.
    tier_->set_parked(true);
    // The caller synchronized the stream, so every copy wait has seen its CopyDone. Once the copy engine is idle every
    // job is in the completion ring, and this caller, the owner now, drains it and releases their COPYING leases (D7).
    tier_->wait_copy_idle_owned(now_ns() + timeout_ns);
    tier_->retire_leases(true);  // a settle pass: the synchronized stream left no signal still to land
    if (tier_->graph_leases_outstanding() > 0) {
      resume_locked();
      return 2;
    }
    return 1;
  }

  // Advisories posted while paused predate the eager use: skip them too.
  void resume() {
    std::lock_guard<std::mutex> caller(tier_->caller_mutex());
    resume_locked();
  }

 private:
  // The owner hands the tier back: its writes happen-before the service's next request through the release of
  // pause_epoch_ (the parked loop acquires it). parked_ is cleared first, so a caller that then takes caller_mutex()
  // sees the service as the owner and queues instead of touching the tier. A timed-out pause never set parked_.
  void resume_locked() {
    // A prefill fill uses the reader the service thread is about to use: join it, and run its epilogue here, on the
    // owner (the fill thread writes no tier state), before the release below hands the tier back.
    tier_->fill_join();
    tier_->skip_advice_posted_so_far();
    tier_->set_parked(false);
    const uint64_t epoch = pause_epoch_.load(std::memory_order_relaxed);
    if (epoch & 1u) pause_epoch_.store(epoch + 1u, std::memory_order_release);
    tier_->request_pause(false);
  }

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
    uint32_t heartbeat = 0;
    uint32_t iterations = 0;
    bool stopped_parked = false;  // stop() came while a caller owned the tier: that caller keeps it
    while (!stop_.load(std::memory_order_relaxed)) {
      // Not every iteration: the word shares the cache line the device polls.
      if ((++iterations & 1023u) == 1u) store_release(page_ + kHeartbeat, ++heartbeat);
      tier_->drain_commands();  // between requests: every queued Python command, in order
      const uint64_t epoch = pause_epoch_.load(std::memory_order_acquire);
      if (epoch & 1u) {
        tier_->drain_commands();  // nothing queued before the pause is left behind
        parked_epoch_.store(epoch, std::memory_order_release);  // the handoff: the pausing caller owns the tier
        while (pause_epoch_.load(std::memory_order_acquire) == epoch && !stop_.load())
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        if (pause_epoch_.load(std::memory_order_acquire) == epoch) {
          stopped_parked = true;  // stopped while parked: touch nothing of the tier the caller owns
          break;
        }
        continue;
      }
      if (tier_->pump_demand() || tier_->pump_prefetch() || tier_->pump_advice()) {
        idle = 0;
        iterations = 0;  // one heartbeat per request served
        continue;
      }
      if (++idle < spin_iters_) {
        _mm_pause();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));  // the idle path (spec L12): kept
      }
    }
    // Plan F14: a command queued after the last iteration's drain is answered here, not left to a waiting caller.
    if (!stopped_parked) tier_->drain_commands();
    tier_->set_counter(kRunning, 0);
  }

  void watch() {
    int64_t fatal_since = 0;
    bool reported = false;
    uint64_t episode = 0;       // the busy episode last seen, 0: idle
    int64_t episode_since = 0;  // when the watchdog first saw it
    uint64_t armed = 0;         // the copy wait last seen armed and not released, 0: none
    int64_t armed_since = 0;
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
      // D6: the clock is read here, every 20 ms, never by the service. One episode held past fatal_wait is a hung
      // request; detection is at most 20 ms late against a 30 s deadline.
      // The copy wait's deadline lives here, not on the copy thread or the service: a copy thread stuck in a driver
      // call, or a service stuck in a read, must still end the device's wait (at most 20 ms late).
      const uint64_t gate = tier_->armed_copy_wait();
      if (gate != armed) {
        armed = gate;
        armed_since = now;
      }
      tier_->release_copy_gate(armed != 0 && now - armed_since > tier_->copy_wait_timeout_ns() ? armed : 0);
      const uint64_t busy = tier_->busy_episode();
      if (busy != episode) {
        episode = busy;
        episode_since = now;
      }
      // Once stop() began, the process is failing stop: only a hung read can still abort.
      const bool fatal_held = !stop_.load() && fatal_since != 0 && now - fatal_since > fatal_wait_ns_;
      const bool stuck = episode != 0 && now - episode_since > fatal_wait_ns_;
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
  uint64_t spin_iters_ = 1;  // idle polls before the idle sleep: idle_budget(spin_ns_), set in start()
  std::thread thread_;
  std::thread watchdog_;
  static constexpr int kPinPending = -1;
  std::atomic<bool> stop_{false};
  std::atomic<bool> watch_stop_{false};
  // The pause handshake: pause_epoch_ is odd while a pause is requested (a new value per pause), and the loop stores
  // into parked_epoch_ the epoch it parked for. Written under the tier's caller_mutex() (pause_epoch_) and by the
  // service thread (parked_epoch_) only.
  std::atomic<uint64_t> pause_epoch_{0};
  std::atomic<uint64_t> parked_epoch_{0};
  std::atomic<int> pin_error_{kPinPending};  // 0 pinned (or not asked), else the errno
};

}  // namespace expert_stream
}  // namespace sglang
