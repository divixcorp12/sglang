// The DSpark draft's CPU experts, the host half of the draft channel (draft_channel.h): one thread, the OpenMP master
// of the draft's team on the draft cores. It reads each posted record, runs the stage's M-row forward over the staged
// x and routes into the stage's out rows, and completes the record through the lease channel (done, then the Dekker
// open of the gate). See LEASE_PROTOCOL.md, "The lease channel".
//
// Idle: for keep_warm_ns after a job the team runs register work, then PAUSE, inside the kernel's keep_warm, which
// watches the channel's head word: the GPU's release store of the next head ends the hold with no syscall. spin_ns
// after the warm window the hold releases the team, and the thread polls the head, spinning its idle budget and then
// with 50 us sleeps (the GPU cannot ring a futex). A negative spin_ns holds the team until the next post, however long.
//
// Failure: a refused forward, a torn or malformed record or a lapped ring (the device posts once and waits) fail-stops;
// a watchdog thread fail-stops when a posted record stays incomplete for fatal_wait_ns.
#pragma once

#include "../draft_channel.h"
#include "cpu_experts.h"
#include "lease_channel.h"
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <future>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sglang::expert_stream::draft {

/// Instr build only (Config::test_hooks): how long the poll path sleeps between its stop and head loads, to make the
/// teardown interleaving deterministic (draft_test_poll_pause).
inline std::atomic<int64_t> g_test_poll_pause_us{0};

class DraftCpuThread {
 public:
  struct Config {
    uint8_t* channel = nullptr;         // kChannelBytes, pinned: the page and the completion block
    const uint8_t* x = nullptr;         // fp16 [stages, kMaxRows, hidden]
    const int32_t* slots = nullptr;     // [stages, kMaxRows, kMaxK]
    const float* weights = nullptr;     // [stages, kMaxRows, kMaxK]
    float* out = nullptr;               // [stages, kMaxRows, hidden]
    int64_t hidden = 0;
    int stages = 0;
    int threads = 1;
    std::vector<int> cores;             // worker i on cores[i]; this thread on cores[0]
    int64_t spin_ns = -1, keep_warm_ns = 0, fatal_wait_ns = 0;
    bool test_hooks = false;            // the instr build's: honour g_test_poll_pause_us
  };

  static constexpr const char* kPrefix = "DSpark draft CPU experts: ";

  explicit DraftCpuThread(Config config) : config_(std::move(config)), layers_(config_.stages) {
    const Config& c = config_;
    if (!c.channel || !c.x || !c.slots || !c.weights || !c.out)
      throw std::runtime_error(std::string(kPrefix) + "the channel and the stage areas are required");
    if (c.hidden <= 0 || c.stages < 1) throw std::runtime_error(std::string(kPrefix) + "no stage or no hidden size");
    if (c.keep_warm_ns < 0 || c.fatal_wait_ns <= 0)
      throw std::runtime_error(std::string(kPrefix) + "the keep-warm window is negative or the fatal wait not positive");
    check_cpu_expert_team(kPrefix, c.cores, c.threads);
  }

  ~DraftCpuThread() {
    stop();
  }

  DraftCpuThread(const DraftCpuThread&) = delete;
  DraftCpuThread& operator=(const DraftCpuThread&) = delete;

  /// Before start(): `stage`'s layer, made by its kernel's make_layer. Every stage shares one kernel.
  void set_layer(int stage, const cpu_experts::ExpertLayer& layer) {
    if (thread_.joinable()) throw std::runtime_error(std::string(kPrefix) + "set_layer after start");
    if (stage < 0 || stage >= config_.stages)
      throw std::runtime_error(std::string(kPrefix) + "stage " + std::to_string(stage) + " is out of range");
    if (layer.kernel == nullptr) throw std::runtime_error(std::string(kPrefix) + "a layer without a kernel");
    if (layer.hidden != config_.hidden)
      throw std::runtime_error(std::string(kPrefix) + "stage " + std::to_string(stage) + "'s layer has hidden " +
                               std::to_string(layer.hidden) + ", the areas " + std::to_string(config_.hidden));
    if (layer.kernel->max_routes() < kMaxK || layer.kernel->max_rows() < kMaxRows)
      throw std::runtime_error(std::string(kPrefix) + "kernel " + layer.kernel->name() + " takes " +
                               std::to_string(layer.kernel->max_rows()) + " rows of " +
                               std::to_string(layer.kernel->max_routes()) + " routes, a draft call up to " +
                               std::to_string(kMaxRows) + " of " + std::to_string(kMaxK));
    for (const cpu_experts::ExpertLayer& other : layers_)
      if (other.kernel != nullptr && other.kernel != layer.kernel)
        throw std::runtime_error(std::string(kPrefix) + "every stage runs on one kernel");
    layers_[stage] = layer;
  }

  /// Starts the thread and its watchdog; refuses unless every stage has a layer. Throws if the thread cannot pin.
  void start() {
    if (thread_.joinable()) throw std::runtime_error(std::string(kPrefix) + "already started");
    for (int s = 0; s < config_.stages; ++s)
      if (layers_[s].kernel == nullptr)
        throw std::runtime_error(std::string(kPrefix) + "stage " + std::to_string(s) + " has no layer");
    completed_.store(channel::head<DraftChannel>(config_.channel), std::memory_order_relaxed);
    std::promise<std::string> started;
    std::future<std::string> pinned = started.get_future();
    thread_ = std::thread([this, &started] { run(started); });
    if (const std::string error = pinned.get(); !error.empty()) {
      thread_.join();
      throw std::runtime_error(std::string(kPrefix) + error);
    }
    watchdog_ = std::thread([this] { watch(); });
  }

  /// Stops and joins the run thread, then the watchdog (a hung forward still fail-stops meanwhile), then opens a gate a wait still holds closed (no completer is left); idempotent.
  /// The device is past its last post at teardown, so moving the head word to end a hold misleads no one.
  void stop() {
    if (!thread_.joinable()) return;
    stop_.store(true, std::memory_order_seq_cst);
    __atomic_fetch_add(reinterpret_cast<uint32_t*>(config_.channel + DraftChannel::kHead), 1u, __ATOMIC_SEQ_CST);
    thread_.join();
    // The watchdog outlives the run thread's stop: a forward that never returns fail-stops within the fatal wait
    // instead of hanging this join.
    watchdog_stop_.store(true, std::memory_order_seq_cst);
    if (watchdog_.joinable()) watchdog_.join();
    channel::open_closed_gate<DraftChannel>(config_.channel);
  }

  int64_t jobs() const {
    return jobs_.load(std::memory_order_relaxed);
  }
  int64_t rows() const {
    return rows_.load(std::memory_order_relaxed);
  }
  int64_t forward_ns() const {
    return forward_ns_.load(std::memory_order_relaxed);
  }
  int64_t holds() const {
    return holds_.load(std::memory_order_relaxed);
  }

 private:
  const uint32_t* head_word() const {
    return reinterpret_cast<const uint32_t*>(config_.channel + DraftChannel::kHead);
  }

  std::string pin() const {
    pthread_setname_np(pthread_self(), "dspark-cpu");
    if (config_.cores.empty()) return "";
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(config_.cores.front(), &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0 ? "" : "cannot pin the draft CPU thread to its core";
  }

  void run(std::promise<std::string>& started) {
    const std::string error = pin();
    started.set_value(error);
    if (!error.empty()) return;
    constexpr int64_t kNever = INT64_MAX;
    uint32_t next = channel::skip_zero(completed_.load(std::memory_order_relaxed) + 1u);
    int64_t warm_until = 0;
    int64_t release_at = config_.spin_ns < 0 ? kNever : now_ns() + config_.spin_ns;
    const uint64_t spin_iters = idle_budget(config_.spin_ns);
    uint64_t idle = 0;
    while (!stop_.load(std::memory_order_acquire)) {
      if (config_.test_hooks)
        if (const int64_t pause = g_test_poll_pause_us.load(std::memory_order_relaxed); pause > 0)
          std::this_thread::sleep_for(std::chrono::microseconds(pause));
      const uint32_t head = channel::head<DraftChannel>(config_.channel);
      if (head != 0 && channel::reached(head, next)) {
        // stop() stores stop_ before it bumps the head word to end a hold: a bump seen here is not a record.
        if (stop_.load(std::memory_order_acquire)) break;
        if (head != next)
          fail_stop(std::string(kPrefix) + "record " + std::to_string(next) + " lapped (head " + std::to_string(head) +
                    "); the device posts one record per wait");
        warm_until = serve(next) + config_.keep_warm_ns;
        release_at = config_.spin_ns < 0 ? kNever : warm_until + config_.spin_ns;
        next = channel::skip_zero(next + 1u);
        idle = 0;
      } else if (release_at != 0) {
        hold(head, warm_until, release_at);
        if (channel::head<DraftChannel>(config_.channel) == head) release_at = 0;  // ran out: poll from now on
      } else if (++idle >= spin_iters) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      } else {
        _mm_pause();
      }
    }
  }

  /// Reads record `seq`, runs its forward, completes it; returns the forward's end.
  int64_t serve(uint32_t seq) {
    alignas(64) uint8_t raw[DraftChannel::kRecordBytes];
    const uint8_t* rec = channel::record_at<DraftChannel>(config_.channel, seq);
    if (!channel::read_seqlocked<DraftChannel>(rec, seq, raw))
      fail_stop(std::string(kPrefix) + "record " + std::to_string(seq) + " torn");
    uint32_t word, epoch;
    std::memcpy(&word, raw + kRecStage, 4);
    std::memcpy(&epoch, raw + kRecEpoch, 4);
    const int stage = static_cast<int>(word & 0xFFFFu), rows = static_cast<int>((word >> 16) & 0xFFu),
              k = static_cast<int>(word >> 24);
    if (stage >= config_.stages || rows < 1 || rows > kMaxRows || k < 1 || k > kMaxK)
      fail_stop(std::string(kPrefix) + "record " + std::to_string(seq) + " malformed (stage " + std::to_string(stage) +
                ", rows " + std::to_string(rows) + ", k " + std::to_string(k) + ")");
    // The slot and weight areas are kMaxK wide per token; the kernel reads [rows, k] contiguous, so compact them.
    int32_t slots[kMaxRows * kMaxK];
    float weights[kMaxRows * kMaxK];
    const int32_t* s = config_.slots + static_cast<int64_t>(stage) * kMaxRows * kMaxK;
    const float* w = config_.weights + static_cast<int64_t>(stage) * kMaxRows * kMaxK;
    for (int t = 0; t < rows; ++t)
      for (int i = 0; i < k; ++i) {
        slots[t * k + i] = s[t * kMaxK + i];
        weights[t * k + i] = w[t * kMaxK + i];
      }
    const cpu_experts::ExpertLayer& layer = layers_[stage];
    cpu_experts::ForwardCall call;
    call.rows = rows;
    call.k = k;
    call.threads = config_.threads;
    call.cores = config_.cores;
    call.x = config_.x + static_cast<int64_t>(stage) * kMaxRows * config_.hidden * 2;
    call.slots = slots;
    call.weights = weights;
    call.out = config_.out + static_cast<int64_t>(stage) * kMaxRows * config_.hidden;
    const int64_t start = now_ns();
    try {
      layer.kernel->forward(layer, call);
    } catch (const std::exception& e) {
      fail_stop(std::string(kPrefix) + "forward of record " + std::to_string(seq) + " (stage " +
                std::to_string(stage) + ") failed: " + e.what());
    }
    const int64_t end = now_ns();
    add(forward_ns_, end - start);
    add(jobs_, 1);
    add(rows_, rows);
    channel::complete<DraftChannel>(config_.channel, seq, static_cast<uint64_t>(epoch) << 32 | seq);
    completed_.store(seq, std::memory_order_release);
    return end;
  }

  /// Holds the team until the head word moves past `head` or the clock reaches release_at.
  void hold(uint32_t head, int64_t warm_until, int64_t release_at) {
    add(holds_, 1);
    try {
      layers_[0].kernel->keep_warm(config_.cores, config_.threads, head_word(), head, warm_until, release_at);
    } catch (const std::exception& e) {
      fail_stop(std::string(kPrefix) + "keep-warm failed: " + e.what());
    }
  }

  /// Every 20 ms: fail-stops when one posted record has stood incomplete for fatal_wait_ns.
  void watch() {
    uint32_t watched = 0;
    int64_t since = 0;
    while (!watchdog_stop_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (watchdog_stop_.load(std::memory_order_acquire)) return;
      const uint32_t head = channel::head<DraftChannel>(config_.channel);
      if (head == 0 || head == completed_.load(std::memory_order_acquire)) {
        watched = 0;
        continue;
      }
      const int64_t now = now_ns();
      if (head != watched) {
        watched = head;
        since = now;
      } else if (now - since >= config_.fatal_wait_ns) {
        fail_stop(std::string(kPrefix) + "record " + std::to_string(head) + " incomplete after " +
                  std::to_string(static_cast<double>(config_.fatal_wait_ns) * 1e-9) + " s (fatal wait)");
      }
    }
  }

  /// A counter only the draft thread writes: no locked add needed.
  static void add(std::atomic<int64_t>& counter, int64_t n) {
    counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }

  Config config_;
  std::vector<cpu_experts::ExpertLayer> layers_;
  std::thread thread_, watchdog_;
  std::atomic<bool> stop_{false}, watchdog_stop_{false};  // the watchdog stops after the run thread has joined
  std::atomic<uint32_t> completed_{0};
  std::atomic<int64_t> jobs_{0}, rows_{0}, forward_ns_{0}, holds_{0};
};

}  // namespace sglang::expert_stream::draft
