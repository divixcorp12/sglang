// Copy workers for the RAM-miss row reader's packing step (plan Task 4, packing-worker variant).
//
// The reader's owner thread keeps every decision: which row is ready, whether the drives delivered
// what its segments read, which bank is free. A worker only executes a copy the owner has already
// vetted, from a bounce slot into slab bytes, and reports that it finished. Nothing here reads or
// writes reader, tier or ring state.
//
// Ownership of a PackJob: the owner arms it and posts it; workers touch `claimed`, `finished` and the
// stamps until the last chunk's `finished` increment, which is the worker's final access. The owner
// may re-arm the job only after it has seen finished == chunks (an acquire load), and must not
// return from the read that posted it (the destination slots and the bounce bank belong to the
// caller and the next read) until every posted job is done.

#pragma once

#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sglang {
namespace exl3_ram_miss {

// One contiguous copy of a row: `bytes` from the bounce slot to a slab.
struct CopyRun {
  uint8_t* dst = nullptr;
  const uint8_t* src = nullptr;
  int64_t bytes = 0;
};

// One row's copy, split into `chunks` byte ranges of the concatenation of its runs. Chunks are
// disjoint, so workers never write the same byte.
struct PackJob {
  // Written by the owner before post(); read-only for workers.
  const CopyRun* runs = nullptr;
  size_t run_count = 0;
  int64_t total = 0;
  unsigned chunks = 1;
  int64_t delay_ns = 0;  // test only: a slow copy, its total time; each chunk sleeps its share
  // Stamps chunk start and end when set (only for a traced read): `clock(clock_arg)`.
  int64_t (*clock)(const void*) = nullptr;
  const void* clock_arg = nullptr;
  // Worker-shared.
  std::atomic<unsigned> claimed{0};
  std::atomic<unsigned> finished{1};  // starts done: a job nobody holds reads done()
  std::atomic<int64_t> first_start{0};  // earliest chunk start, only when traced
  std::atomic<int64_t> last_end{0};     // latest chunk end, only when traced

  void arm(
      const CopyRun* run_list, size_t count, unsigned chunk_count, int64_t delay, int64_t (*clock_fn)(const void*),
      const void* clock_argument) {
    runs = run_list;
    run_count = count;
    total = 0;
    for (size_t i = 0; i < count; ++i) total += run_list[i].bytes;
    chunks = std::max(1u, chunk_count);
    delay_ns = delay;
    clock = clock_fn;
    clock_arg = clock_argument;
    claimed.store(0, std::memory_order_relaxed);
    finished.store(0, std::memory_order_relaxed);
    first_start.store(INT64_MAX, std::memory_order_relaxed);
    last_end.store(0, std::memory_order_relaxed);
  }

  // Acquire: after true, every byte of every chunk is stored and visible to the caller.
  bool done() const { return finished.load(std::memory_order_acquire) == chunks; }
};

// The cores a copy worker may run on: what the creating thread may run on, less the production
// reserve 64-71 (71 is the doorbell's spin core; NVMe completion interrupts land there too).
inline cpu_set_t pack_worker_cpus(const cpu_set_t& inherited) {
  cpu_set_t allowed = inherited;
  for (int core = 64; core <= 71; ++core) CPU_CLR(core, &allowed);
  return allowed;
}

class PackPool {
 public:
  // Throws, with every thread already joined, when there is no allowed core or a worker cannot be pinned.
  PackPool(unsigned workers, const cpu_set_t& inherited, size_t capacity)
      : allowed_(pack_worker_cpus(inherited)), queue_(capacity, nullptr) {
    if (CPU_COUNT(&allowed_) == 0) {
      throw std::runtime_error("exl3 RAM miss: no core is left for the packing workers once cores 64-71 are excluded");
    }
    if (workers == 0) throw std::runtime_error("exl3 RAM miss: a packing pool needs at least one worker");
    try {
      for (unsigned i = 0; i < workers; ++i) threads_.emplace_back([this] { run(); });
      std::unique_lock<std::mutex> lock(mutex_);
      started_cv_.wait(lock, [&] { return started_ == threads_.size(); });
    } catch (...) {
      shutdown();
      throw;
    }
    if (const int error = pin_error_.load()) {
      shutdown();
      throw std::runtime_error(
          std::string("exl3 RAM miss: could not pin a packing worker: ") + std::strerror(error));
    }
  }

  ~PackPool() { shutdown(); }
  PackPool(const PackPool&) = delete;
  PackPool& operator=(const PackPool&) = delete;

  size_t workers() const { return threads_.size(); }

  // The affinity of worker `index` as the kernel reports it (tests).
  cpu_set_t worker_affinity(size_t index) {
    cpu_set_t set;
    CPU_ZERO(&set);
    pthread_getaffinity_np(threads_.at(index).native_handle(), sizeof(set), &set);
    return set;
  }

  // Resize the queue of an idle pool (nothing posted): the reader posts a job per piece with piece streaming.
  void set_capacity(size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ != 0) throw std::runtime_error("exl3 RAM miss: the packing queue was resized while jobs were posted");
    queue_.assign(capacity, nullptr);
    head_ = 0;
  }

  // Hand `job` (armed) to the workers. Jobs are served in the order posted.
  void post(PackJob* job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (count_ == queue_.size()) throw std::runtime_error("exl3 RAM miss: the packing queue overflowed its slot count");
      queue_[(head_ + count_) % queue_.size()] = job;
      ++count_;
    }
    work_cv_.notify_all();
  }

 private:
  void run() {
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(allowed_), &allowed_);
    pthread_setname_np(pthread_self(), "exl3-pack");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (error != 0) pin_error_.store(error);
      ++started_;
    }
    started_cv_.notify_all();
    if (error != 0) return;
    while (true) {
      PackJob* job = nullptr;
      unsigned chunk = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [&] { return stop_ || count_ > 0; });
        if (count_ == 0) return;  // stopping, and nothing is posted
        job = queue_[head_];
        chunk = job->claimed.fetch_add(1, std::memory_order_relaxed);
        if (chunk + 1 == job->chunks) {  // the last chunk is claimed: nobody else needs this job
          head_ = (head_ + 1) % queue_.size();
          --count_;
        }
      }
      copy_chunk(job, chunk);
    }
  }

  void copy_chunk(PackJob* job, unsigned chunk) {
    const int64_t start = job->clock ? job->clock(job->clock_arg) : 0;
    if (job->delay_ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(job->delay_ns / job->chunks));
    // Byte range of the concatenated runs, its inner boundaries on 64 B so two workers never share a line.
    const auto edge = [&](unsigned i) {
      return i >= job->chunks ? job->total : (job->total / static_cast<int64_t>(job->chunks) * i) & ~int64_t{63};
    };
    const int64_t lo = chunk == 0 ? 0 : edge(chunk);
    const int64_t hi = edge(chunk + 1);
    int64_t at = 0;
    for (size_t i = 0; i < job->run_count && at < hi; ++i) {
      const CopyRun& run = job->runs[i];
      const int64_t from = std::max(lo, at), to = std::min(hi, at + run.bytes);
      if (to > from) std::memcpy(run.dst + (from - at), run.src + (from - at), static_cast<size_t>(to - from));
      at += run.bytes;
    }
    // Large copies may use non-temporal stores, which are not ordered by the release below: this
    // core's own fence is what makes them visible before the owner (and then the device) reads them.
    _mm_sfence();
    if (job->clock) {
      const int64_t end = job->clock(job->clock_arg);
      int64_t seen = job->first_start.load(std::memory_order_relaxed);
      while (start < seen && !job->first_start.compare_exchange_weak(seen, start, std::memory_order_relaxed)) {
      }
      seen = job->last_end.load(std::memory_order_relaxed);
      while (end > seen && !job->last_end.compare_exchange_weak(seen, end, std::memory_order_relaxed)) {
      }
    }
    job->finished.fetch_add(1, std::memory_order_release);  // the worker's last access to the job
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    work_cv_.notify_all();
    for (auto& thread : threads_) {
      if (thread.joinable()) thread.join();
    }
    threads_.clear();
  }

  cpu_set_t allowed_;
  std::vector<PackJob*> queue_;  // ring of posted jobs; a job leaves it when its last chunk is claimed
  size_t head_ = 0;
  size_t count_ = 0;
  size_t started_ = 0;
  bool stop_ = false;
  std::atomic<int> pin_error_{0};
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable started_cv_;
  std::vector<std::thread> threads_;
};

}  // namespace exl3_ram_miss
}  // namespace sglang
