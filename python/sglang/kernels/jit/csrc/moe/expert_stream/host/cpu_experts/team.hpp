// The team every CPU expert forward runs on: `threads` workers alive for the Team's lifetime, worker i pinned to
// cores[i] once at creation (no cores: unpinned), the owning thread as worker 0. A job runs through run(); inside one
// every worker may barrier(). Between jobs the workers run the kernel's register work (keep_warm.hpp) for keep_warm_ns
// after the job, then PAUSE, until the next job: no worker sleeps or leaves the team, so a job never waits for a
// wake-up and no OpenMP runtime policy decides when a worker spins.
//
// The host and the quant libraries share a Team through CpuExpertKernel::forward's reference, so its layout is this
// header's, compiled by both sides: atomics, plain ints, a span and function pointers (kernel.hpp's rule).
#pragma once
#include "kernel.hpp"
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace sglang::cpu_experts {

// Throws std::invalid_argument unless every core is in [0, CPU_SETSIZE) and none repeats. Not checked against the
// caller's affinity: the owner may run under a mask that excludes the expert cores, and the workers pin themselves
// outside it; a core that cannot be pinned fails the Team's construction (std::runtime_error).
inline void check_cores(std::span<const int> cores)
{
    for (size_t i = 0; i < cores.size(); ++i) {
        if (cores[i] < 0 || cores[i] >= CPU_SETSIZE)
            throw std::invalid_argument("CPU expert core " + std::to_string(cores[i]) + " is outside [0, CPU_SETSIZE)");
        for (size_t j = 0; j < i; ++j)
            if (cores[j] == cores[i])
                throw std::invalid_argument("CPU expert core " + std::to_string(cores[i]) + " repeats");
    }
}

// Pins the calling thread to `core`; false when it cannot.
inline bool pin_thread(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

// CLOCK_MONOTONIC in ns, the clock of every deadline here (libstdc++'s steady_clock reads it).
inline int64_t team_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

class Team {
public:
    using Body = void (*)(void* ctx, int worker, int workers);

    // Starts the workers and pins the caller to cores[0]. Throws std::invalid_argument for threads < 1 or more threads
    // than `cores`, std::runtime_error when the caller or a worker cannot be pinned (the workers started are stopped
    // first). `cores` must outlive the Team. `kernel` supplies the idle register work (null: PAUSE only). `name` (may
    // be null) names the worker threads as the owner's thread is named, so tooling that selects the owner's threads by
    // name finds its workers.
    Team(std::span<const int> cores, int threads, const CpuExpertKernel* kernel, int64_t keep_warm_ns,
         const char* name = nullptr)
        : cores_(cores), workers_(threads), kernel_(kernel), keep_warm_ns_(keep_warm_ns)
    {
        if (threads < 1 || (!cores.empty() && size_t(threads) > cores.size()))
            throw std::invalid_argument("CPU expert team needs a worker and no more workers than its cores");
        if (!cores.empty() && !pin_thread(cores[0])) throw std::runtime_error("cannot pin CPU expert worker to its core");
        threads_.reserve(size_t(threads - 1));
        for (int w = 1; w < threads; ++w) threads_.emplace_back([this, w, name] { worker_main(w, name); });
        while (started_.load(std::memory_order_acquire) < threads - 1) _mm_pause();
        if (pin_error_.load(std::memory_order_acquire)) {
            stop_workers();
            throw std::runtime_error("cannot pin CPU expert worker to its core");
        }
    }
    ~Team() { stop_workers(); }
    Team(const Team&) = delete;
    Team& operator=(const Team&) = delete;

    int workers() const { return workers_; }
    std::span<const int> cores() const { return cores_; }

    // Runs body(ctx, worker, workers) on every worker, the caller as worker 0, and returns once all have finished. One
    // job at a time, from the owning thread. The body must not throw: a worker's exception ends the process and the
    // caller's would skip the job's end barrier.
    void run(Body body, void* ctx)
    {
        body_ = body;
        ctx_ = ctx;
        generation_.fetch_add(1, std::memory_order_release);
        body(ctx, 0, workers_);
        barrier();
    }
    template <class F>
    void run(F&& f)
    {
        using G = std::remove_reference_t<F>;
        run(&thunk<G>, const_cast<void*>(static_cast<const void*>(std::addressof(f))));
    }

    // Inside a job: returns once every worker has arrived here.
    void barrier()
    {
        const uint32_t phase = phase_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == workers_) {
            arrived_.store(0, std::memory_order_relaxed);
            phase_.store(phase + 1, std::memory_order_release);
        } else {
            while (phase_.load(std::memory_order_acquire) == phase) _mm_pause();
        }
    }

    // Waits until *word != seen or the clock reaches until_ns (INT64_MAX: never): the kernel's register work until
    // warm_until_ns, PAUSE after. The workers' idle wait, on the job word; the owner's, on a word of its own.
    void wait(const uint32_t* word, uint32_t seen, int64_t warm_until_ns, int64_t until_ns) const
    {
        const int64_t warm_until = std::min(warm_until_ns, until_ns);
        if (kernel_ != nullptr && warm_until > team_now_ns())
            sink_.fetch_add(kernel_->warm(word, seen, warm_until), std::memory_order_relaxed);
        for (uint32_t tick = 0; __atomic_load_n(word, __ATOMIC_ACQUIRE) == seen; ++tick) {
            _mm_pause();
            if ((tick & 63) == 0 && until_ns != INT64_MAX && team_now_ns() >= until_ns) return;
        }
    }

private:
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
                  "the warm loop reads the job word as a plain uint32_t");

    template <class F>
    static void thunk(void* ctx, int worker, int workers)
    {
        (*static_cast<F*>(ctx))(worker, workers);
    }

    void worker_main(int w, const char* name)
    {
        if (name != nullptr) pthread_setname_np(pthread_self(), name);
        if (!cores_.empty() && !pin_thread(cores_[size_t(w)])) pin_error_.store(1, std::memory_order_relaxed);
        started_.fetch_add(1, std::memory_order_release);
        const uint32_t* word = reinterpret_cast<const uint32_t*>(&generation_);
        uint32_t seen = 0;
        int64_t warm_until = 0;
        for (;;) {
            wait(word, seen, warm_until, INT64_MAX);
            seen = generation_.load(std::memory_order_acquire);
            const Body body = body_;
            if (body == nullptr) return;
            body(ctx_, w, workers_);
            barrier();
            warm_until = team_now_ns() + keep_warm_ns_;
        }
    }

    void stop_workers()
    {
        if (threads_.empty()) return;
        body_ = nullptr;
        generation_.fetch_add(1, std::memory_order_release);
        for (std::thread& t : threads_) t.join();
        threads_.clear();
    }

    std::span<const int> cores_;
    int workers_;
    const CpuExpertKernel* kernel_;
    int64_t keep_warm_ns_;
    std::vector<std::thread> threads_;
    // The job word: moves once per run() and once at stop; the workers' idle wait watches it.
    alignas(64) std::atomic<uint32_t> generation_{0};
    Body body_ = nullptr;
    void* ctx_ = nullptr;
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<uint32_t> phase_{0};
    alignas(64) std::atomic<int> started_{0};
    std::atomic<int> pin_error_{0};
    mutable std::atomic<int32_t> sink_{0};  // keeps the register work observable
};

// A forward on a team made for this call alone (call.threads workers on call.cores): tests, the bench and other
// harnesses that hold no Team. The engine runs on its own.
inline void CpuExpertKernel::forward(const ExpertLayer& layer, const ForwardCall& call) const
{
    Team team(call.cores, call.threads, this, 0);
    forward(layer, call, team);
}

}  // namespace sglang::cpu_experts
