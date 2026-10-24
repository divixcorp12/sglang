// The team every CPU expert forward runs on: `threads` workers alive for the Team's lifetime, worker i pinned to
// cores[i] once at creation (no cores: unpinned), the owning thread as worker 0. A job runs through run(); inside one
// every worker may barrier(). Between jobs the workers run the kernel's register work (keep_warm.hpp) for keep_warm_ns
// after the job, then PAUSE, until the next job: no worker sleeps or leaves the team, so a job never waits for a
// wake-up and no OpenMP runtime policy decides when a worker spins.
//
// The host and a kernel's translation unit share a Team through the kernel's forward(layer, call, team), so both
// compile this header: atomics, plain ints, a span and function pointers.
#pragma once
#include "clock.hpp"
#include "cores.hpp"
#include "kernel.hpp"
#include "stage_gate.hpp"
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace sglang::cpu_experts {

class Team {
public:
    // The kernel's idle register work (its warm function).
    using Warm = int32_t (*)(const uint32_t* word, uint32_t seen, int64_t deadline_ns);

    // Starts the workers and pins the caller to cores[0]. Throws std::invalid_argument for threads outside [1, 4096],
    // more threads than `cores`, or a core check_cores refuses; std::runtime_error when the caller or a worker cannot
    // be pinned (the workers started are stopped first). `cores` must outlive the Team. `warm` is the idle register
    // work (null: PAUSE only). `name` (may be null) names the worker threads as the owner's thread is named, so
    // tooling that selects the owner's threads by name finds its workers.
    Team(std::span<const int> cores, int threads, Warm warm, int64_t keep_warm_ns, const char* name = nullptr)
        : cores_(cores), workers_(threads), warm_(warm), keep_warm_ns_(keep_warm_ns)
    {
        if (threads < 1 || threads > 4096 || (!cores.empty() && size_t(threads) > cores.size()))
            throw std::invalid_argument("CPU expert team needs a worker and no more workers than its cores");
        check_cores(cores);
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

    // Runs f(worker, workers) on every worker, the caller as worker 0, and returns once all have finished. One job at a
    // time, from the owning thread. f must not throw: a worker's exception ends the process and the caller's would skip
    // the job's end barrier.
    template <class F>
    void run(F&& f)
    {
        using G = std::remove_reference_t<F>;
        body_ = &thunk<G>;
        ctx_ = const_cast<void*>(static_cast<const void*>(std::addressof(f)));
        serving_ = false;
        generation_.fetch_add(1, std::memory_order_release);
        body_(ctx_, 0, workers_);
        barrier();
    }

    // Starts f(worker, workers) on workers 1..n-1 and returns at once; the caller stays worker 0 and does not run f. f
    // is a loop that returns when its owner tells it to, and must outlive that return: join_serve() waits for every
    // worker to leave it. No run() or barrier() in between, and f must not throw.
    template <class F>
    void serve(F&& f)
    {
        using G = std::remove_reference_t<F>;
        body_ = &thunk<G>;
        ctx_ = const_cast<void*>(static_cast<const void*>(std::addressof(f)));
        serving_ = true;
        serving_count_.store(workers_ - 1, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
    }

    // Returns once every worker serve() started has left its loop.
    void join_serve() const
    {
        while (serving_count_.load(std::memory_order_acquire) != 0) _mm_pause();
    }

    int64_t keep_warm_ns() const { return keep_warm_ns_; }

    // Inside a job: returns once every worker has arrived here. It spins and never sleeps: std::barrier parks early
    // arrivals on a futex, and a wake costs microseconds against phases tens of microseconds long.
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
        if (warm_ != nullptr && warm_until > team_now_ns())
            sink_.fetch_add(warm_(word, seen, warm_until), std::memory_order_relaxed);
        for (uint32_t tick = 0; __atomic_load_n(word, __ATOMIC_ACQUIRE) == seen; ++tick) {
            _mm_pause();
            if ((tick & 63) == 0 && until_ns != INT64_MAX && team_now_ns() >= until_ns) return;
        }
    }

private:
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
                  "the warm loop reads the job word as a plain uint32_t");

    using Body = void (*)(void* ctx, int worker, int workers);

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
            if (serving_) serving_count_.fetch_sub(1, std::memory_order_release);
            else barrier();
            warm_until = team_now_ns() + keep_warm_ns_;
        }
    }

    // A null body_ with a generation bump tells the workers to return.
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
    Warm warm_;
    int64_t keep_warm_ns_;
    std::vector<std::thread> threads_;
    // The job word: moves once per run() and once at stop; the workers' idle wait watches it.
    alignas(64) std::atomic<uint32_t> generation_{0};
    Body body_ = nullptr;
    void* ctx_ = nullptr;
    bool serving_ = false;  // the body is serve()'s: workers 1..n-1 only, no end barrier
    std::atomic<int> serving_count_{0};
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<uint32_t> phase_{0};
    alignas(64) std::atomic<int> started_{0};
    std::atomic<int> pin_error_{0};
    mutable std::atomic<int32_t> sink_{0};  // keeps the register work observable
};

}  // namespace sglang::cpu_experts
