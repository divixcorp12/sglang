// The worker cores every forward's team runs on, and the team itself: one OpenMP team per call, the calling thread as
// worker 0, worker i pinned to core i once cores are configured.
#pragma once
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

// Why this thread's last register, free, forward or keep-warm call returned 1 ("" otherwise); each clears it on entry.
// Per library, like the rest of the framework; the C ABI does not export it, a quant's own wrappers read it.
inline std::string& last_error()
{
    static thread_local std::string error;
    return error;
}

// Records `what` as last_error() and returns 1; never throws (it runs in the C ABI's catch blocks).
inline int fail(const char* what) noexcept
{
    try {
        last_error() = what;
    } catch (...) {
    }
    return 1;
}

struct Cores
{
    // Worker i on cores[i], the caller as worker 0. Cores must be distinct and in [0, CPU_SETSIZE); refused (2) once
    // the first forward or keep-warm has frozen them. Not checked against the caller's affinity: the engine configures
    // from a thread whose inherited mask may exclude the expert cores, and the workers pin themselves outside it.
    // A core that cannot be pinned fails the first forward's pin (1).
    static int configure(const int32_t* cores, int32_t n) noexcept
    {
        try {
            if (!cores || n < 1 || n > CPU_SETSIZE) return 2;
            for (int i = 0; i < n; ++i) {
                if (cores[i] < 0 || cores[i] >= CPU_SETSIZE) return 2;
                for (int j = 0; j < i; ++j)
                    if (cores[j] == cores[i]) return 2;
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (started.load(std::memory_order_relaxed)) return 2;
            configured.assign(cores, cores + n);
            return 0;
        } catch (...) {
            return 1;
        }
    }

    static void freeze()
    {
        if (started.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (!started.load(std::memory_order_relaxed)) {
            compute = configured;
            started.store(true, std::memory_order_release);
        }
    }

    // Empty until freeze(), and when no cores were configured (workers then run unpinned).
    static const std::vector<int>& frozen() { return compute; }

    // Sets `error` when the pin fails; the caller checks that worker < frozen().size().
    static void pin(int worker, std::atomic<int>& error)
    {
        if (compute.empty()) return;
        const int core = compute[worker];
        static thread_local int pinned_core = -1;
        if (pinned_core != core || sched_getcpu() != core) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(core, &set);
            if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set))
                error.store(1, std::memory_order_relaxed);
            else
                pinned_core = core;
        }
    }

private:
    static inline std::mutex mutex;
    static inline std::vector<int> configured;
    static inline std::atomic<bool> started{false};
    static inline std::vector<int> compute;  // Immutable after the release store of `started`.
};

// body(worker, workers) on every worker of a pinned team of `threads`. Every pin precedes a barrier and the body runs
// only on a full, fully pinned team, so all workers take the same branch; body must not throw (it runs inside an
// OpenMP region).
template <class Body>
void run_team(int threads, Body&& body)
{
    if (threads < 1) throw std::runtime_error("CPU expert team needs at least one worker");
    Cores::freeze();
    const std::vector<int>& cores = Cores::frozen();
    if (!cores.empty() && size_t(threads) > cores.size())
        throw std::runtime_error("CPU expert worker count exceeds configured cores");
    std::atomic<int> pin_error{0};
    std::atomic<int> actual_workers{0};
    #pragma omp parallel num_threads(threads) shared(body, pin_error, actual_workers)
    {
        const int worker = omp_get_thread_num(), n = omp_get_num_threads();
        if (worker == 0) actual_workers.store(n, std::memory_order_relaxed);
        Cores::pin(worker, pin_error);
        #pragma omp barrier
        if (n == threads && !pin_error.load(std::memory_order_relaxed)) body(worker, n);
    }
    if (pin_error.load()) throw std::runtime_error("cannot pin CPU expert worker to its configured core");
    if (actual_workers.load() != threads)
        throw std::runtime_error("OpenMP returned fewer CPU expert workers than requested");
}

}  // namespace
}  // namespace sglang::cpu_experts
