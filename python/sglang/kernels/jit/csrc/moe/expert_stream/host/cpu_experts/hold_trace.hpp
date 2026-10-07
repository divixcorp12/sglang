// Optional per-worker timing of keep-warm's parallel-region exit and implicit join.
// No worker-side allocation or IO. Normal quant builds instantiate the empty specialization.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace sglang::cpu_experts::hold_trace {
#if (defined(EXL3_MOE_CPU_WORKER_TRACE) && EXL3_MOE_CPU_WORKER_TRACE) || \
    (defined(SGLANG_CPU_EXPERT_HOLD_TRACE) && SGLANG_CPU_EXPERT_HOLD_TRACE)
inline constexpr bool kOn = true;
#else
inline constexpr bool kOn = false;
#endif
inline int64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return int64_t(t.tv_sec) * 1'000'000'000 + t.tv_nsec;
}
inline constexpr int kWorkers = 64;
struct alignas(64) Worker {
    int64_t enter = 0, ready = 0, exit = 0;
    int tid = 0, cpu = -1;
    uint32_t a = 0, b = 0;
};
struct Record {
    Worker stamp;
    int64_t serial, begin, end, warm, release;
    int worker, threads;
    uint32_t seen_a, seen_b;
    bool either;
};
class Buffer {
 public:
    Buffer() {
        const char* prefix = std::getenv("SGLANG_CPU_EXPERT_HOLD_TRACE_PREFIX");
        if (!prefix || !*prefix) return;
        if (const char* raw = std::getenv("SGLANG_CPU_EXPERT_HOLD_TRACE_CAPACITY")) {
            const std::string_view text(raw);
            if (text.empty() || text.find_first_not_of("0123456789") != std::string_view::npos)
                throw std::invalid_argument("hold trace capacity must be in [1, 1048576]");
            char* end = nullptr;
            const auto parsed = std::strtoull(raw, &end, 10);
            if (*end || parsed < 1 || parsed > 1048576)
                throw std::invalid_argument("hold trace capacity must be in [1, 1048576]");
            capacity = size_t(parsed);
        }
        path = std::string(prefix) + "." + std::to_string(getpid()) + "." +
               std::to_string(syscall(SYS_gettid)) + ".jsonl";
        records = std::make_unique<Record[]>(capacity);
    }
    ~Buffer() {
        if (!records) return;
        FILE* out = std::fopen(path.c_str(), "w");
        if (!out) { std::perror(path.c_str()); return; }
        std::fprintf(out, "{\"schema\":1,\"clock\":\"CLOCK_MONOTONIC\",\"kind\":\"keep_warm\"}\n");
        for (size_t i = 0; i < used; ++i) {
            const auto& r = records[i]; const auto& s = r.stamp;
            std::fprintf(out, "{\"hold\":%lld,\"begin\":%lld,\"end\":%lld,\"warm_until\":%lld,\"release_at\":%lld,\"worker\":%d,\"threads\":%d,\"tid\":%d,\"cpu\":%d,\"enter\":%lld,\"ready\":%lld,\"exit\":%lld,\"seen_a\":%u,\"seen_b\":%u,\"observed_a\":%u,\"observed_b\":%u,\"either\":%s}\n",
                (long long)r.serial, (long long)r.begin, (long long)r.end,
                (long long)r.warm, (long long)r.release, r.worker, r.threads, s.tid, s.cpu,
                (long long)s.enter, (long long)s.ready, (long long)s.exit,
                r.seen_a, r.seen_b, s.a, s.b, r.either ? "true" : "false");
        }
        std::fprintf(out, "{\"footer\":true,\"holds_seen\":%lld,\"dropped_holds\":%lld,\"worker_records\":%zu}\n",
                     (long long)seen, (long long)dropped, used);
        std::fclose(out);
    }
    std::string path;
    std::unique_ptr<Record[]> records;
    size_t capacity = 65536, used = 0;
    int64_t seen = 0, dropped = 0;
};
namespace { inline Buffer& buffer() { static thread_local Buffer b; return b; } }
template<bool On> struct Capture;
template<> struct Capture<false> {
    Capture(int, const uint32_t*, uint32_t, const uint32_t*, uint32_t, int64_t, int64_t) {}
    void enter(int) {} void ready(int) {} void exit(int) {} void finish() {}
};
template<> struct Capture<true> {
    Capture(int n, const uint32_t* a, uint32_t seen_a, const uint32_t* other, uint32_t seen_b,
            int64_t warm, int64_t release)
        : b(buffer()), n(n), a(a), other(other), seen_a(seen_a), seen_b(seen_b), warm(warm), release(release) {
        if (n < 1 || n > kWorkers) throw std::invalid_argument("hold tracing supports 1..64 workers");
        begin = now();
    }
    void enter(int w) { workers[w].enter = now(); workers[w].tid = int(syscall(SYS_gettid)); }
    void ready(int w) { workers[w].ready = now(); workers[w].cpu = sched_getcpu(); }
    void exit(int w) {
        auto& s = workers[w];
        s.exit = now();
        s.a = __atomic_load_n(a, __ATOMIC_ACQUIRE);
        s.b = other ? __atomic_load_n(other, __ATOMIC_ACQUIRE) : 0;
    }
    void finish() {
        const int64_t end = now(), serial = ++b.seen;
        if (!b.records) return;
        if (b.used + size_t(n) > b.capacity) { ++b.dropped; return; }
        for (int w = 0; w < n; ++w)
            b.records[b.used++] = Record{workers[w], serial, begin, end, warm, release,
                                         w, n, seen_a, seen_b, other != nullptr};
    }
    Buffer& b;
    std::array<Worker, kWorkers> workers{};
    int n;
    const uint32_t* a;
    const uint32_t* other;
    uint32_t seen_a, seen_b;
    int64_t warm, release, begin;
};
static_assert(std::is_empty_v<Capture<false>>);
} // namespace sglang::cpu_experts::hold_trace
