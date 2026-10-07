// Diagnostic-only per-worker work and barrier timing. The normal plan instantiates the empty specialization.
#pragma once
#include "../../moe/expert_stream/host/trace_gate.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>

#ifndef EXL3_MOE_CPU_WORKER_TRACE
#define EXL3_MOE_CPU_WORKER_TRACE 0
#endif

namespace sglang::exl3_cpu::worker_trace {
inline int64_t clock_ns(clockid_t clock = CLOCK_MONOTONIC) {
    timespec t{};
    if (clock_gettime(clock, &t)) return -1;
    return int64_t(t.tv_sec) * 1'000'000'000 + t.tv_nsec;
}
inline size_t option(const char* name, size_t fallback, size_t maximum) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    const std::string_view text(raw);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string_view::npos)
        throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
    char* end = nullptr;
    const auto value = std::strtoull(raw, &end, 10);
    if (*end || value > maximum) throw std::invalid_argument(std::string(name) + " exceeds its bound");
    return size_t(value);
}
// Keep the plan's phase IDs: 0,1,2,3,5 (4 is an old, now-fused output-transform phase).
inline constexpr int kWorkers = 64, kPhases = 6, kActivePhases = 5;
struct PhaseStamp {
    int64_t begin = 0, work_end = 0, end = 0;
    int64_t cpu_begin = 0, cpu_work_end = 0, cpu_end = 0, units = 0;
};
struct alignas(64) WorkerStamp {
    std::array<PhaseStamp, kPhases> phase;
    rusage initial{};
    int64_t minflt = 0, majflt = 0, nvcsw = 0, nivcsw = 0;
    int64_t enter = 0, ready = 0;
    int tid = 0, cpu = 0;
};
struct Record {
    PhaseStamp stamp;
    int64_t forward_begin, forward_end, team_begin, team_end, serial;
    int64_t minflt, majflt, nvcsw, nivcsw;
    int worker, phase, tid, cpu, threads, rows, chunks;
    int64_t enter, ready;
};

struct ScratchRecord {
    int64_t serial, begin, end, cpu_begin, cpu_end;
    int64_t minflt, majflt, nvcsw, nivcsw;
    size_t growths, initialized_bytes, capacity_added_bytes;
    int rows, chunks;
};
struct GrowthRecord {
    const char* name;
    int64_t serial, begin, end, cpu_begin, cpu_end, minflt, majflt, nvcsw, nivcsw;
    size_t old_size, size, old_capacity, capacity, item_bytes;
    bool moved;
};
inline rusage usage() {
    rusage r{};
    if (getrusage(RUSAGE_THREAD, &r)) r.ru_minflt = -1;
    return r;
}
inline int64_t delta(long first, long last, const rusage& a, const rusage& b) {
    return a.ru_minflt < 0 || b.ru_minflt < 0 ? -1 : last - first;
}

// One writer: the forward's calling thread, after its team joined. A separate buffer per OS leader in this DSO.
class Buffer {
 public:
    Buffer() {
        const char* prefix = std::getenv("SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX");
        if (!prefix || !*prefix) return;
        capacity = option("SGLANG_EXL3_CPU_WORKER_TRACE_CAPACITY", 65536, 1048576);
        if (!capacity) throw std::invalid_argument("worker trace capacity must be positive");
        min_ns = int64_t(option("SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US", 6000, 1000000000)) * 1000;
        path = std::string(prefix) + "." + std::to_string(getpid()) + "." +
               std::to_string(syscall(SYS_gettid)) + ".jsonl";
        events = std::make_unique<Record[]>(capacity);
        if (const char* scratch_prefix = std::getenv("SGLANG_EXL3_CPU_SCRATCH_TRACE_PREFIX")) {
            if (*scratch_prefix) {
                scratch_path = std::string(scratch_prefix) + "." + std::to_string(getpid()) + "." +
                               std::to_string(syscall(SYS_gettid)) + ".jsonl";
                scratch = std::make_unique<ScratchRecord[]>(capacity / kActivePhases);
                growth = std::make_unique<GrowthRecord[]>(kGrowthCapacity);
            }
        }
    }
    ~Buffer() {
        if (!events) return;
        FILE* out = std::fopen(path.c_str(), "w");
        if (!out) { std::perror(path.c_str()); return; }
        std::fprintf(out, "{\"schema\":1,\"clock\":\"CLOCK_MONOTONIC\",\"cpu_clock\":\"CLOCK_THREAD_CPUTIME_ID\",\"min_ns\":%lld,\"capacity\":%zu}\n", (long long)min_ns, capacity);
        for (size_t i = 0; i < used; ++i) {
            const auto& r = events[i]; const auto& s = r.stamp;
            std::fprintf(out, "{\"forward\":%lld,\"forward_begin\":%lld,\"forward_end\":%lld,\"team_begin\":%lld,\"team_end\":%lld,\"worker\":%d,\"phase\":%d,\"tid\":%d,\"cpu\":%d,\"threads\":%d,\"rows\":%d,\"chunks\":%d,\"enter\":%lld,\"ready\":%lld,\"begin\":%lld,\"work_end\":%lld,\"end\":%lld,\"cpu_begin\":%lld,\"cpu_work_end\":%lld,\"cpu_end\":%lld,\"units\":%lld,\"minflt\":%lld,\"majflt\":%lld,\"nvcsw\":%lld,\"nivcsw\":%lld}\n",
                (long long)r.serial, (long long)r.forward_begin, (long long)r.forward_end,
                (long long)r.team_begin, (long long)r.team_end, r.worker, r.phase, r.tid, r.cpu,
                r.threads, r.rows, r.chunks, (long long)r.enter, (long long)r.ready,
                (long long)s.begin, (long long)s.work_end, (long long)s.end,
                (long long)s.cpu_begin, (long long)s.cpu_work_end, (long long)s.cpu_end, (long long)s.units,
                (long long)r.minflt, (long long)r.majflt, (long long)r.nvcsw, (long long)r.nivcsw);
        }
        std::fprintf(out, "{\"footer\":true,\"forwards_seen\":%lld,\"forwards_retained\":%lld,\"dropped_forwards\":%lld,\"worker_phases\":%zu}\n",
                     (long long)seen, (long long)retained, (long long)dropped, used);
        std::fclose(out);
        dump_scratch();
    }
    void dump_scratch() {
        if (!scratch) return;
        FILE* out = std::fopen(scratch_path.c_str(), "w");
        if (!out) { std::perror(scratch_path.c_str()); return; }
        std::fprintf(out, "{\"schema\":1,\"clock\":\"CLOCK_MONOTONIC\",\"cpu_clock\":\"CLOCK_THREAD_CPUTIME_ID\",\"growth_capacity\":%zu}\n", kGrowthCapacity);
        for (size_t i = 0; i < scratch_used; ++i) {
            const auto& r = scratch[i];
            std::fprintf(out, "{\"kind\":\"scratch\",\"forward\":%lld,\"begin\":%lld,\"end\":%lld,\"cpu_begin\":%lld,\"cpu_end\":%lld,\"minflt\":%lld,\"majflt\":%lld,\"nvcsw\":%lld,\"nivcsw\":%lld,\"growths\":%zu,\"initialized_bytes\":%zu,\"capacity_added_bytes\":%zu,\"rows\":%d,\"chunks\":%d}\n",
                (long long)r.serial, (long long)r.begin, (long long)r.end, (long long)r.cpu_begin, (long long)r.cpu_end,
                (long long)r.minflt, (long long)r.majflt, (long long)r.nvcsw, (long long)r.nivcsw,
                r.growths, r.initialized_bytes, r.capacity_added_bytes, r.rows, r.chunks);
        }
        for (size_t i = 0; i < growth_used; ++i) {
            const auto& r = growth[i];
            std::fprintf(out, "{\"kind\":\"growth\",\"forward\":%lld,\"name\":\"%s\",\"begin\":%lld,\"end\":%lld,\"cpu_begin\":%lld,\"cpu_end\":%lld,\"minflt\":%lld,\"majflt\":%lld,\"nvcsw\":%lld,\"nivcsw\":%lld,\"old_size\":%zu,\"size\":%zu,\"old_capacity\":%zu,\"capacity\":%zu,\"item_bytes\":%zu,\"moved\":%s}\n",
                (long long)r.serial, r.name, (long long)r.begin, (long long)r.end, (long long)r.cpu_begin, (long long)r.cpu_end,
                (long long)r.minflt, (long long)r.majflt, (long long)r.nvcsw, (long long)r.nivcsw,
                r.old_size, r.size, r.old_capacity, r.capacity, r.item_bytes, r.moved ? "true" : "false");
        }
        std::fprintf(out, "{\"footer\":true,\"scratch_records\":%zu,\"growth_records\":%zu,\"dropped_growths\":%zu}\n", scratch_used, growth_used, growth_dropped);
        std::fclose(out);
    }
    expert_stream::TraceGate gate;
    static constexpr size_t kGrowthCapacity = 4096;
    std::unique_ptr<ScratchRecord[]> scratch;
    std::unique_ptr<GrowthRecord[]> growth;
    std::string scratch_path;
    size_t scratch_used = 0, growth_used = 0, growth_dropped = 0;
    std::unique_ptr<Record[]> events;
    std::string path;
    size_t capacity = 0, used = 0;
    int64_t min_ns = 0, seen = 0, retained = 0, dropped = 0;
};
// Internal linkage deliberately prevents state merging with another quant DSO.
namespace { inline Buffer& buffer() { static thread_local Buffer b; return b; } }

template<bool On> struct Capture;
template<> struct Capture<false> {
    Capture(int, int, int) {}
    void team_start() {} void worker_start(int) {} void worker_end(int) {}
    void worker_enter(int) {} void worker_ready(int) {}
    void begin(int, int) {} void work_end(int, int) {} void end(int, int) {}
    void add_work(int, int, int64_t) {} void finish() {}
    template<class V> void grow(V& v, size_t n, const char*) { if (v.size() < n) v.resize(n); }
};
template<> struct Capture<true> {
    Capture(int n, int rows, int chunks) : b(buffer()), n(n), rows(rows), chunks(chunks) {
        if (n < 1 || n > kWorkers) throw std::invalid_argument("worker tracing supports 1..64 workers");
        active = b.events && b.gate.enabled();
        forward_begin = active ? clock_ns() : 0;
        if (active && b.scratch) { scratch_cpu_begin = clock_ns(CLOCK_THREAD_CPUTIME_ID); scratch_initial = usage(); }
    }
    template<class V> void grow(V& v, size_t count, const char* name) {
        if (v.size() >= count) return;
        if (!active || !b.scratch) { v.resize(count); return; }
        const auto old_size = v.size(), old_capacity = v.capacity();
        const uintptr_t old_data = reinterpret_cast<uintptr_t>(v.data());
        const auto first = usage();
        const auto begin = clock_ns(), cpu_begin = clock_ns(CLOCK_THREAD_CPUTIME_ID);
        v.resize(count);
        const auto cpu_end = clock_ns(CLOCK_THREAD_CPUTIME_ID), end = clock_ns();
        const auto last = usage();
        ++growths;
        initialized_bytes += (v.size() - old_size) * sizeof(typename V::value_type);
        capacity_added_bytes += (v.capacity() - old_capacity) * sizeof(typename V::value_type);
        if (b.growth_used == b.kGrowthCapacity) { ++b.growth_dropped; return; }
        b.growth[b.growth_used++] = GrowthRecord{name, b.seen + 1, begin, end, cpu_begin, cpu_end,
            delta(first.ru_minflt,last.ru_minflt,first,last), delta(first.ru_majflt,last.ru_majflt,first,last),
            delta(first.ru_nvcsw,last.ru_nvcsw,first,last), delta(first.ru_nivcsw,last.ru_nivcsw,first,last),
            old_size, v.size(), old_capacity, v.capacity(), sizeof(typename V::value_type),
            old_data != reinterpret_cast<uintptr_t>(v.data())};
    }
    void team_start() {
        if (!active) return;
        if (b.scratch) { scratch_cpu_end = clock_ns(CLOCK_THREAD_CPUTIME_ID); scratch_last = usage(); }
        team_begin = clock_ns();
    }
    void worker_enter(int w) {
        if (!active) return;
        auto& s = workers[w];
        s.enter = clock_ns(); s.tid = int(syscall(SYS_gettid));
        if (getrusage(RUSAGE_THREAD, &s.initial)) s.initial.ru_minflt = -1;
    }
    void worker_ready(int w) { if (!active) return; workers[w].ready = clock_ns(); workers[w].cpu = sched_getcpu(); }
    void worker_start(int w) { worker_enter(w); worker_ready(w); }
    void worker_end(int w) {
        if (!active) return;
        auto& s = workers[w]; rusage last{};
        if (s.initial.ru_minflt < 0 || getrusage(RUSAGE_THREAD, &last))
            s.minflt = s.majflt = s.nvcsw = s.nivcsw = -1;
        else {
            s.minflt = last.ru_minflt - s.initial.ru_minflt;
            s.majflt = last.ru_majflt - s.initial.ru_majflt;
            s.nvcsw = last.ru_nvcsw - s.initial.ru_nvcsw;
            s.nivcsw = last.ru_nivcsw - s.initial.ru_nivcsw;
        }
    }
    void begin(int w, int p) {
        if (!active) return;
        auto& s = workers[w].phase[p]; s.begin = clock_ns(); s.cpu_begin = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    }
    void work_end(int w, int p) {
        if (!active) return;
        auto& s = workers[w].phase[p]; s.work_end = clock_ns(); s.cpu_work_end = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    }
    void end(int w, int p) {
        if (!active) return;
        auto& s = workers[w].phase[p]; s.end = clock_ns(); s.cpu_end = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    }
    void add_work(int w, int p, int64_t units) { if (!active) return; workers[w].phase[p].units += units; }
    void finish() {
        const int64_t stop = clock_ns(), serial = ++b.seen;
        if (!active || stop - forward_begin < b.min_ns) return;
        const size_t count = size_t(n) * kActivePhases;
        if (b.used + count > b.capacity) { ++b.dropped; return; } // whole-forward admission
        ++b.retained;
        if (b.scratch) {
            const auto& a = scratch_initial; const auto& z = scratch_last;
            b.scratch[b.scratch_used++] = ScratchRecord{serial, forward_begin, team_begin, scratch_cpu_begin, scratch_cpu_end,
                delta(a.ru_minflt,z.ru_minflt,a,z), delta(a.ru_majflt,z.ru_majflt,a,z),
                delta(a.ru_nvcsw,z.ru_nvcsw,a,z), delta(a.ru_nivcsw,z.ru_nivcsw,a,z),
                growths, initialized_bytes, capacity_added_bytes, rows, chunks};
        }
        for (int w = 0; w < n; ++w) for (int p = 0; p < kPhases; ++p) {
            if (p == 4) continue;
            const auto& s = workers[w];
            b.events[b.used++] = Record{s.phase[p], forward_begin, stop, team_begin, stop, serial,
                s.minflt, s.majflt, s.nvcsw, s.nivcsw, w, p, s.tid, s.cpu, n, rows, chunks, s.enter, s.ready};
        }
    }
    Buffer& b;
    std::array<WorkerStamp, kWorkers> workers{};
    bool active = false;
    rusage scratch_initial{}, scratch_last{};
    int64_t scratch_cpu_begin = 0, scratch_cpu_end = 0;
    size_t growths = 0, initialized_bytes = 0, capacity_added_bytes = 0;
    int n, rows, chunks;
    int64_t forward_begin = 0, team_begin = 0;
};
static_assert(std::is_empty_v<Capture<false>>);
} // namespace sglang::exl3_cpu::worker_trace
