// Layout study (2026-10-22): the DRAM read rate of the EXL3 CPU kernel's weight access pattern, with no arithmetic.
// Each pattern reads exactly the bytes one DSV4.1 forward reads (E experts' gate, up and down 3-bit trellises,
// 13,271,040 B an expert) through the production work split (assign_gemvs<2>: pair-granular contiguous slices of
// the call's GEMVs, gate+up then a barrier then down), from a pool of expert rows far larger than L3, so every call
// reads cold memory. It answers how much of the kernel's time a different persistent layout could recover.
//
//   native-m1    native [k/16][n/16] tiles, the M = 1 grouped traversal (traversal_group: per 128-input block,
//                each of up to 3 groups' 8 rows of 768 B), T0 prefetch of 3 lines two rows ahead
//   native-band  native tiles, the M > 1 register_band walk: P tile pairs (--pairs) down all of k, then the next
//                P pairs, prefetch as above
//   pair-major   tile (kt, nt) at ((nt / 2) * tiles_k + kt) * 2 + nt % 2: a worker's pair range is one contiguous
//                byte range, read front to back (same bytes, same split)
//   contig       each worker reads an equal contiguous slice of every expert row: the streaming ceiling
//
// Build (no torch): g++ -O3 -march=native -fopenmp -std=c++20 layout_stream.cpp -o layout_stream
// Run under numactl --membind=N with --cpus on node N; the pool is first-touched by its pinned workers.
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int H = 5120, I = 2304, kTileBytes = 96, kPairBytes = 192;
constexpr size_t kMatBytes = size_t(H) * I * 3 / 8;  // 4,423,680
constexpr size_t kExpertBytes = 3 * kMatBytes;

struct Mat
{
    int k, n;
};
constexpr Mat kGate{H, I}, kDown{I, H};

std::vector<int> parse_cpus(const std::string& s)
{
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string part = s.substr(i, j - i);
        size_t d = part.find('-');
        if (d == std::string::npos) out.push_back(std::stoi(part));
        else for (int c = std::stoi(part.substr(0, d)); c <= std::stoi(part.substr(d + 1)); ++c) out.push_back(c);
        i = j + 1;
    }
    return out;
}

inline __m512i read_pair(const uint8_t* p, __m512i acc)
{
    acc = _mm512_xor_si512(acc, _mm512_load_si512(p));
    acc = _mm512_xor_si512(acc, _mm512_load_si512(p + 64));
    return _mm512_xor_si512(acc, _mm512_load_si512(p + 128));
}

inline void prefetch_pair(const uint8_t* p)
{
    _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(p + 64), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(p + 128), _MM_HINT_T0);
}

enum class Pattern { NativeM1, NativeBand, PairMajor, Contig };

// One worker's tile range [t0, t1) of one matrix (pair aligned), read the way the pattern reads it.
__m512i read_range(Pattern pat, const uint8_t* m, Mat mat, int t0, int t1, int pairs, bool prefetch, __m512i acc)
{
    const int tiles_k = mat.k / 16, tiles_n = mat.n / 16;
    const size_t row = size_t(tiles_n) * kTileBytes;
    if (pat == Pattern::PairMajor) {
        const uint8_t* a = m + size_t(t0 / 2) * tiles_k * kPairBytes;
        const uint8_t* b = m + size_t(t1 / 2) * tiles_k * kPairBytes;
        for (const uint8_t* p = a; p < b; p += kPairBytes) acc = read_pair(p, acc);
        return acc;
    }
    auto pair_at = [&](int kt, int nt) { return m + size_t(kt) * row + size_t(nt) * kTileBytes; };
    auto column = [&](int n0, int np) {  // register_band<np>: np pairs down all of k
        for (int kt = 0; kt < tiles_k; ++kt)
            for (int b = 0; b < np; ++b) {
                const uint8_t* p = pair_at(kt, n0 + 2 * b);
                if (prefetch && kt + 2 < tiles_k) prefetch_pair(p + 2 * row);
                acc = read_pair(p, acc);
            }
    };
    if (pat == Pattern::NativeBand) {
        for (int n0 = t0; n0 < t1;) {
            const int np = std::min(pairs, (t1 - n0) / 2);
            column(n0, np);
            n0 += 2 * np;
        }
        return acc;
    }
    // NativeM1: compact_pairs at either partial end, whole groups through traversal_tiles (grouped).
    const int a0 = std::min(t1, (t0 + 7) / 8 * 8), a1 = std::max(a0, t1 / 8 * 8);
    if (a0 > t0) column(t0, (a0 - t0) / 2);
    for (int t = a0; t < a1;) {
        const int remaining = (a1 - t) / 8;
        const int groups = std::min(remaining > 4 ? 3 : 4, remaining);
        for (int kb = 0; kb < tiles_k / 8; ++kb)
            for (int g = 0; g < groups; ++g)
                for (int kt = kb * 8; kt < kb * 8 + 8; ++kt)
                    for (int b = 0; b < 4; ++b) {
                        const uint8_t* p = pair_at(kt, t + g * 8 + 2 * b);
                        if (prefetch && kt + 2 < tiles_k) prefetch_pair(p + 2 * row);
                        acc = read_pair(p, acc);
                    }
        t += groups * 8;
    }
    if (t1 > a1) column(a1, (t1 - a1) / 2);
    return acc;
}

// assign_gemvs<2> for total <= 4 * workers (the production calls), else whole-GEMV striding.
template <class F>
void assign(int w, int nw, int total, int tiles_n, F f)
{
    if (total > 4 * nw) {
        for (int j = w; j < total; j += nw) f(j, 0, tiles_n);
        return;
    }
    const int64_t groups = int64_t(total) * (tiles_n / 2);
    const int f0 = int(groups * w / nw) * 2, f1 = int(groups * (w + 1) / nw) * 2;
    for (int j = f0 / tiles_n; j * tiles_n < f1; ++j) f(j, std::max(f0 - j * tiles_n, 0), std::min(f1 - j * tiles_n, tiles_n));
}
}  // namespace

int main(int argc, char** argv)
{
    std::string cpus_s = "28-43", pat_s = "native-m1";
    int experts = 1, pairs = 1, calls = 400, reps = 7, thp = 1, prefetch = 1;
    size_t pool_mb = 3072;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto v = [&](const char* k) { return a.substr(strlen(k)); };
        if (a.starts_with("--cpus=")) cpus_s = v("--cpus=");
        else if (a.starts_with("--pattern=")) pat_s = v("--pattern=");
        else if (a.starts_with("--experts=")) experts = std::stoi(v("--experts="));
        else if (a.starts_with("--pairs=")) pairs = std::stoi(v("--pairs="));
        else if (a.starts_with("--calls=")) calls = std::stoi(v("--calls="));
        else if (a.starts_with("--reps=")) reps = std::stoi(v("--reps="));
        else if (a.starts_with("--pool-mb=")) pool_mb = std::stoul(v("--pool-mb="));
        else if (a.starts_with("--thp=")) thp = std::stoi(v("--thp="));
        else if (a.starts_with("--prefetch=")) prefetch = std::stoi(v("--prefetch="));
        else { fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    Pattern pat = pat_s == "native-m1" ? Pattern::NativeM1 : pat_s == "native-band" ? Pattern::NativeBand
                : pat_s == "pair-major" ? Pattern::PairMajor : pat_s == "contig" ? Pattern::Contig : (Pattern)-1;
    if ((int)pat < 0) { fprintf(stderr, "bad pattern\n"); return 2; }
    const std::vector<int> cpus = parse_cpus(cpus_s);
    const int nw = int(cpus.size());
    const size_t pool_rows = pool_mb * (1 << 20) / kExpertBytes;
    const size_t bytes = pool_rows * kExpertBytes;
    void* base = mmap(nullptr, bytes + (2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    uint8_t* pool = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(base) + (2 << 20) - 1) & ~uintptr_t((2 << 20) - 1));
    madvise(pool, bytes, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);

    std::vector<double> call_us(size_t(reps) * calls);
    long long sink = 0;
    omp_set_num_threads(nw);
#pragma omp parallel reduction(^ : sink)
    {
        const int w = omp_get_thread_num();
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[w], &set);
        if (sched_setaffinity(0, sizeof set, &set)) { perror("affinity"); std::exit(1); }
        // First touch by the workers, slice by slice (so every node-local page is written once).
        for (size_t r = 0; r < pool_rows; ++r) {
            uint8_t* row = pool + r * kExpertBytes;
            const size_t lo = kExpertBytes * w / nw / 4096 * 4096, hi = w + 1 == nw ? kExpertBytes : kExpertBytes * (w + 1) / nw / 4096 * 4096;
            for (size_t o = lo; o < hi; o += 64) row[o] = uint8_t(r * 131 + o);
        }
        __m512i acc = _mm512_setzero_si512();
        size_t next = 0;
#pragma omp barrier
        for (int rep = 0; rep < reps; ++rep)
            for (int c = 0; c < calls; ++c) {
                std::chrono::steady_clock::time_point t0;
#pragma omp barrier
                if (w == 0) t0 = std::chrono::steady_clock::now();
                const size_t first = next;
                next = (next + experts) % (pool_rows - experts);
                auto expert = [&](int e) { return pool + (first + e) * kExpertBytes; };
                if (pat == Pattern::Contig) {
                    for (int e = 0; e < experts; ++e) {
                        const size_t lo = kExpertBytes * w / nw / 64 * 64, hi = kExpertBytes * (w + 1) / nw / 64 * 64;
                        for (size_t o = lo; o + 64 <= hi; o += 64)
                            acc = _mm512_xor_si512(acc, _mm512_load_si512(expert(e) + o));
                    }
                } else {
                    assign(w, nw, 2 * experts, I / 16, [&](int j, int a, int b) {
                        acc = read_range(pat, expert(j / 2) + (j % 2) * kMatBytes, kGate, a, b, pairs, prefetch, acc);
                    });
#pragma omp barrier
                    assign(w, nw, experts, H / 16, [&](int j, int a, int b) {
                        acc = read_range(pat, expert(j) + 2 * kMatBytes, kDown, a, b, pairs, prefetch, acc);
                    });
                }
#pragma omp barrier
                if (w == 0)
                    call_us[size_t(rep) * calls + c] =
                        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            }
        alignas(64) long long lanes[8];
        _mm512_store_si512(lanes, acc);
        for (long long x : lanes) sink ^= x;
    }
    // Per rep: median call; report the median and spread of the rep medians (first rep dropped as warmup if > 1).
    std::vector<double> med;
    for (int rep = reps > 1 ? 1 : 0; rep < reps; ++rep) {
        std::vector<double> v(call_us.begin() + size_t(rep) * calls, call_us.begin() + size_t(rep + 1) * calls);
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        med.push_back(v[v.size() / 2]);
    }
    std::sort(med.begin(), med.end());
    const double gb = double(experts) * kExpertBytes / 1e9;
    const double m = med[med.size() / 2];
    printf("pattern=%s experts=%d pairs=%d workers=%d thp=%d prefetch=%d pool_rows=%zu median_us=%.1f min_us=%.1f "
           "max_us=%.1f GBps=%.2f (min..max %.2f..%.2f) sink=%lld\n",
           pat_s.c_str(), experts, pairs, nw, thp, prefetch, pool_rows, m, med.front(), med.back(), gb / (m * 1e-6),
           gb / (med.back() * 1e-6), gb / (med.front() * 1e-6), sink & 1);
    return 0;
}
