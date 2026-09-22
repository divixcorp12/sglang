// Cross-kernel visibility of host rewrites through ld.global.nc: see NC_VISIBILITY.md (pre-registered).
//
//   nvcc -O2 -std=c++17 -arch=sm_120 -o nc_visibility nc_visibility.cu
//   ./nc_visibility [--iters N] [--char-iters N] [--bw-gib G] [--quick]   (JSON, one object per line)
//
// The design, the cells, the sizes and the decision rule are fixed in NC_VISIBILITY.md before this program
// was written. Nothing here interprets a result; it prints counts.

#include <cuda_runtime.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                                  \
    }                                                                                                \
  } while (0)

namespace {

constexpr int kNC = 0, kCV = 1, kSYS = 2, kPLAIN = 3;
const char* kVariantName[] = {"nc", "cv", "sys", "plain"};

constexpr int kGrid = 8;     // kExpertTransferGridSize
constexpr int kBlock = 256;  // kExpertTransferBlockSize
constexpr size_t kRows = 4;
constexpr size_t kRowBytes = 128 * 1024;
constexpr size_t kRegionBytes = kRows * kRowBytes;  // 512 KiB
constexpr size_t kRegionWords = kRegionBytes / 8;   // 65,536
constexpr size_t kRegionUnits = kRegionBytes / 16;

// ---- device ----

template <int V>
__device__ __forceinline__ void load16(const uint8_t* src, uint64_t& lo, uint64_t& hi) {
  if constexpr (V == kNC) {
    asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(src) : "memory");
  } else if constexpr (V == kCV) {
    asm volatile("ld.global.cv.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(src) : "memory");
  } else if constexpr (V == kSYS) {
    asm volatile("ld.relaxed.sys.global.u64 %0,[%1];" : "=l"(lo) : "l"(src) : "memory");
    asm volatile("ld.relaxed.sys.global.u64 %0,[%1];" : "=l"(hi) : "l"(src + 8) : "memory");
  } else {
    asm volatile("ld.global.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(src) : "memory");
  }
}

template <int V>
__device__ __forceinline__ uint64_t load8(const uint64_t* src) {
  uint64_t v;
  if constexpr (V == kNC) {
    asm volatile("ld.global.nc.u64 %0,[%1];" : "=l"(v) : "l"(src) : "memory");
  } else if constexpr (V == kCV) {
    asm volatile("ld.global.cv.u64 %0,[%1];" : "=l"(v) : "l"(src) : "memory");
  } else if constexpr (V == kSYS) {
    asm volatile("ld.relaxed.sys.global.u64 %0,[%1];" : "=l"(v) : "l"(src) : "memory");
  } else {
    asm volatile("ld.global.u64 %0,[%1];" : "=l"(v) : "l"(src) : "memory");
  }
  return v;
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}

// The production copy shape: 16-byte units, ld from host memory, st.global.cg to the destination.
template <int V>
__global__ void copy_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t units) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t u = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; u < units; u += stride) {
    uint64_t lo, hi;
    load16<V>(src + 16 * u, lo, hi);
    asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(dst + 16 * u), "l"(lo), "l"(hi) : "memory");
  }
}

// counters: [0] words checked, [1] stale (tag - 1), [2] other mismatches, [3] first failing iteration.
__global__ void check_kernel(
    const uint64_t* __restrict__ dst, size_t words, const uint64_t* __restrict__ d_tag, unsigned long long* counters,
    uint64_t expect_offset) {
  const uint64_t tag = *d_tag + expect_offset;
  unsigned long long checked = 0, stale = 0, other = 0;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t w = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; w < words; w += stride) {
    const uint64_t got = dst[w];
    const uint64_t want = (tag << 32) | static_cast<uint32_t>(w);
    ++checked;
    if (got != want) {
      if ((got >> 32) == tag - 1) {
        ++stale;
      } else {
        ++other;
      }
    }
  }
  if (checked) atomicAdd(&counters[0], checked);
  if (stale) atomicAdd(&counters[1], stale);
  if (other) atomicAdd(&counters[2], other);
  if ((stale || other) && expect_offset == 0) atomicMin(&counters[3], static_cast<unsigned long long>(*d_tag));
}

__global__ void bump_kernel(uint64_t* d_tag) { *d_tag += 1; }

// Streams a device buffer: an L2 thrash (read-modify-write) or a busy kernel that keeps the GPU occupied.
__global__ void stream_kernel(uint64_t* buf, size_t words, int passes) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    for (size_t w = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; w < words; w += stride) buf[w] += 1;
  }
}

// C1: a bounded single-thread spin on a host-written flag. The kernel announces it started, then polls.
template <int V>
__global__ void spin_kernel(const uint64_t* flag, volatile uint32_t* started, uint64_t timeout_ns, uint64_t* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const uint64_t begin = global_ns();
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(started), "r"(1u) : "memory");
  uint64_t v = 0;
  while (v == 0 && global_ns() - begin < timeout_ns) v = load8<V>(flag);
  out[0] = v;
  out[1] = global_ns() - begin;
}

// C2: within one kernel, does a second .nc read of a word just overwritten return the old value?
template <int V>
__global__ void intra_kernel(uint64_t* x, int trials, unsigned long long* stale) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  unsigned long long s = 0;
  for (int i = 0; i < trials; ++i) {
    const uint64_t a = load8<V>(x);
    asm volatile("st.global.cg.u64 [%0],%1;" ::"l"(x), "l"(a + 1) : "memory");
    __threadfence();
    const uint64_t b = load8<V>(x);
    if (b == a) ++s;
  }
  *stale = s;
}

// ---- host ----

double now_s() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

uint8_t* host_alloc(size_t bytes) {
  void* p = nullptr;
  if (posix_memalign(&p, 2u << 20, bytes) != 0) {
    std::fprintf(stderr, "posix_memalign failed\n");
    std::exit(2);
  }
  std::memset(p, 0, bytes);  // touch: registration and the first fill must not fault
  CK(cudaHostRegister(p, bytes, 0));  // as _cuda_host_register does
  return static_cast<uint8_t*>(p);
}

void fill_regular(uint64_t* dst, uint64_t tag) {
  for (size_t w = 0; w < kRegionWords; ++w) dst[w] = (tag << 32) | static_cast<uint32_t>(w);
  _mm_sfence();
}

void fill_nt(uint64_t* dst, uint64_t tag) {
  for (size_t w = 0; w < kRegionWords; w += 2) {
    const __m128i v = _mm_set_epi64x(
        static_cast<long long>((tag << 32) | static_cast<uint32_t>(w + 1)), static_cast<long long>((tag << 32) | static_cast<uint32_t>(w)));
    _mm_stream_si128(reinterpret_cast<__m128i*>(dst + w), v);
  }
  _mm_sfence();
}

struct Buffers {
  uint8_t* host = nullptr;
  uint8_t* dev = nullptr;
  uint64_t* d_tag = nullptr;
  unsigned long long* counters = nullptr;  // device, 4 words
  uint64_t* thrash = nullptr;
  size_t thrash_words = 0;
  uint64_t* busy = nullptr;
  size_t busy_words = 0;
};

template <int V>
void launch_trio(const Buffers& b, cudaStream_t s) {
  copy_kernel<V><<<kGrid, kBlock, 0, s>>>(b.host, b.dev, kRegionUnits);
  check_kernel<<<kGrid, kBlock, 0, s>>>(reinterpret_cast<const uint64_t*>(b.dev), kRegionWords, b.d_tag, b.counters, 0);
  bump_kernel<<<1, 1, 0, s>>>(b.d_tag);
}

void launch_trio_v(int v, const Buffers& b, cudaStream_t s) {
  switch (v) {
    case kNC: launch_trio<kNC>(b, s); break;
    case kCV: launch_trio<kCV>(b, s); break;
    case kSYS: launch_trio<kSYS>(b, s); break;
    default: launch_trio<kPLAIN>(b, s); break;
  }
}

struct CellResult {
  unsigned long long iters = 0, checked = 0, stale = 0, other = 0, first_bad = 0;
  double seconds = 0;
};

const char* kModeName[] = {"boundary", "graph", "thrash", "concurrent"};
enum Mode { kBoundary = 0, kGraph = 1, kThrash = 2, kConcurrent = 3 };

CellResult run_cell(int variant, int mode, bool nt, unsigned long long iters, Buffers& b, cudaStream_t s) {
  CK(cudaMemset(b.counters, 0, 4 * sizeof(unsigned long long)));
  const unsigned long long none = ~0ull;
  CK(cudaMemcpy(b.counters + 3, &none, sizeof(none), cudaMemcpyHostToDevice));
  const uint64_t one = 1;
  CK(cudaMemcpy(b.d_tag, &one, sizeof(one), cudaMemcpyHostToDevice));
  auto* host_words = reinterpret_cast<uint64_t*>(b.host);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (mode == kGraph) {
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    launch_trio_v(variant, b, s);
    CK(cudaStreamEndCapture(s, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));
  }
  // Prime: the first iteration reads bytes no earlier kernel has seen; cells measure the steady alternation.
  const double t0 = now_s();
  for (unsigned long long i = 0; i < iters; ++i) {
    const uint64_t tag = i + 1;
    if (mode == kThrash) stream_kernel<<<64, kBlock, 0, s>>>(b.thrash, b.thrash_words, 1);
    if (mode == kConcurrent) stream_kernel<<<64, kBlock, 0, s>>>(b.busy, b.busy_words, 8);
    if (nt) fill_nt(host_words, tag); else fill_regular(host_words, tag);
    if (mode == kGraph) {
      CK(cudaGraphLaunch(exec, s));
    } else {
      launch_trio_v(variant, b, s);
    }
    CK(cudaStreamSynchronize(s));
  }
  CellResult r;
  r.seconds = now_s() - t0;
  unsigned long long c[4];
  CK(cudaMemcpy(c, b.counters, sizeof(c), cudaMemcpyDeviceToHost));
  r.iters = iters;
  r.checked = c[0];
  r.stale = c[1];
  r.other = c[2];
  r.first_bad = c[3];
  if (exec) CK(cudaGraphExecDestroy(exec));
  if (graph) CK(cudaGraphDestroy(graph));
  return r;
}

void print_cell(const char* kind, int variant, const char* mode, bool nt, const CellResult& r) {
  std::printf(
      "{\"kind\":\"%s\",\"variant\":\"%s\",\"mode\":\"%s\",\"host_store\":\"%s\",\"iters\":%llu,\"words_checked\":%llu,"
      "\"expected_words\":%llu,\"stale\":%llu,\"other\":%llu,\"first_bad_iter\":%lld,\"seconds\":%.2f}\n",
      kind, kVariantName[variant], mode, nt ? "nt" : "regular", r.iters, r.checked,
      static_cast<unsigned long long>(r.iters) * kRegionWords, r.stale, r.other,
      r.first_bad == ~0ull ? -1ll : static_cast<long long>(r.first_bad), r.seconds);
  std::fflush(stdout);
}

// C-self: the harness must be able to see staleness. The host writes tag t; the checker is told to expect t + 1.
bool self_check(Buffers& b, cudaStream_t s) {
  CK(cudaMemset(b.counters, 0, 4 * sizeof(unsigned long long)));
  const uint64_t two = 2;
  CK(cudaMemcpy(b.d_tag, &two, sizeof(two), cudaMemcpyHostToDevice));
  fill_regular(reinterpret_cast<uint64_t*>(b.host), 2);  // written with tag 2; the checker below expects 3
  copy_kernel<kNC><<<kGrid, kBlock, 0, s>>>(b.host, b.dev, kRegionUnits);
  check_kernel<<<kGrid, kBlock, 0, s>>>(reinterpret_cast<const uint64_t*>(b.dev), kRegionWords, b.d_tag, b.counters, 1);
  CK(cudaStreamSynchronize(s));
  unsigned long long c[4];
  CK(cudaMemcpy(c, b.counters, sizeof(c), cudaMemcpyDeviceToHost));
  const bool ok = c[0] == kRegionWords && c[1] == kRegionWords && c[2] == 0;
  std::printf(
      "{\"kind\":\"c_self\",\"words_checked\":%llu,\"stale\":%llu,\"other\":%llu,\"expected_words\":%zu,\"pass\":%s}\n",
      c[0], c[1], c[2], kRegionWords, ok ? "true" : "false");
  std::fflush(stdout);
  return ok;
}

template <int V>
void control_c1(uint8_t* flag_host, int repeats) {
  uint32_t* started = nullptr;
  CK(cudaHostAlloc(&started, 4096, cudaHostAllocMapped));
  uint64_t* out = nullptr;
  CK(cudaMalloc(&out, 16));
  std::vector<double> latency_us;
  int saw = 0;
  auto* flag = reinterpret_cast<volatile uint64_t*>(flag_host);
  for (int r = 0; r < repeats; ++r) {
    *flag = 0;
    *started = 0;
    _mm_sfence();
    spin_kernel<V><<<1, 1>>>(reinterpret_cast<const uint64_t*>(flag_host), started, 20'000'000ull, out);
    while (__atomic_load_n(started, __ATOMIC_ACQUIRE) == 0) {}
    const double until = now_s() + 0.002;
    while (now_s() < until) _mm_pause();
    const double set_at = now_s();
    *flag = 1;
    _mm_sfence();
    CK(cudaDeviceSynchronize());
    uint64_t h[2];
    CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost));
    if (h[0] != 0) {
      ++saw;
      latency_us.push_back((now_s() - set_at) * 1e6);  // upper bound: includes the sync
    }
  }
  std::sort(latency_us.begin(), latency_us.end());
  std::printf(
      "{\"kind\":\"c1_host_flag_spin\",\"variant\":\"%s\",\"repeats\":%d,\"saw_flag\":%d,\"median_us\":%.1f}\n",
      kVariantName[V], repeats, saw, latency_us.empty() ? -1.0 : latency_us[latency_us.size() / 2]);
  std::fflush(stdout);
  CK(cudaFree(out));
  CK(cudaFreeHost(started));
}

template <int V>
void control_c2(int trials) {
  uint64_t* x = nullptr;
  unsigned long long* stale = nullptr;
  CK(cudaMalloc(&x, 8));
  CK(cudaMemset(x, 0, 8));
  CK(cudaMalloc(&stale, 8));
  intra_kernel<V><<<1, 1>>>(x, trials, stale);
  CK(cudaDeviceSynchronize());
  unsigned long long s = 0;
  CK(cudaMemcpy(&s, stale, 8, cudaMemcpyDeviceToHost));
  std::printf(
      "{\"kind\":\"c2_intra_kernel_second_read_old\",\"variant\":\"%s\",\"trials\":%d,\"second_read_returned_old\":%llu}\n",
      kVariantName[V], trials, s);
  std::fflush(stdout);
  CK(cudaFree(x));
  CK(cudaFree(stale));
}

// Bandwidth over a working set 8x L2: nc and cv at two geometries, and cudaMemcpyAsync as the reference.
template <int V>
double bandwidth(uint8_t* host, uint8_t* dev, size_t bytes, int grid, int reps, std::vector<double>* all) {
  cudaEvent_t a, e;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&e));
  std::vector<double> gbps;
  for (int r = 0; r < reps + 1; ++r) {
    CK(cudaEventRecord(a));
    copy_kernel<V><<<grid, kBlock>>>(host, dev, bytes / 16);
    CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, e));
    if (r > 0) gbps.push_back(static_cast<double>(bytes) / (ms * 1e-3) / 1e9);
  }
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(e));
  std::sort(gbps.begin(), gbps.end());
  if (all) *all = gbps;
  return gbps[gbps.size() / 2];
}

void bandwidth_cell(double gib, int reps) {
  const size_t bytes = static_cast<size_t>(gib * 1024 * 1024 * 1024);
  uint8_t* host = host_alloc(bytes);
  uint8_t* dev = nullptr;
  CK(cudaMalloc(&dev, bytes));
  for (size_t i = 0; i < bytes / 8; i += 512) reinterpret_cast<uint64_t*>(host)[i] = i;  // page-touch, content irrelevant
  struct Row { const char* name; int grid; double med, mn, mx; };
  std::vector<Row> rows;
  auto add = [&](const char* name, int grid, auto fn) {
    std::vector<double> all;
    const double med = fn(&all);
    rows.push_back({name, grid, med, all.front(), all.back()});
  };
  for (int grid : {kGrid, 128}) {
    add("nc", grid, [&](std::vector<double>* v) { return bandwidth<kNC>(host, dev, bytes, grid, reps, v); });
    add("cv", grid, [&](std::vector<double>* v) { return bandwidth<kCV>(host, dev, bytes, grid, reps, v); });
  }
  // cudaMemcpyAsync reference
  {
    cudaEvent_t a, e;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&e));
    std::vector<double> gbps;
    for (int r = 0; r < reps + 1; ++r) {
      CK(cudaEventRecord(a));
      CK(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice));
      CK(cudaEventRecord(e));
      CK(cudaEventSynchronize(e));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, a, e));
      if (r > 0) gbps.push_back(static_cast<double>(bytes) / (ms * 1e-3) / 1e9);
    }
    std::sort(gbps.begin(), gbps.end());
    rows.push_back({"cudaMemcpyAsync", 0, gbps[gbps.size() / 2], gbps.front(), gbps.back()});
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(e));
  }
  for (const auto& r : rows) {
    std::printf(
        "{\"kind\":\"bandwidth\",\"variant\":\"%s\",\"grid\":%d,\"bytes\":%zu,\"reps\":%d,\"median_gb_s\":%.2f,"
        "\"min_gb_s\":%.2f,\"max_gb_s\":%.2f,\"pcie_gen3_x16_ceiling_gb_s\":15.75,\"above_ceiling\":%s}\n",
        r.name, r.grid, bytes, reps, r.med, r.mn, r.mx, r.mx > 15.75 ? "true" : "false");
  }
  std::fflush(stdout);
  CK(cudaFree(dev));
  CK(cudaHostUnregister(host));
  std::free(host);
}

}  // namespace

int main(int argc, char** argv) {
  unsigned long long iters = 200000, char_iters = 50000;
  double bw_gib = 1.0;
  int c1_repeats = 100, c2_trials = 100000;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--char-iters" && i + 1 < argc) char_iters = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--bw-gib" && i + 1 < argc) bw_gib = std::atof(argv[++i]);
    else if (a == "--quick") quick = true;
  }
  if (quick) {
    iters = 2000;
    char_iters = 1000;
    bw_gib = 0.25;
    c1_repeats = 5;
    c2_trials = 1000;
  }
  int dev_id = 0;
  CK(cudaGetDevice(&dev_id));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev_id));
  int rt = 0, drv = 0;
  CK(cudaRuntimeGetVersion(&rt));
  CK(cudaDriverGetVersion(&drv));
  std::printf(
      "{\"kind\":\"env\",\"device\":\"%s\",\"sm\":\"%d.%d\",\"l2_bytes\":%d,\"runtime\":%d,\"driver\":%d,"
      "\"region_bytes\":%zu,\"iters\":%llu,\"char_iters\":%llu,\"quick\":%s}\n",
      prop.name, prop.major, prop.minor, prop.l2CacheSize, rt, drv, kRegionBytes, iters, char_iters,
      quick ? "true" : "false");
  std::fflush(stdout);

  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  Buffers b;
  b.host = host_alloc(kRegionBytes);
  CK(cudaMalloc(&b.dev, kRegionBytes));
  CK(cudaMalloc(&b.d_tag, 8));
  CK(cudaMalloc(&b.counters, 4 * sizeof(unsigned long long)));
  b.thrash_words = (256ull << 20) / 8;
  CK(cudaMalloc(&b.thrash, b.thrash_words * 8));
  CK(cudaMemset(b.thrash, 0, b.thrash_words * 8));
  b.busy_words = (64ull << 20) / 8;
  CK(cudaMalloc(&b.busy, b.busy_words * 8));
  CK(cudaMemset(b.busy, 0, b.busy_words * 8));

  const bool gate = self_check(b, s);
  int rc = gate ? 0 : 3;

  control_c2<kNC>(c2_trials);
  control_c2<kCV>(c2_trials);
  uint8_t* flag_host = host_alloc(4096);
  control_c1<kNC>(flag_host, c1_repeats);
  control_c1<kCV>(flag_host, c1_repeats);
  control_c1<kSYS>(flag_host, c1_repeats);
  control_c1<kPLAIN>(flag_host, c1_repeats);

  for (int variant : {kNC, kCV}) {
    for (int mode : {kBoundary, kGraph, kThrash, kConcurrent}) {
      for (bool nt : {false, true}) {
        const CellResult r = run_cell(variant, mode, nt, iters, b, s);
        print_cell("primary", variant, kModeName[mode], nt, r);
      }
    }
  }
  for (int variant : {kSYS, kPLAIN}) {
    const CellResult r = run_cell(variant, kBoundary, false, char_iters, b, s);
    print_cell("characterization", variant, kModeName[kBoundary], false, r);
  }
  bandwidth_cell(bw_gib, 10);
  std::printf("{\"kind\":\"done\",\"c_self_pass\":%s}\n", gate ? "true" : "false");
  return rc;
}
