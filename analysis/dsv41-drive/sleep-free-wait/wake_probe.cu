// Wake-latency probe: how long after a host thread writes a pinned word does the next captured kernel start?
//
// Every variant is a captured graph: K0 -> [waiter] -> K2.
//   K0  clears release=0, records T0 = %globaltimer, then stores armed=gen (st.release.sys, pinned host).
//   A   1-block spin shaped like copy_wait (row_copy_kernels.cuh ~640 on master): per iteration a deadline check on
//       %globaltimer, two extra ld.acquire.sys "fatal/shutdown" words, __nanosleep(256), then ld.acquire.sys of the
//       release word. Records its iterations and spin span.
//   B   cuStreamWaitValue32(release, 1, EQ) as a graph mem-op node.
//   B'  as B with CU_STREAM_WAIT_VALUE_FLUSH (only if CAN_FLUSH_REMOTE_WRITES).
//   N   no waiter: K0 -> K2, the floor.
//   K2  records T2 = %globaltimer and the release word it sees.
// An echo thread pinned to one CPU spins on `armed`; on a new value it stores release=1 and nothing else.
// Latency = T2 - T0 (one GPU clock, no host alignment). The host part is identical across variants.
//
// Replays are interleaved one per variant per round, with the order rotated each round. The main thread waits for
// the echo of each replay before launching the next, so no stale release write can land in a later replay.
//
// Usage: wake_probe <replays per variant> <warmup per variant> <echo cpu> <main cpu> <out dir> <echo delay ns>
// A nonzero delay makes the echo busy-wait that long after seeing `armed`, so the waiter is parked when release lands.

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                     \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) {                                                                      \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));        \
      exit(1);                                                                                    \
    }                                                                                             \
  } while (0)
#define CU(x)                                                                                     \
  do {                                                                                            \
    CUresult r_ = (x);                                                                            \
    if (r_ != CUDA_SUCCESS) {                                                                     \
      const char* n_ = nullptr;                                                                   \
      cuGetErrorName(r_, &n_);                                                                    \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, n_ ? n_ : "?");                \
      exit(1);                                                                                    \
    }                                                                                             \
  } while (0)

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Pinned host words, each on its own 128-byte line.
struct HostWords {
  alignas(128) uint32_t armed;
  alignas(128) uint32_t release;
  alignas(128) uint32_t fatal;
  alignas(128) uint32_t shutdown;
};

struct Slots {  // device memory, one per variant
  uint64_t* t0;
  uint64_t* t2;
  uint32_t* seen;      // release as K2 observed it
  uint64_t* iters;     // A only
  uint64_t* spin_ns;   // A only
  uint32_t* timeouts;  // A only
  uint32_t* idx;       // replay index counter
};

__global__ void k0(uint32_t* gen, Slots s, HostWords* hw) {
  if (threadIdx.x != 0) return;
  const uint32_t g = ++gen[0];
  st_relaxed_sys(&hw->release, 0u);
  __threadfence_system();
  s.t0[s.idx[0]] = gtimer();
  st_release_sys(&hw->armed, g);
}

// Shaped like copy_wait's loop: deadline, two abort words, nanosleep, reload.
__global__ void spin_wait(Slots s, HostWords* hw, uint64_t timeout_ns) {
  if (threadIdx.x != 0) return;
  const uint64_t start = gtimer();
  const uint64_t deadline = start + timeout_ns;
  uint64_t n = 0;
  uint32_t word = ld_acquire_sys(&hw->release);
  while (word != 1u) {
    if (static_cast<int64_t>(gtimer() - deadline) >= 0) {
      s.timeouts[0] += 1;
      break;
    }
    if (ld_acquire_sys(&hw->fatal) != 0 || ld_acquire_sys(&hw->shutdown) != 0) break;
    __nanosleep(256);
    word = ld_acquire_sys(&hw->release);
    ++n;
  }
  const uint32_t i = s.idx[0];
  s.iters[i] = n;
  s.spin_ns[i] = gtimer() - start;
}

__global__ void k2(Slots s, HostWords* hw) {
  if (threadIdx.x != 0) return;
  const uint64_t t = gtimer();
  const uint32_t i = s.idx[0];
  s.t2[i] = t;
  s.seen[i] = ld_acquire_sys(&hw->release);
  s.idx[0] = i + 1;
}

// %globaltimer resolution: smallest nonzero step seen by a tight loop.
__global__ void timer_res(uint64_t* out) {
  uint64_t prev = gtimer(), min_step = ~0ull, steps = 0;
  const uint64_t end = prev + 2000000;  // 2 ms
  while (true) {
    uint64_t t = gtimer();
    if (t != prev) {
      min_step = min(min_step, t - prev);
      ++steps;
      prev = t;
      if (t >= end) break;
    }
  }
  out[0] = min_step;
  out[1] = steps;
}

enum Kind { kA, kB, kBf, kN };
static const char* kNames[] = {"A_spin", "B_waitvalue", "Bf_waitvalue_flush", "N_nowait"};

static std::string g_capture_path = "unset";

// Plain capture first; if the driver refuses, fall back to the sleep-free branch's explicit mem-op node.
static void enqueue_wait(cudaStream_t stream, CUdeviceptr addr, unsigned flags) {
  CUresult r = cuStreamWaitValue32(stream, addr, 1u, CU_STREAM_WAIT_VALUE_EQ | flags);
  if (r == CUDA_SUCCESS) {
    if (g_capture_path == "unset") g_capture_path = "plain cuStreamWaitValue32 under stream capture";
    return;
  }
  const char* name = nullptr;
  cuGetErrorName(r, &name);
  fprintf(stderr, "plain capture of cuStreamWaitValue32 failed: %s\n", name ? name : "?");
  exit(2);  // the capture is invalidated; rerun with WAKE_PROBE_EXPLICIT_NODE=1
}

static void enqueue_wait_explicit(cudaStream_t stream, CUdeviceptr addr, unsigned flags) {
  CUstreamCaptureStatus status;
  CUcontext ctx;
  CU(cuStreamGetCtx(stream, &ctx));
  CUgraph graph;
  const CUgraphNode* deps = nullptr;
  const CUgraphEdgeData* edges = nullptr;
  size_t ndeps = 0;
  CU(cuStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps, &edges, &ndeps));
  CUstreamBatchMemOpParams op{};
  op.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
  op.waitValue.address = addr;
  op.waitValue.value = 1u;
  op.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ | flags;
  CUgraphNodeParams p{};
  p.type = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
  p.memOp.ctx = ctx;
  p.memOp.count = 1;
  p.memOp.paramArray = &op;
  CUgraphNode node;
  CU(cuGraphAddNode(&node, graph, deps, edges, ndeps, &p));
  CU(cuStreamUpdateCaptureDependencies(stream, &node, nullptr, 1, CU_STREAM_SET_CAPTURE_DEPENDENCIES));
  g_capture_path = "explicit batch-mem-op graph node (stream_wait.cuh pattern)";
}

static void pin(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    fprintf(stderr, "cannot pin to cpu %d\n", cpu);
    exit(1);
  }
}

static double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  const double k = (v.size() - 1) * q;
  const size_t lo = static_cast<size_t>(k), hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (k - lo);
}

int main(int argc, char** argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s replays warmup echo_cpu main_cpu out_dir echo_delay_ns\n", argv[0]);
    return 1;
  }
  const int replays = atoi(argv[1]), warmup = atoi(argv[2]), echo_cpu = atoi(argv[3]), main_cpu = atoi(argv[4]);
  const std::string out_dir = argv[5];
  const long long delay_ns = atoll(argv[6]);
  const bool explicit_node = getenv("WAKE_PROBE_EXPLICIT_NODE") && atoi(getenv("WAKE_PROBE_EXPLICIT_NODE"));
  pin(main_cpu);

  CK(cudaSetDevice(0));
  CK(cudaFree(0));
  CUdevice dev;
  CU(cuCtxGetDevice(&dev));
  int can_flush = 0, driver = 0;
  CU(cuDeviceGetAttribute(&can_flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, dev));
  CK(cudaDriverGetVersion(&driver));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("echo delay %lld ns\n", delay_ns);
  printf("device %s sm_%d%d, driver API %d, CAN_FLUSH_REMOTE_WRITES=%d\n", prop.name, prop.major, prop.minor, driver,
         can_flush);

  // Timer resolution.
  uint64_t* d_res;
  CK(cudaMalloc(&d_res, 16));
  timer_res<<<1, 1>>>(d_res);
  uint64_t res[2];
  CK(cudaMemcpy(res, d_res, 16, cudaMemcpyDeviceToHost));
  printf("globaltimer: min nonzero step %llu ns, %llu distinct values in 2 ms\n", (unsigned long long)res[0],
         (unsigned long long)res[1]);

  HostWords* hw;
  CK(cudaHostAlloc(&hw, sizeof(HostWords), cudaHostAllocMapped | cudaHostAllocPortable));
  memset(hw, 0, sizeof(HostWords));
  HostWords* hw_dev;
  CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&hw_dev), hw, 0));
  const CUdeviceptr release_dev = reinterpret_cast<CUdeviceptr>(&hw_dev->release);

  uint32_t* d_gen;
  CK(cudaMalloc(&d_gen, 4));
  CK(cudaMemset(d_gen, 0, 4));

  std::vector<Kind> kinds = {kA, kB, kN};
  if (can_flush) kinds.insert(kinds.begin() + 2, kBf);
  const int total = replays + warmup;
  std::vector<Slots> slots(4);
  for (Kind k : kinds) {
    Slots& s = slots[k];
    CK(cudaMalloc(&s.t0, total * 8));
    CK(cudaMalloc(&s.t2, total * 8));
    CK(cudaMalloc(&s.seen, total * 4));
    CK(cudaMalloc(&s.iters, total * 8));
    CK(cudaMalloc(&s.spin_ns, total * 8));
    CK(cudaMalloc(&s.timeouts, 4));
    CK(cudaMalloc(&s.idx, 4));
    CK(cudaMemset(s.iters, 0, total * 8));
    CK(cudaMemset(s.spin_ns, 0, total * 8));
    CK(cudaMemset(s.timeouts, 0, 4));
    CK(cudaMemset(s.idx, 0, 4));
  }

  cudaStream_t stream;
  CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::vector<cudaGraphExec_t> execs(4, nullptr);
  for (Kind k : kinds) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    k0<<<1, 32, 0, stream>>>(d_gen, slots[k], hw_dev);
    if (k == kA) spin_wait<<<1, 32, 0, stream>>>(slots[k], hw_dev, 2000000000ull);
    if (k == kB || k == kBf) {
      const unsigned flags = k == kBf ? CU_STREAM_WAIT_VALUE_FLUSH : 0u;
      if (explicit_node) enqueue_wait_explicit(stream, release_dev, flags);
      else enqueue_wait(stream, release_dev, flags);
    }
    k2<<<1, 32, 0, stream>>>(slots[k], hw_dev);
    CK(cudaStreamEndCapture(stream, &g));
    size_t nodes = 0;
    CK(cudaGraphGetNodes(g, nullptr, &nodes));
    CK(cudaGraphInstantiate(&execs[k], g, 0));
    printf("graph %-20s nodes=%zu\n", kNames[k], nodes);
  }
  printf("B capture path: %s\n", g_capture_path.c_str());

  // Echo thread: spin on armed; on a new value store release=1, then publish which generation it answered.
  std::atomic<uint32_t> echoed{0};
  std::atomic<bool> stop{false};
  std::thread echo([&] {
    pin(echo_cpu);
    std::atomic_ref<uint32_t> armed(hw->armed), release(hw->release);
    uint32_t last = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      const uint32_t a = armed.load(std::memory_order_acquire);
      if (a != last) {
        if (delay_ns > 0) {
          const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(delay_ns);
          while (std::chrono::steady_clock::now() < until) {
          }
        }
        release.store(1u, std::memory_order_release);
        last = a;
        echoed.store(a, std::memory_order_release);
      }
    }
  });

  // Interleaved replays, order rotated per round.
  uint32_t gen = 0;
  const int nk = static_cast<int>(kinds.size());
  for (int r = 0; r < total; ++r) {
    for (int j = 0; j < nk; ++j) {
      const Kind k = kinds[(r + j) % nk];
      CK(cudaGraphLaunch(execs[k], stream));
      ++gen;
      const auto t_start = std::chrono::steady_clock::now();
      while (cudaStreamQuery(stream) == cudaErrorNotReady) {
        if (std::chrono::steady_clock::now() - t_start > std::chrono::seconds(10)) {
          fprintf(stderr, "replay %d of %s hung for 10 s; aborting\n", r, kNames[k]);
          _exit(3);
        }
      }
      CK(cudaGetLastError());
      while (echoed.load(std::memory_order_acquire) != gen) {
      }
    }
  }
  CK(cudaStreamSynchronize(stream));

  // Host turnaround alone: main thread pings a pinned word, the echo-shaped thread answers in another.
  stop.store(true);
  echo.join();
  std::vector<double> host_rtt;
  {
    std::atomic<bool> stop2{false};
    std::atomic_ref<uint32_t> ping(hw->armed), pong(hw->release);
    ping.store(0);
    pong.store(0);
    std::thread echo2([&] {
      pin(echo_cpu);
      uint32_t last = 0;
      while (!stop2.load(std::memory_order_relaxed)) {
        const uint32_t a = ping.load(std::memory_order_acquire);
        if (a != last) {
          pong.store(a, std::memory_order_release);
          last = a;
        }
      }
    });
    for (uint32_t i = 1; i <= 200000; ++i) {
      const auto t = std::chrono::steady_clock::now();
      ping.store(i, std::memory_order_release);
      while (pong.load(std::memory_order_acquire) != i) {
      }
      host_rtt.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t).count());
    }
    stop2.store(true);
    echo2.join();
  }
  printf("host ping-pong (cpu %d <-> cpu %d, pinned words) RTT ns: p50 %.0f p90 %.0f p99 %.0f max %.0f\n",
         main_cpu, echo_cpu, pct(host_rtt, 0.5), pct(host_rtt, 0.9), pct(host_rtt, 0.99),
         *std::max_element(host_rtt.begin(), host_rtt.end()));

  // Results.
  std::vector<std::vector<double>> lat(4);
  double n_p50 = 0, n_p90 = 0, n_p99 = 0;
  for (Kind k : kinds) {
    std::vector<uint64_t> t0(total), t2(total), iters(total), spin(total);
    std::vector<uint32_t> seen(total);
    uint32_t timeouts = 0, idx = 0;
    CK(cudaMemcpy(t0.data(), slots[k].t0, total * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(t2.data(), slots[k].t2, total * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(seen.data(), slots[k].seen, total * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(iters.data(), slots[k].iters, total * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(spin.data(), slots[k].spin_ns, total * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&timeouts, slots[k].timeouts, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&idx, slots[k].idx, 4, cudaMemcpyDeviceToHost));
    if (idx != static_cast<uint32_t>(total)) {
      fprintf(stderr, "%s ran %u replays, expected %d\n", kNames[k], idx, total);
      return 4;
    }
    int bad = 0;
    double iters_sum = 0, spin_sum = 0;
    int spun = 0;
    FILE* f = fopen((out_dir + "/" + kNames[k] + ".csv").c_str(), "w");
    fprintf(f, "replay,t2_minus_t0_ns,seen_release,iters,spin_ns\n");
    for (int i = 0; i < total; ++i) {
      fprintf(f, "%d,%llu,%u,%llu,%llu\n", i, (unsigned long long)(t2[i] - t0[i]), seen[i],
              (unsigned long long)iters[i], (unsigned long long)spin[i]);
      if (i < warmup) continue;
      lat[k].push_back(static_cast<double>(t2[i] - t0[i]) / 1000.0);
      if (k != kN && seen[i] != 1u) ++bad;
      if (iters[i] > 0) {  // loop-rate stats only from replays that actually polled
        ++spun;
        iters_sum += iters[i];
        spin_sum += spin[i];
      }
    }
    fclose(f);
    const auto& v = lat[k];
    const double p50 = pct(v, 0.5), p90 = pct(v, 0.9), p99 = pct(v, 0.99), mx = *std::max_element(v.begin(), v.end());
    if (k == kN) n_p50 = p50, n_p90 = p90, n_p99 = p99;
    printf("RESULT %-20s n=%zu us: p50 %.3f p90 %.3f p99 %.3f p99.9 %.3f max %.3f  min %.3f  bad_release=%d",
           kNames[k], v.size(), p50, p90, p99, pct(v, 0.999), mx, *std::min_element(v.begin(), v.end()), bad);
    if (k == kA) {
      printf("  timeouts=%u  replays_that_polled=%d  iters/polled-replay %.2f  loop %.0f ns/iter"
             "  sys loads %.2f M/s while spinning (3 per iter)",
             timeouts, spun, spun ? iters_sum / spun : 0.0, iters_sum > 0 ? spin_sum / iters_sum : 0.0,
             spin_sum > 0 ? 3.0 * iters_sum / spin_sum * 1000.0 : 0.0);
    }
    printf("\n");
  }
  for (Kind k : kinds) {
    if (k == kN) continue;
    const auto& v = lat[k];
    printf("EXTRA_OVER_N %-20s us: p50 %+.3f p90 %+.3f p99 %+.3f\n", kNames[k], pct(v, 0.5) - n_p50,
           pct(v, 0.9) - n_p90, pct(v, 0.99) - n_p99);
  }
  // A vs B paired by round (same rotation round).
  if (!lat[kA].empty() && !lat[kB].empty()) {
    std::vector<double> d;
    for (size_t i = 0; i < lat[kA].size(); ++i) d.push_back(lat[kB][i] - lat[kA][i]);
    printf("PAIRED B-A per round us: p10 %+.3f p50 %+.3f p90 %+.3f\n", pct(d, 0.1), pct(d, 0.5), pct(d, 0.9));
  }
  return 0;
}
