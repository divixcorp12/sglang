// Per-layer GPU -> CPU -> GPU handoff latency for CPU-computed MoE experts, inside a CUDA graph.
// Per layer the GPU publishes the 5120-wide fp16 hidden state to pinned host memory and a ready flag; a spinning CPU
// worker (optionally running exllamav3's EXL3 CPU expert kernel on it) writes a 5120-wide fp32 output and a done flag;
// the GPU waits for done, pulls the output into VRAM, and a consumer kernel feeds it into the next layer's input.
// Two GPU-side variants: "kernel" (zero-copy stores, spin-wait kernel) and "memop" (cudaMemcpyAsync +
// cuStreamWriteValue32 / cuStreamWaitValue32, no SM spinning). Flags are constant-valued so a graph can replay:
// the GPU writes ready=1, the worker clears ready before it works and sets done=1, the GPU clears done after its wait.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cpu/moe_mul1.h"

namespace {
constexpr int kLayers = 40;
constexpr int kH = 5120;
constexpr int kMaxTopk = 8;
constexpr int kStride = 16;  // u32 words: one 64 B line per flag

struct Shm {
  uint32_t ready[kLayers * kStride];
  uint32_t done[kLayers * kStride];
  uint32_t abort_flag[kStride];
  alignas(64) __half x[kLayers][kH];
  alignas(64) int32_t sel[kLayers][kMaxTopk];
  alignas(64) __half w[kLayers][kMaxTopk];
  alignas(64) float out[kLayers][kH];
};

Shm* g_shm = nullptr;

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void post_kernel(const __half* __restrict__ x, Shm* s, int l) {
  const uint4* src = reinterpret_cast<const uint4*>(x);
  uint4* dst = reinterpret_cast<uint4*>(s->x[l]);
  for (int i = threadIdx.x; i < kH / 8; i += blockDim.x) dst[i] = src[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release_sys(&s->ready[l * kStride], 1u);
  }
}

__global__ void wait_kernel(Shm* s, int l, float* __restrict__ out) {
  __shared__ int ok;
  if (threadIdx.x == 0) {
    const uint64_t start = gtimer();
    ok = 1;
    while (ld_acquire_sys(&s->done[l * kStride]) != 1u) {
      if (gtimer() - start > 2000000000ull) {  // 2 s: the worker is gone, fail the replay instead of hanging
        s->abort_flag[0] = 1;
        ok = 0;
        break;
      }
    }
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    s->done[l * kStride] = 0u;
  }
  __syncthreads();
  if (!ok) return;
  const volatile float4* src = reinterpret_cast<const volatile float4*>(s->out[l]);
  float4* dst = reinterpret_cast<float4*>(out);
  for (int i = threadIdx.x; i < kH / 4; i += blockDim.x) {
    const float4 v = const_cast<const float4&>(src[i]);
    dst[i] = v;
  }
}

// The next layer's input depends on this layer's output, so nothing can run ahead of the handoff.
__global__ void consume_kernel(const float* __restrict__ out, __half* __restrict__ x) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kH; i += gridDim.x * blockDim.x)
    x[i] = __float2half(0.5f * __half2float(x[i]) + 1e-3f * out[i]);
}

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string(what) + ": " + (msg ? msg : "?"));
  }
}

// ---- CPU worker ----
struct Worker {
  std::thread thread;
  std::atomic<bool> stop{false};
  std::vector<double> compute_us;  // per handled layer
  std::atomic<int64_t> handled{0};
};
Worker* g_worker = nullptr;

inline double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void worker_loop(Worker* wk, int64_t handle, int topk, int threads, int cpu, int n_experts) {
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
  Shm* s = g_shm;
  int l = 0;
  uint32_t rot = 0;
  // A fixed stride through the registered experts: no expert repeats within n_experts/topk layers (cold reads).
  while (!wk->stop.load(std::memory_order_relaxed)) {
    volatile uint32_t* ready = &s->ready[l * kStride];
    while (*ready != 1u) {
      if (wk->stop.load(std::memory_order_relaxed)) return;
      _mm_pause();
    }
    *ready = 0u;
    std::atomic_thread_fence(std::memory_order_acquire);
    const double t0 = now_us();
    if (handle >= 0 && topk > 0) {
      for (int k = 0; k < topk; ++k) {
        s->sel[l][k] = static_cast<int32_t>((rot * 97u + static_cast<uint32_t>(k)) % static_cast<uint32_t>(n_experts));
        s->w[l][k] = __float2half(1.0f / topk);
      }
      rot += 1;
      exl3_moe_cpu_forward_raw(handle, reinterpret_cast<const at::Half*>(s->x[l]), s->sel[l],
                               reinterpret_cast<const at::Half*>(s->w[l]), s->out[l], 1, topk, threads);
    } else {
      for (int i = 0; i < kH; ++i) s->out[l][i] = __half2float(s->x[l][i]);
    }
    const double t1 = now_us();
    if (wk->compute_us.size() < 2000000) wk->compute_us.push_back(t1 - t0);
    std::atomic_thread_fence(std::memory_order_release);
    __atomic_store_n(&s->done[l * kStride], 1u, __ATOMIC_RELEASE);
    wk->handled.fetch_add(1, std::memory_order_relaxed);
    l = (l + 1) % kLayers;
  }
}
}  // namespace

void init_shm() {
  if (g_shm) return;
  void* p = nullptr;
  check(cudaHostAlloc(&p, sizeof(Shm), cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
  std::memset(p, 0, sizeof(Shm));
  g_shm = static_cast<Shm*>(p);
}

void reset_flags() {
  for (int i = 0; i < kLayers * kStride; ++i) g_shm->ready[i] = g_shm->done[i] = 0u;
  g_shm->abort_flag[0] = 0u;
}

int64_t aborted() { return g_shm->abort_flag[0]; }

void start_worker(int64_t handle, int64_t topk, int64_t threads, int64_t cpu, int64_t n_experts) {
  if (g_worker) throw std::runtime_error("worker already running");
  reset_flags();
  g_worker = new Worker();
  g_worker->compute_us.reserve(1 << 20);
  g_worker->thread = std::thread(worker_loop, g_worker, handle, static_cast<int>(topk), static_cast<int>(threads),
                                 static_cast<int>(cpu), static_cast<int>(n_experts));
}

// Stops the worker; returns its per-layer CPU work times in microseconds.
std::vector<double> stop_worker() {
  if (!g_worker) return {};
  g_worker->stop.store(true);
  g_worker->thread.join();
  std::vector<double> r = std::move(g_worker->compute_us);
  delete g_worker;
  g_worker = nullptr;
  return r;
}

// Enqueue one layer's handoff on the current stream (called while capturing).
void layer(torch::Tensor x, torch::Tensor out, int64_t l, int64_t mode) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* xs = reinterpret_cast<__half*>(x.data_ptr<at::Half>());
  auto* os = out.data_ptr<float>();
  if (mode == 0) {  // baseline: no handoff
  } else if (mode == 1) {  // kernel: zero-copy stores + spin-wait kernel
    post_kernel<<<1, 256, 0, stream>>>(xs, g_shm, static_cast<int>(l));
    wait_kernel<<<1, 256, 0, stream>>>(g_shm, static_cast<int>(l), os);
  } else if (mode == 2) {  // memop: DMA copies + stream memory operations
    check(cudaMemcpyAsync(g_shm->x[l], xs, kH * sizeof(__half), cudaMemcpyDeviceToHost, stream), "d2h");
    check(cuStreamWriteValue32(reinterpret_cast<CUstream>(stream),
                               reinterpret_cast<CUdeviceptr>(&g_shm->ready[l * kStride]), 1u, 0),
          "write ready");
    check(cuStreamWaitValue32(reinterpret_cast<CUstream>(stream),
                              reinterpret_cast<CUdeviceptr>(&g_shm->done[l * kStride]), 1u, CU_STREAM_WAIT_VALUE_EQ),
          "wait done");
    check(cuStreamWriteValue32(reinterpret_cast<CUstream>(stream),
                               reinterpret_cast<CUdeviceptr>(&g_shm->done[l * kStride]), 0u, 0),
          "clear done");
    check(cudaMemcpyAsync(os, g_shm->out[l], kH * sizeof(float), cudaMemcpyHostToDevice, stream), "h2d");
  } else {
    throw std::runtime_error("bad mode");
  }
  consume_kernel<<<20, 256, 0, stream>>>(os, xs);
  check(cudaGetLastError(), "launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_shm", &init_shm);
  m.def("reset_flags", &reset_flags);
  m.def("aborted", &aborted);
  m.def("start_worker", &start_worker);
  m.def("stop_worker", &stop_worker, py::call_guard<py::gil_scoped_release>());
  m.def("layer", &layer);
  m.def("make_layer", &exl3_moe_cpu_make_layer);
}
