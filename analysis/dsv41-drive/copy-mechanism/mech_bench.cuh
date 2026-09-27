// Copy-mechanism sweep kernels. A job is {source, destination, bytes}, every field a multiple of 16 (the six EXL3
// segment row sizes are multiples of 512). Every host read obeys the lease visibility contract (LEASE_PROTOCOL.md
// E1 amendment): ld.global.cv, or cp.async.cg issued after the acquire, or a bulk read after
// fence.proxy.async.global. The one exception is kind 1 (.nc), the fresh check's negative control, never swept.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace sglang {
namespace mech {

constexpr int kThreads = 256;
constexpr int kKindCv = 0;
constexpr int kKindNc = 1;
constexpr int kKindLdgsts = 2;
constexpr int kKindTma = 3;

struct Job {
  int64_t src, dst, bytes;
};

__device__ __forceinline__ uint64_t now_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// The fresh check's midpoint. Every block counts itself in words[1], then its thread 0 polls words[0] with an
// acquire until the host has rewritten the source and released the word. The block barrier after the poll orders
// the other threads' second-pass reads after it.
__device__ void fresh_barrier(uint32_t* words) {
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd_system(words + 1, 1u);
    while (ld_acquire(words) == 0)
      __nanosleep(1000);
  }
  __syncthreads();
}

template <int W>
struct Vec {
  uint32_t v[W / 4];
};

template <int W, bool kNc>
__device__ __forceinline__ Vec<W> load(const uint8_t* p) {
  Vec<W> r;
  if constexpr (W == 16) {
    if constexpr (kNc) {
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
                   : "l"(p)
                   : "memory");
    } else {
      asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
                   : "l"(p)
                   : "memory");
    }
  } else {
#ifdef MECH_V8
    asm volatile("ld.global.cv.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]),
                   "=r"(r.v[7])
                 : "l"(p)
                 : "memory");
#endif
  }
  return r;
}

template <int W>
__device__ __forceinline__ void store(uint8_t* p, const Vec<W>& r) {
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(r.v[0]), "r"(r.v[1]), "r"(r.v[2]), "r"(r.v[3])
               : "memory");
  if constexpr (W == 32) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p + 16), "r"(r.v[4]), "r"(r.v[5]), "r"(r.v[6]),
                 "r"(r.v[7])
                 : "memory");
  }
}

// S's access pattern generalised: chunks of kThreads * U units dealt round-robin over the grid, U loads issued
// before any store. U = 1, W = 16, grid 8 is exactly stream_copy_slice (row_copy_kernels.cuh).
template <int U, int W, bool kNc>
__global__ __launch_bounds__(kThreads, 1) void sm_kernel(const Job* jobs, int64_t njobs, uint32_t* fresh) {
  for (int pass = 0; pass < (fresh != nullptr ? 2 : 1); ++pass) {
    if (pass == 1) fresh_barrier(fresh);
    for (int64_t j = 0; j < njobs; ++j) {
      const auto src = reinterpret_cast<const uint8_t*>(jobs[j].src);
      const auto dst = reinterpret_cast<uint8_t*>(jobs[j].dst);
      const int64_t units = jobs[j].bytes / W;
      for (int64_t chunk = blockIdx.x; chunk * kThreads * U < units; chunk += gridDim.x) {
        Vec<W> v[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int64_t u = (chunk * U + k) * kThreads + threadIdx.x;
          if (u < units) v[k] = load<W, kNc>(src + W * u);
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int64_t u = (chunk * U + k) * kThreads + threadIdx.x;
          if (u < units) store<W>(dst + W * u, v[k]);
        }
      }
    }
  }
}

// One thread: n serial acquires of a pinned host word, then of a device word (the 711 ns / 117 ns figures).
__global__ void latency_kernel(const uint32_t* host_word, const uint32_t* dev_word, int64_t n, int64_t* out) {
  uint32_t sink = 0;
  const uint64_t t0 = now_ns();
  for (int64_t i = 0; i < n; ++i)
    sink += ld_acquire(host_word);
  const uint64_t t1 = now_ns();
  for (int64_t i = 0; i < n; ++i)
    sink += ld_acquire(dev_word);
  const uint64_t t2 = now_ns();
  out[0] = static_cast<int64_t>(t1 - t0);
  out[1] = static_cast<int64_t>(t2 - t1);
  out[2] = sink;
}

// The device half of a flag round trip: release ping, acquire-poll pong, per round. RTT/2 bounds one-way visibility.
__global__ void pingpong_kernel(uint32_t* ping, const uint32_t* pong, int64_t rounds, int64_t* out) {
  for (int64_t r = 1; r <= rounds; ++r) {
    const uint64_t t0 = now_ns();
    st_release(ping, static_cast<uint32_t>(r));
    while (ld_acquire(pong) != static_cast<uint32_t>(r)) {
    }
    out[r - 1] = static_cast<int64_t>(now_ns() - t0);
  }
}

inline void cpu_relax() {
#if defined(__x86_64__)
  _mm_pause();
#endif
}

// Every copy kernel of this file behind one switch; Tasks 4 (kinds 2, 3) extend it.
inline void launch(int64_t kind, const Job* jobs, int64_t njobs, int64_t grid, int64_t a, int64_t b, uint32_t* fresh,
                   cudaStream_t stream) {
  const int g = static_cast<int>(grid);
  if (kind == kKindNc) {
    host::RuntimeCheck(a == 1 && b == 16, "the .nc control is U=1, W=16 only");
    host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 16, true>, jobs, njobs, fresh);
    return;
  }
  host::RuntimeCheck(kind == kKindCv, "kind: 0 (.cv) or 1 (.nc control); 2 and 3 arrive in Task 4");
  if (b == 16) {
    switch (a) {
      case 1: host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 16, false>, jobs, njobs, fresh); return;
      case 2: host::LaunchKernel(g, kThreads, stream)(sm_kernel<2, 16, false>, jobs, njobs, fresh); return;
      case 4: host::LaunchKernel(g, kThreads, stream)(sm_kernel<4, 16, false>, jobs, njobs, fresh); return;
      case 8: host::LaunchKernel(g, kThreads, stream)(sm_kernel<8, 16, false>, jobs, njobs, fresh); return;
      case 16: host::LaunchKernel(g, kThreads, stream)(sm_kernel<16, 16, false>, jobs, njobs, fresh); return;
    }
  }
#ifdef MECH_V8
  if (b == 32) {
    switch (a) {
      case 1: host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 32, false>, jobs, njobs, fresh); return;
      case 2: host::LaunchKernel(g, kThreads, stream)(sm_kernel<2, 32, false>, jobs, njobs, fresh); return;
      case 4: host::LaunchKernel(g, kThreads, stream)(sm_kernel<4, 32, false>, jobs, njobs, fresh); return;
      case 8: host::LaunchKernel(g, kThreads, stream)(sm_kernel<8, 32, false>, jobs, njobs, fresh); return;
    }
  }
#endif
  host::RuntimeCheck(false, "no .cv kernel for this (unroll, width); width 32 needs the MECH_V8 build");
}

}  // namespace mech

void mech_copy(tvm::ffi::TensorView jobs, int64_t kind, int64_t grid, int64_t a, int64_t b) {
  const auto stream = host::LaunchKernel::resolve_device(jobs.device());
  mech::launch(kind, static_cast<const mech::Job*>(jobs.data_ptr()), jobs.size(0), grid, a, b, nullptr, stream);
}

// Launches `kind` with the fresh midpoint, waits for every block to reach it, rewrites the pinned source with
// `pattern`, releases the flag, and waits for the kernel. The caller checks that the destination holds `pattern`.
void mech_fresh(tvm::ffi::TensorView jobs, tvm::ffi::TensorView words, tvm::ffi::TensorView src,
                tvm::ffi::TensorView pattern, int64_t kind, int64_t grid, int64_t a, int64_t b) {
  auto* w = static_cast<uint32_t*>(words.data_ptr());
  __atomic_store_n(w, 0u, __ATOMIC_RELEASE);
  __atomic_store_n(w + 1, 0u, __ATOMIC_RELEASE);
  const auto stream = host::LaunchKernel::resolve_device(jobs.device());
  mech::launch(kind, static_cast<const mech::Job*>(jobs.data_ptr()), jobs.size(0), grid, a, b, w, stream);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (__atomic_load_n(w + 1, __ATOMIC_ACQUIRE) != static_cast<uint32_t>(grid)) {
    mech::cpu_relax();
    host::RuntimeCheck(std::chrono::steady_clock::now() < deadline, "fresh check: blocks never reached the midpoint");
  }
  std::memcpy(src.data_ptr(), pattern.data_ptr(), static_cast<size_t>(src.size(0)));
  __atomic_store_n(w, 1u, __ATOMIC_RELEASE);
  CHECK_CUDA(cudaStreamSynchronize(stream)) << "fresh check";
}

void mech_latency(tvm::ffi::TensorView host_word, tvm::ffi::TensorView dev_word, tvm::ffi::TensorView out, int64_t n) {
  const auto stream = host::LaunchKernel::resolve_device(out.device());
  host::LaunchKernel(1, 1, stream)(mech::latency_kernel, static_cast<const uint32_t*>(host_word.data_ptr()),
                                   static_cast<const uint32_t*>(dev_word.data_ptr()), n,
                                   static_cast<int64_t*>(out.data_ptr()));
  CHECK_CUDA(cudaStreamSynchronize(stream)) << "latency";
}

void mech_pingpong(tvm::ffi::TensorView words, tvm::ffi::TensorView out, int64_t rounds) {
  auto* w = static_cast<uint32_t*>(words.data_ptr());
  __atomic_store_n(w, 0u, __ATOMIC_RELEASE);
  __atomic_store_n(w + 1, 0u, __ATOMIC_RELEASE);
  const auto stream = host::LaunchKernel::resolve_device(out.device());
  host::LaunchKernel(1, 1, stream)(mech::pingpong_kernel, w, w + 1, rounds, static_cast<int64_t*>(out.data_ptr()));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (uint32_t r = 1; r <= static_cast<uint32_t>(rounds); ++r) {
    while (__atomic_load_n(w, __ATOMIC_ACQUIRE) != r) {
      mech::cpu_relax();
      host::RuntimeCheck(std::chrono::steady_clock::now() < deadline, "ping-pong: the kernel stopped answering");
    }
    __atomic_store_n(w + 1, r, __ATOMIC_RELEASE);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream)) << "ping-pong";
}

}  // namespace sglang
