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
#include <vector>

#include "moe/expert_stream/row_copy_kernels.cuh"  // copy_wait_read: the copy wait's own SM small-copy loop
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
constexpr int kKindWeak = 4;
constexpr int kKindCwReal = 5;
constexpr int kKindLine = 7;
// sm_kernel's load flavour: .cv (swept), .nc (the fresh check's control, never swept), weak (plain ld.global after
// the acquire; swept: the PTX memory model orders it after an acquire that observed the host's release).
constexpr int kLoadCv = 0;
constexpr int kLoadNc = 1;
constexpr int kLoadWeak = 2;

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

__device__ __forceinline__ uint32_t ld_relaxed(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// The fresh check's midpoint. Every block counts itself in words[1], then its thread 0 polls words[0] until the
// host has rewritten the source and released the word. The block barrier after the poll orders the other threads'
// second-pass reads after it. Swept kernels poll with an acquire, as the visibility contract requires. On sm_120 that
// acquire compiles to LDG.E.STRONG.SYS + CCTL.IVALL, which invalidates the SM's whole L1 -- so the .nc control
// (kRelaxed) polls with a relaxed load, no CCTL, or its stale L1 lines would be dropped and it would read fresh bytes
// (the blind check of 2026-09-27). mech_bench.py asserts the CCTL split on the built SASS.
template <bool kRelaxed = false>
__device__ void fresh_barrier(uint32_t* words) {
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd_system(words + 1, 1u);
    if constexpr (kRelaxed) {
      while (ld_relaxed(words) == 0)
        __nanosleep(1000);
    } else {
      while (ld_acquire(words) == 0)
        __nanosleep(1000);
    }
  }
  __syncthreads();
}

template <int W>
struct Vec {
  uint32_t v[W / 4];
};

template <int W, int kLoad>
__device__ __forceinline__ Vec<W> load(const uint8_t* p) {
  Vec<W> r;
  if constexpr (W == 16) {
    if constexpr (kLoad == kLoadNc) {
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
                   : "l"(p)
                   : "memory");
    } else if constexpr (kLoad == kLoadWeak) {
      asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
                   : "l"(p)
                   : "memory");
    } else {
      asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
                   : "l"(p)
                   : "memory");
    }
  } else if constexpr (kLoad == kLoadWeak) {
#ifdef MECH_V8_WEAK
    asm volatile("ld.global.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]),
                   "=r"(r.v[7])
                 : "l"(p)
                 : "memory");
#endif
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
template <int U, int W, int kLoad>
__global__ __launch_bounds__(kThreads, 1) void sm_kernel(const Job* jobs, int64_t njobs, uint32_t* fresh) {
  for (int pass = 0; pass < (fresh != nullptr ? 2 : 1); ++pass) {
    if (pass == 1) fresh_barrier<kLoad == kLoadNc>(fresh);
    for (int64_t j = 0; j < njobs; ++j) {
      const auto src = reinterpret_cast<const uint8_t*>(jobs[j].src);
      const auto dst = reinterpret_cast<uint8_t*>(jobs[j].dst);
      const int64_t units = jobs[j].bytes / W;
      for (int64_t chunk = blockIdx.x; chunk * kThreads * U < units; chunk += gridDim.x) {
        Vec<W> v[U];
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int64_t u = (chunk * U + k) * kThreads + threadIdx.x;
          if (u < units) v[k] = load<W, kLoad>(src + W * u);
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

#if defined(MECH_LDGSTS) || defined(MECH_BULK)
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred done;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n @!done bra WAIT;\n}\n" ::"r"(
          smem(bar)),
      "r"(parity)
      : "memory");
}
#endif

#ifdef MECH_LDGSTS
constexpr int kLdgstsPer = 8;                       // 16-byte units per producer lane per stage
constexpr int kLdgstsUnits = 32 * kLdgstsPer;       // units per stage
constexpr int kLdgstsStage = 16 * kLdgstsUnits;     // 4 KiB
constexpr int kConsumers = kThreads - 32;

// Warp 0 produces: each lane issues its cp.async.cg reads of a stage (L2 only, never L1, so no stale L1 line), then
// arrives on full[slot] when they land (cp.async.mbarrier.arrive.noinc). Warps 1-7 consume: wait full, store the
// stage to the destination, arrive on empty[slot]. In flight per block: at most STAGES * 4 KiB.
template <int STAGES>
__global__ __launch_bounds__(kThreads, 1) void ldgsts_kernel(const Job* jobs, int64_t njobs, uint32_t* fresh) {
  __shared__ alignas(128) uint8_t ring[STAGES][kLdgstsStage];
  __shared__ alignas(8) uint64_t full[STAGES];
  __shared__ alignas(8) uint64_t empty[STAGES];
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 32);
      mbar_init(&empty[s], kConsumers);
    }
  }
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  int64_t local = 0;  // stages this block has used, across passes and jobs: ring position and parity
  for (int pass = 0; pass < (fresh != nullptr ? 2 : 1); ++pass) {
    if (pass == 1) fresh_barrier(fresh);
    for (int64_t j = 0; j < njobs; ++j) {
      const auto src = reinterpret_cast<const uint8_t*>(jobs[j].src);
      const auto dst = reinterpret_cast<uint8_t*>(jobs[j].dst);
      const int64_t units = jobs[j].bytes / 16;
      const int64_t stages = (units + kLdgstsUnits - 1) / kLdgstsUnits;
      for (int64_t g = blockIdx.x; g < stages; g += gridDim.x, ++local) {
        const int slot = static_cast<int>(local % STAGES);
        const uint32_t phase = static_cast<uint32_t>((local / STAGES) & 1);
        const int64_t base = g * kLdgstsUnits;
        if (warp == 0) {
          if (local >= STAGES) mbar_wait(&empty[slot], phase ^ 1u);
          for (int k = 0; k < kLdgstsPer; ++k) {
            const int64_t u = base + k * 32 + lane;
            if (u < units) {
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem(ring[slot] + 16 * (k * 32 + lane))),
                           "l"(src + 16 * u)
                           : "memory");
            }
          }
          asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem(&full[slot])) : "memory");
        } else {
          mbar_wait(&full[slot], phase);
          for (int i = threadIdx.x - 32; i < kLdgstsUnits; i += kConsumers) {
            const int64_t u = base + i;
            if (u < units) {
              const uint4 v = *reinterpret_cast<const uint4*>(ring[slot] + 16 * i);
              asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(dst + 16 * u), "r"(v.x), "r"(v.y),
                           "r"(v.z), "r"(v.w)
                           : "memory");
            }
          }
          asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem(&empty[slot])) : "memory");
        }
      }
    }
  }
}
#endif

#ifdef MECH_BULK
// One thread per block. For each of its chunks: bulk-read host -> shared (async proxy, completion on full[slot]),
// then bulk-write shared -> VRAM. The producer runs STAGES - 1 chunks ahead of the consumer; before reusing a slot it
// waits (wait_group.read 0) for every bulk write to have read its shared source. In flight per block: <= STAGES *
// chunk. The fence.proxy.async.global after the fresh acquire is the contract for async-proxy reads of host bytes.
template <int STAGES>
__global__ __launch_bounds__(1, 1) void tma_kernel(const Job* jobs, int64_t njobs, int64_t chunk, uint32_t* fresh) {
  extern __shared__ __align__(128) uint8_t tma_ring[];
  __shared__ alignas(8) uint64_t full[STAGES];
  for (int s = 0; s < STAGES; ++s)
    mbar_init(&full[s], 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  int64_t local = 0;
  for (int pass = 0; pass < (fresh != nullptr ? 2 : 1); ++pass) {
    if (pass == 1) {
      fresh_barrier(fresh);
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    for (int64_t j = 0; j < njobs; ++j) {
      const auto src = reinterpret_cast<const uint8_t*>(jobs[j].src);
      const auto dst = reinterpret_cast<uint8_t*>(jobs[j].dst);
      const int64_t bytes = jobs[j].bytes;
      const int64_t chunks = (bytes + chunk - 1) / chunk;
      const int64_t mine = chunks > blockIdx.x ? (chunks - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
      for (int64_t k = 0; k < mine + STAGES - 1; ++k) {
        if (k < mine) {
          const int64_t pos = local + k;
          const int slot = static_cast<int>(pos % STAGES);
          const int64_t off = (blockIdx.x + k * gridDim.x) * chunk;
          const uint32_t n = static_cast<uint32_t>(min(chunk, bytes - off));
          if (pos >= STAGES) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem(&full[slot])), "r"(n)
                       : "memory");
          asm volatile(
              "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                  smem(tma_ring + slot * chunk)),
              "l"(src + off), "r"(n), "r"(smem(&full[slot]))
              : "memory");
        }
        const int64_t c = k - (STAGES - 1);
        if (c >= 0) {
          const int64_t pos = local + c;
          const int slot = static_cast<int>(pos % STAGES);
          const int64_t off = (blockIdx.x + c * gridDim.x) * chunk;
          const uint32_t n = static_cast<uint32_t>(min(chunk, bytes - off));
          mbar_wait(&full[slot], static_cast<uint32_t>((pos / STAGES) & 1));
          asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst + off),
                       "r"(smem(tma_ring + slot * chunk)), "r"(n)
                       : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
      local += mine;
    }
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
#endif

inline void cpu_relax() {
#if defined(__x86_64__)
  _mm_pause();
#endif
}

#ifdef MECH_V8
constexpr bool kBuiltV8Cv = true;
#else
constexpr bool kBuiltV8Cv = false;
#endif
#ifdef MECH_V8_WEAK
constexpr bool kBuiltV8Weak = true;
#else
constexpr bool kBuiltV8Weak = false;
#endif

// The swept SM kernels of one load flavour: U in {1,2,4,8,16} at 16 B, {1,2,4,8} at 32 B (when that width is built).
template <int kLoad>
inline bool launch_sm(int g, int64_t a, int64_t b, const Job* jobs, int64_t njobs, uint32_t* fresh, cudaStream_t stream) {
  if (b == 16) {
    switch (a) {
      case 1: host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 16, kLoad>, jobs, njobs, fresh); return true;
      case 2: host::LaunchKernel(g, kThreads, stream)(sm_kernel<2, 16, kLoad>, jobs, njobs, fresh); return true;
      case 4: host::LaunchKernel(g, kThreads, stream)(sm_kernel<4, 16, kLoad>, jobs, njobs, fresh); return true;
      case 8: host::LaunchKernel(g, kThreads, stream)(sm_kernel<8, 16, kLoad>, jobs, njobs, fresh); return true;
      case 16: host::LaunchKernel(g, kThreads, stream)(sm_kernel<16, 16, kLoad>, jobs, njobs, fresh); return true;
    }
  }
  if constexpr ((kLoad == kLoadCv && kBuiltV8Cv) || (kLoad == kLoadWeak && kBuiltV8Weak)) {
    if (b == 32) {
      switch (a) {
        case 1: host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 32, kLoad>, jobs, njobs, fresh); return true;
        case 2: host::LaunchKernel(g, kThreads, stream)(sm_kernel<2, 32, kLoad>, jobs, njobs, fresh); return true;
        case 4: host::LaunchKernel(g, kThreads, stream)(sm_kernel<4, 32, kLoad>, jobs, njobs, fresh); return true;
        case 8: host::LaunchKernel(g, kThreads, stream)(sm_kernel<8, 32, kLoad>, jobs, njobs, fresh); return true;
      }
    }
  }
  return false;
}

// The copy wait's SM small copies as production runs them (exl3_ram_miss_lease_copy_wait_kernel with sm_count > 0):
// one block of kCopyWaitThreads, every job in order through the production copy_wait_read, no barrier between jobs.
// Its 4-deep loop runs only on the part of a tensor that fills 4 x 256 x 16 B = 16 KiB; the rest goes one 16-byte
// unit per thread per round trip (stream_copy16).
__global__ __launch_bounds__(device::expert_stream::kCopyWaitThreads, 1) void cw_real_kernel(const Job* jobs,
                                                                                            int64_t njobs) {
  for (int64_t j = 0; j < njobs; ++j)
    copy_wait_read(reinterpret_cast<const uint8_t*>(jobs[j].src), reinterpret_cast<uint8_t*>(jobs[j].dst), jobs[j].bytes);
}

// Partial-line probe: sm_cv16's loop, but unit u reads the (u % (kTouch / 16))-th 16 B of line u / (kTouch / 16), so
// only the first kTouch bytes of every 128 B line are loaded (.cv) and stored. Tells whether the SM's sysmem path
// fetches whole 128 B lines (useful GB/s scales with kTouch) or 32 B sectors (it does not, down to 32 B).
template <int U, int kTouch>
__global__ __launch_bounds__(kThreads, 1) void line_kernel(const Job* jobs, int64_t njobs) {
  constexpr int kPer = kTouch / 16;
  for (int64_t j = 0; j < njobs; ++j) {
    const auto src = reinterpret_cast<const uint8_t*>(jobs[j].src);
    const auto dst = reinterpret_cast<uint8_t*>(jobs[j].dst);
    const int64_t units = jobs[j].bytes / 128 * kPer;
    for (int64_t chunk = blockIdx.x; chunk * kThreads * U < units; chunk += gridDim.x) {
      Vec<16> v[U];
#pragma unroll
      for (int k = 0; k < U; ++k) {
        const int64_t u = (chunk * U + k) * kThreads + threadIdx.x;
        if (u < units) v[k] = load<16, kLoadCv>(src + 128 * (u / kPer) + 16 * (u % kPer));
      }
#pragma unroll
      for (int k = 0; k < U; ++k) {
        const int64_t u = (chunk * U + k) * kThreads + threadIdx.x;
        if (u < units) store<16>(dst + 128 * (u / kPer) + 16 * (u % kPer), v[k]);
      }
    }
  }
}

template <int U>
inline bool launch_line(int g, int64_t touch, const Job* jobs, int64_t njobs, cudaStream_t stream) {
  switch (touch) {
    case 16: host::LaunchKernel(g, kThreads, stream)(line_kernel<U, 16>, jobs, njobs); return true;
    case 32: host::LaunchKernel(g, kThreads, stream)(line_kernel<U, 32>, jobs, njobs); return true;
    case 64: host::LaunchKernel(g, kThreads, stream)(line_kernel<U, 64>, jobs, njobs); return true;
    case 128: host::LaunchKernel(g, kThreads, stream)(line_kernel<U, 128>, jobs, njobs); return true;
  }
  return false;
}

// Every copy kernel of this file behind one switch; Tasks 4 (kinds 2, 3) extend it.
inline void launch(int64_t kind, const Job* jobs, int64_t njobs, int64_t grid, int64_t a, int64_t b, uint32_t* fresh,
                   cudaStream_t stream) {
  const int g = static_cast<int>(grid);
  if (kind == kKindNc) {
    host::RuntimeCheck(a == 1 && b == 16, "the .nc control is U=1, W=16 only");
    host::LaunchKernel(g, kThreads, stream)(sm_kernel<1, 16, kLoadNc>, jobs, njobs, fresh);
    return;
  }
  if (kind == kKindLine) {
    host::RuntimeCheck(fresh == nullptr, "line: no fresh check");
    if ((a == 4 && launch_line<4>(g, b, jobs, njobs, stream)) || (a == 8 && launch_line<8>(g, b, jobs, njobs, stream)))
      return;
    host::RuntimeCheck(false, "line: unroll 4 or 8, touch 16, 32, 64 or 128");
  }
  if (kind == kKindCwReal) {
    host::RuntimeCheck(grid == 1 && fresh == nullptr, "cw_real: one block, no fresh check (it has no midpoint)");
    host::LaunchKernel(1, device::expert_stream::kCopyWaitThreads, stream)(cw_real_kernel, jobs, njobs);
    return;
  }
  if (kind == kKindLdgsts) {
#ifdef MECH_LDGSTS
    switch (a) {
      case 2: host::LaunchKernel(g, kThreads, stream)(ldgsts_kernel<2>, jobs, njobs, fresh); return;
      case 4: host::LaunchKernel(g, kThreads, stream)(ldgsts_kernel<4>, jobs, njobs, fresh); return;
      case 8: host::LaunchKernel(g, kThreads, stream)(ldgsts_kernel<8>, jobs, njobs, fresh); return;
    }
#endif
    host::RuntimeCheck(false, "ldgsts: stages 2, 4 or 8, in the MECH_LDGSTS build");
  }
  if (kind == kKindTma) {
#ifdef MECH_BULK
    host::RuntimeCheck(b % 16 == 0 && a * b <= 64 * 1024, "tma: chunk a multiple of 16, stages * chunk <= 64 KiB");
    const size_t shared = static_cast<size_t>(a * b);
    switch (a) {
      case 2:
        CHECK_CUDA(cudaFuncSetAttribute(tma_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
        host::LaunchKernel(g, 1, stream, shared)(tma_kernel<2>, jobs, njobs, b, fresh);
        return;
      case 4:
        CHECK_CUDA(cudaFuncSetAttribute(tma_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
        host::LaunchKernel(g, 1, stream, shared)(tma_kernel<4>, jobs, njobs, b, fresh);
        return;
      case 8:
        CHECK_CUDA(cudaFuncSetAttribute(tma_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
        host::LaunchKernel(g, 1, stream, shared)(tma_kernel<8>, jobs, njobs, b, fresh);
        return;
    }
#endif
    host::RuntimeCheck(false, "tma: stages 2, 4 or 8, in the MECH_BULK build");
  }
  host::RuntimeCheck(kind == kKindCv || kind == kKindWeak,
                     "kind: 0 (.cv), 1 (.nc control), 2 (ldgsts), 3 (tma) or 4 (weak)");
  if (kind == kKindCv ? launch_sm<kLoadCv>(g, a, b, jobs, njobs, fresh, stream)
                      : launch_sm<kLoadWeak>(g, a, b, jobs, njobs, fresh, stream))
    return;
  host::RuntimeCheck(false, "no kernel for this (unroll, width); width 32 needs the MECH_V8 / MECH_V8_WEAK build");
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

// Copy-engine baselines, timed on the host as well: the API cost per call is what the service's copy thread pays.
int64_t mech_ce_each(tvm::ffi::TensorView jobs, tvm::ffi::TensorView dev) {
  const auto stream = host::LaunchKernel::resolve_device(dev.device());
  const auto* j = static_cast<const mech::Job*>(jobs.data_ptr());
  const int64_t n = jobs.size(0);
  const auto t0 = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < n; ++i) {
    CHECK_CUDA(cudaMemcpyAsync(reinterpret_cast<void*>(j[i].dst), reinterpret_cast<const void*>(j[i].src),
                               static_cast<size_t>(j[i].bytes), cudaMemcpyHostToDevice, stream))
        << "cudaMemcpyAsync";
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

#ifdef MECH_BATCH
int64_t mech_ce_batch(tvm::ffi::TensorView jobs, tvm::ffi::TensorView dev) {
  const auto stream = host::LaunchKernel::resolve_device(dev.device());
  const auto* j = static_cast<const mech::Job*>(jobs.data_ptr());
  const int64_t n = jobs.size(0);
  std::vector<void*> dsts(n);
  std::vector<const void*> srcs(n);
  std::vector<size_t> sizes(n);
  for (int64_t i = 0; i < n; ++i) {
    dsts[i] = reinterpret_cast<void*>(j[i].dst);
    srcs[i] = reinterpret_cast<const void*>(j[i].src);
    sizes[i] = static_cast<size_t>(j[i].bytes);
  }
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attr_index = 0;
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_CUDA(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), static_cast<size_t>(n), &attr, &attr_index, 1,
                                  stream))
      << "cudaMemcpyBatchAsync";
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
#endif

}  // namespace sglang
