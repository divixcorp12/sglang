// Stand-in W_s (wait) and A_s (acknowledge) kernels for the stage-triple cost g, written to the contract of
// PER_ROW_TRANSFER.md 5.5 at the level that matters for cost (G_MEASUREMENT_PREREG.md section 4). C_s is NOT here:
// it is the production copy_expert_row_segments_gpu_kernel, launched from Python.
//
//   w_kernel  one thread. p == 0: the empty-range exit, no host access, go[0] = 0.
//             p  > 0: p SERIAL system-scope acquire loads of words in a mapped host page, one 64 B line each (each is a
//             PCIe round trip and acquire ordering serialises them), then go[0] = 1. A word that is not yet nonzero
//             would be polled with __nanosleep(256); the harness publishes every word first, so that never runs
//             (registered: "it is always ready"; G_MEASUREMENT_PREREG.md L10 says what this leaves out).
//             spin_ns > 0 adds a busy wait of that long (the positive control, +20 us).
//   a_kernel  one thread. go[0] == 0: acknowledges nothing and returns. Else __threadfence_system() then a
//             st.release.sys to a distinct mapped line (the ack fence).
//   nop_kernel  the 8,000 filler nodes of the empty_base8k variant.
//
// Plain C entry points, called through ctypes on the torch stream so they land in the same captured graph as the copy.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void w_kernel(const uint32_t* page, int p, uint32_t* go, int spin_ns) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (spin_ns > 0) {
    const uint64_t t0 = globaltimer_ns();
    while (globaltimer_ns() - t0 < (uint64_t)spin_ns) {
    }
  }
  if (p == 0) {
    go[0] = 0;
    return;
  }
  for (int i = 0; i < p; ++i) {
    uint32_t v = ld_acquire_sys(page + 16 * i);
    while (v == 0) {
      __nanosleep(256);
      v = ld_acquire_sys(page + 16 * i);
    }
  }
  go[0] = 1;
}

__global__ void a_kernel(const uint32_t* go, uint32_t* ack_line) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (go[0] == 0) return;
  __threadfence_system();
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(ack_line), "r"(1u) : "memory");
}

__global__ void nop_kernel() {}

extern "C" {
int launch_w(void* stream, const void* page, int p, void* go, int spin_ns) {
  w_kernel<<<1, 32, 0, (cudaStream_t)stream>>>((const uint32_t*)page, p, (uint32_t*)go, spin_ns);
  return (int)cudaGetLastError();
}
int launch_a(void* stream, const void* go, void* ack_line) {
  a_kernel<<<1, 32, 0, (cudaStream_t)stream>>>((const uint32_t*)go, (uint32_t*)ack_line);
  return (int)cudaGetLastError();
}
int launch_nop(void* stream) {
  nop_kernel<<<1, 32, 0, (cudaStream_t)stream>>>();
  return (int)cudaGetLastError();
}
}
