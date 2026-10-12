// Capability probes for the copy-mechanism sweep. probe.py builds each probe into a module of its own (one -DPROBE_*
// per build), so an instruction the toolchain rejects fails that probe's build only, and runs each in a process of
// its own, so a fault in one (an illegal address from a bulk read of host memory, say) cannot poison the others.
// Every probe reads a pinned host buffer and writes device memory; probe.py compares the bytes.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

__device__ __forceinline__ uint32_t probe_smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

#if defined(PROBE_V8_WEAK) || defined(PROBE_V8_CV)
// One 32-byte load per thread, stored back as two 16-byte stores so only the load's width is under test.
__global__ void probe_kernel(const uint8_t* src, uint8_t* dst) {
  const uint8_t* s = src + 32 * threadIdx.x;
  uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
#ifdef PROBE_V8_CV
  asm volatile("ld.global.cv.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3), "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
               : "l"(s)
               : "memory");
#else
  asm volatile("ld.global.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3), "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
               : "l"(s)
               : "memory");
#endif
  uint8_t* d = dst + 32 * threadIdx.x;
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(d), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(d + 16), "r"(v4), "r"(v5), "r"(v6), "r"(v7) : "memory");
}
constexpr int kProbeThreads = 256;  // 8 KiB
#endif

#ifdef PROBE_LDGSTS
__global__ void probe_kernel(const uint8_t* src, uint8_t* dst) {
  __shared__ alignas(16) uint8_t buf[256 * 16];
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(probe_smem(buf + 16 * threadIdx.x)),
               "l"(src + 16 * threadIdx.x)
               : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  reinterpret_cast<uint4*>(dst)[threadIdx.x] = reinterpret_cast<const uint4*>(buf)[threadIdx.x];
}
constexpr int kProbeThreads = 256;  // 4 KiB
#endif

#ifdef PROBE_BULK
__global__ void probe_kernel(const uint8_t* src, uint8_t* dst) {
  constexpr uint32_t kBytes = 4096;
  __shared__ alignas(128) uint8_t buf[kBytes];
  __shared__ alignas(8) uint64_t bar;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(probe_smem(&bar)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // The contract's shape: a generic-proxy view of host bytes handed to the async proxy.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(probe_smem(&bar)), "r"(kBytes)
                 : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     probe_smem(buf)),
                 "l"(src), "r"(kBytes), "r"(probe_smem(&bar))
                 : "memory");
  }
  asm volatile(
      "{\n .reg .pred done;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0;\n @!done bra WAIT;\n}\n" ::"r"(
          probe_smem(&bar))
      : "memory");
  reinterpret_cast<uint4*>(dst)[threadIdx.x] = reinterpret_cast<const uint4*>(buf)[threadIdx.x];
}
constexpr int kProbeThreads = 256;  // 4 KiB
#endif

#ifdef PROBE_BATCH
// Host API only: four 1 KiB host-to-device copies in one cudaMemcpyBatchAsync (CUDA 13.4 signature, no failIdx).
void probe_run(tvm::ffi::TensorView src, tvm::ffi::TensorView dst) {
  void* dsts[4];
  const void* srcs[4];
  size_t sizes[4];
  for (int i = 0; i < 4; ++i) {
    dsts[i] = static_cast<uint8_t*>(dst.data_ptr()) + 1024 * i;
    srcs[i] = static_cast<const uint8_t*>(src.data_ptr()) + 1024 * i;
    sizes[i] = 1024;
  }
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attr_index = 0;
  const auto stream = host::LaunchKernel::resolve_device(dst.device());
  CHECK_CUDA(cudaMemcpyBatchAsync(dsts, srcs, sizes, 4, &attr, &attr_index, 1, stream)) << "cudaMemcpyBatchAsync";
}
#else
void probe_run(tvm::ffi::TensorView src, tvm::ffi::TensorView dst) {
  const auto stream = host::LaunchKernel::resolve_device(dst.device());
  host::LaunchKernel(1, kProbeThreads, stream)(
      probe_kernel, static_cast<const uint8_t*>(src.data_ptr()), static_cast<uint8_t*>(dst.data_ptr()));
}
#endif

}  // namespace sglang
