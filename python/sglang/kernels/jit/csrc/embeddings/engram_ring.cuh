// Device side of the Engram device-wait lookup (SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT), which replaces the decode
// graph's Engram host nodes. Protocol and memory-ordering rules: docs/superpowers/plans/2026-09-25-dsv41-engram-no-hostnode.md.
//
// post: one thread advances the lookup's device sequence, copies the step's hash ids into the pinned ids buffer,
// fences, and release-stores the sequence into the control block's post word (kPostSeq).
// wait: thread 0 acquire-spins on kDoneSeq until it equals the sequence or `timeout_ns` of %globaltimer passes.
// A timeout, a failed status, or a fatal word already latched writes the status to `status_dev` (which the caller
// asserts on) and latches kFatalSeq, so the service refuses later requests. On success the block copies the served
// rows from the pinned buffer into `rows_dev`.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace engram_ring_device {

// int32 words of the pinned control block; mirrors engram_host_node.cpp's ring:: and engram_ring.py.
constexpr int64_t kPostSeq = 0;
constexpr int64_t kDoneSeq = 16;
constexpr int64_t kStatus = 17;
constexpr int64_t kFatalSeq = 32;
constexpr int64_t kFatalStatus = 33;
constexpr int32_t kDeviceTimeout = 5;
constexpr int32_t kDeviceSawFatal = 6;
constexpr int kWaitThreads = 256;

__device__ __forceinline__ uint32_t ld_acquire_sys(const int32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void st_release_sys(int32_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint4 ld_volatile_v4(const uint4* address) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(address)
               : "memory");
  return v;
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

// `counter` is pinned host memory that only these kernels read and write (see _EngramDeviceWaitLookup).
// `test_stall_ns` (0 in production) spins after the first id so a test can catch a sequence published too early.
__global__ void post_kernel(const int64_t* __restrict__ ids_dev, int64_t* ids_host, int64_t n, int32_t* control,
                            int32_t* counter, int64_t test_stall_ns) {
  if (threadIdx.x != 0) return;
  volatile int32_t* sequence = counter;
  uint32_t seq = static_cast<uint32_t>(*sequence) + 1u;
  if (seq == 0u) seq = 1u;  // 0 is the control block's initial value, never a posted sequence
  *sequence = static_cast<int32_t>(seq);
  volatile int64_t* out = ids_host;
  for (int64_t i = 0; i < n; ++i) {
    out[i] = ids_dev[i];
    if (i == 0 && test_stall_ns > 0) {
      const uint64_t start = global_ns();
      while (static_cast<int64_t>(global_ns() - start) < test_stall_ns) __nanosleep(1000);
    }
  }
  // The ids are posted writes over PCIe; the fence and the release store both order them before the sequence.
  __threadfence_system();
  st_release_sys(control + kPostSeq, seq);
}

__global__ void wait_kernel(int32_t* control, const int32_t* counter, const uint8_t* rows_host, uint8_t* rows_dev,
                            int64_t bytes, int32_t* status_dev, int64_t timeout_ns) {
  __shared__ int32_t block_status;
  if (threadIdx.x == 0) {
    const uint32_t seq = static_cast<uint32_t>(*reinterpret_cast<const volatile int32_t*>(counter));
    int32_t status;
    if (ld_acquire_sys(control + kFatalSeq) != 0) {
      status = kDeviceSawFatal;
    } else {
      const uint64_t start = global_ns();
      uint32_t done = ld_acquire_sys(control + kDoneSeq);
      while (done != seq && static_cast<int64_t>(global_ns() - start) < timeout_ns) {
        __nanosleep(256);
        done = ld_acquire_sys(control + kDoneSeq);
      }
      // kStatus is read after the acquire load that saw done == seq, so it is the service's status for this seq.
      status = done == seq ? *reinterpret_cast<const volatile int32_t*>(control + kStatus) : kDeviceTimeout;
    }
    if (status != 0) {
      *reinterpret_cast<volatile int32_t*>(control + kFatalStatus) = status;
      st_release_sys(control + kFatalSeq, seq);
    }
    status_dev[0] = status;
    block_status = status;
  }
  __syncthreads();
  if (block_status != 0) return;
  // Thread 0's acquire, then the barrier, orders the other threads' row loads after the service's row stores;
  // the fence keeps that true for the loads of the pinned buffer at system scope.
  __threadfence_system();
  const auto base_in = reinterpret_cast<uintptr_t>(rows_host);
  const auto base_out = reinterpret_cast<uintptr_t>(rows_dev);
  if (((base_in | base_out | static_cast<uintptr_t>(bytes)) & 15u) == 0) {
    const uint4* in = reinterpret_cast<const uint4*>(rows_host);
    uint4* out = reinterpret_cast<uint4*>(rows_dev);
    for (int64_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) out[i] = ld_volatile_v4(in + i);
  } else {
    const volatile uint8_t* in = rows_host;
    for (int64_t i = threadIdx.x; i < bytes; i += blockDim.x) rows_dev[i] = in[i];
  }
}

}  // namespace engram_ring_device

void engram_ring_post(tvm::ffi::TensorView ids_dev, tvm::ffi::TensorView ids_host, tvm::ffi::TensorView control,
                      tvm::ffi::TensorView counter, int64_t test_stall_ns) {
  const auto device = host::LaunchKernel::resolve_device(ids_dev.device());
  host::LaunchKernel(1, 32, device)(
      engram_ring_device::post_kernel,
      static_cast<const int64_t*>(ids_dev.data_ptr()),
      static_cast<int64_t*>(ids_host.data_ptr()),
      static_cast<int64_t>(ids_dev.size(0)),
      static_cast<int32_t*>(control.data_ptr()),
      static_cast<int32_t*>(counter.data_ptr()),
      test_stall_ns);
}

void engram_ring_wait(tvm::ffi::TensorView control, tvm::ffi::TensorView counter, tvm::ffi::TensorView rows_host,
                      tvm::ffi::TensorView rows_dev, tvm::ffi::TensorView status_dev, int64_t timeout_ns) {
  const auto device = host::LaunchKernel::resolve_device(rows_dev.device());
  host::LaunchKernel(1, engram_ring_device::kWaitThreads, device)(
      engram_ring_device::wait_kernel,
      static_cast<int32_t*>(control.data_ptr()),
      static_cast<const int32_t*>(counter.data_ptr()),
      static_cast<const uint8_t*>(rows_host.data_ptr()),
      static_cast<uint8_t*>(rows_dev.data_ptr()),
      static_cast<int64_t>(rows_dev.numel()),
      static_cast<int32_t*>(status_dev.data_ptr()),
      timeout_ns);
}

}  // namespace sglang
