// The lease chain's launch shapes with no protocol: every stage waits for its predecessor under PDL, reads the
// predecessor's word and writes its own, so each edge is a real data dependency, and spins `work_ns` standing in for
// the stage's body. mode 0: no PDL; 1: PDL, trigger implied at exit; 2: PDL, trigger right after the wait (valid per
// the PTX ISA: griddepcontrol.wait waits for the primary grid to COMPLETE and its memory to be visible).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

__device__ __forceinline__ void skel_spin(int64_t ns) {
  uint64_t t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  while (static_cast<int64_t>(t - t0) < ns);
}

// pre_ns: every thread spins this long BEFORE the wait, standing in for a real stage's prologue (parameter loads,
// address setup, mbarrier init), which PDL can overlap with the predecessor and which the plain skeleton lacks.
template <int kMode>
__global__ void skel_stage_kernel(int64_t* words, int64_t index, int64_t work_ns, int64_t pre_ns) {
  if (pre_ns > 0) skel_spin(pre_ns);
  device::PDLWaitPrimary<kMode != 0>();
  device::PDLTriggerSecondary<kMode == 2>();
  if (threadIdx.x != 0) return;
  uint64_t t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  for (uint64_t t = t0; static_cast<int64_t>(t - t0) < work_ns;)
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  if (blockIdx.x != 0) return;
  // Stage 0 counts replays; every later stage copies its predecessor, so after R replays every word reads R.
  words[index] = index == 0 ? words[0] + 1 : words[index - 1];
}

void skel_stage(tvm::ffi::TensorView words, int64_t index, int64_t grid, int64_t block, int64_t work_ns, int64_t mode,
                int64_t pre_ns) {
  const auto stream = host::LaunchKernel::resolve_device(words.device());
  auto* w = static_cast<int64_t*>(words.data_ptr());
  const dim3 g(static_cast<unsigned>(grid)), b(static_cast<unsigned>(block));
  switch (mode) {
    case 0: host::LaunchKernel(g, b, stream)(skel_stage_kernel<0>, w, index, work_ns, pre_ns); return;
    case 1: host::LaunchKernel(g, b, stream).enable_pdl(true)(skel_stage_kernel<1>, w, index, work_ns, pre_ns); return;
    case 2: host::LaunchKernel(g, b, stream).enable_pdl(true)(skel_stage_kernel<2>, w, index, work_ns, pre_ns); return;
  }
  host::RuntimeCheck(false, "mode: 0, 1 or 2");
}

}  // namespace sglang
