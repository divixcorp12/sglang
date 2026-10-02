// REQUIRED BUILD FLAG: none; never -use_fast_math. Fast math flushes subnormal fp32 products to zero, which
// torch's bf16 multiply keeps, and bit parity with the unfused chain breaks. Part of the EXL3 decode cast fusion:
// it keeps each rounding of the unfused chain and only drops a memory trip.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "../expert_stream/tensor_checks.h"
#include <cuda_bf16.h>
#include <stdint.h>

namespace sglang::exl3 {

// Exl3MoEMethod's out.to(bf16) * routed_scaling_factor on the fused MoE's fp32 output. The factor is the float
// torch's mul uses for a Python scalar on a bf16 tensor; this kernel must not be built with fast math, which would
// flush the subnormal products torch keeps.
__global__ void exl3_scale_to_bf16_kernel(
    const float* __restrict__ input, __nv_bfloat16* __restrict__ output, int64_t n, float factor) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(input[i])) * factor);
  }
}

/// \brief Checked launcher for `exl3_scale_to_bf16_kernel`: the fused MoE's fp32 output to bf16, times the routed
/// scaling factor.
///
/// `input` [n] fp32 and `output` [n] bf16 must be CUDA tensors on one device; an empty input launches nothing.
void exl3_scale_to_bf16(tvm::ffi::TensorView input, tvm::ffi::TensorView output, double factor) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto N = SymbolicSize{"n"};
  expert_stream::verify_named("input", TensorMatcher({N}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), input);
  expert_stream::verify_named("output", TensorMatcher({N}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), output);
  const int64_t n = N.unwrap();
  if (n == 0) {
    return;
  }
  constexpr int kThreads = 256;
  LaunchKernel(static_cast<uint32_t>(div_ceil(n, static_cast<int64_t>(kThreads))), kThreads, device.unwrap())(
      exl3_scale_to_bf16_kernel,
      static_cast<const float*>(input.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      n,
      static_cast<float>(factor));
}

}  // namespace sglang::exl3
