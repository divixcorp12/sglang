// REQUIRED BUILD FLAG: -use_fast_math. silu_and_mul (deepseek_v4/silu_and_mul_masked_post_quant.cuh) must compile
// to the same instructions as in silu_and_mul_clamp's module, which builds with it; bit parity depends on that.
// Part of the EXL3 decode cast fusion (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION): it keeps each rounding of the unfused
// chain, in the same order, and only drops the round trips through memory.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "../../deepseek_v4/silu_and_mul_masked_post_quant.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace sglang::exl3 {

// The shared expert's y.to(bf16) -> silu_mul_clamp -> x.to(fp16): gate_up [rows, 2 * inter] fp16 in, the down
// projection's fp16 input [rows, inter] out. Built with the same -use_fast_math as silu_mul_clamp_kernel, whose
// silu_and_mul it calls, so the activation is computed by the same instructions.
template <bool kUsePDL>
__global__
__launch_bounds__(1024, 2) void exl3_silu_mul_clamp_half_kernel(const SiluAndMulClampParams __grid_constant__ params) {
  using namespace device;
  using Vec = AlignedVector<fp16x2_t, 4>;
  const auto row = blockIdx.x / params.blocks_per_row;
  const auto block_in_row = blockIdx.x % params.blocks_per_row;
  const auto vec_id = block_in_row * blockDim.x + threadIdx.x;
  const float limit = params.swiglu_limit;

  PDLWaitPrimary<kUsePDL>();
  if (vec_id < params.out_vecs) {
    const auto input = static_cast<const Vec*>(params.input);
    auto output = static_cast<Vec*>(params.output);
    const auto input_row = row * 2 * params.out_vecs;
    const auto gate = input[input_row + vec_id];
    const auto up = input[input_row + params.out_vecs + vec_id];
    Vec out;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const auto activated = cast<bf16x2_t>(silu_and_mul<true>(to_bf16x2(gate[i]), to_bf16x2(up[i]), limit));
      out[i] = cast<fp16x2_t>(cast<fp32x2_t>(activated));
    }
    output[row * params.out_vecs + vec_id] = out;
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
void exl3_silu_mul_clamp_half(tvm::ffi::TensorView input, tvm::ffi::TensorView output, double swiglu_limit) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto M = SymbolicSize{"rows"};
  auto D = SymbolicSize{"gate_up_dim"};
  auto H = SymbolicSize{"inter"};
  TensorMatcher({M, D}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(input);
  TensorMatcher({M, H}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(output);
  RuntimeCheck(D.unwrap() == 2 * H.unwrap(), "gate_up must be twice as wide as the output");
  const auto inter = static_cast<uint32_t>(H.unwrap());
  RuntimeCheck(inter > 0 && inter % 8 == 0, "inter must be a positive multiple of 8");
  const auto out_vecs = inter / 8;
  const auto threads = std::min(out_vecs, 1024u);
  const auto blocks_per_row = div_ceil(out_vecs, threads);
  const auto params = SiluAndMulClampParams{
      .input = input.data_ptr(),
      .output = output.data_ptr(),
      .swiglu_limit = static_cast<float>(swiglu_limit),
      .out_vecs = out_vecs,
      .blocks_per_row = blocks_per_row,
  };
  LaunchKernel(static_cast<uint32_t>(M.unwrap()) * blocks_per_row, threads, device.unwrap())
      .enable_pdl(kUsePDL)(exl3_silu_mul_clamp_half_kernel<kUsePDL>, params);
}

}  // namespace sglang::exl3
