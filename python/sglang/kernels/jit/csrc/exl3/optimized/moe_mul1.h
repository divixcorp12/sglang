#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <vector>

// CPU-side MoE expert GEMM for mul1 (cb2) EXL3 tensors, standalone from the module code so it
// can be registered and driven directly. A layer is registered once (raw pointers into CPU
// trellis/suh/svh tensors, which the caller must keep alive) and then invoked per forward with
// the routing results. Kernels dispatch at runtime on scalar / AVX2 / AVX-512BW / AVX-512+VNNI;
// the mul1 codebook is affine in a byte-sum, so dequantization and the activation product
// fuse into integer dot products (see exl3_moe_cpu_forward for the math).
//
// Current limits: mul1 codebook only, K in [1, 8]. Gated experts with silu/gelu/swiglu_oai
// (act_limit) or gateless with relu2; optional per-expert biases (uniform per projection).

struct Exl3Projection
{
    const uint16_t* trellis;
    const at::Half* suh;
    const at::Half* svh;
    const at::Half* bias;   // nullable; added after the output transform
    int k;
    int n;
    int bits;
    // Band-contiguous ("swizzled") trellis layout: tile (kt, nt) stored at group nt/8, then
    // kt, then member nt%8, so each 8-tile output band reads as one sequential k-stream
    int swz = 0;
};

// The vendored per-expert kernel (exl3/moe_mul1.cpp) keeps upstream's name: production builds it against
// upstream's cpu/moe_mul1.h, the bench baseline against this header.
using MoeCpuMatrix = Exl3Projection;

struct MoeCpuLayer
{
    std::vector<Exl3Projection> gates;
    std::vector<Exl3Projection> ups;
    std::vector<Exl3Projection> downs;
    // Tensor references keeping the CPU weight storage alive
    std::vector<at::Tensor> refs;
    int num_experts;
    int hidden_size;      // k of gate/up, n of down (unpadded handling is the caller's problem)
    int interm_size;      // n of gate/up, k of down
    int activation;       // 0 = silu, 1 = gelu, 2 = relu2 (gateless), 3 = swiglu_oai
    float act_limit;      // swiglu_oai clamp
};

// Register a layer: per-expert tensor lists (CPU, contiguous). Returns a handle.
int64_t exl3_moe_cpu_make_layer
(
    const std::vector<at::Tensor>& gate_trellis,
    const std::vector<at::Tensor>& gate_suh,
    const std::vector<at::Tensor>& gate_svh,
    const std::vector<at::Tensor>& up_trellis,
    const std::vector<at::Tensor>& up_suh,
    const std::vector<at::Tensor>& up_svh,
    const std::vector<at::Tensor>& down_trellis,
    const std::vector<at::Tensor>& down_suh,
    const std::vector<at::Tensor>& down_svh,
    const std::vector<at::Tensor>& gate_bias,
    const std::vector<at::Tensor>& up_bias,
    const std::vector<at::Tensor>& down_bias,
    int64_t activation,
    double act_limit,
    int64_t swizzled        // caller repacked trellis tensors band-contiguous (K8 exempt)
);

void exl3_moe_cpu_free_layer(int64_t handle);

// Run the routed experts for one forward (upstream's contract):
//   x:        [m, hidden] fp16, CPU
//   selected: [m, top_k] int64, CPU (global expert ids; -1 skips)
//   weights:  [m, top_k] fp16, CPU
//   out:      [m, hidden] fp32, CPU (overwritten)
// This build's kernel takes slab layers only (exl3_cpu_kernel(), kernel.h): exl3_moe_cpu_make_layer, its forwards and
// free_layer refuse, and exist so upstream's bindings link.
void exl3_moe_cpu_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
);

// Raw-pointer variant used by upstream's persistent worker (moe_handoff.cu); refuses, as above
void exl3_moe_cpu_forward_raw
(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* w,
    float* out,
    int rows,
    int topk,
    int threads
);

// Per-phase profiling of the forward, printed to stdout per call.
void exl3_moe_cpu_set_prof(bool enabled);

// Upstream link compatibility only (bindings.cpp, cpu/moe_handoff.cu): both fail if called.
void exl3_moe_cpu_stage_experts(int64_t handle, const uint32_t* expert_ids, int count, uint8_t* dst, int threads);
int64_t exl3_moe_cpu_pool_stress(int threads, int iters, int small, int spin);

// Kernel availability (dispatch happens internally; these are informational, post-env-cap).
// has_avx512_vbmi and has_avx512_bw additionally gate the swizzled weight layout in the child
// loader (the VBMI tier's wide swizzle bands need the byte-gather kernels' low temporary count).
bool exl3_moe_cpu_has_avx2();
bool exl3_moe_cpu_has_avx512_bw();
bool exl3_moe_cpu_has_avx512_vnni();
bool exl3_moe_cpu_has_avx512_vbmi();

