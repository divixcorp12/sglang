# NVFP4 CPU expert plugin

This library implements `CpuExpertForward` from
`kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h`. It is a CPU-only,
batch-one decode kernel behind `Nvfp4CpuQuantTrait`, parallel to the existing
`exl3_cpu/optimized` implementation. The expert engine, lease protocol, job
ring, pinned output buffers and copy completion gate are reused without edits.

## Storage and calculations

Registration takes views of the pinned host tier. A host slot indexes all
slabs; no expert weights are converted or cached at registration. The same
weight and block-scale bytes can be copied to a compatible GPU kernel.

| Slab | Per-slot representation |
| --- | --- |
| `w13_weight` | `uint8[2*N,H/2]`, row-major packed E2M1, low nibble first |
| `w2_weight` | `uint8[H,N/2]`, row-major packed E2M1, low nibble first |
| `w13_blockscale_swizzled` | E4M3 bytes, logical `[2*N,H/16]`, padded to 128 rows and 4 columns then swizzled |
| `w2_blockscale_swizzled` | E4M3 bytes, logical `[H,N/16]`, same swizzle |
| `g1_alphas`, `g2_alphas` | One FP32 GPU GEMM alpha per slot |
| `g1_alphas_up` (optional) | One FP32 alpha per slot if gate/up global scales differ |

For a logical block-scale row `r`, column `g`, and padded column count `G`,
the byte address inside one scale slab row is:

```text
((((r // 128) * (G // 4) + g // 4) * 32 + r % 32) * 4
  + (r % 128) // 32) * 4 + g % 4
```

This matches the checkout's `swizzle_blockscale` reshape/permute. The native
descriptor accepts separate byte strides for each slab so padded host slots
remain supported. The Python adapter requires contiguous tensors with their
exact expected shapes.

W13 row order is explicit: `gate_up`, `up_gate`, or
`up_gate_interleaved64`. The latter reads alternating 64-row up/gate chunks
directly, as used by the packed W13 portion of the CuTe DSL standard path.
It requires N divisible by 64. That path's additional MMA-layout scale tensor
is not read: register the existing 128x4-swizzled scale slab instead.
TRTLLM shuffled weights/scales and Marlin layouts are unsupported even if
their tensors have superficially matching shapes. The caller must identify
the actual preparation path; shapes alone cannot identify a layout.

The CPU calculation is W4A16: FP16 input is decoded to FP32 scratch, followed
by gate/up matvec, ordinary `SiLU(gate) * up`, and down matvec. Results are
weighted and summed in FP32. It avoids activation requantization. This is
not bit-identical to GPU W4A4 inference and requires model-level quality
validation before deployment. There is a portable scalar path and an AVX2
path that expands sixteen FP4 values in registers; both keep weights packed.
Performance has not been benchmarked.

GPU alphas may equal `weight_global_scale * activation_global_scale`.
Pass `inv_input_scale13` and `inv_input_scale2` explicitly to cancel those
activation factors when multiplying full FP16 activations. For weight-only
alphas pass 1.0. These must be layer-wide scalars. A trait currently uses the
same reciprocals and W13 row order for every layer it registers; do not share
one trait across layers with different metadata. The native registration
descriptor itself is per-layer, so a runtime adapter can extend registration
to supply different verified metadata per layer. Per-expert activation
factors would need extra slot-resident scale slabs, because a host slot may
be reused for a different expert; capturing a per-expert array at registration
would be incorrect. Separate gate/up weight scales are supported through
`g1_alphas_up`; omitting it asserts that both halves share the gate alpha.

Only ordinary SiLU is implemented. `act_limit=0` disables clamping; positive
values apply `gate=min(gate,L)` and `up=clamp(up,-L,L)` before SiLU. This is
not an implementation of Bailing's post-SiLU clamp, GPT-OSS's shifted/scaled
SwiGLU, SiTU, GELU, ReLU2 or non-gated experts. The caller must not register
those activation conventions with this descriptor.

## Build and attach

Run from the checkout root:

```sh
python3 python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py \
  --output /absolute/path/libsglang_nvfp4_cpu.so --native
export SGLANG_NVFP4_CPU_LIBRARY=/absolute/path/libsglang_nvfp4_cpu.so
```

C++17 and pthreads are sufficient; no PyTorch, CUDA, GGML, OpenMP or special
GCC version is needed for the library. `--native` enables the build machine's
ISA; omit it for the portable scalar build. Do not distribute native binaries
to incompatible CPUs. The Python trait requires PyTorch for tensor views.
The standalone native code also builds on macOS for correctness tests;
explicit core placement in the production engine requires Linux.

Construct the trait with verified layer metadata:

```python
trait = cpu_trait_for(
    "nvfp4",
    nvfp4_config=dict(
        w13_layout="gate_up",
        inv_input_scale13=1.0,  # only if g1_alphas are weight-only
        inv_input_scale2=1.0,   # only if g2_alphas are weight-only
        separate_up_alpha=False,
    ),
)
```

`CpuExpertService` supplies the activation limit at layer registration, or
standalone callers must set `trait.act_limit=0` explicitly for no clamp.
The trait retains registered tensors and the library, exposes
`native_forward()` as a C function address, and registers each layer once.
The native engine then calls that address without Python involvement.

The plugin and explicit trait factory are implemented here. Automatic
selection from ModelOpt NVFP4 layers, their GPU runners, or server flags is
not wired. Existing EXL3 service activation remains unchanged. A future
runtime hookup must select layout/activation metadata from the actual runner,
carry separate up alphas where needed, and verify layer-wide input scale
factors before constructing the trait. Do not simply enable the EXL3 flags
for an arbitrary NVFP4 model.

## Threading, lifetime and error contract

One process-wide persistent worker pool serves the plugin; one engine thread
is worker zero. Set distinct allowed Linux core IDs before the first forward.
The first forward fixes the maximum thread count; later jobs may use fewer
workers. Another concurrent forward/free/configuration is rejected rather
than racing the pool. Layers have O(H+N) scratch, allocated at registration,
with no expanded-weight storage or per-job helper thread creation.

The engine owns slot leases: it must keep slabs stable until a job finishes.
Once complete, slots may be overwritten; the next job reads their new bytes.
Stop and join the engine before freeing layer handles or dropping tensors.
Do not unload the shared library while its function pointer or workers remain
in use. The callback supports at most eight lanes, skips -1 slots, preserves
routing order (including duplicate slots), and supports overwrite/accumulate.
Callers must supply valid pointer extents and finite activations/block scales;
the native ABI cannot validate the actual allocation behind a raw pointer.

Statuses: 0 success, 1 internal error, 2 invalid arguments, 3 concurrent use.
The engine's existing nonzero-status fail-stop behavior remains applicable.

## Verification

```sh
python3 -m unittest discover -s test/registered/unit/kernels \
  -p test_nvfp4_cpu_experts_native.py -v
# Test the AVX2 build on an x86 host with AVX2:
NVFP4_TEST_NATIVE=1 python3 -m unittest discover \
  -s test/registered/unit/kernels -p test_nvfp4_cpu_experts_native.py -v
```

The eight native tests compare against independently decoded dense experts,
cover finite E4M3 codes and FP4 nibbles, partial scale tiles, slot strides,
three row orders, separate alphas, input-scale cancellation, clamping, lane
routing, accumulation, thread counts, invalid inputs, handle lifecycle and
live slab updates. Three PyTorch tests exercise the real trait and verify
byte identity against the checkout's unchanged production swizzle function.
They skip when PyTorch is absent. Synthetic CPU tests establish arithmetic
and layout compatibility; captured GPU execution, full-model quality and
throughput remain unverified.

For the standalone memory/undefined-behavior harness:

```sh
c++ -std=c++17 -O1 -g -pthread -fsanitize=address,undefined \
  -fno-omit-frame-pointer \
  -Ipython/sglang/srt/layers/quantization/nvfp4_cpu/optimized \
  python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/moe_mul1.cpp \
  test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp -o /tmp/nvfp4-sanitizer
/tmp/nvfp4-sanitizer
```
