# Native NVFP4 CPU expert plugin

Implements the `CpuExpertForward` callback from
`kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h`, parallel to
`exl3_cpu/optimized`. The plugin and build are native C/C++ only. The engine's job ring,
leases, pinned activation/output rows and completion signaling are reused.

Build from the checkout root, on the Linux machine that will run it (GCC with OpenMP):

```sh
python python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py \
  --cxx /opt/rh/gcc-toolset-15/root/usr/bin/g++ --output /absolute/path/libsglang_nvfp4_cpu.so
```

`--main HARNESS.cpp` links a native harness into an executable instead. Inside SGLang,
`sglang.srt.layers.quantization.nvfp4_cpu_ext.nvfp4_cpu_library()` builds the library on first use with `$CXX`
and caches it under `~/.cache/sglang/nvfp4_cpu` by a hash of the sources, flags and compiler. The build targets
baseline x86-64 and holds both ISA tiers: at the first forward the library picks AVX2 on an AVX2/FMA host, else the
scalar loop. `NVFP4_CPU_MAX_ISA=scalar` caps the tier (it never raises it) and `NVFP4_CPU_REPORT_ISA=1` prints the
tier chosen (`nvfp4 isa avx2`) to stderr. The arithmetic needs `-ffp-contract=off` and must never be built with
`-Ofast`.

The native harnesses (`test/registered/unit/kernels/nvfp4_cpu_{sanitizer,ggml_check}.cpp`) build and run under
`test/registered/unit/kernels/test_nvfp4_cpu_build.py`.

See [the benchmark guide](../bench/README.md) for the native benchmark
executable, fixed-team process rounds, fixture format and timing protocol.

See [upstream provenance and adaptation details](../upstream/README.md).

## Attach to the existing engine

Include `cpu_experts_cabi.h` and load the library `build.py` or
`nvfp4_cpu_ext.nvfp4_cpu_library()` builds. Register each layer
using the common `SglangCpuExpertsLayer` descriptor (`cpu_experts_abi.h`, `slab_count` 7) whose `params` points at a
`SglangNvfp4CpuParams` (W13 layout, inverse input scales). Registration stores views; the registrant keeps the slabs
and the parameters alive until `free_layer`. The optimized kernel never repacks or expands full
weight rows.

```cpp
int64_t handle = -1;
// Populate descriptor with this layer's verified layout/scales/slab pointers.
if (sglang_nvfp4_cpu_experts_register_layer(&descriptor, &handle) != 0)
    throw std::runtime_error("NVFP4 CPU layer registration failed");

CpuExpertConfig config;
config.forward = &sglang_nvfp4_cpu_experts_forward;
// Fill config's existing x/out row tables, dimensions, threads and core list.
// Before start/first forward, set the kernel's helper cores:
sglang_nvfp4_cpu_experts_set_cores(cores.data(), cores.size());
CpuExpertEngine engine(config, prefix, thread_name);
engine.set_layer(row, handle);
engine.start();
// Submit existing CpuJob records through the tier's owner.
// During teardown: engine.stop(), then free every registered layer.
```

Check every native return status in actual integration code. The kernel's core
list must match the engine's list and contain at least `config.threads` cores.
Registration supports different descriptors per layer. This change supplies
the native kernel and benchmark; automatic runtime selection is not wired.

## Weight layout and scale contract

| Descriptor slab | Per-host-slot representation |
| --- | --- |
| 0: W13 | packed E2M1 `uint8[2*N,H/2]`, low nibble first |
| 1: W2 | packed E2M1 `uint8[H,N/2]`, low nibble first |
| 2: SF13 | E4M3 logical `[2*N,H/16]`, GPU 128x4 padded/swizzled bytes |
| 3: SF2 | E4M3 logical `[H,N/16]`, same GPU scale layout |
| 4: gate alpha | one FP32 GPU GEMM alpha |
| 5: down alpha | one FP32 GPU GEMM alpha |
| 6: up alpha | optional FP32 alpha; null shares the gate alpha |

Each slab has its own byte stride per host slot. Slab contents may change when
slots are reused; every forward reads the current bytes. Keep all pointers
alive and lease-stable until the job completes. The weight/scale bytes can be
copied unchanged to a compatible GPU kernel.

For a logical scale row r, block column g, and padded column count G, the byte
address is:

```text
((((r / 128) * (G / 4) + g / 4) * 32 + r % 32) * 4
  + (r % 128) / 32) * 4 + g % 4
```

This is the inverse address of the checkout's `swizzle_blockscale` reshape/
permute. W13 row order is explicit: 0 = gate/up halves, 1 = up/gate halves,
2 = alternating 64-row up/gate chunks (N divisible by 64). Read the existing
128x4 scale slab, not the CuTe DSL path's additional MMA scale layout.
TRTLLM shuffled weights/scales and Marlin packing are unsupported. Shapes
cannot identify layout; the registering caller must verify the GPU prep path.

GPU alphas may be `weight_scale * activation_scale`. Register the reciprocal
activation factors as the parameters' `inv_input_scale13` and `inv_input_scale2` to cancel them
for full FP16 CPU input. Use 1.0 for weight-only alphas. Factors may differ
between layer descriptors, but each factor is scalar within one layer. A
per-expert factor must travel with its mutable host slot in a future ABI
extension. Omitting slab6 asserts gate/up share the same global weight scale.

The kernel accepts FP16 input and computes W4A8 with GGML Q8_0 activation
blocks (32 int8 values and a FP16 delta). Input and post-SwiGLU activations are
quantized once per vector, reused across output rows. FP32 gate/up projections, ordinary
`SiLU(gate)*up`, down projection, routing and reduction. Positive `act_limit`
applies `gate=min(gate,L)` and `up=clamp(up,-L,L)` before SiLU; 0 disables it.
Other activation conventions (Bailing post-SiLU clamps, GPT-OSS shifted/scaled
SwiGLU, SiTU, GELU, ReLU2, non-gated experts) must not use this descriptor.
It is not bit-identical to GPU W4A4 or a full-precision CPU activation path.
The benchmark reports Q8 quantization error against a W4A16 reference. Q8
deltas above finite FP16 range and nonfinite activations return status 2;
blocks whose delta rounds to zero contribute zero.

## Threading and lifetime

Each forward runs one OpenMP team of `threads` workers, the calling engine thread as worker 0, in four phases
separated by barriers: every token's input to Q8_0; every routed expert's gate/up rows and SiLU; every intermediate to
Q8_0; every expert's down rows, each token's summed in its routing order into its row of `out`. Configure distinct allowed Linux cores before the first
forward; worker i is pinned to core i (once, then re-checked cheaply). A forward may use any team size up to the
configured cores. If OpenMP forms a smaller team (`OMP_THREAD_LIMIT`, `OMP_DYNAMIC`), the forward returns 1 and
leaves `out` untouched. For latency set `OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE OMP_DYNAMIC=FALSE` and leave
`OMP_PROC_BIND` unset. A concurrent forward or free is rejected (3); cores configured after the first forward or
keep-warm are refused (2). `sglang_nvfp4_cpu_experts_keep_warm` holds the same pinned team in register-only work at
the forward's vector width between calls. Stop/join the engine before freeing
handles or slab storage; do not unload the library while callbacks are in use. The kernel requires Linux and OpenMP.

The forward takes one `SglangCpuExpertsForward` (`cpu_experts_abi.h`
beside the engine, shared with the EXL3 kernel): up to 65536 token rows of up
to eight lanes each. It skips -1 slots, preserves each row's routing order
including duplicate slots, and supports overwrite or accumulation. Rows that
route to the same slot are computed together, up to four per decoded weight
row; every row's output is bitwise its own one-row call's. The engine sends
one row per job. Caller
pointer extents and finite activation/block-scale values are required. Raw
pointers cannot prove allocation size. Status: 0 success, 1 internal error,
2 invalid arguments, 3 concurrent use. The engine's nonzero-status fail-stop
behavior is unchanged.

## Code layout

The layer registry, argument and route validation, worker cores, keep-warm and the five C functions are the shared
CPU experts framework's (`../../cpu_experts_common/`, `ExpertForward<Nvfp4Quant>`). `quant.hpp` holds `Nvfp4Quant`
(slab names and minimum strides, parameter validation, the registered `Layer`, a slot's projections) and the layer
facts; `math.hpp` the ISA-independent arithmetic (`GpuRow`, Q8_0 quantization, the gated SiLU) and `dot_rows<Isa, M>`,
whose tiers are `math_scalar.hpp` and `math_avx2.hpp` (compiled for AVX2 by function attribute); `forward_plan.hpp`
holds the plan, its types (`RouteBinding`, `Chunk`, `ForwardCtx`) and its per-thread scratch, `ForwardArena`;
`kernel.cpp` defines `Nvfp4Quant::dispatch` and the C ABI (one `SGLANG_CPU_EXPERTS_DEFINE_CABI`). A forward is
`ForwardPlan<Shape, Isa>::run` (`forward_plan.hpp`), picked once per call in `Nvfp4Quant::dispatch` from the tier
`ExpertForward` detected: `ForwardPlan<MimoV26ProShape, Isa::Avx2>` when `MimoV26ProShape::accepts` the layer at the
AVX2 tier, else `ForwardPlan<GenericShape, Isa::Avx2>` or `ForwardPlan<GenericShape, Isa::Scalar>`.
An AVX2 plan enters its gate/up and down row loops through `gate_up_avx2`/`down_avx2`, compiled for AVX2 with
everything they call inlined (`flatten`), once per phase per worker; the rest of the library is baseline x86-64.
`PlanTraits<Shape, Isa>` holds the plan's knobs (`kRowUnit`, the split unit). The plan groups a call's routes into
units, one per (token, slot), and units into chunks of up to `kChunkRows` (4) of one slot; `dot_rows<Isa, M>` decodes
each weight row once per chunk. A plan reads every layer fact it may fix
through its Shape (`shapes.hpp`): `GenericShape` from the layer's `LayerInfo`, `MimoV26ProShape` as compile-time
constants (MiMo V2.6 Pro's routed expert: 6144/2048, SiLU with no clamp). Plans read a slot's projections through
`Nvfp4Quant::decode` over the framework's `MoeBufferRows` (the descriptor's slab bases and strides); `SlabRowBytes`
is each slab's minimum stride.

Bit-exact checks for any change here: `test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh` (the avx2 and scalar tiers
of one build against a baseline worktree's dumps), and
`test/registered/unit/kernels/test_nvfp4_cpu_{build,experts}.py`.

## Validation boundaries

The native harness (`test_nvfp4_cpu_build.py` builds and runs it) checks repeated multithreaded forwards, accumulation,
every finite E4M3 encoding, skipped/invalid slots and handle lifecycle. The
benchmark checks every selected layer/count against
an independent decoded-weight Q8 reference before and after timing, and
reports error against a separate FP64 W4A16 reference. The differential GGML
test covers every finite signed scale, varying nibbles, GPU scale row
boundaries, and all partial 64-value block lengths, for both tiers (AVX2 where the host has it). The
standalone sanitizer harness is in
`test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp`; compile it together with
`kernel.cpp` and the vendored C source; `test_nvfp4_cpu_build.py` also runs it under ASan/UBSan where the
compiler can link them.

Synthetic arithmetic/layout checks and benchmark smoke runs are not captured
GPU integration, model-level quality validation or an isolated performance
study. Those remain separate runtime validation steps.
