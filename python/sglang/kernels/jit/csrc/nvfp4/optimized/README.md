# Native NVFP4 CPU expert plugin

Implements the kernel type `Nvfp4CpuKernel` (`entry.h`; `kernels/jit/csrc/moe/expert_stream/host/cpu_experts/binding.hpp`
lists what a kernel type offers) for the CPU expert engine of
`kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h`, parallel to `csrc/exl3/optimized`. The kernel and its
build are native C/C++ only. The engine's job ring, leases, pinned activation/output rows and completion signaling are
reused.

Inside SGLang the kernel is compiled into the expert-stream host module (`kernels/ops/moe/cpu_kernel_build.py`):
`kernel.cpp` and the vendored C (`../upstream/nvfp4.c`, compiled as C through `load_jit`'s `source_cflags`) join the
host's files, each with `-Ofast -march=<resolved>` (`SGLANG_JIT_HOST_MARCH`, else this machine's); the host's own files
keep the host flags. One ISA tier is compiled, chosen at compile time from that `-march`: AVX2 when it has AVX2/FMA/F16C,
else the scalar loop. A lower tier is a build at a lower `-march` (`x86-64-v3` AVX2, `x86-64` scalar); the host module
runs only on CPUs that have its `-march`. The results are not bit-identical to GGML's (FMA contraction and fast math);
the GGML check compares at a tolerance.

For the native harnesses, build from the checkout root, on the Linux machine that will run it (GCC with OpenMP), the
kernel linked with a harness source into an executable:

```sh
python python/sglang/srt/layers/quantization/nvfp4/build.py \
  --cxx /opt/rh/gcc-toolset-15/root/usr/bin/g++ --main HARNESS.cpp --output /absolute/path/harness
```

The native harnesses (`test/registered/unit/kernels/nvfp4_cpu_{sanitizer,ggml_check}.cpp`) build and run under
`test/registered/unit/kernels/test_nvfp4_cpu_build.py`.

See [the benchmark guide](../bench/README.md) for the native benchmark
executable, fixed-team process rounds, fixture format and timing protocol.

See [upstream provenance and adaptation details](../upstream/README.md).

## Attach to the existing engine

The kernel type `Nvfp4CpuKernel` (`entry.h`) is declared where the host sees it and defined in `kernel.cpp`, the kernel's
own translation unit (hidden visibility); the host calls it directly and Python passes no address. The host makes each
layer with the kernel's `make_layer` (`ExpertStreamHost.set_cpu_layer`) from a `CpuExpertLayerSpec`
(`Nvfp4CpuQuantTrait.layer_spec`): seven slabs and the shape (an `ExpertLayer<SglangNvfp4CpuParams>`, `kernel.hpp`) and
the layer's parameter list `[w13_layout, inv_input_scale13 bits, inv_input_scale2 bits]` (the scales as IEEE binary32
bit patterns in int64 values), from which `Nvfp4CpuKernel::params` builds the `SglangNvfp4CpuParams` (`entry.h`) that
`make_layer` validates. A layer is a value of views, its params stored in it; the registrant keeps the slabs alive while
the host may read them. The optimized kernel never repacks or expands full weight rows. A native caller does the same
through `kernel.h`:

```cpp
using Kernel = ::sglang::nvfp4_cpu::Nvfp4CpuKernel;
Kernel::Layer shape;  // this layer's verified slab pointers and strides, its dimensions, and shape.params
const Kernel::Layer layer = Kernel::make_layer(shape);
ForwardCall call;  // rows, k, x, slots, weights, out, out_mode
Kernel::check(layer, call);  // throws std::invalid_argument on a refused call
// `threads` workers, worker i on cores[i] (distinct, at least `threads` of them); a refused team also throws
::sglang::cpu_experts::forward_alone<Kernel>(layer, call, cores, threads);
```

A refused call throws `std::invalid_argument`, a failure another `std::exception`; either leaves `out` untouched.
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
deltas above finite FP16 range and nonfinite activations are refused (`std::invalid_argument`);
blocks whose delta rounds to zero contribute zero.

## Threading and lifetime

Each forward runs on a `Team` (`expert_stream/host/cpu_experts/team.hpp`) of `threads` workers, the calling thread
as worker 0, in four phases separated by the team's barriers: every token's input to Q8_0; every routed expert's
gate/up rows and SiLU; every intermediate to Q8_0; every expert's down rows, each token's summed in its routing order
into its row of `out`. The CPU expert engine keeps one team for its lifetime and passes it to every forward; the
`forward_alone<Nvfp4CpuKernel>(layer, call, cores, threads)` (tests, the bench) makes a team for the call alone,
pinning worker i to `cores[i]` (no cores: unpinned). A team may have any size up to its cores (more is refused:
`std::invalid_argument`); a worker that cannot be pinned fails the call and leaves `out` untouched. No OpenMP
variable reaches the team. Forwards on any cores run at once from different threads. Between jobs the engine's team
runs the kernel's `warm` loop, register-only work at the forward's vector width, then PAUSE. Stop/join the engine
before freeing layers or slab storage. The kernel requires Linux.

The forward takes one `ForwardCall` (`kernel.hpp`, shared with the EXL3 kernel): up to 65536 token rows of up
to eight lanes each. It skips -1 slots, preserves each row's routing order
including duplicate slots, and supports overwrite or accumulation. Rows that
route to the same slot are computed together, up to four per decoded weight
row; every row's output matches its own one-row call's within one Q8_0 quantum of the row's maximum (not bitwise: the
-Ofast build may vectorize a batch differently). The engine sends
one row per job. Caller
pointer extents and finite activation/block-scale values are required. Raw
pointers cannot prove allocation size. A refused call throws `std::invalid_argument`, a failure another
`std::exception`; the engine fails stop on either, as it did on a nonzero status.

## Code layout

Layer and argument validation, worker cores and keep-warm are the shared CPU experts framework's
(`expert_stream/host/cpu_experts/`, `ExpertForward<Nvfp4Quant>`). `quant.hpp` holds `Nvfp4Quant` (slab names and
`row_bytes`, each slab's minimum stride; parameter validation; `Expert`, a slot's three projections read from its
`ExpertRow`); `math.hpp` the ISA-independent arithmetic (`GpuRow`, Q8_0 quantization, the gated SiLU) and
`dot_rows<Isa, M>`, whose tiers are `math_scalar.hpp` and `math_avx2.hpp` (present when the `-march` has AVX2); `forward_plan.hpp`
holds the plan, its types (`RouteBinding`, `Chunk`, `ForwardCtx`) and its per-thread scratch, `ForwardArena`;
`kernel.cpp` defines `Nvfp4Quant::dispatch` and the kernel's accessor (`nvfp4_cpu_kernel`). A forward is
`ForwardPlan<Shape, Isa>::run` (`forward_plan.hpp`), picked once per call in `Nvfp4Quant::dispatch` from the build's
tier: `ForwardPlan<MimoV26ProShape, Isa::Avx2>` when `MimoV26ProShape::accepts` the layer at the AVX2 tier, else
`ForwardPlan<GenericShape, Isa::Avx2>` or `ForwardPlan<GenericShape, Isa::Scalar>`.
An AVX2 plan enters its gate/up and down row loops through `gate_up_avx2`/`down_avx2`, with everything they call
inlined (`flatten`), once per phase per worker.
`kRowUnit` is the plans' split unit. The plan builds the call's `RouteTable` and groups a call's routes into
units, one per (token, slot), and units into chunks of up to `kChunkRows` (4) of one slot; `dot_rows<Isa, M>` decodes
each weight row once per chunk. A plan reads every layer fact it may fix
through its Shape (`shapes.hpp`): `GenericShape` from the `ExpertLayer`, `MimoV26ProShape` as compile-time constants
(MiMo V2.6 Pro's routed expert: 6144/2048, SiLU with no clamp); the W13 layout and input scales are the layer's params.
Plans read a slot's projections through `Nvfp4Quant::expert(layer[slot])`.

Checks for any change here: `test/manual/dsv41/run_nvfp4_cpu_forward_checks.sh` (the avx2 and scalar tiers, each its
own build, against a baseline worktree's dumps at a tolerance), and
`test/registered/unit/kernels/test_nvfp4_cpu_{build,experts}.py`.

## Validation boundaries

The native harness (`test_nvfp4_cpu_build.py` builds and runs it) checks repeated multithreaded forwards, accumulation,
every finite E4M3 encoding, skipped/invalid slots and handle lifecycle. The
benchmark checks every selected layer/count against
an independent decoded-weight Q8 reference before and after timing, and
reports error against a separate FP64 W4A16 reference. The differential GGML
test covers every finite signed scale, varying nibbles, GPU scale row
boundaries, and all partial 64-value block lengths, at a tolerance, for the build's tier. The
standalone sanitizer harness is in
`test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp`; compile it together with
`kernel.cpp` and the vendored C source; `test_nvfp4_cpu_build.py` also runs it under ASan/UBSan where the
compiler can link them.

Synthetic arithmetic/layout checks and benchmark smoke runs are not captured
GPU integration, model-level quality validation or an isolated performance
study. Those remain separate runtime validation steps.
