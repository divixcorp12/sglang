# Native NVFP4 CPU expert plugin

Implements the `CpuExpertForward` callback from
`kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h`, parallel to
`exl3_cpu/optimized`. The plugin and build are C++ only. The engine's job ring,
leases, pinned activation/output rows and completion signaling are reused.

Build from the checkout root:

```sh
src=python/sglang/srt/layers/quantization/nvfp4_cpu
cmake -S "$src" -B /absolute/path/build -DCMAKE_BUILD_TYPE=Release \
  -DNVFP4_CPU_NATIVE=ON
cmake --build /absolute/path/build -j4
ctest --test-dir /absolute/path/build --output-on-failure
```

C++17 and pthreads suffice. No Python, PyTorch, GGML, CUDA, OpenMP or specific
GCC release is required. The shared library is `libsglang_nvfp4_cpu.so` on
Linux. Omit `NVFP4_CPU_NATIVE=ON` for the portable scalar path. Native builds
select AVX2 on supported x86 CPUs; rebuild before moving to a different ISA.

See [the benchmark guide](../bench/README.md) for native baseline/optimized
executables, fixed-team process rounds, fixture format and timing protocol.

## Attach to the existing engine

Include `cpu_experts_cabi.h` and link the shared library or compile
`moe_mul1.cpp` into the same native module, as with EXL3. Register each layer
using a `SglangNvfp4CpuLayer` descriptor. Registration stores views and allocates
only activation/result scratch; it never repacks or expands weights.

```cpp
int64_t handle = -1;
// Populate descriptor with this layer's verified layout/scales/slab pointers.
if (sglang_nvfp4_cpu_experts_register_slabs(&descriptor, &handle) != 0)
    throw std::runtime_error("NVFP4 CPU slab registration failed");

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
activation factors as `inv_input_scale13` and `inv_input_scale2` to cancel them
for full FP16 CPU input. Use 1.0 for weight-only alphas. Factors may differ
between layer descriptors, but each factor is scalar within one layer. A
per-expert factor must travel with its mutable host slot in a future ABI
extension. Omitting slab6 asserts gate/up share the same global weight scale.

The kernel computes W4A16: FP16 input, FP32 gate/up projections, ordinary
`SiLU(gate)*up`, down projection, routing and reduction. Positive `act_limit`
applies `gate=min(gate,L)` and `up=clamp(up,-L,L)` before SiLU; 0 disables it.
Other activation conventions (Bailing post-SiLU clamps, GPT-OSS shifted/scaled
SwiGLU, SiTU, GELU, ReLU2, non-gated experts) must not use this descriptor.
It is not bit-identical to GPU W4A4 activation quantization.

## Threading and lifetime

One process-wide persistent helper pool is owned by the calling engine thread
(worker zero). Configure distinct allowed Linux cores before the first forward.
The first call fixes maximum workers; later calls may use fewer. Concurrent
forward/free/configuration is rejected. Stop/join the engine before freeing
handles or slab storage; do not unload the library while callbacks/workers are
in use. Standalone unpinned arithmetic tests also build on macOS; explicit
production worker placement requires Linux.

The callback accepts up to eight lanes, skips -1 slots, preserves routing order
including duplicate slots, and supports overwrite or accumulation. Caller
pointer extents and finite activation/block-scale values are required. Raw
pointers cannot prove allocation size. Status: 0 success, 1 internal error,
2 invalid arguments, 3 concurrent use. The engine's nonzero-status fail-stop
behavior is unchanged.

## Validation boundaries

The native CTest harness checks repeated multithreaded forwards, accumulation,
every finite E4M3 encoding, skipped/invalid slots and handle lifecycle. The
benchmark checks every selected layer/count against
an independent FP64 decoded-weight reference before and after timing. The
standalone sanitizer harness is in
`test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp`; compile it together with
`moe_mul1.cpp` using `-fsanitize=address,undefined -pthread` for ASan/UBSan.

Synthetic arithmetic/layout checks and benchmark smoke runs are not captured
GPU integration, model-level quality validation or an isolated performance
study. Those remain separate runtime validation steps.
