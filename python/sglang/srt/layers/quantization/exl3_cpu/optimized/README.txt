DSV4.1 CPU expert kernels, clean integration v1
=============================================

This component keeps the selected GCC 15 CPU kernels and native OpenMP
execution. It reads the existing unswizzled 3-bit 96-byte tiles directly.
There is no persistent weight repacking or increase in expert weight storage.

Selection
---------
For a single chunk with H=5120, I=2304, one token, residual activation rows,
128-element quantization blocks, 3-bit unswizzled matrices and AVX512BW:
  one expert: range3_unroll (range-aware neighboring-band traversal);
  multiple experts: unroll_small (one-band traversal).
The measured multiple-expert counts are 3 and 5; counts 2, 4 and above 5
use the same default but have no measured performance claim. Invalid/duplicate
routes and multi-token chunks retain the generic path. Existing scalar, AVX2
and other ISA/bit-width fallbacks are retained.

Both selections include the register decoder, hoisted coefficients, register
Hadamard, parallel 128-element preparation/middle stages, fused down output
transform, cache-line output ownership, compact activation scratch, decoder
unrolling and T0 prefetching. Experiment selectors, alternate persistent weight
layouts and benchmark instrumentation have been removed. Scratch is reused
per calling thread. Workers are individually pinned; a stable assignment avoids
repeating affinity syscalls on every forward. The caller is worker zero. Core configuration is frozen at the first
forward or keep-warm. One forward runs at a time: a forward or free that finds another running returns status 3
rather than wait. Keep-warm takes no lock, so its callers must not overlap it with a forward on the same cores.

Build and link
--------------
Requires Linux, GCC 15, OpenMP and the same PyTorch installation/ABI as the
consumer. -march=native targets the build machine; rebuild on another CPU.
GCC 17 is deliberately not selected: it was not a consistent performance win
and its experimental compiler changed arithmetic in the earlier comparison.

Standalone CPU library on divix01 (no CUDA compilation):

  CUDA_HOME=/usr/local/cuda-13.4 /data/models/slang/.venv/bin/python \
    python/sglang/srt/layers/quantization/exl3_cpu/optimized/build.py \
    --cxx /opt/rh/gcc-toolset-15/root/usr/bin/g++ \
    --output /data/models/exl3_exp/clean_integration/sglang/libexl3_cpu.so

Link the consumer against libexl3_cpu.so, torch_cpu, c10 and OpenMP.
Use moe_mul1.h for the ATen layer-registration API (exl3_moe_cpu_make_layer, one tensor per expert and projection)
and cpu_experts_cabi.h for the service API, the five C functions every CPU expert quant exports
(cpu_experts_common/cabi.hpp): register_layer, free_layer, forward, keep_warm and set_cores.
sglang_exl3_cpu_experts_register_layer takes an SglangCpuExpertsLayer (the engine's
expert_stream/host/cpu_experts_abi.h, shared with the NVFP4 kernel): the pinned tier's six slab base pointers and
per-slot strides, with SglangExl3CpuParams (bits, swizzled) as its params. The kernel keeps no reference: the caller
keeps the slabs alive until it frees the layer. Register and forward through the same library instance: layer
handles belong to that instance's registry, which make_layer's tables share. Packed matrix tensors must remain
alive for the registered layer's lifetime. Configure distinct worker core IDs, within the caller's affinity, before
the first forward or keep-warm. The C ABI forward takes one SglangCpuExpertsForward: rows token rows of FP16
activations, FP32 routing weights (converted to FP16) and FP32 output. Status 0 is success, 1 a kernel error, 2
invalid arguments (a refused call leaves out untouched), 3 concurrent use. The ATen forward and free run through the
same functions and raise on a nonzero status.

SGLang integration
------------------
exl3_ext.py selects this source for residual=1, block=128 and links OpenMP.
With SGLANG_DSV41_CPU_EXPERTS=1 it always does: the accuracy flags can be left
unset, and a flag set to any other value is refused. The extension flavor
resid_b128_cpu_v1 prevents reuse of an old CPU kernel cache. Registration and
the C ABI are compiled into one extension; the CPU service resolves the
existing ABI from that extension. Before starting SGLang, set:

  export SGLANG_EXL3_CPU_CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++
  export CUDA_HOME=/usr/local/cuda-13.4
  export EXL3_MOE_CPU_PIN=0
  export OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 OMP_DYNAMIC=FALSE
  export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE

SGLANG_EXL3_CPU_CXX is used only for this extension's build (exl3_ext sets
CXX around it), so the server's other JIT builds keep their compiler; a global
CXX would move them all to GCC 15. The dsv41_baseline recipe sets it.

Keep the existing EXL3 source/build and CPU-offload settings. Configure the
service with 16 workers and cores 18 through 33 (caller 18, helpers 19..33).
Use NUMA node 1 for CPU weights and scratch, e.g. numactl --membind=1 when
launching the runner. Affinity alone does not place memory on NUMA node 1.
The service now pins its caller to the first configured core. Reserve SMT
siblings 54..69 too when isolating these physical cores. Avoid nested OpenMP
teams and other simultaneous consumers of the same cores.

Code layout
-----------
math.hpp holds the arithmetic every tier shares (state decode, Hadamard, activation quantization) and
math_scalar.hpp, math_avx2.hpp and math_avx512.hpp each tier's GEMV tiles; forward_plan.hpp holds the tier
dispatch, the forward driver and the plans; kernel.cpp holds the public API. Registration, validation, the ISA
tier, the worker cores and keep-warm are cpu_experts_common's ExpertForward<Exl3Quant> (Exl3Quant: quant.hpp).
The tier is min(host, EXL3_MOE_CPU_MAX_ISA), read at the first forward or tier query; EXL3_MOE_CPU_REPORT_ISA=1 prints
"exl3 isa <tier>" to stderr then. A forward is ForwardPlan<Shape, Isa>::run (forward_plan.hpp),
picked once per call in Exl3Quant::dispatch: ForwardPlan<Dsv41Shape, Isa::Bw> when Dsv41Shape::accepts the call on
an AVX-512BW host, else ForwardPlan<GenericShape, I> for the host's tier. PlanTraits<Dsv41Shape, Isa::Bw> is the one
specialization: compact scratch, grouped traversal, wide single-expert quantization. A plan reads every layer fact
through its Shape (shapes.hpp): GenericShape from the layer's LayerInfo, Dsv41Shape as compile-time constants
(5120/2304, 3-bit, gated SiLU, activation limit 10; a layer with any other value takes the generic plan). Plans
read experts through an accessor (quant.hpp): TableExperts over make_layer's per-expert
tables, or StridedExperts<Shape> over sglang_exl3_cpu_experts_register_layer's slab bases and strides.

Bit-exact checks for any change here: test/manual/dsv41/run_exl3_cpu_forward_checks.sh (A/B dumps per ISA tier
against the merge-base, the bare and full-stack benches' frozen references, the CPU expert pool tests).

Provenance and validation
-------------------------
Clean sources originate from:
  /data/models/exl3_exp/records/2026-10-01-thread-count-prepared-v2/unroll_small
The single-expert traversal is selected at runtime; the remaining code uses
the measured unroll_small combination. Source extraction plus integration
changes are recorded under /data/models/exl3_exp/clean_integration.

GCC 15 comparison p50 (microseconds, native full CPU forward):
  experts   original (12 workers)   selected (16 workers)
     1             529.185                347.3425
     3            1391.440                896.846
     5            2294.050               1457.160
These are prior experiment results, not new timings of the clean component.
They exclude Python and GPU transfers and use eight real layers with synthetic
activations. Report:
  /data/models/exl3_exp/compiler_comparison_gcc17_compat/gcc17-comparison-20261001-030640-76920-report/index.html

The standalone clean library was compiled and linked against both benchmark
PyTorch 2.12.1 and SGLang PyTorch 2.13.0. Each passed 24 regular saved-output
comparisons and 216 routed cases against frozen reference outputs, bit exact.
The native harness also checks actual team size and individual worker affinity.
These were untimed NUMA0 checks to avoid the reserved NUMA1 benchmark cores.
The combined CUDA extension has not been rebuilt and clean-library latency has
not yet been remeasured. See clean_integration/record.json for final test status.

License: ../LICENSE.exllamav3 (MIT, original exllamav3 authors).
