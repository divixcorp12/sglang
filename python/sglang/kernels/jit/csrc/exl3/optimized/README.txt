DSV4.1 CPU expert kernels, clean integration v1
=============================================

This component keeps the selected GCC 15 CPU kernels and native OpenMP
execution. It reads the existing unswizzled 3-bit 96-byte tiles directly.
There is no persistent weight repacking or increase in expert weight storage.

Selection
---------
For a call whose chunks each hold 1..CHUNK_M tokens (CHUNK_M = MAX_M / 2; MAX_M is
EXL3_MOE_CPU_MAX_M: 4 in the source, 16 in the server's build via SGLANG_EXL3_CPU_MAX_M), with H=5120, I=2304, residual activation rows,
128-element quantization blocks, 3-bit unswizzled matrices and AVX512BW:
  one expert, one token: range3_unroll (range-aware neighboring-band traversal);
  other one-token chunks: unroll_small (one-band traversal);
  chunks of M > 1 tokens: register_tiles<M>, all M tokens sharing each decoded tile,
  kRegisterBudget[M - 1].pairs tile pairs per call; M = 5..8 through register_band_halves,
  which runs each 128-input block once per output half to hold half the accumulators.
On a VNNI host one-token chunks accumulate the decoded product bytes with vpdpbusd (no byte sum)
and larger chunks with vpdpwssd; on AVX-512BW alone, vpmaddwd + vpaddd. The generic plan splits
calls into chunks of at most 4 tokens (GENERIC_CHUNK_M) whatever MAX_M.
kRegisterBudget (math_avx512.hpp) holds one entry per M and a static_assert keeps each
within 32 zmm; raising MAX_M needs an entry per new M. The measured multiple-expert
counts are 3 and 5; counts 2, 4 and above 5 use the same default but have no
measured performance claim. Every chunk size is bit-identical to the generic plan
(test_exl3_cpu_dsv41_plan_multitoken.py). Invalid routes retain the generic path.
Existing scalar, AVX2 and other ISA/bit-width fallbacks are retained.

Both selections include the register decoder, hoisted coefficients, register Hadamard, parallel 128-element
preparation/middle stages, fused down output transform, cache-line output ownership, compact activation scratch,
decoder unrolling and T0 prefetching. Experiment selectors, alternate persistent weight layouts and benchmark
instrumentation have been removed. Scratch is reused per calling thread. Workers are individually pinned; a stable
assignment avoids repeating affinity syscalls on every forward. The caller is worker zero. Each call carries its
cores (worker i on cores[i]; none: unpinned workers), and forwards on different cores run at once from different
threads. A layer is owned by its holder (the host's per-tier layer table), which frees it only after every engine has
stopped. Keep-warm takes no lock, so its callers must not overlap it with a forward on the same cores.

Build and link
--------------
Requires Linux, GCC 15 and OpenMP. No torch: kernel.cpp and its headers use a plain fp16 type (half.hpp; F16C when the
-march has it, else the exact software conversion) and std exceptions. The build is -Ofast -march=<resolved>
(SGLANG_JIT_HOST_MARCH, else this machine's, via jit/utils/compile/toolchain.py): it targets the build machine, so
rebuild on another CPU.
GCC 17 is deliberately not selected: it was not a consistent performance win
and its experimental compiler changed arithmetic in the earlier comparison.

The server compiles kernel.cpp into the expert-stream host module (kernels/ops/moe/cpu_kernel_build.py): one translation
unit with its own flags (load_jit's source_cflags) beside the host's files, which keep the host flags. The host's
compiler (CXX) must be GCC 15 when CPU experts are on (check_kernel_compiler).

Standalone CPU library on divix01 (no CUDA compilation, no torch):

  /data/models/slang/.venv-upstream/bin/python \
    python/sglang/srt/layers/quantization/exl3/build.py \
    --cxx /opt/rh/gcc-toolset-15/root/usr/bin/g++ \
    --output /data/models/exl3_exp/clean_integration/sglang/libexl3_cpu.so

Link the consumer against libexl3_cpu.so and OpenMP and use kernel.h: the kernel's make_layer over the experts' slabs,
then check and forward (below). The vendored per-expert kernel (../moe_mul1.cpp, which the expert-stream bench builds
as its baseline against baseline_moe_mul1.h) is a separate thing; nothing in this directory declares upstream's API.

The service side is the concrete kernel type Exl3CpuKernel (entry.h; binding.hpp lists what a kernel type offers), declared
where the host sees it and defined in kernel.cpp (hidden visibility). The host calls it directly: Python passes no
address. Python makes each streamed layer with the kernel's make_layer (ExpertStreamHost.set_cpu_layer) from a
CpuExpertLayerSpec (Exl3CpuQuantTrait.layer_spec): the pinned tier's six slab base pointers and per-slot strides, with
the layer's parameter list [bits_w13, bits_w2, swizzled, row_weighted, plan] as int64 values; Exl3CpuKernel::params builds
the SglangExl3CpuParams (entry.h) from them and make_layer validates it. A layer (ExpertLayer<SglangExl3CpuParams>,
kernel.hpp) is a value of views with its params stored in it: the caller keeps the slabs alive. Every forward and
keep-warm carries its worker cores (distinct, in [0, CPU_SETSIZE)); a core that cannot be pinned fails the call. A
refused call throws std::invalid_argument and a failed one another std::exception, leaving out untouched.

SGLang integration
------------------
With SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1 (or the draft's CPU experts) the host module's kernel is always the
residual=1, block=128 flavor: the accuracy flags can be left unset, and a flag set to any other value is refused. The
exllamav3 extension (quantization/exl3/ext.py) no longer builds this kernel: with CPU experts on it compiles upstream's
own cpu/moe_mul1.cpp as it is. Before starting SGLang, set:

  export CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++
  export CUDA_HOME=/usr/local/cuda-13.4
  export SGLANG_EXL3_MOE_CPU_PIN=0
  export OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 OMP_DYNAMIC=FALSE
  export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE

CXX moves every JIT build (host C++, and nvcc's host half through -ccbin) to GCC 15. SGLANG_EXL3_CPU_CXX, when set, must
name the same compiler. The dsv41_baseline recipe sets both.

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
dispatch, the forward driver and the plans; kernel.cpp defines Exl3CpuKernel (entry.h). Registration, validation, the ISA
tier, the worker cores and keep-warm are the CPU experts framework's ExpertForward<Exl3Quant> (Exl3Quant: quant.hpp);
projection.hpp holds Exl3Projection and half.hpp the fp16 type.
The tier is compile-time: the build's -march (isa.hpp's kBuildIsa: AVX512VBMI+VNNI Vbmi, AVX512VNNI Vnni, AVX512BW Bw,
AVX2 Avx2, else Scalar), and only that tier's plans are compiled. A lower tier is a build at a lower -march
(SGLANG_JIT_HOST_MARCH: skylake-avx512 Bw, x86-64-v3 Avx2, x86-64 Scalar). The host installs the worker-trace options
(Exl3CpuKernel::configure) when it loads the module; a native harness calls it itself. A forward is
ForwardPlan<Shape, Isa>::run (forward_plan.hpp),
picked once per layer, at registration: the layer's params name a plan id (ForwardPlan<Dsv41Shape, Isa::Bw>, compiled
from the Bw tier up, or ForwardPlan<GenericShape, I> for the build's tier), Python choosing it from the list
the host's cpu_kernel_plans returns (each plan's facts, ISA floor and availability) by the layer's facts, and make_layer
refusing a plan that does not accept the layer or that this build does not compile (plan_rejects, forward_plan.hpp).
Exl3Quant::dispatch then runs p.plan with no per-call check. In a VNNI build (tier Vnni or Vbmi) the same
plan runs with its Vnni flag set: its register kernels (register_kernels.hpp, included once as reg_bw and once as
reg_vnni by math_avx512.hpp) accumulate with vpdpwssd instead of vpmaddwd + vpaddd, bit-identically (the 16-bit byte sums
and activation pairs are unchanged; both forms wrap mod 2^32). A Bw build runs the reg_bw copy.
PlanTraits<Dsv41Shape, Isa::Bw> is the one
specialization: compact scratch (chunks of 1..CHUNK_M tokens), grouped traversal and wide single-expert quantization (one token).
MAX_M (math.hpp) is one build-time constant, EXL3_MOE_CPU_MAX_M; the generic tiers' tile tables (kTiles) and the
register dispatch (register_tiles_for) are generated from it. A plan reads every layer fact
through its Shape (shapes.hpp): GenericShape from the ExpertLayer, Dsv41Shape as compile-time constants (5120/2304,
3-bit gate/up and down, unswizzled, activation limit 10; a layer with any other value takes the generic plan). The
generic plan decodes gate/up at bits_w13 and down at bits_w2. Every layer is gated SiLU.
Plans read experts through Experts<Shape> (quant.hpp): slot e's gate, up and down, each an Exl3Projection over the
slot's ExpertRow. Exl3Quant::dispatch groups the call's routes by expert itself (a sort of the call's live routes), which
fixes the accumulation order.

Bit-exact checks for any change here: test/manual/dsv41/run_exl3_cpu_forward_checks.sh (A/B dumps per ISA tier, each
a build at its -march, against the merge-base, the bare and full-stack benches' frozen references, the CPU expert pool tests).

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
