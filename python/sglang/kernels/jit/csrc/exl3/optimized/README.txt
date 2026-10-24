DSV4.1 CPU expert kernels, clean integration v1
=============================================

This component keeps the selected GCC 15 CPU kernels and native OpenMP
execution. It reads the existing unswizzled 3-bit 96-byte tiles directly.
There is no persistent weight repacking or increase in expert weight storage.

Selection
---------
Every build quantizes activations the same way: each token row to int8 in 128-input blocks, one scale per block, plus
its residual as a second row (math.hpp's kActRows and kActBlock; there is no other flavor).
For a call whose chunks each hold 1..CHUNK_M tokens (CHUNK_M = MAX_M / 2; MAX_M is
EXL3_MOE_CPU_MAX_M: 4 in the source, 16 in the server's build via SGLANG_EXL3_CPU_MAX_M), with H=5120, I=2304,
3-bit unswizzled matrices and AVX512BW:
  one expert, one token: range3_unroll (range-aware neighboring-band traversal);
  other one-token chunks: unroll_small (one-band traversal);
  chunks of M > 1 tokens: register_tiles<M>, all M tokens sharing each decoded tile,
  kRegisterBudget[M - 1].pairs tile pairs per call; M = 5..8 through register_band_halves,
  which runs each 128-input block once per output half to hold half the accumulators.
On a VNNI host one-token chunks accumulate the decoded product bytes with vpdpbusd (no byte sum)
and larger chunks with vpdpwssd; on AVX-512BW alone, vpmaddwd + vpaddd. The generic plan splits
calls into chunks of at most 4 tokens (GENERIC_CHUNK_M) whatever MAX_M.
kRegisterBudget (register_kernels.hpp) holds one entry per M and a static_assert keeps each
within 32 zmm; raising MAX_M needs an entry per new M. The measured multiple-expert
counts are 3 and 5; counts 2, 4 and above 5 use the same default but have no
measured performance claim. Every chunk size is bit-identical to the generic plan
(test_exl3_cpu_dsv41_plan_multitoken.py). Invalid routes retain the generic path.
Existing scalar, AVX2 and other ISA/bit-width fallbacks are retained.

Both selections include the register decoder, hoisted coefficients, register Hadamard, parallel 128-element
preparation/middle stages, a down output transform fused into the weighted sum, cache-line output ownership, compact
activation scratch, decoder unrolling and T0 prefetching. Experiment selectors, alternate persistent weight layouts and
benchmark instrumentation have been removed. Scratch is reused per job slot. Workers are individually pinned; a stable
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
the layer's parameter list [bits_w13, bits_w2, swizzled, plan] as int64 values; Exl3CpuKernel::params builds
the SglangExl3CpuParams (entry.h) from them and make_layer validates it. A layer (ExpertLayer<SglangExl3CpuParams>,
kernel.hpp) is a value of views with its params stored in it: the caller keeps the slabs alive. Every forward and
keep-warm carries its worker cores (distinct, in [0, CPU_SETSIZE)); a core that cannot be pinned fails the call. A
refused call throws std::invalid_argument and a failed one another std::exception, leaving out untouched.

SGLang integration
------------------
With SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1 (or the draft's CPU experts) the vendored kernel's accuracy flags
(SGLANG_EXL3_CPU_ACT_RESIDUAL, SGLANG_EXL3_CPU_ACT_BLOCK) can be left unset, and a value other than this kernel's
(residual, 128) is refused. The
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
  math.hpp              build constants (kIsa, MAX_M, kActRows, kActBlock), Unroll/with_count, the format tables and
                        scalar decode, the Hadamard transform, PreparedIn and the activation quantization
  math_scalar.hpp       Tiles<Scalar>
  math_avx2.hpp         Tiles<Avx2>
  math_avx512.hpp       Tiles<Bw | Vnni | Vbmi>: one band driver (band_tiles) over BandRows<Isa>, each tier's decode and
                        integer accumulate
  register_kernels.hpp  the 3-bit register kernels (register_tiles<M>, kRegisterBudget): the DSV4.1 plan's GEMVs
  tiles.hpp             run_tiles: a chunk's GEMV on the build's tier (register kernels, or Tiles per 128-input block)
  phases.hpp            ForwardCtx, ForwardArena, ForwardJob and the phases' work besides the GEMVs (prepare, transform, SiLU)
  forward_plan.hpp      PlanTraits, ForwardPlan<Shape> (the five phases' tasks), prepare_job and the plan registry
  shapes.hpp            the plans' Shapes and Chunk; quant.hpp Exl3Quant and Experts<Shape>; projection.hpp
                        Exl3Projection; half.hpp the fp16 type; kernel.cpp Exl3CpuKernel (entry.h)
Registration, validation, the worker cores and keep-warm are the CPU experts framework's ExpertForward<Exl3Quant>.
The tier is compile-time: the build's -march (isa.hpp's kBuildIsa: AVX512VBMI+VNNI Vbmi, AVX512VNNI Vnni, AVX512BW Bw,
AVX2 Avx2, else Scalar), and only that tier is compiled: code holding a tier's instructions sits under isa.hpp's
SGLANG_CPU_EXPERTS_{AVX2,AVX512,VNNI,VBMI} flags, not per-function target attributes. A lower tier is a build at a lower
-march (SGLANG_JIT_HOST_MARCH: skylake-avx512 Bw, x86-64-v3 Avx2, x86-64 Scalar). The host installs the worker-trace
options (Exl3CpuKernel::configure) when it loads the module; a native harness calls it itself (the worker trace is fed by
no job: a job's phases are not one team forward. The instr engine's job trace, SGLANG_EXPERT_STREAM_TRACE's job.prefix,
holds a `cpu_run` event per run of tasks -- slot, phase, worker, tasks and TSC cycles -- from the window). A call is one job (entry.h's job
interface): prepare_job at admission, then five phases of tasks any worker may run, ForwardPlan<Shape> at the build's
tier, picked once per layer, at registration:
  PrepareGateUp  (chunk, gate | up, 128-input block): transform and quantize          80 tasks for one DSV4.1 expert
  GateUp         (chunk, gate | up, tile pair), runs aligned to PlanTraits::kTileAlign  144
  Middle         (chunk, 128-intermediate block): SiLU and the down input             18   (kGatePhase)
  DownTiles      (chunk, tile pair) of the down GEMV, into tout_d; the worker that
                 finishes a 128-output block's last tile transforms it                160
  DownOut        a 16-output unit over every chunk in order: the weighted sum
                 (it zeroes its columns first for kOverwrite)                         320  (kOutPhase)
The scalar tier's PrepareGateUp and Middle tasks are (chunk, gate | up) and a chunk. The grouped multi-band traversal runs
inside a claimed run. A task's arithmetic does not depend on the worker that runs it. The layer's
params name a plan id (ForwardPlan<Dsv41Shape>, compiled from the Bw tier up, or ForwardPlan<GenericShape>), Python
choosing it from the list the host's cpu_kernel_plans returns (each plan's facts, ISA floor and availability) by the
layer's facts, and make_layer refusing a plan that does not accept the layer or that this build does not compile
(plan_rejects, forward_plan.hpp). The tasks then run p.plan with no per-task check. The register kernels
accumulate with the build's instructions: vpdpwssd on a VNNI tier, vpmaddwd + vpaddd on Bw, bit-identically (the 16-bit
byte sums and activation pairs are unchanged; both forms wrap mod 2^32).
PlanTraits<Dsv41Shape> is the one specialization: compact scratch (chunks of 1..CHUNK_M tokens), grouped traversal and
wide single-expert quantization (one token).
MAX_M (math.hpp) is one build-time constant, EXL3_MOE_CPU_MAX_M; the generic tiles' row counts and the register
dispatch (with_count<CHUNK_M>) are generated from it. A plan reads every layer fact
through its Shape (shapes.hpp): GenericShape from the ExpertLayer, Dsv41Shape as compile-time constants (5120/2304,
3-bit gate/up and down, unswizzled, activation limit 10; a layer with any other value takes the generic plan). The
generic plan decodes gate/up at bits_w13 and down at bits_w2. Every layer is gated SiLU.
Plans read experts through Experts<Shape> (quant.hpp): slot e's gate, up and down, each an Exl3Projection over the
slot's ExpertRow. prepare_job groups the call's routes by expert itself (a sort of the call's live routes), which
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

Execution ports (2026-10-24, divix01, Xeon Platinum 8280, vnni build of 922aa59be8, MAX_M 16). Per forward of the
bare bench, from ../../moe/expert_stream/bench/port_counters.sh (two runs of 256 and 4352 iterations, differenced, so
setup cancels) and port_counters.py; user-space counts, uops dispatched per core cycle:
  bench        workers  port 0  port 1  port 5  mem stall  all stalls  cycles/ref  AVX-512 heavy license
  experts:1       1      0.89    0.03    0.77     0.01        0.02        1.31          100%
  experts:5       1      0.86    0.03    0.75     0.06        0.05        1.30          100%
  rows:6/k:6      1      0.79    0.03    0.69     0.07        0.09        1.29          100%
  experts:1      16      0.67    0.02    0.59     0.13        0.24        1.00          100%
  experts:5      16      0.64    0.02    0.56     0.09        0.27        1.00          100%
  rows:6/k:6     16      0.63    0.03    0.56     0.13        0.25        1.01          100%
On one core the decode is port-0 bound (vpdpwssd/vpdpbusd, 512-bit multiplies and shifts), port 1 is fused away at 512
bits, and memory stalls are small. At 16 workers the clock drops to the reference frequency, a quarter of the cycles
stall (9-13% on memory) and a forward costs 34% more summed cycles than on one core. A 256-bit VNNI tier would bring
port 1 back and the lighter frequency license, but doubles every operation and port 5 is already at 0.75-0.77, so it
would likely bind on port 5: not built; estimated a small gain at best. Raw data: divix01:/mnt/nvme1/exl3-tidy/perf1.

License: ../LICENSE.exllamav3 (MIT, original exllamav3 authors).
