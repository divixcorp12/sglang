Native DSV4.1 CPU expert full-forward benchmark
=============================================

For persistent on-demand CPU isolation with sudo-free service start/stop after
one administrator installation, see service/README.txt. That service reserves
the cores only while the benchmark runs and leaves both memory nodes available.

Google Benchmark drives two separate C++ executables:
  exl3_cpu_baseline: existing vendored residual/block-128 kernel, 12 workers;
  exl3_cpu_optimized: clean production winner, 16 workers.
Override --workers=N to compare both with the same count. The baseline uses its
original native pool; optimized uses OpenMP. Separate processes prevent one
backend's idle workers from contaminating the other's measurements. The baseline
source is the repo's exl3_cpu/moe_mul1.cpp, not a new frozen historical snapshot.

There is no Python at configure, build or runtime. These binaries link ATen/c10
from LibTorch or the server's installed torch directory, not Python bindings or
libtorch_python. Google Benchmark v1.9.4 is pinned and fetched at configure if
not installed; set EXL3_FETCH_BENCHMARK=OFF for offline dependency management.
The kernel compiles with the validated GCC15 -Ofast -march=native options;
the harness compiles without fast-math so its finite-output checks work.

Build on divix01
---------------
From the SGLang checkout, set:

  bench=python/sglang/kernels/jit/csrc/moe/expert_stream/bench
  build=/data/models/exl3_exp/google_benchmark/build
  export TMPDIR=/data/models/exl3_exp/google_benchmark
  mkdir -p "$TMPDIR"
  cmake -S "$bench" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
    -DEXL3_TORCH_ROOT=/data/models/slang/.venv/lib/python3.13/site-packages/torch \
    -DEXL3_CXX11_ABI=1
  cmake --build "$build" -j4

Use the same torch C++ ABI as the server; divix01's installation uses ABI=1.
CUDA_HOME is unnecessary for this CPU-only build. Re-running cmake --build
automatically rebuilds changed source/header files. The local CPU's ISA is baked
into -march=native; rebuild on the machine where the benchmark will run.

Inputs, correctness and placement
--------------------------------
Defaults use our original fixture and frozen regular-forward references:
  --fixture=/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin
  --reference-dir=/data/models/exl3_exp/threading
References are reference-e1.bin, reference-e3.bin and reference-e5.bin (FP32,
eight layers in order). Fixture SHA256:
  55de978d5eb3c9d1bc3ac38b5a7937ea724a47fea2c0cf87c04bd1616e188a8c
The fixture uses real weights and synthetic FP16 activations, H5120/I2304,
eight layers and five experts per layer. Swizzled weights, other shapes, truncated
files and trailing bytes are rejected. The reference checks use precisely the
previous harness's SiLU limit=10 and routing coefficients. No fixture/reference
generation is done by this benchmark. Supply paths explicitly on other hosts.

Default --cpus=18-33 and --numa-node=1. Worker zero/caller stays on CPU18;
baseline helpers use19..29, optimized helpers use19..33. --workers selects the
first N CPUs in the list. The runner verifies requested CPUs belong to the node,
are allowed by its affinity/cgroup, and that the real workers are individually
pinned. Do not taskset the entire process to CPU18: helpers need access to all
their CPUs. Reserve SMT siblings54..69 too for isolated timings.

No process-wide NUMA memory policy is imposed. The process inherits its launch
policy, and the CPU affinity is independent of that policy. This allows weights
on node0 and computation on node1. To explicitly test this remote-memory case,
prefix each executable with numactl --membind=0 (scratch also follows that policy).
Without a memory policy, fixture loading occurs on the pinned caller and normally
first-touches node1. This fixture is ordinary CPU RAM, not CUDA-pinned server
slabs; the benchmark excludes PCIe/NVMe transfers and the service job-ring handoff.

Before timing: verify all 24 outputs bit exactly and verify helper affinity.
Before and after each measured run: repeat all eight outputs for that expert
count. Each run warms up with 128 forwards (--warmup-forwards overrides this).
Timed calls are not individually compared with reference output. Setup, fixture
I/O, registration, warmup and comparisons are excluded from reported latency.
--gap-us=N optionally sleeps before each timed call to test wakeup/idle behavior;
the sleep is excluded from the measured interval. The default is uninterrupted
back-to-back forwards, as in the selected winning studies.

Run
---
Inside the existing isolated shell (this script does not alter cgroups):

  bash "$bench/run.sh" "$build" \
    "/data/models/exl3_exp/google_benchmark/run-$(date +%Y%m%d-%H%M%S)"

Default: eight alternating baseline/optimized process rounds, 512 full forwards
per expert count per process, counts1/3/5, rotating layers. Original/baseline is
always included. EXL3_BENCH_ROUNDS changes the process-round count. Extra flags
after the output directory are forwarded to both executables. Example equal-team
comparison: append --workers=16. Existing isolation must include all active CPUs.
The script writes binary hashes, affinity/cgroup context, logs and Google JSON
results to a new directory and refuses to reuse an existing directory.

Single-process examples:
  export EXL3_MOE_CPU_MAX_ISA=bw OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
  export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE
  "$build/exl3_cpu_optimized" --validate-only
  "$build/exl3_cpu_optimized" --benchmark_filter='experts:1/' \
    --benchmark_min_time=512x --benchmark_repetitions=8 \
    --benchmark_out=optimized.json --benchmark_out_format=json
  "$build/exl3_cpu_baseline" --benchmark_min_time=512x

Timing interpretation
---------------------
One iteration is one native service C ABI full forward: routing-weight conversion,
kernel preparation, gate/up, activation/down preparation, down and output reduction,
including OpenMP scheduling/barriers. No Python, GPU work or driver process work
is inside it. Manual timing brackets only this C++ call with steady_clock;
framework/sample recording and layer rotation occur outside that interval.
Google's Time column is the mean measured full-forward wall latency in us.
p50_us, p95_us and p99_us are per-call wall-latency quantiles, calculated outside
timing. The CPU column measures the caller's CPU consumption, not all helpers;
do not interpret it as total team CPU time. Google repetition median rows are
medians of run means, distinct from our per-call p50_us counters. A failed output,
team or affinity check returns nonzero; timings from a failed run are invalid.
For process-level comparisons use the median of p50_us across process rounds,
as in the previous experiments. This migration does not establish a new speedup.

Google Benchmark reference:
https://github.com/google/benchmark/blob/v1.9.4/docs/user_guide.md
