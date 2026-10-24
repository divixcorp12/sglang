Native DSV4.1 CPU expert benchmarks
===================================

Two benchmark families live here:
  - the bare forward: one full CPU-expert forward through the kernel interface,
    baseline kernel against optimized kernel (exl3_cpu_baseline,
    exl3_cpu_optimized);
  - the full stack: the real RamTier, RamThread, copy engine and CPU expert
    engine above the kernel, driven by a simulated device (exl3_full_stack_prod,
    exl3_full_stack_instr). See "Full-stack bench" below.

For persistent on-demand CPU isolation with sudo-free service start/stop after
one administrator installation, see service/README.txt. That service reserves
the cores only while the benchmark runs and leaves both memory nodes available.

Bare-forward benchmark
======================

Google Benchmark drives two separate C++ executables:
  exl3_cpu_baseline:  the existing vendored residual/block-128 kernel, 12 workers;
  exl3_cpu_optimized: the production kernel, 16 workers.
Override --workers=N to compare both with the same count. The baseline uses its
original native pool; the optimized kernel uses OpenMP. Separate processes keep
one backend's idle workers from contaminating the other's measurements. The
baseline source is the repo's csrc/exl3/moe_mul1.cpp, not a frozen snapshot.

There is no Python at configure, build or runtime. The optimized kernel uses no torch; the fixture reader and the
vendored baseline (built against exl3/baseline_moe_mul1.h) link ATen/c10 from LibTorch or the server's installed
torch directory, not Python bindings or libtorch_python. Google Benchmark v1.9.4 is pinned and fetched at configure if
not installed; set EXL3_FETCH_BENCHMARK=OFF for offline dependency management.
The kernel compiles with the validated GCC15 -Ofast -march=native options; the
harness compiles without fast-math so its finite-output checks work.

Build on the reference machine
------------------------------
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

Use the same torch C++ ABI as the server; the reference machine's installation
uses ABI=1. CUDA_HOME is unnecessary for this CPU-only build. Re-running cmake
--build rebuilds changed source and header files. The local CPU's ISA is baked
into -march=native: rebuild on the machine where the benchmark will run.

Inputs, correctness and placement
---------------------------------
Defaults use the original fixture and the frozen regular-forward references:
  --fixture=/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin
  --reference-dir=/data/models/exl3_exp/threading
References are reference-e1.bin, reference-e3.bin and reference-e5.bin (FP32,
eight layers in order). Fixture SHA256:
  55de978d5eb3c9d1bc3ac38b5a7937ea724a47fea2c0cf87c04bd1616e188a8c
The fixture uses real weights and synthetic FP16 activations, H5120/I2304, eight
layers and five experts per layer. Swizzled weights, other shapes, truncated
files and trailing bytes are rejected. The reference checks use SiLU limit=10
and the routing coefficients the references were computed with. The benchmark
generates no fixture or reference: supply the paths explicitly on other hosts.

CPU placement:
  Default --cpus=28-43 and --numa-node=1 (divix01's 2x Xeon 8280: node 1 is
  28-55,84-111; cpu N's SMT sibling is N+56). Worker zero (the caller) stays on
  CPU 28; baseline helpers use 29..39, optimized helpers use 29..43. --workers
  selects the first N CPUs in the list. The runner verifies that the requested
  CPUs belong to the node, are allowed by its affinity/cgroup, and that the real
  workers are individually pinned. Do not taskset the entire process to CPU 28:
  the helpers need access to all their CPUs. Leave SMT siblings 84..99 idle too
  for isolated timings. 52-55,108-111 are kept free for NVMe interrupts.
  --smt (bare and full-stack benches) appends each selected worker CPU's SMT
  sibling after them, so --workers=10 --smt runs 20 workers on 10 physical
  cores: the bench form of SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_SMT.

Memory placement:
  No process-wide NUMA memory policy is imposed. The process inherits its launch
  policy, and CPU affinity is independent of it, so weights can sit on node 0
  while computation runs on node 1. To test that remote-memory case, prefix each
  executable with numactl --membind=0 (scratch follows that policy too). Without
  a memory policy, fixture loading happens on the pinned caller and normally
  first-touches node 1. The fixture is ordinary CPU RAM, not CUDA-pinned server
  slabs; the benchmark excludes PCIe/NVMe transfers and the service job-ring
  handoff.

Checks and timing protocol:
  Before timing: verify all 24 outputs bit-exactly and verify helper affinity.
  Before and after each measured run: repeat all eight outputs for that expert
  count. Each run warms up with 128 forwards (--warmup-forwards overrides this).
  Timed calls are not individually compared with reference output. Setup, fixture
  I/O, registration, warmup and comparisons are excluded from reported latency.
  --gap-us=N optionally sleeps before each timed call to test wakeup/idle
  behavior; the sleep is excluded from the measured interval. The default is
  uninterrupted back-to-back forwards.

Run
---
Inside the existing isolated shell (this script does not alter cgroups):

  bash "$bench/run.sh" "$build" \
    "/data/models/exl3_exp/google_benchmark/run-$(date +%Y%m%d-%H%M%S)"

Default: eight alternating baseline/optimized process rounds, 512 full forwards
per expert count per process, expert counts 1/3/5, rotating layers. The baseline
is always included. EXL3_BENCH_ROUNDS changes the process-round count. Extra
flags after the output directory are forwarded to both executables; for an
equal-team comparison append --workers=16. The existing isolation must include
all active CPUs. The script writes binary hashes, affinity/cgroup context, logs
and Google JSON results to a new directory and refuses to reuse an existing one.

ISA tier: the optimized and full-stack binaries run the tier of their build's
-march (cmake -DEXL3_MARCH=..., default native: vnni on divix01; skylake-avx512
builds bw, x86-64-v3 avx2, x86-64 scalar, though these benches need AVX-512BW). The
vendored baseline detects the host's tier at load; run.sh's EXL3_BENCH_MAX_ISA
(scalar|avx2|bw|vnni|vbmi) caps it as its own EXL3_MOE_CPU_MAX_ISA. Every process
prints "isa tier <tier>" at startup and puts isa_tier in its Google JSON context;
each results directory's isa.txt gathers the lines. Read the tier from there.

Single-process examples (the tier is the build's -march):
  export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
  export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE
  "$build/exl3_cpu_optimized" --validate-only
  "$build/exl3_cpu_optimized" --benchmark_filter='experts:1/' \
    --benchmark_min_time=512x --benchmark_repetitions=8 \
    --benchmark_out=optimized.json --benchmark_out_format=json
  "$build/exl3_cpu_baseline" --benchmark_min_time=512x

Routed workloads and A/B
------------------------
The optimized build also runs M-row calls, as the DSpark draft's CPU thread does:
  --routed=M:k:pattern[,...]  pattern shared<G> (tokens in groups of G share all k slots; shared1 shares none) or
                              random (each token k distinct slots, seeded)
  --routed-slots=48           slab slots per routed layer; slot s holds fixture expert s % 5
  --routed-layers=2           fixture layers the routed workloads rotate over (shared by all of them)
Token t's input is the layer's fixture input rotated by 641 * t elements. There is no frozen reference: validation
checks the outputs are finite and repeat bit for bit. Every optimized benchmark reports plan, the id its
layers were registered on (kernel.h's SglangExl3CpuPlan: 0 generic, 1 DSV4.1), and the context records the build's chunk_m (exl3_cpu_chunk_m).

ab.sh BASE_BUILD HEAD_BUILD NEW_RESULTS_DIR [flags] runs two optimized builds as alternating processes for
EXL3_BENCH_ROUNDS rounds (default 8) and writes summary.tsv (median p50 per build, head/base, the plan each ran).

Timing interpretation
---------------------
One iteration is one full forward through the kernel interface: routing-weight
conversion, kernel preparation, gate/up, activation/down preparation, down and
output reduction, including OpenMP scheduling and barriers. No Python, GPU work
or driver-process work is inside it. Manual timing brackets only this C++ call
with steady_clock; sample recording and layer rotation happen outside it.

Counters and columns:
  Time      Google's mean measured full-forward wall latency, in us.
  p50_us, p95_us, p99_us   per-call wall-latency quantiles, computed outside timing.
  CPU       the caller's CPU consumption only, not the helpers': not the total
            team CPU time.
  Google's repetition median rows are medians of run means, distinct from the
  per-call p50_us counters.

A failed output, team or affinity check returns nonzero, and timings from a
failed run are invalid. For process-level comparisons use the median of p50_us
across process rounds.

Google Benchmark reference:
https://github.com/google/benchmark/blob/v1.9.4/docs/user_guide.md

Full-stack bench
================

exl3_full_stack_prod and exl3_full_stack_instr time the CPU-expert path above
the kernel. A writer thread (the GPU's stand-in, DeviceSim) posts CPU-hit
requests into the lease lanes; the real RamTier, RamThread (busy-polling), copy
engine (host backend) and CPU expert engine serve them with the optimized
kernel; the writer spins on CopyDone. The prod build is the headline; the instr
build compiles in the stage trace, which costs time but breaks the latency down.

Build
-----
Needs the tvm_ffi package's headers and liburing; the targets exist only with
EXL3_TVM_FFI_ROOT:

  cmake -S "$bench" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
    -DEXL3_TORCH_ROOT=/data/models/slang/.venv-upstream/lib/python3.13/site-packages/torch \
    -DEXL3_TVM_FFI_ROOT=/data/models/slang/.venv-upstream/lib/python3.13/site-packages/tvm_ffi
  cmake --build "$build" -j16 --target exl3_full_stack_prod exl3_full_stack_instr

Placement
---------
Defaults (all options):
  --writer-cpu=22    the writer, node 0 (a core of production's helper block
                     22-25); the request page and lease block are first-touched
                     there
  --service-cpu=26   RamThread, busy-polling, with its physical core to itself
                     (production's RAM thread core; sibling 82 stays idle)
  --copy-cpu=27      the copy thread and the watchdog (production's copy core)
  --cpus=28-43       CPU expert worker 0 (the CPU expert thread) and the kernel's
                     15 helpers, node 1 (production's CPU experts take 28-37;
                     the bench takes 16 cores for the same team as the bare
                     bench); slabs, x and outputs are first-touched there
  --host-node=0 --worker-node=1
                     the nodes the runner requires; anything else is refused
                     before setup

Run inside exl3bench.service (partition 22-43,78-99, service/README.txt; the
installed unit keeps the old partition until reinstalled). Off the partition,
use another placement, for example
--writer-cpu=0 --service-cpu=1 --copy-cpu=2 --cpus=3-15 --host-node=0
--worker-node=0 under taskset -c 0-15: it is correct, but not a measurement.
Bare phase: each group's bare forwards run on one persistent Team (the kernel's
own, not a throwaway one per call), so BM_bare no longer times thread creation,
and the bare thread census counts that Team's live helpers.

Two or more NUMA groups
-----------------------
-DEXPERT_STREAM_NODES=N (default 1, 1..8) builds the bench for N groups, as the
wire's node axis: expert e is homed on group e % N, each group has its own
service thread, CPU expert thread, kernel engine and slot range of every row
(StackFixture::kGroupSlots = 8 slots each: 5 experts and 3 staging slots), and
one copy thread completes every group's lanes. Per-group flags take one entry
per group:
  --service-cpu=26,51 --cpus=6-15/28-43 --worker-node=0,1
(',' between groups' service CPUs and nodes, '/' between groups' CPU lists).
Above one group a group's service and workers must sit on its own node; the
writer and the copy thread stay on --host-node. A measured run needs all three
flags; --self-test defaults to --writer-cpu=0 --copy-cpu=1 --service-cpu=2,3
--cpus=4/5, and also runs test_two_groups (a fake forward per group, each seeing
its own engine and only its own slots).

Each group's slots of every slab are bound to that group's node (mbind) before
the row images are read into them, and slot_of(e) names where the tier maps
expert e. Group g writes its parts at floats [2 g hidden, 2 g hidden + 2 hidden)
of the row's output. There is no BM_bare: every group's bare forward (its own
experts, its own engine, called from its worker 0) is recorded before the stack
exists, and each BM_stack and --validate-only call checks that every group's part
equals its bare forward bit for bit (validate_groups) and that the parts' sum
matches reference-e{k}.bin to within 1e-5 (relative above 1.0), since splitting
the experts across engines reassociates the fp32 sum.

Row images
----------
--image-dir (default /data/models/exl3_exp/google_benchmark/full-stack-images)
must accept O_DIRECT. The first run writes eight layer files (~533 MB) and a
.stamp each; later runs reuse files whose stamp names the same fixture (path,
size, mtime) and image size.

Phases
------
Every bare forward runs before the stack exists. The CPU expert thread polls
worker 0's core almost continuously, the kernel runs every caller on that core,
and the kernel's OpenMP team is per calling thread. So the bare phase fills the
slots from the row images (expert e in slot e) and then frees its team; the stack
is built after it and loads every expert through the tier's reader, refusing
unless the tier picks the same slots. BM_bare is registered before BM_stack;
under --benchmark_enable_random_interleaving, a BM_bare run once the stack
exists fails.

Checks
------
--validate-only compares all 24 layer outputs bare, then all 24 through the
stack, bit-exactly with reference-e{1,3,5}.bin (above one group: see "Two or
more NUMA groups"), and requires every thread created
during setup to be pinned to its CPU.

In a timed run, each benchmark compares its 8 outputs before and after it is
timed; BM_stack also requires one CPU job of k lanes per call, no row read and
no overrun. Each phase's thread check runs before that phase is timed (and once
more at the end). Once any benchmark fails, the later ones are skipped, so a
failed process reports no further timings.

--self-test needs no fixture. Run it under taskset -c 0-15 (defaults:
--writer-cpu=0 --service-cpu=1 --copy-cpu=2 --cpus=3). It checks the writer's
records and lane typing against ram_slot_map.type_lanes and drives the real stack
with a fake forward.

Benchmarks and counters
-----------------------
Per k in 1, 3, 5 (experts 0..k-1, cpu_forward.cpp's weights, 8 layers rotated):
  BM_bare/experts:k   the kernel's forward, called from worker 0's CPU, on
                      the stack's slabs (its own layers), slots, x and output
  BM_stack/experts:k  x store, record, gate close, spin until CopyDone == G;
                      t0 before the x store, t1 at CopyDone

Counters: p50_us, p95_us, p99_us per call. BM_stack adds
overhead_p50_us = its p50 - BM_bare's p50 (same process, same k).

The instr build adds, as p50/p95:
  pickup_us    observed - t0           (the record reaching the service)
  service_us   done - observed         (the service handling the record and
                                        submitting the CPU job)
  forward_us   the CPU expert thread's forward time for the call
  handoff_us   (t1 - done) - forward   (CPU job queue, done word, copy thread,
                                        CopyDone, the writer's poll)
and, at p50:
  overhead_vs_forward_p50_us = BM_stack p50 - forward p50
                               (the overhead against the in-stack kernel)
  forward_vs_bare_p50_us     = forward p50 - BM_bare p50

BM_bare runs before the stack exists, with no service, writer or copy thread
spinning. A forward_vs_bare far from 0 means overhead_p50_us carries a
kernel-speed difference.

Fidelity: the writer's stores reach the service by coherence between two node-0
cores, not by PCIe/DDIO, so pickup is a lower bound on the GPU path's. The prod
build's numbers are the headline; the instr build's carry the trace's cost.

Job-stream bench
----------------
BM_jobstream/in_flight:R/tokens:T runs the trace's record pattern through a CPU expert engine of its own (the real
BasicCpuExpertEngine over the fixture's layers, the same 16 pinned workers on 28-43, the driver on the writer CPU), with
no tier, reader or device in between. The driver is the tier owner: it claims a record's sequences, submits the jobs in
sequence order at their landing times, opens staged gates by a timer, and watches done(seq). It uses only claim / submit /
open_stage_two / done, so one source builds against the engine before and after the job window (job_stream.h).

A record (a pure function of its index, rows rotating over the 8 layers):
  hit job     part 0, k = 3, 4, 5 lanes (cycling), submitted at 0
  miss jobs   part 1, 6 - k of them (3, 2, 1), one lane each, the i-th submitted at (i + 1) * 200 us; the first
              overwrites the part, the rest accumulate into it; every second one is staged and its gate opens 100 us
              after the job goes in
  in flight   the next record's hit job goes in as soon as the previous record's last job has, while fewer than R
              records are incomplete; R = 1 waits for the previous record to finish
Cases: R = 1, 2, 4 at 1 token a row, and R = 1, 2 at 32 tokens a row (a full verify: the token table routes each lane to
6 consecutive tokens, --js-per-lane). A stream is 32 records (8 at 32 tokens, --js-records); one iteration is a stream.
Counters: p50_us, p95_us, p99_us of a record's completion time (its first submit to done() passing its last job, over
every record of every iteration) and records_per_s (records over the summed stream wall time).

Why these values (the production trace, 2026-10-18, in the job-window design): a miss job's wait behind its record's hit
job is what 98% of the trace's busy-engine submits are, with a median of 1.4-1.5 ms, so the misses must land while the
hit job (3-5 lanes, about 1 ms on 16 workers) still runs: 200 us apart puts all of a record's misses inside it. The
landing interval and the 100 us gate delay are assumptions of that order, not measured row-read times; --js-land-us and
--js-gate-us change them. The fixture holds 5 experts (slots 0-4), so a record's six lanes reuse them.

Every benchmark first checks every record of a 12-record stream (rows reused, so all three record shapes) against the
same jobs run one at a time through the same engine, bit for bit, and again after the timed iterations; --validate-only
does it for all cases before the stack is built. A failure skips the benchmark.

  bash "$bench/ab_job_stream.sh" BASE_BUILD HEAD_BUILD RESULTS_DIR

alternates the two builds' exl3_full_stack_prod per round (EXL3_BENCH_ROUNDS, default 8), holding cc-gpu.lock, and writes
summary.tsv: per case, the median over rounds of p50, p95 and records/s for each build and head / base. Against the base
(61c5e9399e), check out four files from the head commit in a private worktree of the base and build there:
  git worktree add --detach WT 61c5e9399e
  git -C WT checkout HEAD_COMMIT -- $bench/src/full_stack.cpp $bench/src/job_stream.h \
    $bench/src/cpu_forward.cpp $bench/src/stack_fixture.cpp
The last two are 55f1e94736's fix (the base's own do not build: they pass the removed row_weighted field). The base's
stack.h and self_test.cpp stay: the head's use the window's interface. Never commit that checkout.

Running
-------
From a login shell on the reference machine (no production server running;
service installed):
  bash "$bench/run_full_stack.sh" --launch [BUILD_DIR] [benchmark options...]

It configures BUILD_DIR once (default
/data/models/exl3_exp/google_benchmark/full-stack-build) and builds both
binaries, writes service-command.txt naming this script as the service's job, and
starts exl3bench.service (through sg exl3bench when the shell predates the
install's group membership); follow it with
  journalctl -u exl3bench.service -f
It refuses while the service is active or a production server runs. Remove
service-command.txt to return the service to run.sh. By hand, the job file is:
  printf '%s\n' /bin/bash "$bench/run_full_stack.sh" \
    /data/models/exl3_exp/google_benchmark/full-stack-build \
    > /data/models/exl3_exp/google_benchmark/service-command.txt

run_full_stack.sh BUILD_DIR [NEW_RESULTS_DIR] [options...] alternates prod/instr
processes over EXL3_BENCH_ROUNDS rounds (default 8), 512 calls per benchmark,
writes environment.txt, logs, Google JSON and status.txt, and exits 1 when any
process failed. Without NEW_RESULTS_DIR it writes to EXL3BENCH_RESULTS (the
service's directory). Do not run it while a production server uses CPUs 22-43.

Captured routing shapes (bare optimized forward)
------------------------------------------------
Every forward goes through the kernel's job interface: a one-slot JobWindow over the bench's team (JobRunner,
host/cpu_experts/single_job.hpp), one job at a time.

counts2-1-1-1-1-1-1-1-1-1 reconstructs the captured six-row, ten-chunk,
eleven-route multiplicities. Counts are assigned cyclically to token rows;
unused k lanes are -1. This preserves chunk work sizes, not the original token
activations or expert identities. Controls include uniform and random routes.
It measures isolated native forwards without optional trace counters, not live
decode throughput, GPU transfer overlap, or simultaneous use of both groups.
