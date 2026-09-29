# Configurable expert-stream io_uring experiments

Branch: `codex/sleep-free-lease-wait`. Local worktree:
`/home/dimitri/.codex/worktrees/sleep-free-lease-wait/sglang-nvfp4`.
Remote private worktree: `/data/models/slang/nvfp4-work/wt-sleep-free-lease-wait`.

## What changed

The expert-stream host reader now selects ring setup, submission/completion waiting,
registered files, and read opcode through environment variables. Defaults preserve
the previous configuration. All model files already share one reader/ring per host
tier; this change does not create one ring per file. Options apply to this expert
reader, not the unrelated generic `UringFileReader` or Engram I/O.

Resource registration happens once when a reader opens. Registered buffers and
files remain live until outstanding reads have retired. SQPOLL failure cleanup
submits and retires all prepared reads because its kernel thread can consume the
submission queue concurrently. No mode silently falls back when explicitly
requested capabilities or registration are unavailable.

The optional shared arena groups a layer's named host slabs into one allocation,
while preserving their tensor shapes, page-aligned starts, and NUMA row placement.
It retains the owner and registers/unregisters the CUDA span once. io_uring buffer
registration is a separate operation; CUDA pinning does not substitute for it.

## Options

Prefix every suffix below with `SGLANG_EXPERT_STREAM_URING_`.
Set options before constructing the cache/host reader; use a fresh process for
each benchmark arm. Changing options requires no source edit or JIT variant.

| Suffix | Default | Values / meaning |
|---|---|---|
| `MODE` | `default` | `default`, `iopoll`, `sqpoll`, `sqpoll_iopoll` |
| `QUEUE_DEPTH` | `0` | `0` retains `16 * parts`; otherwise absolute depth `1..32768` |
| `FIXED_FILES` | `0` | `1` registers all opened fds and submits their indices |
| `READ_MODE` | `normal` | `normal`, `fixed`, `readv_fixed` |
| `WAIT_MODE` | `block` | `block` or `spin`; spin spends a CPU polling completions |
| `SQ_THREAD_IDLE_MS` | `10000` | SQPOLL kernel thread's idle timeout in milliseconds (1000 before 2026-09-28; user decision) |
| `SQ_THREAD_CPU` | `-1` | `-1` unpinned; nonnegative CPU requires an SQPOLL mode |
| `DIAGNOSTICS` | `0` | `1` logs effective flags, depths, features and registrations |
| `READ_CUTS` | `auto` | `auto` (on with IOPOLL), `0`, `1`: cut reads into device-sized legs (plan 2026-09-28-iopoll-read-cuts). With cuts on, the default depth grows to 16 x parts x legs per read; an explicit `QUEUE_DEPTH` below that logs once at open (fewer reads in flight than uncut), and one below the widest read's legs is refused |
| `SLAB_ARENA` | `0` | `1` enables the shared per-layer host slab allocation |

Booleans accept only `0` or `1`. Invalid option values fail explicitly.
The logical queue depth is separate from the number of bounce banks/rows; larger
depths cannot create more application work. Kernel SQ/CQ sizes may be rounded.

`normal` uses the existing read/readv path. `fixed` uses `read_fixed` and accepts
scalar reads or a single iovec. It is directly useful for the shard/bounce path;
it refuses multi-iovec row-image reads. `readv_fixed` uses the newer vectored opcode,
including one-vector scalar reads. Every vector in one request must fit within
ONE registered buffer entry; independent slab allocations cannot be treated as
one address span. Row-image experiments therefore need `SLAB_ARENA=1`.

**2026-09-28 update** (plan `docs/superpowers/plans/2026-09-28-reader-crtp-uring-registration.md`, Task 7): the
paragraph above is superseded. `fixed` and `readv_fixed` now fan a read whose iovecs meet several registered buffers
out into one SQE per buffer, submitted together, so multi-iovec row-image reads work in both modes and row images no
longer need `SLAB_ARENA=1`. Separate slabs register without the quadratic pin accounting once each mapping's tail is
bound out to its 2 MiB end (final review Important 1, fixed on the same branch).

`readv_fixed` requires liburing 2.10+ headers and a kernel advertising
`IORING_OP_READV_FIXED`. The reader probes the running kernel. Enabling NVMe poll
queues does not add this opcode. Buffer registration can also fail because of
locked-memory limits or per-buffer size limits (documented kernels limit each
entry to 1 GiB). A full layer arena may exceed that limit: record this as an
unsupported arm rather than shrinking the tier and calling it the same workload.
This implementation does not partition large arenas into registration windows.

IOPOLL requires O_DIRECT plus filesystem/device support. SQPOLL changes submission;
IOPOLL changes device completion. SQPOLL can still need wakeup syscalls, and IOPOLL
without SQPOLL needs kernel entry to make progress. Neither option is a blanket
promise of zero syscalls. SINGLE_ISSUER and DEFER_TASKRUN are deliberately absent:
opening, service, and prefill currently use different submitting threads in turn.

API references: [fixed vectored reads](https://github.com/axboe/liburing/blob/master/man/io_uring_prep_readv_fixed.3),
[registered buffers](https://github.com/axboe/liburing/blob/master/man/io_uring_register_buffers.3),
[buffer limits](https://man7.org/linux/man-pages/man7/io_uring_registered_buffers.7.html).

## Preflight and correctness

On 2026-09-28, divix01 reported kernel `6.12.0-211.60.1.el10_2.x86_64`,
liburing `2.12`, `nvme.poll_queues=1`, and `queue/io_poll=1` on all four
enumerated NVMe namespaces. Recheck after reboot; probe capability instead of
inferring backported features from the kernel version.
The real-kernel matrix passed all 37 cases on this host, including IOPOLL and
READV_FIXED; its 6.12 kernel does advertise the newer opcode.

Commit and push locally, then fetch/fast-forward the private remote worktree.
Never copy an uncommitted tree or test from the production checkout. Follow
`.claude/rules/divix01-run-protocol.md` for interpreter, affinity and lock order.

```bash
cd /data/models/slang/nvfp4-work/wt-sleep-free-lease-wait
git fetch origin
git merge --ff-only origin/codex/sleep-free-lease-wait
git status --short --branch
uname -r
pkg-config --modversion liburing
cat /sys/module/nvme/parameters/poll_queues
cat /sys/block/nvme*n1/queue/io_poll
ulimit -l
PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c 'import sglang; print(sglang.__file__)'
```

Run the real-kernel matrix on an NVMe-backed filesystem, not tmpfs. Its optional
skips distinguish unsupported setup/opcode from a first-read filesystem rejection;
supported modes must pass byte equality, short reads, drain/reuse, concurrent
scalar descriptors, cross-thread ownership, and close/reopen checks.

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_expert_stream_uring_native.py \
  test/registered/unit/kernels/test_expert_stream_uring_integration.py \
  -q -rs -p no:randomly --basetemp=/mnt/nvme2/nvfp4-work/uring-config-native-tests

PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_expert_stream_uring_options.py \
  test/registered/unit/kernels/test_expert_stream_buffer_regions.py \
  test/registered/unit/layers/moe/test_expert_host_slab_arena.py \
  test/registered/unit/layers/moe/test_expert_host_tier.py \
  test/registered/unit/layers/moe/test_host_numa.py \
  -q -p no:randomly
```

Pytest clears its `--basetemp` directory; reserve the named directory exclusively
for this test. Preserve pytest's exit code if piping logs. Before timing, also run
the row-image, tier, and second-layout regression suites with default options.
Use the GPU lock for tests that access CUDA. The fake-liburing suite covers
failure paths and opcode encoding even where the real kernel lacks an opcode.

## Experiment order

Keep model, tier size, NUMA allocation, storage mirrors, piece streaming, prompt
corpus, GPU settings, and cache state identical within each comparison. Use at
least three alternating baseline/candidate pairs after warmup. Record startup
cost separately from steady-state service. Stop an arm on corruption, timeouts,
failed requests, resource-registration errors or unexpected fallback.

| Stage | Compare | Purpose |
|---|---|---|
| A | Explicit defaults; then depth `16`, `32`, `64`, `128` | Find a useful queue depth before adding CPU polling |
| B | Winning depth, fixed files `0` vs `1` | Isolate fd registration |
| C | Shard layout: normal vs fixed, same file setting | Isolate fixed bounce buffers |
| D | Row images: normal with arena `0` vs `1` | Measure allocation-layout effect alone |
| E | Row images with arena: normal vs readv_fixed | Only if opcode AND real-size registration preflight pass |
| F | Fixed workload/opcode/depth: default vs iopoll, block wait | Isolate NVMe polling |
| G | Same workload: default vs sqpoll; then block vs spin | Separate submission polling from userspace busy waiting |
| H | Best individual settings vs sqpoll_iopoll | Test interaction after individual effects are measured |

For SQPOLL, compare idle timeouts (for example `100`, `1000`) only after the
main comparison. Assign a permitted, measured CPU near the drive's NUMA node;
avoid reserved cores 64–71 and account for the SQ thread's CPU separately from
the service thread. Explicit `SQ_THREAD_CPU` is required for SQPOLL experiments
on divix01: unpinned kernel SQ threads can use online CPUs outside the process's
`taskset` mask. The native and integration tests pin their SQ threads to the
lowest CPU in the test process's allowed mask. Do not change CPU placement and polling mode in the same
first comparison. Registered resources can increase startup time and locked RAM.

Capture exact commit and `python/` tree SHA, requested variables, diagnostic
effective flags/SQ/CQ entries/features, actual file paths/layout, kernel/liburing,
poll-queue state, memory-lock limit, drive temperature, CPU affinity and NUMA
placement for each arm. Measure request/first-piece/full-row p50/p95/p99, useful
and submitted bytes, outstanding I/O, service/SQ-thread CPU, syscalls, GPU idle
time, TTFT and decode token latency. Include failures and total CPU consumption;
a lower I/O median alone is not a production win.

## Running the existing full-model campaign

`benchmarks/dsv41_baseline/run_arm.sh` is the existing end-to-end harness. It
accepts explicit `KEY=VALUE` overrides, records them and verifies the process
environment. It requires a production-down, free-GPU window: the harness refuses
to launch if production port 7867 is listening or another GPU compute process
exists. Use a free non-production port (never 7867). Register the current
`python/` tree in `benchmarks/dsv41_baseline/generations.json` before launch;
commit that provenance change locally, push and fast-forward remotely again.
The registry helper is `generations.register(tree_sha, label)`, where tree_sha
comes from `git rev-parse HEAD:python`. The harness requires a clean worktree.

Example baseline after that gate is satisfied:

```bash
OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/rowimg-disk.lock \
  taskset -c 0-63 bash benchmarks/dsv41_baseline/run_arm.sh uring-default 17867 \
  CUDA_HOME=/usr/local/cuda-13.4 \
  SGLANG_EXPERT_STREAM_URING_MODE=default \
  SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=0 \
  SGLANG_EXPERT_STREAM_URING_FIXED_FILES=0 \
  SGLANG_EXPERT_STREAM_URING_READ_MODE=normal \
  SGLANG_EXPERT_STREAM_URING_WAIT_MODE=block \
  SGLANG_EXPERT_STREAM_URING_SQ_THREAD_IDLE_MS=1000 \
  SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU=-1 \
  SGLANG_EXPERT_STREAM_URING_SLAB_ARENA=0 \
  SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS=1
```

The harness acquires the GPU lock itself; do not wrap it in another GPU flock.
Disk lock must come first. A held GPU lock causes the harness to refuse the arm;
retry later. Use distinct arm names and change only the intended variables.
The outer `OMP_NUM_THREADS=8` caps harness work. The server recipe explicitly
sets OMP/MKL threads to 16 and affinity to `0-7,16-17,36-53`; keep those effective
values identical across arms, or pass explicit overrides to change the experiment.
Check the harness's resolved recipe, including shard vs row-image layout, before
choosing stages C–E; these are separate within-layout comparisons.

Do not use `bench_row_scheduling.py` to evaluate these flags: it uses another
reader. `bench_pack_workers.py` exercises this reader but opens it per call, so
its timing includes registration/setup. `exl3_stage_trace_overhead.py` uses a
persistent host but buffered tmpfs fixtures; it is not an NVMe/IOPOLL benchmark.

No full-model performance result is claimed by this implementation. Promote a
configuration only after the same-workload end-to-end comparisons above.

## Verification record

The following checks ran in the private divix01 checkout with the documented
interpreter and `PYTHONPATH`. These are correctness results, not latency results.

- At `a5b4c87bb7`, the native matrix command above (native file alone) passed
  **37 tests** on `/mnt/nvme2`, with no capability skips.
- At the same commit, the five-file allocator/configuration command above passed
  **74 tests and 7 subtests**. The later explicit-CPU metadata regression passed
  **6 tests** locally; it includes empty and populated metadata under a non-CPU
  default device.
- Existing reader regressions: **850 passed, 1 failed** with the command below.
  The failure was `test_the_seqlock_reader_never_accepts_a_torn_record`: its
  one-second stress run accepted only 23 samples versus a requirement of >100;
  it observed **zero torn records**. A subsequent isolated run of the entire
  tier file passed **57 tests**. No seqlock implementation or threshold was changed.

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_exl3_ram_miss_tier.py \
  test/registered/unit/kernels/test_exl3_ram_miss_row_images.py \
  test/registered/unit/kernels/test_expert_stream_second_layout.py \
  test/registered/unit/kernels/test_exl3_ram_miss_pack_workers.py \
  -q -p no:randomly

# Isolated rerun; includes one CUDA-input refusal test, so takes the GPU lock.
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  flock -w 30 /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_exl3_ram_miss_tier.py -q -x -p no:randomly
```

For future runs, also take the GPU lock around the larger invocation if CUDA is
available, because the tier file includes that device-input refusal test.
