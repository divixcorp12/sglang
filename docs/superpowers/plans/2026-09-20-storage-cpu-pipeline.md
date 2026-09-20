# Storage and CPU pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Work through the storage and CPU priorities of
`EXL3_COPY_PIPELINE_HANDOFF.md` §3B-§3E, in an order where each step's value is
established by measurement before the next step's risk is taken.

**Architecture:** Instrument first, then measure scheduling, then restructure.
Stage timing comes first because §3B and §3C are both justified only by where
the time actually goes. The scheduling question (§3D) comes next because it is
pure measurement, needs no code change, and settles an open anomaly. The two
structural changes (§3B overlap, §3C raw-row cache) come last and in that order,
because overlap is reversible and the cache layout change is not.

**Tech Stack:** C++17, liburing, CUDA, PyTorch, Python 3.13, pytest, fio.

**Spec:** `EXL3_COPY_PIPELINE_HANDOFF.md` §3B, §3C, §3D, §3E, plus its §5
ownership contract and §7 measurement plan.

**Companion plan:** `2026-09-20-native-mirror-extents.md` is handoff §3A and
lands first. Task 4 of this plan (§3B) assumes the extent table exists.

## Global Constraints

- divix01 CPU jobs: `taskset -c 0-63`, OMP/MKL threads 16. Cores 64-71 are
  production's, including the doorbell spin core.
- GPU only under `$ANA/gpu-run.sh` (takes `cc-gpu.lock`). Never start or stop
  production.
- Never benchmark a drive that another job is using. Check
  `/proc/diskstats` before and during, and record the idle check in the result.
- Benchmark freshly written data when the workload is freshly written data. A
  stale-file measurement on nvme4 once produced a 3.4x deficit that vanished on
  a fresh file, and a conclusion had to be withdrawn.
- Size any GPU microbenchmark's working set past L2 (~128 MB) or it measures L2.
  Sanity-check implied bandwidth against the card's spec.
- Preserve on every change: release/acquire publication, slot-map visibility,
  timeout fail-stop, watchdog behaviour, short-read and soft-error handling,
  drain-on-error, and ring-empty-on-return.
- Commit with `git commit -F <file>`, never `-m`, no backticks or `$(...)` in
  the body. Stage by name. Trailers:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF
```

---

### Task 1: Stage timing (handoff §7 point 3; its implementation-order item 2)

Nothing below this line should be committed to without it. Today the only
attribution we have is §18.2's node-mode trace (190 ms NVMe / 128 ms gather /
16.8 ms compute) and a per-row CPU bench. Neither separates queueing from
service time, and neither covers the native path.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`
- Modify: `python/sglang/srt/layers/moe/exl3_stream_trace.py` (emit the stages)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_split.py`

**Interfaces:**
- Produces: per-request stage timestamps, monotonic, host clock only, written
  through the existing trace file. Fields: request posted, observed by the
  service, slots reserved, submit, first CQE, last CQE, pack start, pack end,
  publish. Per drive: bytes and extent count.

- [ ] **Step 1: Write the failing test.** Drive the reader through the existing
  fault harness and assert the stage record is emitted once per request, that
  its timestamps are non-decreasing, and that per-drive byte counts sum to the
  request's total bytes.
- [ ] **Step 2: Run, confirm failure. Step 3: Implement.** Use a monotonic host
  clock throughout. Do NOT subtract a GPU timestamp from a host one anywhere.
  Keep the record fixed-size and written on the existing trace path so it costs
  nothing when the trace is off.
- [ ] **Step 4: Tests pass. Step 5: Commit.**
- [ ] **Step 6: Collect a baseline** on divix01: one graph decode arm and one
  eager arm, mirrors off, trace on. Report the split of the RAM-miss wait into
  queueing, submit-to-first-CQE, first-to-last-CQE, packing, and publish.
  **This is the number §3B and §3C are argued from.** Record it in
  `DSV41_REFERENCE.md` and commit.

---

### Task 2: Drive scheduling — whole-row versus within-row (§3D)

Pure measurement, no production code change, and it settles the open anomaly in
§19 (eager + mirrors 25% slower than eager alone while reading 1.54x more
bytes). §3D predicts exactly that shape: splitting helps one row in isolation,
whole-row assignment avoids making every row wait on two drives.

**Files:**
- Modify: `analysis/dsv41-drive/bench_mirror_rows.py` (add the arms below)
- Create: `analysis/dsv41-drive/SCHEDULING.md`

- [ ] **Step 1: Add a whole-row assignment arm.** Rows are handed alternately to
  one root or the other, each read whole from a single drive, so a row waits on
  one drive. Contrast with the existing within-row split. Keep the split
  decision stable over a request and avoid tiny extents, per §3D.
- [ ] **Step 2: Add a batch-depth sweep.** The existing bench times ONE row per
  `read()` call, which is QD1 and is where splitting looks best. Add active
  counts 1, 2, 4, 6, 8 and a larger eager-shaped batch. This is the axis the
  per-row bench never covered and the most likely explanation of the anomaly.
- [ ] **Step 3: Add an asymmetric arm.** Repeat the sweep with a deliberate
  write burst on one root, and with a bandwidth-weighted split
  (`B1 = B*b1/(b1+b2)`, rounded to whole pages) alongside 1:1.
- [ ] **Step 4: Run** on idle drives, freshly written data, with the idle check
  recorded. Report p50/p90/p99 and per-drive bytes and queue depth for every
  arm.
- [ ] **Step 5: Rule.** State which policy wins at which depth, and whether it
  explains §19's anomaly. If whole-row wins at production depth, that is a
  `SplitPolicy` change and gets its own task; do not fold it in here.
- [ ] **Step 6: Commit** the bench, the results and the ruling, and update §19's
  anomaly entry with the outcome either way.

---

### Task 3: NUMA and registered resources audit (§3E)

Cheap, independent, and it can invalidate assumptions the later tasks rest on.

- [ ] **Step 1: Record the topology** on divix01: PCIe generation and negotiated
  width for the GPU and each drive, NUMA node of each drive's controller, of the
  GPU, and of the pinned slabs' first-touch; CUDA, PyTorch, liburing and kernel
  versions; filesystem direct-I/O alignment for each mount.
- [ ] **Step 2: Check whether disk and GPU bandwidth add independently.** Run a
  two-drive read at full rate with and without a concurrent H2D transfer, and
  report whether either degrades. §3E warns a shared root port or inter-socket
  path can prevent them from adding.
- [ ] **Step 3: Audit registration, and correct the record.** The general reader
  (`csrc/io/uring_file_reader.cpp`) already implements registered buffers and
  READ_FIXED and attempts SINGLE_ISSUER plus DEFER_TASKRUN; the native reader
  implements neither. Write down which of those the native reader could reuse
  and what each would cost, including that a DEFER_TASKRUN owner must enter the
  kernel regularly and that the general reader is bound to its creating thread.
  **No code change in this task** — this is the evidence for whether §3E is
  worth doing at all.
- [ ] **Step 4: Commit** as `analysis/dsv41-drive/TOPOLOGY.md`.

---

### Task 4: Bounded completion-driven pipeline (§3B)

Only after Tasks 1-3. This is the first structural change and it changes the
concurrency model, so it carries the §5 ownership contract with it.

**Preconditions, from the handoff and from the companion plan:**
- Completion generations must exist before concurrency increases (§1 conclusion
  6, §5). The companion plan deliberately deferred them to here.
- Per-row publication needs the device delivery protocol to change too: §3B
  notes that optimising CQE processing alone will not overlap H2D, and §5 that a
  whole-request completion word cannot advance past an unfinished earlier
  request. Use per-request completion or a contiguous completed frontier.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh` (the wait/publish side)

- [ ] **Step 1: Write the failing tests first**, covering §5's list: reverse CQE
  order; short positive reads; partial submits; soft and hard errors; a demand
  request arriving during an advisory read; shutdown with I/O in flight; a
  cancelled late completion that must not complete a recycled generation;
  generation wrap; a full RAM cache; repeated overwrite and graph replay.
- [ ] **Step 2: Implement the handoff's own first increment, in its order.**
  Retain the six pinned slabs and the segment layout. Use at least two bounce
  groups with explicit ownership. Pack completed row A while row B's I/O is in
  flight. Publish each fully packed row independently. Release a bounce row only
  after its CPU reader has finished. Let H2D begin on ready rows while later
  rows are still being read.
- [ ] **Step 3: Watch the single-owner hazard.** §3B: polling CQEs and then
  running a long memcpy on the sole I/O owner delays new submissions. Compare a
  packing worker against a bounded single-owner loop, and keep whichever the
  stage timing from Task 1 shows is better. More threads are justified only by
  measured stage overlap.
- [ ] **Step 4: Tests pass**, including every pre-existing fault and thread test.
- [ ] **Step 5: Measure** against Task 1's baseline. Gate: lower cold-miss
  latency with no stale slots and no tail regression.
- [ ] **Step 6: Commit.**

---

### Task 5: Evaluate the full-row CPU copy (§3C)

Last, and explicitly an evaluation before a commitment. This is the largest
single measured per-row cost (1.58 ms fixed, ~8.4 GB/s, 38% of mirrored per-row
time) and the most invasive change in the handoff.

- [ ] **Step 1: Baseline the cheap alternative first.** §3C's own fallback is to
  keep the slab layout and only overlap packing — which Task 4 already does.
  Measure what remains of the 1.58 ms after Task 4 before designing anything.
  If overlap has already hidden it, stop here and record that.
- [ ] **Step 2: Prototype the raw-row pinned cache** behind a flag: read each
  expert's aligned superset directly into its final RAM slot and have the GPU
  segment gather interpret row offsets. This removes the bounce-to-six-slab copy
  and accommodates split-drive reads into disjoint regions of one slot.
- [ ] **Step 3: Account for every hazard §3C names**, in the design document
  before the code: alignment prefix, padded row stride, EOF tail, dropped `mul1`
  scalars, pointer alignment. Packed EXL3 data is not uniformly 16-byte aligned,
  so measure GPU load alignment before and after — moving packing off the CPU
  can make the GPU side worse.
- [ ] **Step 4: Do NOT issue nine O_DIRECT reads into tensor segments.** §3C is
  explicit: their offsets, sizes and destination addresses need not meet direct
  I/O alignment. This is a named trap, not an option.
- [ ] **Step 5: Gate.** Keep it only if end-to-end token latency improves, not
  merely if CPU bytes fall. A GPU-side conversion that replaces a CPU-side one
  at equal cost is not a win.
- [ ] **Step 6: Record and commit** either the change or the negative result.

## Sequencing and why

1. **Task 1 (timing)** — everything else is argued from it.
2. **Task 2 (§3D scheduling)** — pure measurement, no code risk, settles the §19
   anomaly, and may change `SplitPolicy` before §3B builds on it.
3. **Task 3 (§3E audit)** — cheap, and can invalidate assumptions in 4 and 5.
4. **Task 4 (§3B overlap)** — first structural change, reversible.
5. **Task 5 (§3C raw-row)** — largest payoff, least reversible, and its cheap
   alternative is a by-product of Task 4.

This is the handoff's own implementation order (its §8), with its item 2 (stage
timing) pulled to the front where its own text says it belongs, and §3D promoted
above the structural work because of the §19 anomaly.

## What is deliberately still out

- Alternate-root retry on I/O error (§3A): needs completion-safe destination
  ownership; a mirror error still fails the batch.
- GPUDirect Storage (§7): the handoff calls it a later alternative, and it
  changes the inclusive RAM-cache behaviour. Prove native support and benefit
  first.
- Prediction and lookahead (§4B, §6) and split hit/miss compute (§4D): these are
  the answer to §1 conclusion 4, which no amount of storage work addresses, but
  they are a different program with their own measurement needs
  (useful-and-on-time recall, canceled bytes, evicted useful rows).
- Copy-engine DMA versus SM gather (§4A): belongs with Task 4's H2D work, but
  §4A warns device-selected source rows cannot simply be substituted into
  captured memcpy nodes. Needs its own design pass.
