# Storage, CPU, and GPU delivery pipeline implementation plan — revised

> For agentic workers: use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to execute bounded tasks with independent review. This document is a plan, not authorization to launch production workloads or a claim that its proposed interfaces already exist.

**Goal:** Improve EXL3 mirrored storage delivery without changing model arithmetic, first overlapping I/O and CPU packing, then allowing ready rows to reach the GPU while later reads remain outstanding.

**Architecture:** Establish native mirror correctness and controlled measurements first. Separate storage/packing overlap from GPU delivery changes. Keep one graph producer and serialized graph execution; introduce explicit row leases and completion generations before allowing concurrent storage and GPU access. Retain one fused MoE invocation after all required weights are ready.

**Tech stack:** Python/PyTorch, C++17, liburing, CUDA; use the target environment's verified versions.

**Spec:** [EXL3_COPY_PIPELINE_HANDOFF.md](/home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41/EXL3_COPY_PIPELINE_HANDOFF.md), especially §§3–8.

**Relationship to prior plans:** This is an alternative revision of [the original storage plan](/home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41/docs/superpowers/plans/2026-09-20-storage-cpu-pipeline.md); leave that file unchanged. [Native mirror extents](/home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41/docs/superpowers/plans/2026-09-20-native-mirror-extents.md) is the prerequisite, subject to Task 0's corrections. Do not execute both storage plans independently.

**Planning snapshot:** `dsv41`, HEAD `4c9b767e8de604b97a0099912ea61fb0e7e319c9` on 2026-09-20. Python extent-table work has begun; native code still shows eight bounce rows and ring depth 16. Reconcile completed work before execution instead of rebuilding it from the earlier handoff snapshot.

**Reconciled snapshot (Task 0, 2026-09-20):** the branch has moved well past the planning snapshot. `4c9b767e8d` extent table, `18949eafad` per-extent native reader, `b4257dd412` stage timing, `a488c0b259` and `a5547e0ef4` EOF parity, `02d9494e0d` open-time size check and env wiring, `95782ea7aa` Task 4 arm, `f9f15a9f93` the measured result in `DSV41_REFERENCE.md` §19. Native mirroring is proven end to end on decode, not merely implemented: the RAM-miss service attributes 0.00% of the mirror arm's expert bytes to the source drive, with exact byte parity against the mirrors-off arm. Bounce rows are still eight, but ring depth is now `kQueueDepth * parts` and batch depth no longer depends on it.

## Global constraints

- Changes in this phase preserve the six pinned tensor slabs, inclusive RAM protection for GPU-hot experts, byte-exact weights, and existing model arithmetic. Raw-row layout is a separate gated experiment.
- Preserve release/acquire publication, timeout fail-stop, watchdog behavior, short-read handling, soft-error retries, and safe drain on failure.
- Keep the existing blocking reader contract ring-empty-on-return. A new asynchronous service interface may return with owned work outstanding; its explicit `drain`/close boundary must retire all work. Do not impose an empty ring between internal pipeline batches.
- No concurrent graph replays or concurrent fused compute in this phase. Existing shared plan, mailbox, scratch, and fused temporary buffers remain serialized.
- Preallocate descriptors, control records, bounce banks, and captured plan buffers. No new allocation or Python call in graph replay. One owning CPU thread drives each ring.
- On divix01, honor the existing reservation: CPU jobs on cores 0–63, OMP/MKL threads 16; cores 64–71 belong to production. Verify these reservations before running.
- GPU experiments use the established `$ANA/gpu-run.sh` lock. Do not start or stop production. Verify the wrapper and analysis directory before use.
- No uncoordinated competing drive workloads. Record `/proc/diskstats` before/during runs. Deliberate contention is a separate labeled experiment on reserved scratch storage, not an “idle” run.
- Record file age/state, temperature/warmup, filesystem alignment, versions, topology, and cache sizes. Match production checkpoint conditions; distinguish fresh-file tests from existing-file tests.
- Size GPU transfer tests beyond verified L2 capacity and use actual host-to-device traffic. Do not assume an old hardware capacity applies to the current target.
- Do not claim speedup from traced timings alone. Do not commit generated benchmark outputs or implementation automatically merely because this document exists; follow the execution session's authorization. Use truthful authorship, stage named files, and use a message file if committing.

## Coverage and explicit boundaries

| Handoff concern | Deliverable |
|---|---|
| Native mirror bypass, extent accounting, EOF | Task 0 prerequisite acceptance |
| Attribution and matched baselines | Task 1 |
| Extra eager bytes, scheduling, topology | Tasks 2–3 |
| Bounded I/O and CPU packing overlap | Task 4 |
| Incremental GPU delivery and slot leases | Tasks 5–6 |
| CPU scatter removal | Task 7, optional after measurement |
| Blocking promotions/service pauses | Task 8 separate follow-up design and acceptance contract |
| Demand priority and existing advisory behavior | Tasks 4–6; not deferred with predictor research |
| DMA selection, prediction, split compute, GDS | Named follow-ups in Task 9 |

Completing Tasks 0–7 does not establish a fully nonblocking inference system: synchronous promotions remain until Task 8 is implemented. No cold demand can consume weights before those weights arrive.

## Proposed interfaces and ownership

Names below are proposed contracts, not existing APIs. Implement them in a focused native header/source unit if needed rather than placing all state transitions into one large service method. Exact field alignment must be verified against the existing mapped-memory ABI before capture.

```cpp
struct RowKey { uint64_t request; uint32_t lane; uint64_t slot_generation; };
enum class RowState { Reserved, Reading, Packing, Ready, Failed, Retiring };
struct ExtentKey { RowKey row; uint32_t part; uint32_t attempt; };

// Conceptual service operations; handle owns descriptors and destination leases.
RequestHandle submit_rows(RequestDescriptor request);
void progress();                    // submit/reap, bounded packing, consume acknowledgements
void cancel_interest(RequestHandle request);  // does not free in-flight storage
void drain(RequestHandle request);  // no pending I/O, packing, or GPU readers on return
```

- `ExtentKey`/`user_data` resolves through an owned descriptor table; do not assume this structure fits directly in 64 bits. A descriptor index cannot be recycled until its old completion retires.
- A RAM slot has a generation plus I/O, packing, and GPU-reader references. Eviction requires zero references and no inclusive-hot protection. Publication does not release a reader reference.
- A GPU destination remains reserved until its last consumer completes. RAM-copy acknowledgement releases only the source lease, not GPU scratch ownership.
- Request completion and row completion are distinct. Use generation-qualified per-request records; never advance a global completed frontier past an unfinished request.
- Retire completed speculative rows individually. Demand for an in-flight speculative expert attaches to that read and promotes its priority instead of issuing duplicate I/O.

## Task 0: Accept the native mirror prerequisite

**Files:** Existing native-mirror plan; `python/sglang/srt/layers/moe/exl3_ram_miss.py`; `python/sglang/kernels/ops/moe/exl3_ram_miss.py`; `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`; native split/thread and format tests.

- [x] Reconcile the current branch and extent-table commit with the companion plan. Verify the Python-to-native argument bridge, table shapes, source sizes, and row-start metadata, not just the table builder.
- [x] Correct the companion's offset contract: extent offsets derive from `record.aligned_read(PAGE_BYTES)[0] + part_start`, not an assumed `record.offset`. — corrected in `f7250e768b`; the implementation was already right, only the companion's prose was wrong.
- [x] Add bounded SQ preparation and reap/refill. Never dereference a null `io_uring_get_sqe`; count credits by nonempty extents, not rows. Test more extents than ring capacity, including at least three active roots and two-bank preparation. — extents now wait in a queue and are prepared only as credit allows; retries re-enter the same queue and take credit like any other read. **Deviation worth knowing:** with production constants a batch is at most `kBounceRows * parts` = 8×parts against a 16×parts ring, so credit never binds and "more extents than ring capacity" is unreachable as written. A test-only `max_outstanding` cap makes the refill path reachable without resizing the ring; the three-root case is covered by `test_a_full_batch_of_three_part_rows_fits_the_ring`.
- [x] Test no-mirror parity, zero parts, independent part failures/retries, reverse CQEs, source/mirror size mismatch, and byte parity through both demand and advisory service routes. — reverse CQEs are now a fault knob (`reverse_cqes`) rather than an assumption, including a short-read retry under reversal, since a retry re-entering the queue must not be misattributed to another extent.
- [x] Resolve EOF behavior from valid completed bytes, not by declaring either reader authoritative. Clamp synthetic general extents with `max(0, min(length, file_size - offset))`; validate that required expert bytes exist. Do not require undefined bytes past reported EOF to remain unchanged. An entirely-past-EOF synthetic extent is distinct from a valid expert's minimal aligned superset. — `a5547e0ef4`. The `max(0, ...)` was a live bug in the eager Python reader, found only because this item specified the clamp precisely.
- [x] Run the existing native split/thread suites and the matching Python mirror/row-reader suites using the configured project interpreter. Record pass/skip/failure counts; an unavailable O_DIRECT test is not proof of O_DIRECT correctness.
- [x] Confirm on the target that native demand and advisory expert traffic uses the configured mirrors. Attribute expert bytes separately from other process/device traffic; aggregate diskstats alone is not byte-content parity. — §19 follow-up: 13,589 advisory (touch) and 7,211 demand requests, nvme2 at 0.00% of service-attributed expert bytes, byte parity exact, diskstats agreeing only as a cross-check.

**Gate:** Both reader paths return identical required expert bytes, errors never publish partial rows, and native mirroring is proven. No fixed throughput prediction is an acceptance criterion.

## Task 1: Instrument causal stages and build matched baselines

> **Measured-path boundary.** The stage-trace instrumentation (trace schema 3)
> landed inside `5e92db22dc`, a commit whose message describes a documentation
> change. It went there because a concurrent `git commit` without a pathspec
> took another session's staged files out of the shared index; nothing was
> amended, and `0f6bb50ead` carries the message the change should have had.
> The consequence for anyone comparing arms: **`python/` changed at
> `5e92db22dc`**, so arms measured at `f6608901a3` are not directly comparable
> to later ones even with tracing off, because the record grew and the branch
> structure changed. A gap across that boundary is a code-generation change,
> not a regression. Arm results carry a GENERATION label derived from
> `git rev-parse <HEAD>:python` for this reason.


**Files:** Native host service; CUDA RAM-miss kernels and wrapper for optional device timing; `python/sglang/srt/layers/moe/exl3_stream_trace.py`; proposed `analysis/dsv41-drive/PIPELINE_BASELINE.md`; native timing tests.

**Interface:** Bounded trace records keyed by request, row/lane, extent, attempt, root, and generation. Host timestamps use one monotonic clock. GPU intervals use one device clock or profiler domain. A GPU post is not silently labeled with the later host-observation timestamp.

- [ ] Add host timestamps for observation, queue admission, slot reservation, submit, each extent CQE, each row's packing interval, RAM publication, and acknowledgement observation. Separate useful, submitted, completed, retried, and canceled bytes. — **mostly built already** in `b4257dd412`: `observed`, `reserved`, `submit`, `first_cqe`, `last_cqe`, per-row `pack_start`/`pack_end`, `mapped` (RAM publication), `done`, per-extent CQE stamps, and the useful/submitted/completed/retried/cancelled byte split with tests. Outstanding: per-row queue admission, per-extent submit and attempt counts. **Acknowledgement observation is deferred to Task 5** — no device-to-host acknowledgement exists yet, and stamping the host's own completion store as one is the mislabelling this task's interface section forbids.
- [ ] Add device intervals for request publication, row wait, transfer, and final consumer gate using capture-safe instrumentation or a documented profiler procedure. Without calibrated host/device correlation, report durations separately and correlate by IDs only. — **not started; needs the GPU.**
- [ ] Test causal inequalities per row, not a globally sorted stage list. Packing row A before the last CQE of row B is expected overlap. Error/zero-miss/canceled requests have explicit missing-stage markers and terminal status. — **partial.** The overlap case (a row packing while another row's extent is outstanding) and the terminal-status/`missing_stages` markers for no-read, touch, failed, invalid and cancelled all exist. Outstanding: the per-row chain asserted as a chain, under the CQE-reversal and packing-delay fault knobs.
- [x] Export through a bounded host snapshot queue drained outside the hot worker; record dropped trace records. Avoid Python/GIL calls or JSON/file writes from the I/O completion loop. Measure trace-on overhead; verify disabled tracing bypasses timing/record work. — **closed.** The preallocated SPSC ring, its dropped counter and the off-thread drain already existed; `5e92db22dc` added an in-band `dropped_before` so a gap is locatable rather than only counted, a `stamp()` funnel through which every trace clock read passes, a bypass test asserting zero clock reads with tracing off, and a source scan that fails on any `now_ns()` outside the funnel. **Overhead measured** (`6a2643c1b4`): host CPU cost of trace-on minus trace-off, paired and interleaved, 25 reps of 300 requests, four independent runs on four quiet cores. Median +0.5 µs at zero rows read rising to +1.5-2.6 µs at eight rows, i.e. **3.7-5.5% of host CPU time**, on cache-resident buffered tmpfs reads with **no O_DIRECT, no drive and no GPU**, on a **contended box** (load ~5.2). Two artefacts were found and fixed while measuring: a silent cap at 8 ids per request, and a ring-size effect where a 65536-slot ring inflated the 8-row delta several-fold because every push touched a fresh page. **Still unmeasured: trace-off versus UNinstrumented**, which needs an end-to-end arm and is the number that decides whether earlier baselines carry across the instrumentation change.
- [ ] Collect mirrors-off and native-mirrors-on arms with the original barriers. Include cold RAM misses, warm RAM/GPU misses, all-GPU hits, mixed rows, graph decode, and eager prefill. Record Option F and promotion settings. — **partial.** Graph decode, mirrors off and on, and the original barriers (arms at `099eadba33`) are covered; `analysis/dsv41-drive/task1-results/`. The other conditions (cold vs warm misses, all-GPU hits, mixed rows, eager prefill) are **not** separated out, so this stays open.
- [x] Preserve workload, seed, row/request sequence where replayed, RAM/VRAM capacity, policy, and warmup. Run repeated interleaved arms and save both traced attribution and untraced token/latency results. — series `task1`/`task1c`/`task1d`, interleaved, traced and untraced arms in each cell, n=3-5 per cell. `scripts/dsv41/provenance.py` records the resolved environment, the imported tree's HEAD and dirtiness, the reader mode in force and a drive-idle check; `task1_arm_verdict.py` refuses an arm that cannot be attributed. Results in DSV41_REFERENCE section 19.

**Gate:** A baseline distinguishes storage queueing, reader time, packing, GPU wait/transfer, compute, and promotion boundaries. Subsequent overlap changes compare against **native mirroring with old barriers**, not only mirrors-off.

## Task 2: Separate cache-traffic regression from drive scheduling

**Files:** `analysis/dsv41-drive/bench_mirror_rows.py`; its tests; existing end-to-end mirror harness and trace analysis; proposed `analysis/dsv41-drive/SCHEDULING.md`.

- [ ] Repeat eager baseline/mirror runs interleaved with controlled initial cache state. Record RAM/VRAM hits and misses, admissions, evictions, promotions, advisory bytes, and per-session occupancy. Investigate the recorded 1.54× byte increase independently of service latency. — **RETIRED AS NON-REPRODUCIBLE, not resolved.** The gate requires evidence explaining the additional reads and there is none, so this item stays open by its own rule. What the investigation established (`EAGER_ANOMALY.md`, `7f0fecb948`): the ratio is **not** extra reads in the mirror arm. Every session-level arm ever measured reads 380.7-380.9 GiB; the anomaly is one base arm reading roughly 147 GiB **less**. Session bytes agree to 0.1% across seven arms in both orderings at two commits, with identical greedy output sha1s, which argues strongly against machine state, since cache warmth changes timing and not which rows are read. Residual: an uncommitted tree, the environment, or unrecorded state. The `wt-dsv41` reflog shows HEAD unchanged across the whole arm window, so no checkout occurred, but a reflog cannot show uncommitted edits and `~/.bash_history` predates the run. **The durable deliverable is the diagnosis protocol**, not a cause: per session compare implied bytes, (RAM miss + bg rows) x 12.72 MiB, against actual diskstats bytes -- both fall means a different computation, only the actual falls means the reads never reached the device.
- [x] Build a fixed row-request replay with the same requested useful bytes for all storage arms. Implement whole-row assignment within a single concurrent batch, not a loop of serial single-drive reads. — `ff104147ca`: `analysis/dsv41-drive/bench_row_scheduling.py`, 22 tests. Each batch is ONE concurrent `UringFileReader.read` over all extents, so the serial-read trap is avoided, and the run aborts (exit 2) unless every arm requests identical rows, useful bytes and aligned bytes.
- [x] Sweep 1/2/4/6/8 and a larger eager batch with within-row 1:1, whole-row assignment, one-root, and measured weighted-split arms. Report application rows, extents, and per-drive outstanding bytes separately; one application row can issue two extents. — sizes 1/2/4/6/8/32 across all four arms, two runs of ~47.6 GiB, `row-scheduling-run1.json` / `run2.json`. **Gap being closed now:** per-drive BLOCK REQUEST counts are not reported, and the two mirrors differ in `max_sectors_kb` (512 on nvme0, 256 on nvme3 which backs /mnt/nvme4), so an even byte split is not an even request split.
- [ ] Add a controlled asymmetric-load experiment only on reserved scratch space. Keep it separate from quiet-drive results, and rotate/replicate arms under comparable contention. — **designed, not run** (`SCHEDULING.md`): reserved scratch file per mirror device, fixed-rate O_DIRECT interferer at 25%/50%, Latin-square rotation over arms x load placement, separate result file, decision rule stated. Not run because the decision it would inform is already settled by the request model below.
- [x] Report p50/p90/p95/p99, sample counts, per-drive bytes, queue depth, read and packing time. Do not claim scheduling explains excess bytes without a demonstrated mechanism and cache counters. — reported with `n` and a `tail_reliable` flag, cross-checked against whole-run `/proc/diskstats`. No claim is made about the byte anomaly; the harness cannot address it by construction, since every arm requests identical bytes.
- [x] If whole-row assignment wins, specify a batch scheduler interface using row identity and outstanding work. The existing `SplitPolicy.plan(length)` alone does not describe that policy; do not hide a stateful round-robin change in a cached length-only plan. — moot in the winning direction and specified anyway: `SCHEDULING.md` carries a `BatchScheduler.assign(rows, lengths, outstanding)` / `complete(rows)` interface. Whole-row LOST by 1.8x at one size and the measured weighted split was 1-2% slower than 1:1 at every size, so nothing stateful is being introduced into a length-only plan.

**Gate:** Retain static 1:1 unless matched data supports a different policy. Record the eager traffic anomaly as resolved only with evidence explaining the additional reads.

## Task 3: Topology and registered-resource decision

**Files:** Proposed `analysis/dsv41-drive/TOPOLOGY.md`; benchmark helpers only.

- [x] Record GPU/drive topology, negotiated links, NUMA allocation and first-touch, filesystem alignment, versions, pinned-memory budget, and registration limits. — `50cf23b619`, extended and re-verified against current code in `036900c5cd`: `analysis/dsv41-drive/TOPOLOGY.md`. **Two findings that matter more than the inventory:** the mirrors are not symmetric hardware (`max_sectors_kb` 512 on nvme0 versus 256 on nvme3, which backs /mnt/nvme4; 66 file extents versus 3,112 by `filefrag`), and NVMe completion interrupts for several submitter cores land on cores 64-71 including core 71, so `taskset -c 0-63` does not keep interrupts off production's cores.
- [ ] Measure storage alone, current SM transfer alone, and their simultaneous execution. If testing DMA, label it separately from the production SM path. — **designed, not run; deferred to Task 9.** `TOPOLOGY.md` sections 9.A-9.C specify them. Deferred because this task's gate is to choose placement and a budget for Task 4, and Task 4 is already implemented, so the measurement is retrospective. The design records that the existing transfer benchmark's defaults (~50 MB) sit inside L2 and would measure cache rather than HBM.
- [x] Audit existing READ_FIXED and SINGLE_ISSUER/DEFER_TASKRUN support in the general reader versus native service. Preserve creator-thread ownership and regular kernel entries for deferred task work. — `036900c5cd` section 8. The native service has neither flag and no READ_FIXED, and its ring is created on the Python thread but driven by the service thread, so creator-thread ownership is not a property it currently has. The general reader implements both, but the EXL3 path registers no buffers, so its reads are plain `READ`. **Measured negative:** a probe shows a `DEFER_TASKRUN` ring reveals a deferred completion only through `io_uring_get_events()`, `submit_and_wait(1)` or `TASKRUN_FLAG` — never through `submit(0)`, which is the path `reap()` takes whenever a packed row is ready, so credit would starve while rows pack. Recommendation: do not adopt either in the native service.
- [x] Decide whether to run a bounded registered-bounce experiment after Task 4. Measure initialization/registration cost, fallback rate, memory pressure, and end-to-end benefit; registration is not mandatory merely because it exists. — **decided: defer.** Registration is cheap to implement here, since the native bounce is a single `posix_memalign`, so one iovec covers every extent and the fallback rate is zero by construction. Measured registration cost is 59.9 ms with THP or 106.6 ms on 4 KiB pages for 256 MiB; the 213 MB bounce interpolates between the probe's rows. Deferred until a Task 4 timeline shows the owner thread saturated, per the gate's own wording that registration is not mandatory merely because it exists.

**Gate:** Choose placement and a resource budget for Task 4. Record negative results. SQPOLL/IOPOLL remain optional measured experiments, not assumed improvements.

## Task 4: Overlap storage and CPU packing only

**Files:** Native host service; proposed `python/sglang/kernels/jit/csrc/moe/exl3_row_pipeline.h`; wrapper test hooks; `test/registered/unit/kernels/test_exl3_ram_miss_split.py` and `test_exl3_ram_miss_thread.py`.

**Interface:** Implement the owned request/extent contracts above internally. Preserve the old whole-request GPU readiness contract for this milestone. Partial host row completion is not yet permission for an early GPU read.

- [x] Test two banks with reversed CQEs, delayed packing, and poisoned recycled descriptors. Verify bank reuse waits for all I/O and packing references to retire. — `test_a_bank_is_not_reused_until_every_row_in_it_has_packed`, `test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte`, `test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor` (parameterised over CQE reversal), `test_a_short_read_in_either_bank_resubmits_only_its_own_extent`.
- [ ] Allocate two bounded bounce banks and preallocated metadata. Keep SQ/CQ credit handling independent of bank count. Refill submitted I/O before performing bounded packing work; compare a packing worker against the bounded single-owner loop. — **partial.** Banks and preallocated metadata are in `ddcb0d55ff`; credit independence is pinned by `test_ring_credit_is_independent_of_the_banks`. The **packing-worker comparison was never run**, so this stays open.
- [x] Pack complete row A while row B is in flight. Mark each row internally ready, but publish the existing request completion only when all demand rows succeed. Never advance request completion out of order. — `ddcb0d55ff`. Publication is gated per row by `serve()`'s `packed` flags and by `pack_one`'s coverage check (`filled >= needed`), which fails the request rather than publishing bytes no drive delivered. Note the task **gate** ("timeline proves I/O and packing overlap") is a separate artefact and is **not** yet produced; it needs Task 1's stage timestamps.
- [x] Reserve demand capacity. Initial speculative limit: at most one advisory row in flight; stop submitting additional speculative work when demand is queued. Reap already submitted advisory I/O safely and retain independently completed useful rows under the existing eviction/protection rules. — `test_an_advisory_has_one_row_outstanding_and_a_demand_may_have_several`, `test_no_advisory_starts_while_paused`, `test_a_cancelled_advisory_keeps_the_rows_that_completed_and_releases_the_rest`, `test_rows_kept_from_a_cancelled_advisory_follow_the_usual_eviction_rules`.
- [x] Test partial submits, soft/hard errors, short reads, cancellation, pause, full cache, shutdown, and synthetic generation wrap. Cancellation removes interest; it never releases an outstanding kernel destination. — short reads, soft and hard faults, cancellation, pause, full cache, stop/shutdown and `test_the_generation_counter_wraps_and_the_reader_goes_on` are all covered in the split and thread suites. **Known gap:** a submit that consumes nothing while nothing is in flight is deliberately left to the service watchdog rather than guarded, with the signature and the guard recorded in a comment at the `submit_and_wait` call (`be76ba501f`).
- [ ] Run native unit/fault/thread tests and existing GPU graph parity tests. Compare against Task 1's matched native-mirror baseline. — **partial.** Native unit/fault/thread suites pass (281 passed, 0 failed; the commit subset re-verified separately at 230). The **GPU graph parity tests have not been run**, and the comparison against Task 1's baseline became possible only once the old-barrier arms existed; neither is done.

**Gate:** Timeline proves I/O and packing overlap; no changed weights, stale slots, unbounded memory, or statistically meaningful tail regression. This task makes **no H2D-overlap claim**.

> **The timeline half of this gate is MET** (`analysis/dsv41-drive/overlap_timeline.py`, run on the seven schema-2 traces, which agree within noise). Overlap is the norm, not the exception: **1888 of 1888 multi-row requests** have a row packed while another row's read was outstanding, with **zero per-row chain violations**. Packing hidden inside the read window is 51% at p50 for two-row requests and 67-68% for three or four, bounded above by (n-1)/n since the last row cannot pack before its own read ends. The exposed tail is exactly one row's packing, p50 **2.73-2.76 ms**.
>
> **Mirrors-on moves the bottleneck onto the packer**, which is the finding that matters for what comes next. The read window shrinks from 15.3 ms to 6.0 ms at p50 while packing stays at 5.6-5.7 ms per request, so packing covers 18% of the read window with mirrors off and **49-54% with mirrors on**; rows that waited on the packer go from **0 of 4332 to 104-112 of 4332**. Note that 74% of served requests are single-row and cannot overlap within a request at all, which is a property of the blocking-per-request design rather than a defect.
>
> **Two independent analyses converge on this number, in the mirrors-on regime only.** Task 6's precheck, working from the same corpus for a different purpose, found that miss-row completion spread is 2.77 ms with mirrors on -- the same quantity as the exposed tail here, and the basis for calling that spread serial packing. **With mirrors off the spread is 7.4 ms and drive-bound**, so the convergence is a property of the mirrors-on regime, not a general fact about where row spread comes from. Earlier text in this plan, and the commit message that introduced these figures, stated it without that qualifier; the commit cannot be amended, so this correction lives here and in `analysis/dsv41-drive/STORAGE_V2_REPORT.md`. So the packing tail IS the row spread that Task 6 was designed to exploit, and **parallelising packing would shrink Task 4's exposed tail and Task 6's remaining benefit at the same time**. That makes the unchecked packing-worker item below the highest-value remaining work in this task.

## Task 5: Implement and test source leases and device acknowledgements

**Files:** Native service/pipeline header; `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`; `python/sglang/kernels/ops/moe/exl3_ram_miss.py`; `python/sglang/srt/layers/moe/exl3_ram_miss.py`; thread and GPU graph tests. **Plus, found during design and missing from this list:** `python/sglang/srt/layers/moe/expert_stream.py` and `python/sglang/srt/layers/moe/expert_host_tier.py`. `ExpertPinnedHostCache.__init__` registers a `weakref.finalize(..., release_host_slabs)` with the default `atexit=True`, so the slabs are unregistered at every ordinary process exit with no device barrier. **A quarantine that leaves that finalizer attached is silently undone at exit**, so §14's shutdown ordering cannot be implemented without touching these two files.

**Proposed control contract:** For each fixed request lane, publish `{request_generation, slot_generation, host_slot, status}` into stable mapped control storage. Publish readiness last with the existing validated system-scope ordering. The service grants a GPU-reader lease before publication. A matching GPU acknowledgement after that lane's copy releases exactly one lease. Keep record reuse behind acknowledgement retirement.

Include expert identity in the immutable row result and validate it against the lane's request. GPU resolution reads this result, never a subsequently mutable global expert-to-slot map. Deduplicate expert requests before granting lanes, or count one source lease per consuming lane; one acknowledgement cannot release another lane's source.

- [ ] Document aligned fields, single-writer ownership, system release/acquire operations, wrap handling, and request-slot reuse before coding. Use a coherently validated protocol; ordinary Python stores are not its implementation.
- [ ] Initially exercise the acknowledgements after the existing whole-request copy, without changing its scheduling. Include RAM hits as well as newly read rows in source ownership.
- [ ] Inject delayed GPU consumption while forcing RAM admission pressure. A leased source must remain immutable even after its read has completed and while a newer request exists.
- [ ] Handle failure before readiness, timeout before copy, and failure after a subset copied. Suppress dependent compute on any fatal demand error. A skipped GPU copy must not accidentally emit a successful source-consumption acknowledgement.
- [ ] Retire unconsumed leases after a terminal cancellation handshake establishes that no GPU reader can start. Already copied lanes retire through their matching acknowledgements. The worker must keep submitting/reaping and processing cancellations while acknowledgements are outstanding; never synchronously wait for an acknowledgement inside its I/O progress loop.
- [ ] On shutdown, stop admission, drain storage/packing, then establish completion of all GPU readers before freeing their memory. If a CUDA error prevents establishing completion, retain/quarantine allocations until process teardown; never recycle uncertain storage.
- [ ] Preserve Option F's acknowledgement/eviction ordering. Removing its all-hit handshake is a separate optimization after equivalent protection is proven.

**Gate:** Lease-pressure and fault tests prove no reuse before consumption. Concurrent graph execution remains unsupported and guarded. Audit repeated mutable-slot reads through `ld.global.nc` on the deployed GPU; treat this as a visibility test, not a presumption of corruption.

> **The `ld.global.nc` audit is DONE** (`6629148b2d`, pre-registered at
> `afba4dbc2e` before the program was written). Verdict by the mechanical rule:
> **KEEP `nc`** -- "not observed in 1.31e10 words per cell across the eight nc
> cells". The wording is *not observed*, as pre-registered; it is **not** "safe"
> and **not** "coherent".
>
> The result is load-bearing only because the controls show the harness can see
> staleness: within a kernel, `nc` saw a host-written flag **0 of 100** times
> while `cv` saw it 100 of 100 (median 8.6 us); and a second `nc` read of a
> just-overwritten device word returned the stale value **100,000 of 100,000**
> times, where `cv` returned it 0. So `.nc` is cached per SM and the rig detects
> it -- it is across *kernels* that the cache was never seen to serve stale
> bytes. `cv` stays a compile-time switch at no measured bandwidth cost (12.34
> GB/s both ways, against 13.79 for `cudaMemcpyAsync`).
>
> **Constraint this places on Task 6:** any kernel that polls **host** memory
> must use `ld.acquire.sys`, never `.nc`. The existing wait kernel already does.
> **Task 6's per-lane readiness poll must too** -- that is a hard requirement on
> the mechanism, not a tuning choice.
>
> Conditions: PCIe **Gen3 x16** throughout (the link is Gen1 when idle), 1,403
> samples, load1 3.15 rising to 6.68 with substantial foreign load. The
> pre-registration said L2 was "about 128 MB"; it is **96 MiB**. That changes
> nothing here (the visibility region is 512 KiB and the bandwidth region 1 GiB,
> ~10.7x L2), and the pre-registered text was left as written with the
> correction in the result section. **Note the same "~128 MB" figure appears in
> the project's `CLAUDE.md` microbenchmark guidance.**

## Task 6: Start per-row GPU transfers before all reads finish

**Files:** Files from Task 5; `python/sglang/srt/layers/moe/expert_row_plan.py`; `python/sglang/kernels/ops/moe/expert_cache_transfer.py`; `python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh`; graph wrapper/backend tests; `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`.

> **SUMMARY OF THIS TASK'S STANDING** (the detail below is the audit trail; read
> this first):
> 1. The mechanism this plan originally specified -- fixed-order per-row -- is
>    **expected-REJECTED by arithmetic**, and is about 14 ms/step *worse* than
>    the two-phase alternative.
> 2. **Two-phase (hits, then the rest) is the chosen mechanism.** It is
>    justified; per-row is retained below only as the variant that must beat it.
> 3. **Neither mechanism's saving exists today.** Both need an early,
>    device-visible readiness signal that does not exist, confirmed at the
>    source. That signal is also a *safety* requirement, not only a performance
>    one.
> 4. **Task 6 cannot be accepted before Task 5's lease words land**, and it
>    invalidates the standing justification for a race that is benign today.
> 5. Every figure below is conditional on (3) and on assumption A1, which the
>    traces cannot check.
>
> ---
>
> **The mechanism this plan originally specified is not justified by the benefit
> it was designed to exploit.** A pre-registered analysis of the existing graph-decode traces
> (`analysis/dsv41-drive/PER_ROW_PRECHECK_PREREG.txt`, hashed before it ran)
> finds the modelled saving is real -- 19.2 ms/step at random lane order, about
> 7.1% after launch cost -- but that **87% of it is RAM-HIT lanes gathered while
> the NVMe read runs, and only 13% needs miss rows to finish at different
> times**. Per-row transfer's own contribution over a simple **hits-then-rest
> two-phase copy** is at most 5.06 ms/step, about **2.0% gross** -- which is
> *above* the ~1.5% this plan's arms resolve (`task1e-PREDICTIONS.txt` line 10).
> It falls to about 1% only **net of launch cost**, i.e. **at** the design's
> resolution rather than clearly below it. (An earlier revision of this
> blockquote cited 2.55 ms / 1.0% as the gross figure and called it below
> resolution; that was wrong on both counts. 2.55 ms is the registered
> `RANDOM-miss-only`, which is per-row against *batched*, not V2 minus V1.) The two-phase mechanism
> captures nearly the same benefit with no per-lane wait chain, no A4 ordering
> assumption, no partial terminal mask and no head-of-line problem. Prefer it
> unless a measurement shows otherwise.
>
> Two further corrections from the same analysis: **with mirrors on** the
> measured miss-row spread is **Task 4's serial packing** (2.77 ms, against ~2.7
> ms per row of packing), not drive completion order, so this task's premise was
> partly a misattribution of a Task 4 artefact -- and parallelising packing would
> shrink the 13% further, making per-row *less* attractive as Task 4 improves.
> **With mirrors off the spread is 7.4 ms and is drive-bound**, so the
> misattribution claim holds only in the regime this system actually runs in, and
> must not be restated as a general one. The precheck's modelled saving is the
> same either way. The caveat that carries the result is that hit
> lanes are assumed to spread evenly over layers; the traces do not record lanes
> per layer, so an instrumentation item is requested to record the planned lane
> count per request.
>
> **BLOCKING DEPENDENCY.** The 7.5-15% saving, and the 87% share attributed to
> hit lanes, both assume the hit-lane gather can start while the NVMe read is
> still outstanding. Publication today is **per-request**, and `demand_done` is
> stored *after* `read()` returns, so no such early signal exists. The Task 6
> design's own `[REQ 1]` -- asking `LEASE_PROTOCOL` to permit per-lane,
> time-staged publication -- **is** that signal, filed as a protocol request
> rather than as the prerequisite it is. Consequences:
>
> - Every figure in this blockquote is conditional on building `[REQ 1]`.
> - It binds **V1 two-phase exactly as hard as V2 per-row**. V1's advantage is
>   the 87% hit-lane share, which is unreachable without it. Preferring V1 does
>   not avoid this cost, it relocates it.
> - `[REQ 1]`'s cost is currently booked to Task 5 and appears in no Task 6
>   estimate. The cheapest form depends on whether hit lanes are already
>   resolved at plan time and merely kept host-side; that is under independent
>   check and is not yet established.
>
> **CONFIRMED against the source** (`53bc8c7229`). `kDemandDone` is stored in
> exactly one place, `RamTier::pump_demand`, *after* `handle_demand` returns --
> i.e. after `serve()` has finished `RowReader::read()` and published every row.
> `exl3_ram_miss_wait_kernel` polls only that word and has no per-lane or
> per-phase input. `slot_map` does not substitute: `publish_map` for new rows
> also runs after `read()`.
>
> **The early signal is a safety requirement, not only a performance one.** For
> a RAM hit the `slot_map` entry already exists, so it is tempting to gather
> hits from it during the read. That would be a correctness bug. The entry
> carries **no ownership**, and today's safety rests on temporal exclusion that
> early copying is precisely what removes -- an in-progress advisory on the same
> tier evicts unprotected rows. The same early publication that lets the GPU
> start is what grants the lease keeping the slot immutable. Any implementation
> that treats this as a performance-only change, and reads `slot_map` early
> without taking a lease, is reading memory that may be evicted under it.
>
> The change Task 5 must absorb is **when** the `RowResult` words are published
> (at reservation, for hit lanes), not which words exist.
>
> **Cost, split by owner** (`c5c572bd92`). The information the signal must carry
> already exists at the right moment and in the right thread: `RamTier::serve`
> resolves hit lanes authoritatively under `mutex_`, before any read is
> submitted (`tier.expert_slot[expert] >= 0`). It is host-side only, in
> C++-private `Tier` state. So:
> - **(a) small, Task 6's own:** in `serve()`, publish each hit lane's
>   `RowResult` inside the reservation critical section rather than after
>   `read()`. That is the whole of V1's protocol delta.
> - **(b) large, Task 5's:** the words themselves, lease counters and
>   retirement, the eviction predicate, and generations. **None of this
>   exists.**
>
> Two corrections from the first independent review (`PER_ROW_TRANSFER_REVIEW.md`):
> the hit lease must be taken **in the same `mutex_` section as the
> reservation**, not merely "in the reservation step" -- today's `serve()` drops
> the mutex between reservation and `read()`. And V1's own saving bound is
> **33.5 ms**, not the 38.6 ms quoted for early transfer generally, which is
> V2's figure.
>
> **Implementation landmine for V2 only.** Publishing miss rows early means a row
> can be `kReady` and leased when a *later* row of the same request fails.
> `serve()`'s `!ok && !cancelled` cleanup calls `release_locked` on every slot,
> and Task 5 makes releasing a leased slot throw -- so the cleanup must skip
> published lanes or the service thread throws on any mid-request failure. V1
> does not hit this, because its hit lanes are already `kReady`. That is a
> further argument for V1 first.
>
> **Sequencing consequence: Task 6 cannot be accepted before Task 5 (b) lands.**
> Not merely "should follow" -- (b) is what makes (a) safe, per the ownership
> argument above. This orders the remaining plan work regardless of which
> mechanism is chosen.
>
> **The post kernel's `slot_map` hint: benign today, and *why* is the point.**
> It classifies `need` from `slot_map` at post time with no ownership. Checked
> against the source, it cannot silently suppress a demand -- but the two paths
> are safe for different reasons, and only one of them survives this task:
> - **Armed record:** the host re-resolves. `serve()` builds `wanted = protect +
>   need`, and `protect` comes from the layer's routed experts (`topk_ids` via
>   `_apply_graph`), *not* from `need`. Any `wanted` expert with
>   `tier.expert_slot[expert] < 0` at serve time is re-read. This reason is
>   independent of Task 6 and survives it.
> - **Unarmed record (advise off):** the host does **not** re-derive. It is safe
>   only because with advise off no advisories exist, so nothing evicts between
>   post and wait -- **the temporal exclusion of `LEASE_PROTOCOL` §1.1 rule 5,
>   which is precisely the premise this task removes.** Were it to fail today the
>   failure is loud (`unserved_misses` -> `raise_fatal`, fail-stop), not wrong
>   bytes.
>
> So this task does not merely need a lease for the hit lanes it copies early;
> it **invalidates the standing justification for an existing benign race**.
> Any Task 6 work must re-establish safety for the unarmed path rather than
> inherit it. The same argument makes V1's order array `ord` benign -- a wrong
> hint only changes *when* a lane is copied, because correctness comes from the
> per-lane `RowResult`.
>
> Latent and not closed: a planned expert absent from `protect` (a
> `plan_candidates` producer) is suppressed by the hint and never re-derived,
> failing loudly. No production caller exists outside `expert_row_plan.py`
> (`LEASE_PROTOCOL` OPEN 12, seen from the hint side).

**Chosen first mechanism: two-phase (hit lanes, then the rest), with per-row as the variant that must beat it.** Per-row is recorded as *expected-REJECTED by arithmetic, not yet by measurement*: its own contribution over two-phase is at most 5.06 ms/step, **2.0% gross**, falling to about 1% net of launch cost -- against a gate that resolves about 1.5%, so it sits **at** the resolution limit, not below it -- while costing 160 extra stage triples per step across all 40 layers. The strongest argument for the redirect is a different one, currently unregistered: the plan's *original* fixed-order per-row is about **14 ms/step worse than V1** (38.6 - 5.06 - 19.2), because random lane order forfeits roughly half the ideal saving. `analysis/dsv41-drive/PER_ROW_TRANSFER.md` §6.2 lists what would overturn that. The per-row description below is retained as the specification of that variant.

**Baseline correction:** the fair comparison is Task 5 **lease-mode batched**, not today's unleased path. Lease mode arms every `count > 0` record, so all 40 layers pay a service round trip (OPEN 11, unmeasured). That cost must be netted out or any Task 6 arm flatters itself.

**Per-row variant (V2):** Retain the existing SM-driven host gather and one graph stream. Capture a fixed number of lane operations determined by plan capacity. Each active lane waits for its own generation-qualified readiness, copies only that row to its reserved GPU destination, then acknowledges source consumption. Inactive lanes are no-ops. All lane copies precede one fused MoE invocation.

```text
post_request(all demand lanes)
for lane in fixed_capture_capacity:          # unrolled at capture, not Python replay
    wait_row(request_generation, lane)      # fills slot/status, owns no allocator mutation
    copy_one_row_if_valid(lane)             # existing six segments, selected host slot
    acknowledge_consumed_if_copied(lane)    # system publication after transfer completion
validate_all_required_rows_or_fail_stop()
fused_moe()
```

This permits CPU I/O for later rows to overlap the earlier row's SM transfer. It does not promise completion-order consumption: fixed lane order can cause head-of-line waits. It may add kernel launches and must beat the existing batched gather to remain enabled.

- [ ] Add capture-stable lane views/counts, per-lane delivery status, and a row-copy entry point. The all-valid/all-hit path can retain the batched copy only if it obeys the same lease and acknowledgement contract.
- [ ] Ensure no CPU completion depends on work queued behind the GPU's own wait. Use generation-qualified mapped readiness for this mechanism; do not substitute an event that has not been recorded for that generation.
- [ ] Test early row readiness with a deliberately delayed later read. Verify row A's transfer ends before row B's I/O completes and fused compute starts only after all required transfers.
- [ ] Test inactive lanes, arbitrary completion order, RAM hits, repeated overwrite/replay, full cache pressure, fault after partial delivery, stop during SM transfer, and output parity. Maintain a single request timeout budget rather than multiplying the timeout by the lane count.
- [ ] Gate each copy on valid lane readiness and an acquired source lease. Failed, timed-out, inactive, or canceled lanes read no source bytes; setting the fused MoE's `keep` to zero alone does not protect an earlier gather. A final request-success check suppresses compute even when some earlier lanes already copied successfully.
- [ ] Compare batched SM transfer with per-row SM transfer at real miss counts; report kernel overhead, link throughput, SM contention, step latency, and token rate. Sweep launch geometry only as a separate measured variant.

**Gate:** Both demonstrated I/O/H2D overlap and untraced end-to-end benefit at unchanged cache capacity. If launch/head-of-line cost cancels the benefit, retain Task 4 and record Task 6 as rejected; evaluate a readiness-aware gather or native DMA in Task 9 instead of claiming success.

## Task 7: Optional raw-row pinned-cache experiment

**Files:** EXL3 row/layout/source and native table/service code; host-tier layout definitions; segment wrapper/kernel; byte-parity, EOF, graph and capacity tests. Produce `analysis/dsv41-drive/RAW_ROW_CACHE.md` before enabling the variant.

- [ ] Measure residual exposed packing after Tasks 4–6. Existing 1.584 ms packing is about 38% of the saved mirrored CPU-row mean; the 2.550 ms read stage is larger. Neither number is a new native/pinned measurement.
- [ ] Prototype direct reads into leased final raw-row RAM slots behind an opt-in experimental path. Define per-slot stride, row-start prefix, segment source offsets and destination offsets; keep the six GPU tensor outputs unchanged.
- [ ] Preserve page alignment, EOF byte validity, `mul1` omission, and every source-reader lifetime. Never issue arbitrary O_DIRECT segment reads into unaligned tensor destinations.
- [ ] Compare source alignment and GPU transfer efficiency. Account for padded slot capacity: report both equal-memory-budget end-to-end results and equal-row-count diagnostic results.
- [ ] Test byte/model parity and repeated reuse with the same leases. Retain only if net token latency improves without unacceptable tail or memory regression.

**Gate:** A negative result is a valid deliverable. Do not change the checkpoint format or switch production defaults as an incidental part of this experiment.

## Task 8: Required follow-up for blocking promotions

**Owner:** Expert host-tier/hot-cache integration. **Files:** `expert_hot_cache.py`, `expert_stream.py`, `exl3_ram_miss.py`, transfer executor, and promotion/lifetime tests under the existing MoE test directory.

This is explicitly separate from demand delivery, but must be implemented before claiming the original full nonblocking objective is complete. Deliver a follow-up design using Task 5's lease contract; it must specify:

- [ ] Preserve CPU expert IDs for admission instead of the avoidable CPU→GPU→CPU metadata round trip.
- [ ] Asynchronous admission tickets reserve RAM and replacement GPU slots. Old mappings stay readable until new payload copies complete and old consumers retire; account for extra capacity or defer admission when no safe replacement exists.
- [ ] Enqueue promotion transfers on an owned stream, query completed events outside the serving hot path, and publish mappings only at a safe graph boundary with CUDA-visible dependencies.
- [ ] Replace broad native-service pauses only after all consumers hold appropriate leases. Bound promotion in-flight bytes and service priority below demand.
- [ ] Prove no normal-path stream/device synchronization or service-wide pause in promotion submission. Test cancellation, copy failure, insufficient capacity, and old/new map consumers; measure boundary p95/p99 plus miss-rate changes.

**Gate for that follow-up:** Safe asynchronous publication and reduced boundary stalls at matched cache capacity. Existing generic async flags are not evidence of EXL3 support.

## Task 9: Track remaining experiments without implying completion

| Follow-up | Required design/measurement before implementation | Success evidence |
|---|---|---|
| Native DMA vs SM gather | Device-selected destinations, source leases, copy-completion generations, independent worker and CUDA-visible dependencies; no CUDA calls from host callbacks | Transfer/compute overlap and token-latency gain, not isolated bandwidth only |
| Better RAM/GPU lookahead | Real EXL3 pinned-slot resolution, dedicated GPU destinations, pointer-table coverage, exact demand fallback, bounded speculative bytes | Useful-on-time recall, retained reuse, wasted/canceled bytes, demand tails, net throughput |
| Hit/miss or multi-request compute overlap | Per-inflight fused temps/output/scratch and deterministic reduction; separate graph control state | Numerical parity plus gain after lost fusion and extra memory |
| Stream memory waits | Device/driver/graph support, scheduler-visible dependencies, visibility and timeout proof | Better measured cost than the current small wait kernel without deadlock |
| GDS | Proven native platform support, fallback detection, inclusive RAM-cache policy | End-to-end gain and acceptable cache behavior |
| Alternate-root failover/content refresh | Completion-safe retry buffers; immutable manifest/identity validation for refreshed mirrors | Fault recovery without mixed/stale checkpoint bytes |

| Task 3 placement measurements (storage alone, SM transfer alone, simultaneous) | Designed in `TOPOLOGY.md` 9.A-9.C. Needs the GPU lock **and** a quiet box; a contended box invalidates them as surely as a busy GPU | Deferred: this task's gate was to choose placement for Task 4, which is already implemented, so the measurement is retrospective |
| Registered bounce (`READ_FIXED`) in the native service | One iovec covers 100% of extents because the bounce is a single allocation, so the fallback rate is zero by construction; registration measured at 59.9 ms (THP) / 106.6 ms (4 KiB) for 256 MiB | Deferred until a Task 4 timeline shows the owner thread saturated. Registration is not mandatory merely because it exists |
| `SINGLE_ISSUER` / `DEFER_TASKRUN` in the native service | **Measured negative, already recorded.** A probe shows a `DEFER_TASKRUN` ring reveals a deferred completion only through `io_uring_get_events()`, `submit_and_wait(1)` or `TASKRUN_FLAG` — never `submit(0)`, which is the path `reap()` takes whenever a packed row is ready | Rejected for now: adopting it would starve storage credit while rows pack. `IOPOLL` is unexercisable (`poll_queues=0`); `SQPOLL` untried |

These experiments are not requirements for accepting Task 4. Their deferred status must remain visible in the final implementation report.

> **A claim in this plan that is not substantiated.** Task 5's gate says
> "concurrent graph execution remains unsupported and guarded". Reading the
> source found **no such guard** (`LEASE_PROTOCOL.md` OPEN 9), and the post
> kernel updates `state[kPosted]` with a non-atomic read-modify-write. Every
> design that rests on that guard existing — Tasks 5, 6 and 8 — must treat it as
> a requirement to build, not a property to rely on.

## Verification procedure and final acceptance

Use the configured project environment and run focused tests before hardware experiments. A starting native regression invocation is:

```bash
python -m pytest test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py -q
```

Add the existing advisory/service/mirror tests and the new task-specific tests to the invocation as those paths change. Read manual GPU test launch requirements and run them only under the target GPU lock. Do not report skipped hardware tests as passed evidence.

- [ ] Byte parity and model-output parity hold for eager and graph paths.
- [ ] Fault tests cover zero parts, EOF/truncation, retries, queue saturation, stale completion, generation reuse/wrap, partial delivery, shutdown, and full-cache pressure.
- [ ] Timeline proves each claimed overlap independently: I/O/packing for Task 4; I/O/GPU transfer for Task 6.
- [ ] Matched unprofiled runs report tokens/s, p50/p95/p99 step latency, sample counts/variation, traffic, cache misses, CPU cost, pinned/VRAM footprint, and promotion stalls.
- [ ] Original workload and numerical behavior remain unchanged; cache-capacity differences are disclosed.
- [ ] An independent reviewer checks ownership transitions and evidence after each structural milestone. No approval based solely on checklist completion.
- [ ] Final report names accepted, rejected, and deferred work. Storage overlap is not labeled a fully asynchronous inference pipeline while Task 8 remains blocking.
