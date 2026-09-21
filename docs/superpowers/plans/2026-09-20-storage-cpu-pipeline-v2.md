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
- [ ] Export through a bounded host snapshot queue drained outside the hot worker; record dropped trace records. Avoid Python/GIL calls or JSON/file writes from the I/O completion loop. Measure trace-on overhead; verify disabled tracing bypasses timing/record work. — **partial.** A preallocated SPSC ring with a dropped-record counter, drained from Python off the service thread, with no allocation, lock or Python call in the completion loop, already exists. Outstanding: making an overflow locatable in band rather than only as a total; a test that disabled tracing really bypasses the work rather than an argument that it does; and the overhead measurement, which is **blocked on a quiet machine** rather than on the GPU.
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

## Task 5: Implement and test source leases and device acknowledgements

**Files:** Native service/pipeline header; `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`; `python/sglang/kernels/ops/moe/exl3_ram_miss.py`; `python/sglang/srt/layers/moe/exl3_ram_miss.py`; thread and GPU graph tests.

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

## Task 6: Start per-row GPU transfers before all reads finish

**Files:** Files from Task 5; `python/sglang/srt/layers/moe/expert_row_plan.py`; `python/sglang/kernels/ops/moe/expert_cache_transfer.py`; `python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh`; graph wrapper/backend tests; `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`.

**Chosen first mechanism:** Retain the existing SM-driven host gather and one graph stream. Capture a fixed number of lane operations determined by plan capacity. Each active lane waits for its own generation-qualified readiness, copies only that row to its reserved GPU destination, then acknowledges source consumption. Inactive lanes are no-ops. All lane copies precede one fused MoE invocation.

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

These experiments are not requirements for accepting Task 4. Their deferred status must remain visible in the final implementation report.

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
