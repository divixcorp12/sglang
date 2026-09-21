# Storage v2 report: accepted, rejected, deferred and open work

**A living document.** Plan: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md` (the "plan"). Its final acceptance
item reads: *"Final report names accepted, rejected, and deferred work. Storage overlap is not labeled a fully asynchronous
inference pipeline while Task 8 remains blocking."* This is that report. Written 2026-09-21 at `dsv41` HEAD `ec64c07999`, from the
plan's tick state and annotations, the analysis documents in `analysis/dsv41-drive/`, the arm files, and the code at HEAD. **It has
had no independent review pass**; the plan requires one and this is not it.

Status words (a status is a claim about evidence, so each is defined by what it needs):

| word | meaning |
|---|---|
| **ACCEPTED** | the task's gate is met by named evidence (a commit or an artefact) |
| **PARTIAL** | some items are met; the missing evidence is named |
| **REJECTED** | tried, and a named measurement or probe said no |
| **DEFERRED** | not done, with a reason and the place it is tracked |
| **DESIGN ONLY** | a written design exists; no code was written or run for it |
| **OPEN** | not started, or started without a gate-level result |

How to update: change a status only with the commit or file that justifies it; keep the evidence column and the "missing" column
current; add a dated line to section 8. Do not turn a number into a claim by dropping its `n`.

## 0. The plain statements

1. **The pipeline is not fully asynchronous, and must not be described as one.** Promotions still block: on the scheduler thread they
   synchronise the stream, pause the native service across all layers and wait for NVMe reads (`PROMOTION_ASYNC.md` section 1.1).
   Task 8 has a design (`09dcb79564`) and no implementation. Tasks 5 and 6, which would let ready rows reach the GPU before all reads
   finish, are not implemented either. What exists is storage/CPU-packing overlap inside one request (Task 4, two banks), with the
   original whole-request GPU readiness contract unchanged. **No I/O-to-GPU-transfer overlap is claimed.**
2. **What is accepted:** native mirroring reaches decode and is byte-exact (Task 0); the matched baselines (Task 1, cells with `n=5`,
   `4` and `1`); the static 1:1 split retained on measurement (Task 2); the topology and reader audits (Task 3); the two-bank pipeline
   is implemented and unit-tested (Task 4, partial).
3. **What the measurements rejected:** whole-row assignment, the weighted split, `SINGLE_ISSUER|DEFER_TASKRUN` in the native
   service (section 3).
4. **What is deferred:** `READ_FIXED`, the registered bounce, Task 3's placement measurements (to Task 9), the asymmetric-load
   experiment, Tasks 6 and 7, Task 8's implementation, the Task 9 follow-ups.
5. **Effect sizes and their `n`:** mirrors on over off **1.347x (three on-arms) to 1.351x (four)**, off `n=5`; the newer code over the older
   **+3.2 % to +3.5 %** with the **older side `n=1`**, unmonitored, and the delta being two behavioural commits, not two-bank alone.
   Source: `PIPELINE_BASELINE.md` sections 1, 3.2.
6. **Gates that are not met:** Task 1 (device intervals, promotion boundaries), Task 4 (overlap timeline, GPU parity), Task 5, Task 6,
   Task 8, and the final acceptance list except where section 5 says otherwise.

## 1. Status by task

| task | status | strongest evidence | what is missing for the gate |
|---|---|---|---|
| 0 Native mirror prerequisite | **ACCEPTED** | `4c9b767e8d` `18949eafad` `a5547e0ef4` `02d9494e0d`; §19 gate: nvme2 0.00 % of service-attributed expert bytes, 20,800 requests | pass counts for the Task 0 test run are not on the tick; the service-attributed gate is one pair of arms |
| 1 Stage instrumentation and baselines | **PARTIAL** | 17 arms, matched cells (`PIPELINE_BASELINE.md`); host stage timestamps `b4257dd412` | device intervals; per-row chain test; disabled-tracing test and overhead (harness `038abc5b8f` written, **unrun**); the non-graph conditions; promotion-boundary attribution |
| 2 Scheduling vs cache traffic | **ACCEPTED** for the policy decision, **item 1 retired, not resolved** | `ff104147ca`, `SCHEDULING.md`, `EAGER_ANOMALY.md`, `ec64c07999` | evidence explaining the 1.54x (none exists, see 2.3); asymmetric-load experiment (designed, unrun) |
| 3 Topology and registration | **ACCEPTED** (audit, topology, decision); measurements **DEFERRED** | `50cf23b619`, `036900c5cd`, `96149bb89e`, `a4e8a5f6f6` | storage alone / SM alone / simultaneous (designed) |
| 4 Overlap I/O and packing | **PARTIAL** | `ddcb0d55ff`; split+thread suites 281 passed, 0 failed (plan annotation) | packing-worker comparison never run; GPU graph parity not run; the overlap timeline; tail regression |
| 5 Source leases and acknowledgements | **DESIGN ONLY** | `LEASE_PROTOCOL.md` (`cc30fe34f4`, 1,334 lines) | everything after design; the protocol has not been model-checked or run |
| 6 Per-row GPU transfers | **OPEN** (blocked on 5) | none | all |
| 7 Raw-row pinned cache (optional) | **OPEN** (blocked on 4-6 measurement) | none | `RAW_ROW_CACHE.md` does not exist |
| 8 Non-blocking promotions | **DESIGN ONLY** | `PROMOTION_ASYNC.md` (`09dcb79564`, 1,317 lines) | implementation, tests, the gate measurement; depends on Task 5 |
| 9 Follow-ups | **DEFERRED** (tracking table) | plan Task 9 table | all six rows; Task 3's deferrals are not yet in the plan's table (section 2.9) |

## 2. Task by task

### 2.1 Task 0: native mirror prerequisite: ACCEPTED

- **Accepted, with the artefact:** reconcile and bridge check (`4c9b767e8d` extent table, `18949eafad` per-extent native reader); the offset-contract
  correction (`f7250e768b`, prose only; the code was right); bounded SQ preparation and reap/refill with a test-only `max_outstanding` cap to
  reach the refill path that production constants never reach; parity, zero parts, reverse CQEs as a fault knob; EOF handling from valid bytes
  (`a5547e0ef4`, which found a live `max(0, ...)` clamp bug in the eager Python reader); open-time size check and env wiring (`02d9494e0d`).
- **Gate evidence:** service-attributed bytes: nvme2 0.00 %, byte parity exact, extents 9,655 to 19,310 (`DSV41_REFERENCE.md` §19 "The gate"; 20,800 requests,
  13,589 touch and 7,211 demand, zero failed). **That is one pair of arms.** Corroboration that is not attribution: `/proc/diskstats` in **all 12** mirrors-on
  arms of the Task 1 series shows `nvme0` at 197.48 GiB and `nvme4` at 197.46-198.52 GiB, and the mirrors-off arms show nvme2 at 402.59 GiB
  (`R/*.cache.json`, where `R/` is divix01's `.../analysis/dsv41-drive/task1-results/`, see `PIPELINE_BASELINE.md` §0).
- **Not evidenced in the files I read:** pass/skip/failure counts for the Task 0 test run (item 6 is ticked with no counts); model-output parity for the
  *graph* path (eager greedy output sha1s are identical across seven arms in `EAGER_ANOMALY.md`).
- **Superseded:** the single-shot 1.3482x (§19 follow-up) is replaced by the Task 1 cells; the commit message of `ddcb0d55ff` attributes it to two-bank, which is wrong
  and cannot be amended (corrected in §19).

### 2.2 Task 1: instrumentation and baselines: PARTIAL

- **Accepted:** the matched cells (mirrors off `2.903-2.919`, `n=5`; on `3.905-3.956`, `n=4`; older code `3.798`, `n=1`), the harness (`task1-baseline-arms.sh`, `provenance.py`, `task1_arm_verdict.py`), and
  the stage-timestamp set already built in `b4257dd412` (observation, reservation, submit, first/last CQE, per-row pack start/end, publication, done, per-extent CQE, byte split).
- **Read `PIPELINE_BASELINE.md` before quoting any figure.** Its main cautions: the older reference is one arm with no load record; 13 of 17 arms were never sampled for machine load;
  VALID and "not contended" do not mean undisturbed (`task1d-0`: 14.9 % slow with neither flag); the page-cache gate is growth-only; P3, P6 and P7 were falsified and P5 is untested.
- **Missing for the gate** ("distinguish storage queueing, reader time, packing, GPU wait/transfer, compute, promotion boundaries"): device intervals (needs the GPU, not started); per-row queue admission,
  per-extent submit/attempt counts; the per-row causal chain asserted under the CQE-reversal and packing-delay fault knobs; an overflow locatable in band; the disabled-tracing bypass test; the trace-on
  overhead (harness `038abc5b8f`, refuses to run without `--box-clear`, **unrun**); cold/warm misses, all-GPU hits, mixed rows and eager prefill as separate conditions.
- **In flight:** the `task1e` series (gen2 code at `87417376f5`, older code `099eadba33`, three arms each, interleaved, untraced) is **pre-registered** (`task1-results/task1e-PREDICTIONS.txt`, `450f86d17a`) and
  running on divix01 as of 01:35 CDT; no result is judged here. Its rules pre-declare a numeric non-stationarity test and say the outcome may be UNRESOLVED.
- **Code generations:** all 17 baseline arms are gen0 or gen1; `5e92db22dc` changed `python/` (trace schema 3) and starts gen2. A pre-registered check (`rebaseline-PREDICTIONS.txt`) says how to tell whether gen1 baselines carry across; it has not run.

### 2.3 Task 2: scheduling versus cache traffic: policy ACCEPTED, anomaly retired

- **ACCEPTED: retain static 1:1** (`StaticSplitPolicy((1,1))`). `ff104147ca`, `bench_row_scheduling.py` (22 tests), `SCHEDULING.md`; two runs of ~47.6 GiB each, 100 batches, 304 rows per rep, sizes 1/2/4/6/8/32.
- **REJECTED by measurement, whole-row assignment:** one row per batch is **1.8x slower** (p50 4.154 ms against 2.337 ms in run 1; per-batch ratio 1.764, 0 of 96 wins); at 4-32 rows it is
  1-2.3 % *faster* (per-batch ratio 0.977-0.991), which reproduced in both runs but has no established mechanism and small `n` (18 batches at 8 rows, 6 at 32). A hybrid (whole-row only above a threshold) is
  the one variant the data could support and needs an end-to-end arm; it is **not adopted**.
- **REJECTED by measurement, the weighted split:** 1.009-1.022x slower than 1:1 at every size in both runs, although the solo calibration said nvme0 was 3-4 % slower. Unexplained.
- **The request-count asymmetry, stated with its evidence level.** The mirrors differ in `max_sectors_kb` (512 against 256), so one 6.66 MB extent is about 13 against 26 block requests; an even byte split is a 1:2 request
  split (`SCHEDULING.md`, model). **The model is unvalidated** (no `reads_completed` was recorded). **Measured, solo:** the drive with twice the requests was the faster one (nvme4 3456 and 3412 MB/s against nvme0
  3329 and 3293), so request count did not bind at 3.3-3.5 GB/s. The runs lack per-arm page-cache residency, block counts and load, so they cannot be cited as matched evidence for or against a request-count effect.
- **Item 1, the 1.54x eager byte increase: RETIRED AS NON-REPRODUCIBLE, NOT RESOLVED.** There are no extra reads in the mirror arm; the ratio is one base arm (`e-base`) reading about 147 GiB *less* than every other
  arm (`EAGER_ANOMALY.md`, `7f0fecb948`, `ec64c07999`). The gate requires "evidence explaining the additional reads"; there is none, so the item stays open by its own rule. The durable deliverable is a diagnosis protocol,
  not a cause. Do not cite `e-base`'s 257.96 GiB, the 1.54x, or the 25 % regression derived from them (§19 already retires them).
- **DESIGNED, not run:** the asymmetric-load experiment (reserved scratch space, 25 %/50 % interferer, Latin-square rotation). Not run because the decision it would inform is settled.

### 2.4 Task 3: topology and registration: ACCEPTED, measurements DEFERRED

- **Accepted:** the topology record (`50cf23b619`, extended and re-verified in `036900c5cd`), and the audit. Findings that outrank the inventory (`TOPOLOGY.md` headlines):
  `taskset -c 0-63` does not keep NVMe completion interrupts off cores 64-71, including core 71 (irqbalance is active, so a snapshot that was stable over 25 minutes); the two mirrors are different storage devices (66 against 3,112
  file extents, 512 against 256 KiB request ceiling, different model, queue count and filesystem).
- **REJECTED by probe, `SINGLE_ISSUER|DEFER_TASKRUN` in the native service:** on this kernel and liburing, a `DEFER_TASKRUN` ring shows a deferred completion only after `io_uring_get_events()`, `submit_and_wait(1)` or
  with `TASKRUN_FLAG`, never after `submit(0)`, which is the path `reap()` takes whenever a packed row is ready, so credit would starve while rows pack (`TOPOLOGY.md` §8.3, `defer_probe.c`). Separately the native ring is created
  on the Python thread and driven by the service thread, so creator-thread ownership is not a property it has (§8.2). The claim about the native loop is an inference from the code plus the probe, **not a run of the service**.
- **DEFERRED, `READ_FIXED`:** cheap here (one `posix_memalign` bounce, so one iovec covers every extent and the fallback rate is zero by construction), but no benefit was measured: wall time was identical in every arm because the
  drives set the pace, and the CPU saving exists only for a 4 KiB-backed bounce (`TOPOLOGY.md` §3, §8.4a).
- **DEFERRED, the registered-bounce experiment:** until a Task 4 timeline shows the owner thread saturated. Registration cost (measured, CPU only): 59.9 ms with THP or 106.6 ms on 4 KiB pages at 256 MiB; the 213 MB bounce is an interpolation.
- **DEFERRED to Task 9, storage alone / SM transfer alone / simultaneous:** designed in `TOPOLOGY.md` §9.A-9.C. The Task 3 gate ("choose placement and a budget for Task 4") is met only as a **recommendation** (service thread on a node-0
  core, explicit `cpu_core`, no core in 64-71, 213 MB bounce and the unchanged tier), **which is not implemented**: production still calls `start_thread` without `cpu_core` (`srt/layers/moe/exl3_ram_miss.py:383`).
- **Not evidenced:** SQPOLL and IOPOLL are unmeasured (`poll_queues=0`, IOPOLL cannot be exercised here).

### 2.5 Task 4: overlap storage and CPU packing: PARTIAL

- **Accepted, with the artefact:** two bounce banks with reversed CQEs, delayed packing and poisoned recycled descriptors (`ddcb0d55ff`; tests `test_a_bank_is_not_reused_until_every_row_in_it_has_packed`,
  `test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte`, `test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor`, `test_a_short_read_in_either_bank_resubmits_only_its_own_extent`);
  a row is unpublishable unless its bytes were read (`pack_one`'s coverage check); demand reservation with at most one advisory row in flight; the fault matrix (partial submits, soft/hard errors, short reads, cancel, pause, full cache, shutdown, and the
  reader's generation wrap). **Native suites: 281 passed, 0 failed** (split and thread; the commit subset re-verified at 230), as the plan annotation records; I did not rerun them.
- **Missing, by name:** the packing-worker comparison ("never run", so the item is open); the **GPU graph parity tests**, not run; the **timeline that proves I/O and packing overlap** (needs stage timestamps on a real run); a statistically meaningful tail check.
- **Effect, with `n`:** the newer code over the older is **+3.2 % to +3.5 % in mean decode, positive in all four sessions, older side `n=1` and unmonitored**; the step-latency p99 is 0.528 s against 0.552 s (`n=1` older). That delta is `ddcb0d55ff` (two-bank) **plus**
  `cd14545797` (tier and Engram counters) and two comment-only commits; it is not attributed to two-bank alone. The `task1e` series exists to measure it at gen2 and may come back UNRESOLVED.
- **Gate:** "timeline proves I/O and packing overlap; no changed weights, stale slots, unbounded memory, or statistically meaningful tail regression" is **unmet**: the unit tests prove ordering and bounded memory; the timeline and the tail evidence are missing. "This task makes no H2D-overlap claim" holds.
- **Known unguarded path (deliberate):** a submit that consumes nothing while nothing is in flight is left to the service watchdog, with the signature recorded at the `submit_and_wait` call (`be76ba501f`). It is a hang path, not a silent one.

### 2.6 Task 5: source leases and device acknowledgements: DESIGN ONLY

`LEASE_PROTOCOL.md` (`cc30fe34f4`) specifies the protocol, generations and wrap, failure matrix, shutdown quarantine and tests, and lists 13 open questions and 4 owner decisions (`PROMOTION_ASYNC.md` lists 8 of its own). **No code was written or run.** An explicit-state
model (`lease_model.py`, 768 lines) is **untracked and uncommitted**, so it is not evidence yet. Every Task 5 item is unticked. Missing for the gate: lease-pressure and fault tests proving no reuse before consumption; the `ld.global.nc`
visibility experiment on the deployed GPU (needs the GPU); the model check. **The plan's premise "concurrent graph execution remains unsupported and guarded" is unsubstantiated:** the design found no guard, and the post kernel updates `state[kPosted]`
with a non-atomic read-modify-write (`LEASE_PROTOCOL.md` OPEN 9, by reading, not run). A plan that claims a guard that does not exist is worse than one that admits the gap, because every downstream design rests on it; the team lead is recording the correction in the plan itself.

### 2.7 Task 6: per-row GPU transfers: OPEN (blocked on Task 5)

Not started. It needs the lease contract. The gate says to retain Task 4 and record Task 6 as rejected if launch or head-of-line cost cancels the benefit; that has not been evaluated, and its comparison baseline (batched SM transfer at real miss counts,
link throughput, SM contention) has not been measured (Task 3's SM-alone measurement is deferred).

### 2.8 Task 7: raw-row pinned-cache experiment: OPEN (optional, blocked)

Not started; `RAW_ROW_CACHE.md` does not exist. Its first item needs the residual exposed packing after Tasks 4-6, which do not exist to measure. Existing figures are not new native measurements: scatter about 1.58-1.60 ms per row against a 2.55 ms read stage
(`MIRROR_ROWS.md`, pre-Task-4), and at eight rows 13.9 ms packing against 16.2 ms reading (`SCHEDULING.md`). "A negative result is a valid deliverable" applies; none exists.

### 2.9 Task 8 and Task 9

- **Task 8, DESIGN ONLY (`09dcb79564`):** four milestones M0-M3 (preserve CPU expert ids; spare slots, draining and ticketed publication; asynchronous RAM admission at promotion priority; asynchronous decision readback). **Nothing is implemented.** `SGLANG_MOE_HOT_ASYNC_PROMOTIONS`
  stays refused for EXL3. The design says its own throughput case is small: `DSV41_REFERENCE.md` §18.6 measured a mean boundary excess of about 115 ms (about 3.6 ms per token, about 1 % of throughput) and ruled on 2026-09-19 that an async promotion path is not worth taking for the
  gain. **Task 8 is required for a different reason: the plan forbids calling the pipeline fully asynchronous while promotions block.** Its gate is safe asynchronous publication and reduced boundary stalls at matched capacity; its proposed thresholds are marked for the owner to accept.
- **Task 9, DEFERRED:** the plan's six rows (native DMA against SM gather; better lookahead; hit/miss or multi-request compute overlap; stream memory waits; GDS; alternate-root failover) are unstarted. **Task 3's deferrals belong in this table and are not yet there:** storage alone, SM transfer alone, simultaneous execution, the
  registered-bounce experiment, `READ_FIXED`, SQPOLL/IOPOLL. Their deferred status must stay visible here until the plan's table carries them.

## 3. The negatives ledger

Easy to lose, and as valuable as the positives. Each names the evidence that says no.

| what | verdict | evidence | `n` / caveat |
|---|---|---|---|
| Whole-row assignment | rejected | 1.8x slower at one row (`SCHEDULING.md`, 0/96 wins); 1-2.3 % faster at 4-32 rows, unexplained | two runs; 6-144 batches per size |
| Weighted split | rejected | 1.009-1.022x slower at every size, both runs | close to 1:1 by construction |
| Static 1:1 | **retained** | same | as above |
| `SINGLE_ISSUER\|DEFER_TASKRUN` in the native service | rejected | probe `defer_probe.c` (`TOPOLOGY.md` §8.3); native ring's creator is not its driver | probe is CPU-only and deterministic; the native-loop consequence is inferred |
| `READ_FIXED` in the native service | deferred, not rejected | no measured benefit; wall time identical in every arm | Revision 1 probe: CPU-only, one drive |
| Registered bounce | deferred | as above; cost 59.9-106.6 ms once at 256 MiB | interpolated for 213 MB |
| "Two-bank is worth 1.35x" | rejected | that figure is the mirror effect; new-over-old is +3.2 % to +3.5 % (§19 corrected) | older side `n=1` |
| "Two-bank alone is +3.2 %" | not established | the delta is two behavioural commits | isolating it needs a build with only `ddcb0d55ff` |
| "Eager mirrors is slower / 1.54x more bytes" | retired | no extra reads exist (`EAGER_ANOMALY.md`) | the cause of `e-base`'s low bytes is unknown |
| "Tier warming explains e-mirror" | refuted | `DSV41_REFERENCE.md` §19: tier hits capacity inside session 0 in both arms | |
| Boot re-read evicts pages (eviction explanation) | withdrawn | `diskstats` contradicted it | `PIPELINE_BASELINE.md` §7.2 |
| P3, P6, P7 (pre-registered) | falsified | `PIPELINE_BASELINE.md` §6 | P5 untested |
| "VALID / not contended means undisturbed" | rejected | `task1d-0`: 14.9 % slow with neither flag | one arm, the most informative one |
| Page-cache gate sees everything | rejected | growth-only; the `/mnt/nvme4` mirror fell 30.07 to 14.3 GiB inside one timed session of `task1d-3` and the arm was VALID | observation; cause unrecorded |

## 4. Defects found during this work

Three columns matter: how the defect was established (**executed**, **by reading**, **fixed**), whether it is reachable today, and what it does.

### 4.1 Code defects confirmed

| defect | how established | reachable today | consequence | tracked in |
|---|---|---|---|---|
| **Demand and advisory sequence wrap.** The device skips sequence 0 on wrap (`post` kernel and `sim_post`: `if (seq == 0) seq = 1`); the service advances with `next_demand_ += 1u` in `pump_demand` **and** `next_advice_ += 1u` in `pump_advice` (the design document names only the demand side), with no skip, still present at HEAD | **by reading at HEAD, and by execution** by the agent who wrote the stage instrumentation: `demand_head = demand_done = 0xFFFFFFFD`, four records posted through `sim_post`, one phantom iteration observed with `demand_done` stepping `0xFFFFFFFF` -> `0` -> `1`. The test, `test/registered/unit/kernels/test_exl3_ram_miss_wrap.py`, is **untracked and uncommitted**; I did not run it | only after 2^32 requests, and **the rate is the assumption**: about **310 days at 160 requests/s** (40 layers, per `DSV41_REFERENCE.md` §3, at an assumed 4 tok/s; my estimate) or about **41 days at about 1,200 requests/s** (the original estimate, from the instrumentation agent). Neither rate is measured here | **two signatures.** On a *used* page: one overrun (demand) or skipped advisory, and the done word steps back to 0. On a *never-used* page the phantom read **succeeds as an empty touch record with `overruns = 0`**, so the counter anyone would check does not fire. A generation scheme layered on `next_demand_` would be off by one | `LEASE_PROTOCOL.md` D6 (demand only), OPEN 1; **not fixed**. Independent review: real, low |
| **`lanes <= 8` is not enforced at attach: a latent bug.** `Exl3RamMissService.attach` passes `streamer.graph_gather_rows` as the backend's capacity (`srt/layers/moe/exl3_ram_miss.py`, the `Exl3RamMissRowBackend(...)` construction) with no `<= 8` check; `translate` bounds lanes by `planned.numel() = max(capacity, MAX_IDS)`; the post kernel reads `min(count, kMaxIds = 8)` lanes (`exl3_ram_miss.cuh`) | by reading, and by the independent review against the source at HEAD | **not at batch size 1:** DSV4.1 is top-6 (`DSV41_REFERENCE.md` §3), so at most 6 lanes. **Yes for any multi-token graph:** `graph_gather_rows` is tokens x top_k (`expert_route_plan.py`), which exceeds 8 from two tokens | lanes past 8 are never requested from the service; the wait kernel then finds them missing and raises a fatal via `unserved_misses`, so it **fails stop, not silent** | `LEASE_PROTOCOL.md` D7 and OPEN 2 (which this closes: the top-k is 6); the attach-time check is **missing** and not added |
| Past-EOF extent clamped to nothing returned success while publishing stale bounce bytes | executed (found by the item's own precise wording) | was reachable | stale bytes published as a served row, with every observable reporting success | **fixed**, `a5547e0ef4` (the `max(0, ...)` clamp is load-bearing); `ddcb0d55ff` then added a coverage check so a row is unpublishable unless its bytes were read |
| `ensure_started` built its tables with no roots, so the mirror env never reached in-graph reads | found while measuring | was reachable | the feature was inert; §19's first arms measured nothing | **fixed** (`DSV41_REFERENCE.md` §19 "Three defects") |
| Latent io_uring crash at three or more roots (fixed 16-SQE ring against 3 x 8 rows returned a null `get_sqe`) | found while measuring | was reachable with three roots | null dereference | **fixed** (queue and credit, Task 0) |

### 4.2 Design-level defects D1-D7 in `LEASE_PROTOCOL.md`

Established **by reading** in the design document; D6 has since been executed (section 4.1). Numbering note: `PROMOTION_ASYNC.md` uses its own D1-D3, which are different defects (section 4.3); the two series are kept separate so that no citation breaks.
The **corrected severities** are from an **independent review** by the agent who wrote the stage instrumentation in the same source file, checked against the source at HEAD. They are that review's, not the design document's; where the review
says the document overstated, the third column says what.

| id | defect | what the review corrected | corrected severity |
|---|---|---|---|
| D1 | on a wait timeout or failure the translate loop and `host_rows` still run, and the copy runs with `plan.count` unconditionally | mechanism confirmed. Contained by `keep = 0` plus the watchdog abort, **but not designed to be** | **real, low-medium** |
| D2 | the device resolves slots through the mutable service-written slot map | mechanism correct, but the document's "every counter still reports success" is **not reachable on current code**: after a timeout `ok = false` and `keep = 0`, so the counters do not report success. A latent design defect that Task 6 turns live, not a present bug | **real, overstated for today** |
| D3 | no GPU-to-CPU acknowledgement exists (the only one is `demand_done`) | a missing feature more than a defect | **real** |
| D4 | the wait kernel cannot be interrupted | the document's "prompt shutdown impossible by construction" is overstated: the spin is **bounded by `timeout_ns`, default 2,000 ms** | **real, low** |
| D5 | shutdown does not establish that GPU readers finished; slab-unregister finalizers run at exit | torch registers `weakref.finalize`'s atexit hook before `ops.py` registers `_stop_live`, and atexit is LIFO, so `_stop_live` runs **first** and the slab-unregister finalizers second, with no device barrier between (this settles OPEN 13). **But production SIGKILLs the scheduler, so none of these hooks run there.** It is a normal-Python-exit and test-teardown hazard | **real in mechanism; severity depends on the exit path** |
| D6 | the demand sequence wrap (section 4.1) | **confirmed by execution**; also present in `pump_advice`, which the document does not name | **real, low** |
| D7 | lane capacity silently 8 (section 4.1) | not reachable at batch size 1 (top-6); reachable for multi-token graphs; fail-stop | **real, low today** |

### 4.3 Design-level defects in `PROMOTION_ASYNC.md`, by reading, none run

- **D1 destroy-then-copy:** `stage_reassign` reuses the slot indices it just retired, so under any asynchronous publish the evicted experts become misses while replacements are still loading. Correct today only by temporal exclusion.
- **D2 partial enqueue:** if the third of six copy operations raises, the first two are already queued and `abort_promotion` frees the destinations; a later reservation could reuse a slot a queued kernel will still write. Rare; a hazard, not a reproduced bug.
- **D3 a full ticket ring raises** (`RuntimeError`) where an asynchronous submitter needs a deferral.
- **A generation field uploaded to the device and never read:** `FixedRowTransferPlan` uploads `generations`, but no occurrence of "generation" in `expert_cache_transfer.cuh` or its wrapper (by grep). A reader would assume the device enforces it.

### 4.4 Findings that are not code defects

- **NVMe completion interrupts can reach cores 64-71** from submitters on some cores in 0-63, including core 71; `irqbalance` is active (`TOPOLOGY.md` headline).
- **The mirrors are unequal devices** (66 against 3,112 extents; 512 against 256 KiB request ceiling).
- **The page-cache gate is growth-only** and the `/mnt/nvme4` mirror lost 15.8 GiB of residency inside one timed session (`PIPELINE_BASELINE.md` §7.3).
- **13 of 17 baseline arms have no machine-load record.**

## 5. The plan's final acceptance list

| item | status | evidence and what is missing |
|---|---|---|
| Byte and model-output parity for eager and graph paths | **PARTIAL** | eager: identical greedy sha1s in seven arms (`EAGER_ANOMALY.md`); service-attributed byte parity exact in one pair of graph arms. Graph-path output parity for the current code is **not evidenced** (GPU parity tests not run, Task 4) |
| Fault tests: zero parts, EOF, retries, queue saturation, stale completion, generation reuse/wrap, partial delivery, shutdown, full cache | **PARTIAL** | most in the split and thread suites (281 passed). **Demand/advisory sequence wrap is not covered and is buggy** (4.1). Partial delivery and GPU-side shutdown belong to Tasks 5-6 and are absent |
| Timeline proves each claimed overlap independently | **UNMET** | I/O and packing: not produced; I/O and GPU transfer: no such overlap exists |
| Matched unprofiled runs report tokens/s, p50/p95/p99 step latency, `n`, traffic, misses, CPU, footprint, promotion stalls | **PARTIAL** | tokens/s, step latency, `n`, traffic (`diskstats`), CPU seconds: yes for the 17 arms. Promotion stalls, pinned/VRAM footprint per arm: not in the arm files I read |
| Original workload and numerics unchanged; capacity differences disclosed | **PARTIAL** | workload identical across arms by construction; numerics: see the first row. The arm script holds the workload and the 70 GiB tier setting identical; I did not audit every knob per arm |
| Independent reviewer checks ownership transitions and evidence after each structural milestone | **NOT RECORDED** | no review record in the tree for Task 4; this report and `LEASE_PROTOCOL.md`/`PROMOTION_ASYNC.md` were each written by one author and need one |
| Final report names accepted, rejected, deferred work; not labelled fully asynchronous while Task 8 blocks | **THIS DOCUMENT** | sections 0-3; it is not complete until it has been reviewed and updated as the open items close |

## 6. How much each headline number rests on

| number | `n` | monitoring | source |
|---|---|---|---|
| mirrors off 2.903-2.919 tok/s | 5 | unmonitored (no boundary samples) | `PIPELINE_BASELINE.md` §3.1 |
| mirrors on 3.905-3.956 | 4 | unmonitored | same |
| older code 3.798 | **1** | **unmonitored**; four further old arms disturbed | §5 there |
| mirror effect 1.347x-1.351x | 5 and 4 | as above | §3.2 there |
| new over old +3.2 % to +3.5 % | old **1** | as above; delta is two behavioural commits | §3.2 there |
| whole-row 1.8x slower at one row | 144 batches, two runs | conditions not recorded (`SCHEDULING.md`) | `SCHEDULING.md` |
| whole-row 1-2 % faster at 4-32 rows | 6-36 batches per size, two runs | conditions not recorded | same |
| nvme2 0.00 % of expert bytes with mirrors | **one pair of arms** | service-attributed | §19 |
| native suites 281 passed | one recorded run | not rerun here | plan annotation |
| registration cost 59.9-106.6 ms at 256 MiB | median of 7, one run each | CPU only, box not quiet | `TOPOLOGY.md` §7.7 |

## 7. In flight, uncommitted, and who to ask

- `task1e`: running on divix01 (gen2 against older code, 3 v 3, pre-registered); result not judged here. It touches the two-bank attribution.
- **Uncommitted:** `lease_model.py` (a model check for Task 5), `test_exl3_ram_miss_wrap.py` (the wrap test), a modification of `EAGER_ANOMALY.md`, and edits to the native service file and `ops/moe/exl3_ram_miss.py` made by another session. Findings resting on them are marked as such above.
- **Needed from the owner:** the decisions listed in `LEASE_PROTOCOL.md` §19 and `PROMOTION_ASYNC.md` §14; whether to add Task 3's deferrals to the plan's Task 9 table.
- **Not run by design (GPU or quiet box needed):** Task 1 device intervals and trace overhead; Task 3 storage/SM/simultaneous; Task 4 GPU parity; every Task 5-8 test that is not CPU-only.

## 8. Change log

- 2026-09-21: first version, at `ec64c07999`. Written by a single session; not reviewed.
- 2026-09-21: D1-D7 corrected severities added (independent review, attributed); sequence wrap given its execution result, two signatures and both rate assumptions; `lanes <= 8` recorded as a latent bug with a missing attach-time check, DSV4.1 being top-6; OPEN 13 settled. Still not independently reviewed as a whole.
