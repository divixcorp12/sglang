# Storage v2 report: accepted, rejected, deferred and open work

**A living document.** Plan: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md` (the "plan"). Its final acceptance
item reads: *"Final report names accepted, rejected, and deferred work. Storage overlap is not labeled a fully asynchronous
inference pipeline while Task 8 remains blocking."* This is that report. First written 2026-09-21 at `ec64c07999`; **updated the
same day at `dsv41` HEAD `a3b2b317e1`**, from the plan's tick state (read, not trusted: 20 of 59 items checked, counted below), the
analysis documents in `analysis/dsv41-drive/`, the arm files, and the code at HEAD. **This report itself has had no independent review
pass**; the plan requires one and this is not it. Six reviews of *other* work exist (section 5).

Status words (a status is a claim about evidence, so each is defined by what it needs):

| word | meaning |
|---|---|
| **ACCEPTED** | the task's gate is met by named evidence (a commit or an artefact) |
| **PARTIAL** | some items or one half of the gate are met; the missing evidence is named |
| **REJECTED** | tried, and a named measurement or probe said no |
| **DEFERRED** | not done, with a reason and the place it is tracked |
| **DESIGN ONLY** | a written design exists; no code was written or run for it |
| **OPEN** | not started, or started without a gate-level result |

How to update: change a status only with the commit or file that justifies it; keep the evidence column and the "missing" column
current; add a dated line to section 8. Do not turn a number into a claim by dropping its `n`. **A status may not improve beyond what
its evidence supports, even if that leaves the report looking unfinished.**

## 0. The plain statements

1. **The pipeline is not fully asynchronous, and must not be described as one.** Promotions still block: on the scheduler thread they
   synchronise the stream, pause the native service across all layers and wait for NVMe reads (`PROMOTION_ASYNC.md` section 1.1).
   Task 8 has a design, twice reviewed, and no implementation. Tasks 5 and 6, which would let ready rows reach the GPU before all reads
   finish, are not implemented either. What exists is storage/CPU-packing overlap inside one request (Task 4, two banks), with the
   original whole-request GPU readiness contract unchanged. **No I/O-to-GPU-transfer overlap is claimed.**
2. **What is accepted:** native mirroring reaches decode and is byte-exact (Task 0); the matched baselines (Task 1, cells with `n=5`, `4`
   and `1`); the static 1:1 split retained on measurement (Task 2); the topology and reader audits (Task 3). **Task 4 is PARTIAL:** its
   timeline half is now met (section 2.5), its tail, GPU-parity and packing-worker items are not.
3. **What the measurements rejected:** whole-row assignment, the weighted split, `SINGLE_ISSUER|DEFER_TASKRUN` in the native service
   (section 3). **New tonight, a model-based redirect, not a rejection:** Task 6's chosen per-row mechanism is not justified by the benefit
   it was designed to exploit (section 2.7).
4. **What is deferred:** `READ_FIXED`, the registered bounce, Task 3's placement measurements (now in the plan's Task 9 table), the
   asymmetric-load experiment, Task 7, Tasks 6 and 8 implementations, the Task 9 follow-ups.
5. **Effect sizes and their `n`:** mirrors on over off **1.347x (three on-arms) to 1.351x (four)**, off `n=5`. The newer code over the older
   **+3.2 % to +3.5 %** with the **older side `n=1`**, unmonitored, and the delta being two behavioural commits, not two-bank alone.
   **A six-arm series meant to replace that `n=1` returned UNRESOLVED** (section 2.5): three arms ran, two were valid, the pair's 1.047 is not a
   result. Source: `PIPELINE_BASELINE.md` sections 1 and 3.2; `task1-results/task1e-RESULT.txt`.
6. **Three code defects were fixed tonight, all on the measured path** (section 4.1): the sequence-wrap skip, the `lanes <= 8` attach
   refusal, and drive resolution by `st_dev`. They change `python/`: **`HEAD`'s `python/` tree (`8d3312fdf0a7`) is in no code-generation
   manifest** (`clean-reference.json` has gen0-gen2), so arms run from here are a new generation and are not comparable to the baselines without
   the re-baseline check that has not run.
7. **Gates that are not met:** Task 1 (device intervals, promotion boundaries), Task 4 (tail, GPU parity, packing-worker comparison), Task 5,
   Task 6, Task 8, and the final acceptance list except where section 5 says otherwise.

## 1. Status by task

Plan ticks, counted from the plan file: **20 of 59 checked** (Task 0: 7 of 7; Task 1: 2 of 6; Task 2: 4 of 6; Task 3: 3 of 4; Task 4: 4 of 6;
Tasks 5-8: 0 of 23; final acceptance: 0 of 7).

| task | status | strongest evidence | what is missing for the gate |
|---|---|---|---|
| 0 Native mirror prerequisite | **ACCEPTED** | `4c9b767e8d` `18949eafad` `a5547e0ef4` `02d9494e0d`; §19 gate: nvme2 0.00 % of service-attributed expert bytes, 20,800 requests | pass counts for the Task 0 test run are not on the tick; the service-attributed gate is one pair of arms |
| 1 Stage instrumentation and baselines | **PARTIAL** (2 of 6) | 17 arms, matched cells (`PIPELINE_BASELINE.md`); host stage timestamps `b4257dd412`; trace export and bypass closed, **overhead measured** `6a2643c1b4` | device intervals (needs the GPU); per-row queue admission and attempt counts; the per-row chain as a chain under fault knobs; the non-graph conditions; promotion-boundary attribution; **trace-off against uninstrumented, unmeasured** |
| 2 Scheduling vs cache traffic | **ACCEPTED** for the policy decision; **item 1 retired, not resolved** | `ff104147ca`, `SCHEDULING.md`, `EAGER_ANOMALY.md`, `ec64c07999` | evidence explaining the 1.54x (none exists); asymmetric-load experiment (designed, unrun) |
| 3 Topology and registration | **ACCEPTED** (audit, topology, decision); measurements **DEFERRED** | `50cf23b619`, `036900c5cd`, `96149bb89e`, `a4e8a5f6f6` | storage alone / SM alone / simultaneous (designed; tracked in the plan's Task 9 table) |
| 4 Overlap I/O and packing | **PARTIAL** (4 of 6); **timeline half of the gate met** | `ddcb0d55ff`; 281 passed in the split and thread suites (plan); `overlap_timeline.py` `b49cfcb96c`, `a86020e8ce` | packing-worker comparison **never run and now the highest-value remaining item**; GPU graph parity not run; no statistically meaningful tail evidence; the gen2 comparison UNRESOLVED |
| 5 Source leases and acknowledgements | **DESIGN ONLY** (0 of 7) | `LEASE_PROTOCOL.md` incl. the model check and the ordered implementation plan (section 20); `lease_model.py`, 32 tests; the `ld.global.nc` experiment **pre-registered and its program written, not run** | everything after design; lease-pressure and fault tests; the `.nc` visibility result (needs the GPU); a model that has been reviewed, not a proof |
| 6 Per-row GPU transfers | **OPEN** (0 of 6); **mechanism redirected by a model-based precheck** | `3eb95a7423`, `per_row_precheck.py`, `PER_ROW_PRECHECK_PREREG.txt` | any implementation; a measurement of two-phase or per-row; the assumptions in the precheck (section 2.7) |
| 7 Raw-row pinned cache (optional) | **OPEN** (0 of 5) | none | `RAW_ROW_CACHE.md` does not exist; its first item needs residual packing after Tasks 4-6 |
| 8 Non-blocking promotions | **DESIGN ONLY** (0 of 5) | `PROMOTION_ASYNC.md`, reviewed twice (`PROMOTION_ASYNC_REVIEW.md`, `_REVIEW_2.md`) and corrected | implementation, tests with negative controls, the gate measurement; depends on Task 5 |
| 9 Follow-ups | **DEFERRED** (tracking table) | plan Task 9 table, now carrying Task 3's deferrals | all rows |

## 2. Task by task

### 2.1 Task 0: native mirror prerequisite: ACCEPTED

- **Accepted, with the artefact:** reconcile and bridge check (`4c9b767e8d` extent table, `18949eafad` per-extent native reader); the offset-contract
  correction (`f7250e768b`, prose only); bounded SQ preparation and reap/refill with a test-only `max_outstanding` cap to reach the refill path that
  production constants never reach; parity, zero parts, reverse CQEs as a fault knob; EOF handling from valid bytes (`a5547e0ef4`, which found a live
  `max(0, ...)` clamp bug in the eager Python reader); open-time size check and env wiring (`02d9494e0d`).
- **Gate evidence:** service-attributed bytes: nvme2 0.00 %, byte parity exact, extents 9,655 to 19,310 (`DSV41_REFERENCE.md` §19 "The gate"; 20,800 requests,
  13,589 touch and 7,211 demand, zero failed). **That is one pair of arms.** Corroboration that is not attribution: `/proc/diskstats` in **all 12** mirrors-on
  arms of the Task 1 series shows `nvme0` at 197.48 GiB and `nvme4` at 197.46-198.52 GiB, and the mirrors-off arms show nvme2 at 402.59 GiB
  (`R/*.cache.json`, where `R/` is divix01's `.../analysis/dsv41-drive/task1-results/`, see `PIPELINE_BASELINE.md` §0).
- **Not evidenced in the files I read:** pass/skip/failure counts for the Task 0 test run; model-output parity for the *graph* path (eager greedy output sha1s are
  identical across seven arms in `EAGER_ANOMALY.md`).
- **Superseded:** the single-shot 1.3482x (§19 follow-up) is replaced by the Task 1 cells; the commit message of `ddcb0d55ff` attributes it to two-bank, which is wrong
  and cannot be amended (corrected in §19).

### 2.2 Task 1: instrumentation and baselines: PARTIAL

- **Accepted:** the matched cells (mirrors off `2.903-2.919`, `n=5`; on `3.905-3.956`, `n=4`; older code `3.798`, `n=1`), the harness (`task1-baseline-arms.sh`, `provenance.py`,
  `task1_arm_verdict.py`), the host stage-timestamp set (`b4257dd412`), and **the trace export item, now closed**: an in-band `dropped_before` so a gap is locatable, a `stamp()`
  funnel through which every trace clock read passes, a bypass test asserting zero clock reads with tracing off, and a source scan that fails on any `now_ns()` outside the
  funnel (`5e92db22dc`).
- **Trace-on overhead, measured** (`6a2643c1b4`, `b4da256a56`): host CPU cost of trace-on minus trace-off, paired and interleaved, 25 reps of 300 requests, four independent runs on four
  quiet cores: median +0.5 us at zero rows read rising to +1.5-2.6 us at eight rows, **3.7-5.5 % of host CPU time**. **Qualified:** cache-resident buffered reads from tmpfs, no O_DIRECT, no
  drive, no GPU, on a contended box (load about 5.2). Two artefacts were found and fixed while measuring (a silent cap at 8 ids per request; a 65,536-slot ring inflating the 8-row delta
  several-fold). **Still unmeasured: trace-off against an uninstrumented build**, which needs an end-to-end arm and decides whether the earlier baselines carry across.
- **Read `PIPELINE_BASELINE.md` before quoting any figure.** Its main cautions: the older reference is one arm with no load record; 13 of 17 arms were never sampled for machine load;
  VALID and "not contended" do not mean undisturbed (`task1d-0`: 14.9 % slow with neither flag); the page-cache gate is growth-only; P3, P6 and P7 were falsified and P5 is untested.
- **Missing for the gate** ("distinguish storage queueing, reader time, packing, GPU wait/transfer, compute, promotion boundaries"): device intervals (not started, needs the GPU); per-row
  queue admission and per-extent attempt counts; the per-row chain asserted as a chain under the CQE-reversal and packing-delay fault knobs (`overlap_timeline.py` now checks it on real traces,
  see 2.5, but the fault-knob form is not there); cold/warm misses, all-GPU hits, mixed rows and eager prefill as separate conditions.
- **Code generations.** Gen0 (`099eadba33`), gen1 (`f6608901a3`), gen2 (`87417376f5`, trace schema 3), all in `clean-reference.json`. **`HEAD`'s `python/` tree is a fourth**
  (`8d3312fdf0a7`, after the wrap and attach fixes) **and is not in the manifest.** The pre-registered re-baseline check (`rebaseline-PREDICTIONS.txt`) has not run.

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
  arm (`EAGER_ANOMALY.md`, `7f0fecb948`, `ec64c07999`). The gate requires "evidence explaining the additional reads"; there is none, so the item stays open by its own rule. The durable deliverable is a diagnosis
  protocol, not a cause. Do not cite `e-base`'s 257.96 GiB, the 1.54x, or the 25 % regression derived from them.
- **DESIGNED, not run:** the asymmetric-load experiment (reserved scratch space, 25 %/50 % interferer, Latin-square rotation).

### 2.4 Task 3: topology and registration: ACCEPTED, measurements DEFERRED

- **Accepted:** the topology record (`50cf23b619`, extended and re-verified in `036900c5cd`, reorganised in `96149bb89e` and `a4e8a5f6f6`), and the audit. Findings that outrank the inventory:
  `taskset -c 0-63` does not keep NVMe completion interrupts off cores 64-71, including core 71 (irqbalance is active, so a snapshot that was stable over 25 minutes); the two mirrors are different storage
  devices (66 against 3,112 file extents, 512 against 256 KiB request ceiling, different model, queue count and filesystem).
- **REJECTED by probe, `SINGLE_ISSUER|DEFER_TASKRUN` in the native service:** a `DEFER_TASKRUN` ring shows a deferred completion only after `io_uring_get_events()`, `submit_and_wait(1)` or with `TASKRUN_FLAG`, never after
  `submit(0)`, the path `reap()` takes whenever a packed row is ready, so credit would starve while rows pack (`TOPOLOGY.md` §8.3, `defer_probe.c`). Separately the native ring is created on the Python thread and driven by
  the service thread (§8.2). The claim about the native loop is an inference from the code plus the probe, **not a run of the service**.
- **DEFERRED, `READ_FIXED`:** cheap here (one `posix_memalign` bounce, so one iovec covers every extent and the fallback rate is zero by construction), no benefit measured: wall time was identical in every arm because the
  drives set the pace, and the CPU saving exists only for a 4 KiB-backed bounce (`TOPOLOGY.md` §3, §8.4a).
- **DEFERRED, the registered-bounce experiment:** until the owner thread is shown saturated. **This trigger is now closer than when it was written** (section 2.5: with mirrors on the packer is approaching the critical path), but the
  trigger the plan names is a saturated *owner thread*, which the timeline does not measure. Registration cost (measured, CPU only): 59.9 ms with THP or 106.6 ms on 4 KiB pages at 256 MiB.
- **DEFERRED, storage alone / SM alone / simultaneous:** designed in `TOPOLOGY.md` 9.A-9.C, now in the plan's Task 9 table. The Task 3 gate is met only as a **recommendation** (service thread on a node-0 core, explicit
  `cpu_core`, no core in 64-71), **which is not implemented**: production still calls `start_thread` without `cpu_core` (`srt/layers/moe/exl3_ram_miss.py:383`).
- **Not evidenced:** SQPOLL and IOPOLL are unmeasured (`poll_queues=0`).

### 2.5 Task 4: overlap storage and CPU packing: PARTIAL, timeline half of the gate met

- **Accepted, with the artefact:** two bounce banks with reversed CQEs, delayed packing and poisoned recycled descriptors (`ddcb0d55ff`; four named tests); a row is unpublishable unless its bytes were read; demand
  reservation with at most one advisory row in flight; the fault matrix. **Native suites: 281 passed, 0 failed** (split and thread), as the plan annotation records; I did not rerun them.
- **The timeline half of the gate is met** (`overlap_timeline.py`, `b49cfcb96c`, unit tests `test_overlap_timeline.py`; annotation `a86020e8ce`). On the seven schema-2 traces: **1,888 of 1,888 multi-row requests** have a row packed while
  another row's read was outstanding, **zero per-row chain violations**, hidden packing **51 % at p50 for two rows, 67-68 % for three or four** (bounded by (n-1)/n), exposed tail one row's packing, p50 **2.73-2.76 ms**.
  **Caveats that belong beside those numbers:** the seven traces are one workload replayed (each holds 7,211 served requests with the same row-count histogram), so "the seven agree" is agreement across arms, not across workloads;
  **74 % of served requests are single-row and cannot overlap within a request at all** (a property of the blocking-per-request design); and this is an overlap proof, not a speed-up claim.
- **What the timeline shows next:** with mirrors on the read window halves (15.3 to 6.0 ms at p50) while packing stays 5.6-5.7 ms per request, so packing covers 18 % of the read window with mirrors off and **49-54 % with mirrors on**, and rows
  that waited for the packer go from **0 of 4,332 to 104-112 of 4,332**. **The packer is approaching the critical path.** Parallelising packing would shrink Task 4's exposed tail and Task 6's remaining benefit at once, which makes the
  never-run **packing-worker comparison the highest-value remaining item in this task**.
- **Still missing for the gate:** "no changed weights, stale slots, unbounded memory, or statistically meaningful tail regression" is unaddressed by the timeline; the GPU graph parity tests have not been run.
- **Effect, with `n`:** the newer code over the older is **+3.2 % to +3.5 % in mean decode, positive in all four sessions, older side `n=1` and unmonitored**; step-latency p99 0.528 s against 0.552 s (`n=1` older). That delta is `ddcb0d55ff`
  (two-bank) **plus** `cd14545797` (tier and Engram counters) and two comment-only commits; it is not attributed to two-bank alone.
- **The series meant to firm that up returned UNRESOLVED** (`task1e`, `task1-results/task1e-RESULT.txt`): three of six planned arms ran, two were valid (one per cell), the third was refused by the drive-idle gate; the pair's ratio 1.047
  (gen2 new 3.8620 over gen0 old 3.6887, both contended) is **not a result**. Under the registered rules no arm was clean (STRICT admits 0 and 0). **Two of the verdict's gates were mis-calibrated for the box this series had** (the
  contended gate disqualifies every arm where contention is normal; the cross-arm check against one historical reference), **and a third, drive-idle, correctly caught a transient**. The series was stopped by decision after three
  continuations were killed by instructions that crossed in flight, none by the harness. Two amendments were made *during* the series, which a pre-registration should not need; the next series must register calibrated gates before it runs.
  The old-barrier cell stays `n=1`; improving it needs a quiet machine.
- **Known unguarded path (deliberate):** a submit that consumes nothing while nothing is in flight is left to the service watchdog (`be76ba501f`). It is a hang path, not a silent one.

### 2.6 Task 5: source leases and device acknowledgements: DESIGN ONLY

`LEASE_PROTOCOL.md` specifies the protocol, generations and wrap, failure matrix, shutdown quarantine and tests, and section 20 orders the implementation into eight steps (lease mode defaulting off; model traces as regression tests). **No
implementation exists; every Task 5 item is unticked.**
- **Model check.** `lease_model.py` (explicit state; 32 tests, 217 s) finds the epoch hole in an earlier draft of the protocol, reproduces D1 and the wrap defect from today's code, proves A3 necessary by search, and finds 13 mutants. It was
  **transcription-reviewed** (`LEASE_MODEL_REVIEW.md`, `2cbe9b4b92`): faithful where checkable, with one real blind spot (promoter and eager caller modelled as independent threads, which hides a cycle the design contains, F1),
  two designed rules it cannot show necessary (terminal retirement; fail-closed, which an `assert` forbids removing), R6 (shutdown with host leases) unmodelled, and about eight blind spots the limits list did not name. **A pass is a statement
  about the protocol's logic within small bounds, sequentially consistent, not about the memory model.**
- **`ld.global.nc` visibility:** **pre-registered** (`NC_VISIBILITY.md`, `3533e471a5`), program and runner written (`57846c5b8f`), **not run**: it needs the GPU lock. It decides whether the lease-mode copy keeps `ld.global.nc`.
- **The plan's premise "concurrent graph execution remains unsupported and guarded" is unsubstantiated:** no guard was found and the post kernel updates `state[kPosted]` with a non-atomic read-modify-write (`LEASE_PROTOCOL.md` OPEN 9). The plan
  now says so under Task 8; every design that rests on the guard must treat it as a requirement to build.
- **Missing for the gate:** lease-pressure and fault tests proving no reuse before consumption; the `.nc` result; the code.

### 2.7 Task 6: per-row GPU transfers: OPEN; the chosen mechanism is not justified by its own benefit

Not started (0 of 6). **A model-based precheck redirects it** (`3eb95a7423`; pre-registration `PER_ROW_PRECHECK_PREREG.txt` hashed and committed `22dae687f7` before the script ran; `per_row_precheck.py`, result in
`per_row_precheck_result.txt`):
- the modelled saving is real, **19.2 ms per step at random lane order, about 7.1 % of a 257.5 ms step after an assumed 1.0 ms launch cost**;
- **87 % of it is RAM-hit lanes gathered while the NVMe read runs; only 13 % needs miss rows to finish at different times**; per-row's own contribution over a hits-then-rest **two-phase copy** is at most **5.06 ms per step, 2.0 % gross, at best lane order** (`Sigma(m-1)*c/495`), **at or below what the plan's own measurement design can resolve**, and at random lane order per-row is **14.35 ms worse than two-phase** (section 2.7, independent recomputation below). *(An earlier version of this line said "at most 2.55 ms, about 1.0 %": that figure is the random-order miss-only number, not the increment over two-phase; corrected after the recomputation.)*;
- the miss-row spread that motivated per-row is, **with mirrors on, p50 2.77 ms, which matches Task 4's serial packing at about 2.7 ms per row rather than drive completion order** (with mirrors off it is 7.4 ms, a drive-bound read, and the precheck's
  per-step saving is the same either way), so the task's premise was partly a misattribution of a Task 4 artefact in the configuration the project defends, and parallelising packing would shrink the 13 % further; the analysis also reports its own pre-registered rejection test as aimed at the wrong quantity.
**What this is not.** It is a model over the existing schema-2 traces of one workload, with named assumptions: hit lanes spread evenly over the 40 layers (the traces do not record lanes per layer; an instrumentation item is
requested), a hit lane is ready when the service reserves the request, a per-lane copy time of 1.055 ms taken from one measurement (`DSV41_REFERENCE.md` 18.2), and an assumed launch cost. **No two-phase or per-row copy has been built or
measured.** The redirect is a recommendation ("prefer two-phase unless a measurement shows otherwise"); the plan's Task 6 text and gate are unchanged and the task stays OPEN. If per-row is dropped the gate says to retain Task 4 and record
Task 6 as rejected; that has not been done, because the alternative has not been measured either.

**Independent recomputation (`PER_ROW_PRECHECK_REVIEW.md`, `f1cda7ba72`; own code from the model definition, not a re-run of `per_row_precheck.py`; four traces, one workload).**
Every headline number reproduces: BEST 38.6 ms/step (15.0 %), RANDOM 19.17 exact (registered Monte Carlo 19.2), miss-only 2.55, GENEROUS 75.4, the 87/13 split two ways (86.8 % by two-phase over best, 13.3 % by miss-only over RANDOM), the closed
form (0 mismatches in 90,048 cases), the A1 floor at 37 % of modelled hit lanes. It found five things the document had to absorb (none HIGH; each is now in the plan and the design):
- **R1, a misstated comparison.** Per-row over two-phase is **+5.09 ms at best order and -14.35 ms at random order**. The "2.55 ms if lane order is random" was the random-order miss-only figure, a different quantity. The redirect is stronger, not weaker: per-row needs the order array to avoid losing to two-phase.
- **R2, A1 bounded from the data.** Total hit lanes 111.9 per step; A1 credits 31.9 to layers that read; the trace forces **at least 10.8 and at most 63.6** into read layers. At the worst placement two-phase is about **4 %** (clears the 3 % bar) and random-order per-row about **2.8 %** (just under it). A1 does not prop V1 up; it decides the per-row column.
- **R3, the ordering has a different cause.** The "pack order equals ordinal order" result is **per-drive FIFO completion**, not the packer's tie-break: 0 inversions in 1,842 multi-row requests in each of three mirrors-on arms and the mirrors-off arm, 0 of 4,750 consecutive-row pairs on one drive part out of order, same-reap ties in 3.7 % of pairs (0 % mirrors-off). While the effect was thought to be the packer's it needed no assumption; a drive property **is** an assumption (queue depth, hardware, scheduler), and the design lists it.
- **R4, n_eff = 1 for the workload.** `rows_asked`, `vram_miss` and `ram_miss` sequences are identical across all four arms, so "agree to 0.1 ms" is timing agreement only. Across four sessions of the run the RANDOM saving is 5.3 %-8.7 % of step (two-phase 9.3 %-15.5 %): the 7.5 % headline hides a 2.4x range; every session clears 3 %.
- **R5, denominators.** T_step 257.5 ms is the mean of three untraced arms, one of them INVALID (`task1-6`); dropping it alone moves the mean up to 259.3 ms, and the four-`new:on`-arm mean of 254.4 ms is three traced arms and one untraced, not "clean" (the "disturbed" label I gave `task1c-3` was mine: its verdict is VALID with a boot-warm note). The denominator is 254-259 ms depending on the arm set, +-1 % on every percentage, immaterial. The plan's "about 1.5 % resolution" comes from `task1e`'s quiet-box sd and that series is UNRESOLVED, so the real resolution is worse.
- **R6, found after R1-R5: 495 trace lines are 511 decode steps.** Sixteen `graph_step` lines are merged (`steps == 2`, `routed_rows` 480, two steps' `vram_miss`); the precheck divides by 495 and takes `k` from a two-step `vram_miss` on those lines. The registered absolute figures are 3-9 % high (7-8 % for BEST, RANDOM, two-phase): BEST 35.8 (13.9 %), RANDOM 17.8 (6.9 %), two-phase 30.9 (12.0 %), per-row over two-phase +4.93 ms best order and -13.08 ms random order. **No verdict changes** (`PER_ROW_PRECHECK_REVIEW.md`, R6; the figures quoted above in this section are the as-registered ones unless stated).
- **Schema 4 measurement (`task1f-0-new-on-T`, gen4, 511 decode steps; lanes measured, timings from `task1-2`).** Hit lanes in read layers **33.9 per step**, inside R2's bounds (about 10.5-61.6 at 511) and 3.0x the SUPPORT floor; A1 had credited 29.3-31.9. With measured `k`: BEST **40.67 ms (15.8 %)**, RANDOM **20.07 (7.8 %)**, two-phase **35.74 (13.9 %)**; per-row over two-phase +4.93 ms at best order, **-15.67 ms at random order**. The A1-based figures were 13-16 % low after R6's correction (5-7 % below the registered ones); the layer spread was fine, the per-request `k` was not (exact for 36 %). `sum(lanes)` against `sum(vram_miss)` -0.34 % (decode requests only). A2 is untouched: there is still no early readiness signal. See `PER_ROW_PRECHECK_REVIEW.md`. **Supersedes R6's corrected figures for the model's `k`.**
Not recomputed: A2 (the early signal does not exist today), `c = 1.055` and the 1.0 ms launch cost (used, not measured), the 93 ms read-wait sum, and the design sections. Schema 4 (`4e63616666`) records the planned lane count per request and can replace R2's bounds with a measurement once a schema-4 graph-decode trace exists (none does yet; see `PER_ROW_PRECHECK_REVIEW.md`, "Schema 4").

### 2.8 Task 7: raw-row pinned-cache experiment: OPEN (optional, blocked)

Not started; `RAW_ROW_CACHE.md` does not exist. Its first item needs the residual exposed packing after Tasks 4-6. The timeline now gives part of that number for Task 4 alone (exposed tail one row's packing, 2.73-2.76 ms), but
not "after Tasks 5-6". Existing figures: scatter about 1.58-1.60 ms per row against a 2.55 ms read stage (`MIRROR_ROWS.md`, pre-Task-4); at eight rows 13.9 ms packing against 16.2 ms reading (`SCHEDULING.md`). "A negative result is a valid deliverable"
applies; none exists.

### 2.9 Task 8 and Task 9

- **Task 8, DESIGN ONLY:** milestones M0-M3; **nothing implemented**; `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` stays refused for EXL3. The design says its own throughput case is small (`DSV41_REFERENCE.md` §18.6: mean boundary excess about 115 ms, about 3.6 ms
  per token, about 1 % of throughput, ruled on 2026-09-19 not worth an async path for the gain); **it is required because the plan forbids calling the pipeline fully asynchronous while promotions block.** Two reviews exist and were applied:
  the first (`PROMOTION_ASYNC_REVIEW.md`, mine) found a nominal safeguard and re-costed capacity; the second (`PROMOTION_ASYNC_REVIEW_2.md`, sections 5 and 11) found **HIGH** issues, including a safety test that cannot fail on its own bug (B1) and
  a lease release that depends on a host poll (A1). **Two reviews converged on one structural weakness**, the lease release depending on the scheduler thread, found from the state machine by one reviewer and from the model's actor-independence assumption by another
  (`LEASE_MODEL_REVIEW.md` F1); the design now requires that release not depend on the scheduler thread at all (`PROMOTION_ASYNC.md`, team lead's decision, 2026-09-21). A design that has been reviewed twice is a better design, not an
  implemented one.
- **Task 9, DEFERRED:** the plan's six rows are unstarted, and **Task 3's deferrals are now in the plan's table**: storage alone / SM alone / simultaneous, the registered bounce (`READ_FIXED`), and `SINGLE_ISSUER`/`DEFER_TASKRUN` (recorded there as
  a measured negative). Their deferred status must stay visible here.

## 3. The negatives ledger

Easy to lose, and as valuable as the positives. Each names the evidence that says no.

| what | verdict | evidence | `n` / caveat |
|---|---|---|---|
| Whole-row assignment | rejected | 1.8x slower at one row (`SCHEDULING.md`, 0/96 wins); 1-2.3 % faster at 4-32 rows, unexplained | two runs; 6-144 batches per size |
| Weighted split | rejected | 1.009-1.022x slower at every size, both runs | close to 1:1 by construction |
| Static 1:1 | **retained** | same | as above |
| `SINGLE_ISSUER\|DEFER_TASKRUN` in the native service | rejected | probe `defer_probe.c` (`TOPOLOGY.md` §8.3); native ring's creator is not its driver | probe is CPU-only and deterministic; the native-loop consequence is inferred |
| `READ_FIXED` / registered bounce | deferred, not rejected | no measured benefit; wall time identical in every arm | one drive, CPU only |
| **Task 6's per-row mechanism** | **redirected, not rejected** (model-based; arithmetic independently reproduced) | `3eb95a7423`, `f1cda7ba72`: 87 % of the saving is hit lanes; per-row over two-phase is +5.09 ms (2.0 %) at best lane order and -14.35 ms at random order | a model over one workload with named assumptions; nothing built or measured |
| **The premise that miss-row spread comes from drive order** | refuted for mirrors on (with mirrors off the spread is drive-bound, 7.4 ms) | the spread is serial packing, about 2.7 ms per row (`per_row_precheck_result.txt`, `overlap_timeline.py`) | same corpus; schema 2 |
| **The two-bank comparison at gen2 (`task1e`)** | **UNRESOLVED** | 3 of 6 arms, 2 valid, ratio 1.047 is not a result | contended box; two gates mis-calibrated; n=1 per cell |
| "Two-bank is worth 1.35x" | rejected | that figure is the mirror effect; new-over-old is +3.2 % to +3.5 % (§19 corrected) | older side `n=1` |
| "Two-bank alone is +3.2 %" | not established | the delta is two behavioural commits | isolating it needs a build with only `ddcb0d55ff` |
| "Eager mirrors is slower / 1.54x more bytes" | retired | no extra reads exist (`EAGER_ANOMALY.md`) | the cause of `e-base`'s low bytes is unknown |
| "Tier warming explains e-mirror" | refuted | `DSV41_REFERENCE.md` §19 | |
| Boot re-read evicts pages | withdrawn | `diskstats` contradicted it | `PIPELINE_BASELINE.md` §7.2 |
| P3, P6, P7 (pre-registered) | falsified | `PIPELINE_BASELINE.md` §6 | P5 **consistent with, not confirmed** after `task1e` (a Cached fall accompanied every boot-phase fall, but the test does not discriminate) |
| "VALID / not contended means undisturbed" | rejected | `task1d-0`: 14.9 % slow with neither flag; and now the converse, a gate that disqualifies every arm on a box where contention is normal | one arm; one series |
| Page-cache gate sees everything | rejected | growth-only; the `/mnt/nvme4` mirror fell 30.07 to 14.3 GiB inside one timed session and the arm was VALID | observation; cause unrecorded |
| "Trace overhead is negligible" | not established | measured 3.7-5.5 % of host CPU on tmpfs, contended, no O_DIRECT or GPU; trace-off against uninstrumented unmeasured | see 2.2 |
| "The seven traces are seven samples" | rejected | one workload replayed; the same 7,211 requests in each | affects every timeline and precheck figure |

## 4. Defects found during this work

Three columns matter: how the defect was established (**executed**, **by reading**, **fixed**), whether it is reachable today, and what it does.

### 4.1 Code defects: found, and their status

| defect | how established | reachable today | consequence | status |
|---|---|---|---|---|
| **Demand and advisory sequence wrap.** The device skips sequence 0 on wrap; the service advanced with `next_demand_ += 1u` / `next_advice_ += 1u` and resumed a lap at `head - ring + 2`, none skipping 0 | by reading; **executed** by the instrumentation agent (seed `0xFFFFFFFD`, four records through `sim_post`: `demand_done` steps `0xFFFFFFFF` -> `0` -> `1`); **four signatures observed before the fix**: demand ring used page `overruns = 1`; demand fresh page `touch_only = 1`, `overruns = 0`; advisory used page `advisories_skipped = 1`; advisory fresh page `advisories = 5`, `skipped = 0` | only after 2^32 requests: about **310 days at 160 requests/s** (40 layers, per `DSV41_REFERENCE.md` §3, at an assumed 4 tok/s; my estimate) or about **41 days at about 1,200 requests/s** (the instrumentation agent's). Neither rate is measured here | a spurious counter and a stage record once per wrap, a done word stepping back to 0; **not a failed request or a stuck waiter** (`a79ce2eef5`: every waiter still returns 1 pre-fix). On a fresh page the phantom is *served* with no overrun, so the obvious counter does not fire. The lap resume could also land on 0 | **FIXED**, `85cdbf9382`: a `skip_zero()` helper at **four jump sites** (both increments and both lap-skip resumes). Test `test_exl3_ram_miss_wrap.py` (10 cases, used and fresh page, both rings, laps, two controls seeded at `0xFFFFFFFF`): **the four wrap cases and both lap cases were observed failing before the fix, the controls pass**; `a79ce2eef5` adds a real service thread with a waiter on every sequence. Independent review severity: real, low. **Lesson:** a ring with a reserved value must skip it at every place the counter can move, and a resume-after-lap is such a place |
| **`lanes <= 8` not enforced at attach.** `Exl3RamMissService.attach` passed `graph_gather_rows` (tokens x top_k) as capacity; the post kernel reads `min(count, kMaxIds = 8)` lanes | by reading; the independent review | not at batch size 1 (DSV4.1 is top-6, at most 6 lanes); **yes for any multi-token graph** (two tokens of top-6 is 12) | lanes past 8 are never requested; the wait kernel fail-stops on the first not in RAM. **Fail-stop, not silent**, and it gave no error at startup | **FIXED**, `8c78749b35`: `attach` raises `ValueError` before building the device side. Test `test_exl3_ram_miss_attach_lanes.py`: 1, 6, 8 rows attach; 9 and 12 refused with nothing built; **with the check removed both refusal cases fail**; the existing service test still passes (25 passed with the new file). No working configuration is refused (the fused MoE already requires `graph_gather_rows == top_k`) |
| **Arm-harness drive resolution by name.** `provenance.DRIVES` mapped a mount label to a block-device *name* (`nvme4` -> `nvme3n1`); a name-keyed table silently reads nothing if the mapping changes | by reading, from the topology work's finding that there is no `nvme4` device | correct today; a latent trap | a drive read as idle when it is not, in a gate whose job is to detect that | **FIXED**, `c87b8dc181`: resolved by `st_dev` through `drive_conditions.resolve_drive`; raises when a device has no `/proc/diskstats` row; the eager arm driver now also records residency and boundary samples. **A semantic change to know when comparing arms:** `resolve_drive` returns the **partition** (`nvme3n1p1`) where the old table read the **whole disk** (`nvme3n1`), so absolute cumulative sectors differ while per-session deltas agree. The difference is **per drive**, not a constant: whole-disk minus partition was 2,608 sectors on nvme0, 42,400 on nvme2 and 4,296 on nvme4, measured on divix01 on 2026-09-21 (`task1-results/GENERATIONS.txt`, which is the source; not restated independently here). It comes from reads of the parent disk that the partition's counter does not include, and it is a difference of cumulative counters, so it **can grow**; it is not a fixed correction to subtract. No arm comparison used absolute sectors, only deltas. **The commit messages of `c87b8dc181` ("a constant ~4300 sectors") and `d5e0c3423c` carry the superseded figure**; they are history and cannot be amended, so the correction lives here and in `GENERATIONS.txt` Tests: 70 passed in `scripts/`; verified on divix01 against the real mounts (per the commit) |
| Past-EOF extent clamped to nothing returned success while publishing stale bounce bytes | executed | was reachable | stale bytes published as a served row, with every observable reporting success | **fixed**, `a5547e0ef4`; `ddcb0d55ff` added a coverage check so a row is unpublishable unless its bytes were read |
| `ensure_started` built its tables with no roots, so the mirror env never reached in-graph reads | found while measuring | was reachable | the feature was inert; §19's first arms measured nothing | **fixed** (`DSV41_REFERENCE.md` §19 "Three defects") |
| Latent io_uring crash at three or more roots (fixed 16-SQE ring against 3 x 8 rows returned a null `get_sqe`) | found while measuring | was reachable with three roots | null dereference | **fixed** (queue and credit, Task 0) |

**Generation note.** The wrap fix (`exl3_ram_miss_host.cpp`) and the attach refusal (`exl3_ram_miss.py`) change `python/`; the resulting tree is in no manifest (section 2.2). The drive-resolution change is in `scripts/`, not `python/`.

### 4.2 Design-level defects D1-D7 in `LEASE_PROTOCOL.md`

Established **by reading** in the design document; D6 has been executed and fixed (4.1). Numbering note: `PROMOTION_ASYNC.md` uses its own D1-D3 (section 4.3); the two series are kept separate so that no citation breaks. The **corrected severities** are from
an **independent review** by the agent who wrote the stage instrumentation in the same source file, checked against the source at HEAD; they are that review's, not the design document's. That review is recorded in `LEASE_PROTOCOL.md`; I did not find a standalone file.

| id | defect | what the review corrected | corrected severity | status |
|---|---|---|---|---|
| D1 | on a wait timeout or failure the translate loop and `host_rows` still run, and the copy runs with `plan.count` unconditionally | mechanism confirmed; contained by `keep = 0` plus the watchdog abort, **but not designed to be** | **real, low-medium** | open (designed away by Task 5, unimplemented) |
| D2 | the device resolves slots through the mutable service-written slot map | mechanism correct, but "every counter still reports success" is **not reachable on current code**; a latent design defect that Task 6 turns live | **real, overstated for today** | open |
| D3 | no GPU-to-CPU acknowledgement exists | a missing feature more than a defect | **real** | open (Task 5) |
| D4 | the wait kernel cannot be interrupted | "prompt shutdown impossible" overstated: the spin is **bounded by `timeout_ns`, default 2,000 ms** | **real, low** | open |
| D5 | shutdown does not establish that GPU readers finished; slab-unregister finalizers run at exit | `_stop_live` runs first under LIFO atexit, then the slab finalizers, with no device barrier; **production SIGKILLs the scheduler, so none of it runs there** | **real in mechanism; depends on the exit path** | open |
| D6 | the demand sequence wrap | confirmed by execution; also in `pump_advice`; the lap resume too | **real, low** | **FIXED** `85cdbf9382` |
| D7 | lane capacity silently 8 | not reachable at batch size 1; reachable for multi-token graphs; fail-stop | **real, low today** | **FIXED** `8c78749b35` |

### 4.3 Design-level findings in `PROMOTION_ASYNC.md`, by reading, none run

- **From the first review (mine):** `retire(..., consumer_complete)` is a caller assertion passed `True` unconditionally at every call site, not a check (F1, medium); the partial-enqueue hazard is real, mislocated, and weaker than stated (F2); the full-ring raise is
  unreachable today (F3); the section 9.2 deadlock is real for the proposal only (F4); capacity re-costed: the measured configuration has **888** resident slots, 22 or 23 per layer with no seed (F9).
- **From the second review (`PROMOTION_ASYNC_REVIEW_2.md`, sections 5 and 11):** **HIGH:** B1, the DRAINING poisoning test cannot fail on the bug it is for; B2, tests labelled "CPU only" need a GPU to construct their subject; A1, the source lease is released by a host poll, reopening the
  9.2 cycle through the host; A2, publication waits on a ring event the ring may already have re-recorded. Also A3 (`QUARANTINED` is never exited, medium-high) and MEDIUM items. It adds a requirement I would keep in front of every reader: **a safety test that has never been shown to fail is
  not evidence, so every safety rule needs a mutation control.**
- **A cross-document result:** the `lease_model.py` review (F1) found the scheduler-thread cycle from a different starting point than the second promotion review's A1. Neither author could see it from inside their own artefact.
- Numbering: the promotion document's own D1-D3 (destroy-then-copy; partial enqueue; full ring) are distinct from the lease document's D1-D7.

### 4.4 Findings that are not code defects

- **NVMe completion interrupts can reach cores 64-71** from submitters on some cores in 0-63, including core 71; `irqbalance` is active (`TOPOLOGY.md` headline).
- **The mirrors are unequal devices** (66 against 3,112 extents; 512 against 256 KiB request ceiling).
- **The page-cache gate is growth-only**, and the `/mnt/nvme4` mirror lost 15.8 GiB of residency inside one timed session (`PIPELINE_BASELINE.md` §7.3).
- **13 of 17 baseline arms have no machine-load record**, and the `task1e` series showed the harness's contended and cross-arm gates are calibrated for a quiet machine that this box is not.
- **Foreign processes do not stay on their nominal affinities** (`task1e-AMENDMENT.txt`), so no core range avoids them.

## 5. The plan's final acceptance list

| item | status | evidence and what is missing |
|---|---|---|
| Byte and model-output parity for eager and graph paths | **PARTIAL** | eager: identical greedy sha1s in seven arms (`EAGER_ANOMALY.md`); service-attributed byte parity exact in one pair of graph arms. Graph-path output parity for the current code is **not evidenced** (GPU parity tests not run) |
| Fault tests: zero parts, EOF, retries, queue saturation, stale completion, generation reuse/wrap, partial delivery, shutdown, full cache | **PARTIAL, improved** | most are in the split and thread suites (281 passed). **The demand and advisory sequence wrap is now covered** (10 cases, both rings, used and fresh page, laps, real thread; four cases observed failing before the fix). Partial delivery and GPU-side shutdown belong to Tasks 5-6 and are absent |
| Timeline proves each claimed overlap independently | **PARTIAL** | I/O and packing: **proved** (`overlap_timeline.py`; one workload). I/O and GPU transfer: no such overlap exists to prove |
| Matched unprofiled runs report tokens/s, p50/p95/p99 step latency, `n`, traffic, misses, CPU, footprint, promotion stalls | **PARTIAL** | tokens/s, step latency, `n`, traffic, CPU seconds: yes for the 17 baseline arms. **The gen2 comparison did not resolve.** Promotion stalls, pinned/VRAM footprint per arm: not in the arm files I read |
| Original workload and numerics unchanged; capacity differences disclosed | **PARTIAL** | workload and the 70 GiB tier setting identical by the arm script; numerics: see the first row |
| Independent reviewer checks ownership transitions and evidence after each structural milestone | **PARTIAL** | **six reviews exist**, each finding something its author could not see, each bounded by a "not checked" list: the lease-protocol review (severities, in `LEASE_PROTOCOL.md`), `PROMOTION_ASYNC_REVIEW.md`, `PROMOTION_ASYNC_REVIEW_2.md`, `LEASE_MODEL_REVIEW.md`, `PER_ROW_TRANSFER_REVIEW.md` (the Task 6 design) and `PER_ROW_PRECHECK_REVIEW.md` (an independent recomputation of the Task 6 precheck arithmetic, which reproduced it and corrected the per-row-over-two-phase comparison). **They review designs, a model and one calculation.** No review of the *implemented* Task 4 milestone (two banks) is recorded, and this report is unreviewed |
| Final report names accepted, rejected, deferred work; not labelled fully asynchronous while Task 8 blocks | **THIS DOCUMENT** | sections 0-3; not complete until reviewed and updated as the open items close |

## 6. How much each headline number rests on

| number | `n` | monitoring | source |
|---|---|---|---|
| mirrors off 2.903-2.919 tok/s | 5 | unmonitored (no boundary samples) | `PIPELINE_BASELINE.md` §3.1 |
| mirrors on 3.905-3.956 | 4 | unmonitored | same |
| older code 3.798 | **1** | **unmonitored**; four further old arms disturbed; **the attempt to raise it failed** (`task1e`) | §5 there; `task1e-RESULT.txt` |
| mirror effect 1.347x-1.351x | 5 and 4 | as above | §3.2 there |
| new over old +3.2 % to +3.5 % | old **1** | as above; delta is two behavioural commits; **gen2 pair 1.047 is not a result** (n=1 per cell, contended) | §3.2 there; `task1e-RESULT.txt` |
| overlap: 1,888/1,888 multi-row requests, 0 chain violations | one workload, seven arms | schema-2 traces from arms on the same box | `overlap_timeline.py` |
| packer covers 49-54 % of the read window with mirrors on; 104-112 of 4,332 rows wait on it | same | same | same |
| Task 6: saving 20.1 ms/step at random order with **measured** lane counts (registered 19.2; A1 corrected 17.8), 87 % hit lanes, per-row over two-phase +4.93 ms best / -15.67 ms random order | **model**, one workload (n_eff = 1; per-session range 5.3-8.7 %); arithmetic independently reproduced | A2, c = 1.055 ms, launch 1.0 ms assumed; A1 now bounded (10.8-63.6 hit lanes per step in read layers, two-phase about 4 % at the worst placement) | `per_row_precheck_result.txt`, `PER_ROW_PRECHECK_REVIEW.md` |
| trace-on overhead 3.7-5.5 % of host CPU | 4 runs x 25 reps | contended box, tmpfs, no O_DIRECT or GPU | plan Task 1 annotation |
| whole-row 1.8x slower at one row | 144 batches, two runs | conditions not recorded (`SCHEDULING.md`) | `SCHEDULING.md` |
| nvme2 0.00 % of expert bytes with mirrors | **one pair of arms** | service-attributed | §19 |
| native suites 281 passed | one recorded run | not rerun here | plan annotation |
| wrap fix: four wrap and two lap cases fail before, pass after | 10-case test, run once each way | observed by the instrumentation agent, as reported in the commit messages | `85cdbf9382`, `a79ce2eef5` |
| registration cost 59.9-106.6 ms at 256 MiB | median of 7, one run each | CPU only, box not quiet | `TOPOLOGY.md` §7.7 |

## 7. In flight, uncommitted, and who to ask

- **Uncommitted at the time of writing:** a modification of `LEASE_PROTOCOL.md` and of `per_row_precheck_result.txt` by other sessions; findings resting on them are cited only where they are committed.
- **Needed from the owner:** whether to run the packing-worker comparison (Task 4's highest-value item); whether to register calibrated gates for a next old-versus-new series, and whether quiet-box measurement or a contention-tolerant paired design is the right
  instrument for a roughly 3 % effect (open in `task1e-RESULT.txt`); the decisions listed in `LEASE_PROTOCOL.md` §19 and `PROMOTION_ASYNC.md`; a gen3 entry in `clean-reference.json` before any further arm.
- **Not run by design (GPU or quiet box needed):** Task 1 device intervals and the trace-off-versus-uninstrumented arm; the `ld.global.nc` experiment; Task 3 storage/SM/simultaneous; Task 4 GPU parity; every Task 5-8 test that is not CPU-only.

## 8. Change log

- 2026-09-21: first version, at `ec64c07999`. Written by a single session; not reviewed.
- 2026-09-21: D1-D7 corrected severities added (independent review, attributed); sequence wrap given its execution result, two signatures and both rate assumptions; `lanes <= 8` recorded as a latent bug with a missing attach-time check, DSV4.1 being top-6; OPEN 13 settled.
- 2026-09-21: updated at `a3b2b317e1`. Three fixes recorded with their evidence (wrap `85cdbf9382`, attach `8c78749b35`, drive resolution `c87b8dc181`); Task 4's timeline half of the gate met, and the packer identified as approaching the critical path; Task 6's mechanism
  redirected by a pre-registered model-based precheck (task stays OPEN); trace-on overhead measured; the `task1e` series recorded as UNRESOLVED with its calibration findings; `ld.global.nc` experiment pre-registered, not run; Tasks 5 and 8 reviewed (four reviews recorded);
  negatives ledger and defects register extended; `HEAD`'s `python/` tree noted as absent from the generation manifest. **No status improved beyond its evidence: 6 of 10 tasks remain PARTIAL, DESIGN ONLY or OPEN, and the report itself is still unreviewed.**
- 2026-09-21: Task 6 folded with the independent recomputation (R1-R5: the per-row-over-two-phase increment is +5.09 ms best order and -14.35 ms random order, not "2.55 ms"; A1 bounded from the trace; ordering is per-drive FIFO; n_eff = 1; denominators); the drive-resolution offset corrected to per-drive and growable (`e01639eb07`); reviews counted as six. **The report itself is still unreviewed.**
