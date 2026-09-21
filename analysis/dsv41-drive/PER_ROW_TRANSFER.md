# Starting GPU transfers before all reads finish: design and precheck (Task 6)

Status: design only, plus one measurement computed from existing traces. No code was written or run, the GPU was
not used, nothing under `python/` was touched. Written against `dsv41` at `22dae687f7`. Another session is editing
the native service, so cite symbols, not line numbers.

Plan reference: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, "Task 6", "Task 5", "Global constraints".
Foundations: `LEASE_PROTOCOL.md` (Task 5), `PROMOTION_ASYNC.md` (Task 8), `lease_model.py`. Precheck: `PER_ROW_PRECHECK_PREREG.txt`
(sha256 `0c145def...6e15`), `per_row_precheck.py` (sha256 `e81b8b60...4561`), `per_row_precheck_result.txt`.

Legend. **[E]** exists in the tree; the citation names a symbol. **[P]** proposed here; it does not exist. **[M]** measured
by the precheck from existing traces. **[A]** an assumption I could not check. **[OPEN n]** could not determine.
**[REQ n]** a request to another owner (section 9).

---

## 0. What this document concludes

**Read this paragraph and stop only if you must.** Per-row transfer, the mechanism the plan specifies, is **not
justified over a hits-then-rest two-phase copy**. About 87% of the modelled saving is RAM-hit lanes being copied while
the NVMe read runs; only about 13% needs miss rows to finish at different times, and that part (what per-row adds over
two-phase) is at most **5.06 ms per decode step, 2.0%, and 2.55 ms, 1.0%, if lane order is random**: below the
~1.5% the plan's own measurement design can resolve. **But neither saving is available today.** The device learns of
readiness from one word, `demand_done`, which the service stores only after `read()` has returned and every row is
packed (`pump_demand` after `handle_demand`; section 3.1). No per-lane or per-phase early signal exists. The 87% is an
upper bound for a mechanism that **first requires a new early, device-visible, generation-tagged readiness word per hit
lane** (Task 5's `RowResult`, published at *reservation* instead of at the end), which is also what makes the early
copy safe (section 3.1). The comparison to make is therefore two-phase = one early publication of the hit lanes; per-row
= that plus a per-row publication hook inside `read()`, a per-lane wait chain, a partial terminal mask and lane ordering.

**One workload.** The precheck ran on seven traced arms that are one request stream replayed (20,800 requests, 7,211
demands in each). Every statement of the form "the saving is X" below is about this stream and its routing pattern, not
about EXL3 decode in general. A different cache size or routing skew changes the hit-lane count per read layer (A1) and
`Sigma(m-1)` (the per-row-specific part).

1. **The saving is conditional on a blocking dependency that is unbudgeted: per-lane, time-staged publication ([REQ 1]).**
   Every figure below, the 7.5-15% and the 87%, requires that the service publish a lane's readiness, and take its lease,
   *before* the request completes. That does not exist today (section 3.1). **[REQ 1] is a prerequisite for the two-phase
   mechanism V1 exactly as much as for per-row V2**; V1's advantage over V2 is that it needs one early publication instead
   of one per row, not that it needs none. A cost estimate for V1 that books the protocol change as someone else's work is
   wrong. Cost line (section 9): the *timing* change in `serve()` is small (a few stores per hit lane inside a critical
   section that already exists); the *words* it publishes, their generations, the lease counters and retirement are
   Task 5, which is designed but not implemented. So V1's real cost is "Task 5, plus a small reordering", and it is
   unknown until Task 5 lands.
2. **The ceiling for THIS configuration is 19.2 ms/step (random order) to 38.6 ms/step (best order), 7.5% to 15% of a
   257.5 ms mirrors-on step** [M, section 1]. DSV41_REFERENCE 18.3's 20.8 ms (<= 5.3%) was computed for a different
   configuration (the older 391 ms/step arm, G = 121.2, 18.8 RAM misses per step, mirrors off). Please plan against these
   figures and cite both; my BEST figure is uncorrected for 18.3's 74.5% alignment haircut, which is why it is twice
   18.3's. (I first read 221 VRAM misses per step from the first two `graph_step` lines and told the team lead so; over
   all 495 steps it averages **131 [post-hoc]**, which reconciles with 18.2's G = 121 rather than contradicting it.)
3. **The result rests on A1** (hit lanes are spread evenly over the 40 layers), because the trace does not record lanes per
   layer. The SUPPORT bar still holds down to about **37% of the modelled hit lanes, about 0.8 hit lanes per read layer**;
   it fails only if hit lanes sit almost entirely in layers that read nothing. **[REQ 4]** asks for one instrumentation word
   that removes the assumption.
4. **Registered outcome: SUPPORT, and SPREAD-IRRELEVANT.** The pre-registered reject test asked whether miss-row spread was
   large enough to matter. It was, but it was aimed at the wrong quantity: the saving does not come from the miss rows.
   The classification and the redirect both came from the same run, and the redirect is the result.
5. **The spread that exists is Task 4's artefact, not the drives'.** Miss rows become ready about 2.7 ms apart (schema 2,
   traced) because `pack_one` packs one row per loop turn on one thread, and that exceeds `c` (about 1.1 ms). So Task 6's
   premise, "later reads finish later", was partly a misattribution of the packing staircase. **A counter-intuitive
   consequence: the better Task 4 gets (parallel packing, Task 7's raw rows), the smaller the miss-row spread and the
   smaller the 13%, so per-row becomes less attractive, not more.** The hit-lane part does not move.
6. **Head-of-line cannot make per-row slower than batched in overlap terms** (section 2, closed form), only by launch and
   poll cost. It zeroes the miss-row part when the slowest row is first and halves it on average for random lane order.
   In this corpus pack order equalled ordinal order in 100% of 1,888 requests with m >= 2 [post-hoc]. **That is a property
   of the packer, not of the storage**: 100% is the shape a lowest-ordinal tie-break produces, and I did not test it with the
   tie-break changed. Do not read it as "the drives complete in ordinal order". Drive tails would expose it.
7. **A4 as written is not required, and here is what replaces it.** `LEASE_PROTOCOL` A4 says lane `j`'s acknowledgement
   precedes the GPU's wait on lane `j+1`. Two facts make it hold, and the second is the one that matters: (a) stream order
   on one linear captured chain (a checkable graph shape); (b) **reservation is all-or-nothing, so no lane's readiness
   waits on any lease of its own request**. Section 5.2. The 9.2 cycle cannot form: no second stream, no event, no
   `wait_stream`.
8. **I retract my claim that Task 6 must overturn DECIDE 3** (section 9, REQ 3), and I state why: `serve()` reserves all
   slots in one `mutex_` pass, `wanted` excludes every routed and needed expert from victim choice for the whole call,
   and `read()` blocks the single service thread, so nothing evicts in that tier between reservation and return.
   Lease-at-publication is safe for hit lanes as long as they are published before `serve()` returns.
9. **Tracing was on in every trace used, and all seven are schema 2; no schema-3 graph-decode trace exists.** Traced
   request durations are stretched, so the denominator is the untraced step time.

---

## 1. The ceiling, before any mechanism

### 1.1 What was measured, and its limits

`per_row_precheck.py` was frozen and committed (`22dae687f7`) before it read any trace. It reads the seven traced arms in
`task1-results/` on divix01 (read-only, `taskset -c 0-63`, one thread). **Every one is `RAM_MISS_TRACE_SCHEMA` 2**; there
is no schema-3 graph-decode trace. The row-ready stamp used, `row_pack_ns[].end`, means the same in schema 2 and 3
(`record_ram_miss_requests`, `exl3_stream_trace.py`); no schema-1 span was used. **The seven files are one deterministic
request stream (20,800 requests, 7,211 demands in each) replayed with different timing**, so their agreement is
agreement of the run and the drives, not seven workloads.

Model: per request that reads `m >= 1` rows, `k` lanes (`k = round(vram_miss/40)` for its graph step, capped at 6 **[A1]**),
`h = k - m` RAM-hit lanes ready at the service's `reserved` stamp **[A2: this assumes an early readiness signal that does not exist today; section 3.1]**, miss row `j` ready at its `row_pack_ns[j].end`,
lane copy time `c = 1.055 ms` (DSV41_REFERENCE 18.2's measured gather per row). Batched time `T_b = done + k*c`; per-row
time is the chain `e = max(e, ready) + c`. Full definition in the pre-registration.

### 1.2 Result [M] (three mirrors-ON arms; they agree to 0.1 ms; step time 257.5 ms from untraced 3.884 tok/s)

| Quantity, ms per decode step | c = 0.55 | **c = 1.055** | c = 1.6 | share of 257.5 ms step |
|---|---:|---:|---:|---:|
| BEST: hits first, then miss rows in pack order, free launches | 20.2 | **38.6** | 54.9 | **15.0%** |
| RANDOM lane order (64 permutations per request) | 10.1 | **19.2** | 27.7 | **7.5%** |
| RANDOM, miss rows only (no hit lanes) | 1.35 | **2.55** | 3.8 | 1.0% |
| GENEROUS (6 lanes, best order): the registered reject test | 39.7 | 75.4 | 86.9 | 29% |

Registered outcome: **SUPPORT** (RANDOM minus 1 ms of launch cost = 7.1% of step, bar 3%, all three arms) and
**SPREAD-IRRELEVANT** (miss-only / RANDOM = 0.133 < 0.25 in all three arms). Mirrors-OFF (step 344 ms) gives the same
classes. The Task 6 reject test therefore did **not** reject.

`Sigma(m-1)` over the arm's requests is `1407 + 2*352 + 3*70 + 4*11 + 5*2 = 2375` rows over 495 steps, so the
miss-row-only best case is `2375 * 1.055 / 495 = 5.06 ms/step` (2.0%). This is arithmetic on the registered output
(the histogram is `m: {1: 5297, 2: 1407, 3: 352, 4: 70, 5: 11, 6: 2}`), not a separately registered statistic. Its
random-order counterpart, 2.55, is the registered figure and is half, as the closed form of section 2 says it should be.

### 1.3 Relation to DSV41_REFERENCE 18.3

18.3 bounds "gather a layer's RAM-resident missed rows while its NVMe read runs" at **20.8 ms/step, <= 5.3%**, on a
74.5% per-layer alignment, from the older 391 ms/step arm (G = 121.2 VRAM misses, 18.8 RAM misses per step, nvme2 only).
Mine is 19.2 (random) to 38.6 (best) on a 257.5 ms step. The RANDOM figure is within 8% of 18.3's; the BEST figure is
about twice it, because BEST assumes every hit lane in a read layer is copied inside the read wait (94% of requests wait at
least `h*c` [M]; read-wait p50 5.3 ms and p90 8.9 ms against `h*c` of about 2.3 ms are post-hoc and unregistered,
`per_row_precheck_posthoc.py`) and does not apply 18.3's
alignment haircut. **A reader should take 18.3's 20.8 ms as the planning figure and mine as the range it sits in.** The
trace's per-step averages (vram_miss 131 and ram_miss 19.1 are post-hoc; 14.4 read requests per step is 7,139 / 495) are consistent with 18.2's.

Consistency check that used no fitting: modelled gather `131 * 1.055 = 138 ms` + measured read-wait `93 ms/step` + about
17 ms compute (18.2's figure) = 248 ms against a measured 257.5 ms step. The model's `c` and lane count are not
contradicted by the step budget.

### 1.4 What carries the result, and what would break it

- **A2 (an early hit-lane signal exists) is not true of the current tree** (section 3.1); the numbers are an upper bound for the mechanism that adds it.
- **A1 (hit lanes spread evenly over layers) carries it.** The trace records `vram_miss` per step and `layer_ram_rows`
  (RAM rows per layer), not lanes per layer. Robustness: the SUPPORT bar still holds down to about 37% of the modelled
  hit lanes (RANDOM = 2.55 miss-only + 16.7 hit part; the bar needs 8.7 ms), i.e. about 0.8 hit lanes per read layer.
  It fails only if hit lanes sit almost entirely in layers that do not read, which is possible in principle (26
  non-read layers can hold 156 lanes, more than the 112 hit lanes per step) and is not excluded by any data I have.
  **[REQ 4]** asks for one instrumentation word that removes A1.
- A2: a hit lane is ready when the service has reserved the request. This needs the service to publish hit lanes before
  it reads; it does not today (section 6).
- `c = 1.055` is 18.2's number, from a node-mode trace of an older tree; it was not measured for this path in isolation.
- Tracing was on in every trace used, which stretches request durations. The untraced denominator keeps `T_step` honest.
- The precheck models no lease latency, poll latency or per-stage launch gap. Those are the costs of the design (section 8).

---

## 2. Head-of-line, in closed form

Take lanes consumed in a fixed order `1..k` (position `j`), lane `j` ready at `r_j`, uniform copy time `c`, `M = max r_j`.

```text
batched:   T_b = M + k*c
fixed:     T_p = max over j of ( r_j + (k - j + 1)*c )          (the chain e = max(e, r_j) + c, unrolled)
saving  S = T_b - T_p = min over j of [ (M - r_j) + (j - 1)*c ]
```

Consequences.

- **`S >= 0` always.** Every term is nonnegative. Per-row and two-phase can lose only launch and poll cost to the
  batched copy, never overlap. (It assumes the per-lane copy takes no longer than its share of the batched one; see the
  `c` sensitivity in 1.2 and section 6.3.)
- **`S <= (j* - 1)*c` where `j*` is the position of the slowest lane.** If the slowest lane is first, `S = 0`.
- **One miss lane among `k`, at a uniformly random position:** `E[S] = (k - 1)*c / 2`. Half of the ideal `(k - 1)*c` is lost.
  The registered RANDOM/BEST ratio of the miss-only figures (2.55 / 5.06) is this, to rounding.
- **Hits first, misses in completion order** gives `S = (k - 1)*c` when the read outlasts the earlier copies. That is the
  point of the order array (section 4.2): it takes lane order out of the hands of routing.

**When HOL dominates the benefit.** For the *miss-row part*: whenever the read completes out of lane order by more
than `c`. In the corpus it did not: **pack order equals ordinal order in 1,888 of 1,888 requests with m >= 2 in each
of three arms [M, post-hoc, `per_row_precheck_posthoc2.py`]**, and the lane order among miss lanes equals the ordinal order
by construction (both are first appearance in the routed list: `plan_graph_routes`, and `serve()` builds `wanted` from
`protect` first, which `Exl3MoEMethod._apply_graph` fills from `topk_ids`). Two cautions. First, **treat this as a
property of the packer, not of the drives, until shown to survive a changed tie-break (not done)**: `pack_one` packs the *lowest ordinal among rows that are `Ready`*, and rows
that complete in one reaped batch are all `Ready` together. Second, `MIRROR_ROWS.md` measures a per-row tail even on the mirrored arm (p50 4.164 ms, p99 7.387 ms, so a row can
arrive about 3 ms, three copy times, after the median), which is exactly the slow-first-row case where `S` collapses to
the hit part. (Its nvme4 first-touch 18.3 ms is a warming outlier by its own account and is not used.)
For the *hit part*, HOL does not apply, because hit lanes are ready at reservation; it applies only if the order hint
is wrong (a lane that looked like a hit at post time turns out to need a read; section 4.2).

---

## 3. What exists today [E], with corrections found while reading

The device chain per streamed layer, on one stream, in `Exl3RamMissRowBackend` (`srt/layers/moe/exl3_ram_miss.py`) over
`PinnedTierRowBackend.post` (`expert_row_plan.py`):

| Step | Symbol | What it does [E] |
|---|---|---|
| 1 | `exl3_ram_miss_post_kernel` (`exl3_ram_miss.cuh`) | 1 block, thread 0. `need` = planned experts whose slot-map entry is `< 0` (first `min(count, kMaxIds=8)` lanes). `protect` = routed experts, then `need`. `armed = need_count > 0 || advise != 0`. Writes the demand record, releases `demand_head`. |
| 2 | `exl3_ram_miss_wait_kernel` | 1 block, thread 0. Polls `demand_done` with `__nanosleep(256)` until `reached(done, seq)` or `timeout_ns` (`SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`, default 2000, `environ.py`). Then translates every planned expert through `slot_map` into `host_rows`, adds misses to `ram_miss`, writes `keep`. **On timeout it raises fatal, sets `keep = 0` and still runs the translate loop.** |
| 3 | `copy_expert_row_segments_gpu_kernel` (`expert_cache_transfer.cuh`) | **Grid 8 x 256 threads, `__launch_bounds__(256, 1)`**, one launch for all lanes; the row count is read from a device tensor (`count[0]`). With `count <= 64` warps, each row gets a contiguous range of warps; with `count == 1`, that one row gets all 64. `Exl3RamMissRowBackend` inherits `post`, which passes the full `plan.count`, so **the copy runs after a timeout too** (D1 of `LEASE_PROTOCOL`). |
| 4 | fused MoE (`exl3_fused_moe.py`, `route_tables`) | `keep` scales every route weight and, when 0, empties `expert_count`. |

`resolve` and `copy_residual` are no-ops for this backend. Routes: `Exl3MoEMethod._apply_graph`
(`srt/layers/quantization/exl3.py`) copies the live `topk_ids` into `backend.routes` in-graph, so the post kernel's
`protect` includes every routed expert (first 8).

The service, `RamTier::serve` (`exl3_ram_miss_host.cpp`), reserves **all** slots for `wanted = protect + need` in one
`mutex_` pass before any I/O, then calls `RowReader::read(step = kBounceRows = 8, ...)`, which is **one batch**: every
missing row's extents are queued at once, each as one `io_uring_prep_read` of its whole aligned extent (`admit_batch`,
`refill`). `pack_one` packs one row per loop turn and sets `packed[ordinal]`. **Nothing reaches the device until `read()`
returns**: `serve()` then sets `kReady`, calls `publish_map` for every row, and only `handle_demand` stores
`demand_done`. The per-row stamps (`row_pack_ns`, `extent_cqe_ns`) exist in `StageRecord`; the per-row publication does
not. The plan's Task 4 says the same ("Partial host row completion is not yet permission for an early GPU read").

### 3.1 Is there an early device-visible readiness signal today? No, and the modelled saving needs one

**Are hit lanes resolved at plan time, before any read is submitted?** Yes, in two different places, and only one is
authoritative. On the **device**, the post kernel decides `need` from `slot_map` (`ld_volatile`, first `min(count, 8)`
planned lanes) at post time: a racy hint, with no ownership. On the **host**, `RamTier::serve` resolves it authoritatively
under `mutex_` **before any read is submitted**: `tier.expert_slot[expert] >= 0` is a hit (its LRU stamp is touched), otherwise
the expert joins `missing` and gets a `kLoading` slot. That resolution is **host-side only**: it lives in C++-private
`Tier` state and reaches the device only as a side effect of the map after `read()`. So the information the early signal
must carry already exists, at the right moment, in the right thread; the missing piece is publishing it, which is why the
timing change is small and the words it publishes are the real cost.

Checked against the source, because the precheck's A2 (hit lanes ready at the `reserved` stamp) silently assumes it:

- The service stores `kDemandDone` in exactly one place, `RamTier::pump_demand`, **after** `handle_demand` returns, i.e.
  after `serve()` has finished `RowReader::read()`, published every row and set the record status. (The other
  `kDemandDone` uses are the open-time seed and the simulated device's wait.) The advisory word `kAdviseDone` is a different ring.
- `exl3_ram_miss_wait_kernel` polls **only** that word (`ld_acquire_sys(page + kDemandDone)`), then reads the record status,
  then translates through `slot_map`. It has no per-lane or per-phase input.
- `slot_map` does not help: `publish_map` for newly read rows happens after `read()` too, and for a **hit** the map entry
  already exists but carries no ownership: nothing tells the device the service will not evict it. That is the D2 race
  (`LEASE_PROTOCOL` 10.2). An advisory in progress on the same tier evicts non-protected `kReady` rows and protects only
  its own ids, so a device that gathered hits straight from the map during a read would be relying on the temporal
  exclusion that early copying removes.

So the early copy of hit lanes needs **a new signal for both reasons at once**: it is what lets the GPU start (performance)
and it is what grants the lease that keeps the slot immutable (safety). The precheck's numbers bound the *opportunity*; they
do not show it is reachable without that signal. What the signal is, concretely, and what it costs:

| | Two-phase (V1) | Per-row (V2) |
|---|---|---|
| New mapped word | hit lanes' `RowResult.ready` (Task 5's word, tagged with `G56`), published in the reservation critical section of `serve()`, before `read()` | the same, plus one `RowResult.ready` per miss row |
| New service code | publish hit lanes early | that, plus a callback from `pack_one` into publication (a per-row hook inside a blocking `read()`) |
| New device code | stage wait polls the hit lanes' words; one more copy stage | `S` stage waits and a finalize with a partial mask |
| Ordering / HOL problem | none among hit lanes (all ready at reservation); a wrong hint only delays | yes (section 2) |
| Partial terminal mask | needed only if stage 2 fails after stage 1 copied (two lanes groups) | needed for every lane |

Honest framing: not "two-phase gets 87% nearly free" but **two-phase gets about 87% of a modelled 7.5-15% for one early
publication of the hit lanes, and per-row gets the remaining 13% for per-row publication, per-lane waits and ordering**. The
gap in mechanism narrows; the gap in yield does not (the 13% is below what the gate can resolve). Task 5 as designed publishes
every lane after all rows land (whole-request scope), so *when* the words are published is the Task 6 change, not the words.

**Stale or wrong claims in nearby documents, found by reading (each also recorded for its owner):**

- `LEASE_PROTOCOL` D7 ("no `<= kMaxIds` check at attach") is **stale**: `Exl3RamMissService.attach` refuses
  `graph_gather_rows > MAX_IDS` (`8c78749b35`, landed after that design's snapshot). Task 6's lane count `C <= 8` rests on it.
- `LEASE_PROTOCOL` OPEN 12 is **answered for the router-miss producer only**: `_apply_graph` puts the live routed experts
  into `protect`, so planned lanes are within it. A plan built by `plan_candidates` (a prefetch producer) may name
  experts that are not routed, and the design must not rely on the subset property for those.
- The `Exl3RamMissRowBackend` docstring says "`_apply_graph` copies them in before each gather" without naming the
  file; the function is in `srt/layers/quantization/exl3.py`, not `exl3_ram_miss.py`.
- Plan Task 6, "Add capture-stable lane views/counts... and a row-copy entry point": **the row-copy entry point already
  exists.** `copy_expert_row_segments_gpu(segments, source_rows, destination_slots, count)` accepts any 1-D contiguous
  int64 / int32 / int32 tensors of equal length with `count` a single int32 (`_validate_plan`), and a 1-element slice
  of a preallocated tensor satisfies all of that. What is new is per-stage *counts* and *views*, not a kernel.
- `LEASE_PROTOCOL` section 18.3 says per-lane copy, the finalize kernel and a partial terminal mask are **not modelled**.
  Everything in section 5.4-5.5 below that relies on them is therefore unchecked by the model.
- `hold_ordinal` (`ReadFault`; `_fault_tensor` in `ops/moe/exl3_ram_miss.py`) withholds a row's completions "until every
  other row is done". It cannot, on its own, prove that another row's *GPU transfer* ended before the held row's I/O did:
  it releases when the other rows are packed, before any GPU could have copied them. Section 5.3 needs a new gate.

---

## 4. The mechanism

### 4.1 One structure covering batched, two-phase and per-row: stages

A request has `k` lanes (`k <= C`, `C = graph_gather_rows <= 8`). The captured graph holds a fixed number `S` of
**stages**; stage `s` covers a device-determined range of *consumption positions* `[lo_s, hi_s)`, possibly empty. Each
stage is three kernels, plus one finalize kernel per layer:

```text
post                                   // [E] extended: writes ord[], h, k, the request deadline, stage ranges
for s in 0 .. S-1:                     // unrolled at capture in Python, like the plan's sketch; nothing runs in replay
    W_s   wait for the stage's lanes    // reads only mapped lane readiness; writes rows_s[], slots_s[], go[s]
    C_s   copy_expert_row_segments_gpu(segments, rows_s, slots_s, go[s:s+1])     // [E] kernel, unchanged
    A_s   acknowledge lanes copied     // Task 5's ack kernel, restricted to this stage's lanes
F         finalize: success check, terminal mask, keep
fused_moe(keep)
```

`rows_s`, `slots_s` and `go[s:s+1]` are 1-D views of preallocated device tensors, so the copy call needs no new kernel and
no allocation. `slots_s[q]` is the destination scratch row of that lane (`plan.slots[ord[lo_s + q]]`), so **reordering
lanes changes only when a lane is copied, not where it lands**; the route remap that the fused MoE reads is untouched.

| Variant | Stages `S` | Stage ranges | Kernels per layer |
|---|---:|---|---:|
| **V0'** batched, lease mode (Task 5 as designed) | 1 | `[0, k)` | 3 + 1 |
| **V1** two-phase | 2 | `[0, h)` hit lanes, `[h, k)` the rest | 6 + 1 |
| **V2** per-row (the plan's mechanism) | `C` (6 at top-6) | `[j, j+1)` | 18 + 1 |

A request with no reads and all hits is V1 with stage 2 empty: its stage-1 range is all `k` lanes, so **the batched path
survives as a stage covering every lane, under the same lease, acknowledgement and terminal rules** (plan item 1). An
empty stage's three kernels still launch and return at once; that cost is charged to every layer (section 8).

### 4.2 The order array, and why the hint may be wrong

The post kernel already computes `need` (planned experts whose host slot-map entry is `< 0`). **[P]** it also writes
`ord[0..k)`: lane indices with the planned experts *not* in `need` (hits) first, each group in lane order, and `h` (the
hit count). It writes `k` and the request deadline (5.4). `ord` and `h` are a **performance hint**, read from
`slot_map` with `ld_volatile` exactly as `need` is; they carry no safety. Correctness always comes from the per-lane
readiness word (5.5). If a lane the hint called a hit turns out to need a read (`serve()` reads any `protect` expert
not assigned; that is the case its comment calls D12's race), its `RowResult` publishes later and its stage waits for it:
slower, never wrong.

### 4.3 The service side [P]

- **Hit lanes** are published (and leased) in the reservation step, before `read()`. **Miss lanes** are published at
  `pack_one`, when `packed[ordinal]` is set: `RowReader::read` gains a callback (or the tier a hook) invoked from
  `pack_one`, which under `mutex_` sets `kReady`, calls `publish_map`, and publishes the lane's `RowResult`
  (`LEASE_PROTOCOL` 6.1's five-step order, per lane).
- **`demand_done` and the record status stay as the request-level terminal signal.** Stage waits poll them too: a failed
  request must fail fast, not cost a full timeout per lane. (Today the wait reads status only after `demand_done`.)
- **Reads are submitted in consumption order.** `admit_batch` already queues extents in row order and `pack_one` takes
  the lowest ordinal among ready rows; the corpus shows that order holds. V1 does not need more.
- Nothing in the service ever waits on the GPU. Acknowledgements are polled by `retire_leases()` (`LEASE_PROTOCOL` 7.5, 16.1).

### 4.4 Cost of extra stages, all layers, counted up front

`V2 - V1` is 4 more stages, so 12 more kernels per layer, on all 40 layers: **160 more stage triples, 480 more kernel
nodes per step, whether or not any read happens** (67% of layers wait on nothing, from 18.2: 676 of 2,022 layer calls
wait on NVMe). A kernel-node gap of 2-4 microseconds **[A]; I have not measured it on this GPU** puts that at about
1-2 ms per step, before the poll round trip inside each wait. It buys at most the miss-row part, 5.06 ms per step
(best order, section 1.2), so V2 over V1 nets at best a few ms, about 1% of the step, under a 1.5% resolution.
**V2 is expected to be REJECTED**, not for HOL but because it spends launches on every layer to harvest a part that is
1-2% of the step. Section 6.2 states what would overturn that.

---

## 5. The six plan items

### 5.1 Item 1: capture-stable lane views, counts, delivery status, row-copy entry point

- **Capacity** `C = graph_gather_rows`, at most `kMaxIds = 8` [E, enforced at `attach`]. `S` and `C` are capture-time
  constants; a request's `k` and stage ranges are device data.
- **Views [P]:** `rows_s`, `slots_s` (int64 `[C]`, int32 `[C]`) and `go` (int32 `[S]`) are preallocated device tensors
  captured once; the copy for stage `s` is `copy_expert_row_segments_gpu(segments, rows_s, slots_s, go[s:s+1])`, which
  satisfies `_validate_plan` (1-D, contiguous, int64 / int32 / int32, equal length, `count.numel() == 1`).
  Torch slicing runs at capture, not in replay.
- **Delivery status [P]:** device `lane_state[C]` in `{PENDING, COPIED, ACKED, SKIPPED, FAILED}`, written only by the
  stage kernel that owns the lane. The mapped words (`RowResult`, `LaneAck`, `Terminal`) are Task 5's, unchanged.
- **Row-copy entry point:** [E] as above; the kernel's per-thread mapping already handles `count == 1`.
- **The batched path** is a stage covering all lanes (4.1). It obeys the same contract because it is the same code.

### 5.2 Item 2: no CPU completion depends on work queued behind the GPU's own wait; A4; the 9.2 trap

**Mechanism.** CPU to GPU: only the mapped, generation-tagged words (`RowResult.ready` with `G56`, plus
`demand_done` / status). GPU to CPU: only `LaneAck` and `Terminal`, which the service **polls and never waits for**
(`LEASE_PROTOCOL` I5, 16.1). No `cudaEvent`, no `cudaStreamWaitEvent`, no host callback, no second stream. The
plan's "do not substitute an event that has not been recorded for that generation" is met by having none.

**The wait-for graph.** Nodes: stage waits `W_s` (serving stream), the service thread `S`, earlier requests' acknowledgements
`A_prev` (stream-ordered before this request's post), host leases `H` (Task 8), this request's own acks `A_s`.

```text
W_s  -> S              lane readiness
S    -> I/O, packing   (no GPU dependency)
S    -> A_prev, H      only when admission finds every victim leased (deferral, all-or-nothing)
A_prev -> (earlier copies, already issued: stream order)
H    -> (promotion copy on the executor stream; must not depend on the serving stream: A3, PROMOTION_ASYNC 7.2)
A_s  -> C_s -> W_s     (completes before W_{s+1} begins: linear chain)
S    -/-> A_s          NO EDGE: reservation is all-or-nothing, so nothing in this request needs this request's leases
```

There is no path from `W_s` back to itself. The one edge that could close a cycle, `S -> A_s`, does not exist because
**admission reserves every slot the request needs in one `mutex_` pass (as `serve()` does today) or defers the whole
request**, and takes no slot after publishing any lane. That is `LEASE_PROTOCOL` A2, and it is the load-bearing
assumption of this design.

**A4, stated precisely.** `LEASE_PROTOCOL` A4 says lane `j`'s acknowledgement is issued before the GPU waits for lane
`j+1`, "so lane `j+1`'s readiness never waits on an ack of lane `>= j+1`". Two facts make it true here, and the second is
what actually matters:

1. **Stream order on a linear chain.** The stage kernels are launched on one stream during capture, so the captured graph
   has `A_s -> W_{s+1}` as a dependency edge and the chain is linear. *This must be checked, not assumed:* a test
   enumerates the captured graph's kernel-node dependencies (`cudaGraphGetEdges`) and asserts a chain; it fails if any
   stage kernel is ever captured off the serving stream.
2. **No readiness depends on this request's leases** (all-or-nothing reservation, above). Even if stream order were lost,
   no cycle would form, because `S` never waits on `A_s`.

I did not design around A4; I showed it is implied by A2 plus a checkable graph shape, and made the graph shape a test.

**The 9.2 trap.** Once a lease is held, a copy stream must not depend on the serving stream. **Here the copy is on the
serving stream and the lease is released by a kernel later on that same stream.** There is no copy stream to depend on
it. The 9.2 cycle needs a *second* actor whose lease release waits on the serving stream; this design adds none, and
requires Task 8's holder to keep its A3 obligation unchanged. One consequence to record: **hit leases are now held from
reservation to acknowledgement, which can span the whole read wait (p50 5.3 ms, p90 8.9 ms [M])**, longer than in Task 5's
whole-request protocol. That lengthens the window in which an eager `assign`/`release` or a promotion eviction is
refused (`LEASE_PROTOCOL` section 8, R5) and makes Task 8's R4 bound ("hold time bounded by the copy") depend on the
graph, not on a copy.

**Pause and the wait.** `Exl3RamMissService.before_host_use` calls `torch.cuda.current_stream().synchronize()` and only then
`host.pause(...)` [E], so a spinning stage wait is released by a service that is still running. This ordering must not be
reversed by any change; the test is a `before_host_use` while a stage is waiting on a row the held reader has not yet
delivered.

### 5.3 Item 3: early readiness with a deliberately delayed later read

Claim to prove: **row A's transfer ends before row B's I/O completes, and the fused MoE starts only after all required
transfers.** The clock problem: GPU `%globaltimer` and the host's `CLOCK_MONOTONIC` are different clocks and the trace
code says so ("Nothing here compares a GPU clock with the host's"). So the test proves the ordering **causally**, with no
cross-clock comparison.

- **Hook [P, test-only]:** `hold_until_ack_lane` on the reader: row B's completions are withheld until the service's
  `retire_leases()` (or the simulated device's `sim_ack`) has observed lane A's `LaneAck`. If the GPU cannot copy A
  before B is released, the test deadlocks and the hang guard fires; if it passes, A's copy and acknowledgement happened
  while B's read was still outstanding. The existing `hold_ordinal` cannot do this (section 3).
- **Fused starts after all transfers:** a checker kernel in place of the fused MoE reads every destination row and the
  `keep` word; with row B's release delayed past A's ack, the checker must still see B's bytes and `keep == 1`, and with
  B failed it must see `keep == 0` and *no* fused output. Stream order (`F` before `fused_moe`, `F` after every `A_s`) is
  the mechanism; the test is what would catch a graph edge lost.
- **CPU-only variant** with the simulated device (`sim_post`, `sim_wait`, and the `sim_ack` / lane request that
  `LEASE_PROTOCOL` 18.1 adds), which is where most of the matrix in 5.4 belongs.

### 5.4 Item 4: the case matrix, and one timeout budget

| Case | Behaviour designed | Test (CPU unless marked GPU) |
|---|---|---|
| **Inactive lanes** (`i >= k`, empty stage) | stage range empty, `W_s` returns with `go[s] = 0`, `C_s` copies nothing, `A_s` acknowledges nothing (I4) | empty stage leaves `LaneAck` words zero; 0 source bytes read (poisoned slab) |
| **Arbitrary completion order** | service publishes each miss row at its own pack; stages consume in `ord` order; a later-ordinal row that is ready waits behind an earlier slow one | `reverse_cqes`, `hold_ordinal`; assert `S >= 0` (section 2) never negative beyond launch cost |
| **RAM hits** | leased and published at reservation; first stage | hit-only request; lease retires by ack; poisoned-slot check |
| **Repeated overwrite / replay** | request slots reused only after full retirement (`LEASE_PROTOCOL` 11.4); `RowResult` words carry `G56` | more than `16 * R` replays through a generation wrap; a stale `ready` from `G - 2^32` rejected |
| **Full cache pressure** | reservation all-or-nothing; if every victim is leased, the whole request defers; hit leases held to ack | tier with exactly `k` unleased victims; deferral not failure; proceeds after ack |
| **Fault after partial delivery** | copied stages acknowledge; the failing stage sets `req_failed`; later stages skip at once (`go = 0`); `F` publishes a terminal mask of every lane not acknowledged and sets `keep = 0`; fatal raised | lanes `< j` copied and acked, lane `j` faulted: assert the mask names exactly lanes `>= j` (and a ready-but-skipped later lane); the model does **not** cover this |
| **Stop during an SM transfer** | a running copy kernel is finite and cannot be interrupted; `W_s` polls `Header.shutdown` (Task 5's D4 fix) so waits exit within one poll; committed copies finish and acknowledge; quarantine rules of `LEASE_PROTOCOL` 14 unchanged | GPU manual: stop during a large copy; no slab freed before device completion |
| **Output parity** | byte-exact against V0 and against today's path | destination rows byte-equal; fused output bitwise equal on fixed routes and seeds; eager and graph |

**One request timeout budget, not `timeout * lanes`.** The post kernel computes `deadline = globaltimer() + timeout_ns`
once and stores it in a device word (64-bit, so a new tensor or two `state` words: `state` is `int32[]` [E]). Every `W_s`
compares `globaltimer()` to that absolute deadline, not to its own start plus a timeout. Whole-request budget = `timeout_ns`
from the post, shared by every wait and by the copy time between them (copy is milliseconds against a 2,000 ms default;
a test that shortens the timeout must account for it). Once any stage times out or sees `FAILED`, it sets `req_failed`;
every later stage's `W_s` returns at once with `go = 0`, so a failure costs one timeout, not one per stage. Test:
delay lane 0 by 0.9 x timeout and lane 1 by 0.9 x timeout; the request must fail at the timeout from post, not at 1.8 x.

### 5.5 Item 5: gating every copy; the final success check

`W_s` sets `go[s]` to the number of lanes in the stage **only if all of these hold, else 0** (fail closed, I4):

1. `!req_failed && page fatal == 0 && Header.shutdown == 0 && now < deadline`;
2. for every lane in the stage: `RowResult[idx][lane].ready` acquired with `gen == G` and tag `READY`
   (`ld.acquire.sys.u64`), **`RowResult.expert == planned[lane]`** (I2), `0 <= host_slot < capacity`, and the seqlock
   re-read unchanged. `READY` is published only after the lease increment (`LEASE_PROTOCOL` 6.1), so *readiness valid
   implies source lease acquired* by construction; the device never reads `slot_map` for these lanes (D2).

`C_s` receives `go[s]` as its count, so **a failed, timed-out, inactive or cancelled lane reads no source bytes**: the copy
kernel returns at `active_count <= 0`. "Cancelled" is the device's own decision (a stage skipped after `req_failed`); the
service cannot cancel a leased lane (`LEASE_PROTOCOL` 13).

`A_s` acknowledges exactly the lanes with `go[s] > 0` (I4): one device word drives the copy set and the ack set, and
is zero at entry to `W_s`.

**Final request-success check.** `F`, stream-ordered after every `C_s` and `A_s`, computes
`success = !req_failed && sum(go) == k && page fatal == 0 && no VIOLATED ack`. It writes `keep[0]` and only `F` writes
`keep`; no stage kernel does. If `!success`, it publishes `Terminal` with the mask of lanes not acknowledged, raises
fatal, and sets `keep = 0`, which zeroes every route weight and `expert_count` in `route_tables`, so **compute is suppressed
even though earlier lanes already copied**. That is the answer to "setting `keep` to zero alone does not protect an
earlier gather": `keep` protects the *compute*; the per-stage `go` protects each *gather*; neither substitutes for the other.

### 5.6 Item 6: the comparison, specified so it can be falsified

**What must be true for per-row to beat the alternatives, written before anyone measures.**

- **V1 beats V0' iff** enough hit lanes sit in read layers and the read wait outlasts their copy. Predicted gain, **conditional on the early hit-lane signal of section 3.1 existing**,
  **20.8 ms/step (18.3's bound) to 38.6 ms/step (BEST)**, i.e. 8% to 15% of the step, before the costs in section 8.
  Falsified if the untraced tok/s ratio V1 / V0' has a 95% interval containing 1.0, or if the point estimate is below 3%.
- **V2 beats V1 iff** the miss-row gain exceeds the extra stage cost. With `g` the fixed cost of one stage triple
  (three dependent kernel launches plus one poll round trip), the break-even is
  `g* = (miss-row gain) / (160 stages per step)`. Best order: `5.06 ms / 160 = 32 us`; random order: `2.55 / 160 = 16 us`.
  A triple is three kernel-node gaps plus a poll, plausibly 8-14 us **[A, unmeasured]**, so V2's net is positive but
  0.1-1.5% of step time (best order 1.1-1.5%, random order 0.1-0.5%), below what an n=3 v n=3 arm design resolves. **Prediction: V2 is not distinguishable from V1;
  record V2 REJECTED** unless a measurement shows `g` under about 5 us **and** the order is best.
- **Both beat V0' only net of Task 5's own cost.** The correct baseline for Task 6 is **Task 5 in lease mode, batched
  (V0')**, not today's unleased path: lease mode arms every record with `count > 0`, so every layer, including the
  67% that read nothing, pays a service round trip (`LEASE_PROTOCOL` 15, OPEN 11). That is Task 5's gate, but Task 6's
  benefit has to be reported net of it, and it is unmeasured.

**Arms** (interleaved, untraced, n >= 3 clean per cell, unchanged cache capacity, mirrors on, the `task1e` harness and its
stationarity and inconclusive-arm rules): A0 today; A1 V0' (Task 5 batched); A2 V1; A3 V2.

**Metrics.** Mean decode tok/s (headline; interval on the ratio); step-latency p50/p95/p99 (boundary stalls dominate the
mean: 18.6's 2.1 s outlier); **graph node count** and `cudaGraphLaunch` host time per step for each arm (untraced, or
`--cuda-graph-trace=graph` per the project's Nsight notes; **node mode only for per-kernel attribution, never for step
tail or ms/token**); per-layer GPU-side stage timestamps in a separate traced run.

**Microbenchmarks** (GPU lock and crypto-c9's scheduling; not run here):

- **Kernel overhead:** a chain of `S` empty stage triples inside a captured graph, `S = 1, 2, 6`, `S` triples per layer x 40
  layers, `nsys --cuda-graph-trace=node` for the inter-kernel gap only. This gives `g`.
- **Link throughput:** the copy kernel over `count = 1` versus `count = k` rows, working set past L2 (~128 MB; the project rule:
  spread over >= 48 distinct row tensors, ~4.9 GiB) and **checked against the link's spec** (the implied bandwidth is
  the cheapest detector of an L2-resident benchmark). Also settle whether the 12 GB/s in 18.2 is the link or the 8-SM
  gather's own latency limit; if it is the latter, `c` at `count = 1` may differ from `c / k`.
- **SM contention:** with one stream and serialized replays there is none between the stage kernels and compute. It
  exists only against other streams (the prefetch puller, Task 8's executor); measure copy throughput with a
  concurrent promotion copy if Task 8 lands, and say "not applicable" until then.
- **Step latency and token rate:** the arms above. Sweep launch geometry (`kExpertTransferGridSize`) only as a separate,
  labelled variant, as the plan says.

**Pre-registration.** The predicted ranges above, the decision rules, and the analysis script are to be frozen by hash
before the first arm runs, as the four earlier pre-registrations were.

**One instrumentation prerequisite [REQ 4].** Record the request's planned lane count `k` (and `h`) in the stage record.
The device has `count` in the post kernel; the demand record has spare bytes 84..127 (`kRecArmed` is at 80). Without it, A1
stays an assumption and the V1 prediction keeps a wide interval.

---

## 6. Why the ceiling in section 1 can be believed only this far, and what would overturn section 4.4

### 6.1 The two independent reasons the per-row-specific benefit is smaller than it looks

1. **Per-row's own part is the miss rows only**, `Sigma(m-1) * c = 5.06 ms/step` best case (2.0%), because most read
   requests read one row (5,297 of 7,139) and a single miss row has nothing to overlap *with each other*.
2. **Lane order.** Under random lane order, half of even that is lost (2.55 ms, 1.0%; section 2). The order array
   restores it only if reads finish in lane order, which held in this corpus and does not hold for a drive tail.

### 6.2 What would overturn "V2 expected REJECTED"

Any one of: (a) `g` well under 5 us on this GPU (fewer, cheaper launches, or a fused wait+copy+ack per stage, which
needs the last-block-done acknowledgement that `LEASE_PROTOCOL` DECIDE 4 declined); (b) a workload with many
multi-row read requests (a smaller RAM tier or a colder cache makes `Sigma(m-1)` larger: it scales with `m`, not with
`k`); (c) measured link behaviour showing per-row copies faster than `c / k` each. None is measured. The mention in 6.3
is the alternative if V2 fails.

### 6.3 The alternative

A **readiness-aware persistent gather**: one kernel per layer whose blocks claim (lane, chunk) work items as lanes become
ready, with a last-block-done acknowledgement per lane. It has no head-of-line waiting and no extra launches, at the cost
of a larger correctness surface (no co-residency requirement, but a new counting protocol that `lease_model.py` does not
model). The plan lists it as a Task 9 item ("readiness-aware gather"); this document does not design it.

---

## 7. Interactions with Task 5 and Task 8, and what changes in the existing behaviour

- **Task 5 is the foundation and is not implemented.** Everything here assumes its `RowResult` / `LaneAck` / `Terminal`
  words, generations, leases and retirement. If Task 5 changes, section 5 changes.
- **A failed demand today "publishes nothing unless every row landed"** (the comment above `serve()`, which arrived in
  `ddcb0d55ff`, "make a row unpublishable unless its bytes were read"; pinned by
  `test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one` in `test_exl3_ram_miss_split.py`: "the tier releases
  those slots and publishes nothing"). **This is a behaviour change in its own right, not a test update.** With per-row
  publication, rows that packed whole before a failure stay published as ordinary `kReady` rows under the usual
  eviction rules. Why that is acceptable: the guarantee's purpose, as its commit states it, is that **no row is published
  whose bytes were not read**. That is now enforced per row and independently of request outcome, by `pack_one`'s coverage
  check (`filled >= needed` fails the request rather than packing) and by `packed[ordinal]`. **The coverage check had no
  test until `5af2975164`**: before that nothing had shown it fire, because every neighbouring case is refused earlier by
  the table build or `admit_batch`. That commit adds
  `test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing` (single-root and mirrored), and
  reports two negative controls run against it: the check deleted publishes the short row and both parametrisations fail;
  the check weakened by one page also fails both. (I read the commit message and the test; I did not run the controls.)
  So this rationale rests on a tested safeguard, but only as of that commit; a cancelled advisory
  already publishes its completed rows under exactly this rule, so the semantics exist and are accepted. What is
  given up is a simpler statement ("a failed demand leaves the tier unchanged"); anything that relied on it (a test, a
  counter such as `rows_read`, the `kVersion` bump on a failed request) needs an explicit look. I could not find another
  consumer of the old guarantee by reading; that is not proof there is none.
- **`kVersion`** must still bump once per request (Python's `NativePinnedSlotTable.expert_to_slot` rebuilds when it
  moves); do not bump per row.
- **Task 8** sees longer-held hit leases (5.2). Its `lease_on_ready` / R3 window is unaffected.

## 8. Costs not in the ceiling

| Cost | Where it lands | Status |
|---|---|---|
| Lease-mode service round trip on every `count > 0` record (LEASE_PROTOCOL 15, OPEN 11) | all 40 layers, 67% of which read nothing | **unmeasured**; belongs to Task 5, must be netted out of Task 6 |
| Extra stage triples | all layers | `g` unmeasured; 160 triples per step for V2 over V1, 40 more for V1 over V0' |
| Extra graph nodes | `cudaGraphLaunch` host time and capture memory | unmeasured |
| Poll latency inside each `W_s` (`__nanosleep(256)` plus a PCIe read) | each stage | unmeasured |
| Post kernel: `ord`, `h`, `k`, deadline writes | each layer | small |
| Per-row publication under `mutex_` in the service | read layers only | small; `pack_one` already runs on that thread |
| Longer-held hit leases | eager `assign` / promotions | new contention on a scarce resource; not measured |

## 9. Requests to other owners (not edits; nothing outside my file was touched)

**To the `LEASE_PROTOCOL` / `lease_model.py` owner**

- **[REQ 1] BLOCKING DEPENDENCY, not a request to route.** Section 7.2 and 6.1: publication is per lane and staged in time
  (hits at reservation, miss rows at pack). This **is** the early device-visible readiness signal (section 3.1); without it
  no early copy exists, for V1 or V2. `demand_done` remains the request-level terminal, and per-lane waits poll it, so
  section 7.3's "the device never inspects lanes when status is FAILED" (F1) must allow the wait to read status **while**
  waiting on lanes. **Cost: unknown, owned by Task 5.** Two parts. (a) Small, and mine: in `serve()`, publish each hit
  lane's `RowResult` in the reservation critical section instead of after `read()`; a few stores per lane in a section that
  already runs. (b) Large, and Task 5's: the words themselves (`RowResult`, `LaneRequest`, `LaneAck`, `Terminal`), the
  lease counters and their retirement, the eviction predicate, generations and wrap, none of which exists. Task 6 cannot be
  accepted before (b); (a) is the whole of Task 6's protocol delta for V1.
- **[REQ 2]** Extend the model with per-lane / per-stage copy, the finalize kernel, and a **partial terminal mask**
  (`lease_model.py` section "what this does not establish"): a request whose stage `j` fails after stages `< j` copied and
  acknowledged, and a stage skipped because an earlier stage set `req_failed`.
- **[REQ 3]** DECIDE 3 (lease at publication) **stands for the current synchronous service** (reservation is one `mutex_`
  pass with `wanted` protecting every hit, `read()` blocks, nothing else evicts in that tier). It is a **constraint on any
  future asynchronous `progress()` service**: there, hit lanes must be leased at reservation, in the same `mutex_`
  section. Please record it, and record that D7 is closed (`8c78749b35`) and OPEN 12 is answered for the router-miss producer.

**To the tracing owner**

- **[REQ 4]** One instrumentation word: the planned lane count `k` per request in the demand record and `StageRecord`
  (and `h`), so A1 can be tested from a trace. Bump the trace schema; latency spans across schemas are not comparable.
  Also, if Task 6 is built: a schema-3 graph-decode trace with mirrors on, so the precheck can be rerun on the schema it
  will be judged by.

**To the plan**

- **[REQ 5]** Task 6's item list says "per-row transfers". The precheck supports *early* transfers and finds per-row's
  own contribution below the gate's resolution. Suggest the plan's Task 6 title and first mechanism become "two-phase (hit lanes,
  then the rest)", with per-row as the variant that must beat it.

## 10. Assumptions, open questions, decisions

**Assumptions**

- A1 (lanes per layer spread evenly) and A2 (hit lanes ready at `reserved`): section 1.4.
- A3 `c = 1.055 ms` is right for this path; **A4'** a kernel-node gap of 2-4 us; a stage triple costs 8-14 us. Both unmeasured.
- Lane order among miss rows equals pack order in practice (100% in this corpus). **This is a property of `pack_one`'s lowest-ordinal tie-break, not shown to hold for drive completion order.**
- The launch count, not the lane count, is what per-row costs; the SM footprint of a one-row call is the same 8 blocks.

**Open (could not determine)**

- **[OPEN 1]** Whether the 12 GB/s gather is link-limited or limited by the 8-SM kernel's own load latency.
- **[OPEN 2]** The per-layer round trip of lease mode (`LEASE_PROTOCOL` OPEN 11) on the layers that read nothing.
- **[OPEN 3]** Whether the traced request durations overstate the untraced ones, and by how much; `exl3_stage_trace_overhead.py`
  exists but I did not run or read its result.
- **[OPEN 4]** Whether stage waits should each poll one mapped word per lane or one per stage (a per-request "row ready" bitmask
  written by the service would cut poll reads; it is a second writer-owned word on the same line).
- **[OPEN 5]** (narrowed by section 3.1) V1's hit group must wait on the early `RowResult` words: they are the only device-visible
  early signal and the lease grant. What is open is their cost on the common path (a poll round trip per hit stage) and
  whether one phase-level word written after all hit lanes would cut it; it is a second word with the same single writer.

**Decisions I made that the owner may overturn**

- **[DECIDE 1]** Recommend V1 first and V2 expected-REJECTED (sections 0, 4.4, 5.6), on arithmetic, not measurement.
- **[DECIDE 2]** Ack stays a separate kernel per stage, as `LEASE_PROTOCOL` DECIDE 4 chose; a fused ack is the variant in 6.2(a).
- **[DECIDE 3]** Deadline is absolute from the post, and copy time counts against it.
- **[DECIDE 4]** `keep` is written only by `F`.

## 11. What has to exist before this can be accepted

No code was written. A checklist for the implementation, in dependency order: Task 5 (leases, acks, generations, model
check); the per-row publish hook in `RowReader::read` / `serve()`; the post kernel's `ord`, `h`, `k`, deadline; the stage
wait, ack-for-stage and finalize kernels; the stage chain in `Exl3RamMissRowBackend.post` (Python at capture only); the
graph-shape test of 5.2; the `hold_until_ack_lane` hook; the matrix of 5.4; then the arms of 5.6. Any new `SGLANG_*` flag
(for example the stage count) must follow `env-var-conventions`, and the stage count is a capture-time constant, so changing it
needs a fresh graph capture.
