# NVMe-to-RAM prefetch for DSpark forced misses

Date: 2026-10-08. Branch `codex/dsv41-ram-prefetch`, from `master` at `f1ba1a3d18`.

## Problem

Under DSpark with both CPU-expert clients at the 104 GiB pinned tier (`DSV41_REFERENCE.md` §33.12), a verify step is
437 ms p50, and 268 ms of it is the GPU idle inside the verify graph behind the lease gate (`copy_wait`). The gate's
last finisher is a forced NVMe miss 63% of the time: a lane whose expert is in neither VRAM nor RAM is read from the
mirrors into a staging slot and then computed on the CPU, so the whole landing is on the critical path. From
`copy_submit`, per NUMA group: DMA lanes done at 1.2 ms, the hit job at 2.5 ms, a forced miss's row landed at
4.3-4.7 ms p50 (10-11 p90, 19 p99), its CPU job done 1 ms later. Forced misses per group per layer: none in 35%,
one in 30%, two in 18%, three in 9%, four or more in 8% (mean 1.32); the last row lands at 2.2 / 4.4 / 6.7 / 11.1 ms
for 1 / 2 / 3 / 4+ misses. The split calibration and the row-weighted tile assignment (§33.11, the 2026-10-08
row-weighted arms) moved nothing because they shorten work that finishes before the misses land.

The 2026-09-26 study (`NVME_PINNED_PREFETCH_HANDOFF.md`) closed NVMe-to-RAM prefetch at ~2 ms/token for the best
realistic predictor and ~6 ms/token for a perfect one. Its three premises no longer hold:

| Premise then | Now, under DSpark with CPU experts |
|---|---|
| A missed row still crosses the PCIe link, so prefetch removes only the NVMe wait in excess of the link time (6.2 ms/token exposed). | A forced miss is computed on the CPU after landing; nothing of it crosses the link. The whole 2.2 ms per row is exposed, serially per group. |
| One layer of lead is ~2.75 ms, so an h=1 read (2.5 ms) barely lands in time; 2.7 rows/token were still filling when demanded. | A verify layer is ~11 ms (gate 5.3 ms p50 plus GPU work), four times the lead. |
| 0.28 RAM misses per layer (11.3 per token), so the prize per layer is small. | 2.6 forced misses per layer, 104 per verify step. |

Its constraints do hold and this design keeps them: speculative reads must be a strictly lower priority class at the
reader (FIFO arms went negative); wrong prefetches cost evictions (~3 harmful per token in the replay), so unused
speculative rows must be capped and cold; next-token and frequency predictors are useless (precision 0.03); layer 0
cannot be reached within a step.

## Levers that were considered and set aside

- **Intra-layer ordering.** Nothing is left: the reader batches a plan's rows (`kBounceRows` 8, 48 SQEs of ring
  credit), each miss's CPU job is submitted as its row lands, the two groups read in parallel, and `RowReader` lands
  O_DIRECT in the slab. Measured 2026-10-08: the three mirrors deliver 3.2 / 2.8 / 3.2 GB/s alone and the same
  together (no contention), so a row split in equal thirds waits for the SPCC's 4.4 MB at 2.8 GB/s, 1.6 ms, which is
  the 1.7 ms QD1 isolation figure; serving adds ~0.5 ms of record-to-issue and landing-to-submit latency.
- **A fourth mirror on `/mnt/nvme1`** (Crucial P310, Gen3 x4, 2.8 GB/s, 414 GB free against 191 GB of row images;
  it also holds the 64 GB swapfile). Four-way split cuts the slowest share to 3.3 MB, ~1.2 ms at 2.8 GB/s, so a row lands in ~1.7 ms with the ~0.5 ms
  of serving latency (bandwidth alone scales 2.2 to ~1.65; all three current mirrors are Gen3 x4). It is
  independent of this design and is run as its own env-only A/B; the replay here carries it as a timing variant
  (`--nvme-row-ms 1.3`, optimistic against the ~1.7 ms above).
- **Mirror re-weighting.** Tested 2026-09-29 (`analysis/dsv41-drive/mirror-scaling/weight-pair.md`): +0.4 ms/token,
  noise. Not repeated.
- **Prefetch into VRAM** and **device-selected candidates on the lease wire** (the retired
  `SGLANG_DSV41_ENABLE_NATIVE_PREFETCH`, `693fcb3414`). Closed in §25.4; and the device cannot see RAM residency
  or drive load, which is where the policy decisions are.

## Decision

Two phases. Phase 0 is a measurement with no serving code; Phase 1 is built only if Phase 0 clears its bar.

### Phase 0: replay go/no-go

**Capture.** One `dspark-both` arm at master with the stage trace's `GraphRouteLog` and
`SGLANG_DSV41_ROUTER_CAPTURE_PATH` on, over the 8-session suite (`DSV41_MAX_SESSIONS=8`) and a second run over
sessions 8-15 as held-out data, at the 104 GiB tier. It yields, per verify forward and streamed layer: the routed
experts with multiplicity per token, the plan's miss count, the hot set every layer held at the forward's start, and
each token's router input and top-k weights. The arm is a diagnostic, not a throughput number.

Code change: `GraphRouteLog.record_router` (`exl3_stream_trace.py`) takes `[M, H]` and `[M, top_k]` instead of one
token, `RouterCapture` writes `[records, layers, M_max, ...]` with the forward's token count beside its `seq`, and
the route ring's width accepts the verify's lane count. `tier_sim.load_forwards` reads the token count.

**Simulator.** `analysis/dsv41-drive/prefetch-replay/pinned_prefetch_replay.py`, extended rather than rewritten:

- *Chain per NUMA group*, replacing the link-bound layer model. At a layer's post, for each group: DMA lanes done at
  `--dma-ms` (1.2); the hit job ends at `--hit-ms` (2.5); each forced miss row lands serially at `--nvme-row-ms`
  (2.2; variant 1.3) behind any speculative legs in flight, then its CPU job takes `--miss-job-ms` (1.0); the gate
  opens at the max of the three; the layer ends at the slower group plus `--gpu-ms` (~1.0). The next layer posts
  then. The lead for a speculative read is whatever the chain gives, not a constant, so hiding one layer's misses
  shortens the next layer's lead.
- *Tier*: two groups with home node `expert % 2`, per-node capacities from `SGLANG_MOE_PINNED_HOST_NUMA_MB`
  (57344 / 49152 MiB), the draft's resident set pinned on node 0, `filling` slots never victims, `no_victim` when a
  node's evictable set is empty. Validated against the measured 16,486 RAM misses of 108,491 VRAM misses at
  104 GiB before any predictor runs.
- *Calibration*: `--dma-ms`, `--hit-ms`, `--nvme-row-ms`, `--miss-job-ms`, `--gpu-ms` swept once with no predictor
  to match the 2026-10-07 instrumented capture (gate p50 4.9 / p90 9.5 / p99 19.5 ms; last finisher miss 63%, hit
  28%, DMA 9%) and the 2026-10-08 trace (437 ms per step, 268 ms idle). The chosen constants are recorded next to
  the results.
- *Predictors*: `oracle` (h=1, h=2, next), `gate` h=1 and h=2 (layer T+h's gate on each token's layer-T input,
  candidates the union over tokens, up to K per token ranked by margin to the 6th score), `noisy` at p 0.25 / 0.4 /
  0.6. All with the priority queue, in-flight speculative bytes capped at one row, a per-layer spec-share cap, cold
  admission, layer 0 excluded; budgets swept from 1 to 4 rows per layer.
- *Metrics*, as the handoff defines them: precision at target and at any use, speculative rows per token, RAM
  misses per token, harmful evictions, demand delayed and mean delay, late (demanded while filling), exposed and
  saved ms per token, drive busy fraction.

**Go rule.** Build Phase 1 if `gate` h=1 at its best budget saves at least half of what `oracle` h=1 saves, under the
2.2 ms/row calibration. The findings go to `divix01:/data/models/slang/nvfp4-work/ram-prefetch/findings.md`, to
Hindsight, and to `DSV41_REFERENCE.md` §33 whichever way the rule falls.

### Phase 0 result (2026-10-08): go

`DSV41_REFERENCE.md` §33.13. At the 2.2 ms/row calibration `gate` h=1, one candidate per token, **one speculative row
per layer** (the replay's `--budget 1`, a single budget per layer shared by both groups, the highest-margin
candidate wherever it is homed) saves 7.33 / 7.94 ms/token (suite / held-out) against `oracle` h=1's best 12.10 /
12.47: ratios 0.62 / 0.65, against the bar's 0.5 / 0.4. Precision 0.60. Budget 2 saves 6.03, budget 4 2.68, two
per token at budget 8 nothing, h=2 1.82, a FIFO queue -0.53. The alternative calibration (2.0 ms/row) gives 0.54 /
0.57. The replay is optimistic in absolute terms (its gate tail is heavier than measured, and it leaves out the
scoring and issue costs), so the expected live gain is **4-6 ms/token**, not 7.3. Phase 1 below is trimmed to that
arm: h=1 only, one candidate per token, one row per layer.

### Phase 1: host-scored speculative reads, one layer-wide budget

**Input, no device change.** The post kernel (`exl3_ram_miss_post_kernel`, `lease_kernels.cuh`) already stages each
record's `[M, H]` layer input as fp16 for the CPU experts (`cpu_x_dst`, `cpu_hidden`), and each record carries its
layer's hot bitmap (`apply_gpu_hot`). The host additionally loads the 40 gates (`layers.N.ffn.gate.weight` and
`.bias`, 126 MB) once, on NUMA node 0, at service start through the service's FFI. The option therefore requires
`SGLANG_DSV41_CPU_EXPERTS=1`, which the DSpark recipe has.

**One scout.** A single pinned thread on node 0, on a reserved core from `ThreadingConfig`, fed over an SPSC ring
(`spsc_ring.h`) with each verify record's seq and layer once its input is staged. For record (seq, layer T) it scores
layer T+1's gate per live token (`sqrt(softplus(W x)) + b`, M x 384 x 4096 fp16, ~0.1 ms at M=6), takes each token's
`PER_TOKEN` candidates by margin to its 6th score, forms the union ranked by margin, filters (not VRAM-hot by the
record's hot bitmap for layer T+1, one record stale; not layer 0), and emits the top `PER_LAYER` survivors. Each
candidate goes to its **home group's** candidate ring (`expert % 2`, as the tier homes rows), so the budget is
layer-wide across both groups, as the replay's was. The scout touches no tier state: RAM residency and `filling` are
checked at admission, where the tier lives; a candidate admission rejects does not return budget to the scout.

A single scout, not one per group: the scoring is ~0.1 ms against a ~11 ms verify layer, and a per-group budget of
one row is two rows per layer, which the replay puts near its budget-2 arm (6.0 rather than 7.3 ms/token). If the
cross-group ring proves awkward, two scouts with a per-group budget is the fallback, at that cost.

**Admission, on the service thread.** The tier is owned by its group's service thread, so admission stays there. In
the poll loop, when the demand head shows no new record, it pops candidates from its ring, drops any already
RAM-resident or `filling`, claims a victim by the existing rule (`filling` set, never a victim, cold stamp), and
submits the row to a **speculative reader**: a second `RowReader` on its own io_uring ring, in-flight rows capped at
`INFLIGHT_ROWS` per group (default 1). Candidates that arrive while the cap is full wait in the ring, oldest first,
and are dropped once their target layer's demand record has been served. Completions are reaped in the same loop;
a landed row is vetted through the demand path's digest check, then becomes kReady, cold-stamped, and counts against
the layer's `SPEC_SHARE` of unused speculative rows per group (admitting beyond it evicts the oldest unused
speculative row, never a demand row).

**Demand priority.** A demand record is served exactly as today on the demand reader. Speculative legs are never
submitted while a demand read runs (the service thread is inside `read_misses` then), and the in-flight cap bounds
the device-queue delay a demand read can see to one row (~1.4 ms at the three mirrors' ~9.3 GB/s). A demand lane whose
expert is `filling` waits for that landing instead of reading again (promotion), bounded by the same cap; the lane is
typed kKindMissCpu with the filling slot as its staging slot and its CPU job is submitted when the row lands, through
`submit_landed_cpu_misses`.

**Failure handling.** A failed speculative read releases the slot and unmaps the expert (nothing waited on it) and
counts `spec_failed`; a failure during promotion is a demand failure and fail-stops like any demand read
(`fail_record`). The speculative reader holds the watchdog's busy episode while it has legs in flight, so a hung ring
aborts the process as a hung demand does. Shutdown stops the scout, then drains the speculative ring before joining
the demand reader, as `fill_join` does for the fill thread.

**Options**, in `environ.py` under the env-var conventions, default off until a serving arm accepts them:

| Variable | Default | Meaning |
|---|---:|---|
| `SGLANG_DSV41_RAM_PREFETCH` | off | on/off |
| `SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN` | 1 | candidates per live token, by margin |
| `SGLANG_DSV41_RAM_PREFETCH_PER_LAYER` | 1 | speculative rows the scout emits per layer, both groups together |
| `SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE` | 4 | unused speculative rows a layer may hold per group |
| `SGLANG_DSV41_RAM_PREFETCH_INFLIGHT_ROWS` | 1 | speculative rows in flight per group |

Lookahead is fixed at one layer and there is no margin floor: h=2 and wider budgets lost in the replay, and the best
arm had no floor. Both are left out rather than made options.

The launch gate (`expert_stream_requirements_exl3.py`) refuses the option without target CPU experts or row images.
With the option off the request path is today's: no scout, no gate copy, no second ring, no clock reads.

**Observability.** Per-group counters on the service's metrics line: issued, used at target, used later, evicted
unused, failed, dropped (stale or already resident), demand lanes delayed by a speculative leg, promotions; the
scout's candidates per layer and scoring time. InstrBuild events `spec_submit`, `spec_land`, `spec_use` with the
record's seq and row, so the job-trace joins (`miss_latency.py` and kin) attribute them.

## Invariants

- A `filling` slot is never a victim and `release` refuses it (already true for prefill fills).
- A speculative row is published only after its whole read vetted, by the same path as a demand row.
- No speculative leg is submitted while a demand read is in progress.
- In-flight speculative bytes never exceed the cap.
- A demand lane on a `filling` expert yields the same bytes as a fresh read.
- The scout touches no tier state, and emits at most `PER_LAYER` candidates per layer across both groups.
- With the option off, the service's request path is unchanged, and `traced_clock_reads()` stays at today's count.

## Testing

- **Phase 0.** `record_router` and `RouterCapture` with M > 1 (shapes, token count, a replay reading them back);
  `load_forwards` on a verify trace; the simulator's chain model against a hand-computed layer (one miss, two misses,
  a speculative row that lands before and after its demand); the tier model's per-node capacities and `no_victim`.
- **Phase 1, host C++** through the existing `HostCopyBackend` and `DeviceSim` fixtures: the scout's ranking and layer-wide budget against a
  torch reference gate on captured inputs (routes reproduced, as §18.4's self-check did); admission and the spec-share
  cap on a scripted tier; promotion; the in-flight cap under a demand arrival (the `inject`/`inject_fault` hooks delay
  a speculative leg and the test asserts the demand's delay bound); a failed speculative read; shutdown with legs in
  flight. Mutants that must be caught: release of a filling slot, a speculative submit during a demand read, publish
  before vet, a victim taken from a filling slot, the cap ignored, a layer's budget exceeded, a candidate sent to
  the wrong group.
- **Python.** Option parsing and refusals, the launch gate, the metrics line.
- **Full stack.** The expert-stream full-stack benchmark with the speculative reader on, bit-exact against off.

## The A/B

The standard 8-session `dspark-both` suite at 104 GiB through `analysis/dsv41-drive/dspark/both_cpu_ab.py`: arm
`dspark-both` (A) against `dspark-both-prefetch` (B, the replay's best h and K), same run protocol as the row-weighted
arms (private worktree, private `SGLANG_EXL3_BUILD_DIR`, counters off, extension path and hash logged). The fourth
mirror is a separate pair, since it moves the same number. Expected: 4-6 ms/token (Phase 0 result). Acceptance: a median ms/token gain beyond the suite's
~3% run-to-run spread, outputs within the near-tie band, RAM misses per token down, with `demand delayed` and
`harmful evictions` reported. The default stays off until the owner accepts it for production.

## Out of scope

Prefetch into VRAM; a device-side candidate kernel; lookahead beyond one layer; next-token or frequency predictors; layer 0; the trained LLaPor
and APEX scorers (1.3-3.3 ms/token of in-graph scoring, and their capture refuses speculative decoding); any change to
the demand reader's own issue order.
