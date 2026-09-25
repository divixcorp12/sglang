# DSV4.1 EXL3 native next-layer prefetch (h=1 native gate, K=1, copy engine)

**Date:** 2026-09-25. **Branch:** `cc/native-prefetch`, from `shared/cc/dsv41-pinned-numa` at `7333ddf47b` (plus
`e4f81ac774`, the Dsv41Config fix, merged). **Spec:** the "Recommendation: the h=1 native gate at runtime" section of
`2026-09-25-dsv41-router-capture.md`. Evidence: `divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/native-prefetch/`.

## Answer

- **Both gates passed.** The predict step costs ~0.2 ms/token. The replay with the RAM-ready filter still estimates
  **-16.9 ms/token** at the 0.85 budget, against -17.3 without the filter.
- **The runtime path works, but it is slower.** Behind `SGLANG_DSV41_ENABLE_NATIVE_PREFETCH` (off by default), with
  the copy engine on in both arms:
  - A (prefetch off) ran at **120.6 ms/token** (123.2 wall-validated). B (prefetch on) ran at **123.2** (125.9).
  - Both arms had 0 stalls and were 6 of 6 byte-identical to p2a-rowimg.
  - Precision on the device was 0.73, which matches the replay.
- **Why it loses: the idle link is not where the replay put it.** Diagnostic counters time each prefetch:
  - Between the post (after layer T-1's demand chain) and layer T's gather there are only **361 us** of compute.
  - Each ~1 ms copy therefore leaves **620 us** exposed at T's commit, ~20 ms/token over 32.7 prefetches.
  - The 23.8 used rows per token save less than that. They replace copy-engine hit copies, which the chain already
    overlapped with the same layer's NVMe streaming.
  - Rerun at the measured budget of 0.36 rows per layer, the replay itself predicts no gain (120.1 against 120.6).
- **Recommendation: keep the flag off and do not merge as a speedup.** The next step is to issue the prefetch while
  the link idles in T-1's NVMe wait: post it before T-1's chain, and have the service hold it until T-1's demand DMA
  completes (section "Next").

## Gate 1: predict cost

`analysis/dsv41-drive/native-prefetch/predict_cost.py` captures a CUDA graph of 39 predicts. Each has its own
384x5120 bf16 gate, 157 MB in all, more than the 96 MiB L2, so every GEMV reads HBM. Cost per predict is
(graph - empty graph) / 39. RTX 5090, 200 replays, `gate1/{plain,filler}.log`.

| variant | per predict p50 | with a 4 MiB copy before each | per token (39) |
|---|---:|---:|---:|
| `tiny_gemm_bf16` only | 2.81 us | 3.62 us | 0.11-0.14 ms |
| `tiny_gemm_bf16` + `moe_fused_gate` (sqrtsoftplus_log1p) | 5.33 us | 5.78 us | 0.21-0.23 ms |

The GEMV implies ~1.4 TB/s against the card's ~1.8 TB/s: plausible, and not L2-flattered. **Pass** against the
~1 ms/token budget.

The runtime uses the GEMV plus its own plan kernel, not `moe_fused_gate`. The replay's ranking needs the biased order
of the six, and the fused gate returns its winners in id order.

## Gate 2: the RAM-ready filter

`analysis/dsv41-drive/native-prefetch/ram_ready_replay.py` runs on the router-capture trace (varied24, held-out odd
prompts). `gate2/gate2.{json,log,cmd}`, at `577f341725`.

**The pinned tier is replayed and checked against the registers.** The replay runs beside `DirectInsertReplay`:

- 8063 rows split 201/202 per layer.
- LRU by the stamp that every routed expert's request refreshes, inclusive of the VRAM hot set.
- The request's own routes are protected; a VRAM miss the tier lacks is read and admitted. This is
  `take_slot_locked`'s rule.

Its per-(step, layer) NVMe reads equal `graph_step.layer_ram_rows` in **99.94%** of cells: 11.207 against 11.209
reads per token (test split 10.755 against 10.752), aligned by `vram_miss` on 6145 of 6151 steps.

**Rank-1 candidates** (the first non-resident of the gate's top 6, held-out):

| | share RAM-ready |
|---|---:|
| all rank-1 candidates (35.9 per token) | 0.843 |
| ... that T routes (useful) | 0.907 |
| ... that T does not route (wasted) | 0.709 |
| T's VRAM misses | 0.862 |

**The filter costs little, because it removes more waste than hits.** Held-out, K=1, h=1, `PrefetchReplay` and
`link_exposed`, 1.0 ms/row, base 119 (the copy-engine smoke):

| arm | prefetch/tok | precision | exposed @0.85 | ms/tok @0.85 | exposed @1.7 | ms/tok @1.7 |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 | - | 77.99 | 119.0 | 77.99 | 119.0 |
| native gate d6 | 35.75 | 0.69 | 60.73 | 101.7 | 55.36 | 96.4 |
| **native gate d6, RAM-ready** | 33.49 | **0.71** | **61.06** | **102.1** | 56.03 | 97.0 |
| native gate d6, RAM-ready, K=2 | 54.72 | 0.65 | 71.80 | 112.8 | 51.89 | 92.9 |
| oracle K=1 | 34.72 | 1.00 | 50.64 | 91.6 | 45.43 | 86.4 |

**-16.9 ms/token, which clears the ~8 ms bar. Pass.** Offline rank-1 precision rises from 0.678 to 0.694.

## What was built

| Piece | Where |
|---|---|
| Flag, config field | `SGLANG_DSV41_ENABLE_NATIVE_PREFETCH` (`environ.py`, off); `Dsv41Config.enable_native_prefetch` |
| Gate registry | `DeepseekV2MoE.__init__` registers each DSV4 layer's `MoEGate` when the flag is on |
| Hooks | `Exl3MoEMethod._apply_graph`: `commit(T)` before the gather, `predict(T)` for T+1 after it, captured forwards only |
| Runtime | `python/sglang/srt/layers/moe/exl3_native_prefetch.py` (`NativePrefetch`) |
| Kernels | `exl3_native_prefetch.cuh` (plan, commit), wrappers `kernels/ops/moe/exl3_native_prefetch.py` |
| Victims | `GpuResidencyUpdater._rank_victims` also keeps DIRECT's order past the shortlist (7 slots) when the flag is on |
| Service | `exl3_ram_miss_host.cpp`: `pump_prefetch`, a prefetch lease, prefetch copy jobs, judge, 10 counters |

**Predict.** In layer T-1's captured `_apply_graph`, after its gather has been posted and before its fused MoE:

- `tiny_gemm_bf16(x_{T-1}, W_T)` into fp32 logits. This is the router's own call, with the same shape and `max_m`.
- Then the plan kernel (one block):
  1. Scores `sqrt(log1p(exp(x)))` (`x` when x > 20) plus T's bias, as `moe_fused_gate` does.
  2. Takes the top 6.
  3. Keeps the first that is not resident in T (`mapping`) and is READY in the pinned tier. It reads the service's
     published host map through UVA, as a hint; the service re-checks.
  4. Picks the victim: the first of DIRECT's ranked slots past the demand shortlist that holds none of the six. This
     is the replay's rule.
  5. Posts {row, expert, slot} and then the tagged generation on a 256-byte pinned prefetch page. T's residency is not
     touched.

Layers 0..38 predict and targets are 1..39. T=0 (cross-token) is skipped.

**Service.** `pump_prefetch` runs after `pump_demand` in the service loop.

- A new request whose row is READY in the pinned tier gets a lease on that slot, with the same E1 rule as a COPYING
  lane (never a victim while leased). It becomes a one-lane copy job.
- Otherwise the request is SKIPPED (unarmed, not ready, invalid), with no lease and no read. A prefetch never reads
  NVMe.
- The copy thread observes completion with `cuEventQuery`, checks the slot generation (E6), publishes `COPIED`, then
  releases the lease.
- **Demand before prefetch:** the copy thread holds a prefetch job while a demand job is fresh or in flight. The device
  never posts T's demand while T's prefetch is outstanding, so none can queue behind one.
- The next record of the target row judges each copied row *used* or *wasted* against that record's routes. Every
  layer posts a record per forward.

**Commit.** At the start of T's `_apply_graph`, before its gather reads residency:

- If T has a pending request, wait for its done word, under the RAM-miss timeout and the fatal and shutdown words.
- `COPIED` maps the slot: unmap the old expert, map the new one, READY, generation + 1.
- `SKIPPED` changes nothing.
- An abort unmaps the victim, leaves it neither free nor evictable (a late copy may still land there) and raises the
  fatal word.

**Byte identity by construction.** Prefetch changes residency only. The MoE reads the same bytes whether a demand copy
or a prefetch copy wrote the slot.

**Deadlock safety** (LEASE_PROTOCOL.md 7.6, "Module loading"):

- Both kernels launch only while a graph is captured, never eagerly, so they are loaded before the copy engine arms
  (after 16 decode forwards).
- `Exl3RamMissService._arm_copy_engine` calls `NativePrefetch.check_armable()` and refuses to arm until both kernels
  were captured.
- The GEMV is the router's already-loaded `tiny_gemm_bf16` specialisation.
- The flag refuses to start without the copy engine.

## Tests

- **CPU** (`suite/cpu_head_f2683ba3ee.log`), at `f2683ba3ee`: **1396 passed, 414 skipped, 0 failed**, pytest exit 0. Base `7333ddf47b` (the same targets
  without the new file, `suite/cpu_base_7333ddf47b.log`): 1374 passed, 414 skipped, 1 failed
  (`test_one_field_per_knob`, fixed by `e4f81ac774`).

  ```bash
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$WT/python OMP_NUM_THREADS=8 taskset -c 0-31 python -m pytest test/registered/unit/kernels \
    test/registered/unit/layers/moe/test_exl3_ram_miss_{service,shutdown,tables}.py test/registered/unit/layers/moe/test_exl3_stream_trace.py \
    test/registered/unit/test_dsv41_config.py test/registered/unit/layers/moe/test_exl3_native_prefetch.py -q -p no:randomly -rfE \
    --basetemp=/mnt/nvme1/pytest-tmp/<unique>
  ```

- **GPU** (`analysis/dsv41-drive/native-prefetch/gpu_suite.sh`, `cc-gpu.lock`, cores 32-63): the copy-engine suite's
  targets plus `test_exl3_native_prefetch_cuda.py`.
  - At `f2683ba3ee` (`suite/gpu_head_f2683ba3ee.log`): **1947 passed, 1 skipped, 1 error** (only the known `[trace_on]` error).
  - At `426e1d2e10` it gave 1945 passed, 1 skipped, 1 failed, 1 error. The error is the known
    `test_graph_routes_are_logged_only_when_the_stage_trace_is_on[trace_on]`.
  - The failure was `test_work_queued_behind_the_graph_on_other_streams_does_not_hold_the_copy_back[kernel]`: the first
    replay took 2.0 s, the copy wait's deadline. Run alone it passed 3 of 3 at head and 3 of 3 at the base
    (`suite/ce_queued_*`). It is an intermittent in-suite failure of the copy-engine test, not reproduced; see Defects.
  - `test_apply_graph_adds_nothing_unless_router_capture_is_on` passed in both GPU runs.
- **New tests:**
  - `test_exl3_native_prefetch_service.py` (CPU, the real service and the CPU copy backend, 11 tests):
    - a ready row is leased, and COPIED is published only after the observed completion;
    - not-ready, unarmed and invalid requests are skipped with no lease and no read;
    - demand is served before a prefetch posted at the same time;
    - the copy thread holds a prefetch job while a demand job is in flight;
    - a demand whose only victim is under a prefetch copy defers until the copy completes;
    - used and wasted are judged once;
    - the flag refuses to start without the copy engine.
  - `test_exl3_native_prefetch.py` (CPU, 11 tests):
    - the arming guard, both in `NativePrefetch` and in the service;
    - the flag and config field;
    - eager and unbound hooks launch nothing;
    - `bind` refuses a residency updater without the extended ranking;
    - the extended victims continue DIRECT's own order and leave the shortlist unchanged.
  - `test_exl3_native_prefetch_cuda.py` (GPU, 7 tests):
    - the plan kernel matches a Python reference of the rule on 300 random cases;
    - nothing is posted once the page is fatal;
    - commit on COPIED, on SKIPPED, and on a stale done word (timeout, fail-stop, slot taken out);
    - **byte identity**: 120 decode steps replayed from captured graphs (commit graph, then demand chain + MoE gather
      + plan) against the real service and copy engine. Prefetch on and off give byte-identical MoE inputs, equal to
      the checkpoint's rows, and prefetch on has fewer demand misses. It runs plain and with a 512 MB ballast on every
      copy job. After every commit, a snapshot taken before any sync must show every mapped slot holding its expert.
- **Mutants**, each in the private worktree `wt-native-prefetch-mut`, reverted and re-run green:
  - CPU (`suite/mutants_cpu.log`): issuing prefetch jobs in arrival order fails the priority test. COPIED at the grant
    fails the completion test. Dropping the prefetch lease fails the completion and victim tests.
  - GPU, the commit's wait removed (`suite/mut_commit_nowait_*`): **red**: the ballast variant fails "mapped before landing" (expert 8, `w13_trellis`); restored, 2 of 2 green. Two earlier versions of the
    byte test did not catch it. The chain's own copy wait usually drained the stream first, so the snapshot check was
    added.

## Smoke: A then B, once each

Both arms ran `smoke.sh` (the copy-engine template) with the `arm_env` recipe and `SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1`
exported explicitly. B added `SGLANG_DSV41_ENABLE_NATIVE_PREFETCH=1`. Setup:

- `426e1d2e10`, cold server per arm, six greedy requests (prompts 0-2, twice), `cc-gpu.lock` and `rowimg-disk.lock`.
- A at 05:11, B at 05:16.
- Output: `smoke/A426e1d2e10-off` and `smoke/B426e1d2e10-on`.
- `compare_arms.py ref=p2a-rowimg A=... B=...` wrote `smoke/compare_AB.json`; `np_stats.py` wrote `smoke/np_stats_AB.json`.

**Copy engine on in both arms:**

- `env.txt` has `SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1` in both, and `SGLANG_DSV41_ENABLE_NATIVE_PREFETCH=0` or `=1`.
- Both logs show "copy engine armed after 16 decode forwards".
- A: 33.5 jobs and 71.6 lanes per step (953 MB). B: 26.9 jobs and 47.6 demand lanes per step, plus the prefetch jobs
  (1069 MB in all).
- Both: 0 fallbacks, 0 errors, 0 generation mismatches.

| Arm | ms/token, trace (wall-validated) | step p50 / p90 ms | stalls (multi-row > 10 ms) | responses vs p2a-rowimg |
|---|---:|---:|---:|---|
| A, prefetch off | **120.6 (123.2)** | 115.8 / 150.1 | 0 of 1885 | 6 of 6 byte-identical |
| B, prefetch on | **123.2 (125.9)** | 117.7 / 155.8 | 0 of 1897 | 6 of 6 byte-identical |

The p2a-rowimg reference ran at 132.4 (135.2).

**Prefetch counters, B**, from the service over the whole run, warm-up included (584 graph steps):

| requests | issued = copied | skipped unarmed (before arming) | skipped not ready | used | wasted | precision |
|---:|---:|---:|---:|---:|---:|---:|
| 19,548 (33.5/step) | 19,080 (32.7/step) | 468 | 0 | 13,895 (23.8/step) | 5,185 | **0.728** |

- Read-to-completion latency is 988 us per copy. No prefetch job was ever held behind a demand job, as the ordering
  argument predicts.
- The device's plan counters were read lagged at 05:19:40: 16,663 posted, 2,993 plans with no candidate (every one of
  the top 6 resident, or the pinned tier lacked it), 5,038 top-6 entries dropped by the RAM filter, 0 without a victim
  and 0 aborted.
- The 24 fewer demand lanes per step (71.6 to 47.6) match the 23.8 used rows.

**Where the time went.** This is a diagnostic B run, not an arm of the comparison: `smoke/Bdiag1fd396215b-on`, at
`1fd396215b`, with timing counters added to the commit kernel.

- **Window:** 361 us per commit from the post to the target's gather, which is the compute the copy can hide under.
- **Wait:** 620 us per commit spent waiting for the done word: the copy left exposed.
- Precision was 0.727, the same as B's.

At 32.7 commits per token the wait is ~20.3 ms/token exposed. With +2.6 ms/token measured, the 23.8 used rows saved
about 17.7 ms, or ~0.74 ms each rather than 1.0. That split is inferred from the two measurements, not measured
directly.

The replay agrees once its budget is the measured window: at 0.36 rows per layer, native gate + RAM filter gives
77.47 exposed rows against 77.99, 120.1 against 120.6 ms/token (`gate2/gate2_budget036.*`).

## Defects found

1. **The spec's link budget was wrong for this placement.** The replay's 0.85 rows of idle link per layer is
   (116 - 82)/40. Most of that idle link sits inside the demand phase, while S waits for NVMe pieces, not in the
   compute window after the chain. A prefetch posted after T-1's chain sees 0.36 ms. The replay also charges every
   removed demand row a full 1.0 ms, but a copy-engine hit copy overlaps its layer's NVMe streaming.
2. **Intermittent copy-engine test failure.**
   `test_work_queued_behind_the_graph_on_other_streams_does_not_hold_the_copy_back[kernel]` hit its 2 s deadline once
   in the full GPU suite and passed 6 of 6 alone, at head and at the base. The prefetch changes the copy thread's loop
   only for prefetch jobs; demand jobs issue exactly as before.
3. **`smoke.sh nsys-node` writes no report.** The server's shutdown kills its process tree, nsys included, before nsys
   writes the report (`smoke/Bnode3c62b40bcc-on` has none). The copy-engine template has the same wrapper.
4. The clock-read gate test (`test_no_clock_read_bypasses_the_trace_gate`) listed the copy thread's exact lines. It now
   lists the two changed copy-engine lines and the two prefetch latency reads.

## Next (not built)

**Post before the chain, hold until T-1's demand DMA completes.** x_{T-1} exists at T-1's router, before its gather:

1. Post the prefetch there, carrying the demand sequence number that follows it.
2. The service holds it until that demand has been served and its copy-engine job completed. This extends the copy
   thread's hold rule, which already exists.
3. The copy then runs during T-1's NVMe wait and S, where the replay's idle link actually is.

**Risk:** the prefetch DMA shares PCIe with S's SM piece copies. Whether S slows is a measurement, like the NVMe-bound
layers.

**First step:** a replay whose window per layer is T-1's measured demand phase minus its copy-engine time, taken from
the stage records. Build only if that clears ~8 ms/token.

## Reproduce

```bash
# divix01, a private worktree of cc/native-prefetch
N=/data/models/slang/nvfp4-work/direct-two-phase-tests/native-prefetch
analysis/dsv41-drive/native-prefetch/gpu_run.sh $PWD $N/gate1/plain.log python analysis/dsv41-drive/native-prefetch/predict_cost.py
cat $N/gate2/gate2.cmd    # the replay command, with its commit
analysis/dsv41-drive/native-prefetch/smoke.sh off A<commit> $PWD; analysis/dsv41-drive/native-prefetch/smoke.sh on B<commit> $PWD
taskset -c 0-63 python analysis/dsv41-drive/row-images/compare_arms.py ref=$N/../row-images/p2a-rowimg A=... B=...
taskset -c 0-63 python analysis/dsv41-drive/native-prefetch/np_stats.py A=... B=...
```

## Early post: gate 1 fails (2026-09-25)

**Branch** `cc/early-prefetch`, from `shared/cc/dsv41-pinned-numa` at `16a19c5087`. **Evidence:**
`divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/early-prefetch/gate/` (`gate.{json,log,cmd}`), made by
`analysis/dsv41-drive/early-prefetch/window_replay.py` at `07b2e4a907`.

**Verdict: the early post estimates -2.8 ms/token at best (117.8 against A's 120.6). That misses the 8 ms bar, so
nothing was built.** Steps 2-5 of the brief (build, tests, smoke, nsys wrapper fix) were not run, as the brief
requires at a failed gate. Defect 3 above (`smoke.sh nsys-node` writes no report) is therefore still open.

### The replay

It uses A's own link timeline (`smoke/A426e1d2e10-off`, 569 graph decode steps after arming) and B's prefetch
rates (`smoke/B426e1d2e10-on`: 33.4 issued, 24.3 used per step).

- **Per (step, layer):** the service's `observed` post time, the copy-engine job (hit lanes = `lanes - rows_asked`,
  starting at `reserved`), and S's pieces (one per NVMe extent, released at its `extent_cqe_ns` completion).
- **One link, served first come first served, at 0.98 ms per 13.3 MB row.** Three fits agree on this rate:
  - the step's summed copy latency is 0.15 ms/job + 1.00 ms/lane;
  - a no-read layer's post-to-post period is 0.39 ms + 0.98 ms per hit lane;
  - `numa-h2d` measured 13.7 GB/s (0.97 ms/row) for one copy, and zero-copy is no faster.

  So a second concurrent copy adds no bandwidth, and one link is the right model.
- **Model check:** `observed(L+1)` minus L's modelled chain end is the compute after the chain. It should be flat
  across lane and read counts.
  - Without NVMe reads it is 0.36-0.39 ms at every lane count.
  - With reads it is 0.41-0.62 ms: the model ends those chains ~0.2 ms early, which makes it slightly *optimistic*
    about the idle link after them.
- **Early post:** the prefetch for T is released at `max(observed(T-1), CE_end(T-1))`, which is the proposal's hold.
  Its deadline is `observed(T)`, since T's commit precedes its post. It needs 0.988 ms, B's measured copy time.
  - `priority`: it gets only link time demand does not want.
  - `share`: it splits a busy link 50/50, and the delay this adds to T-1's chain is charged.

  A copy not finished by the deadline is charged in full for the rest.
- **Saving:** a used row removes one hit lane from T's copy-engine job. It saves
  `max(CE_end, S_end) - max(CE_end - lane, S_end)`, a full lane only where the copy engine and not NVMe sets T's
  chain.
- **Calibration:** the `late` arm replays B's placement (release 0.361 ms before the deadline).
  - It reproduces B's measured wait: 0.627 ms per prefetch against 620 us.
  - It overstates B's cost: +4.5 against the measured +2.6 ms/token. The saving per used row is modelled at 0.68 ms
    against ~0.74 inferred.
  - That 1.9 ms/token error is carried to the early arms as a constant.

| arm | idle link, T-1 post to T gather (mean / p50 / share >= 1 copy) | exposed per prefetch | fully hidden | exposed + chain delay, ms/tok | saving ms/tok | est. ms/tok raw | calibrated |
|---|---|---:|---:|---:|---:|---:|---:|
| late (B, calibration) | 0.74 / 0.41 ms / 17% | 0.627 ms | 0% | 21.0 | 16.5 | 125.1 | **123.2** (measured 123.2) |
| **early, demand priority** | same | 0.465 ms | 17% | 15.6 | 16.5 | 119.7 | **117.8** |
| early, shared link | same | 0.416 ms | 32% | 13.9 + 4.5 | 16.5 | 122.5 | 120.6 |

### Why

**The idle link is not in T-1's NVMe wait either, for most layers.** Per layer, the link is idle for 0.74 ms on
average between T-1's post and T's gather, but the median is 0.41 ms. Only 17% of layers have a full copy's worth.

- **About two thirds of layers read nothing from NVMe.** For those, the copy engine is the chain's critical path, and
  the link runs back to back from T-1's copy-engine start until its end. Only the ~0.39 ms of compute after it is
  free, the same window the late post already had. So holding the prefetch until T-1's copy-engine job finishes gains
  nothing on these layers.
- **Layers that do read from NVMe have more idle link, but S's pieces use most of it.** A row read from NVMe costs the
  link about 1 ms of piece copies, arriving over the read's ~2.6 ms. That is as much link time as the prefetch itself.
- **The saving has a ceiling:** at 24.3 used rows it is 16.5 ms/token modelled, ~18 inferred, even with nothing
  exposed. On this link, every used row costs 0.98 ms of link time wherever it is moved. It can only win if it lands
  in time the link would otherwise idle, and the replay finds ~0.74 ms of that per layer, against a copy of ~1 ms.
  - The prefetch also adds the 9.1 wasted rows per token, 8.9 ms of link time that demand does not need.

### Recommendation

- **Keep `SGLANG_DSV41_ENABLE_NATIVE_PREFETCH` off and do not build the early post.** A layer-ahead prefetch has
  little left to gain on a link that is ~75% busy with demand (~73 hit lanes and ~11 NVMe rows per token at
  0.98 ms/row, ~120 ms/token).
- **What would move the number is fewer link rows, not earlier ones.** Two ways to get them:
  - a larger VRAM hot set (fewer hit lanes);
  - a faster host-to-device path. `numa-h2d` measures 13.7 GB/s for one copy, against the card's PCIe ceiling, and
    that gap is worth explaining before any further prefetch work.
- **The nsys-node wrapper fix (defect 3) is still needed by the next full-arm trace.** It should be done on its own
  branch.
