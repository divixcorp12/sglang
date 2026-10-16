# Lane tier replay (step 1 of the lane head's next version): handoff, 2026-10-16

Written for a reader with no context. Branch `lane-tier-replay` (off `lane-head` `eb1a14c8c0`). Script:
`analysis/dsv41-drive/dspark/lane_tier_replay.py`. Offline only; no GPU, no new capture. **Status: stopped at the
gate.** P passes, T and E fail. The cause of T's failure is the prefill admissions, which the trace does not record.
Waiting on team-lead and the owner: whether to take a capture with prefill route logging.

## Why

The served lane-head A/B (`/mnt/nvme1/lane-head-ab-20261016`; see `LANE_HEAD_HANDOFF.md` section 0 on `lane-head`)
moved +19.5 VRAM hits/verify but no ms/token. The head gave the GPU lanes to experts whose rows were NVMe-cold, so GPU
stream time rose +51.7 ms/verify and CPU wait fell -46.9. Step 1 was approved: a replay that prices lanes by RAM tier
and is validated on the A/B before it evaluates anything (option (b) "cap to RAM-warm", the head, and an oracle on
C2/base). That evaluation was never reached.

## The gate (as stated and accepted, fixed before computing)

Measured on serve_compare's pair-clean prompts (0, 1, 4, 5, 7; prompts flagged in neither arm). Fits use OFF only.
- **P (pricing).** Per-record costs fit on OFF, applied to ON's observed lane tiers:
  - Δstream/verify within ±20% of +51.68 ms;
  - Δwait/verify within ±20% of -46.92 ms;
  - stream layers/verify within ±1.0 of 12.28 (OFF) and 23.68 (ON);
  - in-sample OFF stream/v and wait/v within ±10%.
- **T (tier simulation).** NVMe-cold GPU lanes/verify within ±15% of observed per arm, and the ON-OFF delta within
  ±20%. Team-lead: T matters most, because C2/base has no observed RAM tier. Report its per-arm error and the delta
  even if they pass.
- **E (end to end).** T's tiers priced by P give Δstream and Δwait within ±25%.
- If a part misses: report why and stop. Do not tune it until it fits.

## Confirmed trace facts (A/B, both arms)

- A record (one layer of one verify) has stream > 50 µs exactly when it has a GPU NVMe lane: 5,942/5,942 (OFF),
  5,605/5,605 (ON).
- GPU NVMe lanes = the record's `evict` incoming experts minus its `miss_expert` (MISS_CPU) experts minus its
  `pool_point` experts.
- The experts a layer inserts into VRAM (route-log hot set at v+1 minus at v, within the misses) are exactly its GPU
  lanes: 12,305 = 12,305 (OFF), 11,631 = 11,631 (ON).
- `wait_start` follows `stream_end` by about 35 µs. A layer's stream + wait is the later of the GPU stream and the CPU
  path.
- Joins:
  - device stamps `seq` = tier events `seq` (= `gen`) = `copy_submit.gen`;
  - `copy_submit` per NUMA group: `seq` = that CPU team's first job seq, `a` = its last forced-miss job, `b` = forced
    misses, `c` = CPU-hit lane mask;
  - `copy_issue.b` = DMA lanes;
  - `cpu_submit.a` = 0 hit part, 1 forced-miss part;
  - team index = group.
- The route log aligns to the verify windows at offset 1.

## Results

### Gate-2 (first model, before the approved fixes; `/mnt/nvme1/lane-tier-replay/gate-2.json`)

- **P failed on wait.** Predicted Δwait -66.9 vs measured -46.9 (+43%).
  - Cause: GPU NVMe lanes and the CPU's forced-miss reads share the drives, so the CPU path slows when the GPU streams
    more. The model had assumed independent, overlapping paths.
  - Fit on ON instead of OFF: a forced miss lands at first byte 2.96 + 3.40 ms per expert, against 2.49 + 1.89 on OFF;
    the GPU stream costs 2.37 ms per NVMe lane against 1.83 on OFF.
  - Records with 4 GPU NVMe lanes: measured wait 2.42 ms against 0.30 predicted (ON).
- **T failed.** A plain per-layer LRU of 170 rows, calibrated on OFF's total NVMe reads:
  - NVMe-cold GPU lanes/verify 12.2 vs 15.1 (OFF) and 17.4 vs 33.2 (ON);
  - delta +5.2 vs +18.1.
  - It matched the total reads (8,751 vs 8,525) but not which misses are cold.
- **E failed.** Δstream +19.8 against +51.7.

### After approved fixes 1 and 2 (`/mnt/nvme1/lane-tier-replay/gate-5.json`, commit `3410d3bc40` and later)

- **P: PASS.**

  | check | predicted | measured |
  |---|---|---|
  | Δstream/verify | +52.59 ms | +51.68 ms |
  | Δwait/verify | -54.15 ms (15%) | -46.92 ms |
  | stream layers, OFF / ON | 12.28 / 23.68 (exact) | 12.28 / 23.68 |
  | in-sample OFF stream | 52.87 | 52.86 |
  | in-sample OFF wait | 143.4 | 136.2 |

  Fitted on OFF, in ms, ± bootstrap SD over 200 resamples of verifies:
  - **GPU stream:** base 0.93±0.11; per NVMe-cold GPU lane 1.47±0.07; per pool-served GPU lane 0.65±0.08; per copy
    lane 0.75±0.02; per NVMe GPU lane × forced CPU miss 0.088±0.011.
  - **CPU hit (RAM-warm) job:** 0.02±0.02 + 0.099±0.019 per lane + 0.439±0.019 per route; it starts 1.12±0.03 ms after
    the post.
  - **CPU forced-miss (NVMe) job:** 0.29±0.01 + 0.261±0.013 per route.
  - **Forced-miss landing:** first byte 2.11±0.17, + 1.76±0.08 per CPU miss, + 1.09±0.07 per GPU NVMe lane of the
    record.
  - **Wait:** ends 1.04±0.03 before max(stream, CPU end).

  Implied read rates for a 13.3 MB row:
  - NVMe GPU lane: 9.0 GB/s;
  - CPU miss: 7.6 GB/s;
  - copy lane: 17.8 GB/s, above a Gen3 x16 slot. That term is the stream's marginal growth per copy lane, not a
    transfer time.
- **T: FAIL** (code-derived tier, rules 1-7 below).

  | NVMe-cold GPU lanes/verify | predicted | observed | error |
  |---|---|---|---|
  | OFF | 9.94 | 15.10 | -34% |
  | ON | 14.07 | 33.22 | -58% |
  | ON - OFF | +4.13 | +18.12 | -77% |

  - Lane-level precision 0.38 / 0.61, recall 0.25 / 0.26.
  - The replay made 990 promotions in OFF, against the counters' 3,201. Most landed rows were already mapped in the
    simulation, a sign that it is too warm.
- **E: FAIL.** Δstream +17.2, Δwait -30.0.
- **Why T fails: departure classification** (`departures` command, `/mnt/nvme1/lane-tier-replay/departures-2.json`).
  For each clean NVMe-cold GPU lane, how its row had left RAM:

  | arm | traced decode eviction | untraced, prefill between (a gap between verifies over 500 ms) | never seen |
  |---|---|---|---|
  | OFF | 616 (41%) | 640 (43%) | 240 (16%) |
  | ON | 1,448 (44%) | 1,205 (36%) | 646 (20%) |

  - No untraced departure happened without a prefill in between.
  - So 56-59% of the cold GPU lanes left RAM through prefill admissions, which emit no event. "Never seen" is most
    likely the same cause, earlier.
  - This is the stop condition: T fails because the prefill routing is missing.

## Approved next steps and how far each got

1. **Shared-drive pricing term.** Done (`693facc98c`). P passes. It is the last term of the stream and landing fits
   above.
2. **Tier rules from the code.** Done.
   - Rules 1-9 with citations are in the `TIER_RULES` comment block in the script (`814cd3684b`).
   - Per-node capacity: from the server log's "Pinned host expert cache startup" line (rows per layer; the bound-bytes
     share per node) and rule 2. That gives about 103 / 87 usable rows per layer on nodes 0 / 1.
   - `spec_land` landings: promoted at landing at the run's measured rate (spec_promoted_early + spec_promoted over
     spec_landed: 0.512 OFF, 0.506 ON), against the row's last hot set.
   - The trace's own evictions outside the verifies are applied as observed.
3. **No new capture.** Respected.
4. **T still fails, and the missing prefill routing is the cause.** Stopped as instructed. The capture team-lead may
   take to the owner needs: per prefill chunk, the experts it admitted to the RAM tier (or its routes), plus the decode
   route log as in this A/B (`spec_margin_capture.py --route-log`, on `lane-head-ab`).
   - An untraced `take_admit_slot_locked` / promotion victim event in the instrumented build would also close it.
5. **Not started:** the evaluation (option (b), the head, the oracle on C2/base), and option (a) (retraining with a
   RAM-tier feature).

## RAM tier rules found (code the A/B ran: `lane-head-ab` `d269f7aa90`)

Paths are under `python/sglang/kernels/jit/csrc/moe/expert_stream/host/` unless given.
1. **Per NUMA group.** A row's slots are one contiguous range per node (`srt/layers/moe/host_numa.py:300`
   group_ranges, `:267` slot_nodes). Expert e belongs to group e % 2 (`../lease_layout.h:52` home, `:148`). A victim
   is taken only in the miss's own group range (`ram_tier.h:2114` take_victim_locked).
2. **Staging and pool slots.** Each group range gives min(8, range - 1) staging slots (`ram_tier.h:1044`
   reserve_staging; the victim lanes under spill, `srt/layers/moe/exl3_ram_miss.py:975`, `:1001`), then `share` (2)
   pool slots (`ram_tier.h:1084`). Neither ever holds a mapped row.
3. **Victim choice.** A free slot first, else the LRU READY slot whose expert is not VRAM-hot (the record's hot bitmap,
   `ram_tier.h:594`), not routed by the record, and not filling (`ram_tier.h:2120-2128`). **A VRAM-resident expert
   keeps its RAM row while hot.** No code path frees a row because it went hot. With no victim, the miss is read but
   not cached (`ram_tier.h:2110`).
4. **Recency.** Every routed expert held in RAM is restamped at each record, VRAM hits included (`ram_tier.h:2188`
   collect_wanted_locked, `:2100` touch_request, `:2090` stamp_routed_locked). The tick is per group.
5. **Every demand miss is cached, GPU or CPU lane** (`ram_tier.h:2998` reserve_victims_locked: forced misses first).
6. **Prefetch.**
   - A landed L+2 row sits in a pool slot. It is served to a GPU lane (`ram_tier.h:2369` point_pooled_locked, called at
     `:3040`) or taken by a forced CPU miss (`:2419` take_pooled_locked, at `:3020`).
   - An unused landed row is promoted when its row's delta is claimed (`ram_tier.h:2919` promote_landed). It takes an
     LRU victim with no record protection and inherits the victim's stamp (`ram_tier.h:2960`).
7. **Full layers** (0, 1) map every expert.
8. **Prefill admissions** (`SGLANG_DSV41_ENABLE_PREFILL_FILLS=1`; `ram_tier.h:399` assign, `:456` fill_begin) take
   `take_admit_slot_locked` (`:1982`) → `take_slot_locked` (`:2038`), an LRU victim that is not hot. No prefill share is
   set, so nothing bounds how many decode rows one prefill displaces. These emit no event.
9. **Only decode demand misses emit `evict`** (`ram_tier.h:2141` trace_evict). Promotion victims are untraced
   (`ram_tier.h:2949`).

## Paths and commands

- **Branch and commit:** `lane-tier-replay`; last code commit `814cd3684b`. This handoff is the next commit.
- **divix01 worktree:** `/data/models/slang/nvfp4-work/wt-lane-tier` (detached at the pushed commit). The code citations
  above refer to `/data/models/slang/nvfp4-work/wt-lane-head-ab` and the laptop worktree `sglang-nvfp4-lane-head-ab`.
- **Inputs:** `/mnt/nvme1/lane-head-ab-20261016/ab-off` and `ab-on`: route log `stages.jsonl`, `events.*` job traces,
  `counters.json`, `server.log`, `probe.json`. `serve_compare.py` is imported from the same directory.
- **Outputs:** `/mnt/nvme1/lane-tier-replay/`:
  - `gate-2.json` (first model);
  - `gate-3.json` (shared-drive P, first code tier);
  - `gate-4.json` / `gate-5.json` (promotions at landing);
  - `departures.json`, `departures-2.json` (`-2` has the prefill-gap marking).
- **Commands** (on divix01):
  ```
  cd /data/models/slang/nvfp4-work/wt-lane-tier
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH=$PWD/python
  /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)"
  taskset -c 56-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/lane_tier_replay.py gate \
      /mnt/nvme1/lane-head-ab-20261016/ab-off /mnt/nvme1/lane-head-ab-20261016/ab-on --json OUT.json
  taskset -c 56-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/lane_tier_replay.py departures \
      /mnt/nvme1/lane-head-ab-20261016/ab-off /mnt/nvme1/lane-head-ab-20261016/ab-on
  ```
  The gate takes about 10 s plus about 1 min of bootstrap. The page cache of the A/B traces was evicted afterwards.
- **Rules:** `.claude/rules/divix01-run-protocol.md`; cores 56-63; `CUDA_VISIBLE_DEVICES=""`; outputs under
  `/mnt/nvme1`; no runs or captures without team-lead; never touch production or `DSV_OPTIMIZATION.md`.
