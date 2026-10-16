# Drip into the hot cache: design note (drip-hot)

Branch `drip-hot`, based on `origin/ram-prefetch-pool-serve` at `fd5dfa5bef`. Design only: no code is written yet.

Path prefixes used below:
- `ES/` is `python/sglang/kernels/jit/csrc/moe/expert_stream/`.
- `MOE/` is `python/sglang/srt/layers/moe/`.
- `RES/` is `python/sglang/kernels/jit/csrc/moe/expert_residency/`.

Captures: `breakdown-20261015/{base,drip}` and `split-ab-20261014/C2`, both under
divix01:`/data/models/slang/nvfp4-work/ram-prefetch/`.

## 0. Summary

1. **The premise behind "land drips in a victim slot and commit residency" is wrong.**
   - A *used* drip already enters residency today. The post types a scratch lane only when the lane has a DIRECT
     victim destination (`ES/lease_device.cuh:402`).
   - C1s copies the scratch slot into that victim (`ES/lease_kernels.cuh:289-292`, `MOE/exl3_ram_miss.py:751-759`).
   - The DIRECT commit leaves out only CPU lanes (`RES/direct_gather.cuh:202`, `ES/lease_device.cuh:104-106`), so
     the used drip is committed.
   - Measured on the drip capture: one verify after their target, **50% of landed drips are VRAM-resident**, which
     matches `drip_used/drip_landed` = 49%. The DSV_OPTIMIZATION.md 3.5 line "a used drip misses again next verify"
     should be retracted.
2. **Why the drip was neutral.**
   - It moved 44 lanes/verify to `HIT_SCRATCH`. Those lanes came out of DMA lanes (95.3 to 71.8) and CPU-RAM lanes
     (329.0 to 299.2).
   - The verify's critical path is the CPU's NVMe reads for forced misses. That share of the last dependency rose
     from 40% to 46%, and its share of wait from 50% to 56%.
   - drip3 vs base: -0.54 ms/token (CI -2.2..+1.8), wait -2.1 ms/verify.
   - The lane-cost fit says 30 fewer CPU-RAM lanes should save about 8 ms/verify. The measured saving is about 2. In
     practice a CPU-RAM lane removed is worth about 0.07 ms here, not the fit's 0.28 ms.
3. **What the drip actually wastes.**
   - 80.6% of landed drips are routed at their target verify; by pick rank this is 86/81/74/68%.
   - Only 49% are used. The rest of the routed ones fall past the spill's 8 victim lanes, where a lane is a forced
     CPU lane and cannot be a scratch lane. That is about 26 routed landings per verify thrown away.
   - Even a used drip takes up one of the 8 victim lanes.
4. **Recommendation: design B, "adopt at the gather".**
   - At the target layer's gather, *before routes are planned*, each landed drip whose expert this forward routes is
     copied from scratch into a hazard-checked shortlist victim and committed. The planner then sees it as a hit.
   - The victim choice uses the hot cache's own shortlist and hazard check, so residency has one authority.
   - Nothing is evicted before routing is known, and an abandoned drip costs no victim.
   - The scratch stays, possibly smaller.
5. **This differs from the owner's literal approval.** That approval was design A: land straight into a victim
   slot, with the scratch gone.
   - A must evict before the target's routing is known.
   - On measured numbers that costs about 11 hits/verify at 2 picks and about 23 at 4 picks.
   - Returning the scratch VRAM (24 slots) is worth about +4.5. So B is the better first build.
   - A is specified in 2.9 in case the owner still wants it, and the replay (section 3) scores both.
6. **Expected gain is small.** B removes about 26 forced CPU lanes per verify, the routed landings that spill
   today. Extending the lanes' victim window would free about 15 more, but that is off by default (2.2, 2.10): the
   freed lanes risk NVMe-cold rows at 1.47 + 1.09 ms each.
   - At the drip3 marginal (0.07 ms/lane) that is about 2 ms/verify, or 0.5 ms/token: below the 1.6 ms/token noise
     of a served pair.
   - At lane-tier-replay's CPU RAM-hit job cost (0.099 per lane + 0.439 per route, at least 0.54 per lane) it is up
     to about 14 ms/verify, or about 3.6 ms/token, if those jobs are on the critical path.
   - The consolidation does not touch the critical path, which is NVMe-cold forced misses.
   - The offline replay gate (section 3) should decide whether to build at all.

## 1. How it works today

### 1.1 One layer's graph gather, in stream order

`MOE/expert_stream.py:_gather_graph` (1574-1739), on the forward's stream:

1. **First streamed layer only: the boundary.** `on_graph_forward` runs `_apply`
   (`MOE/expert_residency_gpu.py:520-532`, `646-695`).
   - It decays `insert_scores` by `0.98^tokens` and adds the window's route counts (650-659).
   - Under DIRECT it re-ranks every layer's victim shortlist (`_rank_victims`, 864-905).
   - The ranking puts free slots first, then residents in ascending key: routed in the last window, insert score,
     -expert (`_expert_rank_keys`, 933-941).
   - With spill, only the first `victim_lanes` columns are valid (900-902). Production uses
     `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES=8` (`benchmarks/dsv41_baseline/arm_env.py:288`): 8 victims per
     layer, one lane per route (36).
2. **Plan.** `plan_graph_routes_fused` reads `expert_to_slot`, the layer's row of `mapping` (1628-1646). A mapped
   expert is a hit, and its route is remapped to its slot. Misses become lanes, sorted by `miss_keys` with the
   highest first (`enable_miss_order`, 913-931).
3. **Destinations.** `fused_gather_destinations` (1665-1672; kernel `RES/direct_gather.cuh:94-161`).
   - A shortlist entry whose slot any route of this forward reads is a hazard and is dropped (`:122-128`).
   - Usable entries move to the front, and live lane k takes usable entry k.
   - With spill, a lane with no victim gets destination -1 (`_idle_destination`, 907-911) and the post makes it a
     forced CPU lane.
4. **The lease chain** (`MOE/exl3_ram_miss.py:post`, 690-772):
   - **post** (`ES/lease_kernels.cuh:141-344`). It applies the RAM-map delta and matches lanes against the drip's
     scratch table (193-195). It types the lanes (`ES/lease_device.cuh:367-451`): forced lanes come from the first
     -1 destination (199-205). It writes the hot sidecar, a bitmap of this layer's VRAM residents from
     `slot_to_expert`, which the RAM tier uses to protect their RAM rows (258-272). Then it publishes the record.
   - **spec_score**, forked onto `spec_stream` (741-747; `kernels/ops/moe/expert_stream_transport.py:3666-3690`).
     It scores the target row and runs the drip select kernel (3738-3762; `ES/spec_score.cuh:281-356`). The select
     picks the `picks` best experts of the target row that are RAM-mapped in the device map and not in that row's
     `slot_to_expert`.
   - **C1** (`HIT_SM`), then **C1s** (`HIT_SCRATCH`: drip scratch to destination, 751-759), then **S**, **CW**, the
     gate wait and **CC**.
   - **spec_join** (771) joins the scorer before anything can post again.
5. **Commit** (1720-1729; `RES/direct_gather.cuh:172-233`). It is forked onto the MoE side stream when that is
   active, which joins before the next layer's gather.
   - Every live, delivered, non-CPU lane is committed: the old expert of its destination is unmapped, then the new
     expert is mapped, its slot set READY and its generation bumped.
   - `delivered` is the plan's count (`MOE/exl3_ram_miss.py:772`).

Residency (`mapping`, `slot_to_expert`, `slot_state`, `slot_generations`, `victims`, `victim_valid`) is device
memory written only by graph kernels, in stream order. No host thread writes it during decode.

- The gather sees a resident only through `mapping`.
- A slot's bytes are written only as a live lane's destination (C1, C1s, S, the DMA), and that destination is always
  a hazard-checked victim. So nothing can overwrite a slot a route of this forward reads.

### 1.2 The drip (`ES/host/gpu_drip.h`; LEASE_PROTOCOL.md "The GPU drip")

- **Threads.** All drip state belongs to the copy thread: `CopyEngine::run` calls `drip_step` between its jobs, with
  `demand_idle` = rings empty and the DMA idle (`ES/host/copy_engine.h:504-519`). The service threads read only
  `ident_` (`gpu_drip.h:137-140`).
- **Step** (`gpu_drip.h:159-173`), in order:
  1. Scan the posted demand records (238-264). A post of a drip's target row ends that drip (286-301, 338-343).
  2. Admit the picks of scored records (419-477). A pick is admitted only if the host mirror still maps it, nothing
     is already dripping it, and a free slot exists. Admission records the mirror generation.
  3. Free quarantined slots (365-382), then poll the one mark in flight (552-575).
  4. Queue a table store (493-509), else issue one chunk of the oldest filling slot (519-550). A chunk is issued
     only when `demand_idle` holds and no device pull window is open (307-315).
- **Source check.** The mirror generation is bumped after every map store (`ES/host/ram_tier.h:1922-1928`), eager
  `publish_map` included (1931). The drip re-checks it before its first chunk and after its last (226-229, 525-529,
  579-583).
- **Publish and release.**
  - A landed slot's word, `tag<<32 | row<<16 | expert`, is stored on the copy stream after its bytes (493-509).
  - The target's post clears it. The slot is reused only once record H+2 is handled (593-598, 365-382).
  - The service fail-stops a scratch lane whose slot does not hold the expert, and counts `drip_used`
    (`ram_tier.h:2221-2231`).
- **RAM tier.** A RAM victim is a free slot, or the LRU READY slot that is not VRAM-hot, not routed by the record,
  and not filling (`ram_tier.h:2109-2136`). VRAM-hot comes from the post's sidecar (`apply_gpu_hot`, 592-598).

### 1.3 What the drip did, measured

Capture `breakdown-20261015/drip` against `base` (`expert_breakdown.json`, 131 vs 130 verifies), with the drip3
config: 24 slots, 4 picks, 256 KiB chunks.

| per verify | base | drip |
|---|---|---|
| VRAM hits (unique) | 348.9 | 358.5 |
| scratch (`HIT_SCRATCH`) lanes | 0 | 44.1 |
| DMA lanes | 95.3 | 71.8 |
| CPU-RAM lanes | 329.0 | 299.2 |
| CPU-NVMe lanes | 71.1 | 71.7 |
| NVMe GPU lanes | 16.6 | 17.7 |
| H2D demand / drip MB | 1466 / 0 | 1185 / 1143 |
| last dependency `cpu_miss:read` (wait share) | 40% (50%) | 46% (56%) |

- **Counters.**
  - issued 15,052; landed 11,827; used 5,770.
  - abandoned 8,147: 8,129 because the target posted, 18 because the source changed.
  - paused 1,703; skipped bytes 83.3 GB; no_slot 0.
  - Per verify: 90 landed, 62 abandoned.
- **Lane cost fit** (`cache-tiers-20261015/drip.lanecost.txt`), in ms/lane: drip 0.02, DMA -0.05, CPU-RAM 0.28,
  CPU-NVMe 1.06.
- **Served result** (DSV_OPTIMIZATION.md 3.2 table): drip3 vs base -0.54 ms/token, clean CI -2.2..+1.8, wait -2.1
  ms/verify.
- **The existing replay over-predicted.** `split-ab-20261014/C2/gpu_drip_replay.txt` forecast 12.6 ms/verify for
  top-4. Any new replay is priced against that miss (section 3).

**Drip precision.** Computed from the drip events and the route records:
- **Alignment.** Route records lag the event clock by 2 records; at that offset 93% of `drip_use` experts are
  routed, so these figures carry about 7% alignment noise.
- **Landed drips:**
  - 80.6% are routed at their target verify.
  - 5.5% are already resident when it starts.
  - One verify later, 50% are resident.
- **By pick rank 0/1/2/3:**
  - routed 0.86 / 0.81 / 0.74 / 0.68;
  - landing rate 0.87 / 0.68 / 0.48 / 0.33.
- **Landed but unrouted at the target:** 21.5% are routed at v+1, and 35.2% within v+1..v+3.

## 2. Proposed flow: design B, adopt at the gather

The drip (picks, copy, pauses, source check, scratch table, quarantine) is unchanged. What changes is where a landed
drip enters residency: at its target layer's gather, before planning, as a hit rather than a lane.

### 2.1 Flow at the target layer's gather (row r)

New kernel `drip_adopt(row r)`: one block, plus C1s's existing segment copy. It runs at the top of `_gather_graph`,
before `plan_graph_routes_fused`, captured.

1. **Find landed drips.** Load the `slots` scratch words (device memory, as `match_scratch` does). Keep the words
   naming row r whose expert this forward routes (`topk_ids`) and that `mapping` does not hold.
2. **Choose victims.** Run the hazard test of `direct_gather_destinations` against this forward's route slots. Take,
   in shortlist order, the usable entries: valid, and no route reads their slot. Adopted drip i takes usable entry i.
   - Mark the taken columns invalid.
   - By default the lanes' window is not extended (2.2), so the lanes find `victim_lanes - k` victims, the GPU-lane
     count today's scratch lanes leave.
   - With `..._ADOPT_EXTEND`, under spill, the kernel validates the next ranked columns past `victim_lanes`, one per
     adoption. That needs `victim_ranked`, the pre-truncation `ranked.values < never` mask, which `_rank_victims`
     keeps only when the knob is on.
3. **Copy.** Write `{scratch slot to victim slot}` pairs into `scratch_rows_1 / scratch_dst_1 / scratch_go_1` and
   launch C1s's `copy_expert_row_segments_gpu(drip_segments, ...)` (today at `MOE/exl3_ram_miss.py:753-759`) here, in
   place of in the chain.
4. **Commit** with `_commit_gather`'s writes: unmap the victim's expert, map the drip expert, set the slot READY, bump
   its generation, and count `drip_adopted`. This is issued after the copy, on the same stream, as `_commit_gather`
   requires (1720-1722).
5. **Plan.** The planner now sees the drip expert as a hit. It gets no lane, does not count against the 8 victim
   lanes, and cannot be a forced CPU lane.

The post is launched with `scratch_slots = 0`, so it never types `HIT_SCRATCH`. The service's scratch-lane check and
`drip_used` become unreachable in this mode. `drip_adopted` replaces `drip_used`.

### 2.2 How a drip claims a victim

It uses the same ranking as insert-on-miss: the boundary's shortlist with the gather's hazard check. Adoptions take
the head of the usable shortlist and the lanes take the entries after them, so the slots evicted in a verify are
still the lowest-ranked ones. This only changes who gets which.

- **Capacity floor.** Spill needs `capacity >= 2 * victim_lanes` (`MOE/expert_residency_gpu.py:403-409`). With the
  extension knob on, up to `picks` adoptions per layer come on top, so it needs `capacity >= 2 * victim_lanes +
  picks`: 16 + 4 = 20 against today's 21-22 slots per layer. The gate checks whichever floor applies.

**Default: no extension (revised after the RAM-tier rules, 2.10).** Adoptions take columns from the lanes' own
`victim_lanes` window and do not extend it. The lanes then keep the GPU-lane count today's drip arm has, since a
scratch lane also used one of the V.
- The gain is then only the routed landings that today spill to the CPU.
- No extra miss is pulled into the GPU prefix, where it could be NVMe-cold.
- Extending the window is a knob (`..._ADOPT_EXTEND`). The replay decides it, priced with the costs in 2.10.
- Without extension the floor is unchanged (`capacity >= 2 * victim_lanes`), because adoptions use at most the V
  columns the lanes would have used.

### 2.3 Commit and visibility

Residency changes only in stream order inside the graph (2.1 steps 3-4), before the plan reads `mapping`. That is
the same place and the same order as a lane's commit. No host write to residency is added.

**Safety against slot reuse.**
- **The victim.** No route of this forward reads it (hazard test). Earlier forwards ended before this gather, in
  stream order. The MoE side stream joins before the next layer's gather, so the commit fork of an earlier layer is
  complete.
- **The scratch slot.** The adopt copy reads it before this record's post. The copy thread clears the word only after
  seeing the post, then reads head H, and frees the slot when record H+2 is handled.
  - Any adopt that read the word before the clear belongs to a record no later than H+1, and its copy completed before
    that record's post.
  - Post H+2 starts after chain H+1 ends.
  - So the existing quarantine rule (`gpu_drip.h:593-598`) covers the new reader unchanged.
- **The device tables.** Scratch words are device memory written by `cuStreamWriteValue64` after the copy stream's
  earlier chunks (`gpu_drip.h:493-509`). The adopt kernel loads them as atomics, so a word it sees has its bytes in
  device memory before the copy kernel launched after it reads them. This is the same argument as `match_scratch`.

### 2.4 Abandoned drips and changed sources

- **Unchanged.** An unlanded drip is abandoned when its target posts. A source change abandons it before the first
  chunk or before publication (`gpu_drip.h:338-361, 525-529, 579-583`).
- **New.** An abandoned drip never touched residency, so it costs link bytes only, as today (83 GB skipped in the
  capture). It never costs a victim. This is the main safety gain over design A.
- **Lapped records.** `lose_all` clears every published word, as today. An adopt that already ran is ordinary
  residency.

### 2.5 The hazard check and current-verify routed residents

The adopt step runs the same hazard test with the same inputs as the gather destinations: the forward's routes and
`expert_to_slot` before adoption. So it never takes a slot a route of this forward reads, and the subsequent
`fused_gather_destinations` sees the taken columns as invalid.

- A resident routed in the previous window ranks last (`_ROUTED_RANK_OFFSET`, `MOE/expert_residency_gpu.py:48`).
- In both captures the first five usable entries were routed in the previous verify 0-3% of the time (section 3).

### 2.6 The score the adopted expert enters with

It enters with its own `insert_scores` entry. That entry exists for every expert (it is `[layers, experts]`, line
290) and gets this forward's routes at the next boundary, exactly like a lane insert. There is no boost.

- **Routed only.** Only experts routed in this forward are adopted, so the next ranking sees them as routed in the
  window, which ranks them last for one verify.
- **Unrouted landed drips.** These are dropped as today (2.8 knob). Their 21.5% next-verify recurrence beats a
  rank-0 victim's, but not by enough to pay for a copy without the replay showing it.

### 2.7 Scratch slots stay, maybe fewer

- **Peak demand.** Drips live from layer t's scoring to the post of target t+2, plus two records of quarantine.
  - At 4 picks that is about 20 of 24 slots (`drip_no_slot` 0 at 24).
  - At 2 picks it is about 10-12, which would return about 12 slots (160 MB) to `SGLANG_MOE_HOT_GPU_MB`.
- **A/B.** The A/B keeps the slot count and hot budget identical between arms. Shrinking is a follow-up arm read off
  `drip_no_slot`.

### 2.8 Knobs

The adopt path has one switch (section 5). A second, `..._ADOPT_UNROUTED`, also adopts landed drips that this
forward does not route. Each one takes a victim only if its key outranks that victim's under `_expert_rank_keys`, so
the ranking stays the one authority. It stays out of v1 unless the replay shows a gain.

### 2.9 Design A, if the owner prefers it: reserve, fill, adopt

The scratch goes away (24 x 13.27 MB = 318 MB back to the hot budget).

1. **Reserve.** The drip select kernel, on `spec_stream` at layer t for row r = t+2, becomes select and reserve.
   - For pick k, it takes the next shortlist-head column of row r under the rule in section 3: unmap the victim, set
     the slot `LOADING` (`HotCacheSlotState.LOADING = 2`, `MOE/expert_hot_cache.py:110-120`), and invalidate the
     column.
   - Under spill, it validates one more ranked column past `victim_lanes`.
   - It writes `{tag, slot, expert}` to a per-row pinned reservation table, under a seqlock, with `R = picks`
     entries per row. A table, not a ring, so it never laps.
   - **Why on `spec_stream` is safe.** Row r's residency is read by row r's gather, row r's commit and the next
     boundary. All three run on the main stream after `spec_join` (`MOE/exl3_ram_miss.py:771`). Nothing on the main
     stream between post t and the join reads row r.
2. **Fill.** The copy thread serves each new tag: it copies the RAM row to `copy-table dst + slot x bytes`, the same
   destination layout as `HIT_COPY` (`ES/host/copy_engine.h:59-64`).
   - It answers every tag with exactly one terminal word in device memory, `tag | landed` or `tag | aborted`, stored
     on the copy stream after any chunk in flight.
   - It aborts on target post, source change, `lose_all` and stop.
3. **Adopt** at row r's gather, before planning. A landed word maps the expert and sets it READY. Aborted, or a
   duplicate (the expert is already mapped), frees the slot.
4. **Settle.** At the top of `_rank_victims`, every boundary first adopts landed reservations and frees aborted ones
   across all rows, then ranks.
   - `LOADING` is neither FREE nor READY, so it is never shortlisted (879-882) and never a destination.
   - No kernel reads a `LOADING` slot. Only the DMA writes it, and the slot leaves `LOADING` only on a terminal word
     ordered after the DMA.
5. **Cost.** Every reservation evicts before routing is known, including the drips that never land.
   - 2 picks: about 77 reservations/verify x P(victim routed) of about 0.14, so about 11 lost hits/verify.
   - 4 picks: about 152 x 0.15, so about 23.
   - The VRAM returned is worth about +4.5 hits (3.5: +1 slot/layer is about +7.5 hits).
   - It also needs the floor `capacity - picks >= 2 * victim_lanes`.
   - Open audit: confirm that no eager EXL3 DIRECT path writes a non-shortlist slot. `_gather_cached`,
     `MOE/expert_stream.py:2457-2460`, and the prefill boundary at `MOE/expert_residency_gpu.py:679-692` look
     clean, but this is unverified.

### 2.10 RAM-tier interplay (TIER_RULES from `lane-tier-replay` 814cd3684b)

These rules come from the TIER_RULES block of `analysis/dsv41-drive/dspark/lane_tier_replay.py`. I re-checked each
citation on this base (`fd5dfa5bef`); the lines match.

- **VRAM-hot experts keep their RAM row.** `take_victim_locked` skips `tier.hot` (`ES/host/ram_tier.h:2126`).
  - The hot set has a fixed size (one per VRAM slot), so an adoption swaps which RAM rows are pinned: the adopted
    expert's row becomes pinned and the VRAM victim's row is released at the row's next post (`apply_gpu_hot`,
    592-598). An adoption costs no extra RAM row; it trades the victim's protection for the drip's.
  - The victim is the one at risk. Once unpinned, it ages out of RAM by LRU and, if it recurs, returns NVMe-cold.
    This is what every lane insert does today; B adds about 26 such swaps per verify.
  - Design A also unpins its victims early: a `LOADING` slot has `slot_to_expert` = -1, so it is not hot.
- **Routed RAM experts are restamped every record**, VRAM hits included (`stamp_routed_locked`, 2090; via
  `collect_wanted_locked`, 2188). A shortlist-head victim was not routed in the last verify, so its RAM stamp is
  older than any recently routed row's. It is among the likelier RAM evictions once unpinned. The replay must model
  this (section 3).
- **Per-node tier**, `home = e % nodes` (`ES/lease_layout.h:52`): about 103 usable rows per layer on node 0 and 87
  on node 1. The drip's RAM source is its home node's slab, so no change.
- **Every demand miss is cached in RAM** (`reserve_victims_locked`, 2998). Prefetch pool rows serve GPU lanes
  (2369) or forced CPU misses (2419). About 50% of landed rows are promoted unused, inheriting their victim's stamp
  (2919, 2960).
  - Drip picks need the device map's `ram_slot >= 0` (`ES/spec_score.cuh:293`) and the host mirror at admission
    (`gpu_drip.h:456-459`). A pool row not yet promoted is never a drip source.
- **Prefill admissions displace any number of decode rows** when no prefill share is set, and emit no event
  (`assign` 399, `fill_begin` 456, `take_admit_slot_locked` 1982, `take_slot_locked` 2038).
  - After a prefill, fewer of a row's experts are RAM-mapped, so the drip has fewer eligible picks. A pick is never
    cold: the RAM-mapped requirement and the generation check (`gpu_drip.h:226-229`) see to that.
  - The risk is on the lane side. In the lane-head A/B, 56-59% of NVMe-cold GPU lanes had left RAM through a
    prefill.
  - That is why 2.2 now defaults to not extending the lanes' window: a lane freed by an adoption would most likely
    be filled after a prefill by exactly such a row.
- **Only decode demand misses emit `evict`** (`trace_evict` 2141). Promotion victims are untraced (2949). The
  replay's RAM model cannot observe promotion evictions directly and must use lane-tier-replay's reconstruction.
- **Measured costs** (lane-tier-replay fit on the lane-head A/B, ms). These replace the flat 0.07/0.28 brackets in
  sections 0 and 3 for pricing:
  - An NVMe-cold GPU lane adds 1.47 to stream, and each one delays a CPU miss's landing by 1.09.
  - A pool-served GPU lane adds 0.65; a copy lane on the link, 0.75.
  - A CPU RAM-hit job costs 0.099 per lane plus 0.439 per route; a forced-miss job, 0.29 per lane plus 0.261 per
    route.
- **What the costs imply.**
  - A spilled, routed landing that B turns into a hit removes a lane from a CPU RAM-hit job: 0.099 plus 0.439 per
    route it carries. That is at least 0.54 ms where that job is on the critical path, and nothing where it is not.
  - Extending the window would add up to 1.47 + 1.09 ms for each freed lane that lands on an NVMe-cold row. That
    outweighs the saving, so the window stays unextended unless the replay shows the freed lanes are mostly RAM-warm.
  - The 0.75 ms cost of a link copy lane confirms the link is not free. Today's drip moved as many bytes as demand,
    which section 4 already gates.

**Rejected, design C: a per-layer spare row, swapped in at the gather.** No eviction happens before routing, but 40
spare rows cost 531 MB (more than the scratch) and cap the drip at one landing per layer per verify. The replay
scores it anyway.

## 3. Churn control

**B's rule.** Adopt a landed drip only if this forward routes it. Its victim is the lowest-ranked usable shortlist
entry, which by the hazard test no route of this forward reads.

- The drip is used now, with certainty; the victim is unused this verify, also with certainty.
- Over the next three verifies, an evicted rank-0..3 victim recurs 26-39% of the time (table below). An adopted drip
  carries a fresh route, and misses recur at about 40% (3.8).
- So B never trades a resident needed in this verify for a drip, and its long-run trade is about even. This is the
  same trade every GPU lane makes today.

**A's rule.** It must decide before routing, so it needs a value comparison. Reserve for pick k into the next head
column only if:
1. the slot is FREE, or its resident was not routed in the last window;
2. the resident's insert score is at most `theta_k`, a per-pick cap set from the replay, with
   `P_land(k) x P_routed(k) > P(victim routed this verify | score)`;
3. a reservation entry is free.

Measured inputs for that comparison:

| | pick 0 | pick 1 | pick 2 | pick 3 |
|---|---|---|---|---|
| P(lands before target) | 0.87 | 0.68 | 0.48 | 0.33 |
| P(routed at target), landed | 0.865 | 0.821 | 0.762 | 0.716 |
| value of reserving | 0.75 | 0.56 | 0.37 | 0.24 |

Victim, by rank in a proxy of the shortlist. The proxy ranks residents by (routed in the previous verify, a 0.98^6
per-verify decayed route score, -expert). C2: 32,400 (verify, layer) pairs; the drip capture agrees within 0.01-0.02.

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 7 | 8 | 11 |
|---|---|---|---|---|---|---|---|---|---|
| P(routed this verify) | 0.114 | 0.146 | 0.167 | 0.186 | 0.214 | 0.245 | 0.288 | 0.316 | 0.397 |
| P(routed within 3 verifies) | 0.262 | 0.321 | 0.362 | 0.394 | 0.438 | 0.477 | 0.533 | 0.551 | 0.603 |
| routed last verify | 0.000 | 0.000 | 0.001 | 0.003 | 0.012 | 0.039 | 0.201 | 0.344 | 0.804 |

- Picks 0-1 clear the rule against any head victim. Picks 2-3 are marginal once the victim's next-3-verify value
  counts. A at 2 picks is the only sane setting.
- These numbers come from a throwaway script, divix01:`/mnt/nvme1/drip-hot-tmp/drip_churn2.py`, run under
  `taskset -c 56-63`. It is not committed; the replay below replaces it.

**Offline replay before any GPU run.** Script `analysis/dsv41-drive/dspark/drip_hot_replay.py` on `drip-hot`,
built on cache-tiers' `vram_cache_replay.py`.

- **Model.** DIRECT shortlist by insert score, the hazard test, `victim_lanes = 8` spill, the fixed split table
  `A` from `breakdown-20261015/run.sh`, and miss-key lane order.
- **Inputs.**
  - The drip capture's real admits and landings, aligned at the offset where `drip_use` is at least 90% routed.
    The replay stops if no offset reaches that.
  - `base` as the no-drip reference.
  - C2 for victim statistics.
  - Optionally, C2 with picks re-scored offline from `router.x.bin` and the target layer's gate weights.
- **RAM tier.** Use lane-tier-replay's TIER_RULES model (2.10) to classify each lane as VRAM, RAM-warm, pool or
  NVMe-cold. Above all, it must follow the VRAM victims B releases, which become evictable from RAM once unpinned.
  If that model has not passed its own gate, report results with RAM state taken from the capture's lane kinds, and
  say so.
- **Arms.** Today (scratch lanes); B without and with the window extension; B with unrouted adoption; A (picks 1-4,
  a theta sweep); and C (1-2 spares).
- **Outputs per verify.**
  - VRAM hits; lanes by kind (DMA, NVMe GPU, CPU-RAM, CPU-NVMe; forced vs split); inserts, evictions and lost hits
    (victims routed in the same verify).
  - Device-to-device copies; link MB.
  - Priced two ways:
    - The lane-tier-replay costs (2.10): CPU RAM-hit job 0.099/lane + 0.439/route; forced miss 0.29 + 0.261;
      NVMe-cold GPU lane +1.47 stream and +1.09 on a CPU miss's landing; pool lane 0.65; link copy lane 0.75.
    - The drip3 empirical marginal (about 0.07 ms per CPU-RAM lane), as the conservative bound.
- **Validation gate.** "Today" must reproduce the drip capture's per-verify kinds within 5%: VRAM 358.5, scratch
  44.1, DMA 71.8, CPU-RAM 299.2, spilled 277. Removing drips must reproduce `base`: 348.9 / 95.3 / 329.0. If not,
  stop and report.
- **Build gate.**
  - Build only if the chosen design cuts CPU lanes by at least 30/verify against "today".
  - It must not add NVMe GPU lanes or stream layers.
  - It must add at most 0.5 ms/verify of GPU time (adopt launches plus copies).
  - Otherwise recommend not building.
- **Box rules.** Cores 56-63, `OMP_NUM_THREADS` set, outputs and `TMPDIR` under `/mnt/nvme1`, page cache evicted
  after reading the captures. No torch is needed; if it is used, `CUDA_VISIBLE_DEVICES=""` comes first.

## 4. PCIe and drive contention

- **Kept as is.**
  - A chunk is issued only when demand is idle.
  - Nothing is issued during an S/C1 pull window, which closes at the copy gate or the next post.
  - One drip mark is in flight at a time, so a demand copy waits behind at most one chunk.
  - These rules live in the copy thread and B does not touch them.
- **Link load.** The drip3 config moved 1.14 GB/verify, as much as demand. With the replay's 2-pick recommendation
  it is about half. The A/B reports drip MB/verify alongside demand.
- **Shared drives.** The drip reads pinned RAM only and never touches the NVMe drives. The lane-head A/B's failure
  mode can still arrive indirectly:
  - Each adoption frees a victim lane, and the next lane in miss-key order joins the GPU prefix.
  - If that lane is NVMe-cold, it becomes an S stream from NVMe, competing with the CPU's forced-miss reads.
  - Today's key mostly favours RAM-warm experts there, which is the lane-head lesson, but this is unmeasured. The
    replay counts NVMe GPU lanes, and the A/B reports stream layers/verify and NVMe GPU lanes as gating metrics.
  - If they rise, the fix is in lane order, not the drip.
- **Host memory bandwidth.** The drip's DMA reads RAM on the nodes whose CPU teams compute CPU-RAM lanes. In the
  capture, CPU-RAM ms/lane was 0.29 in base and 0.28 in drip: no measurable contention at about 4.4 GB/s of drip DMA.
  The A/B keeps the per-prompt CPU ms/lane flag (serve_compare CPUSLOW).

## 5. Flag, gate, counters, tests, A/B

- **Flag.** `SGLANG_DSV41_ENABLE_GPU_DRIP_ADOPT = EnvBool(False)` in `environ.py`'s GPU drip block (2105-2115).
  `ENABLE` is the verb, after `SGLANG_DSV41_ENABLE_LAYER_FUSION`. If design A is chosen instead it becomes
  `SGLANG_DSV41_ENABLE_GPU_DRIP_HOT`.
  - Unset means no adopt kernel, no `victim_ranked`, and the post's `scratch_slots` and C1s exactly as today:
    bit-identical.
- **Launch-gate refusals** (`python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`, plus the runtime
  checks it mirrors):
  - It needs `SGLANG_DSV41_GPU_DRIP_SLOTS > 0` and the drip's existing prerequisites: the copy engine, and RAM
    prefetch with the GPU scorer (`MOE/exl3_ram_miss.py:1554-1562`).
  - It needs `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2` and `SGLANG_DSV41_ENABLE_LAYER_FUSION`.
  - With `..._ADOPT_EXTEND`, it needs every layer's capacity to be at least `2 * victim_lanes + picks`, and names
    the short layer.
  - The gate test asserts the resolved requirements are EXL3, per the run-protocol rule.
- **Counters.**
  - Device, per layer, through `insertion_counters()` so they reach the async residency snapshot: `drip_adopted`,
    and the tripwire `drip_adopt_no_victim`. That keeps out lane-head's untelemetered-counter gap.
  - The host `drip_*` counters stay; `drip_used` reads 0 in this mode.
  - A new InstrBuild event `drip_adopt` (row, expert, victim slot, evicted expert).
- **CPU tests** (registered, `test/registered/unit/kernels/`):
  - A Python reference `drip_adopt_reference` is checked on random shortlists, routes and words. It adopts only
    routed, unmapped, landed experts; never a hazard slot; takes entries in shortlist order; preserves `victim_lanes`
    victims for the lanes; and commits exactly as `_commit_gather` does.
  - Unrouted words are left. With no usable entry, nothing is adopted and the tripwire counts it.
  - `GpuResidencyUpdater` with the flag off: identical tensors, and the existing tests unchanged.
  - Gate refusals.
  - `test_exl3_gpu_drip.py` passes unchanged (the host half is unchanged).
- **GPU tests** (`test/manual/dsv41/test_exl3_gpu_drip_adopt_cuda.py` on `lease_chain_rig.py`):
  - The CUDA adopt kernel matches the reference on 200 random cases.
  - A landed, routed drip is a hit: no lane in the record, the MoE output byte-identical to a resident hit, and
    residency committed.
  - A scratch slot is never refilled under a delayed adopt reader (the pattern of
    `test_a_scratch_slot_is_not_refilled_under_a_delayed_reader`).
  - Captured replays with drips landing between replays.
  - Under spill with k adoptions the lanes still get V victims.
  - Flag-off `test_exl3_gpu_drip_cuda.py` and the residency tests pass.
- **Served A/B** (`analysis/dsv41-drive/dspark/both_cpu_ab.py`):
  - The arm pair is `...-fixedsplit-gpudrip` against a new `...-fixedsplit-gpudrip-adopt`. Picks, chunk, slots and
    hot budget are identical, set to the replay's choice; the flag is the only difference.
  - Both arms have the route-log trace on, under `/mnt/nvme1`.
  - The verdict uses serve_compare's clean-both difference with its CI.
  - Also reported: VRAM hits/verify, adopted/landed, lanes by kind, forced CPU lanes, stream layers/verify, NVMe
    GPU lanes, CPU wait, and drip MB/verify.
  - A third arm that spends any freed scratch VRAM on the hot budget runs only if the pair is positive.

## 6. Risks and open questions

1. **Small gain.** About 0.5-3.6 ms/token per the two pricings, and the low end is the one with served evidence. The
   critical path, NVMe-cold forced misses, is outside the drip's reach. The replay may say not to build. That is a
   valid outcome and should be read as one.
2. **Owner intent.** B keeps a scratch, while the approval named landing in victim slots. This needs an explicit
   go/no-go on B vs A.
3. **Added GPU time.** About 40 adopt launches (about 0.16 ms/verify) plus about 26 extra row copies (about 0.27
   ms/verify) on the stream before planning. The adopt bookkeeping could later fold into the fused plan kernel.
4. **Split tuning.** The fixed split table `A` was tuned without adopted hits. Fewer lanes per layer shifts
   `split[n]`.
5. **Alignment noise.** About 7% (section 1.3), from a 131-verify capture. The precision-by-pick figures should be
   re-checked on a second drip capture if one exists.
6. **Design A's eager-path audit** (2.9) and its pre-route loss. The replay's lost-hit count is the deciding number.
7. **Lane-tier-replay's validity.** The 2.10 costs and TIER_RULES come from a model whose gate had not passed when
   this was written. The drip-hot replay takes them as given and reports which of its conclusions depend on them.
8. **Released VRAM victims.** Each adoption unpins its VRAM victim's RAM row (2.10). A victim that recurs after
   aging out of RAM returns NVMe-cold, at 1.47 + 1.09 ms. Rank-0..3 victims recur within three verifies 26-39% of
   the time, so the replay must count these returns. They are the hidden cost of every extra insert.
9. **After a prefill** fewer drip picks are eligible, since prefill admissions displace decode rows without bound
   and without an event. The A/B should report drip admits per verify against position after a prefill.

## 7. Replay result (2026-10-16): no arm passes the build gate

**Every result here is conditional on lane-tier's stage 0** (prefill admissions, TIER_RULES rule 8), which had not
passed its own gate when this was written.

Run details:
- Script: `analysis/dsv41-drive/dspark/drip_hot_replay.py` at `0ae29b5ef9`.
- It imports `expert_breakdown.py` from expert-breakdown `d79f05e077`, and `lane_tier_replay.py` and its NodeTier
  model from lane-tier-replay `a98b5b527c`.
- Outputs are under divix01:`/mnt/nvme1/drip-hot-replay/`: `gate-{drip,base}`, `arms-drip` (horizon 3),
  `arms-drip-h{1,8}`, and `victims-{c2,drip}`.
- Everything ran on cores 56-63 with no GPU.

**Validation gate: passed.** The post's lane typing is re-run per record on the capture's own lanes.

| | drip capture (131 verifies) | base capture (130 verifies) |
|---|---|---|
| lanes typed the same as the capture | 99.67% | 99.97% |
| per-verify lane mix | every category within 2.0% | every category within 0.04% |

- **What the gate tests.** The VRAM, spill and miss categories match by construction, because the model takes RAM
  hit/miss and the spill boundary from the capture. The test is the split of the remaining lanes into scratch, DMA and
  CPU-RAM: the split table, the miss cut and the drip timing.
- **One fitted parameter.** drip_land's observation lag is 1.4 ms: the copy thread stamps the event at its next poll
  of the store's mark. Without that lag, 22% of scratch lanes were mistimed.
- **Join.** Records join the route lines by lane experts (`layer_misses.match_forwards`), and drips join their target
  record by seq. No time offset is used.

**Arms**, per verify, against "today" (the drip capture as typed):

| arm | adopted (from spill) | lost hits | NVMe GPU lanes | CPU lanes, net of RAM-warm returns | NVMe-cold returns | added GPU ms | ms/verify at 0.07 / 0.28 per CPU-RAM lane | gate |
|---|---|---|---|---|---|---|---|---|
| B, window not extended | 77.1 (32.8) | 0 | -2.77 | **-13.4** | 0.01 | 0.50 | -4.6 / -8.1 | fails CPU |
| B, window extended | 77.1 (32.8) | 0 | **+9.06** | -42.6 | 0.02 | 0.50 | +12.9 / +5.9 | fails NVMe |
| A, 2 picks, no score cap | 53.3 (22.7) | 9.5 | +5.44 | -10.7 | 0 | -0.31 | +8.2 / +7.1 | fails both |
| C, one spare per layer | 30.9 (13.4) | 4.3 | +2.97 | +1.3 | 0 | -0.31 | +4.8 / +5.7 | fails both |

How the arms are priced and counted:
- **Other lane kinds.** CPU-NVMe 1.06, CPU-pool 0.50, DMA 0 (lane_cost_fit on this capture); pool 0.65, NVMe GPU
  1.47 + 1.09 (lane_tier_replay).
- **Returns.** These are released VRAM victims routed again within 3 verifies while the capture still held them: a
  RAM-warm return is priced as a CPU-RAM lane, an NVMe-cold one as an NVMe GPU lane. Horizons of 1 and 8 verifies
  move every figure by less than 2 lanes.
- **A's victims.** A reserved 74.9 victims per verify and wasted 22.4 of them on drips that never landed in time. Its
  9.5 lost hits/verify match the design note's estimate of about 11.

Readings:
1. **B without extension saves fewer than half the CPU lanes the gate asks for.** About 33 adoptions per verify come
   from spilled lanes, but each one shrinks the lanes' victim window by one, so only about 0.5 of a CPU lane is saved
   per spilled adoption.
   - It also moves 2.8 NVMe misses per verify from the GPU stream to forced CPU reads, which are the critical path.
   - Its priced gain of -4.6 to -8.1 ms/verify (about 1.2-2.1 ms/token at 3.83 tokens/verify) is inside the noise of
     a served pair.
   - That gain depends on almost no released victim coming back NVMe-cold. If the 3.5 RAM-warm returns per verify
     were cold instead, the gain is gone (+8 ms). Rule 8, the prefill admissions this tier model lacks, is exactly
     what makes rows cold, so the sign of B's gain is open until stage 0.
2. **B with extension reproduces the lane-head failure.** The freed victim lanes take NVMe-cold misses onto the GPU
   stream (+9.1 per verify).
3. **A loses about 9.5 hits per verify** before routing, and still adds NVMe GPU lanes.
4. **C costs more capacity than it adopts.**

**Recommendation: do not build drip-into-cache.** The drip stays as it is: off by default, neutral when on. If a
lever is wanted here, it is lane order and NVMe-cold rows, not the drip's landing site.

**C2.** It has no lane_kind or drip events, so only the shortlist-head statistics run there (`victims-c2.json`). They
match the drip capture: rank 0-5 recurs 0.114-0.245 of the time in the same verify and 0.262-0.477 within 3.
