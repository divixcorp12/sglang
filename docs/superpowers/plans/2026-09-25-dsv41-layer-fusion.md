# DSV41 decode: fusing the per-layer bookkeeping chains

Goal: cut the ~150 kernels per layer of the EXL3 graph decode (copy-overlap plan section 6, "item 2") with fused
kernels, byte-identical to the unfused path. Branch `cc/layer-fusion` (from `cc/copy-overlap` at `f54ce6f553`, with
`cc/dsv41-pinned-numa` merged at `f25f79e5e4`). Flag: `SGLANG_DSV41_ENABLE_LAYER_FUSION` (default off). Evidence lives
under `divix01:/data/models/slang/nvfp4-work/layer-fusion/`.

**Result:** three JIT kernels replace 89 torch kernels per layer (152 -> 66 in the stage-traced node trace). Greedy responses are byte-identical
in every arm. The 100 GiB smoke runs **131.9 -> 128.7 ms/token (-3.2, -2.4%)**, off/on/on/off in one session.

## 1. The catalog, checked against the trace

Source: `/mnt/nvme1/dsv41-nsys/default-node-20260924-193728.sqlite` (96 steps x 40 layers, 6,112 kernels per step).
Layer 20 of the median step was listed kernel by kernel (`layer20_default.txt`) and every kernel matched to its op.
All steps and layers (`analysis/dsv41-drive/layer-fusion/fusion_ranges.py`, `ranges_default.json`) give the per-layer
figures below; the range is identical in every layer and step. Node-mode kernel times and gaps are inflated for tiny
kernels (CLAUDE.md), so the µs columns rank the chains; the smoke measures the gain.

| Chain | Source (at `f25f79e5e4`) | Kernels / layer | GPU µs / layer | Span µs / layer | Fusion |
|---|---|---:|---:|---:|---|
| DIRECT gather destinations | `expert_residency_gpu.py:716` `gather_destinations`, plus the `expert_to_slot.index_select(0, flat.long())` argument at `expert_stream.py:1157` | 26 | 25.4 | 35.9 | **built** (1 kernel) |
| DIRECT commit | `expert_residency_gpu.py:758` `commit_gather` and `:768` `_commit_gather` | 41 | 49.2 (both rows) | 64.2 (both rows) | **built** (1 kernel) |
| MoE route tables | `exl3_fused_moe.py:69` `route_tables`, `:123` `x16.copy_`, `:125` `out.zero_`, and the remap casts `expert_stream.py:1214` `.to(topk dtype)` and `exl3.py:473` `.long()` | 22 | (in the row above) | (in the row above) | **built** (1 kernel) |
| gemv casts | `exl3_ops.py:90` `x.to(fp16)` and `:99` `y.to(out_dtype)` around each `exl3_gemv` | ~14 | ~23 | -- | not built (section 5) |
| mHC | `_hc_mix_stats_partial`, `_hc_mix_reduce_sinkhorn`, `_mhc_post_split_h`, `_hc_combine_norm`, x2 | 8 | ~22 | -- | not built (section 5) |
| shared-expert glue | `CatArrayBatchedCopy`, `silu_mul_clamp`, cast | 3 | ~2 | -- | not built (section 5) |

Corrections to the catalog:

- **The commit is 41 kernels, not ~22**, and the plan's "MoE prep" row is 22 kernels of `route_tables` plus the staging
  copies around it. The plan's "route plan" row (~29) is really the 26 kernels of `gather_destinations`; the fused
  planner (`plan_unique_routes_kernel`) is already one kernel.
- The go_total `torch.add` (`exl3_ram_miss.py:539`, 1 kernel) sits in the book group but is RAM-miss backend code, and
  was left alone.
- The graph-node gap is ~0.19 µs per kernel in this node-mode trace (p50 of all inter-kernel gaps), not ~1 µs; the
  span column above includes it.

Order by value / effort: the commit (41 kernels), gather destinations (26),
route tables (22). All three are integer bookkeeping plus two exact float conversions (bf16 -> fp16 of `x`, the
keep-scaled fp32 -> fp16 weights), so all three can be byte-identical. No existing fused op in the repo covered any of
them: `plan_unique_routes` (the fused planner) ends where `gather_destinations` starts, and `insert_expert_rows` is the
SCRATCH-stage row copy.

## 2. What was built

`python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh`, wrappers in
`python/sglang/kernels/ops/moe/dsv41_layer_fusion.py`:

- **`direct_gather_destinations`** (1 warp): the hazard test against every route's slot (the `index_select` folded in),
  the stable partition of the shortlist, `live`/`destinations`, the int32 destination slots and the translated remap.
  It writes the remap in the router's dtype, so the gather's `.to(topk_ids.dtype)` becomes a no-op; with a prefetch join
  it keeps int64, as the torch branch does.
- **`direct_commit_gather`** (1 thread): the torch chain is a sequence of scatters whose later writes overwrite earlier
  ones on the dump columns, so it runs serially, in the chain's order, so every final value (dump columns included)
  matches. The leased backend's `delivered`/`keep` narrowing and the truncation tripwire are inside it.
- **`exl3_moe_route_tables`** (grid-stride): `x -> fp16`, the zeroed output, per-slot counts, `inv_order`, the keep-
  scaled fp16 weights in slot order, `det = [start, start, count > 0]` and the int64 remap for `exl3_moe_gather`.
  `expert_start` is computed per slot as the number of routes on lower slots, which is the exclusive cumsum exactly.
  Routes are ranked stably; `torch.argsort` is not stable, which matters only if two routes share a slot. A BS1 DIRECT
  remap has distinct slots except in the case `insertion_truncated` exists to catch: a miss lane inside the count that
  found no live victim goes to slot 0, which can repeat another route's slot. There `inv_order`/`weight_sorted` may
  differ in tie order from torch (the MoE sum over the tied routes is then order-dependent too); the counter stays
  zero when the shortlist guarantee holds.
- Bad indices: torch's `index_select`/`index_add_` device-assert on an out-of-range route id or remap; the kernels do
  not check, so a corrupt id reads out of bounds instead of asserting.

Glue: `expert_residency_gpu.py` (`_init_layer_fusion`, `fused_gather_destinations`, `_fused_commit_gather`, per-layer
output rows so a side-stream commit never reads a reused buffer), `expert_stream.py` (`_gather_graph` branch),
`exl3_fused_moe.py` (`Exl3FusedMoE._fused_route_tables`), `exl3.py` (passes the int32 remap). The flag is read once
where each object's buffers are built. Nothing in the Engram or RAM-miss service code changed.

## 3. Parity

`test/manual/dsv41/test_dsv41_layer_fusion_gpu.py`, **121 passed** at `9df4289771` (`parity2.log`; 97 before the remap-dtype axis was added, `parity1.log`,
`restored_parity.log`):

- gather + commit against the production torch methods on cloned state, exact equality of the remap, destinations,
  live lanes, destination slots and all seven residency tensors (dump columns seeded with garbage). Shapes: width/routes
  6/6 (production), 1/1, 7/5, 13/13, 32/32; capacities 2-64; int32 and int64 routes and, separately, int32 and int64 planner remaps; leased and unleased backends;
  random miss counts, delivered counts, keep 0/1 and out-of-range remap ranks. 60 trials per case.
- the same two kernels captured in a CUDA graph and replayed on 20 new input sets, against the torch chain run eagerly.
- route tables against `route_tables` + the staging copies: routes 1-32, slots 3-1025, hidden 7-5121, int32/int64 remap,
  fp32/bf16 weights, bf16/fp16/fp32 `x`, keep 1/0/0.75; fp16 outputs compared as bits.

Mutants, one per kernel, applied together in the private worktree, then reverted and the suites re-run green: skip the
generation increment (commit), drop the shortlist reorder (gather), rank routes descending (route tables). They fail
24 of 24 gather+commit cases, the capture test, and 60 of 72 route-table cases (the 12 survivors have a single route,
where both orders coincide), and 19 of 31 integration tests with the flag on, which shows the integration tests run the
fused path (`mutant_*.log`).

Integration, flag on vs off (`integ_on.log`, `integ_off.log`, `restored_integ_on.log`): `test_exl3_ram_miss_graph_gpu.py`,
`test_exl3_graph_apply_gpu.py`, `test_moe_side_stream_gpu.py`, `test_exl3_task5_item4_gpu.py`: **35 passed** both ways.
Registered: `test_expert_residency_gpu.py test_expert_graph_gather.py test_exl3_fused_moe.py
test_exl3_ram_miss_service.py test_expert_route_plan_fused.py`: **354 passed, 1 error** flag on, flag off, and at the
merge base `f25f79e5e4` (`reg_base.log`); the error (`test_graph_routes_are_logged_only_when_the_stage_trace_is_on[trace_on]`,
a RAM-miss shutdown fixture) is pre-existing.

Commands (divix01, `run_gpu_tests.sh`: `PYTHONPATH=$WT/python`, `taskset -c 32-63`, under `cc-gpu.lock`, exit code
read from pytest itself): `pytest -q -p no:randomly -rfE <files>`, with `SGLANG_DSV41_ENABLE_LAYER_FUSION=1` for the
flag-on runs.
Registered kernels suite, `pytest -q -p no:randomly -rfE test/registered/unit/kernels` at `c70c5138d2`: **1665 passed,
1 skipped, EXIT 0** (`kernels_suite.log`).

## 4. Measurements

**Smokes**, 100 GiB `arm_env` recipe, cold server per arm, off/on/on/off, one session, at `6c0311b3ea`
(`analysis/dsv41-drive/layer-fusion/smoke.sh`; `compare_arms.py --include-warmup`, `smoke/compare_abba.json`):

| Arm | ms/token, trace (wall-validated) | step p50 / p90 | 1-row submit->done p50 | stalls (multi-row > 10 ms) |
|---|---:|---:|---:|---:|
| off 1 | 131.9 (134.6) | 127.7 / 161.1 | 2580 us | 1 of 2034 |
| on 1 | 128.6 (131.3) | 124.2 / 157.4 | 2602 us | 2 of 2050 |
| on 2 | 128.8 (131.4) | 124.7 / 157.9 | 2587 us | 3 of 2065 |
| off 2 | 132.0 (136.1) | 127.7 / 161.2 | 2586 us | 62 of 1947 |

Responses are **byte-identical**, 6 of 6 for every arm against off 1, and each arm's two reps agree. The gain is
3.2 ms/token on both pairs, with no order effect (off 1 and off 2 agree to 0.1). Off 2's 62 stalls are slow multi-row
reads (its 1-row p99 is 10.7 ms against 3.3 elsewhere) and did not move its ms/token; they are disk-side and not
related to the flag.

`compare_arms.py` needed a fix first (`2feb4d562e`): the stage trace's new `graph_routes` records split every request
block, so every arm reported a trace ms/token of 0.

**Node-mode kernel counts** (counts only, CLAUDE.md), flag on at `6c0311b3ea` (the smokes' commit), flag off at
`2feb4d562e` (the same code; it adds only the `compare_arms.py` fix), same
stage-traced recipe, `fusion_ranges.py` (`ranges_node_{off,on}.json`):

| Trace | Kernels / step | Kernels / layer | A: kernels, span µs | BC: kernels, span µs |
|---|---:|---:|---:|---:|
| flag off (`smoke/node-off`, 489 steps) | 6,196 | 152 | 28, 38.4 | 63, 64.6 |
| flag on (`smoke/node-on`, 566 steps) | 2,756 | 66 | 3, 6.1 | 2, 8.3 |

A is the range after `plan_unique_routes_kernel` up to the post kernel; BC is after the go_total add up to
`exl3_moe_kernel` (p50 over every layer of every step). **86 kernels per layer removed** (3,440 per step), and the two
ranges' span falls by 88.6 µs per layer, 3.5 ms per step in node mode, which lines up with the 3.2 ms/token the
graph-mode smoke measured. Both A counts include `GraphRouteLog.record`'s two `index_copy` kernels, which exist only
because the smoke runs the stage trace (the 09-24 default trace, untraced, has A = 26). Both node runs' responses are
byte-identical to each other and to the off 1 smoke.

## 5. Not built, and why

- **gemv casts** (~14 kernels, ~23 µs/layer). The casts sit on both sides of `exl3_gemm`, whose kernel takes `__half`
  in and writes fp16/fp32 out. Folding them means changing exllamav3's `exl3_gemv` kernel, which is built from the
  pinned external checkout (`exl3_ext.py`, `EXLLAMAV3_COMMIT`), not from this repo. That is a vendoring decision, not a
  layer-glue change.
- **mHC** (8 kernels, ~22 µs/layer). Already Triton; merging `stats_partial` with `reduce_sinkhorn` changes the
  floating-point reduction order (a grid-wide partial sum becomes one reduction), so it could not be byte-identical and
  is the more invasive change for ~2 kernels per mixer.
- **shared-expert glue** (3 kernels, ~2 µs/layer): too small to pay for a kernel.
- The go_total `torch.add` and the option C `routes` copy (1 kernel each per layer) are RAM-miss backend code, which
  this lane does not touch.

## 6. Recommendation

Turn the flag on in `arm_env` for the next measurement lane (this lane does not change `arm_env` defaults). It is exact,
covered by parity and integration tests, and worth 3.2 ms/token. The rest of the per-layer tiny-kernel time is the gemv
casts and mHC; C1 (72 ms/step) is still the decode.
