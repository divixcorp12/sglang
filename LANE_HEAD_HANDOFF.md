# Lane head (v1 gather-time head) — handoff, 2026-10-16

Written for a reader with no context. Branch `lane-head` (off `origin/ram-prefetch-pool-serve` at `a3bca01530`);
analysis scripts and all replay evidence are on branch `cache-tiers` (`04b3d7f793`). Owner approvals and the five
build conditions came through team-lead. Sections 1-8 are the pre-build brief, kept as written; section 0 is what was
built, measured and decided after it, and wins where they differ.

## 0. Outcome (2026-10-16): built, served, no gain; lane held

**Status.** The v1 head is built and tested on `lane-head` (`f4eadc1fa7`) and was served in one A/B. It moved +19.5
VRAM hits/verify but no ms/token: the GPU lanes moved onto NVMe-cold rows. Team-lead's decision: no fused kernel, the
lane is held, no new runs, the `lane_head_gathers` telemetry gap stays unfixed. Recorded in `DSV_OPTIMIZATION.md`
(`a1b107f42f`).

**What was built (differences from sections 2-3, all agreed with team-lead):**
- The file keeps the training's standardization unfolded (`feature_mean`, `feature_scale` = `constant_safe(fs)`,
  `z_scale`); serving computes `(f - fm) / fs` as `next_verify_head.predict` does. The draft odds were constant 1.0 in
  B0 and are fed 1.0 (their term is exactly zero); the loader checks the constant's mean and 1e9 scale.
- The score and age features are the head's own per-request state (`LaneHead.score/last/verify_ct`), advanced by
  its key kernel after the keys are written (score `*= 0.98**live`, `+=` route counts), reset at every PREFILL
  forward (`GpuResidencyUpdater.observe_forward`) and after capture. The updater's `insert_scores` was not what
  training saw: it carries the prefill's route counts and never resets per request.
- A verify's dead rows (DSpark gate) are excluded; a launch with `SGLANG_DSPARK_TOP_PROB_GATE` set, an insert decay
  other than the file's `score_decay` (0.98), or no CPU experts is refused.
- Key = `bits(fp32 P) << 16 | (experts - 1 - e)` (residency_rank_keys' shape); `int64(P * 2^31)` rounded small P to 0.
- Two kernels per layer: PCA in 8 hidden-dim splits, then one block that sums the splits into shared memory and scores
  each route with a warp. `next_verify_head.py` is checked out from `cache-tiers` for the tests.
- Tests: `test/registered/unit/layers/moe/test_lane_head.py` (18; on C2's first 16 verifies the reference gives
  next_verify_head's features bit for bit, P within 1.8e-7 of pred-B, 120/120 lane orders identical),
  `test_exl3_ram_miss_attach_lanes.py` (+2), `test/manual/dsv41/test_lane_head_gpu.py` (8; kernel vs reference P
  within 1e-6, identical orders, graph replay, odds).
- Cost (captured 40-layer graph, 6 tokens): 0.77 ms/verify warm and cold, 19.4 us/layer; PCA 8.5 us + keys 9.4 us
  per layer; floor of two trivial kernels per layer 121 us/verify. fp32 PCA costs the same as bf16, so the served file
  is fp32 (`/data/models/slang/nvfp4-work/ram-prefetch/lane-head/head-B-fp32.safetensors`, sha256 `83312307...`).
  A fused one-kernel design (coalesced PCA, keys in the last block) was estimated at 0.32-0.40 ms/verify; not built.

**The A/B** (`/mnt/nvme1/lane-head-ab-20261016`; branch `lane-head-ab` `d269f7aa90` = lane-head + prefill-merge-chunk
`abaae5fdd0`, throwaway; arms `...-mergechunk-expseg` vs `...-mergechunk-expseg-lanehead`, resolved envs differing
only in `SGLANG_OPT_LANE_HEAD_PATH`, route log on via `spec_margin_capture.py --route-log`, one sample each):
- serve_compare, clean prompts (5): ms/token +0.74, 95% CI -3.11 .. +3.96. Per verify: wall +3.8 ms, stream +51.7,
  wait -46.9; stream layers/verify 12.6 -> 22.9, ms per stream layer +0.11. All 8 prompts: +9.51 (ON prompt 6 had a
  disk stall, x6.0 decode, 32.4 ms nvme3n1 await in its prefill window against ~8 elsewhere).
- `lane_head_ab_report.py` (on `lane-head-ab`, `022acb70b9`; reads the route log and job trace): VRAM hits/verify 388.8 -> 408.3 (+19.5, above the replay's +7.5..+11),
  misses 514.8 -> 497.7, unique routed 903.6 -> 905.9. CPU lanes/verify 366 -> 371, CPU ms/verify 227 -> 223, copy
  submits 79.5 both; RAM-tier rows_read 10,950 vs 11,089. Slots 1,035 -> 1,030. No OOM.
- `lane_head_gathers` did not reach the metrics: the trace builds `residency_gpu` from the async telemetry buffers,
  which lack the counter. The head's run is shown by its "lane head active" log line, `lane_head_bytes`, and the
  hit/stream shifts.

**The finding: the GPU lanes moved onto NVMe-cold rows.** The plan's GPU lanes (a fixed ~80/verify) used to go to the
highest residency keys, which are mostly experts routed recently and so still in the pinned RAM tier: a GPU lane
copied its row from RAM. The head gives them to the experts most likely to be routed at v+1, and it does not see the
RAM tier: many of those are cold, so the GPU lane now streams the row from NVMe (`kMissGpu`). The NVMe reads did not
change in number; they moved from the CPU's wait (-47 ms/verify) to the GPU's stream (+52 ms/verify). The replay that
predicted the gain (`vram_cache_replay.py`, `lane_cost_fit.py`) priced every non-VRAM lane at a flat 0.37 ms, so it
could not see this.

**Next version (sketch; not started, held):**
1. A replay that prices lanes by tier. Per (verify, layer, lane): VRAM hit, RAM-warm miss (row in the pinned tier at
   the gather: GPU copy ~DMA, CPU lane ~0.29 ms), or NVMe-cold miss (GPU stream ~1 ms+ on the critical path, CPU lane
   ~1.1 ms). The RAM tier per verify comes from `ram_prefetch_replay.py`'s calibrated LRU (190 rows/layer reproduces
   51.7 RAM misses/verify) or from a capture's `miss_expert`/`svc_record` events. Validate it on this A/B first: it must
   reproduce OFF and ON's stream and wait per verify (+52/-47 ms) before it ranks anything.
2. Head v1.1, either of:
   - (a) RAM-tier presence as a feature: one more dense feature (row in the pinned tier at the gather), read on device
     from the RAM tier's host-row map the plan already has (`row_backend.host_row_map` / the lease map), and an
     objective that is the lane's saved time, not P(recur): `P(recur) x VRAM-hit value - (cold ? NVMe-on-GPU cost :
     0)`. Needs retraining with the tier state in the extraction (B0 has `miss_expert` events; check coverage).
   - (b) No retraining: cap the head to RAM-warm experts. Key = head P for misses whose row is in RAM, today's
     residency key (below every warm one) for cold misses, so the GPU never takes a cold row the residency order
     would not have given it. Cheapest to try; the tiered replay says first whether the hits left are worth 0.77
     ms/verify.
3. Only if 1 shows a gain above the head's cost: the fused kernel (0.32-0.40 ms/verify estimated), then PDL, then
   the `lane_head_gathers` telemetry, then a served A/B reading hits, stream and wait per verify.

## 1. Goal and status

**Goal.** In DSpark decode of DeepSeek-V4.1 (EXL3), each MoE layer's graph gather plans its miss lanes and the fused
route plan sorts them by an int64 key per expert, highest first; the GPU copies the head of the sorted list (those
rows are inserted into the VRAM hot cache, DIRECT residency) and the CPU experts compute the tail (never inserted).
Today the key is the residency ranking (`GpuResidencyUpdater.miss_keys`, decayed route counts). The lane head
replaces the key with a learned probability that the routed expert is routed again by the *next* verify, so the GPU
copies (and the cache keeps) the experts that will be reused and the CPU computes the rest. Offline replay on
held-out captures: +7.5 hits/verify over the best no-learning table and +11.9 over today's key on C2, +11.0/+13.2
on base, fewer inserts than today; ~1.2-1.4 ms/token traced (below the ~7 ms/token run-to-run noise, so the served
A/B must read VRAM hits and CPU-lane time, not only ms/token).

**Built (commit `8191a92d3b`, untested, never compiled on a GPU):**
- `python/sglang/srt/layers/moe/lane_head.py`: file format (safetensors + JSON metadata `LaneHeadMeta`), validating
  loader `read_lane_head_file`, `read_lane_head_meta`, `lane_head_device_bytes`, the fp32 reference
  `reference_keys` (the test oracle), device state `LaneHead` (weights, `keys[layers, experts]`,
  `last_routed[layers, experts]`, `z` scratch, `launches` counter; `launch`, `record_routed`, `describe`), and the
  per-layer gather hook `LaneHeadGather`.
- `python/sglang/kernels/jit/csrc/moe/lane_head.cuh` + `python/sglang/kernels/ops/moe/lane_head.py`: two kernels
  per layer gather (`lane_head_pca_kernel`: grid (tokens, rank/16) x 256 threads; `lane_head_keys_kernel`: one block
  of 128 threads, one thread per route), JIT wrapper `lane_head_keys(...)` templated on id dtype (int32/int64),
  weight dtype (fp32/bf16) and PDL.
- `analysis/dsv41-drive/dspark/export_lane_head.py`: folds the trained head + PCA into one file (see section 4).
- Edits: `environ.py` (`SGLANG_OPT_LANE_HEAD_PATH = EnvStr(None)`), `expert_hot_cache.py` `from_model` (budget
  subtraction, refusal if the head leaves no budget, `LaneHead` construction after the updater, `lane_head_bytes` in
  the "Expert hot cache startup" log, `manager.lane_head`), `expert_residency_gpu.py` (`self.lane_head`, age update
  in `_apply`), `exl3_ram_miss.py` `attach` (points `_plan_miss_keys` at `lane_head.keys[row]`, installs the hook),
  `expert_stream.py` (`lane_head_hook` attribute; call before `plan_graph_routes_fused` in `_gather_graph`).

**Not built:** tests (none), the exported head file (not yet produced), the divix01 worktree and any run, the
allocation-order change for startup calibration (condition 2, see below), the A/B tooling. The code has been
`py_compile`d only. Expect compile errors in the `.cuh` on first JIT (never built) and small integration mismatches.

**Next concrete step.** Push is done; on divix01: `git -C /data/models/slang/sglang fetch origin && git -C
/data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-lane-head origin/lane-head`, then
`PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES= taskset -c 56-63 /data/models/slang/.venv/bin/python
analysis/dsv41-drive/dspark/export_lane_head.py --head /data/models/slang/nvfp4-work/ram-prefetch/cache-tiers-20261015/head/B
--data /mnt/nvme1/cache-tiers-tmp/head-data --out /mnt/nvme1/cache-tiers-tmp/lane_head_v1.safetensors` (CPU only),
then write and run the CPU tests (section 3), then ask team-lead for a GPU slot for the JIT build, the GPU test and
the microbenchmark.

## 2. Approved design (from the design note, as approved by team-lead)

- **Flag.** `SGLANG_OPT_LANE_HEAD_PATH` (`EnvStr(None)`, in `environ.py` next to
  `SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD`). The path is the switch: unset/empty means nothing is installed and
  `_plan_miss_keys` stays `gpu_residency.miss_keys[row]`, bit-identical to today. Set but unreadable or mismatched
  (format version, feature list, layer ids, expert count, shapes, non-finite, a "constant" feature with a nonzero
  folded weight) raises `ValueError` naming the file, before graph capture. Needs DIRECT residency
  (`SGLANG_MOE_GPU_RESIDENCY_UPDATE=1`, `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2`), CPU experts and the fused plan
  (the existing miss-order requirements); `from_model` refuses without DIRECT.
- **Placement.** `Exl3MoEMethod._apply_graph` (`python/sglang/srt/layers/quantization/exl3/exl3.py` ~line 737) sets
  `backend.cpu_input = (cpu_x, topk_weights)` right before `streamer.gather(topk_ids)`. Inside
  `ExpertStreamer._gather_graph` (`expert_stream.py`), in the fused branch, `self.lane_head_hook(topk_ids)` runs
  immediately before `plan_graph_routes_fused(...)`, on the gather's stream, captured in the breakable decode graph,
  no host node. `plan_unique_routes_kernel` (`expert_route_plan.cuh`) then sorts the residual (miss) lanes by
  `miss_keys[expert]` descending; `type_lanes` (`lease_device.cuh`) gives the CPU the tail as today. Victim ranking
  (`_rank_victims`) is untouched (the replay showed the head's victim-side gain is ~0).
- **Inputs per layer gather.** `x` bf16 `[tokens, hidden=5120]` from `cpu_input[0]`; `topk_ids`, `topk_weights`
  `[tokens, 6]` (ids < 0 are dead rows); `insert_scores[row]` fp32 `[384]` (the boundary's decayed score, before this
  forward's counts, exactly as the offline replay used it); new `last_routed[row]` int32 `[384]` (the forward each
  expert was last routed in, -1 never; written in `GpuResidencyUpdater._apply` from `routed` for gated layers, reset
  to -1 at a prefill boundary = new request); `forwards` int64 `[1]` (the updater's clock; one counted forward per
  verify since drafts are not counted, so forwards == verifies); `mapping[row, :384]` int64 (expert_to_slot, >= 0 =
  resident). Output: `keys[row]` int64 `[384]`, written for routed experts only (the plan reads only routed experts).
- **Key.** `P = sigmoid(logit)`, `key = int64(P * 2^31) * 2^16 + (384 - 1 - e)` (same tie-break shape as
  `residency_rank_keys`).
- **Head math (variant B of the study), per layer L:** `z = (x[i*] - pca_mean_L) @ pca_proj_L` (rank 64; `i*` = the
  token routing e with the largest gate weight, earliest on ties); `logit = expert_bias_L[e] + bias_L + dense_w_L .
  f + z_w_L . z + expert_u_L[e] . z`. The 17 dense features, in order (`lane_head.FEATURES` ==
  `next_verify_head.DENSE`): `cnt1, cnt2, cnt3p` (token count one-hot), `pos0..pos5` (first position one-hot),
  `gate_w` (largest routing weight), `log_score` = log1p(max(score,0)), `score0` (score <= 0), `log_age` =
  log1p(age) (age = forwards since last routed), `first_touch` (never routed), `resident` (expert_to_slot >= 0),
  `live` = live tokens / 6, `odds` (draft acceptance odds; see condition 1). Standardization (`fm`, `fs`, `zs`) is
  folded at export: `dense_w = w/fs`, `bias = b_dense - sum(w*fm/fs)`, `z_w = a/zs`, `expert_u = U/zs`; a feature
  constant in training (`fs` = 1e9 after `next_verify_head.constant_safe`) folds to weight 0.
- **Memory (total over 40 layers, not per layer):** `pca_proj` bf16 40x5120x64 = 26.2 MB; `expert_u` fp32
  40x384x64 = 3.9 MB (the note said 2 MB bf16; the file stores fp32); `pca_mean` 0.8 MB; biases/dense ~0.1 MB; keys
  int64 0.12 MB; last_routed int32 0.06 MB; z scratch 40x8x64 fp32 0.08 MB. `lane_head_device_bytes(meta)`
  computes it from the metadata; `from_model` subtracts it from the hot-cache budget before sizing (so
  `SGLANG_MOE_HOT_GPU_MB` stays the VRAM the two take together; ~2 slots fewer) and logs `lane_head_bytes`.
- **Files to create/edit:** listed in section 1. Tests to add: `test/registered/unit/layers/moe/test_lane_head.py`
  (CPU: file round-trip and refusals; reference ordering == p ordering; the constant-feature agreement test of
  condition 1; a both-flags census test of condition 2), `test/manual/dsv41/test_lane_head_gpu.py` (kernel keys ==
  `reference_keys` on random and captured inputs, P within 1e-3, order identical; microbenchmark per layer with and
  without an L2 flush: 40 distinct weight sets and a 256 MB rotating buffer touched between calls, since the gathers
  evict L2 between layers in serving), `test/manual/dsv41/test_lane_head_capture.py` (reference on C2's
  `router.{ids,w,x}.bin` verifies reproduces the offline predictions' lane order; see section 4 on bf16 PCA).
- **Expected cost.** ~4 MFLOP/layer, launch-dominated: ~5-10 us/layer = 0.2-0.4 ms/verify on the gate->plan
  critical path (the earlier L+2 heads failed at 0.10 ms/layer = 4 ms/verify). Must be measured (condition 3).

## 3. Team-lead's five conditions (all binding)

1. **Draft odds:** serve exactly what training saw. The B0 training corpus has no draft-top-prob, so `odds` was
   constant 1.0 and its weight folds to 0. The exporter records every constant feature and its training value in
   `meta.constant_features` (`{"odds": 1.0}` expected); the kernel takes `odds` from `meta.odds` (wrapper argument,
   kernel `f[16] = odds`); the loader refuses a constant feature with a nonzero folded weight. Still needed: the
   unit test that the exported file's `constant_features["odds"]` equals the value the training shards held and that
   `reference_keys(..., odds=meta.odds)` and the kernel agree.
2. **Startup calibration:** prefill-merge-2's `SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP` (branch
   `origin/prefill-merge-chunk`, `expert_hot_cache.py` lines ~168-215 and ~1523) measures free memory after model
   load and grows the cache toward an upper capacity on VMM memory. The head's bytes must be allocated before that
   measurement and `lane_head_bytes` must show in the startup log (it does). As committed, `LaneHead` is constructed
   after the caches, which is too late if calibration measures inside/after cache allocation: move the allocation to
   right after the budget subtraction in `from_model` (device = the model's; validate layer ids/experts later against
   the updater) or allocate before `from_model` is entered. Write one CPU test with both flags on checking the
   hot-cache census still holds (that branch is not merged into `lane-head`; coordinate the merge order with
   team-lead).
3. **Critical-path cost:** report the microbenchmark per layer with and without the L2 flush; in the served A/B
   report the gate->plan time from the per-layer device stamps (note: today's stamps cover post/S/wait, not
   gate->plan; a stamp may need adding). If the head adds more than 0.5 ms/verify, tell team-lead before the served
   run.
4. **A/B verdict:** report VRAM hits/verify, CPU-lane time per verify and gate->plan time alongside ms/token, with
   `serve_compare.py` as the verdict; the gain must show as hits moving from CPU lanes to VRAM, not only ms/token.
5. **Generalization:** the head was trained on the B0 corpus (100 sessions, `dspark-both-misscut` recipe, 853 slots,
   prefetch off) and evaluated on held-out captures C2 (`split-ab-20261014/C2`, 48 prompts, fixed-split arm, 853
   slots) and `breakdown-20261015/base`; the +7.5..+11 hits came from those two. If startup calibration changes slot
   counts per launch, the `resident` feature's distribution shifts (share of routed experts resident); note it in the
   A/B and retrain if calibration moves the slot count far from 853.

## 4. Where the trained head lives

- Weights: divix01 `/data/models/slang/nvfp4-work/ram-prefetch/cache-tiers-20261015/head/B/L00..L39.npz` (per layer:
  `fm, fs, zs, state_bias.weight [384,1], state_dense.weight [1,17], state_dense.bias [1], state_z.weight [1,64],
  state_u.weight [384,64]`) and `head/B/summary.json` (variant B, 8 epochs, dev log loss per layer; mean 0.528 vs base
  0.685). Variant A (no per-expert direction) in `head/A/`, weaker (precision 0.557 vs 0.595).
- PCA and training shards: `/mnt/nvme1/cache-tiers-tmp/head-data/pca.npz` (`mean [40,5120]`, `proj [40,5120,64]`
  fp32, fit on every 8th training verify, 6000 rows/layer), shards `train-b000..b003/L*.npz`, eval shards `c2/`,
  `base/`, `manifest.json`. Scratch volume; copy to `/data` if it must survive.
- Training command (branch `cache-tiers`, script `analysis/dsv41-drive/dspark/next_verify_head.py`, commits
  `fbdb846930`, `33a8118dca`, `e0aa95a7c7`): `extract --out D --train B0/blocks/b000..b003 --eval c2=... base=...
  --pca-rank 64 --pca-every 8 --pca-rows 6000`; `train --data D --out head/B --variant B --epochs 8`; `predict`;
  `report`. Runner `cache-tiers-20261015/run_head_train.sh`. 5.35M training examples (5,969 consecutive-verify pairs),
  31 s on 4 CPU threads. **Always `CUDA_VISIBLE_DEVICES=""` before importing torch** (the script sets it; see rules).
- Held-out numbers (precision at k = the capture's own GPU-lane count, misses only, `head/{c2,base}.report-B.json`):
  C2 head 0.595 / recall 0.378 vs table 0.471/0.299, score 0.408/0.259, actual 0.420/0.267, oracle 0.851/0.541; base
  0.614/0.393 vs 0.517/0.331, oracle 0.851. Log loss C2 0.544 (head) vs 0.610 (table). Replay hits/verify (test
  forwards, `{c2,base}.replay-headB.txt`): C2 head-k 397.8 vs table-k 390.3, score-k 385.9, next1-k 431.6, belady-k
  460.5; base 373.9 vs 362.9 / 360.7 / 405.4. Inserts 116.8/verify vs today's 121.7.
- bf16 PCA: the offline predictions (`head/c2.pred-B.npz`) used fp32 PCA; the file stores `pca_proj` in bf16, so the
  capture test should regenerate predictions through `reference_keys` and compare lane orders (expect agreement on
  all but near-ties; report the flip count), and rerun `vram_cache_replay.py --pred` with them to confirm the hits.

## 5. Replay findings (all on branch `cache-tiers`, outputs in `cache-tiers-20261015/`)

- **VRAM replay** (`vram_cache_replay.py`, C2 806 verifies, 853 slots = 21-22/layer; `c2.replay*.txt`): actual
  41.1% unique hit rate (361 of 879 unique routed experts/verify), 112 inserts, 405 CPU lanes, 45% of inserts
  evicted before reuse; LRU stack-distance median 32 > capacity. Simulated today's rule 43.7% (model validated).
  Insert-all (CPU experts too): +36 hits for +292 copies/verify, ruled out. Per-layer non-uniform slots: +1-3 hits,
  ruled out. Decay 0.98 -> 0.995: +5-6 hits (cheap arm). Capacity: +1 slot/layer ~ +7.5 hits/verify ~ 2.8 ms/verify
  traced; 853 -> 951 slots ~ +18 hits. Belady-with-bypass 57.2% (+141), one-verify oracle 54.1% (+114), k-limited
  oracle 48.9% (+68 with no extra copies). Rejected-token routes recur within one verify 42% vs 26% for accepted
  routes (rejected positions are regenerated); acceptance-aware bypass loses hits. Cost model (`lane_cost_fit.py`,
  breakdown base, OLS r2 0.81): CPU RAM-hit lane 0.29 ms, NVMe-read lane ~1.1 ms, DMA ~0; mix 0.37 ms per non-VRAM
  lane (traced).
- **NVMe prefetch a verify ahead** (`ram_prefetch_replay.py`, `c2.ramprefetch*.txt`, `c2.ramcalib.json`): a routed
  expert is a RAM hit next verify by construction (0.7% of NVMe demand misses were routed the verify before), so
  recurrence heads cannot feed prefetch. Oracle bounds on top of today's L+2 T3 prefetch: 20 rows/verify save 19.5
  of 50.4 RAM misses, 40 save 33.4, 80 save 44.3 (candidates run out at ~50); timing not the constraint. Near-miss
  gate baseline (ranks 7+ of this verify's tokens): 5% precision, net negative (wasted landings evict useful rows).
- **RAM-tier calibration:** per-layer LRU with VRAM-hot protection reproduces the capture's 51.7 RAM misses/verify
  (33.5 demand + 18.2 covered) at 190 rows/layer (nominal 210; ~20 slots/layer are pool/promotion mechanics).
- **Churn fix refuted:** evicting never-routed landings first is worse (52.9 vs 50.4 misses/verify: a landing is
  unrouted until its target layer runs); stale-first (after its verify passed) saves 0.5. `promote_landed` already
  stamps a promoted row with its victim's old recency, so the real tier is stale-first already. The earlier "13.5
  misses induced by churn" attribution was wrong (a calibration artifact). Cold-miss ages of the experts today's
  prefetch covers: 1.9% last routed 1-2 verifies earlier, 18.2% 3-8, 25.4% 9-16, 24.4% 17-32, 30.0% not in 32.
- miss-cut-replay's independent numbers agree: VRAM pair hit rate 0.51-0.53, residency key a ~6-point reuse
  predictor; keep the greedy split rule, no residency cost term.

## 6. Held items

- **(c) Draft-hidden-state capture** (owner's call, ~2 h of box time, ~16 GB on /mnt/nvme1): B0-style, served recipe
  with prefetch on, 100 sessions, adding per verify the draft's final hidden states for the drafted tokens (~25
  KB/verify) and draft-top-prob. Needed because no capture pairs draft state with the next verify's routes, and B0
  ran with prefetch off.
- **(d) v2 plan** (conditional on (c)): one model per verify, off the critical path during the draft phase
  (end-to-next-begin gap 51.7 ms median, 42.7 p10), inputs = verify v's gate inputs for all layers/tokens (240 x 5120
  bf16 on device), the draft's hidden states for v+1's tokens, a per-(layer, expert) route-history state (16-d,
  updated once per verify), positions and draft confidence; shared 5120->256 projection + 2 attention layers over
  the 240 token-layer vectors + per-(layer, expert) 256-d vectors (~4-5M parameters; per-layer full-dim per-expert
  weights would be 79M on ~350 examples each and overfit); two outputs per (layer, expert): P(routed at v+1) for
  prefetch and P(routed at v+2 | routed at v+1) as the prior the gather-time key reads. Cost ~0.4 GFLOP, <0.2 ms per
  verify in a 40 ms window; GPU training 15-20 min under cc-gpu.lock on a 790 MB projected dataset. Data: 5,969
  pairs; B0 requests run 10 verifies median (p25 7, p75 14; 332 runs of 8+). Ceiling: VRAM lanes 431.6 hits (oracle)
  vs v1 397.8, realistic 405-415; prefetch oracle -19.5 RAM misses at 20 reads, a 50%-precision head +3..+10.

## 7. Rules

- `.claude/rules/divix01-run-protocol.md`: code reaches divix01 only by push + `git worktree add --detach
  /data/models/slang/nvfp4-work/wt-<name> origin/<branch>`; `PYTHONPATH=$PWD/python`, print `sglang.__file__`;
  interpreter `/data/models/slang/.venv/bin/python`; read `${PIPESTATUS[0]}`; run only the affected tests.
- CPU work on cores 56-63 (`taskset -c 56-63`, OMP threads <= 8) while any capture runs; `taskset -c 0-63` only
  when team-lead confirms the box is quiet. GPU work only with a slot from team-lead, under `cc-gpu.lock` on cores
  32-63 (serialized with prefill-merge-2); lock order `rowimg-disk.lock` then `cc-gpu.lock`.
- `CUDA_VISIBLE_DEVICES=""` before importing torch in every offline job: torch's AdamW step calls
  `torch.accelerator.current_stream()` and creates a CUDA context. Incident 2026-10-15 18:32 CDT: the first training
  attempt did this while arm C was serving, failed with CUDA OOM, allocated nothing; team-lead checked arm C's log,
  no harm. `next_verify_head.py` and `ram_prefetch_replay.py` set the variable at import.
- Outputs under `/mnt/nvme1` (TMPDIR `/mnt/nvme1/cache-tiers-tmp`) or `ram-prefetch/cache-tiers-20261015/`; never
  `/tmp` on divix01; never touch `cc-expert-prediction/dsv41-direct-prod`, port 7867, or `DSV_OPTIMIZATION.md`.
- Repo hook: edit tracked files with Edit/Write, not sed/python/redirects. Env vars per
  `.claude/skills/env-var-conventions`; spec-decoding names per `.claude/skills/speculative-naming`; JIT kernels per
  `.claude/skills/add-jit-kernel` (abstractions in `sgl_kernel/`, Doxygen comments, ASCII only).
- Never amend or force-push; stage by name; commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  and the session line.

## 8. Branches and commits

- `lane-head`: `8191a92d3b` (this WIP) + this handoff.
- `cache-tiers` (`04b3d7f793`): `analysis/dsv41-drive/dspark/vram_cache_replay.py` (VRAM policy replay, `--pred`
  hook, predictor report), `lane_cost_fit.py` (traced marginal cost per lane kind), `next_verify_head.py`
  (extract/train/predict/report), `ram_prefetch_replay.py` (RAM tier + prefetch policies, victim rules,
  calibrate-only). Memory notes in `~/.claude/projects/-Users-dnikolaidis-Desktop-divix-sglang-nvfp4/memory/`.
