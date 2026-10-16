# Lane-head (v1 gather-time head) — handoff

Written by team-lead on 2026-10-16. The previous agent, cache-tiers-review, was interrupted twice before it could
write this handoff. **No integration code has been written yet:** branch `lane-head` is `ram-prefetch-pool-serve`
plus this file. The design below was approved; start from it.

## Goal

Integrate the trained v1 head at gather time. Per MoE layer and verify, it scores how likely each routed expert is to
be routed again at v+1. The plan then gives GPU lanes to the likely-recurring experts and CPU lanes (the plan tail)
to the unlikely ones. Victim ranking, insertion, prefetch and the split model stay unchanged.

Expected served gain: +7.5..+11 VRAM hits/verify over today's ordering, about +1.2-1.4 ms/token traced. That sits
at the edge of stall-free noise (SD 1.6 ms/token), so the A/B must show the mechanism (hits and CPU-lane time), not
just ms/token.

## Status / next step

1. Read the skills `speculative-naming` and `env-var-conventions` (see `.claude/rules/modify-component-must-read.md`).
2. Build the design below, then write the CPU tests, then ask team-lead for a GPU slot (GPU test plus microbenchmark),
   then the served A/B.
3. Commit and push to `lane-head` often.

## Approved design

**Flag.** `SGLANG_OPT_LANE_HEAD_PATH = EnvStr(None)` in `environ.py`; the path is the switch.
- None or "": off. Nothing is installed and `_plan_miss_keys` stays `direct.miss_keys`. Behaviour is bit-identical.
- Set but unreadable or shape-mismatched (layers, experts, hidden, PCA rank): `ValueError` at
  `Exl3RamMissService.attach`, naming the path, before graph capture.
- A flag-on launch without CPU experts or without the fused plan is refused, as the existing miss order is.

**Where the keys are written.**
- Today `Exl3RamMissService.attach` sets `streamer._plan_miss_keys = gpu_residency.miss_keys[row]` (int64 [384],
  updated once per boundary in `_rank_victims`).
- `ExpertStreamer._gather_graph` passes it to `plan_graph_routes_fused`, whose `plan_unique_routes_kernel` sorts
  miss lanes by key descending. `type_lanes` gives the CPU the plan tail.
- `Exl3MoEMethod._apply_graph` (exl3.py ~737) sets `backend.cpu_input = (cpu_x, topk_weights)` just before
  `streamer.gather(topk_ids)`, so x and the weights are on device before the plan runs.
- New: one kernel per layer, launched inside `_gather_graph` on the gather's stream right before
  `plan_graph_routes_fused`. It is captured in the breakable decode graph, with no host node.
  - Inputs:
    - x [live, 5120] bf16 from `cpu_input[0]`;
    - `topk_ids` and `topk_weights` [live, 6];
    - `insert_scores[row]` f32 [384], the boundary's decayed score before this forward's counts, exactly as the
      replay used it;
    - a new `last_routed[row]` int32 [384], the forward each expert was last routed in. It is written in
      `GpuResidencyUpdater._apply` from `routed`, only when the head is on.
    - `forwards[1]` and the head weights.
  - Output: `keys[row]` int64 [384], for routed experts only, with `key = int64(P * 2^31) << 16 | (383 - e)` (the
    same tie-break shape as `residency_rank_keys`).
  - `attach` points `_plan_miss_keys` at `keys[row]`. `_rank_victims` is untouched.

**Head math (variant B).**
- z = PCA_L(x_{i*}), where i* is the token routing e with the largest gate weight.
- logit = b[e] + w·f + a·z + U[e]·z.
- f has 17 dense features:
  - token-count one-hot;
  - first-position one-hot;
  - gate weight;
  - log1p(score) and score==0;
  - log1p(age);
  - first touch;
  - resident flag;
  - live/6;
  - draft odds.
- Cost: one 5120x64 GEMV per (layer, token), then a 64-d dot plus 17 MACs per routed expert. About 4 MFLOP/layer,
  launch-dominated; the estimate is 5-10 µs/layer, 0.2-0.4 ms/verify, on the gate→plan critical path. Measure it.

**Files.**
- New:
  - `python/sglang/srt/layers/moe/lane_head.py`: a safetensors head file following the
    `ram_prefetch_learned_head.py` pattern (byte-range reads, no pickle). Metadata: layers, hidden, rank, feature
    list, draft-odds training value, training commit. Device load. Python reference `lane_keys_reference()`.
  - `python/sglang/kernels/jit/csrc/moe/lane_head.cuh` and `python/sglang/kernels/ops/moe/lane_head.py`.
  - `analysis/dsv41-drive/dspark/export_lane_head.py`: `head/B/L*.npz` + `pca.npz` → one file.
- Edited:
  - `environ.py`;
  - `exl3_ram_miss.py` attach: load, refuse, point `_plan_miss_keys`, install the hook on `Exl3RamMissRowBackend`;
  - `expert_stream.py` `_gather_graph`: call the hook when installed;
  - `expert_residency_gpu.py`: the `last_routed` buffer and its update, plus a keys-written counter;
  - `expert_hot_cache.py` `from_model`: budget accounting.

**Memory: ~28.5 MB of persistent VRAM in total** (not per layer):
- PCA 40×5120×64 bf16 = 26.2 MB;
- U 40×384×64 = 2.0 MB;
- the rest is under 0.3 MB.

When the flag is set, `from_model` reads the file metadata and subtracts these bytes from the hot budget before
sizing. The "Expert hot cache startup" log line gains `lane_head_bytes` and the resulting slots.

## Team-lead's five conditions (binding)

1. **Draft odds:** serve exactly what training saw. B0 had draft odds constant, and the feature was **zeroed** in
   training (fix `e0aa95a7c7`; without it C2 precision collapses to 0.32). Store the training value in the file
   metadata, have the kernel use it, and add a test that they agree. Do not feed 1.0.
2. **Startup calibration** (`SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP`, branch `prefill-merge-chunk`, owned by
   prefill-merge-2) measures free memory after model load. The head's bytes must be allocated before calibration
   measures. Add a CPU test with both flags on that checks the hot-cache capacity census still holds.
3. **Critical path:** report the microbenchmark in µs/layer, with and without the L2 flush (40 distinct weight sets
   plus a 256 MB rotating buffer touched between calls; the CLAUDE.md rule on working sets past the 96 MiB L2). If
   it adds more than 0.5 ms/verify, tell team-lead before any served run.
4. **A/B reporting:** the verdict comes from `serve_compare.py` (in `wt-stream-variance`, merged into
   `ram-prefetch-pool-serve` at `91cf7c7c02`). Report VRAM verify hits/verify, CPU-lane time/verify and gate→plan
   time alongside ms/token. Run one plain A/B against the current arm, not a/b/a/b.
5. **Generalization:** state which held-out data the hit numbers come from (below: C2 and base). Calibration can
   change slot counts per launch (1,035 → ~1,047); note whether the "resident" feature's distribution shifts.

## Tests to write

- CPU, `test/registered/unit/layers/moe/test_lane_head.py`: the Python reference on a captured C2 verify
  (`router.{x,ids,w}.bin`) with the exported file must reproduce `next_verify_head.py`'s predictions, and therefore
  the replay's lane order, **exactly** (use the same fp32 op order as `predict`).
- GPU, `test/manual/dsv41/test_lane_head_gpu.py`: kernel keys == reference keys for random and captured inputs, P
  within 1e-3 and an identical order. The microbenchmark goes in the same file.
- Flag off: bit-identical (existing exl3/expert tests unchanged).

## Trained head and data (divix01)

- Analysis dir: `/data/models/slang/nvfp4-work/ram-prefetch/cache-tiers-20261015/`
  - Head weights: `head/B/L00.npz` … `L39.npz` (variant B). Variant A is in `head/A/` and isn't used.
  - Held-out predictions and reports: `head/{c2,base}.pred-B.npz` and `head/{c2,base}.report-B.json`; training log
    `head/train-B.log`.
  - Hit replays: `{c2,base}.replay-headB.{txt,json}`. Baselines: `{c2,base}.replay.{txt,json}`.
  - Training and extraction scripts: `run_head.sh`, `run_head_train.sh` (logs `run_head.extract.log`,
    `run_head_train.log`), `next_verify_head.py`.
- Training shards: `/mnt/nvme1/cache-tiers-tmp/head-data/` (`train-b000…`, `c2`, `base`, `manifest.json`,
  **`pca.npz`**, the PCA64 fit on training rows only), 2.3 GB.
- Training data: B0 corpus `/mnt/nvme1/nvfp4-work/b0-corpus-100-20261012` b000-b003, 5.35M (verify, layer,
  expert) examples over 5,969 consecutive-verify pairs.
- Held out: C2 (658k examples over 752 pairs) and the base capture. Training takes 31 s on 4 threads, CPU only.
- Code: branch `cache-tiers` (last `04b3d7f793`); head fix `e0aa95a7c7`. Scripts in
  `analysis/dsv41-drive/dspark/`:
  - `vram_cache_replay.py`: the VRAM lane/hit replay;
  - `lane_cost_fit.py`: per-lane cost fit;
  - `next_verify_head.py`: extract, train, predict;
  - `ram_prefetch_replay.py`: the RAM-tier replay.
- Verify the paths with `ls` before relying on them.

## Results so far (held-out)

- Precision/recall at k = GPU lanes, C2: head B 0.595/0.378; table 0.471/0.299; today's score 0.408/0.259;
  oracle 0.851/0.541. Variant A (without the per-expert PCA term) reaches 0.557.
- Hits/verify:

  | capture | head B | table-k | today | next1-k oracle |
  |---|---|---|---|---|
  | C2 | 397.8 | 390.3 | 385.9 | 431.6 |
  | base | 373.9 | 362.9 | 360.7 | 405.4 |

- Inserts drop to 116.8/verify from 121.7, so there is no extra link traffic. All the gain is from lane choice.

## RAM-tier replay findings (context; not this build)

- The replay is calibrated at 190 rows/layer (nominal 210, ~20 taken by pool mechanics): 50.4 RAM misses/verify
  against 51.7 measured. Outputs: `c2.ramprefetch*.{txt,json}`, `c2.ramcalib.json`.
- Oracle next-verify prefetch on top of today's saves 19.5 / 33.4 / 44.3 misses at 20 / 40 / 80 reads. The
  near-miss gate is net negative (5% precision).
- **The churn fix is refuted:** speculative-first eviction gives 52.9 (worse), stale-first 49.9. `promote_landed`
  already stamps a landing with its victim's old recency. The earlier "13.5 induced misses" was a replay
  calibration artifact.
- 70% of the misses prefetch covers are cold (not routed in the last 9+ verifies). The lever for prefetch is
  precision.

## Held (owner's call; don't start)

- (c) A capture with the draft's final hidden states per verify (~2 h of box time, 100 sessions, ~16 GB).
- (d) v2: a 384-way next-verify head (shared 5120→256 projection, 2 attention layers over 240 token-layer vectors,
  route-history GRU state, outputs P(routed at v+1) and P(v+2 | v+1)). Only if (c) shows the draft states predict
  new experts well above 35%.

## Rules

- Follow `.claude/rules/divix01-run-protocol.md`:
  - push, then use a fresh worktree on divix01;
  - `PYTHONPATH=$PWD/python`, print `sglang.__file__`, interpreter `/data/models/slang/.venv/bin/python`;
  - read `${PIPESTATUS[0]}`;
  - run only the affected tests.
- Cores and locks:
  - CPU work under `taskset -c 0-63` with `OMP_NUM_THREADS`; offline analysis on cores 56-63; keep 64-71 free.
  - GPU work only with team-lead's OK, via `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`. Take
    `rowimg-disk.lock` before `cc-gpu.lock`.
  - GPU tests need `SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3`.
- Offline torch jobs set `CUDA_VISIBLE_DEVICES=""` before importing torch. (Incident: the first training run touched
  the GPU through AdamW's stream check during a served arm.) Some test utilities break on an empty value; use `99`
  there.
- Outputs go under `/mnt/nvme1` or `/data/models/slang/nvfp4-work/ram-prefetch/`, with `TMPDIR` set there.
- Never touch production (`cc-expert-prediction/dsv41-direct-prod`) or port 7867; harness ports 30017/30032 are
  fine. Never edit `DSV_OPTIMIZATION.md` (team-lead owns it).
- Git: never amend or force-push; stage files by name; use Edit/Write for repo files (a hook blocks sed/python
  edits).
- New behaviour stays off by default.
- Report to team-lead.
