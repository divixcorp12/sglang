# Merge plan: the October 14-16 work into `master` (phase 1, nothing merged yet)

Branch `integrate-1016`, cut from `origin/ram-prefetch-pool-serve` at `975f56fb3e`. Written 2026-10-16 by the
merge-clean agent. Sources: `DSV_OPTIMIZATION.md` (cited as "DO 3.x"), `DSV41_REFERENCE.md` section 33 ("REF 33.x"),
`divix01:/mnt/nvme1/share64-ab-20261016/resolved-env.json` (the best arm's resolved env and argv), and
`git merge-base --is-ancestor` / `git merge-tree --write-tree` against today's `origin`.

The best arm is `dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-promote-poolserve-fixedsplit-pin01-splitmodel-mem-hicache1-mergechunk-expseg`,
served from `lane-head-ab` `022acb70b9` (= `d269f7aa90` + one analysis script). Its code is
`ram-prefetch-pool-serve` (as of `91cf7c7c02`) + `split-model` (`ccb87893fa`) + `prefill-merge-chunk` (up to
`abaae5fdd0`) + `lane-head` code (flag off). `ram-prefetch-pool-serve` itself adds only docs and the prefill share
since then. **`master` (`a7cce264fe`) is an ancestor of `ram-prefetch-pool-serve`, which is 137 commits ahead**, so
the final merge to master is a fast-forward of `integrate-1016`.

## 1. Inventory: branches pushed since 2026-10-14

"master" / "arm" = the tip is an ancestor of `origin/master` / of `d269f7aa90`. "rpps" = of `ram-prefetch-pool-serve`.

| branch | tip | master | arm | rpps | what it does | env flag(s) | evidence | verdict |
|---|---|---|---|---|---|---|---|---|
| `ram-prefetch-pool-serve` | `975f56fb3e` | no (ff-able) | partly (all code, minus the share commit) | — | NVMe-to-RAM prefetch (GPU scorer, L+2 T3 head, idle drive, cap 2), promote (F125), pool serve, GPU scratch drip, conf gate, pin01, JIT build speed, serve_compare; master merged | prefetch family (sect. 4), `SGLANG_MOE_PINNED_HOST_FULL_LAYERS`, `SGLANG_DSV41_GPU_DRIP_SLOTS`, `SGLANG_DSPARK_TOP_PROB_GATE` | pin01 DO 3.1 (70.5 vs 76.2 ms/token, L0+1 20.5 -> 13.0 ms/verify); drip/confgate neutral DO 3.4 re-score; promote REF 33.37 (no gain) | **base of the integration**; pin01 merge+enable; prefetch flags PENDING; drip, conf gate merged-off |
| `prefill-merge-chunk` | `25d8c3e679` | no | yes to `abaae5fdd0` (+2 commits: residue-walk fix `a75bb27afb`, handoff) | no | contains all of `split-model`. Triton FlashMLA merge in token chunks (A3); per-chunk SWA/extra kernels (A2); `empty_cache` on larger prefill; startup hot-cache calibration (VMM, retire/grow, trial capture); prefill peak logger; `-mergechunk*`/`-expseg*` arms; prelaunch evict / memsample | `SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (arm only), `SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS`, `SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP`, `SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL`, `SGLANG_DEBUG_PREFILL_PEAK_LOG` | mergechunk-r2 vs A **-2.02 ms/token, CI -3.7..-0.6** (DO 3.4 re-score); expseg2 vs mergechunk-r2 **-1.79, CI -4.51..-0.64** (DO 3.7); calibration A/B +0.62, CI -1.88..+2.60, safe (DO 3.7 "Served A/B (e)") | **merge**; enable merge budget 128 + expseg + 13144 MiB hot cache; calibration / A2 / empty-cache flags: see sect. 5 |
| `split-model` | `ccb87893fa` | no | **yes** | no | token-weighted CPU/copy split model; residency tiebreak; scheduler mlockall; HiCache host pools on a NUMA node; Engram native cache knob; draft drop-cache fix; prelaunch evict | `SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL`, `..._COSTS`, `..._RESIDENCY_TIEBREAK`, `SGLANG_ENABLE_SCHEDULER_MLOCK`, `SGLANG_HICACHE_HOST_NUMA_NODE`, `SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB` | split model 65.9 vs 67.9 (one pair, pre-serve_compare, DO 3.2); HiCache node 1 fixes node-0 swap, speed unchanged (DO 3.4); tiebreak neutral, parked (DO 3.2); Engram cache keep 5 GiB (DO 3.4) | **merge** (via `prefill-merge-chunk`); enable split model, mlock, HiCache node 1; tiebreak off |
| `split-model-calib` | `e55aeaae6d` | no | no (3 commits) | no | the split model's costs measured at startup (CPU grid, copy lane, landing, stream coupling), clamped; failure -> table; optional live refine | `SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES` (0 = off). Calibration runs whenever the split model is on | validated on the real model, calibrated vs offline agree at the served shape (DO 3.3); `-refine` vs A' +0.1, CI -1.7..+1.9 (DO 3.4) | **merge+enable** (owner rule); refine 0 |
| `cpu-out-rows` | `3be009c47f` | no | no | no | kernel stores a verify job's first write instead of the engine memset | `SGLANG_DSV41_CPU_EXPERTS_STORE_OUT` | bit test passed; bench saves ~10-25 us/job; **never served** (DO 3.10) | **merge-off** |
| `read-amp` | `ca965e7c06` | no | no | no | RAM-miss accounting: `host_fill_rows`, per-root `fill_bytes`, `eager_evictions` (appended words; also fixes the stale `_gather_cached` test stub) | none (observability, always on) | 295 tests passed, 2 mutants caught (DO 3.6) | **merge** (no flag; native counter words appended) — lead to confirm |
| `expert-breakdown` | `d79f05e077` | no | no | no | InstrBuild `lane_kind`/`svc_routed` job-trace events + `expert_breakdown.py` | none (instrumented builds only) | used for `breakdown-20261015` (DO 3.2, 3.5) | **merge-off** (instr-only) — optional |
| `misscut-replay` | `f5823279fd` | no | no | no | offline replays: graded miss cut, split model, tiebreak | none | graded+ ~0.5-1 ms/token, not prioritised (DO 3.9) | analysis only: **merge** (no runtime code) or skip — lead's call |
| `cache-tiers` | `04b3d7f793` | no | no | no | offline replays: VRAM cache, lane cost fit, next-verify head, RAM-prefetch replay | none | DO 3.5, 3.8 | analysis only: **merge** or skip — lead's call |
| `stream-variance` | `79d8147098` | no | yes | yes | `serve_compare.py` | none | DO 3.4 | already in rpps |
| `jit-build-speed` | `d81d37dcf1` | no | yes | yes | JIT compiler launcher, parallel warm-up | `SGLANG_JIT_COMPILER_LAUNCHER` | DO 3.11 | already in rpps |
| `pin-full-layers` | `495e9b3dd3` | no | yes | yes | whole layers 0,1 in the pinned tier | `SGLANG_MOE_PINNED_HOST_FULL_LAYERS` | DO 3.1 | already in rpps; **enable** |
| `gpu-scratch-drip` | `fb6042110e` | no | yes | yes | GPU drip | `SGLANG_DSV41_GPU_DRIP_SLOTS` | drip3/drip4 neutral (DO 3.4 re-score) | already in rpps; off |
| `dspark-conf-gate` | `9f226a6114` | no | yes | yes | DSpark top-prob verify gate | `SGLANG_DSPARK_TOP_PROB_GATE` | confgate run stalled (DO 3.4) | already in rpps; off |
| `diag/pool-skip` | `7906c1cd58` | no | yes | yes | diagnostic | — | — | already in rpps |
| `f125-merge`, `ab/F125-snapshot` | `ee528f2c74`, `b992e11134` | yes | yes | yes | promote (F125) | `SGLANG_DSV41_RAM_PREFETCH_PROMOTE` | REF 33.37 | already in master |
| `prefill-share-default` | `a7cce264fe` | yes | no | yes | prefill share in `base_env` | `SGLANG_DSV41_ENABLE_PREFILL_SHARE` | -3.46 ms/token, CI -4.91..-1.55 | already in master, enabled |
| `prefill-share-size`, `-ps` | `4096b28933`, `57e64ed0e1` | no | no | no | startup-sized share (0.15 x usable rows) | `SGLANG_DSV41_FORCE_PREFILL_SHARE_ROWS` | under test | **don't merge yet**; `-ps` merges cleanly onto the chain below |
| `mirror-weight-calib`, `mirror-weight-ab` | `cb77329dd8`, `dce869d0a2` | no | no | no | startup mirror-weight probe (`-calib` also carries split-model-calib) | `SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION` | +0.04 ms/token, dropped (DO 3.3) | **don't merge** |
| `lane-head`, `lane-head-ab` | `eb1a14c8c0`, `022acb70b9` | no | yes (code) | no | gather-time learned lane head | `SGLANG_OPT_LANE_HEAD_PATH` | +19.5 hits/verify, ms/token neutral; held | **don't merge** |
| `lane-tier-replay` | `1cbfdfc7ef` | no | no | no | tier-priced replays, share sweep, share A/B report | — | DO 3.8 | **don't merge**: contains all of `lane-head` |
| `drip-hot` | `21fe80bc93` | no | no | no | drip-into-hot-cache design + replay | — | fails build gate | **don't merge** |
| `hx-check/*` | — | | | | hx checker scratch refs | | | ignore |

## 2. Env the best arm sets beyond today's master `prod_env()`

Diff of `resolved-env.json["on"]` against master's `prod_env()` (run on divix01 from `git show origin/master:...arm_env.py`).
Values the arm only states at their library default (gate 0, drip 0, tiebreak 0, mirror dynamic 0, decay 0.98,
Engram cache 5, chunk kernels 0, hot calibration 0, lane head "", split layers "", costs "", mirror caps "", prefetch
top-k 0, margin floors "") need no change and are left out. `SGLANG_DSV41_CPU_SPLIT_MISS_CUT=1`/`_MAX=3` are the
library defaults already (`environ.py`), so production has the cut today (and the split model replaces it when on).
The dump omits `CXX`, `EXL3_MOE_CPU_PIN`, `MKL_NUM_THREADS`, `PYTHONDONTWRITEBYTECODE`; `CUDA_VISIBLE_DEVICES=""`
and `CUDA_ARCH` are the recording shell's, not the server's.

| flag / argv | master | best arm | branch | evidence | proposal |
|---|---|---|---|---|---|
| `SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL` | off | 1 | split-model | DO 3.2: 65.9 vs 67.9 ms/token (-2.9%), one `-mem` pair | **enable** (with startup calibration, sect. 5 item 2); miss cut 1/3 stays at its library default, stated explicitly |
| `SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES` | n/a | n/a | split-model-calib | DO 3.4: no gain | leave 0 (default) |
| `SGLANG_ENABLE_SCHEDULER_MLOCK` | off | 1 | split-model | DO 3.4: team major faults 11,877 -> 20-250 | **enable** (risk in sect. 5 item 4) |
| `SGLANG_HICACHE_HOST_NUMA_NODE` | unset | 1 | split-model | DO 3.4: node-0 swap 11.6 GiB -> 5 MB at load; speed unchanged | **enable** |
| `SGLANG_MOE_PINNED_HOST_FULL_LAYERS` | unset | 0,1 | pin-full-layers (in rpps) | DO 3.1 | **enable** |
| `SGLANG_MOE_PINNED_HOST_MB` / `_NUMA_MB` | 106496 / 0:57344,1:49152 | 111530 / 0:59656,1:51874 | pin01 budget (`PIN01_BUDGET`) | DO 3.1 | **enable** with pin01 |
| `SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB` | unset (0) | 128 | prefill-merge-chunk | DO 3.7 / 3.4 re-score: -2.02, CI -3.7..-0.6 | **enable** |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | expandable_segments:True | (env only; arm on prefill-merge-chunk) | DO 3.7 expseg: ~1.5 GB recovered, no cost (-0.26, CI -0.39..-0.02) | **enable** |
| `SGLANG_MOE_HOT_GPU_MB` | 10840 | 13144 (1,035 slots) | prefill-merge-chunk (`EXPSEG_HOT_GPU_MB`) | DO 3.7 expseg2: -1.79, CI -4.51..-0.64; VRAM hit +1.9 pp | **enable** as the start size the startup calibration grows or trims from (sect. 5 item 3) |
| argv `--mem-fraction-static` | 0.78 | 0.8623 | `fraction_keeping_kv` | as above | **enable** |
| argv `--max-total-tokens` | none | 381184 | KV pin | DO 3.7 | **enable** |
| argv `--weight-loader-drop-cache-after-load` | none | present | split-model `-mem` arms (fix `7e77a55ba2`) | DO 3.4: works; shard cache was not the problem | **enable** (harmless, part of the measured arm) |
| `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION` | default **on** | 0 (`-fixedsplit`) | A/B control (REF 33.36) | none; it pins the table between arms | **on** (owner rule; stated explicitly in `dspark_env`, sect. 5 item 2) |
| `SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP` | unset | 0 | prefill-merge-chunk | DO 3.7: served A/B neutral (+0.62, CI -1.88..+2.60), margin held under serving | **enable** (owner rule: replaces the hard-coded 13144 with a measured size; sect. 5 item 3) |
| prefetch family | unset | on | rpps | sect. 4 | **PENDING** |
| `SGLANG_DSV41_EXPERT_JOB_TRACE_*`, `SGLANG_DSV41_EXPERT_TRACE_PATH`, `SGLANG_MOE_HOT_METRICS_FILE` | unset | run paths | measurement | — | **not for production** (sect. 5 item 6) |

Where they go: `dspark_env()` (production is DSpark; none of these was measured on the non-DSpark base recipe), plus
the three argv items in `ServerArgs.argv()` for `dspark=True`. Each gets a comment citing the evidence and a pin in
`test_dspark_recipe.py`, as `a7cce264fe` did.

One structural catch: `both_cpu_ab.fraction_keeping_kv` and `hot_mb_for_margin` read `arm_env.DSPARK_HOT_GPU_MB` and
`DSPARK_MEM_FRACTION_STATIC` as **the OFF -mem arm's measured init** (10840 / 0.78), and every DSpark arm inherits
`dspark_env()`. Proposal: keep those two names at their measured values for the arm math (renamed or copied into
`both_cpu_ab` as the OFF-arm constants), and add production constants for 13144 / 0.8623 / 381184. Historic arms then
keep their meaning; the new flags in `dspark_env()` reach every DSpark arm unless the arm pins them (most already do:
`both_cpu_ab` pins split model, mlock, full layers, HiCache node, merge budget and alloc conf per arm).

## 3. Merge order and expected conflicts

Dry runs chained with `git merge-tree --write-tree` (each step on the previous step's tree) from `975f56fb3e`:

| # | merge | conflicts |
|---|---|---|
| 1 | `prefill-merge-chunk` `25d8c3e679` (brings `split-model`) | **none** |
| 2 | `split-model-calib` `e55aeaae6d` | `analysis/dsv41-drive/dspark/both_cpu_ab.py`, `test/registered/unit/scripts/test_both_cpu_ab.py`: both sides add arms at the same anchor (`-fixedcosts`/`-refine` vs `-mergechunk*`/`-expseg*`). Resolve by keeping both blocks; code files auto-merge (`environ.py` clean) |
| 3 | `read-amp` `ca965e7c06` | none |
| 4 | `cpu-out-rows` `3be009c47f` | none |
| 5 | `expert-breakdown` `d79f05e077` (optional) | none |
| 6 | `misscut-replay`, `cache-tiers` (optional, analysis only) | none |
| 7 | arm_env changes (one commit) | — |
| later | `prefill-share-size-ps` `57e64ed0e1`, once its A/B is decided | none on top of 1-6 |

Then the integration tests (phase 2), then a fast-forward of `master` on your go.

## 4. Prefetch flags: PENDING (code merged, flags not set in arm_env)

Set by the best arm, all in `ram-prefetch-pool-serve` (already the base, so the code is merged by construction):

`SGLANG_DSV41_RAM_PREFETCH=1`, `_SCORER=gpu`, `_HORIZON=2`, `_LEARNED_HEAD=/data/models/slang/nvfp4-work/ram-prefetch/l2-t3-head/t3-f109-calib-b075.safetensors`,
`_PER_LAYER=2`, `_PER_TOKEN=1`, `_SPEC_SHARE=2`, `_IDLE_DRIVE=1`, `_IDLE_PIECES=2`, `_IDLE_SPREAD=1`,
`_IDLE_DEADLINE_US=4000`, `_PROMOTE=1`, `_POOL_SERVE=1` (and `_TOP_K_ONLY=0`, `_MARGIN_FLOORS=""` at defaults).

Evidence on record: L+2 / T3 / cap 2 captures REF 33.29-33.35 (mixed; per-session table in the untracked
`RAM_PREFETCH_FIXED_SPLIT_NEXT_HANDOFF.md` shows cap 2 faster in 4 of 8 sessions); promote REF 33.37: **no gain**
(wall +2.6%, matched -0.14 ms/record, one pair). I found no served pool-serve result in the repo docs. Waiting on
lane-tier's offline prefetch comparison, as you said. DSV41_REFERENCE 33.14-33.37 has no clean served win for the
stack.

**The learned head is a model file outside the repo**, and production would depend on it if the prefetch is enabled:
`divix01:/data/models/slang/nvfp4-work/ram-prefetch/l2-t3-head/t3-f109-calib-b075.safetensors`, 4,758,132 bytes,
mode 0600, dated 2026-10-13, SHA256 `f0d7a1b10e029480bcb9687437d7880135d756ea57f186d9a9cd41729fbd5309`. `ram-prefetch/`
is a scratch tree for run outputs. Proposal: copy it to a production asset path beside the other model inputs,
`/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-prefetch-heads/t3-f109-calib-b075.safetensors` (next to
`dsv41-dspark-draft` and the resident-set JSON), read-only. Pin the path and its SHA256 as constants in `arm_env.py`,
and have `launch_prod.sh` refuse to start on a missing file or a checksum mismatch. At 4.7 MB it could also live in the
repo, but no model weights are tracked in git today, so I would not start here.

## 5. For your decision

1. **The best arm was never served without the prefetch.** Every component's A/B (split model, pin01, merge chunk,
   expseg, prefill share) ran on top of the prefetch + promote + pool-serve stack. If the prefetch stays off,
   production is a combination nobody served.
2. **CPU split calibration: on (owner rule).** The arm ran `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION=0`
   (`-fixedsplit`, an A/B control) and the split model on the **fixed offline costs** (`split-model` predates
   `split-model-calib`). Production gets the table calibration on, stated explicitly, and split-model-calib's startup
   fit, which runs whenever the model is on (+~1.1 s inside the first request; on failure the table types the lanes),
   with refinement off (`REFINE_BATCHES=0`, the default). Calibrated without refinement was **never served as an A/B**;
   the nearest is `-refine` vs fixed costs, +0.1 ms/token (CI -1.7..+1.9). With the model on, the table types lanes
   only when the model is off or has failed.
3. **Hot-cache size: startup calibration on (owner rule).** 13144 MiB comes from `hot_mb_for_margin` on one smoke (3
   slots held back), so it is not a startup value. `SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP=1` measures the size at
   every launch. Its served A/B was safe and neutral (+0.62, CI -1.88..+2.60; ~+11 slots). Points to settle in phase 2:
   - **Bundling.** The served `-expseg-calib` arm also carried A1/A2/A3 (indexer budget 32, chunk kernels, merge
     budget 32). Proposal: enable the calibration with merge budget 128, chunk kernels off and indexer 128, as in the
     best arm, so the only change is the calibration. The calibration measures whatever transient is configured, so it
     is valid either way. A1/A2/A3 stay off until they are measured on their own.
   - **Start size.** The calibration grows or trims from `SGLANG_MOE_HOT_GPU_MB`. Keep 13144 as the start value, and
     keep the fraction 0.8623 and the KV pin 381,184 (the calibration refuses to start if KV falls below the pin).
   - **Cost and state of the code.** Startup takes +18-24 s. The final design (calib11b, `abaae5fdd0`) was
     smoke-tested, not A/B'd. The tip's residue-walk fix `a75bb27afb` has not run on the GPU, so phase 2 runs the
     calibration GPU tests at the tip. A failed calibration falls back to the configured size and does not fail the
     launch.
4. **Node-0 memory with mlock + pin01 budget.** The `-mem` arms left node 0 at 0.04-0.24 GiB free during load
   (expseg 0.04, below the low watermark) and ~0.14 GiB under mlockall. Those runs had `prelaunch_evict.py` and an idle
   box. Node 1 is covered by the launch floor in section 7. Node 0 has no floor today. Proposal: the same check logs
   node 0's MemFree and warns below the arms' observed start value. It does not refuse there, since the arms passed
   with as little.
5. **Split model's own evidence is one pre-serve_compare pair** (65.9 vs 67.9); not re-scored with the stall-aware
   rule. pin01's 70.5 vs 76.2 compares against base-r2, which serve_compare flags as a stalled run. Both are in the
   best arm, so I list them as enable, but neither has a clean-CI number of its own.
6. **The arm was measured on the instrumented host build** (`SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX` set, plus the route
   and metrics files). Production runs without them; expected equal or faster, not measured.
7. **Stays off:**
   - lane head: held;
   - GPU drip and drip-hot: neutral, and drip-hot was not built;
   - residency tiebreak: neutral;
   - decay 0.995: no detectable gain;
   - split-model refinement: no gain;
   - mirror weights: dropped;
   - the DSpark confidence gate: its one served run was a loss, but that run is STALL-flagged by serve_compare (DO
     3.4), so the loss is not evidence against it either. It is simply unmeasured;
   - `SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS` and `SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL`: not measured alone;
   - `cpu-out-rows` (`STORE_OUT`): bit-identical, ~10-25 us/job, never served. It rides along with its flag off
     (clean merge, small, and the next larger served A/B can carry it).
8. **Never acted on, in no arm: `SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM=1`.** REF 33.22 records it as -1.16 ms/token,
   8/8 sessions (2026-09-27), left "for the owner". It was measured before the copy engine, and before DSpark's
   graphed verify. REF (around line 3968) also records an earlier 131.8 vs 131.9 with no gain, part of it lost to C1
   sharing SMs, which the copy engine removes. **Needs a check before any enable, not part of this merge:**
   - does the side stream (shared expert + DIRECT commit) capture in the breakable decode graph with the DSpark
     verify;
   - does it order correctly against the copy engine's per-layer gate and the lease chain;
   - does `expert_stream_requirements` refuse it for EXL3, as it does `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`.
   Then a served pair. The code is already on master (`environ.py`), so no merge is needed.

## 7. Production launch safety: the node-1 floor

The served arms needed node 1 to hold, at launch:
- the HiCache host pools (bound there by `SGLANG_HICACHE_HOST_NUMA_NODE=1`, ~11.75 GiB);
- node 1's 51,874 MiB share of the larger pinned tier;
- the rest of the scheduler.

`prelaunch_evict.py` (on `prefill-merge-chunk`, so it arrives with merge 1) set the floor at **64.2-64.9 GiB of node-1
MemFree** in the arms' records (DO 3.3: "76.64 GiB free, above the 64.21 floor"). The formula is HiCache + rest + 1
GiB margin, where "rest" comes from the arm's last memsample or defaults to arm A's 51.24. It waits up to `--wait-min`,
evicting each minute, and exits 3 if the floor is never met. `launch_prod.sh` has no such step. Another project's page
cache (e.g. mimo-reap's 620 GB on nvme4) can fill node 1, and then the populate/mbind of the HiCache pools pushes node
1 under its low watermark and swaps, which is the failure the arms hit in `-decay995` (DO 3.6).

**Proposal** (phase 2, its own commit, after the arm_env commit):
1. `launch_prod.sh start` runs `prelaunch_evict.py --log <prod log dir>/prelaunch-<ts>.json --wait-min 0 --hicache-gib
   <from arm_env> --margin-gib 1.0` before taking the GPU lock.
   - `--wait-min 0` means one eviction round, then a decision. Production should not sit in a hidden 30-minute wait.
   - It evicts only this model's draft, row images and Engram files, through `posix_fadvise(DONTNEED)`, without root.
     Production is down at that point, so dropping that cache costs no other server of ours anything. lane-tier's
     arms on port 30032 read the same files, so the script logs whether port 30032 is up and skips the eviction (but
     still checks) while it is.
   - A new arm_env constant, `PROD_NODE1_FLOOR_GIB`, is set from the arms' record (rest 51.24 + HiCache 11.75 + margin
     1.0 = 63.99, rounded up to 65). It is used instead of `--previous`, because production has no memsample of its
     own.
2. **Below the floor, refuse, and do not launch.** Exit non-zero with a message giving:
   - the floor and node 1's MemFree and FilePages, before and after eviction;
   - the biggest page-cache holders on node 1, where `fincore`/`numastat -p` can name them without root;
   - the fix: free node 1 (as root, `echo 1 > /proc/sys/vm/drop_caches`, or stop the co-tenant), then rerun.
   - Refusing is better than launching into swap: a swapping launch takes 6+ minutes and serves slowly with no error.
3. **Node 0:** log its MemFree and warn below the arms' observed start (item 5.4). No refusal.
4. A test in `test_launch_prod.py`, run with `DRY_RUN=1` and a fake meminfo, checks three things: the launch calls
   the check before the lock, refuses below the floor, and proceeds above it.

Alternative, if you'd rather not have production evict anything: step 2 alone, a pure refusal check. The operator then
evicts by hand. I recommend the eviction round, because the files it drops are our own and production is down when it
runs.

## 8. Phase 2 test plan (after your go)

Per the run protocol: push `integrate-1016`, divix01 worktree `wt-merge-clean` at the pushed commit and one at the
merge base (`975f56fb3e`), `PYTHONPATH=$PWD/python`, print `sglang.__file__` and `git log -1`, `taskset -c 0-55`,
`OMP_NUM_THREADS=8`, `TMPDIR=/mnt/nvme1/merge-clean/tmp`, read `${PIPESTATUS[0]}`, same selection at both commits and
diff counts.

- CPU: `benchmarks/dsv41_baseline/test_*.py` (from inside that directory); `test/registered/unit/scripts/test_both_cpu_ab.py`;
  `test/registered/unit/kernels/test_exl3_split_model*.py test_exl3_ram_miss*.py test_*expert*.py`;
  hot-cache calibration / VMM / retire / prefill-cache-release / peak-log / merge-chunk unit files; HiCache NUMA and
  memory-lock unit files; read-amp's `test_expert_row_source.py`; cpu-out-rows' CPU tests. Exact selection recorded
  with the result.
- GPU, under `flock cc-gpu.lock taskset -c 32-63`, `SGLANG_EXL3_SRC` set, warm JIT modules first:
  `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`, `test_split_model_calibration_cuda.py`,
  `test_hot_cache_vmm_gpu.py`, the merge-chunk FlashMLA test, `test_exl3_cpu_store_out.py`;
  `test_expert_residency_gpu.py` in its own process (known pollution, DO 3.7).
