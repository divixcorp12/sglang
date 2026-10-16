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
| `SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL` | off | 1 | split-model | DO 3.2: 65.9 vs 67.9 ms/token, one `-mem` pair | **enable** (with startup calibration, sect. 5 item 2) |
| `SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES` | n/a | n/a | split-model-calib | DO 3.4: no gain | leave 0 (default) |
| `SGLANG_ENABLE_SCHEDULER_MLOCK` | off | 1 | split-model | DO 3.4: team major faults 11,877 -> 20-250 | **enable** (risk in sect. 5 item 4) |
| `SGLANG_HICACHE_HOST_NUMA_NODE` | unset | 1 | split-model | DO 3.4: node-0 swap 11.6 GiB -> 5 MB at load; speed unchanged | **enable** |
| `SGLANG_MOE_PINNED_HOST_FULL_LAYERS` | unset | 0,1 | pin-full-layers (in rpps) | DO 3.1 | **enable** |
| `SGLANG_MOE_PINNED_HOST_MB` / `_NUMA_MB` | 106496 / 0:57344,1:49152 | 111530 / 0:59656,1:51874 | pin01 budget (`PIN01_BUDGET`) | DO 3.1 | **enable** with pin01 |
| `SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB` | unset (0) | 128 | prefill-merge-chunk | DO 3.7 / 3.4 re-score: -2.02, CI -3.7..-0.6 | **enable** |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | expandable_segments:True | (env only; arm on prefill-merge-chunk) | DO 3.7 expseg: ~1.5 GB recovered, no cost (-0.26, CI -0.39..-0.02) | **enable** |
| `SGLANG_MOE_HOT_GPU_MB` | 10840 | 13144 (1,035 slots) | prefill-merge-chunk (`EXPSEG_HOT_GPU_MB`) | DO 3.7 expseg2: -1.79, CI -4.51..-0.64; VRAM hit +1.9 pp | **enable** (or calibrate, sect. 5 item 3) |
| argv `--mem-fraction-static` | 0.78 | 0.8623 | `fraction_keeping_kv` | as above | **enable** |
| argv `--max-total-tokens` | none | 381184 | KV pin | DO 3.7 | **enable** |
| argv `--weight-loader-drop-cache-after-load` | none | present | split-model `-mem` arms (fix `7e77a55ba2`) | DO 3.4: works; shard cache was not the problem | **enable** (harmless, part of the measured arm) |
| `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION` | default **on** | 0 (`-fixedsplit`) | A/B control (REF 33.36) | none; it pins the table between arms | **leave default on** (owner rule; sect. 5 item 2) |
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
lane-tier's offline prefetch comparison, as you said. Note the learned head is a file outside the repo.

## 5. For your decision

1. **The best arm was never served without the prefetch.** Every component's A/B (split model, pin01, merge chunk,
   expseg, prefill share) ran on top of the prefetch + promote + pool-serve stack. If the prefetch stays off,
   production is a combination nobody served.
2. **CPU split calibration.** The arm ran `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION=0` (`-fixedsplit`, an A/B
   control) and the split model on the **fixed offline costs** (`split-model` predates `split-model-calib`). With
   split-model-calib merged, the model's costs are always calibrated at startup when it is on (+~1.1 s inside the
   first request; on failure the table types the lanes). Calibrated without refinement was **never served as an A/B**;
   the nearest is `-refine` vs fixed costs, +0.1 ms/token (CI -1.7..+1.9). I propose leaving the table calibration at
   its default (on), which differs from the measured arm; with the model on, the table only types lanes when the
   model is off or failed.
3. **Hot-cache size is a probe-derived constant, not a startup value.** 13144 MiB / 0.8623 / KV 381,184 come from
   `hot_mb_for_margin` on one smoke (3 slots held back). The owner's startup rule points at
   `SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP`: served once, safe, neutral (+0.62, CI -1.88..+2.60; ~+11 slots), but
   with A1/A2/A3 bundled (indexer budget 32, chunk kernels, merge budget 32), +18-24 s startup, and the final design
   (calib11b) was smoke-tested, not A/B'd. The tip's residue-walk fix `a75bb27afb` has no GPU run. Options: (a) ship
   13144 constant (measured), (b) ship calibration on. I lean (a) for this launch.
4. **Node-0 memory with mlock + pin01 budget.** The `-mem` arms left node 0 at 0.04-0.24 GiB free during load
   (expseg 0.04, below the low watermark) and ~0.14 GiB under mlockall; those runs had `prelaunch_evict.py` and an idle
   box. `launch_prod.sh` does no eviction, and production's co-tenants are not the A/B box state. Risk: swap or OOM at
   load. Not a loss, but unmeasured under production conditions.
5. **Split model's own evidence is one pre-serve_compare pair** (65.9 vs 67.9); not re-scored with the stall-aware
   rule. pin01's 70.5 vs 76.2 compares against base-r2, which serve_compare flags as a stalled run. Both are in the
   best arm, so I list them as enable, but neither has a clean-CI number of its own.
6. **The arm was measured on the instrumented host build** (`SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX` set, plus the route
   and metrics files). Production runs without them; expected equal or faster, not measured.
7. **Not served, flag stays off:** `cpu-out-rows` (`STORE_OUT`), `SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS` and
   `SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL` (both only inside the calibration arm or not served), residency
   tiebreak (neutral).

## 6. Phase 2 test plan (after your go)

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
