# Handoff: DSV4.1 / DSpark decode step time — where the L+2 prefetch work stands and what sets the step

Updated 2026-10-13 evening. Laptop checkout `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4`, `master` at
`163337b11c` or later (check `git log` for the 33.33 follow-up). Verify the checkout and the artifacts below before
relying on any number here; `DSV41_REFERENCE.md` is the record, this file is the entry point.

The earlier version of this file was the stage plan for the learned L+2 heads (Stage 1 offline comparison, Stage 2
fast path, Stage 3 served check). All three stages have now been run. That plan is archived at
`~/.hx/specs/F124/RAM_PREFETCH_L2_HEADS_NEXT_HANDOFF.stage-plan-2026-10-13.md`.

## Standing rules (unchanged, they bind every session)

- **Production on divix01 is stopped.** Never start, stop or restart it; never run anything in
  `cc-expert-prediction/dsv41-direct-prod`. Production listens on 7867. Harness experiment servers (ports 30017/30018
  for `both_cpu_ab.py`, 30032 for `spec_margin_capture.py`) are fine. The owner alone changes production defaults:
  RAM prefetch stays default off and the horizon stays 1. Propose, don't flip.
- Run protocol: `.claude/rules/divix01-run-protocol.md`. Code moves only by push to `origin` and a private divix01
  worktree (`git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-<name>
  origin/<branch>`); `PYTHONPATH=$PWD/python`; print `sglang.__file__`. CPU work under `taskset -c 0-63` with
  `OMP_NUM_THREADS`; cores 64-71 stay free. GPU work under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c
  32-63`. Lock order: `rowimg-disk.lock`, then `cc-gpu.lock`. Read `PIPESTATUS` (`$pipestatus[1]` in laptop zsh).
- Every served run needs its python tree registered in the run worktree's untracked
  `benchmarks/dsv41_baseline/generations.json` (copy it from `../wt-l2-ab/`, then `generations.register(tree, label)`).
- Check `ps` and `nvidia-smi` immediately before serving. **The owner also runs other GPU jobs (a REAP job today)
  that do not take `cc-gpu.lock`**; one took 26.6 GiB during our model load and the server died with CUDA OOM
  (attempt kept in `step-time-20261013/failed-oom-1658/`). If load reports `avail mem` well under ~29 GB, stop and ask.
- New outputs only on `/mnt/nvme1` or `/data/models/slang/nvfp4-work/ram-prefetch/`; never nvme0, nvme2 or nvme4
  (expert mirror drives).
- Laptop: never `uv run`, `uv sync` or `pip install`. Tests go through
  `/Users/dnikolaidis/.hx/checks/divix01-pytest [--gpu] <files> -q -p no:randomly`; run only affected tests.
- Git: never amend or force-push; stage by name; never commit `python/uv.lock`, `.python-version` or this file.
- Planning and execution go through hx (hx-brainstorming, hx-planning, hx-running), not superpowers planning or plan
  mode.

## The headline: what sets the verify step (§33.33, today)

Three instrumented captures, in the order prefetch off → native L+2 → prefetch off again, 8 sessions each, broken down
per layer by `layer_misses.py`. Per DSpark verify (about 3.6 tokens), median:

| | off | native L+2 | off2 (clean baseline) |
|---|---:|---:|---:|
| wall ms | 297.3 | 260.6 | 267.1 |
| per-layer gate waits, ms (share) | 238.4 (80%) | 203.7 (78%) | 209.1 (78%) |
| outside the gate waits, ms | 58.9 | 56.9 | 58.0 |
| forced misses / covered by pool / NVMe reads left | 87.9 / 0 / 87.9 | 89.1 / 13.9 / 75.1 | 86.6 / 0 / 86.6 |
| wait of a layer with no miss, ms | 2.87 | 2.67 | 2.79 |
| layer 0 wait (never coverable; drift control), ms | 12.64 | 10.75 | 10.77 |
| lifetime read ms per RAM miss | 2.16 | 1.34 | 1.46 |

A verify is about **42% a fixed per-layer floor** (40 layers × ~2.8 ms, paid even with no miss), **36% NVMe miss
reads** (~87 per verify, ~1.1 ms each) and **22% everything else** (attention, GEMMs, draft, host).

- **Drive read latency moves the step one for one.** The first "off" run followed the REAP job; its reads were 48%
  slower per miss and its verify 11% longer, layer 0 included. That drift is about 5x the prefetch effect. Two arms
  at different drive states cannot be compared. Use layer 0's wait and lifetime `read_ms / ram_misses` as controls.
- **Native L+2 vs off2: -2.4%** (260.6 vs 267.1), in line with F108's 0.984. It covers ~14 of ~88 misses per verify.
  Its uncovered layers are 0.3 ms *faster* than off2's at matched (row, misses): the §33.16 contention cost
  (speculative reads slowing demand reads) is no longer visible with the idle-drive rule.
- **§33.32's "the step is not set by demand NVMe rows" was wrong** and is corrected in place. The prefetch is small
  because it covers one miss in six, not because misses don't matter.
- **The floor's composition was not re-measured.** §33.12 found it was mostly the CPU "hit job" (the CPU computing
  the layer's RAM-resident experts, ~2.5 ms/layer). The captures below hold the per-engine event logs to check it.

## Where the learned L+2 head work stands

| Step | What it found | Record |
|---|---|---|
| B0-B1, R1, R2 | learned L+2 heads beat the native gate offline at cap 2 (T3 dev recall 0.319 vs 0.251); T5/G0 added nothing | §33.25-33.28 |
| F108 | native gate retargeted to L+2 is viable; step ratio 0.984 vs off, 0.958 vs L+1 floors; reverse order deferred by owner | §33.29 |
| F109 (Stage 1) | at the serving cap of 1 read/layer, T3 at budget 0.75 reads 25% fewer rows than the gate at 1.0 for 12.6% more on-time useful rows (calib recall 0.235 vs 0.195, precision 0.523 vs 0.326); frozen T3 threshold 0.1302 | §33.30 |
| F123 (Stage 2) | fused fast path: T0 61.4 us, T3 65.6 us per layer at width 6 (complete path was 137 / 213 us); parity verified; T3 chosen | §33.31 |
| F124 (Stage 3, quick) | T3 served opt-in on the side stream; quick check (one order, 4 sessions, cross-day baseline, **not a verdict**): T3 / fresh native L+2 step ratio 1.026; T3 is the better reader (demand 18.6 vs 20.1 NVMe rows/token, wasted 3.7 vs 7.4) but the expected step gain is ~0.5%, below what the harness can resolve | §33.32 |
| Step-time study | above: the step is the per-layer floor plus the misses; prefetch at cap 1 touches ~14 of ~88 misses | §33.33 |
| T3 cap 2 (`9d26b8d53c`) | head publishes up to 2 blocks (PER_LAYER 1..2, cap-1 threshold kept); one instrumented capture each: off 275.8, T3 cap 1 274.9 (-0.3%), T3 cap 2 264.5 ms/verify (-4.1%), covers 21.8 vs 15.0 misses, 15.6% of used reads still in flight at use; no drift (layer 0 within 3%). Suggestive, not a verdict | §33.34 |
| T3 cap 3 (`db9398a8b7`) | PER_LAYER 3, SPEC_SHARE 3: off 278.2, T3 cap 2 repeat 265.7 (-4.5%), T3 cap 3 277.4 (-0.3%); cap 3 covers 28.5 vs 23.1 misses but NVMe reads slow (1.63 vs 1.51 ms/miss), so it saves less. Cap 2 is the best served setting; a per-rank threshold is the lever for more. Runs in `ram-prefetch/t3-cap3-20261013/` | §33.35 |
| Engram verify in graph (`7413b974de`, default on `00e80ccfc0`) | `SGLANG_DSV41_ENGRAM_VERIFY_IN_GRAPH`: verify graph `segments=1 breaks=0`; gaps before rows 1/14 4.59/5.20 -> 1.07/1.34 ms (8 ms per verify). One pair; wall median dominated by drift. Runs in `ram-prefetch/engram-verify-20261013/`. Note every arm now runs with it on (the default) | §33.35 |

Implication: a better cap-1 predictor can only move the step by a fraction of a percent. A full two-order T3 vs native
A/B would not resolve ~0.5% and is not recommended now.

## Suggested next questions (proposals, none approved; owner decides)

In rough order of expected size. Each should go through hx-brainstorming before code.

1. **What is the per-layer floor now?** ~112 ms per verify. Confirm from the existing captures' event logs
   (`exl3-cpu-exp{0,1}`, `exl3-copy-eng`, `exl3-tier{0,1}`) which finisher sets the gate in zero-miss layers (CPU hit
   job vs DMA vs other), as §33.12 did for the older recipe. Offline, no GPU needed.
2. **Fewer misses without a bigger tier.** The 104 GiB tier is already in every arm (`arm_env.py`
   `PINNED_HOST_MB = 106496`) and host memory has no room for more (§33.12). Options: residency policy (node 0 does
   most of the evictions, §33.12), and verifying fewer wrong positions (§33.16 item 4: 32-59% of a verify's experts
   serve rejected positions; needs ragged/shorter verify graphs).
3. **Faster or more even reads.** Each row waits for its slowest mirror part; the SPCC drive (nvme4) is slower with a
   long p99 (§33.16 item 3). Mirror weights by drive speed lost in §33.19; revisit only with a new idea.
4. **A higher prefetch read budget (cap 2) with T3: built and captured once (§33.34), -4.1% per verify.** Next: the
   native L+2 cap-2 arm (`dspark-both-prefetch-gpu-misscut-idle2-l2-cap2`, exists, never run) as the head's control,
   then a repeated, order-balanced served comparison. Even full cover of rows 1-39 is worth ~19% (§33.16).
5. **Make the A/B harness drift-aware.** Report layer-0 wait and `read_ms / ram_misses` per arm, and refuse or flag a
   pair whose drive state differs. Today's off/off2 swing (11%) dwarfs every prefetch effect measured so far.

## Artifacts on divix01

| What | Path |
|---|---|
| Step-time captures (off, L+2, off2; per-engine event logs, `layer_misses.json`, `counters.json`, `metrics.jsonl`, `server.log`) | `/data/models/slang/nvfp4-work/ram-prefetch/step-time-20261013/{off,l2,off2}/` |
| T3 cap-2 captures (off, T3 cap 1, T3 cap 2; same layout as the step-time ones) | `/data/models/slang/nvfp4-work/ram-prefetch/t3-cap2-20261013/{off,t3,t3cap2}/`, `compare-{t3,t3cap2}-vs-off.{txt,json}` |
| Step-time comparisons | `.../step-time-20261013/compare-off-vs-l2.{txt,json}`, `compare-off2-vs-l2.{txt,json}`, `l2-layers.{txt,json}` |
| Step-time drivers and logs | `.../step-time-20261013/run.sh`, `run.log`, `run2.sh`, `run2.log` |
| OOM'd first attempt (another GPU job) | `.../step-time-20261013/failed-oom-1658/` |
| F124 quick served check | `/data/models/slang/nvfp4-work/ram-prefetch/l2-t3-quick-20261013/` (`report/REPORT.md`; `report-empty-tables/` is the buggy first report) |
| Exported T3 head (sha e675e2bd..., threshold 0.1301608383655548) | `/data/models/slang/nvfp4-work/ram-prefetch/l2-t3-head/t3-f109-calib-b075.safetensors` |
| F108 served A/B (native L+2 vs L+1 floors vs off) | `/data/models/slang/nvfp4-work/ram-prefetch/l2-ab-20261012/` |
| F123 fast-path run | `/mnt/nvme1/nvfp4-work/f123-l2-fast-20261013/` |
| F109 cap-1 offline comparison | `/mnt/nvme1/nvfp4-work/f109-l2-cap1-20261013/` |
| B0 corpus / B1 dataset / R1 / R2 | `/mnt/nvme1/nvfp4-work/{b0-corpus-100-20261012,b1-dataset-20261012,b1-r1-20261012,b1-r2-20261012}/` |
| Older Nsight traces (node mode, older recipes) | `/mnt/nvme1/dsv41-nsys/` (e.g. `dspark-both-20261008-000548.*` for §33.12) |

No Nsight trace was taken today: the instrumented job trace answers the per-layer question directly, and node-mode
nsys distorts graph launches (see CLAUDE.md). The captures' times come from the instrumented build: compare them only
with each other, never with an A/B's ms/token.

Worktrees used today: `wt-step-time` (at `af2dcb2d52`, generation "af2dcb2d52 step-time capture"),
`wt-l2-t3-quick`, `wt-l2-t3-report`, `wt-f109-run`, `wt-f123-run`, `wt-l2-ab`, `wt-f124-finish{,-base}`. They can be
removed once nobody needs them; ask the owner before removing others.

## Code entry points

- `analysis/dsv41-drive/dspark/spec_margin_capture.py` — instrumented served capture of any `both_cpu_ab.py` arm
  (`--arm`, `--no-verify-split`, port 30032); takes both locks itself, accepts an outer `flock` on the disk lock.
- `analysis/dsv41-drive/dspark/layer_misses.py` — per-layer records, forward breakdown, zero-miss floor, cost per
  NVMe read, `--compare` matched by (row, misses).
- `analysis/dsv41-drive/dspark/both_cpu_ab.py` — the arm table (`ARMS`): `dspark-both-misscut` (off),
  `dspark-both-prefetch-gpu-misscut-idle2-l2` (native L+2), `...-idle2-l2-t3` (T3).
- `benchmarks/dsv41_baseline/arm_env.py` — base env every arm inherits (tier size, DSpark flags).
- `analysis/dsv41-drive/dspark/{run_l2_t3_quick.sh,l2_t3_quick_report.py,l2_t3_preflight.py}` — F124 quick check,
  its report (pairs sessions by corpus id) and its quiet-box preflight (refuses on GPU compute processes or >1 MB/s
  expert-drive traffic).
- `analysis/dsv41-drive/dspark/drive_busy.py`, `miss_timeline.py`, `layer_cover.py` — drive and cover studies
  (§33.16, §33.24).

## Housekeeping state

- hx feature F124 is "paused · 21/21 done": finished by hand on master (`e3c8d12a9d`, `bd36a12e95`, `b72467ea67`).
  Don't resume it; resuming recreates a worktree and lands in needs_you. Its plan files are saved in
  `~/.hx/specs/F124/worktree-plans/`.
- A shared `git stash@{0}` and `.hermes/worktrees/F123` on the laptop are not this work's; leave them.
- Recent master commits: `163337b11c` + follow-up (§33.33), `af2dcb2d52` (§33.32), `b9a1b2ac99` (quick report fix),
  `b72467ea67`, `bd36a12e95`, `e3c8d12a9d` (F124 merge).

Production on divix01 is stopped; only the owner can restart it.
