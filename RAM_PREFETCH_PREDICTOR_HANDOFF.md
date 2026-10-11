# DSV4.1 RAM prefetch, next round: a better predictor (handoff)

**Snapshot:** 2026-10-11, `master` at `288fc15aa1`. Written at the end of the October 8-11 prefetch round.
The record of every measurement cited here is `DSV41_REFERENCE.md` §33.12-§33.23. Read it before acting; when this
file and the reference disagree, the reference wins.

## The ask

The NVMe-to-RAM prefetch works mechanically, but no variant has shown a clear decode gain. The last round found that
**prediction** is the binding limit:
- 23% of forced misses never appear among the current predictor's candidates;
- more than half the speculative reads that land are never used, and those wasted reads slow the demand reads on the
  same drives.

This round has two goals.

**Track A.** Close the idle2 question with tonight's A/B. Decide whether idle2, with or without floors, becomes what
`SGLANG_DSV41_RAM_PREFETCH=1` turns on.

**Track B.** Find out whether a **learned predictor** beats the current one, which applies the native gate early.
- The predictor is the handoff's E2 (fresh LLaPor), then E5 (gate scores as features, hard negatives).
- Measure it offline first, on held-out DSV4.1 sessions, at today's read budget.
- Build serving only if it wins clearly.

The design reference for Track B is `DSV4.1 expert_prediction_handoff.md` (2026-09-19). It lays out experiments
E0-E10, the feature definitions, the split discipline and the measurement contract. **Its E1, the native gate
applied early, is what serves today; E2-E5 have never been run on DSV4.1.** Its measured numbers predate DSpark and
CPU experts. Use it for method, not for its numbers.

## Standing rules (non-negotiable)

- **Production on divix01 is stopped.**
  - Never start, stop or restart it.
  - Never run anything in its checkout `cc-expert-prediction/dsv41-direct-prod` (at `f1ba1a3d18`).
  - End every report to the owner with: "Production on divix01 is still stopped; only you can restart it."
- **Follow `.claude/rules/divix01-run-protocol.md`.**
  - Code moves laptop → commit → push to `origin` → a private divix01 worktree. No rsync/scp of trees.
  - Set `PYTHONPATH=$PWD/python`, and print `sglang.__file__` once.
  - CPU work runs under `taskset -c 0-63` with `OMP_NUM_THREADS`. Cores 64-71 stay free.
  - GPU work runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`.
  - Lock order is `rowimg-disk.lock`, then `cc-gpu.lock`.
  - Read `${PIPESTATUS[0]}` on every piped run.
  - Run only the tests your change affects, and record the exact command next to each result.
- **Nothing else may run beside a served A/B.**
  - No tests that touch the GPU, the mirror drives or `/tmp` while one is running. A test run during an A/B
    invalidated one on 2026-10-11.
  - Check `ps -eo cmd | grep -E "launch_server|both_cpu_ab|run_arm.sh" | grep -v grep` first.
  - Do not use `pgrep -f` with that pattern inside an `ssh` command: it matches its own command line.
- **The root volume `/` is ~90% full (7 GB free).** A probe that filled `/tmp` killed an A/B arm on 2026-10-11.
  - Put pytest `--basetemp` and `TMPDIR` under `/data/models/slang/nvfp4-work/<task>/`.
  - Put large captures on `/mnt/nvme1` (the Crucial, not a mirror drive) or under `/data`.
  - Never put captures on `/mnt/nvme0`, `/mnt/nvme2` or `/mnt/nvme4`. Those are the expert mirror roots.
- **Git:** never amend or force-push; stage files by name. Commit trailer exactly:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk
  ```
- **Models:** opus or sonnet only for any agent; **never "fable"**.
- **New features:** plan through hx-brainstorming / hx-planning, never the superpowers planning skills or plan mode.
  Track B's offline study is a spike: it answers a question and its code is throwaway. Any serving design that
  follows is architectural and needs a written spec.

## The larger DSV4.1 story (what this sits inside)

**Model.** DeepSeek-V4.1 Flash, EXL3 3.0 bpw.
- 40 MoE target layers, each with 384 routed experts, top-6, plus a shared expert.
- One expert row is 13.3 MB; all experts total ~205 GB.
- The router input is 5,120 wide: the mHC streams are combined and normalized before the gate.
- Engram sits at layers 1 and 14, and the encoder/decoder switch is at 19→20.

**Host.**
- RTX 5090 (32 GB) in a Gen3 slot; 188 GB RAM shared with other workloads; 72 cores on two NUMA nodes.
- The expert rows are mirrored on three NVMe roots, and every row read splits into thirds across them:

| Mount | Device | Drive | Notes |
|---|---|---|---|
| `/mnt/nvme0` | `nvme0n1` | Samsung 990 EVO Plus | mirror root |
| `/mnt/nvme2` | `nvme1n1` | Samsung 990 EVO Plus | mirror root |
| `/mnt/nvme4` | `nvme2n1` | SPCC | mirror root; DRAM-less, stalls on its own (§33.21) |
| `/mnt/nvme1` | `nvme3n1` | Crucial P310 | not a mirror root; use it for captures |

**Serving recipe** (arm `dspark-both-misscut`; the defaults on master).
- DSpark speculative decoding: block 5, accept length about 3.5-3.7.
- The decode graph is breakable, at batch size 1.
- 10,840 MiB of GPU hot cache, and a **104 GiB RAM tier** (§33.12).
- **CPU experts for both the draft and the target.** A forced RAM miss is read from NVMe and then computed on the
  CPU.
- The miss-aware CPU split cut is on.
- `SGLANG_EXL3_CPU_MAX_M=8`.
- About 75-81 ms/token median on the 8-session A/B set.

**The critical path** (§33.12). 61% of a verify step is GPU idle on the **lease gate**: each layer waits for its
forced NVMe misses to land and be computed. There are about 90 forced misses per verify forward, and roughly
3 in 4 layers have at least one. That is why prefetch is worth pursuing at all.

**What is not the lever.**
- The CPU kernel and the CPU split are off the critical path at this load (§33.20, §33.22).
- Drive stalls cost ~1-2% of mean ms/token but distort A/Bs (§33.23).

## The prefetch so far (October 8-11)

**Mechanism.** After each layer's post, a scorer predicts layer T+1's experts from layer T's MoE input, which is
the native gate applied one layer early.
- The scorer runs on the GPU (`SCORER=gpu`, two kernels in the decode graph, the score kernel ~6 µs, on a side
  stream).
- It publishes up to 8 candidates per record, already filtered past VRAM-hot and RAM-mapped rows.
- Each NUMA group's speculative thread reads the first `per_layer` candidates still unmapped into a small pool.
- A forced miss that finds its row in the pool is "used"; one still mid-read is "boosted".
- Specs: `docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md`, `2026-10-09-dsv41-ram-prefetch-gpu-scorer-design.md`
  and `2026-10-09-dsv41-drive-aware-reads-design.md`.

**Scoreboard** (§33.22; positive = the prefetch arm is faster than the same configuration without prefetch):

| # | Variant | Result |
|---|---|---|
| 1 | CPU scorer | −1.9% |
| 2 | GPU scorer | tie (−1.1%) |
| 3 | GPU + margin floors | tie (+0.6%); precision 0.45 → 0.59, waste 6.9 → 2.8 rows/token |
| 4 | GPU + split cut | about even |
| 5 | GPU + per-drive in-flight cap | −4.5%, dropped |
| 6 | GPU + cut + idle-drive reads | about −7% |
| 7 | **GPU + cut + idle2** | First A/B +3.6% paired median; **repeat −3.7%**. Step time per verify, flagged sessions out: −3.0%, then +2.6%. **A tie** |

**Today's defaults.**
- `SGLANG_DSV41_RAM_PREFETCH=1` alone gives row 3. Prefetch is **off by default**.
- idle2 needs `IDLE_DRIVE=1`, `IDLE_SPREAD=1` and `IDLE_PIECES=2`. It was measured with floors off
  (`MARGIN_FLOORS=""`).

**What the last round established.**
- **The read budget is tight** (§33.16).
  - Decode reads 4.3 GB/s against a ~9.3 GB/s three-drive ceiling.
  - A speculative read on a drive that a demand read needs doubles that read's time.
  - So every wasted read has a cost.
- **There is no headroom for a layer-completing rule.** The layer-cover study (`divix01:/data/models/slang/nvfp4-work/layer-cover/REPORT.md`,
  branch `codex/dsv41-layer-cover`):
  - Misses per layer, rows 1-39: 25% have none, 24% one, 18% two, 12% three, 20% four or more.
  - Live coverage of miss layers: 13.6% fully covered, 47.5% partly, 38.8% not at all.
  - **23% of misses are never a candidate** (top-12), so 42% of miss layers can never be cleared.
  - At today's ~39 reads per verify, reading a layer's predicted misses as a set clears 22% of miss layers against
    17% for top-1, at the same value per read (~0.2 ms).
  - The best value per read (0.36-0.47 ms) comes from a margin cut of 0.1-0.25, which is what the floors already do.
  - A covered read saves 0.33-0.52 ms when it is a layer's last miss, and 0.75-2.1 ms when it comes off a deeper
    queue. Partial coverage is worth more than the "clear the whole layer" model assumed.
  - With the GPU scorer, 21% of used reads were still in flight at use. The read window holds about one extra row per
    layer.
- **Drive stalls** (§33.23).
  - 9 of 272 sessions were disturbed: SPCC 4, Samsung 1, unattributed 4.
  - They raised 4 of 34 arm medians by 10-30%.
  - The A/B driver now prints per-session drive load and `drive_flags`. Re-run or drop flagged sessions before
    pooling.

## Open state you inherit

1. **The idle2 repeat is recorded; idle2 with floors is being re-run.**
   - `ab-idle2-floors-20261011-164535` (recorded in §33.22) showed idle2 to be a tie.
   - Its idle2 + floors arm was aborted at warm-up by the stability gate, with no fault.
   - The re-run is `divix01:/data/models/slang/nvfp4-work/ram-prefetch/ab-idle2-floors-rerun-20261011-174531`.
     Arms in order: `dspark-both-prefetch-gpu-misscut-idle2-floors`, then `dspark-both-misscut`.
   - It runs from worktree `wt-idle2-floors` at `288fc15aa1`, registered in the generation gate as
     "master 288fc15aa1".
   - When it ends, read `summary.json`: per-arm `drive_flags` sit inside each arm's entry. Compare step time per
     verify with flagged sessions excluded, as in §33.22. Ms/token alone mixes in accept-length changes from
     diverging text.
   - Record the result in §33.22, update the scoreboard, then remove `wt-idle2-floors`.
   - If the floors arm again fails warm-up, run it with a longer warm-up rather than reading anything into it.
2. **`codex/dsv41-layer-cover`** (`2c2f7044e0`, laptop worktree `/Users/dnikolaidis/.codex/worktrees/layer-cover`) is
   unmerged.
   - It adds `analysis/dsv41-drive/dspark/layer_cover.py`, `prefetch-replay/layer_cover_oracle.py` and
     `test/registered/unit/scripts/test_layer_cover.py`.
   - Its 5 tests were run only on the laptop with stubs. Run them on divix01, which needs a quiet box, then merge.
   - Also record the layer-cover findings in the reference as §33.24; they are not there yet.
3. **The real-miss check of the layer-cover numbers** has not been run. Item 3 used `verify_replay`'s tier simulation,
   which runs 14.5% below the server's miss count.
   - Run `analysis/dsv41-drive/prefetch-replay/verify_gate_rankings.py` on `cut-timeline-20261009-221958/cut`.
   - It reads ~160 MB of gate tensors that live on `/mnt/nvme2`, so the box must be quiet: no A/B.

## Track A: close idle2

idle2 alone is settled as a tie with no prefetch (two A/Bs, §33.22). Decide idle2 + floors from the re-run, with the
drive flags applied.
- **If idle2-floors beats `dspark-both-misscut`** on step time and the paired median, with flagged sessions re-run
  or excluded:
  1. repeat it once, reversing the arm order;
  2. if it holds, make that configuration what `SGLANG_DSV41_RAM_PREFETCH=1` turns on. The precedent is commit
     `0589e7fae4`, which made the GPU scorer and floors the defaults with tests.
- **If neither wins:** keep prefetch off and record it. Track B decides whether prefetch has a future.

The owner decides any change of production defaults. Propose it; do not flip it unasked.

## Track B: does a learned predictor beat the native gate? (spike)

**Question.** On held-out DSV4.1 sessions, at today's read budget (~1 speculative row per layer, ~39 per verify), does
a learned L+1 head beat the native gate applied early? It must win on all three of:
- recall of the rows that were actually missing from RAM at demand;
- precision of the rows it would read;
- recovery of the 23% of misses that are never a gate candidate.

**Why it might.**
- The gate only sees layer T's input projected through layer T+1's router.
- A head trained on DSV4.1 routes can also use layer T's own chosen experts and weights. The handoff calls this
  "route signature".
- With E5 it can also use the gate's margins, and learn which confident candidates are wrong.
- The pre-DSpark probe (handoff "What has already been measured") showed the gate's NVMe-row recall at only 0.48
  (top-6), with precision falling from 0.68 at rank 1 to 0.19 at ranks 4-6.

**Rough ceiling, to keep expectations honest.** The read budget, not prediction, bounds the gain.
- About one row per layer fits in the window.
- So a better predictor mainly raises precision: fewer wasted reads, less contention.
- A guess, not a measurement: roughly 5-8% ms/token at near-perfect precision, against 0-3.6% today.

### B0. The training capture (the handoff's E0), on the actual serving path

- Capture during DSpark verify forwards on the served EXL3 fused path, with the production recipe
  (`dspark-both-misscut`). Per (forward, layer) record:
  - the canonical normalized 5,120-wide gate input;
  - the native top-6 ids and the weights the runtime uses;
  - which rows were VRAM-hot, RAM-resident or in flight at prediction time and at demand;
  - identity (session, request, forward, verify position, layer).
- Check what already exists before writing anything. `spec_margin_capture.py` and the margin captures
  (`margin-gpu-20261009-035316`, `cut-timeline-20261009-221958/cut`) already dump gate tensors and rankings for a
  few sessions. `verify_gate_rankings.py` reads them. Extend that path rather than reviving the old Qwen
  `expert_prediction/capture.py`.
- Run the handoff's integrity check: recompute each layer's own gate from the captured input and require exact
  route-set matches.
- **Corpus.** Enough whole sessions for train / development / calibration / locked test, split by session, from
  `/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl` (2,674 sessions).
  - Size it first: about 40 layers × verify positions × 10 KB per forward.
  - Keep it on `/mnt/nvme1` or `/data`.
- The capture needs the server, both locks and a quiet box. Use the A/B driver's lock discipline.

### B1. Baselines and the learned head (E1/E2, then E5)

- **Baselines,** all on the same held-out sessions and denominators:
  - the native gate at L+1 (today's scorer), with and without the packaged floors
    (`python/sglang/srt/layers/moe/ram_prefetch_margin_floors.json`);
  - the previous token's routes;
  - training-only popularity.
- **The learned head.** Adapt `python/sglang/srt/layers/moe/expert_prediction/training/llapor.py`, `pca.py`,
  `dataset.py` and `scripts/expert_prediction/training/train_llapor.py`:
  - fresh PCA on 5,120 inputs plus the 384-wide source mask and weights;
  - 384 outputs per (source, target) pair;
  - explicit source/target/horizon identity;
  - PCA and statistics fit on the training split only.
  - The Qwen constants do not apply (`OUTER_SOURCE_LAYERS`, budgets 10/12/16/24/32, recall@16 selection; see the
    handoff's reuse map).
- **E5:** add the gate's scores and margins as features, and hard negatives from training sessions.
- **Metrics** (the handoff's contract, narrowed):
  - recall of NVMe-eligible rows and precision of the rows read, at fixed reads per layer (1, 2, 3) and at
    margin/probability cuts;
  - per-layer calibration;
  - the share of never-candidate misses recovered;
  - the expected saving per read, priced with the layer-cover slopes (0.33-0.52 ms for a last miss, more on deep
    queues);
  - scorer cost: the bytes the head adds in GPU memory, which displace hot slots, and its kernel time, against the
    0.1 ms per layer criterion of §33.15.
- Train under the analysis memory cap (8 GiB, `MemorySwapMax=0`), on CPU cores 0-63, or on the GPU under
  `cc-gpu.lock`.

### B2. Decision

- **Advance to a serving design** only if the head, on the locked test sessions, clearly beats the gate with floors
  at equal reads.
  - Suggested bar: precision +0.15 at rank 1, or the same precision with ≥25% more NVMe-row recall, without losing
    timeliness.
  - Its scorer must fit the decode graph's budget.
- If it advances, write the serving spec through hx-brainstorming. The GPU scorer's slot, pinned candidate page and
  speculative threads already exist, so the head replaces the score kernel and its weights.
- **If it does not:** record that the native gate is the ceiling for same-token L+1 prediction at this budget. The
  remaining levers are then lead time (L+2, E3) or the drives.

## Smaller items, if time allows

- **`spec_moved_root` = 84,652 in the idle2 arm**, against 2,569 for idle.
  - It may be pieces moving between drives while waiting for an idle one.
  - Read the idle-drive reader (`reader_core.h` / the speculative ring) and its counters to see whether this costs
    timeliness.
- **L+2 lead time (E3).** Only if B shows predictions are right but late. First measure precision two layers ahead
  offline. The layer-0 draft prediction already arrived too late (§33.19).
- **`summary.json` ms/token stops at the last content chunk** (§33.23 side finding). Leave it as is for
  comparability.

## Where to start (branches and worktrees)

| What | Where |
|---|---|
| Base for all new work | `origin/master` at `288fc15aa1`. The old branch `codex/dsv41-ram-prefetch-margin` is identical to master. |
| Laptop worktree on that branch | `/Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin` (has an untracked `pyrightconfig.json`; leave it) |
| Unmerged analysis branch | `codex/dsv41-layer-cover` (`2c2f7044e0`), laptop worktree `/Users/dnikolaidis/.codex/worktrees/layer-cover` |
| Running A/B worktree (divix01) | `/data/models/slang/nvfp4-work/wt-idle2-floors` (`288fc15aa1`); remove it after the A/B is recorded |
| divix01 clone | `/data/models/slang/sglang`; make private worktrees under `/data/models/slang/nvfp4-work/wt-<name>` |
| Suggested new branch for Track B | `codex/dsv41-learned-predictor` from `origin/master`, laptop worktree `/Users/dnikolaidis/.codex/worktrees/learned-predictor` |
| A/B driver | `analysis/dsv41-drive/dspark/both_cpu_ab.py` (arms, `REFERENCE`, `drive_flags`), via `benchmarks/dsv41_baseline/run_arm.sh` |
| Generation gate | `benchmarks/dsv41_baseline/generations.py`: `register(<git rev-parse HEAD:python>, label)` in the worktree before an A/B |
| A/B launch shape | `setsid nohup env PYTHONPATH=$W/python OMP_NUM_THREADS=8 flock .../rowimg-disk.lock taskset -c 0-63 python analysis/dsv41-drive/dspark/both_cpu_ab.py $OUT <arm> <arm> ...` |
| Prefetch analysis tools | `analysis/dsv41-drive/dspark/{spec_margin,spec_margin_capture,layer_misses,miss_timeline,verify_split,drive_sessions}.py`, `analysis/dsv41-drive/prefetch-replay/{verify_replay,verify_gate_rankings}.py`, `scripts/dsv41/tier_sim.py` |
| Predictor training code (Qwen-era) | `python/sglang/srt/layers/moe/expert_prediction/` (`training/llapor.py`, `pca.py`, `dataset.py`, `capture*.py`, `serving/`), `scripts/expert_prediction/training/train_llapor.py` |
| Prefetch runtime | `python/sglang/srt/layers/moe/ram_prefetch.py`, the host code under `python/sglang/kernels/jit/csrc/moe/expert_stream/host/` (`ram_tier.h`, `reader_core.h`), env vars in `python/sglang/srt/environ.py` (`SGLANG_DSV41_RAM_PREFETCH_*`) |
| Evidence | `divix01:/data/models/slang/nvfp4-work/{ram-prefetch,layer-cover,stall-cost,critical-path-trace}/` |

Older A/B worktrees on divix01 (`wt-*`, ~140 of them) are earlier work. Leave them unless the owner asks.

## Pitfalls already paid for

- A same-window A/B pair is the minimum. Sessions swing 5-15% between windows, and an imported baseline reads drift
  as an effect.
- One disturbed session flips a pooled result. Use the paired median, and check `drive_flags` before pooling.
- Instrumented-build timings compare only with each other, never with an A/B's tok/s.
- A capture or test that reads the mirror drives during an A/B invalidates it.
- Changing host C++ gives every JIT module a new build key. The first launch compiles; a child test must warm the
  module in its parent (run protocol, "Cold JIT builds").
- The launch gate silently validates NVFP4 rules for a non-directory model path. Assert the resolved requirements are
  EXL3's (run protocol).

## Deliverables

- **Track A:** the A/B result and decision in §33.22, and a proposal to the owner for the prefetch default.
- **Track B:**
  - a capture manifest and integrity report;
  - baseline and learned-head tables on held-out sessions;
  - a plain verdict;
  - a new reference section (§33.25 or next) with artifact paths;
  - if it advances, a written spec for serving the head.
- **Housekeeping:**
  - merge `codex/dsv41-layer-cover` after its tests pass on divix01;
  - add §33.24 (layer cover);
  - remove the worktrees you created.
