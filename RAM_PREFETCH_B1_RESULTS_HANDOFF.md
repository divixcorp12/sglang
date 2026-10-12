# DSV4.1 RAM prefetch: learned-predictor results and the open call (handoff)

**Snapshot:** 2026-10-12, `master` at `d70965052a`. Written at the end of Track B (B0 capture, B1 training and
evaluation, B2 verdict) of `RAM_PREFETCH_PREDICTOR_HANDOFF.md`. The record of every number here is
`DSV41_REFERENCE.md` §33.25 (capture) and §33.26 (results). When this file and the reference disagree, the reference
wins.

> **Update 2026-10-12 (later the same day), `master` after `7175bd5405`.** The design critique below (round 1) was
> accepted with five amendments and run as hx feature F106 ("R1"), followed by an L+2 spike. Their results are in
> **"ROUND 2 RESULTS: R1 and the L+2 spike"** at the end of this file and in `DSV41_REFERENCE.md` §33.27. Short
> version: the learned head is a real, seed-stable gain at equal reads (+15% recall at L+1, +27% at L+2), the R1
> continuation gate fails on scorer cost alone, and the timing replay says **lead time (L+2) is worth more than the
> learned predictor**: the bare native gate moved to L+2 beats today's L+1 gate on on-time useful rows by +18% at
> nominal contention and 2x at adverse. The questions in "What we want from you" below were round 1's; round 2's
> open questions are at the end of the new section.

## What we want from you

A second opinion, not more code. Read this file, then §33.26, and tell the owner:

1. **Is the verdict right?** As written, the bar says "do not advance". One arm of the bar (recall) is met
   decisively, and the verdict fails only on a bootstrap that tests the other arm. Which reading should stand?
2. **Is the thresholded policy real?** The head's advantage shows up only when it chooses *how many* rows to read per
   layer (a probability cut), not when it is held to exactly one read per layer. Does that survive the constraints of
   the serving path (read window, drive contention, timeliness)? What would you measure before believing it?
3. **What next?** Pick among the options at the end, or propose another. Say what you would not do.

Challenge anything below. The previous session wrote the tools, ran them and wrote this file, so it has every
incentive to believe the numbers.

## Standing rules (unchanged; non-negotiable)

`RAM_PREFETCH_PREDICTOR_HANDOFF.md` "Standing rules" applies in full. The short version:
- **Production on divix01 is stopped.** Never start, stop or restart it; never run anything in
  `cc-expert-prediction/dsv41-direct-prod`. End every report with "Production on divix01 is still stopped; only you
  can restart it."
- `.claude/rules/divix01-run-protocol.md`: code moves by push to `origin`, then a private divix01 worktree;
  `PYTHONPATH=$PWD/python`; CPU under `taskset -c 0-63` with `OMP_NUM_THREADS`; GPU under
  `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`; lock order disk then GPU; read `PIPESTATUS`.
- Captures and run outputs only on `/mnt/nvme1`. Never `/mnt/nvme0`, `/mnt/nvme2`, `/mnt/nvme4` (expert mirrors).
- Never amend or force-push; stage by name. Agents: opus or sonnet only. Planning through hx, not plan mode.
- The owner decides production defaults and how to read the bar. Propose; do not flip.
- The laptop cannot run sglang tests; tests go through `~/.hx/checks/divix01-pytest [--gpu] <files> -q -p no:randomly`.
- `pkill -f <pattern>` inside `ssh` kills the ssh's own shell. Kill by PID.

## Background in one screen

- **Model and host.** DeepSeek-V4.1 Flash, EXL3 3.0 bpw: 40 MoE layers, 384 routed experts, top-6; an expert row
  is 13.3 MB. RTX 5090 (32 GB) with a 10.8 GB GPU hot cache, a 104 GiB RAM tier, the rest read from three
  mirrored NVMe drives. DSpark speculative decoding at batch 1, CPU experts for draft and target, ~75-81 ms/token.
- **Why prefetch matters.** 61% of a verify step is GPU idle on the lease gate, waiting for forced NVMe misses
  (~90 per verify forward; ~3 in 4 layers have at least one) (§33.12).
- **What serves today (when prefetch is on; it is off by default).** After layer T's post, the native gate of layer
  T+1 is applied early to layer T's MoE input on the GPU (score kernel ~6 µs, side stream). Up to 8 candidates per
  record go to speculative reader threads, which read the first `per_layer` still-unmapped ones into a small pool.
  "Floors" are packaged per-layer margin cuts (`ram_prefetch_margin_floors.json`) that suppress low-confidence reads.
- **Every served variant tied** (§33.22). GPU scorer + floors: precision 0.45 → 0.59 and waste 6.9 → 2.8
  rows/token, yet ms/token +0.6% (a tie). idle2 variants: +3.6% then −3.7%. Prefetch stays off.
- **Why it ties** (§33.16, §33.24). The read budget is tight: decode already reads 4.3 GB/s of a ~9.3 GB/s ceiling;
  a speculative read on a drive a demand read needs doubles that demand read. About one extra row per layer fits in
  the read window (~39 speculative reads per verify). 23% of misses are never in the gate's top-12. 21% of used
  reads were still in flight at use.
- **The previous handoff's ceiling guess:** 5-8% ms/token at near-perfect precision, against 0-3.6% today. A guess.

## What was done this round

### B0: the training corpus (§33.25)

- Captured on the served path with the production recipe (`dspark-both-misscut`), **prefetch off**, so labels are
  the miss pattern without speculative reads. Per (verify forward, layer, token): the 5,120-wide gate input, native
  top-6 ids and weights, a RAM-residency snapshot of layers T+1..T+4 at the prediction point, and each routed lane's
  true kind at demand (VRAM hot / RAM hit / NVMe miss to GPU / NVMe miss to CPU), with GPU timestamps.
- `divix01:/mnt/nvme1/nvfp4-work/b0-corpus-100-20261012`: 100 sessions (owner cut the target from 300), 16 GB,
  4 server starts, ~70 min. 19 sessions drive-flagged, 4 warm-start.
- Split by whole session, seeded: **train 60 / dev 15 / calib 10 / locked_test 15**.
- Integrity: the gate recomputed from the captured input matches the captured routes in all but 1 of 1,540,800 cells
  (plus 47 bf16 rank-6/7 ties); eviction between snapshot and demand 0.56%; no seq gaps or stamp inversions.

### B1: baselines and the learned head (§33.26, feature F105)

Tools in `analysis/dsv41-drive/prefetch-replay/b1_*.py`, tests beside them.

- **Eligible row** (the thing to predict): a target route that was an NVMe miss at demand (`MISS_GPU` or `MISS_CPU`),
  once per (forward, target layer, expert). Rows that are VRAM-hot, or that the snapshot shows in RAM, are skipped by
  the one shared scoring function, so every scorer has the same denominators on a split (a test asserts it).
- **Scorers.** `gate` (native gate applied early, union over live tokens by best margin to the 6th score), `gate +
  floors`, `prev` (previous token's routes), `pop` (training popularity), and two heads:
  - **E2:** PCA of the 5,120-wide input plus the source layer's 384-wide route mask and weights.
  - **E5:** E2 plus the target gate's scores and margins scattered to 384 wide, with hard negatives (gate top-12
    experts that were not routed) up-weighted in the loss.
  - Both: one MLP shared across layers (`llapor.MiddlePredictor`), with learned source-layer, target-layer and stratum
    embeddings; 384 sigmoid outputs per token, aggregated per forward-layer by noisy-or. One head per horizon.
  - Grid on dev: PCA rank {128, 256} × hidden {256, 512} × (E5) hard-negative weight {1, 2, 4}; AdamW, cosine, 20
    epochs, patience 3. Selection by dev precision at 1 read. **E5 won at every horizon.**
- **Budgets.** Exactly 1, 2 or 3 reads per forward-layer, and probability or margin **cuts** chosen on calib (read
  every candidate above the cut, so the number of reads varies by layer).
- **Discipline.** PCA, popularity and normalisation fit on train only; cuts on calib only; selection frozen and
  hashed (`0018b38b…98df5`) before the locked test was read, once.

### B2: the bar (spec §1.2, from the previous handoff)

Advance to a serving design if, on the locked test at L+1:
- **(precision arm)** precision at 1 read per layer rises by ≥ 0.15 over gate + floors, **or**
  **(recall arm)** at gate + floors' precision, eligible-row recall is ≥ 1.25×;
- **and** the session-bootstrap 95% lower bound of head minus gate + floors is above zero;
- **and** the scorer costs ≤ 0.1 ms per layer.

The spec does not say which arm the bootstrap tests. The implementation bootstraps the precision arm.

## Results (locked test, 15 sessions, L+1)

30,888 forward-layers, 53,619 eligible rows. Pooled; dropping the drive-flagged sessions moves nothing by more than
0.006.

| Scorer | R @1 | P @1 | R @2 | P @2 | R @3 | P @3 | Never-cand. recovered @3 | At calib cut (R / P) |
|---|---|---|---|---|---|---|---|---|
| gate | 0.215 | 0.375 | 0.347 | 0.307 | 0.436 | 0.262 | 0.000 | 0.284 / 0.570 (cut 0.1) |
| gate + floors | 0.188 | 0.520 | 0.284 | 0.488 | 0.336 | 0.468 | 0.000 | 0.317 / 0.532 (cut 0.05) |
| prev | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 | 0.002 | 0.001 | - |
| pop | 0.033 | 0.058 | 0.064 | 0.055 | 0.093 | 0.054 | 0.055 | - |
| E2 | 0.174 | 0.303 | 0.284 | 0.247 | 0.363 | 0.210 | 0.191 | 0.339 / 0.534 (cut 0.27) |
| E5 | 0.232 | 0.403 | 0.381 | 0.331 | 0.486 | 0.281 | 0.126 | 0.507 / 0.534 (cut 0.18); 0.371 / 0.674 (cut 0.36) |

(`prev` is near zero by construction: the previous token's experts were just loaded, so they are almost never misses.)

**Reads actually spent.** The arms compare policies that read different numbers of rows, so this is the table to
judge by:

| Policy | Rows read | Reads per forward-layer | Useful | Precision | Recall |
|---|---|---|---|---|---|
| gate, 1 read | 30,681 | 0.99 | 11,516 | 0.375 | 0.215 |
| gate + floors, 1 read | 19,355 | 0.63 | 10,059 | 0.520 | 0.188 |
| gate + floors, cut 0.05 | 31,981 | 1.04 | 17,016 | 0.532 | 0.317 |
| E5, 1 read | 30,888 | 1.00 | 12,438 | 0.403 | 0.232 |
| **E5, cut 0.36** | **29,517** | **0.96** | **19,903** | **0.674** | **0.371** |
| E5, cut 0.18 | 50,903 | 1.65 | 27,174 | 0.534 | 0.507 |

**Verdict arms as implemented:**

| Arm | Value | Met |
|---|---|---|
| precision @1 | 0.403 vs 0.520 (needs +0.15) | no |
| recall at the floored gate's precision | E5 cut 0.18: 0.507 vs 0.188 = 2.70× (needs 1.25×) | yes |
| bootstrap (precision arm) | median −0.105, 95% CI [−0.122, −0.093] | no |
| cost | 0.059 ms median, 0.060 p90, per layer (6 tokens, CUDA graph, RTX 5090); 4.65 MB bf16 | yes |

**→ DO NOT ADVANCE, as written.** Second capture not warranted (not near the precision bar), though the learning
curve is still rising.

**Post-hoc recall-arm bootstrap** (computed after the locked read on the same per-session tables; 10,000 resamples,
seed 0; both cuts were frozen on calib before the locked read):

| Recall ratio, E5 cut 0.18 against | Point | 95% CI | Sessions E5 higher |
|---|---|---|---|
| gate + floors, 1 read (the arm as written) | 2.70 | [2.47, 2.90] | 15 / 15 |
| gate + floors, cut 0.05 (matched precision 0.532) | 1.60 | [1.47, 1.72] | 15 / 15 |

### L+2 and L+4 (no bar; evidence for the lead-time lever)

| Horizon | Scorer | R @1 | P @1 | R @3 | P @3 | Never-cand. @3 | At cut (R / P) |
|---|---|---|---|---|---|---|---|
| L+2 | gate | 0.190 | 0.315 | 0.377 | 0.213 | 0.000 | 0.265 / 0.476 (cut 0.1) |
| L+2 | E2 | 0.176 | 0.290 | 0.365 | 0.201 | 0.211 | 0.316 / 0.560 (cut 0.3) |
| L+2 | E5 | 0.225 | 0.373 | 0.456 | 0.251 | 0.163 | 0.370 / 0.645 (cut 0.32) |
| L+4 | gate | 0.158 | 0.257 | 0.313 | 0.172 | 0.000 | 0.253 / 0.378 (cut 0.1) |
| L+4 | E2 | 0.174 | 0.284 | 0.361 | 0.196 | 0.224 | 0.314 / 0.557 (cut 0.3) |
| L+4 | E5 | 0.207 | 0.337 | 0.423 | 0.229 | 0.172 | 0.353 / 0.615 (cut 0.38) |

Precision at 1 read by distance: gate 0.375 → 0.315 → 0.257; E5 0.403 → 0.373 → 0.337; E2 0.303 → 0.290 → 0.284
(flat; beats the gate at L+4). Gate + floors was scored at L+1 only.

### Learning curve (dev, L+1 precision at 1 read; nested seeded subsets, each with its own PCA)

| Variant | 15 sessions | 30 | 60 | Still improving (rule D5) |
|---|---|---|---|---|
| E2 | 0.138 | 0.202 | 0.264 | yes |
| E5 | 0.349 | 0.373 | 0.393 | yes |

Most E5 configs reached their best dev epoch at 18-19 of the 20-epoch cap: not fully converged.

## How the previous session reads it (an opinion; argue with it)

1. **At a fixed one read per layer, the gate is close to the ceiling.** E5's top-1 pick is only slightly better than
   the bare gate (0.403 vs 0.375), and worse than the floored gate (0.520), which wins by *declining* to read on
   low-confidence layers.
2. **The head is a much better confidence signal than the gate's margin.** At the same ~1 read per forward-layer on
   average, E5 thresholded at 0.36 reads 29.5k rows with 0.674 precision and finds 19.9k useful rows. The floored
   gate's best threshold reads 32.0k rows with 0.532 precision and finds 17.0k. That is ~17% more useful rows and
   about a third fewer wasted reads (9.6k vs 15.0k). Against the configuration measured in serving (floors, 1 read),
   it is ~2× the useful rows for 1.5× the reads.
3. **So the bar's two arms disagree because they measure different policies.** The precision arm fixes the read
   pattern; the recall arm lets the head choose where to read. The previous handoff's own diagnosis was that wasted
   reads are what hurt, and the thresholded head cuts them. That reading favours the recall arm, but it is the
   owner's call, and the bootstrap for it was computed after the locked read.
4. **The priced savings overstate.** The replay prices a useful row at 0.33-0.52 ms (the "last miss" slope). The
   floored gate's 1-read policy prices at 3.3-5.2 s over the locked test, and the same policy tied in serving. Treat
   replay savings as an ordering, not a forecast.

## Threats to validity (things you should check, not things we know are wrong)

- **Burstiness.** A cut reads 0.96 rows per forward-layer *on average*, but some layers get several and many get
  none. The read window holds about one extra row per layer, and 21% of used reads were already late with one. **The
  distribution of reads per forward-layer at cut 0.36 has not been computed.** If reads bunch onto layers that already
  have deep demand queues, the drive-contention penalty (a speculative read doubles a demand read on the same drive)
  could eat the gain.
- **Feedback.** Labels come from a run with prefetch off. Serving the head changes residency, and with it which rows
  are misses at the next layers. The eviction rate between snapshot and demand is small (0.56%), but a prefetching
  policy changes RAM contents by design.
- **Timeliness.** The head needs the target gate's scores (E5's features), so in serving it runs after the existing
  ~6 µs score kernel, adding ~59 µs per layer on the side stream. Nobody has checked whether that delays the
  candidate page enough to lose reads that land today.
- **Small locked test.** 15 sessions, all from one 100-session capture of one workload (`sessions.jsonl`). The 15/15
  per-session wins are strong for this corpus, not for a new one.
- **Post-hoc statistic.** The recall-arm bootstrap was chosen after seeing the locked tables. The cuts and the frozen
  selection were not, but the decision to report this comparison was.
- **Unconverged head.** The 20-epoch cap was hit; a longer schedule would likely add a little at 1 read. It will not
  close 0.12 of precision.
- **Bookkeeping gap.** `dropped_cells` is 0 in every split, although the one bad capture cell (b000 seq 1283,
  layer 39) was meant to be dropped and counted. Not chased; one cell cannot move these tables.

## Options for what comes next

None of these has been approved or started.

- **(a) Accept "do not advance".** Record that the gate is the ceiling at one read per layer, and move on to lead time
  (L+2, where E5 holds up better than the gate) or the drives.
- **(b) Cheap offline check before any decision** (a spike, CPU only, no locked-test re-selection): on the existing
  locked and calib tables, compute the per-forward-layer read distribution at E5 cut 0.36 and at the floored gate's
  cut 0.05, and replay both through the existing read-window model (`verify_replay` / layer-cover slopes), capping
  reads per layer at the window. If E5's advantage survives a per-layer cap of 1-2 reads, the thresholded policy is
  real for serving purposes.
- **(c) Re-state the bar and advance.** If the owner reads the recall arm as the intended one, write a serving spec
  (hx-brainstorming) in which the head replaces the floors as the read/skip decision: the gate kernel still ranks, and
  the head's probability decides whether to spend the read. The pinned candidate page and speculative threads exist
  already. Then a same-window A/B against `dspark-both-misscut` and against the floored gate, with step time per
  verify and flagged sessions excluded.
- **(d) A cheaper hybrid first.** Use E5 only as a per-layer gate on top of today's floored scorer (read the floored
  gate's top-1 only where E5's probability clears the cut). It keeps the existing ranking and changes one decision.
  It has not been evaluated offline; it needs a small scorer in `b1_baselines` and the calib/dev tables only.
- **(e) More data.** A second, disjoint capture is not warranted by rule D5, but the curve is rising. Only worth it
  if (b) or (c) say the policy matters.

## Where everything is

| What | Where |
|---|---|
| Reference sections | `DSV41_REFERENCE.md` §33.25 (B0), §33.26 (B1/B2, with rerun block) |
| Previous handoff | `RAM_PREFETCH_PREDICTOR_HANDOFF.md` |
| Tools | `analysis/dsv41-drive/prefetch-replay/b0_*.py`, `b1_*.py` (+ `test_b1_*.py`); driver `b1_run.py` |
| Corpus | `divix01:/mnt/nvme1/nvfp4-work/b0-corpus-100-20261012/` (`manifest.json`, `integrity.json`) |
| Dataset | `divix01:/mnt/nvme1/nvfp4-work/b1-dataset-20261012/` (`dataset.json`) |
| Run | `divix01:/mnt/nvme1/nvfp4-work/b1-run-20261012/` — `selection.json` (frozen), `stages/<stage>/`, `stages/report/REPORT.md` (122 KB: per-layer, per-stratum, per-session tables), `stages/report/results.json`, `stages/locked/h{1,2,4}.json` (per-scorer, per-budget cells incl. `session:<id>`) |
| Trained heads | `.../b1-run-20261012/stages/train/heads/h{1,2,4}_<variant>_r<rank>_h<hidden>_w<weight>.pt`; chosen L+1 head `h1_E5_r128_h512_w2.0.pt`, sha256 `2a282600…f22ef67f` |
| Run script and log | `divix01:/mnt/nvme1/nvfp4-work/b1-run-20261012.sh`, `.log` |
| Superseded first run | `.../b1-run-20261012/superseded/` (stopped for speed) |
| Run worktree | `divix01:/data/models/slang/nvfp4-work/wt-b1-run` at `401e7f26e8` |
| hx feature | F105 (story ST8); merged as `326dec616e`, perf fixes `0b2d8e45dd`, `401e7f26e8`; results `d70965052a` |

**Loading the per-session cells** (how the post-hoc bootstrap was computed): `stages/locked/h1.json` is
`{scorer: {"cells": {budget: {key: {forwards, eligible, read, useful, wasted_hot, never_candidates,
never_recovered, saved_ms_lo, saved_ms_hi}}}}}`, with keys `pooled`, `pooled_noflag`, `layer:<id>`,
`stratum:<name>`, `session:<id>`. Recall is `useful / eligible`, precision `useful / read`.

**Run time,** for planning: dataset extraction is reusable; fit + baselines 41 s; train + curve 27 min on the GPU
(device-resident caches, `--compile`); cuts/freeze/locked 6 min; cost 15 s; report 8 s.

## Housekeeping still open

- Pilot corpora `b0-corpus-20261012`, `b0-corpus-v2-20261012`; smoke and failed run dirs
  (`b1-run-20261012-smoke*`); worktrees `wt-b0-capture`, `wt-b1-run`, `wt-hx-*`, `wt-master`; remote hx branches.
  Kept until the owner says otherwise.
- F102 is not merged to master (owner's call).

Production on divix01 is still stopped; only the owner can restart it.

## DESIGN CRITIQUE - round 1

**Date and evidence:** 2026-10-12; source inspection at local `master`
`d70965052a8c5b493d839c37e453a784449861d4`, the existing B1 report and locked-result tables on divix01,
and a read-only calculation on the existing **development** dataset. No predictor was retrained, no serving
experiment was launched, and no production setting was changed for this critique.

**Purpose:** this is the self-contained experimental design for the next offline training iteration, referred to
below as **R1**. The numerical settings in the plan are proposed starting settings, not measured winners. The
owner requested this addition to the handoff; writing it does not mean that implementation, training, a new
capture, or a served A/B has already run. The standing rules above remain in force. A subsequent session can use
this section to implement and run the offline study when directed; any serving integration still needs its own
written design and the repository's hx process. Do not overwrite the historical B1 selection, reports, or verdict.

> 🧠 **From Hindsight memory (Track B1+B2: learned L+1..L+4 expert predictor vs native gate)** — Track B was
> designed to improve future NVMe-miss-row prediction under a tight read budget. This critique checks that design
> against current code and saved artifacts. The remembered unfinished-results status was superseded by §33.26.
> The B0 memory's stratum-crossing exclusion was also stale: current `residency_capture_ref.horizon_targets`
> invalidates a horizon only when there is no target row. Both memory discrepancies were corrected during review.

### R1.1 Recommendation and interpretation of B1

**Keep prefetch off, but continue a bounded offline predictor study.** The recorded B1 check failed as implemented;
it did not establish an accuracy ceiling for the native gate. One incompletely converged architecture, a small
hyperparameter grid, and a mismatched decision statistic cannot establish that ceiling, even for a one-read cap.

The strongest result is selective reading:

| Frozen L+1 policy | Reads | Useful reads | Wasted reads | Precision | Eligible-row recall |
|---|---:|---:|---:|---:|---:|
| Gate + floors, cut 0.05 | 31,981 | 17,016 | 14,965 | 0.532 | 0.317 |
| E5, cut 0.36 | 29,517 | 19,903 | 9,614 | 0.674 | 0.371 |

E5 found **17.0% more useful rows with 7.7% fewer reads and 35.8% fewer wasted reads**. These are saved locked-test
results, not a new selection or a serving-speed claim. The larger 1.60x matched-precision recall result at E5 cut
0.18 also spends substantially more reads; it is not an equal-bandwidth comparison.

The existing bootstrap is specifically a comparison of per-session **precision at reads=1**
(`b1_select.paired_diffs`). It cannot resolve whether a different, thresholded read policy improves the useful
reads obtained within a feasible transfer budget. Retain the historical failed verdict; register the next
policy, objective and bootstrap together before training. The post-hoc recall bootstrap is exploratory evidence,
not a replacement for independent confirmation.

The thresholded policy is promising but unproven in serving. A per-layer cap surviving offline replay establishes
plausibility, not that the serving policy is already real: reader contention, GPU overlap, deadlines, private-pool
capacity and residency feedback remain material.

The alternatives in the original handoff therefore become:

- **(b), constrained offline analysis, first**, alongside the training work defined below.
- **(d), gate ranking with learned admission, as a control**, not an assumed equivalent of the complete head.
  Keeping gate ranking can lose the learned head's alternative ordering and outside-top-12 recovery.
- **(c), serving design, conditional on the offline gate and fresh confirmation.**
- **(e), more data, conditional rather than the first action.** Reuse the corpus for architecture screening;
  a small independent confirmation corpus serves a different purpose from expanding training data.
- Do not adopt **(a)'s ceiling claim**, launch a broad architecture search, or restart production.

### R1.2 Normal decoding, target verification and draft generation are different populations

| Population | Experts and execution | What the predictor should estimate |
|---|---|---|
| Ordinary one-token target decode | Backbone MoE: 384 experts/layer, top-6 per token | Missing rows for one token at a future layer |
| Target verification of DSpark candidates | The **same backbone experts**, evaluated for an anchor and up to five draft positions | Unique missing rows in the union across the live verification block |
| DSpark draft generation itself | Separate draft stages with their own 128-expert/top-3 MoEs | A different routing/residency problem requiring its own labels and output space |

Backbone layers 37-39 supply DSpark context but remain **target-model layers**. They are not the draft generator's
MoEs. A draft expert's numeric ID is not a target expert ID; direct reuse would require a learned cross-model
mapping, not a presumed correspondence. R1 does not train a predictor for the draft generator's own experts.

The verify input construction explicitly puts the anchor at position 0 and draft candidates at positions 1-5
(`dspark_verify_window.compact_verify_ids`; the static verify epilogue likewise treats `candidates[:, 1:]` as
drafts). B0's join keeps request-associated target verification forwards only (`b0_integrity._join`). Consequently:

- The existing corpus can compare **anchor versus draft positions inside verification**.
- It cannot establish a standalone ordinary-decode result. The anchor is a useful proxy for one-token routing,
  but its cache state and surrounding execution come from a speculative-decoding run.
- Every live position's expert demand counts, including positions eventually rejected. Their target computation
  already happens before acceptance is known. Acceptance logs may stratify a diagnostic report, but are neither
  inference features nor grounds for deleting demand labels.
- A known planner-selected live mask may exclude positions that will not execute. Padding must never contribute.
  B0's current development sample contains only width-six blocks; narrower-width generalization is unmeasured.

### R1.3 New development-only token-position measurements

Read-only calculation over
`divix01:/mnt/nvme1/nvfp4-work/b1-dataset-20261012/dev/h1/*.npz`: 15 development sessions,
**45,708 forward-layer groups**, all with six live positions. The calculation uses the stored early target-gate
top-12 IDs, actual target-route IDs, token indices and demand kinds. It does not run the learned head.

| Position | Token rows | Actual route occurrences covered by that token's early top-12 | NVMe-miss route occurrences | NVMe-miss occurrences covered by that token's early top-12 |
|---|---:|---:|---:|---:|
| Anchor, 0 | 45,708 | 0.84506 | 13,265 | 0.70735 |
| Draft 1 | 45,708 | 0.83967 | 13,175 | 0.69397 |
| Draft 2 | 45,708 | 0.83867 | 13,713 | 0.69613 |
| Draft 3 | 45,708 | 0.84028 | 14,986 | 0.71353 |
| Draft 4 | 45,708 | 0.84067 | 16,300 | 0.72104 |
| Draft 5 | 45,708 | 0.83920 | 17,572 | 0.73111 |

**These are occurrence-weighted candidate-coverage fractions, not precision@1, E5 accuracy, or block-level
prefetch recall.** A different token can also nominate a missing expert in the block union, so these fractions
must not be compared directly with B1's union-level never-candidate metric.

For each group let A be the anchor's unique missing experts and D the union of the draft positions' missing
experts. Summing per-group cardinalities gives:

| Quantity | Unique rows, summed across groups |
|---|---:|
| All missing rows, `|A union D|` | 76,073 |
| Anchor-required rows, `|A|` | 13,265 |
| Draft-only rows, `|D minus A|` | 62,808 |
| Shared anchor/draft rows, `|A intersect D|` | 2,523 |

The 76,073 total was asserted equal to `dataset.json`'s development L+1 `eligible_rows`. NVMe misses are
`LaneKind.MISS_GPU = 4` and `MISS_CPU = 5`; `HIT_SM = 2` is **not** a miss. Import the enum or assert these values
when reproducing this calculation.

Reproduction method: compare each `target_ids[token, lane]` with that token's `gate_ids[token, :]`; accumulate
over valid target IDs for route coverage and only kinds 4/5 for miss coverage. Use `n` to recover group boundaries
and `tokens` to distinguish anchor/drafts; deduplicate expert IDs inside each group before accumulating A and D.
Do not sum token-level counts as a substitute for unique-row counts.

**Interpretation:** the drafts are not obviously less predictable by this coverage measure. They create much more
distinct I/O demand: **82.563% of unique missing rows are draft-only**, and position 5 has approximately 32.5% more
miss-route occurrences than the anchor. Improving only anchor prediction would leave most demand uncovered.
The corresponding learned-head position comparison still needs to be measured; the current saved report has
layer/stratum/session summaries, not position summaries.

### R1.4 Layer variation, burstiness and confidence

From the existing locked `stages/locked/h1.json`, fixed-one-read precision for the bare gate versus E5 is:

- Leaving Engram: **0.365 -> 0.525**.
- Interior layers: **0.367 -> 0.386**.
- Layers 37-39: **0.426 -> 0.439**.

This motivates layer-specific capacity and reporting, not a claim that one specific adapter has already helped.
The current `stratum` embedding describes architecture boundaries; it does not describe anchor/draft role.

At the frozen E5 cut 0.36, the saved stratum aggregates are:

| Target stratum | Precision | Recall | Mean reads per forward-layer |
|---|---:|---:|---:|
| Enters Engram | 0.725 | 0.371 | 1.589 |
| Leaves Engram | 0.673 | 0.392 | 1.768 |
| Interior | 0.680 | 0.368 | 0.852 |
| 19->20 | 0.692 | 0.484 | 0.856 |
| Replay start | 0.724 | 0.359 | 0.495 |
| Late, 37-39 | 0.580 | 0.352 | 1.216 |

The pooled mean 0.96 conceals concentrated demand. These averages are **not** the missing per-group histogram,
nor a queue/deadline replay. Those must still be computed.

The score is also not a literal calibrated missing-row probability. For example, the saved E5 calibration bin
0.9-1.0 has 6,071 useful candidates of 7,821, about 77.6%. Token correlations, weighted token losses and the
route-versus-miss objective can all contribute; this review does not identify their individual causal effects.
The frozen threshold's observed precision remains valid even when the numeric score is miscalibrated.

### R1.5 Architecture critique and the proposed model

Current E5 uses fixed PCA, dense source-route mask/weights, sparse top-12 target-gate scores/margins, and small
source-layer/target-layer/stratum embeddings. `llapor.MiddlePredictor` is shared across layers; each horizon has
its own head. Per-token sigmoid outputs are aggregated afterward by noisy-or.

Four limitations are worth isolating:

1. **Training objective and unit of work.** `b1_head.make_batch` labels every non-VRAM target route positive,
   including RAM hits. `head_loss` is token-wise BCE with hard-negative weighting. The shared evaluator rewards
   unique NVMe misses after residency filtering. A frequently repeated expert supplies multiple token positives
   but costs one transfer. Route prediction remains a useful auxiliary task; it should not be the only objective.
2. **Aggregation.** `1 - product(1 - p)` is a useful baseline but does not learn cross-token dependencies. It can
   overstate the evidence supplied by correlated positions. A group loss can directly supervise the event that
   at least one live position needs the row.
3. **Layer specialization.** Expert index e refers to different weights in different layers. Layer embeddings let
   the shared MLP condition its behavior, but a small layer-specific output adapter gives that distinction a
   direct parameterization. Use regularized adapters before forty independent large networks.
4. **Representation.** The selected head reduces 5,120 dimensions to 128 by variance-preserving PCA. A supervised
   projection may retain routing information PCA discards. E5 also discards the early gate's scores outside its
   top 12. The serving kernel already computes the full `[tokens, 384]` score tensor before selection, so a
   full-score correction can access evidence that the present feature vector omits.

The preferred first candidate keeps a small shared token encoder, adds a learned block aggregator and small
target-layer adapters, then tests supervised projection separately:

```text
source gate input x[L], source routes, layer/stratum, token position and live mask
        -> shared token encoder + small target-layer output adapter
        -> per-token expert scores
        -> learned pooling across the live verification block
        -> 384 scores for unique expert-row demand
        -> exact hot/RAM/in-flight filtering, threshold, cap and deadline scheduler

Full-score challenger: early target gate's 384 scores + learned low-rank correction
        -> the same block aggregator and read policy
```

The native model router remains authoritative for execution. The predictor chooses speculative reads only.
Source and target layer, horizon, architecture stratum, token position and residency are distinct concepts;
do not replace one with another or use future target routes as input features.

### R1.6 Data contract, labels and split discipline for the next training run

**Reuse these immutable inputs:**

- Corpus: `/mnt/nvme1/nvfp4-work/b0-corpus-100-20261012`.
- Dataset: `/mnt/nvme1/nvfp4-work/b1-dataset-20261012`.
- Historical run: `/mnt/nvme1/nvfp4-work/b1-run-20261012`.
- Original train/dev/calib/locked session assignment: **60/15/10/15**. Keep warm-start and drive flags.

Use a fresh R1 output directory under `/mnt/nvme1/nvfp4-work/`; record the exact path in the new manifest. Treat
the original locked split as **previously observed historical evaluation**, never as a fresh test set. Do not
load its raw shards for architecture, loss, epoch, threshold, horizon or scheduler selection. Its existing tables
may be quoted descriptively, clearly labeled historical/post-hoc.

Before training:

1. Verify input hashes, session membership, layer IDs, route identity and current code commit; record them.
2. Resolve the original non-tie mismatch bookkeeping gap. Determine whether source features, target labels, or
   both are affected. Write an explicit exclusion manifest keyed by block/seq/layer/token, with reason and all
   affected source/target groups. Apply it identically to every R1 control and candidate in a derived dataset or
   reader overlay; never silently edit B0/B1. If denominators change, regenerate all compared R1 baseline tables.
3. Preserve whole groups during batching. A group key must include session, block/forward identity, source layer
   and horizon; a target expert row is deduplicated within that group's target layer. The existing row-shuffled
   training cache cannot simply be reused unchanged for a block loss.
4. Carry `tokens`, live masks, source/target layer, target stratum, source routes, target IDs/kinds, hot set and RAM
   snapshot. Store future demand kinds and routes as **labels only**. Group IDs, request/session names and drive
   flags are audit metadata, not predictor inputs.
5. Full early gate scores, if required by an arm, are recomputed from captured **source** x and the frozen
   **target** gate weights/bias: `sqrt(softplus(x[L] @ W[L+h].T)) + bias[L+h]`. The weights are already inputs to
   B1 extraction. This requires no new serving capture. Check its top-12 parity against stored gate features.
6. Fit PCA, normalizers, frequency priors and any learned preprocessing on **train only**. Reuse a train-fitted
   PCA across comparable arms. Use dev for model selection and calib for the final policy thresholds only.

For group g, expert e and live token m, define:

```text
route_y[g,m,e] = 1 iff captured target top-6 for token m includes e
miss_y[g,e]    = 1 iff any live token routes e with kind MISS_GPU or MISS_CPU
available[g,e] = not target_hot[g,e] and not target_in_ram_at_prediction[g,e]
```

`miss_y` is a union, not a sum of occurrences. Use `available` to mask the primary loss and to filter every read
policy. Keep all true misses, including snapshot-to-demand evictions, in the historical-compatible evaluation
denominator; additionally report the subset addressable under the prediction-time mask. Otherwise a training
mask could silently inflate recall. Such skipped positives cannot be recovered by this read policy.

Keep groups with no eligible misses: they teach abstention. Mask padding everywhere, including pooling,
auxiliary losses, normalization and metrics. An expert appearing in multiple positions gets one group label and
at most one read. An expert needed by a rejected position remains positive. If MISS_GPU and MISS_CPU overlap in
an expert's group-level attribution, deduplicate total demand and report overlap rather than double-counting.

### R1.7 Concrete architecture and loss defaults

These defaults bound the experiment; change them only in a recorded run-plan revision before evaluating the
change. They are not claims about optimal hyperparameters.

**Token encoder.** Start from the existing E5 rank-128, hidden-512 residual MLP and its 16-dimensional layer/stratum
embeddings. Preserve its source-route features. The position arm adds a learned 8-dimensional embedding for
positions 0-5 and a live-width scalar. Position 0 is the anchor. Do not introduce independent networks for all six
positions. Width-six data does not validate shorter or wider serving modes; use masks correctly but report that
limitation rather than manufacturing new verified width data.

**Block aggregator.** For each expert, form a small feature vector from token sigmoid scores: anchor score and
anchor-present flag; maximum draft score and draft-present flag; overall maximum, mean, standard deviation and
sum; noisy-or; live token count. Concatenate target-layer and stratum embeddings. Feed a shared
`Linear -> GELU -> Linear` MLP with hidden width **32** to one block logit per expert. Define empty draft summaries
as zero with the present flag false. Compute these features differentiably over live positions. No expert is
removed merely because it falls outside the early gate's top 12. Calibrate the final block scores at block level.

The first pooling-only arm freezes the token encoder. The next arm allows gradients through it; this separates
the value of aggregation from retraining the representation. As cheap controls, evaluate max pooling and the
original noisy-or with the same token scores and read-policy calibration.

**Layer adapter.** Add a rank-**16** residual adapter to the shared token classifier:
`z[m] = W_shared v[m] + U[target,h] V[target,h] v[m] + b[target,h]`, where v is the 512-wide shared representation,
V is `[16,512]` and U is `[384,16]`. Initialize U and b to zero and V to a small random value, so the initial
function matches the parent model while gradients can reach the adapter. Keep the shared classifier and layer
embeddings. Horizon-specific models may hold their own adapters. Do not tie target expert IDs across layers.

**Supervised projection.** In a separate arm, make the 128-dimensional projection of x trainable, initialized
from the train-fitted PCA; retain train-only centering. Use a learning rate one tenth of the new-head rate for
this initialized projection. Do not precompute and freeze its projected features in a cache once it is trainable:
the training path must retain or stream raw bf16 x with gradients through the projection.

**Full-score residual challenger.** Use all 384 early target-gate scores/margins, a learned x projection of rank
**64**, and a target-layer correction of width 384. A concrete token score is
`z = a[target] * s_native + b[target] + B[target] GELU(A x + position_embedding)`, with positive scalar a,
per-expert b, A `[64,5120]`, B `[384,64]` and 64-wide position embeddings. Define `s_native` as all 384 biased
early scores minus their token's sixth-highest score; these are full margins, not the stored sparse top-12 vector.
Initialize a=1, b=0 and B=0 with nonzero A. The initial challenger uses these raw margins; any train-only
normalization is a separately recorded ablation with statistics frozen at inference.
Feed these token scores to the same block aggregator. This is an alternative architecture, not evidence that
all its individual ingredients help; compare it against the best simpler arm with equal training data and read
constraints. It must preserve DSV4.1's sqrtsoftplus and selection-bias semantics.

**Primary loss:** per-group BCE-with-logits on `miss_y`, masked by `available`, averaged over available experts
within each group and then over groups. Groups with zero available experts contribute no primary term but are
counted. Start with unweighted primary BCE so there is no deliberate positive-class probability distortion.

**Auxiliary loss for joint arms:** ordinary token-wise BCE on `route_y` for all 384 experts, including hot/RAM
routes, masked only by live-token validity. This separates routing supervision from cache-dependent demand.
Use `loss = block_BCE + 0.25 * route_BCE`. The historical-control arm alone retains B1's original non-VRAM labels
and hard-negative weight 2.0. Record both losses separately; an improvement in token routing is not the endpoint.

**Optional ranking ablation, after screening:** on the best joint architecture, add weight **0.1** times mean
`softplus(z_negative - z_positive)` over available missing-row positives versus the eight highest-scoring
available non-missing experts per group. Skip groups without either side. Compare to the same model with weight
0; do not silently add this to every arm. Recalibrate scores after any change to the loss. Layer-cover or predicted
milliseconds losses are deferred until the timing model is validated.

### R1.8 Ordered ablations and training recipe

Train **L+1 first**, not the full cross-product of horizons, sizes, losses and seeds. The initial screening order is:

| Arm | Change | Question answered |
|---|---|---|
| F0 | Frozen original `h1_E5_r128_h512_w2.0.pt`; rescore dev under new capped policies | How much benefit is already available without retraining? |
| T0 | Same E5 architecture, original labels/loss, longer training | Was the 20-epoch cap hiding improvement? |
| T1 | Freeze T0; train only the block aggregator with block BCE | Does learned group aggregation/admission improve the read policy? |
| T2 | Unfreeze T1's encoder; use block BCE + 0.25 all-route auxiliary BCE | Does direct block-demand supervision improve the representation? |
| T3 | T2 plus token-position embedding and live-width input | Does explicit anchor/draft position help? |
| T4 | T3 plus target-layer rank-16 output adapters | Does layer specialization help? |
| T5 | T4 with trainable rank-128 x projection | Does supervised compression beat fixed PCA? |
| G0 | Rank-64 full-native-score residual challenger from R1.7 | Can a smaller gate-correction architecture outperform the MLP family? |

Also retain bare gate, packaged gate+floors, max/noisy-or pooling controls, and the hybrid that admits the native
gate's highest-ranked available row only when its learned block score clears the cut. Include the hybrid as an
explicitly evaluated policy; do not infer its result from the complete head. `prev` and `pop` remain sanity
references; their weak miss prediction is already established and is not the main research question.

Default training recipe:

- AdamW; new parameters learning rate **1e-3**, weight decay **1e-2**, dropout **0.1** in the token encoder.
- T0: retrain from scratch for up to **60 epochs**, five-epoch linear warmup then cosine decay. Keep its original
  token objective and original best-at-one-read checkpoint as the convergence control; also retain the checkpoint
  best under R1's development policy metric. Name both rather than silently switching the historical criterion.
- T1: freeze T0's checkpoint selected by the R1 capped-policy metric; train the aggregator for up to **30 epochs**,
  with three-epoch linear warmup then cosine. Keep the inherited layer/stratum embedding values frozen in this arm.
- T2-T5: up to **30 epochs** each from the corresponding parent checkpoint, copied rather than overwritten;
  inherited parameters use **1e-4**, new parameters **1e-3**. Use three-epoch warmup then cosine decay. Report this
  as staged fine-tuning and record every parent hash, so accumulated training is visible.
- G0: up to **60 epochs** from scratch, five-epoch warmup then cosine. Do not assume it is cheaper until measured.
- **512 complete groups per optimizer step** where memory permits, random group-order shuffling each epoch.
  Use gradient accumulation to preserve the same effective group batch if raw-x memory requires smaller physical
  batches. No token from a group may be independently shuffled away from its block.
- Screen with seed **0**. Early-stop after **8** epochs without improvement in the arm's named dev selection metric;
  retain train/dev curves, not just the winning point. T0 retains both checkpoint criteria described above.
- Re-run the strongest joint candidate and T0 with seeds **1 and 2** before final selection. If G0 is competitive,
  repeat it as well. Predeclare selection by **median dev policy score across seeds**; the seed-0 checkpoint is the
  deployable candidate, avoiding post-hoc selection of the luckiest initialization. If it is unrepresentative,
  report instability rather than silently picking a different seed. Reproduce the selected candidate's ancestor
  chain with each seed, including its T0 parent; rerunning only the last adapter on a shared seed-0 parent measures
  fine-tuning noise, not full-training variability. Only the already selected ancestry needs these repeat runs.
- If T3 or T4 adds no value, retain the simpler parent as the candidate and run subsequent feature ablations from
  that parent with the changed ancestry recorded. This is a planned sequential screen, not an exhaustive causal
  factorial experiment. A finalist should also get an equal-additional-epochs parent control if its gain could
  plausibly come just from the extra fine-tuning.

The longer control also changes the warmup/batch schedule from historical B1. If attributing its gain specifically
to epoch count, run a 20-epoch T0 control with the same R1 optimizer/batching recipe and report the learning-rate
schedule difference. Otherwise describe the result as an improved training recipe, not a clean epoch-only effect.

**Checkpoint/model-selection metric:** on dev, the best eligible-row recall along the threshold sweep under
**cap 2 and mean reads <= 1 per forward-layer**, with deterministic expert-ID tie breaks. This is a development
selection metric, not a frozen deployable threshold. Recompute each control's curve under the same cap and
budget. Report its realized precision and read count. Break ties by lower complete scorer cost, then fewer
parameters. Also report cap-1 results; do not select on uncapped recall or accuracy of all routes.

For selection, sweep the finite observed group scores (or an exact sorted cumulative curve), not different
hand-chosen threshold grids for different models. A group can read 0, 1 or 2 rows; no fractional reads and no
borrowing information from later groups to decide its actual serving action. A globally sorted curve is only an
offline way to select a single threshold, not an oracle scheduling policy.

After L+1 screening, train the selected architecture and its matched control at **L+2** using the same recipe and
budget. L+4 is an optional final diagnostic only if L+2 demonstrates useful additional lead. Initially compare
one horizon at a time. Combining horizons requires deduplicating `(forward, target layer, expert)` predictions,
in-flight suppression and one shared read budget; it is not equivalent to summing their separate gains.

### R1.9 Read policy, calibration, timing replay and decision rules

Keep prediction and scheduling as separate stages. The initial per-group policy is:

1. Produce block scores for all experts using only information available at the prediction point.
2. Remove VRAM-hot, snapshot-RAM-resident and already prefetched/in-flight rows using the same definitions for
   every policy. Offline snapshot-only evaluation must be labeled as such; it does not know counterfactual future
   residency produced by the policy.
3. Keep candidates above one global threshold per model/horizon, rank by score, break ties by expert ID, and
   truncate to cap **1** or **2**. Do not introduce per-layer or per-position thresholds in the first screen.
4. In timing replay, admit reads only when the modeled transfer capacity, private-pool capacity and deadline
   permit. A late useful row is not an on-time hit; if it arrives partially before demand, price only the modeled
   remaining stall reduction and expose the model assumptions.

Use the original E5 cut 0.36 and gate+floors cut 0.05 as fixed historical reference points. For fair prospective
comparisons, each model/control gets a newly calibrated threshold **for the capped policy**, on calib only after
architecture selection. A threshold from the uncapped policy is not automatically the correct capped threshold.
The primary cap-2 operating budget is mean reads <= 1 per forward-layer on calib; choose the lowest threshold
that respects that budget, breaking equal-read ties toward the higher threshold. Use cap-1 as a separate lane
and disclose its potentially lower read count. Keep actual total reads/bytes next to recall in every table.

The run must produce, for each control and candidate:

- Read-count histogram (0/1/2/3/4+ before capping), p50/p90/p95/p99/max, per-forward totals, consecutive-burst
  lengths, and distributions by layer, stratum and session. Report how many useful proposals each cap drops.
- Useful, wasted, skipped, duplicate/in-flight and eligible unique rows; precision, recall and outside-top-12
  recovery; whole-layer miss coverage as a separate metric.
- Per-position token diagnostic coverage and error rates, plus mutually exclusive anchor-only, draft-only and
  shared **row-set** attribution. Show the head and gate; the table in R1.3 only measures the gate.
- MISS_GPU/MISS_CPU attribution, warm-start and drive-flagged subsets, and known accepted-prefix versus
  reject-only rows where a validated accept-log join is available. Report missing joins; never infer acceptance
  from the token index or use it in the score.
- Reliability bins at the block/missing-row level, especially for candidates surviving the read cap and mask.
  Calibrated thresholds do not by themselves make every output a calibrated probability.
- Prediction-ready time, publication time, read issue/completion, demand deadline, bytes per drive, demand
  queueing penalty, late-hit fraction and on-time useful rows. Count any speculative read that delays demand.

Use existing `verify_replay`/layer-cover tools as components, not as a claim that a complete counterfactual I/O
simulator already exists. The B1 `Group` shards do not carry timestamps; join B0 residency snapshot/demand stamps
by block/seq/layer, and separately locate the applicable read-service/drive traces. Audit units and clock domains.
If detailed queue traces are unavailable, report a bounded sensitivity analysis using measured row-read costs,
zero/nominal/adverse contention and scorer delays. Label it a model, not a measured speedup. Do not price every
useful row at the last-miss slope and call the sum a serving forecast. Prefetch-on residency feedback ultimately
needs a controlled served comparison.

**Measure complete scorer cost.** The saved 0.058752 ms median / 0.05968 ms p90 measures PCA/source feature build,
embeddings, MLP and sigmoid with random gate features supplied beforehand. It excludes the real gate calculation,
gate feature scatter/build, block aggregation, hot/RAM filtering, candidate selection and publication to readers.
Measure all of those together at six live tokens, graph replay and intended numerical precision, including side
stream joins and candidate-buffer lifetime. Report head-only and complete-path cost, parameter/buffer bytes,
prediction-to-publication delay, and interference with target work. The bf16 byte estimate is not evidence that
the original fp32 timing was a bf16 implementation. Test at the prospective serving dtype and record routing
prediction differences caused by that dtype.

**Proposed R1 offline continuation gate** (freeze in `run_plan.json` before the run; it does not rewrite B2):

- Under cap 2, the candidate improves useful-row recall by at least **10% relative** to the best native-gate
  control selected on dev, while satisfying the same mean-read budget. Report comparisons to T0 and F0 as well;
  an architecture change is not justified solely by beating a weak native baseline if a simpler learned control
  is as good.
- At the final calib-chosen operating points, require no lower precision and no more than **5% extra reads**
  relative to that native control. Report absolute counts and the read-budget curves. This tolerance accommodates
  discrete thresholds; it is not permission to hide bandwidth growth.
- Complete scorer median <= **0.1 ms/layer**, with p90/p99 and deadline effects reported. Meeting this bound alone
  is insufficient if the additional delay removes the predicted transfer benefit.
- The advantage survives the timing model's stated constraints and is stable across seeds and the no-drive-flag
  view. The cap-1 lane may lose; report it rather than using its different objective to veto or rescue cap 2.

Dev/calib passage warrants **independent confirmation**, not production adoption. The old locked test has already
informed this critique. If the candidate passes, propose a fresh disjoint-session confirmation capture with a
declared sample size and workload, preserving the no-production-restart boundary. Include an ordinary-decode
capture only if a real ordinary-decode comparison is part of that next approval; do not label anchors as it.
Do not launch this capture as a side effect of training.

Before any fresh confirmation, freeze model/parent hashes, dtype, transforms, masks, thresholds, caps, horizon,
comparators, timing assumptions, exclusions and the exact statistical estimand. Bootstrap **paired whole sessions**
with 10,000 resamples, seed 0. For the chosen capped policy, resample sessions and recompute pooled recall from
summed useful/eligible counts, then head-minus-control recall difference; require its 95% lower bound > 0 alongside
the point/read/cost criteria above. Resample both arms with the same session indices. Also report the recall ratio,
precision difference and read counts, and a block-aware sensitivity view because the original corpus has only four
server starts with shared cache history. Do not report this as the old median-session-precision bootstrap.

If no fresh capture is approved or available, end with **exploratory improvement on reused data; independent
confirmation pending**, not a claim that R1 passed a new locked test. Never reselect a model or threshold on the
confirmation data. A serving A/B remains a separate step comparing verify step time, demand stalls and output
correctness under the same window and workload.

### R1.10 Implementation and launch handoff

No R1 training entry point exists yet. Do not run the old `b1_run.py` with invented flags or overwrite its output
directory. The implementation should add a separate R1 driver/config and preserve the B1 reproduction path.

| Work item | Existing code to reuse or inspect | Required result |
|---|---|---|
| Data/exclusion audit and group batches | `b1_rows.py`, `b1_extract.py`, `b1_dataset.py`, `b1_cache.py`, B0 integrity/capture readers | Stable group masks, explicit exclusions, denominators and train-only transforms |
| Models and losses | `b1_head.py`, `expert_prediction/training/llapor.py`, `pca.py` | Pooling, position conditioning, adapters and separately switchable projection/challenger arms |
| Group score interface | `b1_score.py`, `b1_baselines.py` | One native block score per expert, common filtering, threshold-plus-cap policies and metrics |
| Training and selection | `b1_train.py`, `b1_select.py`, `b1_run.py` | Named arm/parent/seed configs, group-preserving optimizer steps, capped dev selection, freeze manifest |
| Timing and cost | `b1_cost.py`, `verify_replay.py`, DSpark layer-cover tools, `expert_stream_transport.py` | Honest complete-path timing and constrained replay/sensitivity outputs |
| Reporting and checks | `b1_report.py`, existing `test_b1_*.py` | Position/block/layer diagnostics, repeatable controls and a self-contained R1 results handoff |

The current score API expects token scores and aggregates them internally. Add an explicit group-score path for
the new head; do not send already pooled probabilities through noisy-or a second time. Preserve a parity test
showing that all scorers receive identical groups, skip masks and eligible denominators. Reuse immutable raw-x
shards; do not duplicate x for every horizon. For trainable projections, budget raw-x/device-cache memory before
loading it and stream by session/chunk if needed. Historical B1's 27-minute GPU training time is only a baseline,
not an R1 runtime estimate.

Minimum correctness checks before a full run:

- A shared expert used by several tokens contributes one primary label and one read; token auxiliary labels
  remain per-token. Available/hot/RAM masks and eviction-denominator handling agree across controls/candidates.
- Changing a padded token cannot change the loss, prediction or metric; live anchor/draft attribution is correct.
- Group batches never mix tokens from different forwards; cached and uncached group scoring agree.
- Source/target exclusion propagation is explicit, and no future demand/acceptance field reaches predictor inputs.
- Full gate recomputation agrees with stored features within the established tie/precision convention.
- Threshold, cap, deterministic tie-breaking, one-horizon accounting and the session bootstrap are checked on
  small examples where expected reads, unique hits and confidence-interval inputs can be enumerated.
- Tiny training smoke has finite gradients, moves a block loss and saves/reloads an equivalent model. Run only
  affected tests; do not add or run unrelated suites for this experiment.
- Complete GPU scorer uses the intended graph/stream and numerical dtype and covers candidate publication, not
  merely a standalone MLP forward.

Launch sequence for the next session:

1. Read this section, §33.25/§33.26 and `.claude/rules/divix01-run-protocol.md`; inspect HEAD and dirty files.
   The handoff was untracked during critique; preserve it. Confirm the requested scope is the offline R1 study.
2. Implement the bounded study in its own branch (default prefix `codex/`), leaving serving defaults and the
   production checkout untouched. Record this as the next Track B experiment; do not reopen F105's frozen result.
   If the work becomes serving architecture, use hx-brainstorming/hx-planning before that implementation.
3. Follow the existing code-transfer protocol: commit and push code to `origin`, then use a private divix01
   worktree at that pushed commit. Never scp/rsync a source tree. Set `PYTHONPATH=$PWD/python`; verify
   `sglang.__file__` before importing repository code in a run.
4. Check for `launch_server`, `both_cpu_ab` and `run_arm.sh` processes and other work using the needed resources.
   Nothing else may run beside a served A/B. Do not start or stop production to obtain an idle host.
5. Use `~/.hx/checks/divix01-pytest [--gpu] <named affected files> -q -p no:randomly` for the required tests.
   The laptop cannot run sglang tests. Put temporary/runtime output on an allowed work volume.
6. CPU analysis/training preparation uses `OMP_NUM_THREADS` with `taskset -c 0-63`; preserve cores 64-71 for
   completions. Reuse the documented 8-GiB CPU analysis memory cap where appropriate. GPU work uses
   `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`. If both locks are required, acquire
   `rowimg-disk.lock` before `cc-gpu.lock`. Capture the actual command's exit status, including `PIPESTATUS`
   when output is piped. No capture/run output belongs on nvme0/2/4.
7. Create and hash `run_plan.json` in the new output directory **before** training: all arm definitions, seeds,
   parents, epochs, objectives, model-selection metrics, calibration procedure, continuation gate and resource
   settings. Record the real implemented CLI command for every stage; this document deliberately does not pretend
   the future driver is already executable.
8. Run data audit and a small train/dev smoke, then F0/T0, T1-T4, the projection/challenger screen, seed repeats,
   and the selected L+2 comparison. Each stage writes an atomic completion record with input/output hashes;
   resume only when dependencies match. Do not silently skip a failed stage or reuse incompatible cached features.
9. Calibrate the selected architecture and controls, measure complete scorer cost, finish constrained replay,
   freeze the final candidate manifest, and report the R1 continuation gate. Stop before any new serving capture
   or A/B unless that action has separately been authorized.

Expected R1 outputs, all in the new run directory: `run_plan.json`, input/exclusion manifests, group diagnostics,
per-arm configs and parent/checkpoint hashes, per-epoch curves, seed comparisons, per-group candidate/read records
for dev/calib, policy curves, calibration tables, timing/replay assumptions and results, complete-path cost,
`selection.json`, `REPORT.md` and machine-readable `results.json`. Include every exact invocation and exit code.
The report must say whether the gain comes from ranking, abstention, block aggregation, added capacity, training
duration or longer lead, and which of those explanations remains unresolved.

### R1.11 Deferred ideas and source map

- **Later pre-attention prediction:** a more informative input may improve accuracy but loses transfer lead.
  Compare on-time useful rows, not just routing accuracy. R1's first arms retain the current prediction point.
- **Layer 0 from draft tokens:** §33.19 already investigated this. Draft-to-layer-0 demand lead was p50 1.97 ms
  versus roughly 2.2-2.4 ms for a row read. The anchor was known much earlier but covered only a minority of that
  layer's demand. Revisit only with a measured source of additional lead, not another token-ID predictor alone.
- **Full mHC residual information:** DSV4.1 carries four residual streams (20,480 values) while B0 stores the
  collapsed 5,120-wide MoE input. A compact learned summary of the full state could expose information the
  current input cannot recover. It needs new feature capture and cost measurement; widening the current MLP is
  not a substitute. Defer until the existing-data ablations establish the remaining limitation.
- **Small attention-based block encoder:** six-token attention is a plausible follow-up if inexpensive pooling
  saturates. It is not the first model to build, and its complete inference cost still counts.
- **Direct draft-to-target prediction:** draft hidden states may be an additional early signal, but separate
  expert vocabularies and available lead require a new supervised design. Do not assume the draft's selected
  experts are target candidates.

Primary research precedents, consulted during the critique, provide architectural ideas rather than transferable
accuracy or speedup claims for DSV4.1:

- [SpecPrefetch](https://arxiv.org/html/2607.24787v2): layer-specific low-rank expert predictors and transfer-window
  budgeting while preserving the native execution router.
- [Pre-Attention Expert Prediction and Prefetching](https://arxiv.org/abs/2511.10676): same-layer pre-attention
  predictors and ranking-aware objectives; the later input creates a different lead-time tradeoff.
- [SP-MoE](https://arxiv.org/abs/2510.10302): speculative verification expands expert demand and motivates
  prefetching ahead of verification; its use of draft/target correspondence does not establish correspondence
  between DSpark's 128-expert draft stages and DSV4.1's 384-expert backbone layers.

Repository evidence to recheck if HEAD moves:

| File / symbol | What this critique relies on |
|---|---|
| `DSV41_REFERENCE.md` §§2, 10, 33.19, 33.25, 33.26 | Model geometry, DSpark roles, layer-0 lead limit, corpus and historical results |
| `analysis/dsv41-drive/prefetch-replay/b0_integrity.py::_join` | B0 keeps target-verify request forwards |
| `analysis/dsv41-drive/prefetch-replay/b1_rows.py::Group`, `eligible` | Token indices, unique miss labels and dataset layout |
| `analysis/dsv41-drive/prefetch-replay/b1_head.py::make_batch`, `B1Head`, `head_loss` | Fixed PCA, shared token model, embeddings and non-VRAM BCE labels |
| `analysis/dsv41-drive/prefetch-replay/b1_score.py::_aggregate`, `_budgets` | Noisy-or and separate fixed-read/uncapped-cut policies |
| `analysis/dsv41-drive/prefetch-replay/b1_cache.py::take`, `recall_at_1` | Token-indexed batches and original early-stop metric |
| `analysis/dsv41-drive/prefetch-replay/b1_select.py::paired_diffs`, `verdict` | Precision-only bootstrap behind the original verdict |
| `analysis/dsv41-drive/prefetch-replay/b1_cost.py::_make_step` | Supplied random gate features and head-only timing boundary |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py::spec_score`, `_spec_launch`, `spec_join` | Full gate scores, GPU side stream, publication and lifetime joins |
| `python/sglang/srt/layers/moe/residency_capture_ref.py::horizon_targets` | Only absent target rows invalidate a horizon; stratum crossings remain valid |
| `python/sglang/srt/layers/moe/ram_slot_map.py::LaneKind` | MISS_GPU=4, MISS_CPU=5; resident lane types are not miss positives |
| `python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py::compact_verify_ids` | Anchor versus draft verification positions |
| `python/sglang/srt/models/deepseek_v4_dspark.py::_dspark_stage_config`, `DSparkV4Stage` | Separate draft-stage expert configuration |
| `python/sglang/srt/models/deepseek_v4.py::_hc_combine` and MoE boundary code | Four-stream residual versus collapsed MoE input |
| `divix01:/mnt/nvme1/nvfp4-work/b1-run-20261012/stages/locked/h1.json` | Historical pooled/stratum/read/calibration numbers above |
| `divix01:/mnt/nvme1/nvfp4-work/b1-dataset-20261012/dev/h1/` and `dataset.json` | Development-only position coverage and 76,073 unique miss-row reconciliation |

**First experiment to prioritize:** the block-level missing-expert objective with learned pooling, followed by
small target-layer adapters, using the existing corpus. Its success criterion is a better feasible read policy,
with a measured complete scorer cost, not simply a larger token-route accuracy number.

## ROUND 2 RESULTS: R1 and the L+2 spike

**Date and code:** 2026-10-12. hx feature F106 (story ST8), merged as `18e937f060`; results `222fcba5e5`; L+2 spike
`7175bd5405`. Record: `DSV41_REFERENCE.md` §33.27. Run directory `divix01:/mnt/nvme1/nvfp4-work/b1-r1-20261012/`
(`REPORT.md`, `results.json`, `selection.json`, `run_plan.json` + `.sha256`, `spike-l2/result.json`); smoke
`b1-r1-20261012-smoke/`; worktree `divix01:/data/models/slang/nvfp4-work/wt-r1-run`. Every number is **exploratory on
reused data (dev/calib of the B0 corpus); independent confirmation pending**. The old locked test was never loaded.

### What R1 changed relative to the critique (owner-approved amendments)

1. **Stage 1 first, with a stop.** The frozen B1 head (F0) and the native controls were scored under caps before
   any training; training ran only because F0 cleared +10% recall over the best native control under cap 2 and kept
   more on-time useful rows at nominal contention.
2. The label point was reframed: on available experts a route *is* a miss up to the 0.56% eviction rate, so the
   block objective's expected gain is union labels and abstention, not a label fix.
3. Scorer cost is offline only (gate scores through candidate selection); reader publication belongs to a serving spec.
4. The timing replay is a bounded sensitivity model from B0's own stamps: row read 2.2 ms (§33.19) times contention
   1.0 / 2.0 / 3.0 (zero / nominal / adverse; nominal from §33.16's "a speculative read doubles a demand read").
5. Run through hx, not a `codex/` branch.

Frozen dtype for cost: bf16 head and aggregator over an fp32 gate, an fp32 path timed beside it.

### Stage 1 (no training), L+1, cap 2, mean reads <= 1 per forward-layer

| Scorer | Dev recall | Dev precision | Calib recall | Calib precision |
|---|---|---|---|---|
| F0 (B1 head) | 0.317 | 0.528 | 0.330 | 0.578 |
| gate | 0.288 | 0.479 | 0.284 | 0.498 |
| gate + floors | 0.279 | 0.508 | 0.276 | 0.526 |

Stage-2 gate passed narrowly: +10.3% dev recall (bar +10%), on-time useful rows at nominal 15,652 vs 14,821. The cap
shrinks B1's ~17% uncapped equal-read advantage: part of it came from piling reads onto bursty layers. **The cap-1
lane and the hybrid are degenerate** under the frozen "lowest threshold within budget" rule (cap 1 never binds the
mean, so both read on every forward-layer); read cap 2.

### The arm ladder (L+1, seed 0; dev recall at cap 2)

| Arm | Change | Dev recall | Seeds 0/1/2 |
|---|---|---|---|
| F0 | frozen B1 head | 0.317 | - |
| T0 | B1 architecture and loss, retrained (60 epochs) | 0.324 | 0.324 / 0.323 / 0.323 |
| T1 | + learned block aggregator, frozen encoder | 0.326 | 0.326 / 0.325 / 0.324 |
| T2 | + block BCE + 0.25 route BCE, encoder unfrozen | 0.331 | 0.331 / 0.330 / 0.331 |
| **T3** | + position embedding and live width (candidate) | **0.332** | 0.332 / 0.331 / 0.331 |
| T4 | + rank-16 target-layer adapters | 0.331 | - |
| T5, G0 | trainable projection; full-score challenger | **not run** | - |

- T5 and G0 were skipped by the driver's own 8 GiB raw-x cache guard ("stream it instead"; streaming is not
  implemented). Untested, not negative.
- T0 matching F0 rules out a broken training path. All R1 additions together add ~+0.015 recall; the gain is mostly
  the B1 head's ranking (attribution: ranking supported; aggregation, capacity, training length below the bar).
- T3 on calib, at the same 20,787 reads as the gate: recall 0.348 vs 0.284, precision 0.609 vs 0.498.

### Continuation gate: FAIL on cost alone

Met: recall +15.3% (0.3317 vs 0.2877); calib precision not lower; calib reads equal; seed median 0.3308 (min 0.3307);
without flagged sessions 0.3260 vs 0.2856; on-time useful at nominal 16,282 vs 14,834. **Not met: complete bf16 scorer
0.193 ms per layer (bar 0.1).** Head only 0.101 ms; the offline gate alone 0.064 ms, about 10x the serving score
kernel (~6 us), so the complete figure overstates a serving implementation, but the head alone sits at the bound.
bf16 vs fp32 changes the capped read set in 0.9% of groups.

### The L+2 spike: lead time is the bigger lever

Native gate applied two layers early vs R1's L+2 heads, same scoring path, cap rule and per-scorer delays.

| L+2 scorer | Dev recall | Dev precision | Calib recall |
|---|---|---|---|
| gate | 0.251 | 0.401 | 0.251 |
| T0 (L+2) | 0.309 | 0.493 | 0.329 |
| T3 ladder (L+2) | 0.319 | 0.509 | 0.343 |

On-time useful rows (dev; timing model; the gate charged 0.064 ms, heads 0.193 ms):

| Contention | gate L+1 (served today) | T3 L+1 | gate L+2 | T3 L+2 |
|---|---|---|---|---|
| zero | 21,165 | 24,067 | 17,863 | 22,658 |
| nominal | 14,834 | 16,282 | 17,566 | 22,166 |
| adverse | 7,626 | 8,257 | 15,510 | 19,167 |

- At L+1 under contention, a third (nominal) to two thirds (adverse) of the gate's useful reads land **late**. At L+2
  almost none do (nominal: 297 late of 17,863). This matches §33.24's "21% of used reads still in flight at use".
- The bare gate at L+2 is less accurate (recall 0.251 vs 0.288) but beats today's L+1 gate on on-time rows by +18%
  at nominal and 2.0x at adverse. It needs no learned model and no new scorer cost.
- The head's advantage grows with distance (+27% recall at L+2 vs +15% at L+1); T3 at L+2 is +49% / 2.5x over
  today's L+1 gate at nominal / adverse.

**Caveats on the timing model.** It prices each forward-layer independently with a contention multiplier and
does not model cross-layer drive queueing; it ignores the pool holding a row one layer longer (capacity, eviction);
L+2 covers 1,173 fewer forward-layers (one fewer valid source row per forward). It is a model, not a measured
speedup. The served gate+floors prefetch at L+1 measured as a tie (§33.22). That is consistent with this model,
in which much of L+1's useful work arrives late, but it does not prove an L+2 variant will win.

### How this session reads it (an opinion; argue with it)

1. **The learned predictor alone will not move serving.** About +10% on-time useful rows over a served baseline
   that ties is likely below A/B noise, and it costs ~0.1-0.2 ms per layer.
2. **Moving the native gate to L+2 is the cheapest experiment with a plausible serving effect.** The GPU scorer's slot,
   the candidate page and the speculative threads exist; the change is which target gate the score kernel applies
   (W[T+2] on x[T]) and how long the pool keeps a row. B0 already captured horizons up to 4, so an offline check of
   L+4 costs one more spike run.
3. If the L+2 gate wins in serving, the learned head is a second step on top of it (+26% on-time rows over the L+2
   gate in this model), and its cost problem becomes worth solving with a fused kernel.

### Open questions for the next reader

1. Do you trust the timing model's L+1 -> L+2 shift, given that it ignores cross-layer queueing and pool capacity? What
   is the cheapest way to check it before a served A/B (for example, replaying the B0 drive traces, or a short
   instrumented served run with the gate retargeted to L+2)?
2. Should the next served experiment be the native gate at L+2 (no model), with floors re-fitted for L+2, before any
   learned-predictor serving work?
3. Is L+4 worth the same spike? (B1's L+4 gate precision fell to 0.257 at 1 read; more lead may not pay.)
4. Is it worth implementing raw-x streaming to run T5 and G0, given that the ladder adds only ~+0.015 over F0?

### Still open from round 1

- Independent confirmation capture: not taken; none of the above has a fresh locked test.
- Housekeeping as listed under "Housekeeping still open" (plus `wt-r1-run`, `b1-r1-20261012-smoke`).
- F105 Q1 in hx stays open on purpose as evidence for an hx display/replan bug investigation.

Production on divix01 is still stopped; only you can restart it.
