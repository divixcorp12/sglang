# Pipeline baseline: what the Task 1 arms measured, how far to trust them, and what is still open

Written 2026-09-21 for someone who was not here. It is the deliverable Task 1 of
`docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md` names, and it exists to stop the mistakes
`DSV41_REFERENCE.md` section 19 made: single shots quoted as baselines, a reference that was one arm
described as three, and "not contended" read as "undisturbed". It is a document, not a result: no arm was run,
no GPU or drive was used to write it. Everything below is read from the arm files, or recomputed from them
with the commands in Appendix B.

**The numbers are only as good as their `n` and their monitoring, and both are printed next to the number.**
Where a value is one arm, or comes from arms nobody was watching, it says so where it appears.

## 0. Where the data is (and is not)

All 17 baseline arms are in
`/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-drive/task1-results/` on **divix01**
(134 MB; the traces are most of it). Below, `R/` means that directory.

**The arm outputs are not in the git tree.** At `ec64c07999`, `git ls-files analysis/dsv41-drive/task1-results` lists only `clean-reference.json` and
`task1e-PREDICTIONS.txt` (added by `450f86d17a`); the arm jsons, verdicts, caches, traces and the other PREDICTIONS files are on divix01 only. `git log --all -- '*task1-results*'` finds no commit that added them. Section 19 and the plan say "raw output in
`analysis/dsv41-drive/task1-results/`"; that is true of divix01's analysis directory, not of the repository. The
arms are therefore identified below by file name and by the first 12 hex digits of the sha256 of each arm
json (Appendix A), so a later reader can tell whether a file is the one this document describes.

Per arm, `R/<arm>.json` is the driver's report (provenance, per-session timings, step latency, residency),
`.cache.json` the script's before/after residency and `/proc/diskstats` deltas, `.verdict.txt` the verdict tool's
output **at the time**, `.regime.json` the verdict's summary (task1c onward), `.log` the engine log, and
`.trace` / `.trace.cache-stats` the stage trace for traced (`T`) arms only. `t1-0-new-off-T.*` is the tooling
shakedown (`R/t1-0-SHAKEDOWN.txt`): it is not a baseline arm and is not counted anywhere below.

## 1. Read this first

| claim | number | `n` | where it comes from |
|---|---|---|---|
| Mirrors off, mean decode tok/s (new code) | **2.903-2.919** (mean 2.910, sd 0.007) | **5** | `R/task1-{0,1,4,5}-new-off-*.json`, `R/task1c-2-new-off-T.json` |
| Mirrors on (new code) | **3.905-3.956** (mean 3.930, sd 0.021) | **4** | `R/task1-2`, `task1-3`, `task1b-0`, `task1c-0` (`clean-reference.json`) |
| Old barriers, mirrors on (`099eadba33`) | **3.798** | **1** | `R/task1c-1-old-on-U.json`, **and that one arm has no machine-load record** |
| Mirror effect | **1.347x with 3 on-arms, 1.351x with 4** (per-arm ratios 1.338-1.363) | 4 / 5 | means above; section 3 |
| New code over old (two-bank, tier counters, two comment-only commits) | **+3.2 % to +3.5 %**, positive in all four sessions | old `n=1` | section 3, section 5 |

- **The old-barrier reference is one clean arm, not three.** It is corroborated by undisturbed sessions of four
  disturbed arms; that corroboration is not a second clean measurement (section 5).
- **"VALID" and "not contended" mean much less than they sound.** `task1d-0-old-on-U` is VALID, "not contended",
  carries no OUTLIER note, and ran **14.9 % slower** than the clean arm, with two sessions 31 % and 28 % slower
  (section 4).
- **Three of the six pre-registered predictions were falsified: P3, P6 and P7** (P4 is not a prediction). One, P5, is untested. Section 6.
- **13 of the 17 arms have no machine-load record at all**, including the arm used as the only old reference
  (section 8). Their quietness is inferred from tight repeatability, not observed.
- **The two-bank pipeline is not isolated.** "New" versus "old" is four commits under `python/`, one of which is
  two-bank and one adds counters on the tier's hot path (section 3.2).

## 2. What an arm is

Every arm runs the same workload, launched by `analysis/dsv41-drive/task1-baseline-arms.sh` (blob at HEAD; each
arm's `R/*-run.out` header prints the script's and the verdict tool's sha):

- graph decode, `SGLANG_MOE_EXPERT_GRAPH_GATHER=1`, 4 sessions (`--skip 0 --n 4`) of 256 prompt tokens and 128 new
  tokens each, 70 GiB pinned host tier (section 19), the harness `scripts/dsv41/trace_corpus.py` from the new worktree in every
  arm so the old arms have the same driver and fields;
- from a **clean detached worktree at an expected sha** (`wt-task1-old` at `099eadba33`, `wt-task1-new` at the
  arm's sha); the script refuses to start on a dirty tree, with production up (`:7867` listening) or a compute
  process on the GPU;
- through `gpu-run.sh` (`cc-gpu.lock`), which pins the arm to cores 32-63; `OMP/MKL_NUM_THREADS=16`;
- `SGLANG_MOE_EXPERT_MIRROR_DIRS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash` set (**on**) or unset (**off**);
  `SGLANG_DSV41_EXPERT_TRACE_PATH` set (**T**, traced) or unset (**U**);
- reads through `uring_direct` (O_DIRECT), so the ~200-400 GiB an arm reads cannot warm the page cache for the next one
  (this is checked per arm, section 4.1).

**Arm naming.** `<label>-<index>-<code>-<mirror>-<trace>`. `code` is `old` (`099eadba33`) or `new`. The seven
`new` arms of `task1` (0-6) ran at `799ef9d1cb`; the later ones at `f6608901a3` (`task1b`, `task1c`) and `4626789547`
(`task1d-1`). All three `new` shas have the **same `python/` tree**, `815cb72330...` (`git rev-parse <sha>:python`),
which the verdict tool labels "gen1"; the differences between them are harness-only. The `old` arms' tree is
`921bc132c0...` ("gen0"). Sessions carry different prompts, so per-session decode varies from 2.3 to 4.8 tok/s inside one
healthy arm: **compare arms session by session, never mean against a single session.**

**Code generations.** `5e92db22dc` changed `python/` again (stage-trace schema 3; the plan's Task 1 box says so). Arms
measured at gen1 (all 17 here) are **not directly comparable to any arm at that commit or later, even with tracing off**
(`git rev-parse 5e92db22dc:python` = `ee7029642...`, its parent = `815cb72...`). A pre-registered check that would tell whether
the difference matters exists and has not run (section 6, R1).

## 3. The matched baselines

### 3.1 The cells

Source for every row: the named arm files; statistics are computed in Appendix B. "Mean decode tok/s" is the mean of the four
sessions' `decode_tok_s` (`R/<arm>.json` key `mean_decode_tok_s`). Every arm in the table is in `R/clean-reference.json`. Their stored verdicts read VALID except `task1b-0`, which read INVALID under the whole-arm gate then in force and was reclassified by the later timed-phase gate (section 4.1). None has a boundary-sample record (section 8).

| cell | arms (mean decode tok/s) | `n` | mean | sd | range |
|---|---|---:|---:|---:|---|
| new, mirrors **off** | `task1-0-T` 2.9074, `task1-1-T` 2.9172, `task1-4-U` 2.9049, `task1-5-U` 2.9029, `task1c-2-T` 2.9188 | **5** | 2.9102 | 0.0073 (0.25 %) | 2.9029-2.9188 (0.54 %) |
| new, mirrors **on** | `task1-2-T` 3.9053, `task1-3-U` 3.9269, `task1b-0-T` 3.9561, `task1c-0-T` 3.9332 | **4** | 3.9304 | 0.0209 (0.53 %) | 3.9053-3.9561 (1.29 %) |
| old, mirrors **on** | `task1c-1-U` 3.7984 | **1** | 3.7984 | not defined | not defined |

`task1-6-new-on-U` (3.9361) is excluded: its verdict was INVALID under the gate then in force (whole-arm residency +1.40 GiB,
`R/task1-6-new-on-U.verdict.txt`), and it has no per-session residency to re-judge under the later gate. It lies inside the
on-cell range, so nothing suggests it was wrong; it is not evidence either.

**Per session** (mean over the cell's arms, range in brackets; `R/<arm>.json` `per_session[i]`):

| | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| off, tok/s | 2.320 (2.309-2.331) | 3.315 (3.307-3.324) | 2.385 (2.377-2.396) | 3.621 (3.617-3.625) |
| on, tok/s | 3.207 (3.166-3.245) | 4.419 (4.402-4.436) | 3.333 (3.307-3.358) | 4.763 (4.747-4.786) |
| old on (n=1), tok/s | 3.075 | 4.280 | 3.182 | 4.656 |
| off, TTFT s | **96.7 (94.5-99.4)** | 55.5 (55.2-55.8) | 53.7 (53.5-54.2) | 56.4 (56.2-56.7) |
| on, TTFT s | **55.2 (49.8-60.3)** | 30.2 (30.0-30.2) | 29.2 (29.1-29.3) | 30.5 (30.4-30.7) |
| old on (n=1), TTFT s | 55.7 | 30.6 | 29.4 | 30.6 |

**Step latency** (`per_session[i].step_latency`, 127 steps per session; `multi_token_chunks` was 0 in every arm, so the
percentiles are exact). Cell figure = mean over the cell's arms of the mean over the four sessions' percentile; this
averages percentiles and is a summary, not a pooled percentile: off p50 0.324 s / p95 0.603 s / p99 0.750 s; on p50 0.240 /
0.400 / 0.528; old (n=1) 0.248 / 0.431 / 0.552. Process CPU seconds over the four sessions, per arm
(`per_session[i].cpu_s` summed): off 402.0-412.2, on 348.0-359.6, old 359.9.

**Session-0 TTFT** is the one column with visible spread inside the clean on cell (49.8-60.3 s) while the off cell's
is tight (94.5-99.4 s). It is unexplained (section 7.1). Do not pool it across arms without saying so.

### 3.2 The two effects, and how much they rest on

**Mirror effect: 1.347x (three on-arms) to 1.351x (four).** `DSV41_REFERENCE.md` (committed text) reports 1.347x from
`3.92 / 2.91`, where 3.92 is the mean of `task1-2`, `task1-3`, `task1c-0` (3.9218) and excludes `task1b-0` (3.9561), a traced
arm whose boot populated the page cache (section 7.2) and which the reference manifest nonetheless lists as clean. Including it,
3.9304 / 2.9102 = 1.3505. Per arm pair the ratio spans 1.338 (lowest on / highest off) to 1.363 (highest on / lowest off).
Per session (means over arms): 1.382x, 1.333x, 1.397x, 1.315x. The mirror comparison is against the **`/mnt/nvme2` source, whose link
is Gen3 x2** (`DSV41_REFERENCE.md` section 19, "What the gain is measured against"): most of the gain is leaving that link, not
using two drives; that is a per-row bench decomposition, not an end-to-end one.

**New over old: +3.2 % (three on-arms) to +3.5 % (four on-arms).** 3.9218 / 3.7984 = 1.0325; 3.9304 / 3.7984 = 1.0347; per on-arm
+2.8 % to +4.2 %. Per session, means of the four on-arms against the one old arm: **+4.3 %, +3.3 %, +4.7 %, +2.3 %**, all positive.
Three limits on that:

1. **The old side is one arm.** Its own run-to-run spread is unknown. The sign is robust against the *new* cell's spread (0.5 %), not
   against the old arm's unmeasured one. Section 5 reports what the disturbed old arms do and do not add.
2. **"New" is not "two-bank".** `git log --oneline 099eadba33..f6608901a3 -- python/` lists four commits: `ddcb0d55ff`
   (two-bank pipeline, and rows unpublishable unless read), `cd14545797` (pinned-tier and Engram cache counters, on the tier's
   touch/assign path), and two comment-only commits (`be76ba501f`, `6a606e2b33`). The delta is attributed to two-bank because it
   is the one commit that changes the read pipeline; nothing here isolates it from the counters.
3. **Traced and untraced arms are pooled.** Off: traced 2.9144 (n=3) against untraced 2.9039 (n=2), ratio 1.0036; on: traced
   3.9316 (n=3) against untraced 3.9269 (n=1). Tracing looks neutral within noise, at those `n`. That is not a measured
   overhead: `038abc5b8f` added a stage-trace overhead harness that is unrun.

Traces (`R/*.trace`; `task1c-0-new-on-T.trace` holds 20,800 `ram_miss_request` records, 495 `graph_step` records, and its records carry `schema` 2) exist for
the six traced baseline arms (`task1-0`, `-1`, `-2`, `task1b-0`, `task1c-0`, `-2`) and the shakedown. **This document does not analyse them**; per the plan's Task 1 boundary, spans recorded before Task 4 do not
carry across the reader change and were not re-derived here.

## 4. How an arm is judged, and what each judgement does not mean

Everything below is `analysis/dsv41-drive/task1_arm_verdict.py`. **The `.verdict.txt` next to each arm is the tool as it was when the
arm ran**; several checks (CONTENDED, CROSS-ARM, GENERATION) did not exist for the earlier arms. Where this document quotes them for an
earlier arm it says "recomputed", meaning I imported the tool's functions at HEAD and applied them to the arm json (Appendix B). The
arm files themselves are unchanged.

### 4.1 Provenance and gates (what `VALID` means)

`VALID` means only that the arm's own json shows: it imported `sglang` from the intended worktree at the expected, clean sha;
provenance fields were available and no non-secret knob was over-redacted; the reader resolved to `uring_direct`; mirrors, tracing and
`GRAPH_GATHER` match the arm's name; the drives were idle at the start (the driver measures 2 s; the script measures 5 s more); there are
four sessions each with step latency and CPU seconds; and **no directory's page-cache residency grew by more than 1 GiB during the timed
phase** (`engine_ready` to the end of the last session). It means nothing about the machine's load or whether the arm is representative.

What the page-cache gate does not mean:

- **It gates growth only.** A fall is a NOTE. In `task1d-3-old-on-U`, `expert_resident_by_dir` fell by **8.2 GiB** on the source and **15.8 GiB** on
  the `/mnt/nvme4` mirror between the `session_0` and `session_1` samples (section 7.3), and the arm is VALID.
- **It samples once per session** (`fincore`, mincore only). Growth and fall inside one session are invisible.
- **Its independence claim rests on O_DIRECT, not on dropped caches**, because nobody here can drop them. Supporting evidence: whole-arm
  residency change was exactly 0 in 12 of the 17 arms (`.cache.json`, all directories; the rest are the boot-phase movers and `task1d-3` of section 7), and a `dd iflag=direct` test on both filesystems populated zero bytes
  (section 19; and `TOPOLOGY.md` section 7.5, re-run 2026-09-21). "The arm did not warm the cache" is measured; "the arm was
  independent of the cache" is inferred from O_DIRECT.
- **The gate changed during the series.** The first gate was whole-arm and rejected `task1-6` (+1.40 GiB) and `task1b-0` (+4.37 GiB), the first
  of which grew at **boot** (`+5,112.4 MiB at engine_ready`, `R/task1b-0-new-on-T.verdict.txt`; `task1-6`'s phase is unknown, it has only whole-arm residency). Commit `a4946b44f5` moved it to the timed phase and added a boot **regime** label. `task1b-0` then became valid
  (its timed growth was −0.63 GiB, `R/task1b-0-new-on-T.json` `expert_residency`); `task1-6` stayed invalid because it has no per-session
  residency to re-judge. One arm that once failed (`task1b-0`) is therefore in the reference set for a reason that postdates its run.
- **The drive-idle check is not "zero".** The script's 5 s check printed nvme4 at 0-267,059 B/s across arms (`R/task1*-run.out`; the largest
  in `task1d-2`), against a 1,048,576 B/s tolerance. Something reads that drive at up to 0.27 MB/s, which is negligible next to 3.5 GB/s.

### 4.2 The arm-internal OUTLIER note

A session 1-3 whose TTFT exceeds **1.25x the fastest TTFT among the other sessions 1-3** gets `OUTLIER session_i ... a disturbed session, not
gated`. It is a NOTE. **What "no OUTLIER note" does not mean:**

- **Session 0 is exempt** (it is cold), so a slow session 0 is never noted.
- **Only TTFT is compared.** A session that is slow in decode with a normal prefill is invisible. `task1d-0` session 3: TTFT 30.2 s (normal),
  decode 3.329 tok/s against 4.656 in the clean arm (**−28.5 %**), every step slower, p50 0.276 s against 0.206 s, `R/task1d-0-old-on-U.json`.
- **A uniform slowdown of the whole arm is invisible**, because the rule compares an arm's sessions with each other.

### 4.3 The cross-arm check

Pre-registered before any cross-arm computation (`R/crossarm-PREDICTIONS.txt`, written 2026-09-21T00:57:51-05:00): for an arm in its cell
(code x mirror, tracing ignored), each session is compared with the **median of the other reference arms of that cell (leave-one-out)** from
`R/clean-reference.json`; a session ≥ 8 % slower in decode (any session) or ≥ 8 % higher in TTFT (sessions 1-3) is flagged. A cell with no
other reference arm is "not judged". It is a NOTE, never a gate. **Its limits:**

- It is only as good as the reference set, which was chosen from arms already judged VALID and not disturbed **before the rule existed**
  (so the rule partly reproduces the judgement it was built from).
- It cannot see an arm as slow as its references, or any effect under about 8 %.
- With **one** reference arm (the whole old cell) a flag near the threshold is not distinguishable from that arm's own noise; the tool
  says so in the note. Every old-arm flag below carries that caveat.

### 4.4 The CONTENDED header

`CONTENDED contended` means a foreign process using ≥ 50 % CPU **last ran on cores 32-63** (the arm's cores) at a sampled boundary.
`unknown` means there are no samples, or the samples do not record the core. `not contended` means only that no such process was seen.
**It does not mean undisturbed.** The sample is one 0.2 s reading (`scripts/dsv41/provenance.py system_sample`) taken *after* each session
(`trace_corpus.py:217`), so six readings per arm at best, and none inside a timed stream. Recomputed for the arms that have samples:

| arm | CONTENDED (recomputed) | what the samples show |
|---|---|---|
| `task1d-0-old-on-U` | **not contended** | load1 1.9-2.4, largest foreign process `tmux: server` 37 % |
| `task1d-1-new-on-U` | **not contended** | load1 up to 5.34, largest `python` 29 %, `tmux` 41 % |
| `task1d-2-old-on-U` | unknown: busy foreign processes recorded without core | `nimbus_beacon_node` 100 % and 101 %, load1 up to 6.5 |
| `task1d-3-old-on-U` | unknown: same | `nimbus_beacon_node` ~100 % at every boundary, `op-reth` 107 %, load1 up to 8.02 |
| the other 13 arms | unknown: no boundary samples | none exist (section 8) |

**Core and affinity were not recorded in `task1d`'s samples** (added later by `3608384961`), so even the two "unknown" arms cannot be
called contended or not. The pre-registration expected "unknown for every arm recorded so far"; the tool prints `not contended` for
`task1d-0` and `task1d-1` because no process reached 50 %, so that expectation was too broad by two arms.

### 4.5 The two counter-examples, in our own data

**"Not contended" does not mean undisturbed, and "no OUTLIER note" does not mean clean: `task1d-0-old-on-U`.** It printed `VALID`, `CONTENDED not
contended`, no OUTLIER note. Its mean is 3.2339, **14.9 % below** the clean old arm's 3.7984. Session 0: 2.120 tok/s (−31.0 %), TTFT 72.1 s,
p50 step 0.424 s against 0.279 s; session 3: 3.329 tok/s (−28.5 %), TTFT normal. The recomputed cross-arm check flags both (single-reference
caveat). Its six boundary samples show load1 1.9-2.4 and nothing busy: **the cause of two sessions running 30 % slow is not recorded anywhere.**

**Its converse: `task1d-3-old-on-U`.** It carries an OUTLIER note and cross-arm flags, yet its mean decode is 3.7942, **−0.1 %** from the clean arm;
its only deviations are session 1's TTFT (45.1 s against 30.6 s, +47 %) and decode (−3.2 %). A note means "look", not "discard".

## 5. The old-barrier reference: one clean arm, not three

Five old-code arms exist (`099eadba33`). Their per-session decode against the one clean arm (`task1c-1-old-on-U`), `R/<arm>.json`:

| arm (start CDT) | mean tok/s | vs clean | s0 | s1 | s2 | s3 | notes (recomputed) |
|---|---:|---:|---:|---:|---:|---:|---|
| `task1c-1-U` (09-20 23:43) | **3.7984** | (ref) | 3.075 | 4.280 | 3.182 | 4.656 | VALID, no notes, **no boundary samples** |
| `task1c-4-U` (09-21 00:07) | 3.5545 | −6.4 % | −0.5 % | +0.1 % | **−19.6 %** | **−7.3 %** | OUTLIER s2 (TTFT 59.5 s), s3; **no boundary samples** |
| `task1d-0-U` (00:21) | 3.2339 | **−14.9 %** | **−31.0 %** | +0.4 % | +0.2 % | **−28.5 %** | VALID, "not contended", **no OUTLIER** (section 4.5) |
| `task1d-2-U` (00:36) | 3.6106 | −4.9 % | −5.7 % | −1.5 % | −1.0 % | −10.4 % | OUTLIER s3 (TTFT 41.7 s); `nimbus` 100 % |
| `task1d-3-U` (00:44) | 3.7942 | −0.1 % | +2.3 % | −3.2 % | +1.1 % | +0.3 % | OUTLIER s1 (TTFT 45.1 s); `op-reth`, `nimbus`; −24 GiB residency (section 7.3) |

So **four of five old arms were disturbed and are excluded; one is clean.** The corroboration that the ~3.8 level is right comes from the
sessions of those four arms that neither the arm-internal rule nor the cross-arm rule flagged: nine sessions, at −5.7, −1.5, −1.0, −0.5, +0.1,
+0.2, +0.3, +0.4, +2.3 % against the clean arm's values (median +0.1 %; eight of nine within −1.5 % / +2.3 %; the one at −5.7 % is
`task1d-2` session 0, below the 8 % threshold but with TTFT 71.3 s against 55.7 s, so it was probably disturbed as well). That is supporting
evidence that the clean arm is representative, **not a second clean measurement**, and "undisturbed" there means "not flagged by rules that are known
to miss things" (section 4). One earlier single shot, `DSV41_REFERENCE.md`'s native-mirror arm, measured 3.8016 on the same reader (`native-mirror/`,
2026-09-20 02:56-03:03 CDT); the committed text says "four days earlier", but that arm ran the same day as `task1c-1`, about 21 hours before it.
It is n=1, unprovenanced, and agreeing with it is weak evidence.

**More old arms are needed.** Until a second clean old arm exists, treat "+3.2 %" as "about 3 %, positive in all four sessions, old n=1".

## 6. The pre-registered predictions and how they came out

Files: `R/task1c-PREDICTIONS.txt` (2026-09-20 23:37:06-05:00, before task1c ran), `R/task1d-PREDICTIONS.txt` (2026-09-21 00:21:21-05:00, before
task1d), `R/crossarm-PREDICTIONS.txt` (00:57:51, before any cross-arm computation), `R/rebaseline-PREDICTIONS.txt` (01:07:11, before anything ran).
Outcomes below are from the arm files as they are now.

| id | prediction (abridged) | outcome | detail |
|---|---|---|---|
| P1 | in task1c, arms 2-5 have boot growth < 1 GiB in every directory (boot-warm); arm 1 likewise, lower confidence | **survived** | `task1c-0..4` boot growth −3.21, −2.56, 0, 0, 0 GiB (`R/*.regime.json`); all boot-warm. Weak: the prediction allows any fall |
| P2 | every arm's timed-phase growth < 1 GiB | **survived** | all of `task1c` and `task1d` VALID on the timed line; `task1b-0` −0.63 GiB |
| **P3** | new-code mirrors-on arms: boot-warm session-0 TTFT 59-61 s, boot-populated 49-51 s; falsified by a boot-warm arm < 55 s or a boot-populated one > 55 s | **FALSIFIED** | `task1c-0-new-on-T` is boot-warm (boot −3.21 GiB) with session-0 TTFT **50.8 s** |
| P4 | not a prediction (nvme2 read 6.2 GiB at boot in the populated arm against 7.7 in warm arms; more cache growth with fewer device bytes) | recorded | the eviction explanation it doubted was later withdrawn (section 7.2) |
| **P5** | (the team lead's, tested in task1d) pinned-allocation reclaim: where a directory's boot residency falls, `Cached` falls too; refuted by a > 256 MiB fall with `Cached` flat or rising | **UNTESTED** | no task1d arm's boot-phase residency fell (`P5 not testable in this arm` in all four verdicts); boot growth was 0 / +1.11 / 0 / +5.31 GiB |
| **P6** | clean old arms (no OUTLIER, gate valid) within ±1.5 % of 3.798 (3.741-3.855); clean new within ±1.5 % of 3.927 | **FALSIFIED** | `task1d-0` has no OUTLIER note, passes the gate, and is 3.234 (−14.9 %) |
| **P7** | at most 1 of the 4 task1d arms carries an OUTLIER note; falsified by ≥ 2 arms, or by an OUTLIER session with no foreign process ≥ 20 % and load1 < 4 at any boundary | **FALSIFIED** (first clause) | `task1d-1`, `-2`, `-3` all carry OUTLIER notes (**3 of 4**). The second clause did not fire: each has load1 ≥ 4 or a busy process at some boundary |
| R0 (crossarm) | six predicted hits flagged, ten predicted non-hits not flagged | **held**, recomputed | all six flagged (`task1c-3`, `-4`, `task1d-0..3`); none of the nine reference arms, `task1-6`, or `task1c-1` (not judged) flagged. `task1d-3` was expected to be flagged "only via TTFT of session 1"; it also has a +10 % TTFT flag on session 2 |
| R1 (re-baseline) | one gen2, mirrors-on, untraced arm on a quiet box, judged against `[3.905, 3.956]` (strong) or the 95 % interval `[3.856, 4.005]` (weak) | **not run** | no gen2 arm exists in `R/`; inconclusive if the arm is disturbed or CONTENDED is unknown |

Also on the record and not from these files: the native-mirror ceiling of 1.38x, recorded before that run, was met (1.3482x); the per-session
ratios in section 3.2 are 1.315-1.397x, two of them above 1.38x, but that ceiling was stated for mean decode.

**What each failure taught:**

- **P3.** The prediction file says the association came from "n=2 vs n=2 so far" without naming the arms; the arms that fit are `task1-2/-3` (59.8, 60.3 s) against `task1-6/task1b-0` (50.7, 49.8 s). It did not survive the third boot-warm
  arm. Whatever sets session-0 TTFT is not what the regime label records. Do not derive a rule from n=2 against n=2, and do not use the regime as a
  predictor of session 0.
- **P6.** "Clean" defined as "VALID and no OUTLIER note" admits an arm 14.9 % slow. A level band is only meaningful if the definition of clean can reject
  such an arm; that is why the cross-arm rule exists. **A band of ±1.5 % also assumed a quiet box**, which section 8 shows was never observed.
- **P7.** Disturbance is not an independent per-arm probability. Every arm that started at or after **2026-09-21 00:00:24 CDT** (`task1c-3`, `-4`, `task1d-0..3`, six arms) has flagged
  sessions; none of the eleven arms before it does (by the OUTLIER and cross-arm rules, whose reference set is those arms; start times in `R/*-run.out`; the eleven are `task1-0..6`, `task1b-0`, `task1c-0..2`). Something changed on
  the box around midnight. That is an observed clustering in time, with no cause recorded, and **the eleven earlier arms have no load records that could
  confirm they were quiet** (section 8). "0 of the 9 before" in P7's own text undercounts: there were eleven.

## 7. What is still unsettled

### 7.1 Session-0 TTFT

Mirrors on, session-0 TTFT across all arms that ran mirrors on, arm (regime, boot residency change): `task1b-0` 49.8 s (populated, +4.99 GiB), `task1-6` 50.7 s
(INVALID; regime not recorded), `task1c-0` 50.8 s (warm, −3.21), `task1d-3` 52.6 s (populated, +5.31), `task1c-1` old 55.7 s (warm, −2.56), `task1c-3` 59.3 s (warm, 0), `task1-2`
59.8 s and `task1-3` 60.3 s (regime not recorded), `task1c-4` old 59.7 s (warm, 0, disturbed), `task1d-1` 70.3 s (warm, 0, disturbed), `task1d-2` 71.3 s (warm, 0, disturbed),
`task1d-0` 72.1 s (populated, +1.11, disturbed). The **clean** on-cell spans 49.8-60.3 s (`clean-reference.json` arms). Sessions 1-3 are 29-31 s in every clean arm. **No explanation.** The
regime hypothesis (P3) is falsified; disturbance explains the 70-72 s values only by association, and `task1d-3` (52.6 s, disturbed in session 1, not session 0) shows a disturbed arm can have a
normal session 0. The mirrors-off cell does not show it: 94.5-99.4 s, sd 2 %.

### 7.2 Boot-phase page-cache residency

The source directory's residency changes across engine boot by **−3.21 GiB (`task1c-0`) to +5.31 GiB (`task1d-3`)** (`R/*.regime.json`
`boot_growth_bytes`; `task1b-0` +4.99 GiB from its json). These changes **do not track device reads**: in all four `task1d` arms the boot-phase device reads
on nvme2 and nvme0 are identical to the byte (7,807,332,352 B and 7,412,686,848 B, `R/task1d-*.verdict.txt` `BOOT_PHASE`) while the source directory's growth was 0, +1.11, 0 and +5.31 GiB.
The change at `engine_ready` also often reverses at `session_0` (`task1d-0`: +1,131.4 MiB then −1,131.4 MiB; `task1d-3`: +5,441.1 then −5,414.5 MiB). Section 19 quotes the range as "−3.4 to +5.1 GiB",
which I cannot match to one unit (−3.44 is the same change in GB; `task1b-0`'s +5,112.4 MiB is 4.99 GiB) and which omits `task1d-3`'s +5.31.

- **One explanation withdrawn:** that the boot re-read data from disk and evicted pages. `diskstats` contradicted it (P4: fewer device bytes with more cache growth).
- **One hypothesis untested:** the team lead's P5, that the 70 GiB pinned allocation reclaims page cache during boot. It could not be tested in `task1d`: no boot-phase fall occurred. `task1c-0` and `task1c-1`
  had boot-phase falls (−3.21, −2.56 GiB) and `Cached` fell across those arms too (whole-arm 83.1 → 79.7 GiB, 79.7 → 77.1 GiB, `R/task1c-{0,1}*.cache.json`), but those arms predate P5,
  have no boot-phase `meminfo`, and directory pages are part of `Cached`, so a fall in both is expected whatever the cause. They are not a test of P5 and I do not use them as support.

### 7.3 A new observation: the residency of both the source and the mirror collapsed inside a timed session

This is from the recorded files, and it was not in section 19.

- Through **every** arm from `task1b-0` to `task1d-2`, the `/mnt/nvme4` mirror's residency is a constant **32,287,563,776 B (30.07 GiB)** before, after and at every session
  sample (`R/*.cache.json` and `expert_residency`); `/mnt/nvme0`'s is 106,430,464 B (0.10 GiB).
- In `task1d-3-old-on-U` it falls to 14.40 GiB at the `session_1` sample and 14.28 GiB by `session_2` (**−15.8 GiB, −16,046 MiB at session 1**); the source directory falls from 15.44 to 7.23
  GiB at the same sample. `Cached` for the arm goes 79.6 → 72.2 GiB. Foreign load at that boundary: `op-reth` 106.6 %, `nimbus_beacon_node` 98.7 %, `reth-binary` 31.6 %, load1 5.73
  (`R/task1d-3-old-on-U.json` `boundary_samples[session_1]`). Separately, session 1's TTFT was 45.1 s against 31.4 s.
- Subsequent reads on divix01 (`TOPOLOGY.md` section 7.2): 15,335,759,872 B (14.28 GiB) on the mirror in the first read (05:57-06:03 UTC), **10.86 GB at 06:13 UTC**. The 15.34 GB
  figure is `task1d-3`'s recorded end-of-arm value **to the byte** (15,335,759,872 B; the source and `/mnt/nvme0` values also match to the byte), so the state did not change between
  that arm's end (00:51 CDT) and that read, and **the fall is one event inside one arm's timed phase**, not a gradual decline across a series of arms. Any account of the
  mirror's residency that assumes a gradual fall from the earlier "30 GiB" needs to explain that.
- **What it does and does not show.** It coincides with foreign memory and I/O activity, which is consistent with cache pressure from other processes evicting clean pages, but no cause is
  recorded. It happened in the timed phase, not at boot, so it neither supports nor tests P5. It does not invalidate an O_DIRECT arm, but it shows the page-cache gate cannot see falls and that
  **cache residency on both mirrors is time-varying: sample it before and after every arm, do not assume it.**

## 8. What was not monitored

- **13 of the 17 arms have no machine-load record.** Boundary samples (`boundary_samples` in the arm json: load average, meminfo, per-drive sectors, the five busiest foreign processes) exist for
  `task1d-0`, `-1`, `-2`, `-3` only, and only `system_sample` from `4626789547` on. `task1-0..6`, `task1b-0` and `task1c-0..4` have none. **That is thirteen, not the "earlier eleven":** `task1c-3` and
  `task1c-4`, the first two arms in which disturbed sessions were seen, were also never sampled. The earlier eleven were quiet as far as anyone knows, judged only by their run-to-run spread of 0.25-0.53 %
  (section 3.1), which contention does not usually permit; that is inference from the outcome, not observation of the condition.
- **The one old reference arm, `task1c-1`, is in the unmonitored group.** Its being clean rests on its consistency with the other (unmonitored) old-arm sessions and with the 3.8016 single shot.
- **Where samples exist they are sparse:** six 0.2 s readings per arm, between sessions, no core or affinity in `task1d`.
- The drives' idle check is at the start only.
- `TOPOLOGY.md` (section 7.3) notes that NVMe completion interrupts can land on cores 64-71 for submitters on some cores in 0-63; the arms run on cores 32-63, and no interrupt-placement record exists for any of them.

So "matched" here means matched in workload, seed, capacity, policy and code. It does not mean matched in machine load.

## 9. Using these baselines

1. **Compare like with like.** Same code generation (`git rev-parse <sha>:python` looked up in `clean-reference.json` `generations`), same cell, same 4 sessions, paired per session. An arm at gen2 (`5e92db22dc` or later) is a new series until R1 says otherwise.
2. **State `n` and monitoring beside every number.** The three cells are `n=5`, `n=4` and `n=1`; the old cell is unmonitored, and so are the eleven earliest arms.
3. **A new arm counts only if** it is VALID, has no OUTLIER note and no cross-arm flag, its CONTENDED header is `not contended` **and** it has boundary samples that record cores, and it was run while somebody who can see the machine says it is quiet. "Not contended" alone is not enough (section 4.5).
4. **Do not pool session-0 TTFT** and do not read the regime as a predictor of it.
5. **Do not quote the old-barrier gap to two decimals.** Until a second clean old arm exists, "about 3 %, all four sessions positive, old `n=1`, and the delta is four commits, not only two-bank".
6. **Record cache residency of both mirrors and the source before and after** each arm. It moves without our reads (section 7.3).

## 10. Corrections that section 19 of `DSV41_REFERENCE.md` needs

Against the committed text (`bea06e789c` is the last commit to touch it here; if the lead has newer uncommitted text the first two may already be fixed):

1. The table lists mirrors-on as **n=3** (3.905, 3.927, 3.933); `clean-reference.json` lists **four** (adds `task1b-0`, 3.956). The mirror effect is 1.347x with three, 1.351x with four.
2. "**Two old arms ran; one had disturbed sessions**" is stale: five old arms ran and four were disturbed or contended (the same section's later bullet says "four further"). Both cannot stand.
3. The 3.798 arm "measured on the same reader **four days earlier**" than 3.8016 is wrong: the two are about 21 hours apart (section 5).
4. "−3.4 to +5.1 GiB" boot-phase range mixes units and omits `task1d-3` (section 7.2); the values are −3.21 to +5.31 GiB.
5. "The earlier arms have no load record" understates: **thirteen** arms have none, including `task1c-3` and `-4`.
6. It does not mention that the arm outputs are not in git (only `clean-reference.json` and `task1e-PREDICTIONS.txt` are), nor `task1d-3`'s in-session residency collapse (section 7.3).

## Appendix A: arm files (sha256, first 12 hex digits, of `R/<arm>.json`)

Taken 2026-09-21 on divix01. If a file's hash differs from this list, it is not the file this document describes.

| arm | sha256[:12] | | arm | sha256[:12] |
|---|---|---|---|---|
| `task1-0-new-off-T` | `7d5268f6d027` | | `task1c-2-new-off-T` | `f76a9dd7b229` |
| `task1-1-new-off-T` | `af773b1051d8` | | `task1c-3-new-on-U` | `855812de8ccb` |
| `task1-2-new-on-T` | `ead7dcdfd879` | | `task1c-4-old-on-U` | `3002ecc166cf` |
| `task1-3-new-on-U` | `8906978de0b0` | | `task1d-0-old-on-U` | `3899a79705f1` |
| `task1-4-new-off-U` | `2bc71fb2f309` | | `task1d-1-new-on-U` | `6cc89d84c0b4` |
| `task1-5-new-off-U` | `79e55e2a2b57` | | `task1d-2-old-on-U` | `35cc4ebaaf02` |
| `task1-6-new-on-U` | `d9ab293ee913` | | `task1d-3-old-on-U` | `cea825a3a812` |
| `task1b-0-new-on-T` | `2b8c75243f7b` | | `clean-reference.json` | `0c16b04614e8` |
| `task1c-0-new-on-T` | `b591a1d9920d` | | `crossarm-PREDICTIONS.txt` | `0ce65ca8b908` |
| `task1c-1-old-on-U` | `05aaec5fa160` | | `rebaseline-PREDICTIONS.txt` | `d0717c3ac879` |

`task1c-PREDICTIONS.txt` `bc9342263734`, `task1d-PREDICTIONS.txt` `b22b35e42946`.

Arm start times, CDT (`R/*-run.out` `ARM` lines) and whole-arm wall seconds (`.cache.json` `wall_s`): `task1-0` 09-20 13:10:12, 548; `task1-1` 13:19:32, 546; `task1-2` 13:28:49, 382; `task1-3` 13:35:23, 385;
`task1-4` 13:41:59, 553; `task1-5` 13:51:24, 545; `task1-6` 14:00:40, 367; `task1b-0` 23:24:27, 382; `task1c-0` 23:37:06, 383; `task1c-1` 23:43:40, 414; `task1c-2` 23:50:46, 567; `task1c-3` 09-21 00:00:24, 419;
`task1c-4` 00:07:34, 457; `task1d-0` 00:21:21, 446; `task1d-1` 00:28:58, 442; `task1d-2` 00:36:31, 459; `task1d-3` 00:44:22, 416. Clean mirrors-on new arms took 382-385 s; every flagged arm took 416-459 s.

## Appendix B: how the numbers were recomputed

Everything above except the quoted verdict lines is a computation over `R/*.json` and `R/*.cache.json` with the standard library, run on a copy of those files (`tar` of the small files; nothing was
run on divix01 but `sha256sum`, `ls` and `cat`). The recomputed CONTENDED, OUTLIER and CROSS-ARM lines come from importing `analysis/dsv41-drive/task1_arm_verdict.py` and calling
`contention(report)`, `session_outliers(report)` and `cross_arm_outliers(report, path, references)` with `references` taken from `R/clean-reference.json` (paths remapped to the copy).

```python
import json, statistics as st
L = lambda a: json.load(open(f"{a}.json"))
cells = {"new:off": ["task1-0-new-off-T", "task1-1-new-off-T", "task1-4-new-off-U", "task1-5-new-off-U", "task1c-2-new-off-T"],
         "new:on":  ["task1-2-new-on-T", "task1-3-new-on-U", "task1b-0-new-on-T", "task1c-0-new-on-T"],
         "old:on":  ["task1c-1-old-on-U"]}
means = {c: [L(a)["mean_decode_tok_s"] for a in arms] for c, arms in cells.items()}
print(st.mean(means["new:on"]) / st.mean(means["new:off"]))            # 1.3505 (four on-arms)
print(st.mean(means["new:on"]) / means["old:on"][0])                   # 1.0347
# per session: [L(a)["per_session"][i]["decode_tok_s"] for a in arm list], and ["ttft_s"]; step latency: ["step_latency"]["step_s_p50" | "step_s_p95" | "step_s_p99"]
# residency: L(a)["expert_residency"]["before_engine" | "after_engine_ready"], per_session[i]["expert_resident_bytes"], boundary_samples[k]
```
