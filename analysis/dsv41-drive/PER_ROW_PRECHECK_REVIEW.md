# Independent recomputation of the Task 6 precheck arithmetic, 2026-09-21

Subject: `PER_ROW_TRANSFER.md` section 1 (read at `116533cb61`; the paragraphs cited below are unchanged at `0fc5640141`),
`PER_ROW_PRECHECK_PREREG.txt` (sha256 `0c145def...`), `per_row_precheck.py` (`e81b8b60...`), `per_row_precheck_result.txt`.
Method: I wrote my own code from the model definition (pre-registration plus section 1.1) and ran it on the raw traces in
`task1-results/` on divix01, `taskset -c 3/4/5`, `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=` (no GPU, CPU only). I did **not**
run or import `per_row_precheck.py`; I read it only after my numbers existed. Read-only: `PER_ROW_TRANSFER.md` and the
precheck files were not edited. Scripts: `per_row_recompute/` (Appendix). Serena was unavailable; grep and reading.

Traces used: mirrors-on `task1-2-new-on-T`, `task1b-0`, `task1c-0`; mirrors-off `task1c-2-new-off-T`; `task1-3-new-on-U` for
per-session step times.

## Verdict

**Everything in the precheck's headline arithmetic reproduces from the traces.** Nothing I found changes the classification
(SUPPORT, SPREAD-IRRELEVANT) or the redirect (two-phase before per-row). I found one misstated comparison (R1), one claim
that the data can now test rather than assume (R2), one explanation that is right for a different reason than stated (R3),
one independence statement to tighten (R4), two small denominator notes (R5), and, found later, a 3-7% overstatement of every absolute ms/step figure because 16 of the 495 trace lines cover two decode steps (R6). Nothing here is HIGH, and no verdict changes.

## What reproduces

| Quantity | Document / registered | My recomputation | Match |
|---|---:|---:|---|
| BEST, ms/step (share of 257.5) | 38.6 (15.0%) | 38.61 (15.0%) | yes |
| RANDOM, ms/step | 19.2 (Monte Carlo, 64 perms) | **19.17** exact (all permutations per request); 7.45% | yes |
| RANDOM, miss rows only | 2.55 | 2.55 | yes |
| GENEROUS (6 lanes, best order) | 75.4 (29%) | 75.4 (29.3%) | yes |
| miss-only / RANDOM | 0.133 | 0.133 | yes |
| (RANDOM - 1 ms launch) / T_step | 7.1% | 7.06% | yes |
| best-order miss-row saving `Sigma(m-1)*c/495` | 5.06 | 5.09 by simulation | 0.6%; the simulation's DONE can exceed the last row's `R`, the arithmetic assumes equality |
| spread of miss-row readiness, p50 | 2.7 ms, > c | 2.77 ms; fraction >= c = 1.0 | yes |
| hide_ok | 94% | 0.942; read-wait p50 5.29 ms | yes |
| 87% / 13% split | 87% | 86.8% (two-phase / best = 33.53 / 38.61) and 13.3% (2.55 / 19.17) | yes, both ways |
| closed form `S = min_j[(M - r_j) + (j-1)c]` | section 2 | 0 mismatches in 90,048 random cases; `S >= 0` always; `S = 0` when the slowest lane is first; `E[S] = (k-1)c/2` for one miss lane among k; random/best = 0.5 for m = 2, 3 and 0.488 for m = 4 | yes |
| A1 floor: SUPPORT holds down to ~37% of modelled hit lanes | 37%, ~0.8 per read layer | f = 0.369 (11.7 hit lanes per step; 0.81 per read request), registered random-order criterion | yes |
| Mirrors-off gives the same classes | yes | yes (task1c-2) | yes |
| m histogram, 7,139 read requests, 495 steps, 1,842 filtered m>=2 requests | as stated | as stated; the doc's 1,888 is m>=2 without the `graph_step` filter | yes |

"Exact permutation average" is what I computed; the registered Monte Carlo (64 permutations) gave 19.2 against my 19.17, so the
sampling error is negligible.

## Findings

### R1 (medium, misstated comparison): "2.55 ms if lane order is random" is not per-row's increment over two-phase

Section 0 and section 6.1 write: what per-row adds over two-phase "is at most 5.06 ms per decode step, 2.0%, and 2.55 ms, 1.0%,
if lane order is random." 5.06 is right (I get 5.09): it is best-order per-row minus two-phase, 38.61 - 33.53.
2.55 is a different quantity: the random-order **miss-only** figure, per-row with no hit lanes. Per-row **with** hit lanes
at random order is 19.17, and two-phase is 33.53 regardless of order (hits copy first by construction). So per-row over two-phase is

- **+5.09 ms at best order**, and
- **-14.35 ms at random order** (per-row loses to two-phase by 14.35 ms/step, 5.6%).

The conclusion is unchanged, and stronger: per-row does not merely lose half its increment under random order, it loses to
two-phase outright. It needs the order array to break even, which is the document's own point about the order array, but the
number that supports the point is -14.35, not 2.55. The 1.0% belongs where it is used as "the random-order miss-only share" (section
1.2), not as a per-row-over-two-phase bound. Suggested rewording of the two places: "+5.09 ms (2.0%) at best order; at random
order per-row is 14.4 ms *worse* than two-phase".

### R2 (medium, an assumption the trace can bound): A1 does not prop V1 up, but it does decide the per-row column

The document says A1 (hit lanes spread evenly over layers) carries the result and "is not excluded by any data I have"
(26 non-read layers can hold 156 lanes). The trace does contain enough to bound it without A1: `vram_miss` per step,
`layer_ram_rows`, and the per-layer cap of 6 lanes.

- Total hit lanes per step: 111.9. A1 credits 31.9 of them to layers that read (so ~72% of hit lanes sit in layers that read
  nothing and get no benefit under A1 itself; the doc's 37% floor is 37% of these 31.9, that is 11.7 per step).
- A1-free data bounds: the data force at least **10.8** hit lanes per step into read layers (f about 0.34 of the 31.9), and at most
  63.6.
- At that worst-case placement, **two-phase is about 4%** (clears the 3% bar). **Random-order per-row is about 2.8%**, just
  under the registered 3%.
- Two-phase floors (my computation, not registered): >= 3% net of launch cost needs f = 0.26 (8.3 hit lanes per step); >= 1.5%
  needs f = 0.145 (4.6 per step). Both are below the data-forced minimum of 10.8.

So V1's justification survives the worst A1-free placement; the per-row column can fall just below the bar in that same
placement. That is consistent with the document's redirect and adds a margin the document did not have. It does not remove
[REQ 4]: these are bounds, the real placement is unknown. `4e63616666` ("record the planned lane count per request; trace schema 4", an ancestor of `HEAD`) is the [REQ 4] instrumentation; a schema-4 trace would replace
these bounds by a measurement (section "Schema 4" below). The eager per-layer lines in the older traces (200 of them) are no help as an A1 analogue:
they are prefill-regime and all have `ram_miss > 0`.

### R3 (low/medium, right conclusion, different reason): the "pack order equals ordinal order" result comes from drive FIFO, not the tie-break

Section 2 warns that the result should be read as a property of the packer until it survives a changed tie-break. I tested it
against the row completion stamps (`extent_cqe_ns`, max over parts):

- **0 inversions** in 1,842 m>=2 requests, in each of the three mirrors-on arms and in the mirrors-off arm; every request has at
  least two distinct stamps.
- Consecutive rows on one drive part completed out of order in **0 of 4,750** pairs.
- **Same-reap ties** (rows completing in the same reaped batch, where `pack_one`'s lowest-ordinal tie-break decides): 83 of 1,842
  requests (4.5%), 87 of 2,375 adjacent pairs (3.7%) mirrors-on; **0%** mirrors-off.

So the perfect ordering is what per-drive FIFO completion gives; the tie-break is not what produces it, and could change the
order of at most about 4.5% of requests even in the worst case. This supports the document's conclusion (ordinal order is
lane order) and tells the reader which caution matters: it depends on FIFO per drive part, which the design should treat as an
assumption about `RowReader`/io_uring completion, not as a property of the corpus. It also refines the `MIRROR_ROWS.md` reading:
a ~3 ms row tail means a slow row **delays its followers** (bunching), not reordering them. Bunching hurts per-row (later rows
arrive together and the chain restarts) but leaves the order array right.

### R4 (low, independence): "three arms agree to 0.1 ms" and "all three arms, same class" are timing agreement only

The document already says the seven files are one request stream, and so does prereg limit 5. Two phrasings still read as if
the arms were replicates. I confirmed: `rows_asked`, `vram_miss` and `ram_miss` sequences are **identical across all four arms,
mirrors-off included**. n_eff for the workload is 1, and the agreement between arms is agreement of timing on that stream. The
decision rule's "in all three arms" is therefore a robustness check of the timing, not three votes on the workload.

Within the stream there is a large range across the four sessions (same run; step times from `task1-3-new-on-U`: 313.4, 226.3,
300.0, 209.9 ms):

| Session | RANDOM ms/step (share of that session's step) | two-phase ms/step (share) |
|---|---:|---:|
| 1 | 27.15 (8.7%) | 48.68 (15.5%) |
| 2 | 12.97 (5.7%) | 22.25 (9.8%) |
| 3 | 25.47 (8.5%) | 43.71 (14.6%) |
| 4 | 11.17 (5.3%) | 19.59 (9.3%) |

The headline 7.5% hides a 2.4x range across sessions. Every session clears the 3% bar, so SUPPORT stands; a session-level range
should sit beside the headline.

### R5 (low): denominators

- **T_step = 257.5 ms** is built from untraced arms including `task1-6` (marked INVALID in the baseline) and `task1c-3`
  (disturbed). The clean untraced on-arms give **254.4 ms**, a 1.2% relative shift in every percentage: immaterial to every
  verdict.
- The "1.5% the plan's design can resolve" comes from `task1e`'s quiet-box standard deviation. That series is recorded
  UNRESOLVED (contended box), so the resolution on the real box is worse than 1.5%. This sharpens the document's point that the
  per-row-over-two-phase increment (2.0% at best, negative at random) is below what the gate can see.

### R6 (medium, a systematic overstatement I found while chasing the "extra ~2 requests per step"): 495 graph_step lines are 511 decode steps

Found after the first version of this review, on the same traces, by explaining why 20,800 request lines are 42.0 per graph_step against 40 layers.
Sixteen of the 495 `graph_step` lines are merged: `steps == 2`, `routed_rows == 480` (= 2 x 240 for the other 479 lines), `vram_miss` 266.5 on average against
126.5, and the request lines filed under those 16 `forward` ids number 65-81 (about 2 x 40) instead of 38-41. `_trace_step` says why: a lagged register read leaves one
step for the next line and the line counts `steps = round(routed / routed_rows_per_step)`. So the trace holds **511 decode steps in 495 lines** (the client
saw 4 x 127 = 508; I do not know the 3-step gap). The rest of the residual is explained too: four request groups filed under forwards 2, 126, 251, 376 that have
no `graph_step` (374 + 14 + 15 + 15 = 418 lines; one per session, at a 125-forward stride) and the +-1 lag at ordinary boundaries (38-41 per forward). 20,382
requests over 511 steps is 39.9 per step.

Two errors follow in the precheck, both in the direction of a larger saving:
1. **The denominator.** Savings are summed over requests from 511 steps and divided by 495. Every ms/step figure is 3.2% too large (511/495).
2. **The lane count on the 16 merged lines.** `k = round(vram_miss / 40)` uses a two-step `vram_miss` (about 6.7, capped at 6) for each of ~1,280 requests where the truth is about 3.3.
   That over-credits hit lanes on about 6% of requests.

Recomputed with `vram_miss / steps` for `k` and 511 as the denominator (`merged_steps.py`; the four arms agree to 0.1 ms):

| ms per decode step | as registered (/495, k per line) | denominator fixed only (/511) | both fixed |
|---|---:|---:|---:|
| BEST | 38.61 | 37.40 | **35.80** |
| RANDOM | 19.17 | 18.57 | **17.79** |
| two-phase | 33.53 | 32.48 | **30.88** |
| per-row over two-phase, best order | +5.09 | +4.93 | **+4.93** |
| per-row over two-phase, random order | -14.35 | -13.90 | **-13.08** |
| hit lanes credited per step | 31.9 | 30.9 | **29.3** |

Against 257.5 ms: BEST 13.9%, RANDOM 6.9%, two-phase 12.0% (registered 15.0 / 7.5 / 13.0). Miss-only RANDOM 2.55 -> 2.46; `Sigma(m-1)*c` over 511 steps is 4.90, not 5.06.
Total hit lanes per step (R2) 111.9 -> 108.4 and `vram_miss` per step 131 -> 126.9 (nearer 18.2's 121). **No verdict changes**: RANDOM minus 1 ms is 6.5% against the 3% bar,
two-phase clears it, per-row over two-phase is still +4.9 ms best and -13.1 ms random, miss-only/RANDOM stays about 0.13, and the 87/13 split is a ratio of two
quantities that scale together. But every absolute figure in `PER_ROW_TRANSFER.md` section 1 (and everything quoted from it) is 3-7% high, and
**the R2 bounds (10.8 to 63.6, worst-case two-phase about 4%) were computed per 495 lines and are slightly loose in the same direction**. I have not redone them with 511.
Also note that 495 appears as a per-step count in `per_row_precheck.py` itself, not only in the document.

For the schema-4 analysis this matters twice: `schema4_lanes.py` now divides by `sum(steps)`, and the sum check of lanes against `vram_miss` is unaffected (sums), but any per-step
grouping must use the merged lines' `steps`.

## Not recomputed or not verifiable from the traces

- **A2** (hit lanes ready at `reserved`): it is an assumption because the signal does not exist today (the document says so).
- **c = 1.055 ms** (from 18.2, a node-mode trace of an older tree) and the **launch cost of 1.0 ms** (assumed): I used them, I did
  not measure them. The three-value c sweep in 1.2 is scaling I only spot-checked (c = 1.055 exactly).
- The **93 ms/step read-wait sum** in the "no fitting" consistency check: I reproduced the p50 (5.29 ms) but not the sum's
  provenance, and I did not redo the 131 x 1.055 + 93 + 17 = 248 ms addition against the step time.
- Sections 3 to 9 (design, lease, stages, cost, requests): not part of the arithmetic and not reviewed here; see
  `PER_ROW_TRANSFER_REVIEW.md` for that.
- Traces beyond the four arms above (three mirrors-on + one mirrors-off, plus `task1-3-new-on-U` for sessions).

## Schema 4 (`4e63616666`): does it replace R2's bounds with a measurement?

**Yes, and it replaces more than R2.** Read at the source (diff of `4e63616666`, plus `exl3_stream_trace.py` and `exl3_ram_miss.py` at `HEAD`); not
run (no GPU, and no schema-4 trace exists yet, so nothing below has met real data).

What it records. The post kernel stores `count[0]` (the plan's lane count, unclamped) at offset 84 of every demand record; the service copies
it into the stage record (`cur_->lanes`, in `serve()` and in the touch path) and the trace line gets `request.lanes`. The line already carries `layer`.
Per request, **hit lanes = `lanes` - `rows_asked`**, and `lanes` is the planned lanes, so it also replaces the precheck's per-request
`k = min(6, max(m, round(vram_miss/40)))` estimate. That estimate fed every headline figure, not only the A1 bound: 19.17, 33.5 and 38.6 can all be
recomputed with the measured `k` and no A1 and no cap at 6.

Why every layer is covered, checked on the existing schema-3 arm `task1-2-new-on-T`: its 20,800 request lines are 7,211 read demands plus **13,589
`touch` records** (unarmed, no rows read); there are no advisory or no-read lines in that arm. Touch records carry `lanes` too (`cur_->lanes`
in the touch path), so lanes in layers that read nothing are `sum(lanes)` over touches, and lanes in read layers are `sum(lanes - rows_asked)` over
demands. `dropped_before` is 0 on every line. That gives lanes per layer directly. **One number replaces the two bounds:** hit lanes per step in
read layers, against the A1 credit of 31.9 and the data-forced range 10.8-63.6.

Caveats I can already see:
- **The sum check has a tolerance, not equality.** `_trace_step` says a line may pair one step's demand rows with the previous step's. On the schema-3
  trace, `sum(rows_asked)` over decode requests is 9,514 against `sum(ram_miss)` over `graph_step` lines 9,462 (0.5%). Expect `sum(lanes)` against
  `sum(vram_miss)` to agree to about that, not exactly. A large gap would mean `lanes` is not what the commit says.
- **Request-to-step grouping is approximate.** 20,800 requests over 495 steps is 42.0 per step against 40 layers, and 418 lines carry a `forward` with
  no `graph_step`. I do not know what the extra ~2 per step are (warm-up or a boundary effect); the schema-4 analysis restricts to lines whose `forward`
  has a `graph_step`, as the precheck did, and the first trace should reconcile the count.
- **`hits = lanes - rows_asked` assumes `rows_asked` counts every planned lane the service was asked for.** The post kernel requests `min(count, 8)` lanes;
  `attach` refuses `graph_gather_rows > 8` (`8c78749b35`), so `count <= 8` holds in a config that attaches. I check `lanes >= rows_asked` on every request.
- `lanes` up to 8 (histogram will show) means RANDOM at `k > 6` cannot be averaged over all permutations (8! = 40,320 per request); the script samples 64.
- It does not touch A2 (hit lanes ready at `reserved`); there is still no early signal.
- Arms at code before `4e63616666` are not comparable in timing (the commit says so); only placement is wanted here.

**What run is needed.** One graph-decode arm with tracing on at `HEAD` >= `4e63616666`, same prompts and configuration as `task1-2-new-on-T` (mirrors on),
whose request stream is deterministic (identical `rows_asked`/`vram_miss` sequences in all four arms I compared, mirrors-off included), so the
placement it records is this corpus's placement. Only placement is wanted, so a contended box does not matter for this question, and a partial
run gives the placement of the prefix; the full 495 steps allows comparison with the totals above. It needs GPU time scheduled through the usual owner,
and a generation entry in `clean-reference.json` first (the arm script refuses an unregistered generation, `e1d1621394`); the tree's `python/` at
`HEAD` is not in the manifest.

**Script, ready now.** `per_row_recompute/schema4_lanes.py <trace>` refuses a non-schema-4 trace; otherwise it prints the sum checks, hit lanes per step
in read layers and in layers that read nothing, the lane histogram, the `lanes >= rows_asked` check, and BEST / RANDOM / two-phase with the measured `k`
next to the A1-model figures. **Plumbing test only**: run against a copy of `task1-2-new-on-T.trace` with `lanes` injected from the A1 model (a synthetic
file, not a measurement). It reproduced 38.61 / 19.17 / 33.53 ms per step, +5.09 / -14.35 per-row over two-phase, and 31.9 hit lanes per step in
read layers (the A1 credit), and it printed a sum-check mismatch on the synthetic data (68,832 lanes against 64,857 `vram_miss`), so the check can fail. On the real
schema-3 trace it exits with "not a schema-4 trace". It says nothing about the real placement.

### Result: `task1f-0-new-on-T` (schema 4, gen4, mirrors on, 495 graph_step lines = 511 decode steps)

Reported in the order the lead asked for. Everything below divides by `sum(steps)` (511). `task1f` ran under accepted CPU contention (`task1f-PURPOSE.txt`), so it answers
a placement question and **no timing from it is quoted**; see the note on timing at the end.

**1. The measured placement is inside the bounds, and not near an edge.** Hit lanes per decode step in layers that read: **33.9**. The bounds from R2 were 10.8 to 63.6
computed per 495 lines; scaled to 511 they are about 10.5 to 61.6 (scaled, not recomputed, so I have not moved an edge by recomputation, but 33.9 is 46% of the way across and
no edge is within 20 lanes of it). The prediction that the placement lies in the data-forced range held. It also lies well above the registered SUPPORT floor (37% of the A1
credit, 11.3 per step at 511) at **3.0x** that floor, and above A1's own credit (31.9 per 495 lines, 30.9 per 511 steps, 29.3 with `k` from `vram_miss` per step).
Hit lanes in layers that read nothing: 73.9 per step; total 107.9 (R2 estimated 108.4 at 511). So 31% of hit lanes sit in read layers, against the 29% A1 credited. A1's "spread
evenly over layers" is good at the layer level: lanes per layer per step range 2.4 to 4.1 (mean 3.16), first ten layers 3.07 against last ten 3.17.

**2. Checks.**
- The trace is schema 4: all 20,800 request lines carry `schema` 4 and `request.lanes`; `dropped_before` is 0 on every line.
- `sum(lanes)` over decode requests (those whose `forward` has a `graph_step`) is **64,635** against `sum(vram_miss)` **64,857**: -0.34%, inside the 0.5% tolerance and the same order as the
  schema-3 `rows_asked` against `ram_miss` gap (9,514 against 9,462, identical here). Over all 20,800 requests it is 66,875 (+3.1%): the 2,240 lanes filed under forwards 2, 126, 251, 376
  belong to steps with no `graph_step` line, so **the filter matters**. I did not explain the remaining -222 lanes (0.34%); a lag at the four session starts is the candidate, not shown.
- `lanes >= rows_asked` on every read request (0 violations); max `lanes` is 6, so the model's cap of 6 was never binding and exact permutation averaging is valid.
- **Line by line the sum does not hold**: `sum(lanes)` per `forward` equals that line's `vram_miss` in 16 of 495 lines (sd of the difference 10.6; merged lines average -16.6). Per-step grouping through
  `forward` is unreliable, as `_trace_step` says; only run sums are used here.
- Same stream as `task1-2-new-on-T`: `(layer, type, rows_asked, status)` identical for all 20,800 requests in order; 16 requests differ only in `forward` label and 4 `graph_step` lines shift a few
  RAM misses between adjacent steps (the register lag), with equal sums. That is what allows the next item.

**3. Recomputed figures with measured `k`** (`schema4_join.py`: measured lanes from `task1f` joined request by request onto the stage timings of `task1-2-new-on-T`; ms per decode step, shares of 257.5 ms):

| k | BEST | RANDOM | two-phase | hit lanes in read layers | per-row over two-phase, best / random order |
|---|---:|---:|---:|---:|---:|
| A1 as registered (per 495 lines, for reference) | 38.61 (15.0%) | 19.17 (7.4%) | 33.53 (13.0%) | 31.9 | +5.09 / -14.35 |
| A1 with the 511-step denominator (R6, denominator only) | 37.40 (14.5%) | 18.57 (7.2%) | 32.48 (12.6%) | 30.9 | +4.93 / -13.90 |
| A1 corrected (R6, both) | 35.80 (13.9%) | 17.79 (6.9%) | 30.88 (12.0%) | 29.3 | +4.93 / -13.08 |
| **measured** | **40.67 (15.8%)** | **20.07 (7.8%)** | **35.74 (13.9%)** | **33.9** | **+4.93 / -15.67** |

The measurement is 13-16% above the A1-corrected figures and 5-7% above the registered ones (R6's overstatement and A1's understatement happen to nearly cancel against the original numbers, which
is luck, not agreement). Verdicts: SUPPORT holds with more margin ((RANDOM - 1 ms launch) / T = 7.4%); two-phase 13.9%; per-row over two-phase is +4.93 ms (1.9%) at best order and **-15.67 ms at random
order**, so R1 gets stronger; the 87/13 split becomes 87.9% (two-phase / best = 35.74 / 40.67), essentially unchanged. The per-row-over-two-phase best-order increment does not depend on `k` at all (it
is `Sigma(m-1)*c` over steps, 4.93), which is why it is the same in every row.

Where A1 was wrong: not in the layer spread but per request. Against measured `k` on 7,139 read requests, A1's `k` (with `vram_miss` per step) is exact for 36%, too low for 42%, too high for 22%; mean absolute error 0.88 lanes.
A1 assigns every layer in a step the step's average, while the real lane count varies by layer and request (range 1 to 6); the errors mostly cancel in the sum but not to zero, and the net is a 14% undercount of hit lanes in read layers.

**Timing.** `k` is content (a lane count) and does not depend on step duration. The BEST / RANDOM / two-phase figures are timing-derived, because they use the stage stamps (`observed`, `reserved`, `done`, row pack ends). I therefore took
those stamps from `task1-2-new-on-T` and only the lanes from `task1f`. As a check on contamination the same model on `task1f`'s own stamps gives 40.73 / 20.13 / 35.79: within 0.06 ms of the joined figures, so the contended stamps
do not change the model here; I still quote the joined ones. The figures inherit everything else the precheck assumes: A2 (an early readiness signal that does not exist), `c = 1.055`, the 1.0 ms launch cost, one workload (n_eff = 1).

## Appendix: scripts (`per_row_recompute/`)

Run from divix01 with the directory copied over; each prints the numbers above. `D` in `core.py` is the hard-coded results
directory on divix01.

- `core.py`: trace loading, request model, per-request BEST / RANDOM (exact permutations) / two-phase / closed form.
- `closed_form.py`: 90,048 random cases against `S = min_j[...]`.
- `a1_floor.py`: A1 floor and the two-phase floors; A1-free bounds from `layer_ram_rows` and `vram_miss`.
- `ordering_sessions_identity.py`: completion-stamp inversions, per-drive FIFO check, per-session dispersion, cross-arm identity of
  `rows_asked` / `vram_miss` / `ram_miss`.
- `spread_hide_tstep.py`: spread p50, hide_ok, read-wait, T_step variants.
- `reap_ties.py`: same-reap tie counts (mirrors-on and mirrors-off).
- `merged_steps.py`: R6, the merged `graph_step` lines and the recomputation with `steps` (four arms).
- `schema4_join.py`: the schema-4 result (measured lanes joined onto `task1-2` stamps; k error; lanes per layer). Run after `schema4_lanes.py`.
- `schema4_lanes.py`: the measured-`k` version for a schema-4 trace (see the Schema 4 section; plumbing-tested on a synthetic file only).

The core function, for reference:

```python
def per_req(t0, res, done, R, k, c):
    m = len(R); h = k - m
    ready = [res]*h + list(R)
    tb = done + k*c
    best_order = sorted(range(k), key=lambda i:(ready[i], i))
    best = tb - chain(best_order, ready, c, t0)
    perms = list(itertools.permutations(range(k)))
    rand = tb - sum(chain(p, ready, c, t0) for p in perms)/len(perms)
    hit_end = (max(t0,res)+h*c) if h>0 else 0.0
    two = tb - (max(done, hit_end) + m*c)
    cf = min((done - rs[j]) + j*c for j in range(k))   # closed form
    return best, rand, two, cf
```
