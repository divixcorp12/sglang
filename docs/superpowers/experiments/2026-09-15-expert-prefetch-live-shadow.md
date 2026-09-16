# Expert prefetch live shadow validation (Phase A, T6)

Record of the live shadow run only. No go/no-go verdict — see Limitations and Open questions.

## Setup

- **Arms and pinned commits.**
  - `REF`: detached worktree `/data/models/slang/nvfp4-work/cc-expert-prediction/ref-7de955`, HEAD `7de955329a8f41ac336167dcd6fe4e007fc55f2c`. That worktree's checked-out copy of `scripts/expert_prediction/run-shadow-server.sh` was replaced with the branch's Task 5 launcher (`git show d725c54a31:...`) so REF and the S-arms share identical launch flags, then hand-edited with exactly one **uncommitted** line change so REF actually serves its own code instead of the S-arm tree:
    ```diff
    - worktree=$work/cc-expert-prediction/worktree
    + worktree=$work/cc-expert-prediction/ref-7de955
    ```
    That edit lives only in the `ref-7de955` worktree's working copy; the tracked launcher in the main branch worktree was never touched.
  - `S0`/`S1`/`S2`: `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree`, HEAD `d725c54a31bc3bf77d34cb3d555b170297f06da0` for every launch. `S0`: `PREFETCH_PREDICTOR=off`. `S1`: `PREFETCH_PREDICTOR=llapor`. `S2`: `PREFETCH_PREDICTOR=apex`.
- **Budget:** `PREFETCH_BUDGET=2` for S1/S2 (T3's best cell).
- **HOT_GPU_MB:** left at the launcher default (12288) for every cell, equal across arms.
- **Session subset:** `/mnt/nvme2/nvfp4-work/benchmarks/prefetch-shadow/sessions.jsonl`, built by `select_ab_sessions.py --holdout 2 --val 6` from `/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl` — 8 sessions, 29 turns.
- **Fixed warm-up session** (identical every cell, every pass, never varied): `cfq-train-Single_CDW/2015/page_35.pdf-2` — first session in `sessions.jsonl` file order not in the 8-session subset.
- **Token caps:** logprob probe 8 sessions × 96 tokens (`logprob_probe.py --prompts 8 --max-tokens 96`); full capture 768 tokens (`run_capture_sessions.py --max-tokens 768`), same 8-session subset, 29 turns.
- **Cells run:** REF-p1, REF-p2 (logprobs only, no full capture — REF-p2 was a determinism control), S0-p1 (full run split across two launches: logprob probe on the first launch, full 768-token capture on a second launch after a gap was found), S1-p1, S1-p2, S2-p1, S2-p2.

## tok/s

Median decode tok/s over turns with `completion_tokens >= 64`, n=29 turns per arm/pass.

| arm/pass | median tok/s | ms/token | Δ vs S0 (ms/token) |
|---|---|---|---|
| REF-p1 | 15.99 | 62.54 | +0.71 |
| S0-p1 | 16.17 | 61.83 | — (baseline) |
| S1-p1 (llapor) | 15.46 | 64.68 | +2.85 |
| S1-p2 (llapor) | 15.83 | 63.17 | +1.34 |
| S2-p1 (apex) | 15.38 | 65.03 | +3.20 |
| S2-p2 (apex) | 15.35 | 65.15 | +3.32 |

S1−S0 scoring-cost samples: {2.85, 1.34}, mean ≈2.10. S2−S0 scoring-cost samples: {3.20, 3.32}, mean ≈3.26.

## Live budget_recall vs T3 dev split (B=2)

| arm/pass | measured | T3 dev split | Δ |
|---|---|---|---|
| S1-p1 (llapor) | 0.383 | 0.342 | +0.041 |
| S1-p2 (llapor) | 0.389 | 0.342 | +0.047 |
| S2-p1 (apex) | 0.424 | 0.409 | +0.015 |
| S2-p2 (apex) | 0.420 | 0.409 | +0.011 |

All four within ±0.05 of T3's dev split.

## --exact flip counts (`compare_logprobs.py --exact`, 8-session/96-token probe)

**Same-arm (two passes of the identical arm):**

| pair | flips |
|---|---|
| REF-p1 vs REF-p2 | 5 |
| S1-p1 vs S1-p2 | 3 |
| S2-p1 vs S2-p2 | 3 |

Spread: 3-5. **S0 has only one pass — no same-arm baseline is available for it.**

**Cross-arm:**

| pair | flips |
|---|---|
| REF-p1 vs S0-p1 | 5 |
| REF-p2 vs S0-p1 | 4 |
| S0-p1 vs S1-p1 | 5 |
| S0-p1 vs S2-p1 | 5 |
| S0-p1 vs S2-p2 | 5 |

All 8 numbers (3 same-arm + 5 cross-arm) fall in the 3-5 range; no outlier in either direction.

**Which flip sets were margin-audited individually** (token pair, exact-tie check, and position scatter inspected per flip, beyond the margins `compare_logprobs.py` always reports): REF-p1 vs S0-p1, REF-p1 vs REF-p2, REF-p2 vs S0-p1, and S0-p1 vs S1-p1 — these four got the full discriminator.
**Not individually audited** (flip counts and default per-flip margins reported only, no separate token-identity/tie check run): S1-p1 vs S1-p2, S0-p1 vs S2-p1, S2-p1 vs S2-p2, S0-p1 vs S2-p2.

## ConvFinQA answer-level agreement

Primary gate (amended mid-run): a `correct`-field disagreement stops the run only when **both** sides have `finish_reason=stop`. A disagreement where either side hit `finish_reason=length` is recorded as truncation-excluded, not a stop.

**Zero stop-gate mismatches** across all 7 pairs checked: REF-p1 vs S0-p1; S0-p1 vs S1-p1; S0-p1 vs S1-p2; S1-p1 vs S1-p2; S0-p1 vs S2-p1; S2-p1 vs S2-p2; S0-p1 vs S2-p2.

**5 truncation-excluded disagreements**, by session:
- `cfq-val-Single_AON/2011/page_134.pdf-4` — 4 of the 5 (in S0-p1 vs S1-p1; S1-p1 vs S1-p2; S0-p1 vs S2-p1; S0-p1 vs S2-p2).
- `cfq-val-Single_K/2013/page_62.pdf-1` — 1 of the 5 (in REF-p1 vs S0-p1).

## Truncation rate per arm

`finish_reason=length` count out of the fixed 8-session subset, same subset used for every logprob probe and answer-agreement check:

| arm | truncated / 8 |
|---|---|
| REF-p1 | 5 |
| S0-p1 | 6 |
| S1-p1 | 4 |
| S1-p2 | 6 |
| S2-p1 | 5 |
| S2-p2 | 4 |

Range 4-6/8, no arm stands out as truncating materially more or less than the others.

## The two false-alarm gates and how each was resolved

1. **Exactness gate (`--exact` REF vs S0).** Fired first: 5 flips, looked like the S-arm build perturbing the forward with prefetch off — a genuine bug candidate. Resolved by running the same-arm control: REF-p1 vs REF-p2 (both serving identical code, `7de955329a`, nothing between the two passes but a rerun) also produced 5 flips, same signature (same-pair token swaps, min margin ≤0.375 nats, several exact ties, no third candidate, no positional clustering). A second cross-arm sample, REF-p2 vs S0-p1, gave 4 flips with the same signature. Conclusion: decode on this box is not bit-reproducible run to run; the exactness gate was measuring that nondeterminism, not a Task 1-5 code difference. Retired as a pass/fail condition, kept as a diagnostic (the flip-count tables above).
2. **ConvFinQA answer disagreement (the AON session).** Mid-run, `correct` disagreed across arms on `cfq-val-Single_AON/2011/page_134.pdf-4`: S0-p1=False, S1-p1=True, S1-p2=False. Looked like a semantic regression — the primary gate as originally specified stops on any such disagreement. Resolved by inspecting the full transcripts: all three arms derive the identical numeric answer (`11287/8512 - 1 ≈ 0.326010338`, matching `expected=0.32601`) via the identical chain of reasoning; S0-p1 and S1-p2 both hit `finish_reason=length` at the 768-token cap still mid-reasoning-trace (no `ANSWER:` line emitted, hence `correct=False` for lack of a parseable answer, not a wrong number), while S1-p1 converged 70 tokens sooner (`finish_reason=stop`, 698 tokens) and did emit the line. The deciding fact: **S1-p1 disagreed with S1-p2 — the same arm, same commit, same budget, disagreeing with itself** — so the gate as originally written could not distinguish a build difference from within-arm variance. Gate amended: a `correct` disagreement counts as a stop only when both sides reached `finish_reason=stop`; a disagreement involving a `finish_reason=length` side is recorded as a truncation artifact. No token budget was changed to fix this — the 768 cap was kept fixed throughout for comparability.

## Operational notes

- The launcher (`run-shadow-server.sh`) hardcodes `worktree=$work/cc-expert-prediction/worktree` (true since `7de955329a` too, not a Task 5 regression). A literal copy of the launcher dropped into the REF worktree would still `cd` into the S-arm tree and silently serve S0's code under the REF label. Fixed by hand-editing that one line in the REF worktree's own copy only (see Setup); the tracked launcher was never modified.
- The ssh `ControlMaster` socket wedged mid-run (every command reusing the shared control socket hung at connect, while the host itself was reachable and quiet — confirmed by a fresh non-multiplexed connection). Fixed for the rest of the run by adding `-o ControlMaster=no -o ControlPath=none -o ConnectTimeout=10 -o BatchMode=yes` to every ssh invocation.
- Two kills were mis-targeted at the `flock --nonblock cc-gpu.lock env ... sglang serve` wrapper process instead of the actual server pid; both were self-caught (via `ps`/`nvidia-smi` showing the server still resident) and corrected by targeting the real server pid.
- S0's full 768-token capture was initially skipped — only its logprob probe ran before an early hard-stop — leaving a gap in the tok/s and answer-agreement data. Caught before launching S2 and filled by relaunching S0 for the missing capture alone.

## Artifact manifest

All on divix01, confirmed present at the end of the run.

Results / logprobs / metrics — `/mnt/nvme2/nvfp4-work/prefetch-shadow-live/`:
```
ref-p1.jsonl                    ref-p1-logprobs.json            ref-p2-logprobs.json
s0-p1.jsonl                     s0-p1-logprobs.json
s1-p1.jsonl                     s1-p1-logprobs.json             s1-p1-prediction-metrics.jsonl
s1-p2.jsonl                     s1-p2-logprobs.json             s1-p2-prediction-metrics.jsonl
s2-p1.jsonl                     s2-p1-logprobs.json             s2-p1-prediction-metrics.jsonl
s2-p2.jsonl                     s2-p2-logprobs.json             s2-p2-prediction-metrics.jsonl
```
(REF-p2 has no `.jsonl` full-capture by design — logprobs only, for the determinism control.)

Server logs — `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/<cell>/latest/server.log`:
```
prefetch-shadow-ref-p1/latest    prefetch-shadow-ref-p2/latest    prefetch-shadow-s0-p1/latest
prefetch-shadow-s1-p1/latest     prefetch-shadow-s1-p2/latest
prefetch-shadow-s2-p1/latest     prefetch-shadow-s2-p2/latest
```

Input/config:
- A/B subset: `/mnt/nvme2/nvfp4-work/benchmarks/prefetch-shadow/sessions.jsonl`.
- REF worktree: `/data/models/slang/nvfp4-work/cc-expert-prediction/ref-7de955`, HEAD `7de955329a8f41ac336167dcd6fe4e007fc55f2c`, one uncommitted line in its `scripts/expert_prediction/run-shadow-server.sh` (see Setup).

## Limitations

- **The headline scoring-cost numbers are directional, not measured.** REF-p1 vs S0-p1 — two passes with nothing between them but rerun noise — already differ by 0.71 ms/token, and S1-p1 vs S1-p2 — two passes of the identical S1 build — differ by 1.51 ms/token. The S1−S0 delta reported above is ≈2.10 ms/token, the same order of magnitude as that within-arm spread. A signal of 2.10 against a within-arm spread of 1.51 is a direction, not a measurement.
- **APEX's cost is the tighter measurement, and it sits above the break-even band it's compared against.** S2-p1 and S2-p2 agree to 0.12 ms/token (65.03 vs 65.15), noticeably tighter than LLaPor's two passes. That tighter number (≈3.26 ms/token) is the one that sits above the ~2.8-2.9 ms/token break-even quoted in the plan, while LLaPor's noisier, looser number (≈2.10 ms/token, range 1.34-2.85) sits below it.
- **Sample size.** n=29 turns per arm/pass, and most of the subtractions above (tok/s deltas, budget_recall deltas) are single-pass-vs-single-pass; only REF, S1, and S2 have a second pass to check against themselves, and S0 has none.
- **Token-level only.** The tie-break finding (same-pair swaps, near-tie margins, exact ties, no positional clustering) was established at the output-token level from the logprob probes. No MoE router/expert-selection-level confirmation was pulled — declined deliberately to preserve GPU window, since the token-level evidence and the same-arm control already settled whether S0 was implicated, independent of any router-level mechanism.

## Open questions (not resolved here)

- **The break-even the costs above are judged against is itself unsettled.** T3 priced the saving at ~3.15 misses/layer, against a separately measured miss rate of 1.36-2.06 (not measured in this run). If the achievable saving scales with miss count, the ~2.8-2.9 ms/token break-even used above may sit well below that figure, which could invert which side of the line either predictor's measured cost falls on. That re-pricing is not done here and is out of scope for this document.
