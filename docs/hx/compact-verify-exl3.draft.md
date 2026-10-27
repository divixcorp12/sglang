# Compact ragged verify with EXL3 expert streaming (DSV4.1 DSpark)

## Goal
DSV4.1 DSpark serves with `SGLANG_RAGGED_VERIFY_MODE=compact` on the EXL3 expert-streaming stack. The draft
confidence head (with its Markov features) sets each verify's width, so low-confidence drafts are not verified, and
the expert reads and CPU expert work they would trigger never happen.

## Acceptance criteria
- (to be agreed)

## Decisions
- Owner, 2026-10-16: make compact work with EXL3 streaming, rather than building the confidence head under static
  and using it only as a gate input.
- Owner, 2026-10-16: done means correct and working, not faster. Compact runs on the EXL3 streaming stack with no
  launch refusal or crash, gives the same output as static (greedy), and passes tests plus a served smoke. Whether it
  is faster is a later, separate question; a served speed A/B is not in this feature.
- Owner, 2026-10-16: the sentinel approach. Keep static and its 6-wide graph, and drive the dead-row mask from the
  confidence head. Real compact mode is dropped.
- Owner, 2026-10-16: the cut must use the real confidence head with its Markov features (paper eq. 7), not the
  top-prob proxy.
- Owner, 2026-10-16: the STS temperatures are a fitted file, not fitted at startup. Collect once on a served capture
  with the long prompt and thinking on, fit with `dspark_sts_fit.py`, and store the result as a read-only production
  asset with a pinned sha256, checked at launch the way the T3 prefetch head is. The calibration describes the model
  pair, not the hardware.
- Owner, 2026-10-16: the cut is cost-aware (paper section 3.2.2): pick the width w that maximises
  (1 + sum_{k<=w} S_k) / T(w), where S_k is the STS-calibrated cumulative product and T(w) is verify time with w live
  rows, measured at server startup. A fixed-threshold fallback (cut at the first S_k < T) needs the owner's yes before
  it is used or built.

## Rules
- Off by default; production stays on `static` until a served A/B says otherwise (owner decides).
- Follow `.claude/rules/divix01-run-protocol.md` and `.claude/rules/merge-rounds.md`.

## Non-goals
- A served speed A/B of compact vs static, and enabling compact in production.

## Open questions
- (being asked)

## Design
### Background (facts, 2026-10-16)
- The draft checkpoint ships `mtp.2.confidence_head.proj.weight` and a rank-256 Markov head. Under `static`,
  `build_dspark_v4_confidence_head` returns None (`models/deepseek_v4_dspark.py:573`), so the head is never built.
- `static` verifies all 6 tokens (5 drafts + anchor) every step in one fixed-shape graph. `compact` packs verify rows
  to per-request budgets from `HostConfidenceBudgetPlanner` and an SPS step-cost table, with graph tiers by total
  tokens (`speculative/dspark_components/dspark_planner.py:131-200`).
- The EXL3 launch gate refuses non-static modes: "EXL3 expert caching runs a DSpark verify in the decode graph with
  SGLANG_RAGGED_VERIFY_MODE=static only (compact mode reads the host)"
  (`arg_groups/expert_stream_requirements_exl3.py:165`).

### Code survey (2026-10-16, origin/master 9d87cd1275)
- At batch size 1, compact gains nothing. EXL3 forces decode graph bs 1, so the tier grid is [6] and widths 1..6
  replay the same 6-row graph (`decode_cuda_graph_runner.py:526-529`, `ragged_verify.py:34-43`). Compact's padding
  rows are token 0 and still route real experts, because dead-row masking runs only in the static branch
  (`dspark_worker_v2.py:1009-1044`).
- Compact picks the width on the host from confidence lagged 2 steps, with `.cpu().tolist()` syncs
  (`dspark_planner.py:347-353, :501`; `dspark_verify.py:425-430`).
- The confidence head is `Linear(hidden + markov_rank -> 1)` plus a sigmoid. It runs on the device inside the draft
  step from the existing `compute_base_logits` tap, with no extra forward (`deepseek_v4_dspark.py:1062-1084`,
  `dspark.py:412-452`).
- Static already has a device-only per-verify cut: `VerifyDeadRows.set_live_rows` (`layers/moe/verify_dead_rows.py`).
  Dead rows get row 0's experts at weight 0 in the graph and -1 in the RAM-miss routes, so they open no lane, miss
  read, CPU job or prefetch candidate. It is bs-1 only and driven today by `SGLANG_DSPARK_TOP_PROB_GATE`.
- Prior estimate: an oracle width cuts 11-21% per token (`DSV41_REFERENCE.md:7972-7981`).

### Confidence head and STS already in the code (2026-10-16)
- The head is DSpark paper eq. 7, c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]]): `compute_confidence` concatenates the draft
  hidden state with `markov_head.get_prev_embeddings(prev_seq)`, the Markov W1 embedding of the previous draft token
  (`deepseek_v4_dspark.py:1062-1084`, `DSparkConfidenceHead` at `dspark.py:412-452`). The checkpoint has no
  `confidence_head_with_markov` key and the Markov rank is 256, so with_markov is on.
- STS is implemented:
  - collection: `SGLANG_DSPARK_STS_COLLECT_PATH`, identity temperatures required (`dspark_planner.py:96-108`);
  - fit: `python/sglang/benchmark/dspark_sts_fit.py`, a per-position 1D grid of 41 temperatures from 0.1 to 10,
    minimising the ECE of the cumulative product left to right;
  - runtime: `--speculative-dspark-confidence-sts-path` loads per-position temperatures, and `apply_sts` computes
    sigmoid(raw / T_k).
  - It has never been fitted for DSV4.1 EXL3.

### 1. Architecture and components (presented 2026-10-16, pending approval)
The new cut reuses the top-prob gate's path in `dspark_worker_v2.py:1018-1044`. There, a device tensor
`gate_verify_lens` ([bs] int64 in [1, gamma+1]) goes to `begin_gated_static_step`, which caps the accept, and to
`VerifyDeadRows.set_live_rows`, which marks the dead rows. Only the source of `gate_verify_lens` is new.
- **C1, confidence head under static.** When the new flag is on, `build_dspark_v4_confidence_head` builds the head
  under `static` too (today it returns None, `deepseek_v4_dspark.py:573`). The draft step then returns the
  STS-calibrated per-position confidence c_k, [bs, gamma] on the device, through the existing `compute_confidence`.
- **C2, the cost table T(w).** This is a [gamma+1] float tensor on the device: verify milliseconds with w live rows,
  for w = 1..6. It is measured at startup (section 2) and logged.
- **C3, the cut kernel.** S = cumprod(c), E(w) = 1 + sum_{k<w} S_k, and w* = argmax_w E(w) / T(w). It runs entirely on
  the device, with no host read, and outputs the same [bs] int64 as `gate_verify_lens`. It is a tiny torch/Triton op
  next to `gate_verify_lens` in a new module beside `dspark_top_prob_gate.py`. As with the top-prob gate, it applies
  only at bs 1 with greedy sampling; any other batch verifies every row.
- **C4, the STS asset.** A read-only fitted temperature file at `/data/models/slang/prod-assets/dspark-sts/...` with a
  pinned sha256, loaded through the existing `--speculative-dspark-confidence-sts-path` and checked at launch the way
  the T3 head is.
- **C5, the switch and refusals.** One new env flag turns the cut on (name per the env-var-conventions skill). The
  launch refuses:
  - the new cut together with `SGLANG_DSPARK_TOP_PROB_GATE` (one cut source only);
  - the cut without an STS file;
  - a draft checkpoint without `confidence_head` weights;
  - the cut with `SGLANG_RAGGED_VERIFY_MODE` other than static.
- **C6, observability.** Per verify: the chosen w*, the expected E(w*), and the accepted length, plus a counter of
  cut rows, in the existing DSpark metrics and `stages.jsonl`. Tokens lost to a cut can't be observed, because a dead
  row's logits are not valid. They are estimated only from cut-off captures, the same data STS is fitted on.

Section 1 approved by the owner, 2026-10-16. The owner then said to implement immediately with parallel agents, all
tests at the end, so the lead settled section 2 and the hx planning step was skipped at the owner's direction.

### 2. Startup T(w) measurement (lead's decision, 2026-10-16)
- The first `CALIBRATION_VERIFIES = 48` eligible verifies after the copy engine arms, whether warm-up or served, run
  forced widths w = 1..6 round-robin, 8 per width. Interleaving spreads drift and NVMe stalls evenly over the widths.
  A forced width is always safe: acceptance is capped at it, as with any cut.
- Each one is timed with CUDA events on the verify stream, from just before `set_live_rows` to just after the verify
  returns (graph replay, including the lease-gate wait). Elapsed times are read lazily at the next eligible verify,
  so the hot path never syncs.
- T(w) is the median per width, made non-decreasing with a running max. It is frozen after the 48th verify and
  logged once (`dspark confidence cut: T(w) ms = [...]`).
- If any width has no valid sample or a non-positive time, the cut stays off for the server's life, with an ERROR
  log line. Verifies then cover all rows, and the launch is not refused.
- Until T(w) is frozen, the confidence cut is not applied (only forced widths).

### Proposed approach (owner's sentinel idea, approved 2026-10-16)
- Keep `static` and its one 6-wide graph. Drive the existing dead-row mask from the trained confidence head instead
  of the top-prob product. Build the head under static when a new flag is on, compute the cut on the device from the
  head's per-position scores, and feed `set_live_rows`. Acceptance is capped at the cut, as the top-prob gate does.
- Rejected: real compact mode, for the reasons in the code survey.

## Testing
- (to be agreed)
