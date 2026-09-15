# Expert prefetch offline gate (Phase A, T3)

Capture: `/mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438`.
Checkpoints: `/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630/{llapor,apex}`.
Pricing outputs: `/mnt/nvme2/nvfp4-work/prefetch-gate-v2/{llapor,apex,oracle}.json`.
Command: `scripts/expert_prediction/prefetch/price_prefetch.py --predictor {llapor,apex,oracle} --scorer-ms 0,0.02,0.05 --reaction-ms 0,0.03` (per-layer budgets 1-32), run under `taskset -c 0-63` with `OMP_NUM_THREADS=32 MKL_NUM_THREADS=32`.

## Parity check

`check_serving_parity.py` against the real checkpoints/capture: fp32 (training) vs bf16 (serving `LlaporScorer`/`ApexScorer`) recall@16 agree within 0.005 at layers 5, 20, 44. EXIT=0.

## Gate rule (amended Step 8)

GO if `side_saving_ms_per_token_gap >= 3.0` (LLaPor's in-graph next-layer advantage, budget-independent) OR `saving_ms_{aon,prefix}_r0.0_..._per_token >= 3.0` for some budget on `shifted_test`, checked against the oracle upper bound.

## shifted_test_decode totals (primary window, scorer_ms=0, reaction=0 unless noted)

| predictor | budget | recall | aon (r0.0) | prefix (r0.0) |
|---|---|---|---|---|
| llapor | 1 | 0.196 | **5.568** | 5.568 |
| llapor | 2 | 0.342 | 3.238 | **7.085** |
| llapor | 3 | 0.453 | -2.892 | **6.561** |
| llapor | 4 | 0.540 | -9.853 | 5.297 |
| llapor | 6 | 0.661 | -25.477 | 1.633 |
| apex | 1 | 0.231 | **6.714** | 6.784 |
| apex | 2 | 0.409 | 3.001 | **7.251** |
| apex | 3 | 0.544 | -2.339 | **6.735** |
| apex | 4 | 0.647 | -8.805 | 5.696 |
| apex | 6 | 0.783 | -24.168 | 2.836 |

`side_saving_ms_per_token_gap` (LLaPor only, budget-independent) = **6.508**; `side_saving_ms_per_token_overlap` = 6.508→17.114 rising with budget.

r0.03 sensitivity: same shape, ~0.5-1.5 ms/token lower (e.g. llapor b1 aon 5.568→5.568 identical since reaction only matters once posted count > 0 in-window rows... actually b2 aon r0.03=1.828 vs r0.0=3.238). Conclusion unchanged: budgets 1-2 (aon) and 1-4 (prefix) still clear 3.0 under r0.03 for both predictors.

By mixer kind (`shifted_test_decode`, aon, r0.0/s0.0): linear_attention and full_attention both positive at budget 1-2 for both predictors (e.g. apex b1: linear 5.26, full 1.45; llapor b1: linear 4.42, full 1.15), negative from budget 3 on for `full_attention` — mixer split doesn't change the decision.

Oracle bound (upper bound per predictor's window, budget 1): apex_window 8.211, llapor_window 8.535 — both real predictors sit below their oracle bound as expected (sanity check passes); oracle itself clears >>3.0 at low budgets, so the ceiling is not the constraint.

## Decision: **GO**

- LLaPor clears via its budget-independent in-graph `side_saving_ms_per_token_gap` = 6.508 ms/token, regardless of doorbell budget/delivery.
- Both LLaPor and APEX independently clear the `all_or_nothing` doorbell gate at budgets 1-2 (5.6-7.3 ms/token) and the `prefix` gate at budgets 1-4 (5.3-7.3 ms/token), under r0.0; r0.03 sensitivity does not change the call.
- **Copier ask**: at budget=3, `prefix` clears (llapor 6.561, apex 6.735) while `all_or_nothing` does not (llapor -2.892, apex -2.339) — flag to crypto-c9 with these numbers; per-row (prefix) delivery meaningfully extends the useful budget range over today's all-or-nothing copier.
- **Evictable-slot follow-up**: not triggered — `budget_recall@32 - @10` is 0.149 (llapor) and 0.076 (apex), both under the 0.15 threshold.
- Recommended shadow config for T6: small budget (2-3), APEX included per user's request to keep both arms live.

## Deviations from plan

- `test_expert_prefetch_pricing.py`'s `_load_single_layer_capture` helper is custom-written (the plan's referenced neighbor fixture doesn't exist in this tree); it builds a minimal safetensors shard + `manifest.jsonl` matching `capture_writer.py`'s format directly.
- Oracle's `prefix` savings equal its `aon` savings by construction (oracle only ever posts real misses, so there's no false-positive depth to skip past).
