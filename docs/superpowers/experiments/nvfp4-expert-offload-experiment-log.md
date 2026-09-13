# NVFP4 expert-offload experiment log

Qwen3.8-Flash-Next-NVFP4 on divix01 (RTX 5090 32 GiB, PCIe Gen3), bs=1, routed experts streamed
from a host arena into a GPU hot cache. One entry per experiment, newest last. Each entry names
the question, the exact code and settings, where the raw data lives, the numbers, and the verdict,
so a result can be re-read or re-derived without the conversation that produced it.

Paths are on divix01. `work` = `/data/models/slang/nvfp4-work`. Production serves on port 7867
from `work/main-port-probe-7bc4eb` via `work/run-nvfp4-expert-dynamic-hot10g.sh`; its current log
is `work/latest.log`. Plan these experiments serve:
`docs/superpowers/plans/2026-09-13-nextn-graph-capture-offload-plan.md`.

## Index

| # | Date (CDT) | Experiment | Verdict |
|---|---|---|---|
| E1 | 09-13 13:53 | Production route trace + residency replay sweep | Deployed σ=0 / ratio 4 / 8-step updates |
| E2 | 09-13 | Expert-route predictability (offline, from E1) | MTP-style cross-token prefetch not promising |
| E3 | 09-13 14:37 | Vision tower off (`--language-model-only`) | Adopted in production |
| E4 | 09-13 15:09 | Eager NEXTN acceptance + expert traffic | Accept 3.27; 2.28 GiB copied per verify |
| E5 | 09-13 15:26 | Eager non-speculative baseline (matches E4) | 3.13 tok/s; NEXTN 3.0x faster eager |
| E6 | 09-13 16:02 | Traced NEXTN: MTP hidden state as expert predictor | Rejected: recall@10 0.14 vs 0.37 for previous token |
| E7 | 09-13 | Greedy output determinism across runs | Only 3 of 7 replies are stable; token-identity gate unusable |
| E8 | 09-13 16:26 / 16:35 | Phase 1 (Fix A + B) GPU unit tests | 80/2 → 82/0 after test updates |
| E9 | 09-13 16:36 | Phase 1 eager NEXTN rerun | **4x decode regression**; gate failed |
| E10 | 09-13 17:19 | Phase 1 A/B: dedup-off vs fixed64 | Cause: kernel expert count varying per call; fixed64 9.41 tok/s, 2.04 GiB/verify |

## Standard NEXTN workload

`work/cc-spec-nextn/spec-nextn-workload.sh`: 7 single chat requests at temperature 0, thinking off,
in this order — `count` (max 120), `long_prompt` (2,740-token prompt, max 48), `count_again` (120),
`code` (400), `explain` (420), `math` (400), `translate` (120). Reports per-request wall time,
`Decode batch` accept length and gen throughput, and the last hot-cache metrics record. Output
prefixes are the first 160 characters of each reply.

Eager NEXTN launch (E4, E9, E10): `work/cc-spec-nextn/run-nvfp4-expert-spec-nextn.sh` —
NEXTN 3 steps / topk 1 / 4 draft tokens, decode and prefill CUDA graphs disabled, graph gather off,
8 GiB hot cache (3,106 slots), host arena, DMA copy engine, `per_pass` recorder,
`UPDATE_DECODE_FORWARDS=8`, `DECAY_TOKENS=16`, `PROMOTION_SIGMAS=0`, `BENEFIT_RATIO=4`,
`--disable-flashinfer-autotune`, `--language-model-only`, port 7869.

---

## E1 — Production route trace and residency replay sweep

- **Question:** which dynamic-residency settings minimise copy time on the real production route stream?
- **Setup:** production launch with `SGLANG_MOE_ROUTE_TRACE_DIR`; offline replay through `ExpertResidencyPolicy` over 433 setting combinations.
- **Data:** `work/cc-route-trace/run-20260913-135317` (`trace/`, `predict.json`, `sweep.json`, `sweep-prod.json`).
- **Results:** the settings then deployed ranked 49/433. At production geometry, σ=0 / benefit ratio 4 / update every 8 decode forwards modelled −9% copy time. Measured after deploying: 48.3 ms/step vs 55.8 before.
- **Verdict:** deployed `PROMOTION_SIGMAS=0`, `BENEFIT_RATIO=4`, `UPDATE_DECODE_FORWARDS=8`.

## E2 — Expert-route predictability

- **Question:** can a cheap predictor fetch experts before the router asks for them?
- **Data:** E1 trace; scratch `route_predict/evaluate.py`.
- **Results (recall of the true top-10):**
  - One-layer lookahead from the quasi-hidden state: 0.775 at 10 candidates, 0.92 at 20.
  - Multi-layer lookahead decays: 0.66, 0.56, 0.50.
  - Next token from the current hidden state: 0.10–0.15. Previous token's experts: 0.38.
- **Verdict:** cross-token prefetch is not promising; in-token lookahead is capped at roughly 1.10–1.17x.

## E3 — Vision tower off

- **Question:** how much VRAM does `--language-model-only` free, and does production still start?
- **Code:** fork `d9e6720112` (adds `Qwen4ExpForConditionalGeneration` to the language-model-only allowlist); launch script commit `9b9835341` in the crypto repo.
- **Results:** weights 10.05 GB; free after capture 9.99 GB; 22,003 MiB idle, ~27 GiB after requests. First attempt failed because the architecture wasn't allowlisted (~4 min downtime).
- **Verdict:** adopted. Remaining VRAM options: `docs/superpowers/specs/2026-09-13-qwen38-nvfp4-vram-reduction-options.md`.

## E4 — Eager NEXTN acceptance and expert traffic

- **Question:** what accept length does NEXTN get on this model, and how much expert traffic does a verify generate?
- **Code:** fork at the E3 state, before Phase 1. Launch: standard eager NEXTN.
- **Data:** `work/cc-spec-nextn/run-20260913-150935`.
- **Results:**
  - Mean accept length 3.27 (2.55–3.75 per log window).
  - 26.3 requested unique experts per layer per verify; routed hit rate 54%; 2.28 GiB copied per verify (958 GiB over 421 verifies).
  - Decode throughput 6.9–11.5 tok/s per ~128 tokens; decode-heavy requests 9.42 tok/s.
  - Needed `SGLANG_MOE_HOT_GPU_MB=8192`: at 10 GiB the draft's own embed/head left no KV cache.
- **Per request:** count 20.0 s · long_prompt 9.3 s · count_again 9.5 s · code 39.1 s · explain 48.1 s · math 39.3 s · translate 4.4 s.

## E5 — Eager non-speculative baseline

- **Question:** is E4's speedup real, measured against the same eager setup without speculation?
- **Data:** `work/cc-eager-baseline/run-20260913-152619` (E4 launch without the speculative flags, port 7870).
- **Results:** decode-heavy 3.13 tok/s vs NEXTN 9.42 (3.01x). Per request: count 24.0 s · long_prompt 11.0 · count_again 54.2 · code 118.1 · explain 130.0 · math 110.6 · translate 5.5.
- **Verdict:** eager decode is overhead-bound, so the 3x will not carry over to CUDA graphs; the plan's payoff model estimates graphs + NEXTN at 14–25 tok/s vs production 20.7.

## E6 — Traced NEXTN: MTP hidden state as an expert predictor

- **Question:** does the MTP draft's hidden state predict the next token's routed experts well enough to prefetch them?
- **Code:** fork `9397545012` (speculative route trace + MTP hidden trace). Launch `work/cc-spec-nextn-trace/run-nvfp4-expert-spec-nextn-trace.sh` (E4 settings + `SGLANG_MOE_ROUTE_TRACE_SPECULATIVE=1`, 4,096 tokens, port 7871).
- **Data:** `work/cc-spec-nextn-trace/run-20260913-160227` (`trace/`, `trace/mtp/`, `evaluate_mtp.json`); session log `work/cc-spec-nextn-trace/session-20260913.log`. Analysis script `work/cc-spec-nextn/analysis/evaluate_mtp.py`.
- **Results:**
  - Workload: 0 failures, mean accept length 3.24.
  - Trace: target 1,664 tokens / 416 verifies (last partial shard lost on shutdown, fixed in `bbbb66daa1`); MTP 2,679 rows, `closed_by: atexit`.
  - Join: 1,242 of 1,664 verify rows; 416 excluded as last draft slot (expected), 6 without a previous row; accept length checked on 410 verifies, 0 mismatches.
  - Recall of the true top-10, mean over 48 layers:

    | Predictor | @10 | @20 |
    |---|---|---|
    | MTP draft hidden → gate | 0.14 | 0.22 |
    | Previous token's experts | 0.37 | – |
    | MTP interleaved with previous token | 0.29 | 0.44 |
    | Token's own final hidden → gate (proposed ceiling) | 0.11 | 0.18 |

  - Flat across draft depth (0.13–0.15). MTP competes only on layers 44–47 (0.28–0.37), beating the previous token only on layer 47.
- **Verdict:** rejected. Even the token's own final hidden state predicts poorly, because each layer's router sees its own residual.

## E7 — Greedy output determinism

- **Question:** can "greedy outputs token-identical to a pre-fix run" (Phase 1 gate) be checked with this workload?
- **Data:** output prefixes from E4, E5, E6 (session outputs) and E9.
- **Results:** across the three pre-Phase 1 runs only `count`, `count_again` and `translate` are identical every time. `code` and `explain` differ between the two pre-fix NEXTN runs; `math` differs between NEXTN and non-speculative; `long_prompt` differs in all three. E9's `long_prompt` matched E5 word for word.
- **Verdict:** hold only the three stable replies to exact identity; treat the rest as informational.

## E8 — Phase 1 GPU unit tests

- **Code:** fork Phase 1 (Fix A dedup + Fix B verify-driven residency), later committed as `9849c98f17`.
- **Data:** `work/cc-phase1/gpu-tests.log` (first run), `work/cc-phase1/gpu-tests-rerun.log` (second run).
- **Results:** first run 80 passed, 2 failed — two existing CUDA tests still asserted pre-change behaviour (`test_expert_hot_cache.py:631` expected a DRAFT_EXTEND_V2 record; `test_expert_stream.py:129` expected no dedup on 2 rows). After updating them: 82 passed, 0 failed, 12,281 subtests, scratch tree byte-identical to the commit.

## E9 — Phase 1 eager NEXTN rerun

- **Question:** do Fix A and B cut copies per verify below the Phase 1 gate (≤ 1.52 GiB/verify, accept ±0.1, tok/s ≥ 9.42)?
- **Code:** fork `9849c98f17`. Launch: standard eager NEXTN (identical to E4).
- **Data:** `work/cc-spec-nextn/run-20260913-163551`; session log `work/cc-phase1/rerun-session.log`.
- **Results:**

  | | E4 (pre-fix) | E9 (Phase 1) |
  |---|---|---|
  | Decode throughput per ~128 tokens | 6.9–11.5 tok/s | **1.6–2.9 tok/s** |
  | Mean accept length | 3.27 | 3.18 |
  | Copied per verify | 2.28 GiB | 2.08 GiB |
  | Unique misses per layer per verify | – | 16.8 |
  | Routed hit rate | 54% | 43% |
  | Residency boundary updates | 5,085 | 2,322 |

  Per request: count 39.9 s · long_prompt 19.3 · count_again 36.9 · code 147.8 · explain 185.5 · math 160.7 · translate 10.7.
- **Diagnosis:**
  - Every request's extra time fits about +0.9 s per verify; prefill is unchanged.
  - Route recording, residency updates, migrations and draft forwards all do the same work or less per verify.
  - Suspects: Fix A's dedup makes the CUTLASS MoE kernel see a different expert count (≈20–40) on nearly every call, or `torch.unique` plus its host sync.
  - Separately, Fix B drives boundaries every 8 verifies instead of about every 3 draft decodes, which lowers the hit rate.
- **Verdict:** gate failed (2.08 GiB > 1.75 no-go line, and 4x slower). Phase 3 on hold.

## E10 — Phase 1 A/B: dedup-off vs fixed64

- **Question:** is the E9 slowdown Fix A's dedup, and if so is it the changing expert count or `torch.unique`?
- **Code:** `9849c98f17` plus one overlay file per variant, each run from a copied Python tree:
  - `dedup-off` — `expert_route_plan.should_dedup` back to `numel > 64` (a 40-route verify skips dedup); Fix B kept.
  - `fixed64` — dedup kept; `_gather_cached` returns tensors with a constant `max(row_count, 64)` leading dimension from the same staging buffers (padding rows zero-initialised, never routed).
- **Launch:** `work/cc-phase1/ab/run-nextn-variant.sh` (standard eager NEXTN, Python from the variant tree). Session `work/cc-phase1/ab/phase1-ab-session.sh`, log `work/cc-phase1/ab/ab-session.log`; detached `production-guard.sh` relaunches production if the session dies.
- **Data:** `work/cc-phase1/ab/dedup-off/run-20260913-171943`, `work/cc-phase1/ab/fixed64/run-20260913-172744`; overlays in `work/cc-phase1/ab/variants/`.
- **Results — dedup-off:**

  | Request | E4 pre-fix | E9 Phase 1 | dedup-off |
  |---|---|---|---|
  | count | 20.0 s | 39.9 s | 15.4 s |
  | long_prompt | 9.3 s | 19.3 s | 9.0 s |
  | count_again | 9.5 s | 36.9 s | 9.7 s |
  | code | 39.1 s | 147.8 s | 44.4 s |
  | explain | 48.1 s | 185.5 s | 50.5 s |
  | math | 39.3 s | 160.7 s | 38.6 s |
  | translate | 4.4 s | 10.7 s | 4.6 s |

  Mean decode throughput 8.69 tok/s; mean accept length 3.23; 393 verifies. Without dedup every missed route is copied: 23.1 missed experts per layer per verify, routed hit rate 42%, **2.85 GiB copied per verify** (25% more than E4, from Fix B's lower boundary rate).
- **Results — fixed64:**

  | Request | E4 pre-fix | E9 Phase 1 | dedup-off | fixed64 |
  |---|---|---|---|---|
  | count | 20.0 s | 39.9 s | 15.4 s | 15.5 s |
  | long_prompt | 9.3 s | 19.3 s | 9.0 s | 9.1 s |
  | count_again | 9.5 s | 36.9 s | 9.7 s | 8.6 s |
  | code | 39.1 s | 147.8 s | 44.4 s | 37.9 s |
  | explain | 48.1 s | 185.5 s | 50.5 s | 47.0 s |
  | math | 39.3 s | 160.7 s | 38.6 s | 36.1 s |
  | translate | 4.4 s | 10.7 s | 4.6 s | 3.7 s |

  | Per verify (393 verifies each) | dedup-off | fixed64 |
  |---|---|---|
  | Mean decode throughput | 8.69 tok/s | **9.41 tok/s** |
  | Mean accept length | 3.23 | 3.15 |
  | Missed experts copied per layer | 23.1 (every routed miss) | 16.5 (distinct misses) |
  | Routed hit rate | 42% | 44% |
  | Copied per verify | 2.85 GiB | **2.04 GiB** |

- **Verdict:**
  - The E9 regression was the FlashInfer CUTLASS MoE kernel receiving a different expert count on nearly every call. It was not `torch.unique` or its host sync: fixed64 keeps both and is as fast as or faster than every other run.
  - Fix for Phase 1: keep dedup and return a constant `max(row_count, 64)` leading dimension from the eager cached gather. The copy saving holds (2.04 vs 2.85 GiB per verify).
  - Still above the Phase 1 gate (≤ 1.52 GiB per verify): the routed hit rate is 44% vs E4's 54%, because Fix B updates residency every 8 verifies. Next: fixed64 with `UPDATE_DECODE_FORWARDS=2` under speculation (plan step B4).
