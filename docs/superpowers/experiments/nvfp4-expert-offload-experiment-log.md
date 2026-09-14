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
| E11 | 09-13 17:56 | Fixed Phase 1 + Phase 2: cadence 2 at 8 GiB and 10 GiB | Phase 2 passes (draft 2.50 GB, 10 GiB starts, peak 29.8 GiB); 9.9 tok/s; 1.92 GiB/verify; promotions capped (min-residence diagnosis refuted by E12) |
| E12 | 09-13 18:19 | Min residence 2 + cadence 2 at 8 GiB | Rejected: boundaries 2.8x but promotions flat (7,076); 2.01 GiB/verify, 9.31 tok/s; cap is in candidate selection |
| E13 | 09-13 18:40 | Offline residency replay sweep on the E6 NEXTN trace (CPU) | Cap is `DECAY_TOKENS=16`; decay 1 / ratio 3 / min res 0 predicts 10.9 distinct, 1.52 GiB/verify at 8 GiB and 1.36 at 10 GiB |
| E14 | 09-13 18:45 | Phase 3 probe: graphed NEXTN at `29dd4b9bb9`, no code changes | All 3 graphs capture; crashes on the first verify (breakable graph keeps bs rows, not 4); 0/7 requests |
| E15 | 09-13 19:15 | E13's residency policy on the GPU (eager NEXTN): decay 1 / ratio 3 / min res 0 / cadence 2 at 8 GiB and 10 GiB | Replay validated; copies −29–31% (1.49 / 1.37 GiB/verify, byte gate passed) but eager tok/s down (9.19 vs 9.90 at 10 GiB) |
| E16 | 09-13 19:34 | Production residency-policy A/B (decode graph, 10 GiB): deployed vs decay 1 vs cadence 4 / decay 1 / ratio 2 / min res 0 | Positive: decode 7.78 → 9.62 tok/s (+23.7%), 0.682 → 0.489 GiB/token, hit 45.6% → 65.0%; replay bytes within 2.5% |
| E17 | 09-13 20:01 | Offline replay: speed value of FP8-freed expert slots (+1,165 / +1,359) for production, eager NEXTN and graphed NEXTN K=40 | Positive but modest: production +12–14% tok/s (stacks with E16c policy: 7.83 → 11.03 predicted); eager NEXTN +5–10%; graphed K=40 bytes −18% |
| E18 | 09-13 20:02 | Stage 3: graphed NEXTN at K=40 (eager rerun, eager trace, untraced graph + 4,096-token prefill, traced graph) | Graph works (3 graphs, 0 breaks, prefill OK, peak 30.0 GiB, stable replies identical) but **speed gate failed**: 9.03 tok/s vs eager 9.10 and the 22 target. K=40 scratch cuts slots 3,106 → 2,141, raising bytes 1.50 → 1.79 GiB/verify |
| E19 | 09-13 20:36 | Non-speculative decode, E16c policy, production graph, on the Stage 3 workload (direct comparison with E18) | **Plain decode wins:** 10.37 tok/s vs graphed NEXTN 9.03 (+15%), wall 156 s vs 170 s, 0.448 GiB/token vs 0.553 per accepted NEXTN token, peak 26.5 GiB; server left up on 7868 |
| E20 | 09-13 21:23 | Offline replay: FP8-freed slots at the 14 GiB serving baseline (4,957 → 6,236 / 6,466 slots, policy 4/1/2/0) | Positive: +10.6% / +12.5% predicted tok/s (11.42 → 12.63 / 12.85); per-slot tok/s gain barely flattens; extrapolated beyond the E16 fit range |

## Standard NEXTN workload

`work/cc-spec-nextn/spec-nextn-workload.sh`: 7 single chat requests at temperature 0, thinking off,
in this order — `count` (max 120), `long_prompt` (2,740-token prompt, max 48), `count_again` (120),
`code` (400), `explain` (420), `math` (400), `translate` (120). Reports per-request wall time,
`Decode batch` accept length and gen throughput, and the last hot-cache metrics record. Output
prefixes are the first 160 characters of each reply.

**Known defect (found 09-13 in the Phase 3 probe):** `REQUEST_FAILURES` cannot detect a failed request. `curl … | jq … || failures++` takes jq's exit status, so a run with 7 of 7 requests failed still reported `REQUEST_FAILURES=0` and `WORKLOAD_EXIT=0`. E4–E12 are unaffected: every one of their session logs shows real reply text for all 7 requests. Later runs use a corrected copy under `work/cc-phase3/`, and a run counts as passed only when the reply text is present.

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
  - Productionised as `29dd4b9bb9`: padding applies only to deduplicated gathers (single-token decode keeps exactly top_k rows), on the hot-cache, pinned-host and uncached paths.

## E11 — Fixed Phase 1 + Phase 2 on the GPU: residency cadence and draft memory

- **Questions:**
  - E11a: with the constant expert count fix, does updating residency every 2 verifies restore the hit rate and bring copies per verify toward the Phase 1 gate (≤ 1.52 GiB)?
  - E11b: does Phase 2 let NEXTN start with the full 10 GiB hot cache (draft loads at ~2.5 GB instead of 4.86), what is peak GPU memory, and what does the larger cache do to the hit rate?
- **Code:** fork `29dd4b9bb9` (Phase 1 `9849c98f17` + constant expert count fix; Phase 2 `0f311510ce`).
- **Launch:** `work/cc-phase1/e11/run-nextn-config.sh <name> <hot MB> <update decode forwards>` — standard eager NEXTN except the two parameters. E11a: 8192 MB, 2. E11b: 10240 MB, 2.
- **Session:** `work/cc-phase1/e11/e11-session.sh`, run detached; log `work/cc-phase1/e11/e11-session.log`; guard log `work/cc-phase1/e11/production-guard.log`. GPU peak sampled every 2 s into `work/cc-phase1/e11/<name>.peak-mib`.
- **GPU unit test gate:** scratch tree byte-identical to the commit for 8 source and 10 test files; 180 passed, 0 failed, 12,424 subtests (`work/cc-phase1/e11/gpu-tests.log`). Covers the changed CUDA gather assertions, the padded hot-cache hit path, the pinned-host path and the Phase 2 draft-sharing tests.
- **Data:** `work/cc-phase1/e11/e11a-8g-udf2/run-20260913-175703`, `work/cc-phase1/e11/e11b-10g-udf2/run-*`.
- **Results — E11a (8 GiB, cadence 2):**

  | | E10 fixed64 (8 GiB, cadence 8) | E11a (8 GiB, cadence 2) |
  |---|---|---|
  | Mean decode throughput | 9.41 tok/s | 9.82 tok/s |
  | Mean accept length | 3.15 | 3.23 |
  | Routed hit rate | 44% | 42% |
  | Distinct misses per layer per verify | 16.5 | 17.0 |
  | Copied per verify | 2.04 GiB | 2.10 GiB |
  | Speculative migration | – | 16.3 GiB |
  | Draft `Load weight end` mem usage | 4.86 GB | **2.50 GB** |
  | Available after target memory pool | 7.37 GB | **9.75 GB** |
  | Peak GPU (sampled) | – | 28,115 MiB |

  Per request: count 13.3 s · long_prompt 8.5 · count_again 8.5 · code 37.0 · explain 44.5 · math 34.5 · translate 3.8. 0 request failures; `count`, `count_again`, `translate` outputs identical to earlier runs.
  - Phase 2 confirmed on GPU: the draft no longer allocates its own embed/lm_head (−2.36 GB), and 2.38 GB more stays available after the pool.
  - Cadence 2 did not restore the hit rate. Migration (16.3 GiB) is almost exactly E9's 16.6 at cadence 8.
  - Residency counters, summed over 48 layers:

    | Run | Boundary updates | Promotions |
    |---|---|---|
    | E4 (pre-Fix B) | 5,085 | 10,701 |
    | E9 / dedup-off / fixed64 (Fix B, cadence 8) | 2,322–2,324 | 6,948–7,096 |
    | E11a (Fix B, cadence 2) | 3,403 | 7,143 |

  - Diagnosis:
    - Cadence 2 raised boundaries 46% but promotions only 3%, so slot changes are capped elsewhere.
    - Minimum residence fits. The gate is `clock.forwards - _last_update[layer] < SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS` (default 8).
    - Before `9849c98f17` the forward counter advanced on every observed forward, draft decodes included (about 4 per verify cycle), so 8 forwards ≈ 2 verifies.
    - Fix B counts only target forwards, so the same setting now holds a layer's slots for 8 verifies, about 4x stickier. That fits the routed hit rate falling from 54% to 42–44% regardless of cadence.
    - Next test: `MIN_RESIDENCE_FORWARDS=2` with cadence 2 under speculation.
    - **Refuted by E12:** min residence 2 lets every layer update at every cadence opportunity, but promotions stay at ~7,100. Min residence limits boundaries, not promotions.
- **Results — E11b (10 GiB, cadence 2):** `work/cc-phase1/e11/e11b-10g-udf2/run-20260913-180502`

  | | E11a (8 GiB) | E11b (10 GiB) |
  |---|---|---|
  | Hot-cache slots | 3,106 | 3,883 |
  | Draft `Load weight end` mem usage | 2.50 GB | 2.50 GB |
  | Available after target memory pool | 9.75 GB | 7.72 GB |
  | Peak GPU (sampled) | 28,115 MiB | 29,835 MiB |
  | Mean decode throughput | 9.82 tok/s | 9.90 tok/s |
  | Mean accept length | 3.23 | 3.19 |
  | Routed hit rate | 42% | 48% |
  | Distinct misses per layer per verify | 17.0 | 15.5 |
  | Copied per verify | 2.10 GiB | 1.92 GiB |
  | Speculative migration | 16.3 GiB | 19.3 GiB |

  Per request: count 12.7 s · long_prompt 7.8 · count_again 8.1 · code 37.1 · explain 43.5 · math 34.5 · translate 3.6. 0 request failures.
  - Phase 2 passes its GPU gate: NEXTN starts with the full 10 GiB hot cache, the draft loads at 2.50 GB, and peak GPU memory stays at 29,835 MiB (≤ 30.5 GiB).
  - The larger cache lifts the hit rate 6 points and cuts copies 9%, but 1.92 GiB per verify is still above the Phase 1 gate (≤ 1.52).
- **Session:** GPU tests, both runs, and production relaunch completed at 18:15:44 (`SESSION_EXIT=0`); production healthy, 22,005 MiB.

## E12 — Minimum residence 2 under verify-driven residency

- **Question:** E11a showed cadence 2 raises boundaries 46% but promotions only 3%. If `SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS` is the cap (8 target forwards = 8 verifies under Fix B, vs about 2 verifies before), does setting it to 2 bring promotions back toward E4's 10,701, the routed hit rate toward 54%, and copies per verify toward the Phase 1 gate (≤ 1.52 GiB)?
- **Code:** fork `29dd4b9bb9` (same commit as E11; production worktree already there, no GPU test gate).
- **Launch:** `work/cc-phase1/e12/run-nextn-residency.sh <name> <hot MB> <update decode forwards> <min residence forwards>` — standard eager NEXTN except those three parameters. E12a: 8192 MB, 2, 2 (directly comparable to E11a, which differs only in min residence 8).
- **Session:** `work/cc-phase1/e12/e12-session.sh`, run detached; log `work/cc-phase1/e12/e12-session.log`; guard log `work/cc-phase1/e12/production-guard.log`; peak GPU in `work/cc-phase1/e12/e12a-8g-udf2-minres2.peak-mib`. The summary adds residency totals (boundary updates, promotions) summed over 48 layers.
- **Data:** `work/cc-phase1/e12/e12a-8g-udf2-minres2/run-20260913-181943`.
- **Results — E12a (8 GiB, cadence 2, min residence 2):**

  | | E11a (min residence 8) | E12a (min residence 2) |
  |---|---|---|
  | Boundary updates (48 layers) | 3,403 | **9,449** |
  | Promotions (48 layers) | 7,143 | **7,076** |
  | Promotions per boundary | 2.10 | 0.75 |
  | Routed hit rate | 42.1% | 45.3% |
  | Distinct misses per layer per verify | 17.0 | 16.2 |
  | Copied per verify | 2.10 GiB | 2.01 GiB |
  | Speculative migration | 16.3 GiB | 12.4 GiB |
  | Mean decode throughput | 9.82 tok/s | 9.31 tok/s |
  | Mean accept length | 3.23 | 3.14 |
  | Peak GPU (sampled) | 28,115 MiB | 27,795 MiB |

  Per request: count 12.7 s · long_prompt 9.0 · count_again 8.6 · code 37.0 · explain 49.3 (380 tokens vs 364) · math 35.8 · translate 3.8. 0 request failures; `count`, `count_again`, `translate` outputs hash-identical to E11a. Both runs had 393 verifies.
  - Min residence was the boundary limiter: 9,449 ≈ 48 layers × 393 verifies / 2 = 9,432, so nearly every layer now updates at every cadence-2 opportunity.
  - Promotions did not move (7,143 → 7,076). With min residence out of the way, candidate selection admits about 7,100 promotions per workload. The remaining suspects are `SGLANG_MOE_HOT_BENEFIT_RATIO=4` and `SGLANG_MOE_HOT_DECAY_TOKENS=16` (the score window counts tokens, and each verify commits about 3.2).
  - Hit rate +3 points and copies −4% are about the size of the run-to-run spread seen between E10 fixed64 and E11a. Throughput fell 5%, partly from the lower accept length.
  - The server log's 4 Tracebacks are one `post-warmup freeze_gc failed` chain (a startup race against port 7869), which is harmless. Production shows the same chain.
- **Verdict:** rejected. Min residence 2 does not lift promotions, and copies stay at 2.01 GiB per verify, above the Phase 1 gate (≤ 1.52). Phase 3 stays on hold.
- **Session:** production relaunched and healthy at 18:29:45 (`SESSION_EXIT=0`, 22,005 MiB, graph captured).

## E13 — Offline residency replay on the NEXTN verify trace

- **Question:** which residency setting actually caps promotions under verify-driven residency? What is the best achievable distinct misses/layer/verify at 3,106 and 3,883 slots (offline oracle vs causal policy), i.e. is the ≤ 12.3 distinct misses/layer/verify (≤ 1.52 GiB) gate reachable by policy alone? And what is the distribution of distinct misses per verify, which sizes the graph scratch K?
- **Method:** CPU-only replay of the real `ExpertResidencyPolicy` over the E6 trace (`work/cc-spec-nextn-trace/run-20260913-160227/trace/`), reusing E1's `replay.py`/`sweep.py`. Validated first against E10–E12 measured hit rates and promotions. Sweep: benefit ratio, decay tokens, cadence, min residence, sigmas, at both slot counts.
- **Data:** `work/cc-phase1/e13-replay/`. Scripts `spec_trace.py`, `spec_replay.py` (loads the worktree's real `ResidencyBoundaryClock` and `ExpertResidencyPolicy`), `spec_validate.py`, `check_routes.py`, `oracle.py`, `spec_sweep.py`, `pick_rows.py`, `probe_inputs.py`. Outputs in `out/`: `validate*.json`, `oracle.json`, `sweep-popularity2.json` (960-row grid), `sweep-ext-*.json` (1,080-row grid for decay 1–8, ratio 0.5–3), `e13-summary.json`/`.log`. Local copy in the session scratchpad.
- **Validation** (cumulative at forward 400 = 7 prefills + 393 verifies; prompt routes rebuilt from the live run's prefill top-32 popularity):

  | Run | Measured hit / distinct / promotions / boundaries | Replay |
  |---|---|---|
  | E10 (cadence 8, min res 8) | 43.6% / 16.54 / ~7,000 / 2,323 | 43.2% / 16.75 / 5,482 / 2,320 |
  | E11a (cadence 2, min res 8) | 42.1% / 17.00 / 7,143 / 3,403 | 43.3% / 16.74 / 5,402 / 3,720 |
  | E11b (10 GiB) | 47.6% / 15.54 / 8,317 / 3,165 | 48.6% / 15.33 / 6,085 / 3,309 |
  | E12 (cadence 2, min res 2) | 45.3% / 16.24 / 7,076 / 9,449 | 44.8% / 16.44 / 5,581 / 9,449 |

  - Hit rate within ~1 point, distinct misses within ~0.3, boundaries exact for E10/E12.
  - Promotions are 21–27% low. Almost all of the shortfall is in the first 100 forwards, after the 2,740-token prompt, whose routes are not in the trace. From forward 100 to 400, promotions per interval match within ~5%.
  - E1's `replay.py` fed the policy 0/1 indicators above 64 routes; this worktree records raw routes, and the E13 replay does too.
- **Promotion cap, one parameter changed from E11a at a time (window promotions, 8 GiB):**

  | Parameter | Values → promotions |
  |---|---|
  | Cadence 1 / 2 / 4 / 8 | 5,382 / 5,402 / 5,455 / 5,482 |
  | Min residence 0 / 2 / 8 | 5,581 / 5,581 / 5,402 |
  | Benefit ratio 0 / 1 / 2 / 4 | 7,091 / 6,358 / 5,934 / 5,402 |
  | Sigmas 0 / 1 | 5,402 / 4,304 |
  | **Decay tokens 4 / 16 / 64 / 256 / per-boundary** | **10,696 / 5,402 / 2,474 / 1,860 / 7,954** |

  `DECAY_TOKENS=16` integrates ~320 committed tokens (~99 verifies), so rankings move slowly. E10–E12 changed only cadence and min residence, which barely matter, which is why they all landed at ~7,100 promotions. This refutes the E11 min-residence diagnosis for the same reason E12 did.
- **Ceilings (distinct misses/layer/verify, verify demand only):**

  | | 8 GiB (3,106 slots) | 10 GiB (3,883 slots) |
  |---|---|---|
  | Belady, free admission | 5.44 (hit 83.7%) | 4.39 (hit 86.9%) |
  | Future-window top-N, promotions paid | 4.38 + 126 promotions/verify = 0.87 GiB | 4.22 + 84 = 0.74 GiB |
  | Best static set for the whole window | 14.39 (1.78 GiB) | 12.64 (1.58 GiB) |
  | Best causal, real policy | 9.47 distinct (1.65 GiB); lowest total 1.451 GiB at 10.1–10.7 distinct | 7.96 (1.51 GiB); lowest total 1.269 GiB at 8.67 |

  No static hot set can pass the gate; a fast-adapting dynamic policy can.
- **Best settings** (total = distinct misses + promotions, each at 2.64 MiB): cadence 1–2, decay tokens 1, benefit ratio 2–3, min residence 0, sigmas 0. Top row at 8 GiB: cadence 1 / decay 1 / ratio 3 → hit 67.3%, 10.65 distinct, 51.7 promotions/verify, 1.451 GiB. At 10 GiB: cadence 1 / decay 1 / ratio 2 → hit 73.6%, 8.67 distinct, 76 promotions/verify, 1.269 GiB. E11a's settings rank 195–196 of 480. Rankings are stable across three prompt models.
- **Predicted for the recommended configs** (cadence 2, decay 1, ratio 3, min res 0, sigmas 0; "corrected" applies E11's replay error: promotions ×1.32–1.37, distinct ×1.015, hit −1 point):

  | Config | Replay hit / distinct / promotions per verify / total | Corrected | Prompt-model bracket |
  |---|---|---|---|
  | A: 8 GiB | 66.7% / 10.76 / 51.0 / 1.463 GiB | 65.6% / 10.92 / 67.5 / **1.525 GiB** | 1.44–1.61 GiB |
  | B: 10 GiB | 70.9% / 9.44 / 49.2 / 1.295 GiB | 70.0% / 9.57 / 67.3 / **1.358 GiB** | 1.27–1.45 GiB |

  Cadence 1 saves only ~0.01 GiB and doubles the per-boundary score readbacks.
- **Approximations:** routes from the E6 (E4-settings) run, not E10–E12, with a difference equal to run-to-run noise; prompt routes synthesized; accept lengths for 410 of 416 verifies from positions; the first 400 forwards only; oracles ignore prompts.
- **Not observable offline:** the host overhead of decay 1 at cadence 1–2, and the GPU latency of 50–100 synchronous promotion copies per verify.
- **Addendum: graph scratch sizing** (`scratch_k.py`, `out/scratch-k.json`; a verify overflows when any layer's distinct misses exceed the K scratch rows; slot split by the cache's own greedy rule; raw replay, no bias correction; tails rest on 393 verifies):
  - Demand ceiling: the per-verify max over layers of distinct experts requested is p50 38, p95 40, max 40.
  - Per-verify max over 48 layers of distinct misses (p50/p95/p99): E11a settings 31/38/39 at 3,106 slots; recommended policy 26/35/37 at 3,106 and 24/33/35 at 3,883.
  - Recommended policy, distinct misses per (verify, layer) at 3,106 slots: p50 10, p95 24, p99 30, max 38; P(>24) 4.5%, P(>32) 0.37%.
  - Slots vs K, recommended policy (hit / mean distinct / total GiB per verify / whole-verify overflow rate):

    | K, slots | Recommended policy | E11a settings |
    |---|---|---|
    | 16, 3,187 | 67.2% / 10.60 / 1.443 / 0.94 | 43.8% / 16.60 / 2.090 / 1.00 |
    | 20, 3,107 | 66.7% / 10.75 / 1.462 / 0.85 | 43.3% / 16.74 / 2.107 / 0.99 |
    | 24, 3,027 | 66.2% / 10.91 / 1.482 / 0.65 | 42.7% / 16.89 / 2.125 / 0.89 |
    | 32, 2,867 | 65.1% / 11.24 / 1.523 / 0.17 | 41.8% / 17.16 / 2.158 / 0.43 |
    | 40, 2,141 | 59.4% / 12.97 / 1.732 / 0 | 36.9% / 18.46 / 2.315 / 0 |

  - Reading: only K = 40 never overflows a whole verify, and its lost slots cost +1.7 distinct misses and +0.21 GiB/verify vs K = 32. Per layer, overflow is rare, so a per-layer fallback would keep K = 32's numbers.
- **Addendum: production (non-speculative) replay** (`prod/prod_replay.py`, `prod/out/prod-replay.json`). Input is E1's per-pass recorder dump (`work/cc-route-trace/run-20260913-135317`): 6 prefills + 2,141 decode steps, whose decode rows match the route trace exactly, plus the prompt routes. Production geometry is 3,403 slots + 10 scratch rows/layer, with the real `ResidencyBoundaryClock` on 1-token DECODE forwards.
  - Validation is exact: at E1's 3,883 slots and settings, forward 2,100 gives 472,291 hits vs 472,259 measured, and promotions (8,434) and boundaries (6,288) are identical. Promotions and boundaries also match at forwards 500/1,000/1,500/2,000, and all 37 `sweep-prod.json` rows at 3,403 slots agree within 0.02 hit points.
  - Sweep: 288 settings. Total per token = (distinct misses × 48 + promotions per step) × 2.64 MiB.

    | Cadence / decay / ratio / min res | Hit | Misses/layer/step | Promotions/step | decide() calls | Total GiB/token | vs deployed |
    |---|---|---|---|---|---|---|
    | 1 / 1 / 2 / 0 (lowest) | 65.7% | 3.44 | 24.7 | 102,816 | 0.489 | −30.1% |
    | 2 / 1 / 2 / 0 | 65.2% | 3.48 | 24.6 | 51,408 | 0.494 | −29.3% |
    | **4 / 1 / 2 / 0 (recommended)** | **64.5%** | **3.55** | **24.3** | **25,680** | **0.501** | **−28.3%** |
    | 8 / 1 / 4 / 8 (decay only) | 58.5% | 4.15 | 10.2 | 12,816 | 0.539 | −22.9% |
    | 8 / 16 / 4 / 8 (deployed) | 44.6% | 5.54 | 5.35 | 12,816 | 0.699 | rank 240/288 |

  - Decay alone from the deployed settings: 1/2/4/8/16/64 → 0.539/0.540/0.568/0.625/0.699/0.821 GiB/token. Ratio moves the total ≤ 0.02, and cadence and min residence barely move it at decay 16. Feeding the policy 0/1 prompt indicators (E1-era gather) leaves the ranking unchanged: recommended 0.643 → 0.495 (−23%).
  - With decay 1, the 2,753-token prompt boundary promotes ~2,965 rows once (7.8 GiB), included in the totals.
  - Not visible offline: the host cost of 48 `decide()` calls plus a score readback per boundary on the decode path, and whether copy time scales with bytes (E15a's eager slowdown suggests it may not). To be measured on production as E16.
- **Verdict:** the promotion cap is the decay window, not min residence or cadence. Retuning is predicted to take the eager path to ~1.36 GiB/verify at 10 GiB (passes the ≤ 1.52 gate) and ~1.52 at 8 GiB (borderline). To be measured on the GPU as E15.

## E14 — Phase 3 probe: graphed NEXTN without code changes

- **Question:** what actually breaks when NEXTN runs with the production CUDA-graph settings, compared with the plan's section 1 predictions?
- **Code:** fork `29dd4b9bb9`, in a separate divix01 worktree `work/cc-phase3/worktree`. Launch `work/cc-phase3/run-nextn-graph.sh`: E11a settings (8 GiB, cadence 2) plus production's breakable decode graph at bs 1, `SGLANG_MOE_EXPERT_GRAPH_GATHER=1`, `stat` recorder.
- **Data:** session log `work/cc-phase3/stage0/stage0-session.log`; attempt 1 `work/cc-phase3/stage0/s0-ple-stage1/run-20260913-184519/`, attempt 2 `work/cc-phase3/stage0/s0-graph-8g/run-20260913-184534/server.log`.
- **Results:**
  - Attempt 1 (PLE staging on): rejected at startup by `memory_hook.py:181` ("does not support speculative decoding").
  - Attempt 2 (PLE staging off):
    - Only 1,186 hot-cache slots: Phase 2's default scratch of 40 rows/layer takes 5.31 GiB of the 8 GiB (E11a had 3,106 slots).
    - Draft weights 2.50 GB; 9.65 GB free after the pool.
    - Graphs: target verify 1 break (0.18 GB), draft decode 0 breaks (0.27 GB), draft extend 0 breaks (0.14 GB).
    - The first request prefilled, then the server died in `eagle_sample`: `shape '[1, 4]' is invalid for input of size 1`. 0 of 7 requests completed; the peak of 24,255 MiB only covers up to that first prefill.
  - Against the plan's predictions:

    | Predicted breakage | Result |
    |---|---|
    | Scratch too small, so all 48 MoE layers break | Refuted: Phase 2 already sizes 40 rows/layer, and the graph path already dedups with a constant expert-table shape |
    | PLE staging key mismatch at verify replay | Not reachable: staging is rejected with speculation. The 1 verify break is probably the file-backed PLE read (inferred; the log doesn't name it) |
    | `memory_hook.py` rejects PLE staging with speculation | Confirmed |
    | Hyper-connection hidden states held as a Python attribute | Not observed; the crash came first |

  - **New failure:** the breakable graph backend keeps only as many output rows as the graph's size key, and for non-ragged verify that key is the batch size (1). The 4 verify logit rows and hidden states are cut to 1 (`capture_one` → `_output_rows`/`_slice_output`; `_capture_graph_size` returns bs). No flag avoids it: ragged verify keys by token count but only DSpark supports it, and the full decode backend needs PLE staging.
  - Tooling: exposed the workload `REQUEST_FAILURES` defect (see Standard NEXTN workload).
- **Verdict:** graphed NEXTN needs code: output rows sized to bs × tokens per request, a smaller scratch with an overflow flag, and a decision on the PLE break. Production down 18:45–18:54, back healthy (22,005 MiB, `breaks=0`).

## Operations note — `--mem-fraction-static` (09-13 18:58–19:15)

- Operator asked for 0.6 on production and all launchers. Production failed to start at 0.6: `Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.6 ... minimum viable = 0.6433` (log `nvfp4-stream-logs/nvfp4-expert-dynamic-hot10g-20260913-190440.log`).
- How the flag works (`kv_cache_configurator.py:2161-2214`): the KV pool budget is free memory at pool time minus `pre_model_load_memory × (1 − fraction)`, and weights, the expert hot cache and the draft are already charged. The pool itself is capped by `--max-total-tokens 8192` (0.18 GB), so the fraction only moves the startup check; it frees or claims no memory. The post-capture resize (`kv_pool_runtime.py:49-120`) is inactive unless `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` is set.
- Operator then chose 0.95. At 0.95 the startup reserve is 1.52 GB, so the check no longer protects anything: an oversized hot budget now fails with a runtime OOM instead of a startup error, and the ≤ 30.5 GiB peak arithmetic is the only guard.
- An incident along the way: the first relaunches failed because `scp -p` from the crypto-repo copy (mode 100644) stripped the installed script's exec bit, so the relaunch failed with "Permission denied" and wrote no log. About 10 minutes of downtime. Fixed with `chmod 755`; installs now go through a temp file + `mv` + an exec-bit check.
- Production at 0.80 was healthy again at 19:11:12 (3,403 slots, 22,005 MiB, `breaks=0`), then restarted at 0.95 and was healthy at 19:14:40 with identical memory: 3,403 slots, 8,192 KV tokens, 10.67 GB free after the pool, 22,005 MiB used, `breaks=0` (log `nvfp4-stream-logs/nvfp4-expert-dynamic-hot10g-20260913-191154.log`). Launchers at 0.95: production, `cc-phase1/e12/run-nextn-residency.sh`, `cc-phase1/e15/run-nextn-policy.sh`, `cc-phase3/run-nextn-graph.sh`. Backups `*.bak-memfrac080` on divix01.

## E15 — E13's residency policy on the GPU (eager NEXTN)

- **Question:** does the replay's recommended policy reproduce its predictions on the GPU: at 8 GiB, hit ~66%, ~10.9 distinct misses/layer/verify, ~1.52 GiB/verify total; at 10 GiB, hit ~70%, ~9.6 distinct, ~1.36 GiB/verify? What do 50–70 synchronous promotion copies per verify cost in tok/s?
- **Code:** fork `29dd4b9bb9` (production worktree, unchanged).
- **Launch:** `work/cc-phase1/e15/run-nextn-policy.sh <name> <hot MB> <cadence> <min res> <decay tokens> <benefit ratio>`; eager NEXTN like E12 except the policy parameters and `--mem-fraction-static 0.95`. E15a: 8192 MB, 2, 0, 1, 3. E15b: 10240 MB, 2, 0, 1, 3. Sigmas 0.
- **Workload:** fixed copy `work/cc-phase1/e15/spec-nextn-workload.sh` (a request counts as failed if curl fails or the reply has no content).
- **Session:** `work/cc-phase1/e15/e15-session.sh`, detached; waits for production restart 4 (0.95) to finish first; log `work/cc-phase1/e15/e15-session.log`; guard log `work/cc-phase1/e15/production-guard.log`. The summary adds migration and total GiB per verify and promotions per verify.
- **Results — E15a (8 GiB, 3,106 slots):** `work/cc-phase1/e15/e15a-8g-d1r3/run-20260913-191518`

  | | E13 replay prediction (corrected) | E15a measured | E11a (old policy) |
  |---|---|---|---|
  | Routed hit rate | 65.6% | **65.8%** | 42.1% |
  | Distinct misses/layer/verify | 10.92 | **10.96** | 17.00 |
  | Promotions (per verify) | 67.5 | **22,729 (58.0)** | 7,143 (18.2) |
  | Boundary updates | – | 9,456 | 3,403 |
  | Copied per verify (h2d) | – | 1.355 GiB | 2.10 GiB |
  | Migration per verify | – | 0.135 GiB | 0.042 GiB |
  | Total per verify | 1.525 GiB | **1.49 GiB** | 2.14 GiB |
  | Mean accept length | – | 3.23 | 3.23 |
  | Mean decode throughput | – | **8.74 tok/s** | 9.82 tok/s |
  | Peak GPU (sampled) | – | 28,455 MiB | 28,115 MiB |

  Per request: count 15.5 s · long_prompt 7.7 · count_again 9.6 · code 37.6 · explain 54.9 · math 41.8 · translate 4.3. `WORKLOAD_EXIT=0` with the fixed failure counting; 392 verifies.
  - The replay matched hit rate and distinct misses within 0.4 points / 0.04. Promotions landed between its raw (51) and corrected (67.5) figures.
  - Copies per verify fell 29% (h2d 2.10 → 1.355 GiB; total including migration 1.49 GiB, under the 1.52 gate), yet eager throughput fell 11%, and `explain` took 54.9 s vs 44.5. Eager decode time is not set by bytes copied here. Suspects, unmeasured: per-boundary host work (48 `decide()` score readbacks and CPU sorts every 2 verifies) and ~58 synchronous promotion copies per verify. Needs a profile before it can say anything about the graphed path.
- **Contention check:** E13's production replay job ran on divix01 from ~19:21:24 to 19:27:29. It overlapped E15a's `explain` (19:21:12–19:22:07) and `math` (19:22:07–19:22:49), and the 19:20–19:30 sar sample averaged 25% user CPU (~18 of 72 cores) vs 4% the interval before. E15a's throughput is therefore partly confounded; its copy counters are not. E15b's requests (19:27:45–19:30:16) ran after the job ended, and no replay file was written after 19:27:29, so E15b is uncontended.
- **Results — E15b (10 GiB, 3,883 slots):** `work/cc-phase1/e15/e15b-10g-d1r3/run-20260913-192303`

  | | E13 replay prediction (corrected) | E15b measured | E11b (old policy) |
  |---|---|---|---|
  | Routed hit rate | 70.0% | **69.1%** | 47.6% |
  | Distinct misses/layer/verify | 9.57 | **9.99** | 15.54 |
  | Promotions (per verify) | 67.3 | **22,727 (57.8)** | – |
  | Copied per verify (h2d) | – | 1.235 GiB | 1.92 GiB |
  | Migration per verify | – | 0.132 GiB | 0.049 GiB |
  | Total per verify | 1.358 GiB | **1.367 GiB** | 1.97 GiB |
  | Mean accept length | – | 3.28 | 3.19 |
  | Mean decode throughput | – | **9.19 tok/s** | 9.90 tok/s |
  | Peak GPU (sampled) | – | 29,435 MiB | 29,835 MiB |

  Per request (completion tokens): count 12.3 s (111) · long_prompt 8.7 (38) · count_again 9.8 (111) · code 36.9 (400) · explain 50.7 (368) · math 37.5 (400) · translate 4.0 (23). E11b: code 37.1 · explain 43.5 (359) · math 34.5 (400).
- **Outputs:** in both runs the `count`, `count_again` and `translate` replies hash-match E11a/E12 (`f03ad516`, `f03ad516`, `46d5d9ad`). 0 request failures under the fixed counting. Session ended 19:32:47 (`SESSION_EXIT=0`), production relaunched healthy.
- **Verdict:**
  - The replay is validated on the GPU at both sizes: hit rate within 1 point, distinct misses within 0.4, total bytes within 1–3%. Decay 1 cuts copies per verify by 29–31%, and both sizes now pass the Phase 1 byte gate (1.49 and 1.37 GiB ≤ 1.52).
  - Eager throughput did **not** improve: 9.19 vs 9.90 tok/s at 10 GiB, uncontended, with `code` flat and `explain`/`math` slower. In eager NEXTN the decay-1 policy's extra host-side cost outweighs the saved bytes: 3x the promotion copy submissions (6,935 vs 1,987), 58 synchronous promotion rows per verify, and 48 `decide()` readbacks every 2 verifies. This is inferred; a profile would confirm it.
  - Phase 1's throughput gate (≥ 9.42 tok/s) is not met with this policy. The byte gate matters for the graphed path, where per-token host overhead is smaller; that is measured in Stage 3. For production, it is measured directly in E16.

## E16 — Production residency-policy A/B

- **Question:** does E13's production replay prediction (decay 1 cuts copy bytes per decode token by 23–28%) turn into faster graphed production decode, or does the extra boundary host work (more `decide()` calls, ~24 synchronous promotion rows per token) eat the gain as it did in eager NEXTN (E15)?
- **Code:** fork `29dd4b9bb9` (production worktree, unchanged).
- **Launch:** `work/cc-phase1/e16/run-prod-policy.sh <name> <cadence> <decay tokens> <benefit ratio> <min res>`: production's exact settings (10 GiB hot, breakable decode graph at bs 1, graph gather, PLE staging, `per_pass` recorder, `--mem-fraction-static 0.95`) except the policy, on port 7868.

  | Config | Cadence | Decay tokens | Ratio | Min res | Replay GiB/token |
  |---|---|---|---|---|---|
  | E16a deployed | 8 | 16 | 4 | 8 | 0.699 |
  | E16b decay only | 8 | 1 | 4 | 8 | 0.539 (−22.9%) |
  | E16c recommended | 4 | 1 | 2 | 0 | 0.501 (−28.3%) |

- **Workload:** `work/cc-phase1/e16/prod-ab-workload.sh`, the E1 route-trace workload (code 520, summary 220 over the 2,740-token records prompt, math 420, chat 460, translate 180, chat2 400 max tokens; temperature 0), with fixed failure counting. It reports per-request wall time and tokens, mean and median `Decode batch` throughput, and decode hot-cache counters.
- **Session:** `work/cc-phase1/e16/e16-session.sh`, detached; waits for E15 to finish; records host load before and after each config; log `work/cc-phase1/e16/e16-session.log`; guard log `work/cc-phase1/e16/production-guard.log`. The replay agent's divix01 CPU jobs are paused for the duration.
- **Run dirs:** `work/cc-phase1/e16/{e16a-deployed/run-20260913-193412, e16b-decay1/run-20260913-194145, e16c-rec/run-20260913-194838}`.
- **Results** (09-13 19:34–19:58). Every config:
  - captured the graph at bs 1 with `breaks=0`;
  - finished all 6 requests with the same completion token counts (2,093 decode tokens), 0 request failures and workload exit 0.

  Server-log tracebacks are only the startup connection-refused probes, present in every run. The session relaunched production healthy at 19:58:33 (`SESSION_EXIT=0`).

  | Config | Mean / median decode tok/s | Total wall s (6 requests) | Routed hit rate | h2d GiB/token | Migration GiB/token | h2d + migration GiB/token (replay) | Promotions/token |
  |---|---|---|---|---|---|---|---|
  | E16a deployed | 7.78 / 7.95 | 285.4 | 45.6% | 0.672 | 0.010 | 0.682 (0.699) | 5.3 |
  | E16b decay only | 9.47 / 9.77 (+21.7%) | 236.9 (−17.0%) | 58.2% | 0.516 | 0.023 | 0.539 (0.539) | 10.2 |
  | E16c recommended | **9.62 / 9.69 (+23.7%)** | **233.8 (−18.1%)** | **65.0%** | 0.432 | 0.057 | **0.489** (0.501) | 23.4 |

  - **Per-request wall time, a → b → c (s):**

    | Request | E16a | E16b | E16c |
    |---|---|---|---|
    | code | 72.8 | 51.7 | 51.7 |
    | summary | 34.9 | 32.0 | 31.4 |
    | math | 55.4 | 45.3 | 47.6 |
    | chat | 54.0 | 45.9 | 43.8 |
    | translate | 18.0 | 15.6 | 15.0 |
    | chat2 | 50.3 | 46.4 | 44.3 |

  - **Replay accuracy:** the replay's byte predictions were within 2.5% on all three configs. Its hit-rate prediction for E16c was 64.5%, against 65.0% measured.
  - **Caveat:** one run per config, in a fixed order a → b → c, with no repeat. Host load was quiet: 1-minute load average 3.6–5.4, and the replay CPU jobs were paused.
- **Verdict:**
  - **On graphed production decode, cutting copy bytes buys throughput almost one for one.** Bytes fell 21–28%, decode tok/s rose 22–24%, and wall time fell 17–18%. E15's eager slowdown came from eager host overhead, not from the policy itself.
  - **E16c (cadence 4, decay 1, ratio 2, min residence 0) is best.** It promotes 4.4× as many rows as deployed, and the extra host work does not show up as a cost on the graphed path.
  - **Decay tokens is the main lever:** E16b captures ~88% of E16c's gain.
  - **Recommendation:** switch production to E16c's policy, and use it as the residency baseline for the graphed NEXTN Stage 3 comparison.

## E17 — Speed value of FP8-freed expert slots (offline replay)

- **Question:** FP8 on the large BF16 matrices would free about 3–3.5 GiB, which is +1,165 or +1,359 expert slots. Before anyone measures FP8 accuracy, is that extra capacity worth the effort?
- **Code:** `work/cc-phase1/e13-replay/fp8-slots/{fp8_slots.py, fp8_throughput.py}`, the E13 replay engine using the real greedy per-layer split.
  - Every run asserts that resident slots equal the requested count.
  - NEXTN promotions are scaled ×1.14, the E15a correction.
- **Run:** 09-13 20:01:33–20:02:34 on divix01, `taskset -c 64-71`, 4 workers, EXIT=0. It finished before E18 started.
  - Outputs: `out/fp8-slots-replay.json`, `out/fp8-throughput.json`, `out/fp8-slots.log`.
  - The tok/s conversion ran on the laptop.
- **Production cost model**, least squares over E16's three runs: s/token = 0.1347·B + 0.0359, with B in GiB/token.
  - Fitted tok/s is 7.83 / 9.21 / 9.82 against measured 7.78 / 9.47 / 9.62, so residuals are at most 0.26.
  - At 3,403 slots, replay bytes are scaled ×0.976 to match E16.
- **Results, production** (E1 trace, 10 scratch rows per layer; GiB/token and tok/s calibrated):

  | Policy | Slots | Hit | GiB/token | tok/s |
  |---|---|---|---|---|
  | Deployed 8/16/4/8 | 3,403 | 44.6% | 0.682 | 7.83 |
  | | 4,568 | 52.4% | 0.591 | 8.66 (+10.6%) |
  | | 4,762 | 53.6% | 0.577 | 8.80 (+12.4%) |
  | E16c 4/1/2/0 | 3,403 | 64.5% | 0.489 | 9.82 |
  | | 4,568 | 71.1% | 0.406 | 11.03 (+12.4%) |
  | | 4,762 | 72.0% | 0.395 | 11.23 (+14.4%) |

  1/1/2/0 is only 0.01 GiB/token below 4/1/2/0 at every size and would run 8× the `decide()` calls, which the model does not account for.
- **Results, eager NEXTN** (policy 2/1/3/0; L = 3.237):

  | Budget | Slots | Hit | Corrected GiB/verify | tok/s, model (a) / model (b) |
  |---|---|---|---|---|
  | 8 GiB | 3,106 | 66.7% | 1.481 | 8.75 / 8.79 |
  | 8 GiB | 4,271 | 72.7% | 1.240 | 9.59 / 9.41 |
  | 8 GiB | 4,465 | 73.5% | 1.206 | 9.72 / 9.51 |
  | 10 GiB | 3,883 | 70.9% | 1.313 | 9.32 / 9.21 |
  | 10 GiB | 5,048 | 75.7% | 1.115 | 10.09 / 9.77 |
  | 10 GiB | 5,242 | 76.4% | 1.087 | 10.21 / 9.85 |

  - Model (a) uses production's c with F = 0.1706 s/verify.
  - Model (b) is a two-point fit on E15a/E15b: c = 0.1011, F = 0.2186. E15b ran while the E13 replay was also using the CPU, so (b) is less trustworthy.
  - The replay reproduces E15a closely: hit 66.7% vs 65.8% measured, 1.481 vs 1.491 GiB/verify.
- **Results, graphed NEXTN, K = 40** (same policy; graphed F not yet measured):

  | Slots | Hit | Corrected GiB/verify | Per-verify max unique (p50 / p95) | K40 overflow | tok/s at F = 0.05 / 0.10 / 0.15 |
  |---|---|---|---|---|---|
  | 2,141 | 59.4% | 1.750 | 30 / 37 | 0 | 11.33 / 9.64 / 8.39 |
  | 3,306 | 67.9% | 1.436 | 26 / 34 | 0 | 13.30 / 11.03 / 9.42 |
  | 3,500 | 69.0% | 1.393 | 26 / 34 | 0 | 13.62 / 11.25 / 9.59 |

- **Caveats:**
  - The production model is linear in bytes, fitted on 3 points.
  - Production covers the whole E1 session, including the prompt burst.
  - NEXTN covers only the first 400 forwards of the E6 trace.
  - No correction or calibration changes any ranking.
- **Verdict:**
  - **FP8 slots are worth +11–14% production decode tok/s, and the gain stacks with the E16c policy:** 7.83 → 11.03 tok/s predicted, +41% overall. The policy switch (+26%) is larger and costs nothing, so do it first.
  - **Graphed NEXTN at K = 40 gains the most:** bytes fall 18% and tok/s rises 14–17%, depending on F.
  - **Eager NEXTN barely benefits (+5–10%)**, because host overhead dominates.
  - **Next step:** FP8 accuracy (KL / greedy-match first, then GSM8K) is worth measuring. Whether it matters for NEXTN depends on E18's graphed F.

## E18 — Phase 3 Stage 3: graphed NEXTN with 40 scratch rows per layer

- **Question:** does graphed NEXTN verify decode work end to end, and does it clear Phase 3's gates?
  - The graph captures cleanly (0 breaks).
  - Output matches eager.
  - 4,096-token prefill fits under the 30.5 GiB peak.
  - Decode reaches ≥ 22 tok/s.
- **Code:**
  - Fork `29dd4b9bb9` plus the uncommitted Stage 2a/2b changes in `work/cc-phase3/worktree`: the verify graph row count and PLE staging key, the verify trace, and the memory hook.
  - All 11 files match `cc-phase3/stage2b.sha256`.
  - Review verdict: APPROVE WITH REQUIRED CHANGES. The two launch requirements were met at launch. The gate was read by hand, not enforced by the script.
  - The E15a rerun used the production worktree.
- **Runs:** all with NEXTN (3 steps, topk 1, 4 draft tokens), the E15a policy (cadence 2, min res 0, decay 1, ratio 3) and the Stage 3 workload (`cc-phase3/stage3-workload.sh`: the 7 E15 requests, full replies saved).

  | Run | Setup |
  |---|---|
  | `cc-phase3-e15a-rerun` | Eager, 8 GiB hot, untraced |
  | `eager-trace` | Eager, 8 GiB hot, verify trace on |
  | `k40-graph-untraced` | Breakable graphs, graph gather, hot 10,709 MiB = 2,141 slots + 5.3 GiB scratch (40 rows × 48 layers), then one 4,096-token prefill |
  | `k40-graph` | Same as untraced, with the verify trace on |

- **Session:**
  - Started as `cc-phase3/stage3-session.sh` (log `cc-phase3/stage3/stage3-session.log`); T5 GPU test passed 17/17.
  - At the user's request the order changed after `eager-trace`: the session was paused with SIGSTOP and `cc-phase3/stage3-continue.sh` took over. It reuses the session's functions unchanged, appends to the same log, and ran `k40-graph-untraced` before `k40-graph`.
  - Production was not relaunched.
  - Bug in that continuation: it waited for the paused workload with `kill -0`, and a zombie child still answers `kill -0`. Killing the paused session unblocked it after 30 s.
  - Separately, the production guard and the first waiter were both fooled by E16's `SESSION_EXIT=0` line, which the session copies into its own log.
- **Results:** 0 request failures in every run; all 4 server-log tracebacks are startup connection probes.

  | Run | Mean decode tok/s | Accept length | Routed hit rate | h2d + migration GiB/verify | Peak GPU MiB |
  |---|---|---|---|---|---|
  | `cc-phase3-e15a-rerun` | 8.99 | 3.205 | 65.7% | 1.496 | 28,435 |
  | `eager-trace` | 9.10 | 3.239 | 65.7% | 1.502 | 28,155 |
  | `k40-graph-untraced` | **9.03** | 3.226 | 58.0% | 1.785 | 29,957 (30,677 with prefill) |
  | `k40-graph` (traced) | 8.76 | 3.173 | 58.0% | 1.782 | 29,897 |

  - **Graph checks, both graph runs:**
    - Target verify, draft decode and draft extend graphs each captured once at bs=[1], `breaks=0`, 0.22 / 0.25 / 0.14 GB.
    - Verify PLE staging key 4.
    - All 11 decode logs report `cuda graph: True`.
    - The untraced run was confirmed to have no extra env.
  - **Prefill:** 4,096-token prefill `PREFILL_4096_OK`, 5.4 s, peak 30,677 MiB (≤ 30.5 GiB = 31,232 MiB).
  - **Draft model:** its FP8_BLOCK_SCALES experts (`mtp.layers.0.mlp.experts`) run on the Triton fp8 MoE fallback with the default, untuned kernel config. This is the same in every NEXTN run, eager and graphed.
  - **Replies:**
    - The three replies E7 holds to identity (`count`, `count_again`, `translate`; md5 `f03ad516`/`f03ad516`/`46d5d9ad`) are identical in all four runs.
    - Full-text compare, differing replies out of 7:

      | Pair | What differs between them | Differing replies |
      |---|---|---|
      | Rerun vs eager-trace | Worktree and trace | 4 |
      | Eager-trace vs k40-graph | Graph only (both traced) | 3 |
      | k40-graph vs k40-graph-untraced | Trace only | 3 |

    - Graph vs eager differs no more than eager vs eager or graph vs graph, which matches E7's finding that `code`, `explain`, `math` and `long_prompt` are not reproducible run to run.
  - **Verify trace parity, eager-trace vs k40-graph:** 7 comparable requests.
    - Requests 0–2 are identical, and request 6 differs only past the commit.
    - Request 3's first committed divergence is a near-tie (top-2 logit gap 0.25 vs 0.125).
    - Requests 4 and 5 first diverge on a draft token, and so on accept length (gaps 0.125 → 2.0 and 5.75 → 12.75). A draft proposal changes only acceptance, not which tokens are committed.
- **Cost accounting**, E16 copy cost c = 0.1347 s/GiB:
  - eager-trace: 3.239 / 9.10 = 0.356 s/verify, of which 0.202 s is copies, leaving F ≈ 0.154 s/verify.
  - k40-graph-untraced: 3.226 / 9.03 = 0.357 s/verify, of which 0.240 s is copies, leaving F ≈ 0.117 s/verify.
  - So graphing removed about 37 ms/verify of fixed overhead, and the 5.3 GiB of scratch rows added about 38 ms of extra copying.
  - The remaining ≈ 117 ms/verify is ≈ 36 ms per accepted token, the same as production's non-speculative F (0.036 s/token).
- **Verdict:**
  - **Correctness and memory gates: pass.** The graph captures with 0 breaks, stable replies are identical, and the 4,096-token prefill peaks at 30.0 GiB.
  - **Parity:** no graph-specific divergence was found. Every committed divergence lies within run-to-run noise, since eager vs eager differs as much.
  - **Speed gate: failed decisively.** 9.03 tok/s vs 22 needed, and no faster than eager.
  - **NEXTN on this offload setup cannot beat plain decode by much.** A 4-token verify touches ~26 distinct experts per layer, so copies grow with accepted tokens. The graphed fixed cost per accepted token already equals non-speculative decode.
  - **FP8 slots would lift graphed NEXTN only to ≈ 10.4 tok/s** (E17 at F = 0.117).
  - **Next:** measure plain decode on this exact workload (E19) before deciding whether NEXTN stays on the roadmap.

## E19 — Plain decode with the best policy on the Stage 3 workload

- **Question:** on exactly the E18 requests, how does non-speculative decode compare with graphed NEXTN?
  - Setup: production's graph setup plus E16c's policy.
- **Code:** fork `29dd4b9bb9`, production worktree, unchanged.
- **Launch:** `cc-phase1/e16/run-prod-policy.sh e19-nonspec-rec 4 1 2 0` on port 7868. Settings:
  - 10 GiB hot cache: 3,403 slots, 1.33 GiB scratch.
  - Breakable decode graph at bs 1, graph gather, PLE staging, `per_pass` recorder.
  - `--mem-fraction-static 0.95`.
  - Policy: cadence 4, decay 1, ratio 2, min residence 0.
- **Workload:** `cc-phase3/stage3-workload.sh`, the 7 E18 requests, replies saved.
- **Session:** `cc-phase3/e19-session.sh`, log `cc-phase3/e19/e19-session.log`. Run dir `cc-phase1/e16/e19-nonspec-rec/run-20260913-203453`.
  - At the user's request the server was kept up afterwards. The session was paused with SIGSTOP during the `math` request, and `cc-phase3/e19-finish.sh` finished the summary, prefill and compares without stopping the server.
  - The finish script waits on process state, so the zombie child did not stall it, unlike E18's continuation.
- **Results** (E19 finished 20:41:08, `E19_EXIT=0`):
  - 0 request failures, 7 replies saved.
  - 36/36 decode logs report `cuda graph: True`, and the graph was captured with `breaks=0`.
  - The 4 server-log tracebacks are startup connection probes.
  - 4,096-token prefill `PREFILL_4096_OK` in 5.3 s. Peak GPU was 25,975 MiB after the workload and 26,515 MiB during the prefill.

  | Run | Decode tok/s (mean of logs) | Wall s, 7 requests | Completion tokens | Wall tokens/s | GiB per generated token | Hit rate |
  |---|---|---|---|---|---|---|
  | **E19 plain decode, E16c policy** | **10.37** | **156.3** | 1,460 | **9.34** | **0.448** | 68.1% |
  | E18 `k40-graph-untraced` | 9.03 | 169.7 | 1,470 | 8.66 | 0.553 (1.785 / 3.226) | 58.0% |
  | E18 `eager-trace` | 9.10 | 163.1 | 1,454 | 8.91 | 0.464 (1.502 / 3.239) | 65.7% |
  | E18 `cc-phase3-e15a-rerun` | 8.99 | 166.7 | 1,469 | 8.81 | 0.467 (1.496 / 3.205) | 65.7% |

  - **Per-request wall time, E19 vs graphed NEXTN (s):**

    | Request | E19 | Graphed NEXTN |
    |---|---|---|
    | count | 15.0 | 14.1 |
    | long_prompt | 9.6 | 9.8 |
    | count_again | 8.7 | 8.6 |
    | code | 41.3 | 41.3 |
    | explain | 33.8 | 45.7 |
    | math | 43.1 | 45.4 |
    | translate | 4.7 | 4.8 |

    NEXTN only ties plain decode on the high-acceptance `count` requests (accept length ≈ 3.8), and loses on prose.
  - **Replies:** the stable three (`count`, `count_again`, `translate`) are identical to both NEXTN runs. The other replies differ as they do between NEXTN runs (E7): 4 of 7 against the eager rerun, 3 of 7 against graphed NEXTN.
  - **Decode counters:**
    - 1,393 tokens.
    - 3.19 unique misses per layer per token.
    - h2d 0.394 GiB/token plus migration 0.055 GiB/token.
    - 22.8 promotions per token.
- **Verdict:**
  - **Plain decode with the E16c policy beats graphed NEXTN on this workload:** +15% decode tok/s, −8% wall time, −19% GiB per token, and 4.2 GiB less peak memory.
  - **NEXTN costs more bytes per accepted token than plain decode (0.553 vs 0.448 GiB)**, because the 4-token verify's union of experts is large and K=40 scratch shrinks the cache.
  - **Its fixed overhead per accepted token is no lower either** (E18).
  - **Recommendation:** stop the NEXTN graph work and make plain decode with the E16c policy the production config.
  - **Next speed levers,** in order:
    1. Ship the policy (E16: +24%).
    2. FP8 for the BF16 GPU weights, to get more slots (E17: +12%, predicted 11.0 tok/s on the E1 workload).
    3. Reduce the per-token copy cost c.
  - **Server:** the E19 server is left running on port 7868. Production (7867) is down.

## Operations note — serving config after E19 (09-13 20:49–21:05)

- **Launch script:** at the user's request the served model moved to a new script, `/data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh`, run in tmux `cc-nvfp4-dynamic`.
  - Logs go to `cc-e16c-public/run-*/server.log`, with `cc-e16c-public/latest.log` pointing at the current one.
  - The old installed script, `run-nvfp4-expert-dynamic-hot10g.sh`, is unchanged and unused.
- **Settings:** E19's settings, changed step by step:
  - Listens on `0.0.0.0:7867`. A LAN request to `10.0.0.15:7867` returned a reply.
  - `--context-length` and `--max-total-tokens` raised to 65536. KV is 0.75 + 0.75 GB; only the 12 full-attention layers hold KV, about 24 KiB per token. The model's native maximum is 262,144.
  - `SGLANG_MOE_HOT_GPU_MB=14336`: 4,957 slots, 13.7 GiB residency plus 1.33 GiB scratch.
  - `--default-chat-template-kwargs '{"enable_thinking": true}'`. The chat template already defaults to thinking on. `--reasoning-parser auto` returns the thinking in `reasoning_content`.
  - No server-side thinking budget exists: this fork only enforces `max_thinking_tokens` when a request sends it. Thinking is bounded by the request's `max_tokens` and by the context.
- **Headroom check** (21:01–21:03): a 20,688-token prompt with thinking on and `max_tokens` 1500.
  - Idle GPU was 27,609 MiB, and `Memory pool end` reported 5.24 GB available.
  - The request finished in 122.6 s with `finish_reason=length`: all 1,500 completion tokens were reasoning tokens.
  - Peak GPU during the request was **32,055 of 32,607 MiB**.
  - During a following ~9.4k-token prefill, the allocator logged `expandable_segments: memory mapping failed with OOM` (18 MB free). It recovered: health was 200 and the request returned 200.
  - SGLang also warned that the `_causal_conv1d_fwd_kernel` and `_router_triton_kernel` Triton kernels loaded after serving started, with 0.13 GiB free.
  - Prefill staging buffers for misses are about 1.35 GB per set, two sets (510 rows).
  - Decode logs during the request showed 12.5–18.4 tok/s.
- **Decision (user):** keep 14 GiB with 65,536 context despite about 0.5 GB of headroom.
  - Known risk: a long or miss-heavy prefill can OOM the server.
  - Fallback options: 12 GiB (≈ 2 GiB headroom), 13 GiB, or 32,768 context (frees ≈ 0.75 GB).

## E20 — FP8-freed slots at the 14 GiB baseline (offline replay)

- **Question:** at the new serving baseline (14 GiB, 4,957 slots, policy 4/1/2/0), what do FP8's extra slots buy? The two FP8 options add +1,279 slots (all groups except lm_head, +3,373 MiB) or +1,509 slots (all groups, +3,979 MiB).
- **Code:** `work/cc-phase1/e13-replay/fp8-slots/prod_14g.py`, the E17 case 1 method.
  - E1 trace, 10 scratch rows per layer, greedy split.
  - Every case asserts the resident slot count equals the requested count.
- **Run:** 09-13 21:23:15–21:23:35 on divix01, `taskset -c 64-71`, 4 workers, EXIT=0. Outputs: `out/prod-14g.{json,out,log}`.
- **Slot rule:** slots = ⌊MB·2²⁰ / 2,764,808⌋ − 480.
- **tok/s model:** the E16 fit, s/token = 0.1347·B + 0.0359, with bytes calibrated ×0.976.
- **Results:**

  | Policy | Slots | Hit | Misses/token | Promotions/token | Calibrated GiB/token | Calibrated tok/s | vs 4,957 |
  |---|---|---|---|---|---|---|---|
  | 4/1/2/0 | 3,403 | 64.5% | 170.2 | 24.3 | 0.489 | 9.82 | −14.0% |
  | 4/1/2/0 | **4,957** | 72.9% | 130.2 | 22.3 | 0.384 | 11.42 | baseline |
  | 4/1/2/0 | 6,236 | 77.6% | 107.4 | 20.2 | 0.321 | 12.63 | **+10.6%** |
  | 4/1/2/0 | 6,466 | 78.4% | 103.9 | 19.8 | 0.311 | 12.85 | **+12.5%** |
  | 8/16/4/8 | 4,957 | 54.7% | 217.5 | 6.8 | 0.564 | 8.94 | −21.7% |

  - **Marginal gain per +1,000 slots:** +10.5%, then +8.3%, then +7.4% tok/s; bytes saved fall from −0.068 to −0.049 to −0.043 GiB/token.
  - **Why tok/s per slot barely flattens:** tok/s = 1/(c·B + F), so each GiB saved is worth more as B shrinks.
- **Caveats:**
  - B = 0.31–0.38 GiB/token lies outside the fitted range (0.489–0.682), so the +10–12% leans high.
  - The ×0.976 calibration was measured at 3,403 slots.
  - Workload is the E1 session only.
- **Verdict:**
  - FP8's slot value holds at 14 GiB: about +11% predicted.
  - FP8 is now implemented behind `SGLANG_ONLINE_FP8_GROUPS` (design `docs/superpowers/specs/2026-09-13-fp8-nonexpert-weights-design.md`; CPU tests 17/17). Nothing is committed.
  - The GPU accuracy session (phase A, about 70 min) is parked at the user's request until the prefetch design is settled.
