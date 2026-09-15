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
| E21 | 09-14 14:40 | CPU experts spike: kernel microbenchmark (node 1, ≤ 4 threads), `kt_kernel` research, round-trip cost analysis, ik_llama / llama.cpp baselines | Positive pending a GPU prototype: `kt_kernel` native NVFP4 computes 130 experts per token in 25.8 ms at 4 threads (0.5% error), under the 43 ms in-graph break-even, so ~14 tok/s is predicted vs 11.42; Python graph breaks don't win; ik_llama with all experts on CPU decodes 6.5 tok/s at 4 threads and 13.9–16.7 at 18 threads, already above SGLang's 10.37 |
| E22 | 09-14 17:45 | GPU microbenchmark: host→GPU expert copy paths vs the PCIe Gen3 x16 ceiling, including krasis's `cuMemcpyBatchAsync` ANY / prefer-overlap settings | Krasis settings: no gain (production DMA already 12.6–12.7 of 12.8 GiB/s). **The in-graph miss kernel runs at 38–50% of the link** (4.9–6.4 GiB/s), so link-class in-graph copies would save an estimated 22–27 ms/token. The DMA path isn't graph-capturable, and the cost model's 44 ms doesn't match the benchmark's 52–69 ms |
| E23 | 09-14 18:34 | CUDA 13 driver graph nodes for expert copies: BatchMemOp semantics, MEMCPY nodes captured in torch's CUDA graph, host retarget and GPU-predicate IF fan-out | Viable only with a host sync per layer: BatchMemOp nodes can't copy; captured MEMCPY nodes run 11.8 GiB/s byte-exact inside torch's graph, but targets are host-only; IF fan-out ≈ 1.7 s/token at 512 experts. Nets ~15–20 ms/token only with a thin C++ break (estimate) |
| E24 | 09-14 18:44 | Warp-aligned in-graph copy kernel: 23-variant screen (grid, block, load width, layout) and final W1–W4 eager + graph replay on an exclusive GPU | Positive: 11.2–11.5 GiB/s (89% of link) vs the original 5.0–6.6, byte-exact, no graph break; closes ~80% of the gap to the link; estimated 22–37 ms/token saved; integration is a kernel-only change in `expert_cache_transfer.cuh` |
| E25 | 09-14 20:33 | Serving A/B of E24's 16 B kernel: base/proto/base at production settings, logprob margins, in-graph shadow copy under real decode, dynamic-residency-off pair | Positive, cleared: 0 differing bytes over 452,088 real miss rows; saves 17.1–17.3 ms/token (13.4 → 17.4 tok/s, +29%) at production settings and 54.5 ms/token (+63%) with residency off; greedy parity unusable because the stock server is itself nondeterministic |
| E26 | 09-15 | Offline re-rank of the post-kernel ideas: cost model refit from E25's on/off pairs (c_miss 0.1426 → 0.0681 s/GiB), applied to E20's replay | Copy-saving ideas lose about a third of their value; CPU experts at 4 threads now lose (−7%). Order: FP8 slots (+7–9%, implemented), overlapping next-layer copies (ceiling +27–48%, needs a contention microbenchmark), skipping low-weight misses (+8–12% at 20–30%). Gen4/Gen5 host: +27% / +47% |
| E27 | 09-15 01:02 | Copy kernel under decode load: side-stream next-layer copy overlapped with 48 captured NVFP4-MoE layer graphs, eager and graph, 1/3/10/30 rows/layer | ALIVE but window-bound: no measurable contention either way; saving = min(copy, window)/copy, 97.6–97.8% at 3 rows with a 0.76 ms/layer window, 35% with the real kernels' 0.25 ms; byte- and hidden-state-exact. The real serving per-layer GPU time decides the size |
| E28 | 09-15 02:00 | Serving decode profile (nsys, CUDA-graph node trace, no-profiler arm): per-layer window, miss distribution, step composition | Window 0.268 ms, not 0.76; one graph, no breaks; misses/layer mean 2.72 (p90 6). Idea 2 re-priced to ~10 ms/token ceiling (+15–17%). **Every 4th step (residency update) stalls the host ~60–80 ms extra, 38–47% of decode wall time without the profiler, ~70 ms unattributed host CPU**: the largest lever found |
| E29 | 09-15 02:12 | Doorbell copier prototype: in-graph poster → pinned ring → C++ spin thread (`cudaMemcpyBatchAsync`) → chunked GPU waiter with in-graph fallback | Negative for serving: bytes always correct, eager copies 12.7 GiB/s, 54/54 tests (branch `cc/doorbell-prototype` `d57db3e1a1`). The doorbell thread's own `cudaMemcpyBatchAsync` copies land only after the replays launched after them, so every ungated in-graph wait times out. A plain torch side-stream copy is **not** held by graph replay (round 1's claim corrected). The host-gated form works but costs 74 vs 40 ms/token |
| E30 | 09-15 02:39 | Residency-update host stall attribution (instrumented spans, py-spy, no-profiler arm; 233 update steps) | Attributed: the update costs 76 ms p50 / 109 ms p90 on the decode thread, ≈85% Python bookkeeping and synchronous tiny transfers (rank sort, 137 mapping publishes, 274 scalar slot writes, plan uploads), 15% promotion-copy wait; 35.7 ms + 0.616 ms × promotions. Fixes #1–#6 estimated to remove ≈50 ms per update ≈ +20% tok/s |
| E31 | 09-15 04:31 | Residency update inside the decode CUDA graph (`cc/residency-update-fast` `252a888559`, flag `SGLANG_MOE_GPU_RESIDENCY_UPDATE`) vs E30's Python fixes vs base; 175 tests, lockstep replay, 2-run interleaved serving A/B | Positive: gpu +22.3% tok/s (14.87 → 18.18), mean ITL −18.1%, update-step p50 118 → 65 ms (the rest is the in-graph promotion copy); Python fixes alone +15.8%; hit rate unchanged; replay identical uncapped (1 truncated layer at K=64); 0 large-margin flips. Open: normal-step +1.7 ms unresolved. Merged `d43c59b58a` and deployed 09-15 05:12 |
| E32 | 09-15 12:10 | Doorbell hold isolation: thread copy API (batch / per-segment), batch `srcAccessOrder` (stream / during_call / any), thread-created vs torch-created stream; `cc/doorbell-prototype` `696b8cc96b` | **Cause found: the thread's own `cudaStreamCreate` stream.** On a torch-created stream the hold is gone at 3 rows: 0 timeouts per token, requests land 0.64 ms after their layer, token 68 ms in graph / 40.5 ms eager-post, byte-exact, with either copy API. Access order does nothing (`any` = baseline; `during_call` blocks the thread, 40/960 serviced). 30 rows also succeeds once the wait budget covers the 6 ms copy: 0 timeouts, done 6.3 ms after the layer, 339 ms/token. The torch-stream thread keeps 12.3–12.4 GiB/s and beats the in-graph kernel by 4–7% at 1–10 rows. Step-2 estimate ≈2.4 ms/token, below the 3 ms go threshold |

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
  - FP8 is now implemented behind `SGLANG_ONLINE_FP8_GROUPS` (design `docs/superpowers/specs/2026-09-13-fp8-nonexpert-weights-design.md`; CPU tests 17/17). It is committed in `fa66acb84f` but not deployed.
  - The GPU accuracy session (phase A, about 70 min) is parked at the user's request until the prefetch design is settled.

## E21 — CPU experts spike: run cache misses on the CPU instead of copying them

- **Question:** during bs1 decode, can cache-miss experts run on divix01's CPU faster than today's PCIe copy?
  - Today a token costs about 96 ms: about 36 ms of fixed GPU work plus about 52–60 ms of expert copies (E19/E20).
  - At 14 GiB there are 2.71 misses per layer, 130.2 per token (E20).
- **Hardware:**
  - 2× Xeon Gold 6154 (Skylake-SP, 18 cores per socket), AVX-512 F/BW/CD/DQ/VL with no VNNI, AMX or avx512_bf16; 188 GB RAM.
  - PCIe Gen3 x16. The GPU and the production host arena are on NUMA node 0.
- **Four lanes, 09-14 14:40–15:10:**
  - kernel microbenchmark (CPU only, `taskset -c 64-71`, ≤ 4 threads);
  - `kt_kernel` source and docs research;
  - round-trip cost analysis from existing traces;
  - ik_llama baseline.
- **Artifacts:**
  - divix01 `nvfp4-work/cc-cpu-spike/bench/REPORT.md`, with scripts, raw JSONL and command logs (94 + 5 runs, all EXIT=0);
  - `nvfp4-work/cc-ikbench/` (`t4-plegpu.log`);
  - laptop scratchpad `cpu-spike/{kt-kernel-research,ikllama-baseline}.md`.

### Kernel compute for 130 experts per token

- **Setup:** cores 64-71 (node 1), memory on node 1, cold medians. Weights rotate across distinct copies, so they are not warm in L3.
- **Break-even against E20's 11.42 tok/s:** about 43 ms if the CPU call runs inside the CUDA graph, about 22 ms with Python graph breaks.

| Kernel | Output error vs dequantized NVFP4 | 1 thread | 2 threads | 4 threads |
|---|---|---|---|---|
| `kt_kernel` NVFP4 (native, no re-quantization) | 0.52% | 103.4 | 53.8 | **25.8** |
| ggml Q8_0 | 1.5% | 79.3 | 39.4 | 26.5 |
| ggml Q4_K (Q5_0 down) | 10.3% | 54.0 | 30.4 | 17.9 |
| ggml Q4_0 | 13.7% | 57.5 | 32.9 | 18.4 |
| torch bf16 | 0.54% | 151.7 | 91.2 | 55.1 |
| torch fp32 | 0 | 189.1 | 112.4 | 96.5 |
| torch int8 weight-only | 1.5% | 939 | 489 | 251 |

- **Thread scaling, 1→2→4:** `kt_kernel` 1.92× and 4.01×, nearly linear. ggml Q4_K 1.78× and 3.0×. fp32 1.96× at 4 threads, which is bandwidth-bound.
- **Cross-NUMA** (node-0 memory, node-1 cores): 4-bit paths are 1.16–1.27× slower; `kt_kernel` is 1.17× slower.
- **RAM:** all 24,576 experts fit only as 4-bit, about 68 GB. bf16 needs 242 GB and Q8_0 needs 128 GB.
- **`kt_kernel` gotchas:**
  - The NVFP4 output buffer must be bf16. A float32 buffer silently returns garbage (relative error 1.22).
  - Its WorkerPool pins its own threads and calls `numa_bind`, which escapes `taskset`. The benchmark blocked this with an LD_PRELOAD shim plus an affinity assertion.

### Round-trip and graph cost (analysis, existing traces)

- **Transfer:** a 2560-dim bf16 hidden state GPU→CPU→GPU with sync takes about 0.03–0.06 ms per layer, 1.5–3 ms per token. Trace medians: memcpy under 1 µs, `cudaMemcpyAsync` 0.007–0.013 ms, sync 0.002–0.004 ms.
- **Graph breaks:** a Python break costs about 2.65 ms measured (09-12 breakable-graph trace, 144 top-level ops per break), or an assumed ~0.45 ms for a lean break. At 48 breaks per token that is 127 or 22 ms.
- **`kt_kernel` avoids breaks:** it runs CPU submit/sync as `cudaLaunchHostFunc` stream callbacks, and graph capture only preallocates pinned buffers per batch size.
- **Model:** s/token = F + M + 48·(b + r) + C, with F = 35.9 ms, M (promotions) = 7.6 ms, r = 0.03 ms.

| C (CPU compute, ms/token) | In-graph callback | Lean break | Deferred (`kt_max_deferred`) |
|---|---|---|---|
| 10 | 18.2 tok/s | 13.3 | 22.3 |
| 20 | 15.4 | 11.8 | 22.3 |
| 26 (measured `kt_kernel`, 4 threads) | **~14.1** | ~11.0 | 22.3 |
| 40 | 11.8 | 9.5 | 22.3 |

- **Deferral:** upstream defers the lowest-scoring experts and adds their output one layer late. The paper reports −0.5% LiveBench with 6 of 8 deferred. Deferring *cold* experts, which can include top-1, is untested.

### `kt_kernel` capability (source and docs)

- **NVFP4 support:** kt-kernel ≥ 0.7.0.post4 has `--kt-method NVFP4`. It reads this ModelOpt checkpoint directly (`NVFP4SafeTensorLoader`), group 16, AVX2 kernel only on this CPU.
- **Fork glue is stale:** our `kt_ep_wrapper.py` equals upstream sgl-project, which predates kt-kernel 0.7 (`num_gpu_experts` vs `gpu_experts_mask`) and supports fixed IDs 0..N-1 only.
- **kvcache-ai/sglang is further along:** per-layer masks, frequency placement, and a dynamic expert update that is graph-safe.
  - It has no NVFP4 layerwise prefill.
  - It probably lacks the `qwen4_exp` architecture.
- **Blockers:**
  - kt-kernel pins torch 2.9.1, so it must be built from source with `--no-deps`.
  - It keeps its own RAM copy of all experts (issue #2084), next to our ~65 GiB arena.

### Baselines with experts on the CPU

- **Unsloth Studio llama.cpp (09-08/09):** `UD-Q3_K_XL`, 36 unpinned threads, `--fit on` (split not logged). Median decode 12.41 tok/s over 25 requests, sagging from 13–15 to 11.5–12.7; prefill 185 tok/s for prompts of 1k tokens or more. Speculation acceptance was about 1/64.
- **ik_llama sweep (09-14 15:01–15:05):** ji-farthing IQ4_KT, `-ot exps=CPU`, all 480 experts per token on CPU, PLE on host (the loader ignored the GPU request), 2.3 GiB weights on GPU, `-c 10240 -ub 2048 -n 64`, `taskset -c 0-3 -t 4`.

  | Depth | Prefill tok/s, 4 threads | Decode tok/s, 4 threads | Prefill tok/s, 18 threads | Decode tok/s, 18 threads |
  |---|---|---|---|---|
  | 0 | 690.45 | 6.48 | 689.79 | **16.72** |
  | 2048 | 687.68 | 6.57 | 679.58 | **15.78** |
  | 4096 | 652.60 | 6.45 | 653.16 | **16.55** |
  | 6144 | 642.19 | 6.50 | 644.23 | **16.12** |
  | 8192 | 627.33 | 6.69 | 630.84 | **13.87** |

  - **Runs:** the 4-thread run was `taskset -c 0-3`, 15:01–15:05 (`t4-plegpu.log`). The 18-thread run was `taskset -c 0-17 -t 18 -tb 18`, 15:13:45–15:16:01, EXIT=0 (`t18-plegpu.log`; the tag says plegpu but PLE was on host).
  - **Scaling:** decode rises about 2.5× from 4 to 18 threads; prefill does not change with threads.
  - **Noise:** each depth is one 64-token sample, so the 8192 row may be noise.
  - **Memory:** VRAM peak about 6.1 GiB; host RSS peak about 92 GB.
  - **Output quality is not checked:** the loader logged `Oops: tensor with strange name per_layer_token_embd.weight`, and this is a sweep, not a generation.
- **Operations:**
  - SGLang :7867 was stopped 14:57:23–15:09:13 for the 4-thread run (user-approved).
  - It was stopped again at 15:13:24 for the 18-thread run and left down at the user's request.
  - The 18-thread run on cores 0-17 was a one-time exception to the ≤ 4-thread rule, approved by the user.

### Caveats

- Compute timings use synthetic inputs, and harnesses differ: `kt_kernel` includes a Python submit/sync, while ggml times graph compute only.
- Node 1 is not the deployment node. Nothing above 4 threads has been measured on the kernels.
- Untested:
  - whether `kt_kernel` host callbacks replay correctly inside the CUDA graph;
  - whether flashinfer_cutlass skips −1 expert ids;
  - the accuracy of deferring cold experts.

### Verdict

- **Positive, pending a GPU prototype.**
  - Native NVFP4 CPU compute (26 ms at 4 threads) is under the 43 ms in-graph break-even.
  - It predicts about 14 tok/s against 11.42 (+23%), and about 17 tok/s if 8 cores scale linearly. That beats E20's FP8 slot gain (+11%) and the prefetch cap (1.10–1.17×).
  - Python graph breaks do not win (~11 tok/s).
  - **Independent confirmation:** ik_llama with *all* 480 experts per token on the CPU (no GPU expert cache at all) decodes 13.9–16.7 tok/s at 18 threads, already above SGLang's measured 10.37 and E20's predicted 11.42. A hybrid that keeps a hot GPU cache and computes only about 130 misses on the CPU should do better. That comparison is untested, and ik_llama's IQ4_KT is a different quant with output quality unchecked.
- **Next:** a GPU prototype that measures in-graph CPU callbacks, correctness versus eager, and C on node-0 cores. It needs a design that shares one host copy of the experts.

## E22 — Host→GPU expert copy paths against the PCIe link (GPU microbenchmark)

- **Question:** how far are our expert copies from what PCIe Gen3 x16 carries? Does krasis's `cuMemcpyBatchAsync` setup (`src/pcie_batch.rs`: `srcAccessOrder` ANY plus prefer-overlap-with-compute flag) beat production's DMA call?
- **Run:**
  - 09-14 17:4x–18:05 CDT on divix01, with SGLang :7867 already down (user request) and the GPU idle before every run.
  - `taskset -c 64-71`, at most 4 threads, host memory `numactl --membind=0` (production arena placement).
  - Four runs, all EXIT=0: r1 arena-style registered, 2048 rows; r2 `pin_memory`; r3 1 GiB registration chunks; r4 node-1 memory, 1024 rows.
  - 40 iterations each, median and p90.
  - Byte-exact check on all 6 NVFP4 tensors per row, every method, every run: all True.
- **Artifacts:** divix01 `nvfp4-work/cc-pcie-bench/` (`REPORT.md`, `tables.md`, `results.jsonl`, `run_all.log`).
- **Workloads:**
  - W1: one batch of 130 scattered rows (335 MiB).
  - W2: 48 calls of 3 rows (144 rows, 380 MiB), shaped like per-layer calls.
  - A row is 6 segments totalling 2,764,808 bytes.

### Results (r1)

| Method | W1 median / p90 ms | W1 GiB/s | W2 median / p90 ms | W2 GiB/s | W2 ms/call |
|---|---|---|---|---|---|
| M0 link ceiling: 1 GiB contiguous pinned copy | 77.92 / 77.96 | **12.83** | – | – | – |
| M1 in-graph kernel `copy_expert_row_segments_gpu`, eager | 51.86 / 56.42 | 6.45 | 75.83 / 77.01 | 4.89 | 1.58 |
| M2 same kernel inside a CUDA graph (production miss path) | 52.39 / 57.35 | **6.39** | 76.03 / 77.55 | **4.88** | 1.58 |
| M3 production DMA (`transfer_embedding_ranges_direct`, own stream) | 26.40 / 26.42 | **12.68** | 29.48 / 29.50 | **12.58** | 0.61 |
| M3 on torch's default stream (NULL handle → per-range `cudaMemcpyAsync`) | 27.81 / 27.86 | 12.04 | 30.98 / 31.04 | 11.97 | 0.65 |
| Control: `cuMemcpyBatchAsync`, Stream order, flags 0, one batch | 26.56 / 26.60 | 12.60 | 29.01 / 29.04 | 12.78 | 0.60 |
| M4 `cuMemcpyBatchAsync`, ANY, flags 0 | 26.55 / 26.60 | 12.61 | 29.01 / 29.03 | 12.78 | 0.60 |
| M5 `cuMemcpyBatchAsync`, ANY, flags 1 (prefer overlap) | 26.74 / 26.76 | 12.52 | 29.01 / 29.04 | 12.78 | 0.60 |
| M6 CPU `index_select` into pinned staging + one copy | 65.77 / 70.82 | 5.09 | 72.96 / 77.12 | 5.08 | 1.52 |

- **Variant runs:** r2–r4 matched r1 within about 3%.
- **Link:** Gen3 x16 in all 17 samples under a sustained copy loop. Idle read Gen1, which is power-saving downtraining.
- **Contention:** a synthetic ~22.6 ms matmul queue ran on a separate compute stream.
  - The copy on its own stream finished at 27.7–28.0 ms for all flag settings, and compute was not delayed.
  - A copy on the compute stream waited behind it (about 46 ms) in every variant.
  - `flags = 1` changed nothing.

### Verdict

- **Krasis settings: no gain.** Production DMA is already at 99% of the link. ANY and prefer-overlap stayed within ±1.5%. Their small W2 edge also appears in the Stream-order control, so it comes from one batched call instead of 6 per-tensor calls. Expected effect on the 7.6 ms of promotions: about 0 ms (estimate).
- **The in-graph miss kernel is the transfer bottleneck.** It runs at 50% (one 130-row launch) and 38% (48 per-layer launches) of the link.
  - Graph replay equals eager, so launch overhead isn't the limit. The kernel's GPU threads read host memory one uint32 word at a time.
  - At link-class speed, the ~44 ms per token of miss copies would drop to about 17–22 ms, saving 22–27 ms (estimate): 0.335 GiB ÷ 12.6 GiB/s ≈ 27 ms.
  - That is a larger lever than FP8 slots (E20) or prefetch (E2 cap).
- **Obstacle:** the fast DMA path is a host-side driver call and can't be captured in the decode graph. Candidate fixes:
  - a faster in-graph kernel (bigger host reads per GPU thread);
  - CUDA 13 graph batch-memop nodes (`cuGraphAddBatchMemOpNode`, exposed by `cuda.bindings.driver`), with PyTorch integration unverified;
  - per-layer graph breaks with DMA, where break cost (E21: ~0.45–2.65 ms × 48) likely eats the saving.
- **Open discrepancy:** at the benchmark's in-graph rate, 130 rows would cost 52–69 ms, more than the 44 ms the 0.1347 s/GiB cost model implies (7.4 GiB/s effective). Production's real launch shape (dedup, row counts, interleaving) may differ, so reconcile it with a production-shaped trace before relying on the ratio.
- **Caveats:**
  - CPU on node-1 cores; the node-1 run used 1024 rows.
  - M2 timings exclude plan-buffer uploads.
  - The in-graph kernel was not measured under compute contention.
  - The contention load was synthetic, not a real MoE decode.

## E23 — CUDA 13 graph nodes for expert copies (`cuGraphAddBatchMemOpNode`, MEMCPY nodes)

- **Question:** can E22's link-class copy be put inside the decode CUDA graph through the driver graph API (option (b) of E22's obstacle list)? Does it work with PyTorch's CUDA graphs, and can the copy targets come from GPU memory so no host sync is needed?
- **Run:**
  - 09-14 18:16–18:34 CDT on divix01, :7867 down. All GPU runs under `flock cc-gpu.lock` with `GPU_APPS_BEFORE: []` (an earlier p1 attempt exited 3 because another session's server held the GPU, so nothing was measured there).
  - Exits: p1=0, p2_smoke=0, p2=0. Same-process medians. Byte-exact on every correctness check.
  - Sources checked: CUDA 13.4 `cuda.h`, `cuda_device_runtime_api.h`, torch 2.13 `libtorch_cuda.so` imports and `torch/_higher_order_ops/cudagraph_conditional_nodes.py`.
- **Artifacts:** divix01 `nvfp4-work/cc-cugraph/` (`REPORT.md`, `probe1.py`, `probe2.py`, `results.jsonl`, `logs/`, `run.log`).

### Findings

- **BatchMemOp nodes can't copy.** Their ops are WAIT/WRITE_VALUE_32/64, BARRIER, ATOMIC_REDUCTION and FLUSH_REMOTE_WRITES (`cuda.h:571-580`), with only address/value/flags changeable per launch from the host (`cuda.h:21557-21602`). The spike's premise was false.
- **MEMCPY nodes do copy**, one copy per node. They can be retargeted per launch only from the host (`cudaGraphExecMemcpyNodeSetParams1D`: 1-D, same context, no zero length; `cuda.h:22583-22627`). Updates take effect on the next launch.
- **torch 2.13 `CUDAGraph` is this driver API** (`cudaStreamBeginCapture`, `cudaGraphInstantiateWithFlags`, `cudaGraphLaunch`). A pinned `copy_` or `cuMemcpyHtoDAsync_v2` on arena-registered rows, issued during capture, becomes a MEMCPY node in torch's own graph.
  - Replay copies the host bytes as they are at replay time.
  - Nodes interleave with kernels: matmul → copies → kernel reading the rows was exact.
  - 8 rows = 48 nodes, captured in 4.9 ms, instantiated in 0.25 ms.
- **`cuMemcpyBatchAsync` can't be captured:** rc=900, `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
- **No API reads copy pointers from device memory**, so a GPU-computed plan needs a device→host read per MoE layer.
  - Plan sync: 0.025 ms on an idle GPU, 0.246 ms queued behind one matmul. The cost is the graph break around it, not the sync.
  - Retarget: 0.028 ms for 18 nodes (about 1.5 µs per node).
- **The sync-free route (GPU-predicate IF nodes choosing prebuilt copies) is not viable.** IF nodes cost about 7.0 µs each per launch even when nothing copies (E=64: 1.34 ms / 192 IFs; E=128: 2.69 ms / 384 IFs). At 512 experts × 10 slots × 48 layers that is about 1.7 s per token and about 1.47M memcpy nodes (estimate). SWITCH and WHILE don't reduce this.

### Results (same process, medians)

| Method | W1 1×130 rows | W2 48×3 rows |
|---|---|---|
| M0 link ceiling | 12.83 GiB/s | – |
| M2 in-graph kernel | 52.89 ms, 6.33 GiB/s | 76.02 ms (1.584 ms/call), 4.88 GiB/s |
| M3c batch DMA (not capturable) | 26.57 ms, 12.60 GiB/s | 29.01 ms (0.604 ms/call), 12.78 GiB/s |
| **Captured MEMCPY nodes** | **28.41 ms, 11.78 GiB/s** (780 nodes) | **31.43 ms (0.655 ms/call), 11.80 GiB/s** |
| Captured MEMCPY nodes, retargeted before every launch | – | 31.48 ms, 11.78 GiB/s |

A single 3-row launch (18 nodes) took 0.659 ms, so launch overhead is negligible.

### Verdict

- **Viable only with a host sync per MoE layer.** Captured MEMCPY nodes reach 92% of the link inside torch's graph, byte-exact, but choosing their targets needs the routing plan on the host.
- **Integration shape:** per MoE layer, a graph segment up to the routing plan; read the plan device→host; `SetParams1D` on K×6 nodes; disable unused slots (`cuGraphNodeSetEnabled`, `cuda.h:22971`, untested); launch the copies-plus-experts segment. The K_max × 6 per-segment `cuMemcpyHtoDAsync` calls replace `copy_expert_row_segments_gpu`, stay on the graph stream, and must point inside the registered allocations.
- **Net effect (estimate):** W2-shape copies drop to 0.41× of the current kernel, roughly 44 → 18 ms, saving about 26 ms/token. But 48 breaks at E21's Python cost (0.45–2.65 ms each) is 22–127 ms, which eats that. Only a thin C++ break (about 5–10 ms/token, estimate) nets about 15–20 ms/token.
- **Compare with E24:** a faster in-graph kernel needs no break at all. Its correctness-only smoke run (other GPU load present) reached 10.6–11.35 GiB/s, which, if it holds on an exclusive GPU, beats this route.
- **Untested:** `cuGraphNodeSetEnabled`; SGLang's real breakable graph; contention with real decode compute.

## E24 — Warp-aligned in-graph copy kernel (bigger host reads per GPU thread)

- **Question:** can `copy_expert_row_segments_gpu` reach link-class speed inside the CUDA graph by reading more host memory per GPU thread (E22's option (a))?
- **Run:**
  - 09-14 18:16–18:44 CDT on divix01, :7867 down. Detached worktree `cc-copykernel/worktree` at `fa66acb84f`, isolated JIT cache; the only change is a new templated file `expert_cache_transfer_spike.cuh` (grid, block, load width, layout) with distinct symbols.
  - Smoke (18:22, EXIT=0): correctness only; another session's server shared the GPU, so its timings don't count.
  - First screen lost to a bug in the stray-write check (25 GiB allocation, fixed). Re-run screen EXIT=0 at 18:39, final EXIT=0 at 18:44.
  - Every timed run under `flock cc-gpu.lock`, with GPU process snapshots before and after each measurement (`gpu_exclusive` per row).
  - Final: 2048 arena-registered rows, 40 iterations, eager and graph replay, byte-exact on all 6 segments and a stray-write check per row.
- **Artifacts:** divix01 `nvfp4-work/cc-copykernel/` (`REPORT.md`, `final.jsonl`, `screen.jsonl`, `smoke.jsonl`, `bench_ck.py`, `run_screen.sh`, `summarize_ck.py`, `kernel_vs_orig.diff`, `kernel_new_file.diff`, `logs/`).
- **Workloads:** W1 1×130 rows, W2 48×3, W3 48×10 (production-like), W4 48×1.

### Results (final, CUDA-graph replay, median / p90 ms, GiB/s)

| Variant | W1 1×130 | W2 48×3 | W3 48×10 | W4 48×1 |
|---|---|---|---|---|
| Original kernel | 50.94 / 54.34 (6.57) | 73.45 / 76.60 (5.05) | 211.29 / 214.05 (5.85) | 15.38 / 15.71 (8.04) |
| **g8b256u32L2** (32 B loads, contiguous chunk per warp) | **29.41 / 29.43 (11.38)** | **32.39 / 32.40 (11.45)** | **107.45 / 107.49 (11.50)** | **10.95 / 10.96 (11.29)** |
| g8b256u16L1 (16 B loads, interleaved, whole warps per row) | 29.92 / 29.98 (11.19) | 32.58 / 32.59 (11.38) | 107.79 / 107.81 (11.47) | 11.03 / 11.04 (11.21) |
| M0 link ceiling (1 GiB contiguous pinned) | 77.90 ms (12.84) | – | – | – |

- Eager matched graph replay within 0.1 ms.
- The rewrite closes 77% (W1), 82% (W2) and 81% (W3) of the gap to the link, and is within 10–11% of production DMA (E22 M3), which can't be graph-captured (E23).
- Screen: grid 4–32, block 128–512 and interleaved vs contiguous chunks all plateau at ~11.4 GiB/s. Wider loads on the original layout only reach 8.8 GiB/s (W1 smoke). Per-thread contiguous reads were slower than the original.

### Verdict

- **Positive.** The in-graph copy reaches 89% of the link with no graph break and no host sync, byte-exact under replay. It beats E23's captured MEMCPY route, which needs a per-layer break.
- **Why it works:** the original kernel's lanes don't align with the 32-thread warps, so every 4 B host read is its own access. Giving each row whole warps, with a warp's threads reading adjacent host memory, lets the device batch the reads; 16 B or 32 B loads finish the job.
- **Estimated saving:** 22–37 ms per token for ~130 miss rows over 48 layers (estimate): 57 → 29 ms at the W3 shape, 66 → 29 ms at W2, 44 → 22 ms scaling the cost model's figure by the measured ratio. At E20's 14 GiB baseline (s/token ≈ 87.6 ms), the cost-model-consistent 22 ms saving gives roughly 11.4 → 15 tok/s (estimate, unmeasured in serving). The upper 37 ms exceeds the model's 44 ms copy budget's reach and depends on resolving E22's discrepancy.
- **Integration:** change only the lane geometry and load width in `expert_cache_transfer.cuh`. The op signature, `LaunchKernel(8,256)`, `ExpertRowSegments` and `_gather_graph` stay unchanged, with no Python change. Prefer the 16 B `v2.b64` form (hisparse already uses it); the 32 B `v4.b64` form is at most 0.5 ms faster and is used nowhere else. Tests: count 0/1/max, more rows than warps, a misaligned segment, graph replay with a changing count.
- **Caveats:**
  - Synthetic random rows on one GPU; not measured under concurrent decode compute; plan uploads not timed.
  - The remaining ~11% below the link wasn't isolated.
  - E22's discrepancy between the benchmark and the 44 ms cost model is still unreconciled, so the serving gain needs a real decode measurement.

## E25 — Serving A/B of the warp-aligned copy kernel (E24's 16 B variant)

- **Question:** does E24's kernel speed up real decode, and is it correct under real routing?
- **Run:**
  - 09-14 18:58–20:33 CDT on divix01, :7867 down. Detached worktrees at `fa66acb84f`: `base` (unmodified kernel, md5 81ba5d63…), `proto` (only `expert_cache_transfer.cuh` changed to g8b256u16L1, md5 608fce6c…, plus a new test file), `shadow` (proto plus debug shadow copy behind `SGLANG_MOE_SHADOW_COPY_CHECK=1`). Separate JIT caches.
  - Serving flags and expert-stream env copied from `run-nvfp4-e16c-public.sh` (14 GiB cache). One server at a time on 127.0.0.1:31207, each holding `flock cc-gpu.lock` for its life on an otherwise empty GPU; `sglang.__file__` logged from inside each server.
  - Workload: 8 fixed prompts × 512 greedy tokens, same order every run.
  - Waits: another session's server held the GPU at 18:59–19:06. The first base1 launch exited 90 at startup (`sglang/cli/main.py` has no `__main__` guard) without using the GPU; the launcher now calls `main()`.
- **Artifacts:** divix01 `nvfp4-work/cc-copykernel-ab/` (`REPORT.md` md5 72842dcd…, `kernel.patch` md5 7ae646ed…, `shadow_debug.patch`, `run_ab.sh`, `run_followups.sh`, `run-ab-server.sh`, `decode_workload.py`, `margin_ab.py`, `summarize_ab.py`, `shadow_selftest.py`, `negative_check_helper.py`, `servers/`, `logs/`).

### Correctness

- **Kernel tests:** 101 collected, 101 passed, exit 0 (existing expert-cache-transfer tests plus new count 0/1/max, more rows than warps, misaligned segment, graph replay with changing count). The first run had 12 failures from a dtype bug in the test's reference helper. A negative check showed the fixed helper fails on a flipped byte and on a stray write, for all 6 segment types and count 0.
- **Shadow copy in real decode (the gate):** both kernels copied the same miss plan into separate rows inside the decode CUDA graph, compared on the GPU. 196,464 launches (~4,093 forwards × 48 layers), 452,088 real miss rows, **0 differing rows, 0 differing bytes** across 41 snapshots. Launch counts grew by exactly 48 per forward, so the check ran on every replay. Its self-test counted one corrupted byte exactly.
- **Greedy parity can't gate this server:** base1 ≠ base2 ≠ base3 on 8/8 prompts, and proto1 ≠ proto2 likewise.
- **Logprob margins (two-sided, proto2 vs base3 and base_dyn0 vs proto_dyn0):** at every first divergence each run chose the other's runner-up, margin ≤ 0.5 nats. Logprobs come in 0.0625 steps; the median top-1/top-2 margin over all positions is 4.5.
- **Stock-server finding:** base2 (unmodified) flipped at prompt 1, token 2 ('\n' opening a thinking block vs '\n\n' closing `<think>` empty) at a 6.5-nat margin by proto2's logits. The unmodified server has a large nondeterminism source of its own; not investigated.

### Speed (one run per arm)

| Pair | Base ms/token | Proto ms/token | Saved |
|---|---:|---:|---:|
| base1 vs proto1, no logprobs | 74.88 | 57.77 | 17.1 |
| base2 vs proto1, no logprobs | 75.10 | 57.77 | 17.3 |
| base3 vs proto2, logprobs on | 75.81 | 58.72 | 17.1 |
| base_dyn0 vs proto_dyn0, `SGLANG_MOE_HOT_DYNAMIC=0` | 142.22 | 87.74 | 54.5 |

- Production settings: median decode 13.48 / 13.24 → 17.41 tok/s (+29%); TTFT unchanged (1.43 → 1.39 s). Miss load matched across arms: 111.5–111.9 rows (0.288 GiB) per forward.
- Dynamic residency off: 6.77 → 11.01 tok/s (+63%) at 299 rows (0.77–0.785 GiB) per forward. The pre-registered prediction (written 20:20:59, before proto_dyn0 ran) was 45.7 ms; measured 54.5 ms (117%). The saving scales with miss load: 3.2× the saving for 2.7× the load.
- This workload's absolute tok/s differs from E19's 10.37 (different prompts and lengths); only within-experiment deltas count.

### Verdict

- **Positive, cleared.** The kernel is byte-identical to the original under real decode and saves 17.1–17.3 ms/token at production settings, 78% of E24's 22 ms estimate.
- **Cost-model reconciliation (E22):** the implied in-serving cost of the original kernel is about 0.147–0.158 s/GiB (estimate, assuming proto runs at its microbenchmark rate), close to the cost model's 0.1347 s/GiB. E22's microbenchmark overstated the original kernel's small-launch cost.
- **Hand-off:** `kernel.patch` holds only the kernel change and tests, with no Python caller changes, and passes `git apply --check` on the fork at `4b083ae749` (not applied). `shadow_debug.patch` is debug-only.
- **Not run:** repeated runs per arm (no confidence intervals); base self-consistency with dynamic residency off; chat-template or concurrent traffic; per-op copy timing in serving; root cause of the stock server's 6.5-nat flip.
- **End state:** at 20:33:29 no GPU compute process, nothing listening on 31207 or 7867, tmux session gone. The kernel was later committed as `8473a28c88` (101/101 kernel tests at `4b083ae749` + patch) and the serving worktree fast-forwarded to `d0a3e55b16` with :7867 still down.

## E26 — Re-rank the enhancement ideas at the post-E25 copy cost (offline model)

- **Question:** every estimate before E25 priced a miss at the original kernel's cost. At the new cost, which remaining idea is worth building? (Step 1 of `docs/superpowers/plans/2026-09-14-nvfp4-post-copy-kernel-enhancements.md`.)
- **Run:** 09-15 laptop, CPU only, `rerank.py` EXIT=0. Artifacts: divix01 `nvfp4-work/cc-rerank/` (`rerank.py`, `rerank.out`).
- **Model:** per decode token, s = c_miss·D + c_promo·G + F, bytes calibrated ×0.976 (E20). D = in-graph miss GiB, G = DMA promotion GiB.
  - **c_promo = 0.0794 s/GiB:** E22's production DMA rate, 12.6 GiB/s.
  - **c_miss from E25's dynamic-on and dynamic-off runs per kernel** (two equations, c and F unknown): c = (Δs − c_promo·ΔG)/ΔD.
    - Original kernel: (142.22 − 74.88 ms + 0.0794·0.0439 GiB)/(0.7847 − 0.2881 GiB) = **0.1426 s/GiB**.
    - Warp-aligned kernel: (87.74 − 57.77 ms + 0.0794·0.0446 GiB)/(0.7794 − 0.2874 GiB) = **0.0681 s/GiB** (0.48×).
  - **Fit quality:** the two kernels' intercepts on the E25 workload differ by 4.3 ms (30.3 vs 34.7 ms) although only the kernel changed, so treat the model as ±4 ms/token. Each point is one run.
  - **F on the E1 workload:** solving E20's 4,957-slot baseline (11.42 tok/s) with the original kernel's c gives F = 36.4 ms, close to the E16 fit's independent intercept of 35.9 ms.
- **Post-E25 baseline (E1 workload, 4,957 slots, predicted):** 15.83 tok/s = 63.2 ms/token = 22.3 ms miss copies + 4.4 ms promotions + 36.4 ms rest. Each avoided miss saves 0.171 ms (0.358 ms before E25). E25 measured 17.4 tok/s on its own, lighter workload.

### Results (predicted gain vs each era's baseline)

| Idea | Variant | Before E25 (vs 11.42) | After E25 (vs 15.83) | tok/s after |
|---|---|---:|---:|---:|
| 3 FP8 slots | 6,236 slots | +10.6% | +7.3% | 16.99 |
| 3 FP8 slots | 6,466 slots | +12.5% | +8.6% | 17.19 |
| 4 skip low-weight misses | −10% / −20% / −30% / −50% misses | +5.3 / +11.2 / +17.8 / +33.7% | +3.7 / +7.6 / +11.9 / +21.4% | 16.41 / 17.04 / 17.71 / 19.23 |
| 5 CPU experts, in-graph | 4 threads (E21 measured 25.8 ms) | +23.9% | **−7.2%** | 14.69 |
| 5 CPU experts, in-graph | 8 / 18 threads (linear scaling, unmeasured) | +51.5 / +73.0% | +14.4 / +31.5% | 18.12 / 20.82 |
| 2 overlap next-layer copy | recall 0.594 / 0.775 / 0.92 (ceiling) | +42.7 / +64.1 / +86.4% | +26.6 / +37.7 / +48.1% | 20.04 / 21.80 / 23.45 |
| 7 faster host link | Gen4 / Gen5 (copy and promotion time ÷2 / ÷4) | +41.8 / +79.4% | +26.9 / +46.6% | 20.09 / 23.20 |
| 6 last 11% to link | c_miss × 0.89 | n/a | +4.0% | 16.47 |
| 6 promotions | half the promotion time | +4.5% | +3.6% | 16.41 |

- **Recall sources for idea 2:** 0.775 and 0.92 are E2's one-layer lookahead at 10 and 20 candidates; 0.594 is the affinity predictor's recall at M from the expert-prediction shadow smoke (`docs/superpowers/experiments/2026-09-14-expert-prediction-shadow-smoke.md`).
- **The idea 2 rows are a ceiling, not a prediction.** They credit recall × all miss-copy time and assume the copy overlaps compute for free. They ignore: copies of false-positive candidates, the per-layer window (36.4 ms / 48 ≈ 0.76 ms of other work to overlap with), and GPU-thread contention between the copy kernel and compute. E2's earlier 1.10–1.17× cap included such limits.

### Verdict

- **Every copy-saving idea lost about a third of its value**, and CPU experts at 4 threads now lose (−7%). Idea 5 needs ≥8 threads, which the divix01 CPU policy doesn't allow today.
- **Re-ranked software order:**
  1. **Idea 3, FP8 slots (+7–9%):** already implemented (`SGLANG_ONLINE_FP8_GROUPS`); blocked only on the parked GPU accuracy session. Cheapest to ship.
  2. **Idea 2, overlapping copies (ceiling +27–48%):** the largest software lever, but unproven. A copy-kernel-under-decode-load microbenchmark decides whether it's real before any build.
  3. **Idea 4, skipping low-weight misses (+8–12% at 20–30% fewer misses):** needs the gate-weight distribution of misses (the new expert-prediction capture mode records routing) and a quality gate.
  4. **Idea 6, small items (~+4% each).**
  5. **Idea 5, CPU experts:** only with more cores.
- **Hardware (idea 7):** a Gen4 host is worth about +27% and Gen5 about +47%, more than any software idea except the idea 2 ceiling.
- **Caveats:** one run per E25 point; the ±4 ms intercept gap; the E1 workload only; CPU thread scaling beyond 4 is unmeasured; skip fractions are assumptions, not measured.

## E27 — Copy kernel under decode load: does an overlapped next-layer copy contend with compute?

- **Question:** step 2 of the enhancements plan. If layer L+1's miss rows are copied on a side stream while layer L computes, how much of the sequential copy time is actually saved?
- **Pre-registered verdict** on the realized saving (T_seq − T_ovl)/T_copy at 3 rows/layer:
  - ≥50% ALIVE (≈7.7 ms/token at recall 0.7, ≈+13%);
  - 20–50% MARGINAL;
  - <20% DROP.
- **Run:** 09-15 00:18–01:02 CDT, divix01 RTX 5090, exclusive GPU under `cc-gpu.lock`, 7867 down.
  - Artifacts: `work/cc-overlap/` (`REPORT.md`, `bench_overlap.py`, `results.jsonl`, `summary.md`, `cal.jsonl`, `probe_wait.jsonl`, `run.log`).
  - All final runs EXIT=0. A first run was killed for two harness defects (graph `seq` was secretly overlapped; stateful layers made the hidden-state check non-discriminating) and kept as `results_invalid_run1.jsonl`.
- **Copy:** production `copy_expert_row_segments_gpu` from the serving worktree at `d0a3e55b16` (includes `8473a28c88`). Sources are arena-registered rows in the real 6-segment layout, repeated with plain pinned rows.
- **Compute (load level 2, not the real decoder layer):** 48 per-layer CUDA graphs, each containing:
  - the overlay's CUTLASS NVFP4 fused MoE, top-10, bs 1, reading the slots the copy writes;
  - bf16 router, shared expert, hyper-connection, linear-attention and every-4th full-attention ops at checkpoint shapes, with random weights.
  - Real kernels alone take 0.25 ms/layer; 50 dense pads calibrate to 0.764 ms/layer (E26's 36.4 ms ÷ 48).
- **Schedules:**
  - `seq` = copy then compute on the main stream.
  - `ovl` = launch copy(L+1) on the side stream, replay compute(L), then a device-side `wait_stream`.
  - 4 chained tokens × 30 samples, wall time bracketed by synchronize.

### Results (p50, per token; eager / CUDA graph)

| Compute per layer | Rows/layer | Copy GiB/s alone / under compute | Compute ms alone / under copy | seq ms | ovl ms | Realized saving |
|---|---:|---|---|---:|---:|---:|
| 0.76 ms (calibrated) | 1 | 11.20 / 10.89 | 0.765 / 0.765 | 47.9 / 48.2 | 37.1 / 37.3 | 98.0 / 98.2% |
| 0.76 ms (calibrated) | **3** | 11.37 / 11.20 | 0.786 / 0.786 | 69.6 / 70.0 | 37.8 / 38.0 | **97.6 / 97.8%** |
| 0.76 ms (calibrated) | 10 | 11.46 / 11.44 | 0.790 / 0.790 | 144.6 / 145.0 | 108.2 / 108.4 | 33.8 / 34.0% |
| 0.76 ms (calibrated) | 30 | 11.48 / 11.48 | 0.793 / 0.794 | 360.0 / 360.3 | 323.4 / 323.5 | 11.3 / 11.4% |
| 0.25 ms (real kernels only) | 1 | 11.20 / 10.07 | 0.253 / 0.253 | 22.7 / 23.0 | 12.1 / 12.3 | 95.7 / 96.1% |
| 0.25 ms (real kernels only) | 3 | 11.37 / 11.30 | 0.256 / 0.254 | 44.3 / 44.7 | 32.9 / 33.0 | 35.0 / 35.6% |
| 0.25 ms (real kernels only) | 10 | 11.46 / 11.44 | 0.257 / 0.256 | 119.6 / 119.9 | 108.1 / 108.3 | 10.6 / 10.8% |
| 0.25 ms (real kernels only) | 30 | 11.48 / 11.47 | 0.260 / 0.260 | 334.9 / 335.3 | 323.3 / 323.4 | 3.6 / 3.7% |

- **No measurable contention either way.** Compute under a running copy is within ±0.01 ms of compute alone, paired or saturated. Copy speed under compute is within ~0.1 GiB/s of copy alone at 3–30 rows.
- **The saving equals the no-contention ideal, min(copy, window)/copy, within ~2 points.** The per-layer window is the whole answer.
- **Eager and graph are identical.** A copy graph captured on the side stream replays on whichever stream is current at replay.
- **Arena rows and pinned rows are identical.**
- **Correctness, all 20 records:**
  - every copied row in all 48 layers is byte-exact;
  - the overlapped hidden state is bitwise equal to sequential;
  - sequential is deterministic;
  - count 0 changes the hidden state, which proves the equality check can fail.
  - The wait probe's negative control (no wait) sees 3.4–4.8% of the bytes.
- **Not run:** the launch-width sweep (256–2,048 threads) and the host-thread DMA arm were requested after this batch had started.

### Verdict

- **ALIVE on the pre-registered metric** (97.6% eager / 97.8% graph at 3 rows/layer with a 0.76 ms window). It is **conditional on the window**: with only the real batch-1 kernels (0.25 ms/layer) the same point is 35%, MARGINAL.
- **Contention is not the risk.** The in-graph kernel on a side stream costs compute nothing, so moving the copy off the compute cores (a CPU doorbell thread on the copy engine) buys no contention relief. Its only GPU-side edge is link rate: DMA at 12.6 vs 11.4 GiB/s (E22/E24).
- **What decides the size in serving:**
  - How much GPU time a real decode layer takes. The decode graph is one captured graph, so host time between layers is not in the window. E26's 0.76 ms/layer includes LM head, sampling and host work.
  - Burstiness: each layer saves only min(copy, window), so skew lowers the saving below the 3-row point.
  - Lookahead depth: launching copies two or more layers ahead widens the window, at lower recall.
- **Next:** measure the real per-layer GPU timeline in serving (profile a decode step of the production graph), then re-price idea 2 with the measured window, recall and the E1 per-layer miss distribution.
- **Caveats:** proxy non-MoE kernels, random weights; plans are exact rows (no recall, no false-positive copies); promotions and multi-request traffic not modeled.

### Addendum: dedicated copy paths (arms W and H, 09-15 01:15–01:45)

- **Run:** same harness, calibrated 0.76 ms/layer compute, arena rows, 3 and 10 rows/layer. `bench_arms.py`, `results_arms.jsonl`, `summary_arms.md`, report addendum in `REPORT.md`. `arms_full` EXIT=0; 28/28 records pass all four correctness checks.
  - Width variants live in `work/cc-overlap/worktree` (detached `d0a3e55b16` plus an untracked `expert_cache_transfer_width.cuh`).
  - The in-run production-kernel control saved 97.5% eager / 97.6% graph at 3 rows.
- **Arm W (narrower in-graph launch, 256/512/1,024/2,048 threads):** no contention to remove at any width (compute within ±0.01 ms). Narrower launches are slower, not link-limited.

  | Width | GiB/s 3 rows | ovl ms/token 3 rows | Saving 3 rows | GiB/s 10 rows | ovl ms/token 10 rows | Saving 10 rows |
  |---:|---:|---:|---:|---:|---:|---:|
  | 256 | 3.2 | 116.3 | 31.0% | 8.0 | 156.8 | 22.5% |
  | 512 | 7.7 | 49.3 | 74.3% | 5.3 | 236.0 | 15.1% |
  | 1,024 | 11.0 | 37.65 | 98.1% | 11.3 | 110.6 | 32.7% |
  | 2,048 | 11.3 | 37.77 | 97.6% | 11.45 | 108.4 | 33.6% |

  Keep 2,048. Speed follows 32-thread warps per row, which is why 256 beats 512 at 10 rows.
- **Arm H (Python host thread on its own stream, eager copy, compute still 48 captured per-layer graphs):**

  | Path | GiB/s 3 / 10 rows | ovl ms/token 3 / 10 rows | Request→landed 3 / 10 rows |
  |---|---|---|---|
  | Host thread, production DMA op | 12.6 / 12.75 | 38.5 / 97.2 | 0.62 / 2.03 ms |
  | Host thread, per-row `non_blocking` copies | 12.0 / 12.1 | 37.9–38.1 / 102.9–103.2 | – |
  | In-graph kernel (control) | 11.3 / 11.45 | 37.75 / 108.4 | 0.70 / 2.26 ms |

  - **Readings:**
    - The DMA host path is 0.75 ms/token slower than the kernel at 3 rows, because ~0.3 ms of host launch time per copy eats the window. It is 11 ms/token faster at 10 rows, because the link is faster.
    - Under back-to-back compute the host copy drops to 9.9–10.2 GiB/s at 3 rows, and compute is 1–2% slower whenever the thread exists.
    - A Python busy-spin was slightly worse (GIL). A native thread was not measured.
  - **Ordering matters:** main blocked until the host copy was launched, and only then replayed layer L's graph, so every copy was queued *before* its replay launched. The doorbell prototype found the opposite case fails: host copies queued *after* a replay launched were held until that replay ended (18/18 in-graph waits timed out). In serving, the predicted rows are only known inside the replay, so arm H is a best case that also ignores the GPU→host readback.
- **Verdict unchanged:** ALIVE, window-bound. The in-graph 2,048-thread kernel is the practical overlap path. A host-driven path gains link rate only at ≥10 rows/layer, and only if its copies can be queued before the replay that must overlap them.

## E28 — Serving decode profile: the real per-layer window and where decode time goes

- **Question:** E27 showed idea 2's saving is min(copy, window)/copy per layer. How long is the real window in serving, how are misses spread across layers, and where does the rest of a decode step go?
- **Run:** 09-15 01:45–02:00 CDT, divix01, `work/cc-decode-profile/` (`REPORT.md`, `overhead.md`, `analyze.py`, `gaps.py`, `livemiss.py`, `miss_replay.py`, `runs/nsys-20260915-014557/`, `runs/plain-20260915-015447/`, `miss/miss-14g.json`). `session.log` EXIT=0, GPU released after each arm. 7867 stayed down.
- **Config:** a copy of `run-nvfp4-e16c-public.sh` at `d0a3e55b16`, differing only in:
  - port 7897 on 127.0.0.1;
  - own run and log dirs;
  - taskset to NUMA node 0;
  - an nsys wrapper in the profiled arm.
- **Workload:** greedy, 330 decode tokens, prompts of 126 and 5,501 tokens.
- **Profiler:** Nsight Systems 2026.3.2 with `--cuda-graph-trace=node`, 200-step captures via `/start_profile`. A second arm ran with no profiler.
- **Segmentation check:**
  - every step has exactly 48 copy / 48 expand / 48 finalize kernels in launch order, and 12 `kernel_mha` at layers 3, 7, …, 47;
  - capture step periods agree with client inter-token latency within 1.5%.
- **Profiler overhead is large and host-side:** +18% to +52% mean ITL on the same token window, −1.7% on one request. Routing also diverged between arms (94–98 vs 136–150 misses/token), so the profiled host-side times are upper bounds. GPU kernel times in the graph are unaffected.

### Results

- **Decode is one captured graph with no breaks** (`segments=1 breaks=0`), so no host time sits inside any layer.
  - Order within a layer: norm, attention, shared expert, router, route planning (0.21 ms) → the layer's miss copy → MoE kernels (0.05 ms, 10 µs after the copy).
- **The overlap window, copy(L) end → copy(L+1) start, is 0.268 ms p50** (p10 0.265, p90 0.291):
  - 0.267 ms before a linear-attention layer, 0.290 before a full-attention layer;
  - no material change from 126 to 5,501 tokens of context.
  - Whole-token non-copy GPU time is 13.2 ms, plus 0.83 ms after layer 47 (LM head 0.80).
- **Per-layer miss copies:**
  - the kernel costs 0.2239 ms/row + 0.006 ms (11.5 GiB/s), fitted on live kernel durations;
  - a layer's copy is 0.68 ms p50 (linear) and 0.45 ms p50 (full), about 33 ms/token mean in the profiled arm.
- **Misses per layer per token** (E1 trace replayed at 4,957 slots, policy 4/1/2/0; reproduces E20's 130.2/token):

  | Stat | Value |
  |---|---|
  | Mean | 2.72 |
  | p50 / p90 / p99 / max | 2 / 6 / 9 / 10 |
  | Share of cells: 0 misses | 16.8% |
  | Share of cells: 1–3 misses | 51.8% |
  | Share of cells: 4–10 misses | 31.4% |
  | Full-attention layers | 2.11 |
  | Linear-attention layers | 2.93 |
  | Live profile mean | 3.04 |

- **Step pattern:** every 4th decode step is a residency update (exactly spacing 4 in 49/49 and 48/48 captured steps).
  - Normal steps take ~54 ms: 3.5 ms graph launch call, ~44 ms graph, ~4 ms scheduling and input prep.
  - Update steps have a host gap of 106 / 101 ms p50 (short / long), 33–34% of all decode time in the profiled arm.
  - About 30 ms of that gap is visible CUDA work: 160–180 promotion rows on a side stream, stream synchronizes, memcpy waits.
  - The other ~70 ms is unattributed host CPU (no Python sampling).
- **The stall is not a profiler artifact.** Client ITL in the no-profiler arm, tokens 21–330:
  - one slow token every 4 (spacing 4 dominates);
  - slow tokens 105–130 ms p50 against 42–50 ms for normal tokens;
  - slow tokens hold 38–47% of decode wall time.
  - The excess over a normal token is roughly 60–80 ms every 4 tokens, about 15–20 ms/token on average (estimate from ITL, not from a profile).

### Re-pricing idea 2 with the measured window (offline, laptop)

- **Per layer, saving = min(recall·copy(n), 0.268 ms)**, with copy(n) = 0.2239·n + 0.006 ms, over the replay histogram, for 47 layers (layer 0's routes are only known inside the graph).

  | Recall | Saving per token |
  |---|---|
  | 1.0 | 10.1 ms |
  | 0.775 | 9.6 ms |
  | 0.594 | 9.3 ms |

  - The copy total is 29.5 ms/token, so overlap recovers about a third of it.
  - Recall barely matters because the window covers only ~1.2 rows.
  - The hard cap is the 13.2 ms of non-copy GPU time per token.
  - False-positive copies are not charged: they eat the same window and can delay the next layer, so these are ceilings.
- **At the no-profiler arm's ~60–70 ms/token, ~10 ms is +15–17%** (estimate). That is below E26's +27–48% ceiling and roughly level with FP8 slots.

### Verdict

- **Idea 2 is MARGINAL-to-modest in real serving:** window 0.268 ms, not 0.76 ms; a ~10 ms/token ceiling; it needs a layer-ahead predictor, new in-graph code, and handling of false-positive copies.
- **The residency-update step is the largest single cost found so far.** Every 4th step stalls the host ~60–80 ms beyond a normal step, of which ~70 ms (profiled) is unattributed host CPU. Removing most of that stall is worth roughly 15–20 ms/token (estimate, +25–40%), more than any idea on the E26 list.
- **Next:** profile the update step's host CPU (py-spy or Python sampling on the no-profiler config) to attribute the ~70 ms, then decide between cheaper, asynchronous or less frequent updates.
- **Caveats:** one capture per context; profiler overhead on host times; server pinned to node 0 (production unpinned); long context 5,501, not ~4,000 tokens; the E26 cost model's "36.4 ms rest" is now known to hide the update stall and the 13.2 ms of non-copy GPU time.

## E29 — Doorbell copier prototype: GPU posts expert-row requests to pinned memory, a native CPU thread copies them

- **Question:** can the GPU request miss rows through a pinned page, with a C++ spin thread issuing the copies on the copy engine, and a GPU waiter holding the layer until they land? The goal is no host sync, no graph break, and overlap with compute inside the decode CUDA graph.
- **Run:** 09-15, divix01 RTX 5090, under `cc-gpu.lock`. Artifacts in `work/cc-doorbell/`: `REPORT.md`, `doorbell.patch`, `bench_doorbell.py`, `results/bench1.jsonl`, `probe_*.py`, `logs/`, `run.log`.
  - Code lives in `work/cc-doorbell/worktree`, detached at `d0a3e55b16`, as uncommitted new files: `kernels/jit/csrc/moe/expert_doorbell.cuh`, `kernels/ops/moe/expert_doorbell.py`, `test/registered/unit/kernels/test_expert_doorbell_copier.py`.
  - The first round (the numbers below) was authored in that divix01 worktree. A second round followed the requested flow: a local worktree `sglang-nvfp4-worktrees/doorbell` on branch `cc/doorbell-prototype`, pushed to `shared` at `9a95819eac` ("doorbell head-store mode, quiesce, and hold isolation probe"), and tested on divix01 (`br2_*` in `run.log`, `br2_pytest` EXIT=0). Final results are in Round 2 below.
- **Design:**
  - **Request page:** a pinned page holding `head` and `abandoned` words plus a ring of `{seq, count, tag, rows int64[cap], slots int32[cap]}` records.
  - **Poster:** a 1×32 in-graph poster kernel writes the record and publishes `head` last.
  - **Copy thread:** a C++ thread busy-spins on core 71 with no GIL. It issues one `cudaMemcpyBatchAsync` per request on its own stream, then a 4-byte `done` copy queued behind the rows.
  - **Waiter:** a chain of short GPU poll launches (the krasis `__nanosleep(64)` pattern).
    - Chunked because one long waiter launch held any copy queued after it.
    - `cuStreamWaitValue32` was rejected: 801 NOT_SUPPORTED on the 5090, 900 inside torch graph capture, no timeout.
  - **Timeout:** about 512 ms, then about 1 ms in degraded mode. On timeout the waiter marks the request abandoned and falls back to `copy_expert_row_segments_gpu_kernel` from the record.
- **Tests:** 51 collected, 50 passed, 1 failed, EXIT=1 (green5).
  - The failure is `test_posted_rows_match_reference[0]` on `timeouts == 0`, with correct bytes. The first doorbell request in a process can miss its whole wait chain; the cause is unresolved.
  - The negative check can fail: the byte-reference helper flags a flipped byte and a stray write.
  - The RED run failed 46/47 on the missing module.
- **Latency:** 40 iterations, every one byte-exact.

  | Rows | In-graph kernel p50 ms (GiB/s) | Doorbell, eager post and compute, p50 ms | Post → thread sees request, p50 µs | Doorbell inside a graph, p50 ms (all fallback) |
  |---:|---|---:|---:|---:|
  | 1 | 0.243 (10.6) | 0.247 | 28.7 | 1.300 |
  | 3 | 0.697 (11.1) | 0.660 | 30.4 | 1.750 |
  | 10 | 2.274 (11.3) | 2.133 | 60.3 | 3.332 |
  | 30 | 6.761 (11.4) | 6.278 | 71.1 | 7.818 |

  - Batch memcpy after the thread sees the request runs at about 12.7 GiB/s at 30 rows (1.08× the kernel).
  - Waiting on an already-finished request costs 15–25 µs.
  - `PreferOverlapWithCompute` copies ~5.5× faster under main-stream compute, but defers main-stream events behind the thread's copy.
  - The batch memcpy call blocks the thread for the copy's duration from the second request onward.

### Verdict

- **Negative for serving on this machine.** While a CUDA-graph replay runs, the thread's host-issued copies do not run until that replay's GPU work ends.
  - Post plus ~4 ms of in-graph compute: the copy finished at 3.99 ms and compute at 4.00 ms (6/6).
  - The same work run eagerly: the copy finished at 0.36 ms.
  - Every in-graph wait timed out (45/45 per variant) and paid waiter chain plus fallback. Inside a graph the doorbell is slower than the kernel alone.
  - The hold is independent of poll mode and of whether post and wait share a replay. `test_graph_launch_holds_thread_copies_until_launch_ends` asserts it.
- **Consistent with E27 arm H:** host copies overlap a replay only when queued before it launches. Production decode is one unbroken graph (E28), so a doorbell copy can never be queued ahead of the compute it should overlap.
- **Moot anyway:** E27 found no contention for the in-graph kernel, and E28 measured a 0.268 ms window. The in-graph kernel is the overlap path if idea 2 proceeds.
- **Open:** the first-request timeout; slot reservation across timeouts; the single `abandoned` word across tags.

### Round 2 (final): hold isolated to the doorbell thread's copy path

- **Code:** built in the local worktree `sglang-nvfp4-worktrees/doorbell`. Branch `cc/doorbell-prototype` on `shared`, three commits: `867da96dfc`, `9a95819eac` (head-store mode, `quiesce()`, isolation cases), `d57db3e1a1` (probe-only). Tested detached on divix01 in `work/cc-doorbell/worktree`. Report: `work/cc-doorbell/REPORT.md` (copy at `sglang-nvfp4-worktrees/REPORT.md`).
- **Tests:**
  - divix01 `br2_pytest` at `9a95819eac`: 54 collected, 54 passed, EXIT=0. The first-request timeout no longer fails.
  - Negative check: flipped byte → `segment 0`, stray write → `segment 4`.
  - Laptop 3060 is a compile gate only: 51 collected, 42 passed, 9 failed. That is 7 wait timeouts, 1 overlap-timing assertion, and 1 arena test that can't import without a local `common_ops`.
- **Correction to round 1:** a compute-only CUDA graph does **not** hold a host-issued torch side-stream copy queued after its launch.
  - One 4.4 ms replay: extra time beyond an idle copy ≈0.01 ms p50, at both 3 and 30 rows.
  - 48 short replays: a copy queued after replay 0 lands its own duration after layer 0 ends (0.6 ms at 3 rows, 4.9 ms at 30).
  - This matches E27 arm H.
- **What does hold: the doorbell thread's own copies**, issued with `cudaMemcpyBatchAsync` on a stream the C++ thread created itself. Measured over 48 layers × 3 rows × 20 tokens, with compute-only layers at 40 ms/token:

  | Case | Timeouts per token (of 48) | Request done after its layer ended, p50 | Token p50 |
  |---|---:|---:|---:|
  | Post and wait in graph | 48 | 90 ms | 181 ms |
  | Post in graph, eager waits after the token | 46 | 147 ms | 181 ms |
  | Volatile head store | 48 | 90 ms | 181 ms |
  | Noncoherent poll | 48 | 79 ms | 156 ms |
  | `PreferOverlapWithCompute` off | 46 | 147 ms | 180 ms |
  | Compute-only graphs, post launched eagerly between replays | 45 | 144 ms | 179 ms |
  | Host-gated (thread enqueues request j+1 before replay j+1) | 0 | before its layer ended | 74 ms |

  - Requests land near the end of **all** replays launched after them, not only the current replay.
  - Ruled out: graph contents, head store, poll kind, the overlap flag.
  - Not yet tried: the thread issuing `cudaMemcpyAsync` on a torch-created stream.
- **Capture caveat:** one capture failed with `cudaErrorStreamCaptureInvalidated` while the thread was busy. `quiesce()` around every capture fixed it.
- **Latency at `d57db3e1a1` (`br3_bench`):** same shape as round 1.
  - Eager doorbell vs in-graph kernel: 0.663 vs 0.693 ms at 3 rows, 6.273 vs 6.758 ms at 30 rows (≈12.7 vs 11.4 GiB/s).
  - Reaction time 29–75 µs.
  - Inside a graph, always the fallback path.

### Final verdict

- **Negative for serving as built.** Only the host-gated form overlaps, and its per-layer host step makes tokens 74 ms against 40 ms of compute. The hold lies in the thread's own stream and copy path, not in graph replay.
- **Overlap is still possible from a host thread in principle,** with plain torch side-stream copies (arm H). That still needs the plan on the host, which means a readback from inside an unbroken decode graph. The in-graph kernel (E27) needs neither.
- **Not pursued further:** E28's 0.268 ms window caps any overlap at ~10 ms/token, and the residency-update stall (E30) is the larger target.

## E30 — Residency-update host stall attribution

- **Question:** E28 found every 4th decode step stalls the host ~60–80 ms beyond a normal step. Where does that time go, and what removes it?
- **Run:** 09-15 02:15–02:39 CDT, divix01, `work/cc-update-stall/` (`REPORT.md`, `probe.py`, `itl.py`, `pyspy_steps.py`, `microbench.py`, `runs/{plain,instr,pyspy}-*`). Session EXIT=0 and GPU released; 7867 stayed down.
- **Setup:**
  - E28's `serve.sh` unchanged: E16c policy 4/1/2/0, 14 GiB hot cache, graph gather, DMA, port 7897.
  - Three arms: plain; instrumented (`perf_counter_ns` spans in `work/cc-update-stall/worktree`, detached `d0a3e55b16` with probe edits in `expert_hot_cache.py`, `expert_residency.py`, `expert_transfer.py`); and py-spy 0.4.2.
  - Greedy, 330 decode tokens, prompts of 126 and 5,501 tokens.
- **Measured process:** the `sglang::scheduler` main thread, which runs the forward, the expert-distribution hook that performs the update, and sampling.
- **Overhead:**
  - Instrumented vs plain mean ITL: +9.7 / −9.4 / +14.8%.
  - Excess per update step: 71–81 ms instrumented vs 65–76 ms plain.
  - Absolute instrumented times may run up to ~10 ms high; shares are reliable.
  - py-spy native sampling at 500 Hz overloaded, so py-spy is used for ranking only.
  - No GC inside analyzed steps.

### Results (instrumented, 233 update steps vs 688 normal; promotions per update p10/p50/p90 = 29/61/117)

| Component (ms per update step) | p50 | p90 | mean |
|---|---:|---:|---:|
| **Update total** (normal step: 0.51) | **76.0** | **109.3** | **77.9** |
| `advance` + `resident_experts`, 48 layers | 3.0 | 3.2 | 3.0 |
| `decide`, 48 layers | 19.7 | 21.2 | 20.1 |
| … rank sort over 512 experts (`expert_residency.py:163–168`) | 9.2 | 9.8 | 9.5 |
| … `_select_desired` | 3.0 | 3.7 | 3.1 |
| … score readback (GPU wait / `.cpu()` / `.tolist()`) | 0.8 / 1.2 / 0.7 | | |
| … O(C²) eviction membership (`:174–176`) and other | ~3.8 | | |
| `reassign`, 32.6 changed layers | 50.4 | 82.8 | 53.1 |
| … retire loop (incl. mapping publishes) | 8.7 | 15.2 | 9.3 |
| … reserve | 2.7 | 4.6 | 2.9 |
| … `set_rows` plan upload (`expert_transfer.py:85–120`) | 6.3 | 8.1 | 6.1 |
| … copy submission, host side of 6 copy ops | 7.8 | 10.4 | 7.7 |
| … copy submission, GPU wait for the promotion copy | 7.3 | 16.1 | 8.5 |
| … copy submission, other plumbing | ~4.8 | | |
| … `publish_ready` (incl. mapping publishes) | 8.1 | 14.4 | 8.6 |
| Across the above: `_publish_mapping`, 137 calls (`expert_hot_cache.py:159–166`) | 12.5 | 22.6 | 13.5 |
| Across the above: `_set_slot_state` scalar GPU writes, 274 calls (`:168–170`) | 4.8 | 8.6 | 5.2 |

- **Split of the 77.9 ms mean:**
  - GPU waits ≈11.5 ms (15%), of which 8.5 ms is the promotion copy.
  - Synchronous tiny H2D/D2H transfers from Python ≈17–23 ms (22–30%): 137 mapping uploads, 274 scalar slot writes, 48 score readbacks, plan uploads.
  - Pure host CPU (Python, torch dispatch, launches) ≈42–48 ms (55–60%).
  - The py-spy captures rank the same hotspots.
- **Scaling:** update total = 35.7 ms + 0.616 ms × promotions (r = 0.96). Per layer: 0.455 ms p50 with no promotions, fit 0.73 + 0.617 × promotions. At p50 the fixed and per-promotion parts are about half each; at p90 the per-promotion part is 72 of 108 ms.
- **Inherent:** only ~170 MB of promotion bytes per update (≈14 ms of PCIe, 8.5 ms of which blocks the host today), one score readback, and deciding the desired set. The rest is per-row and per-layer Python plumbing.

### Fix candidates (all savings are estimates)

| # | Change | Location (`d0a3e55b16`) | Est. saving per update |
|---|---|---|---|
| 1 | Vectorize `decide`/`advance` across 48 layers: one `[48, 512]` score tensor, one top-k plus one readback, set-based evictions | `expert_hot_cache.py:1198–1206`; `expert_residency.py:143–149`, `:158–185`, `:214–250` | ~18–20 ms |
| 2 | Publish expert→slot mapping once per changed layer (device `index_copy_`), not per ticket | `expert_hot_cache.py:159–166`, via `:260`, `:292`, `:369`, `:399–401` | ~10–12 ms |
| 3 | Batch slot state and generation writes, one pinned copy per layer | `expert_hot_cache.py:168–170`, `:241–244`, `:252`, `:259`, `:270`, `:291` | ~5–6 ms |
| 4 | Persistent pinned plan buffers with a `non_blocking` copy; drop full-capacity `zero_`; ideally one plan for all layers | `expert_transfer.py:85–129`; `expert_hot_cache.py:338–342` | ~5 ms |
| 5 | Cache copy-path validation per (layer, tensor); call the DMA op directly with precomputed ranges | `expert_transfer.py:330–523`, `:592–597`; `expert_dma.py:34–75`, `:107–` | ~6–9 ms |
| 6 | Asynchronous promotions: publish slots on a later forward once the ticket completes (RESERVED→LOADING→READY already exists) | `expert_hot_cache.py:366–370`; `expert_transfer.py:258–274` | ≤8.5 ms; PCIe contention with miss copies unmeasured |
| 7 | Bulk `swap(layer, evict_slots, promote_experts)` replacing retire/reserve/begin/publish bookkeeping | `expert_hot_cache.py:190–293`, `:377–413` | ~4–6 ms after #2–#3 |
| 8 | Rotate 12 layers per forward instead of 48 every 4th forward | `expert_hot_cache.py:1198`; `expert_residency_clock.py:84–111` | 0 mean; cuts the ITL spike to ~¼ |
| 9 | Move the update off the decode thread | same | limited by the GIL until #1–#5 land |

### Verdict

- **Attributed.** The stall is the residency update on the scheduler's decode thread: 76 ms p50 / 109 ms p90 per update. It is mostly per-layer and per-row Python bookkeeping and synchronous tiny transfers (≈85%); waiting on the promotion copy is only 15%.
- **Fixes #1–#6 are estimated to remove ≈50 of ≈78 ms per update:** ≈12 ms/token of mean ITL, or ≈+20% decode tok/s at the plain arm's ~64 ms. That estimate is below E28's rough +25–40%, which assumed most of the stall could go.
- **All are code-local to the hot-cache/residency/transfer modules.** None changes routing or model output, apart from #6's one-forward-later slot publication, which changes which slots are hits.
- **Caveats:** instrumented absolute times may be up to ~10 ms high; routing and promotions diverged between arms; savings are estimated from the breakdown and CPU microbenchmarks, not implemented.

## E31 — Residency update inside the decode CUDA graph (plus the Python-path fixes), serving A/B

- **Question:** E30 attributed the every-4th-step stall to Python bookkeeping in the residency update. How much is gained by:
  - (a) E30's Python fixes #1–#6 (`fix`);
  - (b) moving the whole update onto the GPU, inside the decode graph (`gpu`)?
- **Code:** branch `cc/residency-update-fast` on `shared`, head `252a888559`, based on `d0a3e55b16`.
  - `e493c6a9cc`, `0ef30a6931`: fixes #1–#6. With the flag off this is the live path.
  - `afb18e82ac`: `decide_residency_on_device`, `GpuResidencyUpdater` in `expert_residency_gpu.py`, hooks, flags, tests.
  - `b7f96b5d4f`: test helper.
  - `8839a08ac0`: K+1 swap window so the truncation counter can fire.
  - `252a888559`: review follow-ups: idle flush, count-only when updates are off, exact decay table, ownership guards, and runner guards refusing DP attention, speculative decoding and decode graph bs > 1.
  - Report: `work/cc-update-fix/REPORT.md`. Laptop worktree: `sglang-nvfp4-worktrees/residency-update`.
- **Design (flag `SGLANG_MOE_GPU_RESIDENCY_UPDATE`, default off):**
  - Scores, route counts, the expert→slot mapping, slot state and generations live in `[48, …]` device banks. Caches and policies are rebound to row views, so captured pointers stay stable.
  - Decide is a sorted prefix with composite int64 keys (score bits, then 65535 − id): the old swap loop's exact tie order with no readback.
  - At most K = `SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS` promotions per layer (default 64).
  - A decode update applies decay from a token-indexed table, runs a masked decide, scatters mapping and state, bumps generations, then copies the promotions into hot slots with the in-graph `copy_expert_row_segments_gpu`.
  - It runs inside the first streamed layer's graph gather of the forward after a boundary, before any gather reads the mapping. A device pending flag and counters drive the trigger.
  - Prefill flushes a pending boundary before its first gather and applies qualifying prefill boundaries with the same device update, eager and uncapped.
  - With the flag on, the host lists are not authoritative, so the Python path runs only at startup to populate the seed.
- **Tests (divix01).** Collected counts never dropped (145 → 172 → 175).
  - Existing suite on base: 145 passed, EXIT=0.
  - RED on base: device-decide file import error (EXIT=2), GPU file 5/5 failed (EXIT=1).
  - `gpu6_all` at `252a888559`: 175 passed, EXIT=0. This includes a captured 3-layer decode graph replayed over 17 forwards plus prefills against the Python manager, with byte-exact slot rows.
  - Negative checks: slot-row helpers raise on a flipped byte, a wrong mapping entry and a wrong alpha, at three SHAs. The strict truncation test fails on the old decide. The review tests fail 2/8 on `8839a08ac0`.
  - Prefix equivalence was checked exhaustively on a 5-expert layer (every score vector on 4 tied levels × resident set × capacity × 4 margin/sigma pairs) and on random 48×512 layers.
  - An independent code review found 0 critical, 2 major, 1 medium and 4 minor issues. All were fixed or guarded except metrics folding.
- **Replay at `252a888559`** (E1 trace, CPU, real `GpuResidencyUpdater` in lockstep with base):
  - Uncapped: identical before all 2,148 forwards (resident mask, scores), 47,691 promotions, counters and clock.
  - K = 64: 47,690 promotions, 1 truncated layer, 72.8816% vs 72.8817% hit rate, 130.168 misses/step.
  - K sizing: decode per-layer promotions p99 9, p99.9 54, max 68.
- **Serving A/B (`ab2`):**
  - E28/E30 workload: greedy, 330 decode tokens, prompts of 126 and 5,501 tokens, first 20 tokens dropped, plus a top-2 logprob pass.
  - Order base, fix, gpu, gpu, fix, base, port 7897, under the lock.
  - All drives EXIT=0 and session EXIT=0. `SERVER_EXIT=137` is the harness's stop; the GPU was released to 62 MiB after each arm.
  - Artifacts: `work/cc-update-fix/ab/ab2/{ab_summary.md, margins.md, runs.txt}`, `runs/<arm>-<stamp>/`.

| Metric (mean of 2 runs) | base `d0a3e55b16` | fix (flag off) | gpu (flag on, K=64) |
|---|---:|---:|---:|
| Mean ITL (ms) | 67.27 | 58.09 (−13.6%) | **55.08 (−18.1%)** |
| Update-step ITL p50 / p90 (ms) | 117.9 / 158.3 | 80.8 / 109.0 | **65.3 / 88.7** |
| Normal-step ITL p50 (ms) | 47.31 | 48.27 | 49.00 |
| Decode tok/s | 14.87 | 17.22 (+15.8%) | **18.18 (+22.3%)** |
| Hit rate | 0.688 | 0.689 | 0.685 |
| Misses per token | 149.5 | 149.5 | 151.0 |
| Promotions per update | 89.8 | 90.4 | 88.4 |
| Update excess p50 (update − normal) | 70.5 | 32.5 | 16.3 |
| … promotion copy estimate (rows × 0.2239 ms) | 20.1 | 20.3 | 19.8 |
| … other | 50.4 | 12.3 | −3.5 |

- **Per run:**
  - base 14.58 / 15.16 tok/s, fix 16.98 / 17.45, gpu 18.91 / 17.46;
  - hit rate: base 0.685 / 0.692, gpu 0.693 / 0.678;
  - 1 truncated layer (layer 2) per gpu run, over 425 boundaries per layer.
- **Logprob margins:** 30 pair×request comparisons, 0 large-margin flips. Every first divergence has at least one side ≤ 0.375 nats, and base diverges from base too. The largest gpu-vs-gpu gap (0.81 nats at a near-tie) is inside the spread seen between arms at identical prefixes.

### Verdict

- **Positive.**
  - The in-graph update gives +22.3% decode tok/s (14.87 → 18.18) and −18.1% mean ITL.
  - The update-step spike falls from 118 to 65 ms p50, and all measurable remaining excess is the in-graph promotion copy itself (≈0.185 ms/row).
  - Hit rate, misses and promotions are unchanged within run spread.
  - The Python-path fixes alone give +15.8%.
- **Open:**
  1. Normal-step p50 is +1.7 ms vs base. The masked update runs in every graph forward, and 2 runs can't resolve it (the gpu runs are 4.1 ms apart). Needs a longer A/B or a no-op-step microbenchmark.
  2. Promotion copies run inline, so a large burst stalls that token (the replay's largest decode boundary copied 2,409 rows).
  3. With the flag on, DP attention, speculative decoding and decode bs > 1 are refused.
  4. `copy_rows` / `copy_bytes` / backend metrics are not reproduced.
  5. Two runs per arm on one workload.
- **Not merged or deployed at A/B time.** The serving branch was untouched and 7867 stayed down.

### Deployment (2026-09-15 05:12 CDT, at the user's request)

- **Merge:** `cc/residency-update-fast` merged into `codex/nvfp4-expert-stream-main` as `d43c59b58a` (no conflicts; the branch's only other new commits since `d0a3e55b16` were two benchmark scripts), pushed to `shared`.
- **Serving worktree:** `main-port-probe-7bc4eb` fast-forwarded to `d43c59b58a`; `git status --short -- python` was empty beforehand.
- **Tests in the serving worktree** under `cc-gpu.lock`: 188 passed, 12,042 subtests, EXIT=0, over the copy-kernel, DMA, hot-cache, residency (host, batched, clock, device, GPU) and transfer test files (`work/cc-deploy-gpu-residency/tests.log`).
- **Launch script:** `run-nvfp4-e16c-public.sh` gains `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` and `SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS=64`. Nothing else changed (diff verified, `bash -n` OK). Rollback copy: `run-nvfp4-e16c-public.sh.bak-pre-gpu-residency-20260915`.
- **Relaunch:** tmux `cc-nvfp4-dynamic`, run dir `work/cc-e16c-public/run-20260915-051223/`.
  - Healthy after 180 s on 127.0.0.1 and 10.0.0.15; the log shows `d43c59b58a`.
  - The only tracebacks are the known startup `freeze_gc` connection-refused noise.
  - GPU memory at idle is 28,337 MiB.
- **Post-deploy checks:**
  - Hot-cache metrics show evictions (39) with 0 host promotion copy submissions, so the GPU update path is live.
  - A greedy 400-token decode (thinking off) ran at 20.48 tok/s, mean ITL 48.8 ms, p50 46.7 ms.
  - A 116-token reply was coherent, at 15.8 tok/s over a short window.
  - These are single requests, not an A/B.
- **Crash at 05:21:18: CUDA OOM in eager prefill.**
  - The scheduler died in `expert_stream._gather_cached` → `_staging_buffer`: `torch.zeros` of 790 MiB for `hot_cache_misses` with 799.69 MiB free (30.42 GiB in use, 28.74 GiB allocated by PyTorch).
  - Server log: `SIGQUIT received ... one child failed`.
  - This is the eager (non-graph) miss path on a prefill, not the in-graph residency update. The setup doc already recorded a 32,055 MiB peak with one recoverable allocator OOM on a 20k-token prompt at the 14 GiB cache.
  - Idle memory rose to 28,337 MiB after deployment (27,609 MiB before), which plausibly removed the remaining headroom. The cause of the +728 MiB is not isolated (device residency banks, 14 GiB cache, 65,536-token KV).
- **Config change at the user's request (05:22):** `SGLANG_MOE_HOT_GPU_MB` 14336 → 12288 and `--context-length` / `--max-total-tokens` 65536 → 40000. Backup of the pre-change script: `run-nvfp4-e16c-public.sh.bak-ctx65536-hot14g-20260915`.
- **Relaunch at 05:23:24:** run `work/cc-e16c-public/run-20260915-052324/`, healthy at 05:27:01 on 127.0.0.1 and 10.0.0.15, still at `d43c59b58a`.
  - Hot cache startup: 4,180 slots (from 4,957), 11.56 GiB residency plus 1.33 GiB scratch.
  - Idle GPU memory: 24,891 MiB (from 28,337).
  - The only traceback is the known `freeze_gc` startup noise; 0 OOM.
- **Long-prompt check (05:27:27):** a 32,423-token prompt with 40 completion tokens returned 200 OK and a coherent summary in 322.5 s end to end.
  - Peak GPU memory 30,177 MiB (sampled every 0.5 s), with 0 OOM and 0 SIGQUIT in the log afterwards.
  - One `/health` probe right after timed out (HTTP 000) while the server was serving a queued Hermes request. The next three returned 200. With `--max-running-requests 1`, health can stall behind active work.
  - Not measured: whether decode tok/s drops with 4,180 slots vs 4,957. E20's replay predicts more misses per token at fewer slots.
- **Unrelated client errors in the same window:** Hermes requests with `reasoning_effort: "max"` (Hermes config `reasoning_effort: ultra`) got HTTP 400 "Unexpected reasoning effort max. Supported types are xhigh (default), medium, and low." The same pattern appears in the 2026-09-13 run log. It is not caused by this merge.

## E32 — Doorbell hold isolation: copy API, source access order, stream origin

- **Question:** E29 found the doorbell thread's `cudaMemcpyBatchAsync` copies held until near the end of every CUDA-graph replay launched after them. Plain torch side-stream copies were not held. Is the cause:
  - the batch copy's `srcAccessOrder = Stream` (the source is an immutable pinned arena, so `Any` / `DuringApiCall` are valid),
  - the batch API itself, or
  - the stream the C++ thread created with `cudaStreamCreate`?
- **Code:** `cc/doorbell-prototype` `696b8cc96b`, based on `d57db3e1a1`.
  - The thread takes runtime `copy_api` (`batch` | `per_segment`), `src_access_order` (`stream` | `during_call` | `any`, the CUDA enumerator values 1/2/3) and an optional caller-owned `torch.cuda.Stream`.
  - `stats()` reports the configured arm and the CUDA status of the last failed copy.
  - `probe_hold.py` and `bench_doorbell.py` take `--copy-api`, `--src-access-order` and `--torch-stream`.
  - The probe's doorbell cases now compare the whole destination byte-exact after every token. Every plan maps a source row to one fixed slot, so the result doesn't depend on whether the thread or the fallback landed last. Each case ends by flipping a byte and confirming the check catches it.
- **Tests:**
  - divix01 `e32_pytest`: 61 collected, 61 passed, EXIT=0. That is the 54 existing tests plus 7 copy-arm cases. The arm cases check bytes *before* `wait` launches, so a fallback copy can't hide a wrong thread copy, and they read the configured arm back from the thread.
  - Negative check on the laptop: planted a short per-segment copy and a stream that never reached the thread. 4 of the 7 arm cases went red; reverted, 8 of 8 green.
  - Laptop 3060 compile gate (needs `CUDA_HOME` pointed at the venv's `nvidia/cu13`; system nvcc 12.4 has no batch API): 61 collected, 56 passed, 5 failed (known laptop wait timeouts, overlap timing, arena import).
- **Run:** divix01 RTX 5090, `cc-doorbell/run.sh` under `cc-gpu.lock`, thread on core 71, driver `cc-doorbell/e32.sh` (tmux `cc-doorbell-e32`), results `cc-doorbell/results/e32_*.jsonl`, logs `cc-doorbell/logs/e32_*.log`.
  - `probe_hold.py --iters 20`, 48 layers of 3 × 2048² matmuls (≈40 ms/token of compute), `--timeout-polls 20000`.
  - Cases: `doorbell_layers` (post and wait in graph) and `compute_graphs_eager_post` (compute-only graphs, post launched eagerly). The baseline run also includes `doorbell_layers_gated` and the host torch side-stream controls.
  - Production was already down: SIGTERM at 05:36:02 from session `sglang-nvfp4-ef`, which stopped it for the expert-routing capture, not from this experiment.

### Results, 3 rows (all byte-exact, flip detected, no copy errors)

| Arm | Case | Timeouts / token (of 48) | Request done after its layer ended, p50 | Token p50 | Serviced |
|---|---|---:|---:|---:|---:|
| Baseline: batch, own stream, `stream` | doorbell_layers | 48 | 90.4 ms | 185.1 ms | 960 |
| | compute_graphs_eager_post | 46 | 148.8 ms | 183.9 ms | 960 |
| | doorbell_layers_gated (control) | 0 | −38.0 ms | 75.3 ms | 960 |
| Batch, own stream, `any` | doorbell_layers | 48 | 90.9 ms | 185.6 ms | 960 |
| | compute_graphs_eager_post | 45 | 147.8 ms | 182.0 ms | 960 |
| Batch, own stream, `during_call` | doorbell_layers | 48 | 125.6 ms | 155.5 ms | 40 |
| | compute_graphs_eager_post | 48 | 153.8 ms | 158.5 ms | 41 |
| **(a) per-segment, torch stream** | doorbell_layers | **0** | **0.64 ms** | **68.1 ms** | 960 |
| | compute_graphs_eager_post | **0** | 0.65 ms | **40.5 ms** | 960 |
| **(b) batch, torch stream, `stream`** | doorbell_layers | **0** | **0.64 ms** | **68.0 ms** | 960 |
| | compute_graphs_eager_post | **0** | 0.65 ms | **40.6 ms** | 960 |
| (c) per-segment, own stream | doorbell_layers | 48 | 91.5 ms | 186.4 ms | 960 |
| | compute_graphs_eager_post | 45 | 147.9 ms | 182.5 ms | 960 |
| Batch, torch stream, `any` | doorbell_layers | **0** | 0.64 ms | 67.8 ms | 960 |
| | compute_graphs_eager_post | **0** | 0.64 ms | 40.7 ms | 960 |

Host controls, same run: an idle 3-row copy takes 0.627 ms. A torch side-stream copy queued after replay 0 finishes −0.001 ms after layer 0 ends; queued after all 48 replays, 0.60 ms after layer 0.

### Results, 30 rows

| Arm | Case | Timeouts / token | Done after layer, p50 | Token p50 | Serviced |
|---|---|---:|---:|---:|---:|
| Baseline | doorbell_layers | 48 | 521.0 ms | 1048.7 ms | 960 |
| | compute_graphs_eager_post | 45 | 856.1 ms | 1027.8 ms | 960 |
| | doorbell_layers_gated | 0 | −161.5 ms | 340.6 ms | 960 |
| Batch, own, `any` | doorbell_layers | 48 | 577.4 ms | 1150.3 ms | 960 |
| | compute_graphs_eager_post | 45 | 853.5 ms | 1023.3 ms | 960 |
| Batch, own, `during_call` | doorbell_layers | 48 | 650.1 ms | 760.9 ms | 40 |
| | compute_graphs_eager_post | 48 | 774.1 ms | 777.6 ms | 40 |
| (a) per-segment, torch stream | doorbell_layers | 48 | 8.3 ms | 969.1 ms | 960 |
| | compute_graphs_eager_post | 4 (0–4) | 150.4 ms | 353.1 ms | 960 |
| (b) batch, torch stream | doorbell_layers | 48 | 7.7 ms | 955.3 ms | 960 |
| | compute_graphs_eager_post | 3 | 146.1 ms | 327.6 ms | 960 |
| (c) per-segment, own stream (`e32b`) | doorbell_layers | 48 | 583.9 ms | 968.5 ms | 720 |
| | compute_graphs_eager_post | 45 (42–45) | 834.0 ms | 964.8 ms | 803 |
| Batch, torch stream, `any` (`e32b`) | doorbell_layers | 48 | 7.7 ms | 955.2 ms | 960 |
| | compute_graphs_eager_post | 3 | 145.7 ms | 327.8 ms | 960 |
| **(a) per-segment, torch stream, `--timeout-polls 200000`** | doorbell_layers | **0** | **6.42 ms** | **345.3 ms** | 960 |
| | compute_graphs_eager_post | **0** | 138.2 ms | 310.6 ms | 960 |
| **(b) batch, torch stream, 200000 polls** | doorbell_layers | **0** | **6.34 ms** | **339.4 ms** | 960 |
| | compute_graphs_eager_post | **0** | 133.9 ms | 302.4 ms | 960 |
| Batch, torch stream, `any`, 200000 polls | doorbell_layers | **0** | 6.34 ms | 339.4 ms | 960 |
| | compute_graphs_eager_post | 0 | 134.1 ms | 302.6 ms | 960 |
| (b) at 3 rows, 200000 polls (budget control) | doorbell_layers | 0 | 0.64 ms | 68.0 ms | 960 |

- An idle 30-row copy takes 6.05 ms, and a torch side-stream copy queued after replay 0 lands 4.8 ms after layer 0.
- With a budget that covers the copy, the 30-row torch-stream arms land one copy duration after their layer.
  - Token time is 339 ms = 40 ms of compute + 48 × ~6.2 ms of waits: "compute plus copy time".
- (c) serviced fewer requests than posted (720 and 803 of 960): requests abandoned before the thread reached them are skipped by design. Bytes were exact and the thread drained.
- `e32b` ran 12:48–12:53 after the other session's GPU jobs ended at 12:47:53. Driver `cc-doorbell/e32b.sh`.

### Link rate on a torch stream (`bench_doorbell.py`, 1,024 production-sized arena rows, 40 iterations, all byte-exact, `prefer_overlap` on)

| Rows | In-graph kernel p50 (GiB/s) | Doorbell in graph, torch stream, p50 | Thread copy (GiB/s) | Timeouts (of 45) | Doorbell in graph, own stream (E29 path), p50 | Own-stream timeouts |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.243 ms (10.6) | **0.233 ms** | 0.216 ms (11.9) | 0 | 1.299 ms | 45 |
| 3 | 0.693 ms (11.2) | **0.646 ms** | 0.628 ms (12.3) | 0 | 1.749 ms | 45 |
| 10 | 2.274 ms (11.3) | **2.104 ms** | 2.072 ms (12.4) | 0 | 3.330 ms | 45 |
| 30 | 6.759 ms (11.4) | 12.928 ms | 7.503 ms (10.3) | 45 | 7.841 ms | 45 |

- On a torch stream the in-graph doorbell now beats the in-graph kernel end to end at 1–10 rows, by 4–7%.
- At 30 rows the bench's own `--timeout-polls 20000` budget (≈5 ms) is shorter than the 6.3 ms copy, so it paid the fallback, the same artifact as the probe.
- Eager post → copy complete matches E29 (0.667 / 2.139 / 6.276 ms).
- The bench's "reaction" column is negative here: the Python event spin observes the GPU post after the thread already has. It isn't meaningful and isn't reported.

### E32c: the two controls skipped in E32b (13:20:56–13:23:46, driver `cc-doorbell/e32c.sh`, tmux `cc-doorbell-e32c`, `696b8cc96b`)

All runs under `run.sh` / `cc-gpu.lock`, EXIT 0, byte-exact, flipped byte caught, 0 copy errors. Results `cc-doorbell/results/e32c_*.jsonl`, logs `cc-doorbell/logs/e32c_*.log`.

**(i) Own-stream baseline probe (batch, own stream, `stream`), 30 rows, `--timeout-polls 200000`** (`e32c_base_r30_p200k`):

| Case | Timeouts / token (of 48) | Done after layer, p50 | Token p50 | Serviced |
|---|---:|---:|---:|---:|
| doorbell_layers | 48 | 519.4 ms | 1045.1 ms | 960 |
| compute_graphs_eager_post | 45 (43–45) | 886.9 ms | 1061.9 ms | 960 |

- Still held with a budget that covers the 6.05 ms copy, matching the 20000-poll run (521 ms / 1049 ms).
- Caveat: after a wait's first timeout the waiter drops to degraded ~1 ms budgets (`degraded_polls` 4096) until a wait succeeds, so the 200000-poll budget only governs the first wait. The 519 ms done-after-layer time is what proves the hold. The torch-stream arms never time out, so the caveat doesn't touch them.

**(ii) `bench_doorbell.py`, 1,024 production-sized rows, 40 iterations + 5 warmup, `--timeout-polls 200000`, `prefer_overlap` on, p50 per replay in graph.**

Torch stream, (b) (`e32c_bench_torch_p200k`):

| Rows | In-graph kernel (GiB/s) | Doorbell in graph | Thread copy (GiB/s) | Timeouts (of 45) | Eager post → done |
|---:|---|---:|---:|---:|---:|
| 1 | 0.245 ms (10.5) | 0.239 ms | 0.219 ms (11.8) | 0 | 0.254 ms |
| 3 | 0.702 ms (11.0) | 0.662 ms | 0.633 ms (12.2) | 0 | 0.694 ms |
| 10 | 2.275 ms (11.3) | 2.109 ms | 2.074 ms (12.4) | 0 | 2.147 ms |
| 30 | 6.758 ms (11.4) | **6.241 ms** | 6.204 ms (12.45) | **0** | 6.298 ms |

Own stream (`e32c_bench_own_p200k`):

| Rows | In-graph kernel (GiB/s) | Doorbell in graph | Thread copy (GiB/s) | Timeouts (of 45) | Eager post → done |
|---:|---|---:|---:|---:|---:|
| 1 | 0.242 ms (10.6) | 1.301 ms | 1.495 ms (1.72) | 45 | 0.247 ms |
| 3 | 0.698 ms (11.1) | 1.752 ms | 2.355 ms (3.28) | 45 | 0.664 ms |
| 10 | 2.273 ms (11.3) | 3.333 ms | 5.318 ms (4.84) | 45 | 2.142 ms |
| 30 | 6.757 ms (11.4) | 7.819 ms | 13.815 ms (5.59) | 45 | 6.287 ms |

- On a torch stream the in-graph doorbell beats the in-graph kernel at every size: −2.4% / −5.7% / −7.3% / −7.7% at 1 / 3 / 10 / 30 rows, and 30 rows completes without fallback.
- On its own stream it is held in graph at every size: the thread copy stretches to cover the replay, and the ~1 ms in-graph overhead is the degraded wait plus fallback, matching E29. Eager copies are unaffected on both streams; waiting on an already-completed request takes 16–21 µs.
- The link-rate edge at 30 rows (≈8%) matches the figure used in the step-2 estimate, so the ≈2.4 ms/token estimate stands.
- Production: SIGTERM to pid 787756 at 13:20:22, GPU free 13:20:43, no other session's job on the GPU before or during; relaunched unchanged at 13:24:13 (tmux `cc-nvfp4-prod-e32c`, run `run-20260915-132413`): health 200 at 13:28:05, 4,180 slots, 0 OOM, 24,694 MiB at idle. Down 7 min 43 s.

### Production during E32

- The trainers of session `sglang-nvfp4-ef` held the GPU 12:16–12:47:53, outside `cc-gpu.lock`.
- Relaunch at 12:53:57 (run `run-20260915-125357`) died at 12:58:06 with a CUDA OOM allocating the expert hot cache: `train_apex.py` restarted at 12:54:06 and took 10.3 GiB, leaving 20.24 GB at load instead of 30.53. The script's empty-GPU check runs only at start.
- Relaunch at 13:08:48 (tmux `cc-nvfp4-prod-e32b`, run `run-20260915-130848`): health 200 at 13:13:15, 4,180 slots, 0 OOM, 24.7 GiB at idle. The other session has since agreed to take `cc-gpu.lock` and wait for 7867 health before GPU work.

### Verdict

- **Cause of the E29 hold: the stream the C++ thread creates for itself** (`cudaStreamCreate` on the non-torch thread).
  - With a torch-created stream (`torch.cuda.Stream().cuda_stream` handed to the thread), requests at 3 rows land 0.64 ms after the posting layer. That equals the idle copy time. Every in-graph wait succeeds (0/48 timeouts).
  - Token time is 68 ms in-graph, which is 40 ms of compute plus 48 × ~0.6 ms of waits: the success criterion "40 ms plus copy time". It is 40.5 ms when the post is eager and the waits come after the token.
  - Copy API is irrelevant: (a) ≡ (b), and (c) is held like the baseline.
- **Access order is not the cause** (hypothesis refuted).
  - `any` on the thread's own stream is identical to `stream`, and `any` on a torch stream is identical to `stream` on a torch stream.
  - `during_call` returns no error on driver 610.57. The call itself waits behind the hold, so the thread is stuck in it and serviced only 40 of 960 requests. Unusable.
- **30 rows succeeds too, once the wait budget covers the copy.**
  - With `--timeout-polls 20000` (≈5 ms) the 6.05 ms copy timed out every wait.
  - With 200000 polls: 0/48 timeouts, requests land 6.3–6.4 ms after their layer, token 339–345 ms, byte-exact.
  - (c) stays held at 30 rows (584 ms). The success criterion of NVFP4_DOORBELL_COPIER.md §6 step 1 is met at both 3 and 30 rows.
- **Link rate is kept.** On a torch stream the thread copies at 11.8–12.45 GiB/s, and the in-graph doorbell beats the in-graph kernel at every size: by 2.4–7.7% at 1–30 rows (E32c).
- **Controls completed in E32c:** the own-stream baseline stays held at 30 rows with a 200000-poll budget (519 ms after the layer against a 6.05 ms copy), and the torch-stream bench completes at 30 rows with 0 timeouts.
- **Serving caveat:** `timeout_polls` must cover the largest expected copy. The default in `ExpertDoorbellCopier` is 2,000,000 polls; the probe and bench used 20000.
- **Next (§6 step 2):** re-price against the in-graph kernel for overlap. The go threshold is ≥3 ms/token.
  - Rough E28-based estimate: 2.72 misses/layer × 0.224 ms/row ≈ 0.61 ms of copy per layer. The doorbell's link-rate edge (12.4 vs 11.4 GiB/s, ≈8%) saves ≈0.05 ms/layer, ≈2.4 ms/token over 48 layers.
  - Overlap itself is available to the in-graph kernel too (E27), so the doorbell's edge over it is only that link rate, plus any gain from not holding a GPU kernel thread.
  - This is below the threshold, and is an estimate to confirm with the serving miss distribution, not a measurement.
