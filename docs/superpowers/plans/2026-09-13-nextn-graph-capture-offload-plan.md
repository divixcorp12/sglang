# NEXTN speculative decoding with CUDA graphs under NVFP4 expert offload

Status: plan only, nothing implemented. Written 2026-09-13 from a read-only investigation of branch
`master` plus three measured runs on divix01 (RTX 5090, PCIe Gen3 x16, bs=1).
File:line citations are as of `d9e6720112`.

## Decision summary

- Graph capture with offload pays only if expert **copy bytes per accepted token (B)** reach production's
  level: **B ≤ ~0.46 GiB/token at accept length 3.27** to break even with 20.7 tok/s. Eager NEXTN today
  measures **0.70**.
- **Fix A (dedup routes on multi-token forwards)** is the one lever the code controls: bytes per verify
  2.27 → 1.49–1.99 GiB (−12% to −34%).
- **Fix B (speculative residency boundaries)** is a correctness/cadence fix with a small, unmeasured
  hit-rate effect.
- With both fixes the captured-graph estimate is **14–25 tok/s, realistic ≈ 19**, vs production 20.7.
  Break-even needs **≤ 12.6 unique misses per layer per verify (≤ 1.52 GiB)**; a ship bar of 23.8 tok/s
  needs **≤ 10.5 (≤ 1.26 GiB)**.
- Structural ceiling: 26.3 unique experts per 3.27 accepted tokens = 8.0 per token vs 10 without
  speculation, a **~1.24× copy advantage** at equal hit rate; speculation cannot win by more than ~1.3×
  unless verify gets a better hit rate than decode.
- **Do Phase 0 (offline) and Phase 1 (Fixes A and B, eager) before any graph code.**

## Measurements this plan rests on

| Run (same 7 requests) | Decode-heavy tok/s | Notes |
|---|---|---|
| Production, breakable decode graph, 10 GiB hot cache | ~20.7 | step ≈ 48 ms, copy ≈ 30 ms, compute ≈ 13 ms |
| Eager baseline, 8 GiB hot cache, graph gather off | 3.13 | overhead-bound: ~49 ms copy of ~320 ms/token |
| Eager NEXTN (steps 3, topk 1, 4 draft tokens), same settings | 9.42 | accept length 3.27 (2.55–3.75) |

Eager NEXTN hot-cache totals (`cc-spec-nextn/run-20260913-150935/hot-cache.metrics.jsonl`):
808,320 routed rows, 531,305 unique, 372,061 routed misses, 436,259 hits (54%), 2.28 GiB H2D per verify
over 421 verifies. Per layer per verify: 40 routes, 26.3 unique, 18.4 routed misses, 21.6 routed hits.
Peak GPU: NEXTN 27.2 GiB vs baseline 23.4 GiB (+3.8 GiB, 8 GiB hot cache, no scratch).

Correction to an earlier claim: residency did update during eager NEXTN (107 boundaries, 10,700
promotions, 29.6 GB migration). Draft DECODE forwards (`speculative/eagle_worker_v2.py:757`) pass the
observer at `model_executor/model_runner.py:1764` and advance `_decode_forwards_since_boundary`
(`layers/moe/expert_hot_cache.py:1086-1087`), so a boundary fired every 8 draft forwards ≈ every 4 verify
cycles ≈ 13.1 committed tokens — a working policy on an accidental clock.

## 1. Current state

- NEXTN is an alias for EAGLE (`speculative/spec_registry.py:192-194`) and runs `EAGLEWorkerV2`. The draft
  worker captures draft-decode and draft-extend graphs (`eagle_worker_v2.py:439-594`); the target decode
  runner captures TARGET_VERIFY with `num_draft_tokens` width (`model_executor/runner/decode_cuda_graph_runner.py:292-300`).
- Disabled today: the run script sets `--cuda-graph-backend-decode disabled` and
  `SGLANG_MOE_EXPERT_GRAPH_GATHER=0`; `arg_groups/memory_hook.py:158-162` rejects PLE staging with
  speculation; graph gather needs decode graphs and a host arena and forbids prefetch (`memory_hook.py:124-150`).
- What breaks at 4 verify tokens:
  1. Scratch is `decode.max_bs × top_k` = 10 rows/layer (`model_executor/model_runner.py:730-732`,
     `expert_hot_cache.py:637-638`).
  2. 40 routes exceed it, `serves_graph_gather` is false (`layers/moe/expert_stream.py:537-544`), and
     `fused_moe_triton/layer.py:1573-1585` takes an `eager_on_graph` break on all 48 MoE layers
     (`breakable_cuda_graph.py:219-270`), each with host syncs (`expert_stream.py:776`, `:1011`).
  3. No dedup on either path: eager skips it at `_NO_DEDUP_LIMIT = 64` (`expert_stream.py:21`, `:1018-1019`);
     `_gather_graph` gives each route its own scratch row (`:622-653`).
  4. PLE staging key mismatch at verify replay: capture keys by lookup tokens (`models/qwen4_exp.py:1217-1224`),
     replay passes `_replay_graph_key.size` = bs (`decode_cuda_graph_runner.py:552-553`, `:1437-1441`), so
     `replay_lookup` would raise (`qwen4_exp.py:1276-1280`). Likely; confirm with a test.
  5. Residency boundary clock driven by draft forwards (Fix B).
- Draft side needs no expert scratch: MTP experts are FP8 and streamers attach only on the NVFP4 path
  (`layers/quantization/modelopt_quant.py:2962`); the MTP forward disables the recorder
  (`models/qwen4_exp_mtp.py:209`) and has no PLE layers.

## 2. Memory budget

One expert row = 2,764,808 B; scratch comes out of `SGLANG_MOE_HOT_GPU_MB` (`expert_hot_cache.py:639-646`).

| Scratch rows/layer | Total rows | GiB |
|---|---|---|
| 10 (today) | 480 | 1.24 |
| 16 | 768 | 1.98 |
| 24 | 1,152 | 2.97 |
| 40 (verify, no dedup) | 1,920 | 4.94 |

Meta-allocating the draft `embed_tokens`/`lm_head` saves 2.2 GiB: `init_lm_head` replaces them with the
target's only after the target KV pool is sized (`eagle_worker_v2.py:227-242`, `:318-361`).

| Option | Setup | Peak | Verdict |
|---|---|---|---|
| A | 3,106 slots + 40 rows, no dedup | 32.1 GiB (29.9 with meta head) | does not fit / tight |
| **B (recommended)** | dedup graph gather, K unique-miss rows + overflow flag, 3,106 slots, meta head | 28.0 (K=24), 27.0 (K=16) | fits |
| C | today's 10.74 GB split as 2,300 slots + 40 rows, meta head | ≈ 27.5 | fits, −26% slots, unmeasured hit-rate cost |

Option B re-runs an overflowing verify eagerly; pick K from Phase 0/1 p99 unique misses per layer. If
verify + draft graph pools exceed ~0.5 GiB, use `--chunked-prefill-size 2048`.

## 3. Fix A — dedup routes on multi-token forwards

Eager path:
- `expert_stream.py:1018`: dedup whenever `topk_ids.shape[0] > 1`; single-token decode keeps the no-dedup
  fast path. The `torch.unique(..., return_inverse=True)` remap (`:1024-1025`, `_gather_cached` `:767-812`)
  already serves prefill.
- `expert_stream.py:1033`: `record_routes` must keep routed multiplicity (pass `flat_ids` or bincount with
  the inverse), matching the graph path's `pending_counts.index_add_(flat)` (`:650-652`).
- Stats (`:776-779`, `:987-990`): keep a routed count, or compare runs on `h2d_bytes` and
  `requested_unique_experts` rather than hit %.
- Non-speculative effect: prefill gathers over 64 routes now record routed multiplicity (residency
  scores grow relative to the `SGLANG_MOE_HOT_PROMOTION_SIGMAS` tuning) and bs>1 decode now dedups its
  gathered rows (recorded scores unchanged, gather metrics change).

Graph path (`expert_stream.py:622-653`), device-only: sort flat IDs, mark group starts, build inverse and
unique IDs, look up `expert_to_slot[uniq]`, rank unique misses, remap each route to its slot or
`scratch_base + rank[inverse].clamp_max(K-1)`, and OR `miss_count > K` into a device overflow flag.
Scratch becomes K unique-miss rows and `model_runner.py:730` sizes by `max_bs × num_draft_tokens`.
For 1-token decode this degenerates to today's behaviour with K = 10.

CPU tests (the existing expert stream/cache tests are CUDA-gated, e.g. `test_expert_hot_cache.py:18`, so
extract pure helpers `should_dedup` and `plan_graph_routes`):
- A1 (4,10) IDs with duplicates → `source_ids.numel() == len(set(ids))` (fails under the `<= 64` rule).
- A2 (1,10) → no unique, `source_ids == flat`.
- A3 property test over 200 seeds incl. all-hit/all-miss/all-same: simulated `cache[remap] == experts[ids]`
  and distinct scratch rows used == unique misses.
- A4 `record_routes` counts == `bincount(flat)`.
- A5 overflow flag set at K+1 unique misses, clear at K.
- A6 no host reads under a `TorchDispatchMode` that raises on `_local_scalar_dense`/`item`.

Expected effect: routed misses 18.4/layer (2.27 GiB) → unique misses 12.1 (duplicates split
proportionally, 1.49 GiB, −34%) to ~16 (duplicates concentrated in hot experts, 1.98 GiB, −13%). Eager
≈ 9.9–10.9 tok/s; graph model at L 3.27, F 35: 14.9 → 16.8–21.0 tok/s from dedup alone. Lossless: greedy
outputs must stay token-identical. The overflow flag is read with the accept lengths already copied to
the host (`managers/scheduler_components/batch_result_processor.py:760-778` →
`eagle_worker_v2.py:1456`), so no new sync.

## 4. Fix B — speculative residency boundaries

Facts: `_phase` labels TARGET_VERIFY and DRAFT_EXTEND_V2 "speculative" (`expert_hot_cache.py:872-878`);
only "decode" advances the forward counter (`:1086-1087`); every observer call adds batch_size (1) to
`_tokens_since_boundary` (`:1081-1085`). Per cycle 2 draft decodes + 1 verify + 1 draft-extend count
4 "tokens" against 3.27 committed, so decay runs ~22% fast and cadence silently depends on num_steps.
Accept lengths reach the host only after verify (`eagle_worker_v2.py:1349`, `:1011-1013`,
`batch_result_processor.py:760-778` → `on_verify_complete_cpu` at `eagle_worker_v2.py:1456`).

Changes:
1. Ignore draft-worker forwards in the observer (draft DECODE via `spec_info.is_draft_input()`,
   `speculative/spec_info.py:416`, and DRAFT_EXTEND_V2) before touching any counter.
2. Count TARGET_VERIFY forwards toward `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS`.
3. Add `spec_info.draft_token_num` tokens at verify, then correct by `(accepted − draft_token_num)` in a new
   `ExpertHotCacheManager.on_speculative_commit(accepted_tokens)` called from `on_verify_complete_cpu`;
   boundary decisions happen on the next observer call. Fallback without the hook: count 4, decay 20.
4. Launch with `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=2` under speculation (≈ 6.5 committed tokens,
   close to production's 8-token cadence); keeping 8 would space boundaries ~26 tokens apart.
5. Extract a pure `ResidencyBoundaryClock` for CPU tests.

CPU tests (each fails on current code): B1 two TARGET_VERIFY observations with cadence 2 qualify a
boundary; B2 draft DECODE leaves counters unchanged; B3 DRAFT_EXTEND_V2 ignored; B4 verify(4) then
commit(3) → next `advance()` gets 3; B5 `min_residence_forwards` counts target forwards only; B6 cadence
per committed token independent of num_steps 2/3/5.

Expected effect: cadence 13.1 → 6.5 tokens, exact decay; hit rate +0–4 routed points (unmeasured);
risk of roughly doubling migration bytes (~70 MB per verify today).

## 5. Other graph-capture work

- PLE: pass `bs × captured_req_width` as graph tokens for TARGET_VERIFY (`decode_cuda_graph_runner.py:1441`)
  and drop the rejection at `memory_hook.py:158-162`; `_prepare_ple_batch` already handles verify rows
  (`qwen4_exp.py:155-200`). Keep `--disable-overlap-schedule`.
- Recorder: use `stat` for graph runs; `per_pass` does a `.cpu()` per pass (`eplb/expert_distribution.py:497`).
- Speculative admissions (`expert_hot_cache.py:395-453`) belong to prefetch, which graph gather forbids
  (`expert_stream.py:586-590`): out of scope.
- Remaining sync: the accept-length readback already paid once per step.

## 6. Correctness risks

- GatedDeltaNet rollback: per-token intermediates and conv windows (`mem_cache/memory_pool.py:746-790`,
  `layers/attention/mamba/mamba.py:648-676`), accepted-step scatter (`hybrid_linear_attn_backend.py:1340-1427`),
  PLE short-conv state (`qwen4_exp.py:1146-1170`). Eager accept lengths were sane; graphs need a GPU parity test.
- Hyper-connection hidden states: `last_hc_hidden_states` is a Python attribute (`qwen4_exp.py:2002-2007`);
  under replay it must be a static buffer overwritten in place. The MTP asserts `hc_count × hidden`
  (`qwen4_exp_mtp.py:189`).
- Fix A must be lossless (token-identical greedy output); overflow re-verify must reproduce logits exactly.
- Rejected tokens' routes still bias residency (policy only).

## 7. Payoff model (assumes Fixes A and B)

tok/s = L / (c·Bv + F), with L accept length, Bv GiB copied per verify (B = Bv/L), c ≈ 81 ms/GiB (production
30 ms ↔ 0.37 GiB, Gen3 x16; 64 if production's true figure is 0.47 GiB), F = verify compute 17–25 ms +
3 MTP forwards 9–15 ms + overhead 3–5 ms (realistic 35, range 30–45), Bv = unique misses/layer × 48 × 2.575 MiB.

Break-even vs 20.7 tok/s, Bv ≤ (48.3·L − F)/81 at F = 35, U = 26.3:

| L | Max Bv | Max B | Max unique misses/layer | Unique-expert hit needed |
|---|---|---|---|---|
| 2.5 | 1.06 GiB | 0.42 | 8.8 | 66% |
| 3.0 | 1.36 GiB | 0.45 | 11.3 | 57% |
| **3.27** | **1.52 GiB** | **0.46** | **12.6** | **52%** |
| 3.5 | 1.65 GiB | 0.47 | 13.7 | 48% |
| 4.0 | 1.95 GiB | 0.49 | 16.2 | 38% |

F = 45 at L 3.27 tightens this to ≤ 1.40 GiB (11.6/layer). Ship bar 23.8 tok/s at L 3.27, F 35:
Bv ≤ 1.26 GiB (10.5/layer). Without dedup the same bar needs a routed hit rate ≥ 68.5% (54% measured).

| Scenario | Unique misses/layer | L | F | Bv | Step | tok/s |
|---|---|---|---|---|---|---|
| Best | 11 | 3.5 | 30 ms | 1.36 GiB | 140 ms | 25.0 |
| Realistic | 14 | 3.27 | 35 ms | 1.73 GiB | 175 ms | 18.7 |
| Worst | 16 | 2.9 | 45 ms | 1.98 GiB | 205 ms | 14.1 |

## 8. Phases and go/no-go

**Phase 0 — offline, no fork changes.**
- 0-A: from the per_pass dump, unique misses per layer per verify against the hot set at each boundary
  (seed + promotions): p50/p95/p99, P(>16), P(>24).
- 0-B: replay traces through `ExpertResidencyPolicy` on CPU with Fix B's clock at 2- and 3-verify cadence,
  3,106 and 3,403 slots.
- 0-C: c for the graph path from the production profile (pull-kernel time ÷ graph miss rows).
- 0-D: F from verify + draft graphs captured with all experts on GPU at tiny context.
- **Go** if projected p50 ≥ 23.8 tok/s for any budget option; otherwise stop after Phase 1 at most.

**Phase 1 — Fixes A and B on the eager path.**
- Files: `layers/moe/expert_stream.py`, `layers/moe/expert_hot_cache.py`, `speculative/eagle_worker_v2.py`;
  pure helpers + CPU tests A1–A6, B1–B6; run only those plus `test_expert_hot_cache.py`/`test_expert_stream.py`.
- Rerun eager NEXTN (same script, 8 GiB, same 7 requests): speculative h2d bytes per verify, unique misses
  per layer (new stat), routed hit %, accept length, boundaries, migration GiB, decode-heavy tok/s vs 9.42
  (NEXTN) and 3.13 (baseline). Greedy outputs token-identical to a pre-fix run at temperature 0.
- **Go** if Bv ≤ 1.52 GiB (ideally ≤ 1.26), accept length within ±0.1, identical outputs, tok/s ≥ 9.42.
  **No-go** if Bv > 1.75 GiB.

**Phase 2 — memory.** Meta-allocate draft embed/head (`models/qwen4_exp_mtp.py`, `qwen3_5_mtp.py`), scratch
sized from `num_draft_tokens` (`model_runner.py:730`), validators (`memory_hook.py`). CPU tests: validator
matrix, scratch arithmetic, meta params replaced before first use. **Go** if option B starts and peak stays
≤ 30.5 GiB through a 4,096-token prefill.

**Phase 3 — verify graph** (dedup graph gather, overflow flag, PLE staging key): `expert_stream.py`,
`decode_cuda_graph_runner.py`, `qwen4_exp.py`, `eagle_worker_v2.py`. Tests: CPU PLE key at replay equals
capture; GPU parity on a 2-layer model (eager vs graph verify logits, mamba state after accept, HC hidden
states); `_break_fns` empty. **Go** if zero breaks, 500 greedy tokens identical, overflow re-verify < 5% of
steps, ≥ 22 tok/s.

**Phase 4 — draft graphs + 1-hour soak.** **Ship only if** ≥ 23.8 tok/s sustained and peak ≤ 30.5 GiB.

## Trade-offs

| Option | Pros | Cons |
|---|---|---|
| A+B eager first (recommended) | cheap; measures the deciding number; lossless | no serving speed win by itself |
| Dedup + capped K + overflow re-verify | smallest scratch, fewest bytes | new device kernel logic; overflow steps pay twice |
| 40-row no-dedup scratch | simplest | B ≈ 0.70 (loses); does not fit without the meta head |
| Invest in decode hit rate instead | lowers production B directly | forgoes the ~1.24× union and compute amortization |

## Unknowns and how to resolve them

| Unknown | Resolved by |
|---|---|
| Unique-miss distribution per verify | Phase 0-A |
| Achievable unique hit rate | Phase 0-B |
| Graph copy cost c and non-copy time F | Phase 0-C, 0-D |
| Verify/draft graph pool size | capture log `mem usage=` |
| Draft-extend attention backend | startup log "Capture draft extend CUDA graph" |
| Fix B hit-rate effect | Phase 1 rerun |
