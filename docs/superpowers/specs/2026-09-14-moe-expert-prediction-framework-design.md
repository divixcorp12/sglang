# MoE Expert Prediction Framework (shadow milestone) — Design

Date: 2026-09-14. Branch `master`.

## Purpose

A model-independent place to plug in MoE expert predictors (LLaPor, APEX,
heuristics) so they can be compared on the same live traffic and switched on
or off with one flag. This milestone delivers the shared framework and shadow
scoring only: predictors rank experts, the framework measures them against
the native router, and nothing is transferred.

Companion plans (research design, not yet implemented here):
`crypto/trading-framework/mechanism_hunter/docs/superpowers/plans/2026-09-14-llapor-gpu-only.md`
and `...-apex-gpu-only.md`.

## Non-goals for this milestone

- Prefetch admission, slot leases, transfer scheduling (next milestone; the
  existing `ExpertPrefetchCoordinator` and `SGLANG_MOE_PREFETCH_MAX_CANDIDATES`
  stay untouched until then).
- LLaPor and APEX predictors, their training tools, and checkpoints.
- The offline perfect-predictor replay simulator (separate plan).
- Deducting predictor memory from `SGLANG_MOE_HOT_GPU_MB` (needed once
  prefetch compares latency at equal VRAM).
- Multi-GPU. Tensor, expert, attention-DP, or pipeline parallelism > 1 is
  rejected at startup.

## Requirements

1. Off by default. With `SGLANG_MOE_EXPERT_PREDICTOR` unset or empty, no
   hooks, buffers, or per-forward work exist.
2. No model file edits and no model imports inside the package. Any SGLang
   model whose MoE blocks own a `TopK` and a `FusedMoE` child works.
3. Works under the production launch: breakable decode CUDA graphs with
   `SGLANG_MOE_EXPERT_GRAPH_GATHER=1`.
4. No host synchronization on the forward path; only the periodic metrics
   write reads the device.
5. Several predictors run side by side on the same forwards for A/B
   comparison; their metrics land in one JSONL file.
6. Adding a predictor is one class plus one registration.

## Architecture

```
python/sglang/srt/layers/moe/expert_prediction/
  contracts.py      RouteFeature enum, MoeLayerSpec, feature widths/dtypes
  feature_store.py  fixed-address [max_rows, width] buffers per (layer, feature)
  taps.py           discover TopK+FusedMoE blocks; TopK forward hooks -> store
  adapters.py       per-architecture pre-mixer feature taps (default + Qwen4-Exp)
  base.py           ExpertPredictor ABC, pad_candidates
  popularity.py     same-layer baseline (decayed route counts)
  affinity.py       next-layer baseline (co-occurrence then popularity)
  registry.py       name -> predictor class; build_predictors
  metrics.py        device-only scoring; cumulative totals; JSONL
  runtime.py        ExpertPredictionRuntime: build, on_forward_end, close
```

`ModelRunner` (frozen, orchestration only) gains:
- `maybe_init_expert_prediction()` called in `initialize()` right after the
  expert hot cache and its prefetch hook are set up, so buffers exist before
  CUDA graph capture and KV pool sizing.
- one delegate call `self.expert_prediction_runtime.on_forward_end(forward_batch)`
  inside the expert-distribution recorder's `with` block, after `_forward_raw`.

### Discovery (model independence)

Walk `model.modules()`; a block qualifies when its direct children include
exactly one `TopK` and one `FusedMoE`. `layer_id` comes from the `FusedMoE`
(always required there; some models build `TopK` without one). Routed counts
come from each owner's own config, because e.g. `Qwen2MoeSparseMoeBlock`
builds `TopK` without fused shared experts while its `FusedMoE` counts the
shared slots and appends shared ids only after `TopK` returns:
`num_experts = moe.num_experts - moe.num_fused_shared_experts`,
`top_k = topk.topk_config.top_k - topk.topk_config.num_fused_shared_experts`.
Layers are ordered by `layer_id`; "next layer" means the next tapped MoE
layer, so dense layers in between are skipped.

### Taps and graph safety

A `TopK` forward hook (`with_kwargs=True`) copies `hidden_states` (router
input), `router_logits`, `topk_ids`, and `topk_weights` into `FeatureStore`
buffers with `copy_`. Buffers are allocated once at startup, so copies
recorded during CUDA graph capture write the same storage on every replay.
Python in the hook only runs in eager forwards and during capture, which is
where shape and format checks happen. Top-k tensors keep only the leading
routed columns (fused shared experts are appended after them). Batches with
more rows than `max_rows` are skipped at write time; `max_rows` defaults to
decode CUDA-graph `max_bs x tokens_per_request` so every graph shape fits.

A non-`StandardTopKOutput` result marks the layer unsupported (warning once);
the runtime then stops scoring. It never raises inside a forward.

### Pre-mixer feature (APEX input)

Model-specific, so it goes through an adapter registry keyed by the model's
class name. The decoder layer of each MoE block is the outermost
`nn.ModuleList` element containing it.
- Default: forward pre-hook on the decoder layer taking `hidden_states` from
  kwargs or the first 2-D floating tensor argument of width `hidden_size`.
  Models that pass the residual stream separately should register an adapter.
- `Qwen4ExpForConditionalGeneration`: wrap `attn_hyper_connection.mix` and
  tap `mix(...)[0]`, the tensor the attention or GatedDeltaNet mixer consumes.
Installed only when some enabled predictor requires `PRE_MIXER`.

### Predictor contract

```python
class ExpertPredictor(abc.ABC):
    name: ClassVar[str]
    target_offset: ClassVar[int]            # 0 same layer, 1 next tapped layer
    required_features: ClassVar[frozenset[RouteFeature]]
    def __init__(self, *, specs, device, max_candidates): ...
    def predict(self, *, source_layer, target_layer, store, rows) -> Tensor  # int64 [rows, M], -1 pads
    def observe(self, *, store, rows) -> None                                # online update, after scoring
    state_nbytes: int
```

Within a forward, every `predict` runs before `observe`, so online state never
learns from the routes it is scored on. Both must be device-only.

### Runtime per forward

`on_forward_end(forward_batch)`:
1. Skip unless the forward is DECODE or VERIFY (`classify_forward`), has
   `0 < rows <= max_rows` where `rows = forward_batch.input_ids.shape[0]`,
   and no layer is unsupported. Draft workers never build a runtime.
2. For each predictor and each `(source, target)` pair: `predict`, then
   `score_candidates` against the target layer's `TOPK_IDS`, add to totals.
3. `observe` per predictor.
4. Every `SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL` scored forwards, append
   a JSONL record (one device read). Scoring itself runs only every
   `SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL` eligible forwards, to bound
   per-step host overhead.

Rows are joined by position: within one forward every layer sees the same
tokens in the same order, so no identity keys are needed. Cross-forward
datasets (training capture) will need them; that belongs to the predictor
plans.

The call runs inside the recorder's `with` block, before the hot cache's
per-forward observer reassigns residency, so residency masks reflect the
state the forward's gathers used. Work is enqueued on the current stream
behind the forward.

### Metrics

Per predictor and target layer, cumulative int64 counters:
`rows, routes, hits_at_k, hits_at_m, cold_routes, cold_hits_at_m, cold_candidates`.
- `routes`: valid native routed selections (K per token).
- `hits_at_k` / `hits_at_m`: native selections present in the first K / all M
  candidates.
- Cold: the target layer's hot cache marks the expert non-resident
  (`expert_to_slot < 0`). Without a hot cache every route is cold.
- `cold_candidates`: in-range candidates not resident, counted per token (not
  deduplicated across a batch); a proxy for transfer bytes a prefetcher would
  request.

Totals add `recall_at_k`, `recall_at_m`, `cold_recall_at_m`
(`cold_hits_at_m / cold_routes`), and `cold_precision_at_m`
(`cold_hits_at_m / cold_candidates`).

JSONL record: `{"timestamp_ns", "forwards", "eligible_forwards", "predictors": {name: {"layers": {layer_id: counters}, "total": counters+ratios}}}`.

## Configuration (environ.py, MoE section)

| Variable | Default | Meaning |
|---|---|---|
| `SGLANG_MOE_EXPERT_PREDICTOR` | `()` | Comma list of predictors to shadow-score; empty disables |
| `SGLANG_MOE_EXPERT_PREDICTOR_CANDIDATES` | 16 | M, ranked candidates per token |
| `SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL` | 1 | Score and update predictors every Nth eligible forward |
| `SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS` | 0 | Tap buffer rows; 0 derives from decode graph max_bs x tokens per request |
| `SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL` | 100 | Scored forwards between JSONL records |
| `SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE` | `""` | JSONL path; empty keeps totals in memory only |

A/B: accuracy comparisons run several predictors in one process; latency
comparisons (next milestone) use separate launches differing only in these
variables.

## Known approximations

- Forwards issued outside the scheduler that reach `ModelRunner.forward`
  (server warmup) are scored. Production passes `--skip-server-warmup`.
- `cold_candidates` double counts an expert predicted for several tokens.
- The default pre-mixer adapter reads the decoder's `hidden_states`
  argument; for residual-split decoders that is not the normalized mixer input.
- Shadow predictor compute runs after the forward, so it does not measure
  in-graph predictor latency or lookahead; prefetch will need both.

## Verification

- CPU unit tests on fake TopK/FusedMoE modules for every component (divix01,
  `CUDA_VISIBLE_DEVICES=""`, `taskset -c 64-71`).
- CUDA test: a captured `torch.cuda.CUDAGraph` refreshes tap buffers on
  replay, and `on_forward_end` passes under `torch.cuda.set_sync_debug_mode("error")`.
- Live smoke on an experiment server with production flags: off vs
  `affinity,popularity` give identical greedy output, the decode log reports
  CUDA graph use, the metrics file fills, and decode tok/s is recorded for both.
