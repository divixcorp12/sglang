# DSV4.1 EXL3 VRAM-miss prefetch: offline study (E1 route-only baselines)

**Date:** 2026-09-24/25. **Branch:** `cc/prefetch-study`, based on `shared/cc/hotcache-routes` (`2521e2b53a`).
**Scope:** offline CPU replay only. No runtime code was changed and no GPU was used.
**Question:** can we predict decode VRAM misses early enough to prefetch them into the hot cache during
earlier layers, and how much of the ~31 misses/token gap between the current policy and Belady can that
capture?

## Answer

- **The route-only predictors capture none of the gap.** Previous-token, popularity, recency and cross-layer
  co-occurrence all lose time at the realistic budget, by 0 to 33% of the gap. The best of them, cross-layer
  co-occurrence gated at P ≥ 0.5, gains +2 to +3% (≈1 ms/token) only at the 2x budget. Its held-out precision
  on prefetched rows is 7 to 12%. The previous-token predictor never issues a single row: DIRECT inserts every
  miss, so the previous token's rows are always resident.
- **Perfect prediction captures the whole gap.** The oracle hides ~30 rows/token of the 34 rows of idle link
  at the 0.85-row budget (K=1, h=4: 47.5 exposed, **+99%**). At the 2x budget it goes past Belady (K=2, h=4:
  23.3, +178%). Prefetch does not remove transfers. It moves them into idle link. So the ceiling is the idle
  link, not the Belady number.
- **Break-even precision.** Measured precision of the prefetched rows is ≈0.23 (K=1, 0.85 budget). A
  predictor needs about 0.4 to be worth building (+20%, ~6 ms/token), and about 0.5 to reach +35%
  (~11 ms/token). K=1 is the right width at the realistic budget; K≥2 only pays at the 2x budget with
  precision ≥0.4.
- **Recommendation: build nothing yet. First capture the router input.** Take capture Y below, then score the
  **native gate of layer L+h applied to layer L's router input**. That predictor needs no training, and its
  weights are already on the GPU. It goes through this replay with the `NoisyOracle` precision curve as the
  decision rule. Build the E2 learned L+1 head only if the native gate misses 0.4 rank-1 precision on
  non-resident rows.

## Conflicts with the handoff (`DSV4.1 expert_prediction_handoff.md`, 2026-09-19)

Where the handoff and the current facts disagree, the current facts were used.

| Handoff | Now | Effect on this study |
|---|---|---|
| Objective: NVMe → RAM prefetch (hide NVMe wait) | Objective: RAM → VRAM hot-cache prefetch (hide link time) | The metric is link rows and exposed link rows, not drive reads |
| 888 hot slots (14,336 MiB − 3,048 MiB gather scratch) | 1,128 slots (28–29 per layer); stage DIRECT inserts on miss, no scratch | Capacities come from the route-log header |
| Residency: 32-forward boundary promotions; G 126.9 (c32) | DIRECT insert-on-miss with a per-token decayed score (0.98); G 78.8 (78.0 held-out) | Replayed with `DirectInsertReplay`, which is exact to the capture |
| RAM tier 70 GiB / 5,644 rows; 18.5 RAM misses/token | 100 GiB NUMA pinned tier plus mirrors; 11.26 RAM misses/token (14.3% of G) on varied24 (`graph_step` registers) | NVMe rows need more lead than 1–2 layers (see caveats) |
| ~2.8 tok/s, ~391 ms/step trace; per-layer lead 4.8 ms | 116 ms/token; ~2.9 ms per layer, of which ~0.85 ms has an idle link | Budget of 0.85 rows/layer, and 1.7 as the sensitivity case |
| Gather ≈1.055 ms/row (~12 GB/s) | SM copy ~12.2 GB/s, copy engine ~13.6 GB/s; ~1.0 ms per miss | 1.0 ms/row used (agrees within 5%) |
| E0 capture of hidden states comes first | The captures hold routes, plan misses and hot sets, but no hidden states or logits | E1 is route-only; E2+ needs capture Y |
| Split into train/dev/calibration/locked test by session | Even/odd prompts = train/held-out, as instructed | Tuned values (recency decay, cross-layer gate) were picked on train; no locked test exists |
| Previous-token routes: 0.000 NVMe recall | The same finding one tier up: previous-token rows are always VRAM-resident | 0 rows issued |
| The promotion-stall and boundary-burst discussion | Moot under DIRECT (no boundary promotions) | Not modelled |

## Method

**Data.** `divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/varied24/stages.jsonl`
(run `divix01-1619951-1790303243384310693`): 24 prompts × 256 tokens, 6,153 graph decode steps. Even prompts
are train (3,072 steps) and odd prompts are held-out (3,072 steps). `DirectInsertReplay` gives 78.787
misses/token against 78.783 measured (`varied24/validate.json`). The no-prefetch arm here reproduces the
policy sweep's held-out 77.99.

**`PrefetchReplay`** (`scripts/dsv41/prefetch_sim.py`) is `DirectInsertReplay` plus one step. Just before a
layer's gather, up to K predicted rows that are not resident are inserted. Each evicts the lowest-ranked slot
in that forward's own victim order that satisfies two conditions:

- It is not in the demand shortlist, so demand inserts exactly as before.
- It does not hold a row the predictor ranked above its last pick.

A wrong row is never routed, scores low and is evicted next. Prefetches that evict useful rows show up as
extra demand misses in the replay (pollution). Pollution measured ≈0.13 extra misses per wrong row
(noisy oracle p=0.25). With no candidates, `PrefetchReplay` is bit-identical to `DirectInsertReplay` (tested).

**Link model** (`link_exposed`). Each layer has two phases:

1. **Demand phase.** Exposed: the layer's misses, plus any unfinished prefetch aimed at this layer, which
   the gather must wait for on the serial link.
2. **Compute window.** `budget` rows of idle link. It drains the prefetch queue in FIFO order.

A prefetch for layer T is issued at the start of layer T−h's window, when that layer's routes are known. So
it can use h windows. For T<h it is issued in the previous token's tail, and only within the same request.
Estimated ms/token = 116 + (exposed − 77.99) × 1.0. Gap share = (77.99 − exposed) / (77.99 − 47.23), where
47.23 is the held-out Belady-with-bypass over one global pool (`sweep/varied24_1128.log`,
`opt_bypass_global`). The per-layer Belady is 50.43.

**Predictors.** Each uses only what exists at issue time.

| Predictor | What it does |
|---|---|
| `oracle` | The target's true routes |
| `prev_token` | Same layer, the request's previous token |
| `popularity` | Train-only per-layer counts, frozen |
| `recency d` | Per-layer route score decayed by d per token; d ∈ {0.5, 0.9, 0.98}, with 0.9 best on train |
| `cross_layer` | Train-only co-occurrence from the issue layer's six routes to the target's |
| `cross_layer p` | Gated: max single-source P(j \| i) ≥ p, support ≥ 5 |
| `noisy_oracle p` | A stand-in for a predictor of a given precision. It reads residency; see the docstring |

## Results (held-out steps; budget in rows of idle link per layer)

Baseline: 77.99 misses/token, 116 ms/token. "exposed" is link rows/token on the critical path.

| predictor | K | h | budget | demand | prefetch | precision | G_total | exposed | est ms/tok | gap share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | 1 | 1 | 0.85 | 45.43 | 34.72 | 1.00 | 80.14 | 50.64 | 88.6 | +89% |
| oracle | 1 | 4 | 0.85 | 45.44 | 34.70 | 1.00 | 80.14 | **47.50** | 85.5 | **+99%** |
| oracle | 2 | 4 | 1.7 | 22.47 | 59.93 | 1.00 | 82.40 | **23.25** | 61.3 | **+178%** |
| oracle | 6 | 1 | 0.85 | 0.02 | 86.79 | 1.00 | 86.81 | 56.31 | 94.3 | +70% |
| prev_token | any | any | any | 77.99 | 0.00 | – | 77.99 | 77.99 | 116.0 | 0% |
| popularity | 1 | 1 | 0.85 | 82.24 | 40.00 | 0.01 | 122.24 | 88.23 | 126.2 | −33% |
| popularity | 1 | 1 | 1.7 | 82.24 | 40.00 | 0.01 | 122.24 | 82.24 | 120.2 | −14% |
| recency 0.9 | 1 | 4 | 0.85 | 79.20 | 35.39 | 0.03 | 114.59 | 81.45 | 119.5 | −11% |
| recency 0.9 | 1 | 4 | 1.7 | 79.20 | 35.39 | 0.03 | 114.59 | 79.20 | 117.2 | −4% |
| cross_layer | 1 | 1 | 0.85 | 77.81 | 40.00 | 0.09 | 117.80 | 83.81 | 121.8 | −19% |
| cross_layer | 1 | 1 | 1.7 | 77.81 | 40.00 | 0.09 | 117.80 | 77.81 | 115.8 | +1% |
| cross_layer | 2 | 1 | 1.7 | 78.50 | 79.99 | 0.09 | 158.49 | 90.50 | 128.5 | −41% |
| cross_layer 0.5 | 1 | 1 | 0.85 | 77.16 | 30.10 | 0.12 | 107.26 | 81.67 | 119.7 | −12% |
| cross_layer 0.5 | 1 | 1 | 1.7 | 77.16 | 30.10 | 0.12 | 107.26 | 77.16 | 115.2 | +3% |
| cross_layer 0.5 | 1 | 4 | 0.85 | 77.79 | 28.07 | 0.10 | 105.86 | 77.91 | 115.9 | 0% |
| noisy_oracle 0.25 | 1 | 4 | 0.85 | 73.07 | 38.71 | 0.23 | 111.78 | 77.80 | 115.8 | +1% |
| noisy_oracle 0.4 | 1 | 4 | 0.85 | 67.38 | 37.94 | 0.37 | 105.32 | 71.41 | 109.4 | +21% |
| noisy_oracle 0.5 | 1 | 4 | 0.85 | 63.55 | 37.41 | 0.47 | 100.96 | 67.14 | 105.2 | +35% |
| noisy_oracle 0.6 | 1 | 4 | 0.85 | 59.74 | 36.92 | 0.57 | 96.66 | 62.96 | 101.0 | +49% |
| noisy_oracle 0.75 | 1 | 4 | 0.85 | 54.09 | 36.10 | 0.72 | 90.19 | 56.80 | 94.8 | +69% |
| noisy_oracle 0.5 | 2 | 4 | 1.7 | 52.23 | 72.95 | 0.45 | 125.18 | 57.59 | 95.6 | +66% |
| noisy_oracle 0.75 | 2 | 4 | 1.7 | 36.45 | 67.42 | 0.70 | 103.87 | 38.76 | 76.8 | +128% |

Full tables (every K ∈ {1,2,4,6} × h ∈ {1,2,4} × budget, per predictor, held-out and train):
`divix01:.../hot-cache-policy/prefetch-study/table_test.md` and `table_train.md`.

Patterns worth knowing:

- h matters little for the cost of a single row. It matters for pooling windows: the oracle at K=1 goes from
  +89% (h=1) to +99% (h=4).
- K>1 at the 0.85 budget is always worse, even for the oracle (K=6: +70%). The extra rows cannot be hidden.
- The co-occurrence gate overfits. Its precision is 0.25 on train and 0.12 held-out, so the 24-prompt corpus
  is too small for (layer, expert, expert) statistics.

## What E2 needs: capture Y (smallest addition)

The router input for each streamed layer, and each decode forward, in the existing `GraphRouteLog` ring:

- **Tensors per layer per forward:**
  - `x`: the normalized MoE/router input, bf16 `[5120]`. It is `_hc_combine(..., norm=post_attention_layernorm)`
    at `python/sglang/srt/models/deepseek_v4.py:3683-3690`, asserted bf16/5120 at ~:3697, and reaches the
    method as `dispatch_output.hidden_states`.
  - `topk_weights`: fp32 `[6]`.
  - Optional: `router_logits` fp32 `[384]` (`MoEGate.forward`, `deepseek_v2.py:513-557`) for the E0 gate
    self-check. It can also be recomputed offline from `x` with the checkpoint gate weights and correction
    bias.
- **Size:** ~10.3 KB per layer (+1.5 KB with logits), or ~410 KB per token for 40 layers. Varied24 would be
  ~2.5 GB. That is a binary side file, not JSON lines. A 64-deep device ring is ~26 MB of VRAM, about two hot
  slots, and exists only while the trace is on.
- **Hook:** `GraphRouteLog.record(row, routes, count)` (`python/sglang/srt/layers/moe/exl3_stream_trace.py:153-166`)
  already runs inside the captured graph at every layer's gather. It is reached from
  `Exl3RamMissRowBackend.post` (`python/sglang/srt/layers/moe/exl3_ram_miss.py:506-509`), which is called
  from `_apply_graph` (`python/sglang/srt/layers/quantization/exl3.py:463-468`). There, `topk_ids` and `x`
  both exist before the gather.
  - Change: pass `x` and `topk_weights` into `record` and `index_copy_` them into two new rings beside
    `routes`.
  - `poll`/`_emit` read them back one batch behind, as today.
  - Joins use the existing `seq`/`forward_pass_id`/`rids`.
- **Alternative:** the `RouteTaps` path (`expert_prediction/taps.py:98-120`) already copies ROUTER_INPUT,
  LOGITS, IDS and WEIGHTS graph-safely. It rejects DSV's `StandardTopKOutputPacked` (`:99`), so a one-line
  type fix would make it work. But it writes the older capture schema with no hot-set or plan-miss join. The
  ring extension keeps a single joined log, which this replay needs.

**Estimated value:**

- **Native-gate lookahead.** In the handoff's eager probe it had rank-1 precision on RAM-absent rows of 0.68
  (L+1) and top-6 all-route recall of 0.645 (L+1) and 0.571 (L+2). VRAM-absent rows are a different
  denominator, so treat 0.4–0.6 as the plausible rank-1 range at h=1–2. On this replay that is +20% to +49%
  of the gap: **~6–15 ms/token at the 0.85 budget**, and up to ~20 ms/token at 2x with K=2.
- **E2 (learned L+1 head).** Worth building only if it beats the native gate's measured precision. The
  handoff reports no DSV numbers for it.

## Caveats

- **NVMe-sourced misses.** 11.26/token (14%) are also RAM misses. A prefetch of one first needs the NVMe
  read, which takes ~3.4–7 ms against ~2.9 ms per layer. The replay treats every row as RAM-ready at issue, so
  the oracle ceiling is optimistic by up to ~11 rows/token at h≤2.
- **The idle link is uniform in the model.** It is (116−82)/40 per layer. Engram layers 1 and 14, and the
  step tail, differ. The copy-overlap work may change both the budget and the demand-phase serialization.
- **Prefetch victims are placed outside the demand shortlist.** A runtime could reserve slots differently.
  Pollution is small (~0.13 misses per wrong row), so the conclusions do not depend on this.
- **One corpus, one split.** 12 held-out prompts, and no session-level confidence intervals.

## Reproduce

On divix01, in `wt-prefetch-study`. The commands and commits are also in `prefetch-study/*.cmd`.

```bash
O=/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/prefetch-study
T=/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/varied24/stages.jsonl
# at 51ff391c15: e1_varied24.json (85 arms)
PYTHONPATH=$PWD/python OMP_NUM_THREADS=1 taskset -c 0-63 /data/models/slang/.venv/bin/python \
  analysis/dsv41-drive/prefetch-study/prefetch_baselines.py $T --prompts 24 --belady-test 47.23 --workers 32 \
  --out $O/e1_varied24.json
# at 3918dca71c: e1b_varied24.json (gated cross-layer, noisy oracle)
PYTHONPATH=$PWD/python OMP_NUM_THREADS=1 taskset -c 0-63 /data/models/slang/.venv/bin/python \
  analysis/dsv41-drive/prefetch-study/prefetch_baselines.py $T --prompts 24 --belady-test 47.23 --workers 32 \
  --ks 1 2 4 --horizons 1 2 4 --gates 0.1 0.2 0.3 0.5 --precisions 0.25 0.4 0.5 0.6 0.75 \
  --only cross_layer noisy_oracle --out $O/e1b_varied24.json
/data/models/slang/.venv/bin/python analysis/dsv41-drive/prefetch-study/summarize.py \
  $O/e1_varied24.json $O/e1b_varied24.json > $O/table_test.md
```

Both runs share arms such as `none` and `cross_layer k=1 h=1`, and those give identical numbers.

Tests (37 passed; 14 new in `test_prefetch_sim.py`, 23 existing in `test_tier_sim.py`):

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=4 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_prefetch_sim.py test/manual/dsv41/test_tier_sim.py -q -p no:randomly
```
