# DSV4.1 EXL3 router-input capture and native-gate lookahead

**Date:** 2026-09-24/25. **Branch:** `cc/router-capture`, based on `shared/cc/prefetch-study` (`2db5e23fff`).
**Continues:** `2026-09-25-dsv41-prefetch-study.md`. Its recommendation was the spec for this work: capture the
router input, then score layer L+h's own gate applied to layer L's input.

## Answer

- **The native gate clears the 0.4 bar at every horizon we scored.** Held-out rank-1 precision on
  non-resident rows is **0.68 at h=1**, 0.57 at h=2 and 0.44 at h=4.
- **The best arm is h=1, K=1, prefetching only the gate's own top-6.** It gives **+56% of the 31-miss gap
  (~17 ms/token) at the 0.85-row budget** and +74% (~23 ms/token) at 1.7. With K=2 at the 1.7 budget it
  reaches +86% (~27 ms/token). The measured precision of the prefetched rows is 0.69, which sits on the
  NoisyOracle price curve.
- **Recommendation: build the runtime path for the h=1 native gate (sketch below). Do not build the E2
  learned head now.** Its best case is the remaining 56% → 89% (oracle) gap at h=1. That is worth a later
  look, but only after the native gate runs live.
- **The capture costs nothing measurable.** Decode was 142.5 ms/token against Track B's 143.0 on the same 24
  prompts, including prefill. Generation throughput was 7.99 tok/s in both runs, and all 24 outputs were
  byte-identical.

## Capture design

- **Hook.** `GraphRouteLog.record_router(row, x, topk_weights)`. `Exl3MoEMethod._apply_graph` calls it right
  after `streamer.gather`. By then the gather's post has run the layer's `record`, and row 0's `record` has
  taken the forward's ring slot. It makes two `index_copy_` calls into device rings on the route log's
  slots:
  - `router_x`: bf16 `[depth, 40, 5120]`.
  - `router_w`: fp32 `[depth, 40, 6]`.

  Logits are not captured. The offline self-check below recomputes them and reproduces the routes, so they
  add no information.
- **Readback.** Unchanged. `poll` copies the whole ring through pinned buffers without waiting, one batch
  behind, and `_emit` writes each finished forward.
- **Side files.** `RouterCapture` writes binary files at `SGLANG_DSV41_ROUTER_CAPTURE_PATH`:
  - `<prefix>.x.bin`: raw bf16 `[records, 40, 5120]`.
  - `<prefix>.w.bin`: fp32 `[records, 40, 6]`.
  - `<prefix>.seq.bin`: int64 `[records, 2]`, holding `seq` and the pass id.
  - `<prefix>.json`: shapes, layer ids and run id.

  Each `graph_routes` line names its record as `router: i`. `ROUTE_LOG_SCHEMA` 2 adds that field and nothing
  else. The files are flushed per poll, because the scheduler can be SIGKILLed at shutdown.
- **Opt-in.** The capture needs both `SGLANG_DSV41_ROUTER_CAPTURE_PATH` and the stage trace. Setting it
  without `SGLANG_DSV41_EXPERT_TRACE_PATH` is refused before the service allocates anything. The
  `smoke_routes.sh` switch is `ROUTER_CAPTURE=1`. No `arm_env` default changed.
- **Ring size.**
  - With router capture on, the route ring is 32 deep instead of 64. The router rings are **13.1 MB of VRAM**
    (`router_x` 13,107,200 B, `router_w` 30,720 B), about one hot slot. The whole ring is ~13.5 MB.
  - The pinned host mirror is the same size. Each poll copies ~13.5 MB from device to host.
  - The ring is sized by the first warmup forward. Allocating it after capture or after the first read is
    refused.
  - Varied24 wrote 6,155 records: **2.52 GB, 409.6 KB/token**.
- **Trace-off purity.** `test_apply_graph_adds_nothing_unless_router_capture_is_on` records every aten op
  that `_apply_graph` dispatches:
  - Trace off, and trace on with router capture off, give identical op lists.
  - Router capture adds exactly two `index_copy_` calls, plus views.

## Run

`ROUTER_CAPTURE=1 smoke_routes.sh router-capture <wt> prompts_varied.json 256 1` ran at `e0af0f638d`, in
`divix01:/data/models/slang/nvfp4-work/wt-router-capture`, under `cc-gpu.lock` and `rowimg-disk.lock`. It
used the arm_env defaults and the 100 GiB tier. The run id is `divix01-1984105-1790309672451706496`, and the
output is under `divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture/`.

| | Track B varied24 | this capture |
|---|---:|---:|
| decode graph forwards | 6,153 | 6,153 |
| dropped entries | 0 | 0 |
| DirectInsertReplay G | 78.787 | **78.780** |
| measured G (route log / registers) | 78.783 / 78.759 | 78.777 / 78.755 |
| held-out no-prefetch misses/token | 77.99 | 77.99 |
| held-out Belady (global, bypass) | 47.23 | 47.23 |
| ms/token incl. prefill (24 × 256) | 143.0 | 142.5 (per-prompt Δ −0.12 ± 0.32 s) |
| gen throughput (server log) | 7.99 tok/s | 7.99 tok/s |

## Validation (h=0)

Each layer's checkpoint gate was applied offline, in fp32, to its own captured input. The gate is
`layers.{i}.ffn.gate.weight` plus `.bias` as the score correction, scored as `sqrt(softplus)` + bias, top-6.

- **Routes:** 246,114 of 246,120 (layer, step) route sets were reproduced (**99.998%**). The six that
  differ are near-ties between the production bf16×bf16→fp32 GEMM and the offline fp32 one.
- **Weights:** the captured weights equal the renormalized scores. The maximum absolute error is 7.8e-5 and
  the p99 is 9e-8. The weights sum to 1.0, which means `routed_scaling_factor` 1.5 is applied at the output,
  not in the weights.
- **Hash layers:** the checkpoint has none. All 40 MoE layers have a gate and a bias.

## Precision (held-out, offline, residency from the no-prefetch replay)

The predictor for target T is the first K experts of gate_T(x_{T−h}) that are not resident just before T's
gather, taken from the gate's top-6 (d=6). For T<h the source is the previous token of the same request.

| h | rank-1 precision | rank-1 \| T has a miss | precision K=2 | recall K=1 | recall K=2 | recall top-6 | train rank-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0.678** | 0.738 | 0.594 | 0.312 | 0.476 | 0.580 | 0.652 |
| 2 | **0.573** | 0.636 | 0.484 | 0.270 | 0.404 | 0.492 | 0.543 |
| 4 | **0.438** | 0.496 | 0.357 | 0.212 | 0.316 | 0.393 | 0.407 |

Cutting the ranking to the gate's top-6 (d=6) instead of going deeper (d=12/48) raises precision by 0.02 to
0.06. At h=1, K=1, that is 0.68 against 0.62. With d=6, a layer whose six predicted routes are all resident
gets no prefetch.

## Prefetch replay (held-out; PrefetchReplay + link_exposed, 1.0 ms/row, base 116 ms/token)

| predictor | K | h | budget | demand | prefetch | precision | exposed | est ms/tok | gap share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | – | – | – | 77.99 | 0.00 | – | 77.99 | 116.0 | 0% |
| **native gate d6** | 1 | 1 | 0.85 | 55.36 | 35.75 | 0.69 | **60.73** | **98.7** | **+56%** |
| native gate d6 | 1 | 1 | 1.7 | 55.36 | 35.75 | 0.69 | 55.36 | 93.4 | +74% |
| native gate d6 | 2 | 1 | 0.85 | 43.55 | 61.89 | 0.62 | 75.08 | 113.1 | +9% |
| **native gate d6** | 2 | 1 | 1.7 | 43.55 | 61.89 | 0.62 | **51.40** | **89.4** | **+86%** |
| native gate d6 | 1 | 2 | 0.85 | 58.79 | 36.36 | 0.58 | 62.47 | 100.5 | +50% |
| native gate d6 | 1 | 2 | 1.7 | 58.79 | 36.36 | 0.58 | 58.79 | 96.8 | +62% |
| native gate d6 | 2 | 2 | 1.7 | 49.73 | 63.49 | 0.51 | 52.73 | 90.7 | +82% |
| native gate d6 | 1 | 4 | 0.85 | 63.61 | 37.39 | 0.45 | 67.25 | 105.3 | +35% |
| native gate d6 | 1 | 4 | 1.7 | 63.61 | 37.39 | 0.45 | 63.61 | 101.6 | +47% |
| native gate d6 | 2 | 4 | 1.7 | 57.54 | 66.54 | 0.38 | 59.71 | 97.7 | +59% |
| oracle | 1 | 1 | 0.85 | 45.43 | 34.72 | 1.00 | 50.64 | 88.6 | +89% |
| oracle | 2 | 1 | 1.7 | 22.45 | 59.95 | 1.00 | 29.87 | 67.9 | +156% |
| noisy oracle 0.6 | 1 | 1 | 0.85 | 59.68 | 36.92 | 0.57 | 65.21 | 103.2 | +42% |
| noisy oracle 0.75 | 1 | 1 | 0.85 | 54.03 | 36.13 | 0.72 | 59.45 | 97.5 | +60% |
| noisy oracle 0.75 | 1 | 4 | 0.85 | 54.08 | 36.12 | 0.72 | 56.82 | 94.8 | +69% |

The oracle and NoisyOracle rows reproduce the study's to two decimals, because this capture's replay matches
Track B's. Every arm on both splits is in `router-capture/table_test.md` and `table_train.md`.

- **Against the price curve.** At h=1, K=1, 0.85, the native gate's measured precision of 0.69 falls between
  the NoisyOracle's 0.57 (+42%) and 0.72 (+60%), and gains +56%. Its value comes from its precision, not
  from its choice of which rows to fetch.
- **Horizon.** Precision falls faster with h than lead time helps. The oracle gains +10 points from h=1 to
  h=4. The native gate loses 21 points. h=1 is the design point.
- **K.** At the realistic 0.85 budget, K=2 loses: its extra rows cannot be hidden. K=2 wins only at a 1.7
  budget, which only a faster copy path provides.

## Recommendation: the h=1 native gate at runtime

**Where it runs in the graph.** It runs in layer T−1's `_apply_graph`, after that layer's gather has been
posted and before its fused MoE:

1. **Predict.** Compute `logits = x_{T−1} @ W_Tᵀ` with layer T's gate weight (384×5120 bf16, 3.9 MB, already
   in VRAM). Use the same `tiny_gemm_bf16` + `moe_fused_gate(sqrtsoftplus_log1p, bias_T)` as T's router, top-6.
   That is one GEMV and one top-k per layer. **Its cost must be measured first:** at ~10 µs it would be
   ~0.4 ms/token, against ~17 ms/token saved.
2. **Plan, on the device.** A small kernel does three things:
   - Filters the six predicted ids against layer T's residency bank (`slot_to_expert` / hot page).
   - Keeps only rows that are **RAM-ready in the pinned tier**. At h=1 there is no lead for the NVMe read.
     This drops up to ~14% of candidates, and the replay's figures are optimistic by about that share.
   - Takes the first K=1 and picks the victim by the replay's rule: the lowest-ranked slot that is not in T's
     demand shortlist and not holding a predicted row.
3. **Copy on a second stream.** A graph-captured fork (event record/wait) issues the H2D row copy on a copy
   engine. It then runs concurrently with T−1's expert compute and T's attention, which are the link's idle
   window. Layer T's gather waits on the prefetch's completion event before its planner reads residency. That
   is the link model's rule that an unfinished prefetch delays the target like a miss. It also keeps the
   planner from treating an in-flight row as resident.
4. **Cross-token case (T=0).** The source is the previous token's layer 39. Issue its prefetch in the step
   tail, only when the next step continues the same request. Otherwise skip T=0.

**What it needs from copy-overlap.** This design needs a second stream and a copy engine inside the decode
graph. Both are currently blocked by the Engram host nodes in that graph. Specifically:

- **(a)** A captured side stream for the prefetch copy that can overlap the main stream's compute. The
  Engram host nodes (layers 1 and 14) serialize the graph today.
- **(b)** The copy engine path (~13.6 GB/s, against ~12.2 for the SM copy) so prefetch does not take SMs from
  expert compute.
- **(c)** Demand-over-prefetch priority on the single PCIe link, so a mispredicted row never delays a demand
  miss of layer T−1.
- **(d)** The real idle link per layer after overlap. It sets the budget: at 0.85 use K=1. K=2 pays only if
  overlap lifts the budget toward 1.7 (+86%).

**Before building:** measure the per-layer gate GEMV cost in the graph. Rerun this replay with the
RAM-ready filter, which needs the pinned tier's residency. That residency is not in the route log, so it
needs a `layer_ram_rows`-style snapshot or the stage records.

**The E2 learned head is not specified here.** It is needed only if the live native gate falls short of
this replay. The data it would need is the capture this branch produces.

## Caveats

- **NVMe rows.** 11.26 per token are RAM misses as well as VRAM misses. At h=1, 2.9 ms of lead cannot hide
  their ~3.4 to 7 ms read, so they must be filtered out. The replay treats every row as RAM-ready.
- **The link model.** Its uniform budget is carried over from the study, with the same caveats (Engram
  layers, step tail).
- **One corpus.** Varied24, even/odd split, no confidence intervals. Train rank-1 precision is ~0.03 below
  test at every h, so no fitting leaked. The only choice made was the depth, d=6. It is also the best depth
  on train at the 0.85 budget (63.04 exposed, against 63.24 for d=12). At 1.7, d=12 is 0.4 rows better.

## Reproduce

```bash
# divix01, wt-router-capture
O=/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture
bash $O/run.sh $PWD   # GPU tests under cc-gpu.lock, then ROUTER_CAPTURE=1 smoke_routes.sh (varied24)
PYTHONPATH=$PWD/python OMP_NUM_THREADS=4 taskset -c 0-63 /data/models/slang/.venv/bin/python \
  analysis/dsv41-drive/hot-cache-policy/validate_replay.py $O/stages.jsonl --out $O/validate.json
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python \
  analysis/dsv41-drive/router-capture/router_score.py $O/stages.jsonl $O/router \
  /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40 --prompts 24 --workers 24 --out $O/score.json
/data/models/slang/.venv/bin/python analysis/dsv41-drive/prefetch-study/summarize.py $O/score.json > $O/table_test.md
```

## Tests

```bash
# CPU (GPU hidden): 59 passed, 4 skipped (GPU), 1 failed
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-63 /data/models/slang/.venv/bin/python \
  -m pytest test/registered/unit/layers/moe/test_exl3_stream_trace.py \
  test/registered/unit/layers/moe/test_exl3_ram_miss_service.py -q -p no:randomly
# GPU, under cc-gpu.lock: 2 passed (route log and router capture CUDA-graph replay)
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 32-63 /data/models/slang/.venv/bin/python \
  -m pytest test/registered/unit/layers/moe/test_exl3_stream_trace.py -k captured_graph -q -p no:randomly
# scorer and replays: 39 passed (2 new in test_router_score.py)
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 0-31 /data/models/slang/.venv/bin/python \
  -m pytest test/manual/dsv41/test_router_score.py test/manual/dsv41/test_prefetch_sim.py \
  test/manual/dsv41/test_tier_sim.py -q -p no:randomly
```

The one CPU failure, `test_ram_miss_requests_are_traced_and_skipped_by_tier_sim`
(`KeyError: 'pack_workers'`), also fails at the base `2db5e23fff`. It is not from this branch.

New tests:

- 4 in `test_exl3_stream_trace.py`: the side-file join through warmup, ring wrap and lagging reads; no
  router ring or field when off; the refusals; and the CUDA-graph replay.
- 4 in `test_exl3_ram_miss_service.py`: refusal without the trace, the ring depth and router on/off, and
  `_apply_graph` op purity.
- 2 in `test_router_score.py`.
