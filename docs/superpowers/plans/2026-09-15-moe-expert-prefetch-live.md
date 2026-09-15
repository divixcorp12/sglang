# MoE Expert Prefetch (LLaPor → Prefetch Plan → In-Graph Copy / Doorbell) Live Test Implementation Plan

> **Dependency, stated first:** only the doorbell backend (Task 10) and the two doorbell-on matrix cells wait on crypto-c9's `cc/doorbell-serving` landing on `codex/nvfp4-expert-stream-main`. It is not on `shared` yet. Tasks 1–9, including the default `in_graph_copy` backend and the prefetch-on/doorbell-off live cell, need nothing from it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score next-layer experts with the trained LLaPor predictor (and APEX if feasible) inside the decode CUDA graph. Turn the scores into a backend-independent per-layer prefetch plan. Serve the plan with the existing in-graph copy kernel by default, or with the doorbell copier when it is enabled. Then measure logprob parity and live decode tok/s across every prefetch × doorbell combination.

**Architecture:**
- **Model-independent package.** A new `layers/moe/expert_prediction/serving/` package loads the trained checkpoints, validates them against the live `MoeLayerSpec`s, and builds static-shape bf16 scorers.
- **In-graph scoring through generic hooks.** Scorers attach through one new generic hook, `FeatureStore.after_write`. It fires from the existing TopK taps and pre-mixer adapters. On a decode forward it writes each target layer's top-C expert ids and scores into a stable device bank, with no host sync.
- **One plan interface, two backends.** An in-graph planner turns candidates into a `PrefetchPlan` per target layer: int64 row ids `[B]`, destination slots `[B]`, a device count, and a delivered mask `[B]`. The planner drops resident experts and clamps to budget B by score.
  - `in_graph_copy` (default, doorbell off) copies plan rows into B dedicated prefetch rows per layer with `copy_expert_row_segments_gpu`.
  - `doorbell` (optional, `SGLANG_MOE_EXPERT_DOORBELL=1`) posts the same plan to the doorbell copier.
  - The graph gather treats delivered plan rows as hits. The residual (actual misses minus delivered) goes through the existing in-graph miss copy.
- **Ordering.** The in-graph copy backend ships first. Only the doorbell-backend cells wait on `cc/doorbell-serving`.
- **Gate before live.** An offline replay of the full capture measures recall of non-resident native experts within the budget. It also prices both backends' timelines, and decides go/defer before any live GPU time.

**Tech Stack:** Python 3.13, torch 2.13 (CUDA graphs, `torch.cuda.set_sync_debug_mode`), msgspec, safetensors, SGLang `envs`, the JIT copy kernel `copy_expert_row_segments_gpu`, the breakable decode CUDA graph (`eager_on_graph`), and optionally the doorbell copier (`ExpertDoorbellCopier`, `cc/doorbell-serving`, Task 10 only).

**Spec and research inputs (read before any task):**
- LLaPor serving: `/home/dimitri/data/divix/crypto/trading-framework/mechanism_hunter/docs/superpowers/plans/2026-09-14-llapor-gpu-only.md` §6–8 (read-only).
- APEX serving: `.../2026-09-14-apex-gpu-only.md` §5, Tasks 5–8 (read-only).
- Doorbell: `/home/dimitri/data/divix/crypto/NVFP4_DOORBELL_COPIER.md` §3, §6 (read-only). Prototype API is `shared/cc/doorbell-prototype:python/sglang/kernels/ops/moe/expert_doorbell.py`.
- Trained models: `docs/superpowers/experiments/2026-09-15-expert-predictor-offline-training.md`. Checkpoints are at `/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630/{llapor/pair-NN,apex/layer-NN}`.
- Capture: `/mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438`. Sessions are in `/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl`.
- Timing evidence: `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md` E26–E32 (E27 overlap, E28 window, E29/E32 doorbell).

## Global Constraints

- **Doorbell dependency (D-on cells only):** Task 10 and the matrix cells with `SGLANG_MOE_EXPERT_DOORBELL=1` need the doorbell serving integration from `cc/doorbell-serving` (owner: the crypto-c9 production session) on `codex/nvfp4-expert-stream-main`. Every other task, and the in-graph copy backend, must build, test and run with `SGLANG_MOE_EXPERT_DOORBELL` unset. If the landed doorbell API differs from the one assumed in Design, only `serving/doorbell_backend.py` (Task 10) changes.
- **One plan interface, backends behind it.** The planner (Task 4) owns candidate filtering, the budget clamp and destination rows. A backend only moves bytes and sets `PrefetchPlan.delivered`. No backend reads scores, and the gather reads nothing but the delivered ids.
- **Prefetch never writes hot slots.** Plans target B dedicated rows per layer after the graph-gather scratch rows. The in-graph residency updater (E31) stays the only writer of `expert_to_slot`, slot states and hot rows.
- **Prefetch off must be production.** With `SGLANG_MOE_EXPERT_PREFETCH` unset, the hot-cache layout, the recorded graph kernels, and the logprobs must equal the pre-change build. Every code change is guarded so the off path launches exactly the kernels it launches today.
- **Model-agnostic placement:**
  - Nothing goes in `python/sglang/srt/models/*` and nothing hardcodes Qwen3.8-Next dimensions.
  - Dimensions come from checkpoint manifests, validated against live `MoeLayerSpec` (`layer_id`, `num_experts`, `top_k`, `hidden_size`).
  - Model-specific facts pass only through `python/sglang/srt/layers/moe/expert_prediction/adapters.py`:
    - `_PRE_MIXER_ADAPTERS` / `register_pre_mixer_adapter` (which tensor is pre-mixer);
    - the new `_MIXER_KIND_ADAPTERS` / `register_mixer_kind_adapter` (mixer kind per layer, used only for metric labels).
- **Everything per token lives in the decode CUDA graph.** No `.item()`, `.tolist()`, `.cpu()`, host syncs, Python loops over rows, or dynamic shapes on the decode path. Python runs only at setup, during graph capture, and for eager prefill forwards (which skip scoring). Metric readback happens once per log interval.
- **Frozen `model_runner.py`:** orchestration-only edits (read `.claude/skills/large-class-style/SKILL.md` first). Env vars go through `envs` (read `.claude/skills/env-var-conventions/SKILL.md`). Use `msgspec.Struct`, not dataclass. No defensive `getattr`/`hasattr`.
- **GPU etiquette (hard):**
  - Every GPU process takes `flock` on `/data/models/slang/nvfp4-work/cc-gpu.lock`.
  - Never start while a server on port 7867 is relaunching: if a 7867 process exists, `http://127.0.0.1:7867/health` must be 200 and the run must fit beside production. Only CPU work or GPU unit tests under 1 GiB qualify.
  - Production is taken down only with the user's explicit approval, obtained with AskUserQuestion by the task that needs it. With production down by approval, the GPU must be empty before launch.
  - While the doorbell session owns the GPU, run no GPU work at all. CPU steps set `CUDA_VISIBLE_DEVICES=""`.
- **Code and tests run on divix01.** Laptop commits, then `git push shared HEAD` is followed by a sync of `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree`: `git fetch -q /data/models/slang/nvfp4-work/remotes/sglang-nvfp4.git codex/nvfp4-expert-stream-main && git checkout -q --detach FETCH_HEAD`. Use `ssh -n divix01 '...'` only; never `ssh -t` and never `tmux capture-pane`.
- **Test command (divix01):** `cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && timeout 900 /data/models/slang/.venv/bin/python -m pytest -q -p no:cacheprovider -rfEs <test files>`
- **Commits:** stage by name, `git commit -m "..." -- <paths>`. Never stage `.omc/`, `.superpowers/`, or `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md`. End messages with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NurjMVe2nS3PGBqBr8M8MZ
  ```
- **Never modify the serving worktree or the production launcher.** The serving worktree is `/data/models/slang/nvfp4-work/main-port-probe-7bc4eb`; the production launcher is `run-nvfp4-e16c-public.sh`. Never relaunch production; the production session owns it.

---

## Design

### Timing facts this plan is built on (measured, E27–E32)

| Fact | Value | Source |
|---|---|---|
| Decode step | One unbroken CUDA graph (`segments=1 breaks=0`) | E28 |
| Order within a layer | norm, mixer, shared expert, router/TopK, route planning, the layer's miss copy, MoE kernels | E28 |
| Window from copy(L) end to copy(L+1) start | 0.267 ms before a linear-attention layer, 0.290 ms before a full-attention layer (p50) | E28 |
| In-graph miss copy | 0.2239 ms/row + 0.006 ms (11.5 GiB/s) | E28 |
| Doorbell copy on a torch-created stream | ≈0.209 ms/row (0.628 ms for 3 rows, 12.4 GiB/s) | E32 |
| Doorbell post → thread sees the request | 29–75 µs | E29/E32 |
| Doorbell waiting on an already-landed request | 15–22 µs | E29 |
| Misses per layer per token | mean 2.72, p90 6 at 4,957 slots; production now has 4,180 slots, so expect more | E28 |
| Idea-2 ceiling (perfect recall, free prediction) | ≈10 ms/token (+15%) | E28 |
| Per-row expert bytes | 2,764,808 | server log |
| Graph scratch per layer | 10 rows (1.33 GiB total) | server log |
| Copy under compute | No measurable contention either way (±0.01 ms compute, ~0.1 GiB/s copy); a copy graph captured on a side stream replays on whichever stream is current | E27 |
| Two copies at once (side copy for L+1 during L's own miss copy) | **Unmeasured** — E27 measured copy vs compute only | Task 8 |

### Where each predictor's output can be used

- **LLaPor (target L+1, source L's router features).**
  - Scores and the L+1 plan are ready inside layer L's gather, after the residency update and before L's own route plan and miss copy.
  - **Gap window** (side copy starts after L's miss copy): E28's 0.267/0.290 ms. That fits `floor(0.267 / 0.2299) = 1` row.
  - **Overlap window** (side copy starts before L's miss copy): gap + L's copy time, ≈ 0.267 + 0.006 + 2.72 × 0.2239 ≈ 0.88 ms, about 3 rows. It is valid only if two concurrent copy launches don't slow each other, which Task 8 measures.
- **APEX (target L, source L's pre-mixer).**
  - Scores are ready at layer L's input. The copy must land before L's own gather, so the window is layer L's norm + mixer + shared expert + router + planning, about 0.267 − 0.06 ≈ 0.21 ms (linear) and 0.23 ms (full).
  - In-graph side stream: 0 rows fit (one row is 0.2299 ms). Worse, the scores exist only after the pre-mixer tap, and a break there splits the mixer segment. **Not feasible.**
  - Doorbell: reaction plus one row is 0.03–0.075 + 0.216 ms, also longer than the window. It needs a same-layer post/resolve, and every hit row pays `reaction + n·0.209 − 0.21` ms of wait.
  - `at_target` (copy at L's gather): correct but no faster than baseline.
  - **Decision: APEX is priced offline (Task 3) and deferred from the live test**, unless Task 3 shows ≥3 ms/token *and* the doorbell supports same-layer tags. The scorer is still built and tested (Task 2).

### Prefetch plan interface (backend-independent)

```python
class PrefetchPlan(msgspec.Struct, frozen=True):
    """One target layer's prefetch request. Every tensor is a stable device buffer rewritten in place."""
    target_layer: int
    expert_ids: torch.Tensor         # int64 [B]: row ids to fetch, best score first
    destination_slots: torch.Tensor  # int32 [B]: the target layer's dedicated prefetch rows
    destination_rows: torch.Tensor   # int64 [B]: the same rows, for index_copy_ on device-resident tensors
    count: torch.Tensor              # int32 [1]: min(B, non-resident candidates); rows >= count are padding
    delivered: torch.Tensor          # bool [B]: set by the backend; True only once that row's bytes are safe to read
```

- **Planner** (`serving/plan.py`, in-graph, main stream):
  1. Read the bank's top-C candidates for the target.
  2. Mask residents (`expert_to_slot >= 0`).
  3. Take the top-B non-resident candidates by score; residents sort last.
  4. Write `expert_ids` and `count`, and clear `delivered`.
- **Residual:** actual misses − delivered. The graph gather calls `plan_graph_routes(..., prefetch_ids=where(delivered, expert_ids, -1), prefetch_base)`.
  - Routes to delivered experts remap to their prefetch row.
  - Every other non-resident route takes the existing scratch-row miss copy. `routed_miss_rows` becomes the residual.
  - An undelivered, late, or garbage plan therefore only costs time, never bytes.
- **Dedicated prefetch rows:** `ExpertHotCacheManager.from_model(prefetch_rows=B)`.
  - Scratch per layer becomes graph rows + B, and `enable_graph_gather(graph rows)` is unchanged.
  - Prefetch rows start at `capacity + graph_gather_rows`. The residency updater's dump row (`capacity`, the first graph scratch row) is untouched.
  - The budget deduction is automatic: B × 48 × 2.64 MiB (B = 3 → 380 MiB, B = 10 → 1.27 GiB, taken from hot slots). Live arms keep `HOT_GPU_MB` equal, so prefetch pays for its rows in hot slots; the report counts the lost slots.
- **Streamer hook:** `ExpertStreamer.prefetch: LayerPrefetch | None` (default `None`). In `_gather_graph`, after `residency_update.on_graph_forward` and before `plan_graph_routes`:
  - `prefetch_ids = self.prefetch.resolve()` resolves this layer's plan;
  - `self.prefetch.launch_next()` plans the next layer and starts its copy.
  - With `prefetch is None` the recorded kernels are exactly today's.

### Backends

| Backend | Selected by | Launch (source layer L's gather) | Resolve (target gather) | Status |
|---|---|---|---|---|
| `in_graph_copy` / `at_target` | `SGLANG_MOE_EXPERT_PREFETCH_SCHEDULE=at_target` | nothing | plan the target, copy its `count` rows on the main stream, `delivered = arange < count` | Task 4. A correctness and plumbing mode with zero overlap, so never faster than baseline |
| `in_graph_copy` / `side_stream` (default) | `...SCHEDULE=side_stream` | plan L+1; `main_seq += 1` at the first source; record a main-stream event; **graph break**: `side.wait_event(event)`, replay L+1's captured side graph (per row j: copy 1 row, then `ready[j] = side_seq`) | `delivered = (ready == main_seq) & (arange < count)`, read before the route plan | Task 5 |
| `doorbell` | `SGLANG_MOE_EXPERT_DOORBELL=1` | plan L+1, `copier.post(expert_ids, destination_slots, count, tag=L+1)` | `copier.wait(tag)`, then `delivered = arange < count` | Task 10, **blocked on `cc/doorbell-serving`** |

- **Why `side_stream` is the default schedule rather than copying at L+1 before MoE:** only a copy that runs while the main stream computes can save time.
  - E27: overlap saving equals min(copy, window)/copy, with no contention against compute.
  - `at_target` does the prefetch copy exactly where the miss copy would have run, plus wasted rows, so it can only lose. It stays as a correctness mode and a backend-plumbing check.
  - Task 8 measures both on real modules, and the live matrix runs `at_target` as a probe-only correctness cell.
- **Why a graph break per source layer:** E27 found that a copy graph captured on a side stream replays on whichever stream is current. One captured decode graph therefore cannot hold side-stream kernels.
  - The break is a host launch per layer: record the event, `wait_event`, and a side-graph replay. It is the one per-token Python step this plan adds.
  - It is sync-free. `set_sync_debug_mode("error")` tests enforce that.
  - Task 8 prices its host cost and spikes a multi-stream capture that would remove it.
- **Why the seq protocol is safe:**
  - `main_seq` (main stream, first source gather) and `side_seq` (side stream, first side graph) each advance once per graph-gather decode forward, starting equal.
  - A row counts as delivered only when its ready stamp equals this forward's `main_seq`. The stamp is written on the side stream *after* the row's copy, so a delivered row's bytes are complete.
  - Nothing rewrites a target's prefetch rows until the next forward's side copy. That copy launches only after this forward's sampled token is read back, because the overlap schedule is off, `--cuda-graph-max-bs-decode 1`, and there is no speculative decoding (validated in Task 6).
  - Every race therefore degrades to "not delivered", never to wrong bytes.
  - A non-breakable decode graph backend would never replay the break. Rows would never be delivered, which is safe but useless, so Task 6 refuses `side_stream` unless the decode backend is `breakable`.
- **Doorbell API assumed** from `shared/cc/doorbell-prototype:python/sglang/kernels/ops/moe/expert_doorbell.py`: `ExpertDoorbellCopier(segments, capacity, ...)`, `.post(source_rows int64[cap], destination_slots int32[cap], count int32[1], tag)`, `.wait(tag)`, `.quiesce()` around capture, slots unreadable before `wait`.
  - Whether `post`/`wait` are capturable or need a break is decided by `cc/doorbell-serving`; Task 10 adapts.
  - The handle is assumed to be `expert_hot_cache_manager.doorbell` (`None` when off).

### Scratch versus evictable hot slots

Prefetching into evictable hot slots is **out of scope**:
- The in-graph residency update (E31) owns the mapping, slot states and generations. A second slot writer would need a generation/lease protocol inside the graph.
- Dedicated rows cap B at a few rows per layer (≤ 10 here). Task 3 reports budget recall at 1–10 rows *and* at 16/32. If recall keeps rising well past 10, slot targeting is a follow-up plan, not this one.

### What stays outside the decode graph, and why

1. Checkpoint load, checksum and spec validation, dtype casting, buffer allocation, and side-graph capture: one-time setup.
2. Hook installation, and the Python bodies of hooks during graph capture. Replay re-executes the recorded kernels without Python, as for the existing `RouteTaps`.
3. The `rows <= max_rows` shape check in `FeatureStore.write`. It reads a Python int shape, not device data, and it never runs during replay.
4. Eager prefill forwards larger than the tap buffers. `FeatureStore.write` skips them (no `spill` when capture is off), and the eager gather path ignores `prefetch`. Prefill prefetch is out of scope.
5. Metric readback, once per `SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL` forwards, only when a metrics file is set.
6. **`side_stream` only:** one `eager_on_graph` break per source layer (47 per token). It is a host launch with no device read (see Backends).
7. **`doorbell` only:** the doorbell's C++ copy thread, and whatever host calls its serving API needs.

### Existing Python-side logic this supersedes

- **Host-side next-layer policy:** `expert_prefetch.SparseNextLayerPolicy`, `ExpertPrefetchCoordinator.launch`, and `ExpertHotCacheManager.enable_next_layer_prefetch`.
  - They use `.tolist()` on routes and require the pinned host cache.
  - `ModelRunner.maybe_init_expert_hot_cache` refuses them with graph gather (`SGLANG_MOE_PREFETCH_MAX_CANDIDATES` stays 0).
  - They are not deleted, but live prefetch on graph-gather builds goes through this plan only. `SGLANG_MOE_EXPERT_PREFETCH` and `SGLANG_MOE_PREFETCH_MAX_CANDIDATES > 0` are mutually exclusive (Task 6).
- **Post-forward shadow scoring:** `ExpertPredictionRuntime._score` calls `ExpertPredictor.predict` after the forward. That is too late to feed the same forward's copy, so the live LLaPor/APEX path does not register as an `ExpertPredictor`. Shadow predictors (`affinity`, `popularity`) keep working unchanged.

### Deviations from the research specs (recorded for the write-up)

- **No LLaPor online adaptation, no rollback, no slot leases.** Prefetch writes only dedicated rows, so it needs no slot safety.
- **Candidates come from summed batch scores** (LLaPor sigmoid, APEX softmax masked beyond `top_k + depth(tau)`), the specs' batch extension. At batch 1 this equals per-token ranking.
- **Splits are train/dev/shifted_test** (training write-up), not the specs' 70/10/10/10.
- **APEX is deferred from live** unless Task 3 says otherwise.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `python/sglang/srt/layers/moe/expert_prediction/serving/__init__.py` | package marker | 1 |
| `.../serving/checkpoints.py` | load and verify LLaPor/APEX checkpoints against `MoeLayerSpec` | 1 |
| `.../serving/scorers.py` | static-shape bf16 `LlaporScorer`, `ApexScorer` | 2 |
| `.../serving/candidates.py` | `PrefetchCandidateBank`, `BudgetRecall` device counters | 2 |
| `.../expert_prediction/training/dataset.py` | add `residency_layer` to `load_layer_rows` | 3 |
| `.../expert_prediction/prefetch_pricing.py` | budget recall, doorbell and side-stream timing models (pure torch) | 3 |
| `scripts/expert_prediction/prefetch/check_serving_parity.py` | real checkpoints: serving bf16 vs training fp32 | 3 |
| `scripts/expert_prediction/prefetch/price_prefetch.py` | offline gate report from the capture | 3 |
| `python/sglang/srt/layers/moe/expert_route_plan.py` | `plan_graph_routes(prefetch_ids=, prefetch_base=)`, `GraphRoutePlan.prefetch_hit_routes` | 4 |
| `python/sglang/srt/layers/moe/expert_hot_cache.py` | `from_model(prefetch_rows=)` | 4 |
| `python/sglang/srt/layers/moe/expert_stream.py` | `ExpertStreamer.prefetch` hook, `copy_rows_into_cache`, prefetch counters | 4 |
| `.../serving/plan.py` | `PrefetchPlan`, `PrefetchPlanner`, `LayerPrefetch` protocol | 4 |
| `.../serving/in_graph_copy.py` | `in_graph_copy` backend: `at_target` (4), `side_stream` (5) | 4, 5 |
| `.../expert_prediction/feature_store.py` | `after_write` generic hook | 6 |
| `.../expert_prediction/adapters.py` | `register_mixer_kind_adapter`, `mixer_kinds` | 6 |
| `.../serving/runtime.py` | `PrefetchScoring`: scorers, bank, planner, backend, metrics | 6 |
| `.../expert_prediction/runtime.py` | build `PrefetchScoring` from env; log its metrics | 6 |
| `python/sglang/srt/environ.py` | `SGLANG_MOE_EXPERT_PREFETCH*` | 6 |
| `python/sglang/srt/model_executor/model_runner.py` | prediction gate condition + one `prefetch_rows=` kwarg (orchestration) | 6 |
| `scripts/expert_prediction/run-shadow-server.sh` | doorbell/prefetch/budget/schedule knobs, production-current flags, lock | 7 |
| `scripts/expert_prediction/benchmarks/run_capture_sessions.py` | `--max-tokens`, `--session-ids` | 7 |
| `scripts/expert_prediction/prefetch/select_ab_sessions.py` | fixed A/B subset | 7 |
| `scripts/expert_prediction/prefetch/logprob_probe.py` | greedy top-2 logprob capture per arm | 7 |
| `scripts/expert_prediction/prefetch/compare_logprobs.py` | correctness gate between arms | 7 |
| `scripts/expert_prediction/prefetch/summarize_ab.py` | tok/s, TTFT, budget recall, delivered rows per arm | 7 |
| `scripts/expert_prediction/prefetch/bench_prefetch_schedule.py` | schedule microbench on real layer tensors | 8 |
| `.../serving/doorbell_backend.py` | `doorbell` backend adapter (blocked) | 10 |
| `test/registered/unit/layers/moe/test_expert_prefetch_checkpoints.py` | Task 1 tests (CPU) | 1 |
| `test/registered/unit/layers/moe/test_expert_prefetch_scoring.py` | Task 2 tests (CPU + CUDA) | 2 |
| `test/registered/unit/layers/moe/test_expert_prefetch_pricing.py` | Task 3 tests (CPU) | 3 |
| `test/registered/unit/layers/moe/test_expert_prefetch_plan.py` | Task 4 tests (CPU + CUDA) | 4 |
| `test/registered/unit/layers/moe/test_expert_prefetch_side_stream.py` | Task 5 tests (CUDA) | 5 |
| `test/registered/unit/layers/moe/test_expert_prefetch_runtime.py` | Task 6 tests (CUDA, small) | 6 |
| `docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md` | Task 3 report | 3 |
| `docs/superpowers/experiments/2026-09-1X-expert-prefetch-schedule-bench.md` | Task 8 report | 8 |
| `docs/superpowers/experiments/2026-09-1X-expert-prefetch-live-ab.md` | Task 9 (and 10) report | 9, 10 |

---

### Task 1: Serving checkpoint loader validated against live layer specs

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/__init__.py` (empty)
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/checkpoints.py`
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_checkpoints.py`

**Interfaces:**
- Consumes (existing): `training.llapor.build_predictor(group, *, pca_rank, num_experts)`; `training.apex.Ranker(hidden_size, num_experts)`; `training.apex.OrdinalCDF(hidden_size, num_depths)`; `training.pca.PCAStats(mean, components, explained_variance)`; `contracts.MoeLayerSpec`.
- Checkpoint layout (as written by `scripts/expert_prediction/training/train_{llapor,apex}.py`):
  - `llapor/pair-NN/`: `model.pt` (state dict), `pca.pt` (`{"mean", "components"}`), `manifest.json` (`architecture{group,pca_rank,num_experts}`, `grouping{source_layer,target_layer,group}`, `pca_stats{explained_variance}`, `tensor_checksums{model}`), `DONE`.
  - `apex/layer-NN/`: `ranker.pt`, `cdf.pt`, `manifest.json` (`layer_id`, `architecture{hidden_size,num_experts,top_k}`, `tensor_checksums{ranker}`), `DONE`.
- Produces:
  - `LlaporCheckpoint(source_layer: int, target_layer: int, group: str, pca: PCAStats, model: nn.Module)`
  - `ApexCheckpoint(layer_id: int, top_k: int, ranker: apex.Ranker, cdf: apex.OrdinalCDF)`
  - `state_dict_sha256(state_dict) -> str`
  - `load_prefetch_checkpoints(model_dir: Path, *, predictor: str, specs: Sequence[MoeLayerSpec]) -> dict[int, LlaporCheckpoint | ApexCheckpoint]`, keyed by the layer whose experts are predicted.
  - `PREFETCH_PREDICTORS = ("llapor", "apex")`

- [ ] **Step 1: Write the failing tests**

```python
"""Prefetch checkpoints load only when complete, unmodified and shaped like the live MoE layers."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.training import apex, llapor

HIDDEN, EXPERTS, TOP_K, RANK = 16, 32, 4, 8


def _specs(hidden=HIDDEN):
    return [MoeLayerSpec(layer_id=i, num_experts=EXPERTS, top_k=TOP_K, hidden_size=hidden) for i in (0, 1)]


def _write_llapor(root: Path, *, checksum_override=None, done=True) -> Path:
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import state_dict_sha256

    torch.manual_seed(0)
    directory = root / "llapor" / "pair-00"
    directory.mkdir(parents=True)
    model = llapor.build_predictor("middle", pca_rank=RANK, num_experts=EXPERTS)
    state = model.state_dict()
    torch.save(state, directory / "model.pt")
    torch.save({"mean": torch.randn(HIDDEN), "components": torch.randn(RANK, HIDDEN)}, directory / "pca.pt")
    manifest = {
        "architecture": {"group": "middle", "pca_rank": RANK, "num_experts": EXPERTS},
        "grouping": {"source_layer": 0, "target_layer": 1, "group": "middle"},
        "pca_stats": {"explained_variance": [1.0] * RANK},
        "tensor_checksums": {"model": checksum_override or state_dict_sha256(state)},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if done:
        (directory / "DONE").write_text("ok")
    return directory


def _write_apex(root: Path) -> Path:
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import state_dict_sha256

    directory = root / "apex" / "layer-01"
    directory.mkdir(parents=True)
    ranker = apex.Ranker(HIDDEN, EXPERTS)
    cdf = apex.OrdinalCDF(HIDDEN, EXPERTS - TOP_K + 1)
    torch.save(ranker.state_dict(), directory / "ranker.pt")
    torch.save(cdf.state_dict(), directory / "cdf.pt")
    manifest = {
        "layer_id": 1,
        "architecture": {"hidden_size": HIDDEN, "num_experts": EXPERTS, "top_k": TOP_K},
        "tensor_checksums": {"ranker": state_dict_sha256(ranker.state_dict())},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "DONE").write_text("ok")
    return directory


class TestPrefetchCheckpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_llapor_loads_keyed_by_target_layer(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root)
        loaded = load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())
        self.assertEqual(sorted(loaded), [1])
        self.assertEqual((loaded[1].source_layer, loaded[1].target_layer), (0, 1))
        self.assertFalse(loaded[1].model.training)

    def test_apex_loads_with_cdf(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_apex(self.root)
        loaded = load_prefetch_checkpoints(self.root, predictor="apex", specs=_specs())
        self.assertEqual(loaded[1].top_k, TOP_K)
        self.assertEqual(loaded[1].cdf.thresholds().numel(), EXPERTS - TOP_K + 1)

    def test_checksum_mismatch_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root, checksum_override="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())

    def test_incomplete_checkpoint_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root, done=False)
        with self.assertRaisesRegex(ValueError, "DONE"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())

    def test_hidden_size_mismatch_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root)
        with self.assertRaisesRegex(ValueError, "hidden"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs(hidden=HIDDEN * 2))

    def test_unknown_layer_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_apex(self.root)
        with self.assertRaisesRegex(ValueError, "lacks"):
            load_prefetch_checkpoints(self.root, predictor="apex", specs=_specs()[:1])

    def test_unknown_predictor_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        with self.assertRaisesRegex(ValueError, "unknown prefetch predictor"):
            load_prefetch_checkpoints(self.root, predictor="affinity", specs=_specs())


if __name__ == "__main__":
    unittest.main()
```

Copy the `register_*_ci(...)` line that `test/registered/unit/layers/moe/test_expert_prediction_training.py` uses, and put it after the imports, so the file registers like its CPU neighbour.

- [ ] **Step 2: Run the tests to verify they fail**

Run on divix01, CPU only: `CUDA_VISIBLE_DEVICES="" <test command> test/registered/unit/layers/moe/test_expert_prefetch_checkpoints.py`

Expected: FAIL with `ModuleNotFoundError: ...serving.checkpoints`.

- [ ] **Step 3: Write the implementation**

```python
"""Load trained LLaPor and APEX checkpoints and validate them against the live MoE layer specs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.training import apex, llapor
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats

PREFETCH_PREDICTORS = ("llapor", "apex")


class LlaporCheckpoint(msgspec.Struct, frozen=True):
    source_layer: int
    target_layer: int
    group: str
    pca: PCAStats
    model: torch.nn.Module


class ApexCheckpoint(msgspec.Struct, frozen=True):
    layer_id: int
    top_k: int
    ranker: apex.Ranker
    cdf: apex.OrdinalCDF


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """The digest the training scripts store under ``tensor_checksums``."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].cpu().numpy().tobytes())
    return digest.hexdigest()


def _manifest(directory: Path) -> dict:
    if not (directory / "DONE").exists():
        raise ValueError(f"prefetch checkpoint {directory} is incomplete: no DONE marker")
    return json.loads((directory / "manifest.json").read_text())


def _verified_state(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_sha256(state) != expected_sha256:
        raise ValueError(f"prefetch checkpoint {path} does not match its manifest checksum")
    return state


def _spec(specs: Mapping[int, MoeLayerSpec], layer_id: int, directory: Path) -> MoeLayerSpec:
    if layer_id not in specs:
        raise ValueError(f"prefetch checkpoint {directory} names MoE layer {layer_id}, which the model lacks")
    return specs[layer_id]


def _load_llapor(directory: Path, specs: Mapping[int, MoeLayerSpec]) -> LlaporCheckpoint:
    manifest = _manifest(directory)
    architecture, grouping = manifest["architecture"], manifest["grouping"]
    source = _spec(specs, grouping["source_layer"], directory)
    target = _spec(specs, grouping["target_layer"], directory)
    experts = {architecture["num_experts"], source.num_experts, target.num_experts}
    if len(experts) != 1:
        raise ValueError(f"prefetch checkpoint {directory} expert count {architecture['num_experts']} "
                         f"does not match layers {source.layer_id}/{target.layer_id}")
    pca_state = torch.load(directory / "pca.pt", map_location="cpu", weights_only=True)
    if tuple(pca_state["mean"].shape) != (source.hidden_size,):
        raise ValueError(f"prefetch checkpoint {directory} PCA hidden width {tuple(pca_state['mean'].shape)} "
                         f"does not match layer {source.layer_id} hidden size {source.hidden_size}")
    model = llapor.build_predictor(
        architecture["group"], pca_rank=architecture["pca_rank"], num_experts=target.num_experts
    )
    model.load_state_dict(_verified_state(directory / "model.pt", manifest["tensor_checksums"]["model"]))
    pca = PCAStats(
        mean=pca_state["mean"],
        components=pca_state["components"],
        explained_variance=torch.tensor(manifest["pca_stats"]["explained_variance"]),
    )
    return LlaporCheckpoint(
        source_layer=source.layer_id,
        target_layer=target.layer_id,
        group=architecture["group"],
        pca=pca,
        model=model.eval(),
    )


def _load_apex(directory: Path, specs: Mapping[int, MoeLayerSpec]) -> ApexCheckpoint:
    manifest = _manifest(directory)
    architecture = manifest["architecture"]
    spec = _spec(specs, manifest["layer_id"], directory)
    found = (architecture["hidden_size"], architecture["num_experts"], architecture["top_k"])
    expected = (spec.hidden_size, spec.num_experts, spec.top_k)
    if found != expected:
        raise ValueError(f"prefetch checkpoint {directory} (hidden, experts, top_k)={found} "
                         f"does not match layer {spec.layer_id} {expected}")
    ranker = apex.Ranker(spec.hidden_size, spec.num_experts)
    ranker.load_state_dict(_verified_state(directory / "ranker.pt", manifest["tensor_checksums"]["ranker"]))
    # The training run stores no checksum for cdf.pt.
    cdf = apex.OrdinalCDF(spec.hidden_size, spec.num_experts - spec.top_k + 1)
    cdf.load_state_dict(torch.load(directory / "cdf.pt", map_location="cpu", weights_only=True))
    return ApexCheckpoint(layer_id=spec.layer_id, top_k=spec.top_k, ranker=ranker.eval(), cdf=cdf.eval())


def load_prefetch_checkpoints(
    model_dir: Path, *, predictor: str, specs: Sequence[MoeLayerSpec]
) -> dict[int, LlaporCheckpoint | ApexCheckpoint]:
    """Checkpoints keyed by the MoE layer whose experts they predict."""
    by_layer = {spec.layer_id: spec for spec in specs}
    if predictor == "llapor":
        pairs = [_load_llapor(path, by_layer) for path in sorted((model_dir / "llapor").glob("pair-*"))]
        loaded = {pair.target_layer: pair for pair in pairs}
    elif predictor == "apex":
        layers = [_load_apex(path, by_layer) for path in sorted((model_dir / "apex").glob("layer-*"))]
        loaded = {layer.layer_id: layer for layer in layers}
    else:
        raise ValueError(f"unknown prefetch predictor {predictor!r}; expected one of {PREFETCH_PREDICTORS}")
    if not loaded:
        raise ValueError(f"no {predictor} prefetch checkpoints under {model_dir}")
    return loaded
```

In `test_unknown_predictor_raises`, the loader raises before touching the directory.

- [ ] **Step 4: Run the tests to verify they pass**

Commit, push, sync, then run the Step 2 command. Expected: `7 passed`, `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(moe): load prefetch predictor checkpoints validated against live MoE layer specs" -- \
  python/sglang/srt/layers/moe/expert_prediction/serving/__init__.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/checkpoints.py \
  test/registered/unit/layers/moe/test_expert_prefetch_checkpoints.py
```

---

### Task 2: Static-shape scorers, candidate bank, and budget-recall counters

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/scorers.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/candidates.py`
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_scoring.py`

**Interfaces:**
- Consumes (Task 1): `LlaporCheckpoint`, `ApexCheckpoint`. Consumes (existing): `training.llapor.encode_features`, `training.apex.select_depth`.
- Produces:
  - `LlaporScorer(checkpoint, *, num_experts: int, dtype, device)`, a module. `forward(router_input [rows,H], topk_ids [rows,K] int, topk_weights [rows,K] float) -> float32 [rows,E]` gives independent activation probabilities.
  - `ApexScorer(checkpoint, *, num_experts: int, tau: float, dtype, device)`, a module. `forward(pre_mixer [rows,H]) -> float32 [rows,E]` gives softmax probabilities, zeroed at ranks ≥ `top_k + depth(tau)`.
  - `PrefetchCandidateBank(*, layer_ids: Sequence[int], width: int, device)` has:
    - attributes `ids int64 [layers,width]` and `scores float32 [layers,width]`;
    - `write(target_layer: int, expert_scores: Tensor[rows,E]) -> None`;
    - `ids_for(target_layer) -> Tensor[width]` and `scores_for(target_layer) -> Tensor[width]`.
  - `BudgetRecall(*, layer_ids: Sequence[int], budget: int, device)` has:
    - `observe(*, target_layer, candidate_ids: Tensor[C], topk_ids: Tensor[rows,K], expert_to_slot: Tensor[E]) -> None`;
    - `counts int64 [layers,2]`: column 0 counts non-resident native routes, column 1 counts those covered by the first `budget` non-resident candidates;
    - `snapshot() -> dict[int, tuple[int, int]]`, a host read used only at log intervals.

- [ ] **Step 1: Write the failing tests**

```python
"""Prefetch scoring is device-only: bf16 scorers match training, bank writes and counters never sync."""

import unittest

import torch

from sglang.srt.layers.moe.expert_prediction.training import apex, llapor
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats

HIDDEN, EXPERTS, TOP_K, RANK = 16, 32, 4, 8


def _llapor_checkpoint(group="middle"):
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import LlaporCheckpoint

    torch.manual_seed(1)
    model = llapor.build_predictor(group, pca_rank=RANK, num_experts=EXPERTS).eval()
    pca = PCAStats(mean=torch.randn(HIDDEN), components=torch.randn(RANK, HIDDEN), explained_variance=torch.ones(RANK))
    return LlaporCheckpoint(source_layer=0, target_layer=1, group=group, pca=pca, model=model)


def _apex_checkpoint():
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import ApexCheckpoint

    torch.manual_seed(2)
    return ApexCheckpoint(
        layer_id=1, top_k=TOP_K, ranker=apex.Ranker(HIDDEN, EXPERTS).eval(),
        cdf=apex.OrdinalCDF(HIDDEN, EXPERTS - TOP_K + 1).eval(),
    )


def _route_features(rows=3, device="cpu"):
    generator = torch.Generator().manual_seed(3)
    router_input = torch.randn(rows, HIDDEN, generator=generator)
    topk_ids = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(rows)])
    topk_weights = torch.rand(rows, TOP_K, generator=generator)
    return router_input.to(device), topk_ids.to(device), topk_weights.to(device)


class TestPrefetchScoringCpu(unittest.TestCase):
    def test_llapor_fp32_scorer_equals_training_forward(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        for group in ("outer", "middle"):
            checkpoint = _llapor_checkpoint(group)
            router_input, topk_ids, topk_weights = _route_features()
            u = llapor.encode_features(router_input, topk_ids, topk_weights, pca=checkpoint.pca, num_experts=EXPERTS)
            expected = torch.sigmoid(checkpoint.model(u))
            scorer = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.float32, device=torch.device("cpu"))
            torch.testing.assert_close(scorer(router_input, topk_ids, topk_weights), expected, rtol=1e-5, atol=1e-6)

    def test_apex_scorer_zeroes_ranks_beyond_top_k_plus_depth(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer

        checkpoint = _apex_checkpoint()
        pre_mixer = torch.randn(2, HIDDEN)
        scorer = ApexScorer(checkpoint, num_experts=EXPERTS, tau=0.9, dtype=torch.float32, device=torch.device("cpu"))
        scores = scorer(pre_mixer)
        probabilities = torch.softmax(checkpoint.ranker(pre_mixer), dim=-1)
        depth = apex.select_depth(checkpoint.cdf(pre_mixer), 0.9, EXPERTS - TOP_K)
        for row in range(2):
            kept = (scores[row] > 0).sum().item()
            self.assertEqual(kept, min(EXPERTS, TOP_K + depth[row].item()))
            best = torch.topk(probabilities[row], kept).indices
            torch.testing.assert_close(scores[row, best], probabilities[row, best], rtol=1e-5, atol=1e-6)

    def test_bank_keeps_top_width_of_summed_rows(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank

        bank = PrefetchCandidateBank(layer_ids=[3, 7], width=2, device=torch.device("cpu"))
        scores = torch.zeros(2, EXPERTS)
        scores[0, 5], scores[1, 5], scores[0, 9], scores[1, 11] = 0.4, 0.4, 0.7, 0.6
        bank.write(7, scores)
        self.assertEqual(bank.ids_for(7).tolist(), [5, 9])
        torch.testing.assert_close(bank.scores_for(7), torch.tensor([0.8, 0.7]))
        self.assertEqual(bank.ids_for(3).tolist(), [0, 1])

    def test_budget_recall_counts_nonresident_routes_within_budget(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall

        recall = BudgetRecall(layer_ids=[1], budget=2, device=torch.device("cpu"))
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long)
        expert_to_slot[[2, 4]] = torch.tensor([0, 1])
        candidates = torch.tensor([2, 6, 4, 8, 10], dtype=torch.long)
        topk_ids = torch.tensor([[2, 8, 10, 12]])
        recall.observe(target_layer=1, candidate_ids=candidates, topk_ids=topk_ids, expert_to_slot=expert_to_slot)
        self.assertEqual(recall.snapshot(), {1: (3, 1)})


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchScoringCuda(unittest.TestCase):
    def _parts(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall, PrefetchCandidateBank
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        device = torch.device("cuda")
        scorer = LlaporScorer(_llapor_checkpoint(), num_experts=EXPERTS, dtype=torch.bfloat16, device=device)
        bank = PrefetchCandidateBank(layer_ids=[0, 1], width=6, device=device)
        recall = BudgetRecall(layer_ids=[1], budget=3, device=device)
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long, device=device)
        expert_to_slot[:10] = torch.arange(10, device=device)
        return scorer, bank, recall, expert_to_slot

    def test_decode_step_never_synchronizes(self):
        scorer, bank, recall, expert_to_slot = self._parts()
        router_input, topk_ids, topk_weights = _route_features(rows=1, device="cuda")
        router_input = router_input.to(torch.bfloat16)
        bank.write(1, scorer(router_input, topk_ids, topk_weights))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            bank.write(1, scorer(router_input, topk_ids, topk_weights))
            recall.observe(target_layer=1, candidate_ids=bank.ids_for(1), topk_ids=topk_ids, expert_to_slot=expert_to_slot)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_graph_replay_matches_eager_for_new_routes(self):
        scorer, bank, recall, expert_to_slot = self._parts()
        router_input, topk_ids, topk_weights = _route_features(rows=1, device="cuda")
        router_input = router_input.to(torch.bfloat16)

        def step():
            bank.write(1, scorer(router_input, topk_ids, topk_weights))
            recall.observe(target_layer=1, candidate_ids=bank.ids_for(1), topk_ids=topk_ids, expert_to_slot=expert_to_slot)

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for seed in range(5):
            generator = torch.Generator().manual_seed(100 + seed)
            router_input.copy_(torch.randn(1, HIDDEN, generator=generator).to("cuda", torch.bfloat16))
            topk_ids.copy_(torch.randperm(EXPERTS, generator=generator)[:TOP_K].unsqueeze(0).to("cuda"))
            topk_weights.copy_(torch.rand(1, TOP_K, generator=generator).to("cuda"))
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.topk(scorer(router_input, topk_ids, topk_weights).sum(0), bank.width)
            self.assertEqual(bank.ids_for(1).tolist(), expected.indices.tolist())

    def test_bf16_scorer_agrees_with_fp32_top_candidates(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        checkpoint = _llapor_checkpoint()
        router_input, topk_ids, topk_weights = _route_features(rows=64, device="cuda")
        full = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.float32, device=torch.device("cuda"))
        half = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.bfloat16, device=torch.device("cuda"))
        top_full = torch.topk(full(router_input, topk_ids, topk_weights), 8).indices
        top_half = torch.topk(half(router_input.to(torch.bfloat16), topk_ids, topk_weights), 8).indices
        overlap = (top_full.unsqueeze(2) == top_half.unsqueeze(1)).any(2).float().mean().item()
        self.assertGreaterEqual(overlap, 0.95)


if __name__ == "__main__":
    unittest.main()
```

Register the file with `register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")`, as `test_expert_graph_gather.py` does.

- [ ] **Step 2: Run the CPU tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" <test command> test/registered/unit/layers/moe/test_expert_prefetch_scoring.py`

Expected: FAIL with `ModuleNotFoundError: ...serving.scorers`. CUDA cases skip.

- [ ] **Step 3: Write `scorers.py`**

```python
"""Static-shape prefetch scorers; every op in ``forward`` can be recorded in a decode CUDA graph."""

from __future__ import annotations

import copy

import torch

from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import ApexCheckpoint, LlaporCheckpoint
from sglang.srt.layers.moe.expert_prediction.training import apex


class LlaporScorer(torch.nn.Module):
    """Target-layer expert activation probabilities from one source layer's routing features."""

    def __init__(self, checkpoint: LlaporCheckpoint, *, num_experts: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.num_experts = num_experts
        self.register_buffer("mean", checkpoint.pca.mean.to(device=device, dtype=dtype))
        self.register_buffer("projection", checkpoint.pca.components.T.contiguous().to(device=device, dtype=dtype))
        self.model = copy.deepcopy(checkpoint.model).to(device=device, dtype=dtype).eval()

    def forward(self, router_input: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        h = (router_input.to(self.mean.dtype) - self.mean) @ self.projection
        ids = topk_ids.long()
        mask = torch.zeros((h.shape[0], self.num_experts), dtype=h.dtype, device=h.device)
        route = torch.zeros_like(mask)
        mask.scatter_(1, ids, 1.0)
        route.scatter_(1, ids, topk_weights.to(h.dtype))
        return torch.sigmoid(self.model(torch.cat((h, mask, route), dim=-1)).float())


class ApexScorer(torch.nn.Module):
    """Same-layer softmax probabilities, kept only for ranks below ``top_k + depth(tau)``."""

    def __init__(
        self, checkpoint: ApexCheckpoint, *, num_experts: int, tau: float, dtype: torch.dtype, device: torch.device
    ):
        super().__init__()
        self.top_k = checkpoint.top_k
        self.tau = tau
        self.ranker = copy.deepcopy(checkpoint.ranker.linear).to(device=device, dtype=dtype)
        self.depth_projection = copy.deepcopy(checkpoint.cdf.w).to(device=device, dtype=dtype)
        self.register_buffer("thresholds", checkpoint.cdf.thresholds().detach().float().to(device))
        self.register_buffer("positions", torch.arange(num_experts, device=device))

    def forward(self, pre_mixer: torch.Tensor) -> torch.Tensor:
        x = pre_mixer.to(self.ranker.weight.dtype)
        probabilities = torch.softmax(self.ranker(x).float(), dim=-1)
        cdf_logits = self.thresholds.unsqueeze(0) - self.depth_projection(x).float()
        depth = apex.select_depth(cdf_logits, self.tau, self.thresholds.numel() - 1)
        order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order).scatter_(1, order, self.positions.expand_as(order))
        return probabilities * (rank < (self.top_k + depth).unsqueeze(1))
```

- [ ] **Step 4: Write `candidates.py`**

```python
"""Device buffers the prefetch planner reads, and in-graph counters of prefetch value."""

from __future__ import annotations

from typing import Sequence

import torch


class PrefetchCandidateBank:
    """Per target MoE layer, the ``width`` experts with the most expected routes this forward.

    Rows are rewritten in place so addresses survive CUDA graph capture; ids are
    initialised to distinct experts so an unwritten row is still a valid index.
    """

    def __init__(self, *, layer_ids: Sequence[int], width: int, device: torch.device) -> None:
        if width < 1:
            raise ValueError("prefetch candidate width must be positive")
        self.width = width
        self._rows = {layer_id: row for row, layer_id in enumerate(sorted(layer_ids))}
        self.ids = torch.arange(width, dtype=torch.int64, device=device).repeat(len(self._rows), 1)
        self.scores = torch.zeros((len(self._rows), width), dtype=torch.float32, device=device)

    def write(self, target_layer: int, expert_scores: torch.Tensor) -> None:
        top = torch.topk(expert_scores.sum(dim=0), self.width)
        row = self._rows[target_layer]
        self.ids[row].copy_(top.indices)
        self.scores[row].copy_(top.values)

    def ids_for(self, target_layer: int) -> torch.Tensor:
        return self.ids[self._rows[target_layer]]

    def scores_for(self, target_layer: int) -> torch.Tensor:
        return self.scores[self._rows[target_layer]]


class BudgetRecall:
    """Non-resident native routes, and those the first ``budget`` non-resident candidates cover."""

    def __init__(self, *, layer_ids: Sequence[int], budget: int, device: torch.device) -> None:
        if budget < 1:
            raise ValueError("prefetch budget must be positive")
        self.budget = budget
        self._rows = {layer_id: row for row, layer_id in enumerate(sorted(layer_ids))}
        self.counts = torch.zeros((len(self._rows), 2), dtype=torch.int64, device=device)

    def observe(
        self,
        *,
        target_layer: int,
        candidate_ids: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_to_slot: torch.Tensor,
    ) -> None:
        resident = expert_to_slot >= 0
        offered_mask = ~resident.index_select(0, candidate_ids)
        offered_mask &= torch.cumsum(offered_mask.to(torch.int64), dim=0) <= self.budget
        offered = torch.where(offered_mask, candidate_ids, torch.full_like(candidate_ids, -1))
        native = topk_ids.reshape(-1).long()
        missed = ~resident.index_select(0, native)
        covered = (native.unsqueeze(1) == offered.unsqueeze(0)).any(dim=1)
        row = self.counts[self._rows[target_layer]]
        row[0].add_(missed.sum())
        row[1].add_((missed & covered).sum())

    def snapshot(self) -> dict[int, tuple[int, int]]:
        values = self.counts.cpu().tolist()
        return {layer_id: tuple(values[row]) for layer_id, row in self._rows.items()}
```

- [ ] **Step 5: Run the tests**

Run CPU first with the Step 2 command. Expected: 4 passed, CUDA cases skipped. Then, **only when the GPU is not owned by the doorbell session and 7867 is up and healthy** (these tests use < 100 MiB), take the lock and run: `flock -n /data/models/slang/nvfp4-work/cc-gpu.lock <test command> test/registered/unit/layers/moe/test_expert_prefetch_scoring.py`. Expected: `7 passed`, `EXIT=0`. If the GPU is unavailable, record "CUDA cases pending" in the commit body and run them at the start of Task 4.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(moe): add graph-capturable prefetch scorers, candidate bank and budget recall" -- \
  python/sglang/srt/layers/moe/expert_prediction/serving/scorers.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/candidates.py \
  test/registered/unit/layers/moe/test_expert_prefetch_scoring.py
```

---

### Task 3: Offline gate — budget recall and in-graph/doorbell pricing from the full capture (CPU only)

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_prediction/training/dataset.py` (`LayerRows`, `load_layer_rows`)
- Create: `python/sglang/srt/layers/moe/expert_prediction/prefetch_pricing.py`
- Create: `scripts/expert_prediction/prefetch/check_serving_parity.py`
- Create: `scripts/expert_prediction/prefetch/price_prefetch.py`
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_pricing.py`
- Report: `docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md`

**Interfaces:**
- Consumes: Task 1 `load_prefetch_checkpoints`; Task 2 `LlaporScorer`, `ApexScorer`; existing `dataset.load_session_splits`, `dataset.split_mask`, `capture_schema.FORWARD_RESIDENCY` (int16 `[forwards, layers, experts]`, -1 = non-resident, snapshotted at forward end, which is after that forward's in-graph residency update).
- Produces:
  - `load_layer_rows(..., residency_layer: int | None = None)`, whose `LayerRows.resident` is bool `[rows, experts]` or None;
  - `prefetch_pricing.budget_hits(scores, topk_ids, resident, budget) -> (missed [rows], hits [rows])`;
  - `prefetch_pricing.doorbell_saving_ms(*, hits, budget, window_ms, reaction_ms) -> Tensor[rows] float64`;
  - `prefetch_pricing.side_stream_ready_rows(*, budget, window_ms) -> int`: how many of the plan's rows the side-stream copy lands before the target gather reads readiness (Task 5 copies rows one launch at a time, best score first);
  - the constants `IN_GRAPH_ROW_MS`, `IN_GRAPH_FIXED_MS`, `DOORBELL_ROW_MS`, `DOORBELL_FIXED_MS`, `DOORBELL_COMPLETED_WAIT_MS`.
- Side-stream saving is `budget_hits(..., side_stream_ready_rows(...)).hits × IN_GRAPH_ROW_MS`. Rows that land late are simply not delivered, so they cost no wait. Whether their copy slows the target's residual copy is unmeasured (Task 8).

- [ ] **Step 1: Write the failing pricing tests**

```python
"""Prefetch value model: misses counted against residency, hits bounded by the budget, late requests charged."""

import unittest

import torch


class TestPrefetchPricing(unittest.TestCase):
    def test_budget_hits_skip_residents_and_respect_budget(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import budget_hits

        resident = torch.zeros(1, 8, dtype=torch.bool)
        resident[0, [1, 2]] = True
        scores = torch.tensor([[0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0]])
        topk_ids = torch.tensor([[1, 3, 5, 6]])
        missed, hits = budget_hits(scores, topk_ids, resident, budget=2)
        self.assertEqual((missed.item(), hits.item()), (3, 1))
        missed, hits = budget_hits(scores, topk_ids, resident, budget=3)
        self.assertEqual((missed.item(), hits.item()), (3, 2))

    def test_saving_charges_the_wait_past_the_window(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
            DOORBELL_COMPLETED_WAIT_MS, DOORBELL_FIXED_MS, DOORBELL_ROW_MS, IN_GRAPH_ROW_MS, doorbell_saving_ms,
        )

        hits = torch.tensor([0, 1, 2])
        on_time = doorbell_saving_ms(hits=hits, budget=1, window_ms=1.0, reaction_ms=0.03)
        torch.testing.assert_close(on_time, hits.double() * IN_GRAPH_ROW_MS - DOORBELL_COMPLETED_WAIT_MS)
        late = doorbell_saving_ms(hits=hits, budget=2, window_ms=0.2, reaction_ms=0.05)
        wait = 0.05 + 2 * DOORBELL_ROW_MS + DOORBELL_FIXED_MS - 0.2
        torch.testing.assert_close(late, hits.double() * IN_GRAPH_ROW_MS - wait - DOORBELL_COMPLETED_WAIT_MS)

    def test_side_stream_lands_whole_rows_inside_the_window(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
            IN_GRAPH_FIXED_MS, IN_GRAPH_ROW_MS, side_stream_ready_rows,
        )

        row = IN_GRAPH_ROW_MS + IN_GRAPH_FIXED_MS
        self.assertEqual(side_stream_ready_rows(budget=10, window_ms=0.267), 1)
        self.assertEqual(side_stream_ready_rows(budget=10, window_ms=3 * row + 1e-9), 3)
        self.assertEqual(side_stream_ready_rows(budget=2, window_ms=3 * row), 2)
        self.assertEqual(side_stream_ready_rows(budget=4, window_ms=0.1), 0)


if __name__ == "__main__":
    unittest.main()
```

Add the CPU registration line copied from `test_expert_prediction_training.py`.

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" <test command> test/registered/unit/layers/moe/test_expert_prefetch_pricing.py`. Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `prefetch_pricing.py`**

```python
"""Offline value of expert prefetch: budget recall of cache misses and in-graph/doorbell timing models."""

from __future__ import annotations

import math

import torch

# E28 fit of the in-graph miss copy kernel (11.5 GiB/s): 0.2239 ms/row + 0.006 ms per launch.
IN_GRAPH_ROW_MS = 0.2239
IN_GRAPH_FIXED_MS = 0.006
# E32: doorbell thread on a torch-created stream, 3 rows in 0.628 ms (12.4 GiB/s).
DOORBELL_ROW_MS = 0.209
DOORBELL_FIXED_MS = 0.007
# E29: a wait on an already-landed request costs 15-22 us.
DOORBELL_COMPLETED_WAIT_MS = 0.02


def budget_hits(
    scores: torch.Tensor, topk_ids: torch.Tensor, resident: torch.Tensor, budget: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: native experts missing from the hot cache, and those the top-``budget`` non-resident scores cover."""
    offered = scores.masked_fill(resident, float("-inf")).topk(budget, dim=1).indices
    native = topk_ids.long()
    missed = ~resident.gather(1, native)
    covered = (native.unsqueeze(2) == offered.unsqueeze(1)).any(dim=2)
    return missed.sum(dim=1), (missed & covered).sum(dim=1)


def doorbell_saving_ms(*, hits: torch.Tensor, budget: int, window_ms: float, reaction_ms: float) -> torch.Tensor:
    """In-graph copy time the hits avoid, minus the wait for a ``budget``-row request that lands after the window."""
    landing_ms = reaction_ms + budget * DOORBELL_ROW_MS + DOORBELL_FIXED_MS
    return hits.to(torch.float64) * IN_GRAPH_ROW_MS - max(0.0, landing_ms - window_ms) - DOORBELL_COMPLETED_WAIT_MS


def side_stream_ready_rows(*, budget: int, window_ms: float) -> int:
    """Rows a one-launch-per-row side-stream copy lands within ``window_ms``; later rows are not delivered."""
    return max(0, min(budget, math.floor(window_ms / (IN_GRAPH_ROW_MS + IN_GRAPH_FIXED_MS))))
```

- [ ] **Step 4: Add residency to the capture loader**

In `training/dataset.py`, import `FORWARD_RESIDENCY` next to `FORWARD_KIND`. Add `resident: torch.Tensor | None = None` as the last field of `LayerRows`. Add a keyword `residency_layer: int | None = None` to `load_layer_rows`. Inside the per-shard loop, right after `local_forward` is computed and `is_decode_parts` is appended, add:

```python
            if residency_layer is not None:
                table = handle.get_slice(FORWARD_RESIDENCY)[:, residency_layer : residency_layer + 1, :]
                resident_parts.append(table[local_forward.long(), 0, :] >= 0)
```

Declare `resident_parts: list[torch.Tensor] = []` next to `rid_parts`, and pass it into the returned struct:

```python
        resident=torch.cat(resident_parts) if residency_layer is not None and resident_parts else None,
```

Add this test to `test_expert_prefetch_pricing.py`. It writes a two-forward shard with `safetensors.torch.save_file` and a manifest through `capture_reader`'s format, and asserts `rows.resident` matches the per-forward table. Copy the manifest/shard writing helper from `test/registered/unit/layers/moe/test_expert_prediction_training.py`'s dataset join test, and add the `forward.expert_to_slot` tensor:

```python
    def test_load_layer_rows_joins_residency_by_forward(self):
        # Shard: forwards 0 (decode) and 1 (decode), one row each, layer 0 residency differs per forward.
        residency = torch.full((2, 1, 8), -1, dtype=torch.int16)
        residency[0, 0, 3] = 0
        residency[1, 0, 5] = 0
        rows = _load_single_layer_capture(self.tmp, residency=residency)  # helper built from the training test's writer
        self.assertEqual(rows.resident[0].nonzero().flatten().tolist(), [3])
        self.assertEqual(rows.resident[1].nonzero().flatten().tolist(), [5])
```

`_load_single_layer_capture` calls `load_layer_rows(capture_dir, splits, layer_id=0, features=(RouteFeature.TOPK_IDS,), residency_layer=0)` over the shard built by the copied helper. Run the Step 2 command. Expected: `4 passed`.

- [ ] **Step 5: Real-checkpoint parity script** (`scripts/expert_prediction/prefetch/check_serving_parity.py`)

```python
"""Serving bf16 scorers vs training fp32 forwards on real checkpoints and dev decode rows (CPU)."""

import argparse
import json
from pathlib import Path

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer
from sglang.srt.layers.moe.expert_prediction.training import llapor
from sglang.srt.layers.moe.expert_prediction.training.dataset import load_layer_rows, load_session_splits, split_mask
from sglang.srt.layers.moe.expert_prediction.training.metrics import recall_at_budget

ROWS = 4096


def _specs(capture_dir: Path) -> list[MoeLayerSpec]:
    header = json.loads((capture_dir / "capture.json").read_text())
    return [msgspec.convert(layer, MoeLayerSpec) for layer in header["layers"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layers", default="5,20,44")
    args = parser.parse_args()
    specs = _specs(args.capture_dir)
    splits = load_session_splits(args.sessions)
    llapor_models = load_prefetch_checkpoints(args.model_dir, predictor="llapor", specs=specs)
    apex_models = load_prefetch_checkpoints(args.model_dir, predictor="apex", specs=specs)
    experts = specs[0].num_experts
    report = {}
    cpu = torch.device("cpu")
    for target in (int(layer) for layer in args.layers.split(",")):
        pair = llapor_models[target]
        rows = load_layer_rows(
            args.capture_dir, splits, layer_id=pair.source_layer,
            features=(RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS, RouteFeature.PRE_MIXER),
            next_layer_topk=target,
        )
        keep = (split_mask(rows, splits, "dev") & rows.is_decode).nonzero().flatten()[:ROWS]
        router_input = rows.features["router_input"][keep]
        topk_ids, topk_weights = rows.features["topk_ids"][keep], rows.features["topk_weights"][keep]
        labels = rows.features["next_topk_ids"][keep]
        with torch.no_grad():
            u = llapor.encode_features(router_input, topk_ids, topk_weights, pca=pair.pca, num_experts=experts)
            full = torch.sigmoid(pair.model(u))
            half = LlaporScorer(pair, num_experts=experts, dtype=torch.bfloat16, device=cpu)(
                router_input.to(torch.bfloat16), topk_ids, topk_weights
            )
        report[f"llapor_{target}"] = {
            "recall16_fp32": recall_at_budget(torch.topk(full, 16).indices, labels),
            "recall16_bf16": recall_at_budget(torch.topk(half, 16).indices, labels),
        }
        same = load_layer_rows(args.capture_dir, splits, layer_id=target,
                               features=(RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS))
        keep = (split_mask(same, splits, "dev") & same.is_decode).nonzero().flatten()[:ROWS]
        layer = apex_models[target]
        with torch.no_grad():
            full = torch.softmax(layer.ranker(same.features["pre_mixer"][keep]), dim=-1)
            half = ApexScorer(layer, num_experts=experts, tau=1.0, dtype=torch.bfloat16, device=cpu)(
                same.features["pre_mixer"][keep].to(torch.bfloat16)
            )
        labels = same.features["topk_ids"][keep]
        report[f"apex_{target}"] = {
            "recall16_fp32": recall_at_budget(torch.topk(full, 16).indices, labels),
            "recall16_bf16": recall_at_budget(torch.topk(half, 16).indices, labels),
        }
    print(json.dumps(report, indent=2))
    worst = max(abs(v["recall16_fp32"] - v["recall16_bf16"]) for v in report.values())
    raise SystemExit(0 if worst <= 0.005 else 1)


if __name__ == "__main__":
    main()
```

`tau=1.0` makes the APEX depth the full `E-K` endpoint, so the mask keeps every expert and the comparison isolates bf16.

Run on divix01: `CUDA_VISIBLE_DEVICES="" /data/models/slang/.venv/bin/python scripts/expert_prediction/prefetch/check_serving_parity.py --capture-dir /mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438 --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl --model-dir /mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630`. Expected: EXIT=0, bf16 within 0.005 recall@16 of fp32 at every layer.

- [ ] **Step 6: Pricing script** (`scripts/expert_prediction/prefetch/price_prefetch.py`)

```python
"""Offline prefetch gate: recall of non-resident native experts within a budget, and doorbell ms/token (CPU)."""

import argparse
import json
from pathlib import Path

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
    IN_GRAPH_FIXED_MS, IN_GRAPH_ROW_MS, budget_hits, doorbell_saving_ms, side_stream_ready_rows,
)
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer
from sglang.srt.layers.moe.expert_prediction.training.dataset import load_layer_rows, load_session_splits, split_mask

# E28 p50 windows (copy(L) end -> copy(L+1) start) by the target layer's mixer kind.
WINDOW_MS = {"full_attention": 0.290, "linear_attention": 0.267, "unknown": 0.267}
# APEX must land before its own layer's copy: drop the previous layer's MoE kernels (~0.05 ms) and 10 us of slack.
APEX_WINDOW_REDUCTION_MS = 0.06
BATCH = 4096


def _mixer_kinds(model_config: Path, layer_ids) -> dict[int, str]:
    config = json.loads(model_config.read_text())
    config = config.get("text_config", config)
    layer_types = config.get("layer_types") or []
    return {layer: layer_types[layer] if layer < len(layer_types) else "unknown" for layer in layer_ids}


def _scores(predictor, checkpoint, rows, experts):
    cpu = torch.device("cpu")
    if predictor == "llapor":
        scorer = LlaporScorer(checkpoint, num_experts=experts, dtype=torch.float32, device=cpu)
        inputs = (rows.features["router_input"], rows.features["topk_ids"], rows.features["topk_weights"])
    else:
        scorer = ApexScorer(checkpoint, num_experts=experts, tau=1.0, dtype=torch.float32, device=cpu)
        inputs = (rows.features["pre_mixer"],)
    with torch.no_grad():
        return torch.cat([scorer(*(x[i : i + BATCH] for x in inputs)) for i in range(0, inputs[0].shape[0], BATCH)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--predictor", choices=("llapor", "apex"), required=True)
    parser.add_argument("--budgets", default="1,2,3,4,6,8,10,16,32")
    parser.add_argument("--reaction-ms", default="0.03,0.075")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    header = json.loads((args.capture_dir / "capture.json").read_text())
    specs = [msgspec.convert(layer, MoeLayerSpec) for layer in header["layers"]]
    splits = load_session_splits(args.sessions)
    models = load_prefetch_checkpoints(args.model_dir, predictor=args.predictor, specs=specs)
    kinds = _mixer_kinds(args.model_config, [spec.layer_id for spec in specs])
    budgets = [int(b) for b in args.budgets.split(",")]
    reactions = [float(r) for r in args.reaction_ms.split(",")]
    experts = specs[0].num_experts
    per_layer = {}
    for target, checkpoint in sorted(models.items()):
        if args.predictor == "llapor":
            rows = load_layer_rows(
                args.capture_dir, splits, layer_id=checkpoint.source_layer,
                features=(RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS),
                next_layer_topk=target, residency_layer=target,
            )
            native = rows.features["next_topk_ids"]
        else:
            rows = load_layer_rows(args.capture_dir, splits, layer_id=target,
                                   features=(RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS), residency_layer=target)
            native = rows.features["topk_ids"]
        scores = _scores(args.predictor, checkpoint, rows, experts)
        window = WINDOW_MS[kinds[target]] - (APEX_WINDOW_REDUCTION_MS if args.predictor == "apex" else 0.0)
        per_layer[target] = {"mixer": kinds[target], "window_ms": window}
        for split in ("dev", "shifted_test"):
            keep = (split_mask(rows, splits, split) & rows.is_decode).nonzero().flatten()
            for budget in budgets:
                missed, hits = budget_hits(scores[keep], native[keep], rows.resident[keep], budget)
                entry = {"rows": int(keep.numel()), "missed_per_token": float(missed.double().mean()),
                         "hits_per_token": float(hits.double().mean()),
                         "budget_recall": float(hits.sum() / missed.sum().clamp(min=1))}
                for reaction in reactions:
                    entry[f"saving_ms_r{reaction}"] = float(
                        doorbell_saving_ms(hits=hits, budget=budget, window_ms=window, reaction_ms=reaction).mean()
                    )
                if args.predictor == "llapor":
                    # "gap": the side copy starts after the source layer's own miss copy (E28 window only).
                    # "overlap": it starts before that copy, so the window also spans the source layer's
                    # baseline copy time. Valid only if concurrent copies do not slow each other (Task 8).
                    source_entry = per_layer.get(checkpoint.source_layer, {}).get(f"{split}_b{budgets[0]}", {})
                    source_copy_ms = IN_GRAPH_FIXED_MS + source_entry.get("missed_per_token", 0.0) * IN_GRAPH_ROW_MS
                    for label, side_window in (("gap", window), ("overlap", window + source_copy_ms)):
                        ready = side_stream_ready_rows(budget=budget, window_ms=side_window)
                        side_hits = (budget_hits(scores[keep], native[keep], rows.resident[keep], ready)[1]
                                     if ready else torch.zeros_like(hits))
                        entry[f"side_ready_rows_{label}"] = ready
                        entry[f"side_saving_ms_{label}"] = float(side_hits.double().mean() * IN_GRAPH_ROW_MS)
                per_layer[target][f"{split}_b{budget}"] = entry
        print(json.dumps({"layer": target, **per_layer[target]}), flush=True)
    totals = {}
    for split in ("dev", "shifted_test"):
        for budget in budgets:
            key = f"{split}_b{budget}"
            totals[key] = {
                "budget_recall": sum(v[key]["hits_per_token"] for v in per_layer.values())
                / max(sum(v[key]["missed_per_token"] for v in per_layer.values()), 1e-9),
                **{f"saving_ms_per_token_r{r}": sum(v[key][f"saving_ms_r{r}"] for v in per_layer.values())
                   for r in reactions},
            }
            if args.predictor == "llapor":
                for label in ("gap", "overlap"):
                    totals[key][f"side_saving_ms_per_token_{label}"] = sum(
                        v[key][f"side_saving_ms_{label}"] for v in per_layer.values()
                    )
    args.out.write_text(json.dumps({"predictor": args.predictor, "totals": totals, "layers": per_layer}, indent=2))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the gate on divix01 (CPU only; ~48 layer loads per predictor)**

```bash
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && mkdir -p /mnt/nvme2/nvfp4-work/prefetch-gate && \
  for p in llapor apex; do CUDA_VISIBLE_DEVICES="" nohup /data/models/slang/.venv/bin/python -u scripts/expert_prediction/prefetch/price_prefetch.py \
    --capture-dir /mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438 \
    --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl \
    --model-dir /mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630 \
    --model-config /mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47/config.json \
    --predictor $p --out /mnt/nvme2/nvfp4-work/prefetch-gate/$p.json > /mnt/nvme2/nvfp4-work/prefetch-gate/$p.log 2>&1; done &'
```

Run the two predictors sequentially: each holds ≤ ~10 GB RAM per layer, and divix01 has ~147 GB free. Poll the logs every ~20 min.

- [ ] **Step 8: Decide and report**

Write `docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md` with:
- **Totals:** a table of budget recall and ms/token by budget, split, and reaction.
- **Per-mixer-kind means.**
- **Parity output** from Step 5.
- **A decision under these rules** (all on `shifted_test`, budget ≤ 10):
  - **In-graph copy GO:** `side_saving_ms_per_token_gap` ≥ 3.0 for some budget. The live budget B is its argmax.
  - **Conditional GO:** only `side_saving_ms_per_token_overlap` reaches 3.0. The live test then waits for Task 8 to show that concurrent copies do not slow each other, and B is the overlap argmax.
  - **Doorbell GO:** `saving_ms_per_token_r0.075` ≥ 3.0. It is recorded for Task 10 and does not block Tasks 4–9.
  - **NO-GO:** none of the above. Report to the user before Task 4. Tasks 4–7 may still proceed as plumbing if the user says so, but Task 9 does not run.
  - **APEX live:** needs the doorbell rule *and* a doorbell same-layer post/resolve. Otherwise it is deferred, and the doc states the shortfall. The in-graph side stream cannot serve APEX, because APEX's scores and its own gather sit in the same layer.
  - **Evictable-slot follow-up:** flag it if budget recall at 32 exceeds budget 10 by more than 0.15.

Send the decision to the user; do not start Task 4 on NO-GO without their answer.

- [ ] **Step 9: Commit**

```bash
git commit -m "feat(moe): price live expert prefetch offline from the capture's residency" -- \
  python/sglang/srt/layers/moe/expert_prediction/training/dataset.py \
  python/sglang/srt/layers/moe/expert_prediction/prefetch_pricing.py \
  scripts/expert_prediction/prefetch/check_serving_parity.py \
  scripts/expert_prediction/prefetch/price_prefetch.py \
  test/registered/unit/layers/moe/test_expert_prefetch_pricing.py \
  docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md
```

---

### Task 4: Prefetch plan, in-graph planner, dedicated prefetch rows and gather integration (`in_graph_copy` / `at_target`)

MVP backbone. It needs nothing from the doorbell, and it runs with `SGLANG_MOE_EXPERT_DOORBELL` unset.

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/plan.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/in_graph_copy.py`
- Modify: `python/sglang/srt/layers/moe/expert_route_plan.py` (`GraphRoutePlan`, `plan_graph_routes`)
- Modify: `python/sglang/srt/layers/moe/expert_stream.py` (`ExpertStreamer.__init__`, `_gather_graph`, new `install_prefetch` and `copy_rows_into_cache`)
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py` (`ExpertHotCacheManager.from_model`)
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_plan.py`

**Interfaces:**
- Consumes: Task 2 `PrefetchCandidateBank`; the existing `ExpertStreamer`, `ExpertHotCache`, `plan_graph_routes`, and `copy_expert_row_segments_gpu`.
- Produces:
  - `PrefetchPlan` (see Design);
  - `PrefetchPlanner(*, bank, expert_to_slot: Mapping[int, Tensor], prefetch_base: Mapping[int, int], budget, device)` with `.plans`, `.plan(target_layer)`, `.valid_rows(plan)`, `.delivered_ids(plan)`;
  - the `LayerPrefetch` protocol: `base: int`, `resolve() -> Tensor | None`, `launch_next() -> None`;
  - `InGraphCopyBackend.install(*, planner, streamers, next_target: Mapping[int, int], schedule, device)`. Task 4 implements `schedule="at_target"` only and raises `NotImplementedError` for `side_stream`, which Task 5 replaces;
  - `ExpertStreamer.install_prefetch(hook)`, `ExpertStreamer.prefetch_base` (= `hot_cache.capacity + graph_gather_rows`), `ExpertStreamer.copy_rows_into_cache(source_rows, destination_rows, destination_slots, count)`, and `ExpertStreamer.graph_prefetch_counters` int64 `[2]` (routes served from prefetch rows, delivered rows);
  - `plan_graph_routes(flat, expert_to_slot, scratch_rows, scratch_base, prefetch_ids=None, prefetch_base=0)`, whose `GraphRoutePlan.prefetch_hit_routes` is `None` when `prefetch_ids is None`;
  - `ExpertHotCacheManager.from_model(..., prefetch_rows: int = 0)`.

- [ ] **Step 1: Write the failing tests**

```python
"""Prefetch plans only move bytes into dedicated rows: routes stay byte-exact whatever the plan says."""

import unittest

import torch

from sglang.srt.layers.moe.expert_route_plan import plan_graph_routes

EXPERTS, TOP_K = 8, 4


class TestPrefetchRoutePlanCpu(unittest.TestCase):
    def test_delivered_experts_leave_the_residual_miss_copy(self):
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long)
        expert_to_slot[1] = 0
        flat = torch.tensor([1, 5, 3, 5])
        plan = plan_graph_routes(flat, expert_to_slot, 4, 10, prefetch_ids=torch.tensor([7, 5, -1]), prefetch_base=14)
        self.assertEqual(plan.remap.tolist(), [0, 15, 10, 15])
        self.assertEqual((int(plan.miss_plan_rows), int(plan.routed_miss_rows), int(plan.prefetch_hit_routes)), (1, 1, 2))
        self.assertEqual(plan.source_rows[:1].tolist(), [3])

    def test_resident_expert_wins_over_a_stale_plan_row(self):
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long)
        expert_to_slot[5] = 2
        plan = plan_graph_routes(torch.tensor([5, 6]), expert_to_slot, 2, 10, prefetch_ids=torch.tensor([5]), prefetch_base=12)
        self.assertEqual(plan.remap.tolist(), [2, 10])
        self.assertEqual(int(plan.prefetch_hit_routes), 0)

    def test_no_prefetch_keeps_the_existing_plan(self):
        expert_to_slot = torch.tensor([0, -1, -1, 1, -1, -1, -1, -1])
        flat = torch.tensor([2, 0, 2, 7])
        plan = plan_graph_routes(flat, expert_to_slot, 4, 2)
        self.assertIsNone(plan.prefetch_hit_routes)
        self.assertEqual(plan.remap.tolist(), [2, 0, 2, 3])

    def test_planner_takes_best_nonresident_candidates_and_counts_them(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank
        from sglang.srt.layers.moe.expert_prediction.serving.plan import PrefetchPlanner

        bank = PrefetchCandidateBank(layer_ids=[1], width=4, device=torch.device("cpu"))
        bank.ids[0] = torch.tensor([3, 4, 6, 2])
        bank.scores[0] = torch.tensor([0.9, 0.8, 0.7, 0.1])
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long)
        expert_to_slot[[3, 6]] = torch.tensor([0, 1])
        planner = PrefetchPlanner(bank=bank, expert_to_slot={1: expert_to_slot}, prefetch_base={1: 20},
                                  budget=3, device=torch.device("cpu"))
        plan = planner.plan(1)
        self.assertEqual(plan.expert_ids[:2].tolist(), [4, 2])
        self.assertEqual(plan.count.tolist(), [2])
        self.assertEqual(plan.destination_slots.tolist(), [20, 21, 22])
        plan.delivered.copy_(planner.valid_rows(plan))
        self.assertEqual(planner.delivered_ids(plan).tolist(), [4, 2, -1])
```

For the CUDA class, import `_layer`, `_source_bytes`, `EXPERTS`, `TOP_K` and `NVFP4_STREAM_TENSORS` exactly as `test_expert_graph_gather.py` does. Copy its `_graph_streamer` and `_assert_rows` helpers, building the cache with `scratch_rows=TOP_K + BUDGET` (`BUDGET = 3`). Then add:

```python
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestAtTargetPrefetchCuda(unittest.TestCase):
    def _install(self, layer, bank_ids, bank_scores, resident=(1, 4, 6)):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank
        from sglang.srt.layers.moe.expert_prediction.serving.in_graph_copy import InGraphCopyBackend
        from sglang.srt.layers.moe.expert_prediction.serving.plan import PrefetchPlanner

        streamer, cache = self._graph_streamer(layer, resident)
        device = torch.device("cuda")
        bank = PrefetchCandidateBank(layer_ids=[0], width=len(bank_ids), device=device)
        bank.ids[0].copy_(torch.tensor(bank_ids))
        bank.scores[0].copy_(torch.tensor(bank_scores))
        planner = PrefetchPlanner(bank=bank, expert_to_slot={0: cache.expert_to_slot},
                                  prefetch_base={0: streamer.prefetch_base}, budget=BUDGET, device=device)
        InGraphCopyBackend.install(planner=planner, streamers={0: streamer}, next_target={},
                                   schedule="at_target", device=device)
        return streamer, cache, bank

    def test_predicted_misses_are_served_from_prefetch_rows_byte_exact(self):
        layer = _layer()
        streamer, cache, _ = self._install(layer, [2, 7, 0, 5], [0.9, 0.8, 0.7, 0.6])
        ids = torch.tensor([[2, 7, 4, 3]], dtype=torch.int32, device="cuda")
        compact, tensors = streamer.gather(ids)
        self._assert_rows(layer, ids, compact, tensors)
        base = streamer.prefetch_base
        self.assertEqual(compact.tolist(), [[base, base + 1, cache.expert_to_slot[4].item(), cache.capacity]])
        self.assertEqual(streamer.graph_prefetch_counters.tolist(), [2, 3])
        self.assertEqual(streamer.graph_counters.tolist(), [4, 1])

    def test_garbage_plan_is_byte_exact_under_replay(self):
        layer = _layer()
        streamer, cache, bank = self._install(layer, [0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0])
        ids = torch.tensor([[1, 4, 6, 1]], dtype=torch.int32, device="cuda")
        outputs = {name: torch.empty((TOP_K,) + tuple(t.shape[1:]), dtype=t.dtype, device="cuda")
                   for name, t in cache.tensors.items()}

        def step():
            compact, tensors = streamer.gather(ids)
            for name, output in outputs.items():
                output.copy_(tensors[name][compact.reshape(-1).long()])

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        generator = torch.Generator().manual_seed(7)
        for _ in range(12):
            ids.copy_(torch.randint(0, EXPERTS, (1, TOP_K), generator=generator).to("cuda", torch.int32))
            bank.ids[0].copy_(torch.randperm(EXPERTS, generator=generator)[:4].cuda())
            bank.scores[0].copy_(torch.rand(4, generator=generator).cuda())
            graph.replay()
            torch.cuda.synchronize()
            for name, output in outputs.items():
                self.assertTrue(torch.equal(output.view(torch.uint8).cpu(), _source_bytes(layer, name, ids.reshape(-1))))

    def test_prefetch_gather_never_synchronizes(self):
        streamer, _, _ = self._install(_layer(), [2, 7, 0, 5], [0.9, 0.8, 0.7, 0.6])
        ids = torch.tensor([[2, 7, 4, 3]], dtype=torch.int32, device="cuda")
        streamer.gather(ids)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            streamer.gather(ids)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_manager_adds_prefetch_rows_after_graph_scratch_and_none_when_off(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

        def build(prefetch_rows):
            layer = _layer()
            layer.layer_id = 0
            layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
            model = torch.nn.Module()
            model.add_module("0", layer)
            streamer = layer._nvfp4_expert_streamer
            manager = ExpertHotCacheManager.from_model(
                model, budget_bytes=streamer.bytes_per_expert * (TOP_K + BUDGET + 2), seed_path=None, dynamic=True,
                update_prefill_tokens=16, min_residence_forwards=0, benefit_ratio=1.0, graph_gather_batch_size=1,
                prefetch_rows=prefetch_rows,
            )
            return manager.caches[0], streamer

        cache, streamer = build(BUDGET)
        self.assertEqual((cache.capacity, cache.scratch_rows, streamer.graph_gather_rows), (2, TOP_K + BUDGET, TOP_K))
        self.assertEqual(streamer.prefetch_base, 2 + TOP_K)
        cache, streamer = build(0)
        self.assertEqual((cache.capacity, cache.scratch_rows, streamer.prefetch), (2 + BUDGET, TOP_K, None))
```

In the first test:
- Residents are 1, 4 and 6. The candidates are [2, 7, 0, 5], all non-resident, and B = 3 plans [2, 7, 0].
- Routes 2 and 7 hit prefetch rows. Route 4 is resident. Route 3 is the only residual miss, so it lands in the first graph scratch row.
- The counters therefore read 2 prefetch-hit routes out of 3 delivered rows, and graph misses [4 routes, 1 routed miss].

The last test guards the off path: with `prefetch_rows=0` the slot/scratch split is today's. Register the file with `register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")`.

- [ ] **Step 2: Run to verify failure**

CPU first: `CUDA_VISIBLE_DEVICES="" <test command> test/registered/unit/layers/moe/test_expert_prefetch_plan.py`. Expected: FAIL (`TypeError: plan_graph_routes() got an unexpected keyword argument 'prefetch_ids'`).

- [ ] **Step 3: `plan_graph_routes` and `GraphRoutePlan`**

Add `prefetch_hit_routes: torch.Tensor | None = None` as the last `GraphRoutePlan` field. Leave the dataclass as it is; it is existing code. Change the signature to `..., scratch_base: int, prefetch_ids: torch.Tensor | None = None, prefetch_base: int = 0`. Directly after `hit = slots >= 0`, add:

```python
    prefetch_hit_routes = None
    if prefetch_ids is not None:
        # prefetch_ids holds -1 for undelivered rows, which never equals an expert id.
        match = flat.unsqueeze(1) == prefetch_ids.unsqueeze(0)
        prefetched = match.any(dim=1) & ~hit
        slots = torch.where(prefetched, match.to(torch.uint8).argmax(dim=1) + prefetch_base, slots)
        prefetch_hit_routes = prefetched.sum()
        hit = hit | prefetched
```

Pass `prefetch_hit_routes=prefetch_hit_routes` into the returned plan. Everything below already treats `hit` routes as served. With `prefetch_ids is None` no new kernel is recorded, so the production graph is unchanged. Extend the docstring by one sentence on delivered rows.

- [ ] **Step 4: `ExpertStreamer` hook and row copy**

- **`__init__`:** next to `self.graph_gather_rows = 0` (`expert_stream.py:519`), add `self.prefetch = None` and `self.graph_prefetch_counters = None`.
- **New methods**, after `enable_graph_gather`:

```python
    @property
    def prefetch_base(self) -> int:
        """First hot-cache row after the graph-gather scratch rows."""
        return self.hot_cache.capacity + self.graph_gather_rows

    def install_prefetch(self, hook) -> None:
        """Serve delivered prefetch rows from the graph gather; ``hook`` follows ``serving.plan.LayerPrefetch``."""
        if self.graph_gather_rows < 1:
            raise ValueError("expert prefetch needs the graph gather")
        if self.hot_cache.scratch_rows < self.graph_gather_rows + hook.rows:
            raise ValueError("expert prefetch needs hot cache rows after the graph scratch")
        self.prefetch = hook
        self.graph_prefetch_counters = torch.zeros(2, dtype=torch.int64, device=self.hot_cache.device)

    def copy_rows_into_cache(
        self,
        source_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        destination_slots: torch.Tensor,
        count: torch.Tensor,
    ) -> None:
        """Graph-capturable copy of expert rows into hot-cache rows on the current stream.

        Device-resident tensors copy every row; host rows copy the first ``count``.
        """
        for source, destination in self._graph_device_pairs:
            destination.view(torch.uint8).reshape(destination.shape[0], -1).index_copy_(
                0,
                destination_rows,
                source.view(torch.uint8).reshape(source.shape[0], -1).index_select(0, source_rows),
            )
        if self._graph_row_segments is not None:
            self._copy_row_segments_gpu(self._graph_row_segments, source_rows, destination_slots, count)
```

`LayerPrefetch` therefore also carries `rows: int` (B); add it to the protocol.

- **`_gather_graph`:** replace the `plan = plan_graph_routes(...)` call with:

```python
        prefetch_ids = None
        if self.prefetch is not None:
            prefetch_ids = self.prefetch.resolve()
            self.prefetch.launch_next()
        if prefetch_ids is None:
            plan = plan_graph_routes(flat, cache.expert_to_slot, self.graph_gather_rows, cache.capacity)
        else:
            plan = plan_graph_routes(
                flat, cache.expert_to_slot, self.graph_gather_rows, cache.capacity,
                prefetch_ids=prefetch_ids, prefetch_base=self.prefetch_base,
            )
            self.graph_prefetch_counters[0].add_(plan.prefetch_hit_routes)
            self.graph_prefetch_counters[1].add_((prefetch_ids >= 0).sum())
```

The call must come after `residency_update.on_graph_forward`: the planner reads post-update residency. It must also come before the route plan: the side stream (Task 5) starts the next layer's copy ahead of this layer's miss copy.

- [ ] **Step 5: `from_model(prefetch_rows=)`**

- **Signature:** add `prefetch_rows: int = 0` after `gpu_residency_max_promotions`, and reject negatives.
- **Split the dict:** rename the existing `scratch_rows` dict to `graph_rows`, then build `scratch_rows = {layer_id: rows + prefetch_rows if rows else 0 for layer_id, rows in graph_rows.items()}`.
  - `scratch_rows` keeps driving the budget deduction and `ExpertHotCache(...)`.
  - The `enable_graph_gather` loop iterates `graph_rows`.
- **Guard:** if `prefetch_rows` is set, `graph_gather_batch_size` must be > 0, otherwise raise `ValueError`.
- **Off path:** with `prefetch_rows=0`, every number is today's.

- [ ] **Step 6: `serving/plan.py`**

```python
"""Backend-independent expert prefetch plans, built inside the decode graph from the candidate bank."""

from __future__ import annotations

from typing import Mapping, Protocol

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank


class PrefetchPlan(msgspec.Struct, frozen=True):
    target_layer: int
    expert_ids: torch.Tensor  # int64 [B], best score first
    destination_slots: torch.Tensor  # int32 [B], dedicated prefetch rows of the target layer
    destination_rows: torch.Tensor  # int64 [B], the same rows
    count: torch.Tensor  # int32 [1], rows < count are non-resident candidates
    delivered: torch.Tensor  # bool [B], set by the backend once a row's bytes are safe to read


class LayerPrefetch(Protocol):
    """What one streamed layer's graph gather calls; a backend installs one per streamed layer."""

    rows: int

    def resolve(self) -> torch.Tensor | None: ...

    def launch_next(self) -> None: ...


class PrefetchPlanner:
    """Turns candidate scores into per-target plans; every op is fixed-shape and sync-free."""

    def __init__(
        self,
        *,
        bank: PrefetchCandidateBank,
        expert_to_slot: Mapping[int, torch.Tensor],
        prefetch_base: Mapping[int, int],
        budget: int,
        device: torch.device,
    ) -> None:
        if not 1 <= budget <= bank.width:
            raise ValueError("prefetch budget must be between 1 and the candidate width")
        self.budget = budget
        self._bank = bank
        self._expert_to_slot = dict(expert_to_slot)
        self._positions = torch.arange(budget, device=device)
        self.plans: dict[int, PrefetchPlan] = {}
        for layer_id, base in prefetch_base.items():
            rows = torch.arange(base, base + budget, dtype=torch.int64, device=device)
            self.plans[layer_id] = PrefetchPlan(
                target_layer=layer_id,
                expert_ids=torch.zeros(budget, dtype=torch.int64, device=device),
                destination_slots=rows.to(torch.int32),
                destination_rows=rows,
                count=torch.zeros(1, dtype=torch.int32, device=device),
                delivered=torch.zeros(budget, dtype=torch.bool, device=device),
            )

    def plan(self, target_layer: int) -> PrefetchPlan:
        plan = self.plans[target_layer]
        ids = self._bank.ids_for(target_layer)
        scores = self._bank.scores_for(target_layer)
        nonresident = self._expert_to_slot[target_layer].index_select(0, ids) < 0
        order = torch.topk(torch.where(nonresident, scores, torch.full_like(scores, float("-inf"))), self.budget).indices
        plan.expert_ids.copy_(ids.index_select(0, order))
        plan.count.copy_(nonresident.sum().clamp(max=self.budget).to(torch.int32).reshape(1))
        plan.delivered.zero_()
        return plan

    def valid_rows(self, plan: PrefetchPlan) -> torch.Tensor:
        return self._positions < plan.count

    def delivered_ids(self, plan: PrefetchPlan) -> torch.Tensor:
        return torch.where(plan.delivered, plan.expert_ids, torch.full_like(plan.expert_ids, -1))
```

Scores are non-negative (sigmoid, or masked softmax), so every non-resident candidate outranks the `-inf` residents, and `count` rows are exactly the non-resident prefix.

- [ ] **Step 7: `serving/in_graph_copy.py` (`at_target`)**

```python
"""In-graph copy prefetch backend: plan rows are copied by the JIT row-copy kernel inside the decode graph."""

from __future__ import annotations

from typing import Mapping

import torch

from sglang.srt.layers.moe.expert_prediction.serving.plan import PrefetchPlanner

IN_GRAPH_SCHEDULES = ("side_stream", "at_target")


class _AtTargetLayer:
    """Plans and copies this layer's own rows right before its route plan (no overlap)."""

    def __init__(self, *, planner: PrefetchPlanner, streamer, layer_id: int) -> None:
        self.rows = planner.budget
        self._planner = planner
        self._streamer = streamer
        self._layer_id = layer_id

    def resolve(self) -> torch.Tensor:
        plan = self._planner.plan(self._layer_id)
        self._streamer.copy_rows_into_cache(plan.expert_ids, plan.destination_rows, plan.destination_slots, plan.count)
        plan.delivered.copy_(self._planner.valid_rows(plan))
        return self._planner.delivered_ids(plan)

    def launch_next(self) -> None:
        return None


class InGraphCopyBackend:
    @classmethod
    def install(
        cls,
        *,
        planner: PrefetchPlanner,
        streamers: Mapping[int, object],
        next_target: Mapping[int, int],
        schedule: str,
        device: torch.device,
    ) -> "InGraphCopyBackend":
        if schedule not in IN_GRAPH_SCHEDULES:
            raise ValueError(f"unknown in-graph prefetch schedule {schedule!r}; expected one of {IN_GRAPH_SCHEDULES}")
        if schedule == "side_stream":
            raise NotImplementedError("side_stream prefetch lands in Task 5")
        for layer_id in planner.plans:
            streamers[layer_id].install_prefetch(_AtTargetLayer(planner=planner, streamer=streamers[layer_id], layer_id=layer_id))
        return cls()
```

`next_target` maps source layer → target layer; `at_target` ignores it. It is part of the signature so Task 5 does not change callers.

- [ ] **Step 8: Run the tests**

- CPU: expected `4 passed`, CUDA skipped.
- GPU (< 100 MiB, lock, etiquette): `flock -n /data/models/slang/nvfp4-work/cc-gpu.lock <test command> test/registered/unit/layers/moe/test_expert_prefetch_plan.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency_gpu.py`. Expected: all pass, and the three existing files are unchanged in count.
- If the GPU is unavailable, record "CUDA cases pending" in the commit body.

- [ ] **Step 9: Commit**

```bash
git commit -m "feat(moe): add backend-independent expert prefetch plans served from dedicated hot-cache rows" -- \
  python/sglang/srt/layers/moe/expert_route_plan.py python/sglang/srt/layers/moe/expert_stream.py \
  python/sglang/srt/layers/moe/expert_hot_cache.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/plan.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/in_graph_copy.py \
  test/registered/unit/layers/moe/test_expert_prefetch_plan.py
```

---

### Task 5: `side_stream` schedule — overlap the next layer's copy through one graph break per layer

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_prediction/serving/in_graph_copy.py`
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_side_stream.py`

**Interfaces:**
- Consumes: Task 4; `eager_on_graph` from `model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`.
- Produces:
  - `InGraphCopyBackend.install(schedule="side_stream")`;
  - `InGraphCopyBackend.side_stream: torch.cuda.Stream`;
  - `InGraphCopyBackend.capture_side_graphs()`, called once after install and before decode graph capture.

**Mechanism (see Design → Backends):**
- **Shared state:** `main_seq` and `side_seq` (int64 `[1]`, both 0), plus per target T `ready[T]` (int64 `[B]`, -1) and `row_count[T]` (int32 `[B]`).
- **Source layer L's gather, `launch_next()`, for T = next_target[L]:**
  1. `plan = planner.plan(T)`.
  2. `row_count[T].copy_(planner.valid_rows(plan).to(torch.int32))`.
  3. If L is the first source layer: `main_seq.add_(1)`.
  4. Call the break `_launch(T)`, decorated `eager_on_graph(True)`:
     - `event = torch.cuda.Event()`; `event.record()`;
     - `side.wait_event(event)`;
     - `with torch.cuda.stream(side): side_graphs[T].replay()`.
- **Side graph for T** (captured with `torch.cuda.graph(g, stream=side)` in `capture_side_graphs`):
  - if T is the first target: `side_seq.add_(1)`;
  - for j in range(B): `streamer_T.copy_rows_into_cache(plan.expert_ids[j:j+1], plan.destination_rows[j:j+1], plan.destination_slots[j:j+1], row_count[T][j:j+1])`, then `ready[T][j:j+1].copy_(side_seq)`.
  - Rows copy one launch at a time, best score first, so whole rows become deliverable as they land. `_validate_plan` accepts 1-row contiguous views.
- **Target T's gather, `resolve()`:**
  - `plan.delivered.copy_((ready[T] == main_seq) & planner.valid_rows(plan))`;
  - return `planner.delivered_ids(plan)`.
- A layer that is both a target and a source (every middle layer) resolves first, then launches.

- [ ] **Step 1: Write the failing tests** (CUDA; copy Task 4's fixtures; two layers 0 → 1, each with its own `_layer(seed)` and streamer)

  - **`test_side_stream_delivers_rows_byte_exact_under_breakable_replay`**
    - Build a two-layer forward function: `gather(layer 0 ids)`, then `gather(layer 1 ids)`, copying the outputs.
    - Capture it with the breakable graph harness in `test/registered/unit/model_executor/runner_backend/test_breakable_cuda_graph_backend.py`. Read that file and reuse its capture helper; do not invent one.
    - Replay 12 times with random routes and random bank rows for target 1.
    - After each replay, synchronize and assert both layers' outputs byte-exact against `_source_bytes`, and `graph_prefetch_counters[1]` of layer 1 increasing by `count` when the bank's top candidates are non-resident.
  - **`test_delayed_side_stream_degrades_to_undelivered`**
    - Before one replay, enqueue `torch.cuda._sleep(50_000_000)` on `backend.side_stream`.
    - Assert that replay's layer 1 outputs are byte-exact and `plan.delivered.sum() == 0` for that forward.
    - Assert the next replay (no delay, synchronized) delivers again.
    - This is the derived property the seq protocol exists for.
  - **`test_side_stream_forward_never_synchronizes`:** run the eager two-layer forward once, synchronize, then run it under `set_sync_debug_mode("error")`.
  - **`test_side_stream_refuses_same_layer_targets`:** `next_target={1: 1}` raises `ValueError` ("side_stream needs a later target layer"). Guards against APEX being wired to it.

- [ ] **Step 2: Run to verify failure** (GPU, lock): expect `NotImplementedError: side_stream prefetch lands in Task 5`.

- [ ] **Step 3: Implement** `_SideStreamState`, `_SideStreamLayer(resolve, launch_next)` and `capture_side_graphs` in `in_graph_copy.py`, following the mechanism above.
  - **Validation:** every `next_target[L] > L`; every target's streamer shares one device.
  - **Capture order:** `capture_side_graphs` warms each side step three times on the side stream (as the existing replay tests do) before capturing.
  - **Break wrapper:** decorate a module-level function taking only `(state, target)`, so the break's captured args are plain objects, not tensors that `_weak_ref_if_tensor` would weak-ref.
  - **Logging:** `logger.info("MoE expert prefetch side stream: targets=%d rows=%d breaks_per_forward=%d", ...)`.

- [ ] **Step 4: Run** Task 4's and Task 5's test files under the lock. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(moe): overlap next-layer expert prefetch copies on a side stream from one graph break per layer" -- \
  python/sglang/srt/layers/moe/expert_prediction/serving/in_graph_copy.py \
  test/registered/unit/layers/moe/test_expert_prefetch_side_stream.py
```

---

### Task 6: Live runtime wiring — generic tap hook, scoring runtime, env vars, runner

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_prediction/feature_store.py` (`after_write`)
- Modify: `python/sglang/srt/layers/moe/expert_prediction/adapters.py` (mixer-kind registry)
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/runtime.py` (`from_env`, `build`, `on_forward_end`, `close`)
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py` (orchestration only: one gate condition, one kwarg)
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_runtime.py`; additions to `test_expert_prediction_runtime.py` and `test_expert_prediction_adapters.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4 and 5. Also the existing `FeatureStore`, `TappedMoeLayer`, `install_pre_mixer_taps`, and `ExpertHotCacheManager.{caches, streamers}`.
- Produces:
  - `FeatureStore.after_write: Callable[[int, RouteFeature, int], None] | None`;
  - `register_mixer_kind_adapter(*, architecture, classify)` and `mixer_kinds(*, model, layers)`;
  - `PrefetchScoring.build(...)`, `.from_checkpoints(...)`, `.features_for(predictor)`, `.required_features`, `.metrics_record()`, `.next_target`;
  - `attach_prefetch_backend(*, scoring, manager, budget, schedule, device) -> InGraphCopyBackend`;
  - `expert_prefetch_rows() -> int`;
  - the env vars:
    - `SGLANG_MOE_EXPERT_PREFETCH` (`""|llapor|apex`);
    - `SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR`;
    - `SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES` (16);
    - `SGLANG_MOE_EXPERT_PREFETCH_BUDGET` (3);
    - `SGLANG_MOE_EXPERT_PREFETCH_APEX_TAU` (0.95);
    - `SGLANG_MOE_EXPERT_PREFETCH_SCHEDULE` (`side_stream`).

- [ ] **Step 1: Write the failing tests** (`test_expert_prefetch_runtime.py`)

```python
"""Live prefetch scoring runs from generic tap writes, rewrites stable bank rows, and never synchronizes."""

import unittest

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore

HIDDEN, EXPERTS, TOP_K = 16, 32, 4


class _Cache:
    def __init__(self, device):
        self.expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long, device=device)
        self.expert_to_slot[:6] = torch.arange(6, device=device)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchScoringRuntime(unittest.TestCase):
    def _build(self):
        from sglang.srt.layers.moe.expert_prediction.serving import runtime as serving_runtime
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import LlaporCheckpoint
        from sglang.srt.layers.moe.expert_prediction.training import llapor
        from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats

        device = torch.device("cuda")
        specs = [MoeLayerSpec(layer_id=i, num_experts=EXPERTS, top_k=TOP_K, hidden_size=HIDDEN) for i in range(3)]
        checkpoints = {
            target: LlaporCheckpoint(
                source_layer=target - 1, target_layer=target, group="outer",
                pca=PCAStats(mean=torch.randn(HIDDEN), components=torch.randn(4, HIDDEN), explained_variance=torch.ones(4)),
                model=llapor.build_predictor("outer", pca_rank=4, num_experts=EXPERTS).eval(),
            )
            for target in (1, 2)
        }
        store = FeatureStore(specs=specs, features=serving_runtime.PrefetchScoring.features_for("llapor"),
                             max_rows=1, device=device, hidden_dtype=torch.bfloat16)
        scoring = serving_runtime.PrefetchScoring.from_checkpoints(
            predictor="llapor", checkpoints=checkpoints, specs=specs, store=store,
            hot_caches={i: _Cache(device) for i in range(3)},
            width=8, budget=2, tau=0.95, dtype=torch.bfloat16, device=device,
        )
        return scoring, store

    def _tap(self, store, layer):
        generator = torch.Generator().manual_seed(layer)
        store.write(layer, RouteFeature.ROUTER_INPUT, torch.randn(1, HIDDEN, generator=generator).to("cuda", torch.bfloat16))
        store.write(layer, RouteFeature.TOPK_IDS, torch.randperm(EXPERTS, generator=generator)[:TOP_K].unsqueeze(0).cuda())
        store.write(layer, RouteFeature.TOPK_WEIGHTS, torch.rand(1, TOP_K, generator=generator).cuda())

    def test_source_tap_rewrites_the_next_layers_bank_row_in_place(self):
        scoring, store = self._build()
        self.assertEqual((scoring.targets, scoring.next_target), ([1, 2], {0: 1, 1: 2}))
        ids, scores = scoring.bank.ids_for(2), scoring.bank.scores_for(2)
        pointer = ids.data_ptr()
        self._tap(store, 1)
        torch.cuda.synchronize()
        self.assertEqual(scoring.bank.ids_for(2).data_ptr(), pointer)
        self.assertNotEqual(scores.abs().sum().item(), 0.0)
        self.assertEqual(scoring.bank.scores_for(1).abs().sum().item(), 0.0)

    def test_forward_taps_never_synchronize(self):
        _, store = self._build()
        for layer in range(3):
            self._tap(store, layer)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for layer in range(3):
                self._tap(store, layer)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_replay_updates_candidates_and_metrics_without_python(self):
        scoring, store = self._build()
        inputs = {layer: (torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="cuda"),
                          torch.zeros(1, TOP_K, dtype=torch.long, device="cuda"),
                          torch.zeros(1, TOP_K, device="cuda")) for layer in range(3)}
        for layer, (x, ids, w) in inputs.items():
            ids.copy_(torch.arange(TOP_K, device="cuda").unsqueeze(0) + 6 + layer)

        def forward():
            for layer, (x, ids, w) in inputs.items():
                store.write(layer, RouteFeature.ROUTER_INPUT, x)
                store.write(layer, RouteFeature.TOPK_IDS, ids)
                store.write(layer, RouteFeature.TOPK_WEIGHTS, w)

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        before = scoring.metrics_record()["layers"]["2"]["missed_routes"]
        for _ in range(4):
            inputs[1][0].copy_(torch.randn(1, HIDDEN, device="cuda").to(torch.bfloat16))
            graph.replay()
        torch.cuda.synchronize()
        after = scoring.metrics_record()["layers"]["2"]["missed_routes"]
        self.assertEqual(after - before, 4 * TOP_K)


if __name__ == "__main__":
    unittest.main()
```

Register with `register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")`.

- **The first test** guards the direction of the LLaPor wiring: a source-layer write lands in the *next* layer's row. Swapping source and target would pass the other tests.
- **The replay test:** layer 2's native ids are experts 8..11 and the residents are 0..5, so 4 replays × 4 routes are all misses.
- **Runtime tests** (`test_expert_prediction_runtime.py`): `from_env` raises `ValueError` naming the env var for each of:
  - `SGLANG_MOE_EXPERT_PREFETCH="llapor"` with `expert_hot_cache_manager=None`;
  - `SCHEDULE="side_stream"` with `PREFETCH="apex"`;
  - `side_stream` with `decode_graph_backend="full"`;
  - `PREFETCH` together with `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR`.

  Use `envs.X.override(...)`.
- **Adapter tests** (`test_expert_prediction_adapters.py`): registered and unregistered mixer-kind paths, on that file's fake models.

- [ ] **Step 2: Run to verify failure** (GPU < 200 MiB, lock): `ModuleNotFoundError: ...serving.runtime`.

- [ ] **Step 3: `FeatureStore.after_write`.** In `__init__`, after `self.spill = None`:

```python
        # Called after an in-buffer write with (layer_id, feature, rows); runs at graph capture, not replay.
        self.after_write: Callable[[int, RouteFeature, int], None] | None = None
```

In `write`, the in-buffer branch becomes:

```python
            else:
                buffer[:rows].copy_(flat[:, :width])
                if self.after_write is not None:
                    self.after_write(layer_id, feature, rows)
```

- [ ] **Step 4: Mixer-kind adapter** (`adapters.py`; metric labels only)

```python
MixerKindClassifier = Callable[[nn.Module], str]


def _qwen4exp_mixer_kind(decoder: nn.Module) -> str:
    name = type(decoder).__name__
    return "full_attention" if "Attention" in name else "linear_attention" if "Linear" in name else "unknown"


_MIXER_KIND_ADAPTERS: dict[str, MixerKindClassifier] = {
    "Qwen4ExpForConditionalGeneration": _qwen4exp_mixer_kind,
}


def register_mixer_kind_adapter(*, architecture: str, classify: MixerKindClassifier) -> None:
    if architecture in _MIXER_KIND_ADAPTERS:
        raise ValueError(f"mixer-kind adapter already registered for {architecture}")
    _MIXER_KIND_ADAPTERS[architecture] = classify


def mixer_kinds(*, model: nn.Module, layers: Sequence[TappedMoeLayer]) -> dict[int, str]:
    """An unregistered architecture reports ``unknown`` for every layer."""
    classify = _MIXER_KIND_ADAPTERS.get(type(model).__name__)
    if classify is None:
        return {layer.spec.layer_id: "unknown" for layer in layers}
    return {layer_id: classify(decoder) for layer_id, decoder in decoder_layers_by_moe_layer(model=model, layers=layers).items()}
```

The key matches `_PRE_MIXER_ADAPTERS`. The decoder classes (`models/qwen4_exp.py:1585,1625`) are identified by name, never imported.

- [ ] **Step 5: Write `serving/runtime.py`**

```python
"""Live expert prefetch: score target-layer experts from tap writes inside the decode graph, then plan and copy.

Every per-token op runs from ``FeatureStore.after_write`` and the graph gather's prefetch
hook during eager forwards and graph capture, so replay executes recorded kernels only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall, PrefetchCandidateBank
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.in_graph_copy import InGraphCopyBackend
from sglang.srt.layers.moe.expert_prediction.serving.plan import PrefetchPlanner
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer

logger = logging.getLogger(__name__)

_FEATURES = {
    "llapor": frozenset({RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS}),
    "apex": frozenset({RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS}),
}
# The last feature each tap writes for a layer; scoring fires on it so the others are fresh.
_SCORE_TRIGGER = {"llapor": RouteFeature.TOPK_WEIGHTS, "apex": RouteFeature.PRE_MIXER}


def expert_prefetch_rows() -> int:
    """Dedicated hot-cache rows per layer that ``ExpertHotCacheManager.from_model`` reserves for prefetch."""
    return envs.SGLANG_MOE_EXPERT_PREFETCH_BUDGET.get() if envs.SGLANG_MOE_EXPERT_PREFETCH.get() else 0


class PrefetchScoring:
    """Owns scorers, the candidate bank the planner reads, and budget-recall counters."""

    @staticmethod
    def features_for(predictor: str) -> frozenset[RouteFeature]:
        return _FEATURES[predictor]

    @classmethod
    def build(cls, *, predictor: str, model_dir: Path, specs: Sequence[MoeLayerSpec], **kwargs) -> "PrefetchScoring":
        checkpoints = load_prefetch_checkpoints(model_dir, predictor=predictor, specs=specs)
        return cls.from_checkpoints(predictor=predictor, checkpoints=checkpoints, specs=specs, **kwargs)

    @classmethod
    def from_checkpoints(
        cls,
        *,
        predictor: str,
        checkpoints: Mapping[int, Any],
        specs: Sequence[MoeLayerSpec],
        store: FeatureStore,
        hot_caches: Mapping[int, Any],
        width: int,
        budget: int,
        tau: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PrefetchScoring":
        by_layer = {spec.layer_id: spec for spec in specs}
        if width > min(spec.num_experts for spec in specs):
            raise ValueError("SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES exceeds the expert count")
        missing_caches = sorted(set(checkpoints) - set(hot_caches))
        if missing_caches:
            raise ValueError(f"expert prefetch needs hot caches for layers {missing_caches}")
        if predictor == "llapor":
            scorers = {target: LlaporScorer(c, num_experts=by_layer[target].num_experts, dtype=dtype, device=device)
                       for target, c in checkpoints.items()}
            source_of = {c.source_layer: target for target, c in checkpoints.items()}
            next_target = dict(source_of)
        else:
            scorers = {target: ApexScorer(c, num_experts=by_layer[target].num_experts, tau=tau, dtype=dtype, device=device)
                       for target, c in checkpoints.items()}
            source_of = {target: target for target in checkpoints}
            next_target = {}
        scoring = cls(
            predictor=predictor, scorers=scorers, source_of=source_of, next_target=next_target, store=store,
            hot_caches=hot_caches,
            bank=PrefetchCandidateBank(layer_ids=list(checkpoints), width=width, device=device),
            recall=BudgetRecall(layer_ids=list(checkpoints), budget=budget, device=device),
        )
        store.after_write = scoring._on_write
        logger.info("MoE expert prefetch scoring: predictor=%s targets=%d width=%d budget=%d state_bytes=%d",
                    predictor, len(checkpoints), width, budget, scoring.state_nbytes)
        return scoring

    def __init__(self, *, predictor, scorers, source_of, next_target, store, hot_caches, bank, recall) -> None:
        self.predictor = predictor
        self.targets = sorted(scorers)
        # Source layer -> target layer for a next-layer predictor; empty for same-layer predictors.
        self.next_target = next_target
        self._scorers = scorers
        self._source_of = source_of
        self._store = store
        self._hot_caches = dict(hot_caches)
        self.bank = bank
        self.recall = recall

    @property
    def required_features(self) -> frozenset[RouteFeature]:
        return _FEATURES[self.predictor]

    @property
    def state_nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self._scorers.values() for t in (*s.parameters(), *s.buffers()))

    def _on_write(self, layer_id: int, feature: RouteFeature, rows: int) -> None:
        if feature is RouteFeature.TOPK_IDS and layer_id in self._scorers:
            self.recall.observe(
                target_layer=layer_id, candidate_ids=self.bank.ids_for(layer_id),
                topk_ids=self._store.view(layer_id, RouteFeature.TOPK_IDS, rows),
                expert_to_slot=self._hot_caches[layer_id].expert_to_slot,
            )
        if feature is not _SCORE_TRIGGER[self.predictor] or layer_id not in self._source_of:
            return
        target = self._source_of[layer_id]
        with torch.no_grad():
            if self.predictor == "llapor":
                scores = self._scorers[target](
                    self._store.view(layer_id, RouteFeature.ROUTER_INPUT, rows),
                    self._store.view(layer_id, RouteFeature.TOPK_IDS, rows),
                    self._store.view(layer_id, RouteFeature.TOPK_WEIGHTS, rows),
                )
            else:
                scores = self._scorers[target](self._store.view(layer_id, RouteFeature.PRE_MIXER, rows))
        self.bank.write(target, scores)

    def metrics_record(self, streamers: Mapping[int, Any] | None = None) -> dict:
        """Host read of the device counters; call only at metric log intervals."""
        layers = {str(layer): {"missed_routes": missed, "covered_routes": covered}
                  for layer, (missed, covered) in self.recall.snapshot().items()}
        if streamers is not None:
            served = torch.stack([streamers[layer].graph_prefetch_counters for layer in self.targets]).cpu().tolist()
            for layer, (routes, rows) in zip(self.targets, served):
                layers[str(layer)].update(served_routes=routes, delivered_rows=rows)
        missed = sum(v["missed_routes"] for v in layers.values())
        covered = sum(v["covered_routes"] for v in layers.values())
        return {"predictor": self.predictor, "budget": self.recall.budget,
                "budget_recall": covered / missed if missed else 0.0, "layers": layers}


def attach_prefetch_backend(
    *, scoring: PrefetchScoring, manager: Any, budget: int, schedule: str, device: torch.device
) -> InGraphCopyBackend:
    """Build the planner over the hot caches and install the in-graph copy backend on every target's streamer."""
    planner = PrefetchPlanner(
        bank=scoring.bank,
        expert_to_slot={layer: manager.caches[layer].expert_to_slot for layer in scoring.targets},
        prefetch_base={layer: manager.streamers[layer].prefetch_base for layer in scoring.targets},
        budget=budget,
        device=device,
    )
    backend = InGraphCopyBackend.install(
        planner=planner, streamers=manager.streamers, next_target=scoring.next_target, schedule=schedule, device=device,
    )
    if schedule == "side_stream":
        backend.capture_side_graphs()
    return backend
```

- **Recall vs gather residency:** `recall.observe(L)` reads the live `expert_to_slot` view. The in-graph residency update rewrites it at the first streamed layer's gather, which comes after layer 0's TopK tap. So layer 0's recall uses pre-update residency, while the planner (at gather) sees post-update residency. The difference is one layer out of 47 and only affects the metric.
- **Doorbell:** Task 10 adds its branch in `attach_prefetch_backend`, and only there.

**Launch order within one forward (LLaPor, side_stream):**
1. TopK(L) tap writes ROUTER_INPUT, IDS and WEIGHTS.
2. `_on_write(L, TOPK_IDS)` runs `recall.observe(L)` against the bank row written at L−1.
3. `_on_write(L, TOPK_WEIGHTS)` runs `bank.write(L+1)`.
4. `gather(L)` runs `resolve()` for L's own plan (ready check), then `launch_next()` (plan L+1, break, side copy), then the route plan and residual copy for L.

For APEX with `at_target`: pre-mixer(L) runs `bank.write(L)`, TopK(L) runs `recall.observe(L)`, and `gather(L)` plans and copies L.

- [ ] **Step 6: Env vars, runtime, runner.**
  - **`environ.py`**, directly after `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES` (same section):

```python
    # Live expert prefetch: "" (off), "llapor" or "apex".
    SGLANG_MOE_EXPERT_PREFETCH = EnvStr("")
    SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR = EnvStr("")
    # Candidates per target layer the prefetch planner reads.
    SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES = EnvInt(16)
    # Dedicated hot-cache rows per layer; also the planner's per-layer row budget.
    SGLANG_MOE_EXPERT_PREFETCH_BUDGET = EnvInt(3)
    SGLANG_MOE_EXPERT_PREFETCH_APEX_TAU = EnvFloat(0.95)
    # "side_stream" (overlapped copy, breakable decode graph) or "at_target" (correctness mode).
    SGLANG_MOE_EXPERT_PREFETCH_SCHEDULE = EnvStr("side_stream")
```

  - **`expert_prediction/runtime.py`:**
    - **`from_env`.** Read the six vars. When `PREFETCH` is set, raise `ValueError` naming the env var unless all of these hold:
      - `tokens_per_request == 1`;
      - `decode_max_bs == 1`;
      - `expert_hot_cache_manager is not None`;
      - the model dir is set;
      - `SCHEDULE` is in `IN_GRAPH_SCHEDULES`;
      - `SCHEDULE != "side_stream"` or `PREFETCH == "llapor"`;
      - `SGLANG_MOE_PREFETCH_MAX_CANDIDATES == 0`;
      - `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR` is empty.

      Also pass a `decode_graph_backend: str` argument from the runner. With `side_stream`, raise unless it is `breakable`. Pass `prefetch=PrefetchSettings(predictor, model_dir, width, budget, tau, schedule)` into `build`, where `PrefetchSettings` is a `msgspec.Struct(frozen=True)` in this file.
    - **`build`.** Union `PrefetchScoring.features_for(...)` into the features. After the store and taps are installed:
      - `self.prefetch = PrefetchScoring.build(...)`;
      - `self.prefetch_backend = attach_prefetch_backend(scoring=self.prefetch, manager=manager, budget=..., schedule=..., device=device)`.

      Both default to `None`.
    - **`on_forward_end`.** In the existing metrics block, append `{"prefetch": self.prefetch.metrics_record(manager.streamers), "forwards": self.forwards}` when prefetch is on. For prefetch-only runs, count `forwards` on decode forwards with `rows > 0`.
    - **`close`.** `self.store.after_write = None` when prefetch is on.
  - **`model_runner.py`**, orchestration only (`large-class-style` §1.3):
    - **`maybe_init_expert_prediction` gate:** add `or envs.SGLANG_MOE_EXPERT_PREFETCH.get()`.
    - **`maybe_init_expert_hot_cache`:** add the kwarg `prefetch_rows=expert_prefetch_rows(),`, with a local import next to the existing `ExpertHotCacheManager` import.
    - **`from_env` call:** add `decode_graph_backend=get_exec().graph.cuda_graph_config.decode.backend`. Verify this field name against `cuda_graph_config.py:61-64` (`PhaseConfig.backend`) and the runner's existing `get_exec().graph.cuda_graph_config.decode.max_bs` read.

    No logic enters the runner.

- [ ] **Step 7: Run all prefetch and prediction tests** under the lock: Task 1, 2, 4, 5 and 6 files, plus `test_expert_prediction_runtime.py`, `test_expert_prediction_adapters.py`, `test_expert_prediction_graph.py`, `test_expert_prediction_capture_graph.py`, `test_expert_graph_gather.py`, and `test_expert_residency_gpu.py`. Expected: all pass, and no previously passing test drops.

- [ ] **Step 8: Commit**

```bash
git commit -m "feat(moe): wire live expert prefetch scoring, planning and in-graph copy into serving" -- \
  python/sglang/srt/layers/moe/expert_prediction/feature_store.py \
  python/sglang/srt/layers/moe/expert_prediction/adapters.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py \
  python/sglang/srt/layers/moe/expert_prediction/runtime.py \
  python/sglang/srt/environ.py python/sglang/srt/model_executor/model_runner.py \
  test/registered/unit/layers/moe/test_expert_prefetch_runtime.py \
  test/registered/unit/layers/moe/test_expert_prediction_runtime.py \
  test/registered/unit/layers/moe/test_expert_prediction_adapters.py
```

---

### Task 7: Launcher, A/B subset, driver flags, correctness probe and summary scripts

**Files:**
- Modify: `scripts/expert_prediction/run-shadow-server.sh`
- Modify: `scripts/expert_prediction/benchmarks/run_capture_sessions.py`
- Create: `scripts/expert_prediction/prefetch/select_ab_sessions.py`, `logprob_probe.py`, `compare_logprobs.py`, `summarize_ab.py`

**Launcher env:**
- `PREFETCH=off|llapor|apex` sets `SGLANG_MOE_EXPERT_PREFETCH`.
- `PREFETCH_SCHEDULE` (default `side_stream`), `PREFETCH_BUDGET` (default 3), `PREFETCH_CANDIDATES` (16), `PREFETCH_MODEL_DIR`.
- `HOT_GPU_MB` (default 12288).
- `DOORBELL=0|1` sets `SGLANG_MOE_EXPERT_DOORBELL` **only when `DOORBELL=1`**. When unset, the variable is not exported, so D-off cells equal production's environment.
- `DOORBELL_ENV` is a space-separated `NAME=VALUE` pass-through for doorbell knobs (Task 10).

- [ ] **Step 1: Launcher changes**
  - **Env:** `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1`, `SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS=64`, and the prefetch vars. Build `doorbell_env=()` and append `SGLANG_MOE_EXPERT_DOORBELL=1` only when `DOORBELL=1`. Expand `"${doorbell_env[@]}"` and `${DOORBELL_ENV:-}`.
  - **Prefetch variable:** `prefetch=${PREFETCH:-off}`; `[ "$prefetch" = off ] && prefetch=""`.
  - **Server flags:** `--context-length 40000 --max-total-tokens 40000`. Keep `--cuda-graph-backend-decode breakable --cuda-graph-max-bs-decode 1` (already at lines 110–112).
  - **Refusal:** before the empty-GPU check, `ss -ltn 'sport = :7867' | grep -q LISTEN && { echo "REFUSING_TO_START: production on 7867 is up or relaunching" >&2; exit 1; }`.
  - **Lock:** `exec flock --nonblock /data/models/slang/nvfp4-work/cc-gpu.lock env \`.
  - **Header echo:** `doorbell=`, `prefetch=`, `schedule=`, `budget=`.
  - **Check:** `bash -n` returns 0.

- [ ] **Step 2: Driver flags.**
  - **`run_capture_sessions.py`:** `--max-tokens` (default 4096) replaces the literal. `--session-ids` is a comma list that filters sessions in file order.
  - **Test** against the fake SSE server.

- [ ] **Step 3: `select_ab_sessions.py`, `logprob_probe.py`, `compare_logprobs.py`.** `compare_logprobs.py --exact` fails on any token difference; it is used for the P0D0-vs-REF identity check.

```python
"""Fixed live A/B subset: the first FinanceBench holdout and ConvFinQA val sessions that fit the time budget."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--val", type=int, default=6)
    parser.add_argument("--max-context-chars", type=int, default=80000)
    args = parser.parse_args()
    picked, counts = [], {"holdout": 0, "val": 0}
    limits = {"holdout": args.holdout, "val": args.val}
    with open(args.sessions) as f:
        for line in f:
            session = json.loads(line)
            split = session["split"]
            if split in limits and counts[split] < limits[split] and session["context_chars"] <= args.max_context_chars:
                picked.append(session)
                counts[split] += 1
    with open(args.out, "w") as f:
        for session in picked:
            f.write(json.dumps(session) + "\n")
    print(json.dumps({"sessions": len(picked), "turns": sum(len(s["turns"]) for s in picked), **counts}))


if __name__ == "__main__":
    main()
```

```python
"""Greedy first-turn completions with top-2 logprobs, for the prefetch correctness gate."""

import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = []
    with open(args.sessions) as f:
        sessions = [json.loads(line) for line in f][: args.prompts]
    for session in sessions:
        body = json.dumps({
            "model": "default", "messages": [{"role": "user", "content": session["turns"][0][:6000]}],
            "temperature": 0, "max_tokens": args.max_tokens, "logprobs": True, "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            choice = json.loads(response.read())["choices"][0]
        tokens = [{"token": t["token"], "top": [[c["token"], c["logprob"]] for c in t["top_logprobs"]]}
                  for t in choice["logprobs"]["content"]]
        results.append({"session_id": session["session_id"], "tokens": tokens})
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
```

```python
"""Pass if every divergence between two arms starts at a near-tie (E31 rule: some side's top-2 margin <= 0.375 nats)."""

import argparse
import json

NEAR_TIE_NATS = 0.375


def _margin(entry):
    top = entry["top"]
    return top[0][1] - top[1][1] if len(top) > 1 else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--exact", action="store_true", help="fail on any token difference (identical-build check)")
    args = parser.parse_args()
    base, other = json.load(open(args.base)), json.load(open(args.other))
    failures = []
    for a, b in zip(base, other):
        for index, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
            if x["token"] != y["token"]:
                if args.exact or min(_margin(x), _margin(y)) > NEAR_TIE_NATS:
                    failures.append({"session_id": a["session_id"], "index": index,
                                     "margins": [_margin(x), _margin(y)]})
                break
    print(json.dumps({"compared": len(base), "flips": failures}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: `summarize_ab.py`.**
  - **Metrics:** each `prefetch` record carries `served_routes` and `delivered_rows` summed over layers. Report `delivered_rows_per_token` and `served_routes_per_token` next to `budget_recall`.
  - **Arm keys:** use the cell names below, e.g. `P0D0`, `P1D0`, `P1D0-at_target`.

```python
"""Per arm: median decode tok/s and TTFT over turns with >= 64 completion tokens, plus live prefetch counters."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="arm=results.jsonl[:prefetch-metrics.jsonl]")
    args = parser.parse_args()
    table = {}
    for spec in args.runs:
        arm, paths = spec.split("=", 1)
        results, _, metrics = paths.partition(":")
        turns = [json.loads(line) for line in open(results) if line.strip()]
        good = [t for t in turns if "error" not in t and (t.get("completion_tokens") or 0) >= 64]
        entry = table.setdefault(arm, {"tok_s": [], "ttft": [], "turns": 0, "errors": 0, "prefetch": None})
        entry["tok_s"] += [t["decode_tokens_per_sec"] for t in good if t["decode_tokens_per_sec"]]
        entry["ttft"] += [t["ttft"] for t in good if t["ttft"] is not None]
        entry["turns"] += len(good)
        entry["errors"] += sum("error" in t for t in turns)
        if metrics and Path(metrics).exists():
            records = [json.loads(line) for line in open(metrics) if '"prefetch"' in line]
            if records:
                last = records[-1]
                layers = last["prefetch"]["layers"].values()
                entry["prefetch"] = {
                    "budget_recall": last["prefetch"]["budget_recall"],
                    "delivered_rows_per_token": sum(v["delivered_rows"] for v in layers) / max(last["forwards"], 1),
                    "served_routes_per_token": sum(v["served_routes"] for v in layers) / max(last["forwards"], 1),
                }
    summary = {arm: {"median_decode_tok_s": statistics.median(e["tok_s"]) if e["tok_s"] else None,
                     "median_ttft_s": statistics.median(e["ttft"]) if e["ttft"] else None,
                     "turns": e["turns"], "errors": e["errors"], "prefetch": e["prefetch"]}
               for arm, e in table.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Check and commit.** Run `bash -n`, each script's `--help` on divix01 (CPU), and the fake-SSE driver test. Then:

```bash
git commit -m "feat(nvfp4): add live prefetch matrix launcher knobs, subset, logprob gate and summary" -- \
  scripts/expert_prediction/run-shadow-server.sh scripts/expert_prediction/benchmarks/run_capture_sessions.py \
  scripts/expert_prediction/prefetch/select_ab_sessions.py scripts/expert_prediction/prefetch/logprob_probe.py \
  scripts/expert_prediction/prefetch/compare_logprobs.py scripts/expert_prediction/prefetch/summarize_ab.py
```

---

### Task 8: Schedule microbench — `side_stream` vs `at_target` on real layer tensors, concurrent copies, break cost

GPU needed, < 4 GiB. It fits beside production only if 7867 is healthy and the lock is free; otherwise it runs in Task 9's approved window, before the matrix.

**Files:**
- Create: `scripts/expert_prediction/prefetch/bench_prefetch_schedule.py`
- Report: `docs/superpowers/experiments/2026-09-1X-expert-prefetch-schedule-bench.md`

- [ ] **Step 1: Write the bench.**
  - **Setup:** reuse E27's harness shape (`work/cc-overlap/bench_overlap.py` on divix01; read it first) with real NVFP4 layer tensors registered in the host arena.
  - **Two streamed layers:** L (source) and L+1 (target), each with the graph gather and B prefetch rows, fed by a fixed candidate plan.
  - **Arms**, each captured with the breakable graph backend:
    1. baseline: no prefetch, misses copied in-graph at each gather;
    2. `at_target`;
    3. `side_stream`, launched before L's miss copy (the Task 5 order);
    4. `side_stream`, launched after L's miss copy (event recorded after L's copy; a bench-only variant).
    5. Compute stand-in between gathers: replay a captured 0.267 ms (linear) / 0.290 ms (full) E28 layer-compute graph.
  - **Sweep:** B ∈ {1, 2, 3, 4, 6}, residual misses ∈ {0, 1, 3}, and plan hit fraction ∈ {0, 0.5, 1}.
  - **Measure:**
    - wall ms per two-layer step (CUDA events on the main stream);
    - delivered rows per target (`graph_prefetch_counters[1]`);
    - L's miss copy GiB/s with and without a concurrent side copy, the unmeasured E27 gap;
    - host time per break (`time.perf_counter_ns` around `BreakableCUDAGraph.replay`, minus the 0-break arm).
  - **Correctness:** assert byte-exact rows in every arm.

- [ ] **Step 2: Multi-stream capture spike (≤ 2 h, report only).**
  - Try recording the side copy inside the main decode graph: capture with `torch.cuda.graph(..., stream=main)`, then switch `torch.cuda.stream(side)` inside the capture region.
  - Check whether replay keeps the side kernels on the side stream: side stream busy with `torch.cuda._sleep` while main-stream timestamps advance.
  - Expected from E27: no. If yes, record it as a follow-up that removes the per-layer breaks.

- [ ] **Step 3: Report and decide.**
  - **Contents:** tables per arm; the concurrent-copy slowdown; break host cost per forward (× 47); and the projected ms/token = Σ layers (baseline − arm), using Task 3's per-layer hit rates.
  - **Decision rules:**
    - `side_stream` stays the live default if its projected saving is ≥ 3 ms/token *after* break cost.
    - If concurrent copies slow L's copy by > 10%, the live budget uses Task 3's gap window (`side_ready_rows_gap`).
    - If break cost alone exceeds the saving, Task 9 runs only the correctness cells and reports NO-GO for tok/s.

```bash
git commit -m "docs(moe): benchmark in-graph prefetch schedules on real layer tensors" -- \
  scripts/expert_prediction/prefetch/bench_prefetch_schedule.py \
  docs/superpowers/experiments/2026-09-1X-expert-prefetch-schedule-bench.md
```

---

### Task 9: Live 4-cell matrix — doorbell-off cells now, doorbell-on cells after Task 10 (ask the user first)

**Cells** (`P` = `SGLANG_MOE_EXPERT_PREFETCH`, `D` = `SGLANG_MOE_EXPERT_DOORBELL`):

| Cell | Env | Checks | Runs in |
|---|---|---|---|
| `REF` | server from the pre-change commit `7de955329a`, P/D unset, same launcher flags | reference for "identical to production" | this task |
| `P0D0` | current branch, P unset, D unset | **logprobs exactly equal to REF** (`compare_logprobs.py --exact`) and tok/s within REF's run-to-run noise | this task |
| `P1D0` | `PREFETCH=llapor PREFETCH_SCHEDULE=side_stream PREFETCH_BUDGET=<B> HOT_GPU_MB=11776` | near-tie rule vs P0D0; tok/s delta; delivered/served counters | this task |
| `P1D0-at_target` | as P1D0 with `PREFETCH_SCHEDULE=at_target` | near-tie rule only (probe, no tok/s run) | this task |
| `P0D1` | `DOORBELL=1`, P unset | near-tie rule vs P0D0; tok/s | Task 10 |
| `P1D1` | `DOORBELL=1 PREFETCH=llapor` | near-tie rule vs P0D1; tok/s | Task 10 |

`HOT_GPU_MB=11776` (−512 MiB) holds total VRAM roughly equal for the scorer state (~100 MB) plus bank buffers. Prefetch rows come out of the same budget automatically. Record hot-slot counts per arm from the startup log.

**Preconditions:**
- Task 3 is GO (or conditional GO with Task 8 passing).
- Tasks 4–7 are merged, and all their tests pass on the synced divix01 worktree.
- Task 8's report exists (if Task 8 needs the window, it runs first inside it).

- [ ] **Step 1: Ask the user** (AskUserQuestion) for an approved production-down window.
  - **Estimate:** REF + P0D0 + P1D0 + at_target probe ≈ 4 launches per pass × ~23 min × 2 passes ≈ 3–3.5 h, plus ~1 h if Task 8 runs in the window.
  - **Confirm** that the crypto-c9 production session has released the GPU and will not relaunch 7867 during the window.
  - **Confirm** the live budget B (Task 3/8 argmax) and whether to add a second budget arm (+~50 min).
  - **Worktree:** `REF` needs a detached worktree at `7de955329a` under `/data/models/slang/nvfp4-work/cc-expert-prediction/ref-7de955` (create it with `git worktree add --detach`). Never use the serving worktree.
  - Do nothing further without a yes.

- [ ] **Step 2: Build the subset** (CPU): `select_ab_sessions.py --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl --out /mnt/nvme2/nvfp4-work/benchmarks/prefetch-ab/sessions.jsonl`. It yields 2 FinanceBench + 6 ConvFinQA val sessions, ≈29 turns, ≈17 min per run at `--max-tokens 768`.

- [ ] **Step 3: Arm order and procedure.**
  - **Order:** pass 1 runs REF → P0D0 → P1D0 → at_target-probe; pass 2 runs P1D0 → P0D0 → REF. The at_target probe runs once, so the order is ABC/CBA.
  - **Each launch:**
    1. `ssh -n divix01 'nohup env PREFETCH=... <launcher> prefetch-<cell>-p<pass> 31040 off radix > /dev/null 2>&1 &'`. REF uses the ref worktree's launcher copy.
    2. Wait for `/health` 200 with a Monitor until-loop.
    3. Warm up with one turn outside the subset.
    4. Run `logprob_probe.py`.
    5. Run `run_capture_sessions.py --max-tokens 768` (skipped for the probe cell).
    6. Copy `expert-prediction.metrics.jsonl`.
    7. `pkill -f '[s]glang serve.*--port 31040'` and wait for an empty GPU.
  - **Required log lines:**
    - P1D0: `MoE expert prefetch scoring: predictor=llapor targets=47` and `MoE expert prefetch side stream: targets=47`;
    - P0D0: neither line.
    - Any capture error stops the run with the log excerpt.

- [ ] **Step 4: Correctness gate.**
  - **`--exact`:** `compare_logprobs.py --exact REF-p1 P0D0-p1`. It must pass; a failure means the off path is not production, and the run stops.
  - **Near-tie rule:** `compare_logprobs.py P0D0-p1 P1D0-p1` and `P0D0-p1 at_target`, with `REF-p1` vs `REF-p2` as the noise bound. A large-margin flip beyond REF-vs-REF fails the arm.

- [ ] **Step 5: Summarize** with `summarize_ab.py`, per cell and per pass.
  - **Report:** median decode tok/s, TTFT, errors, budget recall, delivered rows/token, served routes/token, hot slots, and scorer state bytes.
  - **Compare** against Task 3's projection and Task 8's projection.
  - **Target:** ≥ 5% median decode improvement for P1D0 over P0D0 with no correctness failure. With 2 runs per cell, label the result preliminary.

- [ ] **Step 6: Write-up and commit.** Write `docs/superpowers/experiments/2026-09-1X-expert-prefetch-live-ab.md` with setup (commits, flags, B), the cell table, the gate outputs, deviations, the APEX status, and next steps. Commit, push, sync. **Do not relaunch production;** tell the user the GPU is free.

```bash
git commit -m "docs(moe): live LLaPor prefetch matrix, doorbell-off cells" -- docs/superpowers/experiments/2026-09-1X-expert-prefetch-live-ab.md
```

---

### Task 10: `doorbell` backend and doorbell-on cells (BLOCKED on `cc/doorbell-serving`)

**Blocked until:**
- crypto-c9's doorbell serving integration is on `codex/nvfp4-expert-stream-main`;
- `SGLANG_MOE_EXPERT_DOORBELL=1` starts a server whose production-flag logprobs pass the near-tie rule.

Do not start earlier, and do not copy code from the unlanded branch.

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/doorbell_backend.py`
- Modify: `serving/runtime.py` (`attach_prefetch_backend` branch)
- Modify: `expert_prediction/runtime.py` (`from_env` validation)
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_doorbell.py`

- [ ] **Step 1: Read the landed API.** Find the handle on `ExpertHotCacheManager` (assumed `doorbell`), `post`, `wait`, `quiesce`, and whether `post`/`wait` are capturable or host calls. Update this task's steps in the plan file if they differ, then commit the plan edit.
- [ ] **Step 2: Tests** (CUDA, lock), mirroring Task 5:
  - byte-exact rows under replay with a real copier on two layers;
  - an undelivered plan (copier stalled, per the doorbell's own test hooks) is byte-exact;
  - `set_sync_debug_mode("error")` passes, or the test documents the doorbell's own host step;
  - the four env combinations select the right backend: P0D0 installs nothing, P0D1 installs the doorbell's own path only, P1D0 installs `in_graph_copy`, and P1D1 installs `doorbell`.
- [ ] **Step 3: Implement `_DoorbellLayer`.**
  - **`launch_next()`:** `plan = planner.plan(T)`, then `copier.post(plan.expert_ids, plan.destination_slots, plan.count, tag=T)`.
  - **`resolve()`:** `copier.wait(tag=L)`, then `plan.delivered.copy_(planner.valid_rows(plan))`, and return the delivered ids.
  - **Selection:** `attach_prefetch_backend` picks it when `manager.doorbell is not None`. `from_env` needs no new env var: the doorbell's own flag selects the backend.
  - **Destination rows:** they are Task 4's dedicated prefetch rows. If the doorbell insists on its own scratch rows, pass those as `prefetch_base` instead and drop the `prefetch_rows` reservation for this backend.
- [ ] **Step 4: Live cells P0D1 and P1D1.** Follow Task 9's procedure, with a new user approval and a ≈1–1.5 h window: 2 cells × 2 passes, plus a P0D0 anchor launch. Append the rows to the Task 9 write-up.
- [ ] **Step 5: Commit** the three code files, the test, and the write-up update.

---

## Risks

1. **The window may be too short to pay.**
   - The gap window fits 1 row per layer. Only an overlapped side copy (≈0.88 ms, ~3 rows) approaches E28's ~10 ms/token ceiling.
   - Overlap depends on concurrent copy launches not slowing each other, which E27 never measured. Task 8 measures it.
   - A NO-GO at Task 3 or Task 8 is a likely, legitimate outcome.
2. **APEX's same-layer window (~0.21–0.23 ms) fits 0 rows in-graph, and less than reaction + one row on the doorbell.**
   - It is deferred from live. The in-graph side stream cannot serve it structurally, and `at_target` is never faster.
   - It needs a doorbell same-layer post/resolve plus a wait, priced in Task 3.
3. **47 graph breaks per token** (`side_stream`).
   - Each break is a host launch (event, wait_event, side replay), so the decode step becomes 48 segments.
   - If host launch falls behind GPU compute, the GPU idles and the saving evaporates.
   - Task 8 prices it, and the multi-stream capture spike is the escape hatch.
4. **In-graph scoring cost is on the main stream.**
   - The LLaPor middle scorer runs PCA plus three GEMMs plus scatter/cat/sigmoid/topk, ≈10 kernels × 47 layers, which could cost ~5–9 ms/token.
   - P1D0 vs P0D0 measures it end to end. The follow-ups are fusing PCA into the first linear and a fused kernel.
5. **Prefetch rows cost hot slots.** B × 48 × 2.64 MiB comes out of the hot budget (B = 3 → 380 MiB ≈ 144 slots), which adds baseline misses. The planner's value must exceed that; Task 9 reports slot counts.
6. **Off-path drift.** Any recorded-kernel change with prefetch off breaks "identical to production". Task 4 keeps `prefetch_ids is None` on the old code path, and Task 9 gates it with `--exact` against `7de955329a`.
7. **Seq-protocol assumptions.**
   - Safety needs one graph-gather decode forward at a time, with the next launch after readback: no overlap schedule, bs = 1, no speculation.
   - Task 6 validates bs and speculation. If the overlap scheduler is ever enabled with this path, rows could be read while the next forward's side copy rewrites them.
   - Production and the shadow launcher both pass `--disable-overlap-schedule`. Task 6 makes it a hard check: the runner passes the server-args overlap flag into `from_env` (read the exact field name from `server_args`), and `side_stream` is refused when the overlap schedule is on.
8. **Doorbell dependency and contract drift.** Only Task 10's backend file and the launcher pass-through change. Tasks 1–9 are unaffected.
9. **Timing assumptions.**
   - The pricing uses E27/E28/E32 medians at 4,957 slots; production has 4,180.
   - Recall uses the capture's own residency, so it is faithful. The ms model is approximate (±4 ms/token per E26).
10. **Two runs per cell is preliminary**, against spec §8's five matched runs.
11. **GPU etiquette vs live tests.** The live cells need production down. This plan treats only an approved, coordinated window (no 7867 listener, empty GPU) as the exception, confirmed in Task 9 Step 1.

## Decisions

**Made in this plan:**
- **Scoring:** in-graph via `FeatureStore.after_write`. LLaPor fires on TOPK_WEIGHTS(L) and APEX on PRE_MIXER(L).
- **Planner:** backend-independent. It clamps to B by score over non-resident candidates, and the delivered mask is set by the backend.
- **Destination:** dedicated prefetch rows after the graph scratch, not evictable hot slots.
- **Default backend `in_graph_copy`, schedule `side_stream`:**
  - one break per source layer;
  - per-row copy launches best-first, so partial delivery counts;
  - the side copy launches before the source layer's own miss copy;
  - `at_target` is kept as a correctness mode.
- **APEX:** deferred from live, and in-graph side stream refused for it.
- **P0D0 reference:** a server at `7de955329a`, with exact logprob equality.
- **Pricing:** gap and overlap windows both reported; GO at ≥ 3 ms/token on `shifted_test`.

**Need the user (asked at the named step, via AskUserQuestion):**
1. The production-down window for Task 9 (≈3–3.5 h, +1 h if Task 8 needs the GPU), and again for Task 10 (≈1–1.5 h).
2. The live budget B, if Task 3 and Task 8 disagree (gap vs overlap argmax), and whether to add a second budget arm.
3. On Task 3 NO-GO: whether to build Tasks 4–7 anyway as plumbing for the doorbell.
4. Whether a Task 3 result showing recall still rising past B = 10 justifies a follow-up evictable-slot plan.

## Self-review notes

- **Requirement coverage:**
  - model-agnostic placement → `serving/` package, generic `after_write`, adapters only;
  - in-graph with no host syncs → Tasks 2, 4, 5 and 6 sync tests, plus Design "outside the graph" (breaks listed);
  - plan interface (ids `[B]`, slots `[B]`, device count, delivered mask, residual = misses − delivered) → Design and Task 4;
  - `in_graph_copy` default, doorbell unset → Tasks 4–6;
  - `doorbell` optional and blocked → Task 10;
  - schedule choice justified and measured → Design and Task 8;
  - 4-cell matrix → Task 9 (D-off) and Task 10 (D-on);
  - offline recall of non-resident natives within budget → Task 3;
  - APEX feasibility → Design, Task 3 and Risk 2.
- **Project rules:** `msgspec.Struct` for new containers (`PrefetchPlan`, `PrefetchSettings`, checkpoints). The existing `GraphRoutePlan` dataclass is extended, not converted. No `getattr`/`hasattr` in new code. Envs are in `environ.py` next to the predictor block. Runner edits are one gate condition and two kwargs.
- **Names are consistent across tasks:**
  - `PrefetchPlan`, `PrefetchPlanner.plan`/`valid_rows`/`delivered_ids`;
  - `LayerPrefetch.rows`/`resolve`/`launch_next`;
  - `ExpertStreamer.install_prefetch`/`prefetch_base`/`copy_rows_into_cache`/`graph_prefetch_counters`;
  - `InGraphCopyBackend.install`/`capture_side_graphs`/`side_stream`;
  - `PrefetchScoring.build`/`from_checkpoints`/`features_for`/`metrics_record`/`next_target`/`targets`;
  - `attach_prefetch_backend`, `expert_prefetch_rows`;
  - `budget_hits`, `doorbell_saving_ms`, `side_stream_ready_rows`.
- **Specified by contract, not code:** Task 5's side-stream implementation and tests, Task 8's bench, and Task 10. Each depends on an API to be read at implementation time: the breakable capture helper, E27's harness, and the landed doorbell. The mechanism, the assertions, and the failure each test guards are fixed here.
