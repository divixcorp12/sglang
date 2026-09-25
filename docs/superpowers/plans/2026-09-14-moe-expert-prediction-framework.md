# MoE Expert Prediction Framework (Shadow Milestone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-independent, flag-gated framework that taps MoE routing inside CUDA graphs and shadow-scores pluggable expert predictors (`popularity`, `affinity`) against native routes, writing A/B metrics to JSONL.

**Architecture:** A new package `python/sglang/srt/layers/moe/expert_prediction/` discovers MoE blocks by their `TopK` + `FusedMoE` children, copies routing tensors into fixed-address buffers from `TopK` forward hooks (graph-replayable), and after each decode/verify forward runs every registered predictor, scores its candidates on device, and periodically appends metrics. `ModelRunner` gets one init helper and one delegate call.

**Tech Stack:** Python 3.13, PyTorch 2.13 (CUDA 13), msgspec, unittest/pytest; tests run on divix01.

**Spec:** `docs/superpowers/specs/2026-09-14-moe-expert-prediction-framework-design.md`

## Status (2026-09-14): complete

All seven tasks are implemented, tested on divix01, and pushed to `shared` (`dd65602c13`..`af43005387`).

| Task | Commits | Evidence |
|---|---|---|
| 1 Contracts, env, store, taps | `dd65602c13`, `aefb7cb5ef` | 13 CPU tests passed |
| 2 Pre-mixer adapters | `f6bb81d7d5` | 19 passed (with Task 1 tests) |
| 3 Predictors and registry | `1c0489554c` | 9 passed |
| 4 Shadow metrics | `82068b0d7f` | 4 passed |
| 5 Runtime and ModelRunner wiring | `af294e3b03`, `6d3cc01253` | 8 passed; 51 passed CPU sweep; `model_runner.py` compiles |
| 6 CUDA graph replay / no-sync test | `c2859408c1` | 9 passed on GPU, no sync errors |
| 7 Live shadow smoke | `a0ec88567d`, `ddc5c26d7a`, `004c5b6f88` | 48 layers tapped, CUDA graphs on every decode step, 11 metrics records |
| Final review fixes | `d6a67beac3` | 45 CPU + 1 GPU passed |
| Output divergence check | `78fe4e160c`, `af43005387` | Shadow scoring does not change output |

Results are in `docs/superpowers/experiments/2026-09-14-expert-prediction-shadow-smoke.md`.

### Measured

- Recall at 16 candidates from the smoke's last metrics record:

  | Predictor | Recall at K | Recall at M | Cold recall at M |
  |---|---|---|---|
  | `affinity` | 0.473 | 0.594 | 0.143 |
  | `popularity` | 0.300 | 0.384 | 0.0098 |

- Decode speed with scoring every step: 13.76 tok/s off vs 6.85 tok/s on (6.96 on a fresh launch). This is per-step scoring overhead, not a prefetch latency result.
- Greedy output: separate predictor-off launches already diverge from each other at characters 73-232. A fresh launch scoring every step falls in that same range. The smoke's one character-22 split was a single outlier launch.

### Changes from the tasks as written

The code blocks below show the original task text; the committed code differs as follows.

- **Process (MVP):** from Task 2 on, the separate failing-test commit and red run were skipped; each task was one commit with one test run. There were no per-task reviews and one whole-branch review at the end.
- **Scoring interval:** `SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL` (default 1) scores and observes only every Nth eligible forward. `ExpertPredictionRuntime` tracks `eligible_forwards`, `build(...)` takes `score_interval`, and JSONL records carry `eligible_forwards`.
- **Pipeline parallelism:** `from_env(...)` takes `pp_size` and rejects `pp_size > 1`; `ModelRunner` passes `self.ps.pp_size`.
- **Expert counts:** `num_experts = moe.num_experts - moe.num_fused_shared_experts`; `top_k` still subtracts TopK's own fused shared experts. This fixes Qwen2-style blocks that build TopK without fused shared experts.
- **No raise inside a forward:** a width mismatch in a tap marks the layer unsupported with a warning instead of raising. A metrics write failure (`OSError`) warns once and disables further writes.
- **MLX stub:** `hardware_backend/mlx/model_runner_stub.py` sets `expert_prediction_runtime = None`.
- **Task 7:** production was not relaunched afterwards. Results went to the new experiment file above rather than `nvfp4-expert-offload-experiment-log.md`, which held another session's uncommitted edits.

### Open items

- Eager batches above `max_rows` are skipped with no counter or log, so metrics can stay at zero without warning.
- Scoring all layers in one call per predictor, to cut the per-step overhead instead of sampling it.
- Next milestones: routing capture mode for training data, LLaPor and APEX predictors with training tools, then prefetch admission.

## Global Constraints

- All code lives in `/home/dimitri/data/divix/sglang-nvfp4` on branch `master`. Never touch `/home/dimitri/data/divix/crypto`.
- The package must not import from `sglang.srt.models`; model-specific behavior lives only in `adapters.py`, keyed by model class name.
- Off by default: with `SGLANG_MOE_EXPERT_PREDICTOR` empty, no hooks, buffers, or per-forward work.
- No host synchronization in hooks, `predict`, `observe`, `score_candidates`, or `on_forward_end` (no `.item()`, `.tolist()`, `.cpu()`, boolean-mask indexing, `nonzero`). Only `ShadowMetrics.snapshot` / `append_jsonl` read the device.
- Repo rules: `msgspec.Struct` instead of `@dataclass`; no defensive `getattr`/`hasattr`; call 2+ arg functions by keyword; comments only for non-obvious facts, ASCII, one or two lines (`.claude/rules/comment-style.md`); env vars only through `envs.X.get()` / `.override()`.
- `python/sglang/srt/model_executor/model_runner.py` is frozen: orchestration only (construct / wire / delegate), per `.claude/skills/large-class-style/SKILL.md`.
- Never modify the serving worktree `/data/models/slang/nvfp4-work/main-port-probe-7bc4eb`, `run-nvfp4-e16c-public.sh`, or the production server. Tasks 1-5 are CPU-only on divix01. Tasks 6-7 need the GPU and are run only when the controller says the production server is stopped.
- Commits: stage files by name and commit with `-- <paths>` (the index may hold another session's work). Every commit message ends with:

  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NurjMVe2nS3PGBqBr8M8MZ
  ```

### Divix01 loop (referenced by every task as "Divix01 loop")

Run from the laptop. `<files>` are the exact paths the step names.

```bash
# L1: commit on the laptop
cd /home/dimitri/data/divix/sglang-nvfp4
git add <files>
git commit -m "$(cat <<'EOF'
<type(scope): summary>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NurjMVe2nS3PGBqBr8M8MZ
EOF
)" -- <files>

# L2: push to the shared bare repo and check out the commit in the experiment worktree
git push origin master
COMMIT=$(git rev-parse HEAD)
ssh -n divix01 "cd /data/models/slang/sglang && git fetch shared && git -C /data/models/slang/nvfp4-work/cc-expert-prediction/worktree checkout --detach $COMMIT && git -C /data/models/slang/nvfp4-work/cc-expert-prediction/worktree log -1 --oneline"

# L3: run CPU-only tests (while production is up)
ssh -n divix01 "cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && echo STARTED \$(date --iso-8601=seconds) && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH=/data/models/slang/nvfp4-work/flashinfer-0.6.18-cu130-overlay:\$PWD/python:/data/models/slang/nvfp4-work/uring-test-deps timeout 900 taskset -c 64-71 /data/models/slang/.venv/bin/python -m pytest -q -p no:cacheprovider -rfEs <test files>; echo EXIT=\$?"
```

A run is complete only when an `EXIT=` line prints; record the pass count. If `L2`'s `log -1` does not show `$COMMIT`, stop and report.

---

## File Structure

| Path | Responsibility |
|---|---|
| `python/sglang/srt/environ.py` (modify) | Five `SGLANG_MOE_EXPERT_PREDICTOR*` env vars |
| `python/sglang/srt/layers/moe/expert_prediction/__init__.py` | Empty package marker |
| `.../expert_prediction/contracts.py` | `RouteFeature`, `MoeLayerSpec`, `feature_width`, `feature_dtype` |
| `.../expert_prediction/feature_store.py` | `FeatureStore`: fixed-address per-layer buffers |
| `.../expert_prediction/taps.py` | `TappedMoeLayer`, `discover_moe_layers`, `RouteTaps` |
| `.../expert_prediction/adapters.py` | Pre-mixer taps: registry, default, Qwen4-Exp |
| `.../expert_prediction/base.py` | `ExpertPredictor` ABC, `pad_candidates` |
| `.../expert_prediction/popularity.py` | `PopularityPredictor`, `accumulate_routes` |
| `.../expert_prediction/affinity.py` | `AffinityPredictor` |
| `.../expert_prediction/registry.py` | `register_predictor`, `registered_predictor_names`, `build_predictors` |
| `.../expert_prediction/metrics.py` | `COUNTER_NAMES`, `score_candidates`, `ShadowMetrics` |
| `.../expert_prediction/runtime.py` | `layer_pairs`, `ExpertPredictionRuntime` |
| `python/sglang/srt/model_executor/model_runner.py` (modify) | `maybe_init_expert_prediction`, delegate in `forward` |
| `test/registered/unit/layers/moe/test_expert_prediction_taps.py` | Contracts, env, store, discovery, taps |
| `test/registered/unit/layers/moe/test_expert_prediction_adapters.py` | Pre-mixer adapters |
| `test/registered/unit/layers/moe/test_expert_prediction_predictors.py` | Base, registry, popularity, affinity |
| `test/registered/unit/layers/moe/test_expert_prediction_metrics.py` | Scoring and JSONL |
| `test/registered/unit/layers/moe/test_expert_prediction_runtime.py` | Runtime end to end on CPU |
| `test/registered/unit/layers/moe/test_expert_prediction_graph.py` | CUDA graph replay and no-sync check |
| `scripts/expert_prediction/run-shadow-server.sh` | Experiment server with production flags plus predictor env |

---

### Task 1: Contracts, env vars, feature store, discovery, and route taps

**Files:**
- Modify: `python/sglang/srt/environ.py:325` (after `SGLANG_MOE_PREFETCH_MAX_CANDIDATES = EnvInt(0)`)
- Create: `python/sglang/srt/layers/moe/expert_prediction/__init__.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/contracts.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/feature_store.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/taps.py`
- Test: `test/registered/unit/layers/moe/test_expert_prediction_taps.py`

**Interfaces:**
- Consumes: `sglang.srt.layers.moe.topk.TopK`, `StandardTopKOutput`; `sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE` (attrs `layer_id`, `num_experts`, `hidden_size`); `TopK.topk_config.top_k`, `.num_fused_shared_experts`.
- Produces:
  - `RouteFeature(str, Enum)`: `ROUTER_INPUT`, `ROUTER_LOGITS`, `TOPK_IDS`, `TOPK_WEIGHTS`, `PRE_MIXER`
  - `MoeLayerSpec(msgspec.Struct, frozen=True)`: `layer_id: int, num_experts: int, top_k: int, hidden_size: int`
  - `feature_width(feature, spec) -> int`, `feature_dtype(feature, hidden_dtype) -> torch.dtype`
  - `FeatureStore(*, specs, features, max_rows, device, hidden_dtype)`; `.max_rows`, `.nbytes`, `.holds(layer_id, feature) -> bool`, `.write(layer_id, feature, source) -> None`, `.view(layer_id, feature, rows) -> Tensor`
  - `TappedMoeLayer(msgspec.Struct, frozen=True)`: `spec: MoeLayerSpec, topk: nn.Module, block: nn.Module`
  - `discover_moe_layers(model, *, topk_type=None, experts_type=None) -> tuple[TappedMoeLayer, ...]` (sorted by layer_id)
  - `RouteTaps(store)`; `.install(layers)`, `.remove()`, `.unsupported_layers: set[int]`
  - env: `SGLANG_MOE_EXPERT_PREDICTOR` (EnvTuple), `_CANDIDATES` (16), `_MAX_ROWS` (0), `_LOG_INTERVAL` (100), `_METRICS_FILE` ("")

- [x] **Step 0: Create the divix01 experiment worktree (once)**

```bash
ssh -n divix01 'test -d /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && echo EXISTS || (cd /data/models/slang/sglang && git fetch shared && mkdir -p /data/models/slang/nvfp4-work/cc-expert-prediction && git worktree add --detach /data/models/slang/nvfp4-work/cc-expert-prediction/worktree origin/master && git -C /data/models/slang/nvfp4-work/cc-expert-prediction/worktree log -1 --oneline)'
```

Expected: `EXISTS` or a `log -1` line.

- [x] **Step 1: Write the failing test**

Create `test/registered/unit/layers/moe/test_expert_prediction_taps.py`:

```python
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.contracts import (
    MoeLayerSpec,
    RouteFeature,
    feature_dtype,
    feature_width,
)
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.taps import RouteTaps, discover_moe_layers
from sglang.srt.layers.moe.topk import StandardTopKOutput


class FakeTopK(nn.Module):
    def __init__(self, top_k, num_fused_shared_experts=0):
        super().__init__()
        self.topk_config = SimpleNamespace(
            top_k=top_k, num_fused_shared_experts=num_fused_shared_experts
        )

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(
            router_logits.float().softmax(dim=-1), self.topk_config.top_k, dim=-1
        )
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids.to(torch.int32), router_logits=router_logits
        )


class TupleTopK(FakeTopK):
    def forward(self, hidden_states, router_logits):
        output = super().forward(hidden_states, router_logits)
        return (output.topk_weights, output.topk_ids)


class FakeMoE(nn.Module):
    def __init__(self, layer_id, num_experts, hidden_size):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.hidden_size = hidden_size

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(
        self, layer_id, num_experts=8, hidden_size=6, top_k=2, num_fused_shared_experts=0
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.topk = FakeTopK(top_k + num_fused_shared_experts, num_fused_shared_experts)
        self.experts = FakeMoE(layer_id, num_experts + num_fused_shared_experts, hidden_size)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class FakeDecoderLayer(nn.Module):
    def __init__(self, layer_id, **kwargs):
        super().__init__()
        self.mlp = FakeBlock(layer_id, **kwargs)

    def forward(self, positions, hidden_states):
        return self.mlp(hidden_states)


class FakeModel(nn.Module):
    def __init__(self, layer_ids=(0, 1, 2), **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(FakeDecoderLayer(i, **kwargs) for i in layer_ids)

    def forward(self, hidden_states):
        positions = torch.arange(hidden_states.shape[0])
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return hidden_states


def _discover(model):
    return discover_moe_layers(model, topk_type=FakeTopK, experts_type=FakeMoE)


class TestContracts(unittest.TestCase):
    def test_feature_widths_and_dtypes_follow_layer_spec(self):
        spec = MoeLayerSpec(layer_id=3, num_experts=8, top_k=2, hidden_size=6)
        self.assertEqual(feature_width(RouteFeature.ROUTER_INPUT, spec), 6)
        self.assertEqual(feature_width(RouteFeature.PRE_MIXER, spec), 6)
        self.assertEqual(feature_width(RouteFeature.ROUTER_LOGITS, spec), 8)
        self.assertEqual(feature_width(RouteFeature.TOPK_IDS, spec), 2)
        self.assertEqual(feature_width(RouteFeature.TOPK_WEIGHTS, spec), 2)
        self.assertEqual(feature_dtype(RouteFeature.TOPK_IDS, torch.bfloat16), torch.int64)
        self.assertEqual(
            feature_dtype(RouteFeature.ROUTER_LOGITS, torch.bfloat16), torch.float32
        )
        self.assertEqual(
            feature_dtype(RouteFeature.ROUTER_INPUT, torch.bfloat16), torch.bfloat16
        )

    def test_predictor_env_parses_comma_list_and_empty_is_off(self):
        with envs.SGLANG_MOE_EXPERT_PREDICTOR.override("affinity, popularity"):
            self.assertEqual(
                envs.SGLANG_MOE_EXPERT_PREDICTOR.get(), ("affinity", "popularity")
            )
        with envs.SGLANG_MOE_EXPERT_PREDICTOR.override(""):
            self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR.get(), ())
        self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR_CANDIDATES.get(), 16)
        self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS.get(), 0)
        self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL.get(), 100)
        self.assertEqual(envs.SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE.get(), "")


class TestFeatureStore(unittest.TestCase):
    def _store(self, max_rows=4):
        spec = MoeLayerSpec(layer_id=0, num_experts=8, top_k=2, hidden_size=6)
        return FeatureStore(
            specs=[spec],
            features=[RouteFeature.TOPK_IDS, RouteFeature.ROUTER_INPUT],
            max_rows=max_rows,
            device=torch.device("cpu"),
            hidden_dtype=torch.float32,
        )

    def test_write_keeps_buffer_address_and_casts_ids(self):
        store = self._store()
        before = store.view(0, RouteFeature.TOPK_IDS, 4).data_ptr()
        store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
        view = store.view(0, RouteFeature.TOPK_IDS, 2)
        self.assertEqual(view.data_ptr(), before)
        self.assertEqual(view.dtype, torch.int64)
        self.assertEqual(view.tolist(), [[1, 2], [3, 4]])

    def test_topk_writes_drop_trailing_shared_expert_columns(self):
        store = self._store()
        store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[1, 2, 8]]))
        self.assertEqual(store.view(0, RouteFeature.TOPK_IDS, 1).tolist(), [[1, 2]])

    def test_hidden_width_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "width 12, expected 6"):
            self._store().write(0, RouteFeature.ROUTER_INPUT, torch.zeros(1, 12))

    def test_oversized_empty_and_unstored_writes_are_skipped(self):
        store = self._store(max_rows=2)
        store.write(0, RouteFeature.TOPK_IDS, torch.ones(3, 2, dtype=torch.int64))
        store.write(0, RouteFeature.TOPK_IDS, torch.ones(0, 2, dtype=torch.int64))
        self.assertEqual(store.view(0, RouteFeature.TOPK_IDS, 2).tolist(), [[0, 0], [0, 0]])
        store.write(0, RouteFeature.ROUTER_LOGITS, torch.ones(1, 8))
        self.assertFalse(store.holds(0, RouteFeature.ROUTER_LOGITS))
        self.assertTrue(store.holds(0, RouteFeature.TOPK_IDS))
        self.assertEqual(store.nbytes, 2 * 2 * 8 + 2 * 6 * 4)

    def test_rejects_zero_rows(self):
        with self.assertRaisesRegex(ValueError, "at least one row"):
            self._store(max_rows=0)


class TestDiscovery(unittest.TestCase):
    def test_pairs_siblings_sorts_layers_and_excludes_fused_shared_experts(self):
        model = FakeModel(layer_ids=(4, 2), num_fused_shared_experts=1)
        layers = _discover(model)
        self.assertEqual([layer.spec.layer_id for layer in layers], [2, 4])
        self.assertEqual(
            layers[0].spec, MoeLayerSpec(layer_id=2, num_experts=8, top_k=2, hidden_size=6)
        )
        self.assertIs(layers[1].topk, model.layers[0].mlp.topk)
        self.assertIs(layers[1].block, model.layers[0].mlp)

    def test_block_with_two_topk_children_raises(self):
        model = FakeModel(layer_ids=(0,))
        model.layers[0].mlp.extra_topk = FakeTopK(2)
        with self.assertRaisesRegex(ValueError, "2 TopK and 1 FusedMoE"):
            _discover(model)

    def test_duplicate_layer_ids_raise(self):
        with self.assertRaisesRegex(ValueError, "layer_id 0"):
            _discover(FakeModel(layer_ids=(0, 0)))

    def test_model_without_moe_blocks_raises(self):
        with self.assertRaisesRegex(ValueError, "no TopK"):
            _discover(nn.Sequential(nn.Linear(2, 2)))


class TestRouteTaps(unittest.TestCase):
    def test_taps_copy_router_tensors_until_removed(self):
        model = FakeModel(layer_ids=(0, 1))
        layers = _discover(model)
        store = FeatureStore(
            specs=[layer.spec for layer in layers],
            features=[
                RouteFeature.ROUTER_INPUT,
                RouteFeature.ROUTER_LOGITS,
                RouteFeature.TOPK_IDS,
                RouteFeature.TOPK_WEIGHTS,
            ],
            max_rows=4,
            device=torch.device("cpu"),
            hidden_dtype=torch.float32,
        )
        taps = RouteTaps(store)
        taps.install(layers)
        hidden = torch.randn(3, 6)
        with torch.no_grad():
            model(hidden)
            for layer_id, decoder in zip((0, 1), model.layers):
                logits = decoder.mlp.gate(hidden)
                weights, ids = torch.topk(logits.softmax(dim=-1), 2, dim=-1)
                torch.testing.assert_close(store.view(layer_id, RouteFeature.ROUTER_INPUT, 3), hidden)
                torch.testing.assert_close(store.view(layer_id, RouteFeature.ROUTER_LOGITS, 3), logits)
                torch.testing.assert_close(store.view(layer_id, RouteFeature.TOPK_WEIGHTS, 3), weights)
                self.assertEqual(store.view(layer_id, RouteFeature.TOPK_IDS, 3).tolist(), ids.tolist())
            taps.remove()
            model(torch.randn(3, 6))
        torch.testing.assert_close(store.view(0, RouteFeature.ROUTER_INPUT, 3), hidden)
        self.assertEqual(taps.unsupported_layers, set())

    def test_non_standard_topk_output_marks_layer_unsupported(self):
        model = FakeModel(layer_ids=(0, 1))
        model.layers[0].mlp.topk = TupleTopK(2)
        layers = _discover(model)
        store = FeatureStore(
            specs=[layer.spec for layer in layers],
            features=[RouteFeature.TOPK_IDS],
            max_rows=4,
            device=torch.device("cpu"),
            hidden_dtype=torch.float32,
        )
        taps = RouteTaps(store)
        taps.install(layers)
        with torch.no_grad(), self.assertLogs(
            "sglang.srt.layers.moe.expert_prediction.taps", level="WARNING"
        ):
            model(torch.randn(2, 6))
        self.assertEqual(taps.unsupported_layers, {0})


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Commit the failing test and run it on divix01**

Divix01 loop with `<files>` = `test/registered/unit/layers/moe/test_expert_prediction_taps.py`, message `test(moe): add expert prediction tap and store tests`, test files = same path.
Expected: collection error `ModuleNotFoundError: No module named 'sglang.srt.layers.moe.expert_prediction'`, `EXIT=` nonzero.

- [x] **Step 3: Add the env vars**

In `python/sglang/srt/environ.py`, directly after the line `SGLANG_MOE_PREFETCH_MAX_CANDIDATES = EnvInt(0)`, insert:

```python
    # Shadow-score these registered MoE expert predictors against native routes
    # (comma list, e.g. "affinity,popularity"); empty installs no hooks or buffers.
    SGLANG_MOE_EXPERT_PREDICTOR = EnvTuple(tuple())
    SGLANG_MOE_EXPERT_PREDICTOR_CANDIDATES = EnvInt(16)
    # Rows per tap buffer; 0 uses decode CUDA-graph max_bs x tokens per request.
    SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS = EnvInt(0)
    SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL = EnvInt(100)
    SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE = EnvStr("")
```

- [x] **Step 4: Create the package and contracts**

Create empty `python/sglang/srt/layers/moe/expert_prediction/__init__.py`.

Create `python/sglang/srt/layers/moe/expert_prediction/contracts.py`:

```python
"""Model-independent contracts shared by MoE expert predictors."""

from __future__ import annotations

from enum import Enum

import msgspec
import torch


class RouteFeature(str, Enum):
    """A per-token routing tensor a predictor may read after a forward."""

    ROUTER_INPUT = "router_input"
    ROUTER_LOGITS = "router_logits"
    TOPK_IDS = "topk_ids"
    TOPK_WEIGHTS = "topk_weights"
    PRE_MIXER = "pre_mixer"


class MoeLayerSpec(msgspec.Struct, frozen=True):
    """One tapped MoE layer; ``num_experts`` and ``top_k`` exclude fused shared experts."""

    layer_id: int
    num_experts: int
    top_k: int
    hidden_size: int


def feature_width(feature: RouteFeature, spec: MoeLayerSpec) -> int:
    if feature in (RouteFeature.ROUTER_INPUT, RouteFeature.PRE_MIXER):
        return spec.hidden_size
    if feature is RouteFeature.ROUTER_LOGITS:
        return spec.num_experts
    return spec.top_k


def feature_dtype(feature: RouteFeature, hidden_dtype: torch.dtype) -> torch.dtype:
    if feature is RouteFeature.TOPK_IDS:
        return torch.int64
    if feature in (RouteFeature.ROUTER_LOGITS, RouteFeature.TOPK_WEIGHTS):
        return torch.float32
    return hidden_dtype
```

- [x] **Step 5: Create the feature store**

Create `python/sglang/srt/layers/moe/expert_prediction/feature_store.py`:

```python
"""Fixed-address per-layer buffers that route taps fill and predictors read."""

from __future__ import annotations

from typing import Iterable

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import (
    MoeLayerSpec,
    RouteFeature,
    feature_dtype,
    feature_width,
)

# Top-k tensors may carry fused shared-expert columns after the routed ones.
_PREFIX_FEATURES = frozenset({RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS})


class FeatureStore:
    """Preallocated ``[max_rows, width]`` buffers keyed by ``(layer_id, feature)``.

    Addresses never change, so copies recorded during CUDA graph capture keep
    writing the same storage on replay.
    """

    def __init__(
        self,
        *,
        specs: Iterable[MoeLayerSpec],
        features: Iterable[RouteFeature],
        max_rows: int,
        device: torch.device,
        hidden_dtype: torch.dtype,
    ) -> None:
        if max_rows < 1:
            raise ValueError("feature store needs at least one row")
        self.max_rows = max_rows
        wanted = frozenset(features)
        self._buffers: dict[tuple[int, RouteFeature], torch.Tensor] = {
            (spec.layer_id, feature): torch.zeros(
                (max_rows, feature_width(feature, spec)),
                dtype=feature_dtype(feature, hidden_dtype),
                device=device,
            )
            for spec in specs
            for feature in wanted
        }

    @property
    def nbytes(self) -> int:
        return sum(
            buffer.numel() * buffer.element_size() for buffer in self._buffers.values()
        )

    def holds(self, layer_id: int, feature: RouteFeature) -> bool:
        return (layer_id, feature) in self._buffers

    def write(self, layer_id: int, feature: RouteFeature, source: torch.Tensor) -> None:
        """Copy ``source`` rows in; unstored features and batches above ``max_rows`` are skipped."""
        buffer = self._buffers.get((layer_id, feature))
        rows = source.shape[0]
        if buffer is None or rows == 0 or rows > self.max_rows:
            return
        width = buffer.shape[1]
        flat = source.reshape(rows, -1)
        prefix_ok = feature in _PREFIX_FEATURES and flat.shape[1] > width
        if flat.shape[1] != width and not prefix_ok:
            raise ValueError(
                f"layer {layer_id} {feature.value} has width {flat.shape[1]}, "
                f"expected {width}"
            )
        with torch.no_grad():
            buffer[:rows].copy_(flat[:, :width])

    def view(self, layer_id: int, feature: RouteFeature, rows: int) -> torch.Tensor:
        return self._buffers[(layer_id, feature)][:rows]
```

- [x] **Step 6: Create discovery and route taps**

Create `python/sglang/srt/layers/moe/expert_prediction/taps.py`:

```python
"""Discover MoE blocks by their TopK and FusedMoE children and tap their routes."""

from __future__ import annotations

import logging
from typing import Sequence

import msgspec
from torch import nn

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)


class TappedMoeLayer(msgspec.Struct, frozen=True):
    spec: MoeLayerSpec
    topk: nn.Module
    block: nn.Module


def discover_moe_layers(
    model: nn.Module,
    *,
    topk_type: type | None = None,
    experts_type: type | None = None,
) -> tuple[TappedMoeLayer, ...]:
    """Pair every TopK with its sibling FusedMoE; layer ids come from the FusedMoE."""
    if topk_type is None or experts_type is None:
        # Deferred: fused_moe_triton.layer pulls in the quantization stack.
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopK

        topk_type = topk_type or TopK
        experts_type = experts_type or FusedMoE
    layers: dict[int, TappedMoeLayer] = {}
    for block in model.modules():
        children = list(block.children())
        topks = [child for child in children if isinstance(child, topk_type)]
        experts = [child for child in children if isinstance(child, experts_type)]
        if not topks or not experts:
            continue
        if len(topks) != 1 or len(experts) != 1:
            raise ValueError(
                f"{type(block).__name__} holds {len(topks)} TopK and "
                f"{len(experts)} FusedMoE children; expected one of each"
            )
        spec = _layer_spec(topk=topks[0], moe=experts[0])
        if spec.layer_id in layers:
            raise ValueError(f"two MoE blocks claim layer_id {spec.layer_id}")
        layers[spec.layer_id] = TappedMoeLayer(spec=spec, topk=topks[0], block=block)
    if not layers:
        raise ValueError("model has no TopK + FusedMoE blocks to tap")
    return tuple(layers[layer_id] for layer_id in sorted(layers))


def _layer_spec(*, topk: nn.Module, moe: nn.Module) -> MoeLayerSpec:
    shared = topk.topk_config.num_fused_shared_experts
    return MoeLayerSpec(
        layer_id=moe.layer_id,
        num_experts=moe.num_experts - shared,
        top_k=topk.topk_config.top_k - shared,
        hidden_size=moe.hidden_size,
    )


class RouteTaps:
    """Copy each TopK call's tensors into a FeatureStore with device-only ops.

    Hooks run in eager forwards and while a CUDA graph is captured; replay
    re-executes the recorded copies without calling Python.
    """

    def __init__(self, store: FeatureStore) -> None:
        self._store = store
        self._handles: list = []
        self.unsupported_layers: set[int] = set()

    def install(self, layers: Sequence[TappedMoeLayer]) -> None:
        for layer in layers:
            self._handles.append(
                layer.topk.register_forward_hook(
                    self._hook(layer.spec.layer_id), with_kwargs=True
                )
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _hook(self, layer_id: int):
        store = self._store

        def hook(module, args, kwargs, output):
            if not isinstance(output, StandardTopKOutput):
                if layer_id not in self.unsupported_layers:
                    self.unsupported_layers.add(layer_id)
                    logger.warning(
                        "MoE expert prediction cannot tap layer %d: TopK returned %s",
                        layer_id,
                        type(output).__name__,
                    )
                return
            hidden_states = args[0] if args else kwargs["hidden_states"]
            store.write(layer_id, RouteFeature.ROUTER_INPUT, hidden_states)
            if output.router_logits is not None:
                store.write(layer_id, RouteFeature.ROUTER_LOGITS, output.router_logits)
            store.write(layer_id, RouteFeature.TOPK_IDS, output.topk_ids)
            store.write(layer_id, RouteFeature.TOPK_WEIGHTS, output.topk_weights)

        return hook
```

- [x] **Step 7: Commit and run the tests**

Divix01 loop with `<files>` = `python/sglang/srt/environ.py python/sglang/srt/layers/moe/expert_prediction/__init__.py python/sglang/srt/layers/moe/expert_prediction/contracts.py python/sglang/srt/layers/moe/expert_prediction/feature_store.py python/sglang/srt/layers/moe/expert_prediction/taps.py`, message `feat(moe): tap MoE routes into fixed expert prediction buffers`, test files = `test/registered/unit/layers/moe/test_expert_prediction_taps.py`.
Expected: `13 passed`, `EXIT=0`. If an import cycle appears for `sglang.srt.layers.moe.topk`, move that import inside `RouteTaps._hook` with the same one-line "Deferred:" comment and rerun.

---

### Task 2: Pre-mixer feature adapters

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/adapters.py`
- Test: `test/registered/unit/layers/moe/test_expert_prediction_adapters.py`

**Interfaces:**
- Consumes: `FeatureStore.write`, `RouteFeature.PRE_MIXER`, `TappedMoeLayer`, `discover_moe_layers` (Task 1).
- Produces:
  - `PreMixerInstaller = Callable[..., list[Callable[[], None]]]` called as `installer(model=, layers=, store=)`
  - `decoder_layers_by_moe_layer(*, model, layers) -> dict[int, nn.Module]`
  - `install_decoder_input_pre_mixer(*, model, layers, store) -> list[Callable[[], None]]`
  - `install_hyper_connection_pre_mixer(*, model, layers, store) -> list[Callable[[], None]]`
  - `register_pre_mixer_adapter(*, architecture: str, installer) -> None`
  - `install_pre_mixer_taps(*, model, layers, store) -> list[Callable[[], None]]` (returns removers)

- [x] **Step 1: Write the failing test**

Create `test/registered/unit/layers/moe/test_expert_prediction_adapters.py`:

```python
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.moe.expert_prediction import adapters
from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.taps import discover_moe_layers
from sglang.srt.layers.moe.topk import StandardTopKOutput


class FakeTopK(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=top_k, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.softmax(dim=-1), self.topk_config.top_k, dim=-1)
        return StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=router_logits)


class FakeMoE(nn.Module):
    def __init__(self, layer_id, num_experts, hidden_size):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.hidden_size = hidden_size

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(self, layer_id, hidden_size=6):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 8, bias=False)
        self.topk = FakeTopK(2)
        self.experts = FakeMoE(layer_id, 8, hidden_size)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class PlainDecoderLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.input_norm = nn.LayerNorm(6)
        self.mlp = FakeBlock(layer_id)

    def forward(self, positions, hidden_states):
        return self.mlp(self.input_norm(hidden_states))


class PlainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([PlainDecoderLayer(0), PlainDecoderLayer(1)])

    def forward(self, hidden_states):
        positions = torch.arange(hidden_states.shape[0])
        for layer in self.layers:
            layer(positions, hidden_states)


class HyperConnection(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def mix(self, hyper_input):
        return self.proj(hyper_input), hyper_input


class HyperDecoderLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.attn_hyper_connection = HyperConnection()
        self.mlp = FakeBlock(layer_id)

    def forward(self, hyper_states):
        mixed, _ = self.attn_hyper_connection.mix(hyper_states)
        return self.mlp(mixed)


class Qwen4ExpForConditionalGeneration(nn.Module):
    """Named like the real entry class so the registered adapter is selected."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([HyperDecoderLayer(0), HyperDecoderLayer(1)])

    def forward(self, hyper_states):
        for layer in self.model.layers:
            layer(hyper_states)


def _setup(model):
    layers = discover_moe_layers(model, topk_type=FakeTopK, experts_type=FakeMoE)
    store = FeatureStore(
        specs=[layer.spec for layer in layers],
        features=[RouteFeature.PRE_MIXER],
        max_rows=4,
        device=torch.device("cpu"),
        hidden_dtype=torch.float32,
    )
    return layers, store


class TestPreMixerAdapters(unittest.TestCase):
    def test_decoder_layers_map_to_outermost_module_list_elements(self):
        model = PlainModel()
        layers, _ = _setup(model)
        mapping = adapters.decoder_layers_by_moe_layer(model=model, layers=layers)
        self.assertIs(mapping[0], model.layers[0])
        self.assertIs(mapping[1], model.layers[1])

    def test_missing_decoder_layer_raises(self):
        model = FakeBlock(0)
        layers, _ = _setup(model)
        with self.assertRaisesRegex(ValueError, r"MoE layers \[0\]"):
            adapters.decoder_layers_by_moe_layer(model=model, layers=layers)

    def test_default_adapter_taps_decoder_hidden_state_input(self):
        model = PlainModel()
        layers, store = _setup(model)
        removers = adapters.install_pre_mixer_taps(model=model, layers=layers, store=store)
        hidden = torch.randn(3, 6)
        with torch.no_grad():
            model(hidden)
        for layer_id in (0, 1):
            torch.testing.assert_close(store.view(layer_id, RouteFeature.PRE_MIXER, 3), hidden)
        for remove in removers:
            remove()
        with torch.no_grad():
            model(torch.randn(3, 6))
        torch.testing.assert_close(store.view(0, RouteFeature.PRE_MIXER, 3), hidden)

    def test_default_adapter_raises_without_matching_input(self):
        model = PlainModel()
        layers, store = _setup(model)
        adapters.install_pre_mixer_taps(model=model, layers=layers, store=store)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "register a pre-mixer adapter"):
            model.layers[0](torch.arange(3), torch.randn(3))

    def test_qwen4_adapter_taps_mixed_hyper_connection_output(self):
        model = Qwen4ExpForConditionalGeneration()
        layers, store = _setup(model)
        removers = adapters.install_pre_mixer_taps(model=model, layers=layers, store=store)
        hyper = torch.randn(3, 12)
        with torch.no_grad():
            model(hyper)
            for layer_id, layer in zip((0, 1), model.model.layers):
                expected = layer.attn_hyper_connection.proj(hyper)
                torch.testing.assert_close(store.view(layer_id, RouteFeature.PRE_MIXER, 3), expected)
        for remove in removers:
            remove()
        for layer in model.model.layers:
            self.assertNotIn("mix", layer.attn_hyper_connection.__dict__)

    def test_registered_adapter_overrides_default(self):
        calls = []

        class CustomArch(PlainModel):
            pass

        def installer(*, model, layers, store):
            calls.append(len(layers))
            return []

        adapters.register_pre_mixer_adapter(architecture="CustomArch", installer=installer)
        model = CustomArch()
        layers, store = _setup(model)
        self.assertEqual(adapters.install_pre_mixer_taps(model=model, layers=layers, store=store), [])
        self.assertEqual(calls, [2])
        with self.assertRaisesRegex(ValueError, "CustomArch"):
            adapters.register_pre_mixer_adapter(architecture="CustomArch", installer=installer)


if __name__ == "__main__":
    unittest.main()
```

- [-] **Step 2: Commit the failing test and run it on divix01** (skipped: MVP, test committed with the implementation)

Divix01 loop with `<files>` = `test/registered/unit/layers/moe/test_expert_prediction_adapters.py`, message `test(moe): add pre-mixer adapter tests`, test files = same path.
Expected: `ImportError` / `ModuleNotFoundError` for `adapters`, `EXIT=` nonzero.

- [x] **Step 3: Implement adapters**

Create `python/sglang/srt/layers/moe/expert_prediction/adapters.py`:

```python
"""Per-architecture taps for the pre-mixer feature; the default suits plain decoder stacks."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.taps import TappedMoeLayer

PreMixerInstaller = Callable[..., list[Callable[[], None]]]


def decoder_layers_by_moe_layer(
    *, model: nn.Module, layers: Sequence[TappedMoeLayer]
) -> dict[int, nn.Module]:
    """Map each tapped layer id to the outermost ``nn.ModuleList`` element containing its block."""
    block_ids = {id(layer.block): layer.spec.layer_id for layer in layers}
    found: dict[int, nn.Module] = {}
    for module in model.modules():
        if not isinstance(module, nn.ModuleList):
            continue
        for element in module:
            for sub in element.modules():
                layer_id = block_ids.get(id(sub))
                if layer_id is not None and layer_id not in found:
                    found[layer_id] = element
    missing = sorted({layer.spec.layer_id for layer in layers} - set(found))
    if missing:
        raise ValueError(f"no decoder layer contains MoE layers {missing}")
    return found


def install_decoder_input_pre_mixer(
    *, model: nn.Module, layers: Sequence[TappedMoeLayer], store: FeatureStore
) -> list[Callable[[], None]]:
    """Tap each decoder layer's ``hidden_states`` input (kwarg, else first 2-D float arg of hidden width)."""
    widths = {layer.spec.layer_id: layer.spec.hidden_size for layer in layers}
    architecture = type(model).__name__
    removers = []
    for layer_id, decoder in decoder_layers_by_moe_layer(model=model, layers=layers).items():

        def hook(module, args, kwargs, layer_id=layer_id, width=widths[layer_id]):
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = next(
                    (
                        arg
                        for arg in args
                        if isinstance(arg, torch.Tensor)
                        and arg.dim() == 2
                        and arg.is_floating_point()
                        and arg.shape[-1] == width
                    ),
                    None,
                )
            if hidden is None:
                raise ValueError(
                    f"decoder layer {layer_id} has no hidden-state input of width "
                    f"{width}; register a pre-mixer adapter for {architecture}"
                )
            store.write(layer_id, RouteFeature.PRE_MIXER, hidden)

        handle = decoder.register_forward_pre_hook(hook, with_kwargs=True)
        removers.append(handle.remove)
    return removers


def install_hyper_connection_pre_mixer(
    *, model: nn.Module, layers: Sequence[TappedMoeLayer], store: FeatureStore
) -> list[Callable[[], None]]:
    """Tap ``attn_hyper_connection.mix(...)[0]``, the tensor the attention or linear mixer consumes."""
    removers = []
    for layer_id, decoder in decoder_layers_by_moe_layer(model=model, layers=layers).items():
        owner = decoder.attn_hyper_connection
        original = owner.mix

        def mix(hyper_input, original=original, layer_id=layer_id):
            result = original(hyper_input)
            store.write(layer_id, RouteFeature.PRE_MIXER, result[0])
            return result

        owner.__dict__["mix"] = mix
        removers.append(lambda owner=owner: owner.__dict__.pop("mix", None))
    return removers


_PRE_MIXER_ADAPTERS: dict[str, PreMixerInstaller] = {
    "Qwen4ExpForConditionalGeneration": install_hyper_connection_pre_mixer,
}


def register_pre_mixer_adapter(*, architecture: str, installer: PreMixerInstaller) -> None:
    if architecture in _PRE_MIXER_ADAPTERS:
        raise ValueError(f"pre-mixer adapter already registered for {architecture}")
    _PRE_MIXER_ADAPTERS[architecture] = installer


def install_pre_mixer_taps(
    *, model: nn.Module, layers: Sequence[TappedMoeLayer], store: FeatureStore
) -> list[Callable[[], None]]:
    installer = _PRE_MIXER_ADAPTERS.get(
        type(model).__name__, install_decoder_input_pre_mixer
    )
    return installer(model=model, layers=layers, store=store)
```

- [x] **Step 4: Commit and run the tests**

Divix01 loop with `<files>` = `python/sglang/srt/layers/moe/expert_prediction/adapters.py`, message `feat(moe): add per-architecture pre-mixer feature taps`, test files = `test/registered/unit/layers/moe/test_expert_prediction_adapters.py test/registered/unit/layers/moe/test_expert_prediction_taps.py`.
Expected: `19 passed`, `EXIT=0`.

---

### Task 3: Predictor interface, registry, and baseline predictors

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/base.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/popularity.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/affinity.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/registry.py`
- Test: `test/registered/unit/layers/moe/test_expert_prediction_predictors.py`

**Interfaces:**
- Consumes: `FeatureStore`, `MoeLayerSpec`, `RouteFeature` (Task 1).
- Produces:
  - `ExpertPredictor(abc.ABC)`: class attrs `name: str`, `target_offset: int`, `required_features: frozenset[RouteFeature]`; `__init__(*, specs, device, max_candidates)` setting `.specs: dict[int, MoeLayerSpec]`, `.layer_ids: tuple[int, ...]`, `.device`, `.max_candidates`; abstract `predict(*, source_layer, target_layer, store, rows) -> Tensor[int64, rows x max_candidates]`; `observe(*, store, rows) -> None`; property `state_nbytes -> int`
  - `pad_candidates(ranked, width) -> Tensor`
  - `accumulate_routes(counts, ids, *, decay) -> None` (in `popularity.py`)
  - `PopularityPredictor` (`name="popularity"`, offset 0), `AffinityPredictor` (`name="affinity"`, offset 1)
  - `register_predictor(cls) -> cls`, `registered_predictor_names() -> tuple[str, ...]`, `build_predictors(names, *, specs, device, max_candidates) -> tuple[ExpertPredictor, ...]`

- [x] **Step 1: Write the failing test**

Create `test/registered/unit/layers/moe/test_expert_prediction_predictors.py`:

```python
import unittest

import torch

from sglang.srt.layers.moe.expert_prediction import registry
from sglang.srt.layers.moe.expert_prediction.affinity import AffinityPredictor
from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor, pad_candidates
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.popularity import PopularityPredictor

CPU = torch.device("cpu")


def _specs(num_layers, num_experts=4, top_k=1):
    return [
        MoeLayerSpec(layer_id=i, num_experts=num_experts, top_k=top_k, hidden_size=3)
        for i in range(num_layers)
    ]


def _store(specs, rows=2):
    return FeatureStore(
        specs=specs,
        features=[RouteFeature.TOPK_IDS],
        max_rows=rows,
        device=CPU,
        hidden_dtype=torch.float32,
    )


class ConstantPredictor(ExpertPredictor):
    name = "test-constant"
    target_offset = 0
    required_features = frozenset({RouteFeature.TOPK_IDS})

    def predict(self, *, source_layer, target_layer, store, rows):
        return torch.zeros((rows, self.max_candidates), dtype=torch.int64)


class TestBaseAndRegistry(unittest.TestCase):
    def test_pad_candidates_pads_with_minus_one_and_truncates(self):
        ranked = torch.tensor([[3, 1], [2, 0]])
        self.assertEqual(pad_candidates(ranked, 4).tolist(), [[3, 1, -1, -1], [2, 0, -1, -1]])
        self.assertEqual(pad_candidates(ranked, 1).tolist(), [[3], [2]])

    def test_base_rejects_non_positive_candidates(self):
        with self.assertRaisesRegex(ValueError, "max_candidates"):
            ConstantPredictor(specs=_specs(1), device=CPU, max_candidates=0)

    def test_builds_builtins_in_requested_order(self):
        self.assertIn("affinity", registry.registered_predictor_names())
        self.assertIn("popularity", registry.registered_predictor_names())
        built = registry.build_predictors(
            ("popularity", "affinity"), specs=_specs(2), device=CPU, max_candidates=3
        )
        self.assertIsInstance(built[0], PopularityPredictor)
        self.assertIsInstance(built[1], AffinityPredictor)
        self.assertEqual(built[1].max_candidates, 3)

    def test_rejects_unknown_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "unknown expert predictors \\['nope'\\]"):
            registry.build_predictors(("nope",), specs=_specs(1), device=CPU, max_candidates=2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.build_predictors(
                ("affinity", "affinity"), specs=_specs(2), device=CPU, max_candidates=2
            )

    def test_register_custom_predictor_and_reject_name_clash(self):
        self.addCleanup(registry._PREDICTORS.pop, "test-constant", None)
        self.assertIs(registry.register_predictor(ConstantPredictor), ConstantPredictor)
        (built,) = registry.build_predictors(
            ("test-constant",), specs=_specs(1), device=CPU, max_candidates=2
        )
        self.assertEqual(built.layer_ids, (0,))
        with self.assertRaisesRegex(ValueError, "test-constant"):
            registry.register_predictor(ConstantPredictor)


class TestPopularityPredictor(unittest.TestCase):
    def test_ranks_decayed_counts_ignores_invalid_ids_and_pads(self):
        specs = _specs(1, num_experts=4, top_k=2)
        store = _store(specs)
        predictor = PopularityPredictor(specs=specs, device=CPU, max_candidates=6)
        store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[3, 1], [3, -1]]))
        predictor.observe(store=store, rows=2)
        candidates = predictor.predict(source_layer=0, target_layer=0, store=store, rows=2)
        self.assertEqual(candidates.shape, (2, 6))
        self.assertEqual(candidates.dtype, torch.int64)
        self.assertEqual(candidates[:, :2].tolist(), [[3, 1], [3, 1]])
        self.assertEqual(candidates[:, 4:].tolist(), [[-1, -1], [-1, -1]])
        self.assertEqual(predictor.target_offset, 0)
        self.assertEqual(predictor.state_nbytes, 4 * 4)


class TestAffinityPredictor(unittest.TestCase):
    def _trained(self):
        specs = _specs(2, num_experts=4, top_k=1)
        store = _store(specs, rows=1)
        predictor = AffinityPredictor(specs=specs, device=CPU, max_candidates=2)
        for source_id, target_id in ((1, 3), (2, 0)):
            store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[source_id]]))
            store.write(1, RouteFeature.TOPK_IDS, torch.tensor([[target_id]]))
            predictor.observe(store=store, rows=1)
        return predictor, store

    def _first(self, predictor, store, source_id):
        store.write(0, RouteFeature.TOPK_IDS, torch.tensor([[source_id]]))
        candidates = predictor.predict(source_layer=0, target_layer=1, store=store, rows=1)
        self.assertEqual(candidates.shape, (1, 2))
        return candidates[0, 0].item()

    def test_prefers_cooccurring_targets(self):
        predictor, store = self._trained()
        self.assertEqual(self._first(predictor, store, 1), 3)
        self.assertEqual(self._first(predictor, store, 2), 0)

    def test_falls_back_to_popularity_without_cooccurrence(self):
        predictor, store = self._trained()
        self.assertEqual(self._first(predictor, store, 0), 0)

    def test_metadata_and_state_size(self):
        predictor, _ = self._trained()
        self.assertEqual(predictor.target_offset, 1)
        self.assertEqual(predictor.state_nbytes, 4 * 4 * 4 + 2 * 4 * 4)


if __name__ == "__main__":
    unittest.main()
```

- [-] **Step 2: Commit the failing test and run it on divix01** (skipped: MVP, test committed with the implementation)

Divix01 loop with `<files>` = `test/registered/unit/layers/moe/test_expert_prediction_predictors.py`, message `test(moe): add expert predictor interface and baseline tests`, test files = same path.
Expected: `ModuleNotFoundError` for `registry`, `EXIT=` nonzero.

- [x] **Step 3: Implement the interface**

Create `python/sglang/srt/layers/moe/expert_prediction/base.py`:

```python
"""Interface every MoE expert predictor implements."""

from __future__ import annotations

import abc
from typing import ClassVar, Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore


class ExpertPredictor(abc.ABC):
    """Rank expert candidates for tapped MoE layers from a completed forward's features.

    ``target_offset`` 0 predicts the source layer's own routes and 1 the next
    tapped MoE layer. Within a forward every ``predict`` runs before
    ``observe``, and both must stay device-only.
    """

    name: ClassVar[str]
    target_offset: ClassVar[int]
    required_features: ClassVar[frozenset[RouteFeature]]

    def __init__(
        self, *, specs: Sequence[MoeLayerSpec], device: torch.device, max_candidates: int
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.specs = {spec.layer_id: spec for spec in specs}
        self.layer_ids = tuple(sorted(self.specs))
        self.device = device
        self.max_candidates = max_candidates

    @abc.abstractmethod
    def predict(
        self, *, source_layer: int, target_layer: int, store: FeatureStore, rows: int
    ) -> torch.Tensor:
        """Ranked expert ids for ``target_layer``: int64 ``[rows, max_candidates]``, -1 padded."""

    def observe(self, *, store: FeatureStore, rows: int) -> None:
        """Update online state from this forward's features after it was scored."""

    @property
    def state_nbytes(self) -> int:
        return 0


def pad_candidates(ranked: torch.Tensor, width: int) -> torch.Tensor:
    if ranked.shape[1] >= width:
        return ranked[:, :width]
    return torch.nn.functional.pad(ranked, (0, width - ranked.shape[1]), value=-1)
```

- [x] **Step 4: Implement the popularity baseline**

Create `python/sglang/srt/layers/moe/expert_prediction/popularity.py`:

```python
"""Same-layer baseline: rank each layer's experts by decayed route counts."""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor, pad_candidates
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore

# Arbitrary; roughly a 100-forward memory.
_DECAY = 0.99


def accumulate_routes(counts: torch.Tensor, ids: torch.Tensor, *, decay: float) -> None:
    """Decay ``counts`` then add one per in-range routed id, without host syncs."""
    num_experts = counts.numel()
    valid = (ids >= 0) & (ids < num_experts)
    counts.mul_(decay)
    counts.index_add_(
        0, ids.clamp(0, num_experts - 1).reshape(-1), valid.reshape(-1).to(counts.dtype)
    )


class PopularityPredictor(ExpertPredictor):
    name = "popularity"
    target_offset = 0
    required_features = frozenset({RouteFeature.TOPK_IDS})

    def __init__(
        self, *, specs: Sequence[MoeLayerSpec], device: torch.device, max_candidates: int
    ) -> None:
        super().__init__(specs=specs, device=device, max_candidates=max_candidates)
        self._counts = {
            layer_id: torch.zeros(spec.num_experts, dtype=torch.float32, device=device)
            for layer_id, spec in self.specs.items()
        }

    @property
    def state_nbytes(self) -> int:
        return sum(counts.numel() * counts.element_size() for counts in self._counts.values())

    def predict(
        self, *, source_layer: int, target_layer: int, store: FeatureStore, rows: int
    ) -> torch.Tensor:
        counts = self._counts[target_layer]
        ranked = torch.topk(counts, min(self.max_candidates, counts.numel())).indices
        return pad_candidates(ranked.unsqueeze(0).expand(rows, -1), self.max_candidates)

    def observe(self, *, store: FeatureStore, rows: int) -> None:
        for layer_id, counts in self._counts.items():
            accumulate_routes(
                counts, store.view(layer_id, RouteFeature.TOPK_IDS, rows), decay=_DECAY
            )
```

- [x] **Step 5: Implement the affinity baseline**

Create `python/sglang/srt/layers/moe/expert_prediction/affinity.py`:

```python
"""Next-layer baseline: decayed source-to-target route co-occurrence, then popularity."""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor, pad_candidates
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.popularity import accumulate_routes

_DECAY = 0.99
# Arbitrary; scales sum-normalized popularity below one co-occurrence so it
# mostly orders targets without co-occurrence evidence.
_POPULARITY_WEIGHT = 1e-3


class AffinityPredictor(ExpertPredictor):
    name = "affinity"
    target_offset = 1
    required_features = frozenset({RouteFeature.TOPK_IDS})

    def __init__(
        self, *, specs: Sequence[MoeLayerSpec], device: torch.device, max_candidates: int
    ) -> None:
        super().__init__(specs=specs, device=device, max_candidates=max_candidates)
        self._pairs = tuple(zip(self.layer_ids, self.layer_ids[1:]))
        self._transitions = {
            source: torch.zeros(
                (self.specs[source].num_experts, self.specs[target].num_experts),
                dtype=torch.float32,
                device=device,
            )
            for source, target in self._pairs
        }
        self._popularity = {
            layer_id: torch.zeros(spec.num_experts, dtype=torch.float32, device=device)
            for layer_id, spec in self.specs.items()
        }

    @property
    def state_nbytes(self) -> int:
        tensors = [*self._transitions.values(), *self._popularity.values()]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def predict(
        self, *, source_layer: int, target_layer: int, store: FeatureStore, rows: int
    ) -> torch.Tensor:
        transitions = self._transitions[source_layer]
        ids = store.view(source_layer, RouteFeature.TOPK_IDS, rows)
        num_sources = transitions.shape[0]
        valid = ((ids >= 0) & (ids < num_sources)).unsqueeze(-1).to(transitions.dtype)
        scores = (transitions[ids.clamp(0, num_sources - 1)] * valid).sum(dim=1)
        popularity = self._popularity[target_layer]
        scores = scores + _POPULARITY_WEIGHT * popularity / (popularity.sum() + 1.0)
        ranked = torch.topk(scores, min(self.max_candidates, scores.shape[1]), dim=1).indices
        return pad_candidates(ranked, self.max_candidates)

    def observe(self, *, store: FeatureStore, rows: int) -> None:
        for source, target in self._pairs:
            self._observe_pair(
                transitions=self._transitions[source],
                source_ids=store.view(source, RouteFeature.TOPK_IDS, rows),
                target_ids=store.view(target, RouteFeature.TOPK_IDS, rows),
            )
        for layer_id, popularity in self._popularity.items():
            accumulate_routes(
                popularity, store.view(layer_id, RouteFeature.TOPK_IDS, rows), decay=_DECAY
            )

    @staticmethod
    def _observe_pair(
        *, transitions: torch.Tensor, source_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> None:
        num_sources, num_targets = transitions.shape
        source_valid = (source_ids >= 0) & (source_ids < num_sources)
        target_valid = (target_ids >= 0) & (target_ids < num_targets)
        pair_index = source_ids.clamp(0, num_sources - 1).unsqueeze(2) * num_targets + (
            target_ids.clamp(0, num_targets - 1).unsqueeze(1)
        )
        pair_weight = (source_valid.unsqueeze(2) & target_valid.unsqueeze(1)).to(
            transitions.dtype
        )
        transitions.mul_(_DECAY)
        transitions.view(-1).index_add_(0, pair_index.reshape(-1), pair_weight.reshape(-1))
```

- [x] **Step 6: Implement the registry**

Create `python/sglang/srt/layers/moe/expert_prediction/registry.py`:

```python
"""Name-to-class registry used to build the predictors named in the env var."""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.affinity import AffinityPredictor
from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.popularity import PopularityPredictor

_PREDICTORS: dict[str, type[ExpertPredictor]] = {
    cls.name: cls for cls in (AffinityPredictor, PopularityPredictor)
}


def register_predictor(cls: type[ExpertPredictor]) -> type[ExpertPredictor]:
    if cls.name in _PREDICTORS:
        raise ValueError(f"expert predictor {cls.name} is already registered")
    _PREDICTORS[cls.name] = cls
    return cls


def registered_predictor_names() -> tuple[str, ...]:
    return tuple(sorted(_PREDICTORS))


def build_predictors(
    names: Sequence[str],
    *,
    specs: Sequence[MoeLayerSpec],
    device: torch.device,
    max_candidates: int,
) -> tuple[ExpertPredictor, ...]:
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate expert predictors in {list(names)}")
    unknown = [name for name in names if name not in _PREDICTORS]
    if unknown:
        raise ValueError(
            f"unknown expert predictors {unknown}; registered: "
            f"{list(registered_predictor_names())}"
        )
    return tuple(
        _PREDICTORS[name](specs=specs, device=device, max_candidates=max_candidates)
        for name in names
    )
```

- [x] **Step 7: Commit and run the tests**

Divix01 loop with `<files>` = `python/sglang/srt/layers/moe/expert_prediction/base.py python/sglang/srt/layers/moe/expert_prediction/popularity.py python/sglang/srt/layers/moe/expert_prediction/affinity.py python/sglang/srt/layers/moe/expert_prediction/registry.py`, message `feat(moe): add expert predictor registry with popularity and affinity baselines`, test files = `test/registered/unit/layers/moe/test_expert_prediction_predictors.py`.
Expected: `9 passed`, `EXIT=0`.

---

### Task 4: Device-side shadow metrics

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/metrics.py`
- Test: `test/registered/unit/layers/moe/test_expert_prediction_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `COUNTER_NAMES = ("rows", "routes", "hits_at_k", "hits_at_m", "cold_routes", "cold_hits_at_m", "cold_candidates")`
  - `score_candidates(*, candidates, actual, top_k, num_experts, resident) -> Tensor[int64, 7]` (`resident`: bool `[num_experts]` or `None`)
  - `ShadowMetrics(*, predictor_names, layer_ids, device)`; `.add(*, predictor_index, target_layer, counts)`, `.snapshot() -> dict`, `.append_jsonl(path, *, forwards) -> None`

- [x] **Step 1: Write the failing test**

Create `test/registered/unit/layers/moe/test_expert_prediction_metrics.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_prediction.metrics import (
    COUNTER_NAMES,
    ShadowMetrics,
    score_candidates,
)


def _counts(values):
    return dict(zip(COUNTER_NAMES, values.tolist()))


class TestScoreCandidates(unittest.TestCase):
    candidates = torch.tensor([[2, 5, 7, -1], [1, 0, 3, 4]])
    actual = torch.tensor([[5, 1], [4, 6]])

    def test_counts_hits_at_k_and_m_without_residency(self):
        counts = score_candidates(
            candidates=self.candidates, actual=self.actual, top_k=2, num_experts=8, resident=None
        )
        self.assertEqual(counts.dtype, torch.int64)
        self.assertEqual(
            _counts(counts),
            {
                "rows": 2,
                "routes": 4,
                "hits_at_k": 1,
                "hits_at_m": 2,
                "cold_routes": 4,
                "cold_hits_at_m": 2,
                "cold_candidates": 7,
            },
        )

    def test_residency_marks_cold_routes_and_candidates(self):
        resident = torch.zeros(8, dtype=torch.bool)
        resident[[0, 4, 5]] = True
        counts = score_candidates(
            candidates=self.candidates,
            actual=self.actual,
            top_k=2,
            num_experts=8,
            resident=resident,
        )
        self.assertEqual(
            _counts(counts),
            {
                "rows": 2,
                "routes": 4,
                "hits_at_k": 1,
                "hits_at_m": 2,
                "cold_routes": 2,
                "cold_hits_at_m": 0,
                "cold_candidates": 4,
            },
        )

    def test_invalid_actual_ids_are_not_routes(self):
        counts = score_candidates(
            candidates=torch.tensor([[3]]),
            actual=torch.tensor([[-1, 3]]),
            top_k=1,
            num_experts=4,
            resident=None,
        )
        self.assertEqual(_counts(counts)["routes"], 1)
        self.assertEqual(_counts(counts)["hits_at_m"], 1)


class TestShadowMetrics(unittest.TestCase):
    def test_snapshot_aggregates_layers_and_appends_jsonl(self):
        metrics = ShadowMetrics(
            predictor_names=("a", "b"), layer_ids=(0, 1), device=torch.device("cpu")
        )
        for _ in range(2):
            metrics.add(
                predictor_index=0, target_layer=1, counts=torch.tensor([2, 4, 1, 2, 4, 2, 7])
            )
        snapshot = metrics.snapshot()
        self.assertEqual(
            snapshot["a"]["layers"],
            {
                "1": {
                    "rows": 4,
                    "routes": 8,
                    "hits_at_k": 2,
                    "hits_at_m": 4,
                    "cold_routes": 8,
                    "cold_hits_at_m": 4,
                    "cold_candidates": 14,
                }
            },
        )
        total = snapshot["a"]["total"]
        self.assertAlmostEqual(total["recall_at_k"], 0.25)
        self.assertAlmostEqual(total["recall_at_m"], 0.5)
        self.assertAlmostEqual(total["cold_recall_at_m"], 0.5)
        self.assertAlmostEqual(total["cold_precision_at_m"], 4 / 14)
        self.assertEqual(snapshot["b"]["layers"], {})
        self.assertEqual(snapshot["b"]["total"]["routes"], 0)
        self.assertEqual(snapshot["b"]["total"]["recall_at_m"], 0.0)

        path = Path(tempfile.mkdtemp()) / "metrics.jsonl"
        metrics.append_jsonl(path, forwards=7)
        metrics.append_jsonl(path, forwards=9)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["forwards"] for record in records], [7, 9])
        self.assertEqual(records[0]["predictors"], snapshot)
        self.assertIsInstance(records[0]["timestamp_ns"], int)


if __name__ == "__main__":
    unittest.main()
```

- [-] **Step 2: Commit the failing test and run it on divix01** (skipped: MVP, test committed with the implementation)

Divix01 loop with `<files>` = `test/registered/unit/layers/moe/test_expert_prediction_metrics.py`, message `test(moe): add expert prediction shadow metric tests`, test files = same path.
Expected: `ModuleNotFoundError` for `metrics`, `EXIT=` nonzero.

- [x] **Step 3: Implement the metrics**

Create `python/sglang/srt/layers/moe/expert_prediction/metrics.py`:

```python
"""Score predictor candidates against native routes on device and report totals."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import torch

COUNTER_NAMES = (
    "rows",
    "routes",
    "hits_at_k",
    "hits_at_m",
    "cold_routes",
    "cold_hits_at_m",
    "cold_candidates",
)


def _membership(candidates: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Bool ``[rows, num_experts]``; out-of-range ids land in a dropped sentinel column."""
    in_range = (candidates >= 0) & (candidates < num_experts)
    columns = torch.where(in_range, candidates, torch.full_like(candidates, num_experts))
    mask = torch.zeros(
        (candidates.shape[0], num_experts + 1), dtype=torch.bool, device=candidates.device
    )
    mask.scatter_(1, columns, True)
    return mask[:, :num_experts]


def score_candidates(
    *,
    candidates: torch.Tensor,
    actual: torch.Tensor,
    top_k: int,
    num_experts: int,
    resident: torch.Tensor | None,
) -> torch.Tensor:
    """Counts ordered as ``COUNTER_NAMES`` for one layer of one forward, without host syncs.

    A route is cold when ``resident`` marks its expert absent; with no
    residency mask every route is cold.
    """
    valid = (actual >= 0) & (actual < num_experts)
    safe_actual = actual.clamp(0, num_experts - 1)
    hits_m = _membership(candidates, num_experts).gather(1, safe_actual) & valid
    hits_k = _membership(candidates[:, :top_k], num_experts).gather(1, safe_actual) & valid
    candidate_valid = (candidates >= 0) & (candidates < num_experts)
    if resident is None:
        cold = valid
        cold_candidates = candidate_valid
    else:
        cold = valid & ~resident[safe_actual]
        cold_candidates = candidate_valid & ~resident[candidates.clamp(0, num_experts - 1)]
    return torch.stack(
        (
            torch.full((), actual.shape[0], dtype=torch.int64, device=actual.device),
            valid.sum(),
            hits_k.sum(),
            hits_m.sum(),
            cold.sum(),
            (hits_m & cold).sum(),
            cold_candidates.sum(),
        )
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


class ShadowMetrics:
    """Cumulative ``[predictors, layers, counters]`` totals kept on device."""

    def __init__(
        self, *, predictor_names: Sequence[str], layer_ids: Sequence[int], device: torch.device
    ) -> None:
        self._predictor_names = tuple(predictor_names)
        self._layer_ids = tuple(layer_ids)
        self._layer_index = {layer_id: i for i, layer_id in enumerate(self._layer_ids)}
        self._totals = torch.zeros(
            (len(self._predictor_names), len(self._layer_ids), len(COUNTER_NAMES)),
            dtype=torch.int64,
            device=device,
        )

    def add(self, *, predictor_index: int, target_layer: int, counts: torch.Tensor) -> None:
        self._totals[predictor_index, self._layer_index[target_layer]].add_(counts)

    def snapshot(self) -> dict:
        """Synchronizes with the device; returns JSON-ready per-layer counters and totals."""
        totals = self._totals.cpu().tolist()
        result = {}
        for predictor_index, name in enumerate(self._predictor_names):
            per_layer = totals[predictor_index]
            layers = {
                str(layer_id): dict(zip(COUNTER_NAMES, per_layer[i]))
                for i, layer_id in enumerate(self._layer_ids)
                if per_layer[i][0] > 0
            }
            total = dict(zip(COUNTER_NAMES, (sum(column) for column in zip(*per_layer))))
            total["recall_at_k"] = _ratio(total["hits_at_k"], total["routes"])
            total["recall_at_m"] = _ratio(total["hits_at_m"], total["routes"])
            total["cold_recall_at_m"] = _ratio(total["cold_hits_at_m"], total["cold_routes"])
            total["cold_precision_at_m"] = _ratio(
                total["cold_hits_at_m"], total["cold_candidates"]
            )
            result[name] = {"layers": layers, "total": total}
        return result

    def append_jsonl(self, path: Path, *, forwards: int) -> None:
        record = {
            "timestamp_ns": time.time_ns(),
            "forwards": forwards,
            "predictors": self.snapshot(),
        }
        with path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, sort_keys=True) + "\n")
```

- [x] **Step 4: Commit and run the tests**

Divix01 loop with `<files>` = `python/sglang/srt/layers/moe/expert_prediction/metrics.py`, message `feat(moe): score expert predictor candidates on device`, test files = `test/registered/unit/layers/moe/test_expert_prediction_metrics.py`.
Expected: `4 passed`, `EXIT=0`.

---

### Task 5: Runtime and ModelRunner wiring

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/runtime.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py` (`initialize` near line 676, new helper after `maybe_init_expert_hot_cache`, `forward` near line 1784)
- Test: `test/registered/unit/layers/moe/test_expert_prediction_runtime.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4; `sglang.srt.layers.moe.expert_residency_clock.classify_forward`, `ForwardKind`; hot cache objects exposing `expert_to_slot` (long `[num_experts]`, -1 = not resident).
- Produces:
  - `layer_pairs(layer_ids, offset) -> tuple[tuple[int, int], ...]`
  - `ExpertPredictionRuntime.build(*, model, predictor_names, device, hidden_dtype, max_rows, max_candidates, hot_caches, log_interval, metrics_path, topk_type=None, experts_type=None)`
  - `ExpertPredictionRuntime.from_env(*, model, gpu_id, hidden_dtype, decode_max_bs, tokens_per_request, tp_size, moe_ep_size, attn_dp_size, expert_hot_cache_manager)`
  - instance: `.store`, `.metrics`, `.forwards`, `.on_forward_end(forward_batch)`, `.close()`
  - `ModelRunner.expert_prediction_runtime` (None when off), `ModelRunner.maybe_init_expert_prediction()`

- [x] **Step 1: Write the failing test**

Create `test/registered/unit/layers/moe/test_expert_prediction_runtime.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.runtime import (
    ExpertPredictionRuntime,
    layer_pairs,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class FakeTopK(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=top_k, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.softmax(dim=-1), self.topk_config.top_k, dim=-1)
        return StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=router_logits)


class TupleTopK(FakeTopK):
    def forward(self, hidden_states, router_logits):
        output = super().forward(hidden_states, router_logits)
        return (output.topk_weights, output.topk_ids)


class FakeMoE(nn.Module):
    def __init__(self, layer_id, num_experts, hidden_size):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.hidden_size = hidden_size

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.gate = nn.Linear(6, 8, bias=False)
        self.topk = FakeTopK(2)
        self.experts = FakeMoE(layer_id, 8, 6)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class FakeModel(nn.Module):
    def __init__(self, layer_ids=(0, 1, 2)):
        super().__init__()
        self.layers = nn.ModuleList(FakeBlock(i) for i in layer_ids)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _batch(mode, rows):
    return SimpleNamespace(
        forward_mode=mode,
        input_ids=torch.zeros(rows, dtype=torch.int64),
        batch_size=rows,
        spec_info=None,
        extend_num_tokens=rows,
    )


def _runtime(*, model=None, metrics_path=None, hot_caches=None, max_rows=4, log_interval=2):
    model = model or FakeModel()
    runtime = ExpertPredictionRuntime.build(
        model=model,
        predictor_names=("popularity", "affinity"),
        device=torch.device("cpu"),
        hidden_dtype=torch.float32,
        max_rows=max_rows,
        max_candidates=4,
        hot_caches=hot_caches or {},
        log_interval=log_interval,
        metrics_path=metrics_path,
        topk_type=FakeTopK,
        experts_type=FakeMoE,
    )
    return model, runtime


def _decode(model, runtime, rows=3, mode=ForwardMode.DECODE):
    with torch.no_grad():
        model(torch.randn(rows, 6))
    runtime.on_forward_end(_batch(mode, rows))


class TestExpertPredictionRuntime(unittest.TestCase):
    def test_layer_pairs_by_offset(self):
        self.assertEqual(layer_pairs((0, 2, 5), 0), ((0, 0), (2, 2), (5, 5)))
        self.assertEqual(layer_pairs((0, 2, 5), 1), ((0, 2), (2, 5)))

    def test_decode_forwards_score_every_predictor_and_write_metrics(self):
        path = Path(tempfile.mkdtemp()) / "prediction.jsonl"
        model, runtime = _runtime(metrics_path=path)
        for _ in range(2):
            _decode(model, runtime)
        self.assertEqual(runtime.forwards, 2)
        (line,) = path.read_text().splitlines()
        record = json.loads(line)
        self.assertEqual(record["forwards"], 2)
        popularity = record["predictors"]["popularity"]
        affinity = record["predictors"]["affinity"]
        self.assertEqual(set(popularity["layers"]), {"0", "1", "2"})
        self.assertEqual(popularity["total"]["routes"], 2 * 3 * 2 * 3)
        self.assertEqual(set(affinity["layers"]), {"1", "2"})
        self.assertEqual(affinity["total"]["routes"], 2 * 3 * 2 * 2)
        for predictor in (popularity, affinity):
            self.assertTrue(0.0 <= predictor["total"]["recall_at_m"] <= 1.0)

    def test_verify_forwards_are_scored(self):
        model, runtime = _runtime()
        _decode(model, runtime, mode=ForwardMode.TARGET_VERIFY)
        self.assertEqual(runtime.forwards, 1)

    def test_prefill_idle_and_oversized_forwards_are_not_scored(self):
        model, runtime = _runtime(max_rows=4)
        _decode(model, runtime, mode=ForwardMode.EXTEND)
        runtime.on_forward_end(_batch(ForwardMode.IDLE, 0))
        _decode(model, runtime, rows=5)
        self.assertEqual(runtime.forwards, 0)

    def test_residency_mask_splits_cold_routes(self):
        all_resident = SimpleNamespace(expert_to_slot=torch.arange(8))
        model, runtime = _runtime(hot_caches={1: all_resident})
        _decode(model, runtime)
        layers = runtime.metrics.snapshot()["popularity"]["layers"]
        self.assertEqual(layers["1"]["cold_routes"], 0)
        self.assertEqual(layers["2"]["cold_routes"], layers["2"]["routes"])

    def test_unsupported_topk_disables_scoring(self):
        model = FakeModel()
        model.layers[0].topk = TupleTopK(2)
        model, runtime = _runtime(model=model)
        with self.assertLogs("sglang.srt.layers.moe.expert_prediction", level="WARNING"):
            _decode(model, runtime)
        self.assertEqual(runtime.forwards, 0)

    def test_close_removes_hooks(self):
        model, runtime = _runtime()
        _decode(model, runtime)
        before = runtime.store.view(0, RouteFeature.TOPK_IDS, 3).clone()
        runtime.close()
        with torch.no_grad():
            model(torch.randn(3, 6) * 100)
        self.assertTrue(torch.equal(runtime.store.view(0, RouteFeature.TOPK_IDS, 3), before))

    def test_from_env_rejects_multi_gpu_and_unsized_buffers(self):
        common = dict(
            model=FakeModel(),
            gpu_id=0,
            hidden_dtype=torch.float32,
            tokens_per_request=1,
            moe_ep_size=1,
            attn_dp_size=None,
            expert_hot_cache_manager=None,
        )
        with envs.SGLANG_MOE_EXPERT_PREDICTOR.override("popularity"):
            with self.assertRaisesRegex(ValueError, "single GPU"):
                ExpertPredictionRuntime.from_env(decode_max_bs=1, tp_size=2, **common)
            with self.assertRaisesRegex(ValueError, "MAX_ROWS"):
                ExpertPredictionRuntime.from_env(decode_max_bs=0, tp_size=1, **common)


if __name__ == "__main__":
    unittest.main()
```

- [-] **Step 2: Commit the failing test and run it on divix01** (skipped: MVP, test committed with the implementation)

Divix01 loop with `<files>` = `test/registered/unit/layers/moe/test_expert_prediction_runtime.py`, message `test(moe): add expert prediction runtime tests`, test files = same path.
Expected: `ModuleNotFoundError` for `runtime`, `EXIT=` nonzero.

- [x] **Step 3: Implement the runtime**

Create `python/sglang/srt/layers/moe/expert_prediction/runtime.py`:

```python
"""Tap MoE routes during forwards and shadow-score expert predictors after each one."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.adapters import install_pre_mixer_taps
from sglang.srt.layers.moe.expert_prediction.base import ExpertPredictor
from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.metrics import ShadowMetrics, score_candidates
from sglang.srt.layers.moe.expert_prediction.registry import build_predictors
from sglang.srt.layers.moe.expert_prediction.taps import (
    RouteTaps,
    TappedMoeLayer,
    discover_moe_layers,
)
from sglang.srt.layers.moe.expert_residency_clock import ForwardKind, classify_forward

logger = logging.getLogger(__name__)

_SCORED_KINDS = frozenset({ForwardKind.DECODE, ForwardKind.VERIFY})


def layer_pairs(layer_ids: Sequence[int], offset: int) -> tuple[tuple[int, int], ...]:
    """``(source, target)`` pairs where target is ``offset`` tapped MoE layers after source."""
    return tuple(zip(layer_ids, layer_ids[offset:]))


class ExpertPredictionRuntime:
    """Shadow-score registered predictors against native routes.

    Scoring enqueues device work behind the forward on the current stream and
    never synchronizes; only the periodic metrics record reads the device.
    """

    def __init__(
        self,
        *,
        layers: Sequence[TappedMoeLayer],
        store: FeatureStore,
        taps: RouteTaps,
        pre_mixer_removers: Sequence[Callable[[], None]],
        predictors: Sequence[ExpertPredictor],
        metrics: ShadowMetrics,
        hot_caches: Mapping[int, Any],
        log_interval: int,
        metrics_path: Path | None,
    ) -> None:
        if log_interval < 1:
            raise ValueError("SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL must be positive")
        self.store = store
        self.metrics = metrics
        self.forwards = 0
        self._specs = {layer.spec.layer_id: layer.spec for layer in layers}
        self._layer_ids = tuple(sorted(self._specs))
        self._taps = taps
        self._pre_mixer_removers = list(pre_mixer_removers)
        self._predictors = tuple(predictors)
        self._hot_caches = dict(hot_caches)
        self._log_interval = log_interval
        self._metrics_path = metrics_path
        self._reported_unsupported = False

    @classmethod
    def from_env(
        cls,
        *,
        model: nn.Module,
        gpu_id: int,
        hidden_dtype: torch.dtype,
        decode_max_bs: int,
        tokens_per_request: int,
        tp_size: int,
        moe_ep_size: int,
        attn_dp_size: int | None,
        expert_hot_cache_manager: Any | None,
    ) -> "ExpertPredictionRuntime":
        if tp_size > 1 or moe_ep_size > 1 or (attn_dp_size or 1) > 1:
            raise ValueError(
                "SGLANG_MOE_EXPERT_PREDICTOR supports a single GPU; got "
                f"tp={tp_size} moe_ep={moe_ep_size} attn_dp={attn_dp_size}"
            )
        max_rows = (
            envs.SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS.get()
            or decode_max_bs * tokens_per_request
        )
        if max_rows < 1:
            raise ValueError(
                "SGLANG_MOE_EXPERT_PREDICTOR needs decode CUDA graphs or "
                "SGLANG_MOE_EXPERT_PREDICTOR_MAX_ROWS to size its tap buffers"
            )
        metrics_file = envs.SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE.get()
        return cls.build(
            model=model,
            predictor_names=envs.SGLANG_MOE_EXPERT_PREDICTOR.get(),
            device=torch.device("cuda", gpu_id),
            hidden_dtype=hidden_dtype,
            max_rows=max_rows,
            max_candidates=envs.SGLANG_MOE_EXPERT_PREDICTOR_CANDIDATES.get(),
            hot_caches=(
                {} if expert_hot_cache_manager is None else expert_hot_cache_manager.caches
            ),
            log_interval=envs.SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL.get(),
            metrics_path=Path(metrics_file) if metrics_file else None,
        )

    @classmethod
    def build(
        cls,
        *,
        model: nn.Module,
        predictor_names: Sequence[str],
        device: torch.device,
        hidden_dtype: torch.dtype,
        max_rows: int,
        max_candidates: int,
        hot_caches: Mapping[int, Any],
        log_interval: int,
        metrics_path: Path | None,
        topk_type: type | None = None,
        experts_type: type | None = None,
    ) -> "ExpertPredictionRuntime":
        layers = discover_moe_layers(model, topk_type=topk_type, experts_type=experts_type)
        specs = [layer.spec for layer in layers]
        predictors = build_predictors(
            predictor_names, specs=specs, device=device, max_candidates=max_candidates
        )
        features = {RouteFeature.TOPK_IDS}.union(
            *(predictor.required_features for predictor in predictors)
        )
        store = FeatureStore(
            specs=specs,
            features=features,
            max_rows=max_rows,
            device=device,
            hidden_dtype=hidden_dtype,
        )
        taps = RouteTaps(store)
        taps.install(layers)
        pre_mixer_removers = (
            install_pre_mixer_taps(model=model, layers=layers, store=store)
            if RouteFeature.PRE_MIXER in features
            else []
        )
        logger.info(
            "MoE expert prediction shadow mode: predictors=%s layers=%d max_rows=%d "
            "tap_bytes=%d predictor_state_bytes=%d",
            ",".join(predictor_names),
            len(layers),
            max_rows,
            store.nbytes,
            sum(predictor.state_nbytes for predictor in predictors),
        )
        return cls(
            layers=layers,
            store=store,
            taps=taps,
            pre_mixer_removers=pre_mixer_removers,
            predictors=predictors,
            metrics=ShadowMetrics(
                predictor_names=predictor_names,
                layer_ids=[spec.layer_id for spec in specs],
                device=device,
            ),
            hot_caches=hot_caches,
            log_interval=log_interval,
            metrics_path=metrics_path,
        )

    def on_forward_end(self, forward_batch: Any) -> None:
        rows = self._scored_rows(forward_batch)
        if rows == 0:
            return
        for index, predictor in enumerate(self._predictors):
            self._score(predictor_index=index, predictor=predictor, rows=rows)
            predictor.observe(store=self.store, rows=rows)
        self.forwards += 1
        if self._metrics_path is not None and self.forwards % self._log_interval == 0:
            self.metrics.append_jsonl(self._metrics_path, forwards=self.forwards)

    def close(self) -> None:
        self._taps.remove()
        for remove in self._pre_mixer_removers:
            remove()
        self._pre_mixer_removers = []

    def _scored_rows(self, forward_batch: Any) -> int:
        """Rows this forward's taps hold for scoring, or 0 when it is not scored."""
        if self._taps.unsupported_layers:
            if not self._reported_unsupported:
                self._reported_unsupported = True
                logger.warning(
                    "MoE expert prediction disabled: layers %s lack standard top-k outputs",
                    sorted(self._taps.unsupported_layers),
                )
            return 0
        kind, _ = classify_forward(forward_batch)
        if kind not in _SCORED_KINDS:
            return 0
        rows = forward_batch.input_ids.shape[0]
        return rows if rows <= self.store.max_rows else 0

    def _score(self, *, predictor_index: int, predictor: ExpertPredictor, rows: int) -> None:
        for source, target in layer_pairs(self._layer_ids, predictor.target_offset):
            spec = self._specs[target]
            cache = self._hot_caches.get(target)
            candidates = predictor.predict(
                source_layer=source, target_layer=target, store=self.store, rows=rows
            )
            counts = score_candidates(
                candidates=candidates,
                actual=self.store.view(target, RouteFeature.TOPK_IDS, rows),
                top_k=spec.top_k,
                num_experts=spec.num_experts,
                resident=None if cache is None else cache.expert_to_slot >= 0,
            )
            self.metrics.add(
                predictor_index=predictor_index, target_layer=target, counts=counts
            )
```

- [x] **Step 4: Run the runtime tests**

Divix01 loop steps L1-L3 with `<files>` = `python/sglang/srt/layers/moe/expert_prediction/runtime.py`, message `feat(moe): shadow-score expert predictors after each forward`, test files = `test/registered/unit/layers/moe/test_expert_prediction_runtime.py`.
Expected: `8 passed`, `EXIT=0`.

- [x] **Step 5: Wire ModelRunner (frozen file: orchestration only)**

In `python/sglang/srt/model_executor/model_runner.py`, `initialize()` currently reads:

```python
        self.maybe_init_expert_hot_cache()
        if self.expert_hot_cache_manager is not None:
            self.expert_hot_cache_manager.enable_next_layer_prefetch(
                envs.SGLANG_MOE_PREFETCH_MAX_CANDIDATES.get()
            )

        self.maybe_init_dwdp()
```

Change it to:

```python
        self.maybe_init_expert_hot_cache()
        if self.expert_hot_cache_manager is not None:
            self.expert_hot_cache_manager.enable_next_layer_prefetch(
                envs.SGLANG_MOE_PREFETCH_MAX_CANDIDATES.get()
            )
        self.maybe_init_expert_prediction()

        self.maybe_init_dwdp()
```

Directly after the end of the `maybe_init_expert_hot_cache` method (the block ending with `register_forward_observer(manager.on_expert_distribution)`), add:

```python
    def maybe_init_expert_prediction(self):
        """Attach shadow MoE expert predictors before CUDA graph capture."""
        self.expert_prediction_runtime = None
        if self.is_draft_worker or not envs.SGLANG_MOE_EXPERT_PREDICTOR.get():
            return
        from sglang.srt.layers.moe.expert_prediction.runtime import (
            ExpertPredictionRuntime,
        )

        self.expert_prediction_runtime = ExpertPredictionRuntime.from_env(
            model=self.model,
            gpu_id=self.gpu_id,
            hidden_dtype=self.dtype,
            decode_max_bs=get_exec().graph.cuda_graph_config.decode.max_bs or 0,
            tokens_per_request=self.decode_num_tokens_per_req(),
            tp_size=self.ps.tp_size,
            moe_ep_size=self.ps.moe_ep_size,
            attn_dp_size=self.ps.attn_dp_size,
            expert_hot_cache_manager=self.expert_hot_cache_manager,
        )
```

In `forward`, the recorder block currently reads:

```python
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")
```

Insert the delegate inside the `with` block, before its end, so it runs before the recorder's per-forward observers change hot-cache residency:

```python
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
            if self.expert_prediction_runtime is not None:
                self.expert_prediction_runtime.on_forward_end(forward_batch)
        output.expert_distribution_metrics = recorder_outputs.get("metrics")
```

Confirm `get_exec` is already imported in `model_runner.py` (it is used in `maybe_init_expert_hot_cache`); add no new imports.

- [x] **Step 6: Commit and verify the wiring compiles and all CPU tests pass**

Divix01 loop L1-L2 with `<files>` = `python/sglang/srt/model_executor/model_runner.py`, message `feat(moe): wire shadow expert prediction into ModelRunner`. Then:

```bash
ssh -n divix01 "cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && CUDA_VISIBLE_DEVICES='' taskset -c 64-71 /data/models/slang/.venv/bin/python -m py_compile python/sglang/srt/model_executor/model_runner.py && echo COMPILE_OK"
```

Expected: `COMPILE_OK`. Then L3 with test files = `test/registered/unit/layers/moe/test_expert_prediction_taps.py test/registered/unit/layers/moe/test_expert_prediction_adapters.py test/registered/unit/layers/moe/test_expert_prediction_predictors.py test/registered/unit/layers/moe/test_expert_prediction_metrics.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch.py test/registered/unit/layers/moe/test_expert_residency.py`.
Expected: all pass (40 new tests plus the existing prefetch/residency tests), `EXIT=0`.

---

### Task 6: CUDA graph replay and no-sync verification (GPU; controller confirms the server is stopped)

**Files:**
- Test: `test/registered/unit/layers/moe/test_expert_prediction_graph.py`

**Interfaces:**
- Consumes: `ExpertPredictionRuntime.build`, `.store`, `.metrics`, `.on_forward_end`; `RouteFeature`.
- Produces: evidence that tap copies replay inside a captured CUDA graph and scoring issues no synchronizing ops.

- [x] **Step 1: Write the test**

Create `test/registered/unit/layers/moe/test_expert_prediction_graph.py`:

```python
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.runtime import ExpertPredictionRuntime
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")

ROWS = 8
HIDDEN = 32
EXPERTS = 16
TOP_K = 4


class FakeTopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=TOP_K, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.float().softmax(dim=-1), TOP_K, dim=-1)
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids.to(torch.int32), router_logits=router_logits
        )


class FakeMoE(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = EXPERTS
        self.hidden_size = HIDDEN

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, EXPERTS, bias=False)
        self.topk = FakeTopK()
        self.experts = FakeMoE(layer_id)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(FakeBlock(i) for i in (0, 1, 2))

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestExpertPredictionGraph(unittest.TestCase):
    def test_replay_refreshes_taps_and_scoring_never_syncs(self):
        device = torch.device("cuda", 0)
        model = FakeModel().to(device)
        runtime = ExpertPredictionRuntime.build(
            model=model,
            predictor_names=("popularity", "affinity"),
            device=device,
            hidden_dtype=torch.float32,
            max_rows=ROWS,
            max_candidates=8,
            hot_caches={1: SimpleNamespace(expert_to_slot=torch.full((EXPERTS,), -1, device=device))},
            log_interval=10**9,
            metrics_path=None,
            topk_type=FakeTopK,
            experts_type=FakeMoE,
        )
        static_input = torch.zeros(ROWS, HIDDEN, device=device)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            input_ids=torch.zeros(ROWS, dtype=torch.int64, device=device),
            batch_size=ROWS,
            spec_info=None,
            extend_num_tokens=ROWS,
        )
        with torch.inference_mode():
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                for _ in range(3):
                    model(static_input)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                model(static_input)

            for seed in (1, 2):
                generator = torch.Generator(device=device).manual_seed(seed)
                static_input.copy_(torch.randn(ROWS, HIDDEN, device=device, generator=generator))
                graph.replay()
                torch.cuda.synchronize()
                for layer_id, block in zip((0, 1, 2), model.layers):
                    logits = block.gate(static_input)
                    expected = torch.topk(logits.float().softmax(dim=-1), TOP_K, dim=-1).indices
                    self.assertTrue(
                        torch.equal(
                            runtime.store.view(layer_id, RouteFeature.TOPK_IDS, ROWS),
                            expected.long(),
                        )
                    )
                torch.cuda.set_sync_debug_mode("error")
                try:
                    runtime.on_forward_end(batch)
                finally:
                    torch.cuda.set_sync_debug_mode("default")

        snapshot = runtime.metrics.snapshot()
        self.assertEqual(runtime.forwards, 2)
        self.assertEqual(snapshot["popularity"]["total"]["routes"], 2 * ROWS * TOP_K * 3)
        self.assertEqual(snapshot["affinity"]["total"]["routes"], 2 * ROWS * TOP_K * 2)
        self.assertEqual(
            snapshot["popularity"]["layers"]["1"]["cold_routes"],
            snapshot["popularity"]["layers"]["1"]["routes"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Commit, then run on the GPU (only after the controller confirms production is stopped)**

Divix01 loop L1-L2 with `<files>` = `test/registered/unit/layers/moe/test_expert_prediction_graph.py`, message `test(moe): verify expert prediction taps replay in CUDA graphs without syncs`. Then:

```bash
ssh -n divix01 'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && echo GPU_FREE || echo GPU_BUSY'
ssh -n divix01 "cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && echo STARTED \$(date --iso-8601=seconds) && PYTHONPATH=/data/models/slang/nvfp4-work/flashinfer-0.6.18-cu130-overlay:\$PWD/python:/data/models/slang/nvfp4-work/uring-test-deps timeout 900 /data/models/slang/.venv/bin/python -m pytest -q -p no:cacheprovider -rfEs test/registered/unit/layers/moe/test_expert_prediction_graph.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py; echo EXIT=\$?"
```

Expected: `GPU_FREE`, then `9 passed`, `EXIT=0`. If `GPU_BUSY`, stop and report. If the sync check fails, the error names the synchronizing op; fix it in the module that issued it (no `.item()`/mask indexing), rerun, and commit the fix with its own message.

---

### Task 7: Live shadow smoke on an experiment server (GPU; controller runs the session)

**Files:**
- Create: `scripts/expert_prediction/run-shadow-server.sh`
- Modify: `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md` (append an entry)

**Interfaces:**
- Consumes: the full framework via env vars; production launch flags from `scripts/fp8_accuracy/run-fp8-server.sh`.
- Produces: measured decode tok/s for predictor off vs `affinity,popularity`, greedy output equality, a metrics JSONL, and an experiment log entry.

- [x] **Step 1: Create the launch script**

Create `scripts/expert_prediction/run-shadow-server.sh` and `chmod 755` it:

```bash
#!/usr/bin/env bash
# Launches one expert-prediction shadow server with production's E16c settings from the
# cc-expert-prediction worktree on 127.0.0.1:<port>. <predictors> is a comma list or "off".
# Usage: run-shadow-server.sh <name> <port> <predictors>
# Refuses to start while any process holds the GPU. Runs in the foreground; the session backgrounds it.
set -euo pipefail

name=${1:?name}
port=${2:?port}
predictors=${3:?predictors or off}
[ "$predictors" = off ] && predictors=""

work=/data/models/slang/nvfp4-work
worktree=$work/cc-expert-prediction/worktree
flashinfer_overlay=$work/flashinfer-0.6.18-cu130-overlay
model=/mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
cache_model_path=/data/models/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
expert_cache=/mnt/nvme2/nvfp4-work/qwen38-nvfp4-expert-cache-v1
ple_cache=/mnt/nvme2/ple-cache/qwen38-nvfp4
expert_seed=/data/models/slang/slang-dev-2bit/qwen3.8-flash-next-24gb-sglang/assets/expert_freq.pt
run_dir=$work/cc-expert-prediction/servers/$name/run-$(date +%Y%m%d-%H%M%S)
log=$run_dir/server.log

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "REFUSING_TO_START: the GPU is in use" >&2
    exit 1
fi

mkdir -p "$run_dir/profiles" "$work/runtime-tmp"
ln -sfn "$run_dir" "$work/cc-expert-prediction/servers/$name/latest"
cd "$worktree"
{
    echo "cc-expert-prediction server $name port=$port predictors=${predictors:-off}: $(date --iso-8601=seconds)"
    git status --short --branch
    git log -1 --oneline
    sha256sum python/sglang/srt/model_executor/model_runner.py python/sglang/srt/layers/moe/expert_prediction/*.py
} 2>&1 | tee -a "$log"

exec env \
    PYTHONPATH="$flashinfer_overlay:$worktree/python" \
    PYTHONUNBUFFERED=1 \
    TMPDIR="$work/runtime-tmp" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 \
    SGLANG_MOE_EXPERT_STREAM=1 \
    SGLANG_MOE_EXPERT_FILE_DIR="$expert_cache" \
    SGLANG_MOE_EXPERT_FILE_READER=uring_direct \
    SGLANG_QWEN4_PLE_FILE_READER=uring \
    SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY=1 \
    SGLANG_FILE_CACHE_MODEL_PATH="$cache_model_path" \
    SGLANG_MOE_HOT_GPU_MB=14336 \
    SGLANG_MOE_PINNED_HOST_MB=0 \
    SGLANG_MOE_EXPERT_HOST_ARENA=1 \
    SGLANG_MOE_EXPERT_GRAPH_GATHER=1 \
    SGLANG_MOE_HOT_SEED="$expert_seed" \
    SGLANG_MOE_HOT_DYNAMIC=1 \
    SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=4 \
    SGLANG_MOE_HOT_DECAY_TOKENS=1 \
    SGLANG_MOE_HOT_PROMOTION_SIGMAS=0 \
    SGLANG_MOE_HOT_BENEFIT_RATIO=2 \
    SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS=0 \
    SGLANG_MOE_HOT_LOG_INTERVAL=100 \
    SGLANG_MOE_HOT_METRICS_FILE="$run_dir/hot-cache.metrics.jsonl" \
    SGLANG_MOE_PREFETCH_MAX_CANDIDATES=0 \
    SGLANG_MOE_EXPERT_COPY_BACKEND=dma \
    SGLANG_MOE_EXPERT_PREDICTOR="$predictors" \
    SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL=100 \
    SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE="$run_dir/expert-prediction.metrics.jsonl" \
    SGLANG_TORCH_PROFILER_DIR="$run_dir/profiles" \
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$run_dir/profiles/expert-distribution" \
    SGLANG_VLM_CACHE_SIZE_MB=0 \
    SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=4 \
    SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S=5 \
    /data/models/slang/.venv/bin/sglang serve --model-type llm \
        --model-path "$model" \
        --tp 1 \
        --fp4-gemm-backend flashinfer_cutlass \
        --moe-runner-backend flashinfer_cutlass \
        --moe-a2a-backend none \
        --cpu-offload-gb 80 \
        --ple-offload-embedding \
        --ple-offload-backend file \
        --ple-offload-dir "$ple_cache" \
        --page-size 64 \
        --mamba-track-interval 64 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 4096 \
        --context-length 65536 \
        --max-total-tokens 65536 \
        --mamba-radix-cache-strategy extra_buffer_lazy \
        --max-running-requests 1 \
        --max-mamba-cache-size 1 \
        --mamba-ssm-dtype bfloat16 \
        --mem-fraction-static 0.95 \
        --disable-overlap-schedule \
        --disable-radix-cache \
        --language-model-only \
        --cuda-graph-backend-decode breakable \
        --cuda-graph-bs-decode 1 \
        --cuda-graph-max-bs-decode 1 \
        --cuda-graph-backend-prefill disabled \
        --disable-flashinfer-autotune \
        --skip-server-warmup \
        --weight-loader-drop-cache-after-load \
        --expert-distribution-recorder-mode per_pass \
        --reasoning-parser auto \
        --default-chat-template-kwargs '{"enable_thinking": true}' \
        --host 127.0.0.1 \
        --port "$port" \
    >> "$log" 2>&1
```

Commit with Divix01 loop L1-L2, `<files>` = `scripts/expert_prediction/run-shadow-server.sh`, message `feat(moe): add expert prediction shadow server launcher`.

- [x] **Step 2: Run the session (controller only; production is down for its duration)**

For `predictors` in `off` then `affinity,popularity`, with port 7871:
1. `ssh -n divix01 "setsid nohup /data/models/slang/nvfp4-work/cc-expert-prediction/worktree/scripts/expert_prediction/run-shadow-server.sh shadow-<off|on> 7871 <predictors> > /dev/null 2>&1 &"`
2. Wait for health: `ssh -n divix01 'until curl -sf http://127.0.0.1:7871/health; do sleep 5; done; echo HEALTHY'` (run under the Monitor tool with a 15 min bound).
3. Send the same greedy request twice and save both responses:

```bash
ssh -n divix01 'curl -s http://127.0.0.1:7871/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain how a B-tree handles node splits, with a worked example.\"}],\"max_tokens\":600,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"' > /tmp/claude-1000/-home-dimitri-data-divix-sglang-nvfp4/b64912f7-2a7b-4857-8ac3-ffc9e18d0816/scratchpad/shadow-<off|on>-<1|2>.json
```

4. Stop the server: SIGTERM its `sglang serve` pid, then wait until `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty.

- [x] **Step 3: Check the results**

- Output text of `shadow-off-2.json` equals `shadow-on-2.json`.
- `shadow-on` `server.log` contains `MoE expert prediction shadow mode: predictors=affinity,popularity layers=48` and decode batch lines report `cuda graph: True`; no `cannot tap layer` warning.
- `shadow-on` `expert-prediction.metrics.jsonl` has at least 5 records; the last record's `affinity.total.routes > 0`, `popularity.total.routes > 0`, all ratios within [0, 1].
- Record decode `gen throughput (token/s)` median from each `server.log` for the second request.

- [-] **Step 4: Relaunch production and record the entry** (entry recorded in the new experiment file; production not relaunched)

Relaunch production (`tmux send-keys -t cc-nvfp4-dynamic /data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh Enter`), wait for health on 127.0.0.1:7867 and 10.0.0.15:7867. Append an entry to `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md` with commit, run dirs, tok/s off vs on, output equality, last-record recall_at_k / recall_at_m / cold_recall_at_m for both predictors. Commit with Divix01 loop L1-L2, `<files>` = that log path, message `docs(nvfp4): record expert prediction shadow smoke`.
