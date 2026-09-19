# MoE Offload Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--moe-offload-preset {off,graph-gather,doorbell}` replaces ~22 hand-set offload environment variables with one documented, validated preset.

**Architecture:** A pure module (`layers/moe/offload_presets.py`) holds the `msgspec` preset model, the two presets, the merge-and-derive logic and the validation. A thin `arg_groups` hook applies it: it fills unset variables with `envs.X.set`, turns overlap off through `declare_resolution`, and refuses invalid combinations before the weight load. The existing environment-variable read sites are untouched.

**Tech Stack:** Python 3.13, `msgspec`, the `sglang.srt.environ.envs` descriptors, the `arg_groups` resolution pipeline, `unittest`/pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-moe-offload-presets-design.md`

## Global Constraints

- New data containers are `msgspec.Struct`, never `@dataclass` (`.claude/rules/no-dataclasses.md`).
- Every new `SGLANG_*` variable is an `EnvField` on `Envs`, read with `.get()`, overridden in tests with `.override()` (`env-var-conventions` skill).
- No defensive `getattr`/`hasattr` on fields that always exist (`.claude/rules/no-getattr-defensive.md`).
- Comments follow `.claude/rules/comment-style.md`: one or two lines, the fact a reader cannot see, ASCII only.
- `python/sglang/srt/model_executor/model_runner.py` is frozen. Do not edit it.
- Default `--moe-offload-preset off`. With no offload variable set, `off` touches nothing and declares nothing.
- Precedence: explicitly set variable > preset value > `Envs` default.
- Tests run on divix01 (the laptop venv has no pytest). CPU tests use `CUDA_VISIBLE_DEVICES=` and need no GPU lock. GPU runs hold `/data/models/slang/nvfp4-work/cc-gpu.lock`, need an empty `nvidia-smi` census before and after, ports 7867 and 31040-31047 must not be listening, and they run under `taskset -c 0-63`. Never kill a process you did not start.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01ABiGtddQbhsXzYFnNHqhxR
  ```
  Stage files by name. Do not push.

## Running tests on divix01

The branch is not pushed, so tests run in a divix01 worktree fed a patch of the local tree. Set up once:

```bash
C=/data/models/slang/nvfp4-work/cc-expert-prediction
timeout 90 ssh -n -o BatchMode=yes divix01 "git -C /data/models/slang/sglang worktree add -q --detach $C/wt-presets 797be6f678"
```

Sync before every test run (new files need `git add -N` so `git diff` includes them):

```bash
cd /home/dimitri/data/divix/sglang-nvfp4
S=/tmp/claude-1000/-home-dimitri-data-divix-sglang-nvfp4/b33df7b3-ec05-46fa-a1d0-ee10bc177ba2/scratchpad/he
C=/data/models/slang/nvfp4-work/cc-expert-prediction
for f in python/sglang/srt/layers/moe/offload_presets.py python/sglang/srt/arg_groups/moe_offload_hook.py test/registered/unit/layers/moe/test_offload_presets.py; do [ -e $f ] && git add -N $f; done
git diff 797be6f678 -- python/ scripts/ test/ > $S/wt-presets.patch
scp -q -o BatchMode=yes $S/wt-presets.patch divix01:$C/wt-presets.patch
timeout 90 ssh -n -o BatchMode=yes divix01 "cd $C/wt-presets && git reset -q --hard 797be6f678 && git apply --index ../wt-presets.patch && echo SYNCED"
```

CPU test command (referred to below as **RUN_CPU `<pytest args>`**):

```bash
timeout 600 ssh -n -o BatchMode=yes divix01 "cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-presets && CUDA_VISIBLE_DEVICES= taskset -c 0-63 env OMP_NUM_THREADS=8 PYTHONPATH=\$PWD/python /data/models/slang/.venv/bin/python -m pytest -q -p no:cacheprovider <pytest args> 2>&1 | tail -15"
```

## File Structure

| File | Responsibility |
|---|---|
| `python/sglang/srt/environ.py` (modify) | Register `SGLANG_MOE_EXPERT_STREAM`. |
| `python/sglang/srt/layers/moe/expert_stream.py:366` (modify) | Read the stream switch through `envs`. |
| `python/sglang/srt/arg_groups/memory_hook.py:103` (modify) | Same. |
| `python/sglang/srt/layers/moe/offload_presets.py` (create) | Preset model, the two presets, merge/derive, overlap rule, validation. Pure: no reads of process state. |
| `python/sglang/srt/arg_groups/moe_offload_hook.py` (create) | The only impure part: reads `os.environ` and server args, sets variables, declares overlap off, logs. |
| `python/sglang/srt/arg_groups/fields/exec_.py` (modify) | Declare `--moe-offload-preset`. |
| `python/sglang/srt/arg_groups/pipeline.py` (modify) | Call the hook just before `handle_offload_compatibility`. |
| `test/registered/unit/layers/moe/test_offload_presets.py` (create) | CPU tests for the module and the hook. |
| `divix01:/data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh` (modify, not in repo) | Switch prod to the flag. |
| `MOE_EXPERT_TRANSFER.md` (modify) | Record the presets. |

---

### Task 1: Register `SGLANG_MOE_EXPERT_STREAM` in `Envs`

The preset must set this variable through the same `envs` path as every other variable. Today it is read twice with raw `os.environ.get(...) == "1"`.

**Files:**
- Modify: `python/sglang/srt/environ.py` (the `SGLANG_MOE_*` block, next to `SGLANG_MOE_EXPERT_FILE_DIR`)
- Modify: `python/sglang/srt/layers/moe/expert_stream.py:366`
- Modify: `python/sglang/srt/arg_groups/memory_hook.py:103`
- Test: existing `test/registered/unit/test_nvfp4_expert_offload.py`

**Interfaces:**
- Produces: `envs.SGLANG_MOE_EXPERT_STREAM` (an `EnvBool`, default `False`).

- [ ] **Step 1: Add the descriptor**

In `python/sglang/srt/environ.py`, immediately above `    SGLANG_MOE_EXPERT_FILE_DIR = EnvStr("")`:

```python
    # Stream ModelOpt NVFP4 routed experts from host memory; required for hot caching.
    SGLANG_MOE_EXPERT_STREAM = EnvBool(False)
```

- [ ] **Step 2: Replace the two raw reads**

`python/sglang/srt/layers/moe/expert_stream.py:366`. Replace the line
`    return os.environ.get("SGLANG_MOE_EXPERT_STREAM") == "1"`
with
`    return envs.SGLANG_MOE_EXPERT_STREAM.get()`.
Add `from sglang.srt.environ import envs` to the imports if the file does not already import it (`grep -n "^from sglang.srt.environ import envs" python/sglang/srt/layers/moe/expert_stream.py`). Keep `import os` only if `os` is still used elsewhere in the file (`grep -n "os\." ...`).

`python/sglang/srt/arg_groups/memory_hook.py:103`. Replace
`    streaming = os.environ.get("SGLANG_MOE_EXPERT_STREAM") == "1"`
with
`    streaming = envs.SGLANG_MOE_EXPERT_STREAM.get()`.
`envs` is already imported there.

- [ ] **Step 3: Run the existing offload tests**

Sync (see "Running tests on divix01"), then RUN_CPU `test/registered/unit/test_nvfp4_expert_offload.py`
Expected: the same pass count as before the change (61 selected, all pass). `patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STREAM": "1"})` in those tests still works, because `EnvBool.get()` reads `os.environ`.

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/arg_groups/memory_hook.py
git commit -F - <<'EOF'
refactor(moe): read SGLANG_MOE_EXPERT_STREAM through Envs

The two raw os.environ reads become an EnvBool, so a preset can set the
switch through the same path as every other offload variable.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ABiGtddQbhsXzYFnNHqhxR
EOF
```

---

### Task 2: The preset module

**Files:**
- Create: `python/sglang/srt/layers/moe/offload_presets.py`
- Create: `test/registered/unit/layers/moe/test_offload_presets.py`

**Interfaces:**
- Consumes: `envs.SGLANG_MOE_EXPERT_STREAM` (Task 1).
- Produces (used by Task 3):
  - `class MoeOffloadPreset(msgspec.Struct, frozen=True, kw_only=True)`
  - `ENV_NAMES: dict[str, str]` (preset field name to variable name)
  - `OFFLOAD_ENV_NAMES: tuple[str, ...]`
  - `GRAPH_GATHER_PRESET: MoeOffloadPreset`, `DOORBELL_PRESET: MoeOffloadPreset`
  - `PRESETS: dict[str, MoeOffloadPreset | None]` with keys `"off"`, `"graph-gather"`, `"doorbell"`
  - `class ResolvedOffloadEnv(msgspec.Struct, frozen=True)` with fields `effective: dict[str, str]`, `filled: dict[str, str]`, `overridden: dict[str, str]`
  - `preset_env(preset: MoeOffloadPreset) -> dict[str, str]`
  - `explicit_offload_env(environ: Mapping[str, str]) -> dict[str, str]`
  - `resolve_offload_env(preset: MoeOffloadPreset | None, explicit: Mapping[str, str]) -> ResolvedOffloadEnv`
  - `needs_overlap_off(values: Mapping[str, str]) -> bool`
  - `check_offload_config(values: Mapping[str, str], *, speculative: bool, decode_graphs_disabled: bool, decode_max_bs: int | None, tp_size: int, pp_size: int, dp_size: int, dp_attention: bool, allowed_cpus: Collection[int]) -> None`

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/layers/moe/test_offload_presets.py`:

```python
"""MoE offload presets: the preset values, the merge and derivation rules, and validation."""

import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.moe import offload_presets as presets
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The offload variables of divix01:/data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh as
# of 2026-09-19, minus site-specific paths. The graph-gather preset must reproduce them.
PROD_OFFLOAD_ENV = {
    "SGLANG_MOE_EXPERT_STREAM": "1",
    "SGLANG_MOE_EXPERT_FILE_READER": "uring_direct",
    "SGLANG_QWEN4_PLE_FILE_READER": "uring",
    "SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY": "1",
    "SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING": "1",
    "SGLANG_MOE_HOT_GPU_MB": "15360",
    "SGLANG_MOE_PINNED_HOST_MB": "0",
    "SGLANG_MOE_EXPERT_HOST_ARENA": "1",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1",
    "SGLANG_MOE_EXPERT_FUSED_PLAN": "1",
    "SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT": "1",
    "SGLANG_MOE_HOT_DYNAMIC": "1",
    "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": "1",
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2",
    "SGLANG_MOE_HOT_DECAY_TOKENS": "1",
    "SGLANG_MOE_HOT_PROMOTION_SIGMAS": "0",
    "SGLANG_MOE_HOT_BENEFIT_RATIO": "2",
    "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS": "0",
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "1",
    "SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS": "64",
    "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": "0",
    "SGLANG_MOE_EXPERT_COPY_BACKEND": "dma",
}

ALL_CPUS = frozenset(range(72))


def parsed(values):
    return {name: getattr(envs, name).parse(value) for name, value in values.items()}


def check(values, **overrides):
    context = dict(
        speculative=False,
        decode_graphs_disabled=False,
        decode_max_bs=1,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        dp_attention=False,
        allowed_cpus=ALL_CPUS,
    )
    context.update(overrides)
    presets.check_offload_config(values, **context)


class TestPresetValues(unittest.TestCase):
    def test_graph_gather_preset_is_the_prod_offload_env(self):
        self.assertEqual(parsed(presets.preset_env(presets.GRAPH_GATHER_PRESET)), parsed(PROD_OFFLOAD_ENV))

    def test_doorbell_preset_departs_from_graph_gather_only_where_the_doorbell_forces_it(self):
        graph = parsed(presets.preset_env(presets.GRAPH_GATHER_PRESET))
        doorbell = parsed(presets.preset_env(presets.DOORBELL_PRESET))
        changed = {name for name in graph.keys() | doorbell.keys() if graph.get(name) != doorbell.get(name)}
        self.assertEqual(
            changed,
            {
                "SGLANG_MOE_EXPERT_DOORBELL",
                "SGLANG_MOE_EXPERT_DOORBELL_MODE",
                "SGLANG_MOE_EXPERT_DOORBELL_CPU",
                "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE",
                "SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT",
            },
        )

    def test_every_preset_field_names_an_envs_descriptor(self):
        self.assertEqual(set(presets.ENV_NAMES), set(presets.MoeOffloadPreset.__struct_fields__))
        for name in presets.ENV_NAMES.values():
            self.assertTrue(hasattr(envs, name), name)
        self.assertEqual(set(presets.PRESETS), {"off", "graph-gather", "doorbell"})
        self.assertIsNone(presets.PRESETS["off"])


class TestResolution(unittest.TestCase):
    def test_an_explicit_variable_wins_over_the_preset(self):
        resolved = presets.resolve_offload_env(presets.GRAPH_GATHER_PRESET, {"SGLANG_MOE_HOT_GPU_MB": "12288"})
        self.assertEqual(resolved.effective["SGLANG_MOE_HOT_GPU_MB"], "12288")
        self.assertNotIn("SGLANG_MOE_HOT_GPU_MB", resolved.filled)
        self.assertEqual(resolved.overridden, {"SGLANG_MOE_HOT_GPU_MB": "15360"})

    def test_an_explicit_variable_equal_to_the_preset_is_not_an_override(self):
        resolved = presets.resolve_offload_env(presets.GRAPH_GATHER_PRESET, {"SGLANG_MOE_HOT_BENEFIT_RATIO": "2"})
        self.assertEqual(resolved.overridden, {})

    def test_off_with_no_offload_env_sets_nothing(self):
        resolved = presets.resolve_offload_env(None, {})
        self.assertEqual((resolved.effective, resolved.filled, resolved.overridden), ({}, {}, {}))

    def test_insert_on_miss_derives_the_residency_update_and_its_boundary_cadence(self):
        resolved = presets.resolve_offload_env(None, {"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2"})
        self.assertEqual(
            parsed(resolved.filled),
            {
                "SGLANG_MOE_GPU_RESIDENCY_UPDATE": True,
                "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": 1,
                "SGLANG_MOE_HOT_DYNAMIC": True,
            },
        )

    def test_an_explicit_value_contradicting_a_derivation_is_refused(self):
        for explicit, culprit in (
            ({"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "1", "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "0"}, "SGLANG_MOE_GPU_RESIDENCY_UPDATE"),
            ({"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2", "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": "4"}, "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS"),
            ({"SGLANG_MOE_GPU_RESIDENCY_UPDATE": "1", "SGLANG_MOE_HOT_DYNAMIC": "0"}, "SGLANG_MOE_HOT_DYNAMIC"),
        ):
            with self.subTest(culprit=culprit), self.assertRaisesRegex(ValueError, culprit):
                presets.resolve_offload_env(None, explicit)

    def test_explicit_offload_env_keeps_only_offload_variables(self):
        environ = {"SGLANG_MOE_HOT_GPU_MB": "1024", "PATH": "/bin", "SGLANG_LOG_MS": "1"}
        self.assertEqual(presets.explicit_offload_env(environ), {"SGLANG_MOE_HOT_GPU_MB": "1024"})


class TestOverlapRule(unittest.TestCase):
    def test_graph_gather_keeps_overlap_and_doorbell_turns_it_off(self):
        self.assertFalse(presets.needs_overlap_off(presets.preset_env(presets.GRAPH_GATHER_PRESET)))
        self.assertTrue(presets.needs_overlap_off(presets.preset_env(presets.DOORBELL_PRESET)))

    def test_a_hot_cache_without_graph_gather_and_the_residency_update_turns_overlap_off(self):
        self.assertTrue(presets.needs_overlap_off({"SGLANG_MOE_HOT_GPU_MB": "1024"}))
        self.assertFalse(presets.needs_overlap_off({}))


class TestValidation(unittest.TestCase):
    def test_both_presets_pass_their_intended_setups(self):
        check(presets.preset_env(presets.GRAPH_GATHER_PRESET), speculative=True)
        check(presets.preset_env(presets.DOORBELL_PRESET))

    def test_invalid_combinations_are_refused_before_the_weight_load(self):
        graph = presets.preset_env(presets.GRAPH_GATHER_PRESET)
        doorbell = presets.preset_env(presets.DOORBELL_PRESET)
        cases = (
            (doorbell, dict(speculative=True), "speculative"),
            (dict(doorbell, SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE="2"), {}, "INSERT_ON_MISS_STAGE=2"),
            (doorbell, dict(allowed_cpus=frozenset(range(64))), "DOORBELL_CPU=71"),
            (doorbell, dict(tp_size=2), "parallel"),
            (doorbell, dict(dp_attention=True), "parallel"),
            (graph, dict(decode_graphs_disabled=True), "decode CUDA graphs"),
            (graph, dict(decode_max_bs=2), "cuda-graph-max-bs-decode 1"),
            (dict(graph, SGLANG_MOE_EXPERT_STREAM="0"), {}, "SGLANG_MOE_EXPERT_STREAM=1"),
        )
        for values, context, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                check(values, **context)
```

- [ ] **Step 2: Run the tests to verify they fail**

Sync, then RUN_CPU `test/registered/unit/layers/moe/test_offload_presets.py`
Expected: collection error, `ModuleNotFoundError: No module named 'sglang.srt.layers.moe.offload_presets'`.

- [ ] **Step 3: Write the module**

Create `python/sglang/srt/layers/moe/offload_presets.py`:

```python
"""Named MoE expert-offload configurations selected by ``--moe-offload-preset``.

A preset fills the offload variables the user left unset; an explicitly set
variable always wins. ``resolve_offload_env`` then derives the settings other
settings force, and ``check_offload_config`` refuses invalid combinations at
argument resolution, before the weight load. ``#N`` refers to a row of
"Experiment results" in MOE_EXPERT_TRANSFER.md. Everything here is pure; the
``arg_groups`` hook reads and writes the process state.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import msgspec

from sglang.srt.environ import envs


class MoeOffloadPreset(msgspec.Struct, frozen=True, kw_only=True):
    """One offload configuration; a ``None`` field leaves that variable to its ``Envs`` default."""

    # Stream ModelOpt NVFP4 routed experts from host memory; required for hot caching.
    expert_stream: bool | None = None
    # io_uring O_DIRECT reads of the expert file cache.
    expert_file_reader: str | None = None
    # io_uring reads of the file-backed PLE table.
    ple_file_reader: str | None = None
    # Stage PLE rows before decode graph replay instead of breaking the graph.
    ple_stage_before_replay: bool | None = None
    # Token embedding in pinned host memory, shared with the draft: frees 1.18 GB (#33).
    host_token_embedding: bool | None = None
    # Tuned for a 32 GB RTX 5090 with Qwen3.8-Flash-Next-NVFP4 at chunk 4096 (#34);
    # 16,384 OOMs and 15,872 leaves 66 MiB (#36). Override on any other GPU or model.
    hot_gpu_mb: int | None = None
    # 0: the host arena replaces the pinned LRU.
    pinned_host_mb: int | None = None
    # Copy every host expert row into registered memory once.
    expert_host_arena: bool | None = None
    # Host-to-device misses through the CUDA copy engine.
    expert_copy_backend: str | None = None
    # Sync-free in-graph gather of misses; both approaches are built on it.
    expert_graph_gather: bool | None = None
    # Scores decide residency at run time instead of the seed alone.
    hot_dynamic: bool | None = None
    # Residency policy tuned in the E16c/E19 arms.
    hot_decay_tokens: int | None = None
    hot_promotion_sigmas: float | None = None
    hot_benefit_ratio: float | None = None
    hot_min_residence_forwards: int | None = None
    # Residency boundaries run on the device, inside the decode graph.
    gpu_residency_update: bool | None = None
    gpu_residency_max_promotions: int | None = None
    # Insert-on-miss needs a boundary every decode forward.
    hot_update_decode_forwards: int | None = None
    # 0: prefetch was rejected, -4.3% (#5).
    prefetch_max_candidates: int | None = None
    # JIT route planner: about 65 fewer bookkeeping kernels per layer, +6.4%.
    expert_fused_plan: bool | None = None
    # 2 (DIRECT) copies misses straight into victim slots, +4.4% over 1; stage 2 refuses the doorbell.
    insert_on_miss_stage: int | None = None
    # MTP draft experts FP8 to NVFP4 at load: draft 2.46 to 1.45 GB (#30). No effect without such a draft.
    draft_moe_nvfp4_requant: bool | None = None
    # Side-thread copier that starts a miss copy before the graph reaches it.
    expert_doorbell: bool | None = None
    expert_doorbell_mode: str | None = None
    # The spin thread's core; must be in the process's allowed CPUs or it runs unpinned.
    expert_doorbell_cpu: int | None = None


ENV_NAMES: dict[str, str] = {
    "expert_stream": "SGLANG_MOE_EXPERT_STREAM",
    "expert_file_reader": "SGLANG_MOE_EXPERT_FILE_READER",
    "ple_file_reader": "SGLANG_QWEN4_PLE_FILE_READER",
    "ple_stage_before_replay": "SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY",
    "host_token_embedding": "SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING",
    "hot_gpu_mb": "SGLANG_MOE_HOT_GPU_MB",
    "pinned_host_mb": "SGLANG_MOE_PINNED_HOST_MB",
    "expert_host_arena": "SGLANG_MOE_EXPERT_HOST_ARENA",
    "expert_copy_backend": "SGLANG_MOE_EXPERT_COPY_BACKEND",
    "expert_graph_gather": "SGLANG_MOE_EXPERT_GRAPH_GATHER",
    "hot_dynamic": "SGLANG_MOE_HOT_DYNAMIC",
    "hot_decay_tokens": "SGLANG_MOE_HOT_DECAY_TOKENS",
    "hot_promotion_sigmas": "SGLANG_MOE_HOT_PROMOTION_SIGMAS",
    "hot_benefit_ratio": "SGLANG_MOE_HOT_BENEFIT_RATIO",
    "hot_min_residence_forwards": "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS",
    "gpu_residency_update": "SGLANG_MOE_GPU_RESIDENCY_UPDATE",
    "gpu_residency_max_promotions": "SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS",
    "hot_update_decode_forwards": "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS",
    "prefetch_max_candidates": "SGLANG_MOE_PREFETCH_MAX_CANDIDATES",
    "expert_fused_plan": "SGLANG_MOE_EXPERT_FUSED_PLAN",
    "insert_on_miss_stage": "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE",
    "draft_moe_nvfp4_requant": "SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT",
    "expert_doorbell": "SGLANG_MOE_EXPERT_DOORBELL",
    "expert_doorbell_mode": "SGLANG_MOE_EXPERT_DOORBELL_MODE",
    "expert_doorbell_cpu": "SGLANG_MOE_EXPERT_DOORBELL_CPU",
}
OFFLOAD_ENV_NAMES: tuple[str, ...] = tuple(ENV_NAMES.values())

_SHARED = dict(
    expert_stream=True,
    expert_file_reader="uring_direct",
    ple_file_reader="uring",
    ple_stage_before_replay=True,
    host_token_embedding=True,
    hot_gpu_mb=15360,
    pinned_host_mb=0,
    expert_host_arena=True,
    expert_copy_backend="dma",
    expert_graph_gather=True,
    hot_dynamic=True,
    hot_decay_tokens=1,
    hot_promotion_sigmas=0.0,
    hot_benefit_ratio=2.0,
    hot_min_residence_forwards=0,
    gpu_residency_update=True,
    gpu_residency_max_promotions=64,
    hot_update_decode_forwards=1,
    prefetch_max_candidates=0,
    expert_fused_plan=True,
)

# The current best, and prod's config since 2026-09-19: in-graph gather, insert-on-miss
# stage 2 and the fused planner, with overlap scheduling on. 29.30 / 29.88 tok/s median at
# NEXTN-3 on divix01 (#34). Needs decode CUDA graphs at batch size 1; pair it with
# --speculative-algorithm NEXTN, which it does not set.
GRAPH_GATHER_PRESET = MoeOffloadPreset(
    **_SHARED,
    insert_on_miss_stage=2,
    draft_moe_nvfp4_requant=True,
)

# Experimental and unmeasured on this stack. The side-thread doorbell copier on the
# graph-gather base, with insert-on-miss stage 1 because stage 2 refuses it. The doorbell
# requires overlap scheduling off (set automatically), no speculative decoding, no TP, PP,
# DP or DP attention, and its spin core (71) in the process's allowed CPUs.
DOORBELL_PRESET = MoeOffloadPreset(
    **_SHARED,
    insert_on_miss_stage=1,
    expert_doorbell=True,
    expert_doorbell_mode="current",
    expert_doorbell_cpu=71,
)

PRESETS: dict[str, MoeOffloadPreset | None] = {
    "off": None,
    "graph-gather": GRAPH_GATHER_PRESET,
    "doorbell": DOORBELL_PRESET,
}


class ResolvedOffloadEnv(msgspec.Struct, frozen=True):
    """The offload variables after the preset and the derivations are applied."""

    # Every offload variable in play, set explicitly or filled here.
    effective: dict[str, str]
    # The variables this resolution sets.
    filled: dict[str, str]
    # Preset values an explicitly set variable replaced.
    overridden: dict[str, str]


def _env_string(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _value(values: Mapping[str, str], name: str) -> Any:
    descriptor = getattr(envs, name)
    return descriptor.parse(values[name]) if name in values else descriptor.default


def preset_env(preset: MoeOffloadPreset) -> dict[str, str]:
    """The variables ``preset`` sets, as environment strings."""
    return {
        ENV_NAMES[field]: _env_string(value)
        for field in preset.__struct_fields__
        if (value := getattr(preset, field)) is not None
    }


def explicit_offload_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The offload variables already set in ``environ``."""
    return {name: environ[name] for name in OFFLOAD_ENV_NAMES if name in environ}


def _insert_on_miss(values: Mapping[str, str]) -> bool:
    return _value(values, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE") >= 1


def _residency_update(values: Mapping[str, str]) -> bool:
    return _value(values, "SGLANG_MOE_GPU_RESIDENCY_UPDATE")


# (condition, variable, value, why). Ordered: a later rule reads what an earlier one derived.
_DERIVATIONS = (
    (_insert_on_miss, "SGLANG_MOE_GPU_RESIDENCY_UPDATE", "1", "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE>=1"),
    (_insert_on_miss, "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS", "1", "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE>=1"),
    (_residency_update, "SGLANG_MOE_HOT_DYNAMIC", "1", "SGLANG_MOE_GPU_RESIDENCY_UPDATE=1"),
)


def resolve_offload_env(
    preset: MoeOffloadPreset | None, explicit: Mapping[str, str]
) -> ResolvedOffloadEnv:
    """Merge explicit variables over ``preset``, then fill what the derivations force."""
    wanted = {} if preset is None else preset_env(preset)
    filled = {name: value for name, value in wanted.items() if name not in explicit}
    overridden = {
        name: value
        for name, value in wanted.items()
        if name in explicit and _value(explicit, name) != _value(wanted, name)
    }
    effective = {**explicit, **filled}
    for condition, name, value, why in _DERIVATIONS:
        if not condition(effective):
            continue
        if name in explicit:
            if _value(explicit, name) != _value({name: value}, name):
                raise ValueError(f"{why} requires {name}={value}, but it is set to {explicit[name]}")
            continue
        effective[name] = value
        filled[name] = value
    return ResolvedOffloadEnv(effective=effective, filled=filled, overridden=overridden)


def needs_overlap_off(values: Mapping[str, str]) -> bool:
    """Whether these variables require ``--disable-overlap-schedule``."""
    if _value(values, "SGLANG_MOE_EXPERT_DOORBELL"):
        return True
    # memory_hook.handle_offload_compatibility enforces the same rule for the hot cache.
    return _value(values, "SGLANG_MOE_HOT_GPU_MB") > 0 and not (
        _value(values, "SGLANG_MOE_EXPERT_GRAPH_GATHER") and _residency_update(values)
    )


def check_offload_config(
    values: Mapping[str, str],
    *,
    speculative: bool,
    decode_graphs_disabled: bool,
    decode_max_bs: int | None,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    dp_attention: bool,
    allowed_cpus: Collection[int],
) -> None:
    """Refuse combinations that would otherwise fail after the weight load, or silently."""
    if _value(values, "SGLANG_MOE_HOT_GPU_MB") > 0 and not _value(values, "SGLANG_MOE_EXPERT_STREAM"):
        raise ValueError("SGLANG_MOE_HOT_GPU_MB requires SGLANG_MOE_EXPERT_STREAM=1")
    if _value(values, "SGLANG_MOE_EXPERT_GRAPH_GATHER") and decode_graphs_disabled:
        raise ValueError(
            "SGLANG_MOE_EXPERT_GRAPH_GATHER=1 requires decode CUDA graphs; "
            "drop --cuda-graph-backend-decode disabled"
        )
    if _residency_update(values) and (decode_max_bs or 0) > 1:
        raise ValueError("SGLANG_MOE_GPU_RESIDENCY_UPDATE=1 requires --cuda-graph-max-bs-decode 1")
    if not _value(values, "SGLANG_MOE_EXPERT_DOORBELL"):
        return
    if speculative:
        raise ValueError(
            "SGLANG_MOE_EXPERT_DOORBELL cannot run with speculative decoding; drop "
            "--speculative-algorithm or use --moe-offload-preset graph-gather"
        )
    if tp_size > 1 or pp_size > 1 or dp_size > 1 or dp_attention:
        raise ValueError(
            "SGLANG_MOE_EXPERT_DOORBELL runs one spin thread on one core and cannot run "
            "with tensor, pipeline or data parallelism, or DP attention"
        )
    if _value(values, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE") == 2:
        raise ValueError(
            "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 cannot run with SGLANG_MOE_EXPERT_DOORBELL: "
            "the doorbell thread can write a slot after stage 2 has committed it"
        )
    cpu = _value(values, "SGLANG_MOE_EXPERT_DOORBELL_CPU")
    if cpu not in allowed_cpus:
        raise ValueError(
            f"SGLANG_MOE_EXPERT_DOORBELL_CPU={cpu} is outside this process's allowed CPUs "
            f"({min(allowed_cpus)}-{max(allowed_cpus)}), so the spin thread would run "
            "unpinned; set it to an allowed core"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Sync, then RUN_CPU `test/registered/unit/layers/moe/test_offload_presets.py`
Expected: all pass (13 tests, subtests included).

If `test_graph_gather_preset_is_the_prod_offload_env` fails on a value, the preset and the prod script disagree. Re-read the prod script's env block (`ssh divix01 'sed -n 44,90p /data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh'`) and fix the preset, never the literal.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/offload_presets.py test/registered/unit/layers/moe/test_offload_presets.py
git commit -F - <<'EOF'
feat(moe): typed offload presets with derived settings and early validation

MoeOffloadPreset (msgspec) documents every offload variable a preset sets.
GRAPH_GATHER_PRESET reproduces prod's offload config; DOORBELL_PRESET is the
doorbell copier on the same base, experimental and unmeasured.
resolve_offload_env merges explicit variables over a preset and derives what
insert-on-miss and the residency update force; check_offload_config refuses
combinations that would otherwise fail after the weight load, or pin the
doorbell thread nowhere.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ABiGtddQbhsXzYFnNHqhxR
EOF
```

---

### Task 3: The `--moe-offload-preset` flag and its resolution hook

**Files:**
- Modify: `python/sglang/srt/arg_groups/fields/exec_.py`, in `class ExecOffload` after the `ple_offload_dir` field
- Create: `python/sglang/srt/arg_groups/moe_offload_hook.py`
- Modify: `python/sglang/srt/arg_groups/pipeline.py`, before `handle_offload_compatibility` (currently lines 92-94)
- Test: `test/registered/unit/layers/moe/test_offload_presets.py` (append)

**Interfaces:**
- Consumes: everything Task 2 produces.
- Produces: `handle_moe_offload_preset(server_args: Any) -> None`, and the server arg `moe_offload_preset: str` (`"off"`, `"graph-gather"` or `"doorbell"`).

- [ ] **Step 1: Write the failing hook tests**

Append to `test/registered/unit/layers/moe/test_offload_presets.py` (add the imports at the top of the file, next to the existing ones):

```python
import os
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import moe_offload_hook
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
```

```python
class TestPresetHook(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        for target, replacement in (
            ("resolving_view", lambda args: args),
            ("declare_resolution", self.record_declaration),
        ):
            patcher = patch.object(moe_offload_hook, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        affinity = patch.object(moe_offload_hook.os, "sched_getaffinity", return_value=set(range(72)))
        affinity.start()
        self.addCleanup(affinity.stop)
        self.declared = []

    def record_declaration(self, server_args, source, **fields):
        self.declared.append(fields)

    def args(self, preset, **changes):
        values = dict(
            moe_offload_preset=preset,
            disable_overlap_schedule=False,
            speculative_algorithm=None,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="breakable", max_bs=1),
                prefill=PhaseConfig(backend="disabled"),
            ),
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_graph_gather_sets_every_unset_variable_and_keeps_overlap(self):
        os.environ["SGLANG_MOE_HOT_GPU_MB"] = "12288"
        moe_offload_hook.handle_moe_offload_preset(self.args("graph-gather", speculative_algorithm="NEXTN"))
        expected = dict(presets.preset_env(presets.GRAPH_GATHER_PRESET), SGLANG_MOE_HOT_GPU_MB="12288")
        self.assertEqual(presets.explicit_offload_env(os.environ), expected)
        self.assertEqual(self.declared, [])

    def test_doorbell_turns_overlap_off(self):
        moe_offload_hook.handle_moe_offload_preset(self.args("doorbell"))
        self.assertEqual(self.declared, [{"disable_overlap_schedule": True}])
        self.assertEqual(envs.SGLANG_MOE_EXPERT_DOORBELL_CPU.get(), 71)

    def test_off_with_no_offload_env_touches_nothing(self):
        moe_offload_hook.handle_moe_offload_preset(self.args("off"))
        self.assertEqual(presets.explicit_offload_env(os.environ), {})
        self.assertEqual(self.declared, [])

    def test_a_refusal_names_the_preset(self):
        with self.assertRaisesRegex(ValueError, "--moe-offload-preset doorbell: .*speculative"):
            moe_offload_hook.handle_moe_offload_preset(self.args("doorbell", speculative_algorithm="NEXTN"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Sync, then RUN_CPU `test/registered/unit/layers/moe/test_offload_presets.py`
Expected: collection error, `ImportError: cannot import name 'moe_offload_hook'`.

- [ ] **Step 3: Write the hook**

Create `python/sglang/srt/arg_groups/moe_offload_hook.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for ``--moe-offload-preset``."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.environ import envs
from sglang.srt.layers.moe import offload_presets
from sglang.srt.model_executor.cuda_graph_config import Backend

logger = logging.getLogger(__name__)


def handle_moe_offload_preset(server_args: Any) -> None:
    """Fill unset offload variables from the preset, derive forced ones, refuse invalid setups.

    Runs in the launcher before the scheduler and workers are spawned, so they
    inherit every variable set here.
    """
    cfg = resolving_view(server_args)
    name = cfg.moe_offload_preset
    resolved = offload_presets.resolve_offload_env(
        offload_presets.PRESETS[name], offload_presets.explicit_offload_env(os.environ)
    )
    for env_name, value in resolved.filled.items():
        getattr(envs, env_name).set(value)
        logger.info("MoE offload preset %s sets %s=%s", name, env_name, value)
    for env_name, value in resolved.overridden.items():
        logger.info(
            "MoE offload preset %s: %s=%s set explicitly, preset value %s",
            name,
            env_name,
            os.environ[env_name],
            value,
        )
    if offload_presets.needs_overlap_off(resolved.effective) and not cfg.disable_overlap_schedule:
        declare_resolution(server_args, "handle_moe_offload_preset", disable_overlap_schedule=True)
        logger.info("MoE offload preset %s turns overlap scheduling off", name)
    graph = cfg.cuda_graph_config
    try:
        offload_presets.check_offload_config(
            resolved.effective,
            speculative=cfg.speculative_algorithm is not None,
            decode_graphs_disabled=graph is not None and graph.decode.backend == Backend.DISABLED,
            decode_max_bs=None if graph is None else graph.decode.max_bs,
            tp_size=cfg.tp_size,
            pp_size=cfg.pp_size,
            dp_size=cfg.dp_size,
            dp_attention=cfg.enable_dp_attention,
            allowed_cpus=os.sched_getaffinity(0),
        )
    except ValueError as error:
        raise ValueError(f"--moe-offload-preset {name}: {error}") from error
```

- [ ] **Step 4: Declare the flag**

In `python/sglang/srt/arg_groups/fields/exec_.py`, inside `class ExecOffload`, directly after the `ple_offload_dir` field (which ends with `] = None`):

```python
    moe_offload_preset: A[
        str,
        Arg(
            help="Named MoE expert-offload configuration that fills every offload "
            "SGLANG_MOE_* / SGLANG_QWEN4_* variable left unset. 'graph-gather' is the "
            "current best (in-graph gather, insert-on-miss stage 2, fused planner; "
            "tuned for a 32 GB RTX 5090). 'doorbell' is experimental and unmeasured "
            "(side-thread copier; no speculative decoding; turns overlap scheduling "
            "off). 'off' sets nothing. Explicitly set variables win. See "
            "sglang.srt.layers.moe.offload_presets.",
            choices=["off", "graph-gather", "doorbell"],
        ),
    ] = "off"
```

- [ ] **Step 5: Wire the hook into the pipeline**

In `python/sglang/srt/arg_groups/pipeline.py`, replace

```python
    from sglang.srt.arg_groups.memory_hook import handle_offload_compatibility

    handle_offload_compatibility(server_args)
```

with

```python
    from sglang.srt.arg_groups.moe_offload_hook import handle_moe_offload_preset

    # Before the offload checks, which read the variables the preset fills.
    handle_moe_offload_preset(server_args)
    from sglang.srt.arg_groups.memory_hook import handle_offload_compatibility

    handle_offload_compatibility(server_args)
```

- [ ] **Step 6: Run the new tests and the server-args suite**

Sync, then:
RUN_CPU `test/registered/unit/layers/moe/test_offload_presets.py`
Expected: all pass (17 tests).

RUN_CPU `test/registered/unit/server_args test/registered/unit/test_nvfp4_expert_offload.py`
Expected: the same result as on the parent commit. Record the parent's result first: stash nothing; run it on a sync of `HEAD~1` if unsure. A new failure that names `moe_offload_preset` means a field registry or ratchet needs the new field. Follow that test's message (for example add the field to the list it enumerates); do not change the test's intent.

- [ ] **Step 7: Check the CLI end to end without starting a server**

```bash
timeout 300 ssh -n -o BatchMode=yes divix01 "cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-presets && CUDA_VISIBLE_DEVICES= taskset -c 0-63 env PYTHONPATH=\$PWD/python /data/models/slang/.venv/bin/python -c \"
import argparse
from sglang.srt.server_args import ServerArgs
parser = argparse.ArgumentParser(); ServerArgs.add_cli_args(parser)
print(parser.parse_args(['--model-path', 'x', '--moe-offload-preset', 'doorbell']).moe_offload_preset)
\""
```

Expected: prints `doorbell`. If `ServerArgs.add_cli_args` has a different name, find the parser entry point with `grep -n "def add_cli_args\|def prepare_server_args" python/sglang/srt/server_args.py` and use it.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/srt/arg_groups/moe_offload_hook.py python/sglang/srt/arg_groups/fields/exec_.py python/sglang/srt/arg_groups/pipeline.py test/registered/unit/layers/moe/test_offload_presets.py
git commit -F - <<'EOF'
feat(moe): --moe-offload-preset fills the offload variables at argument resolution

The hook runs before the offload checks: it sets every offload variable the
user left unset from the selected preset, logs each value and each explicit
override, turns overlap scheduling off where a rule requires it, and refuses
invalid combinations before the weight load. The default, off, touches nothing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ABiGtddQbhsXzYFnNHqhxR
EOF
```

---

### Task 4: GPU startup smoke test of both presets

Correctness only, no timing. This settles whether the doorbell accepts insert-on-miss stage 1 and the fused planner.

**Files:**
- Create (divix01 only, not in repo): `/data/models/slang/nvfp4-work/cc-expert-prediction/preset-smoke.sh`

**Interfaces:**
- Consumes: the flag from Task 3.

- [ ] **Step 1: Write the smoke script**

Create it locally in the scratchpad as `preset-smoke.sh`, then `scp` it to `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/preset-smoke.sh`:

```bash
#!/usr/bin/env bash
# Start the server with one --moe-offload-preset, send one prompt, stop it. Not a timed arm.
set -uo pipefail
preset=$1; shift
C=/data/models/slang/nvfp4-work/cc-expert-prediction
W=/data/models/slang/nvfp4-work
wt=$C/wt-presets
port=31045
out=$C/matrix/preset-smoke-$preset-$(date +%Y%m%d-%H%M%S)
mkdir -p "$out"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy"; exit 1; }
ss -ltn | grep -qE ':(7867|3104[0-7]) ' && { echo "port busy"; exit 1; }
model=/mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
# Site paths the preset deliberately leaves to the launcher: copy them from the prod script.
eval "$(grep -E '^(expert_cache|expert_seed|ple_cache|cache_model_path|flashinfer_overlay)=' $W/run-nvfp4-e16c-public.sh)"
flock -n $W/cc-gpu.lock taskset -c 0-70 env \
  PYTHONPATH="$flashinfer_overlay:$wt/python" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 SGLANG_MOE_EXPERT_FILE_DIR="$expert_cache" \
  SGLANG_MOE_HOT_SEED="$expert_seed" SGLANG_FILE_CACHE_MODEL_PATH="$cache_model_path" \
  SGLANG_MOE_HOT_METRICS_FILE="$out/hot-cache.metrics.jsonl" \
  /data/models/slang/.venv/bin/sglang serve --model-type llm --model-path "$model" --tp 1 \
    --fp4-gemm-backend flashinfer_cutlass --moe-runner-backend flashinfer_cutlass --moe-a2a-backend none \
    --cpu-offload-gb 80 --ple-offload-embedding --ple-offload-backend file --ple-offload-dir "$ple_cache" \
    --page-size 64 --mamba-track-interval 64 --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
    --context-length 40000 --max-total-tokens 40000 --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-running-requests 1 --max-mamba-cache-size 1 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.95 \
    --disable-radix-cache --language-model-only --cuda-graph-backend-decode breakable --cuda-graph-bs-decode 1 \
    --cuda-graph-max-bs-decode 1 --cuda-graph-backend-prefill disabled --disable-flashinfer-autotune \
    --skip-server-warmup --weight-loader-drop-cache-after-load --expert-distribution-recorder-mode per_pass \
    --moe-offload-preset "$preset" "$@" --host 127.0.0.1 --port $port > "$out/server.log" 2>&1 &
lpid=$!
for _ in $(seq 1 900); do curl -sf -o /dev/null http://127.0.0.1:$port/health && break; kill -0 $lpid 2>/dev/null || break; sleep 1; done
curl -sf http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":64}' \
  | python3 -c 'import json,sys; print("REPLY", json.load(sys.stdin)["choices"][0]["message"])' || echo "NO REPLY"
grep -E "MoE offload preset|Error|error:" "$out/server.log" | head -40
kill -TERM $(pgrep -P $lpid) $lpid 2>/dev/null; wait $lpid 2>/dev/null
for _ in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
echo "census after: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
echo "SMOKE_DONE $out"
```

`taskset -c 0-70` rather than `0-63`: the doorbell preset refuses a spin core outside the allowed set, and core 71 is the doorbell's core. 64-70 stay idle because nothing else is scheduled on them. This matches how production runs the doorbell.

- [ ] **Step 2: Run graph-gather with NEXTN**

Sync, then:

```bash
timeout 1200 ssh -n -o BatchMode=yes divix01 'bash /data/models/slang/nvfp4-work/cc-expert-prediction/preset-smoke.sh graph-gather --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3'
```

Expected: `MoE offload preset graph-gather sets ...` lines for all 22 variables, a coherent `REPLY`, no error, and an empty census after.

- [ ] **Step 3: Run doorbell**

```bash
timeout 1200 ssh -n -o BatchMode=yes divix01 'bash /data/models/slang/nvfp4-work/cc-expert-prediction/preset-smoke.sh doorbell'
```

Expected: a line saying overlap scheduling was turned off, a coherent `REPLY`, and an empty census after. Also confirm the spin thread really landed on core 71: `grep -o "spin_cpu[^,]*" <out>/server.log` (or the hot-cache metrics file) should report `71`.

- [ ] **Step 4: If the doorbell refuses a setting, drop it from the preset**

If the server log shows a startup `ValueError` naming insert-on-miss or the fused planner together with the doorbell:
1. In `offload_presets.py`, change `DOORBELL_PRESET` to override that field (`insert_on_miss_stage=0`, or `expert_fused_plan=False`). Add one line to its comment block saying which startup check refuses it, citing the file and line from the log.
2. Update `test_doorbell_preset_departs_from_graph_gather_only_where_the_doorbell_forces_it` to add that variable to the expected set.
3. If stage 0 is chosen, also check `SGLANG_MOE_GPU_RESIDENCY_UPDATE` and `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS`. Stage 1 required them; stage 0 does not. Keep them unless startup refuses them.
4. Rerun RUN_CPU `test/registered/unit/layers/moe/test_offload_presets.py` and Step 3.
5. Commit:

```bash
git add python/sglang/srt/layers/moe/offload_presets.py test/registered/unit/layers/moe/test_offload_presets.py
git commit -F - <<'EOF'
fix(moe): drop what the doorbell refuses from its preset

<one line: which setting, and the startup check that refuses it>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ABiGtddQbhsXzYFnNHqhxR
EOF
```

If both smoke tests pass unchanged, there is nothing to commit in this task.

---

### Task 5: Switch prod to the flag and record the presets

Prod is not started in this task.

**Files:**
- Modify (divix01, not in repo): `/data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh`
- Create (divix01): `/data/models/slang/nvfp4-work/prod-presets-20260919/` worktree and its patch
- Modify: `MOE_EXPERT_TRANSFER.md`, section "What production runs today, and the delta"

**Interfaces:**
- Consumes: the flag (Task 3) and the smoke results (Task 4).

- [ ] **Step 1: Build the prod worktree from the committed code**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4
S=/tmp/claude-1000/-home-dimitri-data-divix-sglang-nvfp4/b33df7b3-ec05-46fa-a1d0-ee10bc177ba2/scratchpad/he
W=/data/models/slang/nvfp4-work
git status --short -- python scripts test   # must print nothing: prod runs committed code only
git diff 797be6f678 HEAD -- python/ scripts/ test/ > $S/prod-presets-20260919.patch
scp -q -o BatchMode=yes $S/prod-presets-20260919.patch divix01:$W/
timeout 120 ssh -n -o BatchMode=yes divix01 "cd $W && [ ! -e prod-presets-20260919 ] && git -C /data/models/slang/sglang worktree add -q --detach $W/prod-presets-20260919 797be6f678 && cd prod-presets-20260919 && git apply --index ../prod-presets-20260919.patch && grep -c GRAPH_GATHER_PRESET python/sglang/srt/layers/moe/offload_presets.py"
```

Expected: prints a count of at least `2`.

- [ ] **Step 2: Rewrite the prod script's offload block**

Back up first, then remove the 22 variables the preset now sets and add the flag:

```bash
timeout 60 ssh -n -o BatchMode=yes divix01 'cd /data/models/slang/nvfp4-work && f=run-nvfp4-e16c-public.sh && b=$f.bak-pre-presets-20260919 && [ ! -e $b ] && cp -p $f $b && \
sed -i -E \
  -e "s|^worktree=.*|worktree=/data/models/slang/nvfp4-work/prod-presets-20260919|" \
  -e "/^    (SGLANG_MOE_EXPERT_STREAM|SGLANG_MOE_EXPERT_FILE_READER|SGLANG_QWEN4_PLE_FILE_READER|SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY|SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING|SGLANG_MOE_HOT_GPU_MB|SGLANG_MOE_PINNED_HOST_MB|SGLANG_MOE_EXPERT_HOST_ARENA|SGLANG_MOE_EXPERT_GRAPH_GATHER|SGLANG_MOE_EXPERT_FUSED_PLAN|SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT|SGLANG_MOE_HOT_DYNAMIC|SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS|SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE|SGLANG_MOE_HOT_DECAY_TOKENS|SGLANG_MOE_HOT_PROMOTION_SIGMAS|SGLANG_MOE_HOT_BENEFIT_RATIO|SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS|SGLANG_MOE_GPU_RESIDENCY_UPDATE|SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS|SGLANG_MOE_PREFETCH_MAX_CANDIDATES|SGLANG_MOE_EXPERT_COPY_BACKEND)=/d" \
  -e "s|^        --tp 1 \\\\$|        --tp 1 \\\\\n        --moe-offload-preset graph-gather \\\\|" \
  $f && diff $b $f; bash -n $f && echo SYNTAX_OK'
```

Expected: the diff shows exactly 22 removed `SGLANG_*` lines, the changed `worktree=` line and one added `--moe-offload-preset graph-gather` line, followed by `SYNTAX_OK`.

- [ ] **Step 3: Prove the effective environment is unchanged**

Resolve the new script's settings without starting a server, and compare them with the backup's explicit values:

```bash
timeout 300 ssh -n -o BatchMode=yes divix01 'cd /data/models/slang/nvfp4-work && python3 - <<"PYEOF"
import re, subprocess
def env_of(path):
    text = open(path).read()
    return dict(re.findall(r"^\s+((?:SGLANG_MOE|SGLANG_QWEN4|SGLANG_ENABLE_QWEN4|SGLANG_ENABLE_DRAFT)[A-Z0-9_]*)=(\S+) \\\\$", text, re.M))
old = env_of("run-nvfp4-e16c-public.sh.bak-pre-presets-20260919")
new = env_of("run-nvfp4-e16c-public.sh")
code = "import os,json; from sglang.srt.layers.moe import offload_presets as p; r=p.resolve_offload_env(p.PRESETS[\"graph-gather\"], p.explicit_offload_env(os.environ)); print(json.dumps(r.effective))"
out = subprocess.run(["env", "CUDA_VISIBLE_DEVICES=", "PYTHONPATH=prod-presets-20260919/python", *[f"{k}={v}" for k, v in new.items()], "taskset", "-c", "0-63", "/data/models/slang/.venv/bin/python", "-c", code], capture_output=True, text=True, check=True).stdout
import json
effective = json.loads(out.strip().splitlines()[-1])
missing = {k: v for k, v in old.items() if effective.get(k) not in (v,)}
print("MISMATCH" if missing else "IDENTICAL", missing)
PYEOF'
```

Expected: `IDENTICAL {}`. Values the backup quotes, such as `"$metrics"`, are site paths the new script still sets explicitly. If they appear in the mismatch, the regex caught a quoted variable, so compare them by eye. Any real `SGLANG_*` value difference means the preset is wrong: fix it in Task 2's module and repeat.

- [ ] **Step 4: Record the presets in `MOE_EXPERT_TRANSFER.md`**

In "What production runs today, and the delta":
1. Change the worktree paragraph's name from `prod-stage2fix-20260919` to `prod-presets-20260919`, now at the commit that adds the presets.
2. Add this paragraph directly after the settings table:

```markdown
**Since 2026-09-19 the script selects these settings with `--moe-offload-preset
graph-gather`** (`python/sglang/srt/layers/moe/offload_presets.py`) instead of
setting 22 variables by hand; the table above is that preset's content. An
explicitly set variable still wins, and the server logs one `MoE offload preset`
line per value it fills. `--moe-offload-preset doorbell` is the doorbell copier
on the same base (insert-on-miss stage 1, overlap off, no speculative decoding),
smoke-tested for startup only, never timed. Backup of the explicit script:
`.bak-pre-presets-20260919`.
```

3. Commit:

```bash
git add MOE_EXPERT_TRANSFER.md
git commit -F - <<'EOF'
docs(moe): record prod's switch to --moe-offload-preset graph-gather

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ABiGtddQbhsXzYFnNHqhxR
EOF
```

- [ ] **Step 5: Report**

Tell the user: the prod script now uses the flag, the worktree and backup names, that the effective environment is identical, the doorbell smoke outcome (and anything dropped from its preset), and that prod was not started.
