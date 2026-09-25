# MoE Expert Plugins (format + row source seams) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the expert *format* (row schema and dense sources) and the *disk→RAM row source* plugins of the NVFP4 MoE expert-streaming framework, and fix the three framework bugs that fire at large expert sizes. NVFP4 production behaviour stays byte-identical.

**Architecture:** A new `ExpertFormat` protocol (`layers/moe/expert_format.py`) supplies per-tensor `ExpertTensorSpec`s and dynamic dense sources. `DenseLayerFormat` is exactly today's behaviour and is the default. A new `ExpertRowSource` protocol (`layers/moe/expert_row_source.py`) fills host rows. `ExpertFileRowReader` (io_uring over expert files) and `TensorRowSource` (the mmap/tensor fallback) both conform to it. The streamer, the pinned host tier, and the hot cache read shapes from specs and route every host read through one batched `read_host_rows`. On top of that the plan fixes three RAM-tier bugs: slab rounding, O(capacity) LRU, and the out-of-bounds copy. It also adds spec-only formats, a chunked `gather_experts` entry capped by `max_gather_rows`, the `SGLANG_MOE_EXPERT_ROW_SOURCE` knob, and a verify-only `FileTensorCacheGroup.open_verified`.

**Tech Stack:** Python 3.13, PyTorch 2.13 (divix01 venv), Triton, io_uring reader (`sglang.kernels.ops.io.uring_file_reader`), unittest/pytest.

**Spec:** `/home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41/.superpowers/plans-research/dsv41-phase3/D-framework-format-seam.md`. It was written against `dsv41`, whose framework files equal merge-base `b59edf2dc4`. This plan targets `cc/moe-expert-plugins`, which was cut from `e54c84a7c6` and already contains `844bb9d7a5` (in-place prefill staging) and `e1d227a4bd` (slot floor). Every line reference below was re-derived from the `cc/moe-expert-plugins` worktree.

## Global Constraints

Every task's requirements include this section. Implementers see only their own task, so each task repeats the commands it needs.

**Places and branches**
- Laptop worktree: `/home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins`, branch `cc/moe-expert-plugins`. Edit and commit here.
- divix01 test worktree (`$DWT`): `/data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins`, branch `cc/moe-expert-plugins` tracking `shared/cc/moe-expert-plugins`. Task 0 creates it. Never edit files there; only `git pull --ff-only`.
- The laptop has no usable python for these tests. Never run pytest on the laptop.

**Git rules**
- Never commit to `master`. Never merge into it or into `dsv41` from this plan (`dsv41` merges this branch later, in another plan).
- Stage files by name. Never `git add -A`, never `git stash`, never `--amend`, never force-push.
- Each task makes two commits: first the failing tests, then the implementation. The red commit is pushed so divix01 can run it.
- Commit messages end with exactly these two lines:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF
  ```

**divix01 rules**
- CPU jobs run under `taskset -c 0-63` with `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` and `CUDA_VISIBLE_DEVICES=`.
- Cores 64–71 are reserved; core 71 is production's doorbell spin core.
- GPU jobs run only inside Task 10, under `taskset -c 32-63`, holding `/data/models/slang/nvfp4-work/cc-gpu.lock`. No other task touches the GPU or that lock file.
- Never relaunch production. Do no bulk reads of `/mnt/nvme1` or `/mnt/nvme2`.

**Test file rules**
- Test files end with exactly one of the two `__main__` blocks from `test/README.md`. This plan uses the unittest one:
  ```python
  if __name__ == "__main__":
      unittest.main()
  ```
- CPU test files call `register_cpu_ci(est_time=10, suite="base-a-test-cpu")`. CUDA test files call `register_cuda_ci(est_time=60, stage="base-a", runner_config="1-gpu-small")`. Both come from `sglang.test.ci.ci_register`, as the existing expert tests use them.
- A CUDA test is still written in its task, but it is decorated `@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")` and first runs in Task 10.
- **Red steps.** Every red step names the tests that fail and why. A CUDA test is never run red, because no task before Task 10 may use the GPU. Task 10 Step 6 records the one CUDA red that matters, the out-of-bounds read at the base commit. A CPU test that pins existing behaviour and therefore passes at its red commit is named as such, together with the mutation that would make it fail.

**NVFP4 production must be unchanged.** The production config is:
- host arena (`SGLANG_MOE_EXPERT_HOST_ARENA=1`), graph gather, the DMA copy backend;
- `SGLANG_MOE_PINNED_HOST_MB=0`;
- dynamic residency with the GPU residency update;
- eager prefill through `_gather_cached` / `_copy_source_rows`.

The code rules that follow from this:
- Source lookup stays dynamic, because the arena rebinds `layer.<name>.data` after the streamer exists.
- A dense spec's `row_shape` equals `source.shape[1:]` exactly.
- `_gather_graph` and `_check_graph_sources` are not edited.
- `ExpertGatherStats` is constructed positionally, so new fields are trailing and defaulted.
- The attribute `_nvfp4_expert_streamer` is not renamed.
- `model_runner.py` is a frozen core file. Its only edits are the two discovery-helper swaps in Task 1, made after reading `.claude/skills/large-class-style/SKILL.md`.
- New `SGLANG_*` env vars follow `.claude/skills/env-var-conventions/SKILL.md`.

**Standard divix01 CPU test command.** Replace `<TESTS>` with space-separated test paths or node ids:
```bash
git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v <TESTS>'
```

**CPU regression suite (`CPU_SUITE`).** Every task ends by running this suite plus its own new files:
```
test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py
```

- On CPU, most of these tests skip with "CUDA is required". The ones that run are the file-reader, route-plan, offload-config, file-cache, residency and prediction tests.
- One failure is known and pre-existing: `test_expert_transfer.py::TestExpertRowCopySubmission::test_gpu_submission_uses_fallback_for_nonpinned_sources`, which raises `torch.AcceleratorError: ... no CUDA-capable device`. It is not CUDA-gated. On 2026-09-18 the planner ran this exact suite on divix01, over a `git archive` of `e54c84a7c6`, and got `1 failed, 170 passed, 268 skipped`.
- Any other failure is a regression.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `python/sglang/srt/layers/moe/expert_format.py` (new) | `ExpertTensorSpec`, `ExpertFormat` protocol, `DenseLayerFormat`, discovery helpers, knob resolution, graph-support guard | 1, 3, 6 |
| `python/sglang/srt/layers/moe/expert_row_source.py` (new) | `HostSlotLayout`, `RowReadStats`, `ReadTicket`, `CompletedReadTicket`, `ExpertRowSource` protocol, `SynchronousSubmit`, `TensorRowSource` | 2 |
| `python/sglang/srt/layers/moe/expert_host_tier.py` (new) | `PinnedSlotLRU`, page-aligned registered slab allocation, `PinnedGatherResult` | 5 |
| `python/sglang/test/moe_expert_fakes.py` (new) | test doubles: `CountingRowSource`, `SpecOnlyFormat` | 2, 6 |
| `python/sglang/srt/layers/moe/expert_stream.py` | streamer: format/row-source kwargs, spec shapes, batched host reads, read stats, pinned tier rewrite, `gather_experts` | 1, 3, 4, 5, 6, 7 |
| `python/sglang/srt/layers/moe/expert_file_reader.py` | `ExpertFileRowReader` conforms to `ExpertRowSource`, gains `from_group` | 2 |
| `python/sglang/srt/layers/moe/expert_hot_cache.py` | specs in `ExpertHotCache`, discovery, file-bytes property, read counters, spec-only promotions, graph guard | 1, 4, 6 |
| `python/sglang/srt/layers/moe/expert_host_arena.py` | discovery helper, refuses formats without arena support | 1, 6 |
| `python/sglang/srt/model_executor/model_runner.py` (frozen) | two discovery-helper swaps only | 1 |
| `python/sglang/srt/environ.py` | `SGLANG_MOE_EXPERT_ROW_SOURCE` | 3 |
| `python/sglang/srt/model_loader/file_tensor_cache.py` | `FileTensorCacheGroup.open_verified` | 8 |
| `test/registered/unit/layers/moe/test_expert_format.py` (new, CPU) | format, knob and spec-only tests | 1, 3, 6 |
| `test/registered/unit/layers/moe/test_expert_row_source.py` (new, CPU) | row sources, streamer routing, read stats | 2, 3, 4 |
| `test/registered/unit/layers/moe/test_expert_host_tier.py` (new, CPU) | LRU, slabs, CPU pinned tier, out-of-bounds regression | 5 |
| `test/registered/unit/layers/moe/test_expert_gather_experts.py` (new, CPU) | chunked gathers, staging cap | 7 |
| `test/registered/unit/model_loader/test_file_tensor_cache_verified.py` (new, CPU) | verify-only open | 8 |
| `test/registered/unit/layers/moe/test_expert_plugins_cuda.py` (new, CUDA) | every CUDA test of this plan, one class per task | 1, 3, 5, 6, 7 |
| `python/sglang/srt/arg_groups/expert_stream_requirements.py` (new) | per-format server-args requirements registry; NVFP4's moved check; `eager_expert_stream_requirements` | 9 |
| `python/sglang/srt/arg_groups/memory_hook.py` | the expert-caching gate looks up the format's requirements | 9 |
| `test/registered/unit/test_expert_stream_requirements.py` (new, CPU) | gate registry, method resolution, eager-format requirements | 9 |
| `test/registered/unit/layers/moe/test_expert_tier_startup.py` (new, CPU) | pinned manager end to end, tolerant tier options, inclusive hot-slot clamp | 9 |
| `test/registered/unit/test_nvfp4_expert_offload.py` | three ModelOpt-keyed reruns of every hot-cache gate case | 9 |

---

### Task 0: Push the branch and create the divix01 test worktree

**Files:** none in the repository.

**Interfaces:**
- Consumes: nothing.
- Produces: `$DWT` = `/data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins`, on branch `cc/moe-expert-plugins`, tracking `shared/cc/moe-expert-plugins`. It is a worktree of `/data/models/slang/sglang`, set up the same way as `wt-dsv41`, which is a worktree of that repo on `dsv41...shared/dsv41` with remote `shared` = `/data/models/slang/nvfp4-work/remotes/sglang-nvfp4.git`. Also produces a recorded CPU baseline.

- [ ] **Step 1: Check the laptop worktree**

Run: `git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins status --short --branch`
Then run: `git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins log -1 --format=%h -- docs/superpowers/plans/2026-09-18-moe-expert-plugins.md`

Expected:
- The first line of the status is `## cc/moe-expert-plugins`, and no tracked file is modified.
- The log prints a commit. Before dispatching Task 0, the controller commits this plan file on `cc/moe-expert-plugins`, so the branch records which plan revision was executed.

If the log prints nothing, or a tracked file is modified, stop and report.

- [ ] **Step 2: Push the branch to `shared`**

Run: `git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins push -u shared cc/moe-expert-plugins`
Expected: `* [new branch] cc/moe-expert-plugins -> cc/moe-expert-plugins`, and the branch is set up to track `shared/cc/moe-expert-plugins`.

- [ ] **Step 3: Make sure the divix01 repo has no branch of that name yet**

Run: `ssh divix01 'git -C /data/models/slang/sglang rev-parse --verify --quiet refs/heads/cc/moe-expert-plugins; echo rc=$?; test -e /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins; echo exists_rc=$?'`
Expected: `rc=1` and `exists_rc=1`. If either is 0, stop and report; do not delete anything.

- [ ] **Step 4: Fetch the branch and add the worktree**

Run:
```bash
ssh divix01 'git -C /data/models/slang/sglang fetch shared +refs/heads/cc/moe-expert-plugins:refs/remotes/shared/cc/moe-expert-plugins && git -C /data/models/slang/sglang worktree add --track -b cc/moe-expert-plugins /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins shared/cc/moe-expert-plugins && git -C /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins status --short --branch | head -1 && git -C /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins log --oneline -1'
```
Expected: `## cc/moe-expert-plugins...shared/cc/moe-expert-plugins`, and the laptop's HEAD commit.

- [ ] **Step 5: Record the CPU baseline**

Run:
```bash
ssh divix01 'mkdir -p /data/models/slang/nvfp4-work/cc-expert-prediction/logs && cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py -rf 2>&1 | tee /data/models/slang/nvfp4-work/cc-expert-prediction/logs/moe-plugins-cpu-baseline.log | tail -5'
```
Expected: `FAILED test/registered/unit/layers/moe/test_expert_transfer.py::TestExpertRowCopySubmission::test_gpu_submission_uses_fallback_for_nonpinned_sources`, and the summary `1 failed, 170 passed, 268 skipped` (the planner's run of the same base). Write the observed N (passed) and M (skipped) into the task report: they are the baseline for every later task. Any other failure means the base is not clean; stop and report.

No commit in this task.

---

### Task 1: The format seam — `ExpertTensorSpec`, `ExpertFormat`, `DenseLayerFormat`, discovery helpers, file-bytes property

This task is a pure refactor for NVFP4. The streamer, the pinned host tier and the hot cache take their shapes, dtypes and byte counts from specs instead of probing `getattr(layer, name)`. Dense sources are still looked up on every call. Streamer discovery goes through one helper. The pinned-tier gate reads a streamer property that for NVFP4 returns exactly the old layer attribute.

A spec's `residence` is fixed when the streamer is built, whereas the caches used to probe the device when they were built. That makes no difference because no production path moves a streamed tensor between CPU and CUDA after `_attach_expert_streamer`. The arena rebinds CPU to CPU, and `finalize_expert_files` only verifies.

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_format.py`
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`. Edit the imports (top), `ExpertPinnedHostCache.__init__` (`:92-140`), `ExpertPinnedHostCacheManager.from_model` (`:298-308`) and `ExpertStreamer.__init__` (`:502-568`). Add accessors after `__init__`. Also edit `_validate_sources` (`:856-881`), `_gather_cached` (`:979-1022`), `_gather_pinned_host` (`:1105-1115`) and `gather` (`:1268-1274`).
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py`. Edit the import (`:39`), `ExpertHotCache.__init__` (`:134-153`), `_prepare_promotion` (`:468-471`), `ExpertHotCacheManager.from_model` discovery (`:980-990`), and the observer's file bytes (`:2488-2490`).
- Modify: `python/sglang/srt/layers/moe/expert_host_arena.py`, in `from_model` (`:62-67`).
- Modify: `python/sglang/srt/model_executor/model_runner.py`, in `maybe_init_expert_hot_cache` (`:716-719`, `:744-748`). This is a frozen file.
- Test (create): `test/registered/unit/layers/moe/test_expert_format.py` (CPU)
- Test (create): `test/registered/unit/layers/moe/test_expert_plugins_cuda.py` (CUDA)

**Interfaces:**
- Consumes: nothing new.
- Produces (every later task relies on these exact names):
  - `sglang.srt.layers.moe.expert_format`:
    - `STREAMER_ATTRIBUTE = "_nvfp4_expert_streamer"`
    - `FILE_SOURCE_BYTES_ATTRIBUTE = "_nvfp4_file_source_bytes_per_expert"`
    - `GENERIC_ROW_SOURCE_KINDS = ("auto", "files", "tensor")`
  - `ExpertTensorSpec(name: str, row_shape: tuple[int, ...], dtype: torch.dtype, residence: Literal["host", "device"])`, a frozen dataclass with a `.row_bytes -> int` property.
  - `ExpertFormat` (Protocol):
    - attributes `key: str`, `supports_graph_gather: bool`, `supports_host_arena: bool`, `max_gather_rows: Optional[int]`;
    - `tensor_specs(layer) -> tuple[ExpertTensorSpec, ...]`;
    - `num_experts(layer) -> int`;
    - `source(layer, name) -> Optional[torch.Tensor]`;
    - `default_row_source(layer, specs, kind: str) -> Optional[ExpertRowSource]`;
    - `file_source_bytes_per_expert(layer, row_source) -> Optional[int]`.
  - `DenseLayerFormat(tensor_names)` implements the protocol with today's behaviour: `key="dense"`, both `supports_*` True, `max_gather_rows=None`.
  - `expert_streamer_of(module) -> Optional[ExpertStreamer]` and `iter_expert_streamers(model) -> Iterator[ExpertStreamer]`.
  - `ExpertStreamer(layer, tensor_names, *, layer_id=None, format: ExpertFormat | None = None)` gains:
    - attribute `.format`;
    - property `.specs -> tuple[ExpertTensorSpec, ...]`;
    - `.spec(name) -> ExpertTensorSpec`;
    - `.source(name) -> Optional[torch.Tensor]` (a dynamic lookup);
    - property `.file_source_bytes_per_expert -> Optional[int]`.

- [ ] **Step 1: Write the failing CPU tests**

Create `test/registered/unit/layers/moe/test_expert_format.py`:

```python
"""CPU tests for the expert format seam: specs, dense sources and streamer discovery."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertTensorSpec,
    expert_streamer_of,
    iter_expert_streamers,
)
from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXPERTS = 4
HIDDEN = 512
INTERMEDIATE = 256


def _nvfp4_layer(experts=EXPERTS):
    """CPU parameters with the production NVFP4 streamed shapes and dtypes."""
    shapes = {
        "w13_weight": ((experts, 2 * INTERMEDIATE, HIDDEN // 2), torch.uint8),
        "w2_weight": ((experts, HIDDEN, INTERMEDIATE // 2), torch.uint8),
        "w13_blockscale_swizzled": (
            (experts, 2 * INTERMEDIATE, HIDDEN // 16),
            torch.float8_e4m3fn,
        ),
        "w2_blockscale_swizzled": (
            (experts, HIDDEN, INTERMEDIATE // 16),
            torch.float8_e4m3fn,
        ),
        "g1_alphas": ((experts,), torch.float32),
        "g2_alphas": ((experts,), torch.float32),
    }
    generator = torch.Generator().manual_seed(3)
    layer = torch.nn.Module()
    for name, (shape, dtype) in shapes.items():
        if dtype is torch.float32:
            values = torch.rand(shape, generator=generator)
        else:
            values = torch.randint(
                0, 256, shape, dtype=torch.uint8, generator=generator
            ).view(dtype)
        setattr(layer, name, torch.nn.Parameter(values, requires_grad=False))
    return layer


def _legacy_bytes(layer, experts, host_only):
    """ExpertStreamer's byte counts as computed before formats existed."""
    total = 0
    for name in NVFP4_STREAM_TENSORS:
        tensor = getattr(layer, name).data
        if host_only and tensor.device.type != "cpu":
            continue
        total += tensor.numel() * tensor.element_size() // experts
    return total


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


def _as_bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


class TestDenseLayerFormat(unittest.TestCase):
    def test_specs_equal_source_row_shapes_on_an_nvfp4_layer(self):
        layer = _nvfp4_layer()
        specs = DenseLayerFormat(NVFP4_STREAM_TENSORS).tensor_specs(layer)
        self.assertEqual(tuple(spec.name for spec in specs), NVFP4_STREAM_TENSORS)
        for spec in specs:
            source = getattr(layer, spec.name).data
            self.assertEqual(spec.row_shape, tuple(source.shape[1:]))
            self.assertEqual(spec.dtype, source.dtype)
            self.assertEqual(spec.residence, "host")
            self.assertEqual(
                spec.row_bytes, source.numel() * source.element_size() // EXPERTS
            )

    def test_streamer_byte_counts_match_the_legacy_formula(self):
        layer = _nvfp4_layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        self.assertIsInstance(streamer.format, DenseLayerFormat)
        self.assertEqual(streamer.num_experts, EXPERTS)
        self.assertEqual(
            streamer.bytes_per_expert, _legacy_bytes(layer, EXPERTS, host_only=False)
        )
        self.assertEqual(
            streamer.host_bytes_per_expert,
            _legacy_bytes(layer, EXPERTS, host_only=True),
        )
        self.assertEqual(
            streamer.specs, DenseLayerFormat(NVFP4_STREAM_TENSORS).tensor_specs(layer)
        )
        self.assertEqual(streamer.spec("g1_alphas").row_shape, ())

    def test_sources_are_looked_up_on_every_call(self):
        layer = _nvfp4_layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        replacement = layer.w13_weight.data.clone()
        # ExpertHostArena.bind rebinds parameters exactly like this.
        layer.w13_weight.data = replacement
        self.assertEqual(
            streamer.source("w13_weight").data_ptr(), replacement.data_ptr()
        )
        plain = torch.nn.Module()
        plain.rows = torch.zeros(4, 3)
        streamer = ExpertStreamer(plain, ("rows",))
        plain.rows = torch.ones(4, 3)
        self.assertTrue(torch.equal(streamer.source("rows"), torch.ones(4, 3)))

    def test_file_source_bytes_follow_the_layer_attribute(self):
        layer = _nvfp4_layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        self.assertIsNone(streamer.file_source_bytes_per_expert)
        layer._nvfp4_file_source_bytes_per_expert = 12
        self.assertEqual(streamer.file_source_bytes_per_expert, 12)
        # ExpertHostArena.bind drops the attribute like this.
        layer.__dict__.pop("_nvfp4_file_source_bytes_per_expert")
        self.assertIsNone(streamer.file_source_bytes_per_expert)

    def test_invalid_dense_sources_keep_their_errors(self):
        cases = (
            ({"a": torch.zeros(4, 2)}, ("a", "b"), "'b' is missing"),
            ({"a": torch.tensor(1.0)}, ("a",), "has no expert dimension"),
            ({"a": torch.zeros(0, 4)}, ("a",), "has no expert rows"),
            ({"a": torch.zeros(4, 6)[:, ::2]}, ("a",), "must be contiguous"),
            ({"a": torch.zeros(4, 2, device="meta")}, ("a",), "unsupported device"),
            (
                {"a": torch.zeros(4, 2), "b": torch.zeros(5, 2)},
                ("a", "b"),
                "expert count mismatch",
            ),
        )
        for tensors, names, message in cases:
            layer = torch.nn.Module()
            for name, tensor in tensors.items():
                setattr(layer, name, tensor)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ExpertStreamer(layer, names)

    def test_a_format_whose_specs_disagree_with_its_source_is_rejected(self):
        class WrongShape(DenseLayerFormat):
            def tensor_specs(self, layer):
                return tuple(
                    ExpertTensorSpec(
                        spec.name, spec.row_shape + (1,), spec.dtype, spec.residence
                    )
                    for spec in super().tensor_specs(layer)
                )

        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        with self.assertRaisesRegex(ValueError, "does not match its spec"):
            ExpertStreamer(layer, ("rows",), format=WrongShape(("rows",)))

    def test_format_names_must_match_the_streamer_names(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        layer.other = torch.zeros(4, 3)
        with self.assertRaisesRegex(ValueError, "do not match tensor names"):
            ExpertStreamer(layer, ("rows",), format=DenseLayerFormat(("other",)))

    def test_default_row_source_kinds(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        dense = DenseLayerFormat(("rows",))
        specs = dense.tensor_specs(layer)
        with envs.SGLANG_MOE_EXPERT_FILE_READER.override("mmap"):
            self.assertIsNone(dense.default_row_source(layer, specs, "auto"))
            self.assertIsNone(dense.default_row_source(layer, specs, "tensor"))
            with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_FILE_READER"):
                dense.default_row_source(layer, specs, "files")
        with self.assertRaisesRegex(ValueError, "no row source kind 'shards'"):
            dense.default_row_source(layer, specs, "shards")


class TestStreamerDiscovery(unittest.TestCase):
    def test_helpers_find_streamers_in_module_order(self):
        model = torch.nn.Module()
        first, plain, second = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
        first._nvfp4_expert_streamer = "first"
        second._nvfp4_expert_streamer = "second"
        model.add_module("a", first)
        model.add_module("b", plain)
        model.add_module("c", second)
        self.assertEqual(list(iter_expert_streamers(model)), ["first", "second"])
        self.assertEqual(expert_streamer_of(first), "first")
        self.assertIsNone(expert_streamer_of(plain))


class TestSpecStagingShapes(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()

    def tearDown(self):
        expert_stream._STAGING.clear()

    def test_cached_gather_stages_rows_in_their_source_shapes(self):
        layer = _nvfp4_layer(experts=8)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)

        def copy_rows(source_ids, outputs):
            for name, output in outputs.items():
                torch.index_select(getattr(layer, name).data, 0, source_ids, out=output)
            return 0

        streamer._copy_source_rows = copy_rows
        ids = torch.tensor([[1, 5], [5, 2]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_cached(source_ids, compact_ids, ids)
        for name in NVFP4_STREAM_TENSORS:
            source = getattr(layer, name).data
            self.assertEqual(tuple(tensors[name].shape), (64,) + tuple(source.shape[1:]))
            self.assertEqual(tensors[name].dtype, source.dtype)
            self.assertTrue(
                torch.equal(
                    _as_bytes(tensors[name][compact.long()]), _as_bytes(source[ids])
                ),
                name,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Write the CUDA tests (they run in Task 10)**

Create `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`:

```python
"""CUDA tests for the expert format and row-source plugin seams."""

import unittest

import torch

from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-a", runner_config="1-gpu-small")

EXPERTS = 8
_HOST_SHAPES = {
    "w13_weight": (3, 8),
    "w2_weight": (5, 4),
    "w13_blockscale_swizzled": (3, 2),
    "w2_blockscale_swizzled": (5, 1),
}


def _nvfp4_layer(seed=21, pinned=True, experts=EXPERTS):
    """NVFP4-named host rows (optionally pinned) and CUDA alphas, as in test_expert_graph_gather."""
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    for name in NVFP4_STREAM_TENSORS[:4]:
        rows = torch.randint(
            0, 256, (experts,) + _HOST_SHAPES[name], dtype=torch.uint8, generator=generator
        )
        if "blockscale" in name:
            rows = rows.view(torch.float8_e4m3fn)
        if pinned:
            rows = rows.pin_memory()
        setattr(layer, name, torch.nn.Parameter(rows, requires_grad=False))
    for name in NVFP4_STREAM_TENSORS[4:]:
        values = torch.rand(experts, generator=generator).cuda()
        setattr(layer, name, torch.nn.Parameter(values, requires_grad=False))
    layer.top_k = 4
    return layer


def _source_bytes(layer, name, ids):
    source = getattr(layer, name).data
    return source[ids.long().to(source.device)].view(torch.uint8).cpu()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFormatSeamCuda(unittest.TestCase):
    def test_hot_and_pinned_slots_keep_their_source_shapes(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        layer = _nvfp4_layer(pinned=False)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        hot = ExpertHotCache(streamer, 2)
        pinned = ExpertPinnedHostCache(streamer, 2)
        self.assertEqual(hot.device, layer.g1_alphas.device)
        for name in NVFP4_STREAM_TENSORS:
            source = getattr(layer, name).data
            self.assertEqual(tuple(hot.tensors[name].shape[1:]), tuple(source.shape[1:]))
            self.assertEqual(hot.tensors[name].dtype, source.dtype)
        self.assertEqual(pinned.cached_names, NVFP4_STREAM_TENSORS[:4])
        for name in pinned.cached_names:
            source = getattr(layer, name).data
            self.assertEqual(
                tuple(pinned.tensors[name].shape), (2,) + tuple(source.shape[1:])
            )
            self.assertEqual(pinned.tensors[name].dtype, source.dtype)

    def test_graph_gather_still_detects_a_rebound_source(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = _nvfp4_layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        cache = ExpertHotCache(streamer, 3, scratch_rows=4)
        cache.reassign([1, 4, 6])
        streamer.enable_graph_gather(4)
        ids = torch.tensor([[1, 2, 4, 7]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].view(torch.uint8).cpu(),
                    _source_bytes(layer, name, ids),
                ),
                name,
            )
        layer.w13_weight.data = layer.w13_weight.data.clone().pin_memory()
        with self.assertRaisesRegex(RuntimeError, "moved after graph gather"):
            streamer.gather(ids)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py
git commit -m "test(moe): pin dense expert specs, dynamic sources and streamer discovery

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py'
```
Expected: collection of `test_expert_format.py` errors with `ModuleNotFoundError: No module named 'sglang.srt.layers.moe.expert_format'`. Both classes in `test_expert_plugins_cuda.py` skip with "CUDA is required".

- [ ] **Step 4: Create `expert_format.py`**

Create `python/sglang/srt/layers/moe/expert_format.py`:

```python
"""Expert tensor formats: the row schema and dense sources an expert streamer reads.

A format tells :class:`~sglang.srt.layers.moe.expert_stream.ExpertStreamer`
which tensors one expert row holds (:class:`ExpertTensorSpec`), where each
tensor's dense ``[experts, ...]`` source lives if it has one, and which row
source fills host rows. :class:`DenseLayerFormat` is the behaviour every
streamer had before formats existed: every tensor is a layer attribute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
)

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.expert_row_source import ExpertRowSource
    from sglang.srt.layers.moe.expert_stream import ExpertStreamer

# The attribute a quantization method sets on a MoE layer to attach its streamer.
# The name predates other formats; every format uses it so discovery has one home.
STREAMER_ATTRIBUTE = "_nvfp4_expert_streamer"
# Set by the NVFP4 method when every host tensor is a verified expert-file view.
FILE_SOURCE_BYTES_ATTRIBUTE = "_nvfp4_file_source_bytes_per_expert"
# Row source kinds every format accepts; a format may define more.
GENERIC_ROW_SOURCE_KINDS = ("auto", "files", "tensor")


@dataclass(frozen=True)
class ExpertTensorSpec:
    """One streamed tensor's per-expert row: ``row_shape`` elements of ``dtype``.

    ``residence`` is where rows come from: ``host`` rows are read into host
    memory (the pinned tier, pinned staging, or a registered arena), and
    ``device`` rows are indexed from a CUDA source.
    """

    name: str
    row_shape: tuple[int, ...]
    dtype: torch.dtype
    residence: Literal["host", "device"]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row_shape", tuple(int(dimension) for dimension in self.row_shape)
        )
        if not self.name:
            raise ValueError("expert tensor spec needs a name")
        if any(dimension < 0 for dimension in self.row_shape):
            raise ValueError(f"expert tensor spec {self.name!r} has a negative dimension")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"expert tensor spec {self.name!r} dtype must be a torch.dtype")
        if self.residence not in ("host", "device"):
            raise ValueError(
                f"expert tensor spec {self.name!r} residence must be 'host' or 'device'"
            )

    @property
    def row_bytes(self) -> int:
        return math.prod(self.row_shape) * self.dtype.itemsize


class ExpertFormat(Protocol):
    """What an expert streamer needs to know about one layer's expert tensors.

    ``tensor_specs`` returns one spec per streamed tensor, in streamer order.
    ``source`` returns the dense ``[experts, ...]`` tensor of a name at call
    time (the host arena rebinds layer tensors after startup), or None when
    the format has no dense source and only its row source can read rows.
    ``default_row_source`` builds the row source for a knob kind (see
    ``SGLANG_MOE_EXPERT_ROW_SOURCE``) and raises for kinds it does not know.
    ``file_source_bytes_per_expert`` returns the file bytes one expert row
    reads, or None; None keeps eager gathers out of the pinned host tier and
    reports file counters as unknown.
    """

    key: str
    supports_graph_gather: bool
    supports_host_arena: bool
    max_gather_rows: Optional[int]

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]: ...

    def num_experts(self, layer: torch.nn.Module) -> int: ...

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]: ...

    def default_row_source(
        self,
        layer: torch.nn.Module,
        specs: Sequence[ExpertTensorSpec],
        kind: str,
    ) -> Optional["ExpertRowSource"]: ...

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional["ExpertRowSource"]
    ) -> Optional[int]: ...


class DenseLayerFormat:
    """Every streamed tensor is a dense ``[experts, ...]`` attribute of the layer.

    This is the NVFP4 format and the behaviour of every streamer created
    without a format.
    """

    key = "dense"
    supports_graph_gather = True
    supports_host_arena = True
    max_gather_rows: Optional[int] = None

    def __init__(self, tensor_names: Iterable[str]):
        self.tensor_names = tuple(tensor_names)
        if not self.tensor_names:
            raise ValueError("expert streamer requires at least one tensor")

    @staticmethod
    def _dense(layer: torch.nn.Module, name: str) -> torch.Tensor:
        value = getattr(layer, name)
        return value.data if isinstance(value, torch.nn.Parameter) else value

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        return self._dense(layer, name)

    def num_experts(self, layer: torch.nn.Module) -> int:
        expert_count = None
        for name in self.tensor_names:
            if not hasattr(layer, name):
                raise ValueError(f"expert source tensor {name!r} is missing")
            tensor = self._dense(layer, name)
            if tensor.ndim == 0:
                raise ValueError(
                    f"expert source tensor {name!r} has no expert dimension"
                )
            if tensor.shape[0] == 0:
                raise ValueError(f"expert source tensor {name!r} has no expert rows")
            if not tensor.is_contiguous():
                raise ValueError(f"expert source tensor {name!r} must be contiguous")
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError(
                    f"expert source tensor {name!r} uses unsupported device {tensor.device}"
                )
            if expert_count is None:
                expert_count = tensor.shape[0]
            elif tensor.shape[0] != expert_count:
                raise ValueError(
                    f"expert count mismatch for {name!r}: {tensor.shape[0]} != {expert_count}"
                )
        assert expert_count is not None
        return expert_count

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]:
        self.num_experts(layer)
        specs = []
        for name in self.tensor_names:
            tensor = self._dense(layer, name)
            specs.append(
                ExpertTensorSpec(
                    name,
                    tuple(tensor.shape[1:]),
                    tensor.dtype,
                    "host" if tensor.device.type == "cpu" else "device",
                )
            )
        return tuple(specs)

    def default_row_source(
        self,
        layer: torch.nn.Module,
        specs: Sequence[ExpertTensorSpec],
        kind: str,
    ) -> Optional["ExpertRowSource"]:
        # Imported here: the reader pulls in sglang.srt.model_loader, whose
        # package import reaches modelopt_quant, which imports expert_stream.
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader
        from sglang.srt.model_loader.file_row_reader import validate_file_reader_mode

        names = tuple(spec.name for spec in specs)
        if kind == "auto":
            return ExpertFileRowReader.from_layer(layer, names)
        if kind == "files":
            mode = validate_file_reader_mode(envs.SGLANG_MOE_EXPERT_FILE_READER.get())
            if mode == "mmap":
                raise ValueError(
                    "SGLANG_MOE_EXPERT_ROW_SOURCE=files needs "
                    "SGLANG_MOE_EXPERT_FILE_READER=uring or uring_direct"
                )
            return ExpertFileRowReader.from_layer(layer, names, mode=mode)
        if kind == "tensor":
            return None
        raise ValueError(
            f"expert format {self.key!r} has no row source kind {kind!r}; "
            f"choose from {GENERIC_ROW_SOURCE_KINDS}"
        )

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional["ExpertRowSource"]
    ) -> Optional[int]:
        # Exactly the pre-format gate: only the NVFP4 method's verified
        # attribute enables file attribution and the eager pinned tier.
        return getattr(layer, FILE_SOURCE_BYTES_ATTRIBUTE, None)


def expert_streamer_of(module: torch.nn.Module) -> Optional["ExpertStreamer"]:
    """The expert streamer attached to ``module``, or None."""
    return getattr(module, STREAMER_ATTRIBUTE, None)


def iter_expert_streamers(model: torch.nn.Module) -> Iterator["ExpertStreamer"]:
    """Every attached expert streamer of ``model``, in module order."""
    for module in model.modules():
        streamer = expert_streamer_of(module)
        if streamer is not None:
            yield streamer
```

- [ ] **Step 5: Edit `expert_stream.py` — imports**

Old:
```python
from sglang.srt.layers.moe.expert_dma import ExpertDMABackend, _aot_transfer_available
```
New:
```python
from sglang.srt.layers.moe.expert_dma import ExpertDMABackend, _aot_transfer_available
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertFormat,
    ExpertTensorSpec,
    iter_expert_streamers,
)
```

- [ ] **Step 6: Edit `ExpertPinnedHostCache.__init__` to take names, shapes and dtypes from specs**

Old:
```python
        self.cached_names = tuple(
            name
            for name in streamer.tensor_names
            if _tensor_data(getattr(streamer.layer, name)).device.type == "cpu"
        )
        if capacity and not self.cached_names:
            raise ValueError(
                "pinned host cache requires at least one CPU source tensor"
            )
        self.bytes_per_expert = streamer.host_bytes_per_expert
        self.residency_bytes = capacity * self.bytes_per_expert
        devices = {
            _tensor_data(getattr(streamer.layer, name)).device
            for name in streamer.tensor_names
            if _tensor_data(getattr(streamer.layer, name)).device.type == "cuda"
        }
        if len(devices) > 1:
            raise ValueError("pinned host cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            name: torch.empty(
                (capacity,)
                + tuple(_tensor_data(getattr(streamer.layer, name)).shape[1:]),
                dtype=_tensor_data(getattr(streamer.layer, name)).dtype,
                device="cpu",
                pin_memory=True,
            )
            for name in self.cached_names
        }
```
New:
```python
        self.cached_names = tuple(
            spec.name for spec in streamer.specs if spec.residence == "host"
        )
        if capacity and not self.cached_names:
            raise ValueError(
                "pinned host cache requires at least one CPU source tensor"
            )
        self.bytes_per_expert = streamer.host_bytes_per_expert
        self.residency_bytes = capacity * self.bytes_per_expert
        devices = {
            streamer.source(spec.name).device
            for spec in streamer.specs
            if spec.residence == "device"
        }
        if len(devices) > 1:
            raise ValueError("pinned host cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            name: torch.empty(
                (capacity,) + streamer.spec(name).row_shape,
                dtype=streamer.spec(name).dtype,
                device="cpu",
                pin_memory=True,
            )
            for name in self.cached_names
        }
```

- [ ] **Step 7: Edit `ExpertPinnedHostCacheManager.from_model` discovery**

Old:
```python
        streamers = {}
        for module in model.modules():
            streamer = getattr(module, "_nvfp4_expert_streamer", None)
            if streamer is None or streamer.host_bytes_per_expert == 0:
                continue
```
New:
```python
        streamers = {}
        for streamer in iter_expert_streamers(model):
            if streamer.host_bytes_per_expert == 0:
                continue
```

- [ ] **Step 8: Replace `ExpertStreamer.__init__` and add the accessors**

Replace the whole `ExpertStreamer.__init__` method: from `    def __init__(` (the one taking `layer: torch.nn.Module, tensor_names: Iterable[str]`) down to the closing `)` of its `logger.info(...)` call, just above `def serves_graph_gather`. The new code is:

```python
    def __init__(
        self,
        layer: torch.nn.Module,
        tensor_names: Iterable[str],
        *,
        layer_id: int | None = None,
        format: ExpertFormat | None = None,
    ):
        self.layer = layer
        self.layer_id = (
            getattr(layer, "layer_id", None) if layer_id is None else layer_id
        )
        self.tensor_names = tuple(tensor_names)
        if not self.tensor_names:
            raise ValueError("expert streamer requires at least one tensor")
        # The format owns the row schema. Sources stay dynamic lookups because
        # the host arena rebinds layer tensors after this streamer exists.
        self.format = (
            DenseLayerFormat(self.tensor_names) if format is None else format
        )
        self.num_experts = self._validate_sources()
        self.file_row_reader = self.format.default_row_source(
            layer, self.specs, "auto"
        )
        self.hot_cache = None
        self.expert_copy_backend = "gpu"
        self._dma_backend = ExpertDMABackend()
        self.pinned_host_cache = None
        self.residency_policy = None
        self.residency_update = None
        self.row_planner = None
        self.row_plan = None
        self.row_backend = None
        self.row_tag = 0
        self.before_eager_gather = None
        # Set by ExpertPredictionRuntime when SGLANG_MOE_EXPERT_PREFETCH_PULL is on and this
        # layer is a scored prefetch target; see PrefetchPuller.join_target in serving/runtime.py.
        self.prefetch_puller = None
        self.graph_gather_rows = 0
        self.graph_counters: torch.Tensor | None = None
        self.last_gather_stats = ExpertGatherStats()
        self.bytes_per_expert = sum(spec.row_bytes for spec in self.specs)
        self.host_bytes_per_expert = sum(
            spec.row_bytes for spec in self.specs if spec.residence == "host"
        )
        signature = tuple(
            (
                spec.name,
                (self.num_experts,) + spec.row_shape,
                str(spec.dtype),
                spec.residence,
            )
            for spec in self.specs
        )
        if signature not in _LOGGED_SOURCE_SIGNATURES:
            _LOGGED_SOURCE_SIGNATURES.add(signature)
            logger.info(
                "MoE selected-expert streaming active: experts=%d tensors=%s",
                self.num_experts,
                ",".join(self.tensor_names),
            )

    @property
    def specs(self) -> tuple[ExpertTensorSpec, ...]:
        """The format's row specs, in ``tensor_names`` order."""
        return tuple(self._specs.values())

    def spec(self, name: str) -> ExpertTensorSpec:
        return self._specs[name]

    def source(self, name: str) -> torch.Tensor | None:
        """The dense ``[experts, ...]`` source of ``name`` now, or None when it has none."""
        return self.format.source(self.layer, name)

    @property
    def file_source_bytes_per_expert(self) -> int | None:
        """File bytes one expert row reads; None keeps eager gathers out of the pinned tier."""
        return self.format.file_source_bytes_per_expert(
            self.layer, self.file_row_reader
        )
```

- [ ] **Step 9: Replace `_validate_sources`**

Replace the whole `_validate_sources` method (from `    def _validate_sources(self) -> int:` to its `return expert_count`) with:

```python
    def _validate_sources(self) -> int:
        """Check the format's specs against the streamer names and any dense sources."""
        specs = tuple(self.format.tensor_specs(self.layer))
        names = tuple(spec.name for spec in specs)
        if names != self.tensor_names:
            raise ValueError(
                f"expert format specs {names} do not match tensor names {self.tensor_names}"
            )
        expert_count = index(self.format.num_experts(self.layer))
        if expert_count < 1:
            raise ValueError("expert format has no expert rows")
        for spec in specs:
            source = self.format.source(self.layer, spec.name)
            if source is None:
                if spec.residence != "host":
                    raise ValueError(
                        f"expert tensor {spec.name!r} has no dense source, so its "
                        "rows must be host-resident"
                    )
                continue
            if (
                tuple(source.shape) != (expert_count,) + spec.row_shape
                or source.dtype != spec.dtype
            ):
                raise ValueError(
                    f"expert source tensor {spec.name!r} does not match its spec"
                )
            if (source.device.type == "cpu") != (spec.residence == "host"):
                raise ValueError(
                    f"expert source tensor {spec.name!r} on {source.device} does not "
                    f"match its spec's {spec.residence!r} residence"
                )
        self._specs = {spec.name: spec for spec in specs}
        return expert_count
```

- [ ] **Step 10: Edit `_gather_cached` — staging shapes from specs, and the pinned gate through the property**

Old:
```python
        padded = {
            name: _staging_buffer(
                name,
                kernel_rows,
                max(capacity, self.num_experts),
                tuple(source.shape[1:]),
                source.dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
            for source in [_tensor_data(getattr(self.layer, name))]
        }
```
New:
```python
        padded = {
            name: _staging_buffer(
                name,
                kernel_rows,
                max(capacity, self.num_experts),
                self.spec(name).row_shape,
                self.spec(name).dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
        }
```

Old (in `_gather_cached`):
```python
            and pinned_cache.capacity
            and getattr(self.layer, "_nvfp4_file_source_bytes_per_expert", None)
            is not None
        ):
```
New:
```python
            and pinned_cache.capacity
            and self.file_source_bytes_per_expert is not None
        ):
```

- [ ] **Step 11: Edit `_gather_pinned_host` staging shapes**

Old:
```python
                tuple(_tensor_data(getattr(self.layer, name)).shape[1:]),
                _tensor_data(getattr(self.layer, name)).dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
        }
        gathered = {name: buffer[:row_count] for name, buffer in padded.items()}
        hit_positions = hit_mask.nonzero().flatten()
```
New:
```python
                self.spec(name).row_shape,
                self.spec(name).dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
        }
        gathered = {name: buffer[:row_count] for name, buffer in padded.items()}
        hit_positions = hit_mask.nonzero().flatten()
```

- [ ] **Step 12: Edit the pinned gate in `gather`**

Old:
```python
            and self.pinned_host_cache.capacity
            and getattr(self.layer, "_nvfp4_file_source_bytes_per_expert", None)
            is not None
        ):
            return self._gather_pinned_host(source_ids, compact_ids, topk_ids)
```
New:
```python
            and self.pinned_host_cache.capacity
            and self.file_source_bytes_per_expert is not None
        ):
            return self._gather_pinned_host(source_ids, compact_ids, topk_ids)
```
Do not change `enable_graph_gather`, `_gather_graph`, `_check_graph_sources`, `_copy_source_rows`, `_read_host_rows`, `ensure_rows`, or the no-cache loop at the end of `gather` in this task.

- [ ] **Step 13: Edit `expert_hot_cache.py`**

Import. Old:
```python
from sglang.srt.layers.moe.expert_stream import ExpertStreamer, _tensor_data
```
New:
```python
from sglang.srt.layers.moe.expert_format import iter_expert_streamers
from sglang.srt.layers.moe.expert_stream import ExpertStreamer
```

`ExpertHotCache.__init__`. Old:
```python
        devices = {
            _tensor_data(getattr(streamer.layer, name)).device
            for name in streamer.tensor_names
            if _tensor_data(getattr(streamer.layer, name)).device.type == "cuda"
        }
        if len(devices) > 1:
            raise ValueError("hot cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            name: torch.empty(
                (allocation_rows,) + tuple(source.shape[1:]),
                dtype=source.dtype,
                device=self.device,
            )
            for name in streamer.tensor_names
            for source in [_tensor_data(getattr(streamer.layer, name))]
        }
```
New:
```python
        devices = {
            streamer.source(spec.name).device
            for spec in streamer.specs
            if spec.residence == "device"
        }
        if len(devices) > 1:
            raise ValueError("hot cache CUDA sources must share one device")
        self.device = next(
            iter(devices), torch.device("cuda", torch.cuda.current_device())
        )
        self.tensors = {
            spec.name: torch.empty(
                (allocation_rows,) + spec.row_shape,
                dtype=spec.dtype,
                device=self.device,
            )
            for spec in streamer.specs
        }
```

`_prepare_promotion`. Old:
```python
            sources = {
                name: _tensor_data(getattr(self.streamer.layer, name))
                for name in self.streamer.tensor_names
            }
```
New:
```python
            sources = {
                name: self.streamer.source(name)
                for name in self.streamer.tensor_names
            }
```

`ExpertHotCacheManager.from_model`. Old:
```python
        for module in model.modules():
            streamer = getattr(module, "_nvfp4_expert_streamer", None)
            if streamer is None:
                continue
            layer_id = index(streamer.layer_id)
```
New:
```python
        for streamer in iter_expert_streamers(model):
            layer_id = index(streamer.layer_id)
```

Observer file attribution (inside the per-layer loop that sums `counters.gather_copy_engine_bytes`). Old:
```python
            file_bytes = getattr(
                streamer.layer, "_nvfp4_file_source_bytes_per_expert", None
            )
```
New:
```python
            file_bytes = streamer.file_source_bytes_per_expert
```
Then run `grep -n "_tensor_data\|_nvfp4_expert_streamer\|_nvfp4_file_source_bytes_per_expert" python/sglang/srt/layers/moe/expert_hot_cache.py`. Expected: only the docstring line of `ExpertHotCacheManager` (the one that says "Startup may set each layer's `_nvfp4_file_source_bytes_per_expert`").

- [ ] **Step 14: Edit `expert_host_arena.py`**

Add the import after `import torch`:
```python
from sglang.srt.layers.moe.expert_format import iter_expert_streamers
```
Old:
```python
        streamers = [
            streamer
            for module in model.modules()
            for streamer in [getattr(module, "_nvfp4_expert_streamer", None)]
            if streamer is not None
        ]
```
New:
```python
        streamers = list(iter_expert_streamers(model))
```

- [ ] **Step 15: Edit the frozen `model_runner.py` (two discovery swaps only)**

First read `/home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins/.claude/skills/large-class-style/SKILL.md`. These swaps are "Delegate / Coordinate" statements: they call a collaborator's helper and add no logic.

In `maybe_init_expert_hot_cache`. Old:
```python
            streamed = any(
                getattr(module, "_nvfp4_expert_streamer", None) is not None
                for module in self.model.modules()
            )
```
New:
```python
            from sglang.srt.layers.moe.expert_format import iter_expert_streamers

            streamed = next(iter_expert_streamers(self.model), None) is not None
```
Old:
```python
                request_routes = verify_tokens * max(
                    getattr(module._nvfp4_expert_streamer.layer, "top_k", 0) or 0
                    for module in self.model.modules()
                    if getattr(module, "_nvfp4_expert_streamer", None) is not None
                )
```
New:
```python
                request_routes = verify_tokens * max(
                    getattr(streamer.layer, "top_k", 0) or 0
                    for streamer in iter_expert_streamers(self.model)
                )
```
Then run `git diff -- python/sglang/srt/model_executor/model_runner.py`. Expected: exactly these two hunks.

- [ ] **Step 16: Commit, push and run the new tests plus `CPU_SUITE`**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/layers/moe/expert_format.py python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/layers/moe/expert_hot_cache.py python/sglang/srt/layers/moe/expert_host_arena.py python/sglang/srt/model_executor/model_runner.py
git commit -m "feat(moe): describe streamed expert tensors with format specs

The streamer, pinned tier and hot cache take row shapes, dtypes and byte
counts from ExpertTensorSpecs; DenseLayerFormat reproduces the attribute
probes, sources stay dynamic lookups, and discovery goes through
iter_expert_streamers.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py'
```
Expected:
- Every test in `test_expert_format.py` passes (10 tests).
- `test_expert_plugins_cuda.py` skips 2.
- The only failure is the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`.
- The passed count equals the Task 0 baseline N plus 10.
- `GraphGatherStartupSizingTests` in `test_expert_graph_gather_scratch.py` still passes. It drives the edited `model_runner` block.

---

### Task 2: The row-source protocol — `ExpertRowSource`, `RowReadStats`, `ReadTicket`, `HostSlotLayout`, `TensorRowSource`, and `ExpertFileRowReader` conformance with `from_group`

This task adds the disk→RAM seam as a CPU-only module and makes the existing io_uring reader conform to it. `ExpertFileRowReader.read` now returns a `RowReadStats`; every existing caller ignores the return value. The task does not change any caller.

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_row_source.py`
- Create: `python/sglang/test/moe_expert_fakes.py`
- Modify: `python/sglang/srt/layers/moe/expert_file_reader.py` (whole file replaced; `from_layer` body unchanged)
- Test (create): `test/registered/unit/layers/moe/test_expert_row_source.py` (CPU)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces, in `sglang.srt.layers.moe.expert_row_source`:
  - `class HostSlotLayout(Enum)` with members `PER_NAME = "per_name"` and `BLOB = "blob"`. Only `PER_NAME` is implemented.
  - `@dataclass(frozen=True) class RowReadStats`, with fields `rows: int = 0`, `file_bytes: int = 0`, `split_bytes: int = 0`, `read_ns: int = 0` and `split_ns: int = 0`, plus a fieldwise `__add__`.
  - `class ReadTicket(Protocol)`, with `done() -> bool` and `wait() -> RowReadStats`. `wait()` re-raises the read's exception.
  - `class CompletedReadTicket(stats: RowReadStats | None = None, error: BaseException | None = None)`.
  - `class ExpertRowSource(Protocol)`, runtime-checkable, with:
    - attributes `names: tuple[str, ...]`, `num_experts: int`, `host_layouts: frozenset[HostSlotLayout]`, `preferred_batch_rows: int` (0 = no preference), `file_bytes_per_expert: int` and `requires_page_aligned_destinations: bool`;
    - `covers(name) -> bool`;
    - `register_destinations(tensors) -> int`;
    - `read(rows, destinations, destination_rows=None) -> RowReadStats`;
    - `submit(rows, destinations, destination_rows=None) -> ReadTicket`;
    - `close() -> None`.
  - `class SynchronousSubmit`, a mixin whose `submit` runs `read` now and returns a `CompletedReadTicket`. It calls `read(rows, destinations)` with two arguments when `destination_rows is None`.
  - `class TensorRowSource(SynchronousSubmit)`, built as `TensorRowSource(lookup: Callable[[str], Optional[Tensor]], names: Sequence[str], num_experts: int)`. It uses `index_select` from the dense tensors that `lookup` returns at call time.
- Also produces:
  - `ExpertFileRowReader` is an `ExpertRowSource`, with `num_experts`, `file_bytes_per_expert` (the sum of member row bytes), `host_layouts`, `preferred_batch_rows = 0` and `requires_page_aligned_destinations = False`.
  - `ExpertFileRowReader.read(...) -> RowReadStats`.
  - `ExpertFileRowReader.close()`.
  - `ExpertFileRowReader.from_group(group: FileTensorCacheGroup, names: Iterable[str] | None = None, mode: str | None = None) -> ExpertFileRowReader`. It raises `ValueError` for `mmap` mode and for unknown members.
  - `sglang.test.moe_expert_fakes.CountingRowSource(tensors: Mapping[str, Tensor])` records `.calls: list[RowSourceCall(rows, names, destination_rows)]` and reports `read_ns = rows`.

- [ ] **Step 1: Write the test double module**

Create `python/sglang/test/moe_expert_fakes.py`:

```python
"""Test doubles for the MoE expert streaming plugin seams (formats and row sources)."""

from __future__ import annotations

from typing import Iterable, Mapping, NamedTuple, Optional

import torch

from sglang.srt.layers.moe.expert_row_source import (
    HostSlotLayout,
    RowReadStats,
    SynchronousSubmit,
)


class RowSourceCall(NamedTuple):
    rows: list[int]
    names: tuple[str, ...]
    destination_rows: Optional[list[int]]


class CountingRowSource(SynchronousSubmit):
    """A row source over in-memory ``[experts, ...]`` tensors that records every read.

    ``read_ns`` reports one nanosecond per row, so tests can check how stats
    are summed without a clock.
    """

    host_layouts = frozenset({HostSlotLayout.PER_NAME})
    preferred_batch_rows = 0
    requires_page_aligned_destinations = False

    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        self.tensors = dict(tensors)
        self.names = tuple(self.tensors)
        self.num_experts = next(iter(self.tensors.values())).shape[0]
        self.file_bytes_per_expert = sum(self._row_bytes(name) for name in self.names)
        self.calls: list[RowSourceCall] = []
        self.registered: list[torch.Tensor] = []
        self.closed = False

    def _row_bytes(self, name: str) -> int:
        tensor = self.tensors[name]
        return tensor[0].numel() * tensor.element_size()

    def covers(self, name: str) -> bool:
        return name in self.tensors

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int:
        self.registered.extend(tensors)
        return 0

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> RowReadStats:
        missing = [name for name in destinations if name not in self.tensors]
        if missing:
            raise ValueError(f"counting row source does not cover {missing}")
        row_list = [int(row) for row in rows.reshape(-1).tolist()]
        slots = (
            list(range(len(row_list)))
            if destination_rows is None
            else [int(slot) for slot in destination_rows.reshape(-1).tolist()]
        )
        self.calls.append(
            RowSourceCall(
                row_list,
                tuple(destinations),
                None if destination_rows is None else slots,
            )
        )
        for name, destination in destinations.items():
            source = self.tensors[name]
            for row, slot in zip(row_list, slots):
                destination[slot].copy_(source[row])
        return RowReadStats(
            rows=len(row_list),
            file_bytes=len(row_list)
            * sum(self._row_bytes(name) for name in destinations),
            read_ns=len(row_list),
        )

    def close(self) -> None:
        self.closed = True
```

- [ ] **Step 2: Write the failing tests**

Create `test/registered/unit/layers/moe/test_expert_row_source.py`:

```python
"""CPU tests for expert host row sources and how the streamer routes host reads."""

import os
import tempfile
import unittest

import torch

from sglang.srt.layers.moe.expert_row_source import (
    CompletedReadTicket,
    ExpertRowSource,
    HostSlotLayout,
    ReadTicket,
    RowReadStats,
    TensorRowSource,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import CountingRowSource

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4096
HAS_URING = os.path.exists("/usr/include/liburing.h")


class TestRowReadStats(unittest.TestCase):
    def test_stats_add_field_by_field(self):
        total = RowReadStats(1, 2, 3, 4, 5) + RowReadStats(10, 20, 30, 40, 50)
        self.assertEqual(total, RowReadStats(11, 22, 33, 44, 55))
        self.assertEqual(RowReadStats() + RowReadStats(rows=2), RowReadStats(rows=2))


class TestCompletedReadTicket(unittest.TestCase):
    def test_synchronous_submit_returns_a_done_ticket(self):
        rows = torch.arange(12, dtype=torch.uint8).reshape(4, 3)
        source = CountingRowSource({"rows": rows})
        destination = torch.zeros(2, 3, dtype=torch.uint8)
        ticket = source.submit(torch.tensor([3, 1]), {"rows": destination})
        self.assertIsInstance(ticket, ReadTicket)
        self.assertTrue(ticket.done())
        self.assertEqual(ticket.wait().rows, 2)
        self.assertTrue(torch.equal(destination, rows[[3, 1]]))
        self.assertIsNone(source.calls[0].destination_rows)

    def test_a_failed_read_raises_from_wait(self):
        source = CountingRowSource({"rows": torch.zeros(4, 3)})
        ticket = source.submit(torch.tensor([0]), {"missing": torch.zeros(1, 3)})
        self.assertTrue(ticket.done())
        with self.assertRaisesRegex(ValueError, "does not cover"):
            ticket.wait()

    def test_a_ticket_holds_stats_or_an_error(self):
        with self.assertRaises(ValueError):
            CompletedReadTicket()
        with self.assertRaises(ValueError):
            CompletedReadTicket(RowReadStats(), RuntimeError("both"))


class TestTensorRowSource(unittest.TestCase):
    def _source(self):
        tensors = {
            "a": torch.arange(24, dtype=torch.int16).reshape(6, 4),
            "b": torch.arange(6, dtype=torch.float32),
        }
        return tensors, TensorRowSource(tensors.get, ("a", "b"), 6)

    def test_reads_rows_into_destination_slots(self):
        tensors, source = self._source()
        destinations = {
            "a": torch.zeros(4, 4, dtype=torch.int16),
            "b": torch.zeros(4, dtype=torch.float32),
        }
        stats = source.read(torch.tensor([5, 0, 3]), destinations, torch.tensor([2, 0, 3]))
        for name, tensor in tensors.items():
            self.assertTrue(
                torch.equal(destinations[name][[2, 0, 3]], tensor[[5, 0, 3]]), name
            )
            self.assertEqual(destinations[name][1].abs().sum().item(), 0)
        self.assertEqual(stats.rows, 3)
        self.assertEqual(stats.file_bytes, 0)
        self.assertGreaterEqual(stats.read_ns, 0)

    def test_reads_rows_into_leading_rows_without_slots(self):
        tensors, source = self._source()
        destination = torch.zeros(5, 4, dtype=torch.int16)
        source.read(torch.tensor([4, 1]), {"a": destination})
        self.assertTrue(torch.equal(destination[:2], tensors["a"][[4, 1]]))
        self.assertEqual(destination[2:].abs().sum().item(), 0)

    def test_covers_only_names_with_a_dense_source(self):
        tensor = torch.zeros(6, 2)
        source = TensorRowSource({"a": tensor, "b": None}.get, ("a", "b", "c"), 6)
        self.assertTrue(source.covers("a"))
        self.assertFalse(source.covers("b"))
        self.assertFalse(source.covers("c"))
        self.assertFalse(source.covers("d"))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            source.read(torch.tensor([0]), {"b": torch.zeros(1, 2)})

    def test_conforms_to_the_row_source_protocol(self):
        _, source = self._source()
        self.assertIsInstance(source, ExpertRowSource)
        self.assertEqual(source.host_layouts, frozenset({HostSlotLayout.PER_NAME}))
        self.assertEqual(source.register_destinations([torch.zeros(2)]), 0)
        self.assertEqual(source.num_experts, 6)
        self.assertEqual(source.file_bytes_per_expert, 0)
        self.assertIsInstance(source.submit(torch.tensor([1]), {"b": torch.zeros(1)}), CompletedReadTicket)


def _file_group(directory):
    from sglang.srt.model_loader.file_tensor_cache import (
        FileTensorCacheGroup,
        FileTensorSpec,
    )

    specs = (
        FileTensorSpec("w13_trellis", (6, 2, PAGE), (2 * PAGE, PAGE, 1), torch.uint8),
        FileTensorSpec("w2_svh", (6, 40), (40, 1), torch.float16),
    )
    group = FileTensorCacheGroup.open(directory, "row_source_test", {"k": 1}, specs)
    generator = torch.Generator().manual_seed(17)
    for spec in specs:
        target = group.tensors[spec.tag].view(torch.uint8)
        target.copy_(
            torch.randint(0, 256, target.shape, dtype=torch.uint8, generator=generator)
        )
    return group


@unittest.skipUnless(HAS_URING, "io_uring reads need liburing")
class TestExpertFileRowReaderRowSource(unittest.TestCase):
    def test_from_group_reads_named_members_into_slots(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        for mode in ("uring", "uring_direct"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                group = _file_group(directory)
                try:
                    reader = ExpertFileRowReader.from_group(group, mode=mode)
                    self.assertIsInstance(reader, ExpertRowSource)
                    self.assertEqual(reader.names, ("w13_trellis", "w2_svh"))
                    self.assertEqual(reader.num_experts, 6)
                    self.assertEqual(reader.file_bytes_per_expert, 2 * PAGE + 80)
                    destinations = {
                        name: torch.zeros(
                            (4,) + tuple(group.tensors[name].shape[1:]),
                            dtype=group.tensors[name].dtype,
                        )
                        for name in reader.names
                    }
                    rows = torch.tensor([5, 0, 3])
                    slots = torch.tensor([1, 3, 0])
                    stats = reader.read(rows, destinations, slots)
                    for name in reader.names:
                        self.assertTrue(
                            torch.equal(
                                destinations[name][slots].view(torch.uint8),
                                group.tensors[name][rows].view(torch.uint8),
                            ),
                            name,
                        )
                    self.assertEqual(stats.rows, 3)
                    self.assertEqual(stats.file_bytes, 3 * (2 * PAGE + 80))
                    self.assertGreater(stats.read_ns, 0)
                    reader.close()
                finally:
                    group.close()

    def test_from_group_can_cover_a_subset_of_members(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        with tempfile.TemporaryDirectory() as directory:
            group = _file_group(directory)
            try:
                reader = ExpertFileRowReader.from_group(group, ("w2_svh",), mode="uring")
                self.assertEqual(reader.names, ("w2_svh",))
                self.assertFalse(reader.covers("w13_trellis"))
                self.assertEqual(reader.file_bytes_per_expert, 80)
            finally:
                group.close()

    def test_from_group_refuses_mmap_and_unknown_members(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        with tempfile.TemporaryDirectory() as directory:
            group = _file_group(directory)
            try:
                with self.assertRaisesRegex(ValueError, "TensorRowSource"):
                    ExpertFileRowReader.from_group(group, mode="mmap")
                with self.assertRaisesRegex(ValueError, "no member 'w2_suh'"):
                    ExpertFileRowReader.from_group(group, ("w2_suh",), mode="uring")
            finally:
                group.close()

    def test_from_layer_reader_reports_its_file_bytes(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        with tempfile.TemporaryDirectory() as directory:
            group = _file_group(directory)
            try:
                layer = torch.nn.Module()
                for name, tensor in group.tensors.items():
                    parameter = torch.nn.Parameter(tensor, requires_grad=False)
                    parameter._sglang_file_cache_group = group
                    parameter._sglang_file_cache_tag = name
                    setattr(layer, name, parameter)
                reader = ExpertFileRowReader.from_layer(
                    layer, ("w13_trellis", "w2_svh"), mode="uring"
                )
                self.assertEqual(reader.file_bytes_per_expert, 2 * PAGE + 80)
                self.assertEqual(reader.num_experts, 6)
                stats = reader.read(
                    torch.tensor([2]),
                    {"w2_svh": torch.zeros(1, 40, dtype=torch.float16)},
                )
                self.assertEqual((stats.rows, stats.file_bytes), (1, 80))
            finally:
                group.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/test/moe_expert_fakes.py test/registered/unit/layers/moe/test_expert_row_source.py
git commit -m "test(moe): specify expert host row sources and their read stats

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/layers/moe/test_expert_row_source.py'
```
Expected: a collection error, `ModuleNotFoundError: No module named 'sglang.srt.layers.moe.expert_row_source'`.

- [ ] **Step 4: Create `expert_row_source.py`**

Create `python/sglang/srt/layers/moe/expert_row_source.py`:

```python
"""Host row sources: the disk-to-RAM half of expert streaming.

A row source fills host rows of the streamed tensors it covers. It knows
files and extents and never kernels: the pinned host tier and the streamer's
staging paths call it and copy the rows to the GPU themselves. This module is
CPU-only and imports nothing from ``sglang.srt.model_loader``, whose package
import reaches the quantization methods that import the streamer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import torch


class HostSlotLayout(Enum):
    """How a host tier stores one expert row.

    ``PER_NAME`` is one ``[capacity, *row_shape]`` slab per streamed tensor, the
    only layout the framework implements. ``BLOB`` is one byte slot per expert
    holding a source's on-disk row, reserved for a later RAM tier.
    """

    PER_NAME = "per_name"
    BLOB = "blob"


@dataclass(frozen=True)
class RowReadStats:
    """What one or more row reads cost.

    ``file_bytes`` counts bytes read from storage, including any superset
    waste; ``split_bytes`` counts CPU copies that split a read into
    per-tensor rows (0 for sources that read in place).
    """

    rows: int = 0
    file_bytes: int = 0
    split_bytes: int = 0
    read_ns: int = 0
    split_ns: int = 0

    def __add__(self, other: "RowReadStats") -> "RowReadStats":
        return RowReadStats(
            self.rows + other.rows,
            self.file_bytes + other.file_bytes,
            self.split_bytes + other.split_bytes,
            self.read_ns + other.read_ns,
            self.split_ns + other.split_ns,
        )


@runtime_checkable
class ReadTicket(Protocol):
    """A submitted read; ``wait`` returns its stats or raises its error."""

    def done(self) -> bool: ...

    def wait(self) -> RowReadStats: ...


class CompletedReadTicket:
    """A ticket for a read that already ran, successfully or not."""

    def __init__(
        self,
        stats: Optional[RowReadStats] = None,
        error: Optional[BaseException] = None,
    ):
        if (stats is None) == (error is None):
            raise ValueError("a completed read ticket holds either stats or an error")
        self._stats = stats
        self._error = error

    def done(self) -> bool:
        return True

    def wait(self) -> RowReadStats:
        if self._error is not None:
            raise self._error
        assert self._stats is not None
        return self._stats


@runtime_checkable
class ExpertRowSource(Protocol):
    """Fills host rows of the streamed tensors it covers.

    ``read`` copies expert ``rows`` (a CPU integer tensor) of every named
    tensor into ``destinations[name][destination_rows]``, or into the leading
    rows when ``destination_rows`` is None. Destinations are CPU and
    contiguous. It runs synchronously on the calling (model) thread and does
    no CUDA work. A source that reads a whole on-disk expert row per request
    should be given every name it covers in one call. ``submit`` returns a
    ticket; the default (``SynchronousSubmit``) reads immediately.
    ``register_destinations`` registers long-lived host buffers with the
    source's reader and must run on the reader's owner thread.
    ``preferred_batch_rows`` of 0 means no preference.
    """

    names: tuple[str, ...]
    num_experts: int
    host_layouts: frozenset[HostSlotLayout]
    preferred_batch_rows: int
    file_bytes_per_expert: int
    requires_page_aligned_destinations: bool

    def covers(self, name: str) -> bool: ...

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int: ...

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> RowReadStats: ...

    def submit(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> ReadTicket: ...

    def close(self) -> None: ...


class SynchronousSubmit:
    """``submit`` that runs ``read`` at once and returns a completed ticket.

    An error is kept in the ticket and raised by ``wait``, as an asynchronous
    source would report it.
    """

    def submit(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> CompletedReadTicket:
        try:
            if destination_rows is None:
                stats = self.read(rows, destinations)
            else:
                stats = self.read(rows, destinations, destination_rows)
        except Exception as error:
            return CompletedReadTicket(error=error)
        return CompletedReadTicket(stats)


class TensorRowSource(SynchronousSubmit):
    """Rows selected with ``index_select`` from dense ``[experts, ...]`` tensors.

    This is the mmap/tensor fallback: over a file mapping it faults the page
    cache in. ``lookup(name)`` is called on every read, because the host arena
    rebinds layer tensors after startup; a name whose lookup is None is not
    covered. It cannot tell page-cache hits from storage reads, so its stats
    report rows and time but no file bytes.
    """

    host_layouts = frozenset({HostSlotLayout.PER_NAME})
    preferred_batch_rows = 0
    file_bytes_per_expert = 0
    requires_page_aligned_destinations = False

    def __init__(
        self,
        lookup: Callable[[str], Optional[torch.Tensor]],
        names: Sequence[str],
        num_experts: int,
    ):
        self._lookup = lookup
        self.names = tuple(names)
        self.num_experts = int(num_experts)

    def covers(self, name: str) -> bool:
        return name in self.names and self._lookup(name) is not None

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int:
        return 0

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> RowReadStats:
        start = time.perf_counter_ns()
        rows = rows.reshape(-1)
        count = rows.numel()
        for name, destination in destinations.items():
            source = self._lookup(name) if name in self.names else None
            if source is None:
                raise ValueError(f"tensor row source does not cover {name!r}")
            if destination_rows is None:
                torch.index_select(
                    source,
                    0,
                    rows,
                    out=destination
                    if destination.shape[0] == count
                    else destination[:count],
                )
            else:
                for row, slot in zip(rows, destination_rows.reshape(-1)):
                    torch.index_select(
                        source,
                        0,
                        row.reshape(1),
                        out=destination[int(slot) : int(slot) + 1],
                    )
        return RowReadStats(rows=count, read_ns=time.perf_counter_ns() - start)

    def close(self) -> None:
        pass
```

- [ ] **Step 5: Replace `expert_file_reader.py`**

Replace the whole contents of `python/sglang/srt/layers/moe/expert_file_reader.py` with the code below. The `from_layer` body is unchanged from today's file.

```python
"""io_uring reads of expert rows from verified expert files."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_row_source import (
    HostSlotLayout,
    RowReadStats,
    SynchronousSubmit,
)
from sglang.srt.model_loader.file_row_reader import (
    AlignedRowSource,
    read_plans,
    shared_uring_file_reader,
    validate_file_reader_mode,
)

if TYPE_CHECKING:
    from sglang.kernels.ops.io.uring_file_reader import UringFileReader
    from sglang.srt.model_loader.file_tensor_cache import FileTensorCacheGroup

logger = logging.getLogger(__name__)
_LOGGED_MODES: set[str] = set()


class ExpertFileRowReader(SynchronousSubmit):
    """Read host expert rows from the files behind their mapped tensors.

    :meth:`from_layer` covers the tensors bound by the NVFP4 expert file
    cache. Their parameters carry the verified cache group, and row ``e`` of
    a tensor lives at byte ``e * row_bytes`` of its member file because the
    runtime view is contiguous from storage offset zero, which it checks.
    :meth:`from_group` covers members of a group opened directly, such as
    repacked expert files that are not bound as layer parameters. The reader
    is an ``ExpertRowSource`` with the per-name host layout.
    """

    host_layouts = frozenset({HostSlotLayout.PER_NAME})
    # One call is one io_uring batch; the reader pipelines its own queue depth.
    preferred_batch_rows = 0
    # O_DIRECT is used when a destination is page-aligned, buffered reads otherwise.
    requires_page_aligned_destinations = False

    def __init__(
        self,
        reader: UringFileReader,
        sources: Mapping[str, AlignedRowSource],
        mode: str,
    ) -> None:
        self._reader = reader
        self._sources = dict(sources)
        self.mode = mode
        self.registered_bytes = 0
        row_counts = {source.row_count for source in self._sources.values()}
        if len(row_counts) > 1:
            raise ValueError("expert file rows must share one expert count")
        self.num_experts = next(iter(row_counts), 0)
        self.file_bytes_per_expert = sum(
            source.row_bytes for source in self._sources.values()
        )

    @classmethod
    def from_layer(
        cls,
        layer: torch.nn.Module,
        tensor_names: Iterable[str],
        mode: Optional[str] = None,
    ) -> Optional[ExpertFileRowReader]:
        """Build a reader for the CPU expert tensors of ``layer``, or None for ``mmap``."""
        mode = validate_file_reader_mode(
            envs.SGLANG_MOE_EXPERT_FILE_READER.get() if mode is None else mode
        )
        if mode == "mmap":
            return None
        reader: Optional[UringFileReader] = None
        sources: dict[str, AlignedRowSource] = {}
        for name in tensor_names:
            parameter = getattr(layer, name)
            data = (
                parameter.data
                if isinstance(parameter, torch.nn.Parameter)
                else parameter
            )
            if data.device.type != "cpu":
                continue
            group = getattr(parameter, "_sglang_file_cache_group", None)
            tag = getattr(parameter, "_sglang_file_cache_tag", None)
            if group is None or tag is None:
                raise ValueError(
                    f"SGLANG_MOE_EXPERT_FILE_READER={mode} needs file-backed expert "
                    f"tensors, but {name!r} has no expert file; set "
                    "SGLANG_MOE_EXPERT_FILE_DIR or use the mmap reader"
                )
            if (
                data.shape[0] == 0
                or not data.is_contiguous()
                or data.storage_offset() != 0
                or data.data_ptr() != group.tensors[tag].data_ptr()
            ):
                raise ValueError(
                    f"expert tensor {name!r} does not occupy its expert file from "
                    "offset zero"
                )
            if reader is None:
                reader = shared_uring_file_reader()
            sources[name] = AlignedRowSource(
                reader,
                group.paths[tag],
                data[0].numel() * data.element_size(),
                data.shape[0],
                direct=mode == "uring_direct",
            )
        if reader is None:
            return None
        if mode not in _LOGGED_MODES:
            _LOGGED_MODES.add(mode)
            logger.info(
                "MoE expert file reads use io_uring: mode=%s tensors=%s "
                "registered_buffers=%s",
                mode,
                ",".join(sources),
                reader.registered_buffers_supported,
            )
        return cls(reader, sources, mode)

    @classmethod
    def from_group(
        cls,
        group: FileTensorCacheGroup,
        names: Optional[Iterable[str]] = None,
        mode: Optional[str] = None,
    ) -> ExpertFileRowReader:
        """Build a reader for members ``names`` (default: all) of an open cache group.

        Each member must be a contiguous ``[experts, ...]`` tensor; the member
        tag is the streamed tensor name.
        """
        mode = validate_file_reader_mode(
            envs.SGLANG_MOE_EXPERT_FILE_READER.get() if mode is None else mode
        )
        if mode == "mmap":
            raise ValueError(
                "ExpertFileRowReader.from_group reads through io_uring; with "
                "SGLANG_MOE_EXPERT_FILE_READER=mmap read the group's mapped "
                "tensors through a TensorRowSource"
            )
        specs = {spec.tag: spec for spec in group.specs}
        names = tuple(specs) if names is None else tuple(names)
        if not names:
            raise ValueError("ExpertFileRowReader.from_group needs at least one member")
        reader = shared_uring_file_reader()
        sources: dict[str, AlignedRowSource] = {}
        for name in names:
            spec = specs.get(name)
            if spec is None:
                raise ValueError(f"file tensor cache group has no member {name!r}")
            tensor = group.tensors[name]
            if not spec.shape or spec.shape[0] == 0 or not tensor.is_contiguous():
                raise ValueError(
                    f"file tensor cache member {name!r} is not a contiguous "
                    "[experts, ...] tensor"
                )
            sources[name] = AlignedRowSource(
                reader,
                group.paths[name],
                tensor[0].numel() * tensor.element_size(),
                spec.shape[0],
                direct=mode == "uring_direct",
            )
        return cls(reader, sources, mode)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._sources)

    def covers(self, name: str) -> bool:
        return name in self._sources

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int:
        """Register long-lived pinned row tensors; returns the bytes registered."""
        registered = 0
        for tensor in tensors:
            if tensor.numel() and self._reader.register_buffer(tensor):
                registered += tensor.numel() * tensor.element_size()
        self.registered_bytes += registered
        return registered

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> RowReadStats:
        """Read expert ``rows`` of every named tensor in one io_uring batch."""
        missing = [name for name in destinations if name not in self._sources]
        if missing:
            raise ValueError(f"expert file reader does not cover {missing}")
        start = time.perf_counter_ns()
        read_plans(
            self._reader,
            [
                self._sources[name].plan(rows, destination, destination_rows)
                for name, destination in destinations.items()
            ],
        )
        count = rows.numel()
        return RowReadStats(
            rows=count,
            file_bytes=count
            * sum(self._sources[name].row_bytes for name in destinations),
            read_ns=time.perf_counter_ns() - start,
        )

    def close(self) -> None:
        """Nothing to release: the io_uring reader belongs to the process."""
```

- [ ] **Step 6: Commit, push and run the new tests plus `CPU_SUITE`**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/layers/moe/expert_row_source.py python/sglang/srt/layers/moe/expert_file_reader.py
git commit -m "feat(moe): add the expert row-source protocol and make the file reader one

ExpertRowSource, RowReadStats, ReadTicket and HostSlotLayout describe the
disk-to-RAM seam; TensorRowSource is the dense-tensor fallback, and
ExpertFileRowReader reports read stats and opens repacked groups with
from_group.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/layers/moe/test_expert_row_source.py test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py'
```
Expected:
- The 12 tests of `test_expert_row_source.py` pass (4 of them need liburing, which divix01 has).
- The four existing `test_expert_file_reader.py` tests still pass.
- The only failure is the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`.

---

### Task 3: Route every host read through the row source, plus the `SGLANG_MOE_EXPERT_ROW_SOURCE` knob

The streamer gains a `row_source=` kwarg. By default it builds the row source from `SGLANG_MOE_EXPERT_ROW_SOURCE` through the format's `default_row_source(kind)`. `file_row_reader` becomes an alias property of `row_source`, because the host arena and the tests assign and read it.

One method, `read_host_rows`, now serves every host read:
- the pinned tier's admissions;
- `_copy_source_rows`' pageable branch;
- the uncached eager gather.

It makes one row-source call for all covered names, and uses dense-tensor reads for the rest. A tensor whose format has no dense source (`source(name) is None`) is treated as pageable, so its rows come only from the row source.

The readable branches, which serve production prefill, keep their exact code.

**Files:**
- Modify: `python/sglang/srt/environ.py`, after `SGLANG_MOE_EXPERT_FILE_READER = EnvStr("mmap")` (`:502`).
- Modify: `python/sglang/srt/layers/moe/expert_format.py`, adding `resolve_row_source_kind`.
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`. Edit the imports, `ExpertPinnedHostCache.__init__` and `ensure_rows`, `ExpertStreamer.__init__` and its accessors, `_read_host_rows` (removed), `_copy_source_rows`, and the tail of `gather` (it becomes `_gather_uncached`).
- Test (modify): `test/registered/unit/layers/moe/test_expert_row_source.py`, `test/registered/unit/layers/moe/test_expert_format.py`, `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`.

**Interfaces:**
- Consumes (Task 1): `ExpertStreamer.format`, `.specs`, `.spec(name)`, `.source(name)`, `DenseLayerFormat.default_row_source(layer, specs, kind)`.
- Consumes (Task 2): `RowReadStats`, `TensorRowSource(lookup, names, num_experts)`, `CountingRowSource` (tests).
- Produces:
  - `envs.SGLANG_MOE_EXPERT_ROW_SOURCE`, an `EnvStr` with default `"auto"`.
  - `sglang.srt.layers.moe.expert_format.resolve_row_source_kind() -> str`.
  - `ExpertStreamer(layer, tensor_names, *, layer_id=None, format=None, row_source=<default>)`. Omitting `row_source` resolves it from the knob; passing `None` means "no row source".
  - `ExpertStreamer.row_source: ExpertRowSource | None`.
  - `ExpertStreamer.file_row_reader`, a property aliasing `row_source`, with a setter.
  - `ExpertStreamer.__init__` raises `ValueError` when `row_source.num_experts` differs from the layer's.
  - `ExpertStreamer.read_host_rows(rows_cpu: Tensor, destinations: dict[str, Tensor], destination_rows: Tensor | None = None) -> RowReadStats`. Its `rows` field counts the requested rows once, not once per source.
  - `ExpertStreamer._gather_uncached(source_ids, compact_ids, topk_ids)`. This is the no-cache eager gather, moved out of `gather`.

- [ ] **Step 1: Add the failing routing tests to `test_expert_row_source.py`**

Add these imports to the import block of `test/registered/unit/layers/moe/test_expert_row_source.py`:

```python
from types import SimpleNamespace

from sglang.srt.layers.moe.expert_format import DenseLayerFormat
from sglang.srt.layers.moe.expert_stream import ExpertStreamer
```

Insert these helpers and the class above the `if __name__ == "__main__":` block:

```python
def _layer(names=("a", "b", "c"), experts=6):
    layer = torch.nn.Module()
    for position, name in enumerate(names):
        values = torch.arange(experts * 4, dtype=torch.int16).reshape(experts, 4)
        setattr(
            layer,
            name,
            torch.nn.Parameter(values + 100 * position, requires_grad=False),
        )
    return layer


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


class _PartlySpecOnly(DenseLayerFormat):
    """Dense specs, but ``b`` has no dense source."""

    def source(self, layer, name):
        return None if name == "b" else super().source(layer, name)


class TestStreamerRowRouting(unittest.TestCase):
    def test_an_explicit_row_source_is_the_file_row_reader(self):
        layer = _layer()
        source = CountingRowSource({"a": layer.a.data})
        streamer = ExpertStreamer(layer, ("a", "b", "c"), row_source=source)
        self.assertIs(streamer.row_source, source)
        self.assertIs(streamer.file_row_reader, source)
        # ExpertHostArena.bind drops the reader like this.
        streamer.file_row_reader = None
        self.assertIsNone(streamer.row_source)

    def test_covered_names_share_one_row_source_call(self):
        names = tuple("abcdef")
        layer = _layer(names)
        source = CountingRowSource({name: getattr(layer, name).data for name in names})
        streamer = ExpertStreamer(layer, names, row_source=source)
        destinations = {name: torch.zeros(3, 4, dtype=torch.int16) for name in names}
        stats = streamer.read_host_rows(torch.tensor([4, 1, 5]), destinations)
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(source.calls[0].names, names)
        self.assertIsNone(source.calls[0].destination_rows)
        for name in names:
            self.assertTrue(
                torch.equal(destinations[name], getattr(layer, name).data[[4, 1, 5]])
            )
        self.assertEqual(stats.rows, 3)
        self.assertEqual(stats.file_bytes, 3 * source.file_bytes_per_expert)

    def test_uncovered_names_read_their_dense_source(self):
        layer = _layer()
        source = CountingRowSource({"a": layer.a.data})
        streamer = ExpertStreamer(layer, ("a", "b", "c"), row_source=source)
        destinations = {name: torch.zeros(3, 4, dtype=torch.int16) for name in "abc"}
        stats = streamer.read_host_rows(
            torch.tensor([5, 3]), destinations, torch.tensor([2, 0])
        )
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(source.calls[0].names, ("a",))
        self.assertEqual(source.calls[0].destination_rows, [2, 0])
        for name in "abc":
            self.assertTrue(
                torch.equal(destinations[name][[2, 0]], getattr(layer, name).data[[5, 3]])
            )
        self.assertEqual(stats.rows, 2)

    def test_a_tensor_with_neither_source_is_refused(self):
        layer = _layer(("a", "b"))
        streamer = ExpertStreamer(
            layer, ("a", "b"), format=_PartlySpecOnly(("a", "b")), row_source=None
        )
        self.assertIsNone(streamer.source("b"))
        with self.assertRaisesRegex(ValueError, "no row source covers"):
            streamer.read_host_rows(
                torch.tensor([0]), {"b": torch.zeros(1, 4, dtype=torch.int16)}
            )

    def test_a_row_source_must_hold_the_layers_experts(self):
        layer = _layer()
        source = CountingRowSource({"a": torch.zeros(5, 4, dtype=torch.int16)})
        with self.assertRaisesRegex(ValueError, "row source holds 5 experts"):
            ExpertStreamer(layer, ("a", "b", "c"), row_source=source)

    def test_a_row_source_serves_a_tensor_without_a_dense_source(self):
        layer = _layer(("a", "b"))
        source = CountingRowSource({"b": layer.b.data})
        streamer = ExpertStreamer(
            layer, ("a", "b"), format=_PartlySpecOnly(("a", "b")), row_source=source
        )
        destination = torch.zeros(2, 4, dtype=torch.int16)
        streamer.read_host_rows(torch.tensor([1, 4]), {"b": destination})
        self.assertTrue(torch.equal(destination, layer.b.data[[1, 4]]))
```

- [ ] **Step 2: Add the failing knob tests to `test_expert_format.py`**

Insert this class above the `if __name__ == "__main__":` block of `test/registered/unit/layers/moe/test_expert_format.py`:

```python
class TestRowSourceKnob(unittest.TestCase):
    def _layer(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        return layer

    def test_auto_keeps_the_mmap_default_of_no_reader(self):
        with (
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("auto"),
            envs.SGLANG_MOE_EXPERT_FILE_READER.override("mmap"),
        ):
            self.assertIsNone(ExpertStreamer(self._layer(), ("rows",)).row_source)

    def test_tensor_kind_builds_no_reader(self):
        with (
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("tensor"),
            envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"),
        ):
            self.assertIsNone(ExpertStreamer(self._layer(), ("rows",)).row_source)

    def test_files_kind_needs_an_io_uring_reader_and_expert_files(self):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("files"):
            with envs.SGLANG_MOE_EXPERT_FILE_READER.override("mmap"):
                with self.assertRaisesRegex(ValueError, "SGLANG_MOE_EXPERT_FILE_READER"):
                    ExpertStreamer(self._layer(), ("rows",))
            with envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"):
                with self.assertRaisesRegex(ValueError, "has no expert file"):
                    ExpertStreamer(self._layer(), ("rows",))

    def test_unknown_kind_is_refused(self):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
            with self.assertRaisesRegex(ValueError, "no row source kind 'shards'"):
                ExpertStreamer(self._layer(), ("rows",))

    def test_an_explicit_row_source_ignores_the_knob(self):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
            streamer = ExpertStreamer(self._layer(), ("rows",), row_source=None)
        self.assertIsNone(streamer.row_source)
```

- [ ] **Step 3: Add the CUDA routing tests**

Insert this class above the `if __name__ == "__main__":` block of `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`:

```python
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRowSourceRoutingCuda(unittest.TestCase):
    def test_uncached_pageable_rows_read_every_name_in_one_call(self):
        from sglang.test.moe_expert_fakes import CountingRowSource

        layer = _nvfp4_layer(pinned=False)
        source = CountingRowSource(
            {name: getattr(layer, name).data for name in NVFP4_STREAM_TENSORS[:4]}
        )
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS, row_source=source)
        ids = torch.tensor([[6, 1, 3]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(source.calls[0].names, NVFP4_STREAM_TENSORS[:4])
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].view(torch.uint8).cpu(),
                    _source_bytes(layer, name, ids),
                ),
                name,
            )

    def test_cached_misses_read_every_name_in_one_call(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.test.moe_expert_fakes import CountingRowSource

        layer = _nvfp4_layer(pinned=False)
        source = CountingRowSource(
            {name: getattr(layer, name).data for name in NVFP4_STREAM_TENSORS[:4]}
        )
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS, row_source=source)
        ExpertHotCache(streamer, 1).reassign([2])
        before = len(source.calls)
        ids = torch.tensor([[2, 5, 7]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        self.assertEqual(len(source.calls), before + 1)
        self.assertEqual(sorted(source.calls[-1].rows), [5, 7])
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].view(torch.uint8).cpu(),
                    _source_bytes(layer, name, ids),
                ),
                name,
            )
```

- [ ] **Step 4: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add test/registered/unit/layers/moe/test_expert_row_source.py test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py
git commit -m "test(moe): route host expert reads through one row-source call

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/layers/moe/test_expert_row_source.py::TestStreamerRowRouting test/registered/unit/layers/moe/test_expert_format.py::TestRowSourceKnob'
```
Expected: FAIL, 11 tests.
- All 6 `TestStreamerRowRouting` tests fail with `TypeError: ExpertStreamer.__init__() got an unexpected keyword argument 'row_source'`.
- All 5 `TestRowSourceKnob` tests fail with `AttributeError` on `envs.SGLANG_MOE_EXPERT_ROW_SOURCE`.

- [ ] **Step 5: Add the env var**

Follow `/home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins/.claude/skills/env-var-conventions/SKILL.md`: an `EnvStr` in `Envs`, grouped with the expert-streaming entries, read with `.get()`. In `python/sglang/srt/environ.py`:

Old:
```python
    SGLANG_MOE_EXPERT_FILE_READER = EnvStr("mmap")
```
New:
```python
    SGLANG_MOE_EXPERT_FILE_READER = EnvStr("mmap")
    # Where host expert rows are read from: auto | files | tensor, or a kind the
    # expert format defines. auto keeps each format's default (dense NVFP4 layers:
    # their expert files through io_uring, unless SGLANG_MOE_EXPERT_FILE_READER=mmap).
    SGLANG_MOE_EXPERT_ROW_SOURCE = EnvStr("auto")
```

- [ ] **Step 6: Add `resolve_row_source_kind` to `expert_format.py`**

Append to `python/sglang/srt/layers/moe/expert_format.py`:

```python
def resolve_row_source_kind() -> str:
    """The row source kind ``SGLANG_MOE_EXPERT_ROW_SOURCE`` selects (default ``auto``)."""
    kind = envs.SGLANG_MOE_EXPERT_ROW_SOURCE.get().strip()
    if not kind:
        raise ValueError("SGLANG_MOE_EXPERT_ROW_SOURCE must name a row source kind")
    return kind
```

- [ ] **Step 7: Edit `expert_stream.py` — imports and the default sentinel**

Old:
```python
from dataclasses import asdict, dataclass
```
New:
```python
from dataclasses import asdict, dataclass, replace
```
Old:
```python
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertFormat,
    ExpertTensorSpec,
    iter_expert_streamers,
)
```
New:
```python
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertFormat,
    ExpertTensorSpec,
    iter_expert_streamers,
    resolve_row_source_kind,
)
from sglang.srt.layers.moe.expert_row_source import (
    ExpertRowSource,
    RowReadStats,
    TensorRowSource,
)
```
Old:
```python
NVFP4_STREAM_TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
    "g1_alphas",
    "g2_alphas",
)
```
New:
```python
NVFP4_STREAM_TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
    "g1_alphas",
    "g2_alphas",
)
# ExpertStreamer's row_source default: resolve it from SGLANG_MOE_EXPERT_ROW_SOURCE.
_DEFAULT_ROW_SOURCE = object()
```

- [ ] **Step 8: Edit `ExpertPinnedHostCache` — register with, and read through, the row source**

Old (in `__init__`):
```python
        file_row_reader = getattr(streamer, "file_row_reader", None)
        if file_row_reader is not None:
            file_row_reader.register_destinations(self.tensors.values())
```
New:
```python
        row_source = streamer.row_source
        if row_source is not None:
            row_source.register_destinations(self.tensors.values())
```
Old (in `ensure_rows`):
```python
        file_row_reader = getattr(self.streamer, "file_row_reader", None)
        try:
            for name in self.cached_names:
                if file_row_reader is not None and file_row_reader.covers(name):
                    continue
                source = _tensor_data(getattr(self.streamer.layer, name))
                for source_id, slot in zip(source_ids_cpu, slots_cpu):
                    torch.index_select(
                        source,
                        0,
                        source_id.reshape(1),
                        out=self.tensors[name][int(slot) : int(slot) + 1],
                    )
            if file_row_reader is not None:
                file_row_reader.read(
                    source_ids_cpu,
                    {
                        name: self.tensors[name]
                        for name in self.cached_names
                        if file_row_reader.covers(name)
                    },
                    slots_cpu,
                )
        except BaseException:
```
New:
```python
        try:
            self.streamer.read_host_rows(
                source_ids_cpu,
                {name: self.tensors[name] for name in self.cached_names},
                slots_cpu,
            )
        except BaseException:
```

- [ ] **Step 9: Edit `ExpertStreamer.__init__` — the `row_source` kwarg**

Old:
```python
        layer_id: int | None = None,
        format: ExpertFormat | None = None,
    ):
```
New:
```python
        layer_id: int | None = None,
        format: ExpertFormat | None = None,
        row_source: ExpertRowSource | None | object = _DEFAULT_ROW_SOURCE,
    ):
```
Old:
```python
        self.file_row_reader = self.format.default_row_source(
            layer, self.specs, "auto"
        )
```
New:
```python
        if row_source is _DEFAULT_ROW_SOURCE:
            row_source = self.format.default_row_source(
                layer, self.specs, resolve_row_source_kind()
            )
        self.row_source = row_source
        # Dense sources serve every name the row source does not cover.
        self._tensor_rows = TensorRowSource(
            self.source, self.tensor_names, self.num_experts
        )
```

- [ ] **Step 9b: Check the row source's expert count**

Directly after the line `self.row_source = row_source` from Step 9, insert:

```python
        if row_source is not None and row_source.num_experts != self.num_experts:
            raise ValueError(
                f"row source holds {row_source.num_experts} experts, but the layer "
                f"has {self.num_experts}"
            )
```

- [ ] **Step 10: Edit the accessors — the alias property and the file-bytes property**

Old:
```python
    @property
    def file_source_bytes_per_expert(self) -> int | None:
        """File bytes one expert row reads; None keeps eager gathers out of the pinned tier."""
        return self.format.file_source_bytes_per_expert(
            self.layer, self.file_row_reader
        )
```
New:
```python
    @property
    def file_row_reader(self) -> ExpertRowSource | None:
        """Alias of ``row_source``; the host arena drops it by assigning None."""
        return self.row_source

    @file_row_reader.setter
    def file_row_reader(self, value: ExpertRowSource | None) -> None:
        self.row_source = value

    @property
    def file_source_bytes_per_expert(self) -> int | None:
        """File bytes one expert row reads; None keeps eager gathers out of the pinned tier."""
        return self.format.file_source_bytes_per_expert(self.layer, self.row_source)

    def read_host_rows(
        self,
        rows_cpu: torch.Tensor,
        destinations: dict[str, torch.Tensor],
        destination_rows: torch.Tensor | None = None,
    ) -> RowReadStats:
        """Fill host ``destinations`` with expert ``rows_cpu``.

        Every name the row source covers is read in one call, so a source that
        reads a whole on-disk expert row per request pays for it once. The other
        names are read from their dense sources. Rows land in
        ``destination_rows`` of each destination, or in its leading rows.
        """
        row_source = self.row_source
        covered = {
            name: destination
            for name, destination in destinations.items()
            if row_source is not None and row_source.covers(name)
        }
        uncovered = {
            name: destination
            for name, destination in destinations.items()
            if name not in covered
        }
        missing = [name for name in uncovered if not self._tensor_rows.covers(name)]
        if missing:
            raise ValueError(
                f"no row source covers expert tensors {missing} and they have "
                "no dense source"
            )
        stats = RowReadStats()
        if uncovered:
            stats = stats + _read_rows(
                self._tensor_rows, rows_cpu, uncovered, destination_rows
            )
        if covered:
            stats = stats + _read_rows(row_source, rows_cpu, covered, destination_rows)
        if stats.rows:
            stats = replace(stats, rows=rows_cpu.numel())
        return stats
```

Also add this module-level helper just above `class ExpertStreamer:`:

```python
def _read_rows(
    source: ExpertRowSource,
    rows_cpu: torch.Tensor,
    destinations: dict[str, torch.Tensor],
    destination_rows: torch.Tensor | None,
) -> RowReadStats:
    # Without destination rows, call read with two arguments, as the pageable
    # gather path always called the file reader.
    if destination_rows is None:
        return source.read(rows_cpu, destinations)
    return source.read(rows_cpu, destinations, destination_rows)
```

- [ ] **Step 11: Replace `_read_host_rows` with `_read_pageable_rows`**

Replace the whole `_read_host_rows` method (from `    def _read_host_rows(` through `            torch.index_select(source, 0, cpu_ids, out=host_output)`) with:

```python
    def _read_pageable_rows(
        self,
        cpu_ids: torch.Tensor | None,
        outputs: dict[str, torch.Tensor],
        row_count: int,
        capacity: int,
    ) -> None:
        """Read ``outputs``' rows on the host in one batch into pinned staging, then copy them up."""
        assert cpu_ids is not None
        host_outputs = {
            name: _pinned_staging_buffer(
                name,
                row_count,
                capacity,
                self.spec(name).row_shape,
                self.spec(name).dtype,
            )
            for name in outputs
        }
        self.read_host_rows(cpu_ids, host_outputs)
        for name, output in outputs.items():
            output.copy_(host_outputs[name], non_blocking=True)
```

- [ ] **Step 12: Replace `_copy_source_rows`**

Replace the whole `_copy_source_rows` method with the code below. The CUDA, DMA and readable-kernel branches keep today's exact code; only the pageable branch changes.

```python
    def _copy_source_rows(
        self, source_ids: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> int:
        """Fill supplied CUDA rows through the existing bounded host buffers.

        With the ``dma`` copy backend, rows of registered or pinned host tensors
        go through the CUDA copy engine, merged into runs of consecutive rows.
        Graph capture keeps the pull kernel. Rows of pageable host tensors, and
        of tensors without a dense source, are read by the row sources in one
        batch into pinned staging. Returns the bytes the copy engine moved, so
        a build without it shows up as zero.
        """
        row_count = source_ids.numel()
        use_dma = (
            self.expert_copy_backend == "dma"
            and _aot_transfer_available()
            and not torch.cuda.is_current_stream_capturing()
        )
        dma_rows = None
        copy_engine_bytes = 0
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        sources = {name: self.source(name) for name in self.tensor_names}
        host_sources = {
            name: source
            for name, source in sources.items()
            if source is None or source.device.type == "cpu"
        }
        readable = {
            name: source is not None and is_gpu_readable_host_tensor(source)
            for name, source in host_sources.items()
        }
        pageable_source = not all(readable.values())
        cpu_ids = (
            _copy_indices_to_cpu(source_ids, capacity) if pageable_source else None
        )
        pageable_outputs: dict[str, torch.Tensor] = {}
        for name, output in outputs.items():
            source = sources[name]
            if source is not None and source.device.type == "cuda":
                torch.index_select(source, 0, source_ids, out=output)
            elif readable[name] and use_dma and source.ndim >= 2:
                if dma_rows is None:
                    dma_rows = source_ids.tolist()
                self._dma_backend.copy_rows(source, output, dma_rows, range(row_count))
                copy_engine_bytes += (
                    row_count * source.numel() * source.element_size() // self.num_experts
                )
            elif readable[name]:
                row_bytes = source.numel() * source.element_size() // self.num_experts
                _gather_host_rows_kernel[(row_count, triton.cdiv(row_bytes, 1024))](
                    source.view(torch.uint8),
                    source_ids,
                    output.view(torch.uint8),
                    row_bytes,
                    BLOCK=1024,
                )
            else:
                pageable_outputs[name] = output
        if pageable_outputs:
            self._read_pageable_rows(cpu_ids, pageable_outputs, row_count, capacity)
        return copy_engine_bytes
```

- [ ] **Step 13: Move the uncached eager gather out of `gather` into `_gather_uncached`**

In `gather`, replace everything from the line `        row_count = source_ids.numel()` (just after the `next_layer_prefetch(source_ids)` block) to the end of the method with:

```python
        if self.hot_cache is not None and self.hot_cache.capacity:
            return self._gather_cached(source_ids, compact_ids, topk_ids)
        if (
            self.pinned_host_cache is not None
            and self.pinned_host_cache.capacity
            and self.file_source_bytes_per_expert is not None
        ):
            return self._gather_pinned_host(source_ids, compact_ids, topk_ids)
        return self._gather_uncached(source_ids, compact_ids, topk_ids)

    def _gather_uncached(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gather rows straight from their sources, with no hot or pinned cache.

        Returned tensors follow the leading-dimension rule of ``_gather_cached``.
        """
        row_count = source_ids.numel()
        self.last_gather_stats = ExpertGatherStats(
            row_count,
            0,
            row_count,
            row_count * (self.bytes_per_expert - self.host_bytes_per_expert),
            row_count * self.host_bytes_per_expert,
            row_count * self.bytes_per_expert,
            routed_rows=compact_ids.numel(),
            routed_miss_rows=compact_ids.numel(),
            unique_miss_rows=row_count,
        )
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        sources = {name: self.source(name) for name in self.tensor_names}
        pageable_source = any(
            source is None
            or (source.device.type == "cpu" and not is_gpu_readable_host_tensor(source))
            for source in sources.values()
        )
        cpu_ids = (
            _copy_indices_to_cpu(source_ids, capacity) if pageable_source else None
        )
        kernel_rows = capacity if should_dedup(topk_ids) else row_count
        padded: dict[str, torch.Tensor] = {}
        pageable_outputs: dict[str, torch.Tensor] = {}
        for name in self.tensor_names:
            source = sources[name]
            spec = self.spec(name)
            padded[name] = _staging_buffer(
                name,
                kernel_rows,
                capacity,
                spec.row_shape,
                spec.dtype,
                topk_ids.device,
            )
            output = padded[name][:row_count]
            if source is not None and source.device.type == "cuda":
                torch.index_select(source, 0, source_ids, out=output)
            elif source is not None and is_gpu_readable_host_tensor(source):
                source_bytes = source.view(torch.uint8)
                output_bytes = output.view(torch.uint8)
                row_bytes = source_bytes.numel() // self.num_experts
                block = 1024
                _gather_host_rows_kernel[(row_count, triton.cdiv(row_bytes, block))](
                    source_bytes,
                    source_ids,
                    output_bytes,
                    row_bytes,
                    BLOCK=block,
                )
            else:
                pageable_outputs[name] = output
        if pageable_outputs:
            self._read_pageable_rows(cpu_ids, pageable_outputs, row_count, capacity)
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), padded
```

Then run `grep -n "_read_host_rows\|getattr(self.streamer, \"file_row_reader\"" python/sglang/srt/layers/moe/expert_stream.py`. Expected: no output.

- [ ] **Step 14: Commit, push and run the new tests plus `CPU_SUITE`**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/expert_format.py python/sglang/srt/layers/moe/expert_stream.py
git commit -m "feat(moe): read host expert rows through one batched row-source call

ExpertStreamer takes row_source= (default: SGLANG_MOE_EXPERT_ROW_SOURCE via
the format); pinned admissions, pageable staging and the uncached gather
all use read_host_rows, and tensors without a dense source read only
through the row source. Readable-source branches are unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/layers/moe/test_expert_row_source.py test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py'
```
Expected:
- `TestStreamerRowRouting` (6) and `TestRowSourceKnob` (5) pass.
- `test_expert_route_plan.py::TestEagerRouteDedup::test_cached_gather_holds_the_kernel_expert_count_at_the_dedup_limit` still passes; it drives `_gather_cached`.
- The only failure is the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`.

---

### Task 4: Carry row-read stats in `ExpertGatherStats` and the manager counters

Reads a row source makes during an eager gather are added to that gather's `ExpertGatherStats`, as new trailing fields with defaults. The four positional constructions stay valid. Reads outside a gather, such as hot-cache promotions and startup seeding, accumulate in `streamer.background_read_stats`. The hot-cache manager sums the gather fields into new trailing `_OperationalCounters` fields. A gather that made no row reads keeps the exact stats object its path built, so production gathers are unaffected: they read no host rows.

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`. Edit `ExpertGatherStats` (`:50-75`), `ExpertStreamer.__init__`, `read_host_rows`, and the eager dispatch at the end of `gather`.
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py`. Edit `_OperationalCounters` (`:768-819`), add `_add_host_read_counters` after it, and edit the observer loop (`:2482-2486`).
- Test (modify): `test/registered/unit/layers/moe/test_expert_row_source.py`

**Interfaces:**
- Consumes (Task 3): `ExpertStreamer.read_host_rows(...) -> RowReadStats`, `_gather_cached`, `_gather_pinned_host`, `_gather_uncached`, `RowReadStats.__add__`.
- Produces:
  - `ExpertGatherStats` gains trailing fields `host_read_rows`, `host_read_file_bytes`, `host_read_split_bytes`, `host_read_ns` and `host_split_ns`. All are `int = 0`.
  - `ExpertStreamer.background_read_stats: RowReadStats`.
  - `ExpertStreamer._gather_eager_rows(source_ids, compact_ids, topk_ids) -> tuple[Tensor, dict[str, Tensor]]`. It dispatches to the hot, pinned or uncached path and adds the gather's reads to `last_gather_stats`.
  - `_OperationalCounters` gains the same five trailing fields.
  - `expert_hot_cache._add_host_read_counters(counters, stats) -> None`.

- [ ] **Step 1: Add the failing tests**

Add to the import block of `test/registered/unit/layers/moe/test_expert_row_source.py`:

```python
import dataclasses

from sglang.srt.layers.moe.expert_stream import ExpertGatherStats
```

Insert above the `if __name__ == "__main__":` block:

```python
class TestGatherReadStats(unittest.TestCase):
    def test_positional_gather_stats_fields_keep_their_order(self):
        names = [field.name for field in dataclasses.fields(ExpertGatherStats)]
        self.assertEqual(
            names[:15],
            [
                "requested_rows",
                "hot_hit_rows",
                "miss_rows",
                "d2d_bytes",
                "h2d_bytes",
                "source_bytes",
                "pinned_host_hit_rows",
                "pinned_host_miss_rows",
                "pinned_host_populated_bytes",
                "transfer_wait_ns",
                "gather_fallback_used",
                "copy_engine_bytes",
                "routed_rows",
                "routed_miss_rows",
                "unique_miss_rows",
            ],
        )
        self.assertEqual(
            names[15:],
            [
                "host_read_rows",
                "host_read_file_bytes",
                "host_read_split_bytes",
                "host_read_ns",
                "host_split_ns",
            ],
        )
        self.assertEqual(ExpertGatherStats(1, 2).host_read_rows, 0)

    def _all_miss_streamer(self, layer, names, row_source, through_row_source):
        streamer = ExpertStreamer(layer, names, row_source=row_source)
        streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)

        def copy_rows(source_ids, outputs):
            if through_row_source:
                staging = {
                    name: torch.empty_like(output) for name, output in outputs.items()
                }
                streamer.read_host_rows(source_ids.cpu(), staging)
                for name, output in outputs.items():
                    output.copy_(staging[name])
            else:
                for name, output in outputs.items():
                    torch.index_select(
                        getattr(layer, name).data, 0, source_ids, out=output
                    )
            return 0

        streamer._copy_source_rows = copy_rows
        return streamer

    def test_reads_inside_a_gather_land_in_its_stats(self):
        layer = _layer(("a", "b"))
        source = CountingRowSource({"a": layer.a.data, "b": layer.b.data})
        streamer = self._all_miss_streamer(layer, ("a", "b"), source, True)
        ids = torch.tensor([[1, 4], [4, 2]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_eager_rows(source_ids, compact_ids, ids)
        stats = streamer.last_gather_stats
        self.assertEqual(stats.requested_rows, 3)
        self.assertEqual(stats.host_read_rows, 3)
        self.assertEqual(stats.host_read_file_bytes, 3 * source.file_bytes_per_expert)
        self.assertEqual(stats.host_read_ns, 3)
        self.assertEqual(stats.host_read_split_bytes, 0)
        self.assertEqual(streamer.background_read_stats, RowReadStats())
        self.assertTrue(torch.equal(tensors["a"][compact.long()], layer.a.data[ids]))

    def test_a_gather_without_host_reads_keeps_its_stats_object(self):
        layer = _layer(("a",))
        streamer = self._all_miss_streamer(layer, ("a",), None, False)
        created = []
        original = streamer._gather_cached

        def gather_cached(*args):
            result = original(*args)
            created.append(streamer.last_gather_stats)
            return result

        streamer._gather_cached = gather_cached
        ids = torch.tensor([[1, 4], [4, 2]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        streamer._gather_eager_rows(source_ids, compact_ids, ids)
        self.assertIs(streamer.last_gather_stats, created[0])
        self.assertEqual(streamer.last_gather_stats.host_read_rows, 0)

    def test_reads_outside_a_gather_are_background_reads(self):
        layer = _layer(("a",))
        source = CountingRowSource({"a": layer.a.data})
        streamer = ExpertStreamer(layer, ("a",), row_source=source)
        streamer.read_host_rows(
            torch.tensor([0, 1]), {"a": torch.zeros(2, 4, dtype=torch.int16)}
        )
        self.assertEqual(streamer.background_read_stats.rows, 2)
        self.assertEqual(streamer.last_gather_stats.host_read_rows, 0)

    def test_manager_counters_sum_host_reads(self):
        from sglang.srt.layers.moe.expert_hot_cache import (
            _add_host_read_counters,
            _OperationalCounters,
        )

        counters = _OperationalCounters()
        stats = ExpertGatherStats(
            host_read_rows=2,
            host_read_file_bytes=10,
            host_read_split_bytes=3,
            host_read_ns=7,
            host_split_ns=1,
        )
        _add_host_read_counters(counters, stats)
        _add_host_read_counters(counters, stats)
        _add_host_read_counters(counters, SimpleNamespace())
        self.assertEqual(
            (
                counters.host_read_rows,
                counters.host_read_file_bytes,
                counters.host_read_split_bytes,
                counters.host_read_ns,
                counters.host_split_ns,
            ),
            (4, 20, 6, 14, 2),
        )
```

- [ ] **Step 2: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add test/registered/unit/layers/moe/test_expert_row_source.py
git commit -m "test(moe): report row-source reads in gather stats and manager counters

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/layers/moe/test_expert_row_source.py::TestGatherReadStats'
```
Expected: FAIL, all 5 tests. `test_positional_...` fails because `names[15:]` is `[]`. The gather tests fail with `AttributeError: 'ExpertStreamer' object has no attribute '_gather_eager_rows'`. The counters test fails with `ImportError: cannot import name '_add_host_read_counters'`.

- [ ] **Step 3: Add the trailing stats fields**

In `python/sglang/srt/layers/moe/expert_stream.py`, old:
```python
    routed_rows: int = 0
    routed_miss_rows: int = 0
    unique_miss_rows: int = 0


@dataclass
class PinnedHostCacheStats:
```
New:
```python
    routed_rows: int = 0
    routed_miss_rows: int = 0
    unique_miss_rows: int = 0
    # Host rows the row sources read during this gather (summed RowReadStats).
    # Trailing and defaulted: the stats are constructed positionally.
    host_read_rows: int = 0
    host_read_file_bytes: int = 0
    host_read_split_bytes: int = 0
    host_read_ns: int = 0
    host_split_ns: int = 0


@dataclass
class PinnedHostCacheStats:
```

- [ ] **Step 4: Track reads in the streamer**

In `ExpertStreamer.__init__`, old:
```python
        self.last_gather_stats = ExpertGatherStats()
        self.bytes_per_expert = sum(spec.row_bytes for spec in self.specs)
```
New:
```python
        self.last_gather_stats = ExpertGatherStats()
        # Row-source reads outside eager gathers (promotions, seeding, direct calls).
        self.background_read_stats = RowReadStats()
        self._gather_read_stats: RowReadStats | None = None
        self.bytes_per_expert = sum(spec.row_bytes for spec in self.specs)
```

At the end of `read_host_rows`, old:
```python
        if stats.rows:
            stats = replace(stats, rows=rows_cpu.numel())
        return stats
```
New:
```python
        if stats.rows:
            stats = replace(stats, rows=rows_cpu.numel())
        self._record_read(stats)
        return stats

    def _record_read(self, stats: RowReadStats) -> None:
        if self._gather_read_stats is not None:
            self._gather_read_stats = self._gather_read_stats + stats
        else:
            self.background_read_stats = self.background_read_stats + stats
```

- [ ] **Step 5: Wrap the eager dispatch**

In `gather`, old:
```python
        if self.hot_cache is not None and self.hot_cache.capacity:
            return self._gather_cached(source_ids, compact_ids, topk_ids)
        if (
            self.pinned_host_cache is not None
            and self.pinned_host_cache.capacity
            and self.file_source_bytes_per_expert is not None
        ):
            return self._gather_pinned_host(source_ids, compact_ids, topk_ids)
        return self._gather_uncached(source_ids, compact_ids, topk_ids)

    def _gather_uncached(
```
New:
```python
        return self._gather_eager_rows(source_ids, compact_ids, topk_ids)

    def _gather_eager_rows(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one eager gather and add the host reads it caused to its stats."""
        self._gather_read_stats = RowReadStats()
        try:
            result = self._dispatch_eager_rows(source_ids, compact_ids, topk_ids)
            read = self._gather_read_stats
        finally:
            self._gather_read_stats = None
        if read.rows:
            self.last_gather_stats = replace(
                self.last_gather_stats,
                host_read_rows=read.rows,
                host_read_file_bytes=read.file_bytes,
                host_read_split_bytes=read.split_bytes,
                host_read_ns=read.read_ns,
                host_split_ns=read.split_ns,
            )
        return result

    def _dispatch_eager_rows(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.hot_cache is not None and self.hot_cache.capacity:
            return self._gather_cached(source_ids, compact_ids, topk_ids)
        if (
            self.pinned_host_cache is not None
            and self.pinned_host_cache.capacity
            and self.file_source_bytes_per_expert is not None
        ):
            return self._gather_pinned_host(source_ids, compact_ids, topk_ids)
        return self._gather_uncached(source_ids, compact_ids, topk_ids)

    def _gather_uncached(
```

- [ ] **Step 6: Sum the reads into the manager counters**

In `python/sglang/srt/layers/moe/expert_hot_cache.py`, old:
```python
    side_pull_residual_routes: int = 0
    side_pull_useful_precision: float = 0.0


_PHASES = {
```
New:
```python
    side_pull_residual_routes: int = 0
    side_pull_useful_precision: float = 0.0
    host_read_rows: int = 0
    host_read_file_bytes: int = 0
    host_read_split_bytes: int = 0
    host_read_ns: int = 0
    host_split_ns: int = 0


def _add_host_read_counters(counters: _OperationalCounters, stats: Any) -> None:
    """Sum one gather's row-source reads into a phase/layer's counters."""
    counters.host_read_rows += getattr(stats, "host_read_rows", 0)
    counters.host_read_file_bytes += getattr(stats, "host_read_file_bytes", 0)
    counters.host_read_split_bytes += getattr(stats, "host_read_split_bytes", 0)
    counters.host_read_ns += getattr(stats, "host_read_ns", 0)
    counters.host_split_ns += getattr(stats, "host_split_ns", 0)


_PHASES = {
```
Old (observer loop):
```python
            counters.gather_copy_engine_bytes += getattr(stats, "copy_engine_bytes", 0)
            file_bytes = streamer.file_source_bytes_per_expert
```
New:
```python
            counters.gather_copy_engine_bytes += getattr(stats, "copy_engine_bytes", 0)
            _add_host_read_counters(counters, stats)
            file_bytes = streamer.file_source_bytes_per_expert
```

- [ ] **Step 7: Commit, push and run the new tests plus `CPU_SUITE`**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/layers/moe/expert_hot_cache.py
git commit -m "feat(moe): count row-source reads per gather and per manager phase

ExpertGatherStats gains trailing host_read_* fields filled from the
reads an eager gather made, other reads go to background_read_stats, and
_OperationalCounters sums the gather fields. Metrics and trace rows gain
five host_read_* keys, which stay zero under the host arena.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/layers/moe/test_expert_row_source.py test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py'
```
Expected: the 5 `TestGatherReadStats` tests pass. The only failure is the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`.

---

### Task 5: RAM-tier fixes — exact registered slabs, O(1) LRU, `is_pinned` filter and tier options, injected device, transactional admissions, and the `_gather_cached` out-of-bounds read

**The out-of-bounds read still exists after `844bb9d7a5`.** It was re-verified in this worktree. In `_gather_cached`'s pinned branch (`expert_stream.py:1017-1042`):
1. `pinned_cache.ensure_rows(miss_source_ids[~pinned_hit_mask])` admits every miss.
2. `pinned_cache.copy_rows(miss_source_ids, pinned_outputs)` then runs with no residency check.
3. When one call's distinct misses exceed the layer's pinned capacity, `ensure_rows` evicts rows it just admitted. Their `expert_to_slot` is `-1`, so `_gather_host_rows_kernel` reads `src + (-1) * row_bytes`, which is before the slab.

`_gather_pinned_host` already guards against this with its resident/cold split; `_gather_cached` does not.

**The fix** is a pinned-tier method, `gather_rows`, which admits and copies chunk by chunk.
- **Chunk size.** Each chunk is sized when it starts, from `evictable_rows()`: the capacity minus the resident experts that `is_pinned` protects. A protected resident can never make room. `is_pinned` may also grow during a call: an inclusive hierarchy pins the experts the hot cache reserves, so each admitted row can shrink the room left for the next chunk. A chunk holds at most that many distinct experts.
- **Protection.** The whole chunk, hits and misses alike, is protected while it is admitted.
- **Residency check.** Before its copy, every id of the chunk is checked to be resident on the host. A failure raises a clear `RuntimeError` rather than reading slot −1.
- **Single chunk.** A call that fits in one chunk makes the old calls, `lookup(all)`, then `ensure_rows(misses)`, then `copy_rows(all)`, plus that host-side check. Without `is_pinned`, `evictable_rows()` equals `capacity`.
- **No room.** When no slot is evictable, a chunk with misses raises before touching any slot. An all-hit chunk is still copied.

**Protection inside the victim choice.** `ensure_rows(source_ids, protected=())` protects the requested experts plus `protected`, and passes them to `PinnedSlotLRU.assign`, as `exl3_ram_cache.py` on `dsv41` does. The victim is then the oldest expert that is neither `is_pinned` nor protected. A protected expert is evicted only when nothing else can be; that is the legacy in-call reassignment that `ensure_rows`' over-capacity callers relied on.

**Transactional admission.** `ensure_rows` assigns and reads inside one `try`. On any exception, from `assign` or from the read, it releases every slot this call assigned and republishes the device map. So a failed call leaves no expert mapped to an unread slot.

The tier's other fixes:
- **Slabs.** Each is page-aligned `torch.empty` memory registered with `cudaHostRegister` and sized exactly. `pin_memory=True` went through PyTorch's caching host allocator, which rounds each allocation up to a power of two.
- **Slot bookkeeping.** A CPU-testable `PinnedSlotLRU`: an `OrderedDict` for recency plus a heap of free slots. With nothing protected, it makes the same choices as the old clock scan: lowest free slot first, then the least recently used.
- **Tier options.** `ExpertFormat.pinned_tier_options(layer)` returns keyword arguments for `ExpertPinnedHostCache`, such as `is_pinned`. `ExpertPinnedHostCacheManager.from_model` passes them to each layer's tier, so a format reaches the production constructor path with no edit to `model_runner.py`. `DenseLayerFormat` returns `{}`.
- **Device.** An optional `device=` puts the whole tier on the CPU (unregistered slabs) for tests.
- **Copies.** `copy_rows` gains a CPU-output branch.
- **Bounded pinned-only staging.** `_gather_pinned_host` (a pinned tier with no hot cache) now copies through `gather_rows` straight into its one staging set. It used to stage hits, misses and cold rows in three more sets of `max(rows, 64)` rows each, which for EXL3 would be about 4 × 850 MB of VRAM.

NVFP4 production runs `SGLANG_MOE_PINNED_HOST_MB=0`, so it never constructs this tier.

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_host_tier.py`
- Modify: `python/sglang/srt/layers/moe/expert_format.py`, adding the `pinned_tier_options` hook to `ExpertFormat` and `DenseLayerFormat`.
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`:
  - edit the imports;
  - replace the whole `ExpertPinnedHostCache` class;
  - pass tier options in `ExpertPinnedHostCacheManager.from_model`;
  - change the pinned branch of `_gather_cached`;
  - replace `_gather_pinned_host`.
- Test (create): `test/registered/unit/layers/moe/test_expert_host_tier.py` (CPU)
- Test (modify): `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`

**Interfaces:**
- Consumes (Task 1): `streamer.specs`, `streamer.spec(name)`, `streamer.source(name)`, `streamer.host_bytes_per_expert`, `streamer.file_source_bytes_per_expert`.
- Consumes (Task 3): `streamer.row_source`, `streamer.read_host_rows(rows_cpu, destinations, destination_rows)`.
- Produces, in `sglang.srt.layers.moe.expert_host_tier`:
  - `PAGE_BYTES = 4096`.
  - `class PinnedGatherResult(NamedTuple)`, with fields `hit_rows: int`, `miss_rows: int`, `populated_bytes: int` and `fallback_used: bool`.
  - `class PinnedSlotLRU(capacity: int, is_pinned: Callable[[int], bool] | None = None)`, with:
    - `.slot_to_expert: list[int]` and `.expert_to_slot: OrderedDict[int, int]`;
    - `__contains__(expert_id)`, `touch(expert_id)`;
    - `assign(expert_id, protected: Collection[int] = frozenset()) -> tuple[int, int | None]`, returning the slot and the evicted expert or None. The victim is the oldest expert that is neither `is_pinned` nor in `protected`, else the oldest that is not `is_pinned`, else it raises `RuntimeError("every pinned host slot holds a protected expert")`;
    - `release(slot)`;
    - `mapping(num_experts) -> list[int]`.
  - `allocate_host_slab(rows: int, row_shape: tuple[int, ...], dtype: torch.dtype, *, register: bool) -> torch.Tensor`.
  - `release_host_slabs(slabs: Sequence[torch.Tensor]) -> None`.
- Also produces:
  - `ExpertPinnedHostCache(streamer, capacity, *, device=None, is_pinned=None)`, with public attribute `.is_pinned`.
  - `ExpertPinnedHostCache.evictable_rows() -> int`, the capacity minus the resident experts that `is_pinned` protects.
  - `ExpertPinnedHostCache.gather_rows(source_ids, outputs) -> PinnedGatherResult`. Chunks are sized per chunk from `evictable_rows()`, and each chunk's rows are checked resident before the copy.
  - `ExpertPinnedHostCache.ensure_rows(source_ids, protected: Iterable[int] = ()) -> None`.
  - `ExpertPinnedHostCache.close()`.
  - `ExpertFormat.pinned_tier_options(layer) -> Mapping[str, Any]`, implemented on `DenseLayerFormat` as `{}`. `ExpertPinnedHostCacheManager.from_model` passes the result as keyword arguments to each `ExpertPinnedHostCache`.
  - Unchanged public attributes: `streamer`, `capacity`, `cached_names`, `bytes_per_expert`, `residency_bytes`, `device`, `tensors`, `expert_to_slot` (device tensor), `slot_to_expert` (list), `_expert_to_slot` (mapping read by `ExpertHotCache._prepare_promotion`), `stats`, `capacity_for_budget`, `lookup`, `ensure_rows` and `copy_rows`.

- [ ] **Step 1: Write the failing CPU tests**

Create `test/registered/unit/layers/moe/test_expert_host_tier.py`:

```python
"""CPU tests for the pinned host expert tier: slot LRU, slabs and chunked gathers."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_format import DenseLayerFormat
from sglang.srt.layers.moe.expert_host_tier import (
    PAGE_BYTES,
    PinnedSlotLRU,
    allocate_host_slab,
)
from sglang.srt.layers.moe.expert_stream import (
    ExpertPinnedHostCache,
    ExpertPinnedHostCacheManager,
    ExpertStreamer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _LegacyLRU:
    """The pinned tier's slot choice before PinnedSlotLRU (expert_stream.py at e54c84a7c6)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.slot_to_expert = [-1] * capacity
        self.expert_to_slot = {}
        self.last_used = [0] * capacity
        self.clock = 0
        self.evictions = 0

    def touch(self, expert_id):
        self.clock += 1
        self.last_used[self.expert_to_slot[expert_id]] = self.clock

    def admit(self, missing):
        for expert_id in missing:
            free = next(
                (slot for slot, resident in enumerate(self.slot_to_expert) if resident < 0),
                None,
            )
            evicted = free is None
            if free is None:
                free = min(range(self.capacity), key=self.last_used.__getitem__)
            if evicted:
                self.evictions += 1
                self.expert_to_slot.pop(self.slot_to_expert[free], None)
            self.slot_to_expert[free] = expert_id
            self.expert_to_slot[expert_id] = free
            self.clock += 1
            self.last_used[free] = self.clock


class TestPinnedSlotLRU(unittest.TestCase):
    def test_free_slots_go_lowest_first_then_the_least_recent_is_evicted(self):
        lru = PinnedSlotLRU(3)
        self.assertEqual([lru.assign(expert)[0] for expert in (7, 2, 9)], [0, 1, 2])
        lru.touch(7)
        self.assertEqual(lru.assign(4), (1, 2))
        self.assertEqual(lru.slot_to_expert, [7, 4, 9])
        self.assertNotIn(2, lru)
        self.assertEqual(lru.mapping(10)[4], 1)
        self.assertEqual(lru.mapping(10)[2], -1)

    def test_release_frees_the_slot_for_the_next_assignment(self):
        lru = PinnedSlotLRU(2)
        lru.assign(5)
        lru.assign(6)
        lru.release(0)
        self.assertNotIn(5, lru)
        self.assertEqual(lru.assign(8), (0, None))

    def test_is_pinned_protects_experts_from_eviction(self):
        lru = PinnedSlotLRU(2, is_pinned=lambda expert: expert == 1)
        lru.assign(1)
        lru.assign(2)
        self.assertEqual(lru.assign(3), (1, 2))
        self.assertIn(1, lru)
        everything = PinnedSlotLRU(1, is_pinned=lambda expert: True)
        everything.assign(0)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            everything.assign(1)

    def test_the_calls_own_experts_are_evicted_last(self):
        lru = PinnedSlotLRU(3)
        for expert in (4, 5, 6):
            lru.assign(expert)
        self.assertEqual(lru.assign(7, protected={4, 7}), (1, 5))
        self.assertEqual(lru.assign(8, protected={6, 7, 8}), (0, 4))
        # Only the call's own experts are left: the oldest of them goes.
        self.assertEqual(lru.assign(9, protected={6, 7, 8, 9}), (2, 6))

    def test_random_traffic_matches_the_legacy_slot_choice(self):
        generator = random.Random(4)
        legacy, lru = _LegacyLRU(3), PinnedSlotLRU(3)
        evictions = 0
        for _ in range(500):
            request = generator.sample(range(12), generator.randint(1, 5))
            for expert in request:
                if expert in legacy.expert_to_slot:
                    legacy.touch(expert)
                if expert in lru:
                    lru.touch(expert)
            missing = [expert for expert in request if expert not in legacy.expert_to_slot]
            self.assertEqual(missing, [expert for expert in request if expert not in lru])
            legacy.admit(missing)
            for expert in missing:
                evictions += lru.assign(expert)[1] is not None
            self.assertEqual(lru.slot_to_expert, legacy.slot_to_expert)
            self.assertEqual(evictions, legacy.evictions)


class TestHostSlab(unittest.TestCase):
    def test_slabs_are_page_aligned_and_sized_exactly(self):
        slab = allocate_host_slab(3, (1000,), torch.float32, register=False)
        self.assertEqual(tuple(slab.shape), (3, 1000))
        self.assertEqual(slab.dtype, torch.float32)
        self.assertEqual(slab.data_ptr() % PAGE_BYTES, 0)
        self.assertTrue(slab.is_contiguous())
        self.assertEqual(slab.untyped_storage().nbytes(), 3 * 1000 * 4 + PAGE_BYTES)

    def test_empty_and_scalar_rows(self):
        self.assertEqual(
            tuple(allocate_host_slab(0, (4,), torch.uint8, register=False).shape), (0, 4)
        )
        self.assertEqual(
            tuple(allocate_host_slab(5, (), torch.float32, register=False).shape), (5,)
        )


def _host_layer(experts=8):
    layer = torch.nn.Module()
    layer.host_rows = torch.nn.Parameter(
        torch.arange(experts * 12, dtype=torch.uint8).reshape(experts, 3, 4),
        requires_grad=False,
    )
    layer._nvfp4_file_source_bytes_per_expert = 12
    return layer


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


class TestCpuPinnedTier(unittest.TestCase):
    def test_ensure_and_copy_rows_on_a_cpu_tier(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        self.assertIs(streamer.pinned_host_cache, cache)
        self.assertEqual(cache.tensors["host_rows"].data_ptr() % PAGE_BYTES, 0)
        cache.ensure_rows(torch.tensor([1, 3]))
        self.assertEqual(cache.slot_to_expert, [1, 3])
        self.assertEqual(cache._expert_to_slot.get(3, -1), 1)
        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        self.assertFalse(cache.copy_rows(torch.tensor([3, 1]), {"host_rows": output}))
        self.assertTrue(torch.equal(output, layer.host_rows.data[[3, 1]]))
        cache.close()

    def test_gather_rows_keeps_the_legacy_counts(self):
        layer = _host_layer(experts=4)
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        output = torch.zeros(3, 3, 4, dtype=torch.uint8)
        first = cache.gather_rows(torch.tensor([1, 3, 1]), {"host_rows": output})
        self.assertEqual(first.hit_rows, 0)
        self.assertEqual(first.miss_rows, 3)
        self.assertEqual(first.populated_bytes, 24)
        self.assertTrue(torch.equal(output, layer.host_rows.data[[1, 3, 1]]))
        second = cache.gather_rows(torch.tensor([1, 2, 1]), {"host_rows": output})
        self.assertEqual((second.hit_rows, second.miss_rows), (2, 1))
        self.assertEqual(second.populated_bytes, 12)
        self.assertEqual(cache.stats.populated_rows, 3)
        self.assertEqual(cache.stats.evictions, 1)
        self.assertTrue(torch.equal(output, layer.host_rows.data[[1, 2, 1]]))

    def test_misses_beyond_the_capacity_are_copied_correctly(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        ids = torch.tensor([5, 0, 4, 2, 1])
        output = torch.zeros(5, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(ids, {"host_rows": output})
        self.assertEqual((result.hit_rows, result.miss_rows), (0, 5))
        self.assertTrue(torch.equal(output, layer.host_rows.data[ids]))
        self.assertEqual(cache.stats.evictions, 3)

    def test_is_pinned_rows_survive_admissions(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 2, device="cpu", is_pinned=lambda expert: expert == 1
        )
        cache.ensure_rows(torch.tensor([1, 2]))
        cache.ensure_rows(torch.tensor([3]))
        self.assertEqual(cache.slot_to_expert, [1, 3])

    def test_a_protected_resident_shrinks_the_chunks(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 2, device="cpu", is_pinned=lambda expert: expert == 1
        )
        cache.ensure_rows(torch.tensor([1]))
        self.assertEqual(cache.evictable_rows(), 1)
        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(torch.tensor([5, 7]), {"host_rows": output})
        self.assertEqual((result.hit_rows, result.miss_rows), (0, 2))
        self.assertTrue(torch.equal(output, layer.host_rows.data[[5, 7]]))
        self.assertEqual(cache._expert_to_slot.get(1), 0)
        self.assertEqual(cache.slot_to_expert, [1, 7])

    def test_promotion_chunks_of_evictable_rows_are_all_resident(self):
        # ExpertHotCache._load_reserved_in_chunks admits evictable_rows() experts per
        # chunk, and _prepare_promotion then needs every one of them in the tier.
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 3, device="cpu", is_pinned=lambda expert: expert == 1
        )
        cache.ensure_rows(torch.tensor([1]))
        chunk_rows = cache.evictable_rows()
        self.assertEqual(chunk_rows, 2)
        experts = [5, 7, 0, 3, 6]
        for start in range(0, len(experts), chunk_rows):
            chunk = experts[start : start + chunk_rows]
            cache.ensure_rows(torch.tensor(chunk))
            slots = [cache._expert_to_slot.get(expert, -1) for expert in chunk]
            self.assertTrue(all(slot >= 0 for slot in slots), (chunk, slots))
            self.assertTrue(
                torch.equal(
                    cache.tensors["host_rows"][slots], layer.host_rows.data[chunk]
                )
            )
        self.assertIn(1, cache._expert_to_slot)

    def test_every_slot_protected_leaves_the_tier_unchanged(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 2, device="cpu", is_pinned=lambda expert: expert in (1, 2)
        )
        cache.ensure_rows(torch.tensor([1, 2]))
        slots = list(cache.slot_to_expert)
        mapping = dict(cache._expert_to_slot)
        device_map = cache.expert_to_slot.clone()
        populated = cache.stats.populated_rows
        with self.assertRaisesRegex(RuntimeError, "protected"):
            cache.ensure_rows(torch.tensor([5]))
        self.assertEqual(cache.slot_to_expert, slots)
        self.assertEqual(dict(cache._expert_to_slot), mapping)
        self.assertTrue(torch.equal(cache.expert_to_slot, device_map))
        self.assertEqual(cache.stats.populated_rows, populated)
        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            cache.gather_rows(torch.tensor([5, 7]), {"host_rows": output})
        self.assertEqual(cache.slot_to_expert, slots)

    def test_a_failed_assignment_rolls_back_the_calls_slots(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        pinned = {1, 2}
        cache = ExpertPinnedHostCache(
            streamer, 3, device="cpu", is_pinned=lambda expert: expert in pinned
        )
        cache.ensure_rows(torch.tensor([1, 2]))
        pinned.add(5)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            cache.ensure_rows(torch.tensor([5, 7]))
        # 5 took the free slot before 7 found no victim; neither may stay mapped.
        self.assertEqual(cache.slot_to_expert, [1, 2, -1])
        self.assertNotIn(5, cache._expert_to_slot)
        self.assertEqual(int(cache.expert_to_slot[5]), -1)
        pinned.discard(5)
        output = torch.zeros(1, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(torch.tensor([5]), {"host_rows": output})
        self.assertEqual(result.miss_rows, 1)
        self.assertTrue(torch.equal(output, layer.host_rows.data[[5]]))

    def test_a_failed_read_rolls_back_the_calls_slots(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        cache.ensure_rows(torch.tensor([1]))
        with patch.object(streamer, "read_host_rows", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                cache.ensure_rows(torch.tensor([4, 6]))
        # 4 took the free slot and 6 evicted 1; both slots are unread, so both are freed.
        self.assertEqual(cache.slot_to_expert, [-1, -1])
        self.assertEqual(dict(cache._expert_to_slot), {})
        self.assertTrue(torch.equal(cache.expert_to_slot, torch.full((8,), -1)))
        self.assertEqual(cache.stats.evictions, 0)
        output = torch.zeros(2, 3, 4, dtype=torch.uint8)
        cache.gather_rows(torch.tensor([4, 1]), {"host_rows": output})
        self.assertTrue(torch.equal(output, layer.host_rows.data[[4, 1]]))


class TestGrowingProtection(unittest.TestCase):
    """``is_pinned`` whose protected set grows as the tier admits rows.

    An inclusive hierarchy pins the experts the hot cache reserves, so rows
    become protected in the middle of a call.
    """

    def _growing_tier(self, capacity, becomes_pinned):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        pinned = set()
        cache = ExpertPinnedHostCache(
            streamer, capacity, device="cpu", is_pinned=lambda expert: expert in pinned
        )
        read = streamer.read_host_rows

        def read_and_pin(rows_cpu, destinations, destination_rows=None):
            stats = read(rows_cpu, destinations, destination_rows)
            pinned.update(e for e in rows_cpu.tolist() if becomes_pinned(e))
            return stats

        streamer.read_host_rows = read_and_pin
        return layer, cache

    def _assert_tier_consistent(self, layer, cache):
        for slot, expert in enumerate(cache.slot_to_expert):
            if expert >= 0:
                self.assertEqual(cache._expert_to_slot[expert], slot)
                self.assertEqual(int(cache.expert_to_slot[expert]), slot)
                self.assertTrue(
                    torch.equal(cache.tensors["host_rows"][slot], layer.host_rows.data[expert])
                )
        self.assertEqual(
            int((cache.expert_to_slot >= 0).sum()),
            sum(expert >= 0 for expert in cache.slot_to_expert),
        )

    def test_chunks_shrink_as_admitted_rows_become_protected(self):
        # Capacity 3; expert 0 becomes protected once admitted. A chunk size fixed
        # at the call's start (3) would make the second chunk [3, 4, 5] evict one
        # of its own rows; per-chunk sizing gives [0, 1, 2], [3, 4], [5].
        layer, cache = self._growing_tier(3, lambda expert: expert == 0)
        ids = torch.arange(6)
        output = torch.zeros(6, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(ids, {"host_rows": output})
        self.assertEqual((result.hit_rows, result.miss_rows), (0, 6))
        self.assertTrue(torch.equal(output, layer.host_rows.data[ids]))
        self.assertIn(0, cache._expert_to_slot)
        self._assert_tier_consistent(layer, cache)

    def test_a_tier_filling_with_protected_rows_fails_cleanly(self):
        # Capacity 4; even experts become protected once admitted. The chunks are
        # [0..3], [4, 5], [6]; then no slot is evictable and 7 is refused.
        layer, cache = self._growing_tier(4, lambda expert: expert % 2 == 0)
        ids = torch.arange(8)
        output = torch.zeros(8, 3, 4, dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            cache.gather_rows(ids, {"host_rows": output})
        self.assertTrue(torch.equal(output[:7], layer.host_rows.data[:7]))
        self.assertEqual(sorted(cache._expert_to_slot), [0, 2, 4, 6])
        self._assert_tier_consistent(layer, cache)

    def test_an_all_hit_call_needs_no_evictable_slot(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(
            streamer, 2, device="cpu", is_pinned=lambda expert: expert in (1, 2)
        )
        cache.ensure_rows(torch.tensor([1, 2]))
        self.assertEqual(cache.evictable_rows(), 0)
        output = torch.zeros(3, 3, 4, dtype=torch.uint8)
        result = cache.gather_rows(torch.tensor([2, 1, 2]), {"host_rows": output})
        self.assertEqual((result.hit_rows, result.miss_rows), (3, 0))
        self.assertTrue(torch.equal(output, layer.host_rows.data[[2, 1, 2]]))


class TestPinnedTierOptions(unittest.TestCase):
    def test_the_manager_passes_the_format_options_to_each_tier(self):
        def pinned(expert):
            return expert == 3

        class OptionsFormat(DenseLayerFormat):
            def pinned_tier_options(self, layer):
                return {"device": "cpu", "is_pinned": pinned}

        model = torch.nn.Module()
        for layer_id in range(2):
            layer = _host_layer()
            layer.layer_id = layer_id
            layer._nvfp4_expert_streamer = ExpertStreamer(
                layer, ("host_rows",), format=OptionsFormat(("host_rows",))
            )
            model.add_module(str(layer_id), layer)
        manager = ExpertPinnedHostCacheManager.from_model(model, budget_bytes=4 * 12)
        self.assertEqual(sorted(manager.caches), [0, 1])
        for cache in manager.caches.values():
            self.assertEqual(cache.capacity, 2)
            self.assertEqual(cache.device, torch.device("cpu"))
            self.assertIs(cache.is_pinned, pinned)

    def test_the_dense_format_adds_no_options(self):
        options = DenseLayerFormat(("rows",)).pinned_tier_options(torch.nn.Module())
        self.assertEqual(dict(options), {})


class TestCachedGatherPinnedOverflow(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()

    def tearDown(self):
        expert_stream._STAGING.clear()

    def test_cached_gather_with_more_misses_than_pinned_rows(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)
        ExpertPinnedHostCache(streamer, 2, device="cpu")
        ids = torch.tensor([[0, 5], [7, 2], [3, 5]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_cached(source_ids, compact_ids, ids)
        self.assertTrue(
            torch.equal(tensors["host_rows"][compact.long()], layer.host_rows.data[ids])
        )
        stats = streamer.last_gather_stats
        self.assertEqual((stats.pinned_host_hit_rows, stats.pinned_host_miss_rows), (0, 5))
        self.assertEqual(stats.pinned_host_populated_bytes, 5 * 12)
        self.assertEqual(stats.source_bytes, 5 * 12)

    def test_a_pinned_only_gather_stages_one_set_of_rows(self):
        layer = _host_layer()
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        cache.ensure_rows(torch.tensor([5]))
        ids = torch.tensor([[0, 5], [7, 2], [3, 5]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_eager_rows(source_ids, compact_ids, ids)
        self.assertTrue(
            torch.equal(tensors["host_rows"][compact.long()], layer.host_rows.data[ids])
        )
        # One staging set: no hit, miss or cold buffers beside it.
        self.assertEqual({key[0] for key in expert_stream._STAGING}, {"host_rows"})
        self.assertEqual(tensors["host_rows"].shape[0], 64)
        stats = streamer.last_gather_stats
        self.assertEqual((stats.requested_rows, stats.miss_rows), (5, 5))
        # Chunks of 2 evict 5 before its own chunk reaches it, so it counts as a miss
        # (declined review finding M2).
        self.assertEqual((stats.pinned_host_hit_rows, stats.pinned_host_miss_rows), (0, 5))
        self.assertEqual(stats.source_bytes, 5 * 12)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Add the CUDA tests**

Insert above the `if __name__ == "__main__":` block of `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`:

```python
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPinnedTierCuda(unittest.TestCase):
    def test_pinned_slabs_are_registered_page_aligned_and_unrounded(self):
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache
        from sglang.srt.utils.cuda_host_registry import is_gpu_readable_host_tensor

        layer = torch.nn.Module()
        layer.host_rows = torch.nn.Parameter(
            torch.zeros(8, 3000, dtype=torch.uint8), requires_grad=False
        )
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 3)
        slab = cache.tensors["host_rows"]
        self.assertEqual(slab.data_ptr() % 4096, 0)
        self.assertTrue(is_gpu_readable_host_tensor(slab))
        self.assertEqual(slab.untyped_storage().nbytes(), 3 * 3000 + 4096)
        cache.close()
        self.assertFalse(is_gpu_readable_host_tensor(slab))

    def test_cached_gather_copies_misses_beyond_the_pinned_capacity(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        experts = 16
        layer = torch.nn.Module()
        layer.host_rows = torch.nn.Parameter(
            torch.randint(0, 256, (experts, 3, 4), dtype=torch.uint8),
            requires_grad=False,
        )
        layer.gpu_rows = torch.nn.Parameter(
            torch.rand(experts, 5, device="cuda"), requires_grad=False
        )
        layer._nvfp4_file_source_bytes_per_expert = 12
        streamer = ExpertStreamer(layer, ("host_rows", "gpu_rows"))
        ExpertHotCache(streamer, 1).reassign([3])
        pinned = ExpertPinnedHostCache(streamer, 2)
        ids = torch.tensor([[3, 0, 5, 7], [9, 11, 3, 13]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        cpu_ids = ids.long().cpu()
        self.assertTrue(
            torch.equal(tensors["host_rows"][compact.long()].cpu(), layer.host_rows.data[cpu_ids])
        )
        self.assertTrue(
            torch.equal(
                tensors["gpu_rows"][compact.long()].cpu(),
                layer.gpu_rows.data[cpu_ids.cuda()].cpu(),
            )
        )
        stats = streamer.last_gather_stats
        self.assertEqual((stats.hot_hit_rows, stats.pinned_host_miss_rows), (1, 6))
        self.assertEqual(pinned.stats.evictions, 4)
```

- [ ] **Step 3: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add test/registered/unit/layers/moe/test_expert_host_tier.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py
git commit -m "test(moe): pin the pinned tier's slot choice and its overflow copies

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/layers/moe/test_expert_host_tier.py'
```
Expected: a collection error, `ModuleNotFoundError: No module named 'sglang.srt.layers.moe.expert_host_tier'`. This is the red for every test in the file. Two of them pin behaviour that already exists: `test_random_traffic_matches_the_legacy_slot_choice` and `test_gather_rows_keeps_the_legacy_counts`. For those, the mutation check is to make `PinnedSlotLRU._victim` return the newest expert instead of the oldest; both tests then fail.

- [ ] **Step 4: Create `expert_host_tier.py`**

Create `python/sglang/srt/layers/moe/expert_host_tier.py`:

```python
"""CPU-side pieces of the pinned host expert tier: the slot LRU and exact slabs.

They are kept apart from the streamer, and import the CUDA host-registration
helpers only when asked to register, so the slot policy and the slab layout
are testable on a CPU-only host.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from typing import Callable, Collection, NamedTuple, Optional, Sequence

import torch

PAGE_BYTES = 4096


class PinnedGatherResult(NamedTuple):
    """Row counters of one pinned-tier gather."""

    hit_rows: int
    miss_rows: int
    populated_bytes: int
    fallback_used: bool


class PinnedSlotLRU:
    """Slot bookkeeping of a bounded host row cache.

    Free slots are handed out lowest index first. A full cache evicts the
    least recently used expert that neither ``is_pinned`` nor the caller's
    ``protected`` set covers. When only ``protected`` experts can go, the
    oldest of them does: that is an over-capacity call reassigning its own
    slots. ``touch`` and ``assign`` are O(1); only covered experts are
    skipped when choosing a victim.
    """

    def __init__(
        self, capacity: int, is_pinned: Optional[Callable[[int], bool]] = None
    ):
        self.capacity = int(capacity)
        self.is_pinned = is_pinned
        self.slot_to_expert = [-1] * self.capacity
        # Oldest first: iteration order is the eviction order.
        self.expert_to_slot: "OrderedDict[int, int]" = OrderedDict()
        self._free = list(range(self.capacity))
        heapq.heapify(self._free)

    def __contains__(self, expert_id: int) -> bool:
        return expert_id in self.expert_to_slot

    def touch(self, expert_id: int) -> None:
        self.expert_to_slot.move_to_end(expert_id)

    def assign(
        self, expert_id: int, protected: Collection[int] = frozenset()
    ) -> tuple[int, Optional[int]]:
        """Give ``expert_id`` a slot; returns the slot and the evicted expert or None.

        ``protected`` holds the experts of the caller's current request, which
        are evicted only when nothing else can be.
        """
        if expert_id in self.expert_to_slot:
            raise ValueError(f"expert {expert_id} already holds a pinned slot")
        evicted = None
        if self._free:
            slot = heapq.heappop(self._free)
        else:
            evicted = self._victim(protected)
            slot = self.expert_to_slot.pop(evicted)
        self.slot_to_expert[slot] = expert_id
        self.expert_to_slot[expert_id] = slot
        return slot, evicted

    def _victim(self, protected: Collection[int]) -> int:
        fallback = None
        for expert_id in self.expert_to_slot:
            if self.is_pinned is not None and self.is_pinned(expert_id):
                continue
            if expert_id not in protected:
                return expert_id
            if fallback is None:
                fallback = expert_id
        if fallback is not None:
            return fallback
        raise RuntimeError("every pinned host slot holds a protected expert")

    def release(self, slot: int) -> None:
        """Free ``slot``, forgetting its expert (used to roll back a failed read)."""
        expert_id = self.slot_to_expert[slot]
        if expert_id < 0:
            return
        self.expert_to_slot.pop(expert_id, None)
        self.slot_to_expert[slot] = -1
        heapq.heappush(self._free, slot)

    def mapping(self, num_experts: int) -> list[int]:
        """Each expert's slot, or -1."""
        mapping = [-1] * num_experts
        for slot, expert_id in enumerate(self.slot_to_expert):
            if expert_id >= 0:
                mapping[expert_id] = slot
        return mapping


def allocate_host_slab(
    rows: int, row_shape: tuple[int, ...], dtype: torch.dtype, *, register: bool
) -> torch.Tensor:
    """A page-aligned ``[rows, *row_shape]`` host tensor of exactly its size.

    PyTorch's pinned allocator rounds every allocation up to a power of two,
    so a slab just past 1 GiB would pin 2 GiB. This one takes plain host
    memory plus one page of alignment slack, and ``register`` pins it with
    ``cudaHostRegister`` in row-aligned chunks, as the host arena does.
    Page alignment also lets io_uring fill it with ``O_DIRECT``.
    """
    shape = (int(rows),) + tuple(int(dimension) for dimension in row_shape)
    nbytes = math.prod(shape) * dtype.itemsize
    storage = torch.empty(nbytes + PAGE_BYTES, dtype=torch.uint8, device="cpu")
    start = (-storage.data_ptr()) % PAGE_BYTES
    slab = storage[start : start + nbytes].view(dtype).view(shape)
    if register and nbytes:
        # Imported here: expert_stream imports this module while the model
        # loader package is still importing, and mem_cache.pool_host's package
        # import is heavy.
        from sglang.srt.mem_cache.pool_host.common import _cuda_host_register

        _cuda_host_register(slab, registration_granularity_bytes=nbytes // shape[0])
    return slab


def release_host_slabs(slabs: Sequence[torch.Tensor]) -> None:
    """Unregister slabs that ``allocate_host_slab(..., register=True)`` registered."""
    if not slabs:
        return
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

    for slab in slabs:
        _cuda_host_unregister(slab)
```

- [ ] **Step 5: Edit `expert_stream.py` — imports**

Old:
```python
import logging
import json
import os
```
New:
```python
import logging
import json
import os
import weakref
```
Old:
```python
from sglang.srt.layers.moe.expert_row_source import (
    ExpertRowSource,
    RowReadStats,
    TensorRowSource,
)
```
New:
```python
from sglang.srt.layers.moe.expert_host_tier import (
    PinnedGatherResult,
    PinnedSlotLRU,
    allocate_host_slab,
    release_host_slabs,
)
from sglang.srt.layers.moe.expert_row_source import (
    ExpertRowSource,
    RowReadStats,
    TensorRowSource,
)
```

- [ ] **Step 5b: Add the `pinned_tier_options` format hook and pass it through the manager**

In `python/sglang/srt/layers/moe/expert_format.py`, old:
```python
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
)
```
New:
```python
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)
```
Old (the protocol's last method):
```python
    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional["ExpertRowSource"]
    ) -> Optional[int]: ...
```
New:
```python
    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional["ExpertRowSource"]
    ) -> Optional[int]: ...

    def pinned_tier_options(self, layer: torch.nn.Module) -> Mapping[str, Any]:
        """Keyword arguments for this layer's ``ExpertPinnedHostCache``, e.g. ``is_pinned``."""
        ...
```
Old (the end of `DenseLayerFormat.file_source_bytes_per_expert`):
```python
        return getattr(layer, FILE_SOURCE_BYTES_ATTRIBUTE, None)
```
New:
```python
        return getattr(layer, FILE_SOURCE_BYTES_ATTRIBUTE, None)

    def pinned_tier_options(self, layer: torch.nn.Module) -> Mapping[str, Any]:
        # The dense format builds the pinned tier exactly as before formats existed.
        return {}
```

In `python/sglang/srt/layers/moe/expert_stream.py`, `ExpertPinnedHostCacheManager.from_model`, old:
```python
        manager.caches = {
            layer_id: ExpertPinnedHostCache(streamers[layer_id], capacity)
            for layer_id, capacity in capacities.items()
            if capacity
        }
```
New:
```python
        # The format supplies tier options such as an is_pinned filter; the dense
        # format supplies none, so NVFP4 tiers are built exactly as before.
        manager.caches = {
            layer_id: ExpertPinnedHostCache(
                streamers[layer_id],
                capacity,
                **streamers[layer_id].format.pinned_tier_options(
                    streamers[layer_id].layer
                ),
            )
            for layer_id, capacity in capacities.items()
            if capacity
        }
```

- [ ] **Step 6: Replace the `ExpertPinnedHostCache` class**

Replace the whole class, from `class ExpertPinnedHostCache:` down to the line before `class ExpertPinnedHostCacheManager:`, with:

```python
class ExpertPinnedHostCache:
    """Bounded, on-demand pinned host rows shared by one expert layer.

    Each host tensor gets one page-aligned slab registered with CUDA and sized
    to exactly ``capacity`` rows; PyTorch's pinned allocator would round each
    slab up to a power of two. ``device`` holds the slot lookup. It defaults
    to the layer's CUDA source device, else the current CUDA device; a CPU
    ``device`` keeps the tier on the host with unregistered slabs, so it runs
    without a GPU. ``is_pinned(expert_id)`` protects experts from eviction.
    """

    def __init__(
        self,
        streamer: "ExpertStreamer",
        capacity: int,
        *,
        device: torch.device | str | None = None,
        is_pinned=None,
    ):
        capacity = index(capacity)
        if not 0 <= capacity <= streamer.num_experts:
            raise ValueError("pinned host cache capacity must be within expert count")
        self.streamer = streamer
        self.capacity = capacity
        self.cached_names = tuple(
            spec.name for spec in streamer.specs if spec.residence == "host"
        )
        if capacity and not self.cached_names:
            raise ValueError(
                "pinned host cache requires at least one CPU source tensor"
            )
        self.bytes_per_expert = streamer.host_bytes_per_expert
        self.residency_bytes = capacity * self.bytes_per_expert
        if device is None:
            devices = {
                streamer.source(spec.name).device
                for spec in streamer.specs
                if spec.residence == "device"
            }
            if len(devices) > 1:
                raise ValueError("pinned host cache CUDA sources must share one device")
            device = next(
                iter(devices), torch.device("cuda", torch.cuda.current_device())
            )
        self.device = torch.device(device)
        register = self.device.type == "cuda"
        self.tensors: dict[str, torch.Tensor] = {}
        registered: list[torch.Tensor] = []
        try:
            for name in self.cached_names:
                spec = streamer.spec(name)
                slab = allocate_host_slab(
                    capacity, spec.row_shape, spec.dtype, register=register
                )
                self.tensors[name] = slab
                if register and slab.numel():
                    registered.append(slab)
        except BaseException:
            release_host_slabs(registered)
            raise
        # Unregisters the slabs when the cache is collected, at exit, or on close().
        self._release_slabs = weakref.finalize(self, release_host_slabs, registered)
        row_source = streamer.row_source
        if row_source is not None:
            row_source.register_destinations(self.tensors.values())
        self.expert_to_slot = torch.full(
            (streamer.num_experts,), -1, dtype=torch.long, device=self.device
        )
        self.is_pinned = is_pinned
        self._lru = PinnedSlotLRU(capacity, is_pinned=is_pinned)
        self.stats = PinnedHostCacheStats()
        streamer.pinned_host_cache = self

    @staticmethod
    def capacity_for_budget(streamer: "ExpertStreamer", budget_bytes: int) -> int:
        """Round a pinned-host byte budget down to complete expert rows."""
        budget_bytes = index(budget_bytes)
        if budget_bytes < 0:
            raise ValueError("pinned host cache byte budget cannot be negative")
        if streamer.host_bytes_per_expert == 0:
            return 0
        return min(streamer.num_experts, budget_bytes // streamer.host_bytes_per_expert)

    @property
    def slot_to_expert(self) -> list[int]:
        return self._lru.slot_to_expert

    @property
    def _expert_to_slot(self) -> dict[int, int]:
        # ExpertHotCache._prepare_promotion reads resident slots through this.
        return self._lru.expert_to_slot

    def close(self) -> None:
        """Unregister the slabs; the cache must not be used afterwards."""
        self._release_slabs()

    def evictable_rows(self) -> int:
        """Slots a request can use: the capacity minus residents ``is_pinned`` protects."""
        if self.is_pinned is None:
            return self.capacity
        return self.capacity - sum(
            1 for expert_id in self._lru.expert_to_slot if self.is_pinned(expert_id)
        )

    def _refresh_mapping(self) -> None:
        self.expert_to_slot.copy_(
            torch.tensor(
                self._lru.mapping(self.streamer.num_experts),
                dtype=torch.long,
                device=self.device,
            )
        )

    def lookup(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return slots and record row-level hit/miss counters."""
        if source_ids.device != self.device:
            raise ValueError(
                "selected expert IDs must use the pinned cache CUDA device"
            )
        slots = self.expert_to_slot[source_ids.long()]
        hit_mask = slots >= 0
        hit_ids = source_ids[hit_mask].tolist()
        for expert_id in hit_ids:
            self._lru.touch(int(expert_id))
        self.stats.lookup_hits += len(hit_ids)
        self.stats.lookup_misses += source_ids.numel() - len(hit_ids)
        return slots, hit_mask

    def ensure_rows(
        self, source_ids: torch.Tensor, protected: Iterable[int] = ()
    ) -> None:
        """Read missing source rows into pinned slots, evicting least-recently-used rows.

        The requested experts, and any others in ``protected`` (a caller's
        whole chunk), are evicted only when nothing else can be.
        """
        if not self.cached_names or self.capacity == 0 or source_ids.numel() == 0:
            return
        requested = list(dict.fromkeys(int(value) for value in source_ids.tolist()))
        missing = [expert_id for expert_id in requested if expert_id not in self._lru]
        if not missing:
            return
        protected = frozenset(requested).union(int(value) for value in protected)
        assignments = []
        evictions = 0
        # Assignment and read are one transaction: on any failure every slot this
        # call assigned is freed, so no expert stays mapped to a slot never read.
        try:
            for expert_id in missing:
                slot, evicted = self._lru.assign(expert_id, protected)
                evictions += evicted is not None
                assignments.append((expert_id, slot))
            # More misses than slots reassign a slot within this call. Read only the
            # slot's final expert: batched file reads complete in any order.
            final_slots = {slot: expert_id for expert_id, slot in assignments}
            source_ids_cpu = torch.tensor(list(final_slots.values()), dtype=torch.long)
            slots_cpu = torch.tensor(list(final_slots), dtype=torch.long)
            self.streamer.read_host_rows(
                source_ids_cpu,
                {name: self.tensors[name] for name in self.cached_names},
                slots_cpu,
            )
        except BaseException:
            for slot in dict.fromkeys(slot for _, slot in assignments):
                self._lru.release(slot)
            self._refresh_mapping()
            raise
        self._refresh_mapping()
        self.stats.evictions += evictions
        self.stats.populated_rows += len(final_slots)
        self.stats.populated_bytes += len(final_slots) * self.bytes_per_expert

    def copy_rows(
        self, source_ids: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> bool:
        """Gather resident pinned rows directly into CUDA (or, on a CPU tier, CPU) outputs."""
        if source_ids.numel() == 0:
            return False
        slots = self.expert_to_slot[source_ids.long()]
        fallback_used = False
        for name in self.cached_names:
            source = self.tensors[name]
            output = outputs[name]
            if output.device.type == "cpu":
                torch.index_select(source, 0, slots.cpu(), out=output)
            elif source.is_contiguous() and output.is_contiguous():
                row_bytes = source.numel() * source.element_size() // self.capacity
                _gather_host_rows_kernel[
                    (source_ids.numel(), triton.cdiv(row_bytes, 1024))
                ](
                    source.view(torch.uint8),
                    slots,
                    output.view(torch.uint8),
                    row_bytes,
                    BLOCK=1024,
                )
            else:
                fallback_used = True
                slots_cpu = _copy_indices_to_cpu(slots, source_ids.numel())
                host_output = _pinned_staging_buffer(
                    ("pinned_host_cache_fallback", name),
                    source_ids.numel(),
                    max(source_ids.numel(), _NO_DEDUP_LIMIT),
                    tuple(source.shape[1:]),
                    source.dtype,
                )
                torch.index_select(source, 0, slots_cpu, out=host_output)
                output.copy_(host_output, non_blocking=True)
        return fallback_used

    def gather_rows(
        self, source_ids: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> PinnedGatherResult:
        """Copy the rows of ``source_ids`` into the leading rows of ``outputs``, admitting misses.

        Rows are admitted and copied chunk by chunk. Each chunk is sized when
        it starts, from ``evictable_rows()``: residents that ``is_pinned``
        protects never make room, and an admitted row may itself become
        protected (an inclusive hierarchy pins rows the hot cache reserves), so
        the room shrinks during a call. A chunk holds at most that many
        distinct experts, all of them protected while it is admitted, and its
        rows are checked resident before its copy, so no copy reads slot -1.
        A call that fits in one chunk is the pre-chunking sequence ``lookup``,
        ``ensure_rows``, ``copy_rows``, plus that host-side check. With no
        evictable slot, a chunk with misses raises; an all-hit chunk is still
        copied. Each chunk's hit count (``.item()``) syncs the stream on the
        host, so the previous chunk's copy has run before its slots can be
        refilled.
        """
        if self.capacity == 0:
            raise ValueError("pinned host cache has no rows")
        hit_rows = 0
        miss_rows = 0
        populated_before = self.stats.populated_bytes
        fallback_used = False
        total = source_ids.numel()
        start = 0
        while start < total:
            evictable = self.evictable_rows()
            chunk_rows = total - start
            if (
                evictable >= 1
                and chunk_rows > evictable
                and torch.unique(source_ids[start:]).numel() > evictable
            ):
                chunk_rows = evictable
            chunk = source_ids[start : start + chunk_rows]
            _, hit_mask = self.lookup(chunk)
            chunk_hits = int(hit_mask.sum().item())
            hit_rows += chunk_hits
            miss_rows += chunk.numel() - chunk_hits
            chunk_ids = [int(value) for value in chunk.tolist()]
            if chunk_hits < chunk.numel():
                if evictable < 1:
                    raise RuntimeError("every pinned host slot holds a protected expert")
                self.ensure_rows(chunk[~hit_mask], protected=chunk_ids)
            lost = sorted({expert_id for expert_id in chunk_ids if expert_id not in self._lru})
            if lost:
                raise RuntimeError(
                    f"pinned host rows of experts {lost} were evicted before their copy"
                )
            chunk_outputs = {
                name: output[start : start + chunk_rows]
                for name, output in outputs.items()
            }
            fallback_used = self.copy_rows(chunk, chunk_outputs) or fallback_used
            start += chunk_rows
        return PinnedGatherResult(
            hit_rows,
            miss_rows,
            self.stats.populated_bytes - populated_before,
            fallback_used,
        )
```

- [ ] **Step 7: Use `gather_rows` in `_gather_cached`**

Old:
```python
            and self.file_source_bytes_per_expert is not None
        ):
            _, pinned_hit_mask = pinned_cache.lookup(miss_source_ids)
            pinned_hit_rows = int(pinned_hit_mask.sum().item())
            pinned_miss_rows = miss_rows - pinned_hit_rows
            populated_before = pinned_cache.stats.populated_bytes
            if pinned_miss_rows:
                pinned_cache.ensure_rows(miss_source_ids[~pinned_hit_mask])
            pinned_populated_bytes = (
                pinned_cache.stats.populated_bytes - populated_before
            )
            source_bytes = pinned_miss_rows * self.host_bytes_per_expert + miss_rows * (
                self.bytes_per_expert - self.host_bytes_per_expert
            )
            pinned_outputs = {
                name: output
                for name, output in misses.items()
                if name in pinned_cache.cached_names
            }
            gather_fallback_used = pinned_cache.copy_rows(
                miss_source_ids, pinned_outputs
            )
            uncached_outputs = {
```
New:
```python
            and self.file_source_bytes_per_expert is not None
        ):
            pinned_outputs = {
                name: output
                for name, output in misses.items()
                if name in pinned_cache.cached_names
            }
            # Chunked so misses beyond the pinned capacity are never copied from slot -1.
            pinned = pinned_cache.gather_rows(miss_source_ids, pinned_outputs)
            pinned_hit_rows = pinned.hit_rows
            pinned_miss_rows = pinned.miss_rows
            pinned_populated_bytes = pinned.populated_bytes
            gather_fallback_used = pinned.fallback_used
            source_bytes = pinned_miss_rows * self.host_bytes_per_expert + miss_rows * (
                self.bytes_per_expert - self.host_bytes_per_expert
            )
            uncached_outputs = {
```

- [ ] **Step 7b: Replace `_gather_pinned_host` with one staging set filled by `gather_rows`**

Replace the whole `_gather_pinned_host` method (from `    def _gather_pinned_host(` down to the line before `    def _plan_eager_routes(`) with the code below.
- Its counts are unchanged whenever the call fits the tier's evictable slots.
- An over-capacity call used to read its cold rows through `_copy_source_rows`. It now admits them chunk by chunk.
- The three extra staging sets are gone.

```python
    def _gather_pinned_host(
        self,
        source_ids: torch.Tensor,
        compact_ids: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gather routed rows through the pinned host cache.

        Returned tensors follow the leading-dimension rule of ``_gather_cached``.
        Pinned rows are copied straight into the one staging set of
        ``max(rows, NO_DEDUP_LIMIT)`` rows by ``gather_rows``, which admits
        misses in chunks the tier can hold, so a format's ``max_gather_rows``
        bounds this path's VRAM as it bounds ``_gather_cached``'s.
        """
        cache = self.pinned_host_cache
        assert cache is not None
        row_count = source_ids.numel()
        capacity = max(row_count, _NO_DEDUP_LIMIT)
        kernel_rows = capacity if should_dedup(topk_ids) else row_count
        padded = {
            name: _staging_buffer(
                name,
                kernel_rows,
                capacity,
                self.spec(name).row_shape,
                self.spec(name).dtype,
                topk_ids.device,
            )
            for name in self.tensor_names
        }
        gathered = {name: buffer[:row_count] for name, buffer in padded.items()}
        pinned = cache.gather_rows(
            source_ids,
            {
                name: output
                for name, output in gathered.items()
                if name in cache.cached_names
            },
        )
        copy_engine_bytes = 0
        uncached = {
            name: output
            for name, output in gathered.items()
            if name not in cache.cached_names
        }
        if uncached:
            copy_engine_bytes = self._copy_source_rows(source_ids, uncached)
        self.last_gather_stats = ExpertGatherStats(
            row_count,
            0,
            row_count,
            0,
            row_count * self.host_bytes_per_expert,
            pinned.miss_rows * self.bytes_per_expert,
            pinned.hit_rows,
            pinned.miss_rows,
            pinned.populated_bytes,
            copy_engine_bytes=copy_engine_bytes,
            routed_rows=compact_ids.numel(),
            routed_miss_rows=compact_ids.numel(),
            unique_miss_rows=row_count,
        )
        return compact_ids.reshape(topk_ids.shape).to(topk_ids.dtype), padded
```

- [ ] **Step 8: Commit, push and run the new tests plus `CPU_SUITE`**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/layers/moe/expert_host_tier.py python/sglang/srt/layers/moe/expert_format.py python/sglang/srt/layers/moe/expert_stream.py
git commit -m "fix(moe): copy pinned-tier misses beyond capacity in resident chunks

_gather_cached admitted every miss before copying, so a call with more
distinct misses than pinned rows copied evicted rows from slot -1. The
tier now gathers in chunks of its evictable slots and protects a call's
own experts from eviction, rolls back a failed admission, allocates exact
page-aligned registered slabs instead of power-of-two pinned blocks, keeps
slots in an O(1) LRU with an optional is_pinned filter that formats supply
through pinned_tier_options, and accepts a CPU device. The pinned-only
gather stages one set of rows instead of four.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/layers/moe/test_expert_host_tier.py test/registered/unit/layers/moe/test_expert_row_source.py test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py'
```
Expected: all 23 tests in `test_expert_host_tier.py` pass. The only failure is the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`.

---

### Task 6: Spec-only formats — promotions through the pinned tier, arena refusal, graph-flag guard

A format whose `source(layer, name)` is None has no dense tensor. Task 3 already routes every host read of such a tensor through the row source. This task completes spec-only support.

**Hot cache.** A promotion of spec-only rows cannot use the async six-pair copy from dense sources. `stage_reassign` therefore sends spec-only layers to the synchronous `_load_reserved`. When the layer has six tensors and a pinned tier, `_load_reserved` promotes in chunks. Each chunk is sized when it starts, from `pinned_cache.evictable_rows()`, so the tier can hold every row of the chunk even as earlier chunks' rows become protected (Task 5). Each chunk:
1. admits its rows to the pinned tier;
2. copies them from the pinned slabs with the existing six-pair routes;
3. waits on the host before the next chunk can evict them.

If a chunk fails, the later tickets are cancelled, and its promotion is aborted if it is still in flight. Once copies were submitted, the slots are freed only after `torch.cuda.synchronize` has drained them. If the device cannot drain, the promotion stays in flight and the cache refuses further updates rather than reuse a slot a copy may still write.

Without a pinned tier, or with a tensor count other than six, it copies one row at a time through `_copy_source_rows`, which reads through the row source.

**Guards.** `_prepare_promotion` raises instead of passing a None source to the copy routes. `ExpertHostArena.from_model` refuses formats with `supports_host_arena=False` and spec-only streamers. `enable_graph_gather` refuses formats with `supports_graph_gather=False` and spec-only streamers. `ExpertHotCacheManager.from_model` refuses the same whenever a graph flag is set: graph gather, GPU residency update, or doorbell. It refuses before allocating anything.

Every new branch keys on `has_spec_only_tensors` or on a format flag that NVFP4 sets True, so NVFP4 paths are unchanged.

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_format.py`, adding `require_graph_gather_support`.
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`, adding `ExpertStreamer.has_spec_only_tensors` and a guard at the top of `enable_graph_gather`.
- Modify: `python/sglang/srt/layers/moe/expert_host_arena.py` (`from_model`).
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py`, in `_load_reserved`, a new `_load_reserved_in_chunks`, `_prepare_promotion`, `stage_reassign`, and `ExpertHotCacheManager.from_model` (the guard).
- Modify: `python/sglang/test/moe_expert_fakes.py`, adding `SpecOnlyFormat`.
- Test (modify): `test/registered/unit/layers/moe/test_expert_format.py`, `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`.

**Interfaces:**
- Consumes:
  - Task 1: `ExpertFormat` attributes, `iter_expert_streamers`, `ExpertTensorSpec`.
  - Task 3: `read_host_rows`, and `_copy_source_rows` treating None sources as pageable.
  - Task 5: `ExpertPinnedHostCache(streamer, capacity, device=..., is_pinned=...)`, `.gather_rows`, `.evictable_rows()`, `._expert_to_slot`, `.capacity`, and `ExpertFormat.pinned_tier_options`, which `SpecOnlyFormat` implements.
- Produces:
  - `sglang.srt.layers.moe.expert_format.require_graph_gather_support(streamers: Iterable) -> None`. It raises `ValueError` containing "does not support graph gather".
  - `ExpertStreamer.has_spec_only_tensors -> bool` (a property).
  - `ExpertHostArena.from_model` raises `ValueError` containing "does not support the host arena".
  - `ExpertHotCache._load_reserved_in_chunks(tickets, pinned_cache) -> None`. It sizes each chunk from `pinned_cache.evictable_rows()` when that chunk starts. When that is below 1, it cancels the remaining tickets and raises `RuntimeError("... no evictable slots ...")`.
  - `ExpertHotCache._drain_device() -> bool`. It calls `torch.cuda.synchronize(self.device)` and returns False if that raises.
  - `sglang.test.moe_expert_fakes.SpecOnlyFormat(reference: Mapping[str, Tensor])`:
    - `key = "spec_only_test"`, `supports_graph_gather = False`, `supports_host_arena = False`, `max_gather_rows = None`;
    - `source` is always None;
    - `default_row_source` returns `CountingRowSource(reference)` for kinds `auto` and `files`, and raises for any other kind;
    - `file_source_bytes_per_expert` returns `row_source.file_bytes_per_expert`;
    - `pinned_tier_options` returns `{}`.

- [ ] **Step 1: Add `SpecOnlyFormat` to the fakes**

Append to `python/sglang/test/moe_expert_fakes.py`. Also add `ExpertTensorSpec` to its imports, as `from sglang.srt.layers.moe.expert_format import ExpertTensorSpec`.

```python
class SpecOnlyFormat:
    """A format with no dense sources: every host row comes from its row source.

    ``reference`` holds the true ``[experts, ...]`` rows for the row source to
    serve and for tests to compare against; nothing is set on the layer.
    """

    key = "spec_only_test"
    supports_graph_gather = False
    supports_host_arena = False
    max_gather_rows: Optional[int] = None

    def __init__(self, reference: Mapping[str, torch.Tensor]):
        self.reference = dict(reference)

    def tensor_specs(self, layer):
        return tuple(
            ExpertTensorSpec(name, tuple(tensor.shape[1:]), tensor.dtype, "host")
            for name, tensor in self.reference.items()
        )

    def num_experts(self, layer) -> int:
        return next(iter(self.reference.values())).shape[0]

    def source(self, layer, name):
        return None

    def default_row_source(self, layer, specs, kind):
        if kind in ("auto", "files"):
            return CountingRowSource(self.reference)
        raise ValueError(f"expert format {self.key!r} has no row source kind {kind!r}")

    def file_source_bytes_per_expert(self, layer, row_source):
        return None if row_source is None else row_source.file_bytes_per_expert

    def pinned_tier_options(self, layer):
        return {}
```

- [ ] **Step 2: Add the failing CPU tests**

Add to the import block of `test/registered/unit/layers/moe/test_expert_format.py`:

```python
from unittest.mock import patch

from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache, HotCacheSlotTicket
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache
from sglang.test.moe_expert_fakes import CountingRowSource, SpecOnlyFormat
```

Insert above the `if __name__ == "__main__":` block:

```python
def _spec_only_reference(experts=8):
    generator = torch.Generator().manual_seed(9)
    return {
        "w13_trellis": torch.randint(
            -(2**15), 2**15, (experts, 2, 6), dtype=torch.int16, generator=generator
        ),
        "w13_suh": torch.randn(experts, 2, 4, generator=generator).half(),
        "w2_trellis": torch.randint(
            -(2**15), 2**15, (experts, 1, 6), dtype=torch.int16, generator=generator
        ),
    }


class TestSpecOnlyFormat(unittest.TestCase):
    def _streamer(self, row_source=expert_stream._DEFAULT_ROW_SOURCE):
        reference = _spec_only_reference()
        layer = torch.nn.Module()
        layer.layer_id = 0
        streamer = ExpertStreamer(
            layer,
            tuple(reference),
            format=SpecOnlyFormat(reference),
            row_source=row_source,
        )
        return reference, layer, streamer

    def test_specs_describe_the_rows_and_the_row_source_reads_them(self):
        reference, _, streamer = self._streamer()
        row_bytes = sum(t[0].numel() * t.element_size() for t in reference.values())
        self.assertTrue(streamer.has_spec_only_tensors)
        self.assertIsNone(streamer.source("w13_trellis"))
        self.assertIsInstance(streamer.row_source, CountingRowSource)
        self.assertEqual(streamer.num_experts, 8)
        self.assertEqual(streamer.bytes_per_expert, row_bytes)
        self.assertEqual(streamer.host_bytes_per_expert, row_bytes)
        self.assertEqual(streamer.file_source_bytes_per_expert, row_bytes)
        destinations = {
            name: torch.zeros((2,) + tuple(t.shape[1:]), dtype=t.dtype)
            for name, t in reference.items()
        }
        streamer.read_host_rows(torch.tensor([6, 1]), destinations)
        self.assertEqual(len(streamer.row_source.calls), 1)
        for name, tensor in reference.items():
            self.assertTrue(torch.equal(destinations[name], tensor[[6, 1]]), name)

    def test_dense_formats_have_no_spec_only_tensors(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        self.assertFalse(ExpertStreamer(layer, ("rows",)).has_spec_only_tensors)

    def test_the_tensor_kind_is_refused(self):
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("tensor"):
            with self.assertRaisesRegex(ValueError, "no row source kind 'tensor'"):
                self._streamer()

    def test_without_a_row_source_host_reads_are_refused(self):
        _, _, streamer = self._streamer(row_source=None)
        with self.assertRaisesRegex(ValueError, "no row source covers"):
            streamer.read_host_rows(
                torch.tensor([0]), {"w13_trellis": torch.zeros(1, 2, 6, dtype=torch.int16)}
            )

    def test_the_host_arena_refuses_the_format(self):
        from sglang.srt.layers.moe.expert_host_arena import ExpertHostArena

        _, layer, streamer = self._streamer()
        layer._nvfp4_expert_streamer = streamer
        with self.assertRaisesRegex(ValueError, "does not support the host arena"):
            ExpertHostArena.from_model(torch.nn.Sequential(layer))

    def test_graph_gather_is_refused_before_any_cuda_work(self):
        _, _, streamer = self._streamer()
        with self.assertRaisesRegex(ValueError, "does not support graph gather"):
            streamer.enable_graph_gather(4)

    def test_the_hot_cache_manager_refuses_every_graph_flag(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

        _, layer, streamer = self._streamer()
        layer._nvfp4_expert_streamer = streamer
        model = torch.nn.Sequential(layer)
        common = dict(
            budget_bytes=1 << 20,
            seed_path=None,
            dynamic=False,
            update_prefill_tokens=16,
            min_residence_forwards=0,
            benefit_ratio=1.0,
        )
        for flags in (
            dict(graph_gather_batch_size=1),
            dict(gpu_residency_update=True),
            dict(expert_doorbell=True),
        ):
            with self.subTest(**flags):
                with self.assertRaisesRegex(ValueError, "does not support graph gather"):
                    ExpertHotCacheManager.from_model(model, **common, **flags)

    def test_a_cpu_pinned_tier_admits_rows_through_the_row_source(self):
        reference, _, streamer = self._streamer()
        cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
        self.assertEqual(cache.cached_names, tuple(reference))
        outputs = {
            name: torch.zeros((5,) + tuple(t.shape[1:]), dtype=t.dtype)
            for name, t in reference.items()
        }
        ids = torch.tensor([7, 0, 3, 5, 2])
        result = cache.gather_rows(ids, outputs)
        self.assertEqual((result.hit_rows, result.miss_rows), (0, 5))
        for name, tensor in reference.items():
            self.assertTrue(torch.equal(outputs[name], tensor[ids]), name)
        self.assertTrue(
            all(call.names == tuple(reference) for call in streamer.row_source.calls)
        )

    def test_cached_gather_serves_spec_only_misses_from_the_pinned_tier(self):
        reference, _, streamer = self._streamer()
        streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)
        ExpertPinnedHostCache(streamer, 2, device="cpu")
        expert_stream._STAGING.clear()
        ids = torch.tensor([[4, 1], [6, 4]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        compact, tensors = streamer._gather_eager_rows(source_ids, compact_ids, ids)
        expert_stream._STAGING.clear()
        for name, tensor in reference.items():
            self.assertTrue(torch.equal(tensors[name][compact.long()], tensor[ids]), name)
        self.assertEqual(streamer.last_gather_stats.pinned_host_miss_rows, 3)
        self.assertEqual(streamer.last_gather_stats.host_read_rows, 3)


_SIX_NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


class _FixedRoom:
    """A pinned tier stand-in with a fixed number of evictable rows."""

    def __init__(self, rows):
        self.rows = rows

    def evictable_rows(self):
        return self.rows


class TestSpecOnlyPromotion(unittest.TestCase):
    """CPU checks of the hot cache's spec-only promotion control flow.

    ExpertHotCache needs CUDA to construct, so these tests build a bare instance
    with ``__new__`` and set only the attributes the method under test reads.
    """

    def _tickets(self, experts):
        return tuple(
            HotCacheSlotTicket(slot, expert, 1) for slot, expert in enumerate(experts)
        )

    def test_six_spec_only_tensors_promote_in_evictable_chunks(self):
        reference = {
            name: torch.arange(8 * 4, dtype=torch.int16).reshape(8, 4) + position
            for position, name in enumerate(_SIX_NAMES)
        }
        layer = torch.nn.Module()
        layer.layer_id = 0
        streamer = ExpertStreamer(layer, _SIX_NAMES, format=SpecOnlyFormat(reference))
        pinned = ExpertPinnedHostCache(
            streamer, 3, device="cpu", is_pinned=lambda expert: expert == 1
        )
        pinned.ensure_rows(torch.tensor([1]))
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cache.streamer = streamer
        cache._transfer_executor = object()
        calls = []
        cache._load_reserved_in_chunks = lambda tickets, tier: calls.append(
            (tuple(tickets), tier)
        )
        tickets = self._tickets((5, 7, 0))
        cache._load_reserved(tickets)
        self.assertEqual(calls, [(tickets, pinned)])
        self.assertEqual(pinned.evictable_rows(), 2)

    def test_a_failed_chunk_aborts_its_promotion_and_cancels_the_rest(self):
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cache.device = torch.device("cpu")
        cache.promotion_in_flight = None
        promotion = SimpleNamespace(name="first chunk")

        def prepare(tickets):
            cache.promotion_in_flight = promotion
            return promotion

        aborted, cancelled = [], []

        def abort(staged):
            aborted.append(staged)
            cache.promotion_in_flight = None

        cache._prepare_promotion = prepare
        cache.abort_promotion = abort
        cache._cancel_tickets = lambda tickets: cancelled.append(tuple(tickets))
        cache._transfer_executor = SimpleNamespace(
            wait=lambda ticket, stream: (_ for _ in ()).throw(RuntimeError("copy failed"))
        )
        tickets = self._tickets((5, 7, 0))
        drained = []
        with (
            patch("torch.cuda.current_stream", return_value=None),
            patch("torch.cuda.synchronize", side_effect=drained.append),
            patch(
                "sglang.srt.layers.moe.expert_hot_cache.submit_hot_cache_promotions",
                return_value="ticket",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                cache._load_reserved_in_chunks(tickets, _FixedRoom(2))
        # The copies were submitted, so the device is drained before the slots are freed.
        self.assertEqual(drained, [cache.device])
        self.assertEqual(aborted, [promotion])
        self.assertIsNone(cache.promotion_in_flight)
        self.assertEqual(cancelled, [tickets[2:]])

    def test_an_undrainable_device_keeps_the_failed_promotion_in_flight(self):
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cache.device = torch.device("cpu")
        cache.promotion_in_flight = None
        promotion = SimpleNamespace(name="first chunk")

        def prepare(tickets):
            cache.promotion_in_flight = promotion
            return promotion

        aborted, cancelled = [], []
        cache._prepare_promotion = prepare
        cache.abort_promotion = aborted.append
        cache._cancel_tickets = lambda tickets: cancelled.append(tuple(tickets))
        cache._transfer_executor = SimpleNamespace(
            wait=lambda ticket, stream: (_ for _ in ()).throw(RuntimeError("copy failed"))
        )
        tickets = self._tickets((5, 7, 0))
        with (
            patch("torch.cuda.current_stream", return_value=None),
            patch("torch.cuda.synchronize", side_effect=RuntimeError("device lost")),
            patch(
                "sglang.srt.layers.moe.expert_hot_cache.submit_hot_cache_promotions",
                return_value="ticket",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                cache._load_reserved_in_chunks(tickets, _FixedRoom(2))
        # Its slots stay LOADING: no later reservation can reuse them.
        self.assertEqual(aborted, [])
        self.assertIs(cache.promotion_in_flight, promotion)
        self.assertEqual(cancelled, [tickets[2:]])

    def test_no_evictable_slots_cancel_every_ticket(self):
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cancelled = []
        cache._cancel_tickets = lambda tickets: cancelled.append(tuple(tickets))
        tickets = self._tickets((5, 7))
        with self.assertRaisesRegex(RuntimeError, "no evictable slots"):
            cache._load_reserved_in_chunks(tickets, _FixedRoom(0))
        self.assertEqual(cancelled, [tickets])

    def _promoting_cache(self, pinned_tier, reserved):
        """A bare hot cache whose promotion steps mirror the real pinned-tier use.

        ``prepare`` admits the chunk to the tier and requires every row there,
        as ``_prepare_promotion`` does; it then marks the chunk reserved, which
        an inclusive ``is_pinned`` turns into protection.
        """
        cache = ExpertHotCache.__new__(ExpertHotCache)
        cache.device = torch.device("cpu")
        cache.promotion_in_flight = None
        completed, cancelled = [], []

        def prepare(tickets):
            experts = [ticket.expert_id for ticket in tickets]
            pinned_tier.ensure_rows(torch.tensor(experts))
            if any(pinned_tier._expert_to_slot.get(e, -1) < 0 for e in experts):
                raise RuntimeError("promotion needs every row in the pinned host tier")
            reserved.update(experts)
            promotion = SimpleNamespace(experts=tuple(experts))
            cache.promotion_in_flight = promotion
            return promotion

        def complete(promotion):
            cache.promotion_in_flight = None
            completed.append(promotion.experts)

        cache._prepare_promotion = prepare
        cache.complete_promotion = complete
        cache._cancel_tickets = lambda tickets: cancelled.append(tuple(tickets))
        cache.wait_for_slot_publication = lambda: None
        cache._transfer_executor = SimpleNamespace(wait=lambda ticket, stream: None)
        return cache, completed, cancelled

    def _promote(self, cache, tickets, pinned_tier):
        with (
            patch(
                "torch.cuda.current_stream",
                return_value=SimpleNamespace(synchronize=lambda: None),
            ),
            patch(
                "sglang.srt.layers.moe.expert_hot_cache.submit_hot_cache_promotions",
                return_value="ticket",
            ),
        ):
            cache._load_reserved_in_chunks(tickets, pinned_tier)

    def _tier(self, capacity, is_pinned):
        reference = {
            name: torch.arange(8 * 4, dtype=torch.int16).reshape(8, 4) + position
            for position, name in enumerate(_SIX_NAMES)
        }
        layer = torch.nn.Module()
        layer.layer_id = 0
        streamer = ExpertStreamer(layer, _SIX_NAMES, format=SpecOnlyFormat(reference))
        return reference, ExpertPinnedHostCache(
            streamer, capacity, device="cpu", is_pinned=is_pinned
        )

    def test_promotion_chunks_shrink_as_promoted_rows_become_protected(self):
        # Capacity 3; only 5 becomes protected once reserved. A chunk size fixed at
        # the start (3) would make the second chunk [3, 6, 2] evict one of its own
        # rows; per-chunk sizing gives [5, 7, 0], [3, 6], [2].
        reserved = set()
        reference, tier = self._tier(3, lambda expert: expert in reserved and expert == 5)
        cache, completed, cancelled = self._promoting_cache(tier, reserved)
        self._promote(cache, self._tickets((5, 7, 0, 3, 6, 2)), tier)
        self.assertEqual(completed, [(5, 7, 0), (3, 6), (2,)])
        self.assertEqual(cancelled, [])
        for expert, slot in tier._expert_to_slot.items():
            for name, tensor in reference.items():
                self.assertTrue(torch.equal(tier.tensors[name][slot], tensor[expert]))

    def test_an_inclusive_tier_smaller_than_the_promotion_fails_cleanly(self):
        # Every reserved expert is protected (the inclusive hierarchy), so a tier
        # of 4 rows can hold one chunk of 4; the other 2 tickets are cancelled.
        reserved = set()
        reference, tier = self._tier(4, lambda expert: expert in reserved)
        cache, completed, cancelled = self._promoting_cache(tier, reserved)
        tickets = self._tickets((5, 7, 0, 3, 6, 2))
        with self.assertRaisesRegex(RuntimeError, "no evictable slots"):
            self._promote(cache, tickets, tier)
        self.assertEqual(completed, [(5, 7, 0, 3)])
        self.assertEqual(cancelled, [tickets[4:]])
        self.assertIsNone(cache.promotion_in_flight)
        self.assertEqual(sorted(tier._expert_to_slot), [0, 3, 5, 7])
        for expert, slot in tier._expert_to_slot.items():
            for name, tensor in reference.items():
                self.assertTrue(torch.equal(tier.tensors[name][slot], tensor[expert]))
```

- [ ] **Step 3: Add the CUDA tests**

Insert above the `if __name__ == "__main__":` block of `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`:

```python
def _spec_only_reference(names=None, experts=EXPERTS, seed=5):
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "w13_trellis": ((2, 8), torch.int16),
        "w13_suh": ((2, 4), torch.float16),
        "w13_svh": ((2, 2), torch.float16),
        "w2_trellis": ((1, 8), torch.int16),
        "w2_suh": ((1, 2), torch.float16),
        "w2_svh": ((1, 4), torch.float16),
    }
    names = tuple(shapes) if names is None else names
    return {
        name: torch.randint(
            0, 256, (experts,) + shapes[name][0] + (shapes[name][1].itemsize,),
            dtype=torch.uint8, generator=generator,
        ).view(shapes[name][1]).reshape((experts,) + shapes[name][0])
        for name in names
    }


def _spec_only_streamer(reference):
    from sglang.test.moe_expert_fakes import SpecOnlyFormat

    layer = torch.nn.Module()
    layer.layer_id = 0
    return ExpertStreamer(layer, tuple(reference), format=SpecOnlyFormat(reference))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSpecOnlyCuda(unittest.TestCase):
    def _assert_rows(self, reference, ids, compact, tensors):
        for name, tensor in reference.items():
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].cpu().view(torch.uint8),
                    tensor[ids.long().cpu()].view(torch.uint8),
                ),
                name,
            )

    def test_eager_gather_reads_hot_pinned_and_cold_rows_through_the_row_source(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        reference = _spec_only_reference()
        streamer = _spec_only_streamer(reference)
        hot = ExpertHotCache(streamer, 2)
        hot.reassign([0, 1])
        pinned = ExpertPinnedHostCache(streamer, 2)
        pinned.ensure_rows(torch.tensor([2], device="cuda"))
        ids = torch.tensor([[0, 2, 5], [1, 6, 2]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        self._assert_rows(reference, ids, compact, tensors)
        stats = streamer.last_gather_stats
        self.assertEqual(stats.hot_hit_rows, 2)
        self.assertEqual((stats.pinned_host_hit_rows, stats.pinned_host_miss_rows), (1, 2))
        self.assertEqual(stats.host_read_rows, 2)

    def test_promotions_go_through_the_pinned_tier_in_chunks(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        reference = _spec_only_reference()
        streamer = _spec_only_streamer(reference)
        # Expert 2 is protected, so every promotion chunk holds at most 2 rows.
        pinned = ExpertPinnedHostCache(streamer, 3, is_pinned=lambda expert: expert == 2)
        pinned.ensure_rows(torch.tensor([2], device="cuda"))
        self.assertEqual(pinned.evictable_rows(), 2)
        hot = ExpertHotCache(streamer, 5)
        hot.reassign([4, 0, 7, 3, 1])
        self.assertEqual(sorted(hot.resident_experts()), [0, 1, 3, 4, 7])
        self.assertIn(2, pinned._expert_to_slot)
        for slot, expert in enumerate(hot.slot_to_expert):
            for name, tensor in reference.items():
                self.assertTrue(
                    torch.equal(
                        hot.tensors[name][slot].cpu().view(torch.uint8),
                        tensor[expert].view(torch.uint8),
                    ),
                    (name, expert),
                )
        self.assertEqual(pinned.stats.populated_rows, 6)

    def test_non_six_spec_only_promotion_reads_through_the_row_source(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        reference = _spec_only_reference(names=("w13_trellis", "w2_trellis"))
        streamer = _spec_only_streamer(reference)
        hot = ExpertHotCache(streamer, 3)
        hot.reassign([1, 2, 5])
        for slot, expert in enumerate(hot.slot_to_expert):
            for name, tensor in reference.items():
                self.assertTrue(
                    torch.equal(hot.tensors[name][slot].cpu(), tensor[expert]),
                    (name, expert),
                )
```

- [ ] **Step 4: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/test/moe_expert_fakes.py test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py
git commit -m "test(moe): specify spec-only expert formats and their startup guards

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/layers/moe/test_expert_format.py::TestSpecOnlyFormat'
```
Expected: FAIL, 11 tests. The planner's dry run matched this list exactly.
- `test_specs_describe_the_rows_and_the_row_source_reads_them` and `test_dense_formats_have_no_spec_only_tensors` fail with `AttributeError: 'ExpertStreamer' object has no attribute 'has_spec_only_tensors'`.
- `test_the_host_arena_refuses_the_format` and `test_graph_gather_is_refused_before_any_cuda_work` fail because nothing raises the expected `ValueError`. The graph-gather test may instead see an error from the kernels import or the missing hot cache.
- All three subtests of `test_the_hot_cache_manager_refuses_every_graph_flag` fail (`SUBFAILED`). The manager does not raise "does not support graph gather".
- `test_six_spec_only_tensors_promote_in_evictable_chunks` fails with `AttributeError: 'ExpertHotCache' object has no attribute '_transfer_plan'`. `_load_reserved` does not yet send spec-only tickets to `_load_reserved_in_chunks`.
- `test_a_failed_chunk_aborts_its_promotion_and_cancels_the_rest`, `test_an_undrainable_device_keeps_the_failed_promotion_in_flight`, `test_no_evictable_slots_cancel_every_ticket`, `test_promotion_chunks_shrink_as_promoted_rows_become_protected` and `test_an_inclusive_tier_smaller_than_the_promotion_fails_cleanly` fail with `AttributeError` on `_load_reserved_in_chunks`.

These 4 tests pass at this red commit. They pin spec-only behaviour that Tasks 3 and 5 built:
- `test_the_tensor_kind_is_refused` pins that the streamer passes the knob's kind to the format. Mutation check: replace `resolve_row_source_kind()` with `"auto"` in `ExpertStreamer.__init__`, and it fails.
- `test_without_a_row_source_host_reads_are_refused` pins the uncovered-name check in `read_host_rows`. Mutation check: delete its `if missing: raise`, and it fails, because the message becomes "does not cover".
- `test_a_cpu_pinned_tier_admits_rows_through_the_row_source` pins that the tier takes its cached names from specs. Mutation check: derive `cached_names` from `streamer.source(name).device`, and it fails with `AttributeError` on `None`.
- `test_cached_gather_serves_spec_only_misses_from_the_pinned_tier` pins the pinned branch of `_gather_cached`. Mutation check: make that branch call `self._copy_source_rows(miss_source_ids, misses)` instead of `gather_rows`, and it fails on `pinned_host_miss_rows`.

- [ ] **Step 5: Add `require_graph_gather_support`**

Append to `python/sglang/srt/layers/moe/expert_format.py`:

```python
def require_graph_gather_support(streamers: Iterable["ExpertStreamer"]) -> None:
    """Raise unless every streamer's format can serve sync-free graph gathers.

    Graph gather, the GPU residency update and the doorbell all read dense,
    GPU-readable host sources frozen at startup, which spec-only tensors lack.
    """
    for streamer in streamers:
        expert_format = getattr(streamer, "format", None)
        unsupported = (
            expert_format is not None and not expert_format.supports_graph_gather
        ) or getattr(streamer, "has_spec_only_tensors", False)
        if unsupported:
            key = getattr(expert_format, "key", "dense")
            raise ValueError(
                f"expert format {key!r} of layer {streamer.layer_id} does not support "
                "graph gather; unset SGLANG_MOE_EXPERT_GRAPH_GATHER, "
                "SGLANG_MOE_GPU_RESIDENCY_UPDATE and SGLANG_MOE_EXPERT_DOORBELL"
            )
```

- [ ] **Step 6: Edit `expert_stream.py` — the property and the graph guard**

Old:
```python
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertFormat,
    ExpertTensorSpec,
    iter_expert_streamers,
    resolve_row_source_kind,
)
```
New:
```python
from sglang.srt.layers.moe.expert_format import (
    DenseLayerFormat,
    ExpertFormat,
    ExpertTensorSpec,
    iter_expert_streamers,
    require_graph_gather_support,
    resolve_row_source_kind,
)
```
Add after the `source` method of `ExpertStreamer`:

```python
    @property
    def has_spec_only_tensors(self) -> bool:
        """Whether a streamed tensor has no dense source, so only the row source reads it."""
        return any(self.source(name) is None for name in self.tensor_names)
```
In `enable_graph_gather`, old:
```python
        from sglang.kernels.ops.moe.expert_cache_transfer import (
            copy_expert_row_segments_gpu,
            expert_row_segments,
        )

        max_rows = index(max_rows)
```
New:
```python
        require_graph_gather_support((self,))
        from sglang.kernels.ops.moe.expert_cache_transfer import (
            copy_expert_row_segments_gpu,
            expert_row_segments,
        )

        max_rows = index(max_rows)
```

- [ ] **Step 7: Make the host arena refuse the format**

In `python/sglang/srt/layers/moe/expert_host_arena.py`, old:
```python
        streamers = list(iter_expert_streamers(model))
        if not streamers:
            return None
```
New:
```python
        streamers = list(iter_expert_streamers(model))
        if not streamers:
            return None
        for streamer in streamers:
            if not streamer.format.supports_host_arena or streamer.has_spec_only_tensors:
                raise ValueError(
                    f"expert format {streamer.format.key!r} of layer "
                    f"{streamer.layer_id} does not support the host arena; unset "
                    "SGLANG_MOE_EXPERT_HOST_ARENA"
                )
```

- [ ] **Step 8: Hot cache — the manager guard**

In `python/sglang/srt/layers/moe/expert_hot_cache.py`, old:
```python
from sglang.srt.layers.moe.expert_format import iter_expert_streamers
```
New:
```python
from sglang.srt.layers.moe.expert_format import (
    iter_expert_streamers,
    require_graph_gather_support,
)
```
In `ExpertHotCacheManager.from_model`, old:
```python
            streamers[layer_id] = streamer
        if not streamers:
            return None
        seed = None
```
New:
```python
            streamers[layer_id] = streamer
        if not streamers:
            return None
        if index(graph_gather_batch_size) or gpu_residency_update or expert_doorbell:
            require_graph_gather_support(streamers.values())
        seed = None
```

- [ ] **Step 9: Hot cache — spec-only promotions**

`_load_reserved`: replace the whole method (from `    def _load_reserved(self, tickets: Sequence[HotCacheSlotTicket]) -> None:` through its final `self.wait_for_slot_publication()`) with:

```python
    def _load_reserved(self, tickets: Sequence[HotCacheSlotTicket]) -> None:
        """Copy one reserved placement bundle before publishing any slot mapping.

        Tensors without a dense source are read by the streamer's row source:
        through the pinned host tier in chunks of its capacity when the layer
        has six tensors and a pinned tier, else one row at a time into staging.
        """
        if not tickets:
            return
        assert self._transfer_executor is not None
        spec_only = self.streamer.has_spec_only_tensors
        six_tensors = len(self.streamer.tensor_names) == NVFP4_TRANSFER_TENSOR_COUNT
        pinned_cache = self.streamer.pinned_host_cache
        if spec_only and six_tensors and pinned_cache is not None and pinned_cache.capacity:
            self._load_reserved_in_chunks(tickets, pinned_cache)
            return
        if not six_tensors or spec_only:
            for ticket in tickets:
                if not self.begin_loading(ticket):
                    raise RuntimeError("hot cache reservation became stale")
                source_ids = torch.tensor(
                    [ticket.expert_id], dtype=torch.long, device=self.device
                )
                outputs = {
                    name: tensor[ticket.slot : ticket.slot + 1]
                    for name, tensor in self.tensors.items()
                }
                self.streamer._copy_source_rows(source_ids, outputs)
                if not self.publish_ready(ticket):
                    raise RuntimeError("hot cache completion ticket became stale")
            return
        promotion = self._prepare_promotion(tickets)
        current_stream = torch.cuda.current_stream(self.device)
        ticket = submit_hot_cache_promotions([promotion], producer_stream=current_stream)
        self._transfer_executor.wait(ticket, current_stream)
        self.complete_promotion(promotion)
        self.wait_for_slot_publication()

    def _load_reserved_in_chunks(
        self, tickets: Sequence[HotCacheSlotTicket], pinned_cache
    ) -> None:
        """Promote tickets through the pinned tier, one chunk per transfer.

        Each chunk holds at most ``pinned_cache.evictable_rows()`` tickets,
        read when the chunk starts: an inclusive ``is_pinned`` protects the
        rows the hot cache has reserved, so every admitted chunk can shrink the
        room for the next. Each chunk's rows are admitted to the pinned tier,
        copied from its slabs, and waited for on the host before the next chunk
        may evict them. When no slot is evictable, the remaining tickets are
        cancelled and the call raises.

        If a chunk fails, the tickets after it are cancelled, and so is its own
        promotion if it is still in flight. A failed ``_prepare_promotion`` or
        submission has already cancelled its own tickets. Once copies were
        submitted, their slots are freed only after the device has drained
        them. If the device cannot drain (a sticky CUDA error), the promotion
        stays in flight with its slots LOADING. ``stage_reassign`` then refuses
        further updates, so no later reservation can reuse a slot a copy may
        still write.
        """
        tickets = tuple(tickets)
        start = 0
        while start < len(tickets):
            chunk_rows = pinned_cache.evictable_rows()
            if chunk_rows < 1:
                self._cancel_tickets(tickets[start:])
                raise RuntimeError(
                    "the pinned host tier has no evictable slots for hot cache promotions"
                )
            chunk = tickets[start : start + chunk_rows]
            promotion = None
            submitted = False
            try:
                promotion = self._prepare_promotion(chunk)
                current_stream = torch.cuda.current_stream(self.device)
                ticket = submit_hot_cache_promotions(
                    [promotion], producer_stream=current_stream
                )
                submitted = True
                self._transfer_executor.wait(ticket, current_stream)
                current_stream.synchronize()
                self.complete_promotion(promotion)
            except BaseException:
                if promotion is not None and self.promotion_in_flight is promotion:
                    if not submitted or self._drain_device():
                        self.abort_promotion(promotion)
                rest = tickets[start + len(chunk) :]
                if rest:
                    self._cancel_tickets(rest)
                raise
            start += len(chunk)
        self.wait_for_slot_publication()

    def _drain_device(self) -> bool:
        """Wait until every queued copy on the cache's device has run; False if it cannot."""
        try:
            torch.cuda.synchronize(self.device)
        except Exception:
            logger.warning(
                "hot cache promotion copies could not be drained; their slots stay "
                "LOADING and the cache refuses further updates"
            )
            return False
        return True
```

`_prepare_promotion`, old:
```python
            self._transfer_plan.set_rows(
                expert_rows,
                destination_slots,
                [ticket.generation for ticket in tickets],
                secondary_source_rows=secondary_source_rows,
            )
```
New:
```python
            if any(source is None for source in sources.values()):
                raise RuntimeError(
                    "hot cache promotion of expert tensors without a dense source "
                    "needs every row in the pinned host tier; promote at most its "
                    "capacity at once"
                )
            self._transfer_plan.set_rows(
                expert_rows,
                destination_slots,
                [ticket.generation for ticket in tickets],
                secondary_source_rows=secondary_source_rows,
            )
```

`stage_reassign`, old:
```python
                if len(self.streamer.tensor_names) != NVFP4_TRANSFER_TENSOR_COUNT:
                    self._load_reserved(tickets)
                elif tickets:
                    promotion = self._prepare_promotion(tickets)
```
New:
```python
                if (
                    len(self.streamer.tensor_names) != NVFP4_TRANSFER_TENSOR_COUNT
                    or self.streamer.has_spec_only_tensors
                ):
                    self._load_reserved(tickets)
                elif tickets:
                    promotion = self._prepare_promotion(tickets)
```

- [ ] **Step 10: Commit, push and run the new tests plus `CPU_SUITE`**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/layers/moe/expert_format.py python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/layers/moe/expert_host_arena.py python/sglang/srt/layers/moe/expert_hot_cache.py
git commit -m "feat(moe): serve spec-only expert formats and refuse graph and arena modes

Hot-cache promotions of tensors without a dense source go through the
pinned tier in chunks of its capacity (or row by row through the row
source); the host arena and every graph flag refuse formats that do not
support them, before allocating anything.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_host_tier.py test/registered/unit/layers/moe/test_expert_row_source.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py'
```
Expected: the 9 `TestSpecOnlyFormat` tests and the 6 `TestSpecOnlyPromotion` tests pass. The only failure is the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`.

---

### Task 7: `gather_experts` chunked entry and the `max_gather_rows` staging cap

Since `844bb9d7a5`, the first eager gather sizes the staging buffers at `max(capacity, num_experts)` rows. For a format with large rows, that floor is a whole layer of VRAM: an EXL3 layer is 384 × 13.3 MB = 5.1 GB. The format's `max_gather_rows` now caps the floor at `min(num_experts, max_gather_rows)`. The NVFP4 value is None, so the floor stays `num_experts`.

This task also adds `gather_experts(source_ids)`, an eager gather of a given list of distinct experts. Its `iter_gather_experts` generator chunks a large list so that no call stages more than the cap. It also adds `record_routes(topk_ids)`, so a chunked consumer can record a forward's routes once. An eager `gather()` whose distinct experts exceed a set cap raises and points at the chunked entry.

`NO_DEDUP_LIMIT` (64) still pads every deduplicated staging buffer to at least 64 rows, so the staging buffer never exceeds `max(max_gather_rows, 64)` rows.

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`:
  - imports;
  - a new module-level `_sum_gather_stats`;
  - `_gather_cached` (the staging floor);
  - `_plan_eager_routes` (record through `record_routes`);
  - `gather` (the cap check);
  - new methods `record_routes`, `gather_experts`, `iter_gather_experts` and `_staging_floor_rows`.
- Test (create): `test/registered/unit/layers/moe/test_expert_gather_experts.py` (CPU)
- Test (modify): `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`

**Interfaces:**
- Consumes:
  - Task 1: `ExpertFormat.max_gather_rows`, `DenseLayerFormat`.
  - Task 4: `_gather_eager_rows(source_ids, compact_ids, topk_ids)`.
- Produces:
  - `ExpertStreamer.gather_experts(source_ids: Tensor) -> tuple[Tensor, dict[str, Tensor]]`. `source_ids` is 1-D, non-empty, distinct and in range, with at most `max_gather_rows` entries. The call returns `(row_of_source, rows)`, where `rows[name][row_of_source[i]]` is expert `source_ids[i]`.
  - `ExpertStreamer.iter_gather_experts(source_ids: Tensor, chunk_rows: int | None = None) -> Iterator[tuple[Tensor, Tensor, dict[str, Tensor]]]`. It yields `(chunk_ids, row_of_source, rows)`, and once it finishes `last_gather_stats` is the sum over the chunks. It raises `ValueError` on the first `next()` when `source_ids` are not distinct.
  - `ExpertStreamer._plan_eager_routes(topk_ids, record: bool = True)`. `gather` passes `record=False`, checks the cap, and only then calls `record_routes`.
  - `ExpertStreamer.record_routes(topk_ids: Tensor) -> None`.
  - `expert_stream._sum_gather_stats(stats: Sequence[ExpertGatherStats]) -> ExpertGatherStats`.

- [ ] **Step 1: Write the failing CPU tests**

Create `test/registered/unit/layers/moe/test_expert_gather_experts.py`:

```python
"""CPU tests for chunked expert gathers and the format's staging cap."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_format import DenseLayerFormat
from sglang.srt.layers.moe.expert_residency import ExpertResidencyPolicy
from sglang.srt.layers.moe.expert_stream import ExpertGatherStats, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class CappedDenseFormat(DenseLayerFormat):
    def __init__(self, tensor_names, max_gather_rows):
        super().__init__(tensor_names)
        self.max_gather_rows = max_gather_rows


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


def _streamer(experts=16, max_gather_rows=None):
    layer = torch.nn.Module()
    layer.rows = torch.arange(experts * 4, dtype=torch.int32).reshape(experts, 4)
    expert_format = (
        None
        if max_gather_rows is None
        else CappedDenseFormat(("rows",), max_gather_rows)
    )
    streamer = ExpertStreamer(layer, ("rows",), format=expert_format)
    streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)

    def copy_rows(source_ids, outputs):
        for name, output in outputs.items():
            torch.index_select(getattr(layer, name), 0, source_ids.long(), out=output)
        return 0

    streamer._copy_source_rows = copy_rows
    return layer, streamer


class _ClearStaging(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()

    def tearDown(self):
        expert_stream._STAGING.clear()


class TestStagingFloor(_ClearStaging):
    def _staged_rows(self):
        return expert_stream._STAGING[("rows", torch.int32, "cpu", (4,))].shape[0]

    def test_default_floor_stages_a_whole_layer(self):
        _, streamer = _streamer(experts=80)
        ids = torch.tensor([[1, 2], [3, 1]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        streamer._gather_cached(source_ids, compact_ids, ids)
        self.assertEqual(self._staged_rows(), 80)

    def test_max_gather_rows_caps_the_floor(self):
        _, streamer = _streamer(experts=80, max_gather_rows=8)
        ids = torch.tensor([[1, 2], [3, 1]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        streamer._gather_cached(source_ids, compact_ids, ids)
        # NO_DEDUP_LIMIT still pads deduplicated staging to 64 rows.
        self.assertEqual(self._staged_rows(), 64)


class TestGatherExperts(_ClearStaging):
    def test_rows_are_indexed_by_source_position(self):
        layer, streamer = _streamer()
        ids = torch.tensor([9, 2, 5])
        row_of_source, rows = streamer.gather_experts(ids)
        self.assertEqual(tuple(row_of_source.shape), (3,))
        self.assertTrue(torch.equal(rows["rows"][row_of_source.long()], layer.rows[ids]))
        self.assertEqual(streamer.last_gather_stats.requested_rows, 3)

    def test_invalid_ids_are_refused(self):
        _, streamer = _streamer(max_gather_rows=4)
        cases = (
            (torch.tensor([[1, 2]]), "1-D"),
            (torch.tensor([], dtype=torch.long), "nonempty"),
            (torch.tensor([0, 1, 2, 3, 4]), "max_gather_rows"),
            (torch.tensor([16]), "outside"),
            (torch.tensor([3, 3]), "distinct"),
        )
        for ids, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    streamer.gather_experts(ids)

    def test_routes_are_recorded_only_by_record_routes(self):
        _, streamer = _streamer()
        streamer.residency_policy = ExpertResidencyPolicy(16, 4, device="cpu")
        streamer.gather_experts(torch.tensor([1, 2]))
        self.assertEqual(float(streamer.residency_policy.pending_counts.sum()), 0.0)
        streamer.record_routes(torch.tensor([[1, 2], [2, 5]]))
        self.assertEqual(float(streamer.residency_policy.pending_counts.sum()), 4.0)

    def test_prefetch_hooks_are_refused(self):
        _, streamer = _streamer()
        streamer.next_layer_prefetch = lambda ids: None
        with self.assertRaisesRegex(ValueError, "prefetch"):
            streamer.gather_experts(torch.tensor([1]))


class TestIterGatherExperts(_ClearStaging):
    def test_chunks_equal_one_gather_and_sum_their_stats(self):
        layer, streamer = _streamer()
        ids = torch.tensor([3, 9, 1, 14, 6])
        chunks = []
        for chunk, row_of_source, rows in streamer.iter_gather_experts(ids, chunk_rows=2):
            chunks.append((chunk.tolist(), rows["rows"][row_of_source.long()].clone()))
        self.assertEqual([chunk for chunk, _ in chunks], [[3, 9], [1, 14], [6]])
        self.assertTrue(torch.equal(torch.cat([rows for _, rows in chunks]), layer.rows[ids]))
        stats = streamer.last_gather_stats
        self.assertEqual(
            (stats.requested_rows, stats.miss_rows, stats.unique_miss_rows), (5, 5, 5)
        )

    def test_the_format_cap_is_the_default_chunk(self):
        _, streamer = _streamer(max_gather_rows=2)
        sizes = [
            chunk.numel()
            for chunk, _, _ in streamer.iter_gather_experts(torch.tensor([3, 9, 1, 14, 6]))
        ]
        self.assertEqual(sizes, [2, 2, 1])

    def test_chunks_above_the_cap_are_refused(self):
        _, streamer = _streamer(max_gather_rows=2)
        with self.assertRaisesRegex(ValueError, "max_gather_rows"):
            list(streamer.iter_gather_experts(torch.tensor([1, 2, 3]), chunk_rows=3))

    def test_ids_must_be_distinct_across_chunks(self):
        _, streamer = _streamer()
        with self.assertRaisesRegex(ValueError, "distinct"):
            list(streamer.iter_gather_experts(torch.tensor([1, 2, 1]), chunk_rows=2))

    def test_no_ids_yield_nothing(self):
        _, streamer = _streamer()
        self.assertEqual(
            list(streamer.iter_gather_experts(torch.tensor([], dtype=torch.long))), []
        )


class TestSumGatherStats(unittest.TestCase):
    def test_counts_add_and_a_fallback_marks_the_sum(self):
        total = expert_stream._sum_gather_stats(
            [
                ExpertGatherStats(2, 1, 1, gather_fallback_used=False, host_read_rows=1),
                ExpertGatherStats(3, 0, 3, gather_fallback_used=True, host_read_rows=2),
            ]
        )
        self.assertEqual((total.requested_rows, total.hot_hit_rows, total.miss_rows), (5, 1, 4))
        self.assertTrue(total.gather_fallback_used)
        self.assertEqual(total.host_read_rows, 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Add the CUDA tests**

Insert above the `if __name__ == "__main__":` block of `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`:

```python
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGatherExpertsCuda(unittest.TestCase):
    def test_chunked_gather_matches_one_gather_through_hot_and_cold_rows(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = _nvfp4_layer(pinned=False)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        ExpertHotCache(streamer, 2).reassign([1, 4])
        ids = torch.tensor([6, 1, 3, 4, 0], device="cuda")
        row_of_source, rows = streamer.gather_experts(ids)
        whole = {
            name: rows[name][row_of_source.long()].view(torch.uint8).cpu()
            for name in NVFP4_STREAM_TENSORS
        }
        pieces = {name: [] for name in NVFP4_STREAM_TENSORS}
        for _, chunk_rows_of_source, chunk_rows in streamer.iter_gather_experts(
            ids, chunk_rows=2
        ):
            for name in NVFP4_STREAM_TENSORS:
                pieces[name].append(
                    chunk_rows[name][chunk_rows_of_source.long()].view(torch.uint8).cpu()
                )
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(torch.equal(torch.cat(pieces[name]), whole[name]), name)
            self.assertTrue(torch.equal(whole[name], _source_bytes(layer, name, ids)), name)
        stats = streamer.last_gather_stats
        self.assertEqual((stats.requested_rows, stats.hot_hit_rows), (5, 2))

    def test_staging_stays_within_the_cap_and_eager_gathers_above_it_are_refused(self):
        from sglang.srt.layers.moe import expert_stream
        from sglang.srt.layers.moe.expert_format import DenseLayerFormat
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        class CappedDenseFormat(DenseLayerFormat):
            max_gather_rows = 16

        layer = _nvfp4_layer(pinned=False, experts=80)
        streamer = ExpertStreamer(
            layer, NVFP4_STREAM_TENSORS, format=CappedDenseFormat(NVFP4_STREAM_TENSORS)
        )
        ExpertHotCache(streamer, 1).reassign([0])
        expert_stream._STAGING.clear()
        ids = torch.arange(40, device="cuda")
        for _, row_of_source, rows in streamer.iter_gather_experts(ids):
            self.assertLessEqual(row_of_source.numel(), 16)
        self.assertTrue(expert_stream._STAGING)
        self.assertLessEqual(
            max(buffer.shape[0] for buffer in expert_stream._STAGING.values()), 64
        )
        from sglang.srt.layers.moe.expert_residency import ExpertResidencyPolicy

        streamer.residency_policy = ExpertResidencyPolicy(80, 1, device="cuda")
        routes = torch.arange(40, device="cuda", dtype=torch.int32).reshape(10, 4)
        with self.assertRaisesRegex(ValueError, "max_gather_rows"):
            streamer.gather(routes)
        # A refused forward records no routes.
        self.assertEqual(float(streamer.residency_policy.pending_counts.sum()), 0.0)
```

- [ ] **Step 3: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add test/registered/unit/layers/moe/test_expert_gather_experts.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py
git commit -m "test(moe): specify chunked expert gathers and the staging cap

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/layers/moe/test_expert_gather_experts.py'
```
Expected:
- `test_default_floor_stages_a_whole_layer` passes, because it pins `844bb9d7a5`. Mutation check: change `_gather_cached`'s floor from `self.num_experts` to `capacity`, and it fails with `64 != 80`.
- All five subtests of `test_invalid_ids_are_refused` fail (`SUBFAILED`), with `AttributeError` on `gather_experts`.
- `test_max_gather_rows_caps_the_floor` fails with `80 != 64`.
- The `gather_experts`, `iter_gather_experts` and `record_routes` tests fail with `AttributeError`.
- `TestSumGatherStats` fails with `AttributeError: module ... has no attribute '_sum_gather_stats'`.

- [ ] **Step 4: Edit `expert_stream.py` — imports and `_sum_gather_stats`**

Old:
```python
from dataclasses import asdict, dataclass, replace
from operator import index
from typing import Dict, Iterable, Tuple
```
New:
```python
from dataclasses import asdict, dataclass, fields, replace
from operator import index
from typing import Dict, Iterable, Iterator, Sequence, Tuple
```
Old:
```python
@dataclass
class PinnedHostCacheStats:
```
New:
```python
def _sum_gather_stats(stats: Sequence[ExpertGatherStats]) -> ExpertGatherStats:
    """Add gather stats field by field; a fallback in any gather marks the sum."""
    values = {}
    for field in fields(ExpertGatherStats):
        items = [getattr(item, field.name) for item in stats]
        values[field.name] = any(items) if isinstance(items[0], bool) else sum(items)
    return ExpertGatherStats(**values)


@dataclass
class PinnedHostCacheStats:
```

- [ ] **Step 5: Cap the staging floor in `_gather_cached`**

Old:
```python
        # Allocated at a whole layer's experts from the first gather: prefill chunks climb to
        # nearly all of them, and growing one step at a time left every outgrown buffer in
        # the allocator's cache at the prefill peak.
        padded = {
            name: _staging_buffer(
                name,
                kernel_rows,
                max(capacity, self.num_experts),
```
New:
```python
        # Allocated at a whole layer's experts from the first gather: prefill chunks climb to
        # nearly all of them, and growing one step at a time left every outgrown buffer in
        # the allocator's cache at the prefill peak. A format with large rows caps that
        # floor at its max_gather_rows and gathers in chunks (iter_gather_experts).
        padded = {
            name: _staging_buffer(
                name,
                kernel_rows,
                max(capacity, self._staging_floor_rows()),
```
Add this method directly above `def _gather_cached(`:

```python
    def _staging_floor_rows(self) -> int:
        """Rows every eager staging buffer is allocated with from the first gather."""
        cap = self.format.max_gather_rows
        return self.num_experts if cap is None else min(self.num_experts, cap)
```

- [ ] **Step 6: Record routes through `record_routes`**

In `_plan_eager_routes`, old:
```python
    def _plan_eager_routes(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
```
New:
```python
    def _plan_eager_routes(
        self, topk_ids: torch.Tensor, record: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
```
Old:
```python
        residency_policy = self.residency_policy
        if residency_policy is not None and not (
            flat_ids.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            residency_policy.record_routes(flat_ids)
        return source_ids, compact_ids
```
New:
```python
        if record:
            self.record_routes(topk_ids)
        return source_ids, compact_ids

    def record_routes(self, topk_ids: torch.Tensor) -> None:
        """Count every route of a forward in the residency policy; call once per forward.

        Skipped during CUDA stream capture. ``gather`` calls it itself; a
        consumer of ``gather_experts`` calls it with the forward's full
        ``topk_ids``.
        """
        residency_policy = self.residency_policy
        flat_ids = topk_ids.reshape(-1)
        if residency_policy is not None and not (
            flat_ids.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            residency_policy.record_routes(flat_ids)
```

- [ ] **Step 7: The cap check in `gather`, before any route is recorded**

The residency policy must not count the routes of a forward that is refused, so `gather` plans without recording, checks the cap, then records.

Old:
```python
        source_ids, compact_ids = self._plan_eager_routes(topk_ids)
        if prefetch_coordinator is not None:
```
New:
```python
        source_ids, compact_ids = self._plan_eager_routes(topk_ids, record=False)
        cap = self.format.max_gather_rows
        if cap is not None and source_ids.numel() > cap:
            raise ValueError(
                f"eager gather of {source_ids.numel()} experts exceeds the format's "
                f"max_gather_rows={cap}; gather in chunks with iter_gather_experts"
            )
        self.record_routes(topk_ids)
        if prefetch_coordinator is not None:
```

- [ ] **Step 8: Add `gather_experts` and `iter_gather_experts`**

Insert these methods directly above `def _gather_eager_rows(`:

```python
    def gather_experts(
        self, source_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gather the rows of the distinct experts ``source_ids`` for an eager consumer.

        Returns ``(row_of_source, rows)``, where ``rows[name][row_of_source[i]]``
        holds expert ``source_ids[i]``. ``rows`` are the hot cache's slot
        tensors when every expert is resident, else staging buffers that the
        next eager gather of any layer reuses. Routes are not recorded: call
        ``record_routes`` once with the forward's full ``topk_ids``. At most
        the format's ``max_gather_rows`` experts per call; see
        ``iter_gather_experts``.
        """
        if source_ids.ndim != 1:
            raise ValueError("gather_experts needs a 1-D tensor of expert IDs")
        count = source_ids.numel()
        if count == 0:
            raise ValueError("gather_experts needs a nonempty tensor of expert IDs")
        cap = self.format.max_gather_rows
        if cap is not None and count > cap:
            raise ValueError(
                f"gather of {count} experts exceeds the format's max_gather_rows={cap}"
            )
        if (
            getattr(self, "prefetch_coordinator", None) is not None
            or getattr(self, "next_layer_prefetch", None) is not None
        ):
            raise ValueError("gather_experts does not drive expert prefetch")
        if bool(((source_ids < 0) | (source_ids >= self.num_experts)).any().item()):
            raise ValueError(
                f"selected expert ID is outside [0, {self.num_experts - 1}]"
            )
        if torch.unique(source_ids).numel() != count:
            raise ValueError("gather_experts needs distinct expert IDs")
        if self.before_eager_gather is not None:
            self.before_eager_gather()
        compact_ids = _cached_arange(count, source_ids.device, source_ids.dtype)
        row_of_source, rows = self._gather_eager_rows(
            source_ids, compact_ids, source_ids.reshape(1, -1)
        )
        return row_of_source.reshape(-1), rows

    def iter_gather_experts(
        self, source_ids: torch.Tensor, chunk_rows: int | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]:
        """Yield ``(chunk_ids, row_of_source, rows)`` over chunks of distinct experts.

        Each chunk is one ``gather_experts`` call of at most ``chunk_rows``
        experts (default: the format's ``max_gather_rows``, else all at once).
        A chunk's rows share staging with the next chunk, so consume them
        before advancing. When the iteration ends, ``last_gather_stats`` holds
        the sum over its chunks, so observers count the forward once.
        """
        count = source_ids.numel()
        if count == 0:
            return
        if torch.unique(source_ids).numel() != count:
            raise ValueError("iter_gather_experts needs distinct expert IDs")
        cap = self.format.max_gather_rows
        if chunk_rows is None:
            chunk_rows = cap if cap is not None else count
        chunk_rows = index(chunk_rows)
        if chunk_rows < 1:
            raise ValueError("gather chunks need at least one row")
        if cap is not None and chunk_rows > cap:
            raise ValueError(
                f"gather chunks of {chunk_rows} rows exceed the format's "
                f"max_gather_rows={cap}"
            )
        chunk_stats = []
        try:
            for start in range(0, count, chunk_rows):
                chunk = source_ids[start : start + chunk_rows]
                row_of_source, rows = self.gather_experts(chunk)
                chunk_stats.append(self.last_gather_stats)
                yield chunk, row_of_source, rows
        finally:
            if chunk_stats:
                self.last_gather_stats = _sum_gather_stats(chunk_stats)
```

- [ ] **Step 9: Commit, push and run the new tests plus `CPU_SUITE`**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/layers/moe/expert_stream.py
git commit -m "feat(moe): gather given experts in chunks under the format's staging cap

gather_experts stages a list of distinct experts without recording
routes, iter_gather_experts chunks it at the format's max_gather_rows and
sums the chunks' stats, and that cap also bounds the staging floor that
844bb9d7a5 set to a whole layer.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/layers/moe/test_expert_gather_experts.py test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_host_tier.py test/registered/unit/layers/moe/test_expert_row_source.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py'
```
Expected:
- All 12 tests in `test_expert_gather_experts.py` pass.
- `test_expert_route_plan.py::TestEagerRouteDedup::test_residency_records_routed_multiplicity_after_dedup` still passes; it pins `_plan_eager_routes`' recording.
- The only failure is the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`.

---

### Task 8: Verify-only `FileTensorCacheGroup.open_verified`

On any manifest or size mismatch, `FileTensorCacheGroup.open` replaces every member with a fresh sparse file (`file_tensor_cache.py:120-128`). That is right for the NVFP4 loader, which rebuilds its cache. It would destroy a group another tool wrote, such as a later offline repack of EXL3 experts.

`open_verified` maps a completed group only when its manifest and member sizes verify. It never creates, truncates or replaces a file, and a verify-only group refuses `complete()` and `abort()`. `open` itself is not edited.

**Files:**
- Modify: `python/sglang/srt/model_loader/file_tensor_cache.py`. Edit `FileTensorCacheGroup.__init__`, `complete`, `abort`, add a new classmethod `open_verified`, and add a new module function `_group_layout`.
- Test (create): `test/registered/unit/model_loader/test_file_tensor_cache_verified.py` (CPU)

**Interfaces:**
- Consumes (Task 2): `ExpertFileRowReader.from_group(group, names=None, mode=None)`, used by the test only.
- Produces: `FileTensorCacheGroup.open_verified(directory, namespace, cache_identity, specs) -> FileTensorCacheGroup`:
  - it raises `FileNotFoundError` when the directory is missing, and `ValueError` when no group verifies;
  - the returned group has `cache_hit=True` and holds the cache lock until `close()`;
  - its members are private, copy-on-write mappings opened read-only (`_map_tensor(..., shared=False)`), so files may be mode 0444 and writes never reach them;
  - its `complete()` and `abort()` raise `RuntimeError`.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/model_loader/test_file_tensor_cache_verified.py`:

```python
"""CPU tests for the verify-only open of a file tensor cache group."""

import os
import tempfile
import unittest

import torch

from sglang.srt.model_loader.file_tensor_cache import (
    FileTensorCacheGroup,
    FileTensorSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NAMESPACE = "expert_rows_test"
IDENTITY = {"layout": "rows_v1", "layer": 3}
SPECS = (
    FileTensorSpec("w13_trellis", (4, 2, 4096), (8192, 4096, 1), torch.uint8),
    FileTensorSpec("w2_svh", (4, 40), (40, 1), torch.float16),
)


def _stats(paths):
    return {path: (os.stat(path).st_ino, os.stat(path).st_mtime_ns) for path in paths}


def _completed_group(directory):
    group = FileTensorCacheGroup.open(directory, NAMESPACE, IDENTITY, SPECS)
    for index, tensor in enumerate(group.tensors.values()):
        tensor.view(torch.uint8).fill_(index + 7)
    manifest_path, paths = group.manifest_path, dict(group.paths)
    group.complete()
    return manifest_path, paths


class TestOpenVerified(unittest.TestCase):
    def test_a_completed_group_opens_without_touching_its_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, paths = _completed_group(directory)
            before = _stats(list(paths.values()) + [manifest_path])
            group = FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            try:
                self.assertTrue(group.cache_hit)
                self.assertEqual(group.paths, paths)
                self.assertTrue(
                    torch.all(group.tensors["w13_trellis"].view(torch.uint8) == 7)
                )
                self.assertTrue(torch.all(group.tensors["w2_svh"].view(torch.uint8) == 8))
            finally:
                group.close()
            self.assertEqual(_stats(list(paths.values()) + [manifest_path]), before)

    def test_an_unpublished_group_is_refused_and_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            group = FileTensorCacheGroup.open(directory, NAMESPACE, IDENTITY, SPECS)
            manifest_path, paths = group.manifest_path, dict(group.paths)
            group.close()
            before = _stats(paths.values())
            with self.assertRaisesRegex(ValueError, "no verified file tensor cache group"):
                FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            self.assertEqual(_stats(paths.values()), before)
            self.assertFalse(os.path.exists(manifest_path))

    def test_a_corrupt_manifest_is_refused_and_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, paths = _completed_group(directory)
            with open(manifest_path, "w", encoding="utf-8") as stream:
                stream.write("not json")
            before = _stats(list(paths.values()) + [manifest_path])
            with self.assertRaisesRegex(ValueError, "no verified file tensor cache group"):
                FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            self.assertEqual(_stats(list(paths.values()) + [manifest_path]), before)
            with open(manifest_path, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "not json")

    def test_a_different_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            _completed_group(directory)
            with self.assertRaisesRegex(ValueError, "no verified file tensor cache group"):
                FileTensorCacheGroup.open_verified(
                    directory, NAMESPACE, {**IDENTITY, "layer": 4}, SPECS
                )

    def test_a_missing_directory_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "absent")
            with self.assertRaises(FileNotFoundError):
                FileTensorCacheGroup.open_verified(missing, NAMESPACE, IDENTITY, SPECS)
            self.assertFalse(os.path.exists(missing))

    def test_a_verified_group_maps_read_only_files_privately(self):
        with tempfile.TemporaryDirectory() as directory:
            _, paths = _completed_group(directory)
            for path in paths.values():
                os.chmod(path, 0o444)
            try:
                group = FileTensorCacheGroup.open_verified(
                    directory, NAMESPACE, IDENTITY, SPECS
                )
                try:
                    group.tensors["w2_svh"].view(torch.uint8).fill_(1)
                finally:
                    group.close()
                with open(paths["w2_svh"], "rb") as stream:
                    self.assertEqual(set(stream.read()), {8})
            finally:
                for path in paths.values():
                    os.chmod(path, 0o644)

    def test_a_verified_group_cannot_be_published_or_invalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, _ = _completed_group(directory)
            group = FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            try:
                with self.assertRaisesRegex(RuntimeError, "verify-only"):
                    group.complete()
                with self.assertRaisesRegex(RuntimeError, "verify-only"):
                    group.abort()
            finally:
                group.close()
            self.assertTrue(os.path.exists(manifest_path))
            reopened = FileTensorCacheGroup.open_verified(
                directory, NAMESPACE, IDENTITY, SPECS
            )
            reopened.close()

    @unittest.skipUnless(
        os.path.exists("/usr/include/liburing.h"), "io_uring reads need liburing"
    )
    def test_the_file_row_reader_reads_a_verified_group(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        with tempfile.TemporaryDirectory() as directory:
            _completed_group(directory)
            group = FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            try:
                reader = ExpertFileRowReader.from_group(group, mode="uring")
                destination = torch.zeros(2, 40, dtype=torch.float16)
                reader.read(torch.tensor([3, 1]), {"w2_svh": destination})
                self.assertTrue(torch.all(destination.view(torch.uint8) == 8))
            finally:
                group.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add test/registered/unit/model_loader/test_file_tensor_cache_verified.py
git commit -m "test(model_loader): specify a verify-only open of file tensor cache groups

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/model_loader/test_file_tensor_cache_verified.py'
```
Expected: FAIL. Every test fails with `AttributeError: type object 'FileTensorCacheGroup' has no attribute 'open_verified'`.

- [ ] **Step 3: Add the verify-only flag**

In `python/sglang/srt/model_loader/file_tensor_cache.py`, `FileTensorCacheGroup.__init__`, old:
```python
        self._manifest = manifest
        self._lock = lock
        self._closed = False
```
New:
```python
        self._manifest = manifest
        self._lock = lock
        self._closed = False
        # Set by open_verified: the group maps files another writer owns.
        self._verify_only = False
```
In `complete`, old:
```python
        """Durably publish every member as one complete cache group."""
        if self._closed:
            return
```
New:
```python
        """Durably publish every member as one complete cache group."""
        if self._verify_only:
            raise RuntimeError("a verify-only file tensor cache group cannot be published")
        if self._closed:
            return
```
In `abort`, old:
```python
        """Invalidate this group and release its cache lock."""
        if self._closed:
            return
```
New:
```python
        """Invalidate this group and release its cache lock."""
        if self._verify_only:
            raise RuntimeError(
                "a verify-only file tensor cache group cannot be invalidated; close it"
            )
        if self._closed:
            return
```

- [ ] **Step 4: Add `open_verified` and `_group_layout`**

Insert this classmethod directly after the `open` classmethod, before `def complete(self)`:

```python
    @classmethod
    def open_verified(
        cls,
        directory: str | os.PathLike[str],
        namespace: str,
        cache_identity: Mapping[str, Any],
        specs: Sequence[FileTensorSpec],
    ) -> "FileTensorCacheGroup":
        """Map a completed group for reading without creating or replacing any file.

        ``open`` turns any mismatch into a fresh sparse cache miss, which would
        destroy a group another tool wrote. This open maps only a group whose
        manifest and member sizes verify. It raises FileNotFoundError for a
        missing directory and ValueError when no group verifies. The group
        holds the cache lock until ``close``; ``complete`` and ``abort`` raise.
        """
        directory_path = os.path.realpath(os.path.expanduser(os.fspath(directory)))
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(
                f"file tensor cache directory {directory_path} does not exist"
            )
        namespace = str(namespace)
        normalized_specs = tuple(specs)
        if not namespace:
            raise ValueError("file tensor cache namespace must not be empty")
        if not normalized_specs:
            raise ValueError("file tensor cache group must contain at least one tensor")
        tags = [spec.tag for spec in normalized_specs]
        if len(set(tags)) != len(tags):
            raise ValueError("file tensor cache tags must be unique within a group")
        manifest, manifest_path, paths, lock = _group_layout(
            directory_path, namespace, cache_identity, normalized_specs
        )
        lock.acquire()
        try:
            if not _cache_is_valid(manifest_path, manifest, normalized_specs, paths):
                raise ValueError(
                    f"no verified file tensor cache group for namespace {namespace!r} "
                    f"in {directory_path}"
                )
            # A private mapping opens the files read-only: a read-only repack
            # works, and a stray write never reaches another tool's files.
            tensors = {
                spec.tag: _map_tensor(paths[spec.tag], spec, shared=False)
                for spec in normalized_specs
            }
            group = cls(
                cache_hit=True,
                directory=directory_path,
                manifest_path=manifest_path,
                manifest=manifest,
                specs=normalized_specs,
                paths=paths,
                tensors=tensors,
                lock=lock,
            )
            group._verify_only = True
        except BaseException:
            lock.release()
            raise
        _log_cache_event(manifest_path, manifest, "verified_hit")
        return group
```

Give `_map_tensor` a `shared` flag. Old:
```python
def _map_tensor(path: str, spec: FileTensorSpec) -> torch.Tensor:
    storage = torch.from_file(path, shared=True, size=spec.nbytes, dtype=torch.uint8)
```
New:
```python
def _map_tensor(
    path: str, spec: FileTensorSpec, *, shared: bool = True
) -> torch.Tensor:
    # shared=False maps copy-on-write from a read-only descriptor.
    storage = torch.from_file(path, shared=shared, size=spec.nbytes, dtype=torch.uint8)
```

Add this module function directly above `def _build_manifest(`. It computes the same names that `open` computes inline, and the round-trip test (`test_a_completed_group_opens_without_touching_its_files`) checks that they agree.

```python
def _group_layout(
    directory_path: str,
    namespace: str,
    cache_identity: Mapping[str, Any],
    specs: tuple[FileTensorSpec, ...],
) -> tuple[dict[str, Any], str, dict[str, str], Any]:
    """The manifest, manifest path, member paths and lock ``open`` uses for a group."""
    manifest, digest = _build_manifest(namespace, cache_identity, specs)
    stem = f"file_tensor_cache_{_safe_component(namespace)}_{digest}"
    manifest_path = os.path.join(directory_path, f"{stem}.manifest.json")
    paths = {
        spec.tag: os.path.join(
            directory_path,
            f"{stem}_{index:03d}_{_safe_component(spec.tag)}.bin",
        )
        for index, spec in enumerate(specs)
    }
    lock_digest = hashlib.sha256(
        os.path.realpath(manifest_path).encode("utf-8")
    ).hexdigest()
    return manifest, manifest_path, paths, get_lock(f"file-tensor-cache-{lock_digest}")
```

- [ ] **Step 5: Commit, push and run the new tests plus the file-cache suites**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/model_loader/file_tensor_cache.py
git commit -m "feat(model_loader): open a completed file tensor cache group verify-only

open() replaces every member with a sparse file on any mismatch, which
would destroy a group another tool wrote; open_verified maps only a
verified group, never creates or replaces files, and refuses complete()
and abort().

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/model_loader/test_file_tensor_cache_verified.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_row_source.py'
```
Expected: the 8 tests of `test_file_tensor_cache_verified.py` pass. All existing tests of `test_file_tensor_cache.py`, `test_nvfp4_expert_offload.py` and `test_expert_file_reader.py` still pass or skip as in the Task 0 baseline.

---

### Task 9: A format-aware server-args gate, tolerant tier options, and inclusive hot-slot limits

Three startup seams the EXL3 plan needs (review `F-phase3a-r2-review.md`: C1, C2, O2, I1).

**1. The server-args gate is per format.**
- **Today.** `handle_offload_compatibility` (`arg_groups/memory_hook.py:214-265`) raises for any launch that sets `SGLANG_MOE_HOT_GPU_MB` or `SGLANG_MOE_PINNED_HOST_MB`, unless the NVFP4 requirements hold:
  - `SGLANG_MOE_EXPERT_STREAM=1`, `--moe-runner-backend flashinfer_cutlass`, TP1, EP1, `--moe-a2a-backend none`, and no Waterfill;
  - `--disable-overlap-schedule`, unless graph gather runs with the GPU residency update;
  - no batch overlap, `--max-running-requests 1`, no elastic EP and no EPLB;
  - a `stat` or `per_pass` recorder under dynamic residency, the hot-cache policy knobs, and the NVFP4 CUDA-graph rules.
- **The move.** That block moves verbatim into `_check_nvfp4` in a new, light module, `arg_groups/expert_stream_requirements.py`. The only change is that the policy-knob checks become the shared `validate_hot_cache_policy()`, called at the same point.
- **The lookup.** The gate looks up the launch's requirements by quantization method, from what is known before any model loads:
  1. `--quantization`;
  2. the memoised `ModelConfig.quantization` on the second gate run (`pipeline.py:240`, after the model hooks built it);
  3. `quantization_config.quant_method` of a local `<model_path>/config.json`.

  Checked on divix01, 2026-09-18:
  - The production NVFP4 checkpoint's `config.json` has no `quant_method`, and its `ModelConfig.quantization` resolves to `modelopt_mixed`.
  - The DSV4.1 EXL3 checkpoint's `config.json` has `quant_method: exl3`.
- **Outcomes.**
  - An undetermined method keeps the NVFP4 check, which is today's behaviour. That covers the first gate run of an NVFP4 launch and every existing test.
  - Every method whose MoE layers can reach `ModelOptNvFp4FusedMoEMethod._attach_expert_streamer` maps to the NVFP4 check:
    - `modelopt`, `modelopt_fp4` and `modelopt_mixed`;
    - `nvfp4_online`, whose `ModelOptNvFp4OnlineFusedMoEMethod` subclasses it (`nvfp4_online.py:198`);
    - `fp8` and `mxfp8`, whose `Fp8Config` the loader wraps in `HybridFp8NvFp4Config` for hybrid FP8+NVFP4 checkpoints (`model_loader/loader.py:215-242`, `modelopt_quant.py:1682`);
    - `inkling_nvfp4` (`models/inkling_common/quantization/config.py:204`).
  - A determined method that no format registers is rejected with a clear message. No launch passes silently.
- **Registration by other formats.**
  - A format registers its requirements from a plugin module, `sglang.srt.arg_groups.expert_stream_requirements_<method>`. The gate imports that module lazily, the first time it sees the method. The module must import only `expert_stream_requirements`, `sglang.srt.environ` and the standard library, so there is no import cycle and no heavy import at server-args time.
  - `eager_expert_stream_requirements(label, enabled=, enable_hint=)` builds EXL3's shape. It requires eager execution (both CUDA-graph phases disabled), no graph gather, no host arena, and a `stat`/`per_pass` recorder under dynamic residency.
  - It does not require `--max-running-requests 1`, the overlap schedule or `flashinfer_cutlass`.

**2. Tier options are tolerated when missing (C1).**
- The protocol still requires `pinned_tier_options`.
- `ExpertPinnedHostCacheManager.from_model` now reads it through `pinned_tier_options_of(format, layer)`. That treats a missing hook as `{}` and logs one warning per format, instead of raising `AttributeError` mid-startup.
- A CPU test drives `from_model` end to end over two spec-only fake layers, with the device injected through the options.

**3. Inclusive hot-slot limits (I1).**
- **Opt-in.** A format opts in with `inclusive_pinned_tier = True`. Its `is_pinned` protects hot-cache-resident experts, so every hot expert also holds a pinned row.
- **The limit.** `inclusive_hot_slot_limit(streamer)` is `pinned rows - max_gather_rows`, or None when the format does not opt in or the layer has no pinned tier.
- **Clamping.** `ExpertHotCacheManager.from_model` clamps each such layer's hot slots to that limit during selection, so the budget flows to the other layers. It logs each clamped layer.
- **NVFP4.** It is unaffected: `DenseLayerFormat.inclusive_pinned_tier` is False, so every limit is None and the selection loop is unchanged.

**Files:**
- Create: `python/sglang/srt/arg_groups/expert_stream_requirements.py`
- Modify: `python/sglang/srt/arg_groups/memory_hook.py`. Edit the imports, and replace the NVFP4 block at the end of `handle_offload_compatibility`.
- Modify: `python/sglang/srt/layers/moe/expert_format.py`. Add a logger, the `inclusive_pinned_tier` attribute (on the protocol and on `DenseLayerFormat`), `pinned_tier_options_of` and `inclusive_hot_slot_limit`.
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`. Edit the import and `ExpertPinnedHostCacheManager.from_model`.
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py`. Edit the import and the selection loop of `ExpertHotCacheManager.from_model`.
- Modify: `python/sglang/test/moe_expert_fakes.py`. `SpecOnlyFormat` gains `tier_options`, `max_gather_rows` and `inclusive_pinned_tier` arguments.
- Test (create): `test/registered/unit/test_expert_stream_requirements.py` (CPU)
- Test (create): `test/registered/unit/layers/moe/test_expert_tier_startup.py` (CPU)
- Test (modify): `test/registered/unit/test_nvfp4_expert_offload.py`. Append five subclasses of `HotCacheConfigurationTests` that rerun every hot-cache gate case with an NVFP4-streaming method named explicitly: `modelopt_mixed`, `modelopt_fp4`, `ModelOpt`, `nvfp4_online` and `fp8`.

**Interfaces:**
- Consumes:
  - Task 1: `ExpertFormat`, `DenseLayerFormat`, `iter_expert_streamers`.
  - Task 5: `ExpertPinnedHostCache(streamer, capacity, *, device, is_pinned)`, `ExpertPinnedHostCacheManager.from_model`, `pinned_tier_options`.
  - Task 6: `SpecOnlyFormat`, `require_graph_gather_support`.
- Produces, in `sglang.srt.arg_groups.expert_stream_requirements`:
  - `ExpertCacheBudgets(hot_budget_mb: int, pinned_budget_mb: int, graph_gather: bool)`, a frozen dataclass.
  - `ExpertStreamRequirements(label: str, check: Callable[[Any, ExpertCacheBudgets], None])`, a frozen dataclass.
  - `register_expert_stream_requirements(quant_methods: Iterable[str], requirements: ExpertStreamRequirements) -> None`. Methods are case-insensitive. Re-registering the same object is a no-op; registering a different object for a registered method raises `ValueError`.
  - `expert_quant_method(server_args, cfg) -> Optional[str]`.
  - `expert_stream_requirements_for(server_args, cfg) -> ExpertStreamRequirements`.
  - `eager_expert_stream_requirements(label: str, *, enabled: Callable[[], bool], enable_hint: str) -> ExpertStreamRequirements`.
  - `validate_hot_cache_policy() -> None`.
  - Constants: `NVFP4_EXPERT_STREAM_REQUIREMENTS`, `NVFP4_QUANT_METHODS = ("modelopt", "modelopt_fp4", "modelopt_mixed", "nvfp4_online", "fp8", "mxfp8", "inkling_nvfp4")` and `PLUGIN_MODULE_PREFIX = "sglang.srt.arg_groups.expert_stream_requirements_"`.
- Produces, in `sglang.srt.layers.moe.expert_format`:
  - `ExpertFormat.inclusive_pinned_tier: bool`, which is False on `DenseLayerFormat`;
  - `pinned_tier_options_of(expert_format, layer) -> Mapping[str, Any]`;
  - `inclusive_hot_slot_limit(streamer) -> Optional[int]`.
- Produces: `SpecOnlyFormat(reference, *, tier_options=None, max_gather_rows=None, inclusive_pinned_tier=False)`.

- [ ] **Step 1: Write the failing gate tests**

Create `test/registered/unit/test_expert_stream_requirements.py`:

```python
"""CPU tests for the per-format server-args gate of MoE expert caching."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import expert_stream_requirements as requirements_module
from sglang.srt.arg_groups import memory_hook
from sglang.srt.arg_groups.expert_stream_requirements import (
    NVFP4_EXPERT_STREAM_REQUIREMENTS,
    ExpertStreamRequirements,
    eager_expert_stream_requirements,
    expert_quant_method,
    expert_stream_requirements_for,
    register_expert_stream_requirements,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _launch(**changes):
    """Server arguments of an eager launch with neither NVFP4-only setting."""
    values = dict(
        ple_offload_embedding=False,
        cpu_offload_gb=0,
        offload_group_size=0,
        ple_offload_backend=None,
        moe_runner_backend="auto",
        tp_size=1,
        ep_size=1,
        moe_a2a_backend="none",
        disable_overlap_schedule=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        max_running_requests=4,
        expert_distribution_recorder_mode="per_pass",
        enable_waterfill=False,
        enable_eplb=False,
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend="disabled"),
            prefill=PhaseConfig(backend="disabled"),
        ),
    )
    values.update(changes)
    return SimpleNamespace(**values)


class _GateTest(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        view = patch.object(memory_hook, "resolving_view", lambda args: args)
        view.start()
        self.addCleanup(view.stop)
        registry = patch.dict(requirements_module._REGISTRY)
        registry.start()
        self.addCleanup(registry.stop)


class TestQuantMethodResolution(_GateTest):
    def _model_dir(self, directory, quantization_config):
        config = {"architectures": ["X"]}
        if quantization_config is not None:
            config["quantization_config"] = quantization_config
        with open(os.path.join(directory, "config.json"), "w") as stream:
            json.dump(config, stream)
        return directory

    def test_nothing_known_resolves_to_none(self):
        self.assertIsNone(expert_quant_method(SimpleNamespace(), SimpleNamespace()))
        args = SimpleNamespace(quantization=None, model_path="/no/such/model")
        self.assertIsNone(expert_quant_method(args, args))

    def test_config_json_names_the_method(self):
        with tempfile.TemporaryDirectory() as directory:
            self._model_dir(directory, {"quant_method": "EXL3", "bits": 3})
            args = SimpleNamespace(quantization=None, model_path=directory)
            self.assertEqual(expert_quant_method(args, args), "exl3")

    def test_config_json_without_a_method_or_readable_json_resolves_to_none(self):
        # The production NVFP4 checkpoint's quantization_config has no quant_method.
        for quantization_config in ({"quant_algo": "MIXED_PRECISION"}, None):
            with tempfile.TemporaryDirectory() as directory:
                self._model_dir(directory, quantization_config)
                args = SimpleNamespace(quantization=None, model_path=directory)
                self.assertIsNone(expert_quant_method(args, args))
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as stream:
                stream.write("{not json")
            args = SimpleNamespace(quantization=None, model_path=directory)
            self.assertIsNone(expert_quant_method(args, args))

    def test_explicit_quantization_beats_the_model_config_which_beats_config_json(self):
        with tempfile.TemporaryDirectory() as directory:
            self._model_dir(directory, {"quant_method": "exl3"})
            args = SimpleNamespace(
                quantization=None,
                model_path=directory,
                _model_config=SimpleNamespace(quantization="modelopt_mixed"),
            )
            self.assertEqual(expert_quant_method(args, args), "modelopt_mixed")
            args.quantization = "ModelOpt_FP4"
            self.assertEqual(expert_quant_method(args, args), "modelopt_fp4")


class TestRequirementsLookup(_GateTest):
    def test_undetermined_and_modelopt_methods_get_the_nvfp4_requirements(self):
        self.assertIs(
            expert_stream_requirements_for(SimpleNamespace(), SimpleNamespace()),
            NVFP4_EXPERT_STREAM_REQUIREMENTS,
        )
        for method in (
            "modelopt",
            "modelopt_fp4",
            "modelopt_mixed",
            "MODELOPT_MIXED",
            "nvfp4_online",
            "fp8",
            "mxfp8",
            "inkling_nvfp4",
        ):
            args = SimpleNamespace(quantization=method)
            with self.subTest(method=method):
                self.assertIs(
                    expert_stream_requirements_for(args, args),
                    NVFP4_EXPERT_STREAM_REQUIREMENTS,
                )

    def test_an_unknown_method_with_budgets_is_rejected(self):
        os.environ.update(SGLANG_MOE_EXPERT_STREAM="1", SGLANG_MOE_HOT_GPU_MB="1")
        with self.assertRaisesRegex(ValueError, "does not support quantization method 'awq'"):
            memory_hook.handle_offload_compatibility(_launch(quantization="awq"))
        os.environ.update(SGLANG_MOE_HOT_GPU_MB="0", SGLANG_MOE_PINNED_HOST_MB="1")
        with self.assertRaisesRegex(ValueError, "does not support quantization method 'awq'"):
            memory_hook.handle_offload_compatibility(_launch(quantization="awq"))

    def test_an_unknown_method_without_budgets_is_not_checked(self):
        memory_hook.handle_offload_compatibility(_launch(quantization="awq"))

    def test_a_plugin_module_registers_its_method_on_first_use(self):
        imported = []
        requirements = ExpertStreamRequirements("Lazy", lambda cfg, budgets: None)

        def import_module(name):
            imported.append(name)
            register_expert_stream_requirements(("lazy-fmt",), requirements)

        args = SimpleNamespace(quantization="lazy-fmt")
        with patch.object(requirements_module.importlib, "import_module", import_module):
            self.assertIs(expert_stream_requirements_for(args, args), requirements)
            self.assertIs(expert_stream_requirements_for(args, args), requirements)
        self.assertEqual(
            imported, ["sglang.srt.arg_groups.expert_stream_requirements_lazy_fmt"]
        )

    def test_a_plugin_failing_on_its_own_import_is_not_hidden(self):
        def import_module(name):
            raise ModuleNotFoundError("No module named 'missing_dep'", name="missing_dep")

        args = SimpleNamespace(quantization="broken")
        with patch.object(requirements_module.importlib, "import_module", import_module):
            with self.assertRaisesRegex(ModuleNotFoundError, "missing_dep"):
                expert_stream_requirements_for(args, args)

    def test_a_method_cannot_be_registered_twice(self):
        first = ExpertStreamRequirements("A", lambda cfg, budgets: None)
        register_expert_stream_requirements(("twice",), first)
        register_expert_stream_requirements(("twice",), first)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_expert_stream_requirements(
                ("TWICE",), ExpertStreamRequirements("B", lambda cfg, budgets: None)
            )


class TestEagerFormatRequirements(_GateTest):
    """A format registered with ``eager_expert_stream_requirements`` (EXL3's shape)."""

    def setUp(self):
        super().setUp()
        self.streaming = {"on": True}
        register_expert_stream_requirements(
            ("eager_test",),
            eager_expert_stream_requirements(
                "EAGER",
                enabled=lambda: self.streaming["on"],
                enable_hint="the test stream switch",
            ),
        )
        os.environ.update(
            SGLANG_MOE_HOT_GPU_MB="1",
            SGLANG_MOE_PINNED_HOST_MB="1",
            SGLANG_MOE_HOT_DYNAMIC="1",
        )

    def test_it_needs_none_of_the_nvfp4_only_settings(self):
        # No SGLANG_MOE_EXPERT_STREAM, overlap on, 4 running requests, auto backend.
        memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))
        os.environ["SGLANG_MOE_HOT_DYNAMIC"] = "0"
        memory_hook.handle_offload_compatibility(
            _launch(quantization="eager_test", expert_distribution_recorder_mode=None)
        )

    def test_its_own_requirements_are_enforced(self):
        cases = (
            (dict(), {"decode": "breakable"}, "runs eagerly"),
            (dict(), {"prefill": "breakable"}, "runs eagerly"),
            (dict(expert_distribution_recorder_mode=None), {}, "stat or per_pass"),
            (dict(expert_distribution_recorder_mode="per_token"), {}, "stat or per_pass"),
        )
        for changes, graphs, message in cases:
            args = _launch(quantization="eager_test", **changes)
            for phase, backend in graphs.items():
                getattr(args.cuda_graph_config, phase).backend = backend
            with self.subTest(changes=changes, graphs=graphs):
                with self.assertRaisesRegex(ValueError, message):
                    memory_hook.handle_offload_compatibility(args)
        self.streaming["on"] = False
        with self.assertRaisesRegex(ValueError, "requires the test stream switch"):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_graph_gather_and_the_host_arena_are_refused(self):
        # Satisfy the generic graph-gather checks so the format's own refusal is reached.
        os.environ.update(
            SGLANG_MOE_PINNED_HOST_MB="0",
            SGLANG_MOE_EXPERT_GRAPH_GATHER="1",
            SGLANG_MOE_EXPERT_HOST_ARENA="1",
        )
        args = _launch(quantization="eager_test")
        args.cuda_graph_config.decode.backend = "breakable"
        with self.assertRaisesRegex(ValueError, "does not support SGLANG_MOE_EXPERT_GRAPH_GATHER"):
            memory_hook.handle_offload_compatibility(args)
        os.environ["SGLANG_MOE_EXPERT_GRAPH_GATHER"] = "0"
        with self.assertRaisesRegex(ValueError, "does not support SGLANG_MOE_EXPERT_HOST_ARENA"):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_the_policy_knobs_are_checked_for_it_too(self):
        os.environ["SGLANG_MOE_HOT_LOG_INTERVAL"] = "0"
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_HOT_LOG_INTERVAL"):
            memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))


if __name__ == "__main__":
    unittest.main()
```

Append the following to `test/registered/unit/test_nvfp4_expert_offload.py`, directly above its `if __name__ == "__main__":` block. Every existing hot-cache gate case (acceptances and rejections) reruns with the quantization method named, which pins that the ModelOpt methods get exactly today's gate.

```python
class ModelOptMixedHotCacheConfigurationTests(HotCacheConfigurationTests):
    """Every hot-cache gate case again, with the production checkpoint's method named.

    The gate looks up the launch's requirements by quantization method; the
    ModelOpt methods must get exactly the NVFP4 gate the cases above pin.
    """

    quantization = "modelopt_mixed"

    def args(self, **changes):
        changes.setdefault("quantization", self.quantization)
        return super().args(**changes)


class ModelOptFp4HotCacheConfigurationTests(ModelOptMixedHotCacheConfigurationTests):
    quantization = "modelopt_fp4"


class ModelOptAutoHotCacheConfigurationTests(ModelOptMixedHotCacheConfigurationTests):
    quantization = "ModelOpt"


class NvFp4OnlineHotCacheConfigurationTests(ModelOptMixedHotCacheConfigurationTests):
    """nvfp4_online's MoE method subclasses ModelOptNvFp4FusedMoEMethod and streams too."""

    quantization = "nvfp4_online"


class HybridFp8HotCacheConfigurationTests(ModelOptMixedHotCacheConfigurationTests):
    """fp8 checkpoints with NVFP4 experts load through HybridFp8NvFp4Config and stream too."""

    quantization = "fp8"
```

- [ ] **Step 2: Write the failing tier-startup tests**

Create `test/registered/unit/layers/moe/test_expert_tier_startup.py`:

```python
"""CPU tests of the tiers' startup path: the pinned manager and inclusive hot slots."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import expert_hot_cache, expert_stream
from sglang.srt.layers.moe.expert_format import (
    inclusive_hot_slot_limit,
    pinned_tier_options_of,
)
from sglang.srt.layers.moe.expert_hot_cache import (
    ExpertHotCacheManager,
    HotCacheUpdateStats,
)
from sglang.srt.layers.moe.expert_stream import (
    ExpertPinnedHostCache,
    ExpertPinnedHostCacheManager,
    ExpertStreamer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXPERTS = 8


def _reference(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "w13_trellis": torch.randint(
            -(2**15), 2**15, (EXPERTS, 2, 6), dtype=torch.int16, generator=generator
        ),
        "w2_trellis": torch.randint(
            -(2**15), 2**15, (EXPERTS, 1, 6), dtype=torch.int16, generator=generator
        ),
    }


def _model(**format_options):
    """Two spec-only layers attached the way a quantization method attaches them."""
    model = torch.nn.Module()
    references = {}
    for layer_id in range(2):
        references[layer_id] = _reference(layer_id)
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        layer._nvfp4_expert_streamer = ExpertStreamer(
            layer,
            tuple(references[layer_id]),
            format=SpecOnlyFormat(references[layer_id], **format_options),
        )
        model.add_module(str(layer_id), layer)
    return model, references


class TestPinnedManagerStartup(unittest.TestCase):
    def test_the_manager_builds_spec_only_tiers_end_to_end(self):
        model, references = _model(tier_options={"device": "cpu"})
        streamer = model.get_submodule("0")._nvfp4_expert_streamer
        manager = ExpertPinnedHostCacheManager.from_model(
            model, budget_bytes=6 * streamer.host_bytes_per_expert
        )
        self.assertEqual(sorted(manager.caches), [0, 1])
        for layer_id, cache in manager.caches.items():
            self.assertEqual(cache.capacity, 3)
            self.assertEqual(cache.device, torch.device("cpu"))
            outputs = {
                name: torch.zeros((5,) + tuple(tensor.shape[1:]), dtype=tensor.dtype)
                for name, tensor in references[layer_id].items()
            }
            ids = torch.tensor([6, 1, 4, 0, 7])
            result = cache.gather_rows(ids, outputs)
            self.assertEqual(result.miss_rows, 5)
            for name, tensor in references[layer_id].items():
                self.assertTrue(torch.equal(outputs[name], tensor[ids]), (layer_id, name))

    def test_a_format_without_pinned_tier_options_gets_none(self):
        class NoOptions:
            key = "no_options_test"

        with self.assertLogs(
            "sglang.srt.layers.moe.expert_format", level="WARNING"
        ) as logs:
            self.assertEqual(dict(pinned_tier_options_of(NoOptions(), None)), {})
            self.assertEqual(dict(pinned_tier_options_of(NoOptions(), None)), {})
        self.assertEqual(len(logs.records), 1)
        self.assertIn("no_options_test", logs.output[0])

    def test_the_manager_tolerates_a_format_without_pinned_tier_options(self):
        model, _ = _model()
        for layer in model.children():
            streamer = layer._nvfp4_expert_streamer
            streamer.format.pinned_tier_options = None  # hide the hook
        built = []

        def fake_cache(streamer, capacity, **options):
            built.append((streamer.layer_id, capacity, options))
            return SimpleNamespace(capacity=capacity, residency_bytes=0)

        with patch.object(expert_stream, "ExpertPinnedHostCache", fake_cache):
            streamer = model.get_submodule("0")._nvfp4_expert_streamer
            ExpertPinnedHostCacheManager.from_model(
                model, budget_bytes=2 * streamer.host_bytes_per_expert
            )
        self.assertEqual(built, [(0, 1, {}), (1, 1, {})])


class TestInclusiveHotSlotLimit(unittest.TestCase):
    def test_the_limit_is_pinned_rows_minus_gather_rows_when_opted_in(self):
        model, _ = _model(
            tier_options={"device": "cpu"}, max_gather_rows=2, inclusive_pinned_tier=True
        )
        streamer = model.get_submodule("0")._nvfp4_expert_streamer
        self.assertIsNone(inclusive_hot_slot_limit(streamer))  # no pinned tier yet
        ExpertPinnedHostCache(streamer, 5, device="cpu")
        self.assertEqual(inclusive_hot_slot_limit(streamer), 3)
        streamer.format.max_gather_rows = 7
        self.assertEqual(inclusive_hot_slot_limit(streamer), 0)
        streamer.format.max_gather_rows = None
        self.assertEqual(inclusive_hot_slot_limit(streamer), 5)
        streamer.format.inclusive_pinned_tier = False
        self.assertIsNone(inclusive_hot_slot_limit(streamer))

    def test_the_dense_format_does_not_opt_in(self):
        layer = torch.nn.Module()
        layer.rows = torch.zeros(4, 3)
        streamer = ExpertStreamer(layer, ("rows",))
        ExpertPinnedHostCache(streamer, 2, device="cpu")
        self.assertIsNone(inclusive_hot_slot_limit(streamer))


class _FakeHotCache:
    """Stands in for ExpertHotCache, which needs CUDA, in the manager's selection."""

    def __init__(self, streamer, capacity, scratch_rows=0):
        self.streamer = streamer
        self.capacity = capacity
        self.device = torch.device("cpu")
        self.capacity_bytes = capacity * streamer.bytes_per_expert
        self.allocation_bytes = self.capacity_bytes
        self.scratch_bytes = 0
        self.prefetch_pull_bytes = 0
        self.last_copy_submission = None
        self.experts = []

    def reassign(self, expert_ids):
        self.experts = list(expert_ids)
        return HotCacheUpdateStats(len(self.experts), 0, 0)


class TestInclusiveHotSelection(unittest.TestCase):
    def _hot_manager(self, model, slots):
        streamer = model.get_submodule("0")._nvfp4_expert_streamer
        with (
            patch.object(expert_hot_cache, "ExpertHotCache", _FakeHotCache),
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
        ):
            return ExpertHotCacheManager.from_model(
                model,
                budget_bytes=slots * streamer.bytes_per_expert,
                seed_path=None,
                dynamic=False,
                update_prefill_tokens=16,
                min_residence_forwards=0,
                benefit_ratio=1.0,
            )

    def _pinned(self, model, rows):
        for layer_id, capacity in rows.items():
            streamer = model.get_submodule(str(layer_id))._nvfp4_expert_streamer
            ExpertPinnedHostCache(streamer, capacity, device="cpu")

    def test_without_inclusion_the_budget_splits_evenly(self):
        model, _ = _model(tier_options={"device": "cpu"}, max_gather_rows=2)
        self._pinned(model, {0: 5, 1: 8})
        manager = self._hot_manager(model, 8)
        self.assertEqual(
            {layer_id: cache.capacity for layer_id, cache in manager.caches.items()},
            {0: 4, 1: 4},
        )

    def test_an_inclusive_layer_is_clamped_and_its_slots_go_elsewhere(self):
        model, _ = _model(
            tier_options={"device": "cpu"}, max_gather_rows=2, inclusive_pinned_tier=True
        )
        self._pinned(model, {0: 5, 1: 8})
        with self.assertLogs(
            "sglang.srt.layers.moe.expert_hot_cache", level="INFO"
        ) as logs:
            manager = self._hot_manager(model, 8)
        self.assertEqual(
            {layer_id: cache.capacity for layer_id, cache in manager.caches.items()},
            {0: 3, 1: 5},
        )
        clamp_lines = [line for line in logs.output if "inclusive pinned tier" in line]
        self.assertEqual(len(clamp_lines), 1)
        self.assertIn("layer 0", clamp_lines[0])
        for layer_id, cache in manager.caches.items():
            streamer = model.get_submodule(str(layer_id))._nvfp4_expert_streamer
            self.assertLessEqual(
                cache.capacity, streamer.pinned_host_cache.capacity - 2
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Commit the red tests and run them on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add test/registered/unit/test_expert_stream_requirements.py test/registered/unit/layers/moe/test_expert_tier_startup.py test/registered/unit/test_nvfp4_expert_offload.py
git commit -m "test(moe): specify a per-format expert-caching gate and tier startup limits

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -v test/registered/unit/test_expert_stream_requirements.py test/registered/unit/layers/moe/test_expert_tier_startup.py test/registered/unit/test_nvfp4_expert_offload.py'
```
Expected:
- Both new files fail at collection. `test_expert_stream_requirements.py` fails with `ImportError: cannot import name 'expert_stream_requirements' from 'sglang.srt.arg_groups'`, and `test_expert_tier_startup.py` with `ImportError: cannot import name 'inclusive_hot_slot_limit'`.
- The five new keyed classes (`ModelOpt*`, `NvFp4Online*` and `HybridFp8HotCacheConfigurationTests`, 100 tests) pass at this red commit: the gate ignores the quantization method today. They pin behaviour. Mutation check: after Step 6, drop `"modelopt_fp4"` from `NVFP4_QUANT_METHODS`, and `ModelOptFp4HotCacheConfigurationTests` fails with "does not support quantization method 'modelopt_fp4'" on every acceptance case.

- [ ] **Step 4: Create `expert_stream_requirements.py`**

Create `python/sglang/srt/arg_groups/expert_stream_requirements.py`. The body of `_check_nvfp4`, after its first five statements, is the block moved out of `memory_hook.py` in Step 5. It is unchanged except that the eight policy-knob checks became `validate_hot_cache_policy()`.

```python
"""Server-argument requirements of MoE expert caching, per expert format.

``handle_offload_compatibility`` runs while server arguments are processed,
before any model is loaded. Once ``SGLANG_MOE_HOT_GPU_MB`` or
``SGLANG_MOE_PINNED_HOST_MB`` is nonzero, it looks up the requirements of the
launch's expert format here and runs their check.

The format is identified by its quantization method (``expert_quant_method``).
Each method that can stream experts maps to one ``ExpertStreamRequirements``;
this module registers NVFP4's. Another format registers its own by calling
``register_expert_stream_requirements`` at import time of a module named
``sglang.srt.arg_groups.expert_stream_requirements_<method>``. The gate imports
that module on the method's first use. Such a module runs during server-args
processing, so it must import only this module, ``sglang.srt.environ`` and
the standard library: no quantization, model or torch-heavy code.

A launch whose method cannot be determined yet (no ``--quantization``, no
model configuration built, no local ``config.json`` naming a method) keeps the
NVFP4 requirements every expert-caching launch had before formats existed. A
determined method that no format registers is rejected.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend

PLUGIN_MODULE_PREFIX = "sglang.srt.arg_groups.expert_stream_requirements_"
# Quantization methods whose MoE layers can get ModelOptNvFp4FusedMoEMethod, or its
# subclass ModelOptNvFp4OnlineFusedMoEMethod, and so reach _attach_expert_streamer:
# the ModelOpt configs, nvfp4_online, fp8/mxfp8 (Fp8Config, wrapped by the loader in
# HybridFp8NvFp4Config for hybrid FP8+NVFP4 checkpoints) and Inkling's NVFP4 config.
NVFP4_QUANT_METHODS = (
    "modelopt",
    "modelopt_fp4",
    "modelopt_mixed",
    "nvfp4_online",
    "fp8",
    "mxfp8",
    "inkling_nvfp4",
)


@dataclass(frozen=True)
class ExpertCacheBudgets:
    """The expert cache settings the gate has already read and range-checked."""

    hot_budget_mb: int
    pinned_budget_mb: int
    graph_gather: bool


@dataclass(frozen=True)
class ExpertStreamRequirements:
    """How one expert format validates a launch that sets expert cache budgets.

    ``check(cfg, budgets)`` raises ``ValueError`` for an unsupported launch;
    ``cfg`` is the resolving view of the server arguments.
    """

    label: str
    check: Callable[[Any, ExpertCacheBudgets], None]


_REGISTRY: dict[str, ExpertStreamRequirements] = {}


def _normalize(method: Any) -> str:
    key = str(method).strip().lower()
    if not key:
        raise ValueError("quantization method must not be empty")
    return key


def register_expert_stream_requirements(
    quant_methods: Iterable[str], requirements: ExpertStreamRequirements
) -> None:
    """Map each quantization method to ``requirements``; re-registering the same object is a no-op."""
    for method in quant_methods:
        key = _normalize(method)
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not requirements:
            raise ValueError(
                f"expert stream requirements for {key!r} are already registered "
                f"({existing.label})"
            )
        _REGISTRY[key] = requirements


def expert_quant_method(server_args: Any, cfg: Any) -> Optional[str]:
    """The launch's quantization method, from what is known before any model loads.

    In order: ``--quantization``; the quantization of the model configuration,
    when one has been built and memoised (the second gate run, after the model
    hooks); ``quantization_config.quant_method`` of ``<model_path>/config.json``
    when the model path is a local directory. None when none of them names one.
    """
    explicit = getattr(cfg, "quantization", None)
    if explicit:
        return _normalize(explicit)
    from sglang.srt.arg_groups.model_override_base import (
        ResolvedView,
        ResolvingConfig,
        record_of,
    )

    record = server_args
    if isinstance(record, (ResolvedView, ResolvingConfig)):
        record = record_of(record)
    model_config = getattr(record, "_model_config", None)
    resolved = getattr(model_config, "quantization", None)
    if resolved:
        return _normalize(resolved)
    model_path = getattr(cfg, "model_path", None)
    if not isinstance(model_path, str) or not os.path.isdir(model_path):
        return None
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, ValueError):
        return None
    quantization = config.get("quantization_config") if isinstance(config, dict) else None
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    if isinstance(method, str) and method.strip():
        return _normalize(method)
    return None


def _import_plugin(method: str) -> None:
    name = PLUGIN_MODULE_PREFIX + re.sub(r"[^0-9a-z_]", "_", method)
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        # Only this module's absence means "no plugin"; a plugin failing to
        # import one of its own dependencies must surface.
        if error.name != name:
            raise


def expert_stream_requirements_for(
    server_args: Any, cfg: Any
) -> ExpertStreamRequirements:
    """The requirements of the launch's expert format; raises for an unsupported method."""
    method = expert_quant_method(server_args, cfg)
    if method is None:
        return NVFP4_EXPERT_STREAM_REQUIREMENTS
    requirements = _REGISTRY.get(method)
    if requirements is None:
        _import_plugin(method)
        requirements = _REGISTRY.get(method)
    if requirements is None:
        raise ValueError(
            "MoE expert caching (SGLANG_MOE_HOT_GPU_MB, SGLANG_MOE_PINNED_HOST_MB) "
            f"does not support quantization method {method!r}; supported methods: "
            f"{sorted(_REGISTRY)}. A format declares its launch requirements in "
            f"{PLUGIN_MODULE_PREFIX}<method>"
        )
    return requirements


def validate_hot_cache_policy() -> None:
    """Check the hot cache's residency-policy knobs; any format with a hot budget runs this."""
    if envs.SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS.get() < 1:
        raise ValueError("SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS must be positive")
    if envs.SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS.get() < 0:
        raise ValueError("SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS must be nonnegative")
    if envs.SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS.get() < 0:
        raise ValueError("SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS must be nonnegative")
    if envs.SGLANG_MOE_HOT_DECAY_TOKENS.get() < 0:
        raise ValueError("SGLANG_MOE_HOT_DECAY_TOKENS must be nonnegative")
    sigmas = envs.SGLANG_MOE_HOT_PROMOTION_SIGMAS.get()
    if not math.isfinite(sigmas) or sigmas < 0:
        raise ValueError(
            "SGLANG_MOE_HOT_PROMOTION_SIGMAS must be finite and nonnegative"
        )
    ratio = envs.SGLANG_MOE_HOT_BENEFIT_RATIO.get()
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("SGLANG_MOE_HOT_BENEFIT_RATIO must be finite and nonnegative")
    if envs.SGLANG_MOE_HOT_LOG_INTERVAL.get() < 1:
        raise ValueError("SGLANG_MOE_HOT_LOG_INTERVAL must be positive")


def _check_nvfp4(cfg: Any, budgets: ExpertCacheBudgets) -> None:
    # Moved verbatim from handle_offload_compatibility, which read these three
    # values the same way; the stream knob has always been a raw environment read.
    hot_budget_mb = budgets.hot_budget_mb
    pinned_budget_mb = budgets.pinned_budget_mb
    graph_gather = budgets.graph_gather
    streaming = os.environ.get("SGLANG_MOE_EXPERT_STREAM") == "1"
    if not streaming:
        raise ValueError("NVFP4 hot caching requires SGLANG_MOE_EXPERT_STREAM=1")
    if cfg.moe_runner_backend != "flashinfer_cutlass":
        raise ValueError(
            "NVFP4 hot caching requires --moe-runner-backend flashinfer_cutlass"
        )
    if cfg.tp_size != 1:
        raise ValueError("NVFP4 hot caching requires TP size 1")
    if cfg.ep_size != 1:
        raise ValueError("NVFP4 hot caching requires EP size 1")
    if cfg.moe_a2a_backend != "none":
        raise ValueError("NVFP4 hot caching requires --moe-a2a-backend none")
    if cfg.enable_waterfill:
        raise ValueError("NVFP4 hot caching does not support Waterfill")
    # Graph gather with GPU residency keeps route accounting and slot changes on the
    # forward stream, so result processing may trail the next launch.
    if not cfg.disable_overlap_schedule and not (
        graph_gather and envs.SGLANG_MOE_GPU_RESIDENCY_UPDATE.get()
    ):
        raise ValueError(
            "NVFP4 hot caching requires --disable-overlap-schedule unless "
            "SGLANG_MOE_EXPERT_GRAPH_GATHER=1 and SGLANG_MOE_GPU_RESIDENCY_UPDATE=1"
        )
    if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
        raise ValueError("NVFP4 hot caching requires both batch overlap modes disabled")
    if cfg.max_running_requests != 1:
        raise ValueError("NVFP4 hot caching requires --max-running-requests 1")
    if any(
        getattr(cfg, name, None)
        for name in (
            "elastic_ep_backend",
            "elastic_ep_rejoin",
            "ep_join_mode",
            "enable_elastic_expert_backup",
            "elastic_ep_initial_size",
            "max_ep_size",
            "ep_join_rank_offset",
        )
    ):
        raise ValueError("NVFP4 hot caching does not support elastic EP")
    if cfg.enable_eplb:
        raise ValueError("NVFP4 hot caching does not support EPLB")
    if hot_budget_mb:
        recorder = cfg.expert_distribution_recorder_mode
        if envs.SGLANG_MOE_HOT_DYNAMIC.get() and recorder not in ("stat", "per_pass"):
            raise ValueError(
                "Dynamic NVFP4 hot caching requires --expert-distribution-recorder-mode "
                "stat or per_pass"
            )
        validate_hot_cache_policy()
    graph_config = cfg.cuda_graph_config
    decode_backends = (
        (Backend.BREAKABLE, Backend.FULL)
        if graph_gather
        else (Backend.DISABLED, Backend.BREAKABLE)
    )
    if graph_config is not None and (
        graph_config.decode.backend not in decode_backends
        or graph_config.prefill.backend != Backend.DISABLED
    ):
        raise ValueError(
            "NVFP4 hot caching requires decode CUDA graph capture to be "
            f"{' or '.join(decode_backends)}, and prefill CUDA graph capture to be "
            "disabled"
        )
    if pinned_budget_mb and graph_config is not None and (
        graph_config.decode.backend not in (Backend.DISABLED, Backend.BREAKABLE)
        or graph_config.prefill.backend != Backend.DISABLED
    ):
        raise ValueError(
            "NVFP4 pinned host caching requires decode CUDA graph capture to be disabled "
            "or breakable, and prefill CUDA graph capture to be disabled"
        )


NVFP4_EXPERT_STREAM_REQUIREMENTS = ExpertStreamRequirements("NVFP4", _check_nvfp4)
register_expert_stream_requirements(NVFP4_QUANT_METHODS, NVFP4_EXPERT_STREAM_REQUIREMENTS)


def eager_expert_stream_requirements(
    label: str, *, enabled: Callable[[], bool], enable_hint: str
) -> ExpertStreamRequirements:
    """Requirements of a format that streams experts eagerly only.

    The launch must enable the format's streaming (``enabled()``, described by
    ``enable_hint``), capture no CUDA graphs, use neither the graph gather nor
    the host arena, and, for dynamic residency, record routes with the
    ``stat`` or ``per_pass`` recorder. Nothing else is required: not
    ``--max-running-requests 1``, not the overlap schedule setting, not a MoE
    runner backend. A format wanting more composes its own ``check`` around
    this one's.
    """

    def check(cfg: Any, budgets: ExpertCacheBudgets) -> None:
        if not enabled():
            raise ValueError(f"{label} expert caching requires {enable_hint}")
        if budgets.graph_gather:
            raise ValueError(
                f"{label} expert caching does not support "
                "SGLANG_MOE_EXPERT_GRAPH_GATHER; set it to 0"
            )
        if envs.SGLANG_MOE_EXPERT_HOST_ARENA.get():
            raise ValueError(
                f"{label} expert caching does not support "
                "SGLANG_MOE_EXPERT_HOST_ARENA; set it to 0"
            )
        graph_config = cfg.cuda_graph_config
        if graph_config is not None and (
            graph_config.decode.backend != Backend.DISABLED
            or graph_config.prefill.backend != Backend.DISABLED
        ):
            raise ValueError(
                f"{label} expert caching runs eagerly; use --disable-cuda-graph "
                "(or --cuda-graph-backend-decode disabled "
                "--cuda-graph-backend-prefill disabled)"
            )
        if budgets.hot_budget_mb:
            recorder = cfg.expert_distribution_recorder_mode
            if envs.SGLANG_MOE_HOT_DYNAMIC.get() and recorder not in ("stat", "per_pass"):
                raise ValueError(
                    f"Dynamic {label} hot caching requires "
                    "--expert-distribution-recorder-mode stat or per_pass"
                )
            validate_hot_cache_policy()

    return ExpertStreamRequirements(label, check)
```

- [ ] **Step 5: Route `handle_offload_compatibility` through the registry**

In `python/sglang/srt/arg_groups/memory_hook.py`, old:
```python
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
```
New:
```python
from sglang.srt.arg_groups.expert_stream_requirements import (
    ExpertCacheBudgets,
    expert_stream_requirements_for,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
```
Replace everything from the line `    if not streaming:` down to the end of `handle_offload_compatibility`, i.e. the line before `def handle_gpu_memory_settings(`, with:
```python
    # Each expert format declares its own launch requirements; a launch whose
    # quantization method is not known yet keeps the NVFP4 requirements.
    expert_stream_requirements_for(server_args, cfg).check(
        cfg,
        ExpertCacheBudgets(
            hot_budget_mb=hot_budget_mb,
            pinned_budget_mb=pinned_budget_mb,
            graph_gather=graph_gather,
        ),
    )
```
The moved block was the module's only use of `math`, so drop that import. Old:
```python
import logging
import math
import os
```
New:
```python
import logging
import os
```
Then run `python3 -m pyflakes python/sglang/srt/arg_groups/memory_hook.py python/sglang/srt/arg_groups/expert_stream_requirements.py` on the laptop. Expected: no output.

- [ ] **Step 6: Add the tier hooks to `expert_format.py`**

Old:
```python
import math
from dataclasses import dataclass
```
New:
```python
import logging
import math
from dataclasses import dataclass
```
Old:
```python
# The attribute a quantization method sets on a MoE layer to attach its streamer.
```
New:
```python
logger = logging.getLogger(__name__)
_WARNED_WITHOUT_TIER_OPTIONS: set[str] = set()

# The attribute a quantization method sets on a MoE layer to attach its streamer.
```
Old (the protocol's attributes):
```python
    supports_host_arena: bool
    max_gather_rows: Optional[int]

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]: ...
```
New:
```python
    supports_host_arena: bool
    max_gather_rows: Optional[int]
    # True when the format's is_pinned protects its hot-cache experts, so the
    # pinned tier holds every hot row (see inclusive_hot_slot_limit).
    inclusive_pinned_tier: bool

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]: ...
```
Old (`DenseLayerFormat`'s attributes):
```python
    supports_host_arena = True
    max_gather_rows: Optional[int] = None
```
New:
```python
    supports_host_arena = True
    max_gather_rows: Optional[int] = None
    inclusive_pinned_tier = False
```
Append to the end of the file:
```python
def pinned_tier_options_of(expert_format: Any, layer: torch.nn.Module) -> Mapping[str, Any]:
    """``expert_format.pinned_tier_options(layer)``, or no options for a format without the hook.

    The protocol requires the hook; a format missing it gets a default pinned
    tier (no ``is_pinned`` filter) and one warning, instead of failing startup.
    """
    hook = getattr(expert_format, "pinned_tier_options", None)
    if hook is None:
        key = str(getattr(expert_format, "key", type(expert_format).__name__))
        if key not in _WARNED_WITHOUT_TIER_OPTIONS:
            _WARNED_WITHOUT_TIER_OPTIONS.add(key)
            logger.warning(
                "expert format %r has no pinned_tier_options; its pinned host tier "
                "gets default options (no is_pinned filter)",
                key,
            )
        return {}
    return hook(layer)


def inclusive_hot_slot_limit(streamer: "ExpertStreamer") -> Optional[int]:
    """The most hot-cache slots a layer may hold when its pinned tier is inclusive.

    An inclusive pinned tier keeps every hot expert in host memory too,
    protected from eviction, and an eager gather needs room for up to
    ``max_gather_rows`` more rows beside them; so the layer may hold at most
    ``pinned rows - max_gather_rows`` hot slots (never below 0). None when the
    format does not set ``inclusive_pinned_tier`` or the layer has no pinned tier.
    """
    expert_format = streamer.format
    if not getattr(expert_format, "inclusive_pinned_tier", False):
        return None
    cache = streamer.pinned_host_cache
    if cache is None or not cache.capacity:
        return None
    return max(cache.capacity - (expert_format.max_gather_rows or 0), 0)
```

- [ ] **Step 7: Read tier options tolerantly in the pinned manager**

In `python/sglang/srt/layers/moe/expert_stream.py`, old:
```python
    iter_expert_streamers,
    require_graph_gather_support,
    resolve_row_source_kind,
)
```
New:
```python
    iter_expert_streamers,
    pinned_tier_options_of,
    require_graph_gather_support,
    resolve_row_source_kind,
)
```
Old:
```python
                **streamers[layer_id].format.pinned_tier_options(
                    streamers[layer_id].layer
                ),
```
New:
```python
                **pinned_tier_options_of(
                    streamers[layer_id].format, streamers[layer_id].layer
                ),
```

- [ ] **Step 8: Clamp inclusive layers' hot slots during selection**

In `python/sglang/srt/layers/moe/expert_hot_cache.py`, old:
```python
from sglang.srt.layers.moe.expert_format import (
    iter_expert_streamers,
    require_graph_gather_support,
)
```
New:
```python
from sglang.srt.layers.moe.expert_format import (
    inclusive_hot_slot_limit,
    iter_expert_streamers,
    require_graph_gather_support,
)
```
Old:
```python
        chosen = {layer_id: set() for layer_id in streamers}
        for floor_pass in (True, False):
            for _, expert_id, layer_id in candidates:
                if expert_id in chosen[layer_id] or (
                    floor_pass and len(chosen[layer_id]) >= floors[layer_id]
                ):
                    continue
```
New:
```python
        chosen = {layer_id: set() for layer_id in streamers}
        # A format with an inclusive pinned tier keeps every hot expert in host memory
        # too, so its layers hold at most `inclusive_hot_slot_limit` slots and the
        # budget they cannot use goes to other layers. Other formats have no limit.
        slot_limits = {
            layer_id: inclusive_hot_slot_limit(streamer)
            for layer_id, streamer in streamers.items()
        }
        clamped = set()
        for floor_pass in (True, False):
            for _, expert_id, layer_id in candidates:
                if expert_id in chosen[layer_id] or (
                    floor_pass and len(chosen[layer_id]) >= floors[layer_id]
                ):
                    continue
                limit = slot_limits[layer_id]
                if limit is not None and len(chosen[layer_id]) >= limit:
                    clamped.add(layer_id)
                    continue
```
Old:
```python
        for _, expert_id, layer_id in candidates:
            if expert_id in chosen[layer_id]:
                selected[layer_id].append(expert_id)
```
New:
```python
        for _, expert_id, layer_id in candidates:
            if expert_id in chosen[layer_id]:
                selected[layer_id].append(expert_id)
        for layer_id in sorted(clamped):
            streamer = streamers[layer_id]
            logger.info(
                "Expert hot cache clamps layer %d to %d slots: its inclusive pinned "
                "tier holds %d rows and an eager gather stages up to %d more",
                layer_id,
                slot_limits[layer_id],
                streamer.pinned_host_cache.capacity,
                streamer.format.max_gather_rows or 0,
            )
```

- [ ] **Step 9: Give `SpecOnlyFormat` the new options**

In `python/sglang/test/moe_expert_fakes.py`, old:
```python
    def __init__(self, reference: Mapping[str, torch.Tensor]):
        self.reference = dict(reference)
```
New:
```python
    def __init__(
        self,
        reference: Mapping[str, torch.Tensor],
        *,
        tier_options: Optional[Mapping[str, object]] = None,
        max_gather_rows: Optional[int] = None,
        inclusive_pinned_tier: bool = False,
    ):
        self.reference = dict(reference)
        self._tier_options = dict(tier_options or {})
        self.max_gather_rows = max_gather_rows
        self.inclusive_pinned_tier = inclusive_pinned_tier
```
Old:
```python
    def pinned_tier_options(self, layer):
        return {}
```
New:
```python
    def pinned_tier_options(self, layer):
        return dict(self._tier_options)
```

- [ ] **Step 10: Commit, push and run the new tests plus `CPU_SUITE`**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add python/sglang/srt/arg_groups/expert_stream_requirements.py python/sglang/srt/arg_groups/memory_hook.py python/sglang/srt/layers/moe/expert_format.py python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/layers/moe/expert_hot_cache.py python/sglang/test/moe_expert_fakes.py
git commit -m "feat(moe): make the expert-caching gate per format and bound inclusive tiers

handle_offload_compatibility now looks up the launch's requirements by
quantization method; ModelOpt methods, and launches whose method is not
known yet, keep the NVFP4 checks verbatim, other formats register theirs
from a light plugin module, and an unregistered method is rejected. The
pinned manager tolerates a format without pinned_tier_options, and a
format with an inclusive pinned tier has its hot slots per layer clamped
to pinned rows minus max_gather_rows.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF"
git push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/test_expert_stream_requirements.py test/registered/unit/layers/moe/test_expert_tier_startup.py test/registered/unit/layers/moe/test_expert_format.py test/registered/unit/layers/moe/test_expert_host_tier.py test/registered/unit/layers/moe/test_expert_row_source.py test/registered/unit/layers/moe/test_expert_gather_experts.py test/registered/unit/model_loader/test_file_tensor_cache_verified.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py test/registered/unit/layers/moe/test_expert_host_arena.py test/registered/unit/layers/moe/test_expert_file_reader.py test/registered/unit/layers/moe/test_expert_transfer.py test/registered/unit/layers/moe/test_expert_dma.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_graph_gather_scratch.py test/registered/unit/layers/moe/test_expert_residency.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/layers/moe/test_expert_prediction_runtime.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py test/registered/unit/layers/moe/test_expert_residency_clock.py test/registered/unit/layers/moe/test_expert_prediction_capture.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/test_file_row_reader.py test/registered/unit/model_loader/test_file_tensor_cache.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py'
```
Expected:
- All 14 tests of `test_expert_stream_requirements.py` pass, and all 7 of `test_expert_tier_startup.py`.
- The five keyed classes pass, 20 tests each, the same as `HotCacheConfigurationTests`.
- Every existing `HotCacheConfigurationTests`, `OffloadCompatibilityTests`, `RouteTraceConfigurationTests` and `GraphGatherStartupSizingTests` case still passes.
- The only failure is the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`.

---

### Task 10: GPU batch — the new CUDA tests and the full expert suites (controller-run, owner go-ahead required)

This is the only task that uses the GPU. The controller runs it, not an implementer subagent, and only after the owner explicitly approves GPU time in their own message. It runs in one sitting:
- every CUDA test this plan wrote: all classes in `test_expert_plugins_cuda.py`;
- the full existing expert suites, as the no-regression gate for NVFP4.

It waits for a clear GPU and never stops or relaunches production.

**Files:**
- Create, outside the repository: `/data/models/slang/nvfp4-work/cc-expert-prediction/moe-plugins-cuda-tests.sh`. It is modelled on `gating-cuda-tests.sh` in the same directory.
- No repository changes, unless a failure needs a fix. A fix goes into a new task, back through Tasks 1–9's red/green flow.

**Interfaces:**
- Consumes: every commit of Tasks 1–9 on `shared/cc/moe-expert-plugins`.
- Produces: a log directory `/data/models/slang/nvfp4-work/cc-expert-prediction/logs/moe-plugins-cuda-<stamp>/`, with `runner.log` and `suite.log`.

- [ ] **Step 1: Get the owner's explicit go-ahead**

Ask the owner in one message: "The CUDA batch for cc/moe-expert-plugins is ready: about 10–15 minutes on the 5090 under cc-gpu.lock, run only when no production server is up. May I run it?" Do not continue until the owner says yes in their own message.

- [ ] **Step 2: Bring the divix01 worktree to the branch head**

Run: `ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && git log --oneline -1 && git status --short | head -3'`
Expected: the laptop's HEAD commit, and a clean status.

- [ ] **Step 3: Write the runner script on divix01**

Run:
```bash
ssh divix01 'cat > /data/models/slang/nvfp4-work/cc-expert-prediction/moe-plugins-cuda-tests.sh' <<'SCRIPT'
#!/usr/bin/env bash
# CUDA gate for cc/moe-expert-plugins: the plan's CUDA tests plus every expert suite.
set -uo pipefail
work=/data/models/slang/nvfp4-work
base=$work/cc-expert-prediction
wt=$base/wt-moe-plugins
lock=$work/cc-gpu.lock
py=/data/models/slang/.venv/bin/python
stamp=$(date +%Y%m%d-%H%M%S)
out=$base/logs/moe-plugins-cuda-$stamp
mkdir -p "$out"
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$out/runner.log"; }

gpu_clear() {
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] &&
    flock -n "$lock" true &&
    ! ss -ltn | grep -qE ':(7867|3104[0-3]) '
}

log "waiting for a clear GPU"
until gpu_clear; do sleep 30; done
cd "$wt"
log "GPU clear at $(git rev-parse --short HEAD)"
suite=(
  test/registered/unit/layers/moe/test_expert_*.py
  test/registered/unit/layers/moe/test_prefetch_*.py
  test/registered/unit/test_nvfp4_expert_offload.py
  test/registered/unit/test_expert_stream_requirements.py
  test/registered/unit/test_file_row_reader.py
  test/registered/unit/model_loader/test_file_tensor_cache.py
  test/registered/unit/model_loader/test_file_tensor_cache_verified.py
  test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py
  test/registered/unit/models/test_qwen2_moe_bcg_streamer_dispatch.py
  test/registered/unit/kernels/test_expert_doorbell_copier.py
  test/registered/unit/kernels/test_expert_cache_transfer_warp_geometry.py
)
# Doorbell tests spin a copier thread on DOORBELL_SPIN_CORE (default 71, production's
# core); keep it inside this job's cores.
flock -n "$lock" env \
  PYTHONPATH="$wt/python:$base/analysis/dsv41-phase1/pyarrow-shim" \
  OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 DOORBELL_SPIN_CORE=63 \
  taskset -c 32-63 "$py" -m pytest -p no:cacheprovider -v -rfE "${suite[@]}" \
  > "$out/suite.log" 2>&1
rc=$?
log "DONE rc=$rc $(grep -E '(passed|failed)' "$out/suite.log" | tail -1)"
echo "$out"
SCRIPT
ssh divix01 'chmod +x /data/models/slang/nvfp4-work/cc-expert-prediction/moe-plugins-cuda-tests.sh'
```

- [ ] **Step 4: Run it in the background and wait for it to finish**

Run: `ssh divix01 'nohup /data/models/slang/nvfp4-work/cc-expert-prediction/moe-plugins-cuda-tests.sh > /data/models/slang/nvfp4-work/cc-expert-prediction/logs/moe-plugins-cuda-latest.out 2>&1 &'`

Then poll every few minutes with `ssh divix01 'tail -3 /data/models/slang/nvfp4-work/cc-expert-prediction/logs/moe-plugins-cuda-latest.out'` until it prints the `DONE rc=` line and the log directory.

- [ ] **Step 5: Read the results**

Run: `ssh divix01 'd=$(tail -1 /data/models/slang/nvfp4-work/cc-expert-prediction/logs/moe-plugins-cuda-latest.out); grep -E "^(FAILED|ERROR)|passed|failed" $d/suite.log | tail -40; grep -c "test_expert_plugins_cuda.py.*PASSED" $d/suite.log'`

Expected:
- `rc=0`, and no `FAILED` or `ERROR` lines.
- The count of passing plugin CUDA tests is 11: `TestFormatSeamCuda` 2, `TestRowSourceRoutingCuda` 2, `TestPinnedTierCuda` 2, `TestSpecOnlyCuda` 3, `TestGatherExpertsCuda` 2.
- On a GPU, `test_expert_transfer.py::TestExpertRowCopySubmission::test_gpu_submission_uses_fallback_for_nonpinned_sources` passes.

- [ ] **Step 6: Record the out-of-bounds red on the real (Triton) path**

On CPU, `test_expert_host_tier.py::TestCachedGatherPinnedOverflow` is the regression gate for the chunk logic. It exercises the CPU copy branch, where `index_select` raises on slot −1. The bug itself read `src - row_bytes` in the Triton kernel. So, with the lock still free and in the same go-ahead, run the CUDA regression test against the unfixed base code. Build a throwaway detached worktree at `e54c84a7c6` and copy in only the test file; its module-level imports exist at the base.
```bash
ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins-base e54c84a7c6 && cp /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins/test/registered/unit/layers/moe/test_expert_plugins_cuda.py /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins-base/test/registered/unit/layers/moe/ && cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins-base && flock -n /data/models/slang/nvfp4-work/cc-gpu.lock env PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q "test/registered/unit/layers/moe/test_expert_plugins_cuda.py::TestPinnedTierCuda::test_cached_gather_copies_misses_beyond_the_pinned_capacity" 2>&1 | tail -5'
```
Expected: FAIL at the base. Either the row bytes differ (an `assertTrue(torch.equal(...))` failure), or CUDA reports an illegal memory access. Record which one in the task report. The branch run in Step 5 is the green.

Keep the base worktree for Step 7 if anything failed there. Otherwise remove it with `ssh divix01 'git -C /data/models/slang/sglang worktree remove --force /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins-base'`. `--force` is needed because of the copied test file.

- [ ] **Step 7: If anything fails, separate regressions from pre-existing failures**

Stop and report the failing node ids. For each failure, the controller reruns just those node ids at the base commit `e54c84a7c6`, in the throwaway detached worktree from Step 6. It needs a fresh owner go-ahead, and uses the same lock and `taskset -c 32-63`. If Step 6 already removed the worktree, recreate it:
```bash
ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins-base e54c84a7c6'
```
A test that also fails at the base is pre-existing and is reported, not fixed here. A test that fails only on the branch is a regression. It becomes a new red/green task on this branch; do not amend earlier commits. Remove the base worktree afterwards with `ssh divix01 'git -C /data/models/slang/sglang worktree remove --force /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins-base'`.

No commit in this task.

---

## Interfaces produced for the EXL3 plan

The later EXL3 plan is written against exactly these names and signatures. All paths are under `python/sglang/srt/`.

**`layers/moe/expert_format.py`**
```python
STREAMER_ATTRIBUTE = "_nvfp4_expert_streamer"          # EXL3 sets this same attribute
FILE_SOURCE_BYTES_ATTRIBUTE = "_nvfp4_file_source_bytes_per_expert"
GENERIC_ROW_SOURCE_KINDS = ("auto", "files", "tensor")

@dataclass(frozen=True)
class ExpertTensorSpec:
    name: str
    row_shape: tuple[int, ...]            # shape[1:] of one expert row
    dtype: torch.dtype
    residence: Literal["host", "device"]
    @property
    def row_bytes(self) -> int: ...

class ExpertFormat(Protocol):
    key: str
    supports_graph_gather: bool           # EXL3: False
    supports_host_arena: bool             # EXL3: False
    max_gather_rows: Optional[int]        # EXL3: e.g. 64; None = unbounded
    def tensor_specs(self, layer) -> tuple[ExpertTensorSpec, ...]: ...
    def num_experts(self, layer) -> int: ...
    def source(self, layer, name) -> Optional[torch.Tensor]: ...    # None = spec-only (Path S)
    def default_row_source(self, layer, specs, kind: str) -> Optional[ExpertRowSource]: ...
        # kind from SGLANG_MOE_EXPERT_ROW_SOURCE; must handle "auto"/"files"/"tensor"
        # (may raise for one it cannot serve) and may define more, e.g. "shards";
        # unknown kinds raise ValueError
    def file_source_bytes_per_expert(self, layer, row_source) -> Optional[int]: ...
        # non-None enables the eager pinned tier and file counters
    def pinned_tier_options(self, layer) -> Mapping[str, Any]: ...
        # keyword arguments for this layer's ExpertPinnedHostCache, passed by
        # ExpertPinnedHostCacheManager.from_model (the production path; no model_runner edit).
        # EXL3's inclusive hierarchy returns {"is_pinned": <expert is VRAM-resident>};
        # the callable is evaluated at eviction time, so it may look up the hot
        # cache (built after the pinned tier) through expert_streamer_of(layer).hot_cache.

class DenseLayerFormat:                    # NVFP4 / default; key="dense"; pinned_tier_options -> {}
    def __init__(self, tensor_names: Iterable[str]): ...

def expert_streamer_of(module) -> Optional[ExpertStreamer]: ...
def iter_expert_streamers(model) -> Iterator[ExpertStreamer]: ...
def resolve_row_source_kind() -> str: ...
def require_graph_gather_support(streamers: Iterable[ExpertStreamer]) -> None: ...
```

**`layers/moe/expert_row_source.py`**
```python
class HostSlotLayout(Enum):
    PER_NAME = "per_name"                  # implemented
    BLOB = "blob"                          # reserved (T13)

@dataclass(frozen=True)
class RowReadStats:
    rows: int = 0; file_bytes: int = 0; split_bytes: int = 0; read_ns: int = 0; split_ns: int = 0
    def __add__(self, other: RowReadStats) -> RowReadStats: ...

class ReadTicket(Protocol):
    def done(self) -> bool: ...
    def wait(self) -> RowReadStats: ...    # re-raises the read's error

class CompletedReadTicket:
    def __init__(self, stats: RowReadStats | None = None, error: BaseException | None = None): ...

@runtime_checkable
class ExpertRowSource(Protocol):
    names: tuple[str, ...]
    num_experts: int
    host_layouts: frozenset[HostSlotLayout]
    preferred_batch_rows: int              # 0 = no preference
    file_bytes_per_expert: int
    requires_page_aligned_destinations: bool
    def covers(self, name: str) -> bool: ...
    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int: ...
    def read(self, rows: torch.Tensor, destinations: Mapping[str, torch.Tensor],
             destination_rows: torch.Tensor | None = None) -> RowReadStats: ...
    def submit(self, rows, destinations, destination_rows=None) -> ReadTicket: ...
    def close(self) -> None: ...

class SynchronousSubmit: ...                # mixin: submit() = read() now + CompletedReadTicket
class TensorRowSource(SynchronousSubmit):
    def __init__(self, lookup: Callable[[str], Optional[torch.Tensor]],
                 names: Sequence[str], num_experts: int): ...
```
Contract:
- `read` runs on the model thread, synchronously, with no CUDA work.
- `rows` is a CPU integer tensor. Destinations are CPU and contiguous.
- The streamer passes every covered name in one call.
- The streamer checks `num_experts` against the layer's expert count at construction.

The framework never consults `preferred_batch_rows` or `requires_page_aligned_destinations`, and never calls `close()`. The host arena drops a source by assigning None. A single `read` may therefore ask for up to `max(pinned capacity, max_gather_rows, 64)` rows, and a source with a bounded bounce ring must split the work internally.

**`layers/moe/expert_file_reader.py`**
```python
class ExpertFileRowReader(SynchronousSubmit):          # an ExpertRowSource (PER_NAME)
    @classmethod
    def from_layer(cls, layer, tensor_names, mode=None) -> ExpertFileRowReader | None: ...
    @classmethod
    def from_group(cls, group: FileTensorCacheGroup, names: Iterable[str] | None = None,
                   mode: str | None = None) -> ExpertFileRowReader: ...   # mmap mode raises
    def read(self, rows, destinations, destination_rows=None) -> RowReadStats: ...
```

**`layers/moe/expert_stream.py`**
```python
class ExpertStreamer:
    def __init__(self, layer, tensor_names, *, layer_id: int | None = None,
                 format: ExpertFormat | None = None,
                 row_source: ExpertRowSource | None = <resolve from SGLANG_MOE_EXPERT_ROW_SOURCE>): ...
    format: ExpertFormat
    row_source: ExpertRowSource | None       # alias property: file_row_reader
    specs: tuple[ExpertTensorSpec, ...]      # property
    def spec(self, name) -> ExpertTensorSpec: ...
    def source(self, name) -> torch.Tensor | None: ...
    has_spec_only_tensors: bool              # property
    file_source_bytes_per_expert: int | None # property
    background_read_stats: RowReadStats
    def read_host_rows(self, rows_cpu, destinations, destination_rows=None) -> RowReadStats: ...
    def record_routes(self, topk_ids: torch.Tensor) -> None: ...
        # contract: every id in [0, num_experts), on the policy device, with the
        # forward's full multiplicity; the caller filters out negative/sentinel ids
        # (e.g. -1 padding) first -- record_routes does not filter, so NVFP4 prefill
        # pays no extra masked-select. EXL3's _apply_streamed must pass flat[flat >= 0],
        # not the raw topk_ids.
    def gather(self, topk_ids) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...
        # raises when distinct experts > format.max_gather_rows
    def gather_experts(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...
        # 1-D distinct in-range ids, <= max_gather_rows; returns (row_of_source, rows);
        # no route recording; refuses prefetch hooks
    def iter_gather_experts(self, source_ids: torch.Tensor, chunk_rows: int | None = None
                            ) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]: ...
        # consume each chunk before advancing; last_gather_stats = sum over chunks

@dataclass(frozen=True)
class ExpertGatherStats:  # 15 existing positional fields, then:
    host_read_rows: int = 0; host_read_file_bytes: int = 0; host_read_split_bytes: int = 0
    host_read_ns: int = 0; host_split_ns: int = 0

class ExpertPinnedHostCache:
    def __init__(self, streamer, capacity: int, *, device=None,
                 is_pinned: Callable[[int], bool] | None = None): ...
    is_pinned: Callable[[int], bool] | None
    def evictable_rows(self) -> int: ...      # capacity - resident experts is_pinned protects
    def gather_rows(self, source_ids, outputs) -> PinnedGatherResult: ...
        # sizes each chunk from evictable_rows() when it starts (is_pinned may grow
        # mid-call), protects the whole chunk, checks the chunk resident before its
        # copy; RuntimeError when a chunk has misses and no slot is evictable
        # (an all-hit call always succeeds)
    def ensure_rows(self, source_ids, protected: Iterable[int] = ()) -> None: ...
        # protects the requested experts plus `protected`; transactional (rolls
        # back slots and the device map on any failure)
    def close(self) -> None: ...
```
The eager paths, `_gather_cached` and `_gather_pinned_host` (pinned tier without a hot cache), both stage exactly one set of `max(rows, 64)` rows per name. With `max_gather_rows` set, that bounds their VRAM.

**`layers/moe/expert_host_tier.py`**: `PinnedSlotLRU` (with `assign(expert_id, protected=frozenset())`), `allocate_host_slab(rows, row_shape, dtype, *, register)`, `release_host_slabs(slabs)`, `PinnedGatherResult(hit_rows, miss_rows, populated_bytes, fallback_used)`, `PAGE_BYTES`.

**`layers/moe/expert_hot_cache.py`**: `_OperationalCounters` gains `host_read_rows`, `host_read_file_bytes`, `host_read_split_bytes`, `host_read_ns` and `host_split_ns`. They appear in `snapshot_counters()` rows. Spec-only layers promote synchronously. With six tensors and a pinned tier, `_load_reserved_in_chunks(tickets, pinned_cache)` promotes through the tier, sizing each chunk from `evictable_rows()` when it starts. After a failed copy wait it drains the device before freeing the slots, and if the device cannot drain it keeps the promotion in flight. Otherwise they go row by row through the row source.

**`model_loader/file_tensor_cache.py`**
```python
class FileTensorCacheGroup:
    @classmethod
    def open_verified(cls, directory, namespace: str, cache_identity: Mapping[str, Any],
                      specs: Sequence[FileTensorSpec]) -> FileTensorCacheGroup: ...
        # FileNotFoundError: directory missing; ValueError: no verified group;
        # never creates/replaces files; members are private read-only-opened
        # mappings (0444 files work); complete()/abort() raise; close() releases the lock
```

**Knob:** `SGLANG_MOE_EXPERT_ROW_SOURCE`, an `EnvStr("auto")` in `environ.py`. `DenseLayerFormat` handles it as follows:
- `auto`: `ExpertFileRowReader.from_layer`, or None under `SGLANG_MOE_EXPERT_FILE_READER=mmap`;
- `files`: the same, but it raises under `mmap`;
- `tensor`: None;
- anything else raises.

A format may add kinds, such as EXL3 `shards`. `SGLANG_MOE_EXPERT_FILE_READER` still picks buffered or `O_DIRECT` reads.

**Discovery:** EXL3 sets `layer._nvfp4_expert_streamer = ExpertStreamer(layer, names, layer_id=..., format=Exl3ExpertFormat(...))`. The attribute is not renamed.

**Startup: tier options and inclusive hot slots** (`layers/moe/expert_format.py`, Task 9)
```python
class ExpertFormat(Protocol):
    inclusive_pinned_tier: bool   # EXL3: True (its is_pinned protects hot-cache experts); dense: False
def pinned_tier_options_of(expert_format, layer) -> Mapping[str, Any]: ...
    # the manager's read of pinned_tier_options; a format without the hook gets {} and one warning
def inclusive_hot_slot_limit(streamer) -> Optional[int]: ...
    # pinned rows - (max_gather_rows or 0), floored at 0; None unless inclusive_pinned_tier
    # and the layer has a pinned tier. ExpertHotCacheManager.from_model clamps each such
    # layer's hot slots to it during selection (the unused budget goes to other layers)
    # and logs "Expert hot cache clamps layer <id> to <n> slots".
```
The EXL3 format sets `inclusive_pinned_tier = True` and `max_gather_rows`. Its `pinned_tier_options(layer)` returns `{"is_pinned": <expert in the layer's hot cache>}`, reading `expert_streamer_of(layer).hot_cache` when called: the pinned tier is built before the hot cache.

**Server-args gate** (`arg_groups/expert_stream_requirements.py`, Task 9)
```python
@dataclass(frozen=True)
class ExpertCacheBudgets:
    hot_budget_mb: int; pinned_budget_mb: int; graph_gather: bool

@dataclass(frozen=True)
class ExpertStreamRequirements:
    label: str
    check: Callable[[Any, ExpertCacheBudgets], None]   # cfg is the resolving view

PLUGIN_MODULE_PREFIX = "sglang.srt.arg_groups.expert_stream_requirements_"
NVFP4_QUANT_METHODS = ("modelopt", "modelopt_fp4", "modelopt_mixed", "nvfp4_online",
                       "fp8", "mxfp8", "inkling_nvfp4")
NVFP4_EXPERT_STREAM_REQUIREMENTS: ExpertStreamRequirements

def register_expert_stream_requirements(quant_methods: Iterable[str],
                                        requirements: ExpertStreamRequirements) -> None: ...
def expert_quant_method(server_args, cfg) -> Optional[str]: ...
    # --quantization, else the memoised ModelConfig.quantization, else
    # quantization_config.quant_method of a local <model_path>/config.json, else None
def expert_stream_requirements_for(server_args, cfg) -> ExpertStreamRequirements: ...
    # None -> NVFP4 (legacy); registered -> it; else imports
    # PLUGIN_MODULE_PREFIX + method, then raises ValueError for an unregistered method
def eager_expert_stream_requirements(label: str, *, enabled: Callable[[], bool],
                                     enable_hint: str) -> ExpertStreamRequirements: ...
    # requires enabled(), no graph gather, no host arena, both CUDA-graph phases
    # disabled, a stat/per_pass recorder under SGLANG_MOE_HOT_DYNAMIC; runs
    # validate_hot_cache_policy(); requires nothing else
def validate_hot_cache_policy() -> None: ...
```
**Where the EXL3 plan registers.**
- **The module.** It creates `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`. That file imports only `sglang.srt.arg_groups.expert_stream_requirements`, `sglang.srt.environ` and the standard library, and calls at import time:
```python
register_expert_stream_requirements(
    ("exl3",),
    eager_expert_stream_requirements(
        "EXL3",
        enabled=lambda: envs.SGLANG_DSV41_EXPERT_STREAM.get(),
        enable_hint="SGLANG_DSV41_EXPERT_STREAM=1",
    ),
)
```
- **Import timing.** The gate imports that module on the first launch whose method resolves to `exl3`. The DSV4.1 checkpoint's `config.json` names `quant_method: exl3`, so this happens on the first gate run. No framework file names EXL3.

## Design decisions that depart from the research doc

1. **`expert_row_source.py` lives in `layers/moe/`, not `model_loader/`.** Importing any `sglang.srt.model_loader` module runs the package `__init__`, which reaches `modelopt_quant`, which imports `expert_stream`. That is the reason for the existing lazy import comment in `ExpertStreamer.__init__`. `expert_stream` must import the row-source types at module level.
2. **The file-bytes gate is a format hook, `file_source_bytes_per_expert(layer, row_source)`, not "the layer attribute, else the row source's bytes".** Under `uring`, NVFP4 can have an `ExpertFileRowReader` while `_nvfp4_file_source_bytes_per_expert` is absent: it is set only when all four tags verify. The research's rule would newly enable the eager pinned tier there. `DenseLayerFormat` returns exactly the layer attribute.
3. **`default_row_source(layer, specs, kind)` takes the knob's kind.** Generic kinds are handled by each format, and unknown kinds raise inside the format, as the brief asked.
4. **The out-of-bounds fix chunks by evictable slots instead of copying `_gather_pinned_host`'s resident/cold split.** Chunks are sized by `evictable_rows()`, the capacity minus protected residents. The fix only chunks when a call names more distinct experts than that; otherwise the call sequence is the old one exactly. It also keeps `844bb9d7a5`'s in-place miss staging, where the cold split would need extra staging and `index_copy_`.
5. **Row-read stats fields are named `host_read_*`, not `file_*`.** `TensorRowSource` reads are not file reads. Reads outside eager gathers go to `streamer.background_read_stats`, not to the manager counters, so promotions and seeding are not attributed to the next gather.
6. **Spec-only promotions are synchronous.** The async six-pair promotion needs every row in the pinned tier while copies are in flight. `stage_reassign` therefore routes spec-only layers to `_load_reserved`, which promotes in chunks of `evictable_rows()` and waits on the host between chunks. With `SGLANG_MOE_HOT_ASYNC_PROMOTIONS`, spec-only layers still promote synchronously.
7. **The staging cap is `max(min(E, max_gather_rows), NO_DEDUP_LIMIT)`.** Deduplicated gathers still pad to 64 rows. An eager `gather()` over the cap raises instead of staging past it, and chunked consumers use `iter_gather_experts`.
8. **The verify-only open is a new classmethod, `open_verified`, that raises on any mismatch.** It is not a flag on `open`, and `open` itself is untouched. It holds the lock until `close()`, and refuses `complete()` and `abort()`.
9. **`ExpertFileRowReader.from_group` raises in `mmap` mode.** It does not return None; a caller wanting mmap wraps the group's tensors in a `TensorRowSource`.
10. **No `PerNameFileRowSource` alias.** `ExpertFileRowReader` is the per-name file source.
11. **Only the pinned tier takes an injected device.** `ExpertHotCache.__init__` creates CUDA events, pinned upload buffers and a transfer executor, so a CPU device would not make it testable. Its spec-driven shapes are CUDA-tested.
12. **Pinned slabs are unregistered by `weakref.finalize` and `close()`**, rather than an `atexit` hook on the manager, so caches built directly in tests also clean up.
13. **The pinned tier's `copy_rows` gains a CPU-output branch.** It serves the CPU-device tier; production outputs are CUDA and never take it.

### Decisions on the independent review (`dsv41/.superpowers/plans-research/dsv41-phase3/E-plugin-plan-review.md`)

14. **I1, fixed (Task 5, Task 6).** Both suggested fixes are applied:
    - `PinnedSlotLRU.assign` takes `protected`, and `ensure_rows` passes the call's own experts, as `exl3_ram_cache.py` does.
    - `gather_rows` and the spec-only promotions chunk by `evictable_rows()`.

    Protection is soft: when only the call's own experts can go, the oldest of them does. That keeps the legacy in-call reassignment that over-capacity `ensure_rows` calls, such as the existing uring test at capacity 2, rely on. Chunk sizing is what guarantees residency.
    Tests: `test_a_protected_resident_shrinks_the_chunks` (capacity 2, expert 1 protected, misses [5, 7]) and `test_promotion_chunks_of_evictable_rows_are_all_resident`. The CUDA spec-only promotion test also now runs with a protected expert.
15. **I2, fixed (Task 5).** `ensure_rows` assigns and reads inside one `try`. On any exception it frees every slot the call assigned, republishes the device map, and re-raises. Evictions are counted only on success.
    Tests:
    - `test_every_slot_protected_leaves_the_tier_unchanged`;
    - `test_a_failed_assignment_rolls_back_the_calls_slots` (a partial assignment is undone, and the expert is then read fresh, not served stale);
    - `test_a_failed_read_rolls_back_the_calls_slots`.
16. **I3, fixed (Task 5).** `ExpertFormat.pinned_tier_options(layer) -> Mapping[str, Any]`, which is `{}` for `DenseLayerFormat`. `ExpertPinnedHostCacheManager.from_model` passes it into each cache. `model_runner.py` has no new edit.
    Test: `TestPinnedTierOptions`, which also shows a format can place the tier on the CPU through the same hook.
17. **M1, fixed (Task 6), and refined by N2 (item 31).** A failed chunk cancels the later tickets. No evictable slot cancels every remaining ticket and raises. Tested on CPU with a bare `ExpertHotCache.__new__` instance.
18. **M2, declined.** Protecting a whole over-capacity call's resident hits would need the hits copied first, into scattered output positions. That brings back per-call scatter staging, which `844bb9d7a5` removed from this path. Such a call is already over the tier's capacity, and every row is still correct. The cost is one extra storage read per resident expert that an earlier chunk evicts, plus hit counts that reflect the chunked order. The CPU test `test_a_pinned_only_gather_stages_one_set_of_rows` pins that accounting.
19. **M3, fixed (Task 10 Step 6).** The CUDA out-of-bounds test is run once against the base code to record the red on the Triton path. The CPU `TestCachedGatherPinnedOverflow` is named as the chunk-logic gate.
20. **M4, fixed in part.** `ExpertStreamer.__init__` validates `row_source.num_experts` (Task 3). The EXL3 interface section states that `preferred_batch_rows`, `requires_page_aligned_destinations` and `close()` are never consulted, and that a single `read` can be large. Splitting reads by `preferred_batch_rows` is declined: no current source sets it, and the EXL3 source owns its bounce ring.
21. **M5, noted, omission declined.** The Task 4 commit message records the five extra, always-zero keys. Dropping zero-valued keys would make the metrics schema data-dependent, which is worse for consumers than stable extra keys.
22. **M6, fixed as prose; assert declined.** Task 1 states why freezing `residence` at construction is safe. A debug assert in `_copy_source_rows` is declined: that function runs on production prefill, and the plan keeps its readable branches verbatim.
23. **M7, fixed (Task 8).** `open_verified` maps members with `torch.from_file(..., shared=False)`. That is a private, copy-on-write mapping of a read-only descriptor; the planner checked on divix01 that it opens mode-0444 files and that writes do not reach the file. Tested by `test_a_verified_group_maps_read_only_files_privately`.
24. **M8, fixed (Task 7).** `iter_gather_experts` rejects duplicates across chunks. Tested by `test_ids_must_be_distinct_across_chunks`.
25. **M9, fixed (Task 7).** `_plan_eager_routes(topk_ids, record=True)`; `gather` plans with `record=False`, checks the cap, then calls `record_routes`. The existing route-plan test still pins the default recording. The ordering is tested on CUDA (a refused forward leaves `pending_counts` at zero), because `gather` requires CUDA ids.
26. **M10, fixed (Task 5 Step 7b).** `_gather_pinned_host` now copies through `gather_rows` into its single staging set, instead of four sets. With `max_gather_rows` set, the pinned-without-hot-cache path is bounded like `_gather_cached`. Tested by `test_a_pinned_only_gather_stages_one_set_of_rows`. The existing CUDA pinned-only tests, which cover counts, evictions and unique read slots, are the regression gate in Task 10.
27. **M11, fixed (Task 0 Step 1).** The controller commits this plan on `cc/moe-expert-plugins` before dispatching Task 0, and Step 1 checks that the commit exists.
28. **M12, fixed.** Every red step lists its failing tests and the reason, checked against a dry run.
    - Tests that pass at their red commit are named, each with a mutation check: Task 5's two legacy pins, Task 6's four spec-only pins, and Task 7's floor pin.
    - CUDA tests are never run red (Global Constraints), except the out-of-bounds red in Task 10 Step 6.
    - The optional CPU test of `_load_reserved`'s dispatch was added: `test_six_spec_only_tensors_promote_in_evictable_chunks`.

### Decisions on the re-review (same file, "## Re-review")

29. **N1, fixed (Tasks 5 and 6).** The room is now re-measured at every chunk.
    - `gather_rows` re-reads `evictable_rows()` at the start of every chunk and sizes the chunk from it.
    - `ensure_rows` takes `protected=`, and `gather_rows` passes the whole chunk, hits as well as misses.
    - Before each copy, every chunk id is checked resident on the host; a lost row raises `RuntimeError("... evicted before their copy")` instead of reading slot −1.
    - `_load_reserved_in_chunks(tickets, pinned_cache)` re-reads `evictable_rows()` per chunk. With no room it cancels the remaining tickets and raises; the tier stays consistent, and completed chunks stay published.

    CPU tests use an `is_pinned` whose set grows as rows are admitted:
    - gathers: `test_chunks_shrink_as_admitted_rows_become_protected` and `test_a_tier_filling_with_protected_rows_fails_cleanly`;
    - promotions: `test_promotion_chunks_shrink_as_promoted_rows_become_protected` and `test_an_inclusive_tier_smaller_than_the_promotion_fails_cleanly`.

    All four fail when the chunk size is fixed at the start of the call, as the pre-fix plan did; the dry run checked this by mutation.
30. **N3, fixed (Task 5).** With no evictable slot, `gather_rows` raises only for a chunk that has misses; an all-hit call is copied. Tested by `test_an_all_hit_call_needs_no_evictable_slot`.
31. **N2, fixed (Task 6).** The base has no recovery path for a failed wait. `reassign` and `_publish_completed_promotions` simply propagate it, leaving the promotion in flight, so this plan implements the minimal safe behaviour:
    - A failure before submission aborts as before.
    - A failure after submission first drains the device with `torch.cuda.synchronize(self.device)` (`_drain_device`). The copies have then run, and `abort_promotion`'s precondition holds before the slots are freed.
    - If the drain itself fails, which means a sticky CUDA error, the promotion stays in flight with its slots LOADING. `stage_reassign` then refuses further updates, so no later reservation can reuse a slot a copy may still write.

    Tested on CPU with patched `torch.cuda.synchronize`, for both the drain and the no-drain case.

### Decisions for the EXL3 Phase 3a review (`dsv41/.superpowers/plans-research/dsv41-phase3/F-phase3a-r2-review.md`: C1, C2, O2)

32. **C2: a registry keyed by quantization method, in a new light module under `arg_groups/`.** It sits next to its only caller. Putting it in `layers/moe` would pull the MoE runner package into server-args processing.
    - The NVFP4 block moves verbatim, so the NVFP4 gate is provably today's.
    - `HotCacheConfigurationTests` is rerun by subclassing under five method names (100 tests): `modelopt_mixed`, `modelopt_fp4`, `ModelOpt`, `nvfp4_online` and `fp8`. Every existing case still runs with no method named.
    - The NVFP4 method list is every method whose MoE layers can reach `ModelOptNvFp4FusedMoEMethod._attach_expert_streamer`. That is the ModelOpt configs; `nvfp4_online`, through its subclass; `fp8` and `mxfp8`, through the loader's `HybridFp8NvFp4Config`; and `inkling_nvfp4`. The list was found by grepping for the method's subclasses and constructors, and for `_attach_expert_streamer` callers. Each of these launches passes today's gate, so each must keep passing it (re-review 3).
33. **How the format is identified.** In order: `--quantization`, then the memoised `ModelConfig.quantization`, then a local `config.json`'s `quant_method`.
    - This does not build a `ModelConfig` at gate time. That would be a heavy, possibly remote, side effect in a validation hook.
    - The first gate run happens before model hooks. There, NVFP4 production has no method (its `config.json` names none), and EXL3 is recognised from its `config.json`.
34. **An undetermined method keeps the NVFP4 check; a determined but unregistered method is rejected.** Undetermined must keep the NVFP4 check, because that is exactly today's behaviour for NVFP4 production's first gate run and for every existing test (they pass bare `SimpleNamespace` args). A determined method with no requirements fails with a message naming the method and the supported ones, so no format passes silently.
35. **Registration by lazy import of `expert_stream_requirements_<method>`.**
    - The alternative, an import-time side effect in a module the gate already imports, would require the framework to name EXL3.
    - The alternative of registering from the quantization method's own module would run too late, since that code loads with the model, and would import torch-heavy code during server-args processing.
    - Only a missing plugin module counts as "not registered". A plugin that fails on its own imports raises.
36. **`eager_expert_stream_requirements` lives in the framework.** It encodes the owner's EXL3 set: eager only, no graph gather, the recorder under dynamic residency. It adds one refusal that EXL3 already makes at model load, the host arena, so a misconfigured launch fails before any model loads. It deliberately requires neither `--max-running-requests 1`, nor the overlap setting, nor `flashinfer_cutlass`.
37. **Env reads.** No new env var is added. `_check_nvfp4` keeps the raw `os.environ.get("SGLANG_MOE_EXPERT_STREAM")` read it had in `memory_hook.py`: that knob has never been in `Envs`, and moving it there would be an unrelated rename. The EXL3 plugin reads its own knob through `envs`.
38. **C1: the manager tolerates a format without `pinned_tier_options`, with one warning per format; the protocol still requires the hook.** A missing optional hook should not kill a startup after the owner has taken production down. The warning keeps the omission visible, since it silently drops any `is_pinned` filter. A CPU test also drives `ExpertPinnedHostCacheManager.from_model` end to end over two spec-only fake layers, with the device injected through `pinned_tier_options`.
39. **I1: clamp instead of raise, opted in by `inclusive_pinned_tier`.**
    - A seeded selection can legitimately give a skewed layer more hot slots than its tier can hold beside a gather. Raising would stop a launch over a budget split the owner never chose.
    - Clamping inside the selection loop gives the freed budget to other layers, and every clamped layer is logged.
    - NVFP4's limits are all None, so its selection loop is unchanged. `TestInclusiveHotSelection` tests the clamp on CPU with a fake `ExpertHotCache`, and fails if the clamp is removed.

## Follow-ups (not in this plan)

- **EXL3 itself** (the later plan): `Exl3ExpertFormat` with its segment map, `Exl3ShardRowSource` (Path S), the streaming `Exl3MoEMethod`, the offline repack tool (Path R), and the split-vs-repack benchmark (research T3).
- **T12:** async `submit`/`ReadTicket` on a dedicated reader thread, which needs its own thread-owned `UringFileReader`, and FREE/LOADING/READY slot states in the RAM tier. First check whether the TVM-FFI `read` releases the GIL.
- **T13:** the `BLOB` host layout, as a second RAM-tier class with a segmented H2D kernel. Build it only if T3's measurement shows the split matters.
- `_pinned_staging_buffer` still uses `pin_memory=True`, which rounds to a power of two. It is bounded by `max(rows, 64)` rows per name. For EXL3 chunks of 64 × 13.3 MB this is about 1 GiB pinned per staging set; move it to `allocate_host_slab` if that matters.
- Generalising `NVFP4_TRANSFER_TENSOR_COUNT` (six pairs) for formats with another tensor count to use async promotions.
- When `_load_reserved_in_chunks` cannot drain the device after a failed wait, the layer's hot cache stays wedged with a promotion in flight until the process restarts. Revisit this if T12 makes promotions asynchronous for spec-only formats.
- An inclusive `is_pinned` (VRAM residents pinned in RAM) needs the pinned tier's rows per layer to exceed the hot cache's slots. With `inclusive_pinned_tier = True` (Task 9), startup clamps each layer's hot slots to `pinned rows - max_gather_rows`, so the static selection always fits. Dynamic residency never grows a cache past its startup capacity. A layer whose pinned budget is below `max_gather_rows` gets no hot slots; the log line shows it.

## Self-review

**Spec coverage** (brief scope → task):
- T1: specs, `ExpertFormat`, `DenseLayerFormat`, `expert_streamer_of` / `iter_expert_streamers` (attribute kept), `file_source_bytes_per_expert` → Task 1.
- T2: `ExpertRowSource`, `RowReadStats`, `ReadTicket` with a synchronous default `submit`, `HostSlotLayout`, `ExpertFileRowReader` conformance and `from_group`, `TensorRowSource` → Task 2. `ExpertStreamer(row_source=…)` and one batched read in `_read_host_rows` / `_copy_source_rows` → Task 3.
- T6 → Task 5:
  - specs instead of probes;
  - registered page-aligned exact slabs;
  - the O(1) LRU;
  - the `is_pinned` filter;
  - the `_gather_cached` out-of-bounds read, re-verified present after `844bb9d7a5` and fixed in that function.
- T7: spec-only reads → Task 3. Pinned-tier chunked `_prepare_promotion` / `_load_reserved`, arena refusal, and the graph-flag startup guard → Task 6.
- T8: `gather_experts` plus the `max_gather_rows` cap on the `844bb9d7a5` floor → Task 7.
- T10 (generic part): `SGLANG_MOE_EXPERT_ROW_SOURCE` with `auto|files|tensor` and format-defined kinds; unknown kinds raise → Task 3.
- T11 (generic part): the verify-only open → Task 8.
- Stats: trailing defaulted `ExpertGatherStats` fields, summed into the manager counters → Task 4.
- Server-args gate per format (C2), tolerant tier options (C1) and the inclusive hot-slot clamp (I1) → Task 9.
- Excluded, and listed only under Follow-ups: the EXL3 format, shard source, repack and benchmarks; T12; T13.

**Production-path pins.** Each task that edits a production path has a test pinning the old behaviour:
- Task 1: spec shapes equal `source.shape[1:]` on NVFP4 shapes; legacy byte formulas; dynamic sources; the unchanged error messages; staging shapes; the CUDA arena-rebind detection.
- Task 3: the readable branches keep their code verbatim; the existing uring and route-plan tests; the single-call routing.
- Task 4: a no-read gather keeps its exact stats object; the positional field order.
- Task 5: LRU equivalence against a copy of the legacy algorithm on 500 random steps; the legacy pinned counts; the pinned-only gather's counts; `pinned_tier_options` is `{}` for the dense format.
- Task 6: dense formats report `has_spec_only_tensors=False`, and the new branches key on it.
- Task 7: the default floor is still `num_experts`; `_plan_eager_routes` still records by default.
- Task 9: every existing hot-cache gate case runs unchanged and again under five NVFP4-streaming method names; the NVFP4 check is moved verbatim; dense formats have no hot-slot limit.
- Task 10 reruns every existing expert suite on the GPU.

**Frozen file.** `model_runner.py` gets only the two discovery swaps (Task 1, Step 15), after reading `large-class-style`.

**Dry run.** The plan's code was dry-run on 2026-09-18, and again after each review round.
- A script applied every create, old→new edit, insertion and method replacement, in task order, to the files of `e54c84a7c6`. Every old snippet matched exactly once, and every resulting file byte-compiles.
- On divix01, over a scratch `git archive` that has since been deleted, the new CPU tests plus the route-plan, file-reader, graph-gather-scratch, offload and file-cache suites passed at every stage from Task 1 through Task 9. The totals were 110, 122, 133, 138, 161, 176, 188, 196 and (Task 9, which adds `test_expert_stream_requirements.py`, `test_expert_tier_startup.py` and five keyed gate classes) 317 passed.
- A mutation that fixes the chunk size at the start of a call, in both `gather_rows` and `_load_reserved_in_chunks`, makes all four growing-protection tests fail.
- Every red commit failed exactly as its step states.
- The final tree's `CPU_SUITE` result was `1 failed, 270 passed, 268 skipped`: the base's `1 failed, 170 passed, 268 skipped`, plus Task 9's 100 keyed gate tests.
- Two mutations were checked. Dropping `modelopt_fp4` from `NVFP4_QUANT_METHODS` fails `ModelOptFp4HotCacheConfigurationTests`, and removing the clamp fails `TestInclusiveHotSelection`.
- The CUDA classes have not run yet; that is Task 10.

**Placeholders.** None. Every code step has complete code or an exact old→new snippet. Every run step has the command and the expected result.

**Name consistency.** These names are defined and used consistently across tasks:
- Task 1: `specs`, `spec()`, `source()`, `file_source_bytes_per_expert`.
- Task 3: `row_source` with its `file_row_reader` alias; `read_host_rows`; `_DEFAULT_ROW_SOURCE`, which Task 6's tests use.
- Task 4: `_gather_eager_rows`, which Tasks 6 and 7 use.
- Task 5: `gather_rows`, `PinnedGatherResult`, `evictable_rows`, `is_pinned`, `pinned_tier_options`; Task 6 uses `evictable_rows`.
- Task 6: `has_spec_only_tensors`.
- Task 7: `_staging_floor_rows`.
- The `ExpertGatherStats` field names `host_read_*` match the `_OperationalCounters` fields and `_add_host_read_counters`.
