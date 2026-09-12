# NVFP4 File-Backed Expert Hot Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Qwen3.8 ModelOpt NVFP4 expert corpus's anonymous host allocation with verified reusable file mappings, then add a fixed-budget GPU hot-expert tier seeded and updated from SGLang's existing expert-distribution recorder.

**Architecture:** A generic verified file-tensor cache owns sparse mappings, locking, identity validation, and atomic manifests. The ModelOpt loader writes checkpoint data into per-layer mappings once, converts those mappings to the final CUTLASS runtime layout, and skips both large checkpoint copies and transforms on a verified warm hit. The existing `ExpertStreamer` remains the sole runner-facing gather path. A fixed-slot GPU cache accelerates hot rows; mixed hits and file misses are reconstructed into the existing compact staging tensors. SGLang's recorder supplies per-layer routing counts through a forward observer, while the cache adds only residency and byte-transfer counters.

**Tech Stack:** Python 3.13, PyTorch, Triton, SGLang ModelOpt NVFP4 loader, FlashInfer CUTLASS MoE, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-file-backed-expert-cache-design.md`

## Global Constraints

- Preserve the user's unstaged edit in `python/sglang/srt/models/qwen4_exp.py`; never stage or overwrite it.
- Do not stop, restart, or replace the running `cc-nvfp4-stream` server without fresh user confirmation immediately before that action.
- Keep the feature opt-in behind `SGLANG_MOE_EXPERT_STREAM=1` and the new cache environment settings.
- Retain the existing anonymous expert-streaming path when no expert file directory is configured.
- Support only the already-validated prototype envelope: ModelOpt NVFP4, `flashinfer_cutlass`, TP=1, EP=1, `moe-a2a=none`, one running request, overlap scheduling disabled, and CUDA graph capture disabled for routed MoE execution.
- Cache the final runtime layout. A warm hit must not deinterleave or swizzle cached values again.
- Keep `ModelOptNvFp4FusedMoEMethod.apply()` and the FlashInfer CUTLASS quant-info contract unchanged.
- Start with `SGLANG_MOE_HOT_GPU_MB=4096`. The 31.4-GiB GPU must retain headroom for approximately 1.22 GiB of transient compact expert staging and the configured KV/Mamba pools.
- Use focused unit and restart-lifecycle tests. Do not run the production server during implementation tasks.
- Defer transfer overlap, `O_DIRECT`/`io_uring`, and CUDA VMM to later measured phases.

---

## File Structure

### New files

- `python/sglang/srt/model_loader/file_tensor_cache.py` — model-neutral sparse-file mapping, group identity, lock, manifest validation, completion, and abort lifecycle.
- `python/sglang/srt/layers/moe/expert_hot_cache.py` — seed normalization, fixed-slot hot tiers, placement policy, dynamic update gate, and operational statistics.
- `test/registered/unit/model_loader/test_file_tensor_cache.py` — generic cache lifecycle tests.
- `test/registered/unit/layers/moe/test_expert_hot_cache.py` — hot-tier placement, mixed gather, dynamic policy, and accounting tests.
- `docs/superpowers/results/2026-09-12-nvfp4-file-hot-cache-results.md` — cold/warm startup and runtime A/B record after implementation.

### Modified files

- `python/sglang/srt/models/qwen4_exp_ple_table.py` — delegate verified persistence to the generic cache while retaining the PLE API, row prefetcher, RSS trimmer, and staging behavior.
- `python/sglang/srt/model_executor/model_runner_components/load_model_utils.py` — build one stable checkpoint identity for PLE and experts and abort incomplete caches on load failure.
- `python/sglang/srt/model_executor/model_runner.py` — pass cache identity into the offloader and initialize the hot-cache manager before KV/cache sizing.
- `python/sglang/srt/utils/offloader.py` — bind ModelOpt expert parameters to file mappings, wrap large parameter loaders, track exact shard coverage, and publish/abort cache groups.
- `python/sglang/srt/layers/quantization/modelopt_quant.py` — finalize cold mappings into runtime layout and take the transform-free warm-hit path.
- `python/sglang/srt/layers/moe/expert_stream.py` — resolve hot hits and file misses while preserving the existing compact gather result.
- `python/sglang/srt/eplb/expert_distribution.py` — expose already-collected per-forward counts to registered observers without a second top-k counter.
- `python/sglang/srt/environ.py` — declare prototype cache controls.
- `python/sglang/srt/server_args.py` — reject unsupported cache/runtime combinations early.
- Existing focused tests under `test/registered/unit/models/`, `test/registered/unit/layers/quantization/`, and `test/registered/unit/` — regression coverage for PLE, ModelOpt, offloader, and configuration behavior.

---

### Task 1: Extract a model-neutral verified file-tensor cache

**Files:**
- Create: `python/sglang/srt/model_loader/file_tensor_cache.py`
- Create: `test/registered/unit/model_loader/test_file_tensor_cache.py`
- Reference: `python/sglang/srt/models/qwen4_exp_ple_table.py`

- [ ] **Step 1: Write failing lifecycle tests**

Cover one-tensor and grouped caches:

```python
def test_completed_group_reopens_as_hit_and_preserves_bytes(tmp_path):
    specs = [FileTensorSpec("weight", (4, 8), (8, 1), torch.uint8)]
    cold = FileTensorCacheGroup.open(tmp_path, "expert", {"commit": "abc"}, specs)
    cold.tensors["weight"].fill_(17)
    cold.complete()

    warm = FileTensorCacheGroup.open(tmp_path, "expert", {"commit": "abc"}, specs)
    assert warm.cache_hit
    assert torch.equal(warm.tensors["weight"], torch.full((4, 8), 17, dtype=torch.uint8))
    warm.close()


def test_group_is_all_miss_when_one_member_is_invalid(tmp_path):
    specs = [
        FileTensorSpec("weight", (4, 8), (8, 1), torch.uint8),
        FileTensorSpec("scale", (4, 2), (2, 1), torch.float8_e4m3fn),
    ]
    cache = FileTensorCacheGroup.open(tmp_path, "expert", {"commit": "abc"}, specs)
    cache.complete()
    os.truncate(cache.paths["scale"], 1)
    reopened = FileTensorCacheGroup.open(tmp_path, "expert", {"commit": "abc"}, specs)
    assert not reopened.cache_hit
```

Also cover missing manifest, malformed JSON, format-version mismatch, identity mismatch, shape/stride/dtype/nbytes mismatch, incomplete group, idempotent close, abort without manifest, and atomic completion leaving no temporary manifest.

- [ ] **Step 2: Run the new test file and confirm import/API failures**

Run:

```bash
cd /data/models/slang/sglang
.venv/bin/python -m pytest test/registered/unit/model_loader/test_file_tensor_cache.py -q
```

Expected: collection fails because `file_tensor_cache.py` and its public types do not exist.

- [ ] **Step 3: Implement the generic cache API**

Provide these concrete interfaces:

```python
@dataclass(frozen=True)
class FileTensorSpec:
    tag: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int: ...


class FileTensorCacheGroup:
    @classmethod
    def open(
        cls,
        directory: str | os.PathLike[str],
        namespace: str,
        cache_identity: Mapping[str, Any],
        specs: Sequence[FileTensorSpec],
    ) -> "FileTensorCacheGroup": ...

    cache_hit: bool
    tensors: dict[str, torch.Tensor]
    paths: dict[str, str]

    def complete(self) -> None: ...
    def abort(self) -> None: ...
    def close(self) -> None: ...
```

Use SGLang's existing `get_lock()` helper. Hash a canonical JSON payload containing format version, namespace, cache identity, and every tensor spec. Open one lock per group. A verified hit requires an exact manifest and exact file sizes for every member. On a miss, discard the old manifest, resize every sparse file, map each file with `torch.from_file(..., shared=True)`, and expose the requested strided views. `complete()` must fsync all data files, atomically replace and fsync the manifest, then release the lock. `abort()` must leave no valid manifest and release the lock.

- [ ] **Step 4: Run the focused lifecycle tests**

Expected: all tests in `test_file_tensor_cache.py` pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add python/sglang/srt/model_loader/file_tensor_cache.py test/registered/unit/model_loader/test_file_tensor_cache.py
git commit -m "feat: add verified file tensor cache"
```

---

### Task 2: Move PLE persistence onto the generic utility without changing behavior

**Files:**
- Modify: `python/sglang/srt/models/qwen4_exp_ple_table.py`
- Modify: `test/registered/unit/models/test_qwen4_exp_ple_table.py`
- Modify: `test/registered/unit/models/test_qwen4_exp_ple_cache_integration.py`

- [ ] **Step 1: Add compatibility tests around the existing PLE API**

Assert that `allocate_ple_host_table(...)`, `get_ple_file_cache(...)`, and `complete_ple_file_cache(...)` still provide cold miss, durable completion, warm hit, identity invalidation, and preserved bytes. Keep the existing PLE prefetcher, RSS trimmer, and discrete-GPU stager tests unchanged.

- [ ] **Step 2: Run both PLE test files before the refactor**

```bash
.venv/bin/python -m pytest \
  test/registered/unit/models/test_qwen4_exp_ple_table.py \
  test/registered/unit/models/test_qwen4_exp_ple_cache_integration.py -q
```

Expected: current tests pass; the new compatibility assertion fails until the generic cache handle is adapted.

- [ ] **Step 3: Replace the PLE-specific manifest implementation with an adapter**

Keep PLE's public names stable. Implement `PleFileCache` as a narrow adapter around a one-member `FileTensorCacheGroup`:

```python
class PleFileCache:
    def __init__(self, group: FileTensorCacheGroup) -> None:
        self._group = group
        self.cache_hit = group.cache_hit
        self.path = group.paths["ple_table"]

    def complete(self) -> None:
        self._group.complete()

    def close(self) -> None:
        self._group.close()
```

Leave `PleFilePrefetcher`, `PleFileRssTrimmer`, host-access detection, and row staging in `qwen4_exp_ple_table.py`. Delete only duplicated hashing, validation, lock, sparse-file, and manifest helpers.

- [ ] **Step 4: Run the generic and PLE cache tests together**

Expected: all focused cache tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add python/sglang/srt/models/qwen4_exp_ple_table.py test/registered/unit/models/test_qwen4_exp_ple_table.py test/registered/unit/models/test_qwen4_exp_ple_cache_integration.py
git commit -m "refactor: share verified file cache with PLE"
```

---

### Task 3: Supply a stable checkpoint identity to expert storage

**Files:**
- Modify: `python/sglang/srt/model_executor/model_runner_components/load_model_utils.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py`
- Modify: `python/sglang/srt/utils/offloader.py`
- Modify: `test/registered/unit/models/test_qwen4_exp_ple_cache_integration.py`
- Modify: `test/registered/unit/test_nvfp4_expert_offload.py`

- [ ] **Step 1: Write failing identity-plumbing tests**

Rename `_qwen4_exp_ple_cache_identity` to `_checkpoint_cache_identity` and test that it contains the canonical model path, requested revision, and Hugging Face `_commit_hash`. Assert that `create_offloader_from_server_args(..., model_config=...)` receives the same immutable mapping used by PLE.

- [ ] **Step 2: Run the two focused test files and observe signature failures**

- [ ] **Step 3: Generalize identity creation and offloader configuration**

Use this signature:

```python
def create_offloader_from_server_args(
    server_args: ServerArgs,
    dp_rank: int,
    model_config: ModelConfig | None = None,
) -> BaseOffloader:
```

Store a copied checkpoint identity in `OffloaderV1`. Pass `self.model_config` from `ModelRunner.__init__`. Preserve all non-streaming callers by keeping the parameter optional. Use the generalized identity for the PLE text config in `load_model_with_memory_saver()`.

- [ ] **Step 4: Run the identity and offloader tests**

Expected: identical identity inputs produce stable cache keys; path, revision, or commit changes produce different keys.

- [ ] **Step 5: Commit Task 3**

```bash
git add python/sglang/srt/model_executor/model_runner_components/load_model_utils.py python/sglang/srt/model_executor/model_runner.py python/sglang/srt/utils/offloader.py test/registered/unit/models/test_qwen4_exp_ple_cache_integration.py test/registered/unit/test_nvfp4_expert_offload.py
git commit -m "refactor: share checkpoint cache identity"
```

---

### Task 4: Bind cold expert parameters to per-layer file mappings and verify loader coverage

**Files:**
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/utils/offloader.py`
- Modify: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- Modify: `python/sglang/srt/model_executor/model_runner_components/load_model_utils.py`
- Modify: `test/registered/unit/test_nvfp4_expert_offload.py`

- [ ] **Step 1: Write failing file-backed offloader tests**

Cover these requirements:

- only `w13_weight`, `w2_weight`, `w13_weight_scale`, and `w2_weight_scale` are initially file-backed;
- the offloader does not copy uninitialized CUDA contents into a new mapping;
- tiny input/per-tensor scales remain on the normal device/loading path;
- a cold loader call delegates to the original loader and records its `(tensor, expert_id, shard_id)` coverage;
- a warm hit skips only the four large loader targets;
- W13 requires every expert's `w1` and `w3` shards, while W2 requires every expert's `w2` shard;
- incomplete coverage makes `post_init()` abort and raise without publishing a manifest;
- load or postprocess exceptions call `abort()` and release every cache lock.

- [ ] **Step 2: Run the offloader tests and confirm the new cases fail**

- [ ] **Step 3: Add expert file-cache configuration**

Declare `SGLANG_MOE_EXPERT_FILE_DIR` in `environ.py`. Activate file backing only when expert streaming is enabled and the directory is nonempty. Continue using anonymous pageable CPU tensors otherwise.

- [ ] **Step 4: Add per-layer cache binding to `OffloaderV1`**

Change `_iter_streamed_nvfp4_parameters()` to yield `(submodule, parameter_name, parameter)` and derive the stable layer id from `submodule.quant_method.moe_runner_config.layer_id`. Create one four-member `FileTensorCacheGroup` per layer. Allocate mappings in checkpoint-loader shapes, replace `parameter.data` directly, and mark the parameter with:

```python
parameter._sglang_skip_device_loading = True
parameter._sglang_file_cache_hit = cache_group.cache_hit
parameter._sglang_file_cache_group = cache_group
parameter._sglang_file_cache_tag = parameter_name
```

Do not call `cpu_data.copy_(parameter.data)` in file mode.

- [ ] **Step 5: Wrap the existing fused-MoE weight loader**

At the top of `_weight_loader_impl()` or the common entry shared by physical/fused loader paths, inspect the destination parameter attributes. On a hit, return before the large copy. On a miss, call the existing loader unchanged, then report the loaded logical shard to `OffloaderV1.record_expert_shard(...)`. Keep this hook generic enough that all existing stacked `w1`/`w3`/`w2` name mappings still pass through the original implementation.

- [ ] **Step 6: Publish or abort groups at the existing lifecycle boundaries**

`OffloaderV1.post_init()` validates exact per-layer coverage only after `model.load_weights()` and all quantization postprocessing have completed. It then completes every cache group. Add `BaseOffloader.abort()` and invoke it from the exception path around model load/postprocessing in `load_model_with_memory_saver()`.

- [ ] **Step 7: Run offloader tests**

Expected: all prior anonymous-offload tests and new file-backed lifecycle tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/utils/offloader.py python/sglang/srt/layers/moe/fused_moe_triton/layer.py python/sglang/srt/model_executor/model_runner_components/load_model_utils.py test/registered/unit/test_nvfp4_expert_offload.py
git commit -m "feat: file-back ModelOpt NVFP4 experts"
```

---

### Task 5: Persist and reuse the final CUTLASS runtime layout

**Files:**
- Modify: `python/sglang/srt/layers/quantization/modelopt_quant.py`
- Modify: `python/sglang/srt/utils/offloader.py`
- Modify: `test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py`
- Modify: `test/registered/unit/test_nvfp4_expert_offload.py`

- [ ] **Step 1: Write RED tests for final-layout correctness**

Construct deterministic small tensors and prove:

- cold file-backed postprocessing matches anonymous-source deinterleave and swizzle byte-for-byte;
- the mapping contains final `w13_weight`, `w2_weight`, `w13_blockscale_swizzled`, and `w2_blockscale_swizzled` bytes after postprocessing;
- a warm hit does not call `deinterleave_w13()` or `swizzle_blockscale()`;
- cold and warm `ExpertStreamer.gather()` payloads match for fixed routed IDs;
- tiny scales still load and `g1_alphas`/`g2_alphas` are recomputed on warm startup.

- [ ] **Step 2: Run the ModelOpt tests and confirm double-transform failures**

- [ ] **Step 3: Remove redundant derived placeholders in streaming mode**

When expert streaming is enabled, set `layer.w13_blockscale_swizzled` and `layer.w2_blockscale_swizzled` to `None` during `create_weights()`. This avoids allocating the current redundant 7.2-GiB placeholders across 48 layers.

- [ ] **Step 4: Add explicit cold and warm postprocess branches**

Cold path:

1. Load raw checkpoint values into mapped loader views.
2. Run the existing W13 deinterleave.
3. Copy the deinterleaved weight and scale bytes back into their respective mapped storage when the transform returned a new allocation.
4. Run CPU blockscale swizzle.
5. Copy swizzled scale bytes back into the same scale-file storage and bind the parameter plus `*_blockscale_swizzled` alias to the final swizzled shape.
6. Record final tensor shape/stride metadata on the cache group before completion.

Warm path:

1. Skip the four checkpoint copies through Task 4's loader hook.
2. Bind cached storage using final manifest shape/stride metadata.
3. Set `layer._w13_deinterleaved = True`.
4. Alias `w13_blockscale_swizzled`/`w2_blockscale_swizzled` to cached scale mappings.
5. Skip deinterleave and swizzle while retaining the existing tiny-scale and alpha calculations.

Reject a cache if the final layout's byte length differs from its loader layout. The supported Qwen checkpoint has equal byte lengths; do not silently allocate anonymous replacements.

- [ ] **Step 5: Run ModelOpt and offloader tests together**

Expected: cold, warm, and anonymous gathered payloads are identical; no double transformation occurs.

- [ ] **Step 6: Commit Task 5**

```bash
git add python/sglang/srt/layers/quantization/modelopt_quant.py python/sglang/srt/utils/offloader.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py test/registered/unit/test_nvfp4_expert_offload.py
git commit -m "feat: reuse final NVFP4 expert layout"
```

---

### Task 6: Add a fixed-slot GPU hot cache to the existing gather path

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_hot_cache.py`
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`
- Modify: `test/registered/unit/layers/moe/test_expert_stream.py`
- Create: `test/registered/unit/layers/moe/test_expert_hot_cache.py`

- [ ] **Step 1: Write failing fixed-slot and mixed-gather tests**

Cover byte-based capacity, fixed pointer stability across reassignment, unchanged-slot reuse, all-hot direct slot remapping, mixed hot/file reconstruction, duplicate route IDs, prefill deduplication, migration bytes, per-tier hit/miss bytes, and zero-budget fallback.

Representative contract:

```python
cache = ExpertHotCache(streamer, capacity=2)
cache.reassign([3, 7])
ptrs = cache.data_ptrs()

compact_ids, tensors = streamer.gather(torch.tensor([[7, 1]], device="cuda"))
assert torch.equal(tensors["w13_weight"][compact_ids], reference_w13[[7, 1]].cuda())

cache.reassign([3, 9])
assert cache.data_ptrs() == ptrs
```

- [ ] **Step 2: Run the two expert-stream test files and observe missing APIs**

- [ ] **Step 3: Implement `ExpertHotCache` with fixed CUDA slots**

Expose:

```python
@dataclass
class HotCacheUpdateStats:
    promoted_experts: int
    evicted_experts: int
    migration_bytes: int


class ExpertHotCache:
    def __init__(self, streamer: ExpertStreamer, capacity: int): ...
    def reassign(self, expert_ids: Sequence[int]) -> HotCacheUpdateStats: ...
    def lookup(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def data_ptrs(self) -> tuple[int, ...]: ...
```

Allocate every runtime tensor's slot array once. Maintain a CUDA `expert_to_slot` table and CPU `slot_to_expert` metadata. Fill only changed slots by reusing the streamer's existing pageable-to-pinned-to-CUDA helper. Do not allocate replacement slot tensors during updates.

- [ ] **Step 4: Extend `ExpertStreamer.gather()` without changing its return type**

- All-hit path: return hot slot IDs and the fixed slot tensors directly, avoiding compact D2D assembly.
- Mixed path: preserve the current compact ID ordering; copy hot rows D2D into `_STAGING`, gather misses through the existing pinned bounce buffers, and return one complete compact tensor dictionary.
- No-cache path: execute the current implementation byte-for-byte.

Keep `g1_alphas` and `g2_alphas` in the slot set because the CUTLASS quant-info object indexes them with the same compact IDs, even though their source tensors are small.

- [ ] **Step 5: Run expert-stream and hot-cache tests**

Expected: cached and uncached outputs match for all-hot, all-cold, mixed, duplicate, and deduplicated routes.

- [ ] **Step 6: Commit Task 6**

```bash
git add python/sglang/srt/layers/moe/expert_hot_cache.py python/sglang/srt/layers/moe/expert_stream.py test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_hot_cache.py
git commit -m "feat: add fixed-slot NVFP4 expert cache"
```

---

### Task 7: Reuse SGLang's recorder for frequency seeding and dynamic placement

**Files:**
- Modify: `python/sglang/srt/eplb/expert_distribution.py`
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py`
- Modify: `test/registered/unit/layers/moe/test_expert_hot_cache.py`
- Modify: `test/registered/eplb/test_expert_distribution.py`

- [ ] **Step 1: Write failing recorder-observer and policy tests**

Test that one call to the existing `on_select_experts()` produces one shared `global_physical_count` matrix consumed by both the normal accumulator and the cache observer. Verify no second `scatter_add_` or parallel top-k counter is introduced.

Test seed normalization for:

- SGLang stat dump: `{"logical_count": [steps, layers, experts]}`;
- Haberstroh artifact: `{"count": [layers, experts], "mass": [layers, experts], "tokens": int}` using `count` preferentially.

Test dynamic rules: only a qualifying non-speculative extend/prefill triggers; decode, target verification, and small prefill do not; minimum residence interval is honored; reassignment occurs only when estimated source bytes saved exceed migration bytes times the configured benefit ratio.

- [ ] **Step 2: Run the focused recorder and cache-policy tests and confirm failures**

- [ ] **Step 3: Add a forward-observer API to the existing recorder**

Add `register_forward_observer(callback)` to `ExpertDistributionRecorder`. In the real recorder, make collection active when either normal recording or an observer is active. At `_on_forward_pass_end`, call `gatherer.collect()` once, append it to the normal accumulator only when user recording is enabled, and pass the same collected dictionary plus `ForwardBatch` to observers. The no-op recorder keeps a harmless empty implementation.

This preserves SGLang's router hook and counting kernel as the single source of truth and does not reset or consume user-requested recorder output.

- [ ] **Step 4: Implement seed loading and `ExpertHotCacheManager`**

Expose:

```python
class ExpertHotCacheManager:
    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        budget_bytes: int,
        seed_path: str | None,
        dynamic: bool,
        update_prefill_tokens: int,
        min_residence_forwards: int,
        benefit_ratio: float,
    ) -> "ExpertHotCacheManager | None": ...

    def on_expert_distribution(
        self,
        forward_batch: ForwardBatch,
        single_pass_data: Mapping[str, Any],
    ) -> None: ...
```

Discover modules that already own `_nvfp4_expert_streamer`. Compute exact per-layer bytes per slot from all six runtime tensors. Allocate capacity under one global byte budget, initially distributing complete slots across layers in descending expected byte savings. Install a static seed before serving. For a qualifying prefill, rank per-layer experts by the recorder's actual counts, calculate delta migrations and predicted source bytes saved, and mutate fixed slots after the forward has completed.

- [ ] **Step 5: Add structured operational counters**

Track and periodically log JSON-compatible totals for hot hits, file misses, D2D bytes, H2D bytes, file-source bytes, requested unique experts, promotions, evictions, migration bytes, and residency bytes, split by prefill/decode and layer. Do not duplicate the recorder's routing histogram.

- [ ] **Step 6: Run recorder and hot-cache tests**

Expected: the observer sees the recorder's single collected matrix and dynamic updates obey all gates.

- [ ] **Step 7: Commit Task 7**

```bash
git add python/sglang/srt/eplb/expert_distribution.py python/sglang/srt/layers/moe/expert_hot_cache.py test/registered/unit/layers/moe/test_expert_hot_cache.py test/registered/eplb/test_expert_distribution.py
git commit -m "feat: drive expert cache from routing stats"
```

---

### Task 8: Wire configuration, startup allocation, and compatibility checks

**Files:**
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/server_args.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py`
- Modify: `python/sglang/srt/layers/quantization/modelopt_quant.py`
- Modify: `test/registered/unit/test_nvfp4_expert_offload.py`
- Modify: `test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py`

- [ ] **Step 1: Write failing configuration and startup tests**

Validate these environment controls and defaults:

```text
SGLANG_MOE_EXPERT_FILE_DIR=""          disabled
SGLANG_MOE_HOT_GPU_MB=0                disabled
SGLANG_MOE_HOT_SEED=""                 no static seed
SGLANG_MOE_HOT_DYNAMIC=0               disabled
SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS=1024
SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS=8
SGLANG_MOE_HOT_BENEFIT_RATIO=1.0
SGLANG_MOE_HOT_LOG_INTERVAL=100
```

Reject hot caching without expert streaming, dynamic caching without recorder mode `stat`, unsupported backend/TP/EP/A2A, overlap scheduling, more than one running request, or routed-MoE CUDA graph capture. Verify zero budgets leave existing behavior unchanged.

- [ ] **Step 2: Run configuration tests and confirm the new invalid cases are not rejected yet**

- [ ] **Step 3: Initialize the manager before runtime pool sizing**

After `self.load_model()` and `prepare_moe_topk()`, but before KV/cache pool and graph initialization, call `ExpertHotCacheManager.from_model(...)`. Store the manager on `ModelRunner` and register its callback with `get_global_expert_distribution_recorder()` when dynamic mode is enabled. Pass the layer id into each `ExpertStreamer` from `ModelOptNvFp4FusedMoEMethod._attach_expert_streamer()`.

- [ ] **Step 4: Add early compatibility validation**

Extend `_handle_offload_compatibility()` in `server_args.py` to produce direct error messages for every unsupported combination in Step 1. Keep current ModelOpt streaming guards in `_attach_expert_streamer()` as a second line of defense.

- [ ] **Step 5: Run the focused startup/configuration tests**

Expected: valid prototype configuration constructs the manager before memory-pool sizing; invalid combinations fail before loading weights.

- [ ] **Step 6: Commit Task 8**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/server_args.py python/sglang/srt/model_executor/model_runner.py python/sglang/srt/layers/quantization/modelopt_quant.py test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py
git commit -m "feat: wire NVFP4 hot expert residency"
```

---

### Task 9: Focused verification and cold/warm runtime A/B

**Files:**
- Create: `docs/superpowers/results/2026-09-12-nvfp4-file-hot-cache-results.md`
- Verify only: all implementation files from Tasks 1–8

- [ ] **Step 1: Check the patch for accidental placeholders and user-file overlap**

```bash
git diff --check
git status --short
rg -n "TODO|TBD|test\.skip|test\.only|NotImplementedError" \
  python/sglang/srt/model_loader/file_tensor_cache.py \
  python/sglang/srt/layers/moe/expert_hot_cache.py
```

Expected: no whitespace errors or implementation placeholders; `qwen4_exp.py` remains unstaged and unmodified by this work.

- [ ] **Step 2: Run the focused unit suite**

```bash
.venv/bin/python -m pytest \
  test/registered/unit/model_loader/test_file_tensor_cache.py \
  test/registered/unit/models/test_qwen4_exp_ple_table.py \
  test/registered/unit/models/test_qwen4_exp_ple_cache_integration.py \
  test/registered/unit/test_nvfp4_expert_offload.py \
  test/registered/unit/layers/moe/test_expert_stream.py \
  test/registered/unit/layers/moe/test_expert_hot_cache.py \
  test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py \
  test/registered/eplb/test_expert_distribution.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Ask for fresh confirmation before replacing the running server**

Do not proceed to runtime validation until the user explicitly confirms that `cc-nvfp4-stream` may be stopped/restarted.

- [ ] **Step 4: Run a cold file-cache build with no GPU hot tier**

Use a persistent directory under `/data/models/slang/nvfp4-work/`, not `/tmp`. Set `SGLANG_MOE_EXPERT_FILE_DIR`, keep `SGLANG_MOE_HOT_GPU_MB=0`, retain the known-good model/backend arguments, and write a new timestamped log under `/data/models/slang/nvfp4-stream-logs/`. Record startup time, mapped bytes, anonymous RSS, file RSS, swap activity, and manifest publication.

- [ ] **Step 5: Restart against the completed cache and verify the warm hit**

Confirm that all 48 layer groups report verified hits, large checkpoint copies and transforms are skipped, anonymous expert memory falls by at least 60 GiB, and fixed-route smoke outputs match the cold process.

- [ ] **Step 6: Run static 4-GiB hot-cache A/B**

Set `SGLANG_MOE_HOT_GPU_MB=4096` and seed from the existing `expert_freq.pt`. Record cold and warm prefill separately plus warm decode. Include cache coverage, hit/miss bytes, migration bytes, GPU residency, peak VRAM, and throughput. Compare against the known 5.5–5.8 tok/s prefill and 2.8–3.0 tok/s warm-decode baseline.

- [ ] **Step 7: Run dynamic-after-prefill A/B**

Enable `SGLANG_MOE_HOT_DYNAMIC=1` and `--expert-distribution-recorder-mode stat` with the same 4-GiB budget. Run the same prompt/workload. Retain dynamic placement only if source-to-GPU bytes avoided exceeds migration bytes and measured decode latency improves versus static placement.

- [ ] **Step 8: Record results and the next measured decision**

Populate the result document with commands, commit, hardware, cache identity, cold/warm timings, memory readings, throughput, counters, failures, and a decision among:

- keep synchronous mmap plus GPU hot tier;
- add transfer overlap because H2D/wait dominates;
- prototype `io_uring`/`O_DIRECT` because file faults/read latency dominates;
- prototype VMM because compact D2D assembly dominates.

- [ ] **Step 9: Commit the verified result record**

```bash
git add docs/superpowers/results/2026-09-12-nvfp4-file-hot-cache-results.md
git commit -m "docs: record NVFP4 expert cache results"
```
