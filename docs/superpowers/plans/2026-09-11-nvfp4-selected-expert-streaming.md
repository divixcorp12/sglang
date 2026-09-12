# NVFP4 Selected-Expert Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run NVIDIA Qwen3.8-Flash-Next ModelOpt NVFP4 on one 32 GiB Blackwell GPU by keeping decoder routed experts in pinned host memory and staging only routed rows for each MoE invocation.

**Architecture:** Extend V1 CPU offload with an opt-in ModelOpt-NVFP4 expert-only mode, preserve CUTLASS-ready expert rows in pinned host storage, and use a reusable Triton UVA gather to form compact GPU expert tables. Remap routed IDs and pass compact tensors to the existing FlashInfer CUTLASS NVFP4 runner without modifying the fused kernel.

**Tech Stack:** Python 3.13, PyTorch, Triton, FlashInfer CUTLASS MoE, SGLang ModelOpt loader, pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-09-11-nvfp4-selected-expert-streaming-design.md`

## Global Constraints

- Feature gate: `SGLANG_MOE_EXPERT_STREAM=1`; unset preserves stock behavior.
- Initial support: CUDA, TP1, EP1, `moe_a2a_backend=none`, `moe_runner_backend=flashinfer_cutlass`, one running request, no overlap scheduling, and no CUDA graphs.
- Stream only decoder `ModelOptNvFp4FusedMoEMethod` routed experts; do not stream PLE, shared experts, or FP8 MTP experts.
- Preserve the user's existing uncommitted PLE meta-allocation change in `python/sglang/srt/models/qwen4_exp.py`.
- Keep all gather and FlashInfer work ordered on the current CUDA stream.
- Make no hot-cache, VMM, prefetch, multi-request, TP, or EP additions in this milestone.

---

### Task 1: Device-preserving NVFP4 block-scale swizzle

**Files:**
- Modify: `python/sglang/srt/layers/quantization/utils.py`
- Create: `test/registered/unit/layers/quantization/test_nvfp4_swizzle.py`

**Interfaces:**
- Produces: `swizzle_blockscale(scale: torch.Tensor, target_device: Optional[torch.device | str] = "cuda") -> torch.Tensor`.
- Preserves: calling `swizzle_blockscale(scale)` returns CUDA storage as before.
- Adds: `target_device="cpu"` returns the same padded/swizzled values in CPU storage.

- [ ] **Step 1: Write failing CPU-layout tests**

Create tests that build FP8 tensors with shapes `[E, M, K]`, including padding and no-padding cases. Compare `target_device="cpu"` with a pure reshape/permute reference and assert CPU device, exact shape, dtype, and values.

```python
def reference_swizzle(scale):
    e, m, k = scale.shape
    mp = (m + 127) // 128 * 128
    kp = (k + 3) // 4 * 4
    padded = torch.zeros((e, mp, kp), dtype=scale.dtype)
    padded[:, :m, :k] = scale
    return padded.reshape(e, mp // 128, 4, 32, kp // 4, 4).permute(
        0, 1, 4, 3, 2, 5
    ).contiguous().reshape(e, mp, kp)
```

- [ ] **Step 2: Run the focused test and confirm the API is absent**

Run: `/data/models/slang/.venv/bin/python -m pytest test/registered/unit/layers/quantization/test_nvfp4_swizzle.py -q`

Expected: failure because `swizzle_blockscale` does not accept `target_device`.

- [ ] **Step 3: Implement device-preserving allocation**

Allocate `padded_scale` on `scale.device`, perform the existing layout transformation there, then move only the result to `target_device`. Resolve `target_device=None` to `scale.device`; retain `"cuda"` as the default for compatibility.

```python
def swizzle_blockscale(scale, target_device="cuda"):
    source_device = scale.device
    padded_scale = torch.zeros(
        (batch, padded_m, padded_k), dtype=scale.dtype, device=source_device
    )
    # existing reshape and permute
    if target_device is None:
        target_device = source_device
    return result.to(target_device)
```

- [ ] **Step 4: Run CPU tests and an RTX 5090 CPU-vs-CUDA equality test**

Run the focused pytest command, then execute a small CUDA comparison using the editable venv. Expected: all comparisons are exact.

- [ ] **Step 5: Commit Task 1**

Commit only the utility and test with message `feat: preserve device when swizzling NVFP4 scales`.

---

### Task 2: Reusable selected-expert row staging

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_stream.py`
- Create: `test/registered/unit/layers/moe/test_expert_stream.py`

**Interfaces:**
- Produces: `expert_streaming_enabled() -> bool`.
- Produces: `ExpertStreamer(layer, tensor_names: tuple[str, ...])`.
- Produces: `ExpertStreamer.gather(topk_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]`.
- Requires: every source tensor has expert rows on dimension zero and is either CUDA-resident or pinned CPU memory.

- [ ] **Step 1: Write failing remap and validation tests**

Test the pure routing planner with duplicate decode IDs:

```python
topk_ids = torch.tensor([[9, 3, 9, 7]], device="cuda", dtype=torch.int32)
compact_ids, tensors = streamer.gather(topk_ids)
assert compact_ids.tolist() == [[0, 1, 2, 3]]
assert torch.equal(tensors["rows"].cpu(), source[[9, 3, 9, 7]])
```

Add a prefill case above the decode threshold and assert deduplicated rows plus an inverse map. Add failures for pageable CPU input, mismatched expert counts, noncontiguous rows, and out-of-range IDs.

- [ ] **Step 2: Run tests and confirm imports fail**

Run: `/data/models/slang/.venv/bin/python -m pytest test/registered/unit/layers/moe/test_expert_stream.py -q`

Expected: module import failure.

- [ ] **Step 3: Port the generic Haberstroh gather mechanism**

Implement a Triton byte-row gather for pinned host tensors and use `torch.index_select(..., out=...)` for CUDA tensors. Cache one maximum-row staging buffer per tensor name/dtype/device/row shape, slice it to the current row count, and use duplicate-preserving `arange` remapping for 64 or fewer routed entries.

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

- [ ] **Step 4: Add explicit source validation and bounded diagnostics**

Validate at construction and again before first gather. Log source shapes once and staging allocation size once. Raise `ValueError` for malformed tensors and `RuntimeError` for non-pinned CPU tensors.

- [ ] **Step 5: Run focused tests on the RTX 5090**

Expected: duplicate decode and deduplicated prefill gathers exactly match ordinary indexing for uint8, FP8, and FP32 rows.

- [ ] **Step 6: Commit Task 2**

Commit the module and tests with message `feat: add selected expert row staging`.

---

### Task 3: Expert-only V1 CPU offload and configuration guard

**Files:**
- Modify: `python/sglang/srt/utils/offloader.py`
- Modify: `python/sglang/srt/server_args.py`
- Create: `test/registered/unit/test_nvfp4_expert_offload.py`

**Interfaces:**
- Consumes: `expert_streaming_enabled()` from Task 2.
- Produces: `_iter_streamed_nvfp4_parameters(module) -> Iterator[torch.nn.Parameter]`.
- Preserves: ordinary `OffloaderV1` behavior when streaming is disabled.

- [ ] **Step 1: Write failing offloader-selection tests**

Build nested fake modules containing a decoder ModelOpt NVFP4 expert module, an FP8 MTP expert module, a PLE parameter, and a dense parameter. Assert only the six large decoder NVFP4 load-time tensors move to pinned CPU memory and no functional-call wrapper replaces the parent forward.

- [ ] **Step 2: Write failing compatibility tests**

Assert PLE plus `cpu_offload_gb` remains rejected without the environment flag and is accepted with the flag. Assert group offload remains rejected because the first milestone supports only V1 expert-only offload.

- [ ] **Step 3: Implement method-aware parameter selection**

Walk `module.named_modules()`, recognize a direct `quant_method` whose class name is `ModelOptNvFp4FusedMoEMethod`, and select the six load-time parameters with `recurse=False`. Do not select FP8 MTP modules merely because their parameter names resemble routed experts.

```python
NVFP4_OFFLOAD_PARAMETER_NAMES = frozenset({
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
})
```

- [ ] **Step 4: Add the gated V1 behavior**

When streaming is active, move only selected parameters, count their bytes against `cpu_offload_gb`, require the budget to cover each complete parameter, and skip installation of the whole-module forward wrapper. Raise if the configured budget expires partway through an NVFP4 expert module.

- [ ] **Step 5: Relax only the safe PLE compatibility combination**

Permit PLE plus positive `cpu_offload_gb` only when selected-expert streaming is enabled. Continue rejecting PLE plus grouped offload and every generic PLE/offload combination.

- [ ] **Step 6: Run focused tests**

Run: `/data/models/slang/.venv/bin/python -m pytest test/registered/unit/test_nvfp4_expert_offload.py -q`

Expected: all cases pass and the disabled path remains identical.

- [ ] **Step 7: Commit Task 3**

Commit with message `feat: offload only ModelOpt NVFP4 experts`.

---

### Task 4: ModelOpt NVFP4 compact-runner integration

**Files:**
- Modify: `python/sglang/srt/layers/quantization/modelopt_quant.py`
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`
- Create: `test/registered/unit/layers/quantization/test_modelopt_nvfp4_expert_stream.py`

**Interfaces:**
- Consumes: CPU-capable `swizzle_blockscale` and `ExpertStreamer`.
- Produces: a layer attribute `_nvfp4_expert_streamer: Optional[ExpertStreamer]` after weight finalization.
- Produces: compact `FlashInferCutlassMoeQuantInfo` using the six gathered tensors.

- [ ] **Step 1: Write a failing finalization test**

Create a small fake ModelOpt layer whose packed weights and raw block scales are pinned CPU tensors while input and per-tensor scales are CUDA tensors. Run post-load processing and assert final block scales remain pinned CPU tensors while `g1_alphas`, `g2_alphas`, and activation-global scales remain CUDA tensors.

- [ ] **Step 2: Write a failing compact payload test**

Use a fake runner to capture the dispatch output and quantization payload. Assert original IDs are remapped, each per-expert payload has the compact row count, scalar activation scales are unchanged, and routing weights are unchanged.

- [ ] **Step 3: Add streaming-aware finalization**

Detect the opt-in path only when the ModelOpt method uses FlashInfer CUTLASS and `w13_weight` resides on CPU. Swizzle raw block scales on CPU, copy into the already-pinned offloaded parameters through `alias_or_bind_derived_param`, leave the small alpha tensors on CUDA, validate all final sources, and attach `ExpertStreamer`.

- [ ] **Step 4: Add compact apply path**

Before constructing `FlashInferCutlassMoeQuantInfo`, gather the six tensors and structurally replace `topk_ids`. Populate `w13_weight`, `w2_weight`, and the six-element scale list from the gathered tensors plus the two resident scalar activation scales. Leave the non-streaming branch byte-for-byte equivalent.

- [ ] **Step 5: Add strict topology/runtime validation**

At streamer attachment, require TP1, EP1, standard A2A, FlashInfer CUTLASS, and a standard top-k output. Emit actionable errors naming the unsupported setting.

- [ ] **Step 6: Run CPU/unit tests and a synthetic GPU equivalence test**

Run the Task 1-4 test files. Then instantiate a small valid NVFP4 CUTLASS case, evaluate resident and compact forms with duplicate selected IDs, and compare BF16 outputs with exact equality when deterministic finalize allows it or tight documented tolerance otherwise.

- [ ] **Step 7: Commit Task 4**

Commit with message `feat: stream selected ModelOpt NVFP4 experts`.

---

### Task 5: Qwen3.8 startup and generation qualification

**Files:**
- Preserve: `python/sglang/srt/models/qwen4_exp.py`
- Create: `docs/superpowers/results/2026-09-11-nvfp4-expert-stream-bringup.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified launch command, observed memory figures, generation result, and known limitations.

- [ ] **Step 1: Run static and focused verification**

Run `python -m compileall` on modified modules, the four focused pytest files, and `git diff --check`. Confirm the user's PLE change is still present.

- [ ] **Step 2: Check GPU availability without disrupting user processes**

Inspect process and memory state. If another GPU workload is active, report the conflict and do not terminate it. Otherwise continue.

- [ ] **Step 3: Start the minimal server in a `cc-` tmux session**

Use `SGLANG_MOE_EXPERT_STREAM=1`, `--cpu-offload-gb 55`, PLE offload, FP8 PLE override, TP1, FlashInfer CUTLASS, chunked prefill 1024, one request, a small context/token pool, no overlap, no radix cache, disabled CUDA graphs, and no NEXTN. Redirect output to `/tmp/cc-nvfp4-stream.log`.

- [ ] **Step 4: Diagnose startup until the base model is ready**

For each failure, record the exact failing component and make one tested correction. Do not weaken tensor validation or silently fall back to full-table transfer.

- [ ] **Step 5: Run a deterministic generation smoke test**

Send one short completion request, confirm nonempty coherent output, and record GPU memory, host RSS, startup time, and decode throughput.

- [ ] **Step 6: Re-enable NEXTN independently**

Restart with the user's NEXTN settings, repeat the short request, and record acceptance/throughput. If NEXTN fails independently, retain the verified non-speculative launch and document the separate blocker.

- [ ] **Step 7: Restore the target context configuration**

Restore context length 262144, page size 64, Mamba settings, and memory fraction 0.85. Increase token-pool settings only while leaving sufficient staging headroom. Verify startup and one request.

- [ ] **Step 8: Write results and commit qualification evidence**

Record exact commands, versions, memory readings, throughput, limitations, and any deferred optimization work. Commit with message `docs: record NVFP4 expert streaming bring-up`.
