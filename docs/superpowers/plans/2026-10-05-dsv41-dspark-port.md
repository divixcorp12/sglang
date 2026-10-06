# DSV4.1 DSpark on numa-node-distributor (Phase 1: eager, hybrid draft) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DSpark runs eagerly on DSV4.1 EXL3 at the `numa-node-distributor` tip, with the hybrid draft (each stage's
resident set on the GPU, the other routed draft experts on the CPU), and an A/B against the resident draft
reproduces or updates §33.4's result.

**Architecture:** Port `dsv41-dspark-cpu-draft` (`7f22a6bcbb..9361644c82`, 37 commits) onto a branch where
`CpuExpertPool` and the traits' Python forward were deleted (`604a67f99f`) and the EXL3 quantization moved into
`layers/quantization/exl3/`. The draft's multi-row CPU forward moves onto the host module's `kernel_layer` /
`kernel_forward` exports, which already take `rows` (the `CpuExpertKernel` `ForwardCall`) and are promoted from the
instrumented build to production. Everything else is carried over with path and API adaptations.

**Tech Stack:** Python, PyTorch, the expert-stream host module (tvm-ffi JIT C++), the optimized EXL3 CPU kernel,
pytest; divix01 (RTX 5090) for GPU runs.

**Spec:** `DSV41_REFERENCE.md` §33 on this branch, plus the source plan
`docs/superpowers/plans/2026-10-02-dsv41-dspark-hybrid-draft.md` and §33.4 on `dsv41-dspark-cpu-draft`
(`git show 9361644c82:DSV41_REFERENCE.md`). Phase 2 (graphed verify, §33.3 items 1-8) is a separate plan.

## Global Constraints

- Source commit for every ported file: `9361644c82` (`git show 9361644c82:<path>`). zsh: quote `"9361644c82:<path>"`.
- Env vars go through `Envs` in `python/sglang/srt/environ.py` (env-var-conventions skill); names unchanged from the source:
  `SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`, `SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES`, `SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS`,
  `SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH`, `SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH`.
- Core lists parse with `threading_config.parse_cpu_list` (the source's `policy.parse_core_list` no longer exists); cores
  64-71 are refused (`threading_config.check_not_reserved`).
- Target CPU experts (`SGLANG_DSV41_CPU_EXPERTS`) stay refused with speculation; DSpark stays eager (`--cuda-graph-backend-decode disabled`).
- divix01: run from a pulled worktree with `PYTHONPATH=$PWD/python`; CPU jobs under `taskset -c 0-63`; GPU under
  `cc-gpu.lock`; arms take `rowimg-disk.lock` then `cc-gpu.lock`; read `PIPESTATUS` after any pipe.
- Changing the host C++ rebuilds every host module (50-100 s each, cold): warm serially before any `-n 8` run.
- No amend, rebase or stash; push only `dsv41-dspark`.

## Review Focus

1. A draft forward on the production host build with real EXL3 slabs at m = 1..6 rows: equal, bit for bit, to m one-row forwards.
2. A routed id outside the CPU set (resident or fused shared) must never reach the CPU kernel, including -1 and ids >= n_experts.
3. Draft cores overlapping the reserved 64-71 or given fewer than 2 cores: refused at launch, not at first draft call.
4. A draft-only model (the DSpark runner) with `SGLANG_MOE_PINNED_HOST_NUMA_MB` set: no capacity check, no tier.
5. Process exit with the draft runtime started: the worker shuts down and layers are dropped without a hang.

---

### Task 1: The pinned-tier NUMA check skips a model without streamed experts

**Files:** Modify `python/sglang/srt/layers/moe/expert_stream.py` (`ExpertPinnedHostCacheManager.from_model`, ~line 837);
Test `test/registered/unit/layers/moe/test_host_numa.py`.

- [ ] Step 1: add `test_a_model_without_streamed_experts_checks_no_capacity` to `TestManagerPlacement`, verbatim from
  `git diff 7f22a6bcbb 9361644c82 -- test/registered/unit/layers/moe/test_host_numa.py`.
- [ ] Step 2: run `pytest test/registered/unit/layers/moe/test_host_numa.py -k without_streamed -q -p no:randomly`. Expected: FAIL (`ValueError: short`).
- [ ] Step 3: move `placement = pinned_host_placement(budget_bytes)` below `if not streamers: return None` (as `d40d781771`).
- [ ] Step 4: rerun. Expected: PASS. Commit `fix(moe): skip the pinned-tier NUMA check for a model with no streamed experts`.

### Task 2: Production `kernel_layer` / `kernel_forward` / `kernel_error` / `kernel_drop`

**Files:** Modify `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (the four functions,
~lines 820-930, and the header comment at line 15); `python/sglang/kernels/ops/moe/expert_stream_transport.py`
(`TEST_ONLY_EXPORTS`, `kernel_layer`, `kernel_forward`, `kernel_drop` docstrings and `_refuse_test_only` calls).
Test: `test/registered/unit/kernels/test_expert_stream_kernel_exports.py` (new); the manual EXL3 multi-row test in
`test/manual/dsv41/test_cpu_expert_engines_exl3.py`.

**Produces:** `es.kernel_layer(kernel, spec, *, layout="exl3", variant=None) -> int`,
`es.kernel_forward(layer, x[m,H] f16, slots[m,k] i32, weights[m,k] f32, out[m,H] f32, *, threads, cores) -> (status, why)`,
`es.kernel_drop(layer)`, callable on the production build.

- [ ] Step 1: write the registered test: on `variant="prod"`, `kernel_drop(-1)` returns without raising, and
  `kernel_forward(10**6, ...)` raises `RuntimeError` matching `"no layer"` (not `"test-only"`); and the four names are
  absent from `TEST_ONLY_EXPORTS`.
- [ ] Step 2: run it. Expected: FAIL with `"kernel_drop is test-only"`.
- [ ] Step 3: delete the `if constexpr (!Build::kFaults) { test_only(...) } else` wrappers of the four functions
  (bodies unchanged), remove the four names from `TEST_ONLY_EXPORTS` and their `_refuse_test_only` calls, and reword
  "Test only:" to "The DSpark draft's CPU forward (cpu_experts/draft.py), and tests:".
- [ ] Step 4: rerun. Expected: PASS. Add the manual test
  `test_a_multi_row_forward_matches_one_row_forwards` (m = 1..6, k = 3, `variant="prod"`, real slabs from `_random_slabs`):
  m-row output equals the stacked one-row outputs bit for bit. Commit `feat(expert-stream): kernel_layer and kernel_forward in the production host build`.

### Task 3: Env vars, the resident-set file and the launch gate

**Files:** Modify `environ.py` (the source's two hunks); Create `python/sglang/srt/layers/moe/cpu_experts/draft_resident.py`
(verbatim from source); Modify `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` (`_check`);
Tests: `test/registered/unit/kernels/test_dspark_draft_resident.py` (verbatim), `test/registered/unit/test_expert_stream_requirements_exl3.py` (source hunk).

- [ ] Step 1: port both tests; add one: cores `"62-65"` are refused naming core 64.
- [ ] Step 2: run them. Expected: FAIL (import errors / missing env vars).
- [ ] Step 3: port env vars and `draft_resident.py`; port the gate block with `parse_cpu_list` and a
  `check_not_reserved(core)` loop before the `< 2` check.
- [ ] Step 4: rerun. Expected: PASS. Commit `feat(dspark): draft CPU expert env vars, resident-set file and launch rules`.

### Task 4: The draft's CPU runtime on `kernel_forward`

**Files:** Create `python/sglang/srt/layers/moe/cpu_experts/draft.py`; Test `test/registered/unit/kernels/test_dspark_draft_cpu_experts.py`.

**Consumes:** Task 2's exports; Task 3's env vars and `load_resident_set`.
**Produces:** `DraftLayer(slabs, on_cpu, act_limit)`, `DraftCpuExperts(kernel, layers, *, cores, threads, log_every=300)`
with `.submit(key, ids i64 [m,k], x [m,H], weights [m,k]) -> Optional[Future[Tensor f32 [m,H]]]`, `.close()`;
`DraftCpuExpertsRegistry.register(slabs, on_cpu, act_limit, *, layer_id) -> int`, `.runtime()`, `.close()`; `DRAFT_CPU_EXPERTS`.

Change from the source: `CpuExpertPool` is replaced by a `DraftKernel` with `make(slabs, capacity) -> layer id`,
`forward(layer, x16, slots32, w32, out, threads, cores) -> None` (raises `RuntimeError(why)` on a nonzero status) and
`drop(layer)`. The production `DraftKernel` builds `trait.layer_spec(slabs, capacity)` and calls `es.kernel_layer`,
`es.kernel_forward(..., threads=threads, cores=cores)` and `es.kernel_drop`. Slots go to int32 and weights to fp32
(the export's dtypes). The worker thread needs no bind step: `kernel_forward` pins the calling thread to `cores[0]`.

- [ ] Step 1: port the source tests, replacing `_Trait` with a fake `DraftKernel` (same arithmetic: `out = x * sum(valid weights)`), keeping every behaviour test (masking, skips, worker thread identity, stats, registry stage checks, act-limit mismatch).
- [ ] Step 2: run. Expected: FAIL (no module).
- [ ] Step 3: write `draft.py` from the source with the change above.
- [ ] Step 4: rerun. Expected: PASS. Commit `feat(dspark): the draft's CPU expert runtime on the host's kernel_forward`.

### Task 5: Hybrid draft wiring in `Exl3MoEMethod`

**Files:** Modify `python/sglang/srt/layers/quantization/exl3/exl3.py` (`Exl3Config.get_quant_method`, `Exl3MoEMethod`
`__init__`/`create_weights`/`process_weights_after_loading`/`apply`, plus `_attach_cpu_draft`, `_apply_cpu_draft`,
`record_draft_routes`); `python/sglang/srt/models/deepseek_v4_exl3_weights.py` (`is_dspark_draft_expert_module`).
Tests: source hunks of `test/registered/unit/layers/quantization/test_exl3_moe_method.py` and `test_exl3_stream_scope.py`.

- [ ] Step 1: port the test hunks (imports to `sglang.srt.layers.quantization.exl3`).
- [ ] Step 2: run. Expected: FAIL.
- [ ] Step 3: port the source's `exl3.py` hunks into the package file; `exl3_moe_accumulate`, `assert_not_capturing`,
  `Exl3Tensors` import from `.ops`.
- [ ] Step 4: rerun. Expected: PASS. Commit `feat(exl3): hybrid DSpark draft experts and the draft route probe`.

### Task 6: Scripts and the GPU parity test

**Files:** `scripts/dsv41/trace_corpus.py` (`--log-level`), `test/manual/dsv41/test_trace_corpus.py` (hunk);
`analysis/dsv41-drive/dspark/{ab_cpu_draft.py,draft_resident_set.py,draft_routes_report.py}`;
`analysis/dsv41-drive/cpu-experts/draft_bench.py` (port only if it imports nothing deleted; else leave out and ledger);
`test/manual/dsv41/test_dspark_hybrid_draft_gpu.py` (imports `exl3_ops` -> `quantization.exl3.ops`).

`ab_cpu_draft.py`: drop `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING` and any other variable this branch's `environ.py`
no longer defines (check each `COMMON` key with `grep -n KEY python/sglang/srt/environ.py`); draft cores `18-27`
(clear of this branch's driver cores 28-31).

- [ ] Step 1: port `test_trace_corpus.py` hunk; run; expected FAIL; port `trace_corpus.py`; PASS.
- [ ] Step 2: port the analysis scripts and GPU test. Commit `bench(dspark): A/B driver, resident-set tools and hybrid GPU parity test`.

### Task 7: divix01 runs and docs

- [ ] Step 1: push; pull into `wt-dsv41-dspark`; warm host modules serially; run
  `test/registered/unit/kernels/test_dspark_draft_*.py test/registered/unit/kernels/test_expert_stream_kernel_exports.py test/registered/unit/layers/moe/test_host_numa.py test/registered/unit/layers/quantization/test_exl3_moe_method.py test/registered/unit/layers/quantization/test_exl3_stream_scope.py test/registered/unit/test_expert_stream_requirements_exl3.py`. Expected: all pass.
- [ ] Step 2: with `SGLANG_DSV41_CPU_EXPERTS=1 SGLANG_EXL3_SRC=...`: `test/manual/dsv41/test_cpu_expert_engines_exl3.py`, then under `cc-gpu.lock` `test/manual/dsv41/test_dspark_hybrid_draft_gpu.py`. Expected: pass.
- [ ] Step 3: smoke both arms, 1 session, 32 tokens (`AB_SESSIONS=1 AB_NEW_TOKENS=32`). Expected: rc 0, `spec_verify_ct > 0`, the "DSpark CPU experts: 3 draft stages" log line in hybrid.
- [ ] Step 4: the A/B as §33.4 ran it (16 sessions incl. held-out `AB_SKIP`); paired median decode tok/s ratio.
- [ ] Step 5: port §33.4 into `DSV41_REFERENCE.md` and add the rerun at this tip. Commit `docs(dsv41): DSpark hybrid draft on numa-node-distributor`.
