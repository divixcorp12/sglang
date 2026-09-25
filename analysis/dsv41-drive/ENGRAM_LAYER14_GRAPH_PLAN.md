# Layer 14 Engram graph replay implementation plan

> **For agentic workers:** Implement and review each task in order. Work on `master`; preserve unrelated working-tree changes. Use the existing layer-1 host-node path as the implementation template.

**Goal:** Capture the layer-14 Engram file lookup in the batch-1 TP-1 decode graph so both Engram lookups replay with zero eager graph breaks.

**Architecture:** At layer 14's existing consumption point, capture the same ordered D2H hash IDs → native CPU host callback/io_uring row lookup → H2D packed rows → GPU status check/dequant sequence used by layer 1. Each layer gets its own retained pinned staging/context; both contexts use the existing layer-tagged native store and its single io_uring worker. Keep the opt-in flag and all fallback modes unchanged.

**Tech stack:** PyTorch CUDA graphs, CUDA host functions, pinned host memory, C++ native Engram store/liburing, pytest.

**Spec:** [Engram host-node cache/io_uring design](ENGRAM_HOST_NODE_CACHE_URING_PLAN.md) and [lookup optimization plan](ENGRAM_LOOKUP_OPTIMIZATION_PLAN.md).

## Global constraints

- The route remains opt-in under `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1`.
- Scope is batch-1, TP-1 decode on NVIDIA CUDA with the `breakable` decode graph backend. Prefill, extend, target verify, larger batches, other TP sizes, and non-CUDA modes keep their current eager/fallback behavior.
- Do not change the native cache budget, pin the full cache, or add a second cache. Both layers already register in the one native 5 GiB store using distinct `layer_id << 40` tags.
- Do not move the layer-14 lookup before its consumer in this step. This preserves current execution order and isolates the structural graph-break change from an I/O-overlap experiment.
- CUDA host callbacks must not invoke Python, the GIL, or CUDA APIs. The existing native callback submits to the dedicated io_uring worker and returns only when packed rows and status are ready.
- Run GPU-dependent tests and serving checks on `divix01`, in an isolated checkout, under `/data/models/slang/nvfp4-work/cc-gpu.lock`. Do not use the laptop as serving evidence.

## Review focus

- Two retained callback contexts have distinct pinned ID, row, and status buffers and outlive every replay.
- Changing both layers' IDs between replays yields the new rows, including duplicate IDs and IDs that collide numerically across layers.
- A cold layer-14 lookup issues io_uring SQEs/CQEs; repeating it warm issues no new SQEs.
- A layer-14 I/O/ID failure sets nonzero status and cannot expose stale packed rows on replay.
- One-stream replay restriction and graph destruction remain safe with two callbacks; no in-flight staging reuse or callback-after-free.

## Task 1 — Enable the layer-14 capture route

**Files:** Modify `python/sglang/srt/layers/engram.py`; test `test/registered/unit/layers/test_engram_file_table.py`.

**Interface:** Reuse `_capture_engram_file_lookup(file_table, indices) -> torch.Tensor` and `_EngramHostLookupContext` without changing the C++ or table APIs. In `EngramEmbedding.forward`, change only the layer gate from `layer_id == 1` to `layer_id in (1, 14)`; leave every other predicate in place. This gives layer 14 the same capture sequence and separate graph-retained context. Confirm both tables have the same packed row width before assuming the shared-store construction is valid; `EngramFileTable.open` and native `get_shared_store` currently enforce it.

- [ ] Change the combined graph test named `test_layer1_host_node_replays_file_rows_and_layer14_keeps_one_break` to expect **one segment, zero breaks, and two retained contexts**. Rename it to describe both layers.
- [ ] Run that test first; it should fail on the current `layer_id == 1` gate.
- [ ] Change the gate to `layer_id in (1, 14)` and run the test again. Keep the existing changing-ID, duplicate-ID, deduplicated-graph, and cross-stream checks.
- [ ] Add a direct assertion that the two retained contexts' `ids`, `rows`, and `status` pointers differ, and that their native table tags differ. This catches accidental staging aliasing.

The critical code change is:

```python
if (
    getattr(self, "layer_id", None) in (1, 14)
    and getattr(self, "tp_size", None) == 1
    and indices.is_cuda
    and indices.shape[0] == 1
    and getattr(self.file_table, "_host_node_extension", None) is not None
    and getattr(self.file_table, "_native_store", None) is not None
    and forward_batch is not None
    and forward_batch.forward_mode.is_decode()
    and is_breakable_graph_capturing()
):
    return _capture_engram_file_lookup(self.file_table, indices)
```

## Task 2 — Verify fallback and failure behavior

**Files:** Test `test/registered/unit/layers/test_engram_file_table.py` and `test/registered/unit/layers/test_engram_lookup_break.py` only as needed; modify implementation only if a test exposes a real defect.

- [ ] Add a layer-14 test with native route disabled: a captured decode lookup still takes the existing eager break, and a non-captured lookup still uses `EngramFileTable.lookup`.
- [ ] Add a graph replay case with IDs changed independently for both layers, including the same integer ID in both tables; compare each BF16 output to its own layer's fixture bytes. Use different layer-14 fixture contents so a wrong tag cannot pass by coincidence.
- [ ] Add a captured layer-14 invalid-ID replay and assert a failure is surfaced by the captured status check and no preceding output is accepted as fresh. Run it on `divix01`; if PyTorch reports the graph as poisoned after a device assert, construct a fresh graph for subsequent checks.
- [ ] Retain a cold/warm counter assertion for each layer separately: cold layer-14 requests increase native `submitted_sqes` and `completed_cqes`; a repeated warm replay leaves them unchanged. Use disjoint ID sets or delta snapshots so layer-1 traffic cannot mask the result.
- [ ] Re-run the existing non-CUDA/CPU lookup and eager-break tests to ensure the default path is unchanged.

## Task 3 — Validate actual graph and serving impact

**Files:** No product change expected. Save results beside the existing Engram analysis under `analysis/dsv41-drive/`.

- [ ] On `divix01`, run the focused unit tests with the isolated checkout and GPU lock. At minimum run `test_engram_file_table.py`, `test_engram_lookup_break.py`, and `test_engram_row_cache.py`. Verify the result, rather than relying on laptop tests.
- [ ] Run a short real-shard batch-1 decode capture/replay with the opt-in flag and `--cuda-graph-backend-decode breakable --cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1 --cuda-graph-backend-prefill disabled`. Check actual layer-1 and layer-14 outputs against eager lookup for multiple changing ID sets. Confirm one graph segment and zero Engram breaks; report if another subsystem inserts breaks.
- [ ] Compare a fresh matched serving A/B on `divix01`: current one-break commit versus two-host-node candidate, same corpus, checkpoint, GPU settings, cache budget, and request sequence. Record decode token rate/latency, layer callback wait, cache hit/miss and SQE/CQE counters, GPU idle time, and host/device staging bytes. A structural zero-break result alone is not a throughput win.
- [ ] Review CUDA trace ordering: layer 14 should issue D2H only when its model layer is reached, then host lookup, H2D, status check, and dequant before Engram consumption. Check that no stale pinned buffer is read during consecutive replays.
- [ ] Document the measured result and decision. If two callbacks serialize through the worker and regress throughput, keep the correctness implementation opt-in and plan early layer-14 staging/overlap as a separate experiment; do not silently alter the graph schedule in this change.

## Done when

Two-layer graph test reports one segment and zero Engram breaks; changing/cold/warm/failure/fallback tests pass; real-shard output matches eager; and a matched `divix01` serving comparison states whether the change improves decode performance. Commit only the intended implementation, tests, and result note on `master` after checking the working tree.
