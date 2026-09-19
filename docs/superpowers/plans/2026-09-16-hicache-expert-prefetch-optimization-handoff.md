# HiCache-informed expert prefetch optimization handoff

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans when implementing the tasks. This document is a reviewed handoff, not evidence that the changes or benchmarks have been completed.

**Goal:** Reduce expert-request-to-availability latency and total token latency on PCIe Gen 3 ×16 while preserving correct speculative-slot ownership and physical transfer accounting.

**Architecture:** Retain the mapped-host GPU pull and dedicated speculative row. Tighten candidate eligibility, fuse repeated device planning operations, and compare early versus after-demand LLaPor posting. Borrow HiCache's explicit transfer ownership and configurable geometry without transplanting its small-KV-entry copy algorithm.

**Tech stack:** PyTorch, CUDA graphs, CUDA C++ JIT kernels, mapped pinned host memory, optional host-issued DMA.

**Basis:** Source review at `0fe8d526df` on 2026-09-16, the user discussion, and the earlier [side-stream handoff](2026-09-16-side-stream-expert-pull-handoff.md). No new GPU measurements were made for this handoff. The current code already includes fused route planning and real side-stream pulls; older descriptions of scoring-only integration are obsolete.

## Scope and priority

This supplements the earlier handoff. It explicitly supersedes its categorical after-demand posting recommendation: **early LLaPor posting is a valid arm, not a demonstrated performance bug.** Preserve it while comparing alternatives. The empty-eligible-candidate behavior is a concrete issue; geometry and cache-hint changes are experiments.

Implement in this order:

1. Correct eligibility and no-offer state; preserve completion/lifetime invariants.
2. Remove duplicated route/remap/statistics operations from the captured hot path.
3. Compare LLaPor posting placement with identical scoring and one-row budgets.
4. Benchmark transfer geometry and cache hints under concurrent compute.
5. Pursue packing, promotion reuse, or DMA signaling only where traces justify them.

All performance claims must distinguish source-level expectations from measured improvements. Keep one-row speculation initially. Preserve generic deduplication for multi-token routes. Do not expand supported concurrency or replace the copy backend as an incidental part of this work.

## 1. What HiCache does and what applies here

The relevant source roots below are relative to this repository.

| Component | Source | Relevant responsibility |
|---|---|---|
| HiCache wrapper | `python/sglang/kernels/ops/kvcache/hicache.py` | JIT specialization and dispatch |
| HiCache kernel | `python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh` | Indexed KV transfers |
| Staged write-back | `python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh` | GPU relayout followed by D2H DMA |
| HiCache controller | `python/sglang/srt/managers/cache_controller.py` | Queues, events, submissions, acknowledgements |
| HiCache ownership | `python/sglang/srt/mem_cache/hiradix_cache.py` | Allocation, protection, cache state, completion processing |
| Transfer engine | `python/sglang/srt/mem_cache/l2_transfer.py` | Transfer streams and sequencing |
| Expert wrapper | `python/sglang/kernels/ops/moe/expert_cache_transfer.py` | Validation, persistent segment table, JIT dispatch |
| Expert kernel | `python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh` | GPU-selected row pulls into cache slots |

The fair wrapper comparison is `hicache.py` against `expert_cache_transfer.py`, not against the entire expert cache manager. HiCache still has substantial Python orchestration. Its strongest reusable pattern is separating ownership, transport, and readiness.

### Width and threading

HiCache's `kUnroll` changes subgroup width: `kNumThreads = 32 / kUnroll`, and each lane's package is `128 / kNumThreads` bytes. Defaults are 8 lanes/16 bytes for elements at most 512 bytes, 16 lanes/8 bytes through 1024 bytes, and 32 lanes/4 bytes above that. It assigns one subgroup to each KV entry, loads a compile-time-sized entry into `LocalStorage`, and fully unrolls the entry loop.

The expert kernel already uses 16-byte vector loads/stores when aligned, with 4-byte and byte fallbacks. Its default is a fixed eight blocks of 256 threads, versus HiCache's quota of up to two blocks of 1024 on CUDA: both reach 2048 threads when HiCache uses its full quota. HiCache reduces its block count for short index lists; one KV entry launches one block, but only its owning subgroup copies that entry. Experts distribute a small number of large rows across whole warps; one row can use all 64 warps. This is a reasonable shape for multi-megabyte expert payloads.

Do not instantiate HiCache with an entire expert as its element. Its per-thread storage and unrolled code would grow enormously, and only one subgroup would own that row. Adapting the idea requires bounded tiles, not treating megabytes as a small KV entry.

Borrow tunable block count, block size, and bounded unrolling. Preserve coalescing, segment completeness, device-side count, and safe tails. Benchmark both bandwidth and compute interference; fewer blocks can reduce interference but also reduce outstanding PCIe reads.

### Cache instructions

HiCache's helper named `load_nc` emits `ld.global.L1::no_allocate`; ours emits `ld.global.nc`, with `st.global.cg` stores. These are different operations/hints, not different spellings of the same instruction. `.nc` accesses a noncoherent read-only cache; `L1::no_allocate` controls allocation behavior. Neither is a synchronization mechanism.

Compare cache hints independently from launch geometry. Keep immutable host weights and source lifetime guarantees. Do not generalize the expert read-only-source assumption to mutable KV. Consult the [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) and verify instruction support on the deployment toolchain/device.

### Why staged write-back is not a replacement

`staged_write_back.cuh` performs:

```text
scattered GPU KV -> GPU relayout/staging -> larger D2H DMA copies
```

It takes CPU-visible destination indices and uses `cudaMemcpyBatchAsync` when available and suitable, otherwise individual `cudaMemcpyAsync` calls. Its 128 KiB threshold is specific to that implementation, not a universal expert-transfer threshold.

Expert inference weights are immutable and already backed by host memory. Eviction needs no host write-back. Expert cache filling is H2D; promotion from valid GPU scratch is D2D. Neither directly matches staged KV write-back.

Useful ideas are reusable descriptors, layout-aware batching, explicit staging lifetime, and capability/fallback checks. Our segment kernel already transfers multiple weight tensors in one launch. Packing experts into fewer host segments may help DMA submission, but costs layout/storage complexity and must be measured. GPU staging after an H2D gather does not reduce the bytes that already crossed PCIe and introduces an extra device copy.

## 2. Python versus captured GPU work

There are three different costs:

| Phase | What runs | Optimization implication |
|---|---|---|
| Setup | Allocations, JIT loading, segment construction, stream/event creation | Keep outside steady-state execution |
| Eager forward / graph capture | Python hooks dispatch tensor operations and establish dependencies | Python dispatch matters in eager mode; capture records GPU work |
| Graph replay | Captured GPU kernels, copies, and dependencies | Original Python hook bodies do not rerun; recorded operations still cost time |

Moving `where`, comparisons, reductions, copies, and additions into a C++ function that dispatches the same separate GPU operations is not kernel fusion. It may help eager host dispatch, but does not by itself shorten the captured GPU chain. Fusion means a CUDA kernel directly performs several of those computations and writes final outputs in one launch, with less intermediate memory traffic.

Examples from the current path:

| Location | Current work | Candidate change |
|---|---|---|
| `serving/candidates.py:PrefetchCandidateBank.write` | Sum scores, finite/residency masks, stable full sort, index selection, bank copies | Fused eligible top-1 path when only one candidate is needed |
| `serving/runtime.py:PrefetchPuller.post_target` | `where`, sentinel construction, dtype conversion, ID/count copies | One device operation to write posted ID and count |
| `expert_route_plan.py:plan_graph_routes_fused` | Fused routing already exists | Extend this kernel instead of building another post-planning tensor chain |
| `expert_stream.py:_gather_graph` | Residency lookup for join and optionally calibration, after planner already looked up slots | Reuse planner results or emit needed metrics directly |
| `serving/runtime.py:join_target` | Coverage comparison, reductions, residual calculation, waste flag, four counter additions, another remap | Fused planner emits counters and final remap; join retains readiness dependency |

These are source operations, **not measured kernel-launch counts**. Views may be metadata-only, conversions may be no-ops when dtype already matches, and compiler behavior can change the count. Profile the actual captured graph before quoting launch savings.

Do not spend the first iteration eliminating a capture-time `torch.empty` just because it appears in Python. Captured allocation lifetime and graph-pool usage matter, but that is different from a fresh Python allocation on every replay. Prefer persistent outputs where useful for an explicit interface, not as an unsupported latency claim.

### Proposed fusion boundaries

Keep the following dependency stages distinct:

```text
predictor scores
  -> select eligible candidate / write posted metadata
  -> record ready -> side-stream payload pull -> record done

actual target routes
  -> fused residency + coverage + residual plan + remap + outcome counters
  -> wait for payload completion
  -> residual demand copies
  -> expert compute
```

The planner may use metadata before the wait because posting wrote it on the main stream. It may not read the speculative weight payload before the wait. Combining planner and transfer into a single ordinary multiblock kernel is not a trivial extension: it would need a valid grid-wide coordination design. Keep them separate for this work.

## 3. Task A: Make no-offer semantics explicit

**Modify:** `python/sglang/srt/layers/moe/expert_prediction/serving/candidates.py`, `serving/runtime.py` in the same directory. Extend `test/registered/unit/layers/moe/test_expert_prefetch_scoring.py` and `test_expert_prefetch_pull.py`.

Current issue: the bank masks residents/nonfinite scores for sorting but stores valid fallback IDs and original scores when eligibility is empty. Posting uses only the configured mode to set count. In `always` mode, all-resident or all-invalid inputs therefore still copy one row. Valid fallback IDs avoid out-of-bounds access but do not mean a useful offer exists.

**Proposed interface:** add persistent bool bank validity with the same shape as `ids`, exposed by `valid_for(target_layer)`. The ID bank can keep valid fallback indices for existing readers; validity determines whether an offer exists. All consumers claiming an offer, including recall/calibration, must honor this mask. Scores alone are insufficient because resident fallback IDs can have finite original scores.

Reference logic, to be fused later if profitable:

```python
eligible = torch.isfinite(summed) & (expert_to_slot < 0)
ranked = torch.where(eligible, summed, float("-inf"))
order = torch.argsort(ranked, descending=True, stable=True)
selected = order[:width]
bank_valid.copy_(eligible.index_select(0, selected))

candidate = bank_ids[:1]
valid = enabled & bank_valid[:1] & (live_expert_to_slot[candidate] < 0)
posted_id.copy_(torch.where(valid, candidate, -1))
posted_count.copy_(valid.to(torch.int32))
```

This is proposed reference pseudocode, not a callable existing API. Keep sentinel `-1` out of source indexing: the pull kernel must observe count zero before reading source rows. Preserve current `count_zero` diagnostic mode, including its graph nodes and completion event.

- [ ] Add failing tests for all residents, all nonfinite scores, fewer eligible candidates than width, and candidate becoming resident between scoring and posting.
- [ ] Implement explicit validity; update offering/recall/calibration consumers.
- [ ] Ensure a no-offer replay rewrites metadata to `id=-1,count=0`; never retain a prior forward's prediction.
- [ ] Verify stable tie selection by lowest expert ID and resident-hit precedence at target routing.
- [ ] Re-run the focused candidate and pull tests on a CUDA-capable environment. A CUDA skip is not successful GPU validation.

Concrete acceptance cases:

| Scores | Slot map | Expected offer |
|---|---|---|
| `[10, 9, 8]` | `[0, -1, -1]` | expert 1 |
| `[10, 9, 8]` | `[0, 1, 2]` | none |
| `[NaN, +inf, -inf]` | `[-1, -1, -1]` | none |
| `[5, 5, 4]` | `[-1, -1, -1]` | expert 0 |

## 4. Task B: Fuse target outcome bookkeeping into route planning

**Modify:** `python/sglang/kernels/jit/csrc/moe/expert_route_plan.cuh`, `python/sglang/kernels/ops/moe/expert_route_plan.py`, `python/sglang/srt/layers/moe/expert_route_plan.py`, `expert_stream.py`, and `expert_prediction/serving/runtime.py`.

**Tests:** `test/registered/unit/kernels/test_expert_route_plan_fused.py`, `test/registered/unit/layers/moe/test_expert_route_plan.py`, `test_expert_prefetch_pull.py`.

The existing fused kernel already knows each lane's resident, prefetched, and residual status. Add optional persistent outcome counters to that interface rather than recomputing masks in `join_target`. For the BS1 unique-route path, lane zero can derive:

```text
covered_routes = popcount(prefetched_mask)
residual_routes = popcount(residual_mask)
posted_rows = (posted_count == 1)
wasted_rows = posted_rows && (covered_routes == 0)
residual_copy_rows = popcount(residual_mask)
```

Retain logical demand misses as `covered_routes + residual_routes`; successful prefetch is not a permanent cache hit. Preserve the existing cumulative counter ordering `[covered, residual, wasted, posted]`. The proposed wrapper extension passes the real posted count and optional int64 outcome-counter tensor; do not infer a valid offer solely from a fallback candidate ID.

The coverage predicate itself must require `posted_count == 1`, in addition to matching the predicted ID and being nonresident. Test a deliberately stale positive ID with count zero: it must not suppress a demand copy or redirect a route to speculative storage. Normal posting should still write `id=-1,count=0` together for no offer.

After fusion, the fast-path join must still enqueue the completion wait, but must not apply a second `where` remap or add the same counters again. Provide a clear dispatch distinction: planner-owned accounting on the fused path, existing/reference accounting on the generic path. Keep one owner per counter per forward.

- [ ] Add reference-equivalence cases for no offer, correct prediction, wrong prediction, resident prediction, zero residual misses, and 32 routes.
- [ ] Extend CUDA and Python wrappers together; preserve optional-counter behavior and output dtypes.
- [ ] Use the planner's final remap directly after the join on the fused path.
- [ ] Keep multi-token deduplication and route-multiplicity accounting intact on the generic path.
- [ ] Trace replay before/after: verify reduced device work, unchanged dependency edges, and unchanged physical copy counts.

Example: actual routes `[5,2,9,1]`, residents `2->7,1->3`, scratch base 10, speculative slot 20. Prediction 5 yields remap `[20,7,10,3]`, one residual copied row (9), one posted row, one covered route, zero waste. Prediction 8 yields remap `[10,7,11,3]`, two residual copied rows, one posted row, zero covered routes, one wasted row.

For repeated routes `[5,5,9]` with prediction 5 and no residents, the generic path has two covered routes but only one useful physical prefetched row and one distinct residual row. Never use covered-route count as transferred-row count.

## 5. Task C: Fuse candidate selection where the serving configuration permits

Keep the predictor neural network separate initially. Optimize the control work around its output.

**Modify:** candidate/runtime modules above. If profiling justifies a new kernel, create `python/sglang/kernels/jit/csrc/moe/expert_prefetch_plan.cuh` and `python/sglang/kernels/ops/moe/expert_prefetch_plan.py` (proposed new files).

**Proposed operation:** given scores, live residency, and mode, reduce scores across active rows, exclude nonfinite/resident experts, choose the largest eligible score with lowest-ID tie breaking, and write persistent top-1 ID, validity, and count. Define an explicit no-offer output. Match the reference dtype/reduction semantics, especially if more than one token contributes scores.

Do not silently remove wider candidate-bank outputs. Shadow recall, calibration margin, and larger-budget analysis may need top-W or at least top-2 scores. Use a fast top-1 path only when those consumers are disabled or explicitly supported. For full diagnostic mode, retain the reference path until an equivalent fused top-W implementation is tested.

- [ ] Measure candidate-selection node count/time separately from predictor cost.
- [ ] Establish reference equivalence using Task A's cases plus finite negative scores and ties.
- [ ] Implement only the needed serving specialization; preserve a generic fallback.
- [ ] Compare diagnostics-off steady-state latency separately from diagnostic runs.

## 6. Task D: Compare LLaPor posting timing

Immediate posting maximizes lead time. Waiting until source demand finishes protects urgent copies but reduces that lead time. On one PCIe link, simultaneous H2D transfers share bandwidth; contention alone does not establish which schedule minimizes total latency.

Implement two explicit experimental placements, keeping current immediate placement as the baseline:

| Placement | Ready event is recorded | Tradeoff |
|---|---|---|
| Immediate | After prediction/offer preparation at source routing hook | More lead time; can compete with source demand |
| After source demand | After source gather copies, before source expert compute | Protects source demand; shorter speculative window |

**Modify:** `serving/runtime.py` and the relevant gather/forward integration in `expert_stream.py`. Proposed setup-time mode names are `immediate` and `after_source_demand`; these are not existing configuration options. Resolve mode before graph capture and capture a distinct graph for each arm. Do not branch in Python based on device data during replay.

For after-demand mode, the score hook prepares candidate state but does not submit the pull. A hook after source gather submits exactly the target given by LLaPor's existing source-to-target mapping. Keep candidate selection and score cost identical across timing arms; separately label any additional eligibility recheck cost. Verify a source lacking an applicable graph-gather path cannot leave a target waiting on stale metadata or an unrecorded event. Reject unsupported configurations or provide an explicit tested alternative.

Do not apply this placement blindly to APEX: its same-layer pre-mixer prediction has a different overlap window.

- [ ] Test exactly one post per target per forward for both placements, including no-offer paths.
- [ ] Check graph dependencies establish the intended placement, not just Python call ordering.
- [ ] Measure source demand completion, source compute duration, side-pull completion, target wait, and total token latency.
- [ ] Include source zero-miss and miss-heavy workloads; report useful and wasted speculation separately.
- [ ] Choose the default from end-to-end measurements. Do not call immediate posting a correctness bug.

## 7. Ownership and completion acceptance requirements

Dedicated storage prevents overlap with demand scratch; it does not prove the entire lifecycle is safe. Validate all of these:

- Every target-consumed prediction was posted for the current forward, or explicitly reset to no-offer.
- Main-stream metadata publication precedes both planner reads and the side-stream ready dependency.
- All weight components, including any device-resident components, are complete before `done`.
- Target consumption waits for `done`; a no-offer path still preserves capture fork/join structure.
- The initial design joins wrong predictions too. Their wait and PCIe traffic count against performance; no implied cancellation.
- Every fork joins before capture ends and before state reuse, including eager/unsupported/tail paths.
- Independently overlapping graph executions cannot share mutable plans, events, or destination slots. Require separate state or explicitly enforce serialization.
- Residency-map updates have defined ordering relative to selection and routing. Recheck current eligibility at posting where required.
- Permanent promotion/eviction never selects the dedicated speculative row as a destination.
- Warmup/capture counter reset and periodic cumulative snapshots remain correct after fusion.

Use `test/registered/unit/layers/moe/test_expert_gpu_pull.py` and `test_expert_prefetch_pull.py` for actual captured replay tests, changing predictions and counts between replays. Verify byte-exact payloads across every segment and unchanged neighboring rows. Add production-cache allocation coverage: the current pull test's introductory comment still describes a historical missing trailing row; update that stale explanation when touching the test, and do not rely only on the fake cache fixture.

## 8. Transfer tuning, promotion, and optional DMA follow-ups

Once the preceding stages pass, parameterize the expert copy launch and benchmark a small matrix, for example 2/4/8 blocks with 256/512 threads, retaining 8×256 as the baseline. Treat these as trial configurations, not recommended defaults. Keep total warps integral and preserve the one-row and multi-row partitioning formula.

Separately compare existing cache hints against a supported no-allocate variant. Test one row, several rows, count zero, actual heterogeneous weight segments, and alignment tails. Record bandwidth, copy completion latency, source/target compute slowdown, and token latency. A synthetic copy-only win is insufficient.

Promotion remains a separate opportunity: reuse a still-valid GPU scratch row through D2D copy when policy admits that expert, rather than transferring the same host payload again. Prove source lifetime and destination publication ordering. Apply a global promotion byte/time budget; per-layer limits do not bound the total pause. No D2H expert write-back is needed for immutable weights.

For the optional doorbell DMA backend, keep previously discussed zero-count device completion, reduced redundant resolve work, timing-disabled events, sampled tracing, and avoidance of unnecessary host locking as separate changes. Preserve bounded failure behavior and outstanding-DMA slot ownership.

`cuStreamBatchMemOp` batches synchronization operations such as waits and writes; it does not batch expert payload copies. `cudaMemcpyBatchAsync` batches copy submissions and is already relevant to the DMA implementation. Stream-ordered completion writes or batched signaling are optional doorbell experiments, not prerequisites for the event-based side-stream GPU pull. Verify installed-version and graph support; memory waits are not a drop-in replacement for a bounded timeout. Keep CUDA-visible dependencies because memory-operation synchronization alone is not visible to CUDA's scheduler. See [CUDA stream memory operations](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html).

## 9. Measurement and delivery criteria

Use the same model, cache memory budget, predictor checkpoint, scheduler settings, workload, and graph shape across arms. Charge speculative storage against the cache budget. Use repeated warmed measurements and record commit/toolchain/GPU details. Do not run unrelated workloads concurrently.

Minimum comparison sequence:

1. Existing demand-only fused route path.
2. Predictor plus count-zero pull, to isolate control/graph overhead.
3. Corrected one-row immediate pull.
4. Corrected one-row after-demand pull.
5. Winning placement with fused outcome bookkeeping.
6. Candidate-selection fusion, then geometry/cache-hint changes independently.

For each arm report median and p95 token latency, target stalls, source demand time, compute interference, predictor/control cost, actual residual rows, posted rows, useful/wasted rows, and promotion bytes. Physical H2D rows are posted rows plus distinct residual host rows plus separately counted promotions; route counts are not interchangeable with rows. Byte totals derived from row counts are not independent measurements of PCIe wire traffic.

Capture a short GPU timeline/node inventory to establish which operations fusion removed. Measure overall latency with minimal instrumentation separately; profiler overhead must not become the claimed speedup. An operation-count reduction is useful evidence but is not itself a latency result.

Focused test files are listed per task. Run them with the repository's configured Python environment and import setup on an authorized CUDA environment; for example, from the repository root:

```bash
python -m pytest test/registered/unit/layers/moe/test_expert_prefetch_scoring.py test/registered/unit/layers/moe/test_expert_prefetch_pull.py test/registered/unit/layers/moe/test_expert_gpu_pull.py test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/kernels/test_expert_route_plan_fused.py
```

Add `test_expert_prefetch_runtime.py` when changing feature-hook scheduling. A documentation-only handoff does not require running these GPU tests. Remote experiments require the applicable divix01-pilot workflow and operational coordination; this handoff does not start or reserve a remote run.

Before completing implementation, provide the exact configuration, focused correctness results, captured dependency evidence, matched latency results, and any unsupported lifecycle cases. Keep each semantic or performance change independently reviewable. Do not claim that HiCache is faster for expert traffic without measuring an adapted implementation on this workload.

## python in hot path

This additional audit examined source at `ef81a4d426`. No runtime profiling or GPU benchmarks were performed. The findings below describe that revision; subsequent implementations may address them. Priorities reflect source-level opportunities, not measured latency savings. Paths are relative to this repository, and line references identify the audited revision.

### Distinguish replay, eager execution, and surrounding bookkeeping

There is real Python execution and allocation around forwards even when expert transfers execute inside a CUDA graph. The captured transfer wrapper is only part of the path.

| Path | Python runs each forward? | Allocation behavior |
|---|---|---|
| Captured expert planning and transfer | No, during replay | Reuses captured addresses and graph-pool storage |
| The same functions executed eagerly | Yes | Creates intermediate tensors and Python objects |
| Cache observer around graph replay | Yes, when a real recorder is active | Creates CPU/GPU temporaries |
| Periodic metrics and residency updates | Yes, when triggered | Readbacks, lists, dictionaries, and other temporaries |

An allocating tensor operation encountered during capture is not evidence of a fresh Python allocation on every replay. PyTorch preserves captured addresses using graph memory pools. The operation's recorded GPU computation, including initialization, copies, reductions, and intermediate memory traffic, still executes. Likewise, an eager tensor allocation can use a caching allocator; it does not necessarily issue a fresh `cudaMalloc` or host pinning system call. See [PyTorch graph memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#graph-memory-management).

Moving Python tensor calls into a C++ wrapper that dispatches the same separate operations does not eliminate their replay cost. Fuse operations in CUDA, write directly into final destinations, or avoid unnecessary work to reduce the captured GPU chain. Optimize Python dispatch and buffer allocation separately for eager execution and work outside the graph.

### 1. Per-forward statistics outside the graph are the first audit target

`python/sglang/srt/layers/moe/expert_hot_cache.py:1586`, `_accumulate_registers()`, performs the following after each recorder-observed applicable forward:

- Gathers route-count rows and updates popularity.
- Updates a dense adjacent-layer affinity bank with `baddbmm_`.
- Constructs `torch.tensor(eager_gathered, dtype=torch.bool)` on the CPU.
- Pins this tensor and copies it to the GPU.
- Adds and resets graph counters and performs further reductions.

The affinity accumulator is `(layers - 1, experts, experts)` float32. At 48 layers and 512 experts it occupies 47 MiB per phase. `_registers_for()` at line 1561 allocates it on first use of a phase, not every forward; the matrix update recurs every observed forward. The ordinary accumulation body avoids device-to-host synchronization, but still incurs Python dispatch, temporary allocations, a small H2D transfer, and GPU work.

This work runs even when no metrics file is configured. Exact activation matters:

- Dynamic hot cache requires a real recorder configured as `stat` or `per_pass` (`python/sglang/srt/arg_groups/memory_hook.py:255`).
- A static hot cache does not automatically enable a real recorder. The noop recorder ignores observer registration.
- With a real recorder and cache observer, observer callbacks run even when user-facing recording is off. See `python/sglang/srt/eplb/expert_distribution.py:227` and observer registration in `python/sglang/srt/model_executor/model_runner.py:850`.
- The recorder resets count buffers before model execution, then collects counts and invokes the cache observer after model execution. The context around `_forward_raw()` is at `model_runner.py:1906`.

**Recommendation:** separate residency-required counters from optional affinity/history collection. Gate the dense affinity update independently while preserving consumers such as legacy affinity-based prefetch and explicitly requested route-history output. Reuse the small pinned mask and device buffer, with safe reuse ordering, or eliminate that mask upload for graph-only forwards. Prewarm required statistics allocations before measuring steady state.

**Verification:** compare recorder-active static and dynamic configurations with diagnostics enabled/disabled; confirm required residency behavior is unchanged. Trace the entire forward boundary, including post-replay kernels and CPU allocation activity. Do not attribute this cost to every static-cache configuration.

### 2. Periodic metrics perform synchronous readbacks

At the hot-cache logging interval, `expert_hot_cache.py:1895` loops over prefetch targets and calls each target's `PullDeliveryStats.snapshot()`. That method uses `.cpu().tolist()` in `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py:128`. These per-target readbacks happen before `_write_trace()` checks whether a metrics output file exists.

With metrics output configured, additional work includes counter readbacks, popularity/affinity top-k, CPU conversion, Python dictionaries, JSON encoding, and file append (`expert_hot_cache.py:1623` through `_write_trace()` and `snapshot_counters()`). Prediction metrics and calibration have separate periodic snapshots as well.

**Recommendation:** store per-layer counters in a contiguous bank and read them back together. Gate snapshots according to actual consumers, including callers that need in-memory snapshots without a file. If moving serialization off the serving path, hand the worker a completed, owned CPU snapshot; do not let it race mutable GPU or pinned buffers. Preserve cumulative counters, phase attribution, and capture/warmup reset semantics.

**Verification:** compare median and tail token latency across logging boundaries. Confirm a snapshot is neither duplicated nor lost after reset. Batch readbacks before considering more complicated asynchronous snapshot machinery. Periodic spikes are a hypothesis until measured.

### 3. Eager fallback waits for GPU results before transferring experts

`python/sglang/srt/layers/moe/expert_stream.py` dispatches supported shapes to `_gather_graph()` before the eager checks. The following barriers therefore belong to eager fallback, not ordinary captured decode:

| Location | Behavior |
|---|---|
| `gather():1188` | Bounds validation through `.any().item()` |
| `_gather_cached():899` | Hit-count and ID reads through `.item()`/`.tolist()` |
| `_copy_source_rows():855` | DMA path converts GPU-selected IDs to a Python list |
| `_copy_indices_to_cpu():483` | Copies IDs to pinned CPU storage, then explicitly synchronizes the current stream |
| `_gather_pinned_host():1044,1081,1094,1100` | Multiple hit/residency/cold predicates or counts read back to the CPU |

Boolean indexing and `nonzero` also create variable-size outputs in eager paths. CPU-managed cache handling performs Python lists, loops, admission decisions, and temporary tensor construction around these results.

The large GPU staging, pinned staging, and pinned index buffers are already cached: `_staging_buffer():424`, `_pinned_staging_buffer():457`, and `_copy_indices_to_cpu():483` allocate initially or on growth. Do not report fresh full-payload allocations on every transfer.

**Recommendation:** first measure how frequently real serving enters eager fallback. Extend device-count planning and mapped-host gathering to supported eager shapes where worthwhile. Consolidate unavoidable CPU readbacks into one summary when CPU management is required, and reuse classification results. Preserve validation, CPU visibility, deduplication, and source lifetime. Preallocate for expected maximum shapes to avoid first-use/growth latency.

### 4. The captured path has redundant GPU operations and intermediates

At the audited revision, `_gather_graph():669` has these opportunities:

- `topk_ids.reshape(-1).long()` performs a dtype conversion when IDs are int32, although the fused planner accepts int32. Preserve native dtype on the fused path while satisfying the generic path's contract.
- `plan_graph_routes_fused()` creates `remap_out` with `torch.empty` during eager execution/capture. A persistent output can simplify ownership and eager allocation behavior, but does not by itself remove a replay allocation that was never occurring.
- Prefetch join recomputes residency masks, coverage, statistics, and remapping after the planner already knows coverage.
- Optional calibration performs another residency lookup and its own tensor chain.
- GPU-resident weight components use an allocating `index_select` followed by `index_copy_` into scratch (`expert_stream.py:745`). The selected vector includes the plan's valid tail, rather than restricting all work to the device-side residual count.
- The generic planner constructs sorting, masks, reductions, and remap intermediates, then copies outputs into persistent plan tensors. It remains necessary for unsupported fused shapes and duplicate routes.

The newer fused planner API at this revision accepts real prefetch count and outcome counters, but the inspected serving gather call does not yet pass those new arguments; its join still performs separate bookkeeping. Re-check this integration before editing, because it is under active development.

**Recommendation:** wire planner-owned accounting into serving and retain the completion wait at the fast-path join. Pass the real posted count, avoid duplicate counter updates, and use the planner's final remap. Consider a direct indexed D2D kernel that writes final slots and honors the device count for GPU-resident components, preserving their completion ordering. Keep the generic deduplicating path correct.

**Verification:** count actual captured nodes and trace their durations rather than counting Python expressions. Check int32/int64 routes, count zero, successful and wrong predictions, mixed host/device components, and duplicate routes. Compare physical transfer counts and byte-exact weights, not only final model output.

### 5. Predictor preparation and diagnostics add captured work

`python/sglang/srt/layers/moe/expert_prediction/serving/scorers.py:24`, `LlaporScorer.forward()`, builds dense zero-filled mask and route tensors, scatters route features, concatenates inputs, and evaluates the predictor. Candidate selection then adds score reduction, finite/residency masking, sorting, selection, and bank copies. Posting adds validity operations and ID/count writes.

These allocations occur during eager execution/capture. Their recorded initialization and computation recur during replay. Persistent allocations alone do not eliminate the fills or scatters.

`SGLANG_MOE_EXPERT_PREFETCH_SHADOW_RECALL` defaults to true in the audited revision. When prediction scoring is enabled, recall instrumentation adds comparison/reduction/counter operations. Calibration is separately default-off and adds more work when enabled. FeatureStore's buffers are already persistent, but the captured feature copies still execute.

**Recommendation:** identify diagnostic settings in every result and measure production-like diagnostics-off execution separately. After the serving fusion is connected, consider fused feature preparation and eligible top-1 selection. Preserve wider bank outputs, score margins, and recall semantics when those consumers are enabled. Measure predictor compute independently from preparation/control overhead.

### 6. Other Python work executes after graph replay

`model_runner.py:1926` calls prediction `on_forward_end()` after model execution. Unlike the captured feature hooks, this Python method executes per forward. It can perform configured shadow-predictor work and periodic metric/calibration output. See `python/sglang/srt/layers/moe/expert_prediction/runtime.py:326`.

Cache observer work then includes host boundary-clock updates, per-layer inspection of eager statistics, and, depending on configuration:

- Polling pending asynchronous promotion tickets and publishing completed slot metadata (`expert_hot_cache.py:1478`, `:1818`).
- Host residency updates that copy scores to CPU, run NumPy decisions, and submit promotions (`expert_hot_cache.py:1493`, `expert_residency.py:469`).
- Synchronization required before reusing an in-flight pinned metadata-upload buffer.

The GPU residency updater avoids normal graph-decode host score readbacks, but adds its own captured GPU policy work. A device gate that prevents a promotion does not automatically prevent all preceding decision kernels from executing. Measure this separately from host-policy overhead.

With the doorbell backend, the scheduler calls the synchronized fail-stop check after each forward (`python/sglang/srt/managers/scheduler.py:4610`; `expert_hot_cache.py:1407`). It synchronizes even in the no-fault case; without a doorbell the manager returns immediately. This is intentional outstanding-DMA ownership protection. Removing it requires a replacement completion/ownership protocol, not simply deleting the wait. The doorbell's request-serving worker is native code, not a Python callback per expert request.

**Recommendation:** attribute ordinary bookkeeping, boundary work, metrics, and safety synchronization separately. Keep required residency semantics while gating optional diagnostic work. Do not mix backend changes or removal of failure protection into a Python cleanup.

### 7. Wrapper validation and object construction are lower-priority eager costs

The transfer and planner wrappers validate device, dtype, shape, and contiguity. `_check_graph_sources()` checks retained source pointers. There are also dictionary lookups, slicing/view objects, and `ExpertRowDelivery` construction in `expert_row_plan.py`.

These operations execute during eager calls/capture and do not rerun inside ordinary captured replay. Metadata inspection does not itself read a GPU value or force synchronization. Segment tables and their tensor references are already persistent.

**Recommendation:** if eager profiling shows material overhead, use a validated persistent launch object for immutable configuration while retaining checks at public/setup boundaries. Do not sacrifice rebinding/lifetime validation merely to shorten Python code. Optimize the larger costs first.

The doorbell fallback also performs `sub` and `mul` into an existing buffer in `python/sglang/kernels/ops/moe/expert_doorbell.py:522`. It does not allocate a fresh payload, but still records two GPU operations. Folding the arithmetic into resolve or fallback is a separate small fusion opportunity.

### Recommended execution order and audit evidence

1. Gate unnecessary dense affinity/history collection independently of required residency counters.
2. Eliminate or safely reuse the per-forward pinned mask and device temporary.
3. Batch periodic per-layer counter readbacks and gate them by consumers.
4. Finish serving integration of fused planner bookkeeping and native route dtypes.
5. Optimize predictor preparation/candidate selection with diagnostics explicitly controlled.
6. Optimize eager fallback based on its observed frequency and stalls.
7. Tune wrapper dispatch or object construction only if eager profiling warrants it.

Profile the entire model-forward boundary: recorder reset, graph launch/replay, prediction end hooks, observer bookkeeping, promotions, metrics, and scheduler safety checks. Include both steady-state forwards and logging/residency boundaries. Record host allocation activity, CPU self-time, GPU nodes, D2H/H2D traffic, stream waits, and end-to-end median/p95 token latency. Source auditing establishes where work exists; it does not establish the time saved by removing it. No implementation changes or performance validation are implied by this appended audit.
