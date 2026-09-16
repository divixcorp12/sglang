# Side-stream GPU expert pull and fused planning implementation handoff

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` when executing this handoff. Complete and review each independently testable stage before integrating the next. This document authorizes no production restart or deployment.

**Goal:** Reduce single-token expert request-to-use latency on PCIe Gen 3 ×16 by fusing route planning and overlapping a bounded, one-row GPU pull with useful computation.

**Architecture:** Keep the existing graph-safe GPU gather as the demand baseline. Add a single-warp route planner and a distinct side-stream GPU-pull backend, captured into the same decode graph with explicit fork/join dependencies. Begin with at most one predicted nonresident expert per target layer, dedicated speculative scratch, and a join before target-layer consumption.

**Tech stack:** SGLang, PyTorch CUDA graphs, existing TVM-FFI/JIT CUDA kernel wrappers, registered GPU-readable host expert storage, NVFP4 expert weights. Confirm installed versions at execution time; do not infer hardware/API support from documentation alone.

**Spec:** The user selected “properly connect side-stream GPU pull path + fused planning.” This handoff consolidates the conversation's review, one-row design, five latency recommendations, and implementation constraints. Supporting inputs are `MOE_EXPERT_TRANSFER.md`, the September 15 prefetch plan, and the experiment records listed below.

**Status:** Documentation only. No implementation, GPU benchmark, remote command, or production change was performed for this handoff. The original review inspected `a5d45d7068`; the documentation pass observed `9ad5783873` and newer experiment notes. Existing unrelated changes, including `scripts/expert_prediction/prefetch/price_prefetch.py`, were left untouched. Recheck the actual execution checkout before changing code.

## 1. Decision and priorities

I agree with the foundation, but would change the priorities. On PCIe Gen 3 ×16, the largest remaining gains are likely to come from **transferring fewer rows and starting useful transfers earlier**, rather than further tuning the copy kernel.

Priority order:

1. Fuse the single-token route planner and bookkeeping; this benefits ordinary demand gathers independently of prediction.
2. Prove a real multi-stream captured GPU-pull fork/join with synthetic device plans and byte-exact correctness.
3. Connect one-row prediction, fix candidate-selection parity, and measure scoring plus delivery together.
4. Keep doorbell DMA as a separate comparison arm. Improve its fixed overhead independently if worthwhile.
5. Investigate cache admission, promotion reuse, and global promotion budgets as separate changes.

Do not require the doorbell or a CPU spin thread for the selected GPU-pull backend. Do not bundle quantization changes, expert dropping, CPU expert execution, large speculative budgets, or a new residency policy into the initial implementation.

## 2. Evidence and its limits

### 2.1 Link geometry and measured baseline

The reviewed geometry is 48 layers, 512 experts/layer, top-k 10, and 2,764,800 host bytes per expert. Some older benchmark scripts use 2,764,808; derive actual transferred bytes from the source segments and report the difference rather than silently mixing constants. A token requests about 480 expert rows. With no hits, that is about 1.327 GB/token.

The recorded 12 GiB hot-cache configuration has 4,180 resident slots, with scratch charged against the budget. Across two serving workloads, demand misses were about 13.6–20.5%.

The transfer microbenchmark reported **12.081 GB/s GPU gather versus 13.313 GB/s copy-engine DMA**. Gather therefore achieved about 90.7% of the measured DMA rate. These are practical measurements, not a claim that either reaches the theoretical line rate.

| Quantity | Workload A | Workload B |
|---|---:|---:|
| Demand expert bytes/token | 179.9 MB | 272.5 MB |
| Demand miss rows/token | 65.067 | 98.577 |
| Transfer time at 13.313 GB/s, excluding overhead | 13.5 ms | 20.5 ms |
| Recorded total time/token | 35.5 ms | 43.8 ms |
| Calculated gather-to-DMA bandwidth-only saving | 1.4 ms | 2.1 ms |

The last row is calculated as `bytes × (1 / 12.081e9 - 1 / 13.313e9)`; it is not an observed end-to-end speedup. One absolute percentage point less demand miss rate saves approximately `480 × 0.01 × 2,764,800 / 13.313e9 = 0.997 ms/token` of link service time. Actual wall-time savings depend on overlap.

Source: [experiment log, M1/M2 and M3](../experiments/nvfp4-expert-offload-experiment-log.md), particularly original-review lines 1847–1851 and 1885–1907. M3's external PCIe receive rate and derived demand bytes differ by roughly 17–23%; the difference was not attributed. A 1 Hz throughput sample is not a measurement of transfer duty cycle. Two counters derived from the same row count are not independent validation of physical bytes.

### 2.2 Existing transfer and prediction findings

1. **Current doorbell requests have little lead time.** `ExpertStreamer._gather_graph` posts and immediately resolves the same-layer plan (`expert_stream.py`, originally lines 670–673). `model_runner.py` rejects doorbell `next_layer` mode. This is demand DMA substitution, not a completed predictive pipeline.
2. **One row is a sensible first budget.** Recorded windows are approximately 0.21–0.29 ms, depending on predictor placement and mixer kind. The historical gather fit is 0.2239 ms/row + 0.006 ms/launch; doorbell DMA was about 0.209 ms/row. Use the gather fit for GPU-pull pricing, not DMA's smaller number. Scoring, planner cost, capture scheduling, and contention reduce usable lead time.
3. **Concurrent H2D copies share the link.** Do not count a preceding layer's full demand-copy duration as free speculative bandwidth. The copy-versus-compute result in E27 does not establish copy-versus-copy independence.
4. **Large all-or-nothing budgets performed poorly in offline pricing.** Budget 1–2 was useful under some assumptions; larger batches became negative. Prefix pricing counts waiting through the deepest useful candidate, but any submitted speculative tail remains bus traffic. Current doorbell batches have whole-request completion, not per-row prefix completion.
5. **Timeout is not cancellation.** Doorbell drain can report undelivered with committed copies still outstanding. Present fallback writes the same expert bytes into the same slots and fences before another forward. That does not establish safety for overwriting those slots with different speculative residual experts within the same forward. This is a future integration hazard, not proof of current demand-path corruption.
6. **Promotions can transfer an expert twice.** GPU residency copies promoted host rows on the current stream across layers; demand scratch is not reused as the promotion source. An all-layer update can create a blocking burst. Host and GPU residency are alternative policy owners, not necessarily simultaneous duplicate promotions.
7. **Candidate selection differs offline and live.** Offline masks residents before top-k; the live candidate bank truncates first and filters residents later. A useful nonresident candidate beyond the retained width can be lost. Correct this or measure an adequate width; do not assume offline recall transfers unchanged.
8. **Empty doorbell requests still incur host service.** Add a device-only bypass if working on that backend.

### 2.3 Scorer economics: preserve the original finding, incorporate newer evidence

The [offline gate](../experiments/2026-09-15-expert-prefetch-offline-gate.md) reported these budget-1, zero-scorer-cost, all-or-nothing savings: LLaPor 5.568 ms/token and APEX 6.714 ms/token. Its best budget-2 prefix cells were 7.085 and 7.251 ms/token. These are model outputs, not live delivery results.

The original review noted that roughly 50–60 µs/layer scoring was needed to retain the **3 ms/token acceptance target** in those best prefix cells, against an older estimate of 100–190 µs/layer. That threshold is not the point of zero benefit, nor a guarantee for this proposed GPU-pull backend.

Newer [live shadow evidence](../experiments/2026-09-15-expert-prefetch-live-shadow.md), added after the original review, reports:

| Arm | Extra ms/token versus S0, by pass | Budget-2 live recall |
|---|---|---|
| LLaPor | 2.85, 1.34 | 0.383, 0.389 |
| APEX | 3.20, 3.32 | 0.424, 0.420 |

Those runs enabled scoring but **no prefetch copies**. They are end-to-end shadow deltas, not isolated scorer timings. LLaPor's within-arm spread is substantial; S0 has only one pass. Do not present the mean deltas as precise kernel cost or as net prefetch benefit.

The older capture had about 3.15 misses/layer; separate serving measurements had about 1.36–2.06. Reprice by actual workload miss rate, precision, and total traffic. At budget 2, 48 targets would offer 96 rows, about 265 MB/token before adding residual misses. A one-row budget offers at most about 133 MB/token across 48 targets, or about 130 MB for 47 next-layer targets; useful prefetched rows replace demand bytes, wrong predictions add bytes.

The September 15 plan's later notes explicitly warn that earlier threshold tables omit delivery cost. Read those additions before reusing a number. The decisive measurement is **scoring ON + delivery ON versus a matched baseline**, with cache capacity and scheduler settings recorded.

### 2.4 Correct the transfer overview during implementation

`MOE_EXPERT_TRANSFER.md` needs these corrections:

- GPU residency has its own default-off `SGLANG_MOE_GPU_RESIDENCY_UPDATE` flag.
- Async promotion publication has its own default-off `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` flag; a separate copy stream alone does not imply the caller avoids waiting.
- The old eager prefetch coordinator can invoke actual copies through callbacks. Graph gather rejects that integration. The new `expert_prediction/serving` path is the scoring-only path.
- Current-layer doorbell post/resolve does not itself provide next-layer lead time.
- The existing in-graph backend performs the copy at `post` on the current stream. A new side-stream backend must be explicit and separately named.

## 3. Why GPU pull works without a CPU responder

Pinned memory remains system RAM. It must also be GPU-addressable/mapped; pinning alone is not the complete API contract. The current registered expert arena passes `is_gpu_readable_host_tensor` validation and supplies device-accessible addresses.

`expert_cache_transfer.cuh` already performs `ld.global.nc` host loads and device stores. GPU memory transactions cross PCIe without CPU software servicing each request. CPU hardware/memory controllers participate, but no Python or CPU worker must run per expert.

The doorbell worker exists to submit dynamically chosen host-to-device transfers through the **copy engine**. It converts row IDs into DMA commands; it does not copy expert payloads through a CPU staging buffer. Ordinary CUDA kernel code has no general drop-in host `cudaMemcpyAsync` equivalent for this use case.

GPU pull uses SM execution resources; DMA uses the copy engine. The selected design accepts that resource tradeoff to avoid CPU handoff and uses a side branch in the decode graph to overlap pulling with compute. Concurrent execution is possible, not guaranteed: graph dependencies and resource pressure still decide actual overlap.

## 4. Scope and invariants for the initial implementation

- NVIDIA CUDA, one GPU, TP=PP=DP=1, one-token ordinary decode initially. Preserve existing fallback for unsupported shapes, prefill, and speculative verification. Do not silently route duplicate IDs through a unique-ID kernel.
- Provide separate **new default-off flags**, proposed names `SGLANG_MOE_EXPERT_FUSED_PLAN` and `SGLANG_MOE_EXPERT_GPU_PULL_PREFETCH`. Fused planning can run alone; the first GPU-pull prefetch implementation requires fused planning. These names do not exist yet; verify no collision before adding them to `environ.py`.
- Side-stream prefetch requires fused planning, graph gather, GPU-readable stable host sources, prediction enabled, and the new reserved scratch layout. Reject GPU-pull prefetch with fused planning disabled: the existing general residual helper cannot consume the appended speculative slot. Initially also reject simultaneous doorbell selection and GPU-pull prefetch with a clear startup error; keep the doorbell comparison in a separate arm.
- One plan entry per target layer, `count` int32 `[1]` with value 0 or 1, expert ID int64 `[1]`, destination int32 `[1]`. Plans and sources have stable addresses for the graph lifetime.
- No per-token `.item()`, `.tolist()`, `.cpu()`, host event queries, CPU callbacks, Python data-dependent branching, allocation, or host synchronization in the captured path.
- Plan count and ID are overwritten every applicable forward; stale prediction state cannot survive an absent feature update or another graph instance.
- Use the live residency map after any update. Freeze/validate map addresses at capture and recapture if ownership rebinds them; never retain an obsolete tensor accidentally.
- Preserve exact expert weights and routing. No dropped experts, approximate substitutes, changed top-k weights, or new accumulation order as a speed shortcut.
- All side work must join the origin capture before graph completion. No expert source, destination, plan, or feature buffer may be freed or overwritten while a side consumer still uses it.
- Start with no overlapping replays sharing state. Each concurrently executable graph/request must have its own state, or execution must be serialized with explicit dependencies.
- Keep authoring and independent review separate. Do not claim passing tests, safe deployment, or speedup without fresh evidence.

## 5. Implementation map

All paths below are repository-relative to `/home/dimitri/data/divix/sglang-nvfp4`.

| Existing file | Responsibility and expected change |
|---|---|
| `python/sglang/srt/layers/moe/expert_route_plan.py` | Reference planner and specialization dispatch; retain general dedup path. |
| `python/sglang/srt/layers/moe/expert_row_plan.py` | Persistent plans; preserve old backend semantics; adapt residual routing for the new dedicated slot. |
| `python/sglang/srt/layers/moe/expert_stream.py` | Fused demand planning, consumer join, residual gather, source-layer launch hook after demand copies. |
| `python/sglang/srt/layers/moe/expert_hot_cache.py` | Budget/allocate extra scratch, own pipeline lifetime, wire per-layer hooks, report exact resident capacity, and migrate derived byte counters to offered-plus-residual traffic. |
| `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py` | Expose current full scores/feature readiness to the adapter without copying weights here. |
| `python/sglang/srt/layers/moe/expert_prediction/serving/candidates.py` | Residency-before-truncation selection or full-score adapter; preserve independent shadow metrics. |
| `python/sglang/srt/model_executor/model_runner.py` | Validate configuration and orchestrate setup only. |
| `python/sglang/srt/environ.py` | Independent default-off feature gates. |
| `python/sglang/kernels/ops/moe/expert_cache_transfer.py` | Reuse `ExpertRowSegments` and device-count copy wrapper. |
| `python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh` | Existing bandwidth-oriented pull; retain initially. |
| `python/sglang/kernels/ops/moe/expert_doorbell.py` and corresponding `.cuh` | Separate optional latency work, not a dependency of side-stream pull. |

Proposed new files, deliberately small:

- `python/sglang/kernels/jit/csrc/moe/expert_route_plan.cuh`: fused unique-route planner and one-row candidate selection kernels.
- `python/sglang/kernels/ops/moe/expert_route_plan.py`: validated JIT wrappers, following the transfer wrapper's loading conventions.
- `python/sglang/srt/layers/moe/expert_gpu_pull.py`: persistent side-stream, events, target plans, lifecycle and capture integration.
- `test/registered/unit/kernels/test_expert_route_plan_fused.py`: kernel/reference equivalence.
- `test/registered/unit/layers/moe/test_expert_gpu_pull.py`: fork/join, lifecycle, routing and delayed-copy cases.
- `benchmark/kernels/moe/benchmark_expert_request_latency.py`: staged latency and overlap measurements using real row geometry.

Before executing, read applicable instructions and the repository's environment-variable and large-class conventions. Existing tests useful as oracles include `test_expert_route_plan.py`, `test_expert_stream.py`, `test_expert_residency_gpu.py`, `test_expert_prefetch_scoring.py`, `test_expert_prefetch_runtime.py`, `test_expert_cache_transfer.py`, and `test_expert_doorbell_copier.py` in their existing `test/registered/unit` subdirectories.

## 6. Fused planning design

### 6.1 What to remove

For ten unique IDs, the current graph planner performs a general stable sort/dedup, cumulative max, scatter, prefix sum, another sort, gathers, reductions and casts. Approximately 25–30 source-level tensor operations are involved; this is not a measured launch count. `fill_routes` adds two copies. `_gather_graph` adds counters, optional residency score updates, and possible dtype conversions.

Produce persistent outputs directly in one warp kernel. Keep the ordinary route kernel separate from the bulk transfer initially; a one-block planner's writes cannot safely be consumed by other CTAs in the same ordinary kernel without a valid global synchronization design.

Proposed Python wrapper contract (new API, not existing code):

```python
def plan_unique_routes_cuda(
    topk_ids, expert_to_slot, scratch_base,
    source_rows_out, slots_out, count_out, remap_out,
    graph_counters, graph_unique_counters, route_counts,
    prefetch_expert, prefetch_count, prefetch_slot,
) -> None:
    """Write a BS1, K<=32 plan into supplied stable CUDA buffers.

    route_counts is an optional per-expert float32 counter tensor.
    prefetch_expert is int64[1], prefetch_count is int32[1].
    Zero prefetch_count disables coverage; prefetch_slot is a fixed integer.
    Caller has joined prefetch completion before invoking this function.
    """
```

The first implementation supports the observed map dtype and both input/remap ID dtypes used by serving; dispatch at setup, not by copying device values to host. For baseline-only fusion pass permanent zero-prefetch state. Reject unsupported dtypes/shapes at setup and retain the general route path.

Conceptual CUDA algorithm; implement using the project's JIT conventions:

```cpp
unsigned lane = threadIdx.x;
bool active = lane < top_k;
int64_t expert = active ? load_id(topk_ids, lane) : 0;
int64_t slot = active ? expert_to_slot[expert] : -1;
bool hot_hit = active && slot >= 0;
bool prefetched = active && !hot_hit && prefetch_count[0] == 1
                  && expert == prefetch_expert[0];
bool residual = active && !hot_hit && !prefetched;
unsigned misses = __ballot_sync(0xffffffffu, residual);
unsigned earlier = (1u << lane) - 1u;
unsigned rank = __popc(misses & earlier);
unsigned total = __popc(misses);
```

Launch exactly one block of 32 threads per planner invocation; the wrapper must enforce this geometry. All 32 lanes participate in warp collectives. For each active route, write `slot` for a hot hit, the dedicated speculative slot for a covered miss, or `scratch_base + rank` for a residual. Stable-compaction output position is `rank` for a residual and `total + lane - rank` otherwise. This keeps residual experts first and all other IDs in original order, with a fully initialized, in-range tail.

The valid tail matters: `_graph_device_pairs` currently indexes the full source-row vector even beyond the host-copy count. Do not leave those IDs uninitialized. Preserve the complete reference vector in the no-prefetch arm, not just the active prefix.

Write int64 expert IDs, int32 destination slots/count, and remap in the consumer's required dtype. Each active lane can increment its distinct route counter once; use atomics where ownership/concurrency requires them. One lane updates scalar counters. Preserve separate meanings:

- `requested_routes`: all actual routes.
- `hot_hits`: actual routes resident in permanent cache.
- `demand_misses`: actual nonresident routes, including those served by prefetch.
- `prefetch_useful_rows`: actual nonresident rows covered by the joined prefetch.
- `residual_copy_rows`: nonresident rows not covered by prefetch.

Do not relabel a prefetched row as a hot-cache hit. For BS1, unique and routed counts coincide; preserve general-path distinctions elsewhere.

## 7. One-row side-stream GPU pull

### 7.1 Scratch layout and budget

Initially reserve a dedicated speculative slot per eligible target **inside each existing contiguous cache tensor allocation**:

```text
[0, capacity)                      permanent resident slots
[capacity, capacity + demand_rows) existing demand scratch
[capacity + demand_rows]           one speculative row
```

Keep `graph_gather_rows` equal to demand capacity, rather than accidentally enabling a larger batch because an extra allocation row exists. The speculative slot is never published into `expert_to_slot` and is never a residency eviction/promotion destination. The target's actual remap may address it directly after completion.

Across 48 layers, one additional row is about 126.6 MiB at the stated geometry. Reserve it from the same configured cache budget before calculating resident capacities. Scorer memory also consumes VRAM. Compare both (a) equal total VRAM budgets and (b) a baseline with matching resident-slot counts to separate storage effects from scheduling effects. Do not silently increase memory usage or drop an existing demand slot.

Existing `plan_residual_routes` assumes prefetched plan rows live at `scratch_base + r`. It is **not** a drop-in for this appended dedicated slot. Use the extended fused planner above for BS1; modify a general helper only if later supporting larger shapes, with explicit slot mapping tests.

### 7.2 Producer and launch placement

Begin with LLaPor's existing source-to-target mapping; do not assume numerical layer ID + 1 is always the next MoE layer. Read the actual checkpoint/source mapping. The first layer without a producer uses demand only; do not post beyond the final target.

LLaPor currently scores when source-layer top-k features are written, before that layer's demand gather. Keep scoring on its present stream for the first version and charge its cost. Once source L's own demand copies and any required device-resident tensor copies are complete, launch the one-row side pull for target T. This conservative placement avoids competing with L's demand copy.

Use full candidate scores to select the highest-scoring nonresident expert, with deterministic lower-ID tie-breaking and invalid-score masking. A configurable confidence threshold may produce zero rows, but raw model scores are not automatically calibrated probabilities. Initially benchmark fixed best-nonresident selection and separately evaluate a threshold on held-out data.

The target map used by selection must already reflect this forward's residency update. No promotions may race a speculative write. If a host async residency publication could change map storage mid-replay, require existing ownership/fencing or reject that combination until proven safe.

APEX can later use the same copy interface at its pre-mixer hook for the same target layer. Its different window and scoring cost need a separate measurement arm; do not charge it LLaPor's window or post at LLaPor's hook.

### 7.3 Capture the actual fork and join

Create one long-lived side stream and per-target timing-disabled events before warmup/capture. Build events and plans per independently executable graph state; do not share mutable buffers across overlapping graph replays. Capture the side work into the **same** graph, connected by events recorded within that capture. Merely entering a side-stream context around unrelated work is insufficient.

Illustrative integration using existing transfer APIs and preallocated state:

```python
def post_target(target, side_stream):
    origin = torch.cuda.current_stream(target.device)
    target.ready.record(origin)
    with torch.cuda.stream(side_stream):
        side_stream.wait_event(target.ready)
        copy_expert_row_segments_gpu(
            target.segments,
            target.plan.expert_ids,
            target.plan.slots,
            target.plan.count,
        )
        target.done.record(side_stream)

def join_target(target):
    origin = torch.cuda.current_stream(target.device)
    origin.wait_event(target.done)
```

Here `target` is persistent state owned by the proposed `ExpertGpuPullPipeline`: device, existing `ExpertRowSegments`, existing `ExpertRowPlan` of capacity 1, ready/done CUDA events, dedicated slot index, and lifetime references. The selection kernel fills the plan before `post_target`. These Python functions run during eager execution/capture; replay executes the captured dependency graph, not a Python per-layer callback.

If any expert-associated tensors are GPU-resident rather than host-backed, copy their selected row into the same speculative destination on the side stream **before** recording `done`. Completion covers every tensor the expert consumer needs. Skip when count is zero using device predicates or a fused segment kernel, without reading count on the host. Validate tensor row layout and alignment exactly as the demand path does.

Join after the target's router results are available and before its residual planner/weight consumer. The normal compute stream proceeds through source expert compute and target normalization, attention/mixer, shared expert and routing while the side pull runs.

```text
origin:  demand(L) -> post(T) -> expert_compute(L) -> mixer/router(T) -> join(T) -> residual(T) -> expert_compute(T)
                         \                                              /
side:                     ready -> one-row GPU pull -> complete --------
```

The origin must also join any remaining fork before capture ends, including target-disabled/model-tail cases. Capture a count-zero pull when the plan is empty: the kernel returns without payload traffic, but dependency nodes still execute and must be included in overhead measurements.

NVIDIA documents capture fork/join dependencies in [CUDA graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) and [capture constraints](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html). Capture stream names are not proof of physical concurrency on replay; inspect actual kernel intervals and dependency edges.

### 7.4 Consumption, mistakes and completion semantics

Version 1 **always joins the target's one-row side branch**, even if the actual routes do not use that prediction. This intentionally keeps the graph static and the ownership proof simple:

- Needed and already complete: route to the dedicated slot, copy only other misses.
- Needed and still running: join the existing pull; never issue a duplicate demand copy for it.
- Not needed: join completion, then use normal residual planning. Record one wasted row if a row was offered. Any remaining wait is real overhead and must be priced.
- Count zero: no useful or wasted row; normal demand handling after the no-op branch joins.

Do not use the doorbell's 0/1 delivered flag or timeout machinery for this backend. Stream dependency completion supplies visibility; count remains the device-side offered-row indicator. A device fault follows the framework's CUDA error handling; do not pretend an incomplete side kernel was cancelled and continue reading its destination.

Before the next replay writes this target's plan or destination, the prior consumer must have finished. Sequential origin-stream graph execution plus the internal join can establish this; concurrent replay requires separate state or explicit ordering. Source host memory is immutable and registered for the full graph lifetime.

Avoiding the join for wrong predictions is a later design, requiring a graph-supported conditional dependency or another proven ownership mechanism. Dedicated scratch alone prevents overwrite collisions; it does not remove outstanding bus traffic or ensure safe next-replay reuse.

### 7.5 Traffic model

For one target and the initial unconditional join:

```text
offered_rows = 0 or 1
useful_rows = offered row is an actual nonresident expert ? 1 : 0
wasted_rows = offered_rows - useful_rows
residual_rows = actual_nonresident_rows - useful_rows
total_host_rows = offered_rows + residual_rows + separately_counted_promotions
incremental_prefetch_rows = wasted_rows
remaining_join_wait ≈ max(0, actual_side_copy_duration - actual_overlap_window)
```

Net gain includes avoided demand service, remaining join wait, scoring/planning/event cost, compute contention, capacity changes, and downstream transfer contention. The simple window formula is a diagnostic approximation, not an end-to-end estimator. Do not subtract both a modeled wait and the same wasted-copy time again. Count bytes once, and measure the resulting critical path.

**Migrate the existing telemetry explicitly.** `expert_hot_cache.py` currently derives `h2d_bytes` from `unique_missed × host_bytes_per_expert` in its graph counter aggregation (around line 1677 in the reviewed source). This handoff deliberately retains logical demand misses, including prefetch-covered rows. Adding all offered bytes to the old derived value would therefore double-count useful prefetch. Extend the device counters/readback registers and metrics aggregation so transfer accounting uses `offered + residual`, while logical misses remain a separate metric. For a useful prediction with two actual misses, report one offered plus one residual row, not three transferred rows. For a wrong prediction with two actual misses, report one offered plus two residual rows. Attribute promotion bytes separately. Label all row-derived byte totals as derived rather than independent measurements of PCIe traffic.

With one row, all-or-nothing completion already has row granularity. Prefix delivery is unnecessary. Do not queue extra candidates because the side stream is idle; budget expansion requires new evidence.

## 8. The five latency recommendations, retained as implementable work

### 8.1 Fuse unique-route planning and bookkeeping — highest priority

Implement section 6. Expected benefit is fewer GPU nodes before the request and on all-hit layers. No numerical speedup is established. Every 10 µs removed per layer is 0.48 ms/token across 48 layers; use that only as arithmetic, not a forecast.

### 8.2 Eliminate empty requests and residual scalar kernels

For doorbell independently, `undelivered_count` currently launches a subtraction and multiplication. Let the final resolve state output residual count, or let fallback gather read `delivered` and `count` itself. Remove two GPU nodes per layer while preserving disabled/unserviced/drain-timeout behavior. Post should recognize count zero and take a device-complete state that resolve and residual understand without host publication. A captured no-op node is acceptable; a CPU round trip is unnecessary.

For GPU pull, count zero simply returns in the pull kernel. Do not add a host request protocol to implement the zero case.

### 8.3 Reduce fixed wait chains — doorbell comparison only

Current resolve launches growing polling chunks and a drain. Later nodes still execute after early success. E34e's old configuration estimated about 0.95 ms/token for its wait launches; remeasure current configuration rather than booking that as recoverable gain.

Benchmark fewer chunks, or one bounded wait plus bounded drain, on the required supplied stream. Preserve total timeout bounds and committed-copy ownership. Existing cold-launch experiments refute the claim that queued short chunks inherently let blocked copy submissions progress. An infinite persistent spinner is not the proposed optimization.

### 8.4 Reduce completion and CPU housekeeping — doorbell comparison only

Benchmark timing-disabled events, sampled tracing, preallocated pending storage, and an inactive-fault-injection fast path avoiding its mutex. Current `publish` adds an eight-byte H2D completion transfer and an event record. A stream-ordered completion write is an experimental alternative where supported; preserve memory-order guarantees and the actual `{sequence, serviced}` protocol. Consult [CUDA stream memory operations](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html) and query device support.

Host enqueue return is never completion. Do not weaken acquire/release ordering or drop the payload-before-completion fence to gain polling speed. The proposed side-stream pull avoids the entire host completion handshake, but still pays graph dependencies and kernel scheduling.

### 8.5 Copy semantics, layouts and placement — lower priority

Existing E32 experiments found no benefit from `srcAccessOrder=Any` for the hold and found `DuringApiCall` blocking. Batch versus per-segment API choice did not fix it. Preserve supplied PyTorch-created stream behavior for doorbell until contrary serving evidence exists.

Packing expert tensors into fewer contiguous copies might reduce descriptor/submission overhead, but requires compatible source **and destination** layouts. Do not call disjoint segment batching a single contiguous DMA. Repacking must preserve NVFP4 weight/scale strides consumed by existing kernels and be charged for padding and capacity. Benchmark before committing to a layout migration.

Verify effective worker affinity, isolation from competing threads, and NUMA locality of host arena/control pages if evaluating CPU-mediated DMA. Affinity alone does not establish page placement. For GPU pull, host arena locality remains relevant even without a worker. Do not tune affinity or NUMA remotely as part of writing this handoff.

## 9. Cache/promotion follow-ups outside the initial pipeline

Investigate promotion of a valid scratch expert by D2D copy or direct admission into a policy-approved resident slot, with expert ID, generation and lifetime checks. Count avoided host bytes; do not promote from stale scratch. GPU residency currently loops through layers on the current stream; apply a global row/byte/time budget if boundary latency is excessive. Its per-layer maximum of 64 is not a global bound.

Keep host and GPU policy ownership mutually exclusive. Separate residency scoring from transfer scheduling only with a clear publication rule. A speculative slot does not become a resident entry merely because its bytes arrived.

A rolling shared scratch pool could later recover memory, but current contiguous per-layer cache layouts and concurrent consumption make this a separate allocator/layout project. Do not alias buffers across layers without proving every consumer and side writer has completed.

## 10. Staged implementation and verification

### Stage A — fused demand planning, prediction disabled

**Files:** new route-plan wrapper/kernel and fused kernel test; modify `expert_route_plan.py`, `expert_row_plan.py`, `expert_stream.py`, `environ.py`.

**Interface:** section 6 wrapper; permanent count-zero prefetch input; unchanged bulk-copy backend.

- [ ] Add a CUDA equivalence test using IDs `[5, 2, 9, 1]`, map `2 -> 7`, `1 -> 3`, other IDs nonresident, scratch base 10. Expected source rows `[5, 9, 2, 1]`, count 2, remap `[10, 7, 11, 3]`, two hot hits and two demand misses. Verify the absent wrapper fails the test before implementation.
- [ ] Add all-hit, all-miss, K=1/10/32, int32/int64 routing, random unique IDs/maps, and map-change replay cases against `plan_graph_routes`. Keep duplicate/batched cases on the general reference path.
- [ ] Implement the warp algorithm, stable tail and direct counter writes. Add specialization dispatch with feature flag off by default.
- [ ] Run the focused route and stream tests; verify host-backed and GPU-backed tensor segments, graph capture and repeated replay with changing IDs.
- [ ] Benchmark planner-only and full gather latency at 0/1/3/10 misses. Record actual GPU node count from a trace. Have an independent reviewer check semantics before commit/integration.

Test body pattern once the proposed wrapper exists; `run_fused_case` is a new test helper that allocates all outputs/counters, supplies zero-prefetch state, invokes the section 6 wrapper, and returns tensors:

```python
def test_unique_plan_matches_reference():
    ids = torch.tensor([5, 2, 9, 1], device="cuda", dtype=torch.int64)
    mapping = torch.full((16,), -1, device="cuda", dtype=torch.int64)
    mapping[2], mapping[1] = 7, 3
    result = run_fused_case(ids, mapping, scratch_base=10)
    torch.testing.assert_close(result.source_rows, ids.new_tensor([5, 9, 2, 1]))
    torch.testing.assert_close(result.remap, ids.new_tensor([10, 7, 11, 3]))
    assert result.count.item() == 2
```

Host reads here are test assertions after synchronization, not production code. Define the helper in the test file; it is not an existing repository API.

### Stage B — captured side-pull primitive with synthetic plans

**Files:** new `expert_gpu_pull.py`, `test_expert_gpu_pull.py`, request-latency benchmark; reuse existing transfer wrapper/kernel.

**Interface:** persistent target state from section 7.3; `post_target` and `join_target`, with no predictor required by this test harness.

- [ ] Add a test that captures origin work, forks a device-count pull of one row, performs independent compute, joins, then verifies every destination tensor byte against source.
- [ ] Cover count 0 and 1, changing expert ID across at least 100 replays, target tags, graph teardown, and two separately allocated graph states. Ensure the test fails when the join dependency is removed by deliberate test mutation.
- [ ] Implement persistent stream/events and source lifetime ownership; add graph-tail join handling. Warm up all kernels before capture.
- [ ] Introduce a test-only bounded delay in the side branch and verify the consumer cannot see an incomplete row. This is a deterministic dependency test, not a wall-clock sleep assumption.
- [ ] Capture actual concurrent intervals with a compute workload representative of MoE decode; verify physical overlap rather than inferring it from stream creation. Compare against the same operations serialized. Review before integration.

### Stage C — one-row selection and serving integration

**Files:** prediction serving runtime/candidates, `expert_gpu_pull.py`, `expert_hot_cache.py`, `expert_stream.py`, `model_runner.py`, `environ.py`; scoring/runtime/stream tests.

**Interface:** full current scores and source-target mapping feed a device selection kernel; outputs the existing capacity-1 `ExpertRowPlan`. The target fused planner accepts that joined plan and dedicated slot.

- [ ] Test resident-before-selection with the highest 16 scores all resident and expert 17 the best nonresident candidate. Expected candidate is 17 rather than an empty truncated bank. Include all-resident, invalid-score and deterministic tie cases.
- [ ] Reserve the dedicated slot within budget; prove it is excluded from permanent mapping and demand scratch. Verify resident-capacity accounting and model tensor shapes.
- [ ] Wire the source post after its demand copies, target join after actual routing, and covered/residual remap through the fused planner. Keep scoring-only mode available for comparison.
- [ ] Test actual misses `{42,117}` with predicted 42: only 117 crosses demand path. With predicted 93: both actual misses cross demand path and 93 counts as waste. Test count zero, actual all-hit, delayed correct prediction, delayed wrong prediction, and first/last target behavior.
- [ ] Migrate `expert_hot_cache.py` graph telemetry/register aggregation away from treating logical misses as physical demand-copy rows. Assert the useful-prediction example reports two total host rows and the wrong-prediction example reports three; keep logical demand misses at two in both. Verify promotion bytes are separate and every counter name states its derivation.
- [ ] Verify every expert-associated tensor, including device-resident components, comes from the correct expert. Assert original top-k weights and downstream math are unchanged.
- [ ] Exercise alternating eager prefill/decode and graph recapture. Reset/overwrite candidate state so missing hooks cannot reuse previous-forward predictions. Reject unsupported concurrent-state sharing at setup.
- [ ] Run focused integration tests and independent review before live evaluation. Do not enable by default based on synthetic overlap alone.

### Stage D — matched end-to-end evaluation

**Files:** request-latency benchmark and a new dated experiment record; update `MOE_EXPERT_TRANSFER.md` only after implementation behavior is verified.

- [ ] Run a fixed-route microbenchmark at 0/1/3/10 misses using real tensor geometry, registered host memory, byte validation, and serialized versus captured side-pull arms.
- [ ] Measure planner time, request-ready to pull-start, pull duration, join stall, residual demand time, and consumer-ready time with GPU timestamps/trace intervals. Keep CPU issue durations on their own clock; do not subtract Python-observed event time from worker timestamps.
- [ ] Run the matrix in section 11, interleaving/repeating baseline arms and holding prompt set, generation settings, cache initialization, memory budget, scheduling and residency policy fixed.
- [ ] Evaluate useful/wasted bytes, miss rate, cache slots, p50/p95 token latency and full-answer correctness. Distinguish steady decode from prefill and residency boundaries.
- [ ] Publish a result for fused planning independently of prediction. Accept a pipeline only if scoring+delivery gives repeatable net improvement beyond run-to-run spread and meets the stated latency/correctness gate.

### Stage E — optional doorbell overhead changes

Implement section 8.2–8.5 in separate commits/arms after the chosen path is measurable. Each change must pass the existing timeout, stalled-thread, disabled, late-completion and teardown tests. Do not remove a fallback test because the optimized happy path no longer uses its old kernel sequence. Keep the original backend available for A/B and rollback.

## 11. Measurement matrix and acceptance criteria

| Arm | Planner | Scoring | Delivery |
|---|---|---|---|
| A | Current | Off | Current-stream GPU demand pull |
| B | Fused | Off | Current-stream GPU demand pull |
| C | Fused | On | Shadow only, no speculative transfer |
| D | Fused | On | One-row captured side-stream GPU pull + residual demand |
| E | Fused, if compatible | Off | Current-layer doorbell DMA comparison |

Add separate LLaPor and APEX C/D arms when their proper hooks are implemented. An oracle plan is valid only in controlled fixed-route benchmarks or trace replay with future routes explicitly supplied; never present it as a causal serving predictor.

Record per arm: exact commit, flags, hardware/link state, CUDA/PyTorch versions, row bytes, source memory registration, resident slots, demand/speculative scratch bytes, scorer bytes, scheduler settings, update policy, prompt IDs and completion counts. A doorbell arm requiring different scheduler settings must include a matching scheduler control; do not attribute that difference to transfer mechanics.

Metrics:

- End-to-end decode ms/token and tok/s; p50/p95 per-token latency and boundary-step latency separately.
- GPU request/planner/transfer/join/residual intervals and compute slowdown under overlap.
- Offered/useful/wasted/residual/promotion rows and bytes; useful precision and miss coverage separately.
- Actual host segment bytes, external PCIe measurements where available, and the definition of every derived counter.
- Graph nodes, capture compatibility, peak VRAM, warmup/startup cost, and absence of per-token host synchronization.

Correctness gates:

- Exact bytes and remaps for deterministic low-level tests, including delayed-copy and state-reuse cases.
- Expected equality of route decisions/weights under fixed inputs; no intentional approximation.
- Serving logprob/answer checks with repeated same-arm controls. Existing full-model runs have nondeterminism; do not demand impossible bitwise output-token identity or use that observation to excuse a row corruption.
- Report truncation separately. The prior answer gate compared completed answers when both finish reasons were `stop`; incomplete answers are neither a correctness pass nor evidence of a wrong numerical answer.

Performance gate: predeclare the threshold for the new experiment. The older 3 ms/token criterion is a historical gate, not an established achievable target for side-stream pull. At minimum require a repeatable net gain beyond baseline variability, no unacceptable p95 regression, and a complete traffic/capacity accounting. A fused-only win can be accepted even if prediction fails its gate. Do not force the pipeline to ship because this handoff recommends testing it.

## 12. Commands and operational boundaries

This handoff changes documentation only. Before execution, re-read the current checkout's instructions, verify `pwd`, `git status --short --branch`, and relevant handoff/state files. Do not overwrite concurrent edits. Coordinate file ownership before parallel implementation; planner work and independent benchmark work can proceed separately, but integration depends on reviewed interfaces.

Use the repository's SGLang environment, not mechanism_hunter's pytest configuration. Example focused commands **from the SGLang root after activating its validated environment**:

```bash
python -m pytest -q test/registered/unit/layers/moe/test_expert_route_plan.py test/registered/unit/kernels/test_expert_route_plan_fused.py
python -m pytest -q test/registered/unit/kernels/test_expert_cache_transfer.py test/registered/unit/layers/moe/test_expert_gpu_pull.py
python -m pytest -q test/registered/unit/layers/moe/test_expert_stream.py test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_prefetch_scoring.py test/registered/unit/layers/moe/test_expert_prefetch_runtime.py
```

New test paths only exist after their stage is implemented. Confirm collection and expected GPU execution; a GPU test skipped because CUDA is unavailable is not passing GPU evidence. Doorbell changes additionally require `test/registered/unit/kernels/test_expert_doorbell_copier.py` and its existing operational constraints.

For divix01, read and follow the current `divix01-pilot` skill before any remote action. Existing experiment plans serialize GPU work with `/data/models/slang/nvfp4-work/cc-gpu.lock` and reserve production management to its owner. Reconfirm those arrangements; this document is not permission to stop/relaunch a service. Do CPU/static work while GPU access is unavailable. No deployment, pushes, or production mutation are required to deliver this handoff.

## 13. Evidence index for the next engineer

- `MOE_EXPERT_TRANSFER.md`: original overview; corrections in section 2.4.
- `python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh`: direct GPU host loads and fixed-grid, device-count transfer.
- `python/sglang/srt/layers/moe/expert_route_plan.py`: reference deduplication/compaction semantics.
- `python/sglang/srt/layers/moe/expert_row_plan.py`: plan lifetime, in-graph synchronous backend, residual mapping assumptions.
- `python/sglang/srt/layers/moe/expert_stream.py`: actual post/resolve, planning, bookkeeping and device-pair copies.
- `python/sglang/kernels/jit/csrc/moe/expert_doorbell.cuh`: host handoff, wait/drain ownership, batch issue and completion.
- `python/sglang/srt/layers/moe/expert_residency_gpu.py`: current-stream promotion loop and map mutation.
- `python/sglang/srt/layers/moe/expert_prediction/prefetch_pricing.py`: offline assumptions; distinguish all-or-nothing and optimistic prefix.
- `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md`: E27/E28 overlap windows, E32 stream/API experiments, E34/E35 stalls and launch overhead, M1–M3 bandwidth and serving measurements.
- `docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md`: original budget/scorer model.
- `docs/superpowers/experiments/2026-09-15-expert-prefetch-live-shadow.md`: newer live shadow evidence and limitations.
- `docs/superpowers/plans/2026-09-15-moe-expert-prefetch-live.md`: existing integration work, newer traffic/break-even caveats, model-specific hook context.
- [NVIDIA CUDA graph documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): fork/join capture and dependency semantics.
- [NVIDIA asynchronous execution documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html): streams and priority limitations; priorities do not guarantee preemption or execution order.
- [NVIDIA CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): pinned memory and copy/compute overlap.

**Completion deliverables for implementation:** reviewed code with independent flags, focused correctness evidence, a real captured overlap trace, reproducible A/B results including scoring and delivery, explicit scratch/VRAM accounting, and an updated transfer overview accurately separating supported paths from proposed ones.
