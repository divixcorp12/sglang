# NVFP4 residency and shared-transfer implementation plan

> Prototype plan: use focused checks and defer exhaustive edge-case work until
> correctness and performance are demonstrated on divix01.

## Objective

Combine a KTransformers-inspired activation-frequency residency policy with a
HiSparse-inspired fixed transfer plan and one expert-transfer stream per GPU.
Preserve the existing six-tensor ModelOpt NVFP4 gather and its file-backed and
pinned-host fallbacks. Support hit-only CUDA graph replay with eager miss
recovery.

## Task 1: Preserve the laptop mirror and repeatable synchronization

Files:

- Create `scripts/sync-divix01.sh`.
- Maintain the mirrored files under `python/sglang/srt/layers/moe/`.
- Mirror this design and plan into the remote SGLang documentation tree.

The script provides `status`, `pull`, and `push`, uses `scp`, reports SHA-256
hashes, and creates timestamped destination backups. It never controls the
server.

## Task 2: Introduce the fixed transfer control plane

Files:

- Create `python/sglang/srt/layers/moe/expert_transfer.py`.
- Modify `python/sglang/srt/layers/moe/expert_prefetch.py`.

Implement a device-scoped `FixedRowTransferPlan` with stable source,
destination, count, and generation tensors. Implement one shared
`AsyncExpertTransferExecutor` per CUDA device with a bounded ticket/event ring.

The executor owns ordering and completion, while callers provide the six copy
operations. Replace per-layer streams and the single in-flight tuple in the
prefetch coordinator. Do not change transfer kernels in this task.

Focused verification:

- Compile the two modules.
- Exercise ticket allocation, completion, and ring reuse with mocked CUDA
  events where practical.

## Task 3: Make cache-slot lifecycle generation-safe

Files:

- Modify `python/sglang/srt/layers/moe/expert_hot_cache.py`.
- Modify `python/sglang/srt/layers/moe/expert_prefetch.py` only through the
  interface agreed in Task 2.

Add explicit `FREE`, `RESERVED`, `LOADING`, and `READY` slot state plus a
generation per slot. Reserve victims before transfer, publish expert-to-slot
mappings only after ticket completion, and defer reuse until the last consumer
event completes.

Focused verification:

- State transition checks.
- A stale ticket cannot publish or retire a newer slot generation.
- Six-tensor readiness is atomic from the consumer's perspective.

## Task 4: Add activation-aware residency policy

Files:

- Create `python/sglang/srt/layers/moe/expert_residency.py`.
- Modify `python/sglang/srt/layers/moe/expert_stream.py`.

Record exact routing counts without synchronizing GPU values into Python on the
decode hot path. At request or prefill boundaries, apply configurable decay,
select the desired HBM set within budget, and apply promotion hysteresis.

Submit background promotions through the shared transfer coordinator at lower
priority than exact misses. Keep the current exact gather as the correctness
authority. Disable identity-style speculative prefetch by default while leaving
it available for controlled comparisons.

Focused verification:

- Deterministic ranking from synthetic activation counts.
- Hysteresis prevents churn near the cutoff.
- Exact misses outrank background promotions.

## Task 5: Establish the CUDA graph fast-path boundary

Files:

- Modify `python/sglang/srt/layers/moe/expert_stream.py`.
- Modify the narrowest existing breakable-graph integration point only if the
  stable-pool adapter requires it.

Make the captured path cache-hit-only with fixed buffers and on-device ID-to-slot
mapping. Keep transfers, mapping publication, and eviction between graph
replays. Route a miss through an explicit eager recovery path without changing
pool addresses or requiring graph recapture.

Focused verification:

- Search the captured path for `.item()`, `.tolist()`, dynamic allocation, and
  variable-length Python construction.
- Run one small hit-only capture/replay probe on divix01 after explicit approval
  to launch the server.

## Task 6: Compare transfer kernels and collect metrics

Files:

- Add a focused benchmark under the SGLang benchmark/test tree.
- Extend the existing expert metrics output.

Compare the existing Triton row gather, HiSparse planned row copy, and contiguous
copy for actual NVFP4 tensor sizes and miss counts. Record transfer bytes,
ticket latency, pinned/file hit rates, cache hit rate, promotions, evictions,
stale-ticket rejects, and eager graph fallbacks.

Adopt a different row-copy kernel only if it improves or matches end-to-end
behavior. Keep high-volume metrics out of standard server output.

## Integration order

1. Land Task 2's neutral transfer interface.
2. Integrate Task 3's safe slot lifecycle.
3. Add Task 4's policy without enabling background promotion by default.
4. Enable and measure policy-driven promotion.
5. Add the Task 5 graph fast path after hit/miss correctness is demonstrated.
6. Use Task 6 measurements to choose the final transfer kernel and budgets.

## Prototype implementation status

- Tasks 1-3 are implemented.
- Task 4's device-side route accounting, decay, deterministic selection,
  hysteresis, and boundary materialization are implemented. Boundary migration
  remains synchronous so the existing exact gather stays authoritative.
- Priority-aware background submission is deferred until exact-demand misses
  also enter the shared executor. The current executor enqueues immediately in
  FIFO order, so merely attaching a priority enum would not provide real
  scheduling priority. The next narrow step is a per-cache pending-transfer
  registry plus executor admission for both exact and background work.
- Tasks 5-6 remain deliberately deferred until the first live correctness and
  performance run is approved.

## Runtime safety

Implementation and static checks do not authorize starting or stopping the
server. Obtain fresh confirmation before a divix01 runtime launch.
