# NVFP4 activation-aware residency and transfer design

## Goal

Serve Qwen3.8 Flash Next ModelOpt NVFP4 on a single constrained Blackwell GPU
while preserving the checkpoint's exact NVFP4 representation. Keep frequently
used experts resident in HBM, stage colder experts from pinned host memory or a
file-backed source, and retain a CUDA-graph-compatible decode fast path.

## Design decision

Use three independent layers:

1. A KTransformers-inspired residency policy decides which experts deserve HBM.
2. A HiSparse-inspired transfer control plane moves rows safely and efficiently.
3. The existing NVFP4 data plane owns the six ModelOpt tensor families and the
   FlashInfer CUTLASS execution adapter.

KTransformers is a policy reference, not a runtime dependency. Its CPU kernels
cannot directly consume NVIDIA ModelOpt NVFP4, and its GPU/CPU mask does not
represent physical cache slots.

HiSparse is a control-plane reference, not a cache superclass. KV pages and MoE
experts have different identities, sizes, replacement rules, and consumers.

## Residency policy

The initial policy records exact router selections in a fixed
`[num_layers, num_experts]` counter tensor. At a safe scheduling boundary it
updates an exponentially decayed score:

```text
score = decay * score + observed_activations
```

The highest-scoring experts become desired residents subject to the configured
HBM budget. Promotion uses hysteresis so experts near the cutoff do not cause
continuous PCIe churn. Exact current-layer demand always outranks speculative
or background work.

Initial transfer priority is:

1. Exact current-layer misses.
2. High-confidence next-layer prefetches.
3. Background activation-aware promotions.

The previous identity-style next-layer predictor is not a residency authority.
Its observed 17.1% precision, 4.3% recall, and 82.9% wasted speculative bytes
make actual routing frequency the safer first policy.

## Transfer control plane

There is one dedicated expert-transfer CUDA stream per device, shared across all
MoE layers. It is separate from HiSparse's KV-transfer stream to avoid
head-of-line blocking between large expert rows and latency-sensitive KV work.

The control plane uses stable device buffers modeled after HiSparse:

- source row IDs (`int64`)
- destination slot IDs (`int32`)
- request count (`int32`)
- slot generations
- a bounded event/ticket ring

Each ticket represents one atomic expert transfer. All six tensors are enqueued
on the same stream:

- `w13_weight`
- `w2_weight`
- `w13_blockscale_swizzled`
- `w2_blockscale_swizzled`
- `g1_alphas`
- `g2_alphas`

A cache mapping is published only after all six writes complete. A destination
slot cannot be reused until the consumer event for its current generation has
completed.

Slot states are `FREE`, `RESERVED`, `LOADING`, and `READY`.

## Host tiers

Pinned host memory is the preferred source for admitted expert rows. File-backed
storage remains the capacity tier. A file-backed miss is admitted into the
bounded pinned cache before CUDA transfer. Existing SGLang pinned allocation and
registration helpers should be reused instead of introducing another allocator.

The transfer kernel remains replaceable. Benchmark the existing expert Triton
row gather against HiSparse's planned byte-row copy using real NVFP4 row sizes;
the fixed transfer-plan ABI does not require choosing either kernel permanently.

## CUDA graph boundary

The graph fast path is hit-only:

```text
prepare residency outside graph
  -> wait for required transfer tickets
  -> replay fixed decode graph
  -> record consumer completion
  -> rebalance or evict between replays
```

The HBM pools, lookup tables, tensor shapes, and pointers remain stable across
replays. Expert ID to slot remapping occurs on-device. The captured path must not
call `.item()`, `.tolist()`, create variable Python lists, allocate tensors, or
change parameter pointers.

An unresolved miss takes an eager recovery path, loads the expert, and then
returns to graph replay. No graph recapture should be required because slot
addresses and cache capacity remain fixed.

## Prototype boundaries

- Do not enable or subclass `HiSparseCoordinator` for experts.
- Do not share the physical CUDA stream with KV transfers.
- Do not add a `kt_kernel` dependency.
- Do not convert the checkpoint from NVFP4 to MXFP4.
- Do not change the ModelOpt/FlashInfer adapter in the first transfer refactor.
- Prefer focused syntax, unit, and smoke checks over broad test suites while the
  design is still being prototyped.

## Success criteria

- A single per-device expert stream replaces per-layer transfer streams.
- Cache publication happens only after all six tensors are ready.
- Slot reuse is protected by ticket generation and consumer completion.
- Activation counters produce an inspectable desired-residency set.
- Exact-demand misses remain correct through the eager fallback.
- Cache-hit decode can execute without a CUDA graph break caused by Python cache
  management or changing storage addresses.
