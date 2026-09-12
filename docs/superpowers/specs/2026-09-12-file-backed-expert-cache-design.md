# File-backed NVFP4 expert cache design

Date: 2026-09-12

## Objective

Reduce Qwen3.8-Flash-Next ModelOpt NVFP4 expert memory from roughly 72 GiB of
anonymous, swap-prone CPU storage to verified file-backed storage while
preserving selected-expert execution through FlashInfer CUTLASS. Recover warm
decode performance with measured hot-expert residency and transfer overlap.
Use CUDA VMM only if the simpler cache hierarchy proves correct and profiling
shows that compact staging or pointer stability remains a material bottleneck.

This remains an opt-in prototype. The existing anonymous expert streamer is the
fallback and comparison baseline.

## Observed baseline

The current single-RTX-5090 prototype uses TP=1, EP=1, standard top-k dispatch,
FlashInfer CUTLASS MoE, 8,192 total tokens, and one running request.

- The offloader reports 72,000 MiB of host-resident NVFP4 expert tensors.
- A 1,024-token prefill selected 464 of 512 experts.
- Persistent staging capacity reached approximately 1.22 GiB on CUDA and
  another 1.22 GiB in pinned host memory:
  - `w13_weight`: 725.0 MiB
  - `w2_weight`: 362.5 MiB
  - `w13_blockscale_swizzled`: 90.6 MiB
  - `w2_blockscale_swizzled`: 45.3 MiB
- Observed prefill throughput is approximately 5.5-5.8 token/s.
- Observed warm decode throughput is approximately 2.8-3.0 token/s.
- Persistent verified PLE caching is working and its manifest was published.

The 464-expert prefill working set means cold prefill can touch nearly the
entire expert model. Decode and prefill must therefore be measured separately.
A small hot cache is expected to help decode much more than large prefill.

## Approaches considered

### Recommended: staged mmap hierarchy

Back the large expert tensors with verified mmap files, retain the current
bounded pinned and CUDA staging buffers, reuse SGLang's expert-distribution
recorder, then add size-bounded hot tiers and asynchronous staging. This changes
one layer at a time and preserves the existing FlashInfer CUTLASS call shape.

Advantages:

- Immediately converts anonymous expert memory into reclaimable file-backed
  memory.
- Reuses the existing `ExpertStreamer`, checkpoint loader, cache locks, and
  selected-row gather path.
- Produces useful measurements before selecting cache sizes or replacement
  policy.
- Keeps VMM optional.

Trade-off: cold large-prefill latency can be dominated by NVMe reads.

### Alternative: VMM-first expert residency

Reserve each full expert tensor's CUDA virtual address range and dynamically map
physical GPU pages for resident experts. This can preserve stable pointers and
avoid compact D2D assembly, but introduces allocation-granularity waste,
mapping lifecycle, event fencing, CUDA graph constraints, and a larger failure
surface before the file-backed source is proven.

Decision: defer.

### Alternative: static GPU placement only

Keep a fixed set of frequent experts on GPU and use the existing CPU streamer
for all others. This is simple, but does not remove the 72 GiB anonymous source,
adapts poorly to workload shifts, and still requires compact staging when hot
and cold experts are mixed.

Decision: use static frequency placement only as the first policy inside the
recommended hierarchy, not as the storage architecture.

## Architecture

### Shared verified file-tensor storage

Extract the checkpoint identity, locking, sparse-file mapping, and atomic
completion-manifest behavior already proven by the PLE cache into a model-neutral
file-tensor cache utility. Both PLE and expert storage consume that utility.

Each expert tensor identity includes:

- canonical model path;
- revision and Hugging Face commit hash when present;
- layer/module prefix;
- tensor name, shape, stride, dtype, and byte length;
- cache format version.

A cache miss maps a fresh sparse inode. Normal checkpoint loading populates it.
No completion manifest is published until every expected expert tensor for the
model has loaded successfully. A verified hit skips only those expert tensor
copies; scales and unrelated weights remain on the normal loader path.

For the first prototype, file backing applies to the four material tensors:

- `w13_weight`
- `w2_weight`
- `w13_blockscale_swizzled`
- `w2_blockscale_swizzled`

The tiny `g1_alphas` and `g2_alphas` remain ordinary CPU memory.

### Runtime cache hierarchy

The runtime resolves each `(layer, expert)` through three bounded tiers:

1. GPU hot tier
2. Pinned-host hot tier
3. Complete mmap/NVMe backing tier

The existing compact CUDA staging tensors remain the runner-facing interface.
GPU hits are copied D2D into compact staging. Pinned hits use asynchronous H2D.
File misses gather ascending expert rows into the existing pinned bounce buffers
and then transfer asynchronously. The FlashInfer runner continues receiving
compact expert IDs and compact tensors, so its numerical behavior is unchanged.

Initial cache placement is static and frequency-based. Dynamic promotion and
eviction are enabled only after the metrics demonstrate stable locality.
Budgets are byte-based and global rather than a fixed expert count because
expert tensor sizes and available GPU memory determine actual capacity.

### Placement statistics

Reuse SGLang's existing expert-distribution recorder through
`--expert-distribution-recorder-mode stat`. Do not create a second source of
routing truth.

The expert cache adds only operational counters the recorder does not provide:

- hits and misses by tier and layer;
- requested unique experts per forward pass;
- bytes read from file, pinned host, and GPU;
- file gather, H2D, D2D, and wait latency;
- promotions, evictions, and residency bytes;
- prefill versus decode measurements.

Statistics are exportable as structured JSON and periodic server logs so cache
budgets and policies can be selected from evidence.

### Transfer overlap

Split `ExpertStreamer.gather()` into prepare and wait phases backed by two
reusable staging slots.

- CPU/file gathers run in a small bounded worker pool.
- H2D copies run on a dedicated CUDA stream with recorded events.
- CPU gathering of the next tensor overlaps the current tensor's H2D copy.
- The compute stream waits once before the compact expert tensors are consumed.
- A staging slot cannot be reused until its consumer event completes.

This first overlap phase pipelines host work and DMA but does not claim
layer-ahead prefetch: the next layer's routed expert IDs do not exist until its
router executes.

A later runner-specific phase may delay `w2` readiness and transfer it while the
`w13` GEMM and activation execute. That requires an explicit boundary in the
FlashInfer CUTLASS runner and is not part of the first implementation.

### Optional CUDA VMM elasticity

VMM is a measured follow-up, not a prerequisite. If D2D compact assembly or
stable-pointer constraints are significant, adapt SGLang's existing
`cuda_vmm_utils` reservation primitives and MoE DWDP page-pool patterns.

The VMM design reserves the full logical CUDA address span for each large expert
tensor while committing physical GPU pages only for resident experts. Mapping
and eviction occur at CUDA allocation granularity, preserve virtual addresses,
and are fenced against kernels still reading the pages. The original expert IDs
can then remain valid without compact D2D assembly where the backend supports
partially resident logical tables.

VMM is accepted only if it improves measured latency or memory efficiency after
accounting for allocation-granularity waste and map/unmap overhead.

## Configuration

Prototype controls remain opt-in and environment-driven alongside
`SGLANG_MOE_EXPERT_STREAM=1`:

- `SGLANG_MOE_EXPERT_FILE_DIR`
- `SGLANG_MOE_EXPERT_FILE_RSS_BUDGET_GB`
- `SGLANG_MOE_HOT_GPU_MB`
- `SGLANG_MOE_HOT_PINNED_MB`
- `SGLANG_MOE_TRANSFER_OVERLAP=1`
- `SGLANG_MOE_VMM=1`

Zero hot-tier budgets preserve correctness with file-backed selected-row
staging. Transfer overlap and VMM default off. Stable public CLI arguments can
replace environment controls after the prototype is validated.

## Failure behavior

- Missing, malformed, mismatched, or incomplete manifests cause a rebuild.
- A failed cold load leaves no valid completion marker.
- Cache allocation failure reports the exact tensor and required bytes and does
  not silently fall back to anonymous 72 GiB storage.
- Hot-tier allocation failure disables that tier before serving requests.
- An asynchronous gather or copy failure surfaces before the MoE runner consumes
  the staging slot.
- VMM mapping failure rolls back the new mapping and retains the previous valid
  residency state.

## Delivery phases

### Phase 1: correctness and reclaimability

- Generalize the verified file-tensor cache utility.
- File-back the four large expert tensors.
- Preserve the existing synchronous selected-row gather and compact runner
  interface.
- Add verified cold-build completion and warm-start reuse.
- Compare outputs with the anonymous-source streamer for fixed routed IDs.

### Phase 2: statistics and static hot residency

- Connect SGLang's expert-distribution recorder.
- Add tier and transfer metrics.
- Add bounded GPU and pinned-host tiers.
- Seed them from recorded per-layer expert frequency.

### Phase 3: transfer overlap and dynamic policy

- Add double-buffered prepare/wait staging.
- Pipeline CPU/file gathers with H2D copies.
- Enable dynamic promotion only when measured locality supports it.
- Evaluate the separate `w2`-during-`w13` runner change.

### Phase 4: optional VMM

- Prototype fixed-address expert mappings using SGLang's VMM utilities.
- Compare against compact staging on latency, GPU bytes, and complexity.
- Keep VMM disabled if it does not produce a clear measured gain.

## Verification and success criteria

Prototype verification focuses on numerical correctness and measurable resource
behavior:

- Fixed routed IDs produce the same compact tensors and model outputs as the
  current anonymous-source streamer.
- A completed cache survives restart and skips expert checkpoint copies.
- Anonymous expert memory falls by at least 60 GiB.
- Steady-state swap-in and swap-out attributable to experts approach zero.
- File, pinned, and GPU residency stay within configured byte budgets.
- Warm decode throughput after hot residency is within 10% of the current
  2.8-3.0 token/s baseline, or improves it.
- Cold and warm prefill are reported separately at small and 1,024-token chunks;
  no prefill claim is made without both measurements.
- No GPU OOM occurs at the current 8,192-token configuration.

`O_DIRECT` and `io_uring` remain later experiments. They are justified only if
profiles show page-cache faults or synchronous file reads dominate after the
file-backed hierarchy and overlap phases are working.
