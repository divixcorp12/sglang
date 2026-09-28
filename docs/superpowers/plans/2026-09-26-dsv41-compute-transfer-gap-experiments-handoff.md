# DSV4.1 compute-to-transfer gap: experiment handoff

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` when executing this handoff. Run the diagnostics and gated experiments below against one agreed baseline. Preserve source leases, destination lifetimes and numerical behavior.

**Goal:** Test whether shortening or overlapping the computation before the next layer's expert request advances useful expert transfers and improves unprofiled decode latency.

**Status:** Source review and offline analysis of the new indexer-payoff trace completed. No serving arm or production kernel change was made for this handoff. Timing budgets below are observations, not predicted savings.

## 1. What the experiment actually tests

The observed serial dependency is:

```text
layer i transfers → F → routed MoE → combine/mHC
    → next-layer Engram, if present → attention → FFN mHC/norm
    → shared expert → router/top-k/plan → Post → next expert transfers
```

`F` means the end of `exl3_ram_miss_lease_finalize_kernel`; `Post` means the start of `exl3_ram_miss_post_kernel`. These are observable boundaries, not timestamps for the exact request publication store or CPU lease release.

Shortening this chain can advance demand for the next layer. Moving independent shared-expert work to its existing side stream can advance demand without making that work faster. Both are useful hypotheses for reducing transfer-free time.

**Pinned source leases are a separate mechanism.** In the CE path, `copy_completed()` publishes CopyDone after DMA completion; when SM small copies are active, `copy_acked()` waits for a qualifying SmAck generation before `release_copied()` releases the source leases. This does not depend on routed MoE finishing its use of the destination VRAM. Host polling or mutex delay can defer actual retirement, but this trace does not measure that delay. Releasing a source lease never grants permission to overwrite destination VRAM still used by compute.

At constant transferred bytes, a shorter step raises average bytes/second even if the bandwidth during each transfer is unchanged. Count the reduced step time once; the resulting higher average PCIe activity is not a second independent speedup. Faster compute may also expose more storage waiting. That is why the acceptance metric is unprofiled end-to-end latency.

## 2. Baseline and provenance

- Source inspected: `/home/dimitri/data/divix/sglang-nvfp4`, HEAD `db6c99db56b9d267d960cf035f6f3a5d64e4fe34`. Preserve unrelated edits and resolve the intended integrated baseline before execution.
- At the attention/MoE follow-up review, available `origin/master` was `90a20412c1`, three commits ahead, changing only the HiCache analysis driver/equivalence probe. At the RTX 5090 follow-up it is `2bf9919fc7`, five commits ahead, also changing the production HiCache recipe, its test and reference documentation. The inspected attention/MoE kernels remain identical to local master. Freeze the effective recipe within each A/B; do not silently compare the new recipe with the earlier trace. No checkout or pull was performed. The external EXL3 source is available at `/home/dimitri/data/divix/exllamav3`, clean at pinned `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`; `exl3_ext.py` checks that commit before building.
- Trace stem supplied by the user: `/home/dimitri/data/divix/nsys-reports/indexer-payoff-node-20260926-143541`. The actual main export is the adjacent `indexer-payoff-node-20260926-143541.sqlite`; the PCIe report is the adjacent `indexer-payoff-node-20260926-143541-pcie.nsys-rep`.
- Capture environment identifies `f62f7af0b62bf479af5d4535f72d25b8b3dba54f`. This is recorded run metadata, not independent proof of a clean captured source tree. Current-source mappings must be checked against the experiment checkout.
- This is already the **128 MiB prefill-indexer score budget / 17,408 MiB GPU hot cache / memory fraction 0.925** payoff configuration. Do not repeat the older 14,336 MiB baseline accidentally. The recipe landed in `24bbd3baab`.
- `DSV41_REFERENCE.md` around line 4250 records the payoff's historical comparison: fewer expert bytes, largely unchanged NVMe streaming time, and a directional 110.1 → 103.0 ms/token unprofiled result. That comparison was not an interleaved A/B. Combined settings have not been validated at 30k–32k context.
- Retain the active CE, row-image, two-phase, piece-stream, SM-small-copy, DIRECT stage-2, layer/cast-fusion settings; record effective values from the actual process. Keep prefetch off and decode residency updates every forward. Do not sweep cache policy in these compute experiments.

## 3. New trace findings

### Selection and measurement

The main trace contains 110 complete decode graph replays, each with 2,398 kernels and 40 Post/F/MoE instances. Grouping uses `(correlationId, graphId, globalPid, deviceId, contextId)`. The primary selection skips the first 20 chronological graphs and retains 90, matching the existing payoff kernel analysis. This is an explicit analysis cut, not a warmup label proven by NVTX.

Primary scope: process `308046093877248`, device `0`, context `1`; graph stream `142`, off-graph H2D CE stream `141`. Graph span is first kernel start to last kernel end and excludes inter-graph host gaps. Layer numbers below are graph-order ordinals.

| Disjoint wall-time region | Mean ms/step |
|---|---:|
| Prefix before first Post | 2.366 |
| Sum of 40 Post-start → F-end intervals | 90.188 |
| Sum of 39 F-end → next-Post-start intervals | **13.460** |
| Tail after final F | 0.457 |
| Total graph span | **106.471** |

There are **zero expert CE copies overlapping the 3,510 measured F→nextPost intervals**. Their kernel interval union is 13.001 ms/step; only 0.459 ms is between kernels. The typical transition is about 336 µs. This is primarily a dependent GPU-work interval, not an unexplained host scheduling gap.

| Kernel group inside those 39 intervals | Mean summed kernel ms/step |
|---|---:|
| Attention, including projections | **5.751** |
| Routed expert and gather/commit | **4.436** |
| Shared expert | **1.259** |
| Combine/norm | 0.888 |
| Routing and plan | 0.385 |
| Engram | 0.296 |

Groups are inferred from the stable kernel order and checked against current source, not explicitly labeled NVTX ranges. Their sums overlap by 0.014 ms/step, so they are not another exact wall partition.

Largest named kernels in this interval:

- Routed `exl3_moe_kernel`: **3.856 ms/step**, about 99 µs per call.
- Attention `exl3_gemv_int8_sq_kernel`: **2.489 ms/step**, four calls per transition.
- Attention BF16 WMMA `Kernel2`: **1.702 ms/step**, about 43.6 µs per call; likely `wo_a`, subject to attribution/shape confirmation.
- Shared-expert INT8 GEMVs: **1.164 ms/step**, within the 1.259 ms shared block.
- Sparse MLA proper: **0.510 ms/step**. Optimizing only this kernel addresses a small part of the attention block.

Skipping only 10 graphs gives 106.772 ms/step and 13.457 ms for F→nextPost, supporting stability of the compute finding. These profiled durations are not serving latency predictions.

### PCIe cross-check using the same 90 windows

The session-clock offset is **482,882,340 ns**: add this to a main-relative timestamp to query the PCIe report. Sampling is approximately 100 µs. The earlier reference PCIe table used 108 different windows; do not compare its state durations directly with this table.

- Expert CE traffic: **878.395 MB/step**, 132.38 copies/step, **69.162 ms/step** interval union.
- F→nextPost: raw mean RX metric **1.45%**, with **90.23% of sampled values below 5%**. This supports low RX activity during the observed compute chain; it is not an exact idle-time fraction or a percentage of physical Gen3 link capacity.
- No CE copy while S runs: **18.952 ms/step**, mean RX **25.80%**. SM reads of mapped host memory can use PCIe here; “no CE copy” does not mean an idle link.
- No CE copy outside S/CW: **17.919 ms/step**, mean RX **3.10%**; this includes the compute intervals and other boundaries.

GPU UUID and PCI bus identity match between the exports. RX is device-wide and can include other processes. Sample windows straddle kernel/copy boundaries. Empirical GB/s calibration from this selection has only five long CW-contained bursts and is too weak to support a precise bandwidth-headroom claim; use raw RX and exact CUPTI CE byte counts as separate observations.

The adjacent-CE-layer subset also illustrates why “last DMA ended” is not a universal ready marker: last-CE-end→F averages 396 µs but has a 6.8 µs median. A small fraction of long residual transfer-chain waits makes the mean large; storage, SM-copy and acknowledgment contributions require instrumentation. Use F for a conservative transfer-chain boundary and instrument exact readiness when needed.

### A separate unexplained graph-prefix gap

Of the 2.366 ms prefix, **2.047 ms is a repeatable untraced gap** after the initial metadata kernels and before embedding `indexSelectSmallIndex`. All 90 replays have the same gap between graph nodes `8589934606` and `8589934608`. Creation records contain intervening node `8589934607`, cloned from `4294967311`, but do not identify its type.

The representative gap has no recorded kernel, memcpy, memset, CUDA synchronization or NVTX range. `cudaGraphLaunch` spans the preceding graph and ends shortly after this gap. This does not establish CPU compute, a host callback, a residency operation, or a 2 ms optimization opportunity. Diagnose it separately in experiment C0b.

Current source supplies a concrete candidate for the missing node: `decode_cuda_graph_runner.py:1158` prepares attention metadata, calls `_record_in_graph_metadata_prep_done()`, then enters model forward. That helper (around line 479) records an external CUDA event during capture, using `runner_utils/shared_read_event.py`. Embedding follows in `deepseek_v4.py:4735`. An event-record node need not have a kernel/memcpy row. This positional match does **not** prove the node identity or explain its latency. The event protects shared metadata reads against subsequent scheduler writes (`scheduler.py:1936`); do not remove or move it without preserving that ordering. DIRECT residency runs at the first expert gather, later than this boundary, and cannot explain the pre-embedding gap.

## 4. Concrete source targets

Paths are repository-relative; line numbers refer to the inspected HEAD and may move.

| Target | Source and entry point | Why inspect it |
|---|---|---|
| Shared work before routing | `python/sglang/srt/models/deepseek_v2.py:1273–1289`, join/add around 1379 | Existing side-stream fork can let router→Post advance while shared work runs |
| Side-stream lifetime contract | `python/sglang/srt/layers/moe/moe_side_stream.py` | Every fork joins inside the layer, before shared buffers can be reused |
| Attention projection path | `python/sglang/srt/models/deepseek_v4.py:470`, `_apply_wo_a_bf16_matmul`; call around 2528, `wo_b` around 2548 | Likely 1.702 ms Kernel2 block; existing fast-path eligibility excludes usual TP1 eight-group shape |
| Existing grouped GEMV | `python/sglang/kernels/ops/attention/dsv4/wo_a.py:72` | Wrapper derives group count dynamically; isolated eight-group benchmarking is feasible |
| Dense casts and consumers | `deepseek_v4.py:1332`, `_compute_q_b`; `:1398`, `_compute_kv_to_cache`; `:2548`, `wo_b`; `DSV41_REFERENCE.md` around 4341 | Remaining wq_b→q-rope, wkv→k-norm-rope and wo_b→mHC casts are documented candidates; line 1702 is the HIP preparation branch, not the CUDA target |
| Routed fused expert compute | `python/sglang/srt/layers/quantization/exl3.py:472`, `exl3_fused_moe.py:145` | Gather precedes the current fused MoE call; optimize that call only after shape-specific evidence |
| Residency and commit scheduling | `python/sglang/srt/layers/moe/expert_residency_gpu.py`, `_apply`, `_rank_victims`, `commit_gather` | Side-stream flag can also change commit scheduling; distinguish effects and retain ordering |
| Transfer graph | `python/sglang/srt/layers/moe/exl3_ram_miss.py`, `Exl3RamMissRowBackend.post`; `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh` | Post/W1/C1/A1/S/CW/F markers and request publication instrumentation |
| Source lifetime evidence | `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp:4499–4540`, `copy_completed`, `copy_acked`, `release_copied` | Separate DMA/SM completion and actual host retirement from GPU destination use |

### Follow-up ranking from master source inspection

After C1, prioritize bounded experiments in this order. This ranks the cost of obtaining useful evidence and the available opportunity, not measured speedups.

| Order | Concrete experiment | Measured portion in 39 transitions | Initial gate |
|---|---|---:|---|
| 1 | Existing standalone `wo_a_bf16_gemv` on TP1 eight-group BS1 input | 1.702 ms/step, probable attribution | Compare actual shapes before adding dispatch; a stable ≥5 µs/call gain would justify integration (~0.20 ms across 40 calls) |
| 2 | Dense INT8 SQ tuning for `wq_b` and `wo_b` | 1.838 ms/step, positional attribution | Aim for ≥10% aggregate shape-matched improvement (~0.18 ms on measured portion), preserving numerical behavior and fixed workspace ownership |
| 3 | Routed MoE group-width/tile sweep, then static six-route scheduling if justified | 3.856 ms/step | A stable ≥10 µs/call gain (~0.40 ms across 40 calls) merits serving validation; do not spend a rewrite on an unmeasured scheduler cost |
| 4 | Remaining output-cast fusion, starting with q RoPE | One cast per layer per candidate | Prove the launch is removed and outputs are bit-identical; historical expectation is only ~0.1 ms/token per candidate |

These are proposed screening gates, not prior experimental results or final acceptance thresholds. A candidate must beat microbenchmark noise and then satisfy section 6. The routed kernel has the largest measured budget, but greater correctness and implementation cost. `wo_a` has the cheapest decisive test and may already be too bandwidth-efficient to beat.

## 5. Experiment sequence

### C0 — Reproduce the baseline and measure the causal chain

1. Record SHA, effective environment, external EXL3 source/build revision, graph shape, prompt/token corpus, cache slots, NUMA page split, clocks and competing CPU/disk/GPU activity. Confirm the current larger-cache recipe.
2. Run an unprofiled baseline and a short node-mode diagnostic capture. Recompute the same complete-graph selection; keep cold and steady results separate.
3. Collect per-layer relative timestamps for F-end, routed-end, next router/top-k completion, next Post-start/end and next first expert copy. Compare positions relative to graph start and F-end, not absolute timestamps between separate runs.
4. If the proposed benefit is specifically **earlier lease retirement**, add bounded diagnostics for source-reader completion/SmAck observation, `release_copied` lock acquisition/release, outstanding sources, and requests delayed because no reclaimable slot exists. Separate CE and ordinary lease paths. Avoid per-lane logging in the acceptance run and do not equate kernel completion with a lease word being written.
5. Record expert IDs/routes, cache misses, CE bytes/copies, NVMe bytes and CE busy time. A changed transfer workload is a confounder to the compute-only hypothesis.

**Decision:** Earlier Post and first copy with unchanged bytes demonstrate advancing demand. A retirement optimization is justified only if measured release delay blocks a useful next reservation/transfer. Without that evidence, retain retirement logic.

### C0b — Explain the prefix before treating it as an optimization budget

Capture graph topology/node types around the metadata→embedding boundary, preferably from the actual captured SHA. Correlate node creation/callback registration, graph launch, CPU scheduling and any graph stream wait. Compare a minimal diagnostic capture with the node-traced run and unprofiled wall time. CUDA graph debug output or narrow registration instrumentation can resolve node type; do not add broad instrumentation to acceptance runs.

Start with `_record_in_graph_metadata_prep_done()` in `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` and the scheduler's shared-buffer-read-done event wait. Confirm whether missing node `8589934607` is that external event record. Trace its consumer and launch timing while retaining the shared-buffer write-after-read protection. Merely identifying an event node does not show that event recording itself takes 2 ms.

**Decision:** If the 2.047 ms hole is profiling/launch instrumentation, exclude it from optimization estimates. If it is a necessary runtime dependency, identify its producer and protection contract before changing it. DIRECT GPU ranking duration is not evidence that it explains this untraced pre-embedding hole.

### C1 — Retest the existing shared-expert side stream with CE

**Priority: first serving experiment.** Same integrated SHA and recipe, A with `SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM=0`, B with `=1`. Check that the path actually forks: no deferred-shared, fused-shared-inside-SBO, or skip-shared override that bypasses it.

The current shared block contributes 1.259 ms across the 39 exposed transitions. This is an opportunity budget for overlap, not a predicted win. The flag also affects commit/gather scheduling; attribute the combined result and use separate experimental gates only if it is necessary to distinguish the two mechanisms.

**Working expectation, not an experimental result:** a modest improvement, probably below 1 ms/token, with no measurable gain still plausible. The measured shared block averages about 32 µs per transition. Extrapolating that average across all 40 layers gives approximately 1.3 ms/token of shared-compute opportunity, around 1% at current latency. This is an idealized budget for hiding shared work, not a strict bound on the flag's combined effects or a prediction from profiled timings.

| Possible unprofiled outcome | Interpretation |
|---|---|
| 0.3–1.0 ms/token improvement | Working expectation: useful partial overlap after overhead and contention |
| Around 1.3 ms/token improvement | Most shared computation hidden with little added overhead |
| No improvement or regression | Stream dependencies or resource contention consume the overlap benefit |

Shared computation and routing consume the same prepared input, so the side-stream fork can let router→plan→Post proceed independently. Shared work can overlap request preparation and host servicing **before DMA starts**; it need not overlap an actual copy to help. The final combination still waits for both shared and routed results.

Bulk CE DMA does not require the SMs running the shared GEMVs, making this worth retesting after the earlier SM-copy no-go. However, piece streaming still uses SM work, DMA and GEMVs can compete for GPU memory bandwidth, and extra stream dependencies have overhead. Expect earlier demand for experts, not earlier pinned-source lease retirement. Do not expect this experiment alone to fill all PCIe idle time or deliver a several-millisecond gain from hiding the shared block.

Before full serving, run captured CE/side-stream correctness coverage and a repeated-replay soak. Preserve `fork` input readiness, `record_stream` lifetime handling and `join` before combining outputs. Never extend the fork across a layer boundary where buffers may be reused.

Measure whether router/Post and first CE start earlier, how much shared work overlaps S/CW/CE, and whether transfer duration or routed compute slows from contention. The previous SM-copy study found no wall gain; its C1 contention does not settle the current CE configuration. See `DSV41_REFERENCE.md` around 3937.

**Keep only if:** numerical/output checks pass and repeated unprofiled runs show a useful net win with comparable transferred bytes and cache misses. Establish earlier Post/first useful copy, then check that slower transfers or routed compute do not erase the gain. Earlier Post alone is insufficient. The expectation range above is not an acceptance threshold; apply the run-to-run-noise and scheduling-change criteria in section 6. Revert the experimental flag if it adds overhead or destabilizes capture.

### C2 — Attribute and benchmark attention projections before changing kernels

**C2a: wo_a.** Confirm Kernel2 callsite, actual TP1 shape, stride, dtype and batch size. Its demangled implementation is `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8`. Current source falls back to an einsum outside the TP4-specialized `(groups, rank, dim)=(2,1024,4096)` path.

The production shape gate is `deepseek_v4.py:502`; the fallback einsum is at 545. The standalone `wo_a_bf16_gemv` in `python/sglang/kernels/ops/attention/dsv4/wo_a.py:72` already derives group count dynamically and supports `[1,8,4096]` input with `[8,1024,4096]` BF16 weights. Its kernel at line 58 uses one output row per CTA, 8,192 CTAs at this shape, four warps, FP32 multiply/reduction and BF16 store.

First benchmark that existing kernel against the current einsum. Use realistic resident weights, representative input layouts and graph replay, including a rotating set of layer weights to avoid relying solely on same-weight cache reuse. Eight groups of 1024×4096 BF16 weights are 64 MiB; consuming them in 43.6 µs implies about 1.54 TB/s of logical weight traffic. Compare matched measured weight-read throughput before expecting substantial headroom. If the candidate wins, add a separate opt-in gate for EXL3 TP1/BS1 decode on the measured device, exact shape/dtype/contiguity, with the fallback unchanged.

If tuning is needed, compare one, two and four output rows per CTA (`BN`), updating the launch grid accordingly. Measure register pressure and throughput; fewer CTAs can help or hurt. The tree reduction differs from tensor-core einsum, so `enable_fp_fusion=False` does not establish bitwise parity. Require baseline comparison on real activations, downstream output checks and an explicit numerical acceptance decision if bits differ. Existing `test/registered/kernels/ops/attention/test_wo_a_fused.py` covers the different SM100/two-group fused kernel, not this standalone TP1 candidate; add direct coverage before integration.

Do **not** widen the entire TP4 gate or simply enable `SGLANG_DSV41_FUSED_WO_A`. The fused CUDA path is architecture-major-10 gated and contains two-group/2,048-output layouts (`wo_a_fused.cuh` around 735); it is not a simple SM120/TP1 toggle. The standalone Triton GEMV is the bounded experiment.

**C2b: dense INT8 projections.** A follow-up query of the same 90 graph windows found exactly four attention INT8 GEMVs per transition. Their source mapping is positional, not NVTX-proven:

| Ordered call / likely projection | Model shape, input→output | Mean µs/call | ms/step across 39 transitions |
|---|---|---:|---:|
| 1 / wq_a | 5120→1280 | 8.988 | 0.351 |
| 2 / wq_b | 1280→32768 | 22.844 | 0.891 |
| 3 / wkv | 5120→512 | 7.705 | 0.301 |
| 4 / wo_b | 8192→5120 | 24.296 | 0.948 |

Construction is in `deepseek_v4.py` around 870–933. Confirm loaded trellis bits/layout and dispatch before specialization. Prioritize **wq_b and wo_b**, totaling 1.838 ms of the 2.489 ms budget. Keep their input casts/output consumers in the measured chain so a local improvement cannot hide extra conversion overhead.

The relevant external files are under `/home/dimitri/data/divix/exllamav3/exllamav3/exllamav3_ext/quant/`:

- `exl3_gemm.cu:178` selects INT8 SQ before the ordinary forced-shape/autotuner path. Changing `force_shape_idx` or `force_num_sms` in `exl3_ops.py:147` does **not** tune the observed INT8 SQ dispatch.
- `exl3_gemv_int8.cu:133` derives occupancy/grid, K slices and shared-memory requirements; `exl3_gemv_int8_kernel.cuh:976` mirrors the decomposition. Test offline-selected, shape-keyed alternatives here, with explicit agreement between host/device formulas.
- `exl3_gemv_int8_kernel.cuh:920` sums K-split partial results serially. A bounded unroll/prefetch experiment preserving accumulation order is a more conservative numerical change than tree reduction. Measure whether the epilogue matters before investing.
- The fixed per-device workspace at `exl3_gemv_int8.cu:101` and last-contributor counters/fences/reset around kernel line 1027 are part of correctness. Do not resize a workspace referenced by captured graphs or overlap two users of that same workspace on different streams.

Changing slice boundaries also changes per-slice activation quantization; it is not purely scheduling. Start with variants that preserve slices/rounding where feasible, record any changed arithmetic, and compare baseline outputs. Keep `EXL3_INT8_GEMV` mode fixed: residual-corrected and plain INT8 modes are different numerical policies. Existing generic `test_exl3_ops_gpu.py` coverage does not replace full wq_b/wo_b shape, real-input and repeated-capture tests. Preserve the loader's pinned-build provenance through a private experimental source/build pair; do not modify the shared extension in place or bypass its commit check silently.

**C2c: remaining output conversions.** These are smaller but concrete follow-ups. `Exl3LinearMethod.apply` (`exl3.py:209`) currently converts `exl3_gemm_bs1`'s FP16 result to BF16. Use a narrowly scoped half-output path only where its consumer reproduces that BF16 rounding internally; changing the global output contract would affect unrelated linears.

1. **wq_b→Q RoPE:** start at `_compute_q_b` (`deepseek_v4.py:1332`) and `q_rope_store.py:18,77`. For the active no-Q-head-norm path, accept FP16 input but round each loaded value to BF16 before converting to FP32 for RoPE; retain the BF16 output/padding contract. Verify this branch is active. Other Q normalization/cache paths need their own eligibility or fallback.
2. **wkv→K norm/RoPE/cache:** `_compute_kv_to_cache` (`deepseek_v4.py:1398`) calls through `deepseek_v4_memory_pool.py:1988` and `dsv4/elementwise.py:271` into `deepseek_v4/main_norm_rope.cuh`. That kernel currently uses the same template dtype for input and norm weights. Passing FP16 input into the existing entry point is insufficient; preserve BF16 norm-weight interpretation and input BF16 rounding in a separate variant. Require cache-byte parity, including scales and rotary components, across the actual cache layout.
3. **wo_b→mHC:** call at `deepseek_v4.py:2548`; CUDA post dispatch around 2945–3004 and `mhc_post_split_h.py:9,32`, with `mhc_post_combine.py` where eligible. These consumers currently require BF16 input. A half-input variant must round to BF16 before FP32 mixing and preserve FMA/reduction order. Cover every reader reachable under its gate; do not merely remove a dtype assertion.

Common input/internal casts are already fused. The historical estimate is about 0.1 ms/token per remaining output cast, not a measurement of these unimplemented variants. Do not fold flashinfer RMSNorm or mHC reductions into this work merely to remove launches; prior studies found their reduction order could not be reproduced bit-exactly.

**Gate to serving:** shape/layout eligibility, required numerical parity, and stable graph microbenchmark savings at actual batch/shape must all pass. Report aggregate opportunity as measured call count × per-call saving, then validate it end to end. If a kernel is already near its measured data-movement floor, stop that branch.

### C3 — Optimize the existing routed MoE launch, conditionally

The exposed routed kernel itself is 3.856 ms/step, or 98.87 µs/call in the 39 transitions. The actual pinned external source is available; the target is its existing single fused MoE, not a generic rewrite. `Exl3FusedMoE.run` (`exl3_fused_moe.py:145–179`) builds route tables, launches `exl3_moe`, then runs deterministic `exl3_moe_gather`. Four intermediate buffers are reused sequentially across layers.

In the external `quant/` directory, `exl3_moe.cu:264` selects six groups × 28 blocks on the 170-SM device, 512 threads/block and 90 KiB dynamic shared memory. `exl3_moe_kernel.cuh:92` scans slot counts and schedules experts dynamically; gate/up GEMMs run sequentially around 194–197, with five group barriers throughout the expert body and ticket/reset work around 280–299. `exl3_gemm_inner.cuh:830–871` combines split-K contributions with locks/global intermediate sums. These are concrete inspection points, not proof that scheduling or barriers dominate the ~99 µs call.

**C3a: launch/tile sweep first.** Preserve six groups and compare widths 8, 16 and 28 with the current tile fixed. Width needs an experimental source control; it is not an existing serving flag. Then compare the winner with existing `EXL3_MOE_TILE_N=128` versus automatic 256 where eligible (`exl3_moe.cu:22`). Smaller groups trade parallelism for less split-K/barrier overhead. Measure the whole route-table→fused-MoE→gather chain in graph replay, both all-resident and immediately following real copies.

The kernel uses ordinary CUDA launches with software cross-CTA group barriers and requires all grid blocks to be co-resident. Preserve its launch constraint and verify occupancy for each compiled variant, including threads, shared memory and register usage; CTA count alone is not sufficient proof. Six groups × 32 blocks is 192 and exceeds the current 170-SM bound. Preserve lock indexing and barrier/reset behavior. Concurrent kernels can affect residency: keep the initial study isolated and validate any combined side-stream arm separately. Never use a potentially nonresident grid as a benchmark arm. Changing split-K grouping can change rounding even when mathematical intent is unchanged.

**C3b: static BS1/top-six scheduling, only if control overhead is exposed.** Assign one group per route in the existing sorted-slot order, bypassing slot-count scanning and dynamic work tickets. Retain existing GEMM/Hadamard math, weighted scratch, required stage barriers and deterministic gather. Gate to validated BS1/six-distinct-valid-route inputs; retain generic behavior for other cases. Preserve `keep=0` and fail-stop semantics without reading incomplete rows. Validate remap rewrites, padded/invalid routes, repeated replay and layer-buffer reuse before simplifying scheduling/reset state. This is a structural specialization, not permission to remove the intra-expert barriers.

Do **not** change `ROW_TILE=16` to 1: `exl3_gemm_inner.cuh:68` requires M≥16 and already specializes reduction for `size_m<=8`. Do not replace deterministic gather (`exl3_moe.cu:401`) with floating-point atomics. Restructuring gate/up or fusing down/gather is a later, higher-risk branch requiring separately measured evidence.

Require baseline parity for arithmetic-preserving variants, and explicitly assess any changed reduction order. Reuse `test/manual/dsv41/test_exl3_moe_probe_gpu.py` (P2 relative error ≤0.012 and ≤2×loop error+0.001), graph/eager replay after input/row/remap rewrites, split parity and end-to-end checks; passing a tolerance alone does not establish baseline bitwise equivalence. Preserve routing-weight-before-w2 semantics and scratch ownership. For scale, a hypothetical 20% reduction of the measured kernel portion is 0.77 ms/step; 25% is 0.96 ms. These are arithmetic budgets, not forecasts.

**Do not restart resident-first splitting as the default arm.** The prior real-copy study failed its 25 µs/layer gate at every split; much of the apparent gain was graph copy→kernel residual that may not exist with off-graph CE. See `DSV41_REFERENCE.md` around 4907 and the executed RAM-miss frontend plan.

### C4 — Smaller targets only after attribution

Routing/plan (0.385 ms), combine/norm (0.888 ms), and inter-kernel gaps (0.459 ms) are bounded secondary budgets. Reuse the companion threading/lease handoff for Post bitmap, validation and ranking work rather than treating those as new flags. Post publication occurs inside Post, outside the F→Post-start interval used here.

Engram contributes 0.296 ms on average, including 0.104 ms in its wait kernel, with larger tails near the first Engram transition. Investigate it as a tail-latency question if enough observations support it. Do not remove device waits or choose alternate attention streams merely because their names suggest overhead.

### C5 — RTX 5090 / SM120 kernel upgrades

**Architecture boundary.** NVIDIA identifies RTX 5090 as compute capability 12.0. Use SM120 implementations and their actual instruction requirements; do not treat datacenter SM100 kernels as compatible because both products are called Blackwell. [NVIDIA GPU capability list](https://developer.nvidia.com/cuda/gpus).

Current NVIDIA tables give CC12.x a 128 KiB unified cache, with up to **100 KiB usable shared memory per SM**, **99 KiB per block**, 48 resident warps and 64K 32-bit registers per SM. Query the installed device and compiled kernel attributes before choosing occupancy; the 128 KiB unified-cache figure is not an extra 128 KiB shared-memory allowance. [CUDA compute-capability tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications).

**Already implemented:** the pinned EXL3 loader defaults to native `12.0` compilation (`exl3_ext.py:63`, subject to an existing `TORCH_CUDA_ARCH_LIST` override). Dense decode already uses INT8 DP4A. Its SQ kernel actively selects four-stage `cp.async` shared-memory staging for trellis widths 3/5/7 (`exl3_gemv_int8_kernel.cuh:647,705,1021`). Routed MoE already uses asynchronous staging and FP16-input/FP32-accumulating tensor-core MMA (`exl3_gemm_inner.cuh:258,892`). Q RoPE already has PDL support (`q_rope_store.py:18,97`). These are baselines to tune, not missing features to enable.

#### R1 — Retune the existing INT8 memory pipeline for SM120: first priority

Extend C2b with a small sweep of stage depths, for example 2/3/4, and staged versus direct extraction for each loaded bit width. Start with wq_b/wo_b. Current `GEMV_STAGE_D=4` and `gemv_int8_stage_smem()` selection carry 3090-derived tuning comments; that warrants a 5090 comparison but proves no available gain. Keep slice boundaries/activation quantization fixed initially, update host and device shared-memory sizing together, and retain the pipeline's waits and workspace completion protocol.

Collect compiled register/shared-memory use, achieved occupancy, global-load stalls and measured latency. Reject variants that trade fewer instructions for extra spills or stalls. Apply C2b's numerical and ≥10% aggregate screening gates; test stage depth independently from slice-layout changes so the cause of a win is clear.

#### R2 — Specialize routed-MoE resource use for SM120: second priority

Extend C3a with the shared/fragment pipeline stages in `exl3_moe_common.cuh:14` (currently three each), holding other choices fixed first. Audit whether each instantiated BS1 tile needs the full fixed **90 KiB** requested by `exl3_moe.cu`, using the layout calculation/assertion in `exl3_gemm_inner.cuh:73`. If a variant needs less, request its proven required amount; reducing the reservation without proving that every shared-memory region and its alignment requirements fit is unsafe.

The present 90 KiB footprint precludes two such blocks sharing an SM under the documented shared-memory limit. Nevertheless, reducing it does **not** automatically double occupancy or throughput: the current grid has only 168 blocks for 170 SMs, and registers, scheduling and barriers also constrain it. First measure lower-resource variants at the existing safe grid. Any later expansion of the grid needs a new co-residency proof and lock/barrier validation, not simply removal of the current bound. Retain FP32 accumulation; the SM86-only FP16-accumulator option in `exl3_gemm_inner.cuh:11` changes numerical behavior and is not a free 5090 switch.

#### R3 — TMA and warp-specialized staging: conditional larger experiment

SM120 CUTLASS provides TMA-based warp-specialized schedules. Its GeForce GEMM path has a **1×1×1 cluster shape with no multicast**, so do not import a multi-SM multicast design from SM100. [CUTLASS SM120 differences and schedules](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms).

Only prototype TMA loading for a regular packed-trellis tile if R1/R2 profiling shows address generation, load issue or staging stalls are material. Compare complete tile fetch→decode→compute against the existing coalesced `cp.async` pipeline; include descriptor management, alignment/layout changes, barriers and additional shared memory. Keep descriptors/pointers valid across captured remaps and slot-content rewrites. Small BS1 tiles may not amortize the setup. This is a port requiring a measured win, not a flag.

This experiment stages weights already in device memory. It does not replace NVMe→pinned RAM→GPU CE transfers, enlarge the PCIe link, or allow computation to read an unfinished destination. Retain F/readiness and lease ownership requirements.

#### R4 — Complete a producer/consumer PDL pair: lower priority

CUDA PDL can overlap a successor's independent setup with its predecessor's tail, but requires appropriate producer signaling, consumer synchronization and launch/graph dependencies. Concurrency is opportunistic. [CUDA PDL and graph semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

Q RoPE's existing consumer support does not establish overlap with EXL3 GEMV: the active external INT8 sources have no matching explicit PDL producer. Investigate one GEMV→consumer pair only after identifying useful independent setup and a visible launch tail. Preserve output visibility and workspace lifetime, and validate captured edges as well as eager launches. Do not add PDL blindly to S/CW or the software-barrier MoE grid, and never depend on concurrent execution for forward progress.

The trace's 0.459 ms/step of between-kernel time inside F→nextPost is a modest diagnostic budget, not a guaranteed PDL saving or a bound on every possible overlap. The separate 2.047 ms prefix remains C0b's unresolved dependency, not a presumed PDL target. Require a pair-level improvement above measurement noise before integration; consumer flags alone are not evidence.

#### R5 — Native low-precision tensor cores: separate numerical/storage track

SM120 supports narrow-precision and block-scaled tensor-core GEMMs through its `mma.sync` family; NVIDIA supplies GeForce-specific CUTLASS examples. This is distinct from the SM100 `tcgen05`/TMEM design. [CUTLASS SM120 formats and examples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms).

**More contained candidate:** quantize resident BF16 wo_a to FP8 or a validated block-scaled format, benchmark an explicitly SM120-supported kernel and include activation-quantization overhead. This can reduce resident weight bytes as well as compute traffic, but changes numerical behavior. BS1 may underfill tensor-core tiles, so native instructions alone do not guarantee a win. First compare fixed hot-cache capacity; only then test reinvesting measured freed VRAM in expert cache, reporting that second benefit separately. Require activation/output error analysis, representative quality checks, latency including all conversion work, and actual allocated-byte accounting.

**Much larger candidate:** redesign EXL3 routed execution around native INT8/FP8/FP4 tensor-core tiles. The current 3-bit trellis/codebook layout is not an NVFP4 operand. It needs a decoder/packing stage or a different resident/transfer representation; every activation scale, residual correction and affine-codebook correction must remain accounted for. Grouping six distinct experts does not turn them into six rows sharing one weight matrix. Benchmark full decode/packing plus matmul and scratch before considering integration. Repacking transferred experts into a wider format can increase PCIe bytes and reduce cache capacity, defeating the compute saving. Treat an NVFP4/Marlin backend switch as a format/quality project, not a compatible EXL3 kernel toggle.

#### R6 — Build and library verification, before attributing gains to an upgrade

Record effective compiler target, CUDA/compiler, PyTorch/cuBLAS, Triton, CUTLASS and external source/build revisions. The EXL3 extension and SGLang JIT use different build mechanisms: `exl3_ext.py` defaults to `12.0`, while `kernels/jit/utils/arch.py:72` selects `120f` for supported toolchains where feature-specific CUTLASS code requires it. Match the target to the kernel's instructions; architecture/family-specific features have explicit compatibility rules. [NVIDIA compiler-target guidance](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features).

Inspect generated code and resource reports instead of concluding that a kernel named `cutlass_80...` must be running through an obsolete fallback. Compare any compiler/library rebuild as its own arm, with fresh private artifacts and identical model/cache settings. Do not globally change architecture suffixes, install a newer toolchain, or reuse stale JIT artifacts as part of the baseline. Preserve the pinned-source check explicitly for experimental EXL3 builds.

**Suggested order:** retain C1 and the cheap C2a test, add R1 to dense tuning and R2 to routed tuning, then consider R3/R4 only with profiling evidence. Keep R5 separate because it changes precision and potentially cache/transfer behavior. No RTX-specific candidate above has yet been benchmarked in this handoff.

## 6. Run discipline and acceptance

- Use a private experiment checkout and the existing registered serving harness. Read current applicable instructions; preserve unrelated changes. Writing this handoff does not authorize interrupting a live production service.
- Use an available maintenance slot and existing locks. If both are required, acquire `rowimg-disk.lock` before `cc-gpu.lock`; inspect drivers to avoid double acquisition. Never break a live lock.
- Set `DSV41_RUN_ROOT=/mnt/nvme1/compute-transfer-gap` explicitly when using `benchmarks/dsv41_baseline/run_arm.sh`. Store each arm separately with effective configuration and exit status.
- Run unprofiled A–B–B–A comparisons with matched prompt/token work and cache warmup; repeat beyond screening noise. Trace a separate short run for attribution. Fix cache capacity, NUMA placement, storage mirrors, memory fraction and residency cadence.
- Check completed responses, numerical/output parity, graph replay soak, lease/generation/error counters and allocation failures. Include supported prefill/decode shapes affected by a proposed production gate. Do not claim long-context validation from short decode arms.
- Report decode ms/token and throughput, TTFT, step distribution, exact compared token/request counts, CE/NVMe bytes and copies, F→nextPost, Post→first copy, and CE union. Request-p95 needs enough independent requests; graph-step p95 is not request-p95.
- Prefer a repeatable ≥1 ms/token unprofiled improvement for promoting a scheduling change, or another explicitly justified gain above measured run-to-run noise. Smaller kernel wins can be retained when independently stable and inexpensive, but do not claim PCIe benefit without earlier useful transfer activity.
- A byte-count change requires explanation: compute-only speedup and better cache behavior are different mechanisms. If faster compute just moves the stall to storage or resource contention consumes the gain, record a no-go.
- Keep source acknowledgements, generation validation and publication fences unchanged. Keep destination protection until all consumers finish. No lease-retirement shortening is part of C1–C3.

## 7. Saved analysis and reproducibility

Machine-local artifacts are under `/home/dimitri/data/divix/sglang-nvfp4/.omc/artifacts/transfer-gap-indexer-payoff/`:

- `trace/findings.txt`, `analysis.json`: method, distributions and kernel groups.
- `trace/windows.json`, `layer_records.json`, `step_records.json`: exact primary windows and measured records.
- `trace/prefix_analysis.json`: unexplained prefix details and representative timeline.
- `trace/analyze_trace.txt`: executable Python query script for this specific trace; inspect its hardcoded input/output paths before reuse.
- `pcie/analyze_pcie.py`, `summary.json`: aligned RX analysis; read-only SQLite access and interval checks.
- `pcie/indexer-payoff-pcie.sqlite`: export of the companion PCIe report using Nsight Systems 2026.5.1.

These `.omc` artifacts are local and ignored; copy them with the source reports when handing the investigation to another machine. The tables in this document preserve the primary observations independently of that local artifact directory.

To reproduce the PCIe analysis from this checkout:

```bash
python .omc/artifacts/transfer-gap-indexer-payoff/pcie/analyze_pcie.py \
  /home/dimitri/data/divix/nsys-reports/indexer-payoff-node-20260926-143541.sqlite \
  .omc/artifacts/transfer-gap-indexer-payoff/pcie/indexer-payoff-pcie.sqlite \
  .omc/artifacts/transfer-gap-indexer-payoff/trace/windows.json \
  .omc/artifacts/transfer-gap-indexer-payoff/pcie/summary.json \
  --layer-records .omc/artifacts/transfer-gap-indexer-payoff/trace/layer_records.json
```

Deliver an experiment table listing baseline SHA, flag/change, numerical checks, unprofiled latency, transferred bytes, exposed gap, earlier-transfer evidence and keep/no-go decision. Distinguish measured results, source inference and remaining unknowns.
