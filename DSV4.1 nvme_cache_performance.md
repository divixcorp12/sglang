# DSV4.1 NVMe, RAM, and VRAM cache performance

Evidence snapshot: **2026-09-19**, local branch `master`, inspected at `3be5d97b83`. This document records the current transfer path and six optimization priorities. It does not report a new benchmark or an implemented optimization. Historical experiments ran at the commits recorded in [DSV41_REFERENCE.md](DSV41_REFERENCE.md), particularly §§16–18; the local inspection commit is not the provenance of every measurement below.

**Measured** means an existing experiment or artifact reports the result. **Code observation** means the current implementation establishes the behavior. **Derived** means arithmetic or an upper bound calculated from those observations. **Proposed** means work whose benefit has not yet been demonstrated.

## Current state

The working design is **NVMe → bounded pinned RAM cache → VRAM hot cache**. The complete expert set cannot fit in the available RAM. Routed experts occupy about **204.5 GB** on disk; Engram tables occupy another **203 GB**. A raw EXL3 expert contains **13,315,596 bytes**; the six streamed tensors occupy **13,315,584 bytes**. These two row sizes describe different representations and should not be interchanged in accounting. See [DSV41_REFERENCE.md](DSV41_REFERENCE.md), §§3 and 16.2.

The measured configuration allocates **71,680 MiB / 70 GiB** to pinned expert RAM: **5,644 expert rows**, approximately **36.7% of 15,360 experts**. Engram receives **5 GiB** separately. The approximately 90 GB host-memory envelope also has to cover process state, staging, and other allocations; it is not a 90 GB expert cache. The VRAM expert budget is **14,336 MiB**, of which graph-gather scratch consumes **3,048 MiB**, leaving **888 resident expert slots**. The earlier eager configuration had **1,128 slots** because it did not reserve this graph scratch.

The hierarchy is **inclusive**: an expert in VRAM also retains its RAM copy. A VRAM eviction therefore drops the GPU copy without a transfer back to RAM. Distinct expert coverage is the RAM tier's coverage, not the sum of RAM and VRAM rows. This is deliberate behavior in [`Exl3ExpertFormat.inclusive_pinned_tier`](python/sglang/srt/layers/moe/exl3_expert_format.py), lines 146–166.

For each layer in option C decode:

1. The router identifies six experts; the hot cache supplies VRAM hits.
2. The GPU posts the remaining expert IDs to the native RAM-miss service.
3. The service reads absent rows from the original shards into aligned staging, then copies their tensor segments into persistent pinned slabs.
4. The GPU waits for the service and gathers the required RAM rows into VRAM scratch.
5. The fused EXL3 MoE consumes the hot slots and scratch rows.

This path is functional under breakable CUDA graphs at batch size one. Prefill stays eager. [`Exl3RamMissRowBackend.translate`](python/sglang/srt/layers/moe/exl3_ram_miss.py), lines 199–236, posts and waits before the inherited [`PinnedTierRowBackend.post`](python/sglang/srt/layers/moe/expert_row_plan.py), lines 368–372, performs the gather.

### Measurements that should remain distinct

| Evidence | Result | Scope |
|---|---:|---|
| Phase 3b option C corpus, prefetch off | **2.781 tokens/s** | Mean of four session decode rates; 888 hot slots; [reference](DSV41_REFERENCE.md) §17.6 |
| Same corpus with the previous-token advisory predictor | **2.780 tokens/s** | Same capacity; difference is within session variation |
| Earlier eager corpus | **1.664 tokens/s** | Same four sessions, but 1,128 hot slots; not an equal-capacity kernel-only comparison |
| Newer boundary-study `c32` artifact observed during the prior read-only inspection | **2.823090 tokens/s** | `mean_decode_tok_s` in `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-overlap/boundary/corpus-c32.json` |
| Separate node-mode trace | **391 ms/decode step** | Approximately 50.6 steps in one window; [reference](DSV41_REFERENCE.md) §18.2 |

At that prior artifact inspection, `corpus-c0.log` showed the no-decode-boundary arm starting; completed `c0` and `casync` summaries were not available. This is a dated observation, not a statement about their latest execution status. The 2.823 baseline does not establish a promotion-policy winner.

The 391 ms trace is a different workload/window from the 2.781 tokens/s corpus result. Its breakdown is:

| Component | Approximate ms/step | Interpretation |
|---|---:|---|
| RAM-miss service wait | **190** | GPU waiting for absent RAM rows to become usable; includes the service path, not only physical-drive read time |
| Pinned RAM → VRAM gather | **128** | About 121.2 rows/step and 1.055 ms/row |
| Compute | **16.8** | MoE, attention, router, mHC, and other compute |
| Residency-boundary stall, amortized over this trace window | **41** | Mostly a single long interval; not a uniform per-step gap |
| Two Engram graph breaks | **11** | CPU/GPU boundary work |

Rounded components do not exhaustively sum to 391 ms. The window contains approximately **18.8 RAM misses per step**, with **10.16 ms of service wait per RAM-miss row**. Compute, waits, and gathers were serialized. The same session measured approximately 390 ms/step without Nsight in the separate sampling run; that agreement applies to this node-mode experiment, not every profiler configuration. See [DSV41_REFERENCE.md](DSV41_REFERENCE.md), §§18.2–18.3.

## 1. Resolve periodic promotion stalls first

**Measured:** every 32 decode forwards, the current policy can promote experts synchronously. One boundary costs approximately **1.2–2.1 seconds**, equivalent to roughly **40–65 ms/token** when spread over 32 tokens. The sampling run attributes time to synchronous row preparation, copy completion, and CPU policy decisions. See [DSV41_REFERENCE.md](DSV41_REFERENCE.md), §18.6.

**First comparison:** finish interpreting matched `c32` and `c0` arms. `c0` disables decode boundaries while retaining prefill residency updates. Compare decode throughput, boundary latency, VRAM misses, RAM misses, and total expert bytes. Removing the stalls is useful only if the resulting residency policy does not lose more time through extra demand transfers. The existing evidence does not yet establish that tradeoff.

**Code observation:** `SGLANG_MOE_HOT_ASYNC_PROMOTIONS=1` does not make the current EXL3 path asynchronous:

- [`Exl3ExpertFormat.source`](python/sglang/srt/layers/moe/exl3_expert_format.py), lines 140–141, always returns `None`; EXL3 has six streamed tensors, listed at lines 33–40.
- [`ExpertHotCache.stage_reassign`](python/sglang/srt/layers/moe/expert_hot_cache.py), lines 688–694, routes this spec-only source through `_load_reserved`.
- `_load_reserved`, lines 419–424, selects `_load_reserved_in_chunks` for the six-tensor pinned-tier case.
- That loader waits and calls `current_stream.synchronize()` before completing each promotion, lines 475–495. It returns no pending promotion for the manager's asynchronous publication branch at lines 1929–1931.

Consequently, `casync` tests the flag's behavior in this implementation; it does not demonstrate the benefit of a fully asynchronous EXL3 transfer pipeline.

**Proposed:** if decode promotions repay their cost, implement bounded background preparation and transfer, with demand reads taking priority. Retain source-slot protection until the copy completes, keep destination slots unpublished while loading, and publish READY only after the completion event. The current `host_use` scope pauses the RAM-miss thread: see [`NativePinnedSlotTable.before_host_use`](python/sglang/srt/layers/moe/exl3_ram_miss.py), lines 164–183. Keeping that pause open across a background promotion would still obstruct demand service; removing the synchronization without replacing its ownership guarantees risks slot reuse during a copy.

The measured stall is the available opportunity, not a promised gain. Background promotions still consume disk and PCIe bandwidth, and their residency benefit may arrive later.

## 2. Measure the faster drive path before further io_uring tuning

**Measured:** the nvme2 direct-read benchmark sustains approximately **1.64–1.77 GB/s**, or **7.5–8.1 ms/expert** across the reported batch sizes. This is consistent with the Gen3 x2 path. Raising queue depth cannot remove the physical link limit. The benchmark and conditions are recorded in [DSV41_REFERENCE.md](DSV41_REFERENCE.md), §§16.3–16.4.

The owner-selected layout puts experts on **nvme1, Gen3 x4**, and leaves Engram on **nvme2**. At the earlier read-only inspection, the expert copy on nvme1 was absent. The reference's approximately **3.4 ms/expert** for an idle x4 drive remains an estimate. The drive also hosts an `op-reth` workload, and its QLC behavior and competing I/O matter.

**Proposed:** once the expert copy is available, compare the same expert-row reads and serving workload under representative competing load. Record sustained bytes/s, demand-read latency distributions, queue delay, and end-to-end decode throughput. Avoid treating advertised drive bandwidth or a clean idle-drive result as the serving result.

For scale only, **derived** arithmetic using 18.8 misses/step gives about **88 ms/step** from replacing 8.1 ms of physical read time with 3.4 ms, if those timings apply to the same path and all other costs remain unchanged. This is a conditional illustration, not a forecast: the traced 10.16 ms/row service wait includes more than drive reads, and the faster-drive figure is unmeasured on this workload.

## 3. Remove the extra CPU copy between NVMe and persistent RAM

**Code observation:** the native service reads into bounce storage and copies tensor segments into pinned cache slabs. The `memcpy` loop is in [`exl3_ram_miss_host.cpp`](python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp), lines 266–273. This means a RAM miss moves the expert payload through another CPU memory pass before it is published to the GPU.

**Measured, different paths:** the standalone read/split benchmark reports approximately **1.43–1.48 ms/row** for splitting, while the earlier eager corpus reports approximately **2.18 ms/row** in its cold decode arm. These are [reference](DSV41_REFERENCE.md) §§16.4 and 16.8 measurements. The native option C service's split cost has **not** been isolated, so neither value is a measured saving available in the present graph path.

**Proposed:** evaluate a raw BLOB RAM tier: direct reads land in stable aligned cache slots holding the original row bytes, and RAM → VRAM segment gathering selects the required spans and produces the kernel-ready layout. This could remove bounce-to-slab splitting while preserving the original checkpoint shards. It is distinct from repacking the entire model on disk.

The design must account for direct-I/O alignment, per-slot prefix/tail padding, source segment offsets, stable registered addresses, and cache publication only after successful reads. Raw EXL3 tensor offsets are not uniformly 16-byte aligned, so moving layout conversion to the GPU can change gather behavior. The current gather has aligned vector copies plus narrower fallback paths in [`copy_expert_row_lane`](python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh), lines 35–76. Measure the net service-plus-gather cost rather than assuming a removed CPU copy is free performance.

The existing disk-repack experiment did not clear its acceptance threshold on nvme2. A raw RAM representation remains a separate proposed optimization; the approximately 1.5–2.2 ms eager split timings justify investigating it, not declaring its benefit established.

## 4. Make the existing RAM budget more effective

**Code observation:** [`ExpertPinnedHostCacheManager`](python/sglang/srt/layers/moe/expert_stream.py), lines 480–493, distributes slots round-robin across layers until the byte budget is exhausted. With equal expert row sizes this produces approximately **141–142 rows/layer**, irrespective of each layer's reuse pattern or marginal miss reduction.

**Proposed:** use per-layer reuse traces to estimate how many demand misses an additional RAM slot prevents, then allocate the fixed byte budget where the marginal reduction is largest. Evaluate on held-out sessions, including cold starts. A layer with many misses is not automatically the best recipient: its working set may be so diffuse that additional slots yield little reuse.

Retain minimum capacity for current demand routes and inclusive hot residents. The current format protects experts while they are resident or loading and clamps hot capacity against the pinned tier; shrinking a layer's RAM allocation can also reduce its usable VRAM cache. These constraints are documented in [`Exl3ExpertFormat`](python/sglang/srt/layers/moe/exl3_expert_format.py), lines 146–166.

Removing the RAM copies of VRAM-hot experts would increase distinct coverage, but it would also remove the current cheap-eviction guarantee. An exclusive hierarchy would require a deliberate VRAM → RAM demotion or reread policy and new slot-lifetime handling. It should be evaluated as a separate cache design, not treated as an uncomplicated capacity switch.

At the observed service cost, preventing one additional RAM miss per token is worth roughly **10 ms/token** before secondary effects. This is a derived sensitivity, not a measured result for weighted allocation. Record demand misses separately from advisory and promotion reads so cache-policy gains are not hidden by extra background traffic.

## 5. Recover VRAM reserved for per-layer scratch

**Code observation:** every hot cache allocates its own hot slots plus gather scratch in [`ExpertHotCache.__init__`](python/sglang/srt/layers/moe/expert_hot_cache.py), lines 111–155. The manager subtracts all layer scratch bytes from the residency budget, lines 1153–1162. EXL3 requires at least six scratch rows for each BS1 layer in [`exl3_fused_moe_for`](python/sglang/srt/layers/quantization/exl3_fused_moe.py), lines 149–159.

**Measured capacity:** 40 layers × six rows = **240 scratch rows**, totaling **3,195,740,160 bytes / 3,048 MiB**. They consume budget that could otherwise hold hot experts.

**Proposed:** share one six-row staging set across layers whose gather and compute lifetimes do not overlap. For today's serial BS1 path, the **derived theoretical recovery** is:

| Item | Capacity |
|---|---:|
| Existing per-layer scratch | 240 rows |
| Shared six-row scratch | 6 rows |
| Potentially reclaimed | **234 rows / 3,115,846,656 bytes / about 2.90 GiB** |
| Illustrative hot capacity at the same total budget | **888 → about 1,122 slots** |

These are storage calculations, not a measured hot-hit-rate or throughput improvement. Actual allocation must still satisfy per-layer constraints.

EXL3 already accesses expert tensors through pointer tables, but [`slot_pointer_tables`](python/sglang/srt/layers/quantization/exl3_fused_moe.py), lines 55–65, currently derives all addresses from each layer's contiguous cache tensors. Supporting shared scratch requires changes to destinations, pointer construction, and allocation accounting. It cannot be obtained by lowering the current scratch cap: the EXL3 path explicitly requires one scratch row per route. The smaller fused-kernel temporary buffers are already shared across serial layers, lines 30–48, providing an existing lifetime pattern to examine.

Concurrent forwards, transfer lookahead, or speculative verification would require enough independent staging buffers for their overlapping lifetimes. The direct insert-on-miss and GPU residency options are not ready-made EXL3 solutions: the format's composed requirements reject GPU residency update and the expert doorbell in [`expert_stream_requirements.py`](python/sglang/srt/arg_groups/expert_stream_requirements.py), lines 321–329.

## 6. Pursue selective overlap and prefetch

The useful overlap is mostly **transfer with transfer**. There is too little compute to hide the entire I/O path: [DSV41_REFERENCE.md](DSV41_REFERENCE.md), §18.3, bounds in-layer copy/compute overlap at approximately **15.5 ms/token / 4%** of the traced step.

**Proposed first overlap:** gather the layer's already-resident RAM rows while the native service reads its missing rows, then gather the newly completed rows. The current post → wait → gather ordering prevents this. The trace-derived upper bound is approximately **20.8 ms/token / 5.3%**, conditional on the partial per-layer trace alignment; it is not an implemented speedup. Destination ownership, source protection, and the final MoE dependency must remain explicit.

**Measured predictor limitation:** the existing previous-token advisory predictor moved only **30 advisory rows over 511 decode tokens** in the Phase 3b corpus and changed throughput from **2.781 to 2.780 tokens/s**. Its lack of useful missing-row predictions does not establish that the prefetch mechanism itself has no value.

**Proposed selective lookahead:** confidence-gated predictions one or two layers ahead may create lead time for absent RAM rows. The reference's estimated gains are approximately **6% for one layer ahead and 10% for two**, not measured serving improvements. Demand reads must retain priority, and the evaluation must count extra bytes, cache pollution, and useful arrivals before demand. Detailed predictor and speculative-decoding designs should be evaluated against this transfer budget.

The RAM → VRAM gather already moves a 13.3 MB row in about **1.055 ms**, roughly **12.6 GB/s** of payload bandwidth in the trace. Its kernel uses coalesced 16-byte copies when aligned. There is stronger evidence for reducing transferred rows and removing serialized waits than for expecting a large gain from copy-kernel tuning alone.

## Evaluation rules for subsequent work

Keep workload, cache budgets, session order, warmup treatment, and competing drive load explicit. Report decode throughput and tail stalls together with VRAM demand misses, RAM demand misses, physical read bytes, background reads, and promotion traffic. A lower demand-wait counter can coexist with worse throughput if speculation or promotions saturate the shared path.

Do not add the proposed gains as if they were independent. Faster storage changes the value of split removal and prefetch lead time; more VRAM changes demand traffic; weighted RAM allocation changes both misses and promotion costs. Establish each improvement with a matched baseline and then measure their combination.

Preserve the current correctness and failure-path requirements. The truncated-model R4 eager-versus-fused comparison fails, while graph versus debug-eager using the same fused kernel agrees bitwise; an end-to-end fp32 comparison has not resolved the difference. See [DSV41_REFERENCE.md](DSV41_REFERENCE.md), §§17.4 and 17.8. EXL3 graph capture is currently BS1, and the DSV4 alternate-stream overlap remains disabled because of the documented capture defect. Performance results do not close these separate correctness questions.

## NVMe → RAM prefetch has a different error budget

**Assessment:** the owner's proposed direction is sound. Prefetching into RAM can
tolerate more speculative candidates than prefetching into VRAM when there is spare
storage bandwidth and the fetched rows survive until useful. The objective should
be **maximize avoided demand stalls within an I/O and cache budget**, rather than
maximize the number of experts fetched.

A RAM prefetch does **not** make the later RAM → VRAM transfer faster. It moves the
NVMe read and host preparation earlier. A correctly prefetched expert still needs
roughly 1.06 ms of PCIe transfer if it misses VRAM in the measured trace, but can
avoid up to roughly 10 ms of exposed RAM-miss service. These constants describe
that setup; they are not universal or additive guarantees.

| Property | Prefetch into VRAM | Prefetch into RAM |
|---|---|---|
| Speculative resource consumed | GPU-link bandwidth and scarce VRAM | Storage bandwidth, host preparation, and RAM capacity |
| Wrong prediction | May transfer a whole unused expert over PCIe and displace a GPU-hot expert | Need not cross the GPU link; may displace a useful RAM expert |
| Reuse window | Limited by GPU residency and staging lifetime | Can span later layers and subsequent tokens until eviction |
| Useful idle work | Must fit available GPU-link and execution windows | Can run during RAM → VRAM gathers as well as compute, subject to shared topology |
| Remaining bottleneck | Demand misses on either tier | RAM → VRAM transfers still remain |
| Required correctness rule | Native router determines executed experts | Native router determines executed experts; a missed prediction falls back to demand I/O |

The relevant difference from the Qwen transfer case is **where the speculative
traffic goes and which resource has spare capacity**. It is not a universal
relaxation of bandwidth limits. On a saturated storage device, extra wrong reads
still compete with necessary reads. Shared PCIe/root-complex and memory paths can
also couple disk and GPU traffic; overlap must be measured on divix01.

A row not used by the immediately predicted token is not necessarily wasted: it
may be used by a later token before eviction. Conversely, a correct prediction
arriving after its deadline, or evicted before use, may save little or no latency.
Evaluate reuse and timeliness together.

### Existing measurements already address this idea

[DSV41_REFERENCE §18.4](DSV41_REFERENCE.md#184-prefetch-can-a-predictor-hide-the-nvme-reads)
explicitly evaluates **NVMe-tier candidates**, not just VRAM prefetch. Its poor
previous-token result therefore does not rule out a better RAM predictor.

The probe applied a future layer's existing gate to the current layer's MoE input.
It measured candidates, not an implemented lookahead prefetcher's throughput:

| Predictor | Recall of needed NVMe rows | Useful candidates/token | Candidates not routed by that token |
|---|---:|---:|---:|
| Previous token's routes | 0.000 | 0.00 | 0.42 |
| L+1 top-6 | 0.481 | 8.79 | 22.25 |
| L+1 top-12 | 0.683 | 12.47 | 90.53 |
| L+2 top-6 | 0.411 | 7.16 | 26.91 |
| L+1 highest confidence bin | Not separately reported | 4.52 of 7.31 candidates | 2.79 |

The confidence bin is based on a gate-score-margin transform; it is not a calibrated
probability of expert use. These measurements use the probe's routing and cache
state. A live predictor changes both cache contents and later timing.

**Derived traffic illustration:** at 13.3 MB per expert, L+1 top-6's 22.25
immediately unused reads add about **296 MB/token**; top-12's 90.53 add about
**1.21 GB/token**. At the approximately 1.7 GB/s measured drive throughput, the
latter alone takes roughly **0.71 seconds of storage service**. That cannot fit
into a roughly 0.39-second step at unchanged throughput. Some rows may be reused
later; this calculation illustrates why unrestricted breadth is not free.

The reference estimates around 200 ms/step of opportunity outside the RAM-miss
wait. Treat this as a scheduling opportunity estimate, **not a direct measurement
of spare disk capacity**: the 190 ms wait includes CPU preparation/publication,
and promotions, Engram, and other processes also use resources.

The highest-confidence subset is a better starting point. One layer of lead was
estimated at about 4.8 ms and two at about 10 ms, compared with roughly 10 ms of
service per missed row. Two layers can offer enough lead for one row under
favorable conditions, but several candidates queue behind one another. A faster
decode path also shortens future prefetch windows.

### Candidate predictor designs

All three choices below are **proposed**, not implemented or benchmarked here.

1. **Existing gates evaluated early.** Start with the measured L+1 and L+2
   approximations. Filter out RAM/VRAM hits and in-flight reads, then admit only
   high-value candidates. This establishes whether available lead and I/O slack
   are usable before adding training.
2. **A small learned predictor using current target features.** Predict
   `(future layer, expert)` rankings from an earlier hidden state, recent routes,
   and request context. Evaluate horizons such as one, two, and four layers;
   farther horizons offer more lead but may lose accuracy. Runtime cache filtering
   and cost-aware ranking must prevent good predictions of already-cached experts
   from dominating the apparent score.
3. **A next-token or short-window working-set predictor.** Use late target features
   and request history to predict a set likely to be needed over upcoming tokens.
   This is particularly suitable for a persistent RAM cache: immediate-token
   precision is not the only objective. It must beat normal LRU retention and
   account for experts displaced by speculative admission.

There is primary research supporting lightweight early expert predictors:
[Pre-Attention Expert Prediction and Prefetching](https://arxiv.org/abs/2511.10676)
uses pre-attention activations with small learned linear functions and a ranking
objective. [MoE-Infinity](https://arxiv.org/abs/2401.14361) uses activation traces to
guide cache replacement and prefetch. These are design precedents; their published
accuracy or speedups do not establish performance on DSV4.1 or this storage stack.
A same-layer pre-attention predictor may also have too little lead to hide a
whole NVMe read here.

### Proposed admission and scheduling contract

Predicting accurately is only half the design. The RAM service should:

- Keep required demand reads ahead of unsubmitted speculative reads. Use a shallow,
  bounded speculative queue: a demand cannot preempt a read already executing.
- Schedule by estimated use deadline and saved stall, not confidence alone. Account
  for queued bytes, current read latency, predictor overhead, and available lead.
- Deduplicate against ready rows and in-flight reads. If demand requests an in-flight
  prefetched row, join that request and raise its urgency rather than rereading it.
- Limit speculative RAM occupancy and insertion rate. Consider probationary admission
  or lower replacement priority until first actual use; retain existing protection
  for GPU-hot rows and active consumers.
- Publish independently completed rows when safe, retaining them if later speculative
  work is cancelled. Expire stale *requests* without automatically discarding useful
  completed cache entries.
- Reassess the budget as drive contention, RAM pressure, and decode speed change.
  More lookahead should not automatically imply more submitted disk reads.

**Current implementation limitation:** the native advisory path reads one row at a
time but checks for demand only between reads. It reserves victims for its missing
set before reading, and an interrupted advisory releases all its reserved slots,
including already-read rows. The old victims are not restored. See
[`Service::serve` and its advisory path](python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp),
around lines 796–885. A higher-volume predictor needs this admission/publication
behavior addressed alongside prediction; otherwise useful completed work can be
discarded and the demand cache polluted.

An evaluation heuristic is:

```text
expected benefit
  = probability of useful retention × exposed wait avoided by timely completion
    − added demand delay
    − eviction-induced future misses
    − predictor and scheduling overhead
```

There is no single required precision such as 50%. A wrong read using otherwise
idle capacity can have little immediate latency cost, while one queued ahead of
demand or evicting a soon-needed row can be expensive.

## Could DSpark help a RAM predictor?

**Possibly as a source of future-token features, but it is not an existing expert
prefetch head.** There are three distinct proposals:

| Approach | What it could provide | Extra work and cost |
|---|---|---|
| Small predictor on late target hidden states | Next-token/window target-expert scores | New predictor and feature taps; no DSpark draft execution required |
| DSpark-assisted predictor only | Draft hidden features and optionally proposed tokens before future target execution | Draft loading/runtime plus a new mapping to target expert scores |
| Full DSpark speculative decoding plus prefetch | A proposed token block before target verification | All draft costs, multi-token target verification, acceptance/commit accounting, and prefetch integration |

### What the code establishes

- DSV4.1's DSpark configuration taps target layers **37, 38, and 39**. Target capture
  records completed post-layer states, averaged over mHC streams; these are not
  each target layer's router inputs.
  [Target capture](python/sglang/srt/models/deepseek_v4.py), around lines 4815–4822;
  [reference §14.3](DSV41_REFERENCE.md#143-dspark-target-layers-and-planner).
- The draft has **three stages, each with 128 experts**, whereas the target has
  **40 layers, each with 384 experts**. Its separate `mtp.*` weights and routers
  do not give a direct identity mapping from draft expert IDs to target IDs.
  [Draft stages and loader](python/sglang/srt/models/deepseek_v4_dspark.py).
- Draft `forward` returns final hidden states. Those features and optional token
  proposals could feed a separately trained target-route predictor; exact target
  routes are still unknown until the target computes its own intermediate states.
  [Draft output](python/sglang/srt/models/deepseek_v4_dspark.py), around lines 988–1009;
  [draft/verify orchestration](python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py).
- DSpark's confidence head concerns token proposals, not calibrated probability
  that a particular target expert will be needed.
  [`compute_confidence`](python/sglang/srt/models/deepseek_v4_dspark.py), around lines 1046–1068.

**Timing consequence:** taps near the end of a target pass cannot hide that pass's
earlier-layer misses. They can condition predictions for subsequent tokens. For
same-token L+1/L+2 prefetch, earlier target features are the more direct signal.

**Representation consequence:** applying target gates directly to draft hidden
states is unvalidated. The hidden spaces, stages, and expert identities differ.
A learned or calibrated mapping to target `(layer, expert)` scores is a research
proposal. Predicted token IDs alone also do not reveal exact target expert routes.

### Cost and integration limits

Resident draft experts alone cost approximately **6.8 GB** by the byte calculation
in reference §18.5, before the rest of the draft runtime. Taking that budget from
target hot residency could substantially increase target transfers. Streaming the
draft instead introduces additional disk and GPU-link traffic; it is not free.

The documented EXL3 blockers remain:

- The current `dsv41-full40` serving directory omits the MTP draft weights.
- The draft loader lacks the target loader's EXL3 adaptation.
- The process-wide EXL3 streamer assumes the target's 384-expert layout and rejects
  the draft's 128-expert layers.
- Full speculative decoding additionally needs compatible multi-token verification:
  the current option C path is sized for one token.

Predictor-only DSpark use could avoid multi-token target verification and might
skip vocabulary projection/Markov token sampling if only draft features are used.
It still requires draft stages, feature-context plumbing, loading/layout work,
memory, and execution time. No such predictor-only mode is implemented or measured.

**Recommendation:** first compare cheap lookahead and a small predictor over
existing target features. Keep DSpark-assisted prediction as a separate research
arm, especially if DSpark will later run for speculation anyway. The previously
parked DSpark implementation remains parked; this document records the option
without implementing it.

Do not transfer Qwen's speculative acceptance break-even to predictor-only use.
A cache predictor that preserves native routing is judged by net latency and
traffic, not token acceptance. Full speculative decoding must additionally measure
acceptance and physical bytes per accepted token on DSV4.1.

## Evaluation before choosing a predictor

Use recorded traces first, then paired serving measurements if a prototype is
authorized. Proposed sequence:

1. Replay a rolling prediction horizon against realistic read queues, RAM capacity,
   inclusivity protections, and eviction. Include a same-budget baseline without
   prediction and an oracle future-access upper bound. Static candidate recall
   alone cannot model the changed cache.
2. Compare gate lookahead, learned early-target features, and late-target next-token
   features on held-out sessions. Add DSpark features only as a separately costed arm.
3. Sweep confidence thresholds, layer horizons, speculative bytes in flight, and
   probationary capacity. Include long prompts and warm sessions as well as the
   existing cold 256-token-prompt corpus.
4. Check native-route/output preservation, slot lifetime/publication, cancellations,
   and demand fallback. Predictor errors must affect timing rather than expert
   selection.
5. Measure end-to-end benefit after charging predictor compute, extra model memory,
   CPU work, all reads, and co-tenant contention. Re-evaluate after drive, layout,
   or cache-size changes; their benefits cannot be added independently.

Report:

- Exposed RAM-miss service time/token and absolute demand disk misses/token.
- Useful prefetches completed before demand, late prefetches with partial wait saved,
  and reuse before eviction across subsequent tokens.
- Total physical disk bytes/token: demand, speculative, promotion, cancelled, and
  reread traffic; distinguish cancelled reads from completed rows retained in cache.
- RAM evictions caused by speculation and subsequent demand misses attributable to
  them, using a counterfactual baseline or controlled replay.
- GPU-link bytes/token, predictor cost, RAM/VRAM footprint, mean throughput, and
  p95/p99 inter-token latency.

Existing service `rows_read` counters count successfully published request rows;
interrupted advisory reads can have incurred physical I/O without increasing that
counter. They are insufficient by themselves to measure speculative bandwidth.
The existing R4 fused-versus-eager numerical divergence remains a separate
validation limitation; prefetch experiments must not silently treat it as resolved.
