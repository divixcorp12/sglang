# DSV4.1 MoE RAM-miss service trace plan

The first native-stage and matched graph-node captures are recorded in
[the 23 September measurement](MOE_SERVICE_TRACE_MEASUREMENT_20260923.md).
Their 2,598 demand waits total 19.541 s; native CPU service accounts for
19.482 s, leaving a 59.596 ms signed boundary residual around CPU service.
This evidence prioritizes the read and packing investigation below.

## Goal and baseline

Attribute the 19.418 s spent in 4,400 `exl3_ram_miss_wait_kernel` calls in the
[110-replay graph-node capture](ENGRAM_DECODE_GRAPH_NODE_MEASUREMENT_20260923.md).
The same capture spent 18.213 s in the subsequent pinned-host row-copy kernel.
Those are profiler event sums, not an unprofiled throughput measurement.

The measured setup used a 50 GiB MoE pinned tier, a 5 GiB Engram cache,
`uring_direct`, and RAM-miss leases off. Hold those settings, the model, prompts,
warmup, mirror layout, and process affinity fixed for the diagnostic. Record the
actual `SGLANG_DSV41_RAM_MISS_PACK_WORKERS` value; its default is zero, but the
trace's packing interpretation depends on the value used by the server. Run on
**divix01**, using the pinned commit and isolated GPU procedure in
[ENGRAM_DECODE_TIME_HANDOFF.md](../../ENGRAM_DECODE_TIME_HANDOFF.md).

## 1. Use the stage trace that already exists

First rerun the exact diagnostic workload with
`SGLANG_DSV41_EXPERT_TRACE_PATH=<unique divix01 JSONL path>` and keep that file.
The native stage ring is enabled before the service thread starts in
[`exl3_ram_miss.py`](../../python/sglang/srt/layers/moe/exl3_ram_miss.py);
[`exl3_stream_trace.py`](../../python/sglang/srt/layers/moe/exl3_stream_trace.py)
writes schema-5 `ram_miss_request` records. If using `bench_arm.py`, pass
`--keep-trace`; its default pinned-tier size differs from this capture, so set
the resource arguments explicitly. Keep this run separate from the final
throughput comparison.

For each **demand** request, report count, layer, sequence, status, `rows_asked`,
`lanes`, backlog, bytes, per-drive bytes, retries, and these intervals in
microseconds:

| Interval | Meaning and existing source |
| --- | --- |
| `observed → reserved` | Record validation, cache lookup, slot reservation; native `StageRecord`. Service wake time is **before** `observed`. |
| `reserved → submit` | Reader setup and SQE preparation before the first submit call. This stamp is not the syscall's return. |
| `submit → first_cqe` and `submit → last_cqe` | Read/reap window; use per-extent `submit`, `cqe`, and attempts to expose stragglers. |
| Per-row last `cqe → pack_start → pack_end` | Ready-to-pack lag and bounce-to-pinned copy. |
| `pack_end → mapped → done` | Slot-map publication, fences, and the CPU completion signal. |
| `observed → done` | Whole CPU service interval. |

Separate `rows_asked == 0` from `io_uring` file reads; also separate demand from
advisory and touch records. Report p50/p95/p99, totals, and top layers for each
group, plus a timeline of the worst requests. A CQE stamp is when the service
**reaped** it, not necessarily when the drive finished. Rows pack while other
reads are pending, so do not add packing and read-window durations as if they
were serial. The existing [overlap analyser](overlap_timeline.py) handles the
per-row ordering and refuses invalid worker-mode interpretations.
Submit, CQE, and packing intervals are **N/A** on requests with no read;
do not subtract their zero stamps. If the first run points to submit-call
overhead, add optional pre/post `RowReader::submit()` stamps in the diagnostic
build before attributing time to the kernel's `io_uring` handoff.

Validate the trace before drawing conclusions: schema 5, no dropped stage
records in the measured window (`dropped_before` **and** native
`trace_dropped()`), row/extent coverage within the 16/32 stamp limits for any
per-row analysis, one-to-one sequence joins, and stage timestamps ordered
according to the record contract. Report failures, overruns, and GPU/CPU
timeout counters; reject a clean-run attribution if any occurred. Use the
benchmark's measured-window markers or explicit time boundaries to exclude
startup and warmup; `forward` labels can lag one decode step under overlap
scheduling. `SGLANG_DSV41_EXPERT_TRACE_PATH` also enables a per-batch GPU
readback in `_trace_step`, so compare **two matched node-level captures**,
one with and one without that trace, before using the instrumented capture to
explain the earlier 19.418 s. Also compare matched unprofiled runs to
quantify serving overhead. Do not use node-level Nsight for throughput.

## 2. Connect GPU publication and wait exit to the CPU timeline

The existing record begins at the CPU's first observation, and its `done`
stamp is just after the CPU release-store to `demand_done`. A graph-node trace
gives post, wait, and copy **kernel intervals**, but the CPU may observe the
request before the post kernel ends. Thus `observed - post_kernel_end` is not
the posting delay.

Make a diagnostic-only instrumentation mode, disabled by default:

1. In [`exl3_ram_miss_host.cpp`](../../python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp), emit native NVTX marks keyed by request `seq` and kind at **first** observation and immediately after the `demand_done` release-store. The row is not parsed at the first-observation point; attach it when joining to the eventual stage record. If a lease request is deferred, distinguish retries from its first observation. Retain the existing `CLOCK_MONOTONIC` stage stamps and measure each NVTX mark's offset from its nearby stamp. Add a native stamp just before the release-store; it and the existing `done` stamp bracket the signal. Update [`STAGE_FIELDS`](../../python/sglang/kernels/ops/moe/exl3_ram_miss.py) and bump the schema in [`exl3_stream_trace.py`](../../python/sglang/srt/layers/moe/exl3_stream_trace.py), with decoding tests. Keep NVTX outside the hot path when disabled.
2. In [`exl3_ram_miss.cuh`](../../python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh), optionally record `{seq, row, armed, post_before_store, post_after_store, wait_enter, wait_saw_done, wait_exit}` from thread 0 using the existing `%globaltimer` helper. Bracket the `demand_head` release-store with the two post stamps; the store executes between them. Stamp `wait_saw_done` at the successful acquire poll. Record `wait_saw_done` as N/A for an unarmed touch that never polls. Use a preallocated fixed device ring with full sequence/generation tags; never allocate, synchronize, or copy per layer inside the captured graph. Drain it only after a safe replay boundary, and detect overwrite rather than silently using partial data. Update the Python wrapper in [`ops/moe/exl3_ram_miss.py`](../../python/sglang/kernels/ops/moe/exl3_ram_miss.py) to allocate/pass that optional buffer before capture.
3. Run the same short node-level Nsight capture with CUDA, NVTX, and OS runtime tracing. Join GPU diagnostic records, CPU stage records, NVTX marks, and post/wait/copy graph nodes by sequence plus row and occurrence order. Calibrate `%globaltimer` to Nsight's timeline using matched kernel start/end samples; validate that mapped in-kernel stamps fall within their kernel intervals and report the alignment error. NVTX marks similarly anchor host `CLOCK_MONOTONIC` stamps. Never subtract raw GPU and CPU clocks directly. Reject ambiguous or unmatched requests, and report join coverage.

The joined report should expose: post-kernel work and publication; publication
to CPU observation; observation to first submit call; that call to the last reaped CQE;
packing and map publication with overlap shown; CPU completion store to GPU
observation; GPU wait-kernel tail; and pinned-host copy. If the existing trace
already shows a clear dominant CPU stage, the device ring can be deferred; the
first run is valuable without it.

The post and completion brackets locate the release-store instructions, not
the instant another processor sees the cache line. Compute cross-device gaps
as bounds, with clock-calibration error included; CPU observation may precede
`post_after_store` without indicating a bad join. Compare a fully instrumented
node capture with both the stage-trace-only node capture and the node-only
capture, since NVTX marks and device writes can alter the short gaps being
measured.

For the central 19.418 s attribution, intersect each CPU stage interval with
that request's GPU `[wait_enter, wait_saw_done]` polling interval on the
calibrated timeline. CPU service may begin before the wait kernel. Show
overlapping reads and packing as intervals, and reconcile their **union** plus
publication/observation and signal/visibility gaps against the GPU polling
duration. Report `wait_saw_done → wait_exit` separately as GPU validation and
slot translation. For an unarmed request, the polling interval is N/A even
though its wait kernel still executes.

## 3. Decide what to optimize, then compare a candidate

* Large publication-to-observation gap: inspect the service thread's 5 ms spin
  then 50 µs sleep, CPU affinity, and scheduler contention before changing I/O.
* Large SQE-to-CQE window: examine per-drive bytes, queue depth, retries,
  stragglers, mirrors, and RAM hit rate. Test cache size or reader settings one
  variable at a time.
* Large CQE-to-pack or pack-to-map tail: inspect bounce-buffer placement,
  packing workers, NUMA traffic, and slot-map publication.
* Large completion-store-to-wait-exit gap: inspect the system-scope polling and
  visibility path. Preserve its release/acquire and fail-stop protocol.
* Large row-copy share after service improvements: measure bytes per copied
  lane and effective bandwidth under the same drive load, then test a copy
  engine candidate only if the possible gain warrants its ownership machinery.

Use a matched unprofiled A/B on divix01 for any chosen change: same commit
except the candidate, cache sizes, warmup, sessions, drive layout, and GPU
availability; repeat enough to resolve the observed effect. Keep correctness
gates for expert output, cache eviction/reuse, timeout/fail-stop, and graph
replay. A short Nsight capture can explain a change but cannot be its throughput
verdict.

Deliver an analyzer under `analysis/dsv41-drive/` that reads the JSONL stage
records, optional device trace, and Nsight SQLite export, then writes a
per-request table and a summary with clock/join/coverage diagnostics. Save the
run settings, raw trace paths, and results beside this plan. If diagnostic
fields change the native record, test schema decoding, ring wrap/loss detection,
unarmed and no-read requests, and captured graph replay with hits, NVMe misses,
and slot reuse. Confirm that tracing disabled adds no graph nodes or trace
writes. The current measurement has leases off; a copy-engine port must prove
slot lifetime safety independently rather than infer it from this trace.

## Qwen gather versus DSV4.1 doorbells

These are different stages. Qwen's fast **in-graph gather** copied from a
complete 63.3 GiB pinned arena indexed directly by expert ID. DSV4.1 already
uses the same GPU row-copy kernel: its fixed pinned slabs are indexed by a
dynamic expert-to-slot map, and the RAM-miss service fills absent slots from
NVMe before the copy. Partial residency therefore does not prevent capture or
require per-expert static addresses. The current graph path is in
[`PinnedTierRowBackend`](../../python/sglang/srt/layers/moe/expert_row_plan.py)
and [`Exl3RamMissRowBackend`](../../python/sglang/srt/layers/moe/exl3_ram_miss.py).

Qwen's separate `ExpertDoorbellCopier` used CPU-scheduled copy-engine DMA from
the dense arena. It cannot be selected for EXL3 as-is because its source
indices are expert IDs, whereas EXL3 only has mutable cache slots; the
[`require_graph_gather_support` guard](../../python/sglang/srt/layers/moe/expert_format.py)
rejects it. A port would require **post-fill expert-to-slot translation**, a
stable source slot through DMA completion, and destination-slot lifetime
protection through completion or cancellation. A late copy must not write a
reused GPU slot. The existing lease protocol is relevant to that design.

The historical Qwen copy-engine microbenchmark measured 13.313 GB/s versus
12.081 GB/s for the in-kernel gather (about 10% more bandwidth), but no
controlled served decode A/B isolated that copier. Qwen's larger throughput
gain bundled several other changes. Even if that microbenchmark's ratio held
here and no extra coordination cost existed, it would save roughly 1.7 s of
the measured 40.4 s replay span, or about 4%, while leaving the 19.418 s
RAM-miss wait untouched. Treat that as a rough prioritization estimate, not a
DSV4.1 prediction. Source: [Qwen transfer notes](../../MOE_EXPERT_TRANSFER.md)
and [doorbell brief](../../MERGE_BRIEF_DOORBELL.md).
