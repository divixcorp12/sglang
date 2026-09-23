# DSV4.1 asynchronous CPU readbacks, 23 September 2026

## Changes

The optional MoE stream trace now copies cumulative GPU graph counters to one pinned CPU buffer and polls a CUDA event on later scheduler checks. It coalesces samples while the buffer is busy. At orderly shutdown, the service pauses its native reader after the existing GPU barrier, then records final GPU and demand-row totals. A failed shutdown quarantines the buffer. This removes the trace's per-batch `.tolist()` wait from serving; final shutdown may still wait.

The dynamic MoE residency policy has an opt-in `SGLANG_MOE_ASYNC_RESIDENCY_SCORES=1` path. At a boundary, it advances scores on the GPU, enqueues a pinned score snapshot, and returns. A later forward polls the event and chooses promotions from the completed snapshot and the cache's current resident set. Boundaries that arrive while a copy is pending advance scores and request a fresh snapshot. Capture reset drains and discards a pending capture-era snapshot before serving. The default remains `0` until a full served throughput comparison; `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` is incompatible with this EXL3 partial pinned tier and was not used.

## Verification on divix01

The isolated test checkout was at detached diagnostic commit `c39f1ab473490a5e0ed9dd08b487b045ae5cdfb2`, based on the real-server code `d337301dd1163ec46a5a76c2d8da41cdc34d19f4`. It did not change the existing benchmark checkout or production server. Targeted residency, hot-cache, and trace tests: **100 passed, 27 subtests passed**. Shutdown, telemetry, clock, and benchmark tests: **143 passed, 12,000 subtests passed**. Independent code review found two lifecycle issues; both were fixed and re-reviewed before this diagnostic run.

The two-session Nsight arm used four row-packing workers, `uring_direct`, the 50 GiB partial MoE tier, native Engram graph host nodes, and `SGLANG_MOE_ASYNC_RESIDENCY_SCORES=1`. Its run folder is `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/sync-removal-stage-20260923b/run-20260923-104620`; report and SQLite export are `/mnt/nvme1/dsv41-nsys/sync-removal-stage-20260923b-20260923-104635.nsys-rep` and the matching `.sqlite` on divix01. Preflight, environment verification (33 variables), health, and warmup gates passed. The shortened result gate rejected **2 results instead of 8**, with **zero request errors**, as expected. The two completion texts, reasoning texts, and token counts matched the earlier two-session `moe-pack4-stage-20260923-081207` diagnostic.

Nsight recorded five 61,440-byte GPU-to-CPU score copies. Their matched `cudaMemcpyAsync_v3020` API calls took **0.0128, 0.0108, 0.0562, 0.0271, and 0.0213 ms**, **0.128 ms total**. The earlier sampled run measured three blocking score readbacks totaling **0.895 s** during decode. These runs differ in tracing mode and serving conditions, so this establishes removal of the long host API stall in the short diagnostic, not a throughput gain. The Engram hash-ID D2H graph nodes and any prefill readbacks remain separate data dependencies.

## Remaining gate

Run a matched, unprofiled eight-session comparison on divix01 with the async score flag off and on, holding the four-worker packer and other server settings fixed. Compare decode rate, output parity, MoE RAM-miss rows, promotions, and residency-boundary latency. The prior whole-arm page-cache gate counts startup residency growth, so resolve that gate before calling either arm formally valid. The async policy may apply a boundary decision one or more forwards later when its score copy is pending; that timing difference needs the serving comparison.
