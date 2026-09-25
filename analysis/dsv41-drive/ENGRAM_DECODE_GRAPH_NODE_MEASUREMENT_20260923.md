# Decode graph-node measurement, 2026-09-23

The previous [sampled served-path measurement](ENGRAM_DECODE_SAMPLED_MEASUREMENT_20260923.md)
found that expensive visible `.tolist()` readbacks happen in prefill, while
decode time falls inside one CUDA graph replay per token. This diagnostic
capture traces the graph's individual nodes to attribute that decode time.

## Capture and validity

- divix01 worktree commit: `d337301dd1163ec46a5a76c2d8da41cdc34d19f4`.
- Key server settings: `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1`,
  `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=0`, 5 GiB native Engram cache,
  `SGLANG_MOE_PINNED_HOST_MB=51200`, `SGLANG_MOE_EXPERT_FILE_READER=uring_direct`.
- Nsight 2026.3.2: `--cuda-graph-trace=node:host-only`, CUDA/NVTX/OSRT tracing,
  no CPU sampling. A temporary launcher wrapper replaced the harness's
  hardcoded graph-level setting for this run; the wrapper was removed after
  capture. Four warmup rounds completed before collection.
- Two timed requests generated 7 and 103 tokens. The result gate rejected
  two records because it expects eight; both requests had no reported error.
- Report on divix01:
  `/mnt/nvme1/dsv41-nsys/graph-node-attrib2-20260923-20260923-004107.nsys-rep`.
  SQLite export: same stem with `.sqlite`. Run directory:
  `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/graph-node-attrib2-20260923/run-20260923-004052`.

The export contains 110 graph launches, 220 graph host callbacks, 569,580
graph kernel events, and 58,080 graph memcpy events. The two temporal clusters
of 7 and 103 replays match the requests' generated token counts. For each
callback, `CUDA_HOST_CALLBACK` matches one graph-host stream activity on
`(start, end, correlationId)`; the measured intervals are identical.
Graph activities were grouped by replay correlation ID. Kernel names were
resolved through `StringIds`; Engram nodes were labeled by their repeated
execution order and adjacent D2H/H2D node IDs.

## In-graph attribution

| Graph activity | Calls across 110 replays | Summed duration | Per replay mean | What it does |
| --- | ---: | ---: | ---: | --- |
| `exl3_ram_miss_wait_kernel` | 4,400 | 19.418 s | 176.5 ms | GPU polls the CPU-served MoE demand doorbell |
| `copy_expert_row_segments_gpu_kernel` | 4,400 | 18.213 s | 165.6 ms | GPU reads mapped pinned-host MoE expert rows into device slots |
| All other graph kernels | 560,780 | 1.859 s | 16.9 ms | Includes model math and bookkeeping |
| Engram layer 1 host callback | 110 | 427.207 ms | 3.884 ms | Cache lookup and possible `io_uring` row fetch |
| Engram layer 14 host callback | 110 | 260.101 ms | 2.365 ms | Same path for the second Engram layer |

The two MoE kernels total **37.630 s**, or **95.3% of summed graph kernel
duration** (39.489 s). Their per-replay combined duration has median
330.390 ms and p95 496.640 ms (linearly interpolated). Summed first-to-last
graph activity spans
40.394 s across the 110 replays; the MoE pair is 93.2% of that sum.
The two Engram callbacks together total 687.308 ms, or 1.7% of the
summed replay spans. Layer 1 callback p50/p95 is 3.927/4.389 ms;
layer 14 is 2.164/4.126 ms. The first layer-1 callback of each timed
request is an outlier at about 32 ms.

The host nodes are identified by their order on every replay and adjacent
memcpy nodes: **192-byte hash-ID D2H → host callback → 6,336-byte packed-row
H2D → 4-byte status H2D**, once for each Engram layer. All six Engram memcpy
nodes together take about 1.03 ms of GPU transfer time across the full trace.
The MoE row-copy is a GPU kernel, so its host-memory traffic does not appear
in the CUDA memcpy summary. `_gather_host_rows_kernel`, another MoE path,
appears only during eager work outside the graph.

## Interpretation and limit

In this measured configuration, decode graph time is dominated by **MoE RAM
miss waiting and pinned-host expert-row copying**, not Engram table lookup.
The wait kernel's duration includes waiting until a host service signals completion;
it does not by itself identify how much was disk I/O, CPU work, queueing, or
doorbell latency. The Engram callback combines its own queue wait, cache work,
and `io_uring` reads; this trace does not split those components either. The
graph trace also has no byte count for the dynamic MoE row-copy kernel, so it
cannot establish that kernel's effective PCIe bandwidth.

Node-level tracing perturbs execution. The repository's earlier calibration
measured roughly 0.77 µs of tracing cost per graph node, about 6 ms per decode
step, so these are attribution figures, **not a throughput comparison**.
Summed event durations can overlap across streams; the percentages above
state their denominators explicitly. The next diagnostic, if optimizing
decode, is to time the MoE RAM-miss service from demand post through
`io_uring` completion and doorbell signal, then compare that timeline with
the 19.418 s GPU wait-kernel total.
