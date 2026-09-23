# DSV4.1 MoE service trace, 23 September 2026

## Capture

The trace was collected on **divix01** from the clean `d337301dd1163ec46a5a76c2d8da41cdc34d19f4` benchmark worktree, with the same two timed synthetic sessions used for the earlier 110-replay node capture: 7 and 103 completion tokens. The benchmark arm was `moe-stage-20260923-062054`. Its raw native trace is `/mnt/nvme1/dsv41-nsys/moe-stage-20260923-062054-expert.jsonl`; the run folder is `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/moe-stage-20260923-062054/run-20260923-012054`. The capture used the 50 GiB MoE pinned tier, 5 GiB Engram cache, `uring_direct`, leases off, `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1`, and `SGLANG_DSV41_EXPERT_TRACE_PATH` set. Native records show `pack_workers=0`. The benchmark's eight-session result gate fails intentionally after `DSV41_MAX_SESSIONS=2`; both collected results had no errors.

The analyzer selected 4,400 records: 110 complete 40-layer cycles with contiguous request sequences 31481–35880. Selection uses native `CLOCK_MONOTONIC` stamps, not delayed JSONL drain times or the `forward` label. The last service record finishes 461 ms and 414 ms after the two respective result markers. The analyzer includes a bounded 1 s service tail and rejects overlapping session selections. All records have schema 5 and successful status; there are no selected dropped records or untraced rows or extents. The server reported zero read errors and zero overruns. These checks support attribution of **this diagnostic run**.

## Native service stages

Of the 4,400 layer requests, 2,598 demand requests read NVMe rows and 1,802 were touches. The demands requested 4,476 rows and completed 59.619 GB of reads, split evenly between devices 259:3 and 259:5 (29.808 GB and 29.811 GB). There were zero retry and cancelled bytes.

| Interval across 2,598 read demands | Total | Mean per demand | p95 |
| --- | ---: | ---: | ---: |
| CPU observed → done | 18.863 s | 7.260 ms | 13.704 ms |
| Observed → reservation | 0.018 s | 0.007 ms | 0.010 ms |
| Reservation → first submit stamp | 0.006 s | 0.002 ms | 0.003 ms |
| Submit → first reaped CQE | 5.573 s | 2.145 ms | 2.316 ms |
| First → last reaped CQE | 5.804 s | 2.234 ms | 6.473 ms |
| Submit → last reaped CQE | 11.376 s | 4.379 ms | 8.718 ms |
| Last CQE → pack end, exposed tail | 7.453 s | 2.869 ms | 2.937 ms |
| Pack end → map publication | 0.007 s | 0.003 ms | 0.004 ms |
| Map publication → CPU completion | 0.001 s | 0.001 ms | 0.001 ms |

The read window overlaps row packing. The exposed tail is the time after the last CQE until the last row is packed; it does not represent all packing work. CQE time is when the service reaped completion, not the precise drive completion time. The longest demand took 19.838 ms at layer 3, sequence 35844: 14.398 ms from submit to last CQE and 5.421 ms of exposed packing tail. The native stage record begins when the CPU first observes the demand, so this table does not yet measure GPU publication to CPU observation. The previous **separate** node capture measured 19.418 s in GPU waits and 18.213 s in pinned-row copy; the current native totals must be paired with a same-run node capture before interpreting the residual.

The largest aggregate CPU service layers were layer 0 (0.946 s over 90 read demands), 19 (0.783 s over 92), 23 (0.768 s over 86), and 39 (0.727 s over 84). Per-demand means differ because each layer asks for a different number of rows. The demand row counts were 1: 1,449; 2: 679; 3: 291; 4: 119; 5: 40; 6: 20.

## Matched GPU node join

The second arm, `moe-stage-node-20260923-063547`, captured native stages **and** node-level Nsight for the same 7 + 103 token workload. Raw artifacts on divix01 are `/mnt/nvme1/dsv41-nsys/moe-stage-node-20260923-063547-expert.jsonl` and `/mnt/nvme1/dsv41-nsys/moe-stage-node-20260923-063547-20260923-013602.nsys-rep`; its SQLite export has the same stem with `.sqlite`. The selected and joined per-layer rows are `moe-stage-node-20260923-063547-selected.jsonl` and `moe-stage-node-20260923-063547-joined.jsonl` in the same directory. The run folder is `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/moe-stage-node-20260923-063547/run-20260923-013547`.

The join found exactly 110 correlation groups, each with 40 post, wait, and pinned-row copy nodes, and paired them by layer with 4,400 contiguous native requests. All 2,598 demands pass the broad host/GPU clock plausibility checks: CPU observation falls after the GPU post begins and before its wait ends, and CPU completion precedes the wait end. The result gate again reports two records, zero errors; native records are complete with no selected drops, failures, or untraced extents.

| Same-run measurement | Total | Mean per relevant request |
| --- | ---: | ---: |
| GPU post kernels, all 4,400 layers | 37.819 ms | 8.6 µs |
| GPU wait kernels, all 4,400 layers | 19.550 s | 4.443 ms |
| GPU wait kernels, 2,598 demands | 19.541 s | 7.522 ms |
| Native CPU service, same 2,598 demands | 19.482 s | 7.499 ms |
| Demand GPU wait minus CPU service | **59.596 ms** | **22.9 µs** |
| Pinned-row GPU copy, all 4,400 layers | 18.262 s | 4.150 ms |

The 59.596 ms difference is an **unassigned signed boundary residual** around CPU service: for each demand, `(CPU observed − GPU wait start) + (GPU wait end − CPU done)`. It is about **0.3%** of demand wait time. It does not separately measure publication-to-observation or completion visibility, and the GPU post-kernel duration is outside the wait. This paired duration difference does not depend on a precise GPU-to-CPU clock offset. In the same run, native submit-to-last-CQE windows total **11.893 s** and exposed packing tails **7.560 s**; map-publication-to-CPU-completion totals **1.109 ms**. Reads and packing overlap within a request. The dominant measured CPU intervals support testing read/packing time or miss count first.

Nsight's `ANALYSIS_DETAILS.startTime` plus kernel-relative timestamps appears to share the host's monotonic time base. Exploratory means are 17.4 µs from post end to CPU observation and 5.7 µs from CPU completion to wait end. The containment check is only a **millisecond-scale plausibility check**, not a calibration precise enough to assign those small gaps separately. The optional NVTX/device timestamp ring in [the trace plan](MOE_SERVICE_TRACE_PLAN.md) would be needed to make that split quantitative. The current 59.6 ms combined residual makes it a lower priority than the read and packing work.

For context, the earlier node-only capture on the same commit and two-session workload measured 19.418 s of wait and 18.213 s of pinned-row copy. The stage-enabled node capture measured 19.550 s and 18.262 s, respectively. This comparison suggests modest trace perturbation at the node level, but it is not a randomized throughput A/B; do not use it to predict unprofiled throughput.
