# Project notes

To use nsys refer to '/opt/nvidia/nsight-systems/2026.5.1/skills/nsight-systems/SKILL.md'

## Nsight Systems traces

- Use `--cuda-graph-trace=graph` for decode traces, not `node`. Node mode makes `cudaGraphLaunch` cost ~0.77 µs per traced graph node (8,153 nodes → ~6 ms of host time per decode step), which shows up as a fake ~6 ms GPU-idle gap at the end of every step and inflates traced ms/token by ~7 ms (traced 74.1 vs untraced 66.8 ms/token). Use `node` only when a question needs per-kernel attribution inside the graph, and do not read step-tail idle or ms/token from such a trace. Evidence: `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/step-tail/`.
- The converse: **a graph-mode trace's kernel table leaves out the graph body.** Kernel summaries from it cover only eager work (prefill, sampling), so never rank kernels from one. Check first: if no kernel row has a nonzero `graphId` and summed kernel time is far below wall time, you are looking at prefill. Evidence: `MOE_EXPERT_TRANSFER.md`, "Nsight decode trace 2026-09-18".
- **A graph-mode report's `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` table can describe the warm-up, not the capture.** In `engram-on-trace-20260922-221602`, all 1,298 rows carry negative timestamps and correlation IDs strictly below the captured window's minimum: they are flushed pre-capture events on a different time base. Read naively they give a confident per-graph-execution average that describes nothing in the traced region. Check before using the table: `MIN(start)` against the other tables' `MIN(start)`, and whether its correlation IDs overlap `CUPTI_ACTIVITY_KIND_RUNTIME`'s. Summed graph time exceeding wall time is the tell.
- **`cudaMemcpyAsync` host time is not transfer time, and in this workload it is mostly not even a transfer.** Join each call to its GPU-side record by `correlationId` and read `bytes` before concluding anything about bandwidth: the same trace's biggest memcpy costs are 31-byte and 2 KB device-to-host readbacks (`.item()`) that block the host 841 us and 145 ms respectively while moving nothing. Evidence: `DSV41_REFERENCE.md` section 21.
- Bound every trace analysis's memory. On the laptop, `/tmp` is a RAM-backed tmpfs, and the Nsight skill writes its Parquet/DuckDB cache to `/tmp/nvidia/nsight_systems/nsys-skill-cache`. Loading a 250 MB node-mode decode report there out-of-memoried the laptop and crashed VS Code.
  - Analyse traces on divix01 where they live, under `taskset -c 0-63`. Copy a report to the laptop only when asked, and then only to a disk path, never `/tmp` or the scratchpad.
  - Cap memory for any local run: `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 <cmd>`. For DuckDB, also set `SET memory_limit='4GB'; SET threads=4`.
  - Narrow the data before loading it. Use `--filter-time`/`--filter-nvtx` or a short capture (`--duration`), and bound queries (`LIMIT`, pre-filtered CTEs).
  - Delete the skill cache and any copied report when done.

## GPU microbenchmarks on the RTX 5090

- **Size the working set past L2 (96 MiB) or you are measuring L2, not HBM.** A
  benchmark that reuses one tensor across repetitions stays resident and reports
  bandwidth the real workload will never see. This produced two recorded
  constants that understate production cost by 1.5x and 2.3x
  (`expert_residency_gpu.py`, the `0.007`/`0.054` ms/row pair); the fix was to
  spread the same work over 48 separate layer tensors for a 4.9 GiB footprint.
- **Sanity-check the implied bandwidth against the card's spec.** An impossible
  number is the cheapest detector available — a 3.1 TB/s result is what exposed
  the cache-resident benchmark above. HBM and PCIe ceilings both work for this.
- PCIe transfers cannot be flattered by L2, so H2D figures are unaffected; this
  applies to device-to-device work.
