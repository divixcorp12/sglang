# Project notes

## Nsight Systems traces

- Use `--cuda-graph-trace=graph` for decode traces, not `node`. Node mode makes `cudaGraphLaunch` cost ~0.77 µs per traced graph node (8,153 nodes → ~6 ms of host time per decode step), which shows up as a fake ~6 ms GPU-idle gap at the end of every step and inflates traced ms/token by ~7 ms (traced 74.1 vs untraced 66.8 ms/token). Use `node` only when a question needs per-kernel attribution inside the graph, and do not read step-tail idle or ms/token from such a trace. Evidence: `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/step-tail/`.
- Bound every trace analysis's memory. On the laptop, `/tmp` is a RAM-backed tmpfs, and the Nsight skill writes its Parquet/DuckDB cache to `/tmp/nvidia/nsight_systems/nsys-skill-cache`. Loading a 250 MB node-mode decode report there out-of-memoried the laptop and crashed VS Code.
  - Analyse traces on divix01 where they live, under `taskset -c 0-63`. Copy a report to the laptop only when asked, and then only to a disk path, never `/tmp` or the scratchpad.
  - Cap memory for any local run: `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 <cmd>`. For DuckDB, also set `SET memory_limit='4GB'; SET threads=4`.
  - Narrow the data before loading it. Use `--filter-time`/`--filter-nvtx` or a short capture (`--duration`), and bound queries (`LIMIT`, pre-filtered CTEs).
  - Delete the skill cache and any copied report when done.
