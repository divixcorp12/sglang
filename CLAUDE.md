# Project notes

## Nsight Systems traces

- Use `--cuda-graph-trace=graph` for decode traces, not `node`. Node mode makes `cudaGraphLaunch` cost ~0.77 µs per traced graph node (8,153 nodes → ~6 ms of host time per decode step), which shows up as a fake ~6 ms GPU-idle gap at the end of every step and inflates traced ms/token by ~7 ms (traced 74.1 vs untraced 66.8 ms/token). Use `node` only when a question needs per-kernel attribution inside the graph, and do not read step-tail idle or ms/token from such a trace. Evidence: `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/step-tail/`.
