# Sampled served-path measurement, 2026-09-23

This diagnostic run answers the callsite question in
`ENGRAM_DECODE_TIME_HANDOFF.md`. It used the layer-1+14 graph build on divix01,
remote worktree commit `d337301dd1163ec46a5a76c2d8da41cdc34d19f4`, with
`SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1` and
`SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=0`. Nsight collected process-tree CPU
sampling, Python sampling and CUDA backtraces (`memory:80000`). The two timed
sessions generated 7 and 103 tokens. The harness result gate rejected the arm
because it expects eight sessions; this is a trace, not a throughput verdict.

Artifacts on divix01:

- Report: `/mnt/nvme1/dsv41-nsys/layer14-attrib-20260923-20260923-001143.nsys-rep`
- SQLite export: `/mnt/nvme1/dsv41-nsys/layer14-attrib-20260923-20260923-001143.sqlite`
- Run: `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/layer14-attrib-20260923/run-20260923-001128`

The ordinary SQLite export omits NVTX extended payload blobs. An export with
`--include-blobs=true` was used to match Python backtraces to CUDA calls by
timestamp and thread ID.

## Measured attribution

| Phase / call | Count | Host CUDA API time | GPU transfer time | Source |
| --- | ---: | ---: | ---: | --- |
| 256-byte device-to-host copies | 408 distinct GPU copies | 12.635 s in 205 long outer calls | 0.182 ms total | 204 sampled backtraces: `exl3.py:497`, `experts = chunk.tolist()`; one: line 498 |
| 4-byte device-to-host copies | 17,424 distinct GPU copies | 0.194 s in the inner runtime records | about 6 ms total | Scalar readbacks; exact Python callsite not established |
| 60 KiB device-to-host copies during decode | 3 | 0.895 s | about 0.014 ms | `expert_residency.py:471`, `decide_residency_policies`: `.cpu().numpy()` |

All 408 of the 256-byte copies occur in the two prefill windows, before the
respective decode graph launches. The 205 long calls split 108/97 across the
two prefills and account for 6.646/5.989 s of host API time. Their native
callchains all contain PyTorch's `THPVariable_tolist`. The 203 short copies
immediately follow long calls and are consistent with the adjacent
`row_of_source.tolist()` at `exl3.py:498`, but their exact Python callsite was
not sampled because of the 80 µs backtrace threshold.

Nsight shows 408 inner `cudaMemcpyAsync_v3020` records (12.642 s) and 205
outer `cudaMemcpyAsync` records (12.635 s) for the **same** 408 copies. Adding
those records would double-count the long calls. The transfer itself takes
microseconds; the long API time includes waiting for preceding GPU work and
cannot be assumed to be fully recoverable by deleting the readback.

There are 110 decode graph launches in two temporal clusters of 7 and 103,
matching the requests' generated token counts. Their API calls sum to
37.191 s across two decode intervals totaling
38.704 s. The three 60 KiB residency-policy readbacks occur between graph
launches in the second interval and occupy 0.895 s of the remaining
1.513 s. No 256-byte or 4-byte eager copy occurs within either decode
interval. Graph-internal copies and kernels are not exposed as ordinary
memcpy/kernel rows in this report, so the 37.191 s graph-launch time cannot
be split into GPU compute, transfer, Engram host-node work or launch wait.

## Consequence

The handoff's proposed `.item()`/event-wait change is not supported by this
measurement as the main decode optimization. The long visible readback is
the ExL3 MoE `chunk.tolist()` in prefill, relevant to time to first token.
For decode, the subsequent
[graph-node measurement](ENGRAM_DECODE_GRAPH_NODE_MEASUREMENT_20260923.md)
shows that MoE RAM-miss waiting and pinned-host row copying dominate the
traced graph. This sampled run alone could not see that work;
the separate residency-policy `.cpu()` readback is measurable but small
relative to the graph-launch span. CPU sampling also places substantial
prefill activity in PyTorch's CPU `direct_copy_kernel`, but its native stacks
are broken and this run does not establish a precise Python callsite for it.
