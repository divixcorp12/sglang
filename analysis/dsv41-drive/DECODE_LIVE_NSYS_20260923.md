# DSV4.1 live decode Nsight capture, 2026-09-23

## Setup and validity

On divix01, the production DSV4.1 server was briefly restarted under Nsight
Systems 2026.3.2 with the same model, 32,768-token context, prefix cache,
50 GiB pinned MoE tier, 14 GiB GPU hot tier, EXL3 `uring_direct` reader,
RAM-miss leases, GPU residency updates, DIRECT insert-on-miss stage 2, and
decode update interval 1. The checkout was `709d819356` on
`codex/nvfp4-expert-stream-main`. Node-level CUDA graph tracing was enabled
only after server startup and a 260-token warm-up request. The capture contains
a 7-token request followed by a 100-token request with the same 260-token
prompt and `ignore_eos=true`. Both returned HTTP 200; the long request finished
at its 100-token limit in 32.845 s. Its prefill reused 256 cached tokens.

Raw files on divix01:

- `/mnt/nvme1/dsv41-nsys/decode-live-20260923-2013.nsys-rep`
- `/mnt/nvme1/dsv41-nsys/decode-live-20260923-2013.sqlite`
- `/mnt/nvme1/dsv41-nsys/decode-live-20260923-expert.jsonl`
- `/mnt/nvme1/dsv41-nsys/decode-live-20260923-selected.jsonl`

The selected native suffix has 4,280 contiguous requests in exact 40-layer
cycles, matching 107 traced decode replays. Its final 4,000 records match the
100-token request. They report zero failed, dropped, or untraced requests. The
production server was restarted after capture; port 7867 initially returned
HTTP 200 for `/v1/models`, selected `UnifiedRadixCache`, and retained the four
DIRECT-mode activation flags. At 20:19:26 the restored scheduler received
SIGTERM (exit -15), and the server exited. The user later confirmed they had
shut the server down. Port 7867 was left down by request, and the automatic
restart was cancelled. The profiled and briefly restored launches each allocated 1,128
hot-cache slots. Nsight reduced the profiled KV token pool (78,080 versus
88,064 after restoration); the short request stayed far below either limit.

## 100-token decode result

| Stage | Total in node-level trace | Per replay |
| --- | ---: | ---: |
| EXL3 leased GPU wait node | 18.155 s | 181.6 ms |
| Pinned expert-row copy kernel | 9.826 s | 98.3 ms |
| EXL3 MoE kernel | 0.384 s | 3.84 ms |
| EXL3 int8 GEMV kernels | approximately 0.46 s | approximately 4.6 ms |

The wait and copy kernels account for 27.981 s, 85.2% of the 32.845 s
profiled HTTP request. This ratio describes the node-level trace, whose
per-node overhead perturbs wall time; it is not an unprofiled throughput
estimate. The other GPU kernels, host scheduling, prefill remainder, and gaps
occupy the rest of the response interval.

The native records for these 100 replays show 3,575 demand requests and 425
touches. Of the demands, 2,314 read NVMe and 1,261 required no read. Successful
reads totaled **54.810 GB**. For read demands, submit-to-last-CQE windows
totaled 11.190 s and the exposed packing tails after the last CQE totaled
6.866 s. These intervals overlap work within a request; their sum is close to
the 18.091 s observed-to-done service total. Native `pack_workers=0` and
`pack_split=0` for all 4,000 records.

The prior `moe_service_node_analysis.py` join rejects this capture because its
one-correlation-group-per-replay assumption does not hold: the current
breakable graph has three correlation groups per replay. The totals above
come directly from the SQLite kernel table, with the first 280 layer nodes
excluded for the 7-token request. The native suffix was independently checked
for contiguous sequence and 40-layer cycles. No per-layer host/GPU clock
claim depends on the rejected join.

## Next optimization

The current production recipe leaves `SGLANG_DSV41_RAM_MISS_PACK_WORKERS` at
its inline default of 0. A previous leases-off, stage-OFF comparison found a
16.6% median decode improvement with 4 workers and reduced exposed pack tail
from 7.453 to 2.574 s on a matched 110-replay diagnostic. This capture shows
6.866 s of exposed pack tail in the current DIRECT mode. The next controlled
test is a paired unprofiled 0-versus-4 worker run **with the current DIRECT
flags and 32,768 context held fixed**. If it wins without read errors or
response drift, enable four workers in production, then remeasure the pinned
row-copy kernel, which already costs 9.826 s per 100 replays here.
