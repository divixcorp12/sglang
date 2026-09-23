# DSV4.1 async residency sync profile, 23 September 2026

## Question and controls

Where do GPU-to-CPU host waits remain with
`SGLANG_MOE_ASYNC_RESIDENCY_SCORES=1`, compared with the same path with the flag
off? No comparable flag-off Nsight capture existed, so a fresh off baseline was
captured before on. These are **two-session diagnostics**, not throughput arms.

Both arms ran on divix01 from the same clean isolated checkout
`e8aa86b3436a705617afa98090d04066c4f4762d` and identical Python tree
`8a5130d70997d2e6da52fd6d4e868481580d191e`. The checkout includes the
corrected timed page-cache verdict. The expected 33 server environment variables
differed only in `SGLANG_MOE_ASYNC_RESIDENCY_SCORES` (0 versus 1). Both used the
four-worker RAM-miss row packer, Engram native host nodes and io_uring,
`uring_direct` expert reads, leases off, and the 50 GiB MoE pinned tier. The
optional `SGLANG_DSV41_EXPERT_TRACE_PATH` was unset. Nsight used the same CPU
sampling and CUDA/Python backtrace options in each arm, starting collection after
the warm-up readiness gate.

| Arm | Run directory on divix01 | Nsight report and SQLite export |
| --- | --- | --- |
| Off | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/syncscore-off-profile-20260923/run-20260923-131555` | `/mnt/nvme1/dsv41-nsys/syncscore-off-profile-20260923-20260923-131610.nsys-rep` and matching `.sqlite` |
| On | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/syncscore-on-profile-20260923/run-20260923-133134` | `/mnt/nvme1/dsv41-nsys/syncscore-on-profile-20260923-20260923-133150.nsys-rep` and matching `.sqlite` |

Each arm recorded the same 7-token and 103-token completions, with exact content,
reasoning, and finish-reason parity; zero request errors and zero timed JIT events.
The eight-session result gate rejected each intentionally shortened arm after two
records. Warm-up needed eight rounds off and five on. The first short session had
different start clocks (2902 versus 2947 MHz); this profile is used for API wait
attribution, not a throughput A/B.

## Long decode window

The 103-token request's decode cluster was selected from consecutive
`cudaGraphLaunch` calls in each SQLite export. Relative to the first recorded CUDA
runtime API, it spans approximately 87.540–118.858 s off and 100.930–131.499 s
on. The corresponding CUDA runtime and memcpy records show:

| Host API in long decode | Off | On |
| --- | ---: | ---: |
| Three 61,440-byte score D2H host calls | 277.172 + 316.166 + 222.878 = **816.216 ms** | 0.027 + 0.026 + 0.023 = **0.076 ms** |
| Three long `cudaStreamSynchronize_v3020` calls at policy boundaries | none; all stream-sync calls total 47.699 ms | 274.276 + 207.493 + 248.003 = **729.772 ms** (783.350 ms across all stream-sync calls) |
| `cudaGraphLaunch` host calls | 103 calls, 29.890 s total | 102 calls, 29.246 s total |

The score transfer itself takes approximately 0.004–0.005 ms on the GPU in either
arm. Off uses a pageable destination (`dstKind=0`), and on uses a pinned
destination (`dstKind=1`). On, each pinned copy finishes before the corresponding
long stream synchronization begins. For the first such boundary, both occur on
stream 13; no GPU memcpy runs during the subsequent wait. A long synchronization
thus remains immediately after each score snapshot. The profiler does not prove
that it is the same dependency or that 816 ms was removed from the decode critical
path. The graph-launch API time covers graph execution and host-node work;
it is not a separately additive GPU-to-CPU synchronization total.

Nsight did not attach a CUDA callchain to the stream-sync calls. CPU sampling shows
`c10::cuda::CUDAStream::synchronize()` in their native stack, but the Python caller
is unresolved. Source inspection finds stream synchronization in the hot-cache
promotion and pinned-tier host-use paths; the exact caller at these three
boundaries remains a hypothesis. A useful next diagnostic is narrow NVTX or
timestamp instrumentation around those sites, then another short capture.

## Exclusions and interpretation

Both profiles also contain 204 pageable 256-byte D2H `.tolist()` readbacks with
about 12.5 seconds of host API time. They precede the long decode cluster and must
not be ranked as long-decode stalls. A final `cudaEventSynchronize` of about
0.64/0.63 seconds occurs after the last graph launch and is excluded from the
decode cut. The short session and prefill/setup work require separate attribution
before optimization decisions. `cudaMemcpyAsync` and `cudaMemcpyAsync_v3020` are
nested views of some calls in the export, so their durations must not be summed.

The earlier unprofiled eight-session off/on/off comparison still shows a 3.5–5.1%
median paired gain (excluding its seven-token prompt) and exact output parity.
This profile narrows the mechanism: asynchronous readback shortens the copy call,
while a following stream barrier still waits. Leave the flag opt-in while locating
that barrier and testing whether it can overlap with useful work.
