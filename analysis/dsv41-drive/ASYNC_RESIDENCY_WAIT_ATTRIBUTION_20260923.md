# DSV4.1 async residency wait attribution, 23 September 2026

## Diagnostic and controls

The matched off/on profile found three long `cudaStreamSynchronize_v3020` calls
after async residency score readbacks, but could not identify their Python caller.
This follow-up labels the candidate waits with opt-in NVTX ranges and repeats only
the **on** two-session diagnostic. It is a caller attribution run, not a throughput
comparison.

The run used divix01's isolated diagnostic checkout at
`fb8385a0fe0fd2eea36a6848bcee5f2ab0aaa657` (Python tree
`cc79afadcbffcdb21fff053bfdc2971a5fd2f1d3`), corresponding to the NVTX
source change on `codex/nvfp4-expert-stream-main` at `c88567866d`. The checkout
passed the clean-tree and registered-generation gates. The harness verified all
34 expected server variables against the live process. Relevant settings were
`SGLANG_MOE_ASYNC_RESIDENCY_SCORES=1`,
`SGLANG_DSV41_SYNC_WAIT_NVTX=1`, four RAM-miss pack workers, Engram host-node
io_uring on, RAM-miss leases off, expert doorbell off, and the same 50 GiB MoE
pinned tier as the earlier profiles. Nsight collected CUDA, NVTX, and OS runtime
events after the warm-up gate.

| Artifact on divix01 | Path |
| --- | --- |
| Run records | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/syncwait-nvtx-on-20260923/run-20260923-141419` |
| Nsight report | `/mnt/nvme1/dsv41-nsys/syncwait-nvtx-on-20260923-20260923-141434.nsys-rep` |
| SQLite export | `/mnt/nvme1/dsv41-nsys/syncwait-nvtx-on-20260923-20260923-141434.sqlite` |

The two timed requests again generated 7 and 103 tokens. Their content,
reasoning, finish reasons, and token counts exactly match the saved flag-on run.
Both had zero request errors and zero timed JIT events. Start/end SM clocks were
2947/2940 and 2940/2940 MHz. The run intentionally stopped after two sessions;
the harness's eight-session result gate therefore failed as expected, after
writing the trace. Port 7878 was closed and no compute process remained.

## Long-decode result

The SQLite export has 110 `cudaGraphLaunch` calls in two clusters: 7 for the
short request and 103 for the long request. The long decode window excludes its
first launch, which belongs to prefill/setup, and spans graph calls 8–109 at
approximately 87.524–117.981 seconds on the profiler clock. Matching each long
stream-sync API to an NVTX range on the same thread gives:

| Stream-sync start (s) | API wait (ms) | Containing NVTX ranges |
| ---: | ---: | --- |
| 97.391982 | 278.927 | `dsv41.ram_miss.before_host_use_stream_sync` inside `dsv41.hot_cache.host_use` |
| 107.503658 | 202.797 | same |
| 116.050254 | 251.654 | same |

These three waits total **733.38 ms** before rounding the table entries. Across
the decode window there were 44 hot-cache host-use ranges. Their RAM-miss
stream-sync portions totaled
734.890 ms, so the three calls account for nearly all of that wait. The
promotion-copy stream-sync ranges totaled 47.165 ms, with a 2.008 ms maximum;
the RAM-miss thread-pause ranges totaled 4.860 ms. The enclosing host-use range
total is inclusive and must not be added to its nested waits. No long decode
sync was attributed to the expert-index fallback.

Three 61,440-byte score D2H transfers remained pinned and asynchronous: their
host `cudaMemcpyAsync_v3020` calls took 0.048, 0.026, and 0.024 ms, while each
GPU copy took about 0.004 ms. The first copy began on the GPU roughly 273 ms
after its host API returned. This supports the earlier observation that the
readback call is short while queued stream work continues; it does not make
the later host-use wait an independently additive cost.

The earlier flag-on profile measured 729.772 ms in its three policy-boundary
stream waits. The new 733.38 ms total is consistent with that pattern, but the
traces have different instrumentation and code generations, so their durations
are not a timing A/B.

## Correctness constraint and next experiment

`Exl3RamMissService.before_host_use` drains the serving stream **before**
pausing the native RAM-miss reader. The reader must remain able to serve any
already-queued graph demand until those graphs complete; pausing it first could
strand an in-flight graph. The later promotion-copy wait keeps pinned source
rows owned until all six tensor copies complete. Replacing either wait with a
blocking event at the same point does not remove the dependency, and dropping
it would violate row ownership.

The next useful experiment is a nonblocking serving-stream readiness query at
a **pre-forward** boundary, before the next graph is queued. If the stream is
still busy, defer the entire residency promotion and retain the pending score
snapshot and current slot mappings. If ready, apply the promotion with the
existing safety barrier as a guard. First measure how often that pre-forward
query finds the stream ready and how old deferred decisions become; a policy
that never finds a safe boundary would need a bounded fallback. Only then test
the scheduling change with graph output parity, delayed RAM-miss injection,
and pinned-row ownership checks, followed by an unprofiled throughput A/B.
