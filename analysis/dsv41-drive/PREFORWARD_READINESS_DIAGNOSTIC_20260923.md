# DSV4.1 pre-forward readiness diagnostic, 23 September 2026

## Measurement question and limits

At the actual pre-forward point, is the serving CUDA stream already ready while an asynchronous residency score event is complete? This was an observational query-only probe. It did not defer score decisions, mutate residency policy, or change the required `before_host_use` stream drain.

The existing end-of-forward path can apply the score decision and drain queued work before a later pre-forward sample. Therefore the observed readiness fractions do not predict a counterfactual wait reduction or throughput gain. The 2-session sample is also too small to characterize readiness over a serving workload.

## Instrumentation

The diagnostic is opt-in with `SGLANG_MOE_PREFORWARD_READINESS_DIAGNOSTIC=1` and an explicit `SGLANG_MOE_PREFORWARD_READINESS_PATH`. It registers a separate observer at the start of `ExpertDistributionRecorder.with_forward_pass`, immediately before `_forward_raw`. It queries `torch.cuda.current_stream(cache.device).query()` and, only while an async residency snapshot is pending, `backend.ready(_async_residency_event)`. Query exceptions are counted and cannot change serving behavior.

The collector retains up to one hour of 100 ms monotonic buckets. The background writer atomically replaces the JSON snapshot once per second; it does no per-forward file I/O. The harness recorded monotonic start/end boundaries for each timed request. Warmup/capture were not part of these intervals. Analysis below includes only buckets fully contained in each interval and splits prefill from decode/verify. The diagnostic JSON is about 500 KB; periodic serialization, the diagnostic queries, and GIL contention remain possible small observer effects on marginal readiness observations.

## Run and gates

- Ran the prepared HTTP arm on divix01 at port 7878 from isolated checkout `/data/models/slang/nvfp4-work/wt-sync-removal`, temporary commit `e204f0afdd08dcfc1caf53d74886c83ffa930272` (parent `fb8385a0fe0fd2eea36a6848bcee5f2ab0aaa657`). Python tree `4c45ea8285bc832d9b9079ba5048c5616d2a87fc` was registered as `preforward-readiness-diagnostic`.
- Preflight observed 63 MiB GPU memory used by GNOME/Xwayland only, no SGLang server, and no listener on ports 7867 or 7878. Production checkout/port were not touched. After the run, 7878 had no listener and the GPU was back to 63 MiB display-only.
- Recipe overrides: 4 row-pack workers, Engram host-node io_uring enabled, leases disabled, async residency scores enabled, 50 GiB MoE pinned tier, expert doorbell disabled. `SGLANG_DSV41_EXPERT_TRACE_PATH` was unset. All 35 expected server variables matched `/proc/<pid>/environ`.
- Python-generation and clean-checkout preflight passed. Four warmup rounds reached stable clock (2947 MHz) and decode rate (3.813 tok/s) with zero JIT compile events. Both timed sessions had zero compile events and 2955 MHz start/end clocks.
- The two timed results had zero request errors. Exact `content`, completion-token count, finish reason, truncation, and correctness fields matched the corresponding rows in the saved async-scores-on results. The 2-record result gate then failed as expected because the normal gate requires 8 records.

## Readiness results

Source files: [readiness JSON](artifacts/preforward-readiness-20260923/pre-forward-readiness-20260923.json), [timed windows](artifacts/preforward-readiness-20260923/run-20260923-155330/pre-forward-readiness-windows.jsonl), [results](artifacts/preforward-readiness-20260923/run-20260923-155330/results.jsonl), [saved async-scores-on comparison](artifacts/preforward-readiness-20260923/async-scores-on-reference-results.jsonl), and [complete run directory](artifacts/preforward-readiness-20260923/run-20260923-155330/).

| Timed request | Fully contained buckets | Pre-forward ready / samples | Decode ready / samples | Prefill ready / samples | Pending event ∩ stream ready / pending samples | Pending event ready / event-query samples |
|---|---:|---:|---:|---:|---:|---:|
| CDW page 35 | 7 | 3 / 8 (37.5%) | 2 / 7 (28.6%) | 1 / 1 | 2 / 2 | 2 / 2 |
| ETR page 261 | 100 | 6 / 104 (5.8%) | 5 / 103 (4.9%) | 1 / 1 | 1 / 7 | 4 / 7 |
| Combined | 107 | 9 / 112 (8.0%) | 7 / 110 (6.4%) | 2 / 2 | 3 / 9 | 6 / 9 |

Across the 9 pending samples, the serving stream was ready 3 times and the event was ready 6 times. Of the 3 joint-ready observations, 2 were decode samples (2 / 8 pending decode samples) and 1 was prefill (1 / 1). Each of the 3 tracked pending-decision keys first showed joint readiness at age 0 forwards; CDW had 2 such keys and ETR had 1. This poll-age value is an observation under the existing policy, not a count of forwards a deferred promotion could safely wait. Verify had 0 samples. Stream query errors: 0; pending-event query errors: 0; capture skips: 0; draft skips: 0. No populated partial boundary bucket was excluded. The last included full bucket in session 2 ended 343 ms before its recorded request-end boundary.

The harness freshness gate accepted a snapshot whose logged completion timestamp was 206,069,270 ns after session 2 ended. The preserved final JSON has `snapshot_monotonic_ns` 4,782,211,471 ns after that boundary. The first diagnostic schema did not record when bucket copying began, so the logged timestamp alone cannot prove the copied aggregate had started after the request boundary. Since the fully contained cutoff ends 343 ms before the boundary and the saved JSON has a later final flush, the available rows are useful as an observational sample, but final-window completeness is not strictly proven. The later isolated diagnostic commit `7bff2a478a7edc530c6c6445a9c54e49ab6b298c` records `snapshot_started_monotonic_ns` before copying and requires it to follow the request-end boundary for future runs.

The observations show a few pending-event/stream joint-ready pre-forward points (3 / 9 pending samples) but most sampled pre-forwards found the stream busy (103 / 112). The pending sample count is small, and current policy decisions plus the required drain affect subsequent samples. This is not enough evidence to conclude that deferring residency promotions would avoid meaningful waiting.

## Raw artifact inventory

The durable canonical artifacts are on divix01 at `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/preforward-readiness-20260923/run-20260923-155330/` and `/mnt/nvme1/dsv41-nsys/pre-forward-readiness-20260923.json`; the saved on-reference is `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/async-scores-on-20260923/run-20260923-112355/results.jsonl`. This workspace also has a local mirror under `analysis/dsv41-drive/artifacts/preforward-readiness-20260923/`, but the repository ignores `artifacts/`, so the mirror is not tracked and may not exist in other checkouts. The key local files are [readiness JSON](artifacts/preforward-readiness-20260923/pre-forward-readiness-20260923.json), [timed windows](artifacts/preforward-readiness-20260923/run-20260923-155330/pre-forward-readiness-windows.jsonl), [results](artifacts/preforward-readiness-20260923/run-20260923-155330/results.jsonl), and [saved async-scores-on comparison](artifacts/preforward-readiness-20260923/async-scores-on-reference-results.jsonl). The run directory includes the four warmup result files, synthetic session list, boundary samples, compile and clock checks, expected and actual server environments, and server/arm logs.

The isolated diagnostic checkout contains only temporary diagnostic work; the shared local branch was not committed. No performance optimization was implemented.
