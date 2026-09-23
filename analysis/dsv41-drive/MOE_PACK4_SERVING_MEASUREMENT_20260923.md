# DSV4.1 four-worker row-packing measurement, 23 September 2026

## Setup

Two sequential, unprofiled eight-session arms ran on **divix01** at clean code commit `d337301dd1163ec46a5a76c2d8da41cdc34d19f4`. Both used the same synthetic session file recipe, 50 GiB MoE pinned tier, 5 GiB Engram cache, `uring_direct` expert reader, `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1`, and RAM-miss leases off. The only intended serving flag difference was `SGLANG_DSV41_RAM_MISS_PACK_WORKERS=0` versus `4`; both actual server environments matched their expected environments. The four-worker flag selects four byte-range chunks per row.

Raw run folders:

- Inline: `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/moe-pack0-20260923-074054/run-20260923-024054`
- Four workers: `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/moe-pack4-20260923-074054/run-20260923-025551`

Both result gates passed: eight results, zero errors. Both arms had zero timed JIT compile events, identical starting GPU tenancy (63 MiB, no production process), and compatible per-session SM clocks (all starts 2955 MHz; ends 2947–2955 MHz). The warmup gate declared stable decode rates of 3.156 and 3.889 tokens/s, respectively, at 2947 MHz.

## Paired decode throughput

| Session, in benchmark order | Generated tokens | Inline tok/s | Four workers tok/s | Ratio |
| --- | ---: | ---: | ---: | ---: |
| CDW | 7 | 2.3079 | 2.7163 | 1.177 |
| ETR | 103 | 2.9013 | 3.3497 | 1.155 |
| TSCO | 128 | 2.9630 | 3.3148 | 1.119 |
| BKR | 51 | 2.6428 | 3.0600 | 1.158 |
| K | 33 | 2.7298 | 3.2432 | 1.188 |
| DISCA | 34 | 2.7551 | 3.2649 | 1.185 |
| WRK | 75 | 2.9722 | 3.4535 | 1.162 |
| VLO | 54 | 2.5145 | 2.9443 | 1.171 |

The **median of eight paired ratios is 1.1664**, a 16.6% decode-rate gain. All eight prompts improved; the smallest gain was 11.9%. The median paired TTFT change was −0.22 s. The server process accumulated 519.10 s of CPU time during inline timed sessions and 573.64 s with four workers, about 10.5% more. This is one arm per condition, so it does not measure run-to-run variability under divix01's changing CPU contention.

## Validity limitation

Both official arm verdicts are **invalid** because their whole-arm expert-shard page-cache check compares before server startup to after the timed set. Inline residency was 0.605 GB before startup, 5.148 GB when ready, and 4.172 GB after timing; the cache *fell* by 0.909 GiB during its timed window. Four-worker residency was 4.172 GB before startup and 5.979 GB both when ready and after timing; it did *not grow* during its timed window. The actual server environment in both arms says `SGLANG_MOE_EXPERT_FILE_READER=uring_direct`. The gate's recorded growth therefore happened during startup, but its whole-arm rule still rejected both reports. Do not label these arms formally valid or use them as the final production-enablement gate. The page-cache starting states differed between sequential arms; `O_DIRECT` timed reads and stable warmup reduce that concern but do not replace a repeated paired experiment.

## Native-stage confirmation

A separate two-session diagnostic run, `moe-pack4-stage-20260923-081207`, enabled `SGLANG_DSV41_EXPERT_TRACE_PATH=/mnt/nvme1/dsv41-nsys/moe-pack4-stage-20260923-081207-expert.jsonl` with four workers. Its run folder is `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/moe-pack4-stage-20260923-081207/run-20260923-031207`. The eight-session result gate rejected this shortened diagnostic by design after **two records with zero errors**. The stage analyzer selected 4,400 complete layer records across the same 7 + 103 generated tokens; all selected records have valid schema, successful status, no dropped records, and complete row/extent coverage. Server counters show zero read errors and overruns. Native trace mode is `(pack_workers, pack_split)=(4,4)`.

Compared with the earlier inline two-session [stage capture](MOE_SERVICE_TRACE_MEASUREMENT_20260923.md), the worker run read a similar number of demands (2,604 versus 2,598) and bytes (59.819 versus 59.619 GB). Totals across read demands were:

| Native stage | Inline | Four workers | Change |
| --- | ---: | ---: | ---: |
| CPU observed → done | 18.863 s | 13.750 s | −27.1% |
| Submit → last reaped CQE | 11.376 s | 11.148 s | −2.0% |
| Last CQE → pack end, exposed tail | 7.453 s | 2.574 s | **−65.5%** |
| Exposed tail p50 per read demand | 2.737 ms | 0.980 ms | −64.2% |
| Exposed tail p95 per read demand | 2.938 ms | 1.109 ms | −62.2% |

This supports row splitting as the mechanism for the throughput gain. The read-window change is small enough to treat as run variation, and the stage captures were taken at different times. The stage trace adds a GPU readback and should not be used as the throughput verdict; the unprofiled eight-session arms above provide that comparison.
