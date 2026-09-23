# DSV4.1 async residency score throughput, 23 September 2026

## Setup

Ran three sequential, unprofiled eight-session HTTP arms on divix01 in the order
`off → on → off`, each with a fresh server on port 7878. All used the same clean
diagnostic checkout `c39f1ab473490a5e0ed9dd08b487b045ae5cdfb2`, whose
`python/` tree `8a5130d70997d2e6da52fd6d4e868481580d191e` is identical to
local commit `d7e2297a61`. The expected server environments differed only in
`SGLANG_MOE_ASYNC_RESIDENCY_SCORES` (0, 1, 0); all 33 values were verified against
the live server process. The recipe used four RAM-miss row-packing workers, native
Engram graph host nodes with io_uring, `uring_direct` expert reads, a 50 GiB partial
MoE pinned tier, and RAM-miss leases disabled. The production server was not running,
and all three arms started with 63 MiB of GPU memory in use.

| Arm | Result directory on divix01 | Warm-up rounds | Own median decode tok/s |
| --- | --- | ---: | ---: |
| Off 1 | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/async-scores-off-20260923/run-20260923-111003` | 3 | 3.3641 |
| On | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/async-scores-on-20260923/run-20260923-112355` | 5 | 3.5387 |
| Off 2 | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/async-scores-off-repeat-20260923/run-20260923-114006` | 4 | 3.2629 |

The harness compares **matched session IDs**, not the arm medians above. All arms
completed eight records with zero request errors and zero compilation events during
timing. GPU clock samples were 2947–2955 MHz. All eight pairs matched exactly on
content, reasoning, completion token count, and finish reason.

| Session | Off 1 tok/s | On tok/s | Off 2 tok/s |
| --- | ---: | ---: | ---: |
| Single_CDW (7 output tokens) | 2.8126 | 3.8548 | 2.7227 |
| Single_ETR | 3.3876 | 3.4892 | 3.3782 |
| Single_TSCO | 3.4019 | 3.4349 | 3.3738 |
| Double_BKR | 3.1338 | 3.2780 | 3.1138 |
| Single_K | 3.3558 | 3.6471 | 3.2520 |
| Single_DISCA | 3.3724 | 3.5882 | 3.2739 |
| Single_WRK | 3.5577 | 3.6421 | 3.4884 |
| Single_VLO | 3.0325 | 3.1398 | 2.9866 |

The async arm won 8/8 matched sessions against each off arm. Its median paired
rate ratio was **1.0407** against Off 1 and **1.0520** against Off 2. Excluding the
seven-token `Single_CDW` session, whose decode rate is especially noisy, the median
paired ratios were **1.0354** and **1.0513**, respectively. These are observed
differences in one `off → on → off` sequence, not a confidence interval or a
production throughput guarantee.

## Validity limits

The existing verdict compares expert-shard page-cache residency before server start
with residency after the timed set. It rejected Off 1 (3.75 GiB growth) and On
(2.06 GiB growth), although the page-cache residency was unchanged from `server_ready`
through the timed set in both. Off 2 began with a warmed cache and passed the cache
check (`valid except the acknowledged step-latency gap: True`). The timed page-cache
levels differed across arms: approximately 6.38, 8.44, and 8.42 GiB. The reverse
Off 2 result makes simple cache warming an unlikely explanation for the gain, but
the first two arms do not meet the harness's formal whole-arm gate.

The HTTP harness has an acknowledged absence of engine-side per-step latency in all
three arms. Warm-up reached the stability gate after three, five, and four rounds;
this difference could affect state at the start of timing. For a formal go/no-go
decision on enabling the flag by default, repair the page-cache gate to judge the
timed window (while still recording startup growth), then repeat a matched run.

The original run files (`results.jsonl`, `clocks.jsonl`, `compile.jsonl`,
`expected-env.json`, `report.json`, `verdict.txt`, and `server.log`) remain in the
directories above. The verdict failure prevented the first two arms from writing
`run-manifest.json`; the raw results and reports were preserved.
