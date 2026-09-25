# DSV4.1 EXL3 DIRECT insert-on-miss: implementation and first serving measurement

Measured on divix01, 2026-09-23. The implementation is on
`master`. All three serving arms used commit
`165a48fa4f1784dfdb72843ab8db424397dab706` and Python tree
`9822a387d6b4610948efdaac79106a534fec102c`. The final test-only commits
through `e22f725d18` leave that Python tree unchanged.

## Mode and configuration

The native EXL3 RAM-miss service now receives a graph-published bitmap of the
GPU hot set before it chooses a pinned-tier eviction victim. A successful
single-phase leased row copy is acknowledged before the GPU updater publishes
the inserted resident. Eager host use refreshes its hot set from the GPU at
the existing synchronization boundary. DIRECT reserves no expert scratch
rows, making their GPU allocation available for residents. The default mode
remains OFF.

The existing `benchmarks/dsv41_baseline/run_arm.sh` recipe supplied the same
model, corpus, `uring_direct` file reader, 50 GiB pinned tier, 14 GiB GPU
budget, breakable batch-1 decode graph, and CPU placement to each arm:

| Arm | Overrides | Resident slots | Scratch bytes |
| --- | --- | ---: | ---: |
| A, current recipe | none | 888 | 3,195,740,160 |
| B, lease control | `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=1` | 888 | 3,195,740,160 |
| C, DIRECT | B plus `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1`, `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2`, `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1` | 1,128 | 0 |

The actual server environments were checked against `/proc/<pid>/environ`.
No other GPU compute tenant was present when each arm started; the reports
record some CPU contention from unrelated processes. The corpus checksum,
generation, graph capture, readiness clock, and compile gates passed.

## Eight-session result

| Comparison | Median paired decode ratio | Wins | One-sided sign-test p | Arms' median decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Current → leases | 1.0134 | 8/8 | 0.00390625 | 2.6328 → 2.6912 |
| Current → DIRECT | **1.1631** | **8/8** | **0.00390625** | **2.6328 → 2.9697** |
| Leases → DIRECT | **1.1533** | **8/8** | **0.00390625** | **2.6912 → 2.9697** |

Median TTFT was 37.331 s (A), 37.084 s (B), and 36.768 s (C). Each
arm returned eight results without response errors. The native service had
zero read errors, overruns, and late-after-fatal events in each run. The
complete result, clocks, environment, and validity report are in these
divix01 directories:

- A: `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/direct-insert-a/run-20260923-182553`
- B: `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/direct-insert-b/run-20260923-181115`
- C: `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/direct-insert-c/run-20260923-175724`

The native shutdown totals show 36,240 NVMe rows read in B and 35,767 in C.
They include warm-up, which was three rounds in both arms. B granted and
acknowledged 125,910 leases; C granted and acknowledged 80,289. These lease
counts suggest fewer pinned-to-GPU copy sources, but the harness did not
measure H2D bytes directly. A ran four warm-up rounds, so its 40,980 total
NVMe rows are not directly comparable to B or C.

## Correctness and limits

On divix01, the focused native, GPU-residency, server-requirements, and fused
MoE suite passed: 168 tests and 15 subtests. The captured DIRECT GPU test
passed all four combinations of generic/fused routes and forced timeout/
post-copy lease-generation violation. It checks source row bytes, numerical
MoE output, insertion, next-replay hit, eviction, refetch, eager pinned
admission, and fail-stop without resident publication. An independent code
review found no remaining concrete P1/P2 issue.

The serving harness marks strict validity false because its HTTP path does
not expose engine-side per-step latency and its report builder cannot observe
the actual server's imported `sglang` file or resolved environment defaults.
Its relaxed verdict excludes both acknowledged gaps and was true for all
three arms; the paired checker accepted the equal GPU tenancy, clocks, and
compile records. The live server's environment was verified separately from
`/proc/<pid>/environ`. Neither per-session H2D bytes nor a complete numerical
eager-prefill comparison was collected. The first result supports keeping
DIRECT available as an explicit opt-in while those instrumentation gaps are
closed; it does not change the default recipe.
