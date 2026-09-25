# DSV41 decode: final arms of 2026-09-25

The final full-arm benchmark of the day's decode work, and a node-mode Nsight trace of the best arm.

- **A**: the `arm_env` recipe as is. Layer fusion and the Engram device wait are on; the copy engine is off.
- **B**: A plus `SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1`, passed as a `run_arm.sh` `KEY=VAL` override so the
  harness checked it against the live server's `/proc/<pid>/environ`.

**Result.** B is 5-6% faster per session: 110.8 against 117.8 ms/token on the long session, 112.4 against 119.3
pooled. Both sessions produce byte-identical outputs in A and B. The copy engine armed, took every RAM-hit lane,
and recorded 0 fallbacks.

Code is branch `cc/final-arms` from `cc/dsv41-pinned-numa` at `58c2785d4d`. Evidence is on divix01 under
`/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/final-{A,B,B-nsys-node}/` and
`/mnt/nvme1/final-arms/`.

## 1. Harness change: node-mode Nsight in `run_arm.sh` (`38b8e93a82`)

- **`NSYS_CUDA_GRAPH_TRACE=graph|node`**, default `graph`. `nsys_capture.graph_trace_mode` validates it against the
  arm's full resolved environment, not only the overrides. It refuses `graph` whenever
  `SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE` is true (graph-mode tracing deadlocks the copy wait) and refuses any
  unknown mode.
- **`NSYS_TMPDIR=/mnt/nvme1/nsys-tmp`** is exported for the launch, `start` and `stop`. `NSYS_OUT_DIR` must be under
  `/mnt/nvme1/`.
- **Report before shutdown.** `run_arm.sh` already ran `nsys stop` before `stop_server`. It now also waits for the
  `.nsys-rep` to exist and stop growing, up to 30 min, before the server is stopped. It warns on a report under
  10 MB; ~361 KB is the known out-of-scratch signature. The live run shows the ordering holds: the 368 MB report was
  generated, then the result gate and verdict ran, then the server stopped.
- **Tests.** 17 new tests in `test_dsv41_baseline.py`.

  ```
  cd benchmarks/dsv41_baseline && OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
    test_dsv41_baseline.py -q -p no:randomly -p no:cacheprovider --basetemp=/mnt/nvme1/final-arms/pytest/base
  ```

  This gave **139 passed**, EXIT=0 read from `PIPESTATUS[0]`.

## 2. Protocol

- **Worktree.** Private worktree `/data/models/slang/nvfp4-work/wt-final-arms`, detached at `38b8e93a82`. It was
  clean for preflight; the only untracked file was `generations.json`. `sglang.__file__` resolved to
  `.../wt-final-arms/python/sglang/__init__.py`.
- **Generation.** Python tree `5e063f2206` was registered as `final-arms-38b8e93a82`.
- **Driver.** `/mnt/nvme1/final-arms/drive_ab.sh` took `rowimg-disk.lock`, because the recipe reads the row images.
  It waited for `cc-gpu.lock` to be free, then ran each arm on port 30021:

  ```
  EXPECT_SHA=38b8e93a821dd60a863bd90bec3d9f6480ee4ba1 bash benchmarks/dsv41_baseline/run_arm.sh final-A 30021
  EXPECT_SHA=38b8e93a821dd60a863bd90bec3d9f6480ee4ba1 bash benchmarks/dsv41_baseline/run_arm.sh final-B 30021 \
    SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1
  ```

- **Order.** A then B, once each, with no interleaving (06:11-06:21).
- **Gates.** The harness's gates ran on both arms: preflight, generation, corpus sha256, health, env (42 and 43
  vars matched), and readiness.
  - Readiness took A 3 warm-up rounds, with the clock at 2955 MHz and decode settling at 9.64 tok/s.
  - It took B 4 rounds, with the clock at 2940 MHz and decode at 10.52 tok/s.
  - No compile events occurred in either arm.
- **Timed set.** The default: corpus indices 0 and 1.
  - Session 0 (`...CDW/2015/page_35.pdf-2`) stops after **7 tokens** in both arms, so its per-token figures rest on
    6 decode steps.
  - Session 1 (`...ETR/2004/page_261.pdf-1`) runs 103 tokens and carries the comparison.

## 3. Metrics

Commands:

```
python benchmarks/dsv41_baseline/paired.py $A $B --a-name A-ce-off --b-name B-ce-on
python analysis/dsv41-drive/final-arms/arm_metrics.py A=$A B=$B --json /mnt/nvme1/final-arms/out/arm_metrics_AB.json
```

`$A` is `servers/final-A/run-20260925-061142` and `$B` is `servers/final-B/run-20260925-061636`.

The step p50 and p90 below are the client-side inter-token gaps (`chunk_times`). This is the harness's only per-step
source. It is not engine step latency (`client_latency.py`).

**Stalls** are gaps of at least 0.5 s (about 4x a step) and at least 2 s (the RAM-miss deadline).

| | A (copy engine off) | B (copy engine on) | B vs A |
|---|---:|---:|---:|
| session 1 (103 tok): decode tok/s | 8.491 | **9.024** | x1.063 |
| session 1: ms/token | 117.8 | **110.8** | -7.0 |
| session 0 (7 tok): decode tok/s | 6.846 | 7.175 | x1.048 |
| session 0: ms/token | 146.1 | 139.4 | -6.7 |
| pooled decode tok/s (tokens / decode seconds) | 8.379 | 8.897 | |
| pooled ms/token | 119.3 | **112.4** | -6.9 |
| TTFT, s (session 0 / 1) | 21.40 / 17.71 | 20.89 / 17.68 | |
| step p50 / p90 ms, session 1 | 116.3 / 145.3 | 109.5 / 137.3 | |
| step p50 / p90 ms, pooled | 117.6 / 153.9 | 110.7 / 148.0 | |
| step max ms | 246.8 | 235.2 | |
| stalls >= 0.5 s / >= 2 s | 0 / 0 | 0 / 0 | |
| outputs vs the other arm | | **2 of 2 byte-identical** | |

### Paired comparison

`paired.py` reports B winning 2 of 2 sessions, with ratios 1.048 and 1.063 (median 1.055). The one-sided sign test
gives p = 0.25, the floor for two sessions. The README's noise floor for this protocol is still unmeasured, so the
significance is directional.

The size of the gain agrees with the copy-engine smoke, which used different requests: 125.5 -> 119.0 ms/token, or
-6.5.

### Verdict

Command: `verdict.judge` with `manifest=generations.load()`, each arm as the other's cross-arm reference, and
`traced=False`. Both arms returned `valid_except_acknowledged_gaps: True`, with no unacknowledged problems.

- **Generation**: `final-arms-38b8e93a82` (tree `5e063f2206`) in both arms.
- **Compile**: 0 events per session in both.
- **Clock** (per-session `clocks.sm` start): A 2970, 2970 MHz; B 2970, 2962 MHz. These are within the 3% gate, and
  `paired.py` accepted them.
- **Contention**:
  - A flags `conda(101%)` on core 55 and the server's own `sglang::scheduler(105%)` on core 37, both at
    `server_ready`. It also flags `sglang::scheduler(98%)` on core 46 during session 0.
  - B flags `tmux: server(90%)` on core 37 at `server_ready`, and nothing during a session.
  - A `ps` at 06:16 showed questdb's `java` at ~90% on core 55 and byobu/tmux on 35 and 59. These co-tenants sit
    inside the server's 32-63 core range. They are recorded, not gated, and bear on A more than on B.
- **Session and cross-arm outliers**: none flagged.
- **Acknowledged** (6 per arm, as in every HTTP arm): no engine step latency (2), server-side provenance fields
  unavailable, and the three `resolved to None` server checks.

### Copy-engine arm proof (B's `server.log`)

- 06:18:16: `exl3 RAM miss thread started: ... leases on, row images on, copy engine on`.
- 06:19:25: `exl3 RAM miss copy engine armed after 16 decode forwards since capture`.
- The counters at shutdown cover the whole server lifetime, warm-up included:

  | counter | value |
  |---|---:|
  | `copy_jobs` | 21,814 |
  | `copy_lanes` | 47,874 |
  | `leases_copied` | 47,874 |
  | `copy_bytes` | 637.5 GB (13.3 MB per lane, one slot) |
  | mean grant-to-completion | 2.24 ms per job |
  | max grant-to-completion | 6.27 ms |
  | `copy_fallbacks` | **0** |
  | `copy_errors` | 0 |
  | `copy_generation_mismatches` | 0 |
  | `leases_voided` | 0 |

  Of 53,347 leases granted, 47,874 were released by the copy thread and 5,473 by acknowledgements.
- A's log: `copy engine off`, every `copy_*` counter 0, and 43,247 of 43,247 leases acknowledged.

## 4. Nsight trace of B (node mode; per-kernel attribution only)

### Capture

```
NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node EXPECT_SHA=a87fe49206... bash benchmarks/dsv41_baseline/run_arm.sh \
  final-B-nsys-node 30021 SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1
```

- The capture covers the timed set only; warm-up ran untraced. The report is
  `/mnt/nvme1/dsv41-nsys/final-B-nsys-node-20260925-062249.nsys-rep` (368 MB).
- The traced run passed every gate and verdict.
- Its outputs are byte-identical to B's.
- It armed after 16 forwards, with `copy_fallbacks` 0 (47,886 lanes).
- **Do not read ms/token or step-tail idle from this run** (CLAUDE.md).

### Analysis

All analysis ran on divix01 under `taskset -c 0-63` and `systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0`.

- **Nsight skill.** The laptop's 2026.5.1 skill pack ran against divix01's nsys 2026.3.2. `NSYS_TMPDIR` was
  `/mnt/nvme1/nsys-tmp`, and duckdb, pyarrow and pandas were installed with `pip --target` into a private directory.
  - `report-doctor` passed: 56 tables, 598,730 kernels, 57.5 s timeline, and 100% runtime-kernel correlation.
  - `report-fact --type memcpy_summary --metric total_bytes` reported H2D as 50,971 operations, 111.9 GB and 8.90 s.
- **Group breakdown.** It needs the section 1.1 layer cuts, which are procedural, so it used a sqlite export
  (`nsys export --type sqlite`, 1.16 GB beside the report). The scripts that cut it:
  - `analysis/dsv41-drive/copy-overlap/layer_chain.py`: the same script, groups and cuts as section 1.1;
  - `analysis/dsv41-drive/copy-engine/ce_trace.py --skip 0`;
  - `analysis/dsv41-drive/final-arms/ce_overlap.py`.
- **Outputs** are under `/mnt/nvme1/final-arms/out/`.
- **Cleanup.** The skill cache and the skill copy were deleted afterwards.

**Anomaly: `ce_trace.py` had never run on a real trace.** It failed with `no such column: graphId`, because nsys
2026.3's export carries `graphNodeId`, not `graphId`, on `CUPTI_ACTIVITY_KIND_MEMCPY`. It is fixed in `ac7490e2a1`.
The fix was verified: the out-of-graph copies are all `graphNodeId IS NULL`, and the graph's D2D nodes on stream 142
are excluded.

### Per-step breakdown

The trace holds 110 decode steps, one graph each. Every graph kernel is on one stream (142). Copy-engine copies are
on stream 141.

| Group | Kernels / step (before) | now | ms / step (before) | now, mean | now, p50 |
|---|---:|---:|---:|---:|---:|
| attn | 1,485 | 1,489 | 7.50 | 7.73 | 7.18 |
| shared | 440 | 440 | 1.67 | 1.62 | 1.62 |
| route | 1,254 | **254** | 1.39 | **0.45** | 0.45 |
| chain | 280 | 320 | 95.15 | 102.50 | 96.92 |
| book | 2,560 | **160** | 2.00 | **0.35** | 0.35 |
| moe | 80 | 80 | 4.02 | 4.11 | 4.11 |
| **total** | **6,112** | **2,756** | 111.7 of a 116.5 span | | 114.5 span p50 (120.0 mean) |

Chain detail, in ms per step (mean / p50):

| Kernel | Before | Now |
|---|---:|---:|
| post | 0.82 | 0.84 / 0.83 |
| W1 (`stream_hit_wait`) | 0.57 | 1.31 / 1.19 |
| **C1** (SM copy) | **72.26** | **0.026** / 0.026 |
| stage acks | 0.14 | 0.077 / 0.067 |
| **S** (`lease_stream`) | **21.30** | **46.38** / 39.88 |
| **CW** (copy wait, new) | n/a | **53.81** / 52.70 |
| F | 0.06 | 0.06 / 0.06 |

### Copy-engine memcpy activity

The copies are H2D, outside the graph, on stream 141.

| | per step, mean / p50 / p90 |
|---|---:|
| copies (about 6 segments per lane, ~76 lanes) | 458 / 447 / 570 |
| MB | 1,017 / 992 / 1,265 |
| copy-engine busy (union), ms | 80.9 / 77.8 / 102.8 |
| of it under any graph kernel, ms | 80.9 / 77.8 (**99.98%**) |
| under CW, ms | 53.5 |
| under S, ms | 26.7 |
| under W1, ms | 0.73 |
| under any compute (attn, shared, route, book, moe), ms | ~0 |

- **Totals**: 50,418 copies and 111.9 GB over the trace, at **12.57 GB/s** while busy. That is the copy-engine
  rate measured in section 1.2's bench (13.5 GB/s), less the S-side traffic sharing the link.
- **Driver calls**:

  | call | count | median | p99 | max |
  |---|---:|---:|---:|---:|
  | `cuMemcpyAsync` | 50,418 | 4.1 us | 12.6 us | |
  | `cuEventQuery` | 10.26 M (~178 k/s, the copy thread's completion polling) | 0.54 us | | 2.4 ms |

### Against section 1.1 (before today's work)

- **C1 72.26 -> 0.03 ms.** The copy engine takes essentially every RAM-hit lane. C1 still runs on the lanes the
  service does not hand over, but it is empty.
- **The copy moved but did not shrink.** The copy engine is busy 78-81 ms per step, the same order as the old C1,
  because the bytes are the same (~1 GB per token) and the link rate is similar (12.6 GB/s against C1's
  12.1-12.3).
- **The gain is the overlap with S.** 26.7 ms of copy per step now runs under S, which previously ran after C1.
  The exposed remainder shows up as CW's 53.8 ms wait.
- **S 21.3 -> 46.4 ms (39.9 p50).** S now shares the H2D link with the copy engine for the ~27 ms they overlap, so
  its own zero-copy reads slow down. The chain as a whole is unchanged in size (95 -> 97 ms p50) but carries both
  copies.
- **Kernels per step 6,112 -> 2,756.** Layer fusion replaced route's and book's elementwise chains: route
  1.39 -> 0.45 ms and book 2.00 -> 0.35 ms, for -2.6 ms per step. The Engram host nodes (1.6 ms before) are gone.
  attn now holds 2 `wait_kernel` per step (0.56 ms), next to `_engram_gate_kernel`, which is presumably the Engram
  device wait.
- **attn, shared and moe are unchanged** (7.2-7.7, 1.6, 4.1 ms).

## 5. Where the remaining time per token goes (B, ~111 ms/token untraced)

Node-trace kernel time per step at p50, which adds up to ~110 of the 114.5 ms span:

| Component | ms/step | Share |
|---|---:|---:|
| H2D link, copy engine: ~1.0 GB of RAM-hit expert rows at 12.6 GB/s (27 under S, 53 exposed as CW) | ~78-81 | ~70% |
| S beyond the copy engine's share: the lanes' own streamed pieces and NVMe waits (p50 39.9 - 23.6; mean 46.4 - 26.7) | ~16-20 | 14-17% |
| Compute: attention 7.2, MoE GEMM 4.1, shared expert 1.6, routing and bookkeeping 0.8 | ~13.7 | ~12% |
| post, W1, acks, F | ~2.2 | ~2% |

- **Decode is PCIe-bound on RAM-miss expert rows.** About 76 lanes, of 13.3 MB each, cross the H2D link every
  token. The copy engine keeps the link near its measured ceiling for ~70% of the step.
- **The copy engine overlaps nothing but the chain's own waits.** 99.98% of its busy time lies under CW, S or W1,
  and ~0 under attention or the MoE GEMM, because the next layer's attention needs this layer's MoE output. At most
  ~13.7 ms of compute is left to hide copies behind, even with perfect cross-layer prefetch.
- **The remaining levers** are therefore fewer bytes per token (a higher VRAM hit rate, smaller rows) or a faster
  link, not more overlap. The early-post gate reached the same conclusion from the link side: the link is ~70%
  busy, 82.5 of ~120 ms (`c6fd5ba27e`).

## 6. Anomalies

- **The timed set is thin.** Session 0 generates 7 tokens, and the sign test cannot go below p = 0.25 on two
  sessions. The B-over-A gain is consistent across both sessions and matches the smoke and the design estimate,
  but the protocol's noise floor is still unmeasured (README).
- **Co-tenant CPU load inside the server's cores 32-63.** questdb's `java` (~90%), byobu/tmux and a `conda` process
  are running there. It was recorded as contention, mostly in A.
- **`cudaGraphLaunch` blocks the host for ~one step.** The median is 109 ms, 110 calls. The host is never the
  bottleneck here: it waits inside the launch for the previous step. This is recorded, not investigated.
- **`ce_trace.py` graphId bug**, fixed in `ac7490e2a1` (section 4).
- **The copy-engine flag stays off by default.** Section 10.5 of `2026-09-25-dsv41-copy-compute-overlap.md` still
  applies, and `arm_env.py` is unchanged. These arms add a second clean run: 0 fallbacks, 0 errors, no stall of
  0.5 s or more, and identical outputs.
