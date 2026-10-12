# io_uring option campaign: end-to-end screen, 2026-09-28

**Verdict: keep the defaults.** No option beats S0 by the 1.5 ms/token needed to go to confirmation, so step 2 was
skipped. The campaign used 9 arms: S0–S6, then a fresh S0b next to the S7 the user added.
- Queue depth 128, the slab arena and SQPOLL (block or spin) are all within ±0.7 ms/token of S0, with byte-identical
  output.
- IOPOLL is clearly worse: +10.2 ms/token pooled, TTFT +1.4 s. Its reads are punted to io-wq worker threads.
- SQPOLL costs a full core (98% system time on the SQ thread), and spin waiting adds about 11 points to the service
  thread, for no latency gain.
- S7 (`sqpoll_iopoll` + spin) matches a fresh S0b at 104.4 vs 104.5 ms/token, and costs a full SQ core plus a
  spinning service thread.
- Fixed vectored reads (S3) are **unsupported** at this tier size: each per-layer arena is 2.69 GB, above the
  kernel's 1 GiB per registered buffer, and registration fails with EFAULT.

## Code, harness, preflight

- **Commit.** Every arm ran the same commit: `arm/sleepfree-cand` at `f3ba371ad4`, which is `32d9a21316` (the
  `565a76c9fc` registration) plus the sleep-free results commit. `HEAD:python` = `8244785c9bb5` (registered as
  `sleepfree-cand-565a76c9fc`). Worktree: `/data/models/slang/nvfp4-work/wt-uring-camp` on divix01.
- **Recipe.** The base `arm_env.py` with row images (multi-iovec readv into the slabs), leases, two-phase, piece
  streaming and the copy engine. There is no stage trace, so no per-read latency percentiles exist.
- **Launch.** Each arm: `EXPECT_SHA=<HEAD> OMP_NUM_THREADS=8 flock rowimg-disk.lock taskset -c 0-63 bash
  benchmarks/dsv41_baseline/run_arm.sh uring-<arm> 30031 <all nine SGLANG_EXPERT_STREAM_URING_* set explicitly>`.
  - Wrapper: `divix01:/mnt/nvme1/sf-ab/arm2.sh`. It checks the driver, production, the port, foreign processes (by
    comm/exe name and exact argv element), the GPU and the GPU lock, then runs the per-thread sampler
    `tsample.py`.
  - Chain: `screen_chain.sh`. Report scripts: `arm_report.py` and
    `analysis/dsv41-drive/final-arms/arm_metrics.py`.
- **Explicit defaults (S0).** `MODE=default QUEUE_DEPTH=0 FIXED_FILES=0 READ_MODE=normal WAIT_MODE=block
  SQ_THREAD_IDLE_MS=1000 SQ_THREAD_CPU=-1 SLAB_ARENA=0 DIAGNOSTICS=1`. Each screen arm changes only the keys named
  below.
- **Host.** Kernel `6.12.0-211.60.1.el10_2.x86_64`, liburing 2.12, `nvme.poll_queues=1`, `io_poll=1` on all four
  namespaces, `ulimit -l` unlimited. All four NVMe drives are on NUMA node 1. The driver was 615.71.09 before every
  arm, production (7867) was down, and CPU 20 was online and ~90% idle, with nothing pinned to it.
- **Real-kernel matrix.** `test_expert_stream_uring_native.py` + `test_expert_stream_uring_integration.py`, run
  under the disk lock with `--basetemp=/mnt/nvme2/nvfp4-work/uring-camp-native-tests`: **45 passed, 0 skipped**.

## Screen (2 timed sessions per arm; SM clock 2970 MHz at the start of every session in every arm)

| Arm | Change | Session 0 ms/token (TTFT) | Session 1 ms/token (TTFT) | **Pooled ms/token** | Δ vs S0 | Output vs S0 |
|---|---|---|---|---|---|---|
| S0 | defaults | 127.5 (8.90 s) | 102.5 (8.27 s) | **104.2** | — | — |
| S1 | `QUEUE_DEPTH=128` | 128.4 (8.63 s) | 102.5 (8.07 s) | **104.2** | 0.0 | byte-identical |
| S2 | `SLAB_ARENA=1` | 128.5 (8.70 s) | 102.3 (8.09 s) | **104.0** | −0.2 | byte-identical |
| S3 | `SLAB_ARENA=1 FIXED_FILES=1 READ_MODE=readv_fixed` | **unsupported**: refused at startup | | | | |
| S4 | `MODE=iopoll` | 149.7 (10.47 s) | 111.9 (9.42 s) | **114.4** | **+10.2** | byte-identical |
| S5 | `MODE=sqpoll SQ_THREAD_CPU=20` | 129.4 (8.63 s) | 103.2 (8.07 s) | **104.9** | +0.7 | byte-identical |
| S6 | as S5 + `WAIT_MODE=spin` | 128.9 (8.59 s) | 102.7 (8.06 s) | **104.5** | +0.3 | byte-identical |

Session 0 decodes only 7 tokens (6 gaps), so session 1 (85 tokens) carries the pooled figure. Every completed arm
has run_arm rc=0, no stalls ≥0.5 s, no fatals or tracebacks, and "valid except the acknowledged step-latency gap".

**S1 depth choice.** S0's diagnostics show an effective default depth of 32 (`16 * parts`, so parts=2). That is
small, so per the plan S1 used 128. The ring grew to match (sq 128 / cq 256) with no latency effect.

**S3 refusal.** Startup failed, 87 s after `server_args`, with
`RuntimeError: expert stream registering fixed buffers (regions=40, bytes=107363784704, largest=2689753088; check
RLIMIT_MEMLOCK and kernel per-buffer limits, commonly 1 GiB): Bad address (-14)`. That is 40 per-layer arenas of
up to 2.69 GB each against the 1 GiB per-buffer limit, and memlock is unlimited. The refusal is explicit with no
fallback, and the tier was not shrunk. `READ_MODE=fixed` alone does not apply because row images use multi-iovec
reads, so stage C was not run.

## Diagnostics (effective settings, from `expert stream io_uring:` in each server.log)

| Arm | mode / read / wait | requested → effective flags | depth | SQ / CQ entries | fixed files / buffers | SQ thread |
|---|---|---|---|---|---|---|
| S0 | default / normal / block | 0x0 → 0x10000 | 32 | 32 / 64 | 0 / 0 | — |
| S1 | default / normal / block | 0x0 → 0x10000 | 128 | 128 / 256 | 0 / 0 | — |
| S2 | default / normal / block | 0x0 → 0x10000 | 32 | 32 / 64 | 0 / 0 | — |
| S4 | iopoll / normal / block | 0x1 → 0x10001 | 32 | 32 / 64 | 0 / 0 | — |
| S5 | sqpoll / normal / block | 0x6 → 0x10006 | 32 | 32 / 64 | 0 / 0 | idle 1000 ms, cpu 20 |
| S6 | sqpoll / normal / spin | 0x6 → 0x10006 | 32 | 32 / 64 | 0 / 0 | idle 1000 ms, cpu 20 |

`0x10000` is `IORING_SETUP_NO_SQARRAY`, which liburing adds itself. `0x6` is `SQPOLL|SQ_AFF`, and `0x1` is `IOPOLL`.
Features were `0x3ffff` everywhere. Every arm logged one line: one ring, created once, with no reset.

## CPU, service counters, startup

Per-thread utime/stime come from `/proc/<pid>/task/*/stat`, sampled every 2 s. The window runs from the harness's
"ready" line to the last timed result, 28.6–28.7 s (32.7 s for S4). The harness `cpu_s` is the server process tree
per timed session.

| Arm | harness cpu_s (s0, s1) | tree CPU in window | ram-miss service thread | SQ thread | io-wq workers | copy-engine thread | rows_read / served / read_errors | startup to "fired up" |
|---|---|---|---|---|---|---|---|---|
| S0 | 10.2, 31.6 | 42.8 s | 25.4% (6.96 u / 0.33 s) | — | 0 | 34.8% | 4164 / 2928 / 0 | 150.2 s |
| S1 | 10.1, 31.4 | 43.1 s | 25.6% | — | 0 | 34.9% | 4153 / 2940 / 0 | 139.9 s |
| S2 | 10.3, 31.2 | 42.9 s | 25.4% | — | 0 | 34.8% | 4184 / 2941 / 0 | 140.7 s |
| S4 | 23.9, 50.2 | 50.1 s | 32.1% (6.09 u / **4.40 s**) | — | **12 threads, 4.3 s** | 32.7% | 4155 / 2919 / 0 | 146.5 s |
| S5 | 19.4, 48.0 | 70.9 s | 24.7% | **98.3%** (28.1 s system) | 0 | 35.1% | 4182 / 2935 / 0 | 150.0 s |
| S6 | 23.7, 53.9 | 74.0 s | **36.3%** (10.37 u / 0.00 s) | **98.8%** | 0 | 34.9% | 4253 / 2974 / 0 | 140.4 s |

- The scheduler's main thread was 82–84% in every arm.
- An idle `iou-sqp` thread (0 CPU) exists in every arm, in the launcher process, not the scheduler. It belongs to
  another ring, not this reader.
- Startup is measured from server process start to "The server is fired up". It varies by ±5 s across arms with no
  pattern tied to the options, so ring setup is not a measurable part of it.
- **S4 (IOPOLL).** The service thread's system time rises from ~0.3 s to 4.4 s. Twelve `iou-wrk-*` workers appear,
  most of them children of the service thread, which means reads were punted to io-wq instead of completing by
  polling. There were no errors, but latency and TTFT both got worse.
- **S5/S6 (SQPOLL).** The SQ thread never reaches its 1000 ms idle timeout during decode. It burns a whole core in
  system time. With WAIT_MODE=spin the service thread also busy-waits (all utime).

Other service counters were all 0 in every arm: late_after_fatal, copy_errors, copy_fallbacks and
slots_quarantined.

## Validity notes

- S5 waited ~6 min on `rowimg-disk.lock` before starting, while another job held it. The lock serializes disk work,
  so that job did not overlap any timed window.
- My chain log's `rc=` column is wrong (the `$(date)` in the echo resets `$?`). The per-arm logs carry the true
  run_arm rc: 0 for every arm except S3, which is 1.
- Run dirs (`/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/`):
  - `uring-S0-default/run-20260928-030535`
  - `uring-S1-depth128/run-20260928-031109`
  - `uring-S2-arena/run-20260928-031523`
  - `uring-S3-arena-fixedfiles-readvfixed/run-20260928-031942`
  - `uring-S4-iopoll/run-20260928-032205`
  - `uring-S5-sqpoll-cpu20/run-20260928-033255`
  - `uring-S6-sqpoll-cpu20-spin/run-20260928-034018`
  - `uring-S0b-default/run-20260928-034816`
  - `uring-S7-sqpolliopoll-cpu20-spin/run-20260928-035259`
- Per-arm logs, thread samples and report JSON are in `divix01:/mnt/nvme1/sf-ab/uring-*`.

## S7: `MODE=sqpoll_iopoll WAIT_MODE=spin` (added by the user), run next to a fresh S0b

S7 used `SQ_THREAD_CPU=20` and `SQ_THREAD_IDLE_MS=1000`, with everything else at the explicit defaults. The arms ran
back to back, in the order S0b then S7, in a fresh worktree `/data/models/slang/nvfp4-work/wt-uring-s7` at the same
`f3ba371ad4` (tree `8244785c`). CPU 20 was ~99% idle beforehand.

| Arm | Session 0 ms/token (TTFT) | Session 1 ms/token (TTFT) | **Pooled** | Output | rows_read / served / errors | startup |
|---|---|---|---|---|---|---|
| S0b (defaults) | 129.1 (8.64 s) | 102.7 (8.08 s) | **104.5** | byte-identical to S0 | 4221 / 2939 / 0 | 148.7 s |
| S7 | 128.8 (8.63 s) | 102.7 (8.06 s) | **104.4** | byte-identical to S0 and S0b | 4210 / 2960 / 0 | 145.1 s |

The SM clock was 2970 MHz at both sessions' start in both arms. S7 is −0.1 ms/token vs S0b, which is noise. S0b
reproduces S0 (104.2) within 0.3 ms/token.

- **Diagnostics.** `mode=sqpoll_iopoll read_mode=normal wait=spin requested_flags=0x7 effective_flags=0x10007
  depth=32 sq/cq=32/64 sq_thread_idle_ms=1000 sq_thread_cpu=20`. `0x7` is `IOPOLL|SQPOLL|SQ_AFF`, so **both SQPOLL
  and IOPOLL are in effect**.
- **CPU over the 28.6 s timed window.**
  - SQ thread `iou-sqp`: **98.1%** (28.06 s, all system time).
  - Service thread `exl3-ram-miss`: **36.2%** (10.34 s, all user time: the spinning completion wait), against 25.5%
    in S0b.
  - Copy engine: 34.9%, the same as S0b. Scheduler main thread: 82.9%.
  - Process-tree CPU: 75.4 s, against 43.2 s in S0b. Harness cpu_s: 24.4 s and 54.8 s, against 10.3 s and 31.4 s.
- **io-wq punting.** Yes, but less than S4. Six `iou-wrk-714815` workers appeared, children of the SQ thread
  (714815). They used 4.45 s of CPU over the whole run, 1.45 s of it in the timed window. S0b had none.

## Why S4 (IOPOLL alone) lost 10 ms/token (likely cause)

The likely reason is that IOPOLL reads on these files were **punted to io-wq** rather than completed by polling.
- Over the run, S4 spawned 14 `iou-wrk-600353` workers, children of the service thread 600353, using 12.7 s of CPU.
  That is on top of ~16 other short-lived workers.
- In the timed window the service thread's system time rose from ~0.3 s to **4.4 s**, and 12 workers used 4.3 s.
- The defaults, S1, S2, S5 and S6 had no workers at all.

A punted read takes an extra thread handoff and wakeup on the RAM-miss critical path. That fits both the +10.2
ms/token and the +1.4 s TTFT. S7 also punts, but from the SQ thread and less, and its busy SQ thread plus spinning
waiter hide the handoff, so it shows no loss. The workers and system time are measured; the causal link to latency
is inferred, because no per-read latency is available without the stage trace.

## Step 2

Skipped: no screen arm, S7 included, beat S0 by at least 1.5 ms/token.
