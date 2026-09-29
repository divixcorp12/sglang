# IOPOLL wait fix and read cuts: decode arms A/B/C/D/A2, 2026-09-28

Plan `docs/superpowers/plans/2026-09-28-iopoll-read-cuts.md`, Task 7. Diagnosis: `analysis/dsv41-drive/iopoll/diagnosis.md`.

**Verdict: the IOPOLL regression is gone, and nothing is promoted.**
- **C (IOPOLL).** With the wait fix and read cuts, `MODE=iopoll` decodes at **101.23 ms/token**, **+0.62** against the A/A2 baseline of 100.61, with byte-identical output. It had **0 io-wq workers over the whole server run**, against 12–14 in the Codex S4 arm (+10.2 ms/token then).
- **B (cuts in default mode).** It lands at **+0.14** and does not clear the −1.5 ms/token promotion bar, so `READ_CUTS` stays `auto` (on only under IOPOLL). The −40 µs per row that cuts saved in the microbenchmark does not show up in decode.
- **D (`sqpoll_iopoll`).** It is neutral at **+0.18** and spends a full core on the SQ thread (24.3 CPU-s in a 24.5 s window).
- **Validity.** Every arm has byte-identical output to A, `read_errors=0` and run_arm rc=0. Drift A2 − A is −0.68 ms/token, inside the plan's 1.0 validity bound.
- **No confirmation pass.** No arm reached Δ ≤ −1.0, the plan's trigger for one.

## What ran

| | |
|---|---|
| commit / python tree | `452c6ade7e` / `b33e351fa4ce037b9a4d4d4dfdd79f67580bb89f` (generation label `iopoll-read-cuts`, registered for the arms and reverted afterwards) |
| worktree | `divix01:/data/models/slang/nvfp4-work/wt-iopoll-cuts` (detached, clean) |
| driver | `analysis/dsv41-drive/iopoll-cuts/drive_iopoll_cuts_arms.sh`; output `divix01:/mnt/nvme1/iopoll-cuts/` (`driver.log`, `arms-report.json`, `<arm>-uring.txt`, `<arm>-threads.jsonl`, clocks, memory) |
| host | kernel `6.12.0-211.60.1.el10_2`, NVIDIA 615.71.09, production (7867) down, EXL3 gate resolved `EXL3` before every arm |
| tier | **full** `0:61440,1:40960 / 102400`. Gates before A: node 0 82222 + 6080 = 88302 MiB (need 80896), node 1 47250 + 35919 = 83169 MiB (need 55056); rechecked before every arm |
| time | A 20:53:47 → A2 end 21:18:16 CDT, back to back |

All ten `SGLANG_EXPERT_STREAM_URING_*` knobs were set explicitly in every arm:

| Arm | Overrides |
|---|---|
| A, A2 | `MODE=default READ_CUTS=0 WAIT_MODE=block QUEUE_DEPTH=0 FIXED_FILES=0 READ_MODE=normal SQ_THREAD_IDLE_MS=10000 SQ_THREAD_CPU=-1 SLAB_ARENA=0 DIAGNOSTICS=1` |
| B | as A, `READ_CUTS=1` |
| C | as B, `MODE=iopoll` (`WAIT_MODE=block`, which the wait fix turns into `effective_wait=reap`) |
| D | as B, `MODE=sqpoll_iopoll SQ_THREAD_CPU=20` |

A runs this commit with cuts off in default mode. The golden and reader suites show that this prepares master's SQE set and writes master's bytes.

## Decode (2 timed sessions; pooled = total decode s / total decode tokens)

| Arm | session 0 ms/token (7 tok) | session 1 ms/token (84 tok) | **pooled** | **Δ vs mean(A, A2)** | TTFT s (s0 / s1) | output vs A |
|---|---|---|---|---|---|---|
| A | 125.05 | 99.23 | **100.95** | +0.34 | 8.17 / 7.69 | — |
| B | 124.39 | 99.06 | **100.75** | **+0.14** | 8.17 / 7.68 | identical |
| C | 125.37 | 99.50 | **101.23** | **+0.62** | 8.18 / 7.72 | identical |
| D | 125.01 | 99.06 | **100.79** | **+0.18** | 8.16 / 7.67 | identical |
| A2 | 124.15 | 98.56 | **100.27** | −0.34 | 8.20 / 7.70 | identical |

The SM clock was 2947–2970 MHz in every timed window (median 2962).

## io_uring, read cuts, threads

The thread window is the timed window (~24.5 s), sampled every 2 s over the server process tree. "Whole run" means every sample from `server pid=` to shutdown.

| Arm | effective flags / wait | SQ / CQ entries (requested depth) | read cuts: cut_bytes per drive | io-wq workers (window / whole run), CPU s | SQ thread CPU s | RAM-miss service CPU s | rows_read / served / read_errors |
|---|---|---|---|---|---|---|---|
| A | 0x10000 / block | 64 / 128 (48) | off | 0 / 0, 0.0 | — | 7.34 | 4223 / 2937 / 0 |
| B | 0x10000 / block | 2048 / 4096 (1056) | nvme0n1 520192, nvme2n1 **262144**, nvme3n1 520192 (sysfs) | 0 / 0, 0.0 | — | 7.53 | 4265 / 2977 / 0 |
| C | 0x10001 / **reap** | 2048 / 4096 (1056) | same | **0 / 0**, 0.0 | — | 8.15 | 4133 / 2899 / 0 |
| D | 0x10007 / block (idle 10000 ms, cpu 20) | 2048 / 4096 (1056) | same | 0 / 0, 0.0 | **24.34** | 7.17 | 4119 / 2919 / 0 |
| A2 | 0x10000 / block | 64 / 128 (48) | off | 0 / 0, 0.0 | — | 6.48 | 4122 / 2926 / 0 |

- **Depth.** With cuts the default credit is 16 · 3 parts · leg_stride 22 = 1056 (ring 2048/4096), as the plan's depth rule sets it.
- **Per-drive limits.** Each drive is cut at its own sysfs limit. `check_modes` required exactly 262144 + 2 × 520192 for B, C and D, and passed.
- **Against Codex.** S4 (`MODE=iopoll`, uncut, block wait) ran 12–14 `iou-wrk` workers using 4.3 s of CPU in the window, and S7 (`sqpoll_iopoll`) ran 6. C and D run none.
- **Service-thread CPU.** C's service thread uses a little more CPU (8.15 s against A's 7.34 and A2's 6.48), which is expected, because the reap loop spins while waiting. D's SQ thread is busy for the whole window, as in Codex S5–S7.
- **Sampling caveat.** The 2 s sampling could miss a worker that lives under 2 s. io-wq workers linger idle for seconds after their last request, and Codex's 2 s sampler saw S4's workers, so zero here is meaningful.

## Promotion rule and outcome

The rule: promote an arm only at Δ ≤ −1.5 ms/token vs mean(A, A2), with identical output. Run 3 alternating pairs first if any arm lands at Δ ≤ −1.0.

| Arm | Outcome |
|---|---|
| B | +0.14, **not promoted**: `READ_CUTS` stays `auto`. Turning cuts on for every mode stays the user's decision; these numbers do not support it. |
| C | +0.62, **not promoted**: the recipe keeps `MODE=default`. The fix still stands. IOPOLL no longer costs 10 ms/token; it costs a spinning waiter and gains nothing measurable. |
| D | +0.18, **not promoted**: a full SQ core for no gain. |

**Limits of this pass.** One pass gives one sample per arm. A2 bounds drift at 0.68 ms/token, which is smaller than the 1.5 ms bar but larger than B's and D's deltas. B, C and D are therefore indistinguishable from A at this resolution. The claim that holds is "C removed the +10 ms regression", not "C is 0.6 ms slower".

## Open item (not in scope)

**The SPCC drive (nvme2n1, `/mnt/nvme4`) has slow episodes.** In 7 of 24 multi-row microbenchmark runs its per-SQE p50 was 10–35 ms (`analysis/dsv41-drive/iopoll/diagnosis.md`, "System").
- **Health (user's smart-log):** healthy, with 0 media errors, 19% used and no thermal throttling.
- **Likely cause:** it is DRAM-less, with a 64 MiB HMB serving a 191 GB row-image span, so FTL map misses are the likely cause.
- **Parked:** a mirror-weight test is parked for later.
