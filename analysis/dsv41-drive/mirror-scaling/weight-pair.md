# Does down-weighting the SPCC mirror improve decode? (W 1:0.9:1 vs U 1:1:1)

2026-09-29, divix01, after the 02:02 CDT reboot. Three balanced alternating pairs plus one discarded warm-up arm.

## Verdict

**No. Weighting the SPCC 0.9 does not improve decode, so keep equal weights as the recipe default.**

- **Result.** W is **+0.38 ms/token** against U over the paired deltas (sd 0.46, range +0.01 to +0.89). All three pairs
  land at or above zero. The one-sided sign test for "W slower" gives p = 0.125, so this is not a detected regression
  either. It sits inside the pair-to-pair noise: U arms span 0.32 ms/token and W arms 0.56.
- **Against the predicted gain.** The disk bench predicts a 0.16 ms QD1 row p50 cut. Its ~12 row-equivalents of
  critical-path reads per token turn that into about −1.9 ms/token. The best pair measured +0.01, so a gain of that
  size is excluded by all three pairs.
- **Byte identity.** Output is byte-identical in every arm (both turns, reasoning and content, against U1).

## Numbers

Pooled ms/token = total decode s / total decode tokens over the 2 timed sessions (6 + 84 decode tokens; session 1 dominates).

| Arm | order | pooled ms/token | session 0 (CDW p35) | session 1 (ETR p261) | TTFT s1 (s) | SM clock median (min) MHz |
|---|---|---|---|---|---|---|
| X0 (U, warm-up, discarded) | 0 | 100.06 | 123.41 | 98.39 | 7.69 | 2962 (2940) |
| W1 | 1 | 100.65 | 124.80 | 98.93 | 7.76 | 2962 (2940) |
| U1 | 2 | 100.41 | 124.19 | 98.71 | 7.68 | 2962 (2940) |
| U2 | 3 | 100.54 | 124.10 | 98.86 | 7.71 | 2955 (2940) |
| W2 | 4 | 100.55 | 124.12 | 98.87 | 7.75 | 2955 (2940) |
| W3 | 5 | 101.11 | 125.15 | 99.39 | 7.76 | 2962 (2940) |
| U3 | 6 | 100.22 | 123.70 | 98.54 | 7.69 | 2959 (2940) |

| | W | U |
|---|---|---|
| mean pooled ms/token (n=3) | **100.77** (sd 0.30) | **100.39** (sd 0.16) |

| Pair (order) | W − U pooled | session 0 | session 1 |
|---|---|---|---|
| 1 (W1, U1) | +0.24 | +0.62 | +0.21 |
| 2 (U2, W2) | +0.01 | +0.01 | +0.01 |
| 3 (W3, U3) | +0.89 | +1.46 | +0.85 |
| **mean (sd)** | **+0.38 (0.46)** | | |

### RAM misses and drives

The RAM-miss counters cover the server lifetime: 3 warm-up rounds, prefill and the timed set, 326 generated tokens.
Drive numbers cover the timed window (server ready to the last session's end, ~27.3 s, prefill included), from
`/proc/diskstats`.

| Arm | served RAM misses / generated token | rows_read | read_ms | share nvme0 / **SPCC** / nvme2 % | MB/s nvme0 / **SPCC** / nvme2 | device ms per read request nvme0 / **SPCC** / nvme2 |
|---|---|---|---|---|---|---|
| W1 | 8.98 | 4147 | 2693 | 34.4 / **30.9** / 34.6 | 884 / **794** / 890 | 8.20 / **5.62** / 4.81 |
| W2 | 9.02 | 4176 | 2694 | 34.4 / **30.9** / 34.6 | 891 / **800** / 896 | 8.20 / **5.63** / 4.79 |
| W3 | 8.94 | 4080 | 2719 | 34.4 / **30.9** / 34.6 | 887 / **797** / 893 | 8.21 / **5.61** / 4.80 |
| U1 | 8.97 | 4133 | 2658 | 33.3 / **33.2** / 33.5 | 866 / **865** / 871 | 7.52 / **7.02** / 4.32 |
| U2 | 9.03 | 4212 | 2634 | 33.3 / **33.2** / 33.5 | 860 / **859** / 865 | 7.53 / **7.02** / 4.30 |
| U3 | 9.01 | 4199 | 2672 | 33.3 / **33.2** / 33.5 | 863 / **861** / 868 | 7.54 / **6.99** / 4.30 |

- **The weights take effect.** The SPCC's share falls from 33.2% to 30.9%, which is 0.9/2.9 = 31.0% as intended.
- **The miss count is unchanged.** RAM misses per token are the same in both arms (8.94–9.03). The weighting changes
  only where the bytes come from.
- **The latency moves to the Samsungs.** The SPCC's mean device time per read request falls from 7.0 to 5.6 ms. The
  Samsungs rise: nvme0 goes 7.5 → 8.2 ms and nvme2 goes 4.3 → 4.8 ms. The service's cumulative `read_ms` rises
  ~1.5% (2655 → 2702 mean). Under decode's concurrent reads the SPCC is not the straggler the QD1 bench measured.
- **nvme0 is not comparable to nvme2.** nvme0 sees half the other Samsung's IOPS at the same MB/s, so its requests are
  twice the size, and its per-request time is not a like-for-like latency. It was not investigated.
- **Row read latency p50/p99 is not logged by the server**, so this pass cannot report it. The diskstats per-request
  time is the available proxy.

## Validity

- **Every arm passed.** Each arm has run_arm rc 0, `read_errors` 0, no compile event in a timed session and its env
  verified from `/proc/<pid>/environ`, including `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS`. Each ran at tier **full**
  `0:61440,1:40960 / 102400`, and both NUMA gates passed before every arm (`tier-gate.jsonl`). The lowest was U2/W2 on
  node 0, at 81074 + 6374 and 81843 + 5774 MiB against the 80896 needed.
- **Locks and co-tenants.** The locks were taken per arm and released between arms. Other lanes ran in the gaps: X0 →
  W1 waited 51 min, and W2 → W3 waited 11 min. No GPU process was present at any arm's start.
- **Clocks.** The SM clock was 2940–2962 MHz in every timed window.
- **Warm-up arm.** X0 (U weights) ran first after the reboot, with a cold page cache, and is excluded from every pair.
  Its 100.06 sits inside U's range.

## Deviation: `DSV41_MEM_FRACTION_STATIC=0.91` in every arm (recipe: 0.90)

The recipe did not start at 0.90, before or after the reboot. It failed at the scheduler's KV sizing:

```
RuntimeError: The DSV4 SWA pool cap (10752 tokens, 0.23 GB) leaves no room for the full KV pool within the
available 0.14 GB. [0.17 GB after the reboot]
```

| launch | `Load weight begin` avail | weights | KV `available_bytes` | outcome |
|---|---|---|---|---|
| 09-28 20:53 iopoll-cuts A (452c6ade7e, 0.90) | 29.31 GB | 9.96 GB | 0.27 GB | ran; full KV pool 26,624 tokens |
| 09-29 01:16 W1 attempt 1 (0.90) | 29.21 GB | 10.00 GB | 0.14 GB | refused |
| 09-29 02:14 X0 after the reboot (0.90) | 29.23 GB | 9.99 GB | 0.17 GB | refused |

- **What changed.** The launch starts ~0.10 GB lower in free GPU memory and the weights take ~0.04 GB more. No other
  process held the GPU (62–65 MiB used at start, the same as 09-28). The code diff from `452c6ade7e` to master is
  host-only (`reader_core.h`, `read_cuts.h`, `ffi_exports.h`).
- **Uniform change (lead-approved).** All seven arms ran at 0.91. It changes only the KV pool size, and decode here is
  BS1 at a short context.

KV sizing at 0.91, per arm:

| | X0 | W1 | U1 | U2 | W2 | W3 | U3 |
|---|---|---|---|---|---|---|---|
| available_bytes GB | 0.52 | 0.60 | 0.59 | 0.66 | 0.64 | 0.46 | 0.62 |
| full KV pool, tokens | 191,488 | 241,664 | 235,264 | 283,392 | 265,984 | 145,920 | 254,208 |

**Open recipe-margin issue.** The margin at 0.90 was ~0.04 GB (0.27 against 0.23), and it no longer holds on this
box. At a fixed config, available_bytes varies by 0.20 GB from launch to launch (0.46–0.66 above), which is five times
that margin. The cause of the lower pre-load free memory is not identified. The recipe (`MEM_FRACTION_STATIC=0.90`,
`SGLANG_MOE_HOT_GPU_MB=16100`) needs more headroom before the next arm that uses it unmodified.

## Timeline (CDT)

| time | event |
|---|---|
| 01:16–01:19 | W1 attempt 1 (0.90) refused at KV sizing |
| ~01:40 | Another lane's io_uring clone-buffers bug leaked ~150 GiB of pinned host memory. The driver was paused at 01:47 before any further launch. |
| 02:02 | divix01 rebooted |
| 02:14 | X0 at 0.90 refused at KV sizing (0.17 GB) |
| 02:47–02:51 | X0 at 0.91 (warm-up, discarded) |
| 03:42–04:20 | W1, U1, U2, W2, W3, U3 |

## How it ran

- **Code.**
  - Branch `cc/spcc-weight-pair`, driver commit `56dc922455`.
  - Python tree `41c40c486810079cbff51fdb804dbf3fd5b1fbef`, identical to `origin/master` `ba01695c35`. The branch adds
    only this directory's files.
  - Generation label `spcc-weight-pair`, registered untracked in the divix01 worktree and removed with it.
- **Arms.**
  - W: `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=1:0.9:1`.
  - U: `1:1:1`. `parse_mirror_weights` maps unset to exactly `(1.0,) * 3`, so U is the default. It is set explicitly so
    the two arms' env differs in that value only.
  - Everything else is `arm_env.base_env()`: 3 mirrors, default block wait, untraced.
- **Driver.** `drive_weight_pairs.sh` (template: `iopoll-cuts/drive_iopoll_cuts_arms.sh`).
  - It takes `rowimg-disk.lock` per arm, then polls for `cc-gpu.lock`, which `run_arm.sh` takes.
  - Before each arm it runs the foreign-process gates by exe name and the EXL3 gate, then settles the tier per pair.
  - The `/proc/diskstats` sampler runs on the mirror drives, resolved from the mounts: after the reboot `/mnt/nvme2`
    is nvme1n1; the SPCC is still nvme2n1.
- **Report.** `weight_pair_report.py`, reading from `divix01:/mnt/nvme1/spcc-weight/`: `pairs-report.json`,
  `driver.log`, `tier-gate.jsonl`, `<arm>-diskstats.jsonl` and `<arm>-clocks.csv`. The failed pre-reboot and 0.90
  launches are in `failed-W1-1/`, `prepause/` and `failed-X0-090/`.

```bash
# divix01, worktree at 56dc922455
DSV41_MEM_FRACTION_STATIC=0.91 taskset -c 0-63 env OMP_NUM_THREADS=8 \
  bash analysis/dsv41-drive/mirror-scaling/drive_weight_pairs.sh $WT 56dc922455ce6dc2c28c93e06682585d48c198a7 /mnt/nvme1/spcc-weight
```
