# Does down-weighting the SPCC mirror improve decode? (W 1:0.9:1 vs U 1:1:1)

2026-09-29, divix01. **Status: paused, no arm completed.** Resumes after divix01's reboot, on the team lead's go.

## Question

The disk-only bench (`results.md` on `cc/mirror-scaling-bench`) found the SPCC (`/mnt/nvme4`, nvme2n1) finishing last in
~90% of three-root rows. Weighting it 1:0.9:1 cut QD1 row p50 by 9.5%, and QD2 was neutral. RAM misses are ~21–30 ms of
~100–110 ms/token, so any decode effect should be small. That is why the design uses paired arms.

## Design

- **Code.** The harness is on branch `cc/spcc-weight-pair`, cut from `origin/master` `ba01695c35`. Its python tree is
  `41c40c486810079cbff51fdb804dbf3fd5b1fbef`, identical to master's; the branch adds only files under this directory.
  - The driver commit is `8e63a0aafd`, and the KV-sizing report fields came in `d5396cd59b`.
  - The divix01 worktree is `wt-spcc-weight`, with generation label `spcc-weight-pair`, registered untracked.
- **Arms.**
  - W: `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=1:0.9:1`.
  - U: `1:1:1`. `parse_mirror_weights` maps an unset value to exactly `(1.0,) * 3`, so U is the default. It is set
    explicitly so that the two arms' env differs in that one value.
  - Everything else is the standard recipe from `arm_env.base_env()`: 3 mirrors, default block wait, untraced.
- **Order.** Balanced alternating pairs: W1 U1 / U2 W2 / W3 U3, optionally U4 W4 / W5 U5.
- **Tier.** Full `0:61440,1:40960` when both NUMA gates pass. The gates are node 0 ≥ share + 4096 + 15360 and node 1 ≥
  share + 4096 + 10000. Otherwise the fallback is `0:51200,1:40960`. The tier is settled at a pair's first arm, and the
  pair's second arm must pass the same tier's gates; a pair's tier is never cut.
- **Driver.** `drive_weight_pairs.sh`, from the iopoll-cuts template.
  - It takes `rowimg-disk.lock` per arm and polls for `cc-gpu.lock`, which `run_arm.sh` then takes. Both locks are
    released between arms.
  - Before each arm it checks for foreign GPU processes by exe name and for foreign pytest, nvcc and cc1plus.
  - It runs the EXL3 gate and a `/proc/diskstats` sampler on the three mirror drives, and requires `read_errors == 0`.
- **Report.** `weight_pair_report.py` computes the following per arm:
  - pooled and per-session ms/token, and TTFT;
  - SM clocks;
  - RAM-miss counters (server lifetime) and misses served per generated token;
  - per-drive read share and mean device read time per request over the timed window;
  - KV sizing (`available_bytes`, `full_token`);
  - the per-pair W − U deltas with their spread, and byte identity against the first U arm.
- **Not logged by the server.** Row read latency p50/p99 is not in the server's logs. The per-drive read time from
  diskstats is the available proxy.

## What happened

| time (CDT) | event |
|---|---|
| 01:16 | W1 attempt 1 launched at tier **full**. Gates: node 0 84968 + 187 = 85155 MiB (need 80896), node 1 52764 + 24611 = 77375 MiB (need 55056). |
| 01:19 | **W1 failed at GPU KV sizing** (below). run_arm rc=1, and the driver stopped. |
| 01:21 | The driver restarted for one unchanged retry of W1, then queued for the locks. |
| 01:38 | The retry's gate failed on host memory: node 0 1216 + 712 MiB, node 1 22508 + 1671 MiB. The driver released its locks and waited. |
| 01:47 | The driver was stopped on the lead's PAUSE. It had not launched a server since 01:19. |

**No arm completed, so there are no decode numbers, no paired delta and no verdict yet.** No arm ran under the host-memory
pressure, so none needs to be marked invalid.

### W1 attempt 1: GPU KV sizing failure

```
RuntimeError: The DSV4 SWA pool cap (10752 tokens, 0.23 GB) leaves no room for the full KV pool within the
available 0.14 GB. Reduce --max-running-requests, lower --swa-prefix-tails or SGLANG_SWA_EVICTION_INTERVAL,
or increase --mem-fraction-static.
```

Run dir: `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/W1/run-20260929-011626`.
Driver logs: `/mnt/nvme1/spcc-weight/failed-W1-1/`.

**Not contention.** No compute app held the GPU. `memory.used` was 62 MiB at run start, the same as on 09-28. The code
diff between the last good arms (`452c6ade7e`) and master (`ba01695c35`) is host-only (`reader_core.h`, `read_cuts.h`,
`ffi_exports.h`).

## Open issue: the recipe's GPU memory margin is ~0.04 GB

This compares W1 with the last good recipe arm (iopoll-cuts A, 2026-09-28 20:53, same recipe, `MEM_FRACTION_STATIC=0.90`):

| | 09-28 A | 09-29 W1 | Δ |
|---|---|---|---|
| `Load weight begin` avail mem | 29.31 GB | 29.21 GB | −0.10 GB |
| weights (`mem usage` at load end) | 9.96 GB | 10.00 GB | +0.04 GB |
| hot cache allocation | 16870844928 B | 16870844928 B | 0 |
| KV `available_bytes` | 0.27 GB | 0.14 GB | −0.13 GB |
| SWA fixed cap | 0.23 GB | 0.23 GB | |

The good arm cleared the SWA cap by only 0.04 GB (0.27 against 0.23), and its full KV pool was 26,624 tokens. A 0.10 GB
drop in pre-load free memory plus 0.04 GB more for weights was enough to refuse the launch. The cause of the pre-load
drop is not identified. The recipe is fragile at `0.90` with `SGLANG_MOE_HOT_GPU_MB=16100`.

**Agreed fallback (lead):** after the reboot, W1 gets one unchanged try. If KV sizing still fails, every arm runs at
`DSV41_MEM_FRACTION_STATIC=0.91`, uniformly, recorded as a deviation. Each arm's `available_bytes` and KV pool size will
be recorded either way.

## Resume

```bash
# divix01, after the lead's go
git -C /data/models/slang/nvfp4-work/wt-spcc-weight fetch origin && \
  git -C /data/models/slang/nvfp4-work/wt-spcc-weight checkout --detach origin/cc/spcc-weight-pair
WT=/data/models/slang/nvfp4-work/wt-spcc-weight
setsid nohup taskset -c 0-63 env OMP_NUM_THREADS=8 [DSV41_MEM_FRACTION_STATIC=0.91] \
  bash $WT/analysis/dsv41-drive/mirror-scaling/drive_weight_pairs.sh $WT $(git -C $WT rev-parse HEAD) \
  /mnt/nvme1/spcc-weight > /mnt/nvme1/spcc-weight/driver.log 2>&1 < /dev/null &
```
