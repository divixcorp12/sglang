# Three mirror roots: blocked by piece streaming's part limit, 2026-09-28

**Status: a 3-root run is not possible on current master with the production recipe.** B1 refused at startup:

```
RuntimeError: exl3 RAM miss: piece streaming reads at most kPieces / kSubReads mirror parts
```

`reader_base.h:61-62` sets `kSubReads = 4` and `kPieces = 8`, so piece streaming accepts at most 8 / 4 = **2** mirror
parts. The check is `row_reader.h:108` and it is raised from `enable_piece_stream`. The recipe sets
`SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM=1`, so every 3-root arm refuses. The limit is the code's own guard, not a
problem with the `/mnt/nvme2/dsv41_flash` root, which `build_row_images.py --verify` passed on all three roots. The
planned fix is a code change: per-part sub-reads = kPieces / parts. The 3-drive run waits for that change.

The planned A/B was cut. What ran: A1 (2 roots), B1 (3 roots, refused), and A2 (2 roots, already started when the
refusal was found; kept as a second same-code reference). B2, A3 and B3 were not run.

## Code and harness

- `arm/mirror3` at `4185bf17c8`: origin/master `1732aff4ba` (the production recipe with PDL and layer-major prefill)
  plus one commit registering python tree `e4f5d5bec2e5` as `mirror3-1732aff4ba`. Worktree:
  `/data/models/slang/nvfp4-work/wt-mirror3` on divix01.
- Launch, via the wrapper `divix01:/mnt/nvme1/sf-ab/arm2.sh`:
  `EXPECT_SHA=<HEAD> OMP_NUM_THREADS=8 flock rowimg-disk.lock taskset -c 0-63 bash benchmarks/dsv41_baseline/run_arm.sh <arm> 30031 [override]`.
  The wrapper checks the driver (615.71.09), production (7867 down), the port, foreign processes (comm/exe names and
  exact argv elements), the GPU and the GPU lock. It also runs a 2 s sampler of per-thread CPU and `/proc/diskstats`.
- A: no override, so the recipe's `SGLANG_MOE_EXPERT_MIRROR_DIRS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash`.
  B: `SGLANG_MOE_EXPERT_MIRROR_DIRS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash:/mnt/nvme2/dsv41_flash`.
  `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS` was unset in both.
- Device map: `/mnt/nvme0` = nvme0n1, `/mnt/nvme4` = nvme2n1, `/mnt/nvme2` = nvme3n1 (the 990 EVO Plus, Gen3 x4).
  `/mnt/nvme2` also holds the source checkpoint (`EXPERT_DIR`), read only at startup, and the Engram table directory.

## B1 (3 roots): refused

Run dir `mirror3-B1/run-20260928-040412`. `server_args` was logged at 04:04:58, and the scheduler raised the error
above at 04:06:18, during hot-cache startup (`expert_hot_cache._load_reserved` → `exl3_ram_miss.ensure_started` →
`enable_piece_stream`). run_arm rc=1. No requests were served, and no timed data exists.

## A1 and A2 (2 roots): single-run references, not a paired result

The SM clock was 2970 MHz at the start of every session. Both runs are byte-identical to each other.

| Run | Session 0 ms/token (TTFT) | Session 1 ms/token (TTFT) | **Pooled** | rows_read / served | read errors, timeouts, fatals | startup |
|---|---|---|---|---|---|---|
| A1 `run-20260928-035827` | 128.8 (8.65 s) | 102.6 (8.09 s) | **104.4** | 4047 / 2900 | 0 / 0 / 0 | 230.1 s |
| A2 `run-20260928-040631` | 127.8 (8.60 s) | 101.9 (8.10 s) | **103.6** | 4182 / 2938 | 0 / 0 / 0 | 134.5 s |

Session 0 decodes only 7 tokens, so session 1 (85 tokens) carries the pooled figure. Both runs were "valid except
the acknowledged step-latency gap". late_after_fatal, copy_errors and slots_quarantined were 0. A1's startup
includes a colder page cache, since it followed the row-image build and verify.

### Per-drive reads over the timed window (`/proc/diskstats` deltas, 28.6 s)

| Drive | A1 read | A1 rate | A1 reads | A2 read | A2 rate |
|---|---|---|---|---|---|
| nvme0n1 (`/mnt/nvme0`, mirror) | 35.00 GB | 1223 MB/s | 84,163 | 35.25 GB | 1234 MB/s |
| nvme2n1 (`/mnt/nvme4`, mirror) | 34.98 GB | 1222 MB/s | 164,965 | 35.23 GB | 1233 MB/s |
| nvme3n1 (`/mnt/nvme2`, not a mirror here) | 0.18 GB | 6 MB/s | 42,224 | 0.18 GB | 6 MB/s |

The two mirrors split the byte load evenly, as equal shares should. The nvme3n1 traffic is ~4 KB reads, most likely
the Engram table on that drive. It is not expert rows, because the 2-root recipe does not read that root.

**Reading the rates.** ~1.2 GB/s per mirror drive is about a third of the ~3.5 GB/s per drive that the row-image
verify sustained. That average covers the whole window, including ~17 s of TTFT across the two sessions. So it
shows the drives are far from saturated on average. By itself it does not prove the path is latency-bound rather
than bandwidth-bound, because short bursts may be much faster.

## Next

The 3-drive run is waiting on the piece-streaming change: sub-reads per part = kPieces / parts. `kRowPieces = 8` in
`row_copy_kernels.cuh` is the device-side twin of `kPieces`. Once the change is merged, rerun B (and A on the same
commit as a reference). Run dirs, logs and samples are in `divix01:/mnt/nvme1/sf-ab/mirror3-*`.
