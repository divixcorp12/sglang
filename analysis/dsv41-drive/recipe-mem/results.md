# Recipe GPU memory on driver 615.71.09: MEM_FRACTION_STATIC 0.885, SGLANG_MOE_HOT_GPU_MB 15400

2026-09-29, divix01. Why the old recipe stopped fitting is in `diagnosis.md`: driver 615.71.09's EAGER context is
~0.5 GiB larger. The candidates tried, in order:

| recipe (fraction / hot MiB) | commit | KV available (GB) | free after decode graph capture (GB) | outcome |
|---|---|---|---|---|
| 0.90 / 16100 (old) | master | 0.14–0.38 | 2.33–2.37 | KV sizing refused some launches (floor 0.23 GB) |
| 0.91 / 16100 | `5ee8a23565` | 0.46–0.66 | 1.92–1.97 | **16k layer-major OOM** |
| 0.895 / 15400 | `4482837694` | 0.71–0.88 | 2.26–2.32 | chunked 16k: **2 OOM retries**, 1 MiB free at peak; stopped |
| **0.885 / 15400** | **`926cfb58c9`** | **0.46–0.59** | **2.67–2.72** | **all checks pass** |

- **Why 0.895 failed.** Its hot-cache cut went into the KV pool (311k–425k tokens), not prefill headroom.
- **Why 0.885 works.** It spends ~320 MiB of that KV spare outside the static budget. Headroom after capture is now
  0.2 GB above the old driver's 0.90 (2.46–2.47).

## Verification at 0.885 / 15400 (`926cfb58c9`, no override)

`run_checks.sh`, all steps rc 0, in a fresh worktree under the locks (disk, then GPU), with the NUMA gates passing
before every launch. The pass stops on an allocator OOM retry or on under 64 MiB free at a peak.

**Free at a peak is 32,202 MiB − peak `memory.used`.** 32,202 MiB is what CUDA reaches: torch's total is
33766572032 B, and 0.895's chunked 16k peaked at 32,201 MiB `memory.used`. **DSV41_REFERENCE §27.18's "CUDA can use
only 32,150 MiB" is wrong by 52 MiB.** Its "~16 MiB margin" for the 09-26 chunked 16k at 0.90 was really ~68 MiB.

### Startup and KV sizing: 9 launches (the decode arm plus 8 smokes)

| launch | Load weight begin (GB) | KV available (GB) | KV pool (tokens) | free after capture (GB) |
|---|---|---|---|---|
| decode arm | 29.40 | 0.55 | 205,056 | 2.69 |
| lm16k / lm16k-2 | 29.34 / 29.36 | 0.46 / 0.50 | 150,784 / 175,104 | 2.72 / 2.71 |
| lm64k / lm64k-2 | 29.42 / 29.39 | 0.56 / 0.54 | 216,320 / 200,960 | 2.67 / 2.67 |
| ch16k / ch16k-2 | 29.43 / 29.41 | 0.59 / 0.57 | 232,448 / 219,648 | 2.67 / 2.67 |
| ch64k / ch64k-2 | 29.32 / 29.40 | 0.49 / 0.55 | 169,728 / 206,592 | 2.70 / 2.68 |

- **Every launch clears the KV floor.** The lowest available is 0.46 GB, against the 0.23 GB SWA floor.
- **The pool is smaller than the old recipe's.** It holds 150k–232k tokens, against ~387k at 0.90 on the old driver.
  One 64k prompt fits several times over.
- **Context capped to 131,072.** The GPU KV pool (150k–232k tokens) had fallen below `CONTEXT_LENGTH` 262,144. The
  cause is the driver 615.71.09 memory loss plus the smaller fraction and hot cache. The recipe's context is now
  capped at 131,072, below the smallest measured pool with margin for launch-to-launch variation. See "Context cap"
  below.

### Prefill: 4096-token chunks; lm = recipe (≥ 8192 uncached tokens run layer-major), ch = `SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0`

| prompt | run 1: TTFT s, peak MiB, free MiB, retries | run 2: TTFT s, peak MiB, free MiB, retries |
|---|---|---|
| lm 16,384 | 83.9, 31,589, **613**, 0 | 83.9, 31,607, **595**, 0 |
| lm 65,536 | 339.9, 31,649, **553**, 0 | 322.9, 31,665, **537**, 0 |
| ch 16,384 | 88.9, 31,973, **229**, 0 | 89.2, 31,969, **233**, 0 |
| ch 65,536 | 352.8, 31,949, **253**, 0 | 352.6, 31,945, **257**, 0 |

- **No OOM, no allocator retry.** Every prompt generated its 64 tokens, and every peak had ≥ 229 MiB free, above the
  128 MiB goal.
- **Free at Triton loads.** The lowest `free device mem` at a serving-time Triton load was 0.22 GiB (ch16k).
- **Tightest case.** The chunked path is the tightest; it is the path every prompt of fewer than 8,192 uncached tokens
  takes.

### Decode arm (`run_arm.sh`, standard 2 sessions, untraced)

| recipe | pooled ms/token | session 0 / 1 | output |
|---|---|---|---|
| 0.91 / 16100 (`mem091`) | 100.58 | 123.85 / 98.91 | |
| 0.895 / 15400 (`mem0895`) | 101.89 | 123.9 / 100.31 | |
| **0.885 / 15400 (`mem0885`)** | **102.44** | 124.4 / 100.88 | byte-identical to `mem091` and to the weight-pair arm U1 |

- **Cost of the change.** The 700 MiB smaller hot cache costs about **+1.9 ms/token**, in line with §27.7's slope. It
  is one arm per recipe against the weight pairs' ±0.3 spread, so treat it as an estimate.
- **Validity.** The arm passed run_arm's gates (rc 0; clocks 2962–2970 MHz), and `cuda graph: True` appears in the
  decode lines.

## Where things are

- **Logs.** `divix01:/mnt/nvme1/recipe-mem/checks-0885.log`, `peaks.jsonl`, `gates.jsonl`;
  `/mnt/nvme1/prefill-chunk/mem0885-*/` (`server.log`, `vram.csv`, `long.json`, `retries.txt`); and the decode arm at
  `dsv41-baseline/servers/mem0885/run-20260929-111245`.
- **Earlier candidates.** `mem091-*` and `mem0895-*` beside them.
- **Cleanup.** The worktree `wt-recipe-mem` and its generations registration (`recipe-mem-0885`) were removed.

```bash
# divix01, worktree at 926cfb58c9
taskset -c 0-63 env OMP_NUM_THREADS=8 bash analysis/dsv41-drive/recipe-mem/run_checks.sh $WT 926cfb58c93e5e14c44d7b2d62e7b9d596041887
```
