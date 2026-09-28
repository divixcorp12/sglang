# io_uring registration decode arms: R0 (defaults) vs R3 (arena + readv_fixed + fixed files), 2026-09-28

Plan 2026-09-28-reader-crtp-uring-registration, Task 9 (the fresh pair after Task 10).

**Verdict: keep the defaults.** R3 is **+0.19 ms/token** against R0 (101.65 vs 101.46 pooled) with byte-identical
output, so it does not beat R0 by the 1.5 ms/token the rule requires. It does not go to a confirmation plan.
Registration works end to end at the full tier: 280 chunks, 107.4 GB, in 59.2 s, with ~1 in 3 fixed reads fanned
out across chunks. It moved nothing measurable in decode. It costs about 60 s more startup, and the first full-tier
attempt at it hung the server (see "How we got here").

## What ran

| | R0 | R3 |
|---|---|---|
| io_uring knobs (all nine set explicitly) | `MODE=default QUEUE_DEPTH=0 FIXED_FILES=0 READ_MODE=normal WAIT_MODE=block SQ_THREAD_IDLE_MS=1000 SQ_THREAD_CPU=-1 SLAB_ARENA=0 DIAGNOSTICS=1` | as R0 but `SLAB_ARENA=1 READ_MODE=readv_fixed FIXED_FILES=1` |
| commit / python tree | `86fff9b245` / `434ece471b21cfd57d943b5d759aa720217ba3d0` (Task 10 head `87d9b1bd25` + registration + driver) | same |
| generation label | `reader-crtp-uring-reg-numa2m` | same |
| run dir (`/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline/servers/`) | `R0/run-20260928-173215` | `R3/run-20260928-173630` |
| run_arm rc / read_errors | 0 / 0 | 0 / 0 |

**Harness and host:**
- Worktree `divix01:/data/models/slang/nvfp4-work/wt-reader-crtp` (detached, clean).
- Driver `analysis/dsv41-drive/uring-reg/drive_uring_reg_arms.sh`, with output in `divix01:/mnt/nvme1/uring-reg-numa2m/`:
  `driver.log`, `arms-report.json`, `node0-gate.jsonl`, `memory.jsonl`, `<arm>-startup-mem.jsonl`,
  `<arm>-stacks.txt`, `<arm>-uring.txt` and the clock CSVs.
- Host: kernel `6.12.0-211.60.1.el10_2`, THP `[always]`, NVIDIA driver 615.71.09. Production (7867) was down.
- EXL3 gate: the worktree's `expert_stream_requirements_for` resolved `EXL3` before each arm.
- Arms ran back to back: R0 from 17:32:15 to 17:36:20, R3 from 17:36:30 to 17:41:35 CDT.

**Launch** (no outer flock; the driver takes `rowimg-disk.lock` itself, and run_arm.sh takes `cc-gpu.lock`):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-reader-crtp && mkdir -p /mnt/nvme1/uring-reg-numa2m \
  && OMP_NUM_THREADS=8 setsid nohup taskset -c 0-63 bash analysis/dsv41-drive/uring-reg/drive_uring_reg_arms.sh \
     $PWD $(git rev-parse HEAD) /mnt/nvme1/uring-reg-numa2m 30031 > /mnt/nvme1/uring-reg-numa2m/driver.log 2>&1 < /dev/null &'
```

## Tier and node gates

The tier is the **full recipe tier, `0:61440,1:40960 / 102400`**, set as a driver override in both arms. Before R0
the driver chose it over the 90 GiB fallback because both nodes cleared its gates. The gates count MemFree plus page
cache, i.e. `Active(file)` + `Inactive(file)`, in MiB.

| before | node 0 available | need (61440+4096+15360) | node 1 available | need (40960+4096+10000) | ok |
|---|---|---|---|---|---|
| R0 | 80927 + 7307 = 88234 | 80896 | 54606 + 28422 = 83028 | 55056 | yes / yes |
| R3 | 80852 + 7488 = 88340 | 80896 | 54658 + 28404 = 83062 | 55056 | yes / yes |

The node-1 footprint of 10000 MiB was observed on the first run's R0. Node-1 MemFree fell from 51231 to 1193 MiB at
"ready" with a 40960 MiB share, so about 9078 MiB beyond the share.

- **Comparable** to references at the full recipe tier (102400), e.g. the first run's R0 below.
- **Not comparable** to the 98304 tier (Task 5, 102.40/102.56) or to the cancelled 90 GiB re-run (103.69).
- The server's startup line reports `bound_bytes` 64,424,509,440 on node 0 and 42,939,310,080 on node 1 (Task 10's
  2 MiB-aligned binding).

## Decode (2 timed sessions, one turn each; pooled = total decode s / total decode tokens)

| session | tokens | R0 ms/token | R3 ms/token | Δ (R3 − R0) | R0 TTFT s | R3 TTFT s |
|---|---|---|---|---|---|---|
| cfq-train-Single_CDW/2015/page_35.pdf-2 | 6 | 125.22 | 125.33 | +0.11 | 8.211 | 8.198 |
| cfq-train-Single_ETR/2004/page_261.pdf-1 | 84 | 99.76 | 99.96 | +0.20 | 7.704 | 7.698 |
| **pooled** | 90 | **101.46** | **101.65** | **+0.19** | median 7.957 | median 7.948 |

- SM clock over the timed window was min 2940, median 2955, max 2955 MHz in both arms (limit 3135). At session
  start it was 2955 in both arms and both sessions.
- **Output identity:** every timed turn's `(reasoning, content)` is byte-identical between R0 and R3, 2 of 2
  (`mirror3_report.identity`).

## Reader, registration, counters

| | R0 | R3 |
|---|---|---|
| io_uring setup | `read_mode=normal fixed_files=0 fixed_buffers=0 registered_bytes=0` | `read_mode=readv_fixed fixed_files=120 fixed_buffers=280 registered_bytes=107,363,553,792` |
| regions / chunks / largest chunk | 0 / 0 / 0 | **240 / 280 / 1,070,530,560 B** (cap 1 GiB) |
| register_ms | 0.0 | **59,231** |
| fixed_reads / fixed_cuts / fanout_sqes | n/a (not captured: no fixed reads) | **101,208 / 33,736 / 118,076** (last line at close; a periodic line at 65,536 reads also logged) |
| rows_read / served / read_errors | 4209 / 2944 / 0 | 4262 / 2957 / 0 |
| copy_errors, copy_fallbacks, overruns, late_after_fatal, piece refusals | all 0 | all 0 |

The RAM-miss counters cover the server's lifetime: warm-up, prefill and the timed set. `fixed_cuts` is the number of
fixed reads whose destination spanned more than one registered chunk and so fanned out: 33,736 of 101,208 (33%),
issuing 118,076 SQEs for those reads.

## Startup and memory

| | R0 | R3 |
|---|---|---|
| first server log line → "fired up" | **113 s** (17:32:56 → 17:34:49) | **172 s** (17:37:11 → 17:40:03) |
| "Load weight end" → "Pinned host expert cache startup" | 33 s | 33 s |
| "Pinned host expert cache startup" → "fired up" | 34 s | **92 s** (registration 59.2 s) |
| node-0 / node-1 MemFree at "ready" (MiB) | 1946 / 3453 | 5396 / 3932 |
| VmallocUsed at "ready" (Δ vs baseline 2975 MiB) | 4289 (+1314) | 4490 (+1515; **+201 vs R0**) |
| Slab at "ready" (Δ vs baseline 5412 MiB) | 5558 (+146) | 5460 (+48; −98 vs R0) |
| startup vmstat Δ (launch → ready): thp_fault_fallback | +21,546 | +21,484 |
| compact_stall / compact_fail | +50 / +48 | +102 / +98 |
| allocstall (sum) / pgscan_direct | +27 / +35,623 | +54 / +81,990 |

- Memory baseline before R0: node-0 MemFree 80927 MiB, node-1 54606 MiB.
- The registration stop rule (300 s after the pinned-cache line) never fired. The driver saw the setup line 60 s
  after the pinned-cache line in R3, and 5 s after it in R0.
- `/proc/<pid>/task/*/stack` is root-only here, so all per-minute stack samples read "not readable".
- The THP fault fallback during startup is about the same in both arms: ~21.5k faults fell back to 4 KiB. That is a
  property of free-memory fragmentation at launch, not of the arm. R3 pays for it in registration time, because
  mixed-folio chunks do not coalesce (see Task 10), and R0 does not.

## Placement-change check (Task 10 changed NUMA binding)

This pair ran on the full tier, so the new R0 is compared with **Task 9's first-run R0**. That run used the full tier,
the pre-fix placement with page-rounded `mbind` splits, the same prompts, and the same commit apart from
`host_numa.py`. It is at `/mnt/nvme1/uring-reg`, run dir `R0/run-20260928-160757`.

| | first-run R0 (pre-fix placement) | new R0 (2 MiB-aligned placement) |
|---|---|---|
| pooled ms/token | 101.89 | 101.46 (−0.43) |
| session ms/token | 125.50 / 100.21 | 125.22 / 99.76 |
| median TTFT s | 7.964 | 7.957 |
| rows_read / served / read_errors | 4139 / 2926 / 0 | 4209 / 2944 / 0 |
| output | | **byte-identical, 2 of 2 turns** |

**This is the check on the placement change.** It found no output change and no slowdown: −0.43 ms/token is inside
run-to-run noise. The new R0's output is also byte-identical to the cancelled 90 GiB re-run's R0.

## Decision rule and limits

The rule: R3 goes to a confirmation plan (≥3 alternating pairs, with arms that separate arena, fixed buffers and
fixed files) **only if it beats R0 by ≥1.5 ms/token with identical output**. Otherwise the defaults stay. Output is
identical, but R3 is 0.19 ms/token **slower**, so **the defaults stay. Registration moved nothing.**

- **Drift is not controlled.** There is no repeated baseline arm, so a change in machine state between R0 and R3
  (clocks, page cache, drive temperature, memory fragmentation) cannot be told apart from an effect. The
  2026-09-28 campaign's S0 vs S0b differed by 0.3 ms/token, which gives a rough scale only.
- **A win could not have been attributed to a single mode.** R3 changes the arena layout, fixed buffers and fixed
  files at once. There is no win here, so there is nothing to separate.
- The timed set is the harness default: 2 sessions and 90 decode tokens, and the 84-token session dominates.
- The verdicts carry the same acknowledged problems as earlier runs. For example, provenance "imported sglang from
  None" (the provenance field is unavailable), and there is no engine-side step latency.

## How we got here: the R3 startup hang and the fix

1. **First run, full tier** (`/mnt/nvme1/uring-reg`, commit `8398e579e2`, tree `c008dddb31`).
   - R0 passed. At its "ready", node-0 MemFree was 1881 MiB and node-1 1193 MiB.
   - R3 logged "Pinned host expert cache startup" and then nothing for 13+ minutes: no error and no io_uring line.
     run_arm.sh aborted at its 900 s health gate.
2. **Root cause: quadratic pin accounting in the 6.12 kernel's buffer registration.**
   - `io_sqe_buffer_register` coalesces a chunk's pages into one entry per folio only if all its folios have the
     same size.
   - `io_buffer_account_pin` → `headpage_already_acct` then walks every bvec of every buffer already registered,
     for each new compound head page.
   - Before Task 10, `host_numa.allocate_bound` placed its per-node `mbind` splits on page-rounded row boundaries.
     These split the arena's VMAs at non-2 MiB addresses. The pages there were 4 KiB, so chunks spanning them mixed
     folio sizes, stayed uncoalesced at 262,144 bvecs per GiB, and each paid a walk over all earlier chunks.
   - Repro, outside the server: production `RegisteredBufferTable`, 1 GiB chunks, node 0, no GPU and no disk.

     | memory | 8 GiB | 16 GiB | 32 GiB | per-chunk cost |
     |---|---|---|---|---|
     | THP, coalesced | 0.16 s | — | 1.46 s | 3 → 88 ms |
     | **THP with a mid-chunk 4 KiB split** (the production shape) | **33.9 s** | **139.2 s** | fit ≈ 570 s | **0.26 s + 1.13 s × (earlier chunks)** |
     | 4 KiB only (MADV_NOHUGEPAGE) | 0.31 s | — | 0.61 s | flat |

   - On the live server: in the 90 GiB re-run, R3's scheduler thread was state R, 100% kernel time (+501 stime ticks
     in 5 s) and wchan 0. Its stack was `io_uring_register_buffers_update_tag` ← `RegisteredBufferTable::add` ←
     `UringReader::register_resources` ← `configure_resources` ← `ReaderCore::open`.
   - Kernel stacks and perf are root-only on this box, so the kernel frame is inferred from the v6.12 source and
     the exact linear-per-chunk growth.
   - Excluded: the arena owner is never a registration region (the regions are the 240 named slabs with their true
     row bytes); a userland loop or lock; memory pressure.
3. **90 GiB re-run** (`/mnt/nvme1/uring-reg-90g`, `0:51200,1:40960`).
   - R0 finished at 103.69 ms/token, with a 90 GiB tier that is not comparable.
   - R3 sat in `register_resources` with the same signature, and was **cancelled** by directive.
4. **Task 10 fix** (`40452dacc2`): 2 MiB-aligned mapping bases and node-change boundaries (`host_numa.plan_bindings`).
   - Harness registration at the 90 GiB tier layout went from an extrapolated hour to **58.4 s**. At 8/16/32 GiB it
     takes 0.14/0.54/2.06 s. The per-GiB growth is 3.9 ms per earlier GiB, the coalesced-THP floor, against 906
     for the old-alignment control.
   - About **46 s of the 58.4 s** comes from THP fault fallback: 7,480 fallbacks, or 14.6 GiB of 4 KiB pages. Those
     mixed-folio chunks still pay the walk.
   - **MADV_COLLAPSE** (measured only) did not recover them: 63 of 267 calls failed (ENOMEM on the big slabs),
     `thp_collapse_alloc` +18 against `thp_collapse_alloc_failed` +46, and registration after it took 47.3 s with
     the same shape.
   - In this pair, R3 registered the full tier in **59.2 s**.

Remaining risk: R3's startup cost depends on free-memory fragmentation at launch, through the THP fault fallback.
Anyone re-enabling registration should expect about a minute of registration or more, and it varies.
MADV_NOHUGEPAGE on the slabs would make registration linear (0.6 s per 32 GiB in the repro), but it was not tried
in decode.
