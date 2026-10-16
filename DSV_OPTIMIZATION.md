# DSV4.1 decode optimization: state of play (handoff)

**Snapshot:** 2026-10-15, base branch `ram-prefetch-pool-serve` at `e02b8b9e0e`. This is a living collection of the
October 14-15 round: what was measured, what was built, what is still running and what was decided. Numbers are
per verify unless they say ms/token. Run directories are under
`divix01:/data/models/slang/nvfp4-work/ram-prefetch/` unless a full path is given. `DSV41_REFERENCE.md` remains the
record of older measurements; when the two disagree on an older number, the reference wins.

**Production on divix01 is stopped. Only the owner restarts it.** Everything new below is opt-in and off by
default; the owner decides production defaults.

## 1. How to read the numbers

- **Noise:** identical runs differ by about 7 ms/token (base 69.3 vs base-r2 76.2 on the same code). Only
  differences above that, or differences with a matching mechanism in `step_budget` (wait / stream / post), count.
  The owner chose single A/B pairs ("no a/b/a/b"); where an OFF sample already exists, only the ON arm is run.
- **Output divergence between arms is ordinary nondeterminism:** identical OFF pairs flip at the same near-tie
  tokens (DISCA token 25, ETR 31, BKR 18).
- **Memory state confounds runs.** Until the `-mem` arms (section 3), node 0 swapped during load and decode, and runs
  were not comparable across box states. Compare `-mem` arms only with `-mem` arms.
- **Traced vs untraced:** replay costs are traced marginals; untraced is roughly 0.9x.

## 2. Results table (DSpark arms, split-model OFF/ON)

| run | ms/token | median verify | mean verify | wait | stream | post | DMA busy |
|---|---|---|---|---|---|---|---|
| base (pre-pin01) | 69.3 | 266.9 | 286.5 | | | | |
| base-r2 (compacted box) | 76.2 | 274.0 | | | | | |
| pin01-r2 | 70.5 | 266.3 | 291.1 | 158.4 | 46.4 | 4.8 | 37.6% |
| splitmodel | 66.0 | 247.1 | 274.7 | 133.4 | 52.2 | 6.0 | 54.0% |
| splitmodel-r2 | 72.1 | 263.0 | 301.5 | 141.4 | 55.7 | 6.0 | 51.6% |
| splitmodel-stream | 72.3 | 261.4 | 318.0 | 146.5 | 45.2 | 6.1 | 44.6% |
| **pin01-mem (OFF)** | **67.9** | **256.7** | **277.7** | 149.5 | 42.8 | 4.9 | 39.1% |
| **splitmodel-stream-mem (ON)** | **65.9** | **247.3** | **268.6** | 140.1 | 42.3 | 6.1 | 47.3% |

Runs: `cores-ab-20261014/` (pin01, splitmodel*, *-mem), `breakdown-20261015/{base,drip}`.

## 3. Landed and in-flight work, by topic

### 3.1 Whole layers 0 and 1 in the pinned tier (pin01) — MERGED

- Branch `pin-full-layers`, merged as `e02b8b9e0e`. Env `SGLANG_MOE_PINNED_HOST_FULL_LAYERS` (off by default).
- pin01-r2 70.5 vs base-r2 76.2 ms/token on the same box state; layers 0+1 went from 20.5 to 13.0 ms per verify;
  rows_read -16%.
- Merge tests: 4 failures pre-exist at `b5ced020b5` (stage_trace_causal clock audit, hotpath_shim lock check,
  sync_primitives device headers, row_source `gather_cached` stub). plugins_cuda split chunk passes under the GPU lock.

### 3.2 Split model: per-record CPU/copy split by a cost model — branch `split-model`, NOT merged

- Commits `cffc86e9ed` (feature), `80d4447834` (tests), `c8c006fe38` (stream-aware, `stream_lane=0.65`,
  `stream_base=0`), `ea78dfdc3c` (mlock + `-mem` arms), `66ba201557` (HiCache NUMA + Engram cache knob),
  `1ddfaae39e` (hicache arms).
- Env `SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL` (off), costs `SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS`. Rule: token-weighted
  greedy in `lease_device.cuh` `type_lanes`/`model_cpu_lanes`, mirrored in `ram_slot_map.py`.
- **Clean pair (both `-mem`): 65.9 vs 67.9 ms/token (-2.9%)**, median verify -9.4 ms. Wait -9.4 (replay predicted
  -10.2), stream -0.6 (predicted +0.3). One pair.
- Rule counters (ON): 5,238 records typed by the model; to_cpu [2529, 2143], to_copy [3769, 3617].
- **VRAM effect (offline, from counters):** hit rate unchanged (ON 0.516-0.529 vs OFF 0.510-0.519, layers 2+),
  unique non-VRAM experts per forward unchanged (507-525), lanes per verify unchanged (535-551); +15 (stream-aware)
  to +29 (first rule) VRAM inserts per forward, costing link time only.
- **Residency order:** the greedy rule inverts residency order on 25% of node-records (~40 pairs/verify), but the
  residency key barely predicts reuse (next-verify reuse 34.6% CPU vs 40.2% copy lanes under the table; 34.0 / 39.6
  under the rule). Residency-respecting variants: slack 2 = 17.0, slack 1 = 16.8 (21% inverted), slack 0 = 13.4 ms
  saving vs the table, against 16.9 for greedy. **Decision: keep greedy; no residency cost term.**
- **In flight — miss_key tiebreak** (owner approved 2026-10-15): among lanes with equal predicted finish, the copy
  engine takes the higher miss_key. Own env var, off. Cache-tiers replay: heaviest-first costs ~1 hit/verify,
  light-first or random 9-13 (~1 ms/token); the tiebreak guarantees the copy engine never gets the low-score lanes
  systematically. Offline re-score first, no serving unless asked.
  - **Built:** `ae1b27c189`, `SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK` (off; refused without the split model).
    The model's lane cost depends only on (node, tokens), so a tie is "same node, same token count" with no
    tolerance. Within each such group the CPU keeps its count but takes the lowest residency keys, so predicted
    finish times are unchanged by construction. Device `model_cpu_lanes` (cost words now 5N+6) plus Python mirror.
  - **Replay (breakdown base):** saving unchanged (+16.94 ms/verify untraced, as greedy); inverted node-records
    25.4% -> 13.9% (40.5 -> 17.9 pairs/verify); copy-lane next-verify reuse 0.396 -> 0.399 (table 0.402). Better
    than slack 1 (20.9%, +16.75). The remaining inversions are between lanes with different token counts.
    CPU tests 124 passed. GPU parity (`test/manual/dsv41/test_exl3_lease_kernels_cuda.py`, at 7e77a55ba2): 52
    passed; the mutant with the device tiebreak disabled fails 5 of 10 split-model tests, and reverting restores green.
    No serving run yet.
  - **Arm D** (`splitmodel-stream-mem-tiebreak`, worktree at `f9cf8b5cd0`, which also carries the draft
    drop-cache fix), vs arm A:

    | run | ms/token | median verify | mean verify | wait | stream | VRAM hit (layers 2+) |
    |---|---|---|---|---|---|---|
    | A | 65.9 | 254.5 | 272.5 | 139.9 | 42.9 | 0.512 |
    | D | 75.4 | 268.9 | 320.2 | 145.7 | 50.5 | 0.515 |

    - The VRAM hit rate didn't move.
    - **Not a cost-word bug** (checked offline):
      - `to_cpu`/`to_copy` count lanes where the model disagrees with the table, so the tiebreak lowers both. The
        net lanes moved to copy are unchanged (0.552 vs 0.565 per record).
      - D's logged cost words equal A's, plus the flags word.
      - The job-trace split matches A within 0.5% (CPU hit lanes 318.3 vs 320.0, DMA 116.2 vs 115.9).
    - What remains is the draft drop-cache fix or run noise. Arm A' (A at D's code, tiebreak off) is the new OFF
      sample.
    - **A'** (`splitmodel-stream-mem-hicache1-r2`, `ccb87893fa`, warm target load):

      | run | ms/token | median verify | mean verify | wait | stream | VRAM hit |
      |---|---|---|---|---|---|---|
      | A' | 66.8 | 241.8 | 269.7 | 141.4 | 41.3 | 0.514 |

      - A' reproduces A, so the draft drop-cache fix costs nothing.
      - D vs A': stream +9.2, wait +4.3, mean verify +50 ms. Lane mix, VRAM hit and decode memory are all the same.
      - Stream has swung this much between identical runs before (drip4 55 vs 42).
    - **Decision: tiebreak parked, off.** It is built and tested but showed no VRAM benefit. D's +9 ms stream is
      unexplained and one-sample; no repeat was run.
    - Open: A' read 7.35 GiB from sdb3 in an 11 s load window, which is too much for SATA. Which file was it?

### 3.3 Startup calibration of the split-model constants — branch `split-model-calib` (`7171570959`), NOT merged

- Owner requirement: "All the offsets and values should be calculated at server startup via a calibration routine."
- Extends `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION`: CPU grid {1,2,4} experts x {1,2,6} tokens fitting a/b/c per
  team; copy lane per node; stream coupling with 1/2/4 copy lanes; NVMe landing; live refinement (retune), clamps,
  logging, fallback to the table. Commits `7171570959`, `fe0daed0c4`. **Done and validated on the real model; no
  A/B yet.**
- Runs at copy-engine arming in the paused calibration window (+1.09 s, inside the first request).
  - **CPU:** per-team hit jobs {1,2,4} experts x {1,2,6} tokens and miss jobs, least-squares a/b/c.
  - **Copy lane:** DMA through a private backend.
  - **Landing:** O_DIRECT reads through the node's reader.
  - **Stream coupling:** a new `calibration_pull` kernel started behind 0/1/2/4 copy lanes.
- **Guardrails:** each constant is clamped to `RANGES_MS`. A non-finite or far-out value fails the calibration, and
  the model then stays off (device sentinel copy_lane < 0, so the table types the lanes). It never falls back to
  constants. `SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS` overrides named costs.
- **Live refinement:** `SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES` (0 = off) refits the CPU costs from seqlocked
  per-job moments, moving 25% of the way each time. Untested on a live server.
- **Calibrated vs offline (ms):**

  | constant | calibrated | offline |
  |---|---|---|
  | hit team0 a/b/c | 0.182 / 0.163 / 0.246 | 0.016 / 0.154 / 0.395 |
  | hit team1 a/b/c | -0.124 / 0.278 / 0.267 | -0.050 / 0.179 / 0.390 |
  | miss team0 a/c | 0.215 / 0.238 | 0.331 / 0.193 |
  | miss team1 a/c | 0.229 / 0.224 | 0.291 / 0.245 |
  | copy_lane | 0.968 | 1.05 |
  | landing | 0.264 + 1.574·i | 0.8 + 1.6·i |
  | stream | 0.135 + 0.906/lane | 0 + 0.65/lane |

  Totals agree at the served shape. first_byte is biased ~0.5 ms low, because the startup read misses the RAM
  thread's pickup.
- Tests: 111 + 80 CPU passed, 13 GPU passed.
- Next: an A/B with calibration + refinement against the fixed-cost split model; and the mirror-weight startup
  probe (`SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION`, approved), from read-amp's nvme3 finding.
- **Mirror-weight probe built** (`mirror-weight-calib` `f828356056`, not yet run).
  - Runs once per process in `exl3_mirror_config`, before the row tables are built.
  - Reads 4 O_DIRECT reads in flight per root, all roots at once, one reader-piece-sized sub-read each, 3 rounds
    of 0.15 s.
  - Weight = root throughput / mean, clamped to [0.5, 2]. Rounds disagreeing by >25%, a dead root or a missing
    manifest leave equal shares, with a warning. Explicit weights win.
  - Expected from read-amp's busy times: ~1.056 : 0.875 : 1.069 (nvme0 : nvme3 : nvme2).
  - **Validated 2026-10-15 19:52-20:01 (`split-calib-20261015/mirror1/`): runs, but over-corrects. Do not use.**
    - Probe: 0.47 s, noise check passed, no clamp.
    - Measured nvme0 3.48, nvme4 (the SPCC, read-amp's "nvme3") 2.04, nvme2 3.41 GB/s. That gives weights
      1.168 : 0.687 : 1.146 (1.7x).
    - Served busy time puts the SPCC only ~1.21x slower. These weights would leave it ~30% less busy than the
      Samsungs: the reverse imbalance.
    - Likely cause: short isolated QD4 random reads flatter the Samsungs more than bursty served demand does.
    - Fix needed: a probe that reproduces served demand (paced bursts of one row's sub-reads at the served queue
      depth), or weights taken from live busy time.
    - Same start: split-model calibration took 1.04 s and agreed with run1, apart from team1's hit intercept
      (0.159 vs -0.124).
  - **Reworked probe** (`mirror-weight-calib` `9e7ca5691a`, flag off, pushed; not yet run on the box):
    - A round is 40 bursts. Each burst puts one row's 8 sub-reads per root in flight, each root at its own share.
    - Each root is timed until its last sub-read lands. The probe then idles for as long as the burst took
      (`IDLE_RATIO=1.0`).
    - 3 rounds. A root's rate is its share divided by its median burst time.
    - Clamp, noise check and fallbacks are unchanged.
    - New fake-drive test (2x slow root): weights come out ~2:1, at most 8 sub-reads in flight per root, each root
      reads its own share, and bursts don't overlap.
    - `IDLE_RATIO` and queue depth 8 are assumptions. The validation start after the peak logger checks for ~1.21x.
    - **Validation #2** (2026-10-15 22:46-22:53, `mirror2`): 11 CPU tests passed.
      - Probe in 0.58 s: nvme0 2.80, nvme4 2.13, nvme2 2.67 GB/s, giving weights **1.1052 : 0.8416 : 1.0532**.
      - That is 1.31x, against a target of ~1.21x: much closer than the first probe's 1.7x.
      - **No stall during the probe** (22:50:05-06): read await 0.69 / 0.77 / 0.98 ms, so the SPCC was 1.34x the
        Samsungs, consistent with 1.31x. All roots passed the 25% round check.
      - **Adopted for a served A/B as is.** With a served truth of 1.21x, these weights leave the SPCC ~5% less busy
        than the Samsungs, against ~21% more busy with equal shares.
      - Caveat: a startup probe can't see the 20-60 s stalls. Static weights can't fix those.
      - Served A/B: `run_mirrorab.sh` (worktree `wt-mirror-ab`, `a988627f06`, the mergechunk-r2 code plus the probe
        and per-round logging), arm `...-mergechunk-mirrorcal`, against mergechunk-r2. Started first, ahead of the
        bigger-cache run.
- **Mirror weight A/B result (2026-10-15 23:14-23:24, `...-mergechunk-mirrorcal` vs mergechunk-r2): NOT adopted.**
  - serve_compare: **+0.04 ms/token (CI -0.72..+0.63)**, 8/8 prompts clean. Stream -1.44 and wait -1.10 ms/verify.
  - Per-root busy (nvme0 / SPCC / nvme2):

    | run | nvme0 | SPCC | nvme2 |
    |---|---|---|---|
    | r2, equal shares | 86.5 | 88.3 | 85.6 s |
    | mirrorcal | 93.3 | 74.9 | 89.4 s |

    - **With equal shares and no stall, the drives are already balanced.** The weights push the SPCC 17% under,
      and the busiest root gets busier.
  - So read-amp's 1.21x was stall episodes or the old box settings, not throughput. The probe sees 1.3x only in
    isolated bursts.
  - **Static mirror weights are dropped.** Kept as an opt-in diagnostic (`mirror-weight-calib` `cb77329dd8`,
    unmerged). The startup line is a cheap per-drive health readout.
  - Stalls need dynamic routing (`SGLANG_MOE_EXPERT_MIRROR_DYNAMIC`) or fixing the drive.
- **Refinement verdict (split-calib):** the `-refine` run published only 2 refinements, so it ended about 44% of the
  way to the served fit: a weak test. Even a fair one is unlikely to beat ~1 ms/token. **Not worth pursuing.**
  Startup calibration stays useful: it matches the offline fits within noise with no hard-coded constants.
- **A/B arms** (`f58bd94a44`):
  - `...-pin01-splitmodel-fixedcosts`: all old offline fits, calibration skipped.
  - `...-pin01-splitmodel-refine`: calibration + `REFINE_BATCHES=50`.
  - Plain `...-splitmodel`: calibrated, no refinement.
  - Rebased onto the `-mem-hicache1` base via a merge of split-model (`e55aeaae6d`): `...-hicache1-fixedcosts`
    and `...-hicache1-refine`. Plain `-hicache1` now calibrates on this branch.
  - The fixed-cost arm publishes the same device words as A', so only `-refine` runs, against A'.
  - CPU tests after the merge: 187 passed.
- **`-refine` run started 2026-10-15 22:09** (`cores-ab-20261014/run_refine.sh`, a copy of `run_a2.sh` with only
  the worktree `wt-split-calib-refine` at `e55aeaae6d`, the arm and the output name changed). Compared against A''.
  - GPU tests before launch: 13 passed, EXIT=0, from `test_split_model_calibration_cuda.py` and
    `test_exl3_lease_kernels_cuda.py` with `-k "split_model or calibration_pull or calibrate_fits"`.
  - Prelaunch eviction left node 1 with 76.64 GiB free, above the 64.21 floor.
  - It started ~1h45m late: the waiter was never queued after the mirror run.

### 3.4 Node-0 memory pressure — branch `split-model`

- **Symptom:** during load node 0 drops below its low watermark and swaps ~10-12 GiB; in decode the CPU expert team
  threads took major faults (11,877 on team0 in splitmodel-stream). Node 0 held ~85 GiB of the scheduler vs 52 on
  node 1; node 1 keeps ~15 GiB free. Only kswapd0 reclaims.
- **mlock (`SGLANG_ENABLE_SCHEDULER_MLOCK`, off):** `mlockall(MCL_CURRENT|MCL_FUTURE|MCL_ONFAULT)` in
  `run_scheduler_process` after `Scheduler(...)` (`sglang/srt/utils/memory_lock.py`; `Scheduler.__init__` untouched).
  Team faults fell to 20-250 per run; load 4 min instead of ~10; OFF itself gained 2.6 ms/token. It cannot stop the
  load-time swap (that happens before the lock). Locked and resident: 80-83 GiB node 0, 51 GiB node 1.
- **`--weight-loader-drop-cache-after-load`** in both `-mem` arms: works, but shard page cache was not the problem
  (fincore: 0 GiB of shards / row images at ready).
- **Cause found:** ~11.4-12.1 GB of node-0 Shmem is the **HiCache host KV pools** (`mmap_allocator.py`,
  MAP_SHARED|MAP_ANONYMOUS + MAP_POPULATE + cudaHostRegister, first touch on node 0). The launch has
  `--enable-hierarchical-cache`, `hicache_ratio 14`, `max_running_requests 1`. A separate ~9 GiB of page cache during
  the weight load is transient and gone by ready.
- **Owner decisions:** keep HiCache enabled; keep the KV pool at 381,184 tokens.
- **Built (`66ba201557`):** `SGLANG_HICACHE_HOST_NUMA_NODE` (unset = unchanged; mbind before populate via
  MADV_POPULATE_WRITE, logged per pool; move_pages test) and `SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB` (default 5 =
  the old hard-coded `5 << 30`; 0 = no cache, bit-identical rows; hit/miss logged per 60 s and at exit). 52 + 66 tests
  passed.
- **Measured (`cores-ab-20261014/run_hicache_ab.sh`), ON only vs splitmodel-stream-mem (65.9):**

  | run | ms/token | median verify | mean verify | wait | stream | tokens |
  |---|---|---|---|---|---|---|
  | A: ON-mem + HiCache on node 1 | 65.9 | 254.5 | 272.5 | 139.9 | 42.9 | 491 |
  | B: A + Engram native cache 0 | 69.2 | 249.7 | 270.6 | 147.0 | 42.4 | 509 |

  - **A fixes node-0 memory:** load swap-out 1,263 pages (~5 MB) against 3.0 M (11.6 GiB) for ON-mem; node 0 stayed
    above its low watermark (load min 35,422 pages vs low 34,934); team-thread major faults 0. HiCache Shmem moved to
    node 1 (numa_maps `/dev/zero (deleted) [bind:1]` 11.5 GiB). Node 1 is now the tighter node in decode (0.5-1.7 GiB
    free) but never neared its low watermark; kswapd1 idle. Scheduler per node 77.9 / 64.0 GiB (was 84.3 / 52.2).
    Throughput unchanged, as expected (headroom, not compute). Load took 6:19 vs 3:19-4:12 elsewhere; cause being
    checked.
  - **Engram cache:** first served hit rate. Arm A per 60 s window rises ~0.10 -> 0.29 through decode (run total
    0.087 incl. prefill warm-up; 831k accesses). B (cache 0) was +3.3 ms/token and wait +7.1 ms/verify, though median
    verify was -4.8 ms and B's sessions flipped to 509 tokens: one run cannot size it. **Decision: keep the Engram
    cache at 5 GiB.**
  - Found: `--weight-loader-drop-cache-after-load` left the DSpark draft's 7.2 GB shard cached (4-5.6 GiB at
    ready): the iterator's fadvise runs while a tensor from that shard is still mapped. Fixed in `7e77a55ba2`:
    `DefaultModelLoader` drops the shards again after `load_weights_and_postprocess`. fincore test reproduces the
    bug and the fix.
  - Arm A's 6:19 load was not HiCache (pools took 6 s, as elsewhere): target shard 1 of 40 took 2:21 with no memory
    pressure, most likely storage contention. The sampler now records /proc/diskstats.
  - C: A plus `SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY=0.995` (section 3.6): did not run with A/B; running next.
- **Box settings changed by the owner (2026-10-15 ~19:50 CDT):**
  - `kernel.numa_balancing=0`, persisted in `/etc/sysctl.d/90-dsv41.conf`.
  - `IRQBALANCE_BANNED_CPULIST=0-63` in `/etc/sysconfig/irqbalance`, so the eno1 IRQs 239-243 now sit on 64-71
    (they were on 14, 3, 9, 36, 5). NVMe per-CPU queue IRQs stay kernel-managed per core, as expected.
  - `compaction_proactiveness` is back at 20.
  - The owner's `swapoff -a` was OOM-killed mid-way by prefill-merge's run 3; swap stays active, ~3.1 GB used.
  - **Every run before this used the old settings.** A'' (A' rerun) is the new baseline and also measures the change.
  - **A''** (`splitmodel-stream-mem-hicache1-r3`, same code and env as A'):

    | run | ms/token | median verify | mean verify | wait | stream | tokens/verify |
    |---|---|---|---|---|---|---|
    | A'' | 70.9 | 249.2 | 273.4 | 137.8 | 46.7 | 3.52 |
    | A' | 66.8 | 241.8 | 269.7 | 141.4 | 41.3 | 3.60 |

    - No speed-up. Lower acceptance accounts for ~1.5 ms/token; the rest is stream +5.3 ms/verify against wait
      -3.6. VRAM, split and lanes are unchanged.
    - CPU expert ms/expert: team 0 is 5% slower, team 1 5% faster, so no consistent effect.
    - Decode swap-in fell from 13,563 to 757 pages and major faults from 3,528 to 459. That is real, but invisible
      in time.
  - **Measurement limit:** the stream lane moves 5-9 ms per verify between runs with identical split, VRAM and
    lanes (A vs A', A' vs D, A' vs A''). A single pair cannot resolve effects below ~5 ms/token.
    - **Owner decision (2026-10-15):** investigate the stream-lane variance. The interleaved in-server A/B harness
      is not wanted.
    - Agent `stream-variance` (branch `stream-variance`, outputs `ram-prefetch/stream-variance-20261015/`) starts
      offline:
      - decompose stream ms/verify into bytes x rate per run, prompt and window;
      - check the candidates: clocks/power, PCIe link, pinned-pool NUMA/THP, concurrent DMA, text divergence,
        hot-cache contents;
      - only if needed, a read-only GPU clock/PCIe sampler riding along on runs already queued.

      Goal: make one pair resolve ~2 ms/token.
    - **Interim result: the variance is the estimator plus SPCC read stalls. Clocks and PCIe are not involved.**
      Outputs: `ram-prefetch/stream-variance-20261015/`.
      - "Stream" is the NVMe read latency of GPU-miss rows. Per layer, stream / svc read time has a median of
        0.99-1.03. Clocks, PCIe, pinned-pool NUMA/THP and launch jitter don't show up in it.
      - **Estimator flaw:** step_budget/ab_compare skip 8 verifies, which leaves 8-14 warm-up/health verifies of
        75-125 ms each. A median over 8 prompts also has a CI of about ±7-8 ms/verify.
        - Serving-only stream means: A 47.2, A' 48.0, A'' 48.8. A' to A'' is +0.8, not +5.3 ms/verify.
        - ab_compare's ms/token numerator includes warm-up steps, but its denominator counts only serving tokens.
      - **SPCC stalls:** nvme3n1 = `/mnt/nvme4`, DRAM-less/HMB, 89% full.
        - Episodes last 20-60 s. Read await goes from ~7 ms to 25-61 ms, and throughput falls to 0.5-0.6 GB/s.
        - The Samsungs are capped at the same rate, because every row waits for its SPCC third.
        - Seen in A'', C, D and -refine; not in A', mergechunk or mergechunk-r2.
        - Shutdown root-1 `demand_busy` flags them: 99-127 s against 87.7-88.5 s clean.
        - A stall in serving decode makes that prompt's ms/stream-layer 1.5-3x (D prompt 1: 515 vs 233 ms/verify).
      - **"Identical" pairs re-scored**, dropping prompts with ms/stream-layer above 1.3x the run median:

        | pair | all prompts | stalled prompts dropped |
        |---|---|---|
        | base vs base-r2 | +6.5 | -0.2 |
        | A' vs D | +11.9 | -0.5 |
        | drip3 vs drip4 | +11.6 | +0.1 |
        | pin01 vs pin01-r2 | -5.4 | +2.7 |
        | A' vs A'' | +2.2 | +2.2 |

        - With stalls removed, identical pairs differ by SD ≈1.6 ms/token, max 2.8.
        - **The tiebreak (D) and drip4 losses were stall artifacts.**
        - The residual (±0.4 layers/verify from text and residency) is about ±2 ms/verify.
      - `-refine` vs A': ms/token +0.1. Stream is 7 ms/verify lower (fewer stream layers, 4.0 ms/layer), offset by
        wait.
      - **Root needed (owner):** `sudo nvme smart-log /dev/nvme3` before and after a run that stalls.
        - If the thermal counters rise, the drive is throttling.
        - If not, it is GC/SLC on a full DRAM-less drive, and the lever is fstrim or freeing space.
        - The only writer to that drive is Ray's `/mnt/nvme4/ray_tmp`, at about 2 writes/s.
      - **Final:**
        - **Tool:** `analysis/dsv41-drive/dspark/serve_compare.py RUN_A RUN_B` (`stream-variance` `2ba6d98f44`).
          - Takes serving prompts by session and reports per-prompt ms/token, stream, layers, ms/layer and wait.
          - Flags a STALL when a prompt's ms/stream-layer exceeds 1.3x the run's median, and prints root busy.
          - Reports the all-prompts and clean-both differences with a prompt-block bootstrap CI (typically ±2
            ms/token).
        - Every STALL-flagged serving prompt belongs to a stalled run: base-r2, splitmodel-r2, pin01, pin01-gpu01,
          D, drip4 and confgate. No clean run has one.
        - **A second excursion type, on the CPU side:**
          - In A'', prompt 5 alone makes the +2.2 ms/token over A' (89 vs 69 ms/token). Its stream is normal, but
            both CPU teams were 25-43% slower per lane.
          - It coincides with a node-0 swap-in burst at 20:15:30 (pswpin 137, 95 major faults in 5 s). Team
            threads took no faults.
          - Without that prompt, A' vs A'' is about -1.
        - **`-refine` vs A':** ms/token +0.1 (CI -1.7..+1.9). Stream -7.1 ms/verify (0.9 fewer stream layers,
          4.02 ms/layer), wait +4.4. Its stall fell outside serving. **Refinement: no measurable gain.**
        - **New verdict rule:**
          - (a) Use serve_compare means, paired by prompt, quoting the clean-both difference with its CI. No more
            skip-8 medians.
          - (b) Stall gate: if root-1 busy is more than 5% above the Samsungs', drop STALL-flagged prompts from both
            arms. If more than 2 are flagged, rerun the arm.
          - (d) A per-prompt CPU ms/lane flag (requested).
        - **Drive fix (owner):** run smart-log first. If it is GC, fstrim and free space; if thermal, add airflow.
          For production, dynamic or busy-weighted mirror shares could route around a stalled root (pointer only,
          not built).
        - **serve_compare MERGED** into `ram-prefetch-pool-serve` (`91cf7c7c02`, tool `79d8147098`).
          - It adds a CPUSLOW flag: either team's ms/lane above 1.2x the run's median prompt, corroborated by the
            swap-in/major-fault rate and node-0 MemFree.
          - It replaces ab_compare and step_budget medians for verdicts. ab_compare is an untracked run-dir script,
            now marked deprecated.
        - **Re-scored pairs** (first minus second, ms/token; clean = no flag in either arm; 95% prompt-block CI):

          | pair | all | clean (n) | clean CI | stream/v | wait/v | flags |
          |---|---|---|---|---|---|---|
          | A - A'' | -0.80 | +1.72 (7) | -0.7..+4.7 | -2.2 | +6.9 | A'' p5 CPUSLOW (swap x22) |
          | refine - A'' | -2.11 | +0.43 (7) | -2.6..+3.3 | -8.7 | +8.8 | A'' p5 |
          | mergechunk-r2 - A'' | -2.82 | +0.27 (7) | -2.0..+3.6 | +1.2 | -1.7 | A'' p5 |
          | **mergechunk-r2 - A** | -2.02 | **-2.02 (8)** | **-3.7..-0.6** | +4.1 | -9.2 | none |
          | D - A' | +11.89 | -0.47 (7) | -1.9..+2.2 | +0.5 | +1.1 | D p1 STALL |
          | C - A | +0.49 | -0.56 (6) | -1.4..+0.1 | +2.8 | -0.8 | C p0, p5 CPUSLOW |
          | drip3 - base | -0.54 | -0.54 (8) | -2.2..+1.8 | -0.2 | -2.1 | none |
          | drip4 - base | +11.09 | -0.64 (7) | -2.1..+0.7 | +1.2 | -4.4 | drip4 p7 STALL |

          - On clean prompts every pair is within ±2 ms/token.
          - **Only merge-chunk at 11721 MiB vs A clears zero: -2.0 ms/token (CI -3.7..-0.6).** Wait falls by 9.2
            ms/verify and stream rises by 4.1. A real but small gain.
          - Refine moves 8.7 ms/verify from stream to wait, with no net change.
          - Tiebreak (D), decay 0.995 (C) and drip3/drip4 are all neutral, not losses.
        - **SPCC production tax:**
          - Frequency: 5 episodes in 0.59 h of drive-active time, about 8.5 per hour and 20-51 s each (~6% of
            active time). 6.6% of traced serving decode fell in a stall.
          - Cost: a hit prompt runs at 1.94x expected ms/token (1.14-2.98x).
          - **Tax: ~3.2% of decode time, about 1.9 ms/token on average.**
          - Caveats: small sample, and the test runs are prefill-heavy. The cause awaits the owner's smart-log.
        - CPUSLOW causes: memory pressure fits A'' p5 (node 0 at 918 MiB), but C's two flags have only weak swap
          corroboration, so the cause is open.
  - Resolved: the ~7 GiB read from sdb3 at launch is the DSpark draft, read cold each launch because it is evicted
    and dropped (~20 s at ~470 MB/s). The target load is warm.
- **In-server allocator profiling is stopped:** the snapshot dump grows the node-0-bound heap and OOM-kills.
  The 3120 MiB attribution uses a gated peak logger instead: `SGLANG_DEBUG_PREFILL_PEAK_LOG`, default off,
  `debug_utils/prefill_peak_log.py`, `187cf1b589`..`7951fc18eb`.
  - It wraps the DSpark prefill root, the target forward, each layer's attention/MLP, logits, the sampler, the
    draft-KV injection and the draft forward. Per call it logs allocated and reserved before, at peak and after.
  - No history, no stacks, no syncs.
  - 5 CPU tests passed, and a mutant is caught. Its run goes last in the queue.

### 3.5 VRAM hot cache review — branch `cache-tiers` (`4d81ad411f`, `a4c674a2b9`)

Offline replay (`analysis/dsv41-drive/dspark/vram_cache_replay.py`, `lane_cost_fit.py`) of `split-ab-20261014/C2`
(806 verifies, 853 slots) and `breakdown-20261015/base`. Outputs `cache-tiers-20261015/`. Simulated today-rule 43.7%
vs real 41.1%: read deltas, not absolutes.

- **Today:** 878.7 unique routed experts/verify; 361 VRAM hits (41.1% unique, 54.4% token-weighted); 112 inserts;
  45% of inserts evicted before reuse. LRU stack distance median 32 > 22 slots/layer: **capacity-starved**.
- **Lane cost (OLS, r2 0.81):** CPU RAM hit 0.29 ms, NVMe-read lane ~1.1 ms, DMA ~0, SM copy 0.88; mix 0.37 ms per
  non-VRAM lane. The CPU side is the critical path; the copy engine is slack.
- **Where VRAM goes:** target weights 9.05 GB, draft 2.10, hot cache 10.6 GiB, KV + pools ~0.73 GB for ~345k tokens,
  graphs 0.36, **5.1-5.2 GB free after init = prefill headroom** for the triton fp32 merge.
- **Ranked:**
  1. Capacity (section 3.7): 853 -> 951 slots ~1.9 ms/token; each further +40 slots ~0.8 ms/token; ~15000 MiB ~6
     ms/token. Uniform per layer (non-uniform: +1-3 hits only).
  2. Next-verify route predictor (section 3.8): +68 hits/verify (~7 ms/token) using only today's GPU-lane count, +114
     if it also drives inserts. Largest prize.
  3. Split model: greedy + miss_key tiebreak (section 3.2).
  4. Decay 0.98 -> 0.995: +5.5 hits/verify (C2), +3.6 (base), ~0.6 ms/token.
- **Not recommended:** inserting CPU-computed experts into VRAM (score-all +36 hits for +292 inserts = +3.9 GB/verify
  H2D; budgeted/conditional/lazy variants 0-3 hits or worse); per-layer slot allocation; acceptance-aware bypass
  (rejected-only routes recur 42% vs 26% for accepted); cutting layer 0/1 slots under pin01; changing the RAM tier's
  protection of VRAM-hot rows (11.3 GB; evicted VRAM experts recur within 8 verifies 58%).
- **GPU drip explained:** drips land in scratch, not residency, so a used drip misses again next verify (measured
  neutral, 69.9 vs 69.3). If kept: land drips in a victim slot and commit residency.

### 3.6 Hot-cache decay 0.995 — arm C: no detectable gain

Existing env `SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY` (0.98 today). Replay predicted +0.5-1 point of routed-pair VRAM
hit rate. Arm C (`splitmodel-stream-mem-decay995`) measured 0.514 vs A 0.512: +0.2 points, inside the spread between
identical configs (0.512-0.516). Insertions per forward were 140.9 vs 141.5. Timing (67.6 ms/token, mean verify 315)
is confounded: the previous server left 23.2 GiB of file cache on node 1, both nodes went below low during load,
kswapd1 ran, and 1.8 GiB was swapped out in decode. Team threads still took 0 major faults. **Not adopted; leave at
0.98.**

**Lesson for every run:** evict leftover file cache before launch, without root (posix_fadvise DONTNEED over the
model, draft, row-image and Engram files). Log per-node free memory, and wait if node 1 lacks room for the HiCache
pools. Committed as `analysis/dsv41-drive/dspark/prelaunch_evict.py`, with `memsample.py` (numa_maps, diskstats) and
`memsum.py`, in `f5fb51875b` on `split-model`. The node-1 floor = HiCache pools + "rest" + margin. "Rest" comes from
the arm's previous memsample (`--previous`) or falls back to arm A's 51.24 GiB, and the source used is logged.
Eviction drops the cache of shared model files for everyone, so it runs only when no other server is up. The sampler now records /proc/diskstats. The model shards are on sdb3 (SATA root volume), which fits arm A's
first-shard stall.

**Cold target load:** the target's ~9 GiB of non-expert weights come from sdb3 (SATA) at ~55 MB/s when not
cached, about +150 s per launch. Arms A and D both paid it. dsv41-full40 is now left out of the eviction list.

**Read amplification: none (agent `read-amp`).** Arm D's NVMe mirrors read ~895 GiB against 209 GiB in
`rows_read` + `spec_landed`. The reader's own per-root byte counters ("drive load" at stop) match diskstats to within
0.1 GiB on nvme0/nvme3; nvme2's +15.8 GiB is weight load and Engram.
- **The counters only cover decode.** With `SGLANG_DSV41_ENABLE_PREFILL_FILLS=1`, prefill's prefetched rows are
  counted as pinned hits; `pinned_admissions` sees them.
- **The "262 s of decode" window was mostly prefill.** The run had only 50.8 s of verify time.
- Breakdown:

  | consumer | GiB | share |
  |---|---|---|
  | prefill (11 warm-up + 3 health + 8 serving requests; 48-68 GiB, ~4,500 rows, ~113 rows/layer per request) | ~640 | 72% |
  | decode demand | 134.6 | 15% |
  | spec prefetch (incl. 6.4 abandoned) | 80.9 | 9% |
  | startup fills | 23.5 | 3% |
  | weight load + Engram | ~16 | 2% |

- The row split works: each row is read once, a third from each mirror.
- **Avoidable:** only the spec rows that landed unused, 45-54 GiB (5-6%). They are gated to idle drives (1.3-3.1 s
  overlap with demand per root), so removing them would save well under 1% of verify time.
- **Prefill churn:** 46,900 eager evictions across 8,782 RAM slots (prefill share 0). Decode misses per verify are
  flat by position after a prefill (pos 0 67.6, pos 1 77.6, pos 10+ 82.9). The first verify is slower (0.60 vs ~0.36 s)
  but not from reads, so it is likely a one-time first-verify cost. Not pursued.
- **Counter fix built** (branch `read-amp` on split-model f9cf8b5cd0: `3965123c2c`, `ca965e7c06`; not merged).
  Observability only, and every existing field keeps its name, offset and meaning.
  - `host_fill_rows` in the gather stats and metrics: a prefill's real reads = host_read_rows + host_fill_rows.
  - Per-root `fill_bytes` (part of demand_bytes), appended after the original 47 snapshot words and as a new key in
    the shutdown line.
  - `eager_evictions`, appended to the enum; evictions minus eager_evictions = the service's count.
  - Tests: 295 passed. 1 failure pre-existed at the base (a stale `_gather_cached` stub) and is fixed in
    `ca965e7c06`. Two mutants are caught.
- **Mirror lead (sent to split-calib):** nvme3 (SPCC, max_sectors_kb 256) was busy 104.5 s vs 86.6 / 85.6 s on the
  two Samsung 990 EVO Plus drives at equal bytes. Busy-time-balanced mirror weights (`SGLANG_MOE_EXPERT_MIRROR_WEIGHTS`)
  could cut demand-read time by up to ~10% (~0.1 ms per NVMe lane), ideally derived by a startup throughput probe.

### 3.7 Prefill VRAM spike / hot-cache capacity — agent `prefill-merge`, branch `prefill-merge-chunk` (not pushed yet)

- `benchmarks/dsv41_baseline/arm_env.py`: the triton FlashMLA prefill merge runs in fp32 and costs ~0.9 GiB at a 2048
  chunk and 1.75 GiB at 4096 over flashinfer, so DSpark's prefill peak sits 4,808 MiB above init. The DSpark arms
  therefore run `DSPARK_MEM_FRACTION_STATIC=0.78`, `DSPARK_HOT_GPU_MB=10840` (12040 before) — the "1.2 GB".
  Owner TODO option C: chunk the merge over tokens, bit-identical.
- **Cause confirmed** (H=64, D=512, TP1): `flash_mla_sparse_decode_triton` holds `out` and `out_extra` (bf16, 256
  MiB each at 4096 tokens). `_merge_partial_attn` then holds 3 fp32 copies at its peak (1536 MiB): 2048 MiB in total
  vs flashinfer's 256, i.e. +1.75 GiB at 4096. `_apply_attn_sink` has the same peak on its own, and layers 0 and 1
  have only the sink. Layer-major prefill doesn't apply to DSpark (`SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0`).
- **Built** (branch `prefill-merge-chunk` on `split-model` 1ddfaae39e: `618ef2682e`, `22beb490ed`, `7a2dd5cc4b`,
  `b475901dfd`): `SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB` (0 = off). Tokens per chunk = budget //
  (3·H·D·4), so 128 MiB gives 341 tokens. Merge and sink write each token chunk in place, and decode is untouched.
  GPU test: torch.equal on out and lse through the real kernel at 4096 / 2065 / 171 tokens, budgets 64 MiB and 1
  MiB; 50 passed. The mutant that drops the last partial chunk is caught (9 failed), then reverted to green.
  `test_flash_mla_backends.py` has 2 pre-existing failures (identical at base).
- **Peaks** (startup warm-up sets the peak; a 30k prompt adds <= 2 MiB):
  - budget 0: 31885 of 32607 MiB, 4337 above init.
  - budget 128: 30679, 3120 above init (frees 1217 MiB).
  - `-mergechunk` arm (hot 12040 = 948 slots, KV pinned 381,184): 31977 peak, 630 MiB free, no OOM.
- **Launch:** `SGLANG_MOE_HOT_GPU_MB=12040`, budget 128, `--max-total-tokens 381184`, fraction 0.8256 computed by
  `fraction_keeping_kv()` with a 0.12 GiB launch-jitter margin. Unpinned KV floats between launches (339,712 in
  the OFF sample, 377k-403k in the probes), hence the pin.
- **Next:**
  - ON arm on top of the HiCache-on-node-1 arm, against arm A (65.9). Runs after arm C.
  - Then find the remaining 3120 MiB above init, the path from 12040 toward ~15000 MiB.
  - Watch item: HiCache host pools scale with the device KV, so they grow ~12% on node 1.
- **Hot-cache size from measured headroom** (`hot_mb_for_margin`, `2ece3a6d1d`): 630 MiB free at peak with 948
  slots, minus 10 MiB for the worst free-before-load seen, leaves 111.8 MiB over the 508 MiB margin, i.e. 8 slots.
  So **12140 MiB (956 slots)**, fraction 0.8290, KV pinned 381,184. This is the ON arm queued after arm C
  (`run_mergechunk.sh`).
- **Why not ~15000 yet:** the 3120 MiB that remains above init comes from the startup warm-up shapes (1 ... 4096
  tokens). First cut from the nvidia-smi timeline:
  - ~810 MiB: persistent MoE expert staging (w13 540 + w2 270 MiB, 64 rows), allocated on the first prefill.
  - ~780 MiB: the 257- and 400-token prefills plus copy-engine arming.
  - ~1.5 GB: the 2048, 2049 and 4096 prefills.
  An allocator-history probe (`1c258a85ea`) will attribute it exactly after the A/B.
- **Allocator-probe attempts:**
  - Run 1, on the `-mergechunk` arm with mlockall: the allocator-snapshot dump got the scheduler host-OOM-killed
    (MEMORY_POLICY nodemask=0, anon-rss 127 GiB).
  - **Finding: under mlockall the `-mem` arms leave node 0 with only ~0.14 GiB free**, so any extra node-0
    allocation risks an OOM kill.
  - Run 2: the pinned tier was cut to 20 GiB, and startup refused it (layer 0 needs 66 node-0 slots). The probe then
    held both locks for 18 min; fixed in `4bada59bd0`.
  - Run 3 (dspark-both, default tier, no mlockall, one 4096-token prompt) is queued after A', with a node-0 free check
    before the dump.
  - Offline so far: the torch prefill indexer is already capped at 128 MB (~384 MiB transient). The 810 MiB of MoE
    staging is persistent.
- **ON run at 12140 MiB / 956 slots** (`splitmodel-stream-mem-hicache1-mergechunk`, vs arm A):
  - Result: 66.4 vs 65.9 ms/token, step median 241.1 vs 251.3, tokens per verify 3.46 vs 3.56. No gain detectable on
    one sample.
  - Confounded by swapping: 2.05 GiB swapped out in decode (A: ~0). The run started with 26 GiB of buff/cache (A: 16),
    and node 1's low watermark was boosted (62211 vs 36099 pages; watermark_boost_factor 15000; min_free_kbytes 90112
    is the THP-recommended value). It ran before the pre-launch eviction step existed.
  - HiCache pools grew only 0.29 GB with the KV pin (A ran at KV 367,872). Unpinned, this fraction would have given KV
    460,288.
- **Headroom correction:** the probes measured total − used. On this card used + free = 32,203 MiB (the driver
  keeps 404 MiB), and arm_env's 508 MiB margin is nvidia-smi memory.free. On that basis:

  | run | memory.free at the warm-up peak (MiB) |
  |---|---|
  | fix off | 318 (already below 508) |
  | chunked | 1524 |
  | 948 slots | 226 |
  | 956 slots | 134, no OOM |

  Re-derived (`918b3560fe`): 923 slots = **11721 MiB**, fraction 0.8150, KV 381,184. That is +70 slots over 10840,
  not +103.
- **11721 MiB run** (`splitmodel-stream-mem-hicache1-mergechunk-r2`, 2026-10-15 20:17-20:24, one sample):

  | run | ms/token | tok/verify | median step | fwd wall | gate wait | misses/fwd | covered |
  |---|---|---|---|---|---|---|---|
  | A | 65.9 | 3.56 | 251.3 | 243.7 | 191.5 | 77.8 | 16.5 |
  | A'' | 70.9 | 3.52 | 248.4 | 240.3 | 187.7 | 74.6 | 15.4 |
  | mergechunk-r2 | **63.9** | 3.56 | 235.0 | 229.0 | 177.0 | 76.8 | 16.5 |

  - **Safety: clean.**
    - GPU: peak 31,605 MiB used, 598 MiB free (margin 508), no OOM, 923 slots, KV 381,184.
    - Node 0: low point 0.72 GiB in decode, 0.24 GiB at load (just above the boosted watermark).
    - Swap: out 7.5 MiB at load, 75 pages in decode. Swap used stayed flat at about 2.9 GiB carried over from before
      the run.
  - **Speed: unresolved.**
    - Faster than A'' by 7.0 ms/token, with forward and gate wait each about 11 ms lower.
    - But misses/forward did not fall (76.8 vs 74.6), so the extra slots don't explain it.
    - Within the 65.9-70.9 spread between identical runs.
    - Treat as a safety pass, not a speed result.
  - Text diverges only at the known-unstable points (ETR, DISCA); the merge itself is bit-tested.
  - **Re-scored with serve_compare: -2.0 ms/token vs A (CI -3.7..-0.6), the only clean gain so far.**
- **Peak logger run** (2026-10-15 22:35-22:44, `-mergechunk` arm at 11721 MiB, `wt-prefill-merge-dev` at
  `7951fc18eb`; report `prefill_peak_report.py` at `3c1fc23778`; artifacts `/mnt/nvme1/prefill-merge/peaklog/`):
  - **Most of the warm-up spike is caching-allocator reserve, not live tensors.**
    - Reserved grows 24,766 -> 28,242 MiB across the warm-up shapes. The growth per shape: 400 tokens +136, 2048
      +784, 2049 +260 (pure fragmentation: the allocated peak is unchanged), 4096 +1,480. Later prompts add 0.
    - The allocated peak is 26,363, at the 4096-token prefill.
    - Of the ~3.1-3.5 GB above init:
      - 813 MiB is persistent MoE staging;
      - ~1,459 MiB is the live prefill transient;
      - **~1,879 MiB is reserve backing no live tensor.** Up to ~140 more hot slots if recovered (estimate).
  - Live transient at 4096: attention 903 MiB per compressed layer (600 on L0/L1) and MoE 431. About 556 MiB
    stays live across the layer loop (residual/mHC streams and aux capture).
  - GPU: peak 31,721 used, **482 MiB free, below the 508 margin.** Free-before-load was 29.33 GiB (the minimum
    seen). No OOM, but `hot_mb_for_margin`'s 0.04 GiB jitter allowance is too small.
  - SPCC: no stall in the window. Read await had a median of 8.6 ms and a max of 13.4.
  - **Next (approved):**
    - A smoke run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the logger on, queued after
      split-calib. A compatibility check comes first: graphs, copy engine, IPC, mapped host memory.
    - Offline: largest-shape-first warm-up or `empty_cache` between shapes, behind a default-off flag.
    - A proposal for deriving the margin jitter from startup measurements, not a constant.
  - **expandable_segments audit (torch 2.13.0+cu130, TP1): no blocker found.**
    - Graphs capture into torch's private pool.
    - Graph-pool borrow is the one risky path (it reads segment layout), and it is off
      (`SGLANG_ENABLE_GRAPH_POOL_BORROW=False`).
    - There is no custom allocator or MemPool in this config.
    - CUDA IPC is only in TP>1 all-reduce.
    - Host registration is unaffected, and the copy engine uses stable tensor pointers.
    - Smoke queued: `cores-ab-20261014/run_expseg.sh`, after split-calib.
  - **Option 2 built, off:** `SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL` (`b9811b4d98`,
    `mem_cache/prefill_cache_release.py`).
    - It calls `empty_cache()` before each new high-water prefill size: 11 times across the warm-up, never in
      serving.
    - empty_cache was chosen over reordering the warm-up, because a reorder changes route and residency state.
    - 74 CPU tests passed. The `>` -> `>=` mutant is caught.
  - **Margin derivation:**
    - Growth from init to the warm-up peak is stable per arm: 3,522-3,573 MiB across 3 launches.
    - The jitter is in init headroom: 164 MiB between two 923-slot launches. Free-before-load explains only part
      of it.
    - Design (a), trimming after the warm-up, **is not possible.**
      - Startup order: host tier -> RAM-miss service -> hot cache -> KV pinned -> both graphs captured ->
        warm-up -> copy engine armed.
      - Capture bakes in the hot-cache tensor addresses, the scratch-row positions, the residency tables and the
        RAM-miss lease tables.
      - So a trim that frees memory would need a recapture; masking slots frees nothing.
    - **Owner decision: (a′), calibrate before graph capture on every launch,** behind a default-off flag.
      - After KV pinning, call empty_cache, then run one dummy extend at the largest chunk (derived from config).
      - Measure the lowest free memory and the reserved peak, then size the hot cache and the tables depending on
        it before capture. The margin comes from measured values.
      - The forward must leave no route, residency, KV or Engram state behind.
      - Step 1 is a map of the capacity-dependent init and a reorder proposal, before any code. `model_runner.py`
        stays a hook call at most.
      - The expandable_segments smoke still runs first.
    - **Map (step 1) found a blocker for rebuilding:**
      - The native copy engine's `set_table` is set-once per row (`copy_engine.h:368`), and it holds hot-cache
        addresses from `from_model`.
      - There is no teardown for the manager, updater or recorder, and RAM-miss service shutdown is one-way.
      - Hook point: `scheduler.init_model_worker`, between attention-backend init and `init_all_cuda_graphs`.
    - **Decision (team-lead): design (C), VMM tail-release**, flag `SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP`
      (off).
      - Hot-cache tensors are allocated at an upper-bound capacity on CUDA VMM (2 MiB pages).
      - At the hook, a calibration EXTEND at `--chunked-prefill-size` runs twice through the graph runner's
        `_dummy_run`. The second run measures fragmentation in-launch.
      - Then tail slots are retired: evicted, marked retired, capacity lowered everywhere through one authority,
        pages unmapped. All before capture and copy-engine arming.
      - Base addresses stay valid, so the copy table, segment tables and pointer tables don't change.
      - The margin is checked again after capture and after the warm-up peak (refuse healthy if short).
      - No-state guards: no route recording, recorder, inserts, pinned-tier admissions or Engram cache.
      - Rejected: (A) deferred construction (it starts the RAM-miss service before plan_graph_gather), and a full
        rebuild (it needs a native set_table replace and teardown).
  - **expandable_segments smoke** (`splitmodel-stream-mem-hicache1-mergechunk-expseg`, 2026-10-15 22:55-23:03),
    read from the log:
    - **Peak used 30,163 MiB, 2,040 MiB free** (482-598 before).
    - Reserved peak 26,730 (28,242 before); allocated peak 26,292.
    - Graphs captured, 0 OOM lines.
    - **That recovers ~1.5 GB, about 115 slots.** The reserve gap at the peak falls from 1,879 to 438 MiB. The
      2049-token prefill now adds 0 (was +260) and the 4096-token prefill +822 (was +1,480). Init headroom is 4.59
      GB, against 3.91-4.07.
    - **Speed (serve_compare vs mergechunk-r2):**
      - Clean prompts (6/8): **-0.26 ms/token (CI -0.39..-0.02), so no cost.**
      - All prompts: +17.1. Prompts 0 and 1 were hit by an SPCC stall (22:59:33-23:01:00, await 30-62 ms; root 1
        busy 135.8 s against ~86 s).
    - **Node-0 pressure during load, worse than r2:**
      - Zone free fell to 0.04 GiB, below the low watermark. 410 MiB was swapped out and kswapd0 ran 132 ticks.
      - The teams took 10 major faults each, against 0.
      - Scheduler VmLck is 715 GiB against 389: expandable segments' VA reservations count as locked VMAs under
        mlockall ONFAULT. That should not use RAM in itself, but it is not yet isolated from launch-to-launch
        variation.
  - **expseg2: expandable_segments plus 1,035 slots (13,144 MiB, +112)** (2026-10-15 23:29-23:36, code
    `23d8931dcd`):
    - **serve_compare vs mergechunk-r2: -1.79 ms/token (CI -4.51..-0.64), with all 8 prompts clean.**
      - Wall -6.5, wait -8.6 and stream +2.4 ms/verify.
      - Prompt 4 carries much of the mean (-14.8). The other prompts run +0.07..-2.07.
      - **Against A this stacks to about -3.8 ms/token** (mergechunk-r2 was -2.0 vs A).
    - **VRAM verify hit rate 41.71% -> 43.61% (+1.9 pp).** Miss rows -3.0%, misses/forward 76.8 -> 73.2, NVMe left
      60.4 -> 57.7. Prefill hit rate 11.8% -> 13.2%.
    - GPU:
      - 706 MiB free at the warm-up peak; the prediction held after scaling for free-before-load (29.47 GiB).
      - Graphs captured, 0 OOM, 0 kernel OOM kills.
      - Allocator reserve gap: 439 MiB.
    - **Node 0 during load:** a minimum of 0.14 GiB, above the min watermark (r2 0.24, expseg 0.04), and 14 MiB
      swapped out. So expseg's deep dip was launch variance, not the allocator.
      - VmPTE tracks RSS (236-290 MiB).
      - The 715 GiB VmLck is unpopulated VA reservations: no RAM or page-table cost.
      - The -mem arms leave node 0 little room during load whatever the allocator.
    - **Candidate for the owner:** the `-mergechunk-expseg` arm at 1,035 slots (merge budget plus
      `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Defaults are unchanged; that is the owner's call.
  - **(C) step 1 pushed** (`bf68f4ca52`, flag off): `layers/moe/hot_cache_vmm.py` `VmmHotCacheBacking`, with one VA
    reservation per tensor in 2 MiB granules. `release_tail(keep_rows)` unmaps whole granules past the kept rows.
    - 23 CPU tests passed.
    - GPU test `test/manual/dsv41/test_hot_cache_vmm_gpu.py`: **2 passed, EXIT=0** (10 s). Kept addresses and
      contents hold, free memory rises by the released bytes, and it coexists with expandable_segments (0 alloc
      retries).
    - **Headroom left at expseg2:** 15 slots at that launch, but only 4 at the lowest free-before-load seen (the
      static jitter allowance costs 143 MiB). So (C) is worth about 4-15 slots per launch.
    - **The bigger lever is the 1,459 MiB live prefill transient (~115 slots):** attention 903 MiB plus 556 MiB held
      across layers. A scoping note on bit-identical cuts comes after step 2.
    - **Step 2 pushed** (`dff9c6fe42`, `layers/moe/hot_cache_retire.py` `retire_hot_slots(manager, keep)`): the only
      place capacity drops.
      - It evicts tail experts, lowers every holder, re-pushes the host hot sets, re-ranks victims and releases the
        VMM tail.
      - It refuses before any change if any of these hold:
        - the updater is enabled, meaning capture has happened;
        - a capture is in progress;
        - the copy engine is armed;
        - the cache is not VMM-backed, or not DIRECT;
        - keep is below twice the victim lanes;
        - a capacity attribute is not in the census.
      - 8 CPU tests. The census test parses the real classes' source for `*capacity*` attributes. The mutant that
        skips `policy.capacity` is caught.
      - Not yet tested on a real manager; that comes with the step-3 GPU smoke.
    - **Step 3 findings and decisions:**
      - `_dummy_run` forces TARGET_VERIFY on a speculative runner, and its graph runners don't exist until capture.
        So the calibration module builds its own one-sequence EXTEND ForwardBatch, mirroring `_dummy_run`'s extend
        branch, with no base_runner change.
      - The prefill MoE admits every RAM-tier miss into the pinned tier; there is no staging-only path.
      - **Decision: route calibration tokens round-robin to experts already resident in the RAM tier,** with the
        real route count and about the real distinct-expert count. No admissions, no NVMe reads.
      - The approximation is checked: the post-capture check logs the calibration peak beside the real warm-up
        4096-token peak, and flags a gap larger than the fragmentation term.
      - The Engram and route-recording skips are calibration-context checks.
    - **Step 3 first cut pushed** (`1551a883ba`, `5ac2200cea`, flag off):
      - `layers/moe/hot_cache_calibration.py` `maybe_calibrate_hot_cache`, returning a frozen `CalibrationResult`;
        `hot_cache_dummy_extend.py` builds the EXTEND batch.
      - Two measured runs. lowest free = F0 - (P1 - R0) - non-torch growth; margin = max(P1 - A1, P2 - R1).
        `plan_keep` retires from the fullest layer first, never below 2x the victim lanes.
      - Scheduler hooks: calibration before `init_all_cuda_graphs`; `require_hot_cache_headroom` after capture
        refuses to start if free memory is below the margin. `model_runner.py` is untouched.
      - 109 CPU tests passed.
      - Decided:
        - post-warm-up check is log-only, with the calibration peak shown beside the real warm-up peak;
        - capture memory goes into the pre-capture sizing, from an eager TARGET_VERIFY dummy peak.
      - Pending: wiring routing (i) and the capture estimate, stating the upper-bound capacity derivation, then a
        ~8 min GPU smoke.
  - **Step-3 calibration smoke** (2026-10-15 23:55 - 10-16 00:03, `c439ee2326`, expseg arm at 1,035 slots, flag on):
    - **The dummy EXTEND runs on the speculative target.** 2 calibration runs, no traceback, 0 OOM.
    - Measured: lowest free 1,017.9 MiB; margin 331 (the allocator gap of run 1; run 2 grew 0); non-torch growth 88.
      Retire 0.
    - **Routing (i) against the real warm-up 4096 prefill:** allocated +46 MiB, reserved +152 MiB, below the margin.
      The approximation holds.
    - **Bug A:** the post-capture check read 2,172 MiB free because capture released the calibration reserve. Fix:
      project from the reserved peak.
    - **Bug B:** the VMM backing rounds each of the 6 tensors x 40 layers to 2 MiB, costing ~140 MiB.
      - So the KV pin missed: 372,480 tokens instead of 381,184.
      - Free memory at the warm-up peak was 414 MiB, against 706 without the flag.
      - Fix: VMM only for the large tensors, and a hard refusal if KV is below the pin.
    - **Approved next:**
      1. Fix B.
      2. Fix A.
      3. The capture estimate (TARGET_VERIFY dummy).
      4. **Grow:** reserve VA for the derived upper bound (pinned-tier rows minus max_gather_rows), map physical memory
         for the budget, then let calibration trim or grow through one authority.
         - This launch had room for ~30-40 more slots after allowing for capture and the warm-up delta.
  - **Prefill transient cuts (scoping note; estimates at 4096 tokens):**

    | id | cut | saves | bit-identical | code |
    |---|---|---|---|---|
    | A1 | indexer score budget 128 -> 32 MB | ~290 MiB if it is in the peak window | yes (§27.7) | env |
    | A3 | merge budget 128 -> 32 MB | ~96 MiB | yes | env |
    | A2 | per-chunk SWA/extra kernels into one output | ~240 MiB | yes (grid per token/head) | `flash_mla_sm120_triton.py` |
    | B1 | in-place hc_post residual | ~160 MiB if both mHC copies are live | yes | `deepseek_v4.py` hc_pre/hc_post |
    | B2 | free DSpark aux capture earlier | ~0 at the peak | yes | — |
    | A4 | chunk the whole attention block over tokens | ~500-700 MiB | **no** (GEMM tactics by M) | MQALayer.forward |

    - Bit-identical potential: ~0.6-0.8 GB, 50-60 slots. MoE cuts would reorder fp32 sums, so they are skipped.
    - **Decided:**
      1. The A0 measurement run now: finer peak-logger wrappers on the expseg2 config.
      2. A1 + A3.
      3. A2.
      4. B1 if A0 confirms the double copy.
      - A4 is skipped unless the others fall well short; that would be an owner call.
    - The served check pairs the cuts with the calibration flag, so the freed memory goes to the hot cache.
    - **A0 measured** (`/mnt/nvme1/prefill-merge/a0`, `1d0e669e7f`, expseg config, KV 381,184, 662 MiB free, 0 OOM):
      layer 2 at 4096 tokens.
      - q (wq_b output) holds 256 MiB. wq_b's own transient is +522.
      - **The attention peak, +633 over q, is in the FlashMLA core:** out 256 + out_extra 256 + the 128 merge
        chunk. Layer 1 is +330.
      - Not at the peak: wo_b +200, MoE +431, hc_post +160 at the layer end.
      - **A1 (indexer) does not touch the peak; dropped.**
      - **A2 with A3 inside: about -370 MiB (~29 slots)** before wq_b's transient becomes the peak. The next target
        is wq_b's extra 266 MiB temporary.
      - The 556 MiB held across layers is still unresolved: the hc_pre path didn't fire. A wrapper goes into the next
        instrumented smoke.
    - **Calibration fixes pushed** (`863ad432ba`):
      - The post-capture check now projects from the reserved peak.
      - VMM backing only for tensors of 64 MiB or more.
      - KV below `--max-total-tokens` is refused.
      - 45 CPU tests passed.
    - Next: A2 code plus its torch.equal GPU test, the TARGET_VERIFY capture estimate, then one combined GPU slot
      (VMM test rerun, A2 test, step-3 smoke rerun).
    - **A2 built** (`aaf76413be`, `f334c7fd44`): `SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS` (off), together with
      the merge budget.
      - The SWA and extra kernels, the merge and the sink run per token chunk into one preallocated output.
      - Chunk = budget / (16 x H x D), so 32 MiB gives 64-token chunks.
      - Expected: about -370 MiB at the attention peak.
      - CPU tests: 20 passed, 56 GPU-skipped.
      - **Capture estimate wired** (`4c075ca89c`):
        - An eager `BaseRunner._dummy_run` TARGET_VERIFY forward, with routes on the VRAM residents, so there are no
          inserts.
        - Sizing requires margin + capture estimate. The cached Exl3FusedMoE pointer tables are dropped afterwards.
        - Draft capture (0.04-0.25 GB) is not estimated; the post-capture check is the backstop.
        - 77 CPU tests passed.
      - **calib2** (2026-10-16 00:16-00:21, `/mnt/nvme1/prefill-merge/calib2`):
        - **GPU tests: 66 passed, 12 skipped, EXIT=0.** A2 is torch.equal at all three sizes and both budgets. The
          per-call peak at 4096 tokens is 289 MiB, against 583 (merge-only chunking) and 2,054 (one pass).
        - **Combined smoke** (expseg, calibration + A2 + merge budget 32): KV pin held at 381,184, 1,035 slots,
          trimmed 0, 0 OOM.
          - Calibration: lowest free 1,526; margin 298; capture estimate 25; post-capture projection 1,098, OK.
          - **memory.free at the warm-up peak: 876 MiB** (expseg2 706).
        - **Prefill transient 1,459 -> 1,259 MiB (attention 903 -> 704).** The indexer at a 128 MiB budget is now
          the binding op, so **A1 (indexer budget 32) is back in, at about -150 MiB more.**
        - Prefill time for the 4096 prompt: 19.70 s against 19.35 s (+1.8%), partly the logger. A clean A/B is
          needed before choosing the budget.
        - Calibration findings:
          - The capture estimate is too small: 25 MiB against ~130 measured (graph-pool static buffers, and the
            draft is not covered).
          - The calibration-vs-real comparison must use transients, not absolute peaks.
        - **Grow headroom at this launch: about 450 MiB, ~35 slots.**
      - **Grow built** (`acc829fc2f`, `75eac8b26f`, `6c7902bb7f`):
        - The upper bound is the pinned tier's rows minus max_gather_rows (`inclusive_hot_slot_limit`). VA is
          reserved for it, and only the configured slots are backed with physical memory.
        - `set_hot_capacity(manager, keep)` is the single authority both ways. The inverse census test checks that
          grow 20 -> 26 -> 20 restores every holder.
        - Calibration runs the warm-up's shapes (2048, 2049, 4096, 4096), then grows or trims by the measured
          headroom. The real-vs-calibration comparison now uses transients.
        - The peak logger now wraps `forward_hc_pre_from_prev` and `_hc_combine`, for the 556 MiB.
        - 150 CPU tests passed.
        - **calib3:** GPU tests, then the expseg smoke with calibration (grow), A1 (indexer 32), A2+A3 and the
          logger. Started 2026-10-16.
      - **prefill-merge replaced by `prefill-merge-2`** (owner request, context nearly full). Handoff:
        `PREFILL_MERGE_HANDOFF.md` on `prefill-merge-chunk` (`594d8aaf8b`).
        - calib3 is in flight (pid 1948706, outputs in `/mnt/nvme1/prefill-merge/calib3/`).
        - The new agent's first job is the calib3 report.
      - **calib3 crashed at startup** (2026-10-16 00:33:58):
        - The GPU tests had passed (66 passed).
        - The crash was `cuMemCreate(local): CUDA_ERROR_OUT_OF_MEMORY` in the grow path (`set_backed_rows`).
        - The KV pin held at 381,184.
        - **Cause:** grow was sized from free memory measured *before* the capture-estimate verify run (1,482 MiB),
          and that run then took about 400 MiB more (1,482 -> 1,082).
        - It planned +83 slots (1,054 MiB) and mapped 35 layers before running out.
        - The calibration summary was logged only after grow, so the crash swallowed it.
        - The agent's monitor missed the crash: `cut` block-buffered the ssh pipe.
        - **Fixes approved:**
          1. Size from free memory measured last: after all runs, synchronize and `empty_cache`, then price slots
             by granule-rounded bytes.
          2. Grow per layer with a free-memory guard; on a short check or an OOM, unmap the partial layer, keep the
             rest and log the shortfall. Never crash. Fault-injection tests come first.
          3. Log the summary before `set_hot_capacity`.
          4. The capture estimate counts reserve growth and the free-memory drop, not just the allocated peak.
        - **Fixes pushed** (`6bd1f06eb2`):
          - `settle()` sizes from F after every run, with `bytes_to_back` granule pricing.
          - Guarded per-layer grow with rollback on OOM (`HotCacheMapOutOfMemory`).
          - The summary is logged before the resize.
          - The capture estimate is the max of the allocated-peak and reserve-peak deltas.
          - 72 CPU tests passed, including fault injection: OOM on the first layer, OOM part-way, guard trip,
            rollback, and end to end. The three mutants (no rollback, no guard, unsettled sizing) are each caught.
        - **calib4 refused at startup** (2026-10-16 01:04:13): `KvPinMissed`, KV 372,480 against the 381,184 pin.
          - This is the refusal working as designed, but it is the same shortfall as the first smoke, so fix B did
            not recover it.
          - Suspect: buffers sized to the upper bound since grow (residency banks, upload buffers, transfer plan),
            or the VA reservation itself.
          - **Diagnosis:** the 4 small scale tensors (suh/svh, 44,544 B per row) are on torch.empty at the
            upper-bound rows (Σ ≈ 6,227 rows, 277 MB), against 46 MB at capacity. **That +232 MB is the whole
            shortfall.**
            - The KV-sizing headroom by run: expseg2 1.19 GB; calib2 1.09 (fix B worked); calib3 0.87 (passed by ~30
              MB); calib4 0.83 (refused).
            - Banks and index buffers are under 1 MB. The VA reservation has no physical cost.
          - **Fix approved, calib5:**
            - Bound the upper capacity by the possible grow budget B = total memory x (1 - mem_fraction_static),
              about 4.5 GB or ~340 slots, spread over the layers within the tier caps.
            - Log Σupper and B.
            - Refuse early, before the hot cache allocates, if the pin can't fit.
          - The watcher had caught the refusal 25 s after it happened, but the agent wasn't woken. It now waits in
            the foreground for short runs.
          - Fix pushed (`ed1ceb5aeb`, `2d12a5ac1d`): the grow bound, a draft-load memory probe, and
            test_expert_tier_startup's fake now takes upper_capacity. 103 CPU tests passed.
          - **calib5 attempt 1 stopped at its GPU-test gate:** 1 failed, 137 passed. The smoke is rerunning with
            the residency tests in their own process.
          - **calib5 attempt 2** (2026-10-16 01:36-01:39): GPU tests passed.
            - The grow bound is 386 slots (non-static memory 4,434 MiB). The plan was +32 slots, the bound binding
              on 0 layers.
            - Then the census post-check refused: `spec_score.hot_capacity is 26, not 27` on layer 25.
            - **Cause: the check itself is wrong, not the grow.** A row's SpecScoreRow holds the capacity of its
              prefetch *target* layer (L+2), not its own. Layer 25 grew to 27 while its target, layer 27, stayed at
              26, so 26 was correct. This was the first launch where a capacity actually changed. The CPU fakes
              set `spec_score=None`, so no test covered it.
            - Other calib5 numbers: KV pin OK (available 1.05 GB, natural pool 513,536). Verify-run drop 430 MiB,
              400 of it outside torch. Margin 574 MiB, because a repeated 4096 run still grew the reserve by
              +320 MiB (to look at later). Draft load is all torch weights (+2,179 MiB).
            - Fix approved: compare against the target layer's capacity, add a test on the real `_bind_spec_scores`
              path, and on a mismatch roll the grow back with a loud log instead of killing startup. Then calib6.
          - **calib6** (2026-10-16, `53832f12e8`, `/mnt/nvme1/prefill-merge/calib6/`): **started and served.**
            0 OOM lines. GPU tests 66 passed and 12 skipped; residency tests 72 passed.
            - KV pinned at 381,184 (natural pool 462,592).
            - Hot cache grew 1,035 → **1,059** (+24 slots, +288 MiB). The grow bound bound on 0 layers, with no
              rollback.
            - Calibration: F 2,489.9, regrow 1,595.8, verify drop 432 (402 outside torch), margin 574.4,
              keep_free 2,200.2.
            - **Capture under-estimated:** it took ~214 MiB against a 30 MiB estimate. The margin absorbed the
              gap: the post-capture projection gives 589.9 free at a 4,096-token prefill.
            - **Free at the warm-up peak: 472 MiB** (nvidia-smi), below the 508 floor (expseg2 had 706). It fit
              by luck, not by design, because the margin covered the capture miss.
            - Real vs calibration transient: ≈ +6 MiB once calibration's base is corrected for the 813 MiB of
              MoE staging. Reserved peak 14,768 against 14,674 (+94).
            - The cross-layer 556 MiB is now 481, the carried 4-stream residual state. B1 (in-place hc_post)
              would not lower the peak, so it is **dropped**.
            - Attention peak with A1: 584 (A0 903, calib2 704). The binding op is now wq_b (522).
            - Prefill of a 4,096-token prompt: 19.79 s with the logger on (calib2 19.70; without A2/A3 19.35).
              Not a clean A/B.
            - **Next:**
              - (b) Fix the watch's base and hook.
              - (a) Fix the capture estimate. First check whether a final trim or grow can run after capture,
                against measured free memory, safely with VMM. Otherwise derive the estimate from the
                graph-pool buffers plus the draft capture.
              - (e) One served A/B against expseg2.
              - Later: a clean A2/A1 timing run and wq_b's temporary.
            - (b) done (`fcd9d85c3d`). The watch was on the right forward, but the peak logger's
              per-call `reset_peak_memory_stats` wiped it. It now keeps the peak across those resets. The
              transient base is now the peak run's own entry, after staging: +4 MiB on calib6.
            - (a) **A post-capture resize is unsafe:** captured graphs bake in `hot_capacity`, through kernel
              parameters passed by value in spec_score/spec_head_tail, ExpertStreamDevice.post, and
              row_planner.scratch_base. Instead, capture will be estimated before capture by running eager dummy
              forwards inside fresh `torch.cuda.MemPool`s.
              - calib6's 214 MiB: +188 graph pools (torch reserved) plus ~26 outside torch.
              - calib7 adds log-only capture probes (`dd974488c9`), then calib8 validates the estimate.
          - **calib7** (2026-10-16 02:12, `dd974488c9`): served with 0 OOM lines, but **peak free was 38 MiB**
            (nvidia-smi), a near-OOM. Prelaunch waited ~13 min: node 1 had 8.7 GiB of reclaimable slab,
            partly from my own `find`, until the user dropped the caches.
            - Grew +66 slots (1,035 → 1,101) on a margin of 334 (calib6: 574).
            - Capture probes: two graphs, both bs=1.
              - Target verify: −172 free (146 torch reserved, 26 outside torch).
              - Draft: −42.
              - Total 214 again, against a 30 estimate.
            - **The margin is unstable:** reserve growth on the repeated 4,096 run was +320 in calib6 and +60 in
              calib7. The real warm-up's reserved peak is steady (~14,750), because its small shapes (1 to 400)
              set the allocator history before the 2048 run.
            - The fixed watch works: real transient vs calibration +41; reserved peak +312.
            - **Next:**
              - (a1) Estimate capture with eager dummies in fresh MemPools, taking the larger of that and the
                last probe.
              - (a2) Calibration replays the full real warm-up shape sequence. The margin comes from the worst
                repeated 4,096 run plus the measured noise between repeats.
              - calib8 must show peak free ≥ margin. The arm has to beat 1,035 slots at expseg2's peak free
                (706) to be worth it.
            - (a2) is pushed (`95709bcd05`). Calibration replays `DSV41_PREFILL_WARMUP_SIZES` plus 2 full chunks.
              The margin is the worst gap plus the measured spread between the full-chunk peaks. The post-capture
              check now warns and never refuses. Carrying a value across launches was dropped: it would break
              the startup-only rule.
            - (a1) The MemPool estimate is ruled out, because eager runs reuse freed blocks (~30 vs 146). Chosen:
              **(C) a throwaway trial capture** of target verify plus the draft into a private graph pool, measured,
              then `backend.cleanup()`. It is exact and touches no hot-path kernels; startup cost is ~3-6 s.
              Needs:
              - bypassing `init_cuda_graphs`, so the residency updater stays off;
              - an idempotent DSpark sampler hook;
              - a swap of the global pool;
              - `forget_graph_path_state` after the trial.
              Rejected fallback (A): bake the upper bound into the graphs and trim after capture. That touches
              hot-path kernels and needs a writer audit.
          - **calib8** (2026-10-16 02:38, `7811f88be2`): crashed at startup in the first replayed shape (1 token).
            - The dummy extend ran under `torch.inference_mode()`, and the EXL3 cast-fusion M==1 path reads
              `_version`. Fix: `no_grad`. Earlier runs never had a 1-token shape.
            - **The trial capture works:** estimate 212 MiB against the real capture's 214 (within 1%). The free
              drop was 712, of which 500 is outside torch and persistent (first use, already in F). The trial
              takes 8.0 s.
            - Open item: 148 MiB of the attention backend's capture metadata stays pinned after the trial
              teardown. That makes the grow 148 MiB more conservative than needed.
            - Also requested: any exception during calibration falls back to the configured size, logged at
              ERROR, instead of killing the launch. Rerun as calib8b.
          - **calib8b** (`0588d70bf6`): **served, safe by design.** 0 OOM lines, no rollback.
            - **Peak free 692 MiB ≥ margin 508.5** (worst chunk gap 488.5 + spread 20).
            - The replayed reserve peak is 14,722 against the real 14,758 (+36). calib7 was −312.
            - Slots 1,035 → 1,049 (+14): about even with expseg2 (706 free at 1,035).
            - The pinned 168 MiB residue held back ~13 slots. It is counted against F and not credited in the
              estimate. The real capture's net was 98, because it frees the residue. The trial's gross cost of
              232 is within 8% of the real 214.
            - Startup +18 s against calib7 (trial 8.8 s, longer replay 7 s). Prefill of a 4,096 prompt: 19.96 s.
            - Decision: the 706 is **not** a floor; the calibrated margin is the rule. The served A/B samples
              GPU free memory every 1 s. If serving dips below the warm-up peak, calibration is missing a source.
            - Next: release the residue (drop `cuda_graph_metadata_of_bucket_and_bs` after the trial), expecting
              ~+27 slots. Then calib9, then the served A/B against expseg2.
          - **calib9** (`0df35739c1`): served, peak free 682 ≥ 508.5, slots 1,047 (+12).
            - Dropping the metadata dict freed nothing: the residue is still 168. It is memory the real capture
              *reuses* (its own new allocation is +100). Candidates: `_forward_input_buffer_pool` and the
              backend's capture-state buffers.
            - The estimate of 232 overcounts by ~134, which is safe but costs ~10 slots. The "new allocation"
              estimate (64) would be 34 under, which is unsafe, so it was rejected.
            - Decision: keep 232 and run the served A/B against expseg2 now. Its main purpose is to check the
              margin under load, with GPU free memory sampled every 1 s. Release the residue holders later, with
              a per-holder diagnostic.
          - **Served A/B (e)** (2026-10-16 03:03-03:20, `prefill-merge-ab-20261016/`, `c6d3128a40`, logger off):
            `-mergechunk-expseg` (OFF, 1,035 slots) against `-expseg-calib` (ON: calibration + A1/A2/A3, 1,046
            slots).
            - **serve_compare, 5 of 8 clean prompts: +0.62 ms/token, CI −1.88..+2.60. No cost and no gain.**
              Stall-flagged: OFF prompt 0; ON prompts 2 and 5.
            - VRAM verify hit rate 43.72% against 43.69%: +11 slots (~1%) is invisible.
            - **Margin held under serving.** Min free over the whole run was OFF 580, ON 624 (margin 508.5).
              Serving never went below the warm-up minimum: the allocator keeps the warm-up reserve.
            - Node 0 min free during load: OFF 230 MiB, ON 135 MiB (watermark 43). 0 OOM in both.
            - Long-prompt prefill cost can't be measured from this prompt set (548-1,406 tokens, NVMe-dominated).
            - **Verdict:** calibration is safe and free, but with today's numbers it buys only ~+11 slots, which
              doesn't move ms/token. Its value is automation and safety, not speed. Next: release the 168 MiB
              residue (~+13 slots), write the final handoff, and wrap up the lane. (c) and (d) (wq_b) are
              deferred.
            - Follow-up checks:
              - ON's 2x-slow prefills (prompts 3 and 4, 30 s against 15 s) sit inside an nvme3n1 (SPCC) stall
                during prefill: read await 47-52 ms against the ~9 ms baseline. OFF has the same thing on
                prompts 0 and 1.
              - **serve_compare's STALL flag covers decode only, so stalls confined to prefill go unflagged.**
                Decode ms/token is unaffected, but any prefill-time comparison has to check nvme3n1 await per
                prefill window.
              - This run alone had 4 SPCC stall episodes.
          - **SPCC (nvme3, `/mnt/nvme4`) smart-log, 2026-10-16, from the owner:** **not thermal.** Every thermal
            throttle counter and every warning/critical temperature time is 0 for the drive's life. It reads 34 °C
            (sensor 1 46 °C). Media errors 0, 20% used, 148.7 TB written over 8,385 h, 127 unsafe shutdowns.
            - The A/B arms wrote only 12-13 MB to it while reading ~308 GB each. No concurrent writer causes the
              stalls.
            - Fill: 89% (1.6 TB used, 206 GB free): mimo-reap 620 GB, mimi-v26-pro 535 GB, dsv41_flash 395 GB,
              ray_tmp 28 GB. fstrim.timer is weekly; it last ran 10-12.
            - **Remaining suspects:** background GC on a nearly full DRAM-less drive, or its HMB mapping-cache
              misses on random reads spread over a 1.6 TB span. Both get better with less data on the drive.
              - Node 0's load minimum comes from the draft weight load in both arms, not from the trial capture
                or the calibration replay.
- **Known issue (pre-existing test pollution, not this work):**
  `test_expert_residency_gpu.py::TestInsertOnMissDirect::test_the_metrics_trace_reports_the_overflow_and_the_graphed_verifies`
  fails with graphed_verify_ct 1 vs 2 when run in one pytest process after the merge-chunk and VMM GPU tests.
  - It passes in isolation at every commit, including base `1ddfaae39e`.
  - Owner: the residency tests. Run it in its own process until fixed. The combined
        smoke must report prefill ms per 4096 chunk with and without A2: 64-token chunks mean many more launches.

### 3.8 Next-verify route predictor study — `cache-tiers` agent, offline

Owner approved 2026-10-15. (1) find or design data pairing draft hidden states with the next verify's routes (F109 /
F123 L+2 heads, Track B in `RAM_PREFETCH_PREDICTOR_HANDOFF.md`); (2) score no-learning baselines in the replay
(repeat-this-verify, decayed frequency, draft MoE routes) as victim protection in `_rank_victims` and as GPU-lane
choice; report; (3) a learned head only if a clear gap to the +68-hit oracle remains. Gather-time signals already
tested (token count, draft confidence, prior score, acceptance) give +0 to -1 hits.

**Step (2) result:**
- **Reframing:** a prediction of v+1 made from the draft's state after verify v arrives too late to choose v's lanes,
  and it is redundant at v+1 (the hazard check already keeps routed residents). The useful predictor runs at verify
  v's own gather: which of v's misses and residents recur at v+1. Draft-state heads could only drive a pull (a
  next-verify drip).
- **Data:** no capture pairs draft hidden states with next-verify routes. Aligned and sufficient for a gather-time
  head: C2 `router.{ids,w,x}.bin` (818 forwards x 40 layers x 6 tokens, top-6 ids, gate weights, 5120-d gate inputs,
  2.0 GB) and the B0 corpus `/mnt/nvme1/nvfp4-work/b0-corpus-100-20261012` (6,412 verify forwards, 16 GB). No new
  capture needed.
- **Baselines (C2 held-out, k = ~112 GPU lanes/verify):**
  - "Recurs next verify" precision / recall among the k first-ranked misses: actual 0.396 / 0.266; today's score
    0.444 / 0.299; feature table (token count, earliest position, acceptance odds from draft top_prob, prior score,
    gate weight) 0.501 / 0.337; oracle 0.817 / 0.549.
  - Replay hits/verify: score-k 385.9, table-k 390.3 (+4.4), next1-k oracle 431.6 (+45.7, ~4.5 ms/token), belady-k
    460.5.
  - Position and acceptance odds carry the table (position-0 misses recur 6-15%, later positions 25-48%).
  - Baselines close ~10% of the gap. The missing signal is expert identity in context, which only the gate inputs
    carry.
- **Step (3) approved and running:** a per-layer head P(miss recurs at v+1 | PCA64 of the routing tokens' gate
  inputs, position / odds features, per-(layer, expert) bias). Trained CPU-only on B0, held-out C2. Estimated serving
  cost ~2.5 MFLOP per layer, similar to the spec_score kernels. No integration until the precision is in.
  - Extraction done 18:32 (cores 56-63): B0 b000-b003 gives 5.35M (verify, layer, expert) examples over 5,969
    consecutive-verify pairs. Held-out C2: 658k examples over 752 pairs. PCA64 is fit on training rows only.
    Shards: `/mnt/nvme1/cache-tiers-tmp/head-data`, 2.3 GB.
  - **Incident:** the first training attempt touched the GPU during arm C (decay 0.995) at about 18:32:20-40.
    - torch AdamW's `torch.accelerator.current_stream()` check tried to create a CUDA context, which failed with OOM,
      so nothing was allocated.
    - Arm C's server log has no error there, only the usual late-Triton-load warnings, which already appear at 18:31.
      Its timing was already marked confounded.
    - Fixed: `CUDA_VISIBLE_DEVICES=""` is now set before torch import, in both the script and the runner.
    - **Rule:** any offline torch job sets `CUDA_VISIBLE_DEVICES=""`.
  - **Interim (precision only; hit replay pending):** head B on held-out data, scored at k = the capture's GPU-lane
    count.

    | predictor | C2 precision / recall | base |
    |---|---|---|
    | head B | **0.595 / 0.378** | 0.614 / 0.393 |
    | table | 0.471 / 0.299 | |
    | today's score key | 0.408 / 0.259 | |
    | capture | 0.420 / 0.267 | |
    | oracle | 0.851 / 0.541 | |

    - It closes about a third of the table-to-oracle precision gap.
    - Log loss on C2 misses: 0.544 for the head vs 0.610 for the table.
    - Variant A, without the per-expert PCA direction, reaches 0.557, so that term matters.
    - Training: 31 s on 4 threads.
    - Per layer: 384 biases, 17 dense weights, 64 + 384x64 weights, and a 5120x64 PCA.
    - Fix (`e0aa95a7c7`): draft odds is a constant feature in B0, so it is zeroed rather than standardized. Without
      this, C2 precision collapsed to 0.32.
    - Next: replay `--pred` for head-k hits/verify against score-k 385.9, table-k 390.3 and next1-k 431.6.
  - **Head v1 replay (held-out, hits/verify; 0.37 ms/hit traced):**

    | capture | head-k | table-k | score-k (today) | next1-k oracle |
    |---|---|---|---|---|
    | C2 | **397.8** | 390.3 | 385.9 | 431.6 |
    | base | **373.9** | 362.9 | 360.7 | 405.4 |

    - Over today's score: +11.9 / +13.2 hits, about +4.4-4.9 ms/verify or ~1.2-1.4 ms/token. That is a clear,
      consistent gain, but modest; it sits at the edge of the stall-free noise (SD 1.6 ms/token).
    - All of the gain comes from lane choice; the head doesn't touch victims.
    - Inserts fall to 116.8/verify from 121.7 (wasted 41% vs 44%), so no extra link traffic.
    - The open gap to the k-oracle is 34 hits (~3.4 ms/token): what v2 would chase. **The owner's condition is met,
      so the v2 plan goes ahead.**
    - Artifacts: `head/B`, `{c2,base}.replay-headB.*`, on `cache-tiers` through `e0aa95a7c7`.
  - **NVMe prefetch question, premise check (C2):**
    - After L+2 T3 prefetch, 33.5 unique NVMe demand misses per verify remain.
    - Of those, only 0.7% were routed in the previous verify: a forced miss lands in pinned RAM, so an expert
      routed at v is a RAM hit at v+1 by construction.
    - **The v1 head and table-k score only experts routed at v, so they can't help prefetch (≤1%).** That needs a
      384-way next-verify predictor over unrouted experts: the v2 output shape. The no-learning analogue is the
      gate's near-miss ranks 7-12.
    - **Side finding:** 18.9% of NVMe misses are rows routed 2-8 verifies earlier and since evicted from the RAM
      tier. That fits the 40 prefetch landings/verify (26 unused) churning the LRU faster than demand does.
    - **RAM-tier LRU replay calibrated** (`cache-tiers` `80c815adeb`): 190 rows/layer reproduces 50.4 RAM
      misses/verify, against 51.7 measured.
      - **Today's L+2 prefetch nets only 4.7 misses/verify for 40 reads/verify.** It covers ~18.2, but its 40
        landings raise tier evictions from 49.8 to 54.1, and the induced re-misses take back most of the gain.
        RAM misses go from 55.1 without prefetch to 50.4 with it.
    - **RAM replay results** (2026-10-16, `c2.ramprefetch.{txt,json}`; 805 verifies):
      - **Oracle next-verify prefetch on top of today's** saves 19.5 / 33.4 / 44.3 of 50.4 RAM misses at
        20 / 40 / 80 reads per verify. Timing isn't the limit (51.7 ms median gap); bandwidth and precision are.
        The 20-row oracle is worth ~5 ms/token.
      - **The near-miss gate is net negative:** 5% precision, and RAM misses go *up*, to 51.6 / 53.0 / 56.2.
        Experts just below this verify's top 6 don't predict the next verify's new experts.
      - **Churn:** each landing at today's precision causes ~0.34 extra misses. That caps any low-precision
        prefetch.
      - **v2 verdict:** the only plausible input for predicting new experts is the draft's hidden states for
        v+1, and no capture has them. v2 needs a new capture (~2 h of box, owner's call) and comes after the
        churn fix. VRAM-lane v2 is worth about +1-2 ms/token over v1.
      - **Ranking:**
        - (a) Churn fix: evict speculative landings that haven't been routed first, in `take_victim_locked`.
          Replay running now.
        - (b) Integrate v1 at gather time (+1.2-1.4 ms/token).
        - (c) The draft-hidden-state capture.
        - (d) v2.
      - **Decision:** (a) first, built only if the replay recovers ≥ 5 misses/verify; also replay it with 20
        landings in place of 40. (b) to (d) wait until (a) has a served A/B.
      - **(a) refuted and dropped** (`c2.ramprefetch-victims.txt`):
        - Speculative-first eviction comes out at 52.9 misses/verify (+2.5, worse): a landing counts as
          unrouted until its target layer runs, so it gets evicted before use.
        - Stale-first comes out at 49.9 (−0.5).
        - **The "13.5 induced by churn" claim is retracted.** `promote_landed` already stamps a landing with
          its victim's old recency, so unused landings are already the first to be evicted. The
          18.2-against-4.7 gap is a calibration ambiguity in the replay's ideal LRU.
        - Of the misses prefetch covers, 70% are experts not routed in the last 9+ verifies: genuinely cold
          rows. Only ~3.6 misses/verify could ever be helped by an eviction rule.
        - The 20-landing variant can't be measured from this capture, because spec_land has no pick index.
        - The lever for L+2 prefetch is prediction precision, not the RAM tier.
      - **Next: (b) v1 gather-time integration**, flag off by default, with a design note before code.
        (c) and (d) stay held.
      - **(b) design approved** (branch `lane-head`; ~1.5 days to tests passing):
        - Flag `SGLANG_OPT_LANE_HEAD_PATH`: the path is the switch, unset means off and bit-identical.
        - One kernel per layer inside `_gather_graph`, right before `plan_graph_routes_fused`, writes
          `keys[row]` for the routed experts. `_plan_miss_keys` then points at them, so CPU lanes take the
          experts least likely to recur. Victim ranking is unchanged.
        - +28.5 MB VRAM in total (PCA 26.2 MB), subtracted from the hot budget in `from_model` and logged.
        - Cost estimate 5-10 µs/layer, to be measured past L2.
        - The A/B reports VRAM hits per verify, CPU-lane time and gate→plan time alongside ms/token.
      - **Lane-head agent restarted** (opus, fresh context, 2026-10-16). It builds on the WIP `8191a92d3b`.
        - Its review found two bugs: dead rows were counted as routes, and the odds check fails on the real file.
        - **Fixed training/serving skew:** training used a per-request score (starting at 0, verify-only,
          0.98^live decay) and a per-request age. Serving's `insert_scores` includes prefill counts and never
          resets: ~30 after a 2k prefill where training saw 0. The head now keeps its own state, updated in
          training's exact order and reset at each prefill.
        - Draft odds: training fed the constant 1.0; serving feeds the stored training value.
        - A launch with the flag on is refused if the top-prob gate is on, or if the insert decay differs from
          training.
        - Kept independent of `prefill-merge-chunk`. A test with both flags on comes after both land.
  - **Owner decisions (2026-10-15):**
    - **v2 head:** plan it only if the replay shows clear hit gains. A plan only; no GPU training without approval.
      - GPU-trained on the full 5120-dim gate inputs.
      - Attention across the verify's tokens and layers, plus route history.
      - Runs once per verify, off the critical path (side stream during the draft).
    - **Next-verify head as an NVMe-to-RAM prefetch source (offline replay):**
      - Today every arm runs L+2 T3 prefetch at cap 2. Per forward that leaves 76.8 misses, 16.5 covered and 60.4
        NVMe demand reads. 35% of landed prefetches are used, and 26% of used ones are still in flight at use.
      - A verify-ahead prediction (~250 ms of lead time) could read during the draft phase.
      - The replay measures demand reads caught, alone and on top of L+2, at 0.5x/1x/2x of today's spec-read
        budget. It also counts wasted reads and RAM-tier eviction effects, with table-k and the next1-k oracle as
        bounds.

### 3.9 Graded miss cut — branch `misscut-replay` (`5014aaff0a`, `23c09c180c`, `f2e2e0c80b`, `760736c33b`, `7d105fabfe`), NOT merged

The safe graded+ variant gains ~0.5-1 ms/token. Not prioritised.

### 3.10 CPU expert compute gap — agent `cpu-expert-gap`, branch `cpu-out-rows` (`3be009c47f`)

- Compile flags identical (GCC 15 `-Ofast -march=native` via `exl3/ext.py`). One expert x one token: 0.41-0.47 ms in
  the bench vs ~0.57 median in the server; shared tokens +0.23 ms server vs +0.15 bench. The 6-row output memset costs
  19/38 us per job.
- **Store-on-first-write built** (`SGLANG_DSV41_CPU_EXPERTS_STORE_OUT`, off).
  - With the flag on, the kernel stores `w*src + zero` on each token's first write, with `zero` behind an asm
    barrier so -Ofast keeps the bits identical (including -0). Later writes add.
  - Only rows of unrouted tokens are zeroed, and the engine memset is skipped. NVFP4 already overwrites every row.
  - Unit tests: 165 passed.
  - **Bit test passed** (2026-10-15 20:03-20:08, `wt-cpu-out-rows` at `901ab0fead`, `KERNEL EXIT=0` from
    PIPESTATUS): `test/manual/dsv41/test_exl3_cpu_store_out.py`, with 11 cases x threads 1/4/10 plus -0. It compares
    the old zero-then-accumulate, the new store_first and the kernel overwrite bit for bit on int32 views, then
    again after a second accumulating job. Unrouted tokens read exactly +0. `test_exl3_cpu_dsv41_plan_multitoken.py`
    is still green, giving 51 passed.
  - **Bench** (`bench_cpu_store_out.py`: cores 6-15, 10 threads, node 0, real EXL3 kernel via `run_job`, 6 rounds of
    alternated arms, 50 reps):
    - k=1 median: 472 us off vs 448 us on.
    - Paired deltas without outlier rounds: 5-37 us.
    - k=2/3 deltas move with the outlier rounds, so they are noise. k=4 shows no difference.
    - **Saving: ~10-25 us per job**, which matches the 19 us memset. That is far below the 1 ms/verify bar, so no
      served arm.
  - Verdict: correct and bit-identical, but too small to serve alone. Kept off and unmerged on `cpu-out-rows` until
    a larger served A/B can carry it.
  - Two quiet-box rounds had one cell 30-80% slow, the same right-tail variance as in the gap analysis.
  - Logs: `ram-prefetch/cpu-expert-gap-20261015/cpu-out-rows/`.
- Other suspects, not yet tested: NUMA balancing on (TLB IPIs near prefill), IRQ 239 on core 14, swap (now addressed
  by `-mem`). Proposed benches: numa_balancing A/B, perf clock check, busy-socket bench.
- Facts: no CPU core copies expert weights to the GPU (copy engine = `cuMemcpyAsync` DMA; stream kernel and drip are
  GPU pulls; `HostCopyBackend` memcpy only when `device < 0`). `spec_promote` is a pointer remap into the RAM tier, no
  bytes copied. `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` is refused for EXL3. CPU keep-warm
  `SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US=2000` is on by default in all runs.

### 3.11 JIT build speed — MERGED as `0923ed8e97` (`99e54e2bf3`, `fc653e9a8a`, `d81d37dcf1`)

- Cold build of the 9 expert-stream host modules:
  - serial (old): 614.7 s;
  - parallel warm (`python -m sglang.test.warm_expert_stream_modules --variants instr,prod --lanes 8,16 --nodes
    1,2`): 109.5 s;
  - parallel + launcher, ccache cold: 105.0 s;
  - from another worktree with `SGLANG_JIT_COMPILER_LAUNCHER=ccache`: 22.2 s (9 direct hits);
  - all cached: 0.1 s.
- Tests: 53 + 22 passed.
- Only the launcher gets cross-worktree hits: it reaches the tree through a `sglang-src` symlink, so paths match.
  The system-wide masquerade only caches nvcc's host `c++` subcalls with per-worktree paths.
- nvcc through the launcher is untested on the box.
- `hash_dir = false` is added to `~/.config/ccache/ccache.conf`; with -g, the cwd would otherwise defeat hits.
- A suite runner should run the warm-up command once before `pytest -n 8`.

- Opt-in `SGLANG_JIT_COMPILER_LAUNCHER` (ccache for nvcc builds) and parallel warm-up. Cold-build timings running
  (serial, parallel, ccache); may be inflated by concurrent test compiles.
- Owner: merge into `ram-prefetch-pool-serve` once timings are in (if tests pass).
- divix01 has ccache 4.11.3 system-wide (`/etc/profile.d/ccache.sh`), `~/.config/ccache/ccache.conf` max_size 50G.
  Build-time measurements must set `CCACHE_DISABLE=1`.

## 4. Ideas on record, not started

- Predict layer 0/1 experts; aim RAM prefetch at CPU misses.
- Move part of the pinned tier from node 0 to node 1 (`SGLANG_MOE_PINNED_HOST_NUMA_MB`) — superseded by the HiCache
  NUMA move unless arm A leaves node 0 short.
- In-server interleaved A/B harness for noise (offered; owner chose other work).
- RAM prefetch `promote_landed`: ~42 rows land and ~20 are used per verify; a RAM-tier precision concern.

## 5. Box rules for anyone continuing

`.claude/rules/divix01-run-protocol.md` applies: push, then a fresh divix01 worktree; `PYTHONPATH=$PWD/python` and
print `sglang.__file__`; CPU under `taskset -c 0-63` with `OMP_NUM_THREADS`; GPU under
`flock cc-gpu.lock taskset -c 32-63`, disk lock first; quiet `nvidia-smi` and ~29 GB model-load avail mem before
serving; outputs under `/mnt/nvme1` or `ram-prefetch/`; never touch `cc-expert-prediction/dsv41-direct-prod` or
port 7867.
