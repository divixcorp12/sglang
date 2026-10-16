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
  - Training (variants B then A), C2/base report and the replay with `--pred` are running CPU-only.

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
