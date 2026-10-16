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

### 3.3 Startup calibration of the split-model constants — branch `split-model-calib` (`7171570959`), NOT merged

- Owner requirement: "All the offsets and values should be calculated at server startup via a calibration routine."
- Extends `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION`: CPU grid {1,2,4} experts x {1,2,6} tokens fitting a/b/c per
  team; copy lane per node; stream coupling with 1/2/4 copy lanes; NVMe landing; live refinement (retune), clamps,
  logging, fallback to the table. Status: validation (short server start logging constants, no A/B); offline fits
  sent for comparison. **Pending report.**

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
- **Not done (root, owner's):** `kernel.numa_balancing=0`; IRQ 239 (`eno1-tx-0`) off core 14 (a team0 core);
  optionally `swapoff -a && swapon -a`; possibly revert `compaction_proactiveness` to 20 (was set to 100).

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

**Read amplification (open, agent `read-amp`):** on a clean, idle-box run (arm D), the three NVMe mirror
partitions read ~895 GiB (831 GiB in 262 s of decode, ~3.2 GB/s). The server's own counters account for 209 GiB
(rows_read 10,858 + spec_landed 6,008, x 13.3 MB), so the disks read ~4.3x more. Even a full row read from all
three mirrors would leave ~1.4x unexplained. Arm C showed the same. The agent is breaking the bytes down by phase,
device and consumer before proposing a fix.

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
  not +103. Derived, not yet run; on hold until the 3120 MiB attribution is in.

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

### 3.9 Graded miss cut — branch `misscut-replay` (`5014aaff0a`, `23c09c180c`, `f2e2e0c80b`, `760736c33b`, `7d105fabfe`), NOT merged

The safe graded+ variant gains ~0.5-1 ms/token. Not prioritised.

### 3.10 CPU expert compute gap — agent `cpu-expert-gap`, branch `cpu-out-rows` (`3be009c47f`)

- Compile flags identical (GCC 15 `-Ofast -march=native` via `exl3/ext.py`). One expert x one token: 0.41-0.47 ms in
  the bench vs ~0.57 median in the server; shared tokens +0.23 ms server vs +0.15 bench. The 6-row output memset costs
  19/38 us per job.
- In flight: remove per-job output-row zeroing (store-on-first-write, else zero unwritten rows only, else AVX-512 NT
  stores + sfence), env-gated off (`SGLANG_DSV41_CPU_EXPERTS_STORE_OUT`), bit-identical test. Unit tests running; then
  a box slot to measure.
- Other suspects, not yet tested: NUMA balancing on (TLB IPIs near prefill), IRQ 239 on core 14, swap (now addressed
  by `-mem`). Proposed benches: numa_balancing A/B, perf clock check, busy-socket bench.
- Facts: no CPU core copies expert weights to the GPU (copy engine = `cuMemcpyAsync` DMA; stream kernel and drip are
  GPU pulls; `HostCopyBackend` memcpy only when `device < 0`). `spec_promote` is a pointer remap into the RAM tier, no
  bytes copied. `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` is refused for EXL3. CPU keep-warm
  `SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US=2000` is on by default in all runs.

### 3.11 JIT build speed — branch `jit-build-speed` (`99e54e2bf3`, `fc653e9a8a`, ... `d81d37dcf1`)

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
