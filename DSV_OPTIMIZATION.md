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
  - Found: `--weight-loader-drop-cache-after-load` does not reach the DSpark draft loader (4-5.6 GiB of draft
    shard cache at ready). Fix in progress.
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

### 3.6 Hot-cache decay 0.995 — arm C in flight

Existing env `SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY` (0.98 today). Below serving noise, so the evidence is the
routed-pair VRAM hit rate from the hot-cache metrics counters (expect +0.5-1 point vs arm A); ms/token must only show
no regression.

### 3.7 Prefill VRAM spike / hot-cache capacity — agent `prefill-merge`, branch `prefill-merge-chunk` (not pushed yet)

- `benchmarks/dsv41_baseline/arm_env.py`: the triton FlashMLA prefill merge runs in fp32 and costs ~0.9 GiB at a 2048
  chunk and 1.75 GiB at 4096 over flashinfer, so DSpark's prefill peak sits 4,808 MiB above init. The DSpark arms
  therefore run `DSPARK_MEM_FRACTION_STATIC=0.78`, `DSPARK_HOT_GPU_MB=10840` (12040 before) — the "1.2 GB".
  Owner TODO option C: chunk the merge over tokens, bit-identical.
- Task: confirm the cause, build the chunked merge (env-gated off, exact-equality GPU test), measure peaks (4096
  warm-up, 30k prompt) on/off, derive the largest hot cache that keeps KV at exactly 381,184 tokens with a margin like
  today's 508 MiB (computed, not guessed), add the arm, ask for a slot, run ON only vs splitmodel-stream-mem.

### 3.8 Next-verify route predictor study — `cache-tiers` agent, offline

Owner approved 2026-10-15. (1) find or design data pairing draft hidden states with the next verify's routes (F109 /
F123 L+2 heads, Track B in `RAM_PREFETCH_PREDICTOR_HANDOFF.md`); (2) score no-learning baselines in the replay
(repeat-this-verify, decayed frequency, draft MoE routes) as victim protection in `_rank_victims` and as GPU-lane
choice; report; (3) a learned head only if a clear gap to the +68-hit oracle remains. Gather-time signals already
tested (token count, draft confidence, prior score, acceptance) give +0 to -1 hits.

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
