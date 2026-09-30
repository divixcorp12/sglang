# DSV4.1 decode: compute RAM-resident experts on the CPU

**Status:** plan, not started (2026-09-29). Evidence base: `DSV41_REFERENCE.md` §28 (kernel and handoff
microbenchmarks on DSV4.1 shapes), §25.1/§27.3 (where decode time goes), experiment log E21/E26 (the earlier
Qwen/NVFP4 spike and why it lost at 4 threads).

**Goal.** At batch-1 decode, stop copying RAM-tier experts over the Gen3 link. For each layer, send the hidden
state (10 KB) to the host, compute some or all of the layer's RAM-resident experts on the CPU from the pinned tier,
and return the partial sum (20 KB). The GPU keeps computing its hot-cache experts and the shared expert meanwhile.

**Why now.** ~97 ms/token today (user session 10.35 tok/s). ~68-76 RAM-hit rows of 13.3 MB cross the link per token,
~67-80 ms; NVMe exposure ~13-20 ms; all GPU compute ~14 ms (§25.1, §27.3). One expert costs ~0.98-1.11 ms on the
link and **0.40-0.46 ms on 12 node-1 cores** (0.28 ms on 18 node-0 cores), and the handoff costs 16-23 us per layer
(§28). §28.3 estimates 50-60 ms/token [estimate].

---

## 1. Library choice

| Option | Reads our EXL3 rows as-is | Extra RAM | Fast path on Xeon 6154 (AVX-512BW, no VNNI/AMX) | Graph-safe handoff | Verdict |
|---|---|---|---|---|---|
| **exllamav3 `cpu/moe_mul1.cpp` + `moe_handoff.cu`** (MIT, turboderp, pin `02aef45`) | **yes**, the same trellis/suh/svh | **none**, computes in place from the pinned tier | AVX-512BW tier, measured 3.3 GB/s/core, 1.4% rel L2 | yes, memop flag write/wait, fused issue/collect | **use** |
| ktransformers / `kt_kernel` | no (AMX/AVX-512 int4/int8, GGUF, NVFP4) | a second full copy: ~270 GB at 4-bit, does not fit ~90 GB | its AMX/VNNI kernels do not run here; AVX2 fallback only | `cudaLaunchHostFunc` callbacks | reject; borrow deferral only |
| llama.cpp / ik_llama (`-ot exps=CPU`) | no (GGUF/IQ quants) | second copy, as above | yes | not an SGLang component | reject; baseline only (E21) |
| Custom kernel | yes | none | whatever we write | ours | only if exllamav3's kernel misses a gate |

**Take from exllamav3:** the kernel (`exl3_moe_cpu_forward_raw`), its thread pool, and the ideas behind its hybrid
split (`modules/block_sparse_mlp_cpu.py`, `model/moe_cpu_host.py`):
- `-1` sentinels for GPU-resident picks;
- GATED jobs that skip an empty CPU share;
- issue before and collect after the GPU's own expert work;
- between-generation hot/cold swap sweeps.

**Don't take `MoeCpuHost` wholesale.**
- It assumes every expert is RAM-resident, in its own child-process arena with a static tail split. We have three
  tiers (NVMe → 100 GiB pinned tier → 16 GB VRAM hot cache), and residency changes every step.
- Run the pool **in-process** instead, over our existing pinned slabs.
- Signal the GPU through our existing stream-ordered gate (`cuStreamWaitValue32` plus host release, §29.14). That is
  the same memop mechanism as upstream's `exl3_moe_flag_wait` and §28.2's "memop" arm.

**Vendoring.** Copy `moe_mul1.{h,cpp}` into the fork's JIT/ext tree with its MIT header and the upstream commit
recorded. Do not import the checkout under `cc-exl3-cpu-bench/`.

### Kernel adaptations (all small; each gets a CPU unit test against a fp32 dequant reference)

1. **Clamped SwiGLU.** DSV4.1 uses `swiglu_limit` 10: the gate is clamped from above and the up projection to
   ±limit before `silu(gate)*up`. That is not upstream's `swiglu_oai` (activation 3). Add activation 4 and test it
   against the GPU path's activation.
2. **Slab-backed layer, not a registered expert list.**
   - `make_layer` wants per-expert tensors registered once. Our rows live at `ExpertPinnedHostCache.tensors[name][slot]`,
     and residency changes.
   - Build the `MoeCpuMatrix` triples per job from the `[L,6]` slab base table (`exl3_ram_miss.py:_slab_table`) plus
     host slot × stride.
   - Jobs carry host slot ids, not expert ids.
3. **Fused `w13` slab.** The tier stores `w13_trellis/suh/svh`, gate and up together, while the kernel wants separate
   gate and up matrices. Work out the exact fused layout (tile order along n, and whether the two `suh` agree), then
   either pass an n-offset and stride into `MoeCpuMatrix` or run one n=2·I matrix and split before the activation.
   **This is the first thing to verify.**
4. **Native layout.** The GPU copies the same slab rows, so the band-contiguous "swizzle" cannot be applied in place.
   §28's throughput numbers may have used the swizzled layout (`bench.py weights(..., swizzled_random=True)`), so
   **re-measure the native layout**. If swizzling is worth >15%, consider a swizzled layout only for rows the CPU owns.
5. **Affinity.** Keep `EXL3_MOE_CPU_PIN=0` semantics and pin the workers ourselves to the assigned cores. Refuse a
   single-core pool (the §28 livelock trap).

---

## 2. How many experts go to the CPU, and which ones

**Never GPU-resident experts:** those cost ~0.02 ms each on the GPU. The choice is only over a layer's RAM-tier hits
(and later its NVMe misses, once they land in the tier).

### Per-layer cost model

For a layer with `n` RAM hits, `k` of which go to the CPU:

```
T_cpu(k)  = h + k · c_cpu(threads, numa)            h ≈ 0.02 ms handoff (§28.2)
T_link(k) = (n − k) · c_link                        c_link ≈ 0.98–1.11 ms/row (§28.1, §3)
T_layer   ≈ max(T_cpu(k), T_link(k) + t_gpu_moe, t_gpu_resident + t_shared)
k*(n)     = argmin_k T_layer                        ≈ n · c_link / (c_cpu + c_link) ≈ 0.7 n
```

At 12 node-1 threads (c_cpu ≈ 0.44), k* is 1→1, 2→2, 3→2, 4→3, 6→4. **Up to two RAM hits per layer all go to the
CPU; above that the link takes the remainder in parallel.**

Today's mean is ~1.7-1.9 RAM hits per layer. So in the common case the link carries almost nothing and the CPU
finishes a layer's share in ~0.9 ms instead of ~2 ms of copies [estimate].

**Which k:** prefer rows whose host slot sits in node-1 memory, which is local to the pool. The tier is 60 GiB on
node 0 and 40 GiB on node 1. Node-1 cores reading node-0 memory is unmeasured, and it would load the memory
controller the GPU's DMA also reads (CPU load on node 0 cut H2D by 27-43%, §25.4).

### Deciding the numbers: measure, model, then sweep

1. **Calibrate** (CPU only, no server, ~½ day). Measure `c_cpu` on the 12 assigned node-1 cores (threads {8, 10, 12}) × {node-1-local,
   node-1 cores reading node-0 memory} × {1..6 experts/call}, on the native layout with real rows mmapped from the
   checkpoint. Measure it again with a synthetic NVMe read load, and again with a copy-engine H2D load, running at
   the same time. `c_link` is already measured.
2. **Model offline** (laptop or divix01 CPU). Extend `scripts/dsv41/tier_sim.py` with the CPU cost model, then replay
   captured routing (the route log's `record_router`) at today's recipe: hot 16080 MB, 100 GiB tier.
   - Per layer, record the distribution of `n` (RAM hits) and NVMe misses.
   - Compare predicted ms/token for these policies:
     - off;
     - all RAM hits to the CPU;
     - `k*(n)` split;
     - `k*(n)` plus NUMA preference;
     - also computing NVMe-landed rows on the CPU.
   - Sweep threads.
   - Pre-register the gate: go only if the predicted gain is ≥15%.
3. **Runtime table.** At startup, run a ~1 s micro-calibration of `c_cpu` on the chosen cores and build `k*(n)` for
   n = 0..6 as a device table. The post kernel, which already knows routes and residency on the device, reads it,
   so the decision stays inside the graph with no host round trip.
   - An EWMA of measured job time (the worker timestamps done) re-tunes the table between requests.
   - It never changes mid-graph.
4. **Sweep served A/B arms** (`run_arm.sh`, paired sessions). off / cap 1 / cap 2 / `k*` table, then thread counts
   {8, 10, 12} on the assigned cores. Byte-identity cannot be the correctness check (§3 below), so use the quality gate plus ms/token.

---

## 3. KV cache interaction

**Direct: none.** Routed experts are a position-wise FFN with no per-token state. Attention, the compressed KV,
indexer, SWA windows, radix/HiCache all stay on the GPU exactly as today, and the CPU never reads or writes KV. The
indirect effects:

- **Numerics.**
  - The CPU kernel quantizes activations to int8: 1.4% relative L2 per expert against fp32, where upstream's
    tolerance is 5%.
  - So decode hidden states, and the KV written from them, differ slightly from the GPU-only path. Radix and HiCache
    reuse stays valid, in the same sense as any non-batch-invariant kernel, **but arm outputs are no longer
    byte-identical**.
  - Replace the byte-identity check with a quality gate: teacher-forced logprob/KL of CPU-on against CPU-off over
    corpus sessions, plus a greedy-match rate.
- **Prefill stays on the GPU path.** CPU cost scales with tokens × experts, while a link copy is paid once per chunk.
  So the CPU path runs only at decode (M == 1; the recipe is batch 1). A prompt's KV is built by the GPU path, and
  the decode that follows uses the CPU path. That mix is fine.
- **Speculative decoding (DSpark), when it lands.** A 6-token verify is M = 6. The CPU cost grows with it, so the
  `k*` table needs an M dimension or the CPU path is off for M > 1.
- **Host memory.**
  - The CPU path adds only small pinned buffers (hidden, weights, output per layer slot).
  - It shares memory bandwidth with the HiCache host pool (~9 GB, write_through DMA on node 0), the Engram host
    tables, and NVMe DMA into the tier.
  - Calibration step 1 measures this.
- **VRAM.**
  - Nothing new is needed.
  - Fewer RAM-hit copies means the miss/scratch rows see less traffic. Any VRAM that frees up later goes to the hot
    cache or the KV pool (the pool-versus-context guard in §29.13 still applies).

---

## 4. GPU-side integration (decode graph)

These are the changes found in the current code (`exl3.py:_apply_graph`, `expert_stream.py:_gather_graph`,
`exl3_ram_miss.py`, `lease_kernels.cuh`, `exl3_fused_moe.py`).

1. **Post carries the CPU share.** Today the post (`lease_kernels.cuh`, LaneRequest) publishes expert ids and
   destination slots but **not the top-k weights**.
   - Add per-route weights plus a CPU-assigned bit, or a small separate pinned region.
   - Add a D2H of the fp16 hidden state (10 KB), then a gate release.
   - RAM hits assigned to the CPU get no C1/CW copy, and the copy thread skips them.
2. **Fused MoE skips CPU routes.** `Exl3FusedMoE` counts routes per slot and skips a slot only when its count is 0.
   It has no per-route `-1`, and a weight of 0 still computes.
   - Remap CPU-assigned routes to a sentinel slot whose count is forced to 0. `expert_count` has `slots+1` entries;
     verify whether the last is usable, or add a mask to `route_tables`.
3. **Issue early, collect late** (exllamav3's `cpu_split_submit` / `cpu_split_combine` order):
   `router → post + D2H + release → GPU resident experts + shared expert → wait(CPU done gate) → H2D 20 KB → add`.
   - Apply the routing weights the same way on both halves, including `routed_scaling_factor`.
   - The wait uses §29.14's stream-ordered gate: no SM spinning, and `abort_copy_waits` covers teardown.
4. **Worker.** An in-process C++ pool (not Python, no GIL) on the assigned cores. It polls the per-layer job slot,
   resolves host slots to slab pointers, runs the kernel and releases the done gate.
   - One job per layer. The next layer cannot start before this one's output, so there is no cross-layer pipelining
     without deferral (§6).
5. **Residency still learns.** An expert computed on the CPU counts as a hit for hot-cache scoring. With the link
   mostly idle, promotions to VRAM run as background H2D. Today they happen device to device from the scratch copy,
   which will no longer exist for CPU rows. The parked prefetch (§26.2 item 7) becomes viable again, since the link
   is now idle.
6. **Flags and refusals.**
   - New `SGLANG_DSV41_CPU_EXPERTS*` env vars. Read the `env-var-conventions` skill before touching `environ.py`.
   - The launch gate (`expert_stream_requirements`) refuses the flag off EXL3, at batch > 1, or without explicit
     cores.
   - Default off until the arms pass.

---

## 5. Phases and gates

| Phase | Work | Where | Gate to continue |
|---|---|---|---|
| **P0** kernel fit | Adaptations 1-5; real-weight accuracy per expert vs fp32 reference; native-layout throughput; calibration step 1 | CPU only on divix01 (`taskset`, node-1 cores), private worktree | Rel L2 ≤ 2% on real rows; native `c_cpu` ≤ 0.5 ms at 12 threads |
| **P1** offline model | `tier_sim` CPU model + routing replay; policy table | CPU | Predicted ≥15% ms/token gain |
| **P2** in-graph prototype | §4 items 1-4 behind the flag; fixed policy "RAM hits → CPU, cap `k*`"; GPU unit tests (graph capture/replay, gate abort, eager vs graph equivalence) | GPU under `cc-gpu.lock` | Teacher-forced KL within budget (set in P0); A/B arms win paired sessions |
| **P3** policy and link | Runtime `k*` table + EWMA; NUMA-aware choice; background promotions over the idle link; NVMe-landed rows computed on CPU | GPU | Arms beat P2 |
| **P4** optional | Node-0 cores or split sockets; tier rebalance toward node 1; one-layer deferral of low-weight CPU experts (quality-gated); prefetch revisit | GPU | Each item on its own arm |

**Rough budget [estimate]:**
- ~70 RAM hits × 0.44 ms plus 40 × 0.02 ms of handoff ≈ 32 ms of CPU time per token, against ~70 ms of link now.
- With NVMe exposure (~17 ms) and compute (~14 ms) unchanged, decode lands near **55-65 ms/token (15-18 tok/s)**.
- P3/P4 aim at §28.3's 50-60 ms/token.

---

## P0 results (2026-09-29, `c7b6ca4ab4`, `analysis/dsv41-drive/cpu-experts/p0_real.py`)

**Setup.**
- Production was down. The fork's own `exl3_ext()` build (exllamav3 `02aef45`) ran on real DSV4.1 rows read from the
  EXL3 shards, laid out as the pinned tier's slabs (`w13_*` `[N, 2, ...]`, gate = part 0, up = part 1).
- Activation 0 with `act_limit` 10, as the graph path passes it.
- Cores 18-29, `numactl --membind=1`, `EXL3_MOE_CPU_PIN=0`.
- Raw records are in `p0_results.jsonl` beside the script on divix01 (`wt-cpu-experts`). Logs are
  `p0_acc-r1.log` and `p0_perf-r1-{native,swizzled}.log`.

**Plan corrections found on the way:**
- **No SwiGLU work is needed.** Upstream's activation 0 with a nonzero `act_limit` computes
  `min(silu(g), lim) · clamp(u, ±lim)`, the same as the GPU graph path's exllamav3 kernel.
  (The eager torch reference in `exl3_ops.py` clamps before the SiLU, a negligible difference.)
- **No w13 adapter is needed.** `w13_*[slot, part]` is contiguous per projection, and the GPU's
  `slot_pointer_tables` uses the same views.
- **No vendoring is needed.** `exl3_ext()` already builds `cpu/moe_mul1.cpp` and `moe_handoff.cu` and binds
  `exl3_moe_cpu_*`, `exl3_moe_flag_*` and `moe_unswizzle_trellis`.
- **The codebook matches.** Every loaded tensor's `mul1` is `0x83DCD12D`, the constant the CPU kernel hard-codes.

**Accuracy** (`accuracy 0,19,39 12 acc-r1`, under `cc-gpu.lock`).
- Setup: 12 random experts per layer, 8 cases each, top-1 and top-6, at input scale 1 and at 8 (scale 8 drives
  values past the clamp).
- Relative L2 against the fp32 reconstruct reference (`exl3_linear_reference`):

| Path | p50 over groups | worst max |
|---|---:|---:|
| CPU, AVX-512BW tier (what would serve) | 1.38-1.86% | 2.23% (layer 39, scale 8, top-6) |
| GPU `exl3_linear` (today's production kernel family) | 1.05-1.38% | 1.51% |
| CPU, scalar fp32 tier | 0.16-0.44% | 1.25% |
| CPU BW against GPU | 1.71-2.30% | 2.75% |

- The CPU path is ~1.3x the GPU path's own quantization error, with every output finite. The int8 activations
  account for the gap: the scalar tier is 4-8x closer to the reference than either.
- The ≤2% gate passes at the median and is marginal at the worst case. The deciding gate stays the teacher-forced
  KL of P2.

**Throughput** (`perf 19 384 8,10,12 1,2,6 {native,swizzled}`). Cold, over 384 real experts (5.1 GB). ms per
expert, p50 (p90 ≤ 1.1x p50):

| Threads | native, 1 / 2 / 6 per call | swizzled, 1 / 2 / 6 per call |
|---:|---|---|
| 8 | 0.717 / 0.658 / 0.644 | 0.593 / 0.539 / 0.529 |
| 10 | 0.622 / 0.594 / 0.543 | 0.514 / 0.485 / 0.455 |
| 12 | **0.555 / 0.510 / 0.475** | **0.450 / 0.411 / 0.385** |

- **Native misses the ≤0.5 ms gate at decode's 1-2 experts per layer** (0.51-0.56 ms). It is still ~1.9x
  faster per expert than the link (~1.0 ms). Swizzled matches §28's 0.40-0.46 and is 19-23% faster.
- **Swizzling is worth taking, and does not need a second copy.** Upstream stores the CPU arena swizzled and
  unswizzles on the GPU when staging to VRAM (`moe_unswizzle_trellis`, bound in `exl3_ext`). Applied here, the
  pinned tier (and the row images it reads from NVMe) would be swizzled, and every H2D row copy would gain one
  device-side unswizzle into its VRAM slot. Price that in P1 against a ~20% smaller `c_cpu`, and measure the
  unswizzle kernel's cost per row on the 5090.
- **With native `c_cpu` ≈ 0.53 ms, `k*(n)` becomes ~0.65·n:** n = 1 → 1, 2 → 1-2 (a tie), 3 → 2, 4 → 3,
  6 → 4. With swizzled (0.42), n = 2 → 2.

**Still open in P0:**
- `c_cpu` under concurrent NVMe DMA into node-1 slabs and under copy-engine H2D load.
- Node-1 cores reading node-0 memory.
- Both need the RAM-miss service running, and are easiest to take in P2's prototype.

---

## 6. Risks and open questions

- **Cores (owner decision 2026-09-29): 12 node-1 physical cores, e.g. 18-29.** They are dedicated to the CPU-expert pool;
  30-35 stay for co-tenants and CPU jobs. Expected c_cpu ≈ 0.40-0.45 ms per expert (§28.1). Pin the co-tenants off
  18-29 before any arm, because spinning workers on shared cores add jitter to the critical path. P0 calibration uses
  exactly these cores.
- **Node-1 DRAM is the ceiling.** It reads 35 GB/s against node 0's 62 GB/s, and 12 threads already reach ~92% of
  it. NVMe DMA into node-1 slabs competes with the kernel.
- **The CPU is on the critical path of every layer.** A late worker stalls the GPU. We need the gate abort (2 s) and
  a per-step fallback: if the pool is down, route everything to the link.
- **Quality with real weights and the clamped SwiGLU is untested** (§28, "Not established").
- **E26 lesson.** At 4 threads, CPU experts lost once the copy kernel got faster. Re-run P1 against the current
  link cost before P2, and keep the policy table data-driven so the choice flips back automatically if the link
  improves.

---

## P1 results (2026-09-29, `645ffdfb83`, `scripts/dsv41/cpu_expert_sim.py`)

**Setup.**
- Trace: the router-capture route log (`direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl`), 6,153
  decode tokens, hot cache 28-29 slots per layer as logged, pinned tier 8,063 rows (100 GiB).
- VRAM is `tier_sim.DirectInsertReplay` (DIRECT insert-on-miss, started from the first logged hot set); the pinned
  tier is `ram_replay.py`'s RamTier (`base` policy). Prefill forwards are replayed, decode is counted.
- Per decode token and layer: n = misses VRAM, hits RAM; m = misses both.
- Model checks against the run: VRAM misses/token 78.78 (replay) vs 78.78 (route log); m = 11.32 vs 11.265 measured
  RAM misses/token.
- Cost model as §2, with c_cpu(k) interpolated through the P0 points at 1/2/6 experts per call. Defaults:
  `c_link` 1.0, `h` 0.02, GPU 14 ms, NVMe exposure 1.5 ms per miss (11.3 × 1.5 ≈ 17 ms, inside §27.3's 13-20 ms).

```bash
# divix01, wt-cpu-p1 at 645ffdfb83, PYTHONPATH=$PWD/python OMP_NUM_THREADS=8, taskset -c 0-17,30-63, 30 s
O=/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture
$PY scripts/dsv41/cpu_expert_sim.py $O/stages.jsonl --out /mnt/nvme1/cpu-p1/native.json
# swizzled table: --nvme-ms 1.5 --c-cpu-table '{"8":[0.593,0.539,0.529],"10":[0.514,0.485,0.455],"12":[0.450,0.411,0.385]}'
# GPU term off:   --nvme-ms 1.5 --gpu-ms 0
```

**Where n and m fall** (over 246,120 token-layers; 0.7 mean m and up to 3.0 mean n per layer; layer 0 is the heaviest):

| n | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| share of layers | 17.8% | 31.1% | 27.1% | 15.4% | 6.3% | 1.9% | 0.4% |

- Mean n = **67.5 per token** (1.69 per layer), mean m = 11.3 per token (0.28 per layer). m histogram 0..6:
  188,602 / 47,348 / 8,530 / 1,362 / 239 / 36 / 3.
- Per-layer means and histograms are in `native.json` (`mean_n_per_layer`, `n_hist_per_layer`).
- 76% of layers have n ≤ 2, the range where `k*` sends everything to the CPU.

**Predicted ms/token** (12 threads, NVMe 1.5 ms/miss, GPU 14 ms), gain against `off` [estimate]:

| Policy | native ms/token | gain | swizzled ms/token | gain |
|---|---:|---:|---:|---:|
| off | 109.76 | 0.0% | 109.76 | 0.0% |
| all RAM hits to CPU | 71.35 | 35.0% | 65.05 | 40.7% |
| **`k*(n)` (`split_table`)** | **70.63** | **35.6%** | 64.28 | 41.4% |
| `k*` cap 1 | 82.49 | 24.8% | 81.31 | 25.9% |
| `k*` cap 2 | 73.16 | 33.3% | 67.80 | 38.2% |
| `k*` exact (argmin with k-dependent c_cpu and m) | 67.17 | 38.8% | 62.90 | 42.7% |
| `k*` + NVMe-landed rows on CPU | 65.85 | 40.0% | 61.02 | 44.4% |

- `split_table` uses a scalar c_cpu (the value at 2 per call), as the P3 runtime table would. `k*` exact uses the
  k-dependent cost and counts m on the link side, so it is the ceiling for a table without an m dimension.
- `k*` + NVMe assumes the m landed rows are computed on the CPU (rows_cpu = k + m) and their read latency stays fully
  exposed: the two add, no overlap. A landed row therefore leaves the link.
- NUMA preference is not modeled: the trace has no host-slot node.
- Not modeled either: c_cpu under NVMe DMA or copy-engine load (P0 open), any swizzled-layout unswizzle cost on H2D
  rows (the swizzled columns take P0's numbers as given), and the GPU work the CPU share overlaps with.

**Thread sweep** (native P0 table, NVMe 1.5), `k*` gain and ms/token; other policies in `native.json`:

| Threads | c_cpu at 1/2/6 | off | all-CPU | `k*` | `k*` exact | `k*` + NVMe |
|---:|---|---:|---:|---:|---:|---:|
| 8 | 0.717 / 0.658 / 0.644 | 109.76 | 80.56 (26.6%) | 74.28 (32.3%) | 72.07 (34.3%) | 72.65 (33.8%) |
| 10 | 0.622 / 0.594 / 0.543 | 109.76 | 75.75 (31.0%) | 72.29 (34.1%) | 69.47 (36.7%) | 69.08 (37.1%) |
| 12 | 0.575 / 0.518 / 0.475 | 109.76 | 71.35 (35.0%) | 70.63 (35.6%) | 67.17 (38.8%) | 65.85 (40.0%) |

**Sensitivity.**
- NVMe exposure at 12 threads, `k*` gain: 1.5 ms/miss 35.6%, 3.4 ms 29.8%, 7.0 ms 22.7%. The higher figures are
  tier_sim's whole-row serial upper bounds and put `off` at 131 and 172 ms/token, well above the 97 ms measured.
- GPU term 0 (`--gpu-ms 0`): `off` 95.76 ms, near the ~97 measured, and `k*` 56.63 ms, a 40.9% gain.
  The 14 ms term is therefore conservative for the percentage.
- `k*` cap 1 is the weakest policy in every column (24.8% at 12 threads): with n = 2-3 in 43% of layers, a one-expert
  cap leaves a serial link copy next to the CPU work. The cap 2 arm is within 2.3 points of uncapped.
- The modeled `off` ms/token (109.8) is 13% above the 97 measured, so absolute ms/token here is high by about that
  much. The gain is the quantity to read.

**Gate (pre-registered: predicted gain ≥ 15%): PASS.**
- `k*` predicts **35.6%** (70.6 vs 109.8 ms/token) at 12 native threads, and 32.3% at 8.
- Every policy clears 15% in every column of the sweep at 1.5 ms/miss (the worst is cap 1 at 8 threads, 23.6%),
  and the 7.0 ms/miss upper bound still gives `k*` 20.6-22.7%.
- The absolute predicted level (~71 ms, 14 tok/s) is above the plan's 55-65 ms [estimate]; the P3 items and the
  swizzled layout (64 ms) are what move it there.
- P2 should still measure the interaction the model omits: c_cpu under concurrent NVMe DMA and H2D load.
