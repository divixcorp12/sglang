# NVMe → pinned-host-tier prefetch: investigation handoff (2026-09-26)

**Status: closed, no-go.** Prefetching predicted expert rows from NVMe into the pinned host tier would save about
**2 ms/token (~2%)** with the best realistic predictor. Even a perfect predictor saves at most **~6 ms/token**. Both
are below the 8 ms/token bar that earlier prefetch studies used to decide whether to build.

This document stands alone: it restates the system, the question, every measurement and calculation, the simulator,
and all results, so a reader needs nothing else. File paths are listed at the end so the work can be reproduced.

---

## 1. The system

DeepSeek-V4.1-Flash, EXL3 3.0 bpw experts, served by our SGLang fork on divix01 (one RTX 5090, batch 1 decode).

| Fact | Value |
|---|---|
| MoE layers | 40, all MoE |
| Routed experts | 384 per layer, top-6 per token, plus 1 always-on shared expert |
| One expert row (all tensors of one expert) | 13,315,584 B ≈ 13.3 MB |
| All of one layer's experts | 384 × 13.3 MB = **5.11 GB** |
| All routed experts | 204.5 GB |
| GPU ↔ host link | PCIe Gen3 x16 (the only slot type the HPE DL380 Gen10 has) |
| Link ceiling | copy engine 13.67 GB/s (measured up to 13.75); SM zero-copy reads 12.23 GB/s |

**Memory tiers, fastest first**

1. **GPU hot cache** in VRAM: 14,336 MiB in the recipe (16,282 MiB in the one-off hot launcher). Its residency is
   dynamic. With `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` and `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2`, the in-graph
   updater picks victim slots and the gather writes missed rows straight into them.
2. **Pinned host tier** in RAM: 100 GiB = 8,063 rows. `SGLANG_MOE_PINNED_HOST_NUMA_MB=0:61440,1:40960` places
   60 GiB on NUMA node 0 and 40 GiB on node 1. `ExpertPinnedHostCacheManager.from_model` deals rows round-robin, so
   every layer gets 201–202 rows whatever its miss rate. The RamTier victim rule is LRU-like by stamp, and a slot
   still being filled is never a victim.
3. **NVMe**: two mirrors, and every row read is split across both.
   - nvme0n1 (`/mnt/nvme0`) serves 416 kB per request.
   - nvme3n1 (`/mnt/nvme4`, SPCC, 256 KB maximum transfer) serves 212 kB per request.
   - nvme2n1 serves only 4 kB Engram lookups.
   - All NVMe hangs off socket 1.

**Decode step anatomy.** A step takes ~110 ms, about 2.75 ms per layer. Each layer's chain is
post → W1 → S → CW → MoE:
- **S** (the stream kernel) waits for NVMe-read rows and pulls streamed pieces over PCIe with SM loads.
- **CW** (copy wait) waits for the copy engine's RAM-hit copies. The copy engine runs on stream 141, one
  `cuMemcpyAsync` per tensor, six per row.

A row that misses VRAM crosses the link no matter where it came from. A RAM hit crosses as a copy-engine copy; an
NVMe miss is read into a pinned slot, then pulled by S.

## 2. The question and how it arose

1. **The link.** In `prod-flags-node-20260925-211044`, stream 141's host-to-device copies did not look like they ran
   at line rate. The per-copy analysis showed the link was fine:
   - Of the 50,448 copies, 16,816 large ones (4.4 MB and 8.8 MB) carry 99.7% of the bytes. 73% of those run at
     12–13.75 GB/s; the other 27% run at 10.8–12 GB/s because they overlap `exl3_ram_miss_lease_stream_kernel`,
     whose SM reads share the link.
   - During that overlap the link is ~96% saturated (13.2 GB/s total), so the true contention loss is ~0.8 ms/step.
   - The 33,632 small copies (4.6–20 KB) run at 1.5–5 GB/s because per-copy overhead dominates.
   - The large idle stretches of the link are the real loss.
2. **The idea.** Use idle link time for prefetch. Prefetch into VRAM was already studied and closed (§4 below).
3. **The refinement.** Instead of prefetching into VRAM, start the **NVMe read into pinned host memory earlier**,
   so the wait is hidden and the GPU link is never touched. First the whole next layer; then, once the arithmetic
   ruled that out (§3.1), a predicted subset.

## 3. Calculations from measured data

### 3.1 Whole next layer: impossible by bandwidth

- One layer is 384 × 13.3 MB = **5.11 GB**.
- NVMe while busy reads 3.3–3.9 GB/s per mirror. Decode measured 3.33 + 2.67 = **6.0 GB/s** combined; the best
  case is ~7.5 GB/s.
- One layer therefore takes **0.68–0.85 s** to read, against **~2.75 ms** of decode time per layer: **250–310× too
  slow**.
- Capacity rules it out too: the 100 GiB pinned tier holds only about 21 layers' worth of experts.

### 3.2 What NVMe bandwidth is spare (makes a subset feasible)

- Mirrors are busy only **23% (nvme0n1) and 29% (nvme3n1)** of decode. The low averages mean nothing is known yet,
  not a shallow queue: when a layer's misses are known, both drives run flat out at their x4 link rate.
- Per 110 ms step at ~6 GB/s, NVMe can move ~0.66 GB ≈ 50 rows. Decode uses ~11.3 rows/token (§5.3), leaving
  **~38 rows per step** of headroom.
- One row takes ~2 ms at best across both mirrors (~2.5 ms fits the model best, §5.3).

### 3.3 Link and NVMe time in a decode step (trace `prod-flags-node-20260925-211044`, steady state)

Steady state is steps 22–109 of session 1 (88 steps); the cold start is steps 7–21. Node-mode tracing inflates
each step by ~7 ms, so per-kernel and per-copy numbers are valid but absolute step times read high.

| Link state | ms/step | Link rate |
|---|---:|---:|
| Copy engine copying, GPU in CW | 55.0 | 13.62 GB/s |
| Copy engine copying, GPU in S | 24.2 | 12.79 GB/s |
| No copy, GPU in S (NVMe waits; S's own SM pulls) | 16.9 | 7.9 GB/s (25% of samples under 5% RX) |
| No copy, compute | 14.6 (+2.6 idle, +0.65 other waits) | 0.9 GB/s |

- The PCIe RX calibration comes from clean copies: 13.69 GB/s reads as 43.2% RX, so 100% ≈ 31.7 GB/s. Samples
  every 100 µs; treat as ±2–3%.
- **S (NVMe wait) totals 41.1 ms/step steady and 48.5 ms/step cold.** 17.35 ms of it has no copy in flight; 5.2 ms
  of that is in layers with no RAM-hit copies at all.
- **Worst S by layer (steady):** L0 2.70 ms/step (median 1.33 — it waits *every* step), L19 2.11, L39 2.05, L23 1.96,
  L13 1.67. Apart from L0 these are rare stalls: median 0.08–0.09 ms. In the cold start, layers 0, 1, 13, 19 and 27
  all have a median above 1 ms.
- **Stream-141 gaps within a step:**

  | Gap | Count/step | ms/step | GPU mostly in |
  |---|---:|---:|---|
  | <10 µs | 415 | 0.67 | S / CW (within rows) |
  | 10–100 µs | 1.4 | 0.05 | – |
  | 0.1–0.5 ms | 22.6 | 8.64 | compute (90%) |
  | 0.5–1 ms | 3.5 | 2.62 | compute (75%) |
  | 1–2 ms | 4.4 | 6.20 | S (65%) |
  | >2 ms | 4.6 | 17.35 | S (70%) |

  Gaps of 1 ms or more total 20.0 ms/step, all between layers: 69% in S, 21% in compute, 9% GPU idle.
- **Layer 0 can't be predicted within a token.** Its gate is learned (`layers.0.ffn.gate.weight`, `.bias`, no
  `tid2eid`), the config has no hash layers, and nothing runs before it to hide reads under. That makes the only
  layer with a routine NVMe wait the one that within-token lookahead can't reach.

### 3.4 Ceiling arithmetic

Removing a RAM miss does **not** remove its link time. The row still crosses PCIe, just as a copy-engine copy
instead of an SM pull. Prefetch can only remove the NVMe wait **in excess of** the layer's link-bound copy time.

- The model (§5) puts the time from a layer's RAM-hit copies ending to the layer ending at 18.35 ms/token, against
  17.35 ms measured "no copy, S".
- Of that, 11.33 missed rows × 1.07 ms ≈ 12.1 ms is those rows' own transfer over the link, which is needed anyway.
- That leaves **~6.2 ms/token of truly exposed NVMe wait**, the most a pinned-tier prefetch could ever remove.

## 4. Prior closed studies

These are from `DSV41_REFERENCE.md` §25.4. All are prefetch **into VRAM**, except the residency replay.

| Study | Result |
|---|---|
| Route-only predictors | Useless. Prefetch into VRAM breaks even at precision ~0.25 and is worth building at ≥0.4 |
| Native-gate lookahead | Layer T's own gate applied to layer T−1's input: 0.68 rank-1 precision on non-*VRAM*-resident rows |
| Native prefetch, built (`SGLANG_DSV41_ENABLE_NATIVE_PREFETCH`, off) | Precision 0.73 live, byte-identical, but **123.2 vs 120.6 ms/token**. Only ~0.36 ms of compute sits between the post and the next gather, so ~0.62 ms of each ~1 ms copy was exposed |
| Early-post prefetch | Replay at best 117.8 vs 120.6 ms/token, below the 8 ms bar. 68% of layers have no NVMe wait to hide under |
| Residency policy (Track B) | Exact replay matches measurement. The current policy is the best online policy found. +1 GiB of hot cache saves 2–2.7 misses/token. Belady's bound is 31 misses/token lower |
| Per-layer RAM split (branch `cc/pinned-layer-weights`) | No gain: decode NVMe rows went up 3.8%. Probably prefill churn |

The pinned-tier destination avoids the two things that killed prefetch into VRAM: GPU link contention, and too
little compute to hide the copy under. Its costs are different: NVMe queueing, pinned-tier evictions, and extra
memory traffic on node 0. §25.4 measured that CPU memory load on the GPU's NUMA node cuts H2D bandwidth by 27–43%.

## 5. The replay study

### 5.1 Data

- **Router capture** at
  `divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture/`:
  - `stages.jsonl`: the route log, with each forward's routes, hot set and phase;
  - `router.x.bin`, `router.w.bin`, `router.seq.bin`: each layer's router input, per decode step.
  - It has 6,153 decode steps. Its measured RAM misses (`graph_step` counts, same run) are **11.265 per decode
    token**.
- **Gate weights** from `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40`
  (`layers.{i}.ffn.gate.weight`, `.bias`), read by byte range so a `ulimit -v` run can afford it.
- **Self-check:** applying each layer's gate to its own captured input reproduces **0.999984** of the captured
  route sets, so the capture and weights are consistent.

### 5.2 Predictor rankings (`gate_rankings.py`)

For every decode step s and target layer T, the script stores the top-12 experts of
`softplus(x_src · W_T^T).sqrt() + bias_T`, the biased score the model routes on, for horizons h = 0..4:
- **h ≥ 1:** `x_src` is the same token's layer T−h input. For T < h it is the same request's previous token at layer
  40+T−h; if the previous step belongs to another request, the entry is invalid.
- **h = 0:** layer T's gate on its own input. Ranks 1–6 are the routes; ranks 7–12 are the near misses the next-token
  predictor uses.

Output: `rank.npz`, with `order[6153, 40, 5, 12]` int16 and `valid[6153, 40, 5]`.

### 5.3 Simulator (`pinned_prefetch_replay.py`)

**Tier.** This is the RamTier replay from `prefill-evict/ram_replay.py` (previously validated: 11.33 against 11.26
measured RAM misses per token on varied24), extended with fills:
- Per-layer capacity comes from `ram_rows_per_layer(8063, 40, 384)`, round-robin.
- A read claims its slot at issue. The row is ready when its read completes, and a filling slot is never a victim.
- Prefill forwards are replayed untimed with the base policy (chunks of 64, admitted at MRU); the NVMe queue drains
  before each one.

**NVMe queue.**
- One server for both mirrors, because every row is split across them. `--nvme-row-ms` per row (default 2.5), served
  in `--pieces` pieces (default 8), with optional `--nvme-lat-ms` from issue to first piece.
- `fifo` serves pieces in issue order.
- `prio` serves demand pieces before speculative ones. A speculative piece already started finishes (non-preemptive
  at piece granularity). A speculative row still queued when a demand for it arrives is promoted into the demand
  queue.
- Demand delay: a demand row is delayed by the speculative piece in service when it arrives, plus every speculative
  piece served before it completes.

**Layer timing.**
- Layer T posts at time t. Its VRAM-miss rows cross the link one after another at `--link-row-ms` = 1.07 ms each
  (copy-engine rate for a 13.3 MB row), RAM hits first.
- An NVMe row is pulled as its pieces land, so the layer's S+CW ends at
  `max(t + n_rows × 1.07, last NVMe arrival + 1.07 / pieces)`.
- **Exposed NVMe wait** = end − link-bound end. This is the only thing a pinned-tier prefetch can remove.
- The next layer posts `--compute-ms` = 0.35 later; each step adds `--step-ms` = 2.0 of tail.

**Predictors.** All are issued at a layer's post, once its router input exists, and each picks the first K
candidates not already in RAM, filling, or VRAM-hot.

| Predictor | What it predicts | Lead |
|---|---|---|
| `gate` | Layer T+h's gate on layer T's input, from its top-6 (`--depth 6`) | h layers (~2.75 ms per layer) |
| `near` | Next token, same layer: layer T's gate on this token's layer-T input, ranks 1–12. Ranks 1–6 were just admitted, so in effect the near misses | 1 step (~110 ms) |
| `freq` | Next token: an online per-layer decayed count (0.98) of VRAM misses | 1 step |
| `oracle` | The target's true VRAM-miss rows that are not in RAM | h layers or next step |
| `noisy` | Oracle-sized issue where each row is right with probability p, otherwise a random wrong expert | as chosen |

**Options.**
- `--budget`: speculative rows per decode step (default 16).
- `--admit mru|cold`: `cold` stamps a prefetched row 0 until first use, so an unused prefetch is the next victim.
- `--spec-share N`: once a layer holds N unused prefetched rows, a new prefetch evicts the LRU of those, so a wrong
  prefetch displaces no demand row (as `SGLANG_DSV41_ENABLE_PREFILL_SHARE` does for prefill).
- `--layers`: restrict targets, e.g. layer 0 only.

**Metrics.**
- **Precision (target):** the prefetched row was used at exactly the (step, layer) it targeted.
- **Precision (any use):** it was used at any later point before eviction.
- **Harmful evictions:** demand misses of rows that a speculative admission had evicted.
- **Demand delayed / mean delay:** as defined for the queue above.
- **Late:** a prefetched row was demanded while still filling.
- **Saved ms/token** = baseline exposed (6.24) − arm exposed.

### 5.4 Calibration

The timing model is first order. Its constants were picked by sweeping NVMe latency and row time with no prefetch
(`calib.arms`) and matching measured quantities. RAM misses are the same in every row (11.33/token; 10.81 steady;
VRAM misses 78.78 against 78.77 measured).

| NVMe model | Exposed ms/tok | After-hits ms/tok (meas. 17.35) | Step ms (meas. ~110) | NVMe busy (meas. 23–29%) |
|---|---:|---:|---:|---:|
| **row 2.5 ms, 8 pieces, 0 latency (chosen)** | **6.24** | **18.35** | 106.5 | **26.7%** |
| row 2.0 ms, 1 piece | 7.64 | 19.75 | 107.9 | 21.1% |
| row 2.5 ms, 1 piece | 11.99 | 24.10 | 112.3 | 25.3% |
| row 3.5 ms, 8 pieces | 14.43 | 26.54 | 114.7 | 34.7% |
| row 2.5 ms, +0.5 ms latency | 8.88 | 20.99 | 109.2 | 26.0% |
| row 2.5 ms, +1.0 ms latency | 12.47 | 24.58 | 112.8 | 25.2% |
| row 2.5 ms, +1.5 ms latency | 16.24 | 28.35 | 116.5 | 24.4% |
| row 2.5 ms, +2.0 ms latency | 20.45 | 32.56 | 120.7 | 23.5% |
| row 2.5 ms, +3.0 ms latency | 29.34 | 41.45 | 129.6 | 21.9% |
| row 3.5 ms, +1.0 ms latency | 22.43 | 34.54 | 122.7 | 32.4% |

- The chosen model matches after-hits wait within 1 ms, NVMe busy within range, and step time within ~3%.
- **Known miss:** it is calibrated only in aggregate. It puts layer 0's exposed wait at 0.26 ms/token (top per-layer
  exposed: L1 0.257, L0 0.255, L5 0.199, L37 0.188, L4 0.181), against a measured L0 S of 2.7 ms/step. The model has
  no NVMe latency tail, and L0 has no preceding compute.
- **Baseline per-layer RAM misses/token:** L0 0.704, L1 0.549, L2 0.326, L3 0.270, L4 0.347. Total 11.37.

## 6. Results

Every arm, with the priority queue, a budget of 16 speculative rows per step, MRU admission, and the chosen
calibration, unless the name says otherwise:
- `_fifo`: FIFO queue.
- `_cold`: cold admission.
- `_bN`: a budget of N rows per step.
- `_sN`: spec-share N.
- `_L0`: layer 0 only.
- `r20_`: 2.0 ms per row.
- `nobudget`: unlimited budget.

"Saved" is against `none` (11.33 misses, 6.24 exposed ms).

| arm | prec (target) | prec (any use) | spec rows/tok | RAM misses/tok | saved/tok | harmful evict/tok | demand delayed/tok | mean delay ms | exposed ms/tok | saved ms/tok | late/tok | NVMe busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.00 | 0.00 | 0.00 | 11.33 | +0.00 | 0.00 | 0.00 | 0.00 | 6.24 | +0.00 | 0.00 | 0.27 |
| gate_h1_k1 | 0.40 | 0.67 | 9.22 | 7.40 | +3.93 | 3.16 | 2.02 | 0.12 | 4.23 | +2.02 | 2.72 | 0.40 |
| gate_h1_k1_b40 | 0.41 | 0.68 | 9.74 | 7.04 | +4.29 | 3.18 | 2.20 | 0.12 | 4.04 | +2.20 | 2.95 | 0.40 |
| gate_h1_k1_s2 | 0.31 | 0.36 | 11.65 | 7.32 | +4.01 | 2.84 | 2.41 | 0.13 | 4.27 | +1.98 | 2.72 | 0.45 |
| gate_h1_k1_s4 | 0.32 | 0.41 | 11.10 | 7.22 | +4.11 | 2.83 | 2.28 | 0.12 | 4.16 | +2.09 | 2.67 | 0.44 |
| gate_h1_k1_s4_b40 | 0.34 | 0.42 | 11.82 | 6.80 | +4.53 | 2.82 | 2.50 | 0.12 | 3.94 | **+2.30** | 2.94 | 0.45 |
| gate_h1_k1_s4_fifo | 0.32 | 0.41 | 11.10 | 7.22 | +4.11 | 2.83 | 2.11 | 1.18 | 5.09 | +1.15 | 2.66 | 0.44 |
| gate_h1_k1_s8 | 0.34 | 0.48 | 10.44 | 7.17 | +4.16 | 2.81 | 2.15 | 0.12 | 4.07 | +2.18 | 2.65 | 0.42 |
| gate_h1_k1_L0 | 0.00 | 0.25 | 0.59 | 11.51 | -0.18 | 0.33 | 0.10 | 0.04 | 6.38 | -0.13 | 0.00 | 0.28 |
| gate_h1_k2 | 0.38 | 0.65 | 9.94 | 7.34 | +3.99 | 3.27 | 1.92 | 0.13 | 4.46 | +1.79 | 2.91 | 0.41 |
| gate_h1_k2_fifo | 0.38 | 0.65 | 9.94 | 7.34 | +3.99 | 3.27 | 1.66 | 1.71 | 5.72 | +0.53 | 2.88 | 0.41 |
| gate_h1_k2_s4 | 0.28 | 0.35 | 12.68 | 7.26 | +4.07 | 2.90 | 2.39 | 0.13 | 4.50 | +1.75 | 2.80 | 0.48 |
| gate_h2_k1 | 0.30 | 0.59 | 9.76 | 8.36 | +2.97 | 3.52 | 2.31 | 0.13 | 4.42 | +1.82 | 0.78 | 0.43 |
| gate_h2_k1_s4 | 0.22 | 0.29 | 12.53 | 8.06 | +3.27 | 2.95 | 2.67 | 0.13 | 4.25 | +1.99 | 0.78 | 0.49 |
| gate_h2_k2 | 0.28 | 0.57 | 10.61 | 8.39 | +2.94 | 3.68 | 2.52 | 0.13 | 4.67 | +1.58 | 1.02 | 0.45 |
| gate_h2_k2_L0 | 0.00 | 0.23 | 0.85 | 11.58 | -0.25 | 0.43 | 0.28 | 0.12 | 6.44 | -0.20 | 0.00 | 0.29 |
| gate_h2_k2_b4 | 0.20 | 0.50 | 3.99 | 10.67 | +0.66 | 1.72 | 1.03 | 0.13 | 5.86 | +0.38 | 0.27 | 0.35 |
| gate_h2_k2_b8 | 0.24 | 0.54 | 7.43 | 9.65 | +1.68 | 2.97 | 1.70 | 0.13 | 5.31 | +0.94 | 0.60 | 0.41 |
| gate_h2_k2_b40 | 0.30 | 0.59 | 11.84 | 7.72 | +3.61 | 3.75 | 2.85 | 0.13 | 4.39 | +1.85 | 1.31 | 0.47 |
| gate_h2_k2_cold | 0.17 | 0.19 | 15.16 | 8.57 | +2.76 | 3.83 | 3.36 | 0.14 | 5.06 | +1.19 | 1.13 | 0.56 |
| gate_h2_k2_fifo | 0.28 | 0.57 | 10.61 | 8.39 | +2.94 | 3.68 | 2.04 | 1.85 | 6.64 | -0.40 | 0.53 | 0.45 |
| gate_h2_k2_s4 | 0.19 | 0.25 | 14.35 | 8.19 | +3.14 | 3.00 | 3.12 | 0.14 | 4.65 | +1.59 | 1.04 | 0.54 |
| gate_h2_k2_s4_b40 | 0.20 | 0.26 | 17.53 | 7.24 | +4.09 | 3.15 | 3.95 | 0.14 | 4.35 | +1.89 | 1.49 | 0.59 |
| gate_h3_k1 | 0.24 | 0.54 | 10.22 | 8.97 | +2.36 | 3.81 | 2.72 | 0.13 | 4.70 | +1.54 | 0.30 | 0.46 |
| gate_h3_k2 | 0.23 | 0.52 | 11.07 | 9.00 | +2.33 | 3.92 | 2.94 | 0.14 | 4.91 | +1.34 | 0.47 | 0.48 |
| gate_h4_k1 | 0.21 | 0.51 | 10.74 | 9.35 | +1.98 | 3.99 | 3.02 | 0.13 | 4.97 | +1.27 | 0.15 | 0.48 |
| gate_h4_k2 | 0.20 | 0.49 | 11.62 | 9.38 | +1.95 | 4.12 | 3.29 | 0.14 | 5.16 | +1.08 | 0.26 | 0.50 |
| near_k1 | 0.03 | 0.44 | 14.62 | 9.76 | +1.58 | 4.55 | 4.10 | 0.14 | 5.21 | +1.04 | 0.00 | 0.58 |
| near_k2 | 0.03 | 0.42 | 15.26 | 10.10 | +1.23 | 4.59 | 4.50 | 0.14 | 5.53 | +0.71 | 0.00 | 0.60 |
| near_k2_L0 | 0.02 | 0.41 | 1.02 | 11.36 | -0.03 | 0.40 | 0.44 | 0.13 | 6.31 | -0.07 | 0.00 | 0.29 |
| near_k2_cold | 0.02 | 0.04 | 15.93 | 10.84 | +0.49 | 3.68 | 5.16 | 0.15 | 6.21 | +0.04 | 0.00 | 0.63 |
| near_k2_fifo | 0.03 | 0.42 | 15.26 | 10.10 | +1.23 | 4.59 | 3.09 | 2.65 | 11.12 | -4.87 | 0.00 | 0.57 |
| near_k2_s4 | 0.02 | 0.08 | 15.92 | 10.50 | +0.83 | 3.36 | 4.81 | 0.15 | 5.88 | +0.36 | 0.00 | 0.62 |
| near_k4 | 0.03 | 0.41 | 15.33 | 10.20 | +1.13 | 4.62 | 4.60 | 0.14 | 5.62 | +0.62 | 0.00 | 0.60 |
| near_k4_L0 | 0.02 | 0.40 | 1.13 | 11.38 | -0.05 | 0.44 | 0.50 | 0.13 | 6.33 | -0.09 | 0.00 | 0.29 |
| freq_k1 | 0.00 | 0.46 | 0.02 | 11.32 | +0.01 | 0.00 | 0.09 | 0.14 | 6.24 | +0.00 | 0.00 | 0.27 |
| freq_k2 | 0.00 | 0.47 | 0.02 | 11.32 | +0.01 | 0.00 | 0.09 | 0.14 | 6.24 | +0.00 | 0.00 | 0.27 |
| freq_k2_cold | 0.00 | 0.40 | 0.02 | 11.32 | +0.01 | 0.00 | 0.09 | 0.14 | 6.24 | +0.00 | 0.00 | 0.27 |
| freq_k4_L0 | 0.00 | 0.33 | 0.00 | 11.33 | +0.00 | 0.00 | 0.00 | 0.15 | 6.24 | -0.00 | 0.00 | 0.27 |
| oracle_h1 | 1.00 | 1.00 | 9.70 | 1.66 | +9.67 | 0.77 | 0.06 | 0.11 | 2.00 | +4.25 | 6.68 | 0.28 |
| oracle_h1_fifo | 1.00 | 1.00 | 9.70 | 1.66 | +9.67 | 0.77 | 0.06 | 1.64 | 2.00 | +4.25 | 6.68 | 0.28 |
| oracle_h2 | 1.00 | 1.00 | 9.70 | 1.65 | +9.68 | 0.78 | 0.08 | 0.14 | 1.21 | +5.04 | 1.88 | 0.28 |
| oracle_h4 | 1.00 | 1.00 | 9.71 | 1.64 | +9.69 | 0.78 | 0.16 | 0.14 | 0.97 | +5.27 | 0.31 | 0.28 |
| oracle_next | 1.00 | 1.00 | 9.63 | 1.73 | +9.60 | 0.79 | 0.46 | 0.14 | 0.99 | +5.25 | 0.00 | 0.28 |
| oracle_next_L0 | 1.00 | 1.00 | 0.70 | 10.64 | +0.69 | 0.01 | 0.17 | 0.11 | 6.00 | +0.24 | 0.00 | 0.27 |
| oracle_next_nobudget | 1.00 | 1.00 | 11.21 | 0.16 | +11.17 | 0.08 | 0.04 | 0.13 | 0.08 | **+6.16** | 0.04 | 0.28 |
| noisy_h2_p10 | 0.10 | 0.32 | 10.82 | 12.18 | -0.85 | 4.83 | 6.91 | 0.14 | 7.63 | -1.39 | 0.49 | 0.53 |
| noisy_h2_p10_s4 | 0.10 | 0.13 | 9.76 | 10.57 | +0.76 | 3.88 | 5.66 | 0.14 | 6.34 | -0.09 | 0.40 | 0.48 |
| noisy_h2_p25 | 0.25 | 0.44 | 10.66 | 10.36 | +0.97 | 4.36 | 5.36 | 0.14 | 6.50 | -0.25 | 1.08 | 0.49 |
| noisy_h2_p25_fifo | 0.25 | 0.44 | 10.66 | 10.36 | +0.97 | 4.36 | 3.72 | 2.09 | 10.48 | -4.24 | 0.47 | 0.48 |
| noisy_h2_p25_s4 | 0.25 | 0.28 | 9.73 | 9.07 | +2.26 | 3.63 | 4.47 | 0.14 | 5.45 | +0.79 | 0.90 | 0.45 |
| noisy_h2_p40 | 0.40 | 0.56 | 10.48 | 8.51 | +2.82 | 3.79 | 3.99 | 0.14 | 5.28 | +0.96 | 1.49 | 0.45 |
| noisy_h2_p60 | 0.60 | 0.71 | 10.22 | 6.13 | +5.20 | 2.94 | 2.36 | 0.14 | 3.77 | +2.47 | 1.82 | 0.39 |
| noisy_h2_p80 | 0.80 | 0.86 | 9.95 | 3.92 | +7.41 | 1.99 | 1.15 | 0.14 | 2.38 | +3.87 | 1.89 | 0.34 |
| noisy_next_p10 | 0.10 | 0.32 | 10.81 | 12.24 | -0.91 | 4.86 | 4.43 | 0.14 | 7.14 | -0.90 | 0.00 | 0.54 |
| noisy_next_p10_s4 | 0.10 | 0.13 | 9.67 | 10.57 | +0.76 | 3.85 | 3.42 | 0.14 | 5.95 | +0.30 | 0.00 | 0.48 |
| noisy_next_p25 | 0.25 | 0.44 | 10.57 | 10.28 | +1.05 | 4.30 | 3.41 | 0.14 | 5.67 | +0.57 | 0.00 | 0.49 |
| noisy_next_p25_fifo | 0.25 | 0.44 | 10.57 | 10.28 | +1.05 | 4.30 | 2.48 | 1.99 | 8.81 | -2.56 | 0.00 | 0.48 |
| noisy_next_p25_s4 | 0.25 | 0.28 | 9.67 | 9.09 | +2.24 | 3.64 | 2.75 | 0.13 | 4.89 | +1.35 | 0.00 | 0.45 |
| noisy_next_p40 | 0.40 | 0.56 | 10.40 | 8.50 | +2.83 | 3.77 | 2.61 | 0.14 | 4.47 | +1.77 | 0.00 | 0.45 |
| noisy_next_p60 | 0.60 | 0.71 | 10.14 | 6.20 | +5.13 | 2.96 | 1.75 | 0.13 | 3.07 | +3.17 | 0.00 | 0.40 |
| r20_none | 0.00 | 0.00 | 0.00 | 11.33 | +0.00 | 0.00 | 0.00 | 0.00 | 2.89 | +0.00 | 0.00 | 0.22 |
| r20_gate_h1_k1 | 0.40 | 0.67 | 9.22 | 7.40 | +3.93 | 3.16 | 1.25 | 0.09 | 1.78 | +1.11 | 1.72 | 0.33 |
| r20_gate_h1_k1_s4 | 0.32 | 0.41 | 11.10 | 7.22 | +4.11 | 2.83 | 1.41 | 0.09 | 1.76 | +1.13 | 1.71 | 0.36 |
| r20_gate_h2_k1 | 0.30 | 0.59 | 9.76 | 8.36 | +2.97 | 3.52 | 1.46 | 0.09 | 1.90 | +0.99 | 0.48 | 0.36 |
| r20_oracle_h1 | 1.00 | 1.00 | 9.70 | 1.66 | +9.67 | 0.77 | 0.03 | 0.10 | 0.81 | +2.08 | 4.19 | 0.23 |
| r20_oracle_next | 1.00 | 1.00 | 9.63 | 1.73 | +9.60 | 0.79 | 0.30 | 0.08 | 0.46 | +2.43 | 0.00 | 0.23 |

The r20 rows are under the alternative 2.0 ms-per-row calibration, so their baseline is 2.89 exposed ms, not 6.24.

## 7. What the results say

1. **The ceiling is low.** A perfect predictor with no budget removes almost every RAM miss (11.33 → 0.16/token)
   but saves only **6.16 ms/token**, because the rows still cross the link (§3.4). With the realistic 16-row budget,
   the oracle saves 4.3–5.3 ms/token. Under the 2.0 ms/row calibration the whole prize halves (baseline 2.89 ms,
   oracle 2.1–2.4).
2. **The best realistic arm is one-layer gate lookahead with K=1**, optionally with a 4-row spec share and a 40-row
   budget:
   - It saves **~2.0–2.3 ms/token** by removing ~4 misses/token, 1.1 ms/token under the 2.0 ms/row calibration.
   - It costs ~123–157 MB/token of extra NVMe reads, which roughly doubles NVMe traffic from the 151 MB/token
     baseline, and ~3 harmful evictions/token.
   - About 2.7 prefetched rows/token are still filling when demanded.
3. **Precision is lower than §25.4's 0.68.** Here a candidate must be absent from *RAM*, not just VRAM, and those
   experts are colder. The gate reaches 0.40 at h=1, falling to 0.30, 0.24 and 0.21 at h=2, 3 and 4. Longer lead
   loses precision faster than it gains hiding time.
4. **Next-token predictors are useless.**
   - `near` has 0.03 precision at its exact target.
   - `freq` issues almost nothing: its candidates are already in RAM, which the LRU tier already exploits.
5. **Nothing helps layer 0.** Even a perfect layer-0-only predictor saves 0.24 ms/token in the model, and every
   real layer-0 arm is negative. (The model under-attributes L0; see §5.4.)
6. **Demand reads must have priority.** FIFO raises mean demand delay from ~0.12 ms to 1.2–2.7 ms:
   - gate h1 falls from +1.79 to +0.53;
   - gate h2 from +1.58 to −0.40;
   - near_k2 from +0.71 to −4.87;
   - a p=0.25 predictor from about 0 to −2.6 to −4.2.
   Speculative reads have to be a strictly lower priority class at the reader, and must never delay a real miss by
   more than one piece.
7. **Wrong prefetches cost mainly evictions,** ~3 harmful evictions/token. Capping unused prefetched rows per layer
   (spec-share 4) recovers a little. Cold admission is worse.
8. **Break-even precision** at the exact target, priority queue:
   - ~**0.28** at h=2;
   - ~**0.19** for next-token lead;
   - below **0.12** with spec-share 4.
   With FIFO, precision 0.25 already loses 2.6–4.2 ms/token.
9. **Not modelled:** NVMe busy rises from 27% to 40–45%. That is ~1.2–1.5 GB/s of extra DMA writes into the pinned
   tier, 60% of which sits on NUMA node 0, where §25.4 measured CPU memory load cutting H2D by 27–43%. This can only
   make things worse.

## 8. Decision and what to try instead

**Don't build NVMe → pinned prefetch as a project.** At best it is worth ~2 ms/token (2%), at the edge of what a
single A/B arm can resolve.

**If you want the one cheap arm:** the built-but-off native prefetch (`SGLANG_DSV41_ENABLE_NATIVE_PREFETCH`) already
runs a layer's gate on the previous layer's input. If its not-in-RAM candidates can be posted to the RAM-miss
service as **low-priority** reads with little plumbing, try h=1, K=1, spec-share 4.
- Expect ~2 ms/token (1 ms/token if rows really read at 2.0 ms).
- Verify with node-mode traces, not a single A/B.
- The reader must serve demand pieces first; with FIFO this loses.

**What moves more, in this model:**
- **Fewer RAM misses:** a bigger pinned tier or hot cache (+1 GiB of hot cache saves 2–2.7 misses/token), or keeping
  prefill from evicting decode's rows.
- **Faster NVMe:** a third mirror, or keeping co-tenants off the SPCC mirror. A Ray cluster's temp and spill
  directory `/mnt/nvme4/ray_tmp` is the likely source of 200–576 MB/s write bursts during decode, not proven.
- **Fewer bytes over the link per token:** the link carries ~1.14 GB/step and is saturated whenever it copies.

## 9. Related measurements from the same session

- **Merging each row's six copies into fewer:** investigated, not built.
  - The six copies are per tensor, not deliberate chunking: `CopyEngine::issue`,
    `exl3_ram_miss_host.cpp:3398-3405`, issues one copy per `CopyEntry` from `ExpertRowSegments`.
  - `cuMemcpyBatchAsync` gave no gain.
  - One copy per row would save ~0.55–1.44 ms/token but needs a slab layout redesign.
  - An agent is building the cheaper variant: the CW kernel fetches the four small tensors (44.5 KB) with SM loads,
    and the lease is released after both the large copies and those reads.
- **Async hot-cache promotions:** no-go. In the recipe, promotion is in-graph (the insert-on-miss updater) and costs
  ~0.35–0.4 ms/step. There is no host path left to make async.
- **Link-loss ranking** (`divix01:/mnt/nvme1/prefill-opt/link-idle/`):
  - idle link time is at most ~23.7 ms/token, mostly unusable;
  - small-segment merging is worth 0.66 ms on the critical path, 1.44 ms at most;
  - slow-copy contention is ~0.8 ms.

## 10. Reproducing

Everything runs on divix01, CPU only, from a private worktree of `/data/models/slang/sglang` at `8b43ed3dea` or
later. The one used was `/data/models/slang/nvfp4-work/wt-prefetch-replay`.

```bash
WT=/data/models/slang/nvfp4-work/wt-prefetch-replay
O=/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture
M=/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40
cd $WT; ulimit -v 16000000; export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= PYTHONPATH=$WT/python
PY=/data/models/slang/.venv/bin/python

# 1. Gate rankings (h = 0..4, depth 12); prints the h=0 self-check (0.999984)
taskset -c 18-25 $PY analysis/dsv41-drive/prefetch-replay/gate_rankings.py $O/stages.jsonl $O/router $M \
    --out /mnt/nvme1/prefetch-replay/rank.npz
# 2. Baseline, no prefetch (11.33 RAM misses/token, 6.24 exposed ms/token)
taskset -c 26-27 $PY analysis/dsv41-drive/prefetch-replay/pinned_prefetch_replay.py $O/stages.jsonl \
    --out /mnt/nvme1/prefetch-replay/none.json
# 3. Arms (one line per arm in the .arms files); then the table
OMP_NUM_THREADS=1 taskset -c 18-35,54-63 $PY analysis/dsv41-drive/prefetch-replay/run_arms.py $O/stages.jsonl \
    /mnt/nvme1/prefetch-replay/rank.npz analysis/dsv41-drive/prefetch-replay/main.arms \
    /mnt/nvme1/prefetch-replay/main.jsonl --workers 12
#    (b2.arms the same way into b2.jsonl), then one table over both:
$PY analysis/dsv41-drive/prefetch-replay/summarize.py /mnt/nvme1/prefetch-replay/main.jsonl \
    /mnt/nvme1/prefetch-replay/b2.jsonl
```

- Calibration: run `pinned_prefetch_replay.py` once per line of `calib.arms`, as `calib.sh` does. A single arm is,
  for example, `pinned_prefetch_replay.py $O/stages.jsonl --ranks rank.npz --predictor gate --h 1 --k 1
  --spec-share 4 --budget 40`.
- RamTier validation: `analysis/dsv41-drive/prefill-evict/ram_replay.py $O/stages.jsonl --arms base`.

## 11. File inventory

**Repo** (`analysis/dsv41-drive/prefetch-replay/`, on master):

| File | Contents |
|---|---|
| `pinned_prefetch_replay.py` | The simulator (§5.3) |
| `gate_rankings.py` | Gate rankings (§5.2) |
| `run_arms.py` | Parallel arm runner |
| `summarize.py` | Builds the results table |
| `main.arms`, `b2.arms`, `calib.arms` | Arm definitions |
| `all_table.md` | The full table of §6 |

The simulator imports `scripts/dsv41/tier_sim.py` (`load_forwards`, `ram_rows_per_layer`), and `gate_rankings.py`
also imports `analysis/dsv41-drive/router-capture/router_score.py` and `prefetch_sim.py`.

**divix01 outputs** (`/mnt/nvme1/prefetch-replay/`):
- `rank.npz`;
- `none.json`, the baseline with per-layer exposed ms and misses;
- `main.jsonl` and `b2.jsonl`, all arms' full reports;
- `calib_*.json`;
- `base_rc.json`, the RamTier validation;
- `all_table.md`, `main_table.md`;
- run scripts `ranks.sh`, `none.sh`, `arms.sh`, `calib.sh`, `base.sh`.

**Traces and trace analyses:**
- `/mnt/nvme1/dsv41-nsys/prod-flags-node-20260925-211044.sqlite` (the production recipe, node mode);
- `/mnt/nvme1/prefill-opt/h2d_rate.py`, `h2d_slow.py` (per-copy link analysis);
- `/mnt/nvme1/prefill-opt/link-idle/calc1..5*.py` and `calc*.txt` (link states, gaps, slow copies, small segments,
  S per layer).

**Background** (`DSV41_REFERENCE.md`):
- §18.6 (lease / `host_use` contract, promotions);
- §25.4 (closed prefetch studies, link and NUMA);
- §27.3 (decode link use);
- §27.4 item 5 (per-layer RAM split);
- §27.10 (NVMe load and mirrors).

## 12. Open questions, if this is ever revisited

- **The NVMe tail.** L0's measured 2.7 ms/step of S is ~10× what the model attributes. A per-read latency
  distribution (from the reader's own timestamps) would say whether a tail exists that prefetch could hide better
  than the mean-based model suggests.
- **The NUMA cost** of ~1.3 GB/s of extra DMA writes into node 0 on H2D bandwidth is unmeasured.
- **Better predictors:** anything that beats 0.40 precision on not-in-RAM rows at h=1. The noisy-oracle rows give the
  payoff curve: p=0.6 saves ~2.5 ms, p=0.8 saves ~3.9 ms.
