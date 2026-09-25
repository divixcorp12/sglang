# DSV4.1 decode: computing experts as they land

**Date:** 2026-09-25. **Status:** plan only, not started.
**Continues:** `2026-09-25-dsv41-copy-compute-overlap.md`. This plan specifies that doc's item 1c.
**Depends on:**
- 1b: the copy engine, on `cc/copy-engine` and being verified now.
- The h=1 native-gate prefetch in `2026-09-25-dsv41-router-capture.md`.

**Base:** `cc/dsv41-pinned-numa` at `d2e9e2f596`, which has layer fusion and the Engram device wait on by default. It runs
at 125.5 ms/token.

## 1. Where we are

Each layer runs its routed experts only after every missed row has arrived. The chain on the single decode stream is:

```
attention -> route -> post -> W1 -> C1 (RAM-hit rows) -> S (NVMe rows) -> F -> exl3_moe (all 6 routed experts) -> gather -> next layer
```

- One `exl3_moe` launch computes all six routed experts, with grid 28. It covers ~160 of 240 lanes per token that were
  already resident in VRAM, which is about 4 of 6 per layer. Those lanes wait for the layer's slowest miss.
- Piece streaming publishes NVMe pieces as they land, but only S reads them. No expert compute starts early.
- The costs per token are:
  - Copies: ~93 ms (C1 72.3 ms, S 21.3 ms).
  - `exl3_moe`: 3.85 ms, about 16 us per expert GEMM.
  - One 13.3 MB row takes ~0.98 ms to copy at 13.55 GB/s.

## 2. What this can buy, and why it's small

PCIe is the wall. The last row of a layer still gates the layer, and at least its own GEMM runs after it lands.
Starting compute sooner only hides compute, never copy.

| Variant | What moves off the critical path | Ceiling today | Ceiling after prefetch |
|---|---|---:|---:|
| **A. Resident first** | GEMMs of experts resident when the layer starts, run while the copies are in flight | 160/240 x 3.85 = **~2.6 ms/token** | ~185/240 x 3.85 = **~3.0 ms/token** |
| B. Three tiers | adds the RAM-hit experts' GEMMs, run while an NVMe row is still being read | +~0.2-0.4 ms/token | similar |
| C. Every expert on arrival | adds each earlier miss's GEMM under the next miss's copy | +~0.5-0.7 ms/token | smaller |

Notes on the ceilings:
- **The resident share after prefetch** uses the native gate's demand of 55.4 misses/token, against 78 without it
  (router-capture table, h=1 K=1).
- **Prefetch makes A worth more.** More of each layer is resident when the layer starts.
- **The ceilings are gross.** A second launch costs something at M=1. The MoE kernel is latency-bound, and 40
  extra launches may cost ~0.2-0.4 ms/token.
- **1a is a warning.** The shared expert on a side stream was measured and gave nothing at the wall, because the
  overlapped work came back as a slower C1. The difference here is that 1b takes the copy off the SMs, so compute
  should no longer slow it. That is a hypothesis for this plan to test, not a result.

**Order of work: 1b → prefetch → A. Do B or C only if A's measured gain matches its ceiling.**

## 3. Design A: resident first

### 3.1 One stream, reordered; no side stream

With 1b, the service issues the copies on the copy engine and the graph waits on a completion word. W1 and C1
become one wait kernel. The decode stream then no longer carries the copies, so reordering is enough:

```
post -> exl3_moe(resident mask) -> WAIT(completion) -> [S wait] -> F -> exl3_moe(missed mask) -> gather
```

- The resident launch runs while the copy engine moves the misses.
- The wait sits after the resident launch, so it spins only for the time that remains.
- Everything stays in one captured graph on one stream. This avoids the side-stream hazards
  (`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`) and reuses the shared temp buffers safely. `shared_temps` assumes that
  launches never overlap, and here they don't.

### 3.2 Why the split can stay byte-identical

This is the property that decides whether A is safe, and it has to be verified in the kernel source before any
graph change:

- Under the FUSED_DET path, `exl3_moe` writes each route's result into its own fp32 `scratch` row at `det[0]`
  (`expert_start`). `exl3_moe_gather` then sums the rows in a fixed order.
- If each route's scratch row depends only on its own slot's weights and input, the sum is the same however many
  launches produce it: two launches give the same bytes as one.
- **Pass the full-route `expert_start` table with a masked `expert_count`.** The scratch placement must not move when
  some slots are masked off. Today `route_tables` derives `expert_start` from the same count
  (`exl3_fused_moe.py:84-87`), so it has to be split into two parts:
  - a placement table, from all routes;
  - a work mask, one per launch.
- **To verify in exllamav3's `exl3_moe`:**
  1. The count and start arrays are read independently.
  2. With `NUM_ACTIVE = 6`, a launch whose mask has fewer than 6 active slots does its active slots' work unchanged.
  3. The kernel writes nothing to `out` or to other routes' scratch rows.
  4. Per-slot work does not depend on the concurrency index it is scheduled on (`exl3_moe_max_concurrency`).

  If any of these fails, stop and report it.

### 3.3 `keep` and the fatal path

`keep` comes from F, after S, and the resident launch now runs before F. So:

- The resident launch computes into scratch with unscaled weights, or with weights that carry no `keep`.
  `keep` is applied only where the gather reads.
- Today the layer-fusion kernel writes `weight_sorted = fp16(w * keep)` (`dsv41_layer_fusion.cuh:211`) and zeroes the
  counts when `keep` is 0.
  - **Question 1:** does `exl3_moe` itself read the weights? The call passes `weight_sorted` to both kernels.
  - If only the gather applies them, `keep` moves to the gather's weights and the resident scratch is simply
    ignored when `keep` = 0.
  - If `exl3_moe` applies them, the resident launch needs `w` and the gather needs `keep` as a separate factor.
- **Question 2:** is `w * 1.0f` then fp16 exactly the current path? It is for `keep` ∈ {0, 1}. Check that no other
  value of `keep` occurs.
- **The fatal path must keep its current meaning:** "drops the failed layer and every later layer and runs no expert"
  (`test_exl3_task5_item4_gpu.py`).
  - After the split, the resident experts of a failed layer have already run. That is harmless: they read only
    resident rows, which are valid.
  - But their output must not reach the residual. The test's assertion that no expert runs has to become "no
    expert's output is used", or the resident launch has to be gated on the device-side `keep` of the previous
    layer. Choose one and write it into the test.

### 3.4 Route tables before F

- `remap` (DIRECT's destination slots) is known after the route group, before post, so the placement table can be
  built early.
- Each hit or miss mask comes from the plan: the lanes that post publishes as misses (1b's `dst_slot`) against all
  routed lanes.
- Build both masks in the fused route-tables kernel, which already holds this data. That adds no kernel.

### 3.5 Switch

- Add `SGLANG_DSV41_ENABLE_RESIDENT_FIRST_MOE = EnvBool(False)` in `environ.py`, following the env-var conventions.
- It requires both 1b's copy-engine flag and layer fusion. Refuse it at launch without them. Without 1b, C1 is on
  the SMs and a resident launch would slow the copy, as it did in 1a.

## 4. Designs B and C (only if A pays)

- **B, three tiers.** The chain becomes: post → resident launch → wait for C1's completion → launch for the
  RAM-hit misses → wait for S → launch for the NVMe misses → gather.
  - It helps only in layers with an NVMe miss: 11.3 NVMe rows per token over 40 layers, so about a quarter of
    layers.
  - It needs a separate completion word for the RAM-hit tier and one for the NVMe tier. 1b's service already
    retires them separately.
- **C, per-expert arrival.** A single persistent MoE launch whose CTAs each spin on their own lane's completion word
  before loading that expert's weights.
  - It is the only variant that truly computes each expert the moment it lands.
  - It adds ~0.5-0.7 ms/token over A.
  - It costs a spin-and-deadline path inside the MoE kernel, which is exllamav3's code, not ours.
  - Spinning CTAs hold SMs, but the copy engine needs none, so this is not the host-node deadlock.
  - Build it only with measured evidence that A's gain is real.

## 5. Interaction with prefetch

- Prefetched rows that have landed count as resident, which is why A's ceiling rises after prefetch.
- The router-capture design makes layer T's planner wait for T's prefetch copy before it reads residency. Under A
  that wait delays the resident launch too. Once the per-lane completion words exist, a refinement is to treat an
  in-flight prefetched row as a miss with a head start rather than blocking the planner on it. Decide this after
  prefetch runs live. It changes the planner, not this kernel.

## 6. Steps

1. **Kernel check (no graph change).** Answer 3.2's four questions and 3.3's two questions from exllamav3's source.
   Write a GPU parity test comparing `exl3_moe` once with all six against two launches (resident mask, then missed
   mask) followed by one gather, and assert bitwise equality over random routes, hit/miss splits and `keep` ∈ {0, 1}.
   **Stop if it isn't bitwise.**
2. **Microbench the cost of the split at M=1.** Use real slot pointer tables and a working set past L2 (CLAUDE.md, GPU
   microbenchmarks). Report µs per layer for one launch against two. If two launches cost more than ~60 µs over
   one, the resident share can't pay for it: stop.
3. **Build A behind the flag**, on top of 1b and prefetch. Cover:
   - the route-table split;
   - the reordered chain in `_apply_graph`, fused path only;
   - `keep` moved to the gather;
   - the fatal-path test updated per 3.3.
4. **Tests:**
   - the CPU suite (`test/registered/unit/kernels`) and the RAM-miss GPU suite, against the base;
   - the new parity test;
   - the fatal-path and victim tests with the flag on and off.
5. **Smoke:** a single A then B run, flag off then flag on, not ABBA. Report:
   - ms/token and stalls;
   - 6/6 byte-identity against p2a-rowimg (`compare_arms.py`);
   - optionally a node-mode trace to show the resident launch overlapping the copy. Use node mode only for
     attribution; no graph-mode nsys with the copy engine on.
6. **Decide:** make A the default if it is byte-identical and saves ≥ 1 ms/token. Consider B or C only if A's
   measured gain is within ~30% of its ceiling.

## 7. Risks

- **The kernel is not ours.** If exllamav3's `exl3_moe` couples slots (shared accumulators, or work split that depends
  on the active count), the byte-identity premise fails and so does A. Step 1 settles it before any integration work.
- **Launch overhead at M=1** may equal the gain (step 2).
- **1a's precedent:** a measured overlap with no wall gain. Step 5's smoke is the only verdict that counts.
- **Dropped-layer semantics** change subtly (3.3). The test must say which meaning holds.
