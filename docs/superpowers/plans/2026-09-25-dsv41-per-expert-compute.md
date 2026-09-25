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

## 8. Gate results

**Date:** 2026-09-25. **Branch:** `cc/resident-first`, from `e4f81ac774`. **Run:** divix01, RTX 5090, under
`gpu-run.sh`, private worktree at `50cd50160e`, `sglang.__file__` in that worktree. exllamav3 at the pinned
`02aef45cd6`. Commands and outputs: `analysis/dsv41-drive/resident-first/` (`run.sh`, `parity-run2.txt`,
`bench-run2.txt`, `bench-run2.json`). An earlier run at `c1b54c0b99` gave the same numbers to within 1 us/layer.

**Verdict: step 1 passes with two corrections to 3.2/3.3; step 2 fails the gate. Stop A.**

### 8.1 Step 1: the kernel, from exllamav3's source

Sources: `exllamav3_ext/quant/exl3_moe.cu` (host: `exl3_moe`, `exl3_moe_gather`, `exl3_moe_max_concurrency`),
`exl3_moe_kernel.cuh` (the kernel), `exl3_gemm_inner.cuh` (the split-K GEMM it calls).

1. **Count and start read independently? Only the scratch row does.** The kernel never reads a start table
   for its inputs. It walks `expert_count` over every slot and keeps a running prefix (`start = end; end +=
   expert_count[e]`), then reads `token_sorted[start + row]` and `weight_sorted[start + row]`. Only the output row
   comes from `fused_base[e]` (`det[0]`). So a masked count keeps the placement where it is, but moves each route's
   weight index. With the full-route `weight_sorted`, a masked route reads its neighbour's weight.
   `test_full_weight_table_misplaces_masked_weights` shows the output changes.
   - **Correction to 3.2:** each launch needs its own compacted weight table, holding the j-th masked route's
     weight (slot order) at index j. `token_sorted` is all zeros at BS1, so it needs nothing.
2. **Fewer than 6 active under NUM_ACTIVE = 6: yes, unchanged.** `num_active` only sizes the grid:
   `num_groups = min(concurrency, 64, num_active)`, `group_size = min(num_sms / num_groups, 32)`. That is 6 groups
   of 28 SMs on the 5090, whatever the mask. Groups whose ticket matches no active slot scan the slots and retire.
   - **Trap:** `num_active` must stay 6 in both launches. A launch sized by its own active count gets wider groups.
     The GEMM's split-K slicing follows the group width (`slice_beg = tiles * blockIdx.x / gridDim.x`), so the
     partial sums reduce in a different grouping. Measured: `num_active = 4` on a 4+2 split changed the bytes in
     4 of 4 route sets.
3. **Writes only its own scratch rows: yes.** In the FUSED_DET path (`output_scratch` set), `had_d_out` writes only
   `output_scratch[(fused_base[e] + row) * hidden]` for slots it processes. It never writes `output_state`; the
   atomic add into `out` is the non-deterministic branch only. It also writes the per-group temps, the GEMM locks
   and the self-resetting scheduler words. Two launches on one stream serialize, so sharing these is safe.
   The parity test checks this directly: `out` stays zero after each launch, and the rows of routes a launch does
   not own stay NaN-poisoned.
4. **Per-slot work independent of concurrency index and scheduling: yes.** The group index offsets only the temp
   buffers and lock rows. Tickets assign experts to groups dynamically, but an expert's arithmetic depends only on
   `group_size` (the split-K slicing) and the lock-ordered reduction inside its group, which is fixed. See item 2
   for the one dependence, the group width.
5. **Does `exl3_moe` read `weight_sorted`? Yes; it applies the weight.** `had_d_out` scales by
   `0.0884 * weight`. The gather with slot kind 1 uses weight 1.0; only kind 2 multiplies by `weight_sorted`.
   - **Correction to 3.3:** keep cannot move into the gather's weights under kind 1. It goes into the gather's
     slot kind: `kind = det[2] * (keep > 0)`. A dropped layer then reads no scratch row, which is what the zeroed
     count gives today. The resident launch takes `fp16(w)` with no keep. For keep = 1 that is bit-for-bit today's
     `fp16(w * 1.0f)`.
   - The missed launch runs after F, so its count is multiplied by keep as today, and a dropped layer runs no
     missed expert.
6. **keep in {0, 1} only: yes.** Every write is a literal:
   - `exl3_ram_miss.cuh:541` (`ok ? 1.0f : 0.0f`), `:674` and `:1536` (1.0f), `:688`, `:1566` (finalize, the sole
     writer when F is on) and `:1663` (0.0f);
   - `expert_row_plan.py:357` `torch.ones`, and `:365` a bool cast.
   `go_total` only feeds the finalize kernel's `served` test. No other value is written.

**Parity test** (`test/manual/dsv41/test_exl3_moe_split_parity_cuda.py`, real layer-3 EXL3 rows, 12 slots):
- Reference: `Exl3FusedMoE.run`, with layer fusion off and on.
- Split: full-route placement from `route_tables(keep = 1)`, resident launch, missed launch, one gather with the
  keep-gated kind.
- Coverage: every hit count 0-6, keep 0 and 1, 4 random route sets each. 56 cases per reference.
- Result: all 112 cases bitwise equal in `out` (and in every scratch row for keep = 1). After each stage, `out`
  was untouched and exactly the expected rows were written.
- Command: `pytest -q -s -p no:randomly --basetemp=... test/manual/dsv41/test_exl3_moe_split_parity_cuda.py`,
  4 passed, EXIT=0.

### 8.2 Step 2: the cost of the split at M=1

Bench: `analysis/dsv41-drive/resident-first/split_launch_bench.py`, `--replays 300`, one CUDA graph of 40 layers per
arm, arms interleaved round-robin, medians.
- Real 13.32 MB EXL3 rows, 8 distinct per layer behind 46-entry slot pointer tables.
- A token touches 3.20 GB. That is far past L2, and the one-launch arm implies 790 GB/s, below the 5090's
  ~1.8 TB/s HBM.

| Arm | us/token | us/layer | vs one, us/layer | vs one, us/token |
|---|---:|---:|---:|---:|
| one (production `run`, layer fusion) | 4045 | 101.1 | - | - |
| route tables + gather, no launch | 362 | 9.1 | -92.1 | -3683 |
| two launches, static tables, 5+1 | 7725 | 193.1 | **+92.0** | **+3680** |
| two launches, static tables, 4+2 | 7668 | 191.7 | **+90.6** | **+3623** |
| two launches, static tables, 3+3 | 7624 | 190.6 | **+89.5** | **+3579** |
| two launches, torch-built tables, 4+2 | 9435 | 235.9 | +134.7 | +5390 |
| resident launch only, 4 experts | 4049 | 101.2 | +0.1 | +4 |
| missed launch only, 2 experts | 4034 | 100.9 | -0.3 | -11 |
| missed launch only, 1 expert (5+1) | 4014 | 100.4 | -0.8 | -31 |
| missed launch, 2 experts, `num_active = 2` (not bitwise) | 3737 | 93.4 | -7.7 | -308 |

- **An `exl3_moe` launch costs ~92 us at M=1 whether it holds 1 or 6 experts.** The six run in parallel, one per
  28-SM group, and a launch lasts as long as one expert does. The "~16 us per expert GEMM" in section 1 is
  3.85 ms / 240, a throughput average. It is not a latency that shrinks with fewer experts.
- **Two launches cost +89.5 to +92.0 us per layer, +3.6 ms per token**, even with the masks free (static tables).
  That is over the ~60 us gate in every split. Building the masks with torch ops adds another ~44 us per layer. An
  extended route-tables kernel would remove that part, but not the launch.
- **Even perfect overlap buys nothing.** Suppose the resident launch hid entirely under the copy. The critical
  path after the wait is then the missed launch. That launch costs 100.4-100.9 us per layer against 101.1 for
  today's single launch, a saving of at most ~0.8 us/layer, ~30 us/token. The ~2.6 ms/token ceiling in section 2
  assumed launch time scales with the number of experts, and it does not.
- Widening the missed launch's groups (`num_active = m`) saves ~7.7 us/layer, ~0.3 ms/token at best. That breaks
  byte-identity (8.1, item 2) and is still below step 6's 1 ms/token bar.

### 8.3 Verdicts

- **Step 1: PASS.** A split can be bitwise identical, with two design changes:
  - per-launch compacted weights;
  - keep applied through the gather's slot kind, not its weights;
  - plus `num_active` kept at 6 in both launches.
- **Step 2: FAIL. Stop Design A.** The split costs ~90 us per layer (~3.6 ms per token) serially. Its best case,
  full overlap of the resident launch, saves under 0.1 ms per token, because one launch's latency does not depend
  on how many experts it holds.
- **B and C inherit the problem.** B adds a third launch; C is a single persistent launch, so it avoids a second
  launch's cost, but each expert still takes ~92 us from the moment it starts. After the last miss lands, the tail
  is one expert's full latency either way.
- **What could still pay** is a faster single-expert path: one expert on more than 28 SMs, or a smaller row tile.
  That is exllamav3 kernel work, outside this plan.
