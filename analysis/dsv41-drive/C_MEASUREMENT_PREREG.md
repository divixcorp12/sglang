# Pre-registration: the per-row gather time `c` (Task 6), 2026-09-21

**Status: design only.** Nothing has been run on the GPU. Nothing is to run until (1) the lead has accepted this document, (2) the harness has been written and its sha256 appended to
section 9 in a second commit, (3) `cc-gpu.lock` is held and crypto-c9 has scheduled the window on a quiet box. CPU work so far: the analysis script and its synthetic self-test, run
under `taskset -c 0-63`, one thread. Author: t3-topology. Companion to `G_MEASUREMENT_PREREG.md`; the decision rule below is fixed by the hash in section 9.

## 0. What `c` decides, and why it is the binding unmeasured input

`c = 1.055 ms` enters three places. (a) **Per-row's gross increment over two-phase**, `+4.93 ms` per step at best order, is proportional to it (a 10% error moves it 0.5 ms). (b) **V1's ceiling**,
`33.9 hit lanes x c = 35.74 ms`, is `c` times a count by construction, and (c) whether the hit copies **fit inside the read wait** (`h * c` against the wait). For the per-row label, `G_MEASUREMENT_PREREG.md`
2.3 already shows the crossing `g*` moves about 3 us per 10% of `c`. **One further effect is larger than it looks and is not a bandwidth question:** the model's per-row stage copies one row per
launch (`count = 1`), where the batched copy moves `k` rows in one. Per-row's exposed copy after the last row arrives is `T(1)`, two-phase's is `T(m)`; if a single-row launch costs `delta` more than its share of the
batched one (a latency limit at `count = 1`, which `PER_ROW_TRANSFER.md` OPEN 1 leaves unmeasured), per-row loses `13.97 read requests per step x delta`. **`delta = 80 us` costs 1.1 ms per step, the whole
of `G*` = 1.114 ms.** So the measurement is of the *function* `T(n)`, not of one number.

## 1. What `c = 1.055 ms` is today

**It is neither the idealised bytes-over-bandwidth figure nor a measurement of this path.**

- **Derivation.** `DSV41_REFERENCE.md` 18.2: the gather kernel totals **128 ms per step** in a node-mode `nsys` trace of an **older tree** (`wt-dsv41` at `76829dff55`, mirrors off, 391 ms per step, prefetch off),
  divided by **G = 121.2 VRAM misses per step**, a row count taken from `graph_step` lines aligned to the kernel window at **74.5% per-layer agreement**. `128 / 121.2 = 1.056`. So it is an in-situ per-row average of the
  batched gather at `count = k` (k mostly 2-4), under whatever slab placement, page backing and source-cache state that arm had, with an imprecise row denominator. It is the kernel's time, not link arithmetic; but it is one
  arm's average from another tree.
- **The link "ceiling" cited beside it is weaker than it reads.** "12.02 GB/s" is `2,764,800 B / 0.23 ms` (`MOE_EXPERT_TRANSFER.md` line 433): a per-row rate for the **older 2.76 MB NVFP4 row**. It is not a measurement of
  the DSV4.1 link, and the reference table's "1.11 ms per miss row" is that rate applied to 13.3 MB. `PER_ROW_TRANSFER.md` 1.4 and `PER_ROW_TRANSFER_REVIEW.md` G1 read it as "the measured link ceiling, `c` within 5% of it"; but `c`
  (13,315,584 B / 1.055 ms = **12.62 GB/s**) is *above* 12.02, so it is not "under the ceiling".
- **The only direct measurement of this kernel on this box** is `NC_VISIBILITY.md`'s bandwidth cell: the production geometry (grid 8 x 256), a 1 GiB region (10.7 x the **96 MiB** L2), **12.34 GB/s**, and `cudaMemcpyAsync` 13.79 GB/s,
  link Gen3 x16 with 15.75 GB/s spec. As a row time: **1.079 ms** (kernel), 0.966 ms (copy engine). That is a contiguous streaming copy, not the six-segment random-row gather.
- **The three do not agree, and the largest gap is the row denominator.** With the later corpus's 126.9 VRAM-miss rows per step (`128 / 126.9 = 1.009 ms`, 13.2 GB/s) the in-situ figure would exceed the 12.34 GB/s the
  same kernel achieves in isolation. So **the existing numbers bound `c` to roughly 0.97-1.08 ms and do not choose within it.**

| `c` (ms) | source | GB/s | gross (ms/step) | `g*` all exposed / hit stages hidden (us) |
|---:|---|---:|---:|---:|
| 0.966 | copy engine, 13.79 GB/s | 13.79 | 4.5 | 4.4 / 5.0 |
| 1.009 | 18.2's 128 ms over the later corpus's 126.9 rows | 13.2 | 4.7 | 5.6 / 6.4 |
| **1.055** | 18.2 as used | 12.62 | 4.93 | **6.98 / 8.03** |
| 1.079 | this kernel, isolated, 12.34 GB/s | 12.34 | 5.04 | 7.7 / 8.8 |

## 2. What is measured

`T(n)`: the **GPU-timeline time of one launch of the production `copy_expert_row_segments_gpu_kernel`** (same source, grid 8 x 256) moving `n` rows, `n = 1..6` (the schema-4 trace's maximum lane count is 6), each row **13,315,584 B in the
six streamed segments** (`w13_trellis`, `w13_suh`, `w13_svh`, `w2_trellis`, `w2_suh`, `w2_svh`; `DSV41_REFERENCE.md` 16.2; the segment table is read from the loaded layout, and the harness checks that it sums to the
constant), from **distinct random rows of a pinned host region allocated and registered as the service allocates its slabs**, to **distinct VRAM scratch slots cycled over 64 slots** (so neither side stays L2-resident), event-timed
on an otherwise idle stream. From it:

- `c_m`, the **marginal per-row time**, and `f`, the fixed per-launch cost: least squares `T(n) = f + c_m * n` over `n = 1..6`. **The registered `c` is `c_m`.** `f` should be about 6 us (the old E28 fit) and is the same quantity
  as the copy kernel's active-path fixed cost inside `g_a` in `G_MEASUREMENT_PREREG.md`; an `f` far from that is reported as a discrepancy between the two documents.
- **Linearity at `n = 1`**: `|T(1) - (f + c_m)| / T(1)`. Flagged above 3% ("`count = 1` is not the marginal cost").
- Implied GB/s by `n`, each checked against the ceilings (gates).

## 3. Arms

Primary: **`sm` / `cold` / `idle` / `eager`**, pinned source on **node 0** and on **node 1** (the GPU is on node 0; the four drives on node 1; where production's six slabs actually landed is unknown, `TOPOLOGY.md` 1.5 and 7.2), `n = 1..6`.
Secondary, each with a stated purpose:

| arm | purpose |
|---|---|
| `hot` (the CPU `memcpy`s a row into the slab immediately before the copy, as the service does when it packs a miss row) | production's miss rows are just-packed; its hit rows are cold. A different source-cache state may change `c`. |
| `repeat` (the same rows every launch) | **the L2 control**: if it is not visibly faster than `cold`, either L2 does not hold host reads on this part or the harness cannot see it. Informational, not gating; it is why the cold arms must not repeat rows. |
| `ce` (`cudaMemcpyAsync` of the same rows, labelled "copy engine, not the production path") | the yardstick for the impossible-number gate (kernel bandwidth must not exceed it). |
| `graph` (the same launch captured and replayed, `n = 3`) | the production path is a graph node; the eager figure is used only if the two agree (gate). |
| `nvme` load (the CPU-only `uring_probe`-geometry reader on the drives, node-1 bounce, running throughout; `n in {1, 3, 6}`) | **V1's whole premise is copying hit rows while the NVMe reads DMA into host memory.** The 18.2 trace measured the gather when nothing else ran ("everything is serialized"), so a `c` under concurrent drive-to-host DMA has never been seen. Absorbs `TOPOLOGY.md` 9.C. |

Order randomised within each of **5 interleaved passes** (A B B A); every repetition reported with median and min-max, never a mean of two.

## 4. Decision rule

### 4.1 Gates (any failure: INVALID, no number quoted)

Row bytes equal 13,315,584; **no row re-read within 100 rows** (1.33 GB = 13.9 x L2) in any non-`repeat` SM cell; link Gen3 and P0 at the start and end of every cell; no other GPU process; no foreign process above 10% of a core; `p99 / p50 <= 1.25` per cell;
**no implied bandwidth above 15.75 GB/s (spec) or above 1.03 x the measured copy-engine figure** (the impossible-number detector; L2 residency would trip it); SM bandwidth at `n >= 2` at least 8 GB/s (else the harness or the box is broken);
graph within 3% of eager at `n = 3`; both nodes present with 5 passes and every `n`.

### 4.2 The label (computed by the frozen script, per node, then together)

From `T(n)` the script reruns the precheck model on the **measured lanes** (`task1f`) joined onto the **stage timings** (`task1-2`), with copy cost `T(n)` in place of `k * c`: a batched launch of `n` rows costs `T(n)`, a per-row stage costs `T(1)`;
two-phase copies hits (`T(h)`) then the rest (`T(m)`); per-row (best order) is the chain of `T(1)` stages. It reports: **V1 ceiling** (`sum T(h)` per step), **gross** (two-phase minus per-row, per step), and the **non-linearity penalty** on layers that read nothing
(`k * T(1) - T(k) - (k - 1) * f`; the `(k - 1) * f` part is already in `g_a`). Net-before-`g` `= gross - penalty`, and

```text
g*_X = (net - 3.816 ms) / 159.55 * 1000    us  (all extra triples exposed)
g*_H = (net - 3.816 ms) / 138.69 * 1000    us  (the 20.86 hit-lane stages in read layers hidden)
```

| label | condition | consequence for `PER_ROW_TRANSFER.md` / the plan |
|---|---|---|
| **STANDS-FOR-ALL-ASSUMED-g** | `g*_H <= 8` us: below the plan's assumed 8-14 us even with class C hidden | "expected-REJECTED" stands across the whole assumed range of `g`. In uniform-`c` terms: `c <= 1.054` ms. |
| **WITHDRAWN-FOR-ALL-ASSUMED-g** | `g*_X >= 14` us | best-order per-row clears the bar for every `g` in the assumed range; the label is withdrawn and V2 becomes arm A3 (still needs `g`, the order array, FIFO). In uniform-`c` terms: `c >= about 1.30`. |
| **INTERMEDIATE** | otherwise | the verdict turns on `g`; only A3 (with `g` measured on the real `W_s`/`A_s`) decides. `c` between about 1.054 and 1.30. |
| **STRADDLES-NODES** | the two node arms fall in different labels | report both; production's node mix (a per-slab node share the owner must supply) decides; do not average. |

**At the model's own `c = 1.055` the label is INTERMEDIATE by 0.03 us** (`g*_H = 8.03`). The measurement decides which side; a measured `c_m` within about 2% of 1.055 will not move the label with any confidence, and I will say so
rather than pick a side. The edges are `g*` in us, not `c`; they are stated in `c` only for orientation.

### 4.3 Other outputs and their rules

- **`c_m`, `f`, `T(1)` linearity** replace `1.055` in the plan and the design; the model figures are recomputed from `T(n)` (not by rescaling).
- **`nvme`-load ratio** `rho = T_load(n) / T_idle(n)`: `rho <= 1.05`: no change; `1.05 < rho <= 1.20`: recompute `hide_ok` and V1's ceiling with `T_load` for copies that overlap a read; `rho > 1.20`: V1's hiding claim fails in proportion and the V1 ceiling is restated with `T_load`.
- **`hot` versus `cold`**: a difference above 5% is reported and the per-row chain (miss rows, just packed) uses `hot`, V1's hit stage (cold rows) uses `cold`.

## 5. Predictions (mine, before the run; not part of the rule)

- `c_m` **1.05-1.10 ms** (12.1-12.7 GB/s), centre about 1.08 (the isolated kernel figure); `f` about 6 us; `T(1)` on the line within 2%.
- Node 1 (remote to the GPU) 0-10% slower than node 0.
- `hot` within 5% of `cold`; `rho` 1.00-1.15.
- Under these: **INTERMEDIATE** (`g*_H` about 8-9 us). I expect `c` to sit close enough to the edge that the label does not change, which would be an honest null: the measurement would then have removed an
  assumed input, not moved the verdict.

## 6. What could make the measurement lie

| # | how | control |
|---|---|---|
| L1 | **L2 is 96 MiB, not ~128 MB, and a benchmark that reuses a tensor measures L2.** This has already produced two recorded constants that understate cost by 1.5x and 2.3x (`expert_residency_gpu.py`, the 0.007 / 0.054 ms pair). Unlike `g`, `c` moves data and is exposed to this. | Distinct rows drawn without replacement from at least 2 GiB per node (150 rows), reuse distance at least 100 rows; the `repeat` arm shows the size of the effect; the impossible-number gates (15.75 GB/s spec, 1.03 x measured copy engine). |
| L2 | **Host-side caches**: the CPU's 24.75 MB L3 and the just-written rows (`hot`). | `cold` rows are never touched by the CPU near the copy; `hot` is a labelled arm. |
| L3 | **Page backing and registration.** THP versus 4 KiB pages changes GPU TLB reach on 13 MB rows spread over 75 GB. | The production slab allocator and `cudaHostRegister` path; `AnonHugePages` of the region recorded per node. |
| L4 | **NUMA placement**, and first-touch spilling to the other node when the target is full (node 1 has about 1.29 GiB free, so a 2 GiB node-1 region reclaims page cache). | Allocate under `numactl --membind`, touch, verify with `move_pages`/`numa_maps` **of the harness's own process** before any timing; reclaim recorded; both nodes measured. The production slabs' actual node shares are asked of the owner, not read from the production process. |
| L5 | **The link idles at Gen1 / P8.** | Untimed load first; Gen and pstate at the start and end of every cell (gate). |
| L6 | **Foreign load** (`nimbus`, `reth` at 90% and 50% of a core have been on this box) shares memory controllers and links with the arm; `c` is memory-bandwidth-sensitive. | Quiet-box slot; foreign load recorded per cell (gate); `p99 / p50` gate. |
| L7 | **Other GPU processes** (production holds 25-30 GiB and serves). | crypto-c9 stops or confirms idle; `pmon` per cell (gate). |
| L8 | **A single event pair around one launch** includes launch and event overhead. | It lands in `f`, which is reported and compared with the 6 us prior; the slope `c_m` is unaffected. |
| L9 | **The older harness reported 12.34 GB/s for every repetition (min = max)**, which is either a very stable figure or a reduction that hides variation. | Every launch timed and kept (200 per cell); the distribution and `p99 / p50` reported. |
| L10 | **Eager versus graph.** | Graph arm and gate. |
| L11 | **`n = 1` may not be linear.** | The fit residual at `n = 1` is reported and enters the model through `T(1)` directly, not through `c_m`. |
| L12 | **Load arm realism.** A reader that is not the production `RowReader` has a different DMA pattern. | Production geometry (6.5 MiB extents, queue depth 16 per drive, both mirrors); recorded disk bytes per `/proc/diskstats`; the reader is **not** the service. It bounds the effect, it does not reproduce a decode step. |

## 7. Size of the run and sharing

- **GPU time about 8 minutes of launches:** per pass, `2 nodes x 6 n x 200 launches x about 3.5 ms` (SM cold) is 8 s, the same for `hot`, about 8 s for the copy engine, a few seconds each for `repeat` and `graph`: about 40 s; **5 passes = 3.5 min**; the `nvme` arm
  (2 nodes x `n in {1, 3, 6}` x 200 launches, five passes, reader running) about 1.5 min. Plus about 3 minutes of allocation, registration and first-touch for 2 x 2 GiB, and 30 s of warm-up. **Request 15 minutes.**
- **GPU memory about 1.2 GiB** (context, 64 scratch slots of 13.3 MB = 0.85 GB). Request **3 GiB**. **Host: 4 GiB pinned (2 per node)**; production's 75 GB tier is resident, so the harness must not run beside a serving process that would be squeezed (a quiet-box condition already).
- **Shares a window with:** `TOPOLOGY.md` 9.B is **absorbed** (it asked for the SM gather at an honest working set on both nodes; this is that, with the real segments and counts 1-6) and 9.C's storage-plus-SM question is the `nvme` arm. **Retire 9.B and 9.C** if this is accepted. 9.A (storage alone, no GPU) is independent and can run
  in the same quiet hour. Nothing else pending needs the GPU concurrently, and nothing should run concurrently with this. The `g` run is cancelled (`g` moves to arm A3).
- **No decode arm is needed**, and none can substitute: the existing traces carry no device intervals, so an in-situ `c` cannot be recovered from them (Task 1's device intervals are not built).
- Order: this document accepted; harness written and its hash committed; `cc-gpu.lock`; crypto-c9 messaged with 15 minutes, 3 GiB and 4 GiB of host memory; run; frozen script; verdict word reported first.

## 8. What it will not settle

`g` (moved to arm A3), **A2** (there is still no early readiness signal), the 1.0 ms launch cost, the gate's real resolution (1.5% is a quiet-box figure on an UNRESOLVED series), **production's slab node shares** (owner input), whether the
production `c` differs between a traced and an untraced arm (this is a standalone kernel measurement, so it also does not reproduce a decode step's concurrent CPU packing and polling), and `d`, the poll detection delay.

## 9. Frozen artefacts

| file | sha256 |
|---|---|
| `c_measurement/c_analysis.py` (gates, `T(n)` fits, the model recompute on measured lanes, the label; `--selftest` produces STANDS / INTERMEDIATE / WITHDRAWN and five INVALID cases from synthetic `T(n)` with known `c`, needs the divix01 traces) | `4657702c0b63d7956fc699bf99ee16c1bbf4810ebf8ef9774652a1c375f7c21c` |
| the harness | *to be appended before the lock is taken* |

Self-test, 2026-09-21 on divix01, CPU: `c = 1.00` gives STANDS (`g*_H` 6.22), `c = 1.055` gives INTERMEDIATE (`g*_X` 7.01, `g*_H` 8.06, gross 4.92: the earlier model's 6.98 / 8.03 / 4.93 within `f`), `c = 1.15` INTERMEDIATE, `c = 1.40` WITHDRAWN, and an
impossible bandwidth, an idle link, row reuse, foreign load and a wrong row size are INVALID. **That tests the logic and the arithmetic reproduction, not the GPU.**
