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
**no implied bandwidth above 15.75 GB/s (spec) or above 1.03 x 13.79 GB/s** (13.79 is the copy-engine figure measured in `NC_VISIBILITY.md`, a registered constant; the run's own `ce` arm is reported beside it and a `ce` figure more than 5% off 13.79 is noted as a change of box state) (the impossible-number detector; L2 residency would trip it); SM bandwidth at `n >= 2` at least 8 GB/s (else the harness or the box is broken);
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
| `c_measurement/c_harness.py` (the harness; `--dry-run` for CPU tests) | `6b7313b05132798cdae4ca77401280eaa010b8961a3f6c8fd8249264c3519b8b` (**supersedes `171302f8…` after the amendment in section 12**) |
| `c_measurement/nvme_load_reader.py` (the `nvme` arm's background reader) | `9ac57d78957d5657fdeccb70d0406c4e972ff3c14c4626b46088d1c66fdb08ba` |
| `c_measurement/quiet_check.py` (the pre-flight: is the box quiet, on which cores; reads /proc only) | `0190708bf9ab4bf37a573f80a8a088b193e7d82d252d6af72143dffd310df6f4` |
| `c_measurement/prebuild_jit.py` (compiles the gather kernel's JIT module with no GPU) | `015215022edf78dd69f98002e1c1d7bbc982fa2662cd5cca09ab7091d4fbb782` |
| `c_measurement/test_c_harness.py` (7 CPU tests, including the harness-to-analysis hand-off) | `94082c546ec5a9e777f0466bde2d1e6ac0fdf5a00e9b727abfca4bbf213c84ad` (9 CPU tests; supersedes `3fe04030…`) |

Self-test, 2026-09-21 on divix01, CPU: `c = 1.00` gives STANDS (`g*_H` 6.22), `c = 1.055` gives INTERMEDIATE (`g*_X` 7.01, `g*_H` 8.06, gross 4.92: the earlier model's 6.98 / 8.03 / 4.93 within `f`), `c = 1.15` INTERMEDIATE, `c = 1.40` WITHDRAWN, and an
impossible bandwidth, an idle link, row reuse, foreign load and a wrong row size are INVALID. **That tests the logic and the arithmetic reproduction, not the GPU.**

## 10. The harness as built (added when it was written; the registered design above is unchanged)

**Disclosed implementation choices** (also in the harness docstring): the "idle stream" is realised as a 600 us spin kernel before every timed launch so the host is always ahead of the GPU and its launch latency stays outside `T`;
the plan tensors are updated by device-to-device copies outside the timed window, for eager and graph alike; each cell's 200 launches are two visits of 100 (ABBA), after 20 discarded warm-up launches per visit; the six segment sizes
are derived from the dimensions (hidden 5120, intermediate 2304) and checked against the measured 13,315,584 B, not read from a loaded layout; `hot` packs with a one-thread torch copy on the launching thread, not the service thread; the `ce` arm's `T`
includes host enqueue of 6 x n copies, so it is a lower bound on the copy engine's speed, not its peak (its role is only the yardstick for the gate, which uses the registered 13.79 GB/s).
**Reuse distance is measured, not assumed:** each cell records the minimum reuse distance of the time-ordered rows read from that node's slabs (a permutation ring, so 150), and gate 4.1 reads it.
**Harness-level refusals, additional to the frozen gates:** a node's slabs must be at least 99% on the requested node by `move_pages`; each node's first launch is checked to have moved the intended rows (row ids are stamped in the first 8 bytes); the `nvme` arm writes
`results.INVALID` (exit 3) if the reader did not average 1.0 GB/s in every load window. `--skip-arms hot|repeat|ce|graph` exists as a **declared deviation** recorded in `meta.json`; the frozen analysis still requires the primary arms.

**What was tested, on CPU (2026-09-21):** 7 tests pass (segment sum; ring and reuse distance; ABBA coverage; the arm list; a dry run feeds `c_analysis.py` and it recovers the synthetic `c = 1.08`, `f = 0.006` and the load arm's 8% ratio; each of six gates fires on
bad harness output; the foreign-load meter sees a spinning process). On divix01: the harness's imports resolve against `wt-task1-new` with the engine's `/data/models/slang/.venv` Python; **the gather kernel's JIT module was built with no GPU** by `prebuild_jit.py` (6.2 s, cached under `~/.cache/sglang/jit/sm120f`), so the window has no compile; the NUMA helpers
bound 24 MiB slabs to node 0 and node 1 and `move_pages` confirmed 1.00 and 0.99. **Not tested, because it needs the GPU or the drives:** the real device path end to end (CUDA context, registration, the copy check, event timing, the graph capture, the copy-engine arm), and the reader's real I/O (its imports resolve).

**Known risks for the window.**
1. **P-state.** Gate 4.1 requires P0 at the start of every cell. The `hot` arm syncs after every launch and the GPU is idle while the CPU packs, so it may sit below P0. If `hot` cells alone fail the gate the whole run is INVALID by the registered rule; the fallback is `--skip-arms hot` with the deviation named in the report. I would look at the first pass's states before deciding.
2. **Page cache.** Node 1 had 734 MiB free (node 0: 6.3 GiB) on 2026-09-21. Binding 2 GiB there evicts about 1.3 GiB of page cache; that is a side effect on any later arm's cold/warm state (`PIPELINE_BASELINE.md`'s regime notes). Run this window **after** any pending timed arm, or re-warm.
3. **Threads.** `gpu-run.sh` pins the harness to cores 32-63, which span both nodes, so the launching thread may run on either; `cpu` is recorded per cell.

**Command (for whoever holds the lock).** Copy `c_measurement/` (with `bench_mirror_rows.py`, `bench_row_scheduling.py`, `drive_conditions.py` beside its parent) to a scratch directory on divix01 keeping the relative layout, then:

```text
CC=/data/models/slang/nvfp4-work/cc-expert-prediction
env CUDA_HOME=/usr/local/cuda-13.2 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$CC/wt-task1-new/python \
  $CC/analysis/dsv41-phase3b/gpu-run.sh /data/models/slang/.venv/bin/python <dir>/c_measurement/c_harness.py \
  --out <outdir> --repo $CC/wt-task1-new --with-nvme --reader-cpus 18-25
python3 <dir>/c_measurement/c_analysis.py <outdir>/results.jsonl        # the frozen verdict
```

`--reader-cpus 18-25` is node 1 outside cores 32-63 and 64-71. First minute: the same command with `--check-only` (no `--with-nvme`) does the allocation, registration and copy check and exits.

## 11. Retirement of the earlier design

`TOPOLOGY.md` 9.B is **retired**, superseded by this document; 9.C's GPU side is carried by the `nvme` arm (its remote-bounce placement matrix stays open and unscheduled); **9.A (storage alone, no GPU) stays as an independent item.** The status lines in `TOPOLOGY.md` say so.

## 12. Amendment before any data: what "quiet" means, the idle baseline for `rho`, and how contention is handled (2026-09-21)

Prompted by the lead's note that the box carried a load average of 27.8 (36 when I looked: every core from 0 to 63 was between 20% and 100% busy) while the GPU itself was free. **Nothing has been run; this
changes the registered design only by making three things explicit, all before a launch, and the harness hash in section 9 is the one that includes them.**

**1. Decision: the registered run is the QUIET run, and it waits for a quiet box. I do not register a loaded run.** The measurement is not GPU-bound in the sense that would let host contention be ignored: the source rows are read *from host
memory over PCIe* (memory controllers, the IIO and the pinned-page path are shared with every other lane), the `hot` arm is CPU packing, the `nvme` arm is drive-to-host DMA, and the effect being sought is at the 2-8% level that a busy box moves. A loaded pass would have an
unrepeatable, unrecorded mixture of foreign load, so its difference from the quiet pass could not be interpreted; it would be a caveat, not a second data point. (The `nvme` arm is the *designed* contention, and it is controlled.) If the lead nonetheless wants a
loaded pass as an extra, it is reported beside the quiet run, labelled contended, never merged and never used by the rule.

**2. How quiet.** The frozen gate (4.1) is "no foreign process above 10% of a core"; `foreign_max_core_pct` is now **defined** as the largest per-core sum of foreign thread CPU over the cores the run uses (the harness's and the reader's), not the box-wide maximum, so a
permanent daemon on a core we do not use (nimbus, reth) does not trip it and the same daemon migrating onto one of our cores does. To have margin the pre-flight asks for half of it: **`quiet_check.py` must print GO: the eight harness cores (node 0, within 32-63) and the eight
reader cores (node 1, outside 32-63, never 64-71) it picks are each under 5% busy over 10 s, and the NVMe devices read under 0.02 GB/s.** It is run at the start of the window and again after, and both outputs are kept. Beyond the gate I ask the lanes to **suspend CPU-heavy and
memory-heavy jobs box-wide for the ~15 minutes** (the gate cannot see memory-bandwidth contention on cores we do not use); the harness then launches on the quiet cores through `taskset -c <picked>`, inside `gpu-run.sh`'s 32-63 mask, and the reader on its picked cores.
Today's `quiet_check.py` result on divix01 was NO-GO (load 34.8; the quietest eight harness cores were 20-27% busy, the reader's 10-56%).

**3. The idle baseline for `rho = T_load / T_idle`.** If other lanes read the drives during the idle cells, `T_idle` is not idle and `rho` is understated, which flatters the hiding claim. Two harness-level rules, additional to the frozen gates and treated the same way (INVALID, no number):
(a) **every idle-arm cell records the NVMe read rate** (`/proc/diskstats`, all whole `nvme*n*` devices, over the cell's own window) **and the run is INVALID if any idle cell read faster than 0.02 GB/s** (`results.INVALID`, exit 3);
(b) **in every load cell the drives' bytes must equal the reader's own bytes to within 10%**, or another lane used the drives during the load window and the `nvme` arm is INVALID. The reader is direct (O_DIRECT through `uring_direct`), so the page cache of other lanes does not enter it.
This establishes the idle baseline by measurement in the same window, not by assumption; a baseline taken on a box where three lanes are doing I/O fails (a).

**4. What contention would still get through.** Memory-bandwidth pressure from cores outside our set, and the drives' queues seen by a lane that reads a different device set: (a) sees all whole NVMe devices, so the second is covered; the first is not gated, which is why the request to suspend box-wide jobs is for the window and not only for our cores. The per-cell
`p99 / p50` gate and the recorded foreign-thread name (`foreign_where`) are how a contaminated cell would show.
