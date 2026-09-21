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
| `c_measurement/c_harness.py` (the harness; `--dry-run` for CPU tests) | `35dcf8215ae5c2c92b56afde2c0a1598b6d40cee1ef2dd92f47951ad52ebe010` (**IN FORCE: amendment 4, section 15**) |
| `c_measurement/nvme_load_reader.py` (the `nvme` arm's background reader) | `9ac57d78957d5657fdeccb70d0406c4e972ff3c14c4626b46088d1c66fdb08ba` |
| `c_measurement/quiet_check.py` (the pre-flight: candidate cores from /sys, the idle-core check, and the CPU-only `--rehearse` of the frozen gate) | `f557d035d9c82fddfbe130cb7366771366a50933275d5c4038d5a43277fef7d3` |
| `c_measurement/verify_hashes.py` (checks the files against this table; run it immediately before the window) | `26bb113eb3699d6d48440f795656e5fe077e51d617dd3c75535efe931141f542` |
| `c_measurement/prebuild_jit.py` (compiles the gather kernel's JIT module with no GPU) | `015215022edf78dd69f98002e1c1d7bbc982fa2662cd5cca09ab7091d4fbb782` |
| `c_measurement/test_c_harness.py` (7 CPU tests, including the harness-to-analysis hand-off) | `ccb5e6eae19a9f21eab364a2727d2f137c1d507a1f9a82c733ac945714ad1c6c` (17 CPU tests) |

**Superseded hashes: none of these is the registered harness.** `c_harness.py` `171302f871ee45d2…` (first version, commit `8313506f25`) was superseded by amendment 1 (section 12, `c2eddc482b`); `6b7313b05132798c…` (amendment 1) by amendment 2 (section 13, `cb436aec3b`);
`d39270e19a7fcd90…` (amendment 2) by amendment 3 (section 14, `bc02ab9ddf`); `adcd041f5660ef77…` (amendment 3) by amendment 4 (section 15). Earlier `test_c_harness.py` hashes (`3fe04030…`, `94082c54…`, `e71c672a…`, `a8f96f0f…`) and `quiet_check.py` `0190708b…`, `e50bd7ab…` are superseded the same way.
**`c_analysis.py` has one commit (`05c5510392`) and has not been touched since: `4657702c0b63d7956fc699bf99ee16c1bbf4810ebf8ef9774652a1c375f7c21c`. It is what turns numbers into a verdict, and it is the one file that does not move.**

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

**Partly superseded by sections 14 and 15: the 0.02 GB/s idle-drive criterion (item 2's GO condition and item 3(a)) and the idea that the box should be quiesced are withdrawn; the drift rules replace them. Read section 14 first.**

Prompted by the lead's note that the box carried a load average of 27.8 (36 when I looked: every core from 0 to 63 was between 20% and 100% busy) while the GPU itself was free. **Nothing has been run; this
changes the registered design only by making three things explicit, all before a launch, and the harness hash in section 9 is the one that includes them.**

**1. Decision: the registered run is the QUIET run, and it waits for a quiet box. I do not register a loaded run.** The measurement is not GPU-bound in the sense that would let host contention be ignored: the source rows are read *from host
memory over PCIe* (memory controllers, the IIO and the pinned-page path are shared with every other lane), the `hot` arm is CPU packing, the `nvme` arm is drive-to-host DMA, and the effect being sought is at the 2-8% level that a busy box moves. A loaded pass would have an
unrepeatable, unrecorded mixture of foreign load, so its difference from the quiet pass could not be interpreted; it would be a caveat, not a second data point. (The `nvme` arm is the *designed* contention, and it is controlled.) If the lead nonetheless wants a
loaded pass as an extra, it is reported beside the quiet run, labelled contended, never merged and never used by the rule.

**2. How quiet.** The frozen gate (4.1) is "no foreign process above 10% of a core"; `foreign_max_core_pct` is now **defined** as the largest per-core sum of foreign thread CPU over the cores the run uses (the harness's and the reader's), not the box-wide maximum, so a
permanent daemon on a core we do not use (nimbus, reth) does not trip it and the same daemon migrating onto one of our cores does.
**KNOWN UNGATED FAILURE MODE, stated here beside the redefinition and not elsewhere: restricting the count to our cores makes this gate BLIND to memory-bandwidth and memory-controller contention from daemons on cores we do not use, which is exactly the coupling that matters for `c`, because the source rows cross PCIe and share controllers with them. Nothing in the run gates it; it is bounded only by the box-drift rule (idle versus load arms) and by the recorded foreign CPU.** To have margin the pre-flight asks for half of it: **`quiet_check.py` must print GO: the eight harness cores (node 0, within 32-63) and the eight
reader cores (node 1, outside 32-63, never 64-71) it picks are each under 5% busy over 10 s, and the NVMe devices read under 0.02 GB/s.** It is run at the start of the window and again after, and both outputs are kept. Beyond the gate I ask the lanes to **suspend CPU-heavy and
memory-heavy jobs box-wide for the ~15 minutes** (the gate cannot see memory-bandwidth contention on cores we do not use); the harness then launches on the quiet cores through `taskset -c <picked>`, inside `gpu-run.sh`'s 32-63 mask, and the reader on its picked cores.
Today's `quiet_check.py` result on divix01 was NO-GO (load 34.8; the quietest eight harness cores were 20-27% busy, the reader's 10-56%).

**3. The idle baseline for `rho = T_load / T_idle`.** If other lanes read the drives during the idle cells, `T_idle` is not idle and `rho` is understated, which flatters the hiding claim. Two harness-level rules, additional to the frozen gates and treated the same way (INVALID, no number):
(a) **every idle-arm cell records the NVMe read rate** (`/proc/diskstats`, all whole `nvme*n*` devices, over the cell's own window) **and the run is INVALID if any idle cell read faster than 0.02 GB/s** (`results.INVALID`, exit 3);
(b) **in every load cell the drives' bytes must equal the reader's own bytes to within 10%**, or another lane used the drives during the load window and the `nvme` arm is INVALID. The reader is direct (O_DIRECT through `uring_direct`), so the page cache of other lanes does not enter it.
This establishes the idle baseline by measurement in the same window, not by assumption; a baseline taken on a box where three lanes are doing I/O fails (a).

**4. What contention would still get through.** Memory-bandwidth pressure from cores outside our set, and the drives' queues seen by a lane that reads a different device set: (a) sees all whole NVMe devices, so the second is covered; the first is not gated, which is why the request to suspend box-wide jobs is for the window and not only for our cores. The per-cell
`p99 / p50` gate and the recorded foreign-thread name (`foreign_where`) are how a contaminated cell would show.

## 13. Amendment before any data, second: the P-state fallback, a persistent foreign process, and how `rho` is worded (2026-09-21)

**P-state fallback (authorised by the lead, three conditions).** The run is made as registered first. If gate 4.1 fails on `hot` cells only, the failure is **observed and recorded first**: the INVALID result of the full run is kept in its own output directory
(`results.jsonl`, `meta.json`, the analysis output) **before** any re-run. A re-run in the same window may then use `--skip-arms hot`. `meta.json` of that run records the deviation (`skipped_arms`, and `NOT_MEASURED` lists `hot`), and **`hot` is reported NOT MEASURED, not absent and not
"not applicable"**. Wherever `c` is quoted afterwards (the report, `PER_ROW_TRANSFER.md`, the plan) the limitation travels with it: **a `c` measured without `hot` is a `c` for the cold path; anything that depends on host-cache state is outside it.** A `hot` cell that sits below P0 because the GPU
idles while the CPU packs describes that arm's duty cycle, not only the instrument; it would still not be comparable with the other cells, which is why the gate refuses it. No pre-emptive skip is permitted.

**A persistent foreign process.** `aggregate-runner`, running as root at about 246% of a CPU, owned by nobody on the team, is a floor under everything: **a truly idle baseline is not available today.** It does not block the run, and it is handled by recording and by a gate, not by assumption:
- **Recorded per cell** (`results.jsonl`): the box-wide foreign CPU in cores (`box_foreign_cores`: every non-own thread, with the kernel threads that carry *our* I/O, `kworker`, `ksoftirqd`, `irq/`, `nvme`, `iou-`, excluded so the load arm's own I/O is not counted against it), the five biggest foreign
  processes by name and CPU for each visit (`foreign_top`, where `aggregate-runner` will appear), and the load average at the start and end of the cell.
- **Recorded per node in `meta.json`:** MemFree, FilePages and Active/Inactive(file) before and after the slab allocation, so a later cold/warm measurement can tell whether it inherited this run's eviction.
- **A new harness-level gate, INVALID and no number:** if, within a pass, the mean box-wide foreign CPU of the load cells differs from that of the idle cells by **more than 1.0 core**, the `nvme` ratio is invalid (`results.INVALID`, exit 3; the per-pass means are in `meta.json`).

**How `rho` is worded.** The idle baseline is measured on a box that is not idle, so `T_idle` is inflated and `rho = T_load / T_idle` is biased **low, in the direction that flatters V1** (it is harder to trip 1.05 and 1.20). Therefore: **a `rho` just below 1.05 is reported as "no effect detectable above a floor we did not control",
never as "no effect".** The floor is stated with its numbers (the recorded foreign cores in the idle cells and the process names).

**When the `nvme` arm is deferred instead of run.** Once the lanes are quiesced the lead reports the residual load at that moment; it is recorded as the actual starting condition. If `quiet_check.py` does not print GO for the reader's cores or the drives, or the residual floor is high enough that I judge a ratio cannot be
defended, the run goes without `--with-nvme` and `nvme` is **reported NOT MEASURED** (`meta.json` `NOT_MEASURED`), for a genuinely quiet box; the other arms are unaffected. Cores 64-71 are never used.

**Order of the report** (as required): the `--check-only` result first, and if it finds a problem I stop and tell the lead before proceeding; then the verdict word; and if `c_m` is within about 2% of 1.055 I say so plainly and report the label as unchanged.

## 14. Correction before any data, third (items 1 and 2 SUPERSEDED by section 15): the box's permanent load is the CONDITION, not contamination (2026-09-21, lead's correction)

`nimbus_beacon_node` (Ethereum mainnet consensus), `op-reth` (Optimism full node) and `aggregate-runner` (`aggregate-runner.service`, a feed scraper) are the user's own permanent infrastructure, and production normally runs beside them. **They are not stopped and nobody is asked to stop them.**
Measuring `c` with them running is measuring the environment V1 would ship into; the run is quoted as **"`c` measured under the box's steady-state load"** and `meta.json` records that load rather than assuming it away. This **supersedes** what sections 12 and 13 said about quiet, in three places:

1. **No absolute idle-drive limit.** Section 12's rule that an idle cell reading above 0.02 GB/s makes the run INVALID is **withdrawn**: the services do real drive I/O. In its place the refusal is on **drift**: within a pass, foreign NVMe traffic (reads plus writes over every whole NVMe device, the reader's own bytes removed from the load cells) may differ between the idle cells and the load cells by at most **0.10 GB/s**,
   and box-wide foreign CPU by at most **1.0 core** (section 13, kept). Either drift writes `results.INVALID` for the `nvme` ratio (exit 3). **A steady load cancels in `rho`; a drifting one invalidates it.** Both services burst (sync, peer churn), so drift is a live possibility, not a formality.
2. **Their I/O is sampled per cell, not only their CPU.** Per visit: read and write bytes of `nimbus_beacon_n`, `op-reth`, `reth-binary`, `aggregate-runner` and any process among the top CPU users, from `/proc/<pid>/io` (**readable only for our own uid; a process it cannot read is recorded as "unreadable", never as zero**), and the per-device NVMe read/write deltas. On 2026-09-21 only `nimbus` was readable
   (0.06 MB/s written); `op-reth` and `reth-binary` were not, so their share is bounded only through the per-device totals. `meta.json` also records the command lines of the top foreign processes at the end.
3. **The `nvme` arm is not deferred because the box is busy.** It is deferred (reported NOT MEASURED) only if `quiet_check.py` cannot find reader cores under 5% busy, or a drift gate fires and the run is not repeated. "The box is not quiet" alone is no longer a reason.

**What the steady state looks like (observed 2026-09-21, `quiet_check.py`, 8 s).** Foreign CPU was **21 cores** (kernel I/O threads excluded), not the roughly 4 that `ps` suggested: `java` (QuestDB, `questdb.jar`, about 14 cores of `ilpwriter` threads), `aggregate-runner` (about 3), one pytest lane of ours, `nimbus_beacon_node` and `op-reth`. NVMe traffic across all devices was
0.018 GB/s read and 0.008 GB/s written, so drive I/O was small at that moment. Load average 23.5-35.9 through the afternoon.

**A tension I cannot resolve by design and am recording before the run.** The frozen per-core gate (`foreign_max_core_pct` at most 10% on the cores we use) is not a drift rule; with a 21-core foreign floor a service thread landing on one of our cores for 100 ms of a 1-second visit fails it. Cores are chosen as the quietest by `quiet_check.py` (four harness cores on node 0, four reader cores on node 1) to minimise that.
**If the run is INVALID and the only failing gate is `foreign_max_core_pct`, and the recorded `foreign_where` names a steady service on one of our cores, I ask for one re-run on freshly picked cores, with both runs reported.** I will not loosen the gate or drop cells to get a VALID run. That request is not a permission I take; it needs the lead's yes after seeing the failure.

## 15. Amendment before any data, fourth: retractions, what the per-cell gates can and cannot catch, and the retry rule (2026-09-21)

**1. Retractions (the lead measured the drives).** `/data`, where `nimbus_beacon_node`, `op-reth` and `reth-binary` keep their data, is the LVM root volume (`/dev/mapper/rl00-root`), not an NVMe mount; the harness drives are `nvme0n1`, `nvme1n1`, `nvme2n1` and `nvme3n1` (mounts nvme0, nvme1, nvme2, nvme4), and a 10 s
`/proc/diskstats` sample showed 0.0000 GB/s on three of them and 0.0132 GB/s on `nvme1n1`, which is one of our own lanes. So the services do **not** touch the drives the harness reads. **Withdrawn:** section 14's per-process `/proc/<pid>/io` sampling (removed from the harness), its 0.10 GB/s drive-drift gate, and its statement that the 0.02 GB/s idle rule is withdrawn.
**Restored:** an idle-arm cell that moved more than **0.02 GB/s** (reads plus writes over the whole NVMe devices) is contaminated and the run writes `results.INVALID` (exit 3); a load cell must match the reader's own bytes within 10%. Per-device deltas are still recorded. `quiet_check.py` again requires the drives under 0.02 GB/s for GO.

**2. What the per-cell gates catch about a daemon migrating onto a harness core mid-run (the lead's question).** Three different things, so it matters which:
- **The foreign-CPU measure (`foreign_max_core_pct`, per visit) catches it, at the visit where it happens.** It is per-core foreign CPU from `/proc/stat` (user + nice + system ticks, our own threads subtracted, irq/softirq/steal/iowait excluded so our own NVMe and GPU interrupt time is not counted against us). **Resolution:** ticks are 10 ms, so a core with fewer than 3 foreign ticks in a window is reported as at most 9.9% (one or two ticks in a 100-200 ms visit are not evidence of 10%).
- **`p99 / p50` does NOT do this job.** It is a frozen cell-level gate that fails the whole run; it only sees launch-side stalls (a preempted launching thread lets the GPU run past the 600 us spin, so that launch's `T` is inflated), and only if at least two of a cell's 200 launches are affected. A uniform slowdown (memory-bandwidth contention, an SMT sibling) does not move it at all. So it is not a mid-run migration detector.
- **Retry rule (new, harness-level, declared before data).** A visit (100 launches) that fails an *environmental* check is re-run in place, up to 3 attempts: foreign CPU above 10% on our cores, another process on the GPU, or the PCIe link not at Gen3. **No `T` is looked at** (a value-based retry would select on the result), the P-state is deliberately not a retry criterion (it is the registered fallback, not a transient), and every failed attempt is written to `retries.jsonl`
  with its reason and the foreign core and load. If all 3 attempts fail the last is kept and the frozen gate judges it. `meta.json` records `visits`, `retried_visits` and `retry_fraction`; a fraction above 10% is reported with the result. **This converts a migration onto a harness core from a lost run into a re-run visit; it does not touch memory-bandwidth contention, which stays an ungated failure mode (section 12, item 2).**

**3. Steady state as actually measured.** `quiet_check.py` (thread scan, 2 s) at 04:5x: foreign CPU **22 cores**: QuestDB `java` about **14 cores**, `aggregate-runner` about 3.3, `nimbus_beacon_node` 1.1, one lane's pytest 0.8, `op-reth` 0.5, `reth-binary` 0.4. The lead's `ps`-based estimate of about 4.9 cores understates it, because `ps` `%CPU` is a lifetime average and QuestDB's ingest was running.
This matters to the box-drift gate: **QuestDB alone moving by a core or more between the idle and load arms of a pass would make the `nvme` ratio INVALID.** If a pass fails that gate marginally, first check whether it is one daemon breathing (`aggregate-runner` alone swings by about 0.2 cores) before concluding that a lane collided. The per-pass top-foreign list and command lines are in `meta.json` `steady_state_by_pass`.

**4. Hash discipline.** Section 9 lists the harness in force and, separately, every superseded hash with the amendment that superseded it. `verify_hashes.py` checks the files against the section 9 table and is run **immediately before the window**, not only at staging, together with a plain `sha256sum` of `c_analysis.py` against `4657702c…c21c`.

## 16. Amendment before any data, fifth: the re-run conditions, node membership, and what "GO" means (2026-09-21)

**1. The one re-run, with the lead's five conditions registered here before any run.** If the full run is INVALID **and the only failing gate is `foreign_max_core_pct`**, then:
(1) **exactly one** re-run is permitted; if the second attempt fails the same gate the answer is **NOT MEASURED** and we stop (two is a core-placement retry; three would be fitting the placement to the result);
(2) the **INVALID run is reported in full**, in its own output directory, beside the second, not as a footnote;
(3) **`foreign_where` must name a steady service** (nimbus, op-reth, reth-binary, aggregate-runner, QuestDB `java`, the Ray `log_monitor.py`, and the like) **on one of our cores**; if the cause is one of our own lanes the re-run is not authorised and the lead is told instead, to clear the lane;
(4) the new cores are picked by `quiet_check.py` **on its own criteria from a fresh sample**, never chosen to avoid where the daemon was seen;
(5) no gate is loosened and no cell is dropped to obtain a VALID run. (This is in addition to the per-visit retry rule of section 15, which acts inside one run; and it does not cover the P-state fallback, which has its own conditions in section 13.)

**2. Node membership comes from `/sys`, not from a split point.** This box's NUMA layout is interleaved (node 0: cores 0-17 and 36-53; node 1: 18-35 and 54-71). `quiet_check.py` reads `node_cpus()` from `/sys/devices/system/node/node*/cpulist` and derives the candidates in one pure function (`candidates()`): harness = node 0 within 32-63 = **cores 36-53 (18 candidates)**; reader = node 1, outside 32-63, never 64-71 = **cores 18-31 (14 candidates)**.
A test pins that (and shows a contiguous split would give 32-35 instead). The cores picked in earlier pre-flights (harness 43, 49, 52, 53; reader 18, 19, 23, 27) were inside those sets; the pre-flight now also prints the candidate lists so a wrong set is visible.

**3. What "GO" means, and why it is not my 5% idle-core margin.** The frozen gate is 10% *foreign CPU on the cores we use while we run*, not "the core is idle beforehand". The launching thread occupies one core fully, and foreign threads are placed on idle cores by the scheduler, so a core that looks 17-27% busy in a survey (the team's own OMP lanes and the services spread over the mask) says little about how much foreign CPU a cell will see on the core we are on.
The idle-core figure stays in the pre-flight as an **informational** line. **The registered GO is a CPU-only dress rehearsal (`quiet_check.py --rehearse 40`): a spinner stands in for our launching thread and for the reader on the picked cores, and the foreign CPU on the mask is measured, by the same code the harness uses, in 0.5 s windows; GO needs at most 5% of the windows above 10% on either mask (a visit has 3 attempts, so a per-window failure rate `p` costs `p^3` per visit) and the NVMe drives under 0.02 GB/s.**
The harness mask is the two quietest harness candidates and the reader's the three quietest reader candidates (`taskset -c` inside `gpu-run.sh`'s 32-63; the harness needs one core for the launching thread and one spare). If no pair or triple meets it, the answer is that this box cannot host the measurement today and the window does not open. The rehearsal runs a few seconds of spinners on three cores; it needs the lead's yes to be run on divix01.

**4. The record.** The list of standing services in `meta.json` (`steady_state_by_pass`, from the per-pass thread scan) will include, besides `java` (QuestDB, measured flat at about 14.4 cores in three samples: 1444%, 1448%, 1430%), `aggregate-runner`, `nimbus_beacon_node`, `op-reth` and `reth-binary`, a **Ray `log_monitor.py`** (about 14% of a core, on core 70, which is inside the 64-71 band we never touch and outside our masks). The flat QuestDB figure also lowers the risk of the box-drift gate: there is no ingest burst to swing it.

## 17. Amendment 6, made after the first rehearsal and before any measurement: which cores and which drives the gates judge, with the SMT correction (2026-09-21)

**Status: ACCEPTED by the lead in principle, with one mandatory correction (item 5); NOT used today** (today's window did not open; see item 6). The files are in `c_measurement/proposed_amendment6/` and become the harness in force, with section 9 flipped in a separate commit, only when a run is next attempted; until then the harness in force is still `35dcf8215ae5c2c9…` (section 9).

**What happened, in the order it happened.**
1. **The first rehearsal (the in-force code, run 04:58 on divix01, `--rehearse 40`) said NO-GO: 72% of 80 windows above 10% on the harness mask [36, 38] and 66% on the reader mask [25, 27, 30].** No lane of ours was running (a `pgrep` for pytest found none), so this is not a false negative caused by us; against the in-force gate as implemented, the box fails.
2. **A diagnostic** (CPU-only, 20 windows of 0.5 s on each of cores 36 and 46 with a spinner on the core, foreign CPU by the harness's own code): **foreign ticks on the core we occupy were 0 in 19 of 20 windows and 1 tick in the twentieth (core 36), and 0 in all 20 (core 46).** The NO-GO came from the *spare* core of the mask (core 38, 14-20% busy from the services), on which our thread was not running.
3. **The implementation had counted every core of the allowed mask.** Section 12 says the gate counts "the cores the run uses". A spare core on which our threads did nothing cannot contend with us for CPU. The proposed code counts the cores **where our own threads accrued at least 10% of the window** (all cores of the set when we have no presence at all, so a plain survey stays conservative), which is the text of section 12 and not a new gate.
4. **A second defect, found in the same rehearsal:** its NVMe criterion summed **all** whole NVMe devices, and `nvme1n1` (0.019 GB/s read, 0.005 written, not read by this harness, not one of ours by the lead's measurement) alone exceeded 0.02 GB/s. The rules exist to protect `rho`'s baseline on the drives the reader reads; the proposal judges the devices holding the mirror roots and the source checkpoint (`nvme0n1`, `nvme3n1`, `nvme2n1`, resolved through `st_dev` and `/sys/dev/block`, never by mount label) and **records every device**.

**Disclosure of the ordering.** Both changes were made after seeing a rehearsal result. They are made before any measurement of `c`, they follow from the text already registered in section 12 (cores "the run uses") and from the stated purpose of the drive rule, and the rehearsal that motivated them is reported here in full, including that the in-force code said NO-GO. That is still a decision made with a result in view, and the lead should weigh it as such.

**The rehearsal under the proposed code** (two runs, both with no lane of ours running, load 25-32, foreign CPU 20.6-29.9 cores box-wide, QuestDB and the other services present): harness mask **0% of 80 windows above 10%**, reader mask **0% of 80**; NVMe on the watched drives **0.0000 GB/s** read and written (all devices: 0.017 read, 0.008 written). The first of the two runs printed NO-GO only because of the all-devices NVMe criterion of item 4; the second, with the watched-drive rule, prints **GO**.

**What the proposal does not change.** `c_analysis.py` (`4657702c…`) and every frozen gate value; the per-visit retry rule; the box-drift gate; the ungated memory-bandwidth failure mode (section 12), which the used-cores rule does not touch and does not weaken any more than the mask restriction already did.

**5. The lead's SMT correction (required after the proposal, and it cuts against my result).** SMT is active on this box (2 threads per core, 18 cores per socket, 2 sockets; e.g. `thread_siblings_list` of cpu42 is 6,42; cpu44 is 8,44; cpu36 is 0,36). A logical CPU is half a physical core, so "the cores the run uses" are **physical cores**: my used-core rule was too wide on spare CPUs of the mask **and too narrow on the SMT siblings outside it**.
The lead measured foreign CPU on the siblings of my proposed masks at 29.7%, 44.9%, 13.9%, 31.9%, 7.7% and 100% while my rehearsal reported 0% of windows above 10%; both were true, which is the hole. My diagnosis (foreign ticks on the occupied CPU are 0) was correct about why spare-core counting was wrong and incomplete about where the contention went: **the scheduler moves foreign threads off a CPU we spin on, but not off its idle sibling.**
Implemented (v2 of the files below): foreign CPU is summed over each used logical CPU **and its thread sibling(s)** (from `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`) against the same 10% gate; `quiet_check.py` scores candidates by **physical core, worst sibling**, prints both members of each pair, and needs both siblings quiet; and **any CPU whose sibling lies in the reserved 64-71 band is excluded permanently in `candidates()`** (reader cpus 28-31 pair with 64-67, cpu35 with cpu71 itself, the doorbell spin core; the reader set is 10 CPUs, not 14).
Conditions the lead set: report all three rehearsals; keep the ordering disclosure and add that this correction was required after the proposal; no further amendment after a result without coming back to the lead first; `c_analysis.py` stays `4657702c…`.

**6. The three rehearsals (full list in `c_measurement/rehearsal_log.md`).** (1) in-force code, clean box: **NO-GO** 72%/66%. (2) the proposal, before the SMT correction: GO-shaped, **but the runs that produced it are contaminated (a relaunched order sweep) or unverified, so this GO is not evidence**; it is reported here because it shows how the un-corrected rule behaved (it certified physical cores whose siblings were 30-100% foreign-busy).
(3) **the SMT-corrected code, 05:20:54-05:22:20, `pgrep` for `orderplug|pytest|sweep3` = 0 at the start and 0 at the end, load 27.9-30.5, foreign 21.1 cores: NO-GO, 40% of 80 windows above 10% on the harness mask [46, 50] and 40% on the reader mask [18, 23, 26]; physical cores with both siblings under 10%: harness 0 (need 2), reader 0 (need 3); the best pair's worst side was 12% (harness) and 21% (reader).**
On the box as it is, **the frozen gate cannot be met today.** Whether that makes the measurement NOT MEASURED or lets it proceed on a declared sibling load turns on whether a steady sibling load biases `c_m` absolutely; that question and its proposed pilot are in the lead thread, not decided here.

**Files and hashes (v3: SMT-corrected plus the lane guard of item 7; 22 CPU tests pass on them; `quiet_check.py` and `test_c_harness.py` supersede the v2 versions of those two files in commit `a6e410bbbe`; `c_harness.py` is unchanged from v2):** 

**7. The lane guard (accepted by the lead: "a pre-flight that cannot detect contamination at its own start is the defect we have just spent an hour paying for").** `quiet_check.py` (v3) scans `/proc` for the team's own lanes (command lines containing `pytest`, `orderplug`, `sweep`, `mutate` or `xargs -P`, excluding itself, its parents and the lead's `sweep_guard` watchdog) **before sampling anything: any hit aborts with exit 2**; it records the count again **at the end of the survey and at the end of the rehearsal, and a nonzero end count voids the result** ("CONTAMINATED DURING") whatever the windows said.
A mid-run arrival during the measurement itself is visible afterwards in `meta.json` `steady_state_by_pass` (the per-pass thread scan names the top foreign processes and command lines). The first version of the guard flagged the lead's own watchdog (`/tmp/sweep_guard.sh`) at once, which is the pattern doing its job; the watchdog is now excluded by name. The lead's cause for the repeated relaunches (an automatic retry loop, eleven seconds apart) is recorded here as the lead's finding, not mine.

**8. `gpu-run.sh` and the masks compose (checked on divix01).** `taskset -c 32-63 flock -n -E 75 "$LOCK" "$@"` applies an affinity mask, not a cpuset (`cpuset.cpus.effective` is 0-71), so an inner `taskset` may narrow or move it: `taskset -c 32-63 taskset -c 46,50 …` gives [46, 50] and the reader child's `taskset -c 22,23` under a parent pinned to 32-63 gives [22, 23]. So the harness mask (inside 32-63) and the reader mask (outside it) are both honoured; the outer 32-63 is only a default.

| file | sha256 |
|---|---|
| `proposed_amendment6/c_harness.py` | `ca12a3ae35d6a454c9a63507298862290c4c0729b1788219fc54e892ef31a15f` |
| `proposed_amendment6/quiet_check.py` | `1e715a8929b0451e4664e0a7a5297c0e3c95a4fea9a4308b6c9b86269297a569` |
| `proposed_amendment6/test_c_harness.py` | `b735ca4723768fdcb45f87360a31f538a24bb08d8f0ed35a55ac1561f4a32904` |

## 18. The SMT-sibling pilot: pre-registration (2026-09-21; approved by the lead; nothing has been run)

**Question.** Does a busy SMT sibling of the launching CPU shift the GPU-side time `T` of one production gather launch (cold rows, node 0)? This decides whether a steady sibling load can be treated as a *declared condition* on this box (it enters `T` not at all) or whether it makes any absolute `c_m` measured under sibling load meaningless. It changes **no gate**: a "not in T" verdict is an argument for a future amendment that goes to the lead first. `c_analysis.py` stays `4657702c…`.

**Design (fixed before any run).** One process pinned to **one** logical CPU `L` (`taskset -c L`; refuses to start otherwise); its SMT sibling `S` is read from `/sys`. A spinner process is pinned to `S` and switched **on and off with SIGCONT / SIGSTOP** (it stays alive, so a switch costs microseconds and nothing is forked).
The measured kernel and path are the harness's `RealDevice.run_visit` unchanged: production `copy_expert_row_segments_gpu_kernel`, cold distinct rows from the service-allocated slabs on node 0 (2 GiB), 64 scratch slots, a spin kernel before every timed launch, 20 discarded warm-up launches then 100 timed launches per visit. `n = 3` and `n = 6`.
Per rep and per `n`: visits **A, B, B, A** (A = spinner stopped, B = spinner running). **20 reps.** About 160 visits, about 0.4-1 s each, plus 0.3 s for the spinner to settle before each B: roughly 2 minutes of GPU time and about 3 minutes with allocation.
`nvidia-smi` samplers are moved off `L` and `S`. Only node 0 is allocated (no eviction of node 1's page cache).

**Per visit it records** (not just that a spinner was started): the spinner's **achieved busy fraction** (its own ticks over the window), the **foreign busy fraction of the sibling** (busy ticks of `S` from `/proc/stat` minus the spinner's ticks) and **of `L`** (busy ticks minus our own), with the harness's tick-resolution floor (fewer than 3 ticks is reported as at most 9.9%).

**Validity (registered).** A visit is valid iff the foreign busy fraction of `L` and of `S` are both under 10% **and** (B: the spinner achieved at least 90% of the window; A: the spinner accrued zero ticks). A **foreign burst on the sibling in an A visit would make A look like B, so it is excluded, not counted**; the same for B. A `(rep, n)` unit is valid iff all four of its visits are.
**At least 8 valid reps per `n`, else INVALID.** The whole run is INVALID (no verdict quoted) if the PCIe link is not at Gen3 throughout, another process used the GPU, a lane of ours ran at the start (abort) or the end (`pgrep`-equivalent, `quiet_check.lane_processes()`).
The lead's reason for this: a burst landing on the sibling *specifically* corrupts the contrast, not the level, and the failure direction that matters is a false "not in T", which would license running on this box.

**Statistic and verdict.** `shift(rep, n) = median(T of the two B visits) / median(T of the two A visits) - 1`; per `n` the mean over valid reps and a two-sided 95% t-interval (df = valid reps - 1). **Verdict: INSENSITIVE** if both `n` have the whole interval inside +-0.5%; **SENSITIVE** if either `n` has the whole interval outside +-0.5%; **INCONCLUSIVE** otherwise. The verdict word is reported first.
INSENSITIVE means a busy sibling did not move `T` by more than 0.5% (the sibling can then be a recorded, declared condition for the cold-SM `c_m`); SENSITIVE means a sibling load above 10% makes an absolute `c_m` unmeasurable on this box while the services run (a run needs a quiet pair, which does not exist today); INCONCLUSIVE means neither, and the pilot is repeated or not, at the lead's call.

**Two biases that both favour V1, stated together so a marginal pro-V1 result is read with both in view.** (a) A non-idle `T_idle` (the box's steady services) is inflated, so `rho = T_load / T_idle` is understated (section 13). (b) The `nvme` arm's reader loses throughput when its SMT sibling is busy, so the concurrent load is weaker than production's could be. Neither is measured; both point the same way.

**What it does not do.** It does not measure `c`; it does not test the `hot`, `ce`, graph or `nvme` arms (the sibling can matter for `ce`, which includes host enqueue in `T`, for `hot` through cache state and for `nvme` through reader throughput; the pilot speaks only to the cold SM launch); it does not change any gate.

**Frozen files** (commit and hashes as of this section; the pilot is run from the staged copies and `verify_hashes`-style checked immediately before):

| file | sha256 |
|---|---|
| `sibling_pilot/sibling_pilot.py` (the runner (GPU; needs the lock)) | `4b9219521077bb8e3ccd251b8047dd895bba2d278222f4dacba88ff5e2afdff0` |
| `sibling_pilot/sibling_pilot_analysis.py` (the frozen rule and verdict (CPU; `--selftest` passes all ten cases)) | `5d85f67c946e55445598421ff79765be0e40fb748a131e55988fd8d9b24b3bc5` |
| `sibling_pilot/test_sibling_pilot.py` (3 CPU tests incl. the dry-run plumbing) | `8709257809e254459a7079743b79525250d702a2934cc1c12142527edcf4f076` |
| `proposed_amendment6/c_harness.py` (v4: only change from v3 is `RealDevice.setup` taking `nodes`, so the pilot allocates node 0 only) | `ca12a3ae35d6a454c9a63507298862290c4c0729b1788219fc54e892ef31a15f` |
| `proposed_amendment6/quiet_check.py` (unchanged from v3 (`lane_processes()` is used by the pilot)) | `1e715a8929b0451e4664e0a7a5297c0e3c95a4fea9a4308b6c9b86269297a569` |

### 18.1 Run 1 of the pilot: INVALID, twice over; no verdict is quoted (2026-09-21, 05:37:55-05:40:26, raw output on divix01 in `c_measurement_run/pilot_out_053754/`, `meta.json` in `sibling_pilot/run1_meta.json`)

**Verdict word: INVALID.** Two independent registered reasons, either sufficient:
1. **Fewer than 8 valid reps per `n`: 7 of 20 for `n = 3` and 7 of 20 for `n = 6`** (126 of 160 visits valid). The B arm was clean and the instrument behaved (spinner achieved **99%** of the window in every B visit, minimum 99; the sibling's foreign busy in B was at most 2.1%; the launching CPU's foreign busy at most 4.3%; the A spinner accrued 0 ticks in every A visit). **The A arm was not: the sibling of a spinning CPU is foreign-busy in A** (median 8.5%, **maximum 93.2%**), because the scheduler moves foreign threads off the CPU we spin on and onto its idle sibling, exactly the hole the lead measured for the main gate.
   Reps with any A visit at 10% or more were excluded, as registered; 13 of 20 remained excluded for each `n`.
2. **A "lane of ours" at the END** (`quiet_check.lane_processes()`): two short-lived `bash -c test -f /data/models/slang/nvfp4-work/t1-ordersweep/sweep.done` pollers (the order-sweep lane's release probe, matched by the `sweep` substring). They are almost certainly harmless (no CPU), but the rule as registered is any hit, so the run is voided by it too. I do not reinterpret the rule after the fact.
Also recorded: link Gen3 throughout, P1 (busy), SM clock 2760-2970 MHz, no other GPU process; the sweep's `t1py` unit was stopped by the lead before the run, `pgrep` was clean at the start (the in-harness guard passed).

**No shift and no verdict was computed or looked at**: only the validity statistics above were read from `visits.jsonl`. **Bugs found on the first real GPU launches** (before any timed data, so they cost nothing but the minute): `RealDevice._plan` used a bare `torch` (`NameError`) and `run_visit` called `os.sched_getcpu`, which does not exist. Fixed in `proposed_amendment6/c_harness.py` (v6, hash updated above). **The in-force harness `35dcf8215ae5c2c9…` carries both bugs**; had the main run gone ahead it would have crashed at `--check-only`. `pyflakes` now clean on the undefined-name class of error; an attribute error like the second is not caught by it, which is why the first real launch is the test.

**What this does and does not say.** It says nothing about whether a busy sibling moves `T`. It says something about the box that the pilot's design depends on: **on divix01 the idle SMT sibling of a CPU we occupy is foreign-busy above 10% in roughly a third of 0.5-0.9 s windows even on the quietest physical core the survey found**, so a clean A arm is available only about a third of the time.
**Options, none taken without the lead:** re-run with more reps (about 40, so that about 14 valid are expected against the minimum of 8: a change of a registered parameter after an INVALID, so it needs the lead's yes and a new hash), and either stop the sweep lane's `test -f` pollers or exclude pure `test -f` probes from the lane pattern (a guard change; it would be made before, not after, looking at any shift).
| `proposed_amendment6/test_c_harness.py` (25 CPU tests, incl. the stdlib-attribute existence check, the `_plan`/launch-CPU test and the guard-narrowing test) | `b6815a802c84e9bf8ec831dcb1ec3f23c855cb9e315108fc7a023860910b27a3` |

### 18.2 What the first real launch showed about the instrument, and the guard change (2026-09-21)

**The `c` run would have crashed on its first execution.** `RealDevice._plan` used an undefined `torch` and `run_visit` called `os.sched_getcpu`, which does not exist; both are in the **in-force** harness `35dcf8215ae5c2c9…`, the one gated, hashed and amended five times today, and the real path had never run on a GPU. So **had the box passed the gate the `c` run would have failed at `--check-only`**: today's NOT MEASURED was over-determined, by the box (the gate is unsatisfiable) **and** by an instrument that could not have completed. The natural reading of the day, "divix01 was too busy", is incomplete.
**No measurement existed when either fix was made** (both crashed before any timed launch was recorded), so repairing them is not amending a gate in response to data; the new hash is recorded above and the in-force harness is noted as carrying both bugs.
**Why 19-22 CPU tests passed while the real path was broken:** they exercised everything except the thing the harness exists to do. They asserted outcomes (a dry run producing a `results.jsonl`, gates firing on synthetic input) that a non-functional GPU path also produces, and nothing witnessed that the real launch path ran at all. Both defects are of the kind a smoke test on the real path catches in one second: an import-scope error and an environment-dependent name. **Two tests now stand in the list before the next attempt** (both need no GPU): `test_stdlib_attributes_used_by_the_scripts_exist` (every `module.attr` used on an imported stdlib module must exist; it would have caught `os.sched_getcpu`) and `test_plan_and_launch_cpu_construct_without_a_gpu` (constructs the plan and reads the launch CPU; it would have caught both). This is a worked example, from our own tooling, of the two-part rule of the cannot-fail taxonomy (`ac5af3223b`): a test that asserts something a broken path also satisfies, with no witness that the path under test ran.

**The guard change, made before any re-run.** Run 1's INVALID reason 2 was a false positive: `bash -c test -f /data/models/slang/nvfp4-work/t1-ordersweep/sweep.done` (pids 3274284 and 3274313), a file-existence poll that matched the `sweep` substring of a path. The guard now judges **what is executed, not which words appear**: a process whose command (or `bash -c` body) begins with `test`, `[`, `ls`, `cat`, `grep`, `tail`, `head`, `sleep`, `pgrep`, `stat`, `wc`, `echo`, `sed`, `awk`, `ssh`, `systemctl`, `date`, `readlink`, `find` or `ps` is not a lane, while a launcher (`bash -c cd … && systemd-run … sweep_py.py`), `python -m pytest`, and `xargs -P` still are. The lead's remark that a command-line regex is a proxy for "is a process consuming CPU that could contend with us" stands; a CPU-time-based check remains the better guard if this proves fiddly again.
**Run 1 stays as it is:** its directory is kept, the INVALID marker stands, no shift was computed. **Disclosure:** before the instruction not to analyse `visits.jsonl` arrived I had read its validity statistics (spinner achieved fraction, foreign busy fractions, valid-rep counts) to decide whether a re-run was feasible; I read no `T` value and computed no shift.

### 18.3 Ray writes to a watched drive (from the lead)

`/mnt/nvme4/ray_tmp` is on `/dev/nvme3n1p1`, and `nvme3n1` is in the watched set (mirror roots plus source). Ray's `log_monitor` and `dashboard` write session logs there with `--logging-rotate-bytes=536870912`, so its traffic is bursty by construction; measured quiet at 0.0000 GB/s over 10 s today. **A Ray rotation can trip the idle-cell rule or break the load-cell match for a reason that is neither us nor a lane**; the per-cell per-device bytes (`drive_by_dev`) are recorded, and **Ray is a known writer to a watched drive**, to be named in `meta.json` for the `c` run. If rotations prove frequent enough to make the `nvme` arm unreliable, the remedy is to move the source checkpoint off `nvme3n1`, not to loosen the rule. (`nvme1n1` read a steady 0.0266 GB/s with our lanes stopped; it is not watched and its reader is unidentified.)
