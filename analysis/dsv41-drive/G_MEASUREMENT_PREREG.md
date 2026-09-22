# Pre-registration: the per-stage cost `g` (Task 6, V2 over V1), 2026-09-21

**Status: design only. Nothing has been run, nothing is to run until (1) the lead has accepted this document, (2) the harness has been written and its sha256
appended to section 9 in a second commit before any GPU time is taken, (3) `cc-gpu.lock` is held and crypto-c9 has scheduled the window.** CPU work so far:
the exposure counts (section 1.3) and the analysis script with a synthetic self-test (section 9), both under `taskset -c 0-63`, one thread. Author: t3-topology.
This document was written before any `g` number exists, and the decision rule in section 2 is fixed by the hash in section 9: a result is judged by it, not by
how the number looks afterwards.

## 0. Why this measurement, and what it can and cannot decide

Per-row's rejection at best order rests on one unmeasured number. Best-order per-row over two-phase is +4.93 ms per decode step gross (k-free: `Sigma(m-1)*c`
over 511 steps); the plan's bar is 1.5% of 254.4 ms = **3.816 ms**; so best-order per-row can carry at most **`G*` = 4.93 - 3.816 = 1.114 ms per step** of extra stage cost and
still clear the bar. If the plan's "160 extra triples, all exposed, one `g`" held, that is `g* = 1.114 ms / 160 = 6.96 us`, and the assumed 8-14 us is 1 us above it.

**What follows from that arithmetic, before any measurement:**

- **Two conditions in the plan's `160 x g` are wrong or unstated, and they move the crossing by more than the margin.** Section 1.
- **`g*` is not a property of the GPU alone.** The gross 4.93 is proportional to `c` (1.055 ms per row, taken from one older trace and never measured for this path). A 10%
  error in `c` moves the crossing by about 3 us, three times the 1 us by which the assumed 8-14 us range sits above it (table in section 2.3). **A measurement of `g` can settle the verdict
  only if `g` lands far from the crossing; it cannot rescue or condemn per-row against an error in `c`.** This is stated here so nobody expects it later.
- The real `W_s` and `A_s` kernels do not exist (Task 5 is unimplemented), so the measurement uses stand-ins. What it can give is a **bounded** `g`, not the production `g`.
  The only direct test is the traced arm A3 (`PER_ROW_TRANSFER.md` 5.6). If that is judged too weak to be worth a GPU slot, that is a legitimate answer and section 8 says so.

## 1. What `g` is

### 1.1 What the plan and `PER_ROW_TRANSFER.md` 5.6 mean, and where "about 5 us" came from

5.6: "`g` the fixed cost of one stage triple (three dependent kernel launches plus one poll round trip)"; "plausibly 8-14 us **[A, unmeasured]**"; and "record V2 REJECTED
unless a measurement shows `g` under about 5 us and the order is best". That is neither the marginal cost of one more triple, nor an amortised cost at realistic counts: it
is a *composition* (three launches + a poll), applied to **all 160** extra triples as if every one were identical and fully on the critical path.

**"About 5 us" cannot be reproduced from 5.6's own inputs.** I tried the registered gross (4.93 and 5.06 and 5.09), bars of 1.0%, 1.5% and 2.0%, denominators 254.4, 257.5 and
259.3: `g*` is 6.7-8.0 us at 1.5%, about 14.9 us at 1.0%, and zero or negative at 2.0%; none is 5. `PLAN_TASK6_REVIEW.md` reached 6.96 the same way. I do not know its origin. One
guess, labelled as a guess and not checked: `DSV41_REFERENCE.md` 18.2 records the existing wait kernel's median as 5 us on a layer that waits for nothing, and that number may have
been carried over as `g`. If so it is a single kernel, not a triple. **Nothing below uses 5 us.**

### 1.2 The definition registered here

`g` is a **marginal, exposed, step-time cost**: the increase in the GPU-side time of a replay of a captured graph per additional stage triple placed in series in that
graph, `g = d(T_replay)/dN` at N in the realistic range, in microseconds. Two kinds of triple, because they are different objects:

- **`g_e`, an empty stage triple**: `W_s` finds the stage range empty and returns with `go[s] = 0`, `C_s` is the real copy kernel called with `count = 0`, `A_s` acknowledges
  nothing (`PER_ROW_TRANSFER.md` 4.1: "an empty stage's three kernels still launch and return at once"). No host word is read. This is three dependent kernel launches.
- **`g_a(p)`, an active stage triple** (one lane, its readiness word already published, so no waiting): `W_s` reads `p` host words with system-scope acquire loads *in
  series* (each is a PCIe round trip, and acquire ordering serialises them), writes `go[s] = 1`; `C_s` is the real copy kernel with `count = 1` moving a **4 KiB** row (data
  movement is `c`, a separate quantity; this measures the launch and fixed cost of a copy that moves nothing significant); `A_s` issues `__threadfence_system()` and a
  `st.release.sys` to a mapped word. **Registered `p = 4`**: the fail-closed conditions of 5.5, read as written (`page fatal`, `Header.shutdown`, `RowResult.ready` acquire,
  seqlock re-read). `p = 1` (the optimistic reading: one word) and `p = 6` (a second lane's word and one more) are measured as sensitivity and cannot change the registered verdict.

`g` is **not** the launch gap alone (which 5.6's own microbenchmark bullet would measure: "a chain of `S` empty stage triples ... `nsys --cuda-graph-trace=node` for the
inter-kernel gap only"). An empty chain measured by gaps omits the poll round trips, the ack fence and the copy kernel's active-path cost, all of which are in `g_a`. **This
design supersedes that bullet** if the owner agrees.

### 1.3 Not all 160 triples are exposed and not all are the same kind (counted from the measured lanes)

V1 has `S = 2` stages, V2 has `S = 6`; V2 minus V1 is 4 extra stages on every layer, 39.89 layers per step in the trace, **159.55 triples per step**. Classified with the measured
per-layer lane counts of `task1f` (`g_exposure_counts.py`, per decode step over 511 steps):

| class | what | per step | exposed? |
|---|---|---:|---|
| A | extra **empty** stages (stage index beyond the layer's `k`; they sit at the tail) | **85.10** | always: after the last active stage nothing hides them |
| B | extra **active** stages in layers that read nothing | **48.93** | always: the GPU waits for nothing else |
| C | extra **hit-lane** stages in layers that read | **20.86** | **hidden** if the chain of hit stages finishes inside the read wait (the precheck's `hide_ok` was 94% under A1's `k`; not recomputed with measured `k`) |
| D | extra **miss-row** stages in layers that read | **4.65** | counted exposed (upper bound: exposed if rows bunch and the chain is the bottleneck) |

So the plan's `160 x g` treats class C as cost when it is probably hidden, and prices classes A (empty) and B/C/D (active) at one `g` when they differ by the poll round trips
and the copy's active path. With one uniform `g`: **all exposed** the crossing is 6.98 us; **C hidden** it is **8.03 us**, the low end of the plan's assumed range. Registered
accounting, both bounds always computed:

```text
G_X (all exposed)  = ( 85.10 * g_e + (48.93 + 20.86 + 4.65) * g_a ) / 1000     ms per step
G_H (C hidden)     = ( 85.10 * g_e + (48.93         + 4.65) * g_a ) / 1000     ms per step
net at best order  = 4.93 - G          (per-row over two-phase, before the resolution bar)
```

## 2. The decision rule (registered before any run)

### 2.1 Gates: the run is INVALID, and no number is quoted, if any fails

1. Every registered variant present, `N in {0, 40, 80, 160, 320, 640}` (`empty_base8k`: `{0, 160, 640}`), **3 independent processes**, and the node count equals `3N` (+8,000 for `empty_base8k`) in every cell (nothing optimised away).
2. **Positive control**: the `control20` variant (the empty triple with a 20 us spin added) must show a slope `20 +- 1` us above `empty` in every process. A harness that cannot recover a known 20 us is not measuring.
3. PCIe link generation is 3 at the start and end of every cell (the link idles at Gen1; see section 3, L3).
4. No other process used the GPU during any cell (`nvidia-smi pmon` sampled around each cell).
5. SM clock within 10% of its cell maximum throughout a cell.
6. **Linearity**: the slope on `N <= 160` and on `N >= 160` agree within 10% (a per-cell check, not a fit assumption).
7. **Production size**: the slope with 8,000 filler nodes underneath (`empty_base8k`) is within 15% of the bare-chain slope.

### 2.2 Verdicts

Slopes come from the OLS fit of per-N median replay time against N; each process's slope has a bootstrap 95% interval over batches; across the three processes the interval
reported is the **union** (lowest lower bound, highest upper bound), which is conservative on purpose. `G_X` and `G_H` are computed with interval arithmetic from `g_e` and
`g_a(p=4)`. `G* = 1.114 ms`.

| verdict | condition | what I will conclude |
|---|---|---|
| **CLEARS** | `G_X` upper bound `<= G*` | Even with every extra triple exposed, best-order per-row nets at least the bar (under the model's `c`, gross 4.93 and bar 3.816). **The "expected-REJECTED" label is withdrawn** and V2 is admitted as arm A3. It is not accepted: order array, per-drive FIFO and Task 5 still stand. |
| **REJECTION-STANDS** | `G_H` lower bound `>= G*` | Even with class C hidden, per-row's net is below the bar. The label stands **on a measured (stand-in) `g`**, no longer only on an assumed one. |
| **EXPOSURE-DEPENDENT** | `G_H` upper bound `< G* < G_X` lower bound | `g` is known and the verdict turns on whether class C is hidden, which a synthetic graph cannot show. The label stays "not distinguishable"; only the traced arm A3 decides. |
| **UNRESOLVED** | anything else (an interval contains `G*`) | The label stays "not distinguishable". **More replays will not help** (the intervals are already tight; the uncertainty is systematic); do not re-run for precision. |

The p-sensitivity (`active_p1`, `active_p6`) is reported, and if the verdict would differ at `p = 1` or `p = 6` I will say "depends on the design of `W_s`", not give one verdict.
A value that comes back "about 7 us" is resolved by this table, not by argument: 7 us is a *uniform-g* figure and this rule prices `g_e` and `g_a` separately.

### 2.3 Sensitivity of the crossing to inputs this measurement does not touch (uniform `g`, us)

| `c` (ms) | gross | `T` = 254.4: all exposed / C hidden | `T` = 259.3: all exposed / C hidden |
|---:|---:|---:|---:|
| 0.90 | 4.21 | 2.44 / 2.81 | 1.98 / 2.28 |
| 0.95 | 4.44 | 3.91 / 4.49 | 3.45 / 3.96 |
| **1.055** | **4.93** | **6.98 / 8.03** | 6.52 / 7.50 |
| 1.16 | 5.42 | 10.06 / 11.57 | 9.60 / 11.04 |
| 1.25 | 5.84 | 12.69 / 14.60 | 12.23 / 14.07 |

**The registered rule uses `c = 1.055`, gross 4.93 and `T = 254.4`; that row is the verdict.** The other rows are here so a reader can see that a 10% error in `c` swings the crossing from 2.4 to 12.7 us.

## 3. Predictions (mine, stated before the run; not part of the rule)

- `g_e` **5-10 us**: three dependent kernel-node launches at an assumed 1.5-3 us each (the plan's own A4' is 2-4 us, unmeasured), plus small kernel bodies.
- `g_a(p = 4)` **10-20 us**: `g_e` plus four serial PCIe reads (about 1-1.5 us each), the ack fence and store (about 1-2 us) and the copy's active-path fixed cost (the old E28 fit had 0.006 ms
  per launch, `MOE_EXPERT_TRANSFER.md`; an older tree).
- Priors that are *not* measurements of `g`: the existing wait kernel's median of 5 us on a layer that waits for nothing (`DSV41_REFERENCE.md` 18.2), the post kernel's 7.5 us mean, and
  `hit_path_us_per_layer` = 18.24 (`test_overheads`, `test_exl3_ram_miss_cuda.py`): that last is an **eager** loop of 400 steps, so it includes host launch cost and is not comparable.
- Under these predictions `G` is about 1.3-2.4 ms, above `G* = 1.114`: **REJECTION-STANDS is the expected verdict.** It would be a result only if it did *not* come out so.

## 4. The harness (to be written after this is accepted; its hash goes in section 9)

A standalone program, no engine, no model, no service thread. One process builds, captures with `cudaStreamBeginCapture` on one stream, and replays:

- **Kernels.** `C_s` is the **production** `copy_expert_row_segments_gpu_kernel` (same source, same launch geometry, grid 8 x 256), fed a synthetic segments table of one 4 KiB row and a device `count` tensor (0 or 1).
  `W_s` and `A_s` are stand-ins written to the 5.5 contract at the level that matters for cost: `W_s` = one thread, `p` serial `ld.acquire.sys` loads of words in a mapped host page (one 64 B line
  each), a device store of `go[s]`, `__nanosleep(256)` only if a word is not ready (registered: it is always ready); `A_s` = `__threadfence_system()` then `st.release.sys` to a distinct mapped line.
  The empty variants take the `go == 0` / empty-range exit with no host access.
- **Host page.** Allocated as the service allocates its page (pinned, mapped, on the GPU's NUMA node); the node of the page and of the GPU are recorded.
- **Variants** (each a captured chain of `N` triples in series): `empty` (`g_e`), `active_p1`, `active_p4` (registered), `active_p6`, `control20` (`empty` with a 20 us spin in `W_s`), `empty_base8k` (8,000 filler 1-thread kernels first, then `N` empty triples).
- **Sweep.** `N in {0, 40, 80, 160, 320, 640}` (a triple per layer-stage x 40 layers is 40-240; 640 tests the extrapolation).
- **Timing.** One `cudaEvent` pair around a batch of `R = 50` back-to-back `cudaGraphLaunch` replays; 20 batches per cell; time per replay = batch / R. The order of cells is randomised within each process (so clock or thermal drift does not line up with `N`). Host launch time per replay is also recorded but is not part of `g`.
- **Warm-up** 2 s of load on the same stream before the first cell (link and clocks up), then re-checked per cell (gates 3-5).
- **Three processes**, run one after another with the lock held, so between-process variance is in the interval.
- **Recorded per cell**: variant, `N`, `R`, batch times, link generation start/end, SM MHz min/max, other GPU processes, node count, NUMA nodes, driver and CUDA versions, and the harness hash.
- **Nothing here uses Nsight.** `g` comes from event timing and a slope. If a cross-check with `nsys` is wanted it is a separate, later, short run in `--cuda-graph-trace=node`, used **only** for GPU-side gaps and never for host time or kernel ranking, per the project note.

## 5. What could make the measurement lie, and the control for each

| # | how it lies | control |
|---|---|---|
| L1 | **L2 hides memory cost.** L2 is **96 MiB**, not the ~128 MB in the project `CLAUDE.md` (measured by the `ld.global.nc` result). A benchmark that reuses one tensor measures L2, not HBM; that produced two recorded constants that understate cost by 1.5x and 2.3x. | The primary quantity moves **no data** (empty copies) or 4 KiB, so L2 does not enter `g`. This is deliberate. The `c` microbenchmark (link throughput at `count = 1`) is a different quantity, is **not** in this run, and must size its working set past 96 MiB (at least 1 GiB, spread over 48 row tensors) and check the implied bandwidth against the link spec. |
| L2 | **A graph-mode Nsight trace's kernel table omits the graph body**, and node mode charges ~0.77 us of host time per node. | Event-timed slopes, no Nsight for `g`. |
| L3 | **The PCIe link idles at Gen1**; a poll measured on an idle link is a Gen1 round trip, not the Gen3 one production sees. | Warm-up load; link generation read at the start and end of every cell; gate 3. |
| L4 | **Clocks and thermal drift** line up with `N` if cells run in order. | Randomised cell order; SM MHz recorded; gate 5. |
| L5 | **Optimised away or empty by accident**: an empty kernel elided, a captured node dropped, a load hoisted out. | Node count `= 3N` (gate 1); the positive control (gate 2). |
| L6 | **Not linear**: launch queues, graph-launch modes and kernel-node scheduling change with size, so a slope on 640 nodes is not the slope at production size. | Linearity gate; the 8,000-node base variant (gates 6-7). |
| L7 | **Another process on the GPU.** Production shares this card (25-30 GiB in use, and it serves). Any concurrent kernel time-slices with this run. | crypto-c9 stops or confirms idle; `pmon` around every cell; gate 4. |
| L8 | **Host page on the wrong NUMA node** raises every poll. | The service's allocator, the GPU-local node, both nodes recorded. (No wrong-node control run; that would be a second experiment.) |
| L9 | **Stand-in fidelity.** The real `W_s`/`A_s` will read more or fewer words and add checks. | `p in {1, 4, 6}`; the verdict is registered at `p = 4` and reported as "depends on `W_s`" if `p = 1` or `6` flip it. **The stand-ins bound `g`; only A3 measures it.** |
| L10 | **Ready-at-launch is the best case for polling.** A wait that really waits adds a detection delay `d` after the publish (poll period plus round trip), which this run never sees. | `d` is the same in V1 and V2 for the last-arriving row and is assumed to cancel in `V2 - V1`; that is an assumption, not measured here. A `d` measurement (publish from the host after a known delay) is possible in the same window but is **not registered** and would need its own rule. |
| L11 | **Both `g` and the exposure model are inputs to `G`.** A correct `g` with a wrong exposure model gives a wrong `G`. | Both accountings (`G_X`, `G_H`) are always computed; EXPOSURE-DEPENDENT is a verdict, not a failure. |
| L12 | **Event timing and back-to-back replays** include inter-graph host gaps. | They do not depend on `N` and cancel in the slope; the host launch time per replay is recorded separately. |

## 6. Size of the run (the user's standing rule: as short and non-redundant as possible)

- **A synthetic graph swept over `N`, not a decode arm.** No model, no NVMe, no service thread; a decode arm (about 2 minutes of decode plus 10-15 minutes of boot) would also blur
  `g` with everything else in the step.
- **GPU time about 6 minutes of replays** (per process about 2 minutes: 5 variants x 6 `N` x 1,000 replays at 0.3-13 ms each, plus the 8,000-node cells at about 15 ms per replay), **three processes, plus 30 s of warm-up and about 1 minute of context creation each: 12 minutes.** Request **15 minutes**.
- **GPU memory under 1 GiB** (a CUDA context, a 1 MiB mapped page, tensors of a few KiB, a graph of at most 10,000 nodes). Request **2 GiB** of headroom.
- **Compile beforehand, CPU only**, so the window contains no compilation: the harness and the copy kernel's JIT build under `taskset -c 0-63`, no lock needed.
- **Order of steps**: this document accepted; harness written; its hash appended (section 9) and committed; `cc-gpu.lock` taken; crypto-c9 messaged with 15 minutes and 2 GiB; run; analysis by the frozen script; report the verdict word first.

## 7. What it will not settle

- **`c`.** The gross 4.93 is `Sigma(m-1) * c`; `c = 1.055 ms` is from one older node-mode trace and has not been measured for this path (or at `count = 1`). Section 2.3 shows it moves the crossing by more than `g` does.
- **A2.** There is still no early readiness signal, no per-lane publication, and no measurement here creates one.
- **The 1.0 ms launch cost** used in the ceiling figures (it is not `g`) stays assumed.
- **The bar.** The 1.5% resolution comes from `task1e`'s quiet-box standard deviation on a series recorded UNRESOLVED; the real resolution is worse, which favours rejection.
- **Class C's exposure**, and **`d`** (L10): not settled by a synthetic graph.
- **The production `W_s`/`A_s` `g`.** Stand-ins only.

## 8. If the measurement cannot be made cleanly

`g_e` can: its three kernels exist in spirit and the copy kernel is production code. **`g_a` cannot be made clean**, because the kernels that define it are unbuilt; the design bounds it
with `p`. If the owner judges that a bounded stand-in `g_a` is not worth a 15-minute window, the honest alternative is to build the harness as part of Task 6's first step (the kernels
are needed there anyway) and measure `g` on the real `W_s`/`A_s` inside arm A3. Whichever is chosen, **the measured `g` will not by itself overturn or uphold the label unless it lands
outside about 4-10 us**, because of `c` (section 2.3).

## 9. Frozen artefacts

| file | sha256 |
|---|---|
| `g_measurement/g_analysis.py` (the decision rule and gates; `--selftest` reproduces all four verdicts and five INVALID cases on synthetic data) | `660220c2600f66848d0c72b031223c3386b71776e514def851ff2babe05e0081` |
| `g_measurement/g_exposure_counts.py` (the counts in 1.3, from the measured lanes of `task1f`) | `ff08d3d99857b98962b79da7342571ab977cfcac227440caaaf063140cf3d6dd` |
| the harness | *to be appended before the lock is taken* |

Self-test (run 2026-09-21, CPU): CLEARS, REJECTION-STANDS, EXPOSURE-DEPENDENT, UNRESOLVED and the INVALID cases (bad positive control, idle link, foreign GPU user, non-linear, 8,000-node base slope) all produce the
expected verdict on synthetic data with known `g`. **That tests the logic of the rule, not the GPU.**

## Amendment, 2026-09-21 (after the lead's request to confirm or drop the guess in 1.1)

**The guess about "about 5 us" is dropped, not confirmed.** In the first version of `PER_ROW_TRANSFER.md` (`116533cb61`) "about 5 us" appears twice (5.6 and 6.2) with no derivation; `git log -S` finds it introduced in that commit and
carried through `d22af00980`, `8c78c3d645`, `344b15623b` and `64e01cae78`. Its neighbours are the "8-14 us" range and a "kernel-node gap of 2-4 us [A]"; the 18.2 wait-kernel median of 5 us is not cited anywhere in that document. There is
therefore no textual link between the two 5's, and the guess has nothing under it. **Origin: unknown.** Nothing else in this document changes; this note is appended rather than edited into 1.1 so the registered text stays as frozen.

**Status of the `g` run: cancelled by the lead** (message of 2026-09-21): `g` is to be measured on the real `W_s`/`A_s` inside arm A3. This document is kept as the design A3 will use; `g_analysis.py` and its self-test stand.
