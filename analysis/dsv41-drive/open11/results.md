# OPEN 11: the per-layer cost of arming every `count > 0` record (advise off)

Measured 2026-09-21 on divix01 (RTX 5090), under `gpu-run.sh` (holds `cc-gpu.lock`), `taskset -c 0-63`,
`OMP_NUM_THREADS=1`, production not running. Script: `open11_arming_cost.py`, run three times.
Code at `36c4a78906`. Harness: `LAYERS=2, EXPERTS=16, CAPACITY=8, TOP_K=6`, three all-resident lanes,
300 timed repetitions after 30 warm-up, `rows_read` during timing **0** in every run (they really are hits).

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| A lease OFF, all-hit, **unarmed** (p50 ms) | 0.1855 | 0.1858 | 0.1846 |
| C lease ON, all-hit, **armed** (p50 ms) | 0.1934 | 0.1937 | 0.1930 |
| **delta per layer (p50)** | **+0.0079** | **+0.0079** | **+0.0084** |
| delta per layer (p90) | +0.0056 | +0.0080 | +0.0083 |
| x40 layers (p50 ms/step) | +0.315 | +0.317 | +0.336 |

**Answer: about 8 us per all-hit layer, or ~0.32 ms per step at 40 layers.** The three runs agree to within 6%.

**Phase attribution** (p50 ms, with a `synchronize` between phases, so absolutes are inflated and only the
differences are meaningful; figures identical to 3 decimal places across all three runs):

| phase | lease OFF | lease ON | delta |
|---|---|---|---|
| post | 0.0205 | 0.0215 | +0.001 |
| **wait** | **0.0190** | **0.0300** | **+0.011** |
| copy | 0.1655 | 0.1658 | 0.000 |
| ack | n/a | 0.0142 | +0.014 |

**The sum of the phase deltas (~26 us) is three times the end-to-end delta (~8 us), and that is the useful
finding.** Measured in isolation the acknowledgement kernel costs ~14 us, but in the real stream it does not
serialise: it is launched after the copy and is largely hidden. The copy is unchanged, as it must be. The whole
of the exposed cost is the wait, which grows by ~11 us because an armed all-hit request now blocks on the
service round trip that an unarmed one skipped.

## What the number means, and its limits

* **It is an upper bound on the per-step cost, attained only when every layer is all-hit.** A layer with a miss
  was already armed before lease mode (`need_count > 0`) and pays nothing extra. The x40 figure therefore
  describes the warm-cache steady state, which is the case that matters, rather than an average.
* **Against Task 6's budget it is material: ~0.32 ms is about 28-30% of the 1.114 ms `G*`** that Task 6 is
  trying to win. Section 17.2's requirement that Task 6's benefit be measured net of OPEN 11 now has a number.
* **Against absolute step latency it is small**: ~0.5% of a ~66.8 ms/token decode step.
* **Not production geometry.** Two layers, three lanes, capacity 8. The x40 is a linear extrapolation that
  assumes per-layer costs are additive and do not overlap between layers; in a real decode the wait may overlap
  other work (making it smaller) or contend (making it larger). It is labelled as extrapolation, not measured
  at 40 layers.
* **Not the backend.** `Exl3RamMissRowBackend` has no lease `post` override and there is no environment switch,
  so step 5 of section 20.1 has not landed and a switch-on/switch-off serving comparison is impossible today.
  This is the kernel-plus-real-service level, which is the level section 20.1 item 13 names ("a measurement,
  not a test"). A serving-path number must be taken again when step 5 lands.
* **Box conditions:** load 3-4.5 with QuestDB, nimbus and reth running; the service thread spins
  (`spin_us=5000`), so the round trip is a spin handoff. A different spin setting would move the ~11 us.

---

# Re-taken through the real backend (2026-09-21, step 5 of section 20.1)

Script `open11_serving_path.py`; raw output `serving_path/run{1,2,3}_lease{0,1}.json.gz`. Code at the commit that
carries this file. Same box conditions as above (divix01, RTX 5090, `gpu-run.sh`, `taskset -c 0-63`,
`OMP_NUM_THREADS=1`, production stopped), but **load average 8-11** this time, so the eager view is noisier.
`Exl3RamMissService` and `Exl3RamMissRowBackend` are driven by the real switch `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES`,
one process per switch value, three alternating runs each. Route `[3, 5, 7, 4, 0, 1]`: four RAM lanes plus two
VRAM-hot experts (which are not lanes); all four are resident before timing and `rows_read` during timing is **0** in
every timed run. `leases_granted == leases_acked` at exit, `leases_voided == 0`, `fatal == 0`.

| view | what is timed | delta per layer, p50, runs 1/2/3 |
|---|---|---|
| **eager** | `backend.post`, then a synchronize, wall clock: the method of the 8 us figure above | -2.6 / +10.5 / +9.5 us (noisy) |
| **layer** | one whole `Exl3MoEMethod._apply_graph` per CUDA graph replay | +15.5 / +17.6 / +16.6 us |
| **chain** | 40 `backend.post` in one graph, 1.5 ms GPU spin between them | +17.9 / +16.3 / +17.1 us |
| **packed** | the same 40 back to back, no spin | +17.1 / +17.0 / +16.7 us |

**In a CUDA graph the added cost is about 17 us per all-hit layer, about 0.67-0.69 ms per 40-layer step. That is
roughly twice the ~8 us / ~0.32 ms of the kernel-and-service measurement above, and it is not reconciled here.**
The three graph views agree with each other to within ~1 us and across runs; the eager view through the real backend
lands near the old 8 us but its three runs span -2.6 to +10.5 us, so it neither confirms nor contradicts it.

What this does and does not say:

* **The old figure was an eager figure.** It was taken with a synchronize per step, where the CPU launch cost of
  the kernels sits in front of the GPU work. Serving replays a graph, and in a graph the GPU-side service round trip
  and the acknowledgement kernel run back to back on the GPU with nothing hiding them. That is a hypothesis for the
  gap, **not tested**: nothing here isolates the ack kernel from the wait in graph mode. The gap could also be the
  4-lane versus 3-lane plan, or the real backend's extra work (the `planned` copy), which the eager old harness did
  not do.
* **The 1.5 ms spacing changes nothing** (chain +17 us versus packed +17 us), so the cost is not being paid for a
  ring slot waiting on an unretired acknowledgement: `deferred` and `deferred_reuse` are 0 in the run counters. The
  cost is the per-layer wait itself.
* **Against absolute step latency:** ~0.68 ms is ~1.0% of a ~66.8 ms/token step. **Against Task 6's `G*` of
  1.114 ms it is about 60%**, not the 28-30% the kernel-level number gave. Section 17.2's requirement that Task 6's
  benefit be measured net of this cost now has a number twice as large as the one written there.
* **Still not production geometry.** One layer's row, four lanes, a fake 8-row checkpoint; "40 layers" is 40 posts
  to one row in one graph (chain, packed) or 40 times the single-layer delta. A real 40-layer decode has different
  rows, different lane counts, real MoE compute between layers and `advise` possibly on. The all-hit case is the
  upper bound: a layer with a miss was armed before lease mode.
* **Not measured:** a full serving run (`bench_serving`/the real model), lease mode with `advise` on in a graph,
  and any cost when the RAM tier is under eviction pressure.

---

# Re-taken on an exclusively held card (2026-09-21)

The two sections above were measured while the box was shared: the first at load 3-4.5, the second at load
8-11. That second run's percentiles were not trustworthy, and chasing the discrepancy cost real time -- a p50
delta of +90.5 us against a min-to-min delta of +14.8 us for the same measurement. Percentile numbers need an
exclusively held card, so this re-take waited for one.

Conditions: divix01, RTX 5090, **no other process on the GPU** (63 MiB used, no compute apps), load average
4.2-4.5 and steady across all six runs, `gpu-run.sh` holding `cc-gpu.lock`, `OMP_NUM_THREADS=1`, `PYTHONPATH`
at the tree under test. Script `open11_serving_path.py --reps 300`, three runs per arm, raw output in
`serving_path_exclusive/run{1,2,3}_lease{0,1}.json.gz`. `rows_read` during timing is **0** in every run and
`leases_granted == leases_acked == 36264` in every lease-on run, with no fatal.

| view | lease OFF p50 (ms) | lease ON p50 (ms) | delta per post (us) | min-to-min (us) |
|---|---|---|---|---|
| layer (one in-graph MoE layer) | 0.3501 | 0.3672 | **+17.10** | +17.19 |
| chain (40 posts, 1.5 ms spacing) | 68.28 | 68.96 | **+17.09** | +16.98 |
| packed (40 posts, no spacing) | 8.466 | 9.144 | **+16.79** | +16.84 |
| eager (one post, synchronize) | 0.2514 | 0.2612 | +9.74 | +10.15 |

**The three graph views agree at 16.8-17.1 us per armed layer, and p50 now agrees with min to within 0.4 us.**
Run-to-run spread is 0.0002 ms. That agreement is the point of the re-take: under contention the p50 and the
min disagreed by a factor of six, and neither could be quoted.

## Two corrections to what is recorded above

* **The cost is ~17 us per armed layer, not ~8 us.** The 8 us figure came from `open11_arming_cost.py`, which
  drives a hand-written step. Through `Exl3RamMissRowBackend` and the real switch it is about twice that. The
  backend path, not the measurement noise, is the difference.
* **The x40 figure is now measured rather than extrapolated, and it is ~0.68 ms/step, not ~0.32 ms.** `chain`
  and `packed` each run 40 posts in one graph: +0.684 ms and +0.672 ms respectively. The earlier 0.32 ms was
  8 us linearly extrapolated, and the caveat recorded against it ("assumes per-layer costs are additive")
  turns out to have been the smaller error.

**Consequence for Task 6: OPEN 11 is about 61% of the 1.114 ms `G*`, not 28-30%.** Section 17.2's requirement
that Task 6's benefit be reported net of OPEN 11 now bites roughly twice as hard as recorded. Still an upper
bound -- it is attained only when every layer is all-hit, and a layer with a miss was already armed -- and
still ~1% of a ~66.8 ms/token decode step in absolute terms.
