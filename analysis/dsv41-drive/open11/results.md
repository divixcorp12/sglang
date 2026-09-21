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
