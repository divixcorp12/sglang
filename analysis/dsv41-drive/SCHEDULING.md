# Drive scheduling for batched row reads — 2026-09-20

Plan: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, Task 2, the
part that needs no model run. Script `bench_row_scheduling.py`; raw results
`row-scheduling-run1.json`, `row-scheduling-run2.json`; tests
`test_bench_row_scheduling.py` (22 pass).

CPU/IO only, no GPU. divix01, `CUDA_VISIBLE_DEVICES=`, `taskset -c 0-63`,
`OMP_NUM_THREADS=8`, O_DIRECT, layer 19, page-aligned destinations. Mirrors
`/mnt/nvme0/dsv41_flash` and `/mnt/nvme4/dsv41_flash` (`nvme3n1`). Both were idle
before run 1 (nvme0 0 sectors and nvme4 16 sectors read over 10 s); during both
runs the diskstats deltas matched the harness's own bytes (below).

## What was measured

One fixed replay (`ad92c28366896b0d`, seed 1234): 100 batches, 304 application
rows per rep, batch sizes 1/2/4/6/8/32, every batch a set of distinct experts of
one layer. Every arm replays exactly that sequence, and each batch is **one
concurrent `_submit`** (a single `UringFileReader.read` over all of the batch's
extents), like `Exl3RowReader.read_split`, not a loop of single reads.

| arm | what one row becomes |
|---|---|
| within-row 1:1 | two extents, one per root, `StaticSplitPolicy((1,1))` — production's policy |
| whole-row | one extent on one root; root = least outstanding bytes in the batch, ties by bytes served so far |
| one-root nvme0 | one extent on nvme0 (weights 1:0) |
| weighted | two extents, weights from a one-root calibration (0.96:1 run 1, 0.97:1 run 2) |

The harness stops (exit 2) unless every arm requested the same rows, the same
useful bytes (`record.nbytes`) and the same aligned bytes in every batch, and
the per-drive bytes of each batch sum to the requested bytes. Data were
compared across arms for every rep-0 row, and 104 row reads per run matched a
plain `pread` of the source checkpoint. A submit that transfers nothing fails
that check (tested).

Three quantities are kept apart. **Application rows** is what the caller asked
for; **extents** is what was submitted (within-row: 2 per row); **per-drive
bytes** is what each root was asked for in the batch. The most extents in one
batch was 64 (32 rows, within-row) against a ring depth of 128, so all of a
batch's extents were outstanding at once, and a drive's outstanding bytes equal
its per-drive bytes for that batch.

## Byte volume, and the cross-check

Run 1 and run 2 each moved **51.15 GB (47.6 GiB)** from the mirrors: run 1
31.73 GB from nvme0 and 19.42 GB from nvme4 (one-root piles onto nvme0), run 2
31.76 and 19.38 GB. Each also read 0.34 GB from the source for ground truth
(run 2's came from the page cache, since run 1 had just read the same rows;
that only affects the reference read, not the timed ones). `/proc/diskstats`
read deltas agree with the harness's own byte counts to within 0.3 MB on nvme4
and exactly on nvme0, so the drives were not doing other work and the O_DIRECT
reads reached the drives. Per-drive rate in the two-drive arms was about
3.3 GB/s, the same as the solo one-root rate (3.3–3.5 GB/s), so there is no
sign of cache-resident reads.

## Result (read p50 in ms; n = batches over 3 reps)

Both runs, same command. Run 1 / run 2 where they differ.

| rows | n | within-row 1:1 | whole-row | one-root nvme0 | weighted |
|---|---|---|---|---|---|
| 1 | 144 | 2.337 / 2.337 | **4.154 / 4.175** | 4.043 / 4.065 | 2.378 / 2.349 |
| 2 | 72 | 4.354 / 4.355 | 4.310 / 4.298 | 7.866 / 7.891 | 4.428 / 4.423 |
| 4 | 36 | 8.343 / 8.310 | 8.199 / 8.184 | 15.405 / 15.412 | 8.533 / 8.386 |
| 6 | 24 | 12.371 / 12.359 | 12.045 / 11.994 | 22.894 / 22.936 | 12.507 / 12.489 |
| 8 | 18 | 16.152 / 16.200 | 15.856 / 15.925 | 30.346 / 30.339 | 16.329 / 16.321 |
| 32 | 6 | 62.907 / 62.964 | 62.116 / 61.988 | 121.816 / 121.837 | 64.152 / 63.557 |

Whole-row against within-row on the *same batch* (median of the per-batch
ratio, reps 1–2; wins / pairs; ratio below 1 is faster):

| rows | run 1 | run 2 |
|---|---|---|
| 1 | 1.764 (0/96) | 1.777 (0/96) |
| 2 | 0.989 (41/48) | 0.995 (31/48) |
| 4 | 0.983 (19/24) | 0.991 (20/24) |
| 6 | 0.977 (16/16) | 0.978 (16/16) |
| 8 | 0.984 (12/12) | 0.985 (11/12) |
| 32 | 0.990 (4/4) | 0.987 (4/4) |

Weighted against within-row: 1.009–1.022 slower at every size, both runs.

### What the data support

- **Whole-row is 1.8x slower at one row per batch.** One extent on one drive is
  one-root: 4.15 ms against 4.04 ms for one-root, so the stateful alternation
  between batches spreads *load* across drives but not a single row's latency.
  How often a production batch is a single row is not in this data.
- **At 4–32 rows whole-row is 1–2.3% faster, and this reproduces**: 0.3 ms of
  16 ms at 8 rows, 0.8–1.0 ms of 63 ms at 32. The direction held in both
  invocations, in 16/16 paired batches at 6 rows and 11–12 of 12 at 8. Two rows
  is a tie.
  These are small effects on the same rows repeated across reps, so treat the
  sign-test p-values as an upper bound on the evidence.
- **The mechanism is not established.** It is not per-drive imbalance: both arms
  request identical bytes per drive (53/53 MB at 8 rows). The arms differ in
  extents per drive (4 against 8 at 8 rows) and in extent size, and both
  saturate at about 3.3 GB/s per drive, which is consistent with a small
  per-extent cost. That is a hypothesis; no counter here tests it.
- **The measured weighted split was slower than 1:1 by ~1–2%** in every size and
  both runs, although the calibration said nvme0 was 3–4% slower solo (3329 vs
  3456 MB/s, then 3293 vs 3412) and so gave it fewer bytes. Solo one-root rates
  did not predict the concurrent optimum. The arm is close to 1:1 by
  construction, so it says little about weighting in general. Unexplained.
- **One-root is 1.7–1.9x slower than either two-drive arm at every size**, which
  is the expected drive-bound result and a sanity check of the harness.
- **Rep 0 was slower for whole-row and within-row at one row** (e.g. 4.52 vs
  4.05/4.11 ms, run 1; 4.64 vs 4.18/4.06, run 2), but not for one-root, which
  never reads nvme4. `MIRROR_ROWS.md` recorded an nvme4 first-touch outlier;
  that would fit, and it was not tested. The percentile columns include rep 0.
- **Packing is not a scheduling quantity but is now the same size as the read**:
  the scatter is 0.6–0.8 ms per row in a batch of 1 and 1.6–1.7 ms per row at
  8+ rows (13.9 ms packing against 16.2 ms reading at 8 rows). It is identical
  across arms, so it does not affect the comparison, but it belongs to Task 4.

### What the data do not show

- **Nothing about the 1.54x eager byte increase.** Every arm requests the same
  bytes by construction, so scheduling cannot be shown to explain, or to fail to
  explain, any extra bytes. That needs serving arms with cache counters and has
  not been run. The anomaly stays unresolved.
- **Tails.** p95/p99 are printed with n, but below 100 batches (every size
  but one row here; n = 144) p99 is the sample maximum, and p95 is near it.
- **Per-drive service time.** The reader returns only the total byte count, so
  which drive finished last in a batch is not observed. Only per-drive bytes and
  the batch's wall time are.
- **Asymmetric load** (Task 2's scratch-space experiment) has not been run.
- **32 rows is not a production batch**: the bounce ring is `BOUNCE_ROWS = 8`
  and production runs a larger eager read as serial batches of 8. The 32-row
  point is what a bigger ring would give.

## Decision

Retain static 1:1 (`StaticSplitPolicy((1,1))`). Whole-row loses by 1.8x at one
row and gains 1–2% at 4 or more rows; the data give no basis to change the
default. If the batch-size mix in production turns out to be dominated by 4+
row eager batches, a hybrid (whole-row only above a row threshold) is the one
variant this data could support, and it would need an end-to-end arm.

## Interface a whole-row policy would need

`SplitPolicy.plan(length)` is length-only and, in `Exl3RowReader.read_split`,
called once per row with nothing else in scope. Whole-row assignment cannot be
expressed there: which root a row takes depends on the other rows in the batch
and on what each root has already been asked for, and the alternation at one row
per batch needs state that survives between calls. Putting that in the policy
object would hide a mutable round-robin inside something callers treat as a pure
function of length. Keep it separate:

```python
class BatchScheduler:
    # rows: identity of every row in this batch, in request order
    # outstanding: bytes already in flight per root (0s for a synchronous batch)
    # returns one Extent list per row: (root, file offset, dest offset, length)
    def assign(self, rows: Sequence[RowKey], lengths: Sequence[int],
               outstanding: Sequence[int]) -> list[list[Extent]]: ...
    def complete(self, rows: Sequence[RowKey]) -> None: ...  # retire outstanding work
```

`RowKey` is the plan's `RowKey`, so an asynchronous service can pass real
in-flight bytes and retire them per row. The harness's `WholeRowPlanner`
implements this for the synchronous case (`assign` plus a `served` history
standing in for `outstanding`). A `SplitPolicy` remains right for the
within-row policies, which really are a function of length.

## Reproduce

```
python bench_row_scheduling.py --dry-run     # plan, per-drive GiB, total I/O; reads nothing
python bench_row_scheduling.py               # ~47.6 GiB from the mirrors, 0.2 from the source
```

`--max-gib` (default 80) refuses to start a run whose planned reads exceed it.
