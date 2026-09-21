# Drive scheduling for batched row reads — 2026-09-20, conditions added 2026-09-21

Plan: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, Task 2. Script
`bench_row_scheduling.py` with `drive_conditions.py`; raw results
`row-scheduling-run1.json`, `row-scheduling-run2.json`; tests
`test_bench_row_scheduling.py` and `test_drive_conditions.py`.

**Result: keep static 1:1.** No policy tested here beats it by enough to change the
default. This document does not resolve the 1.54x eager byte increase; it cannot
(next section).

## What this analysis is not

Every arm requests the same rows and the same bytes by construction, and the harness
stops (exit 2) if it does not. Scheduling therefore cannot add or remove a byte, so a
scheduling replay says nothing about the 1.54x eager traffic increase, which is
Task 2's first item and needs cache counters (hits, misses, admissions, evictions,
promotions, advisory bytes) from the eager serving path. Building this replay must
not be recorded as having investigated that anomaly. It stays unresolved.

## Limits of the existing evidence

`row-scheduling-run1.json` and `-run2.json` were taken without three things that a
matched comparison needs: **per-arm page-cache residency of the mirror shards**
(`fincore` before and after each arm), **per-drive block-request counts**, and the
**load average and foreign CPU use**. The arms were interleaved in one process, so
the whole-run `/proc/diskstats` deltas that were recorded cannot be split per arm
either. The runs therefore cannot be cited as matched evidence for, or against, any
effect of request count, cache state or load. What they do support is exactly the
property the verdict rests on: identical requested bytes in every batch for every arm
(enforced, not assumed), and the timing of each arm under whatever conditions held.
The missing conditions are a limitation of the evidence, not a reason to discard the
verdict. One caution: the nvme4 mirror was reported on 2026-09-21 to hold about 15 GB
of page-cache residency from earlier writes, varying over time. O_DIRECT was reported
to bypass the cache on both filesystems (`dd iflag=direct` against a buffered control),
so residency should not affect these reads; it was not sampled during either run, so
that is not shown here.

The harness now records all three per (rep, size, arm) block (below), so a re-run, if
one is ever wanted, is conditioned. None is planned.

## The two mirrors are not symmetric hardware

**Device resolution trap:** the `/proc/diskstats` row is the *partition*
(`nvme0n1p1`, `nvme3n1p1`) while the queue limits live on the *parent disk*
(`nvme0n1`, `nvme3n1`); a script keyed on a device name reads the wrong thing or
nothing. The harness resolves both.

Resolved at run time from `/proc/self/mountinfo`, `/proc/diskstats` and
`/sys/dev/block/<major>:<minor>` by `os.stat().st_dev`, never by name (there is no
`nvme4` block device; `/mnt/nvme4` is `nvme3n1`, and diskstats keys the partition):

| mirror | device | model | fs | `max_sectors_kb` | `max_segments` |
|---|---|---|---|---|---|
| `/mnt/nvme0/dsv41_flash` | `nvme0n1p1` | Samsung SSD 990 EVO Plus 2TB | xfs | 512 | 128 |
| `/mnt/nvme4/dsv41_flash` | `nvme3n1p1` | SPCC M.2 PCIe SSD | ext4 | 256 | 65 |

The same 6.66 MB extent (one row on one drive) becomes about 13 block requests on
nvme0 and 26 on nvme3. A 50/50 byte split is a 1:2 request split.

### Request-count model applied to run 1 and run 2 (no drive I/O)

**Status: an unvalidated prediction, not a measurement.** Its three limits:
(1) it has not been checked against real block counts, because the existing runs
recorded no `reads_completed`; (2) it counts only the `max_sectors_kb` size cap and
ignores segment merging and how xfs and ext4 build bios, either of which can move the
real count; (3) run 1 and run 2 lack per-arm conditions (previous section) and stand
only for the identical-requested-bytes verdict. A small validation run (below, under
Reproduce; about 0.12 GB) is approved for a time when the drives are free and is not
required to close the scheduling items.

**Finding 1, measured: request count does not bind at 3.3-3.5 GB/s.** The drive with
twice the requests per byte was the faster one solo. The calibration read nvme0 (13
requests per extent) at 3329 MB/s in run 1 and 3293 in run 2, and nvme4 (26 per extent)
at 3456 and 3412, so nvme4 was 3-4% faster; per-row solo p50 in `MIRROR_ROWS.md` was
5.916 ms (nvme0) against 6.007 ms (nvme4), a parity. This is a measured fact and does
not depend on the model, though the 13 and 26 that label it do.

**Finding 2, modelled: the request asymmetry is real but arm-invariant.** A
byte-balanced split is a 1:2 request split, and both arms whose verdict matters carry
the same one (busiest over quietest 2.00 for within-row, 1.96 for whole-row). It is a
property of the drive pair at a given byte split, not of the scheduling policy, so it
cannot explain whole-row's 1-2% gain at 4 or more rows or its 1.8x loss at one row
(one extent on one drive, a byte effect). A real effect that cannot tell the
alternatives apart is a negative result for the decision here: "1:2 request imbalance"
should not be read as a reason to change the policy.

The one arm this cannot rule out is a request-balanced split (about 2:1 bytes toward
nvme0). Finding 1 gives no reason to expect it to win, since nvme3 is not the slower
drive.

**How the model was built and why it is credible.** `bench_row_scheduling.py
--model-requests --weights 0.96:1` (and `0.97:1` for run 2) re-plans the recorded
replay (digest `ad92c28366896b0d`, 100 batches, 304 rows per rep) through the same
planners, resetting stateful planners per batch-size group exactly as the run did, and
predicts requests as `ceil(extent / max_sectors_kb)` per extent. It read only the
checkpoint headers. Its per-arm per-drive bytes reproduce the recorded ones (1.89/1.89
GiB within-row and whole-row, 3.77/0 one-root, 1.85/1.92 and 1.86/1.91 weighted), so
the plan is the one that ran. Predicted requests per batch, nvme0 / nvme3:

| rows | within-row 1:1 | whole-row | weighted 0.96:1 |
|---|---|---|---|
| 1 | 13.0 / 26.0 | 13.0 / 25.5 | 13.0 / 26.0 |
| 2 | 26 / 52 | 26 / 51 | 26 / 52 |
| 4 | 52 / 104 | 52 / 102 | 52 / 104 |
| 6 | 78 / 156 | 78 / 153 | 78 / 156 |
| 8 | 104 / 208 | 104 / 204 | 104 / 208 |
| 32 | 416 / 832 | 416 / 816 | 416 / 832 |

Request share nvme0 / nvme3 is 0.33 / 0.67 for within-row and weighted and 0.34 / 0.66
for whole-row. One-root nvme0 puts all requests (26 per row) on nvme0. The harness
prints predicted and measured requests side by side in a conditioned run and marks a
block `?` when they differ by more than 25% (arbitrary threshold).

## What was measured (runs 1 and 2)

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

## What a conditioned run records

Each (rep, size, arm) block is bracketed by `drive_conditions.ConditionProbe`, and
every summary row carries these, per drive unless noted:

- device name, model, filesystem, `max_sectors_kb`, `max_segments`, `chunk_sectors`
  (`report["drives"]`); a root with no `/proc/diskstats` row stops the run before any
  read instead of reading zero;
- block requests: model, and measured `reads_completed` and `reads_merged` over the
  block; mean in-flight requests while the drive was busy (`weighted_io_ms / io_ms`);
- page-cache residency of the shards under the mirror root before and after the block
  (`fincore`, outside the timed interval); a block whose residency moved more than
  64 MiB (arbitrary) is marked `!`, and an unmeasured residency counts as moved, not
  as cold;
- 1-minute load average before and after, cores' worth of CPU used by other
  processes on cpus 0-63 and on all cpus (from `/proc/stat` minus this process), and
  the cpus this process was allowed;
- application rows, extents and per-drive bytes stay separate; plus p50/p90/p95/p99,
  sample counts (`n`), read and pack time as before.

The reader does not expose per-extent completion times, so per-drive service time is
still not measured: queue depth and request counts are the block-level view, batch
wall time is the only latency.

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

`outstanding` here is this reader's own in-flight bytes per root. It does not include
another process's reads, so it cannot steer around a drive that someone else is
loading (see the asymmetric-load design below). And because the two mirrors differ in
`max_sectors_kb`, equal outstanding bytes are unequal outstanding block requests; a
policy that wanted request balance would have to be given the per-root request cap
as well. The data above give no reason to want that.

## Asymmetric-load experiment (design only; not built, not run)

Question: does the ranking of within-row 1:1, whole-row and the measured weighted
split change when one mirror is busy with someone else's reads? The interface below
counts only this reader's own outstanding bytes, so a least-outstanding rule cannot
see foreign load; if a load-adaptive policy is ever wanted it would need completion
timing fed back, which the reader does not expose today. That is a hypothesis to test,
not a finding.

- **Reserved scratch space is required.** The interferer must not read the mirror
  shards (that would move their page cache and contend with production). It needs its
  own scratch file per mirror device, created once on the same filesystem as the
  mirror root, of a size well past the drive's cache (arbitrary: 16 GiB each), and
  space reserved for it by whoever owns the machine. Do not run it on `/mnt/nvme1` or
  `/mnt/nvme2`.
- **Interferer.** An O_DIRECT reader of the scratch file at a fixed rate (for
  example `fio --rate`), at two levels (25% and 50% of a solo drive's rate; arbitrary),
  on one mirror at a time. Its own achieved bytes per block are logged, and
  subtracted from the device's diskstats delta to keep the harness's own bytes
  attributable.
- **Arms and rotation.** within-row 1:1, whole-row and weighted, crossed with load on
  nvme0, load on nvme3, and none, in a Latin-square order across at least three
  replicates so no arm always meets a warming or cooling drive. The weighted arm is
  calibrated once under quiet conditions and not re-tuned under load; a second variant
  re-calibrated under load is a separate arm.
- **Batch sizes 4 and 8 only** (where whole-row was measured to differ at all), 48 rows
  per size, 3 reps: about 35 GB of mirror reads across the 54 blocks, plus the
  interferer's own bytes.
- **Separate results.** Written to their own file with the load placement and level in
  the header, never merged into a quiet-drive table, and never labelled quiet unless
  the foreign reads on both devices were near zero.
- **Decision rule.** Change the default only if a policy beats 1:1 by a margin that is
  reproduced across replicates on paired per-batch ratios in the loaded cases and does
  not lose in the quiet case. Otherwise keep static 1:1.

Needs before running: the reserved scratch space, a window with no production
traffic on the drives under test, and a small interferer launcher plus a load tag in
the harness header; neither exists yet.

## Reproduce

Everything runs on divix01 under `taskset -c 0-63` with `CUDA_VISIBLE_DEVICES=`,
`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`. Do not start a run while other drive work
is going on; the harness reports load and foreign CPU but cannot make a contended
machine quiet.

```
# no drive I/O beyond checkpoint headers: prints the request model per arm and size
python bench_row_scheduling.py --model-requests --weights 0.96:1 --output model.json

python bench_row_scheduling.py --dry-run     # plan, per-drive GiB, total I/O; reads nothing

# a conditioned run: about 47.6 GiB from the mirrors, 0.2 from the source
python bench_row_scheduling.py --output row-scheduling-runN.json

# smaller check that the request model matches the block layer (about 0.12 GB in total,
# one extent per drive per row, so 13 / 26 requests per row are expected)
python bench_row_scheduling.py --arms within_row --sizes 1 --rows-per-size 8 \
    --reps 1 --warmup-batches 1 --verify-rows 0 --weights 1:1 --output check.json
```

`--max-gib` (default 80) refuses to start a run whose planned reads exceed it. Tests
need a real filesystem for O_DIRECT, so pass `--basetemp` on a disk:

```
python -m pytest analysis/dsv41-drive/test_bench_row_scheduling.py \
    analysis/dsv41-drive/test_drive_conditions.py -p no:cacheprovider --basetemp <disk dir>
```
