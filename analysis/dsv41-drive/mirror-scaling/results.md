# Do expert row reads scale from 1 to 2 to 3 NVMe mirror roots?

2026-09-28, divix01. Branch `cc/mirror-scaling-bench` (from `origin/cc/iopoll-diag`). Disk only: no GPU, no server,
no root. Every run held `rowimg-disk.lock` and ran under `taskset -c 18-35` (NUMA node 1, the drives' node).

## Verdict

**Reads scale as expected once each drive's fixed per-read latency is counted.** A per-drive model
`part_time = a + b * part_bytes`, fitted at QD1, predicts every root set's row p50 within 5%. Bandwidth scales almost
linearly: 3.53 GB/s on 1 drive, 6.84–7.03 on 2, and 10.1 on 3 (2.86×).

- **Why the naive estimate was off.** It said 1.95 ms on 2 roots and 1.3 ms on 3, but assumed zero fixed cost per
  read. The fixed cost is about 0.25 ms on the Samsungs and 0.47 ms on the SPCC. So 3 roots cannot go below about
  1.5 ms however the bytes are split.
- **The change that went to production gained what the drives allow.** The old pair {nvme0, SPCC} took 2.39 ms at
  QD1. All three roots at equal weights take 1.73 ms: 0.66 ms (−28%) less per row. The naive estimate said −33%.
- **The SPCC is the straggler.** At equal weights it finishes last in 88–90% of three-root rows. It reads as fast
  as a Samsung (both 3.65 GB/s at the margin) but starts about 0.23 ms later.
- **Down-weighting the SPCC to about 0.9 is the one cheap disk-side gain.** QD1 row p50 goes 1.69 → 1.53 ms (−9.5%)
  with p99 unchanged, and QD2 goes 2.61 → 2.64 ms (neutral). The old proposal of 0.5 is too much: the Samsungs then
  become the stragglers and the row is no faster (1.74 ms).
- **What this says about decode.** Decode fell 8.2 ms/token (109.9 → 101.7) for 0.66 ms less per row. So about 12
  row-equivalents per token of serial row-read time sit on decode's critical path. That is roughly 30 ms/token before
  the change and 21 ms after, about a quarter of the token. Reads scaled as expected, so the small decode gain means
  **misses are not the dominant decode cost**. Most of the ~45 rows read per token overlap compute or each other.
- **The SPCC also has slow episodes.** For tens of seconds at a time, it serves reads at 10–47 ms SQE p99 while the
  Samsungs sit idle. That is a tail risk for decode that no weight fixes (see "SPCC slow episodes").

## Method

`mirror_bench.c` issues the production reader's direct-mode row reads. Production runs with piece streaming on,
`URING_MODE=default`, `WAIT_MODE=block` and `READ_CUTS=0`, read from the live server's environment. The bench's
row reads follow these production rules:

- **Split across roots.** It splits the row's page-rounded `row_stride` across the roots by weight, exactly as
  `exl3_read_split.ReadSplit` does: pages floored per weight, and the leftover pages to the first largest weight.
  Each part is clipped at `image_bytes`, as `exl3_ram_miss.py` does for the row-image tables.
- **Sub-reads per part.** It cuts each reading part into `sub_reads_per_part(reading) = min(4, 8/reading)` sub-reads,
  as `piece_geometry.h::split_part` does. That gives 4 sub-reads on 1 or 2 roots and 2 on 3 roots, so 4, 8 and 6
  SQEs per row.
- **SQE shape.** Each sub-read is one O_DIRECT `READV` whose iovecs are the six per-name slab rows.

These choices were fixed across every run:

- **Wait mode.** `MODE=default` and `WAIT=block` (`io_uring_submit_and_wait(1)`), held constant. That is
  production's setting. `diagnosis.md` showed default + spin is only ~57 µs faster, so the wait mode cannot change a
  scaling conclusion.
- **Rows.** Rows are uniform over (layer 0–39, expert 0–383), with a fixed seed per (rep, QD), so every root set reads
  the same rows. O_DIRECT, no page cache.
- **Matrix.** 8 root sets × QD {1, 2, 4} rows in flight × 3 reps.
  - Rows per cell: 2000 at QD1, 3000 at QD2 and 4000 at QD4.
  - Root sets are interleaved inside each rep, so drift hits every set alike.
  - Reps agree to ±0.03 ms at QD1.
- **Weight sweep.** 1:w:1 with w in {1, 0.9, 0.8, 0.7, 0.6}, QD {1, 2}, 3 reps, 3000 rows per cell.
- **Clean runs.** All 102 cells: rc 0, 0 errors, 0 short reads, and 0 MB of foreign I/O on any drive
  (diskstats bytes = bench bytes).

```bash
# divix01, in /data/models/slang/nvfp4-work/wt-mirror-bench
gcc -O2 -pthread -o /mnt/nvme1/mirror-scaling/mb analysis/dsv41-drive/mirror-scaling/mirror_bench.c -luring
bash analysis/dsv41-drive/mirror-scaling/run_matrix.sh ./mb matrix.jsonl raw.csv 3
bash analysis/dsv41-drive/mirror-scaling/run_weights.sh ./mb weights.jsonl weights-raw.csv 3
python3 analysis/dsv41-drive/mirror-scaling/analyze.py results/matrix.jsonl results/matrix-raw.csv.gz
```

The mirror_bench.c at `a7f0b89029` produced these numbers. The later io_uring-options version keeps the same defaults
and the same row sequence. Raw data is in `results/`:

- `matrix.jsonl` and `weights.jsonl`: one line per cell;
- `*-raw.csv.gz`: one line per row, with label, row, layer, expert, row µs and each root's part-completion µs;
- `run.log`.

Drive names: nvme0 = nvme0n1, Samsung 990 EVO Plus (/mnt/nvme0). SPCC = nvme2n1, DRAM-less, 64 MiB HMB (/mnt/nvme4).
nvme2 = nvme3n1, Samsung 990 EVO Plus (/mnt/nvme2).

## Scaling table (median of 3 reps)

| root set | QD | row p50 ms | p90 | p99 | max | GB/s | per drive: GB/s, util, part p50, last-finisher share |
|---|---|---|---|---|---|---|---|
| nvme0 | 1 | 3.92 | 4.06 | 4.70 | 5.4 | 3.36 | 3.36, 0.97 |
| SPCC | 1 | 4.13 | 4.41 | 4.89 | 6.8 | 3.18 | 3.18, 0.97 |
| nvme2 | 1 | 3.89 | 3.94 | 4.25 | 5.0 | 3.40 | 3.40, 0.97 |
| nvme0+SPCC (old prod) | 1 | 2.39 | 2.70 | 3.03 | 5.0 | 5.45 | nvme0 2.73, 0.83, 2.04 ms, 3% · SPCC 2.72, 0.95, 2.38 ms, **97%** |
| nvme0+nvme2 | 1 | 2.14 | 2.22 | 2.60 | 3.3 | 6.21 | nvme0 3.10, 0.93, 2.03 ms, 8% · nvme2 3.10, 0.95, 2.13 ms, 92% |
| SPCC+nvme2 | 1 | 2.19 | 2.33 | 2.71 | 4.5 | 5.96 | SPCC 2.98, 0.96, 2.18 ms, 77% · nvme2 2.98, 0.92, 2.13 ms, 23% |
| all 3, 1:1:1 (prod) | 1 | **1.73** | 2.00 | 2.32 | 3.3 | 7.45 | nvme0 2.49, 0.79, 1.42 ms, 5% · SPCC 2.48, 0.94, 1.71 ms, **90%** · nvme2 2.48, 0.81, 1.52 ms, 5% |
| all 3, 1:0.5:1 | 1 | 1.74 | 1.92 | 2.54 | 3.1 | 7.43 | nvme0 2.97, 0.93, 1.66 ms, 22% · SPCC 1.49, 0.66, 1.20 ms, 1% · nvme2 2.97, 0.93, 1.72 ms, 77% |
| nvme0 | 2 | 7.51 | 7.73 | 8.37 | 9.1 | 3.53 | |
| SPCC | 2 | 7.61 | 7.99 | 8.85 | 12.3 | 3.48 | |
| nvme2 | 2 | 7.51 | 7.65 | 8.08 | 8.7 | 3.54 | |
| nvme0+SPCC | 2 | 3.87 | 4.25 | 4.93 | 6.4 | 6.85 | SPCC last 99% |
| nvme0+nvme2 | 2 | 3.77 | 3.86 | 4.23 | 5.3 | 7.03 | |
| SPCC+nvme2 | 2 | 3.87 | 4.14 | 4.94 | 6.7 | 6.84 | SPCC last 99% |
| all 3, 1:1:1 | 2 | 2.61 | 3.01 | 3.60 | 4.5 | 10.07 | SPCC last 77% |
| all 3, 1:0.5:1 | 2 | 3.05 | 3.37 | 3.95 | 4.9 | 8.55 | nvme0 last 100% |
| nvme0 | 4 | 15.04 | 15.37 | 16.08 | 16.8 | 3.53 | |
| SPCC | 4 | 15.22 | 15.68 | 19.45 | 24.9 | 3.49 | |
| nvme2 | 4 | 15.02 | 15.18 | 15.70 | 16.8 | 3.54 | |
| nvme0+SPCC | 4 | 7.75 | 8.20 | 9.19 | 12.3 | 6.84 | SPCC last 99% |
| nvme0+nvme2 | 4 | 7.55 | 7.67 | 8.07 | 8.9 | 7.03 | |
| SPCC+nvme2 | 4 | 7.74 | 8.09 | 9.42 | 12.7 | 6.86 | SPCC last 100% |
| all 3, 1:1:1 | 4 | 5.25 | 5.66 | 6.21 | 7.5 | 10.11 | SPCC last 99% |
| all 3, 1:0.5:1 | 4 | 6.13 | 6.60 | 7.23 | 8.3 | 8.56 | nvme0 last 100% |

- **QD1 against 1/N.** QD1 row p50 goes 3.9 → 2.14–2.39 → 1.73 ms. Pure 1/N scaling would give 1.95 and 1.30 ms.
  The gap is the fixed cost (next section), not a failure to scale.
- **QD ≥ 2 is bandwidth-bound and scales almost perfectly.** 3.53 → 6.84–7.03 → 10.1 GB/s. The per-drive rate is
  3.35–3.53 GB/s in every set.
- **The SPCC gates bandwidth at equal weights.** With equal shares, every drive moves the same bytes per row, so the
  set runs at the slowest drive's pace. That costs 3.37 vs 3.53 GB/s per drive (−5%) against a Samsung-only triple.

## Fit: `part_time = a + b · part_bytes` at QD1

Fitted by least squares over every QD1 cell's per-drive part-completion p50. Part sizes range from 2.66 MB to 13.3 MB.

| drive type | a (fixed) | b | marginal bandwidth | R² | n |
|---|---|---|---|---|---|
| Samsung 990 EVO Plus | **245 µs** | 0.275 µs/KB | 3.64 GB/s | 0.998 | 30 |
| SPCC | **471 µs** | 0.274 µs/KB | 3.65 GB/s | 0.995 | 15 |

| root set | predicted row = max over drives (ms) | measured p50 | measured / predicted |
|---|---|---|---|
| nvme0 | 3.91 | 3.92 | 1.00 |
| SPCC | 4.12 | 4.13 | 1.00 |
| nvme2 | 3.91 | 3.89 | 1.00 |
| nvme0+SPCC | 2.30 | 2.39 | 1.04 |
| nvme0+nvme2 | 2.08 | 2.14 | 1.03 |
| SPCC+nvme2 | 2.30 | 2.19 | 0.95 |
| all 3, 1:1:1 | 1.69 | 1.73 | 1.02 |
| all 3, 1:0.5:1 | 1.71 | 1.74 | 1.02 |

- **The limits of an even split.** Even with zero skew, a 3-way split cannot beat 245 + 4.44 MB / 3.64 GB/s =
  1.47 ms on three Samsungs. With the SPCC it cannot beat 1.69 ms at equal weights.
- **The ideal SPCC weight.** The two lines cross where the SPCC carries 0.82× a Samsung's share, which predicts
  1.51 ms. The sweep confirms it: w 0.8–0.9 measures 1.53–1.55 ms.
- **The SPCC's +226 µs is a fixed cost, not bandwidth.** `diagnosis.md`'s flat reads show the same pattern: 4 KiB is
  131 vs 50 µs, and 256 KiB is 588 vs 144 µs. Its 256 KiB `max_sectors_kb` also doubles the requests per SQE: 7–14
  against 4–8 on the Samsungs. Neither changes its streaming rate. The likely causes are a slower first-byte time
  (DRAM-less: a map lookup through the 64 MiB HMB on random rows across 191 GB) and per-request overhead. Separating
  them would need a 4 KiB random-read test, which is outside this bench.

## Straggler analysis

At QD1, pooled over the 3 reps (6000 rows per set), from `results/matrix-raw.csv.gz`:

| root set | row p50 / p99 ms | part p50 / p99 by root (ms) | row p50 / p99 if the last part had finished with the second-to-last |
|---|---|---|---|
| nvme0+SPCC | 2.39 / 3.02 | nvme0 2.04/2.60 · SPCC 2.38/3.01 | 2.04 / 2.42 |
| nvme0+nvme2 | 2.14 / 2.61 | 2.03/2.41 · 2.13/2.44 | 2.03 / 2.16 |
| SPCC+nvme2 | 2.19 / 2.82 | SPCC 2.18/2.70 · nvme2 2.13/2.46 | 2.12 / 2.26 |
| 1:1:1 | 1.73 / 2.39 | nvme0 1.42/2.11 · SPCC 1.71/2.30 · nvme2 1.52/2.04 | 1.53 / 1.86 |
| 1:0.5:1 | 1.74 / 2.54 | nvme0 1.66/2.50 · SPCC 1.20/1.70 · nvme2 1.72/2.08 | 1.66 / 1.87 |

- **How often the SPCC is last.** At equal weights it finishes last in 90% of rows at QD1 and 77–99% at QD2/4. In the
  old pair it was last in 97%.
- **What rebalancing it is worth.** If the SPCC had kept pace, the 1:1:1 row p50 would have been 1.53 ms instead of
  1.73 (−0.20 ms), and p99 1.86 instead of 2.39. The weight sweep recovers almost all of this without new hardware:

| SPCC weight (1:w:1) | QD1 p50 / p90 / p99 ms | QD1 GB/s | last finisher at QD1 (nvme0 / SPCC / nvme2) | QD2 p50 / p99 ms | QD2 GB/s |
|---|---|---|---|---|---|
| 1.0 (prod) | 1.69 / 1.96 / 2.26 | 7.65 | 7% / **88%** / 6% | **2.61** / 3.49 | **10.09** |
| **0.9** | **1.53** / 1.77 / 2.25 | **8.36** | 20% / 51% / 30% | 2.64 / 3.50 | 9.88 |
| 0.8 | 1.55 / 1.74 / 2.25 | 8.30 | 22% / 13% / 66% | 2.73 / 3.65 | 9.57 |
| 0.7 | 1.58 / 1.77 / 2.32 | 8.17 | 32% / 3% / 65% | 2.82 / 3.63 | 9.26 |
| 0.6 | 1.64 / 1.81 / 2.38 | 7.90 | 31% / 1% / 68% | 2.92 / 3.75 | 8.95 |

- **w = 0.9 is balanced.** No drive is last most of the time. It gives −0.16 ms (−9.5%) at QD1 and costs 2% of QD2
  bandwidth (+0.03 ms at QD2 p50).
- **Lower weights lose bandwidth at QD ≥ 2.** They trade it 1:1 for nothing: the Samsungs become the stragglers.
- **Why this sweep's 1:1:1 is 1.69 ms, not 1.73.** It is a different seed and set of rows. Every weight inside the
  sweep used the same rows.

## Other possible limits, checked

- **Submit/reap thread.** The whole process uses 0.03–0.11 CPU-s per wall-second, i.e. ≤ 11% of one core, even at
  10 GB/s. It is not a limit.
- **Request splitting at `max_sectors_kb`.** Every SQE splits: 4–8 block requests per SQE on the Samsungs (512 KiB)
  and 7–14 on the SPCC (256 KiB). Splitting does not cost bandwidth, because both drive types stream at the same
  3.65 GB/s. `diagnosis.md` measured cutting reads to device size in default mode at −40 µs per row: a small
  contribution to fixed cost, not a scaling limit. The follow-on `uring-sweep.md` measures it on this bench.
- **Per-SQE fixed cost.** This is the only real deviation from 1/N (the `a` term above). A row's parts run in
  parallel, so the fixed cost is paid once per row per drive, not per root added.
- **IRQ cores 64–71.** In this bench the completion interrupts did not land there. `irqs_cpu64plus` is 0 on the
  Samsungs in every cell, and 0–17k of 20–220k on the SPCC. NVMe completion vectors follow the submitting CPU's
  hardware queue (managed affinity), and the bench submits from 18–35. The interrupt rate is about one per 3 block
  requests. Nothing points at IRQs as a limit.

## SPCC slow episodes (tail risk, not scaling)

Twice during these runs, the SPCC entered the slow state that `diagnosis.md` recorded:

- **Weight sweep, rep 3, QD2.** It lasted about 30 s over three consecutive cells.
  - w0.9: SPCC part p50 11.4 ms, SQE p99 21.8 ms, 2.2 GB/s total, Samsung utilization 0.25.
  - w0.8: SQE p99 20.4 ms.
  - w1: SQE p99 10.6 ms.
  - By w0.7 it had recovered.
- **Matrix, rep 2, QD4.** SQE p99 was 40–47 ms in the two sets that contained the SPCC.

The medians above are over 3 reps and are unaffected. But for the rows that land in such an episode, the row takes as
long as the SPCC does, whatever its weight. The Samsungs never showed this.

## Recommended next steps

1. **Decode pair: `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=1:0.9:1` against the current 1:1:1**, alternating, at least 3
   pairs.
   - Disk-side prediction: −0.16 ms per row at QD1. At the ~12 row-equivalents per token that the 3rd root's gain
     implies, that is about −2 ms/token at best. It could be smaller, if decode's reads are mostly QD ≥ 2, where
     w0.9 is neutral.
   - No code change is needed.
2. **A traced decode pair (graph mode)** to measure the read critical path per token directly. That would confirm or
   correct the "~21 ms/token of serial read time" inference above, which rests on decode being QD1-like.
3. **Hardware.** Replacing the SPCC with a Samsung-class drive would remove both the 226 µs fixed-cost gap (1.73 →
   ~1.47 ms per row at QD1) and the slow episodes. It is worth doing only if step 2 shows reads are a large enough
   share of the token.
