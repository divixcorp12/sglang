# io_uring settings sweep on the 3-root mirror read

2026-09-28, divix01. Branch `cc/mirror-scaling-bench`. This follows `results.md`. It is disk only: no GPU, no server,
no root. Every cell held `rowimg-disk.lock` and ran on cores 19–35 (NUMA node 1). Core 18 is the node-1 SQ-thread
core and core 2 the node-0 one; cores 64–71 are untouched.

## Verdict

**No io_uring setting is worth a decode pair on its own.** All 48 combos land within 100 µs of each other at QD1:
1519–1596 µs at 1:0.9:1, and 1596–1736 µs at 1:1:1. At QD2 and QD4 they are identical, because the drives are
saturated.

- **Production is one of the slowest settings.** Production is `default.block.cuts0.normal`: `MODE=default`,
  `WAIT_MODE=block`, `READ_CUTS=0`, `READ_MODE=normal`. It ranks 47th of 48, at 1736 / 1575 µs (1:1:1 / 1:0.9:1).
- **The best robust combination is `default.spin.cuts1.rvf-arena+ff`.** That is `WAIT_MODE=spin`, `READ_CUTS=1`,
  `READ_MODE=readv_fixed`, `SLAB_ARENA=1` and `FIXED_FILES=1`. It runs at 1650 / 1524 µs: **−86 / −51 µs**, or −68 µs
  on average (−4%). It costs one spinning core while reads are in flight: 0.12 CPU-s per GB, against 0.011.
- **The best combination without a spinning core is `default.block.cuts1.rvf-arena+ff`.** It runs at 1681 / 1552 µs:
  −55 / −23 µs, −38 µs on average. CPU is 0.016 CPU-s/GB.
- **All the gains come from three small effects that add up.** Averaged over the combos that hold each value:
  - some thread spinning on completions (spin wait, IOPOLL reap or an SQPOLL thread): −16 to −22 µs;
  - `readv_fixed` with fixed files: −20 µs;
  - read cuts: −8 µs.
- **Nothing else matters.** SQ-thread placement on node 1 vs node 0 differs by 5 µs. Fixed files alone do nothing.
  A 4096-entry ring changes nothing at QD1 or QD4.
- **Against the SPCC weight, these gains are small.** Moving 1:1:1 → 1:0.9:1 at the production setting is worth
  −161 µs, 2–4× any io_uring gain.
  - The best no-spin setting plus the weight gives 1736 → 1552 µs (−184 µs).
  - At the ~12 row-equivalents per token that `results.md` infers, that is at most about −2.2 ms/token, and nearly
    all of it is the weight.
  - The io_uring part alone would be about −0.5 to −0.8 ms/token, under the 1.5 ms promotion bar. A spinning waiter
    has already measured neutral in decode (S6, `iopoll/diagnosis.md`).
- **Recommendation.** Run the decode pair from `results.md` (`MIRROR_WEIGHTS=1:0.9:1` against 1:1:1). If a third arm
  fits, make it the weight plus `READ_MODE=readv_fixed FIXED_FILES=1 READ_CUTS=1 SLAB_ARENA=1`, with block wait. That
  combination adds no CPU. Mind its costs: registered buffers are charged to memlock, and registering them takes about
  59 s at tier scale (`uring_reader.h`'s own note). Do not spend an arm on SQPOLL or IOPOLL: they cost a full core
  and gain nothing over a spinning waiter.

## Method

`mirror_bench.c` (commit `990bd3b37e` onward) gained the options of master `ba01695c35`'s `UringOptions`,
`UringReader` and `read_cuts.h`:

- **Ring mode.** `--mode default|iopoll|sqpoll|sqpoll_iopoll`. `--sq-cpu N` sets `IORING_SETUP_SQ_AFF`. The SQ idle
  time is 10 000 ms, master's new default.
- **Wait.** `--wait block` calls `io_uring_submit_and_wait(1)`. `--wait spin` submits, then polls the CQ with a pause
  hint.
  - IOPOLL without SQPOLL always takes master's reap loop (`polls_in_wait`): submit, then `io_uring_get_events`
    (min_complete=0) passes. Its wait is labelled `reap`.
  - SQPOLL+IOPOLL uses block only. The SQ thread drives the polling, so a spin wait has nothing to add.
- **Read cuts.** `--cuts` implements `cut_legs` as master does. It cuts at `cut_bytes = min(max_sectors_kb,
  (max_segments − 1) pages)` and at every iovec join off `virt_boundary_mask`, which gives 520 192 B on the Samsungs
  and 262 144 B on the SPCC. The ring default then becomes `16 × parts × leg_stride`, as `ReaderCore::queue_depth()`
  sets it.
- **Buffers.** `--read-mode readv_fixed`, with `--arena` selecting the slab layout.
  - `--arena 0` registers each of the six per-name slabs as its own buffer, as production does with `SLAB_ARENA=0`.
    A READV_FIXED leg is then the run of iovecs inside one buffer (`fixed_legs`), so a sub-read fans out per slab:
    11 SQEs per row uncut.
  - `--arena 1` places the six slabs in one 2 MiB-aligned THP allocation, registered as a single 213 MB buffer
    (`SLAB_ARENA=1`, under the 1 GiB limit). No fan-out.
  - **What was registered is a bench arena of the production shape, not the real tier.** Each slab holds 16 row
    slots, not the full tier.
- **Fixed files.** `--fixed-files` registers the 120 row-image files (40 layers × 3 roots).
- **Ring size.** `--ring N` overrides the ring entries. The ring depth is also the SQE credit: a row starts only when
  its SQEs fit beside those already in flight.
- **CPU accounting.** Per-thread CPU comes from `/proc/self/task/*/schedstat`. It covers the submitting thread (which
  is also the waiter), the `iou-sqp-*` SQ thread, and the `iou-wrk-*` io-wq workers, and is reported per GB read.

**Matrix.**

- **Combos (48).**
  - default × {block, spin} × cuts {0, 1} × buffers {normal, normal+ff, rvf-slabs+ff, rvf-arena+ff}: 16.
  - iopoll (reap) × cuts × the same 4 buffer settings: 8.
  - sqpoll × SQ cpu {node 1: 18, node 0: 2} × {block, spin} × cuts × {normal, rvf-arena+ff}: 16.
  - sqpoll_iopoll × SQ cpu × cuts × {normal, rvf-arena+ff}: 8.
- **Cells.** Each combo runs at weights 1:1:1 and 1:0.9:1, at QD {1, 2}, for 2 reps, with 2000 rows per cell. That
  is 384 cells in one shuffled order (`random.Random(20260928)`).
- **Same rows.** Every combo reads the same rows for a given (weight, QD, rep).
- **Top phase.** The 4 leading combos plus production run at QD4 with the default ring, and at QD1 and QD4 with a
  4096-entry ring (2 reps).

```bash
# divix01, /mnt/nvme1/mirror-scaling; bench built from wt-mirror-bench at the branch head
gcc -O2 -pthread -o mb2 $W/analysis/dsv41-drive/mirror-scaling/mirror_bench.c -luring
taskset -c 0-17 python3 $W/analysis/dsv41-drive/mirror-scaling/run_uring_sweep.py ./mb2 uring-main.jsonl main
taskset -c 0-17 python3 .../run_uring_sweep.py ./mb2 uring-repairN.jsonl repair --labels-file results/uring-episodes-N.txt
taskset -c 0-17 python3 .../run_uring_sweep.py ./mb2 uring-top.jsonl top --top <5 combo labels>
python3 analysis/dsv41-drive/mirror-scaling/analyze_uring.py results/uring-main.jsonl.gz results/uring-repair{1,2,3}.jsonl.gz results/uring-top.jsonl.gz
```

All 492 runs exited 0, with 0 errors and 0 short reads.

### SPCC slow episodes: excluded and re-run

48 of the 384 main cells (12.5%) landed in an SPCC slow episode.

- **What an episode looks like.** The SPCC is at 0.97–0.99 utilization with SQE p50 of 3–10 ms, the Samsungs are
  starved at 0.2–0.58 utilization, and the row p50 is 1.9–5.6× the class median.
- **Detection is unambiguous.** Clean cells sit within 1.12× of their (weight, QD) median with Samsung utilization of
  at least 0.71; episode cells are far outside both. `analyze_uring.py::episode` flags a cell when its p50 is more
  than 1.5× the median or the lowest Samsung utilization is below 0.65.
- **Episodes cluster in time.** They came in runs of 2–8 consecutive cells, roughly every 3–5 minutes of continuous
  reading.
- **Re-runs under continuous load kept hitting them.** The flagged cells were re-run: 48 cells, then 4, then 3.
  Rounds 1 and 2 ran straight after continuous load, and 4 of their 52 cells, then 3 of 4, were caught again.
- **After a cool-down they were clean.** After a 3-minute idle, all 3 remaining cells came back clean.
- **Interpretation.** That fits a thermal or sustained-load throttle on the DRAM-less SPCC. It is not confirmed: the
  drive's temperature and throttle counters need root (`nvme smart-log`), and it has no hwmon node.
- **The final tables use each cell's latest clean record.** After exclusion, the two reps of a cell differ by a
  median of 5 µs (p90 22 µs).
- **For decode, this matters more than any io_uring setting.** About 12% of sustained-read time ran 2–5× slower here.
  Under decode's intermittent load it may be rarer, but an episode stalls every row that touches the SPCC.

## Ranked table (QD1 row p50, mean of 2 reps; rank key = mean of the two weights)

Combo labels are `mode[@sq-node].wait.cutsN.buffers`: `rvf` = `READ_MODE=readv_fixed`, `ff` = fixed files, `slabs` or
`arena` = `SLAB_ARENA=0` or `1`. "vs base" is against production, `default.block.cuts0.normal`. CPU columns are
CPU-s per GB read. "SQE p50" is from row submit to CQE, at 1:1:1.

| # | combo | QD1 p50 1:1:1 / 1:0.9:1 (us) | vs base | QD1 p90 / p99 | QD2 p50 1:1:1 / 1:0.9:1 | QD2 p99 | QD2 GB/s | CPU-s/GB QD1: submit + SQ + io-wq = total | QD2 total | io-wq workers | SPCC last (1:1:1 / 1:0.9:1) | SQE p50 nvme0/SPCC/nvme2 (1:1:1) | SQEs/row |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `iopoll.reap.cuts0.rvf-slabs+ff` | 1596 / 1519 | -97 | 1788 / 2278 | 2582 / 2644 | 3450 | 10.01 | 0.121 + 0.000 + 0.004 = 0.124 | 0.103 | 2 | 0.87 / 0.64 | 1310 / 585 / 742 | 11.0 |
| 2 | `default.spin.cuts1.rvf-arena+ff` | 1650 / 1524 | -68 | 1821 / 2231 | 2600 / 2641 | 3533 | 9.96 | 0.122 + 0.000 + 0.000 = 0.122 | 0.099 | 0 | 0.85 / 0.49 | 930 / 1251 / 1093 | 40.4 |
| 3 | `iopoll.reap.cuts1.rvf-arena+ff` | 1653 / 1524 | -67 | 1826 / 2210 | 2601 / 2641 | 3473 | 9.99 | 0.122 + 0.000 + 0.000 = 0.122 | 0.100 | 0 | 0.85 / 0.51 | 931 / 1249 / 1089 | 40.4 |
| 4 | `default.spin.cuts0.rvf-arena+ff` | 1659 / 1523 | -64 | 1834 / 2356 | 2605 / 2643 | 3444 | 9.99 | 0.123 + 0.000 + 0.000 = 0.123 | 0.099 | 0 | 0.88 / 0.54 | 1306 / 1523 / 1337 | 6.0 |
| 5 | `sqpoll_iopoll@n1.block.cuts1.rvf-arena+ff` | 1658 / 1526 | -63 | 1835 / 2386 | 2595 / 2639 | 3438 | 10.00 | 0.006 + 0.124 + 0.000 = 0.130 | 0.106 | 0 | 0.81 / 0.52 | 933 / 1253 / 1105 | 40.4 |
| 6 | `iopoll.reap.cuts0.rvf-arena+ff` | 1659 / 1527 | -63 | 1838 / 2352 | 2605 / 2644 | 3512 | 9.96 | 0.123 + 0.000 + 0.003 = 0.127 | 0.103 | 1 | 0.81 / 0.53 | 1310 / 1511 / 1326 | 6.0 |
| 7 | `sqpoll_iopoll@n0.block.cuts1.rvf-arena+ff` | 1658 / 1529 | -61 | 1827 / 2243 | 2604 / 2639 | 3465 | 9.99 | 0.012 + 0.123 + 0.000 = 0.136 | 0.112 | 0 | 0.82 / 0.51 | 938 / 1254 / 1107 | 40.4 |
| 8 | `iopoll.reap.cuts1.rvf-slabs+ff` | 1661 / 1527 | -61 | 1837 / 2280 | 2605 / 2641 | 3460 | 9.99 | 0.123 + 0.000 + 0.000 = 0.123 | 0.100 | 0 | 0.84 / 0.51 | 932 / 1292 / 1096 | 42.0 |
| 9 | `sqpoll@n1.spin.cuts0.rvf-arena+ff` | 1665 / 1524 | -61 | 1842 / 2316 | 2607 / 2644 | 3680 | 9.94 | 0.123 + 0.124 + 0.000 = 0.248 | 0.201 | 0 | 0.85 / 0.52 | 1305 / 1526 / 1334 | 6.0 |
| 10 | `sqpoll@n0.spin.cuts1.rvf-arena+ff` | 1661 / 1528 | -60 | 1828 / 2263 | 2602 / 2643 | 3608 | 9.96 | 0.123 + 0.123 + 0.000 = 0.246 | 0.200 | 0 | 0.81 / 0.49 | 948 / 1253 / 1109 | 40.4 |
| 11 | `sqpoll@n1.block.cuts0.rvf-arena+ff` | 1665 / 1526 | -59 | 1840 / 2353 | 2611 / 2645 | 4740 | 9.86 | 0.001 + 0.124 + 0.000 = 0.125 | 0.103 | 0 | 0.87 / 0.52 | 1308 / 1529 / 1342 | 6.0 |
| 12 | `sqpoll@n0.spin.cuts0.rvf-arena+ff` | 1669 / 1523 | -59 | 1831 / 2263 | 2608 / 2642 | 3457 | 9.98 | 0.123 + 0.123 + 0.000 = 0.246 | 0.200 | 0 | 0.83 / 0.52 | 1305 / 1525 / 1329 | 6.0 |
| 13 | `sqpoll@n1.block.cuts1.rvf-arena+ff` | 1657 / 1535 | -59 | 1845 / 4662 | 2602 / 2641 | 3544 | 9.96 | 0.006 + 0.129 + 0.000 = 0.136 | 0.106 | 0 | 0.83 / 0.51 | 935 / 1254 / 1104 | 40.4 |
| 14 | `sqpoll@n1.spin.cuts1.rvf-arena+ff` | 1663 / 1529 | -59 | 1828 / 2225 | 2604 / 2642 | 3492 | 9.99 | 0.123 + 0.123 + 0.000 = 0.246 | 0.200 | 0 | 0.81 / 0.50 | 932 / 1255 / 1108 | 40.4 |
| 15 | `sqpoll_iopoll@n1.block.cuts0.rvf-arena+ff` | 1658 / 1538 | -57 | 1850 / 2363 | 2608 / 2644 | 3821 | 9.93 | 0.001 + 0.124 + 0.013 = 0.139 | 0.139 | 1 | 0.82 / 0.49 | 1313 / 1513 / 1335 | 6.0 |
| 16 | `sqpoll@n0.block.cuts1.rvf-arena+ff` | 1662 / 1535 | -56 | 1841 / 2338 | 2601 / 2641 | 3473 | 9.96 | 0.013 + 0.124 + 0.000 = 0.137 | 0.113 | 0 | 0.82 / 0.51 | 938 / 1258 / 1114 | 40.4 |
| 17 | `sqpoll@n0.block.cuts0.rvf-arena+ff` | 1670 / 1529 | -56 | 1837 / 2283 | 2612 / 2646 | 3661 | 9.94 | 0.003 + 0.123 + 0.000 = 0.126 | 0.103 | 0 | 0.85 / 0.53 | 1313 / 1531 / 1340 | 6.0 |
| 18 | `default.spin.cuts1.rvf-slabs+ff` | 1669 / 1537 | -53 | 1834 / 2214 | 2606 / 2642 | 3959 | 9.92 | 0.122 + 0.000 + 0.000 = 0.122 | 0.100 | 0 | 0.84 / 0.53 | 932 / 1294 / 1104 | 42.0 |
| 19 | `iopoll.reap.cuts1.normal` | 1667 / 1539 | -52 | 1845 / 2333 | 2593 / 2664 | 3641 | 9.91 | 0.124 + 0.000 + 0.000 = 0.124 | 0.101 | 0 | 0.80 / 0.47 | 934 / 1262 / 1132 | 40.4 |
| 20 | `default.spin.cuts0.rvf-slabs+ff` | 1682 / 1526 | -51 | 1842 / 2274 | 2605 / 2660 | 3628 | 9.90 | 0.123 + 0.000 + 0.000 = 0.123 | 0.100 | 0 | 0.85 / 0.52 | 1307 / 1548 / 1408 | 11.0 |
| 21 | `sqpoll@n1.spin.cuts1.normal` | 1671 / 1540 | -50 | 1834 / 2235 | 2611 / 2642 | 3471 | 9.98 | 0.124 + 0.124 + 0.000 = 0.248 | 0.199 | 0 | 0.80 / 0.46 | 934 / 1265 / 1139 | 40.4 |
| 22 | `sqpoll@n0.spin.cuts1.normal` | 1671 / 1540 | -50 | 1844 / 2601 | 2616 / 2672 | 4135 | 9.77 | 0.125 + 0.125 + 0.000 = 0.250 | 0.204 | 0 | 0.81 / 0.45 | 942 / 1269 / 1129 | 40.4 |
| 23 | `default.spin.cuts1.normal` | 1672 / 1540 | -49 | 1839 / 2239 | 2600 / 2641 | 3437 | 10.00 | 0.123 + 0.000 + 0.000 = 0.123 | 0.099 | 0 | 0.79 / 0.47 | 935 / 1263 / 1139 | 40.4 |
| 24 | `default.spin.cuts1.normal+ff` | 1670 / 1543 | -48 | 1834 / 2218 | 2600 / 2642 | 3445 | 9.99 | 0.123 + 0.000 + 0.000 = 0.123 | 0.099 | 0 | 0.83 / 0.50 | 936 / 1264 / 1125 | 40.4 |
| 25 | `sqpoll_iopoll@n1.block.cuts1.normal` | 1674 / 1540 | -48 | 1836 / 2250 | 2601 / 2640 | 3466 | 9.99 | 0.006 + 0.124 + 0.000 = 0.130 | 0.106 | 0 | 0.80 / 0.50 | 936 / 1265 / 1134 | 40.4 |
| 26 | `sqpoll_iopoll@n0.block.cuts1.normal` | 1673 / 1542 | -48 | 1854 / 2823 | 2611 / 2670 | 4367 | 9.76 | 0.013 + 0.126 + 0.000 = 0.139 | 0.114 | 0 | 0.84 / 0.50 | 947 / 1274 / 1124 | 40.4 |
| 27 | `sqpoll@n0.spin.cuts0.normal` | 1683 / 1534 | -47 | 1856 / 2536 | 2607 / 2673 | 3790 | 9.83 | 0.125 + 0.125 + 0.000 = 0.251 | 0.203 | 0 | 0.86 / 0.50 | 1315 / 1542 / 1354 | 6.0 |
| 28 | `iopoll.reap.cuts1.normal+ff` | 1674 / 1542 | -47 | 1837 / 2251 | 2608 / 2641 | 6418 | 9.49 | 0.124 + 0.000 + 0.000 = 0.124 | 0.106 | 0 | 0.79 / 0.50 | 951 / 1262 / 1131 | 40.4 |
| 29 | `sqpoll@n0.block.cuts1.normal` | 1673 / 1549 | -44 | 1852 / 2663 | 2601 / 2658 | 4188 | 9.83 | 0.013 + 0.126 + 0.000 = 0.139 | 0.114 | 0 | 0.82 / 0.50 | 948 / 1274 / 1134 | 40.4 |
| 30 | `sqpoll@n1.block.cuts1.normal` | 1679 / 1544 | -44 | 1842 / 2238 | 2610 / 2642 | 3709 | 9.93 | 0.006 + 0.124 + 0.000 = 0.131 | 0.107 | 0 | 0.79 / 0.48 | 937 / 1269 / 1144 | 40.4 |
| 31 | `default.spin.cuts0.normal` | 1689 / 1537 | -42 | 1854 / 2445 | 2606 / 2644 | 3941 | 9.92 | 0.124 + 0.000 + 0.000 = 0.124 | 0.100 | 0 | 0.87 / 0.52 | 1312 / 1550 / 1363 | 6.0 |
| 32 | `iopoll.reap.cuts0.normal` | 1680 / 1548 | -41 | 1849 / 2287 | 2609 / 2642 | 3475 | 9.98 | 0.124 + 0.000 + 0.006 = 0.130 | 0.106 | 2 | 0.84 / 0.51 | 1318 / 1535 / 1364 | 6.0 |
| 33 | `sqpoll@n1.spin.cuts0.normal` | 1691 / 1540 | -40 | 1857 / 2414 | 2601 / 2643 | 3442 | 10.00 | 0.125 + 0.126 + 0.000 = 0.250 | 0.200 | 0 | 0.85 / 0.51 | 1310 / 1555 / 1360 | 6.0 |
| 34 | `sqpoll@n1.block.cuts0.normal` | 1692 / 1540 | -39 | 1858 / 2287 | 2614 / 2643 | 3492 | 9.97 | 0.001 + 0.125 + 0.000 = 0.126 | 0.102 | 0 | 0.88 / 0.50 | 1314 / 1560 / 1368 | 6.0 |
| 35 | `default.block.cuts1.rvf-arena+ff` | 1681 / 1552 | -38 | 1844 / 2245 | 2599 / 2642 | 3447 | 9.99 | 0.016 + 0.000 + 0.000 = 0.016 | 0.015 | 0 | 0.80 / 0.48 | 956 / 1268 / 1127 | 40.4 |
| 36 | `sqpoll@n0.block.cuts0.normal` | 1694 / 1541 | -38 | 1863 / 2505 | 2608 / 2673 | 3794 | 9.82 | 0.003 + 0.126 + 0.000 = 0.129 | 0.104 | 0 | 0.85 / 0.50 | 1316 / 1553 / 1362 | 6.0 |
| 37 | `default.spin.cuts0.normal+ff` | 1693 / 1541 | -38 | 1861 / 2381 | 2604 / 2644 | 3964 | 9.92 | 0.125 + 0.000 + 0.000 = 0.125 | 0.100 | 0 | 0.86 / 0.52 | 1312 / 1552 / 1358 | 6.0 |
| 38 | `iopoll.reap.cuts0.normal+ff` | 1681 / 1558 | -36 | 1862 / 2403 | 2607 / 2643 | 3474 | 9.98 | 0.125 + 0.000 + 0.006 = 0.131 | 0.106 | 2 | 0.83 / 0.54 | 1318 / 1536 / 1353 | 6.0 |
| 39 | `default.block.cuts1.rvf-slabs+ff` | 1684 / 1559 | -34 | 1852 / 2256 | 2611 / 2642 | 3474 | 9.98 | 0.016 + 0.000 + 0.000 = 0.016 | 0.016 | 0 | 0.83 / 0.49 | 940 / 1310 / 1120 | 42.0 |
| 40 | `sqpoll_iopoll@n0.block.cuts0.rvf-arena+ff` | 1688 / 1562 | -30 | 1862 / 2300 | 2604 / 2640 | 3447 | 9.98 | 0.003 + 0.125 + 0.007 = 0.135 | 0.111 | 2 | 0.82 / 0.52 | 1323 / 1538 / 1351 | 6.0 |
| 41 | `default.block.cuts0.rvf-arena+ff` | 1706 / 1548 | -28 | 1864 / 2301 | 2617 / 2642 | 3862 | 9.92 | 0.008 + 0.000 + 0.000 = 0.008 | 0.008 | 0 | 0.82 / 0.51 | 1307 / 1545 / 1333 | 6.0 |
| 42 | `default.block.cuts0.rvf-slabs+ff` | 1717 / 1544 | -25 | 1871 / 2290 | 2611 / 2642 | 3472 | 9.98 | 0.009 + 0.000 + 0.000 = 0.009 | 0.009 | 0 | 0.88 / 0.48 | 1311 / 1589 / 1432 | 11.0 |
| 43 | `sqpoll_iopoll@n1.block.cuts0.normal` | 1690 / 1574 | -23 | 1867 / 2318 | 2618 / 2646 | 3459 | 9.98 | 0.001 + 0.126 + 0.018 = 0.145 | 0.140 | 2 | 0.81 / 0.51 | 1316 / 1539 / 1368 | 6.0 |
| 44 | `default.block.cuts1.normal+ff` | 1689 / 1579 | -21 | 1862 / 2246 | 2615 / 2641 | 3477 | 9.98 | 0.018 + 0.000 + 0.000 = 0.018 | 0.018 | 0 | 0.80 / 0.44 | 944 / 1282 / 1147 | 40.4 |
| 45 | `default.block.cuts1.normal` | 1698 / 1590 | -11 | 1878 / 2539 | 2602 / 2642 | 3762 | 9.93 | 0.019 + 0.000 + 0.000 = 0.019 | 0.018 | 0 | 0.76 / 0.43 | 945 / 1284 / 1164 | 40.4 |
| 46 | `default.block.cuts0.normal+ff` | 1723 / 1583 | -2 | 1883 / 2299 | 2605 / 2642 | 3494 | 9.96 | 0.010 + 0.000 + 0.000 = 0.010 | 0.011 | 0 | 0.89 / 0.45 | 1315 / 1570 / 1367 | 6.0 |
| 47 | `default.block.cuts0.normal` | 1736 / 1575 | +0 | 1892 / 2383 | 2609 / 2642 | 3458 | 9.99 | 0.011 + 0.000 + 0.000 = 0.011 | 0.010 | 0 | 0.89 / 0.48 | 1315 / 1578 / 1370 | 6.0 |
| 48 | `sqpoll_iopoll@n0.block.cuts0.normal` | 1717 / 1617 | +12 | 1908 / 2566 | 2625 / 2661 | 4096 | 9.75 | 0.003 + 0.130 + 0.012 = 0.145 | 0.119 | 2 | 0.81 / 0.43 | 1335 / 1563 / 1381 | 6.0 |

Notes on the table:

- **#1 (`iopoll.reap.cuts0.rvf-slabs+ff`) is reproducible but geometry-specific.** Its 1:1:1 reps are 1596.1 and
  1596.3 µs. At 1:1:1 the slab fan-out gives the SPCC small READV_FIXED legs, which IOPOLL completes by polling, while
  its large uncut legs are punted to one io-wq worker. That cuts the SPCC's part p50 to about 1580 µs, from 1625–1657
  in the other top combos. At 1:0.9:1 the part boundaries move and the edge disappears: 1519 against 1523–1529. It
  also relies on the io-wq punt path that `iopoll/diagnosis.md` warns about. Do not pick it.
- **Every combo with a thread spinning on completions costs about 0.12 CPU-s per GB.** That is one full core while
  reads are in flight, and the rows with 0.24–0.25 are two cores. The combos without one cost 0.008–0.019.
- **An SQPOLL thread spins for the whole idle window.** It keeps spinning for 10 s after the last read, which in
  decode means always.
- **Straggler share.** The SPCC finishes last in 76–89% of 1:1:1 rows in every combo, and in 43–64% at 1:0.9:1. No
  io_uring setting changes which drive is last; only the weight does.

## Main effects (mean of QD1 p50 over the two weights, us; over the combos that have the value)


## Top combos: QD4, and a 4096-entry ring

| combo | weights | QD4 p50 / p99 / GB/s (default ring) | QD1 p50 (ring 4096) | QD4 p50 / p99 / GB/s (ring 4096) |
|---|---|---|---|---|
| `default.block.cuts0.normal` | w1 | 5231 / 6082 / 10.15 | 1735 | 5238 / 6064 / 10.14 |
| `default.block.cuts0.normal` | w0.9 | 5302 / 6283 / 9.90 | 1573 | 5306 / 6305 / 9.90 |
| `default.block.cuts1.rvf-arena+ff` | w1 | 5229 / 6029 / 10.15 | 1669 | 5218 / 6070 / 10.16 |
| `default.block.cuts1.rvf-arena+ff` | w0.9 | 5302 / 6276 / 9.90 | 1550 | 5301 / 6274 / 9.90 |
| `default.spin.cuts1.rvf-arena+ff` | w1 | 5231 / 6069 / 10.14 | 1657 | 5234 / 6054 / 10.15 |
| `default.spin.cuts1.rvf-arena+ff` | w0.9 | 5302 / 6274 / 9.90 | 1528 | 5312 / 6282 / 9.88 |
| `iopoll.reap.cuts0.rvf-slabs+ff` | w0.9 | 5301 / 6281 / 9.90 | 1518 | 5304 / 6286 / 9.90 |
| `sqpoll_iopoll@n1.block.cuts1.rvf-arena+ff` | w1 | 5226 / 6157 / 10.16 | 1656 | 5235 / 6185 / 10.14 |
| `sqpoll_iopoll@n1.block.cuts1.rvf-arena+ff` | w0.9 | 5301 / 6266 / 9.90 | 1525 | 5300 / 6271 / 9.90 |

- **QD4 is bandwidth-bound and identical across combos.** At 1:1:1 every combo gives 10.14–10.16 GB/s with a row p50
  of 5.22–5.24 ms. At 1:0.9:1 every combo gives 9.88–9.90 GB/s. The 1:0.9:1 weight costs 2.5% of QD4 bandwidth.
- **The larger ring is neutral.** A 4096-entry ring changes nothing: QD1 is within ±15 µs of the default ring, and
  QD4 matches.
- **The ring is never full.** The default ring (64 entries uncut, 1024 cut) already holds more than QD4's SQEs.
- **Missing row.** The `iopoll.reap.cuts0.rvf-slabs+ff` 1:1:1 row is absent because both of its default-ring QD4 reps
  landed in an SPCC episode.
