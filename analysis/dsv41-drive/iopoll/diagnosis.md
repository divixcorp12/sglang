# Why `SGLANG_EXPERT_STREAM_URING_MODE=iopoll` is slower on divix01

2026-09-28. Branch `cc/iopoll-diag` from `origin/master` `e1559b948b`. Diagnosis only: production code, the
recipe and master are untouched.

## Verdict

IOPOLL costs time through two effects that only hurt together.

1. **Every production read is punted to io-wq.** One row-image sub-read is a single ~2.2 MB `READV`. The drives take
   at most 512 KiB per request (Samsung 990 EVO Plus) or 256 KiB (SPCC), so the block layer has to split the bio.
   Under IOPOLL, io_uring issues the read non-blocking, and the polled bio carries `REQ_NOWAIT`. The block layer
   refuses to split a `REQ_NOWAIT` bio and returns `-EAGAIN`, so io_uring re-issues the read from an io-wq worker.
   A join between iovecs that is not page-aligned (the 9216, 4608 and 10240 B slab rows) crosses the NVMe
   `virt_boundary` (4095). That forces the same split, even for small requests. Default mode never sets
   `REQ_NOWAIT` on a read, so it splits silently and punts nothing.
2. **The punted reads then queue behind a lock the waiter holds while it polls.** `WAIT_MODE=block` waits with
   `io_uring_submit_and_wait(ring, n)` (`uring_reader.h:203`), which is `io_uring_enter(GETEVENTS,
   min_complete=n)`. On an IOPOLL ring, that call polls the device inside the kernel until n completions arrive,
   holding the ring's `uring_lock`. Each io-wq worker needs that lock to queue its issued read on the poll list,
   and it spins (state R) until the waiter lets go. So the punted reads are issued one after another, each behind
   earlier completions. The drive that gets its reads last waits longest: nvme3's SQE p50 goes 1.39 → 2.8–3.0 ms
   while nvme0 stays at 1.30 ms. Row p50 goes 1.70 → 3.2 ms, and the workers burn ~0.8 CPU each.

**Evidence that (2) is the cost.** Waiting with `GETEVENTS min_complete=0` instead makes one polling pass per
call and drops the lock between passes (the reader's own `WAIT_MODE=spin` path, `uring_reader.h:215`). With it,
the same punted production shape runs at a row p50 of **1.61 ms**, below default mode's 1.69 ms. io-wq CPU falls
from ~2.0 to ~0.1 CPU-s per 2 s. Cutting SQEs so that nothing punts fixes it too, even with the blocking wait
(1.61 ms, 0 workers).

**Known and unknown.** The size threshold, the gap effect, the worker count, the worker CPU, the per-drive
ordering, and the block vs spin wait result are measured. The named kernel paths come from upstream io_uring and
block source (the ~6.15 era this EL10 kernel backports), and they fit every measurement. They are **not** confirmed
by kernel stacks on this box, which need root (see "Root-only confirmation" below).

The Codex campaign's finding that IOPOLL punts to io-wq is right but incomplete. The punt costs almost nothing on
its own. It costs 1.5 ms per row only when it meets the blocking poller. That also explains why their S7
(`sqpoll_iopoll` + spin) was neutral: the SQ thread polls and releases the lock between passes.

## Host facts (read 2026-09-28)

| root | device | model | fs | `max_sectors_kb` = `max_hw_sectors_kb` | `max_segments` | `virt_boundary_mask` | hctx (1 poll) |
|---|---|---|---|---|---|---|---|
| /mnt/nvme0 | nvme0n1 | Samsung 990 EVO Plus 2TB | xfs | 512 | 128 | 4095 | 16 |
| /mnt/nvme4 | nvme2n1 | SPCC M.2 PCIe SSD | ext4 | **256** | 65 | 4095 | 31 |
| /mnt/nvme2 | nvme3n1 | Samsung 990 EVO Plus 2TB | xfs | 512 | 128 | 4095 | 16 |

- Other host settings: `nvme.poll_queues=1`, `io_poll=1`, kernel `6.12.0-211.60.1.el10_2`, liburing 2.12.
- Unprivileged profiling: `perf_event_paranoid=2`, so there is no kernel profiling. `bpftrace` is absent.
- `max_sectors_kb` already equals `max_hw_sectors_kb`, which is set by the drive's MDTS, so **no system setting can
  remove the split.**
- The row geometry comes from `layer-000.rows.digests.json`: a 13,315,584 B image with a stride of 13,316,096.
  - The segments' row_bytes are 8847360, 20480, 9216, 4423680, 4608 and 10240.
  - Three mirror parts at `per_part=2` give 6 sub-reads per row, each one ~2.2 MB `READV` with up to 4 iovecs.

## Microbenchmark

`iopoll_bench.c` depends only on liburing. `run_matrix.sh` runs every variant under `rowimg-disk.lock` with
`taskset -c 18-35` (the drives' NUMA node). Raw JSON lines are in `results/`. The reported fields are:

- latency percentiles per op;
- p50 and p99 per drive, measured per SQE;
- CPU for the process and for the submitting thread;
- `iou-wrk-*` threads: how many were seen, the peak live count, and their CPU, from `/proc/self/task`;
- the run state and wchan sampled from each worker;
- block requests per SQE, from `/proc/diskstats`.

- `flat`: SQEs of one size into one page-aligned buffer, at QD SQEs in flight.
- `row`: the reader's direct-mode shape, with 6 slab allocations and `SLOTS=16` destination rows per slab.
  - Each row is 3 parts over 3 drives, then `split_part` gives 6 `READV` SQEs.
  - Their iovecs are the slab rows, as in `RowReader::image_iovecs` (`row_reader.h:207`).
  - `--cut N` cuts each sub-read into SQEs of at most N bytes, and at every iovec join that is not on a page
    boundary. `--no-gap-cut` keeps the size cut but skips the gap cut.
  - `--op read` makes one `READ` per iovec.
- `--wait block` uses `io_uring_submit_and_wait(1)`, like `WAIT_MODE=block`. `--wait spin` submits, then loops
  `io_uring_get_events` (min_complete=0) until a CQE appears, like `WAIT_MODE=spin` with IOPOLL.

```bash
# divix01, in /data/models/slang/nvfp4-work/wt-iopoll-diag (origin/cc/iopoll-diag)
gcc -O2 -pthread -o /mnt/nvme1/iopoll-diag/ib analysis/dsv41-drive/iopoll/iopoll_bench.c -luring
OMP_NUM_THREADS=8 bash analysis/dsv41-drive/iopoll/run_matrix.sh /mnt/nvme1/iopoll-diag/ib out.jsonl flat row threads fixed repeat qd wait
```

Every variant ran for 2 s against `layer-003.rows` on the three mirror roots. All runs exited 0, with 0 errors and 0
short reads.

### 1. Punting starts exactly at the device's request limit (flat, QD1, block wait)

| SQE size | nvme0 (512 KiB) p50 default / iopoll µs | io-wq workers | reqs/SQE | nvme4 (256 KiB) p50 default / iopoll µs | io-wq workers | reqs/SQE |
|---|---|---|---|---|---|---|
| 4 KiB | 50 / **39** | 0 | 1 | 131 / 118 | 0 | 1 |
| 64 KiB | 88 / 79 | 0 | 1 | 303 / 298 | 0 | 1 |
| 256 KiB | 144 / 132 | 0 | 1 | 588 / 563 | 0 | 1 |
| 512 KiB | 210 / 199 | 0 | 1 | 658 / 653 | **1** | 2 |
| 1 MiB | 361 / 353 | **1** | 2 | 749 / 741 | 1 | 4 |
| 2.1 MiB | 696 / 687 | 1 | 5 | 1038 / 1028 | 1 | 9 |

- Workers appear exactly when an SQE needs more than one block request. On xfs and on ext4 alike, READV polls
  properly below the limit, so neither the filesystem nor readv is the cause.
- IOPOLL only helps small requests: 4 KiB goes −11 µs, and at 256 KiB and above the gain is ≤ 12 µs.
- It keeps the waiting thread 100% busy: 1.99 CPU-s per 2 s, against 0.03–0.3 CPU-s in default mode.
- At QD8, the punted sizes run two workers at ~1 CPU each (1.96 CPU-s). Bandwidth is unchanged, because the drive
  is the ceiling at 3.57 GB/s.

### 2. The production row shape (QD1, 5 interleaved reps; median of the per-run p50s)

| variant | SQEs/row | default p50 µs | iopoll (block wait) p50 µs | iopoll io-wq workers / CPU-s |
|---|---|---|---|---|
| production (`READV` ~2.2 MB) | 6 | 1696 | **3233** (2249–3334) | 1–2 / 1.56–1.93 |
| cut 512 KiB + page-gap cuts | 33 | 1714 | 1641 | 1–3 / 0.75 (the SPCC's 256 KiB limit) |
| cut 256 KiB + page-gap cuts | 57 | 1652 | **1622** | **0 / 0** |
| cut 256 KiB, no gap cuts | 54 | 1660 | 1602 | 1 / 0.31 (the gaps alone punt) |
| `READ` per iovec, cut 256 KiB (single runs) | 59 | 1679 | 1641 | 0 / 0 |

- Per-drive SQE p50 for the production shape under IOPOLL: nvme0 is 1305 µs in both modes. nvme3n1, whose reads
  are submitted last, goes from **1391 to 2816–3056 µs**. That is the serial, lock-gated worker handoff.
- Fixed files change nothing. Production shape: iopoll 3243 µs with 2 workers, default 1716. Cut to 256 KiB:
  iopoll 1611, default 1648.

### 3. The waiter, not the punt, is the cost (QD1, 3 interleaved reps; median of the per-run p50s)

| variant | default + block | default + spin | iopoll + block | iopoll + spin |
|---|---|---|---|---|
| production shape p50 µs | 1687 | 1630 | **2095** (p99 3.1–3.6 ms) | **1609** |
| production io-wq CPU-s | 0 | 0 | 1.91–2.57 | **0.09–0.11** |
| cut 256 KiB p50 µs | 1641 | 1616 | 1608 | 1614 |
| waiting thread CPU-s / 2 s | 0.16 | 1.98 | 1.99 | 1.99 |
| flat 2.1 MiB QD8 io-wq CPU-s | — | — | 1.97 | 0.03 |

- Worker samples show R (running) in every sample not spent idle in `io_wq_worker`, and D only once. The workers
  spin; they do not sleep on I/O (`results/wchan.jsonl`).
- Everything below 1.69 ms in this table comes from a spinning waiter, and default mode + spin gets most of it
  (−57 µs). IOPOLL adds −10 to −20 µs at most on this shape.

### 4. Not the cause

- **One poll queue per device.** With the non-punting shapes, 1, 2 and 4 threads (each with its own IOPOLL ring,
  sharing the one poll queue) show 0 workers and the same bandwidth as default: 3.56 GB/s at flat 256 KiB, QD8. CPU
  grows by one core per polling thread.
- **Queue depth.** At QD2 and QD4 rows, both modes reach the same 10.2 GB/s three-drive ceiling, but the production
  shape under IOPOLL has a p99 of 14.5 ms against 6.1 ms. IOPOLL cannot raise bandwidth.
- **A non-poll hctx.** Not observable without debugfs. If polled bios had gone to a queue with interrupts, the reads
  that fit the limit would not poll cleanly with zero workers, and they do.

## Proposed fix

### Code (ours to do; none is applied here)

1. **Never block-wait on an IOPOLL ring without SQPOLL.** This alone removes the regression.
   - In `UringReader::submit` (`uring_reader.h:201-220`), when `options_.iopoll() && !options_.sqpoll()`, take the
     `io_uring_get_events` loop even when `WAIT_MODE=block`. The alternative is to refuse `MODE=iopoll
     WAIT_MODE=block` at `from_env`, the same way other unsupported combinations are refused.
   - `wait_one` (`:453-460`, `io_uring_wait_cqe`, used by `drain`) has the same shape. It is a cold path, but it
     should get the same change.
   - Expected effect: 3.23 → 1.61 ms per row at QD1 (table 3). The price is a spinning waiter, which an IOPOLL
     ring has regardless.
2. **Cut reads into legs the device can take whole.** This is for any IOPOLL use, and it is optional in default
   mode.
   - At `open()`, read each fd's `queue/max_sectors_kb` and `virt_boundary_mask` via `fstat` →
     `/sys/dev/block/M:m`, using the parent disk for a partition.
   - In `ReaderCore`'s prepare step (`reader_core.h:1024-1027`), cut every `READV` at that size and at every iovec
     join that is not page-aligned. Reuse the leg machinery fixed reads already have (`fixed_legs`,
     `legs_inflight`, `reader_core.h:640-654`), so a sub-read is still vetted and published only once all its
     legs land.
   - Expected effect: 0 io-wq workers under IOPOLL even with the blocking wait (1.62 ms; table 2). In default mode,
     −40 µs per row at QD1 (1652 vs 1696 µs over 5 reps; about −2.5%).
   - Cost: ~10× the SQEs (57 against 6 per row) and about twice the submitter CPU at saturation (0.33 against 0.16
     CPU-s per 2 s). The ring depth must grow to match (`16 * parts` today).
3. **Recipe advice (no change proposed here): keep `MODE=default`.** For 2.2 MB, bandwidth-bound reads, IOPOLL gains
   10–20 µs per row over default + spin, and it costs a full core. S6 (SQPOLL + spin) already showed that a spinning
   waiter does not reach decode (+0.3 ms/token).

### System (needs root; for the user)

- **None is needed for the fix.** Raising `max_sectors_kb` is impossible: it already equals the hardware limit.
- `nvme.poll_queues=N` (a module parameter, set by a reload or reboot) helps only if several threads poll at
  once. One poll queue served four polling threads without loss here. Leave it at 1 unless the reader moves to one
  polling ring per drive or per thread.
- **The SPCC drive (nvme2n1, /mnt/nvme4) has slow episodes.**
  - In 7 of the 24 QD2/QD4 runs of the `qd` phase, and 3 of 8 QD4 runs of the `row` phase, its SQE p50 rose to 10–35 ms and total bandwidth fell to ~1.4 GB/s in both
    modes. Flat QD8 1–2 MiB reads on it dropped to ~0.65 GB/s. The Samsung drives never did this.
  - Because it is one of the three mirror roots, it gates whole-row latency.
  - Worth `nvme smart-log /dev/nvme2` (for thermal-throttle counts) and a look at the drive before its next long
    run. Moving that mirror to another Samsung-class drive would remove the tail.
- **Security.** `/usr/local/sbin/nsys-profile` (NOPASSWD) is `exec nsys "$@"`. `nsys profile <any command>` therefore
  runs that command as root, which makes the wrapper passwordless root. Restrict it to fixed arguments if that is
  not intended. It was used here once, for a 2 s CPU-sampling capture of the benchmark. That capture yielded only
  user-mode frames (no kernel samples), and it was deleted.

### Root-only confirmation of the lock mechanism (optional)

During `ib --mode iopoll --wait block --workload row ...`, the prediction is:

- `cat /proc/<iou-wrk tid>/stack`, or `perf record -g -t <tid>`, would show `__mutex_lock` /
  `mutex_optimistic_spin` under `io_iopoll_req_issued` ← `io_wq_submit_work`.
- `perf stat -e block:block_split,io_uring:io_uring_queue_async_work` would count one async punt per SQE in the
  production shape, and none with `--cut 262144`.

## Expected decode impact

- **Fix 1 (with IOPOLL):** removes the +10.2 ms/token. Predicted to land within ±1 ms/token of S0 (104.2), since
  at QD1 IOPOLL + spin is ≤ default + block. The gain beyond S0 is bounded by what S6 showed for a spinning waiter
  (neutral).
- **Fix 2 in default mode:** −40 µs per row read.
  - Upper bound: if all ~45 rows read per token (S0's rows_read / tokens, prefill included) were serial on the
    critical path, that is ≤ ~1.8 ms/token.
  - Realistically, much less, because reads overlap across three drives and with compute.
- **No decode arm has been run.** A two-arm confirmation would settle it: S0 against `MODE=iopoll WAIT_MODE=spin`,
  3 alternating pairs. That arm needs no code change, because the reader's spin path already makes the
  `min_complete=0` call. Fix 2 would need its own arm after implementation.

## Open questions

- The lock-spin mechanism is inferred from upstream source plus user-visible counters. Kernel stacks (root) would
  confirm it.
- Why S7 still punted (6 workers, 1.45 s): SQPOLL issues the reads non-blocking too, so the split refusal
  applies. Its SQ thread releases `uring_lock` between passes, which is consistent with S7 staying neutral.
- The benchmark's parts are page-aligned thirds of the row image. The reader's real part cut (2 MiB-aligned NUMA
  splits) gives slightly different sub-read sizes. Every one of them is still far above 512 KiB, so the conclusion
  does not depend on this.
- The SPCC slow episodes: thermal, SLC exhaustion, or firmware? Not investigated beyond the counters above.
