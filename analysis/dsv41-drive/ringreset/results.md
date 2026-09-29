# Ring reset without re-registration: results (branch `cc/ringreset`)

Plan: `2026-09-29-ring-reset-nop-drain`; its working copy is `plan.md` in the lead's scratchpad. The branch was cut
from `origin/master` = `ba01695c35`.

Commits:

- `0931fbe42f`: the RED commit. It adds the `registrations()` seam, the native harness, and the big-slab `reset`
  mode.
- `e5291d8c52`: the fix. Unconsumed SQEs are discarded as NOPs on the same ring. A fixed mode throws when the kernel
  refuses that NOP submit.
- `0a1f682b9e`: review fixes. The NOP submit retries soft errors for 2 s; `close()` survives a refused drain; the NOP
  tag is `~0 - 1`; a test hook, `publish_sq_without_enter`, is added.
- `2c4a598712`: the fake-liburing driver contract now asserts the NOP drain.

> **Timing provenance.** The numbers below were re-taken after divix01's reboot (2026-09-29 02:05-02:56 CDT, about
> 133 GiB free, swap unused). A first set was taken after about 01:15 under memory pressure: another agent's
> clone-buffers experiment had leaked about 150 GiB of pinned memory and swap was in use. That set appears only in
> the "Earlier runs" note and is not valid as a timing.

## The defect

When a failed submit left SQEs that the kernel had never consumed, `UringReader::drain`:

1. closed the ring;
2. re-created it;
3. re-registered every fixed buffer chunk and file (`uring_reader.h:271-292` at `ba01695c35`).

At tier scale the re-registration took 59,231 ms for 107,363,553,792 B in 280 chunks
(`analysis/dsv41-drive/uring-reg/results.md`, arm R3). The cost comes from the kernel's pin accounting
(`io_buffer_account_pin` → `headpage_already_acct`). That is past the 30 s `fatal_wait`. The watchdog bounds one
request's time in service (`ram_thread.h:168-171`), and the drain runs inside `reader_.read()`, so the process
aborts with "a request stayed in service".

## The fix

Without SQPOLL, the kernel reads an SQE only inside `io_uring_enter`. So the unconsumed positions
`[sqe_tail - n, sqe_tail)` still belong to userspace. `drain()` now:

- zeroes those SQEs;
- rewrites them as `IORING_OP_NOP`;
- submits and retires them on the same ring.

The buffer table, the file table and the ring fd all survive.

If the kernel refuses the NOP submit (hard error, or soft errors for 2 s), the behaviour depends on the read mode:

- **Fixed read mode:** the reader closes and throws, naming the cause: "refused the NOP drain … re-registering the
  fixed buffers would exceed the watchdog's fatal_wait". User decision, 2026-09-29.
- **Other modes:** they keep the plain reset, which has no buffer table to rebuild.

SQPOLL modes are unchanged; their drain never reset.

Kernel support on divix01 (`6.12.0-211.60.1.el10_2`, liburing 2.12): `io_nop`, `io_register_clone_buffers` and
`io_sync_cancel` are in kallsyms, and `IORING_REGISTER_CLONE_BUFFERS = 30` is in the uapi header. The fix needs only
NOP, which every io_uring kernel has.

## Registrations: before and after

| Harness case (`test_expert_stream_ring_reset.py`) | before (`0931fbe42f`) | after (`2c4a598712`) |
|---|---|---|
| normal / fixed / readv_fixed × unconsumed, mixed | `REGISTRATIONS 2` (6 of 6; laptop, and divix01 RED run: 8 failed / 2 passed as expected) | `REGISTRATIONS 1` |
| published: SQEs on the kernel tail, unconsumed (review fix) | not present | `REGISTRATIONS 1` (3 of 3) |
| `test_ring_reset_mid_fan_out[unconsumed]`, reader end-to-end, `register_ms=` lines | 2 | 1 |
| refused NOP drain, normal mode | not present | `REGISTRATIONS 2` (plain reset) |
| refused NOP drain, fixed / readv_fixed | not present | `RAISED … refused the NOP drain …`, reader closed |

In every case the abandoned rows stay unwritten (0xcd fill). The next read on the same ring gets exactly one
completion, its own, with the right bytes.

## Reset time at 1.5 GiB (divix01, after the reboot)

Test: `test/manual/dsv41/test_expert_stream_fixed_buffers_big.py::test_a_ring_reset_keeps_the_big_slab_registered`.
It uses a 1,613,631,488 B slab in 3 chunks and runs under `rowimg-disk.lock`, `numactl --membind=1` and
`taskset -c 0-63`. Each worktree printed its `sglang.__file__`.

| commit | MODE / READ_MODE | drain_ms | register_ms | registrations |
|---|---|---|---|---|
| before `0931fbe42f` | default / fixed | 595.5 | 555.0 | 2 |
| before `0931fbe42f` | default / readv_fixed | 898.6 | 839.4 | 2 |
| before `0931fbe42f` | iopoll / readv_fixed | 780.1 | 777.7 | 2 |
| after `f8307deb21` | default / fixed | 0.011 | 739.3 | 1 |
| after `f8307deb21` | default / readv_fixed | 0.017 | 731.6 | 1 |
| after `f8307deb21` | iopoll / readv_fixed | 0.012 | 565.5 | 1 |

- **Before the fix,** a reset costs one full registration plus the ring teardown: `drain_ms ≈ register_ms + 3-60 ms`.
- **After the fix,** the drain takes 11-17 µs and never registers.

**Extrapolated to 90 GiB:**

- **Before:** measured production was 59.2 s at 100 GiB (uring-reg R3), so linear scaling gives about 53 s at
  90 GiB. That is a lower bound, because the pin accounting is superlinear when THP falls back to 4 KiB. Linear
  scaling from 1.5 GiB gives 31-47 s at 90 GiB (60 × 0.52-0.78 s), which also runs past the 30 s `fatal_wait`.
- **After:** about 10-20 µs. The drain touches only the unconsumed SQEs (at most the queue depth), whatever the
  tier size.

**Earlier runs, under memory pressure, not valid as timings:**

- before: fixed 535.9 / 535.3 ms; readv_fixed 109.3 / 91.6 ms (drain / register);
- after: 0.010-0.013 ms, with `registrations` of 2 before and 1 after, the same as the re-take.

## Mutants (divix01, private worktree `wt-ringreset-mut` at `0a1f682b9e`, removed afterwards)

Each mutant was run against `test_expert_stream_ring_reset.py`. The baseline was 15 passed, and it was green again
after every revert.

| # | mutant in `uring_reader.h` | result |
|---|---|---|
| M1 | no NOP rewrite (the original reads are submitted) | killed, 9 failed ("a discarded read wrote row") |
| M2 | rewrite starts at `tail - n + 1` | killed, 9 failed |
| M3 | no `memset` (stale `IOSQE_FIXED_FILE` / `buf_index`) | **survived on the real kernel**: 6.12 accepts a NOP carrying them. Now killed by the fake-liburing contract (`2c4a598712` asserts `flags == 0`, `buf_index == 0`). The memset stays as a defense. |
| M4 | NOP completions not retired | killed, 9 failed ("a completion other than row 5's surfaced") |
| M5 | NOP flush followed by the old reset | killed, 9 failed (`REGISTRATIONS 2`) |
| M6 | fixed-mode throw disabled (slow reset instead) | killed, 2 failed |
| M7 | rewrite from `sq.sqe_head` (misses published SQEs) | killed, 3 failed (the `published` cases) |
| M8 | `close()` terminates on any drain throw | killed, 1 failed (`refused_close`, readv_fixed) |

## Suites (divix01)

Command:

```
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly
```

`sglang.__file__` was printed under each worktree.

| tree | result |
|---|---|
| master `ba01695c35` (`wt-ringreset-base`) | 1933 passed, 23 skipped, EXIT=0 |
| `0a1f682b9e` | 1947 passed, 2 failed, 23 skipped. The failures were both `test_uring_reader_driver_contract` cases: its fake liburing had no SQ ring to rewrite, and it asserted the old reset. Fixed in `2c4a598712`. |
| head `f8307deb21` (after the reboot) | **1949 passed**, 23 skipped, EXIT=0 |

The delta is +16 passed: the 15 cases of the new `test_expert_stream_ring_reset.py`, plus
`test_a_refused_nop_drain_through_the_reader` in `test_expert_stream_fixed_buffers.py`. There are no new failures
or skips. The master count was taken before 01:15, before the memory pressure.

Targeted run at head `2c4a598712`, CPU, `taskset -c 0-63`: `test_expert_stream_uring_options.py`,
`test_expert_stream_ring_reset.py` and `test_expert_stream_fixed_buffers.py` gave 48 passed, EXIT=0.

## Pre-existing, not caused by this branch

- **`test_big_fixed_slab_real_kernel` with `MODE=iopoll` times out after 300 s at master `ba01695c35` as well**
  (02:22 CDT, after the reboot). The head times out the same way. So the hang is a pre-existing problem between the
  big-slab harness's `settle` loop and IOPOLL, not this branch. The `reset` case under IOPOLL passes at the head
  (`registrations=1`, `drain_ms=0.012`). The IOPOLL NOP drain therefore works on a polled NVMe queue.
- `test_big_fixed_slab_memlock_refused` failed at master before the reboot, when ring creation itself hit ENOMEM
  under `RLIMIT_MEMLOCK=262144`. It **passes after the reboot** at the head in both default modes. The earlier
  failure was environmental.

## Verdict

The branch is ready to merge, as far as this defect is concerned:

- the unit, contract and suite tests are green;
- the re-registration is gone (registrations 2 → 1);
- the reset cost fell from one registration to microseconds;
- 8 mutants were run, and all are killed, M3 through the fake contract.

Two conflicts with `cc/hotpath-zero-overhead` are expected, both small:

- `read_fault.h`: the `ReadFault` field, the word-30 decode and the layout comment;
- `reader_core.h`: `set_fault`'s `SubmitFault` initializer and condition.

This branch also touches `file_reader.h` (`SubmitFault`), `faulty_reader.h` and
`test_expert_stream_uring_options.py` (the fake).

Open follow-up, unrelated to this fix: the IOPOLL `real_kernel` harness hang, which also happens at master.
