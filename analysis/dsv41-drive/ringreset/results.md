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

> **Memory-pressure caveat.** From about 01:15 CDT on 2026-09-29, divix01 had about 150 GiB of pinned host memory
> leaked by another agent's clone-buffers experiment: node 0 had 1.2 GiB free and swap was in use. **Every divix01
> timing number below was taken after 01:15 and is not valid as a timing. Re-take them after the reboot.**
> Pass/fail results were checked the same way, but repeat them too.

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

## Reset time at 1.5 GiB (divix01, under memory pressure: re-take)

These come from `test/manual/dsv41/test_expert_stream_fixed_buffers_big.py::test_a_ring_reset_keeps_the_big_slab_registered`,
with a 1,613,631,488 B slab in 3 chunks, run under `numactl --membind=1`.

| commit | MODE / READ_MODE | drain_ms | register_ms | registrations |
|---|---|---|---|---|
| before `0931fbe42f` | default / fixed | 535.9 | 535.3 | 2 |
| before `0931fbe42f` | default / readv_fixed | 109.3 | 91.6 | 2 |
| after `e5291d8c52` | default / fixed | 0.010 | 774.5 | 1 |
| after `e5291d8c52` | default / readv_fixed | 0.011 | 742.4 | 1 |
| after `e5291d8c52` | iopoll / readv_fixed | 0.013 | 1403.0 | 1 |

Before the fix a reset costs one full registration: `drain_ms ≈ register_ms`. After it, the drain costs about
10 µs whatever the tier size. The `register_ms` spread (92-1403 ms for the same 1.5 GiB) shows the memory pressure
and THP state, which is why these numbers need re-taking.

**Extrapolated to 90 GiB:**

- **Before:** measured production was 59.2 s at 100 GiB. Scaling linearly gives about 53 s at 90 GiB. That is a
  lower bound, because the pin accounting is superlinear when THP falls back to 4 KiB. Scaling this run's
  1.5 GiB figures linearly gives 5.5-46 s, which shows how much the pressured numbers vary.
- **After:** about 10 µs, independent of size. The drain touches only the ≤ queue-depth unconsumed SQEs and never
  the table.

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
| head `2c4a598712` | PENDING (queued behind `cc-gpu.lock`) |

Targeted run at head `2c4a598712`, CPU, `taskset -c 0-63`: `test_expert_stream_uring_options.py`,
`test_expert_stream_ring_reset.py` and `test_expert_stream_fixed_buffers.py` gave 48 passed, EXIT=0.

## Pre-existing, not caused by this branch

- `test_big_fixed_slab_memlock_refused` fails at master `ba01695c35` as well. Under `RLIMIT_MEMLOCK=262144` the
  ring's own creation already fails with ENOMEM ("creating the ring for registered buffers or files"), before any
  registration. So the test's expected "registering fixed buffers" refusal never appears.
- With `MODE=iopoll`, `test_big_fixed_slab_real_kernel` ran into its 300 s subprocess timeout at the branch head.
  The master run of the same case was stopped when the lead held big-slab runs, so it is not yet known whether
  master hangs too. Re-check both after the reboot: this may be a pre-existing IOPOLL issue in the harness's
  `settle` loop, or it may be the memory pressure. The `reset` case, in the same file and mode, passed
  (`registrations=1`, `drain_ms=0.013`).

## To re-take after the reboot

1. The big-slab `reset` case, before (`0931fbe42f`) and after (head): default/fixed, default/readv_fixed and
   iopoll/readv_fixed.
2. The IOPOLL `test_big_fixed_slab_real_kernel` case at master and at head.
3. The kernels suite at head, if the queued run finished under pressure.
