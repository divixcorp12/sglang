# Wake-latency probe: spinning copy_wait vs a stream wait-value node

**Verdict: CW should stay a spinning kernel.** The rule was to move CW only if B beats A by at least 1 µs at the
median and is no worse at p99. That never holds at any park length tested. B is slower at the median everywhere
except 200 µs, where it is 0.38 µs faster: under the threshold. Short and long parks are clearly worse for B:
+1.7 µs at 5 µs, +5.3 µs at 20 µs, +7.7 µs at 1 ms. B' (`CU_STREAM_WAIT_VALUE_FLUSH`) is unsupported on this card
(`CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES=0`).

## Setup

- Probe: `wake_probe.cu` and `run_wake_probe.sh` in this directory, branch `probe/stream-wait-wake` at `b724cb9040`
  (first run at `d35d4f808f`). Run from the divix01 worktree `/data/models/slang/nvfp4-work/wt-wake-probe`.
- RTX 5090 (sm_120), driver 615.71.09 (driver API 13040), nvcc 13.4 (`cuda_13.4.r13.4/compiler.38501229_0`).
  `%globaltimer` steps at 32 ns (62,500 distinct values in 2 ms).
- Each variant is a captured graph, **K0 → waiter → K2**, all kernels 1 block × 32 threads with thread 0 working:
  - K0 clears `release=0`, records T0 and stores `armed=gen` (st.release.sys) to pinned host memory.
  - **A** copies CW's loop shape (`row_copy_kernels.cuh` on master, `lease_copy_wait`). Each iteration does a
    `%globaltimer` deadline check, `ld.acquire.sys` of two abort words, `__nanosleep(256)`, then `ld.acquire.sys`
    of the release word.
  - **B** is `cuStreamWaitValue32(release, 1, EQ)`. Plain capture works on CUDA 13.4: the explicit mem-op node
    helper from the sleep-free branch was not needed. The graph has 3 nodes.
  - **N** is K0 → K2 with no waiter.
- An echo thread pinned to CPU 40 spins on `armed`, waits the configured **echo delay D**, and stores `release=1`.
  The main thread is on CPU 42. Both CPUs are in 32-63 and on the GPU's NUMA node 0 (the GPU's CPU affinity is
  0-17 and 36-53).
- Latency = T2 − T0 on the GPU clock. The host part is identical for every variant.
- Replays: 20,000 measured per variant plus 2,000 warm-up, interleaved one per variant per round, with the order
  rotated each round. The main thread waits for each replay's echo before the next launch. Every A and B replay saw
  `release==1` in K2, and A had 0 timeouts.
- Command (in the worktree, after the foreign-process gate: comm/exe names plus exact argv elements; production
  port 7867 was down and the GPU idle):
  ```bash
  flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
    bash analysis/dsv41-drive/sleep-free-wait/run_wake_probe.sh /mnt/nvme1/sf-ab/wake-<ts> 20000 2000 <delays ns...>
  ```
  Logs and per-replay CSVs are in `divix01:/mnt/nvme1/sf-ab/wake-20260928-025937` (D=0 on `d35d4f808f`),
  `wake-20260928-030029` (D = 0, 20, 200 µs) and `wake-20260928-030109` (D = 5, 10, 50, 100, 1000 µs).

**Why the echo delay.** At D=0 the host answers in ~0.2 µs, before the spin kernel has even launched. Only 47 of
20,000 A replays ever polled, so D=0 measures the launch chain, not a wake. CW waits for a DMA the service
completes much later (the service's own counters give a mean `copy_latency` of ~2.2 ms per copy job in the last
end-to-end arm), so the parked cases (D ≥ 5 µs) are the ones that matter.

## Results (µs, T2 − T0; N = 0.896 p50 / 0.928 p99 at every D)

| Echo delay D | A p50 | A p90 | A p99 | A max | B p50 | B p90 | B p99 | B max | B − A p50 | B − A p99 | Paired B−A p50 (p10 / p90) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.240 | 2.240 | 2.240 | 45.8 | 2.912 | 4.352 | 4.608 | 18.1 | +0.67 | +2.37 | +0.70 (+0.32 / +2.11) |
| 5 µs | 7.296 | 8.320 | 9.536 | 42.9 | 9.024 | 15.648 | 16.064 | 19.8 | +1.73 | +6.53 | +1.76 (+0.80 / +8.38) |
| 10 µs | 12.608 | 13.376 | 14.464 | 23.4 | 15.744 | 17.216 | 17.344 | 37.9 | +3.14 | +2.88 | +3.14 (+2.46 / +4.80) |
| 20 µs | 22.688 | 23.200 | 24.256 | 53.8 | 27.968 | 29.472 | 29.600 | 41.7 | +5.28 | +5.34 | +5.50 (+4.74 / +6.88) |
| 50 µs | 53.344 | 53.472 | 53.856 | 71.2 | 54.080 | 54.528 | 54.880 | 66.4 | +0.74 | +1.02 | +0.74 (+0.45 / +1.25) |
| 100 µs | 102.304 | 104.448 | 104.544 | 125.2 | 103.488 | 103.968 | 104.576 | 115.6 | +1.18 | +0.03 | +0.90 (−1.22 / +1.70) |
| 200 µs | 204.320 | 204.480 | 204.544 | 235.1 | 203.936 | 204.128 | 204.288 | 225.9 | −0.38 | −0.26 | −0.38 (−1.09 / +1.31) |
| 1 ms | 1003.040 | 1003.232 | 1003.616 | 1010.8 | 1010.720 | 1010.944 | 1011.040 | 1014.1 | +7.68 | +7.42 | +7.58 (−0.16 / +8.03) |

The D=0 row is from the second run. The first run at `d35d4f808f` agrees within 0.04 µs at p50 and p99.

Reading it:
- **A wakes in a steady ~2.3–4.3 µs past D**, with a tight spread (p50 to p99 usually under 1.5 µs). That is
  consistent with its ~2.5 µs loop plus the K2 launch.
- **B's excess over D is not monotonic in D**: ~4 µs at 5 µs, ~8 µs at 20 µs, ~4 µs at 50–200 µs, then ~10.7 µs
  at 1 ms. That fits a front-end poll interval that changes with wait length, not a wake on write. The probe did
  not examine the internal mechanism. At short parks B's p90 to p99 also spreads out (5 µs: p50 9.0, p99 16.1).
- **No-wait floor N** (the K0 → K2 launch gap): 0.896 µs p50 and 0.928 µs p99 at every D. A's extra over N at D=0
  is +1.31 µs, and B's is +2.02 µs at p50 and +3.68 µs at p99.
- **Host turnaround** (a pure CPU 42 ↔ CPU 40 ping-pong on the same pinned words): RTT p50 211–236 ns and p99
  226–247 ns, so about 0.11 µs one way. Each run saw one multi-ms max from a host preemption. This part is shared by
  all variants.

## Spinning cost (A)

The poll loop runs at 2.4–2.9 µs per iteration (2.56 µs at 1 ms). Each iteration makes 3 `ld.acquire.sys` reads
of pinned host memory, which gives **1.0–1.25 M PCIe reads/s** while CW waits, of which ~0.4 M/s are polls of the
release word. That is about a quarter of the ~4 M/s estimate: each system-scope load is a serialized PCIe round
trip of roughly 0.75 µs, which dominates the 256 ns nanosleep. Any bandwidth effect was not measured. A also holds
one SM thread block resident for the whole wait. B holds none.

## Scope and caveats

- Each replay starts from an idle stream (the main thread synchronizes and waits for the echo), and there is no
  concurrent GPU work or PCIe traffic. Under load, both A's PCIe round trips and B's front-end polling may behave
  differently.
- Only the release condition was tested. Real CW also exits on deadline, fatal and shutdown. B has no timeout, so a
  producer must complete every request (the constraint `stream_wait.cuh` documents).
- One GPU, one driver, one run per delay set.
