# Copy-mechanism sweep: divix01 results (RTX 5090, sm_120f, CUDA 13.4 toolkit / torch cu13.0, PCIe Gen3 x16)

Full run 2026-09-27T18:37 at commit 6c75e547b2, in the private worktree `wt-xfer`. `sglang.__file__` was
`/data/models/slang/nvfp4-work/wt-xfer/python/sglang/__init__.py`. The link was Gen3 x16 at start. 30 timed reps per
cell, 167 cells. `mech_bench.py` exited 0, and so did `mech_report.py`: no cell above the ceiling, no unsafe method,
fresh check not blind.

## 1. `mech_report.py gen3/divix01.jsonl` (verbatim)

```
# divix01: theoretical 15.75 GB/s, measured ceiling 13.686 GB/s
serial acquire 780.096 ns, flag RTT p50 1440 ns, min 1408 ns
BDP (ceiling x RTT min, primary) 19270 B; BDP (ceiling x serial acquire) 10676 B

| method | best GB/s | knee (bytes in flight) | share of measured |
|---|---:|---:|---:|
| ce_batch | 13.686 | None | 1.0 |
| ce_batch_small | 2.684 | None | 0.196 |
| ce_each | 13.622 | None | 0.995 |
| ce_each_small | 3.979 | None | 0.291 |
| cw_real | 3.808 | 4096 | 0.278 |
| ldgsts | 12.308 | 32768 | 0.899 |
| sm_cv16 | 12.309 | 16384 | 0.899 |
| sm_cv32 | 12.309 | 32768 | 0.899 |
| sm_small | 6.69 | 32768 | 0.489 |
| sm_weak16 | 12.31 | 16384 | 0.899 |
| sm_weak32 | 12.309 | 32768 | 0.899 |
| tma | 12.302 | 32768 | 0.899 |

named cells: {'cw_pattern': 11.599, 's_pattern': 12.294}

best GB/s by bytes in flight (K = the method's knee)

| method | 4 KiB | 8 KiB | 16 KiB | 32 KiB | 64 KiB | 128 KiB | 256 KiB | 512 KiB | 1024 KiB | 2048 KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cw_real | 3.808 K |  |  |  |  |  |  |  |  |  |
| ldgsts |  | 5.679 | 10.311 | 12.291 K | 12.308 | 12.308 | 12.308 | 12.308 |  |  |
| sm_cv16 | 4.956 | 8.691 | 11.832 K | 12.298 | 12.309 | 12.307 | 12.309 | 12.308 | 12.305 | 12.304 |
| sm_cv32 |  | 8.301 | 11.648 | 12.303 K | 12.308 | 12.309 | 12.308 | 12.307 | 12.306 | 12.305 |
| sm_small |  |  |  | 6.69 K |  |  |  |  |  |  |
| sm_weak16 | 4.497 | 8.368 | 11.714 K | 12.3 | 12.305 | 12.31 | 12.308 | 12.308 | 12.304 | 12.303 |
| sm_weak32 |  | 8.258 | 11.652 | 12.303 K | 12.309 | 12.308 | 12.307 | 12.308 | 12.308 | 12.303 |
| tma |  | 5.873 | 10.474 | 12.279 K | 12.298 | 12.302 | 12.302 | 12.298 | 12.282 |  |
```

The fresh check passed as required. The `.nc` control read stale bytes (`fresh: false`). Every swept method read the
bytes the host rewrote mid-kernel: `sm_cv16`, `sm_cv32`, `sm_weak16`, `sm_weak32`, `ldgsts` and `tma`. The SASS
assertion also held: the control kernel has no `CCTL`, and all 24 swept copy kernels have one. **What the fresh check can and cannot see.** The only negative control is `.nc`, so the check is
shown to detect a stale L1 line and nothing else. `cp.async.cg` (ldgsts) and TMA bypass L1, so their `fresh: true`
would come out the same without the acquire or without `fence.proxy.async.global`. A missing proxy fence has no
control here. Their safety rests on code review of the fence and acquire placement, not on this check. Read latency: a serial
acquire of a pinned host word takes 780.1 ns; a device word takes 108.9 ns. Flag round trip: p50 1440 ns, min 1408 ns
(240 rounds).

## 2. Commands

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement \
  && rm -f analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/mnt/nvme1/tmp-ests/xfer-tmp flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python analysis/dsv41-drive/copy-mechanism/mech_bench.py --repo $PWD \
     --probe analysis/dsv41-drive/copy-mechanism/gen3/probe.json --out analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl \
     2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}" \
  && python3 analysis/dsv41-drive/copy-mechanism/mech_report.py analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl; echo REPORT_EXIT=$?'
# probe (gen3/probe.json), run at 94f3ff5863 in wt-xfer-a under the same lock, TMPDIR and PYTHONPATH:
python analysis/dsv41-drive/copy-mechanism/probe.py --repo $PWD --out analysis/dsv41-drive/copy-mechanism/gen3/probe.json
```

`TMPDIR` is on NVMe because divix01's root volume was full on 2026-09-27. A first probe run without it failed every
build (nvcc could not write `/tmp`); that run was killed and its output discarded.

## 3. Probe (`gen3/probe.json`)

A probe is ok when it built, ran, and copied the right bytes.

| probe | what | ok | SASS load opcodes |
|---|---|---|---|
| `v8_weak` | 32-byte weak `ld.global.v8.u32` of host memory | yes | `LDG.E.ENL2.256` |
| `v8_cv` | 32-byte `ld.global.cv.v8.u32` of host memory | yes | `LDG.E.ENL2.256.STRONG.SYS` |
| `ldgsts` | `cp.async.cg` host -> shared | yes | `LDGSTS.E.BYPASS.128`, `LDGDEPBAR` |
| `bulk` | `cp.async.bulk` (TMA) host -> shared, after `fence.proxy.async.global` | yes | `UBLKCP.S.G` |
| `batch` | `cudaMemcpyBatchAsync`, 4 x 1 KiB H2D | yes | (host API) |

Every mechanism exists on sm_120 and works from pinned host memory. `batch` must be issued on a non-default stream:
on the legacy NULL stream it returns `invalid argument`, as the CUDA 13.4 header documents.

Two harness defects were found and fixed before the run:

- **The `.nc` control was blind.** The fresh check's midpoint polled with `ld.acquire.sys`, which on sm_120 compiles
  to `LDG.E.STRONG.SYS` + `CCTL.IVALL`. The CCTL invalidates the whole SM's L1, so the control's `.nc` second pass read
  fresh bytes. The harness's own acquire was hiding the staleness it was meant to show. The control now polls with
  `ld.relaxed.sys` (no CCTL), and the sweep refuses to run unless the SASS shows that split (`README.md`).
- **The batch probe ran on torch's default stream.** Its first run reported `invalid argument`, which is the
  legacy-NULL-stream refusal, not a missing mechanism.

## 4. Part A rows of the Decision table (divix01, Gen3)

Every row is decided per host. A "no" on Gen3 says nothing about Gen5. The rows that also need Part C say so. Part C
is gated on the expert-stream-native-sync merge and has not run.

- **Deeper unroll in S (U=1 -> 4).** Condition: `sm_cv16` G8 U4 >= 1.05 x `s_pattern`. Measured 12.306 vs 12.294
  (1.001x < 1.05) -> **no (on Gen3)**, whatever Part C's copy share turns out to be. G8 U1..U16 spans only
  12.294-12.308, and `sm_weak16` G8 matches it.
- **cp.async ring or TMA in S.** Condition: best `ldgsts`/`tma` cell with <= 32 KiB per block in flight >= 1.10 x the
  best `sm_cv*` cell. Measured: best `ldgsts` G8 stages 2 = 12.308, against best `sm_cv*` 12.309 (0.9999x < 1.10)
  -> **no (on Gen3)**. Both passed the fresh check. No bulk path beat 12.5 GB/s: TMA's best cell is 12.302. Every SM
  path stops at about 12.31 GB/s, 0.90 of the copy engine's 13.69.
- **Wider copy-wait SM reads.** `cw_pattern` is contiguous and 4-deep, which is **not CW's real shape**. It ran at
  11.599 GB/s, below 0.90 x the measured ceiling (12.317), so the A half of the condition is met, but that decides
  nothing on Gen3. Every SM cell, including G32 U16 with 2 MiB in flight, is below 0.90 x the measured ceiling,
  because the measured ceiling is the copy engine's 13.686 while every SM path plateaus at 12.31 (0.899). Against the
  SM plateau, `cw_pattern` is 11.599 / 12.309 = 0.942, which does not meet the condition. The threshold should compare
  against the SM plateau (a note for the plan owner). What the cell does show is
  that a single block with 16 KiB in flight is bound by latency, not by the link. 16384 B / 1408 ns (the minimum
  round trip) = 11.64 GB/s, the same as the measured 11.6. Every 16 KiB cell lands in 10.3-11.8 GB/s whatever its
  grid or width. The streaming kernel (S, 32 KiB in flight) is bound by the link on Gen3 instead: 32 KiB cells run at
  12.27-12.30 GB/s.

  CW's real shape is the `cw_real` cell: the production `copy_wait_read` over CW's four small tensors per lane. That
  is 20,480 B through one 4-deep 16 KiB pass, then 9,216, 10,240 and 4,608 B through the 16-byte fallback, at about
  4 KiB in flight. Per lane: 1 lane 22.1 us (2.02 GB/s), 2 lanes 16.3 us (2.73), 4 lanes 13.3 us (3.36), 8 lanes
  11.7 us (3.81).

  Those reads are hidden. DSV41_REFERENCE.md §27.15 (node-mode trace `sm-small-B-node-20260926-034122`) records CW
  spin at 54.6 ms, spent waiting on CopyDone, and CW ending 4.96 us (p50 per layer) after the last copy lands. The C
  half (`cw_wait_ns` > 0 on >= 10% of copy-engine requests) needs Part C. -> **Not decided on the measured
  conditions (C pending). No CW change is recommended.**
- **`cudaMemcpyBatchAsync` in the copy thread.** Condition: `ce_batch_small` >= 1.5 x `ce_each_small` GB/s, or host ns
  per call <= 0.7 x, at 4 lanes. Measured 2.354 vs 3.218 GB/s (0.73x) and host 57,083 vs 40,660 ns (1.40x) -> **no
  (on Gen3)**, whatever Part C finds. Caveat: the small-segment copy-engine GB/s mostly measures host enqueue
  time, because `e0` is recorded before the API loop (4 lanes: host 40.7 us vs event 55 us for `ce_each_small`, 57.1
  vs 76 us for `ce_batch_small`). The two measures are therefore not independent, and the verdict rests on the
  host-time arm (1.40x). The batch call is slower at every lane count: 1 lane 1.33 vs
  1.45 GB/s; 8 lanes 2.68 vs 3.98 GB/s, host 101.7 vs 74.8 us. On the large segments batch matches per-segment
  copies on the device (13.69 vs 13.62 GB/s at 4 rows) but costs 1.70x the host time (85.6 vs 50.3 us).
- **Fewer, larger pieces / more, smaller pieces.** Condition: Part C only -> **pending (Part C gated)**.
- **PDL on the chain** (Part B, `../chain-pdl/results.md`). The skeleton bound is 1.48 us/layer (mode 2, work_ns 2000),
  below the 2 us gate -> **no (bound below threshold)**.

### Did the Gen3 expectation hold?

- **Every SM method at 12.1-12.4 GB/s:** held. `sm_cv16`, `sm_cv32`, `sm_weak16`, `sm_weak32`, `ldgsts` and `tma`
  all plateau at 12.30-12.31.
- **Copy engine at 13.5-13.8:** held. `ce_each` 13.54-13.62, `ce_batch` 13.58-13.69.
- **Knees near the bandwidth-delay product of latency x ceiling, about 8.7 KiB:** did not hold. The 16 B methods reach
  95% of their plateau at 16 KiB in flight; the 32 B, `ldgsts` and `tma` methods only at 32 KiB. The cells that break
  the expectation are every 8 KiB one, at 5.7-8.7 GB/s. The serial acquire (780 ns) understates the latency a loaded
  copy sees. `in_flight / GB/s` gives the effective time per window: 0.83 us at 4 KiB, 0.94 us at 8 KiB, 1.41 us at
  16 KiB (= the 1408 ns round trip), and 2.66 us at 32 KiB, where the link rather than latency is the limit. The
  primary BDP is therefore the ceiling x the minimum round trip: 19,270 B, not 10,676 B. The Gen5 argument at the top
  of the plan used the 711 ns figure, so re-derive its BDP from that host's measured round trip.
- **A bulk path above 12.5 GB/s:** none. TMA's best is 12.302 and `ldgsts`'s 12.308.

### Weak loads after the acquire vs `.cv`

`sm_weak*` (plain `ld.global` ordered after the acquire) is **contract-valid**. `.nc` is **not contract-valid,
informational only**.

Weak does not beat `.cv`. Best 12.310 vs 12.309 GB/s; the same knees (16 KiB at 16 B, 32 KiB at 32 B); slightly
lower at small in-flight sizes (4 KiB: 4.50 vs 4.96 GB/s).

### Small segments (the four small tensors, 44,544 B per lane)

The copy-engine columns below are bound by host enqueue time (see the batch row above).

| lanes | `ce_each_small` GB/s (host us) | `ce_batch_small` GB/s (host us) | `sm_small` GB/s (G8 U1) | `cw_real` GB/s (us per lane) |
|---:|---:|---:|---:|---:|
| 1 | 1.448 (16.3) | 1.329 (18.3) | 2.111 | 2.019 (22.1) |
| 2 | | | | 2.731 (16.3) |
| 4 | 3.218 (40.7) | 2.354 (57.1) | 5.364 | 3.360 (13.3) |
| 8 | 3.979 (74.8) | 2.684 (101.7) | 6.690 | 3.808 (11.7) |
