# CPU/DMA split calibration at startup

## Goal

Choose the CPU expert split (`split[n]`: how many of a layer's `n` eligible lanes the CPU computes, the rest going
over PCIe) from costs measured on the loaded model and this machine, instead of the fixed constants
`SGLANG_DSV41_CPU_EXPERTS_CPU_MS=0.52`, `_LINK_MS=1.0`, `_HANDOFF_MS=0.02`. Measure once at startup, print the result,
and use that split statically for the life of the server.

Success: on divix01, the calibrated split's decode ms/token is no worse than the constants' split, and the printed
tables are plausible against the expert size and the Gen3 x16 link.

## Non-goals

- No measurement on the decode path. Nothing in `serve_record`, the copy thread or the CPU engine's job loop gains a
  clock read, counter or branch.
- No re-calibration while serving. `retune()` is off when calibration ran.
- No separate model for CPU misses vs GPU misses. Both pay the same NVMe read first; the split uses the hit
  measurements for them (see "Known limits").

## When it runs

Once, at the moment the RAM-miss service arms the copy engine (`_arm_copy_engine`, `exl3_ram_miss.py:997`), just
before `host.arm_copy_engine()`. By then:

- warm-up has run, so the CPU rows are registered (`CpuExpertService.register` is lazy: it needs the layers'
  activation limit) and the kernel's cores are set;
- the copy tables are set;
- the copy engine is not armed yet, so the device types no CPU or copy-engine lanes (`host_lanes` requires
  `copy_armed`): the CPU engine and the copy thread are idle, and calibration competes with nothing.

It runs on the Python thread at a batch boundary, with the RAM thread paused (`pause()`/`resume()`), so the caller
owns the tier and may submit CPU jobs. Expected cost: about 2-3 s, once.

## What it measures

All three passes use one calibration row: the first registered CPU row whose RAM tier has at least 8 slots, and its
host slots 0..7. Slot contents do not matter for timing; the bytes are real pinned memory of the real size and
format. Each measurement is repeated 10 times after one discarded warm-up, and the mean is used.

1. **CPU pass: `cpu[k]`, k = 1..8.** A `CpuJob` with `k` lanes (slots 0..k-1, weight 1) submitted to the live
   `CpuExpertEngine` and waited for with `done(seq)`. Timed from submit to observed done, so queueing and wake-up are
   included. Output goes to the row's part-0 output buffer, which nothing reads before the copy engine is armed.
2. **Link pass: `link[m]`, m = 1..8.** The DMA of `m` experts' bytes from the row's pinned host slots into a scratch
   VRAM buffer, through a dedicated `CudaCopyBackend` (its own stream and completion word): the same
   `cuMemcpyAsync` per copy-table entry that the copy engine issues, then `mark` and poll `query`. Timed from the
   first issue to observed done.
3. **Concurrent pass: `both[n][k]`, n = 1..8, k = 0..n.** The CPU job of `k` lanes (slots 0..k-1) and the DMA of the
   other `n - k` experts (slots k..n-1) started together, timed until both are observed done. This is what a decode
   layer waits for, including the contention between the CPU kernel and the DMA for host memory bandwidth.
   `both[n][0]` is `link[n]` measured again; `both[n][n]` is `cpu[n]` measured again.

Pass 3 alone decides the split; passes 1 and 2 are printed to explain it.

### Isolation from production state

- The CPU jobs go through the engine's own thread. The Python thread never calls the kernel's `forward`: libgomp
  keeps one worker pool per calling thread, and with `GOMP_SPINCOUNT=INFINITE` a second pool would spin forever on
  the CPU-expert cores.
- The DMA uses its own backend instance and stream, never the copy engine's: copy-engine jobs publish CopyDone to the
  device.
- The scratch buffer (8 experts' bytes) is allocated for the calibration and freed after it. No VRAM expert slot is
  written.
- CPU job sequences come from the engine's normal `claim()`, so `done()` stays monotonic for production jobs.

## Choosing the split

```
split[0] = 0
split[n] = argmin over k in 0..n of  both[n][k]      (n = 1..8)
```

A tie within 2% goes to the larger `k`: equal layer time, and the link stays free for the GPU's own misses. The
`handoff` constant is no longer used when calibration ran; its cost is inside `both`. `split[n] <= n` holds by
construction.

The table is pushed once with `host.set_cpu_split(split)` and stored as `CpuExpertService.split`.

## Output

One block, to stdout and to the log at INFO:

```
CPU experts calibration: row 0, expert 12.3 MiB, 10 reps
  cpu  ms k=1..8:  0.51 0.88 1.27 ...
  link ms m=1..8:  0.98 1.93 2.89 ...
  layer ms n=1..8 at chosen k: 0.53 1.01 1.30 ...
  split n=0..8: 0 1 1 2 3 3 4 5 5
```

The full `both[n][k]` grid goes to the log at DEBUG.

## Configuration

- `SGLANG_DSV41_CPU_EXPERTS_SPLIT` (existing): an explicit table still wins, and calibration is skipped.
- `SGLANG_DSV41_CPU_EXPERTS_CALIBRATE` (new, bool, default on): off restores today's behavior, the constants model
  plus `retune()`.
- `SGLANG_DSV41_CPU_EXPERTS_CALIBRATE_REPS` (new, int, default 10).
- The constants (`_CPU_MS`, `_LINK_MS`, `_HANDOFF_MS`) remain the initial split until calibration replaces it, and
  the whole split when calibration is off.

New env vars follow `.claude/skills/env-var-conventions`.

## Failure handling

Calibration never stops the server from serving. If it cannot run (no registered row with 8 slots, scratch
allocation fails, the backend fails to initialise), it logs a warning naming the reason and keeps the constants'
split. A CPU job or DMA that does not finish within 1 s is a failure of the same kind. Failures inside the CPU
engine itself keep their existing fail-stop.

## Components

- **C++ (`host/`):** one host method, run by the caller that paused the tier: `calibrate_cpu_split(row, reps,
  scratch_ptr, scratch_bytes)` returning the three tables as one float tensor. It owns the dedicated
  `CudaCopyBackend`, builds `CpuJob`s, and reads the row's copy table for source addresses and entry sizes. Exported
  through `ffi_exports.h` next to `set_cpu_split`.
- **Python (`cpu_experts/`):** `policy.py` gains `split_from_grid(both, tie=0.02)`; `service.py` gains
  `calibrate()`, which allocates the scratch buffer, pauses, calls the host method, resumes, chooses the split,
  prints it, pushes it and disables `retune()`; `exl3_ram_miss.py` calls it from `_arm_copy_engine` before arming.

## Testing

- Python unit tests for `split_from_grid`: argmin, the 2% tie rule, `split[n] <= n`, monotonic and non-monotonic
  grids.
- A C++/FFI test with the host copy backend (no GPU): calibration returns positive times of the right shape, the
  engine's `done()` sequence keeps working for jobs after calibration, and the failure paths report instead of
  throwing.
- A GPU test on divix01 (`test/manual/dsv41`): calibration on the real model completes, `link[m]` grows with `m`,
  and the split is valid.
- An A/B of decode ms/token on divix01: the calibrated split against the constants' split, same prompt set.

## Known limits

- One calibration row stands for all layers. Every streamed layer has the same expert size and format, so this
  should hold; per-layer calibration can come later if layer times differ.
- Startup conditions (idle box, no other GPU traffic) may differ from serving. If the static split proves wrong in
  practice, the next step is the live measurement design discussed before this one.
- CPU misses vs GPU misses are decided with hit measurements.
