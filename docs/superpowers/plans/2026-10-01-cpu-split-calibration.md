# CPU/DMA Split Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the CPU expert cost, the DMA link cost and their concurrent cost on the loaded model once at startup, choose the CPU split from those measurements, print it, and keep it static.

**Architecture:** A C++ host method (`RamTier::calibrate_cpu_split`, core loop in a new header `host/split_calibration.h`) times CPU jobs through the live `CpuExpertEngine` and DMA through a dedicated copy backend into a scratch buffer, filling a `[10, 9]` float64 grid of mean ms. Python (`cpu_experts/policy.py`, `service.py`) turns the grid into `split[n]`, prints it and pushes it once; `exl3_ram_miss.py` runs it right before the copy engine arms, with the RAM thread paused.

**Tech Stack:** C++20 host headers (tvm-ffi exports, JIT-built by `load_jit`), CUDA driver API via the existing `CudaCopyBackend`, Python/PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-cpu-split-calibration-design.md`

## Global Constraints

- Nothing on the decode path gains a clock read, counter or branch: `serve_record`, the copy thread's loop and `CpuExpertEngine::run()` are unchanged.
- Calibration runs once, right before `host.arm_copy_engine()`, with the RAM thread paused; `retune()` is off once it ran.
- `SGLANG_DSV41_CPU_EXPERTS_SPLIT` set: calibration is skipped.
- Repetitions: 10 timed after one discarded warm-up per cell (`SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS`, default 10).
- Measurement timeout: 1 s per run. Any failure logs a warning and keeps the constants' split; it never stops serving.
- Tie rule: within 2% of the best time, the larger `k` wins.
- The Python thread never calls the kernel's `forward`; CPU calibration jobs go through the engine's thread. The DMA never uses the live copy engine. No VRAM expert slot is written.
- Env vars follow `.claude/skills/env-var-conventions` (`EnvField` on `Envs`, verb category, `envs.X.get()`, `envs.X.override()` in tests).
- Code is written on the laptop in `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4` on `master`, committed, pushed to `origin`, and run on divix01 in the private worktree `/data/models/slang/nvfp4-work/wt-cpusplit` (`.claude/rules/divix01-run-protocol.md`). Pushing `master` to `origin` needs the user's go-ahead once at execution start.
- Every CPU job runs under `taskset -c 0-63` with `OMP_NUM_THREADS=8`; GPU work under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`. Read `PIPESTATUS`, never a pipeline's status.
- Never run tests in the production checkout `cc-expert-prediction/dsv41-direct-prod`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF`. Stage files by name. Never amend.

### The divix01 run command

Every "Run on divix01" step below uses this form, with `<TESTS>` replaced:

```bash
git push origin master && ssh divix01 'set -o pipefail; W=/data/models/slang/nvfp4-work/wt-cpusplit; \
  git -C /data/models/slang/sglang fetch -q origin && git -C $W checkout -q --detach origin/master && cd $W && \
  git log -1 --oneline && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest <TESTS> -q -p no:randomly 2>&1 | tail -25; echo "EXIT=${PIPESTATUS[0]}"'
```

## Review Focus

1. A copy table with an SM entry (`sm_mask`): the calibration DMA must skip that entry exactly as production's DMA does, so the per-expert bytes and link times describe what decode copies. Pinned by `test_copy_expert_bytes_leaves_out_sm_entries` (Task 2).
2. A measurement that times out leaves a CPU job still running in the engine; a later calibration, and production jobs, must still complete in order. Pinned by `test_a_timed_out_calibration_reports_and_a_later_one_completes` (Task 2).
3. Calibration called while the RAM thread runs and the caller has not paused it must be refused, not race the service thread on `claim()`. Pinned by `test_calibration_needs_the_tier_owner` (Task 2).
4. Calibration that fails (backend error, out of memory, timeout) must keep the configured split and let the server serve. Pinned by `test_failed_calibration_warns_and_keeps_the_split` (Task 3).
5. Calibration's own CPU jobs add to `cpu_stats()`; the stats baseline must be re-taken so the periodic log and any later retune do not count them. Pinned by `test_calibration_pushes_the_measured_split_once_and_stops_retuning` (Task 3).

---

### Task 1: Split policy from a measured grid

**Files:**
- Modify: `python/sglang/srt/layers/moe/cpu_experts/policy.py` (append after `split_table`)
- Test: `test/registered/unit/kernels/test_cpu_expert_pool.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `split_from_grid(grid: Sequence[Sequence[float]], tie: float = 0.02) -> list[int]`. `grid` is the calibration grid as nested lists, `len(grid) == lanes + 2`, each row `lanes + 1` long: `grid[0][k]` = cpu ms for k lanes, `grid[1][m]` = link ms for m lanes, `grid[1 + n][k]` = concurrent ms for n lanes with k on the CPU (k = 0..n). Returns `split[0..lanes]`.
  - `format_calibration(grid, split, *, row: int, expert_bytes: int, reps: int) -> str`, the printed block.

- [ ] **Step 1: Write the failing tests**

Add to the import block of `test/registered/unit/kernels/test_cpu_expert_pool.py`:

```python
from sglang.srt.layers.moe.cpu_experts.policy import (
    format_calibration,
    k_star,
    parse_core_list,
    split_from_grid,
    split_table,
)
```

Append:

```python
def _grid(both, lanes=8):
    """A calibration grid whose concurrent rows come from both(n, k); cpu and link rows from the n == k and k == 0 cells."""
    grid = [[0.0] * (lanes + 1) for _ in range(lanes + 2)]
    for n in range(1, lanes + 1):
        for k in range(n + 1):
            grid[1 + n][k] = both(n, k)
        grid[0][n] = both(n, n)
        grid[1][n] = both(n, 0)
    return grid


def test_split_from_grid_takes_the_fastest_k_per_n():
    # A CPU twice as fast as the link per expert, both paths in parallel; exact ties go to the larger k.
    grid = _grid(lambda n, k: max(0.5 * k, 1.0 * (n - k)))
    expected = [0] + [min(range(n + 1), key=lambda k: (max(0.5 * k, 1.0 * (n - k)), -k)) for n in range(1, 9)]
    assert split_from_grid(grid) == expected


def test_split_from_grid_breaks_a_near_tie_toward_the_cpu():
    # k = 1 is 1% slower than k = 0 for every n: inside the 2% tie, so the larger k wins; at 3% it does not.
    near = _grid(lambda n, k: 1.0 + 0.01 * k if k <= 1 else 9.0)
    far = _grid(lambda n, k: 1.0 + 0.03 * k if k <= 1 else 9.0)
    assert split_from_grid(near) == [0] + [1] * 8
    assert split_from_grid(far) == [0] * 9


def test_split_from_grid_never_exceeds_n_and_ignores_cells_past_n():
    # Cells k > n are unused (0.0 in the C++ grid); a 0.0 there must not win.
    grid = _grid(lambda n, k: 10.0 - k)  # more CPU is always faster
    assert split_from_grid(grid) == list(range(9))
    assert all(0 <= k <= n for n, k in enumerate(split_from_grid(grid)))


def test_split_from_grid_handles_a_non_monotonic_grid():
    # A contention dip at k = 2 of n = 4.
    grid = _grid(lambda n, k: 1.0 if (n, k) == (4, 2) else 3.0 + k)
    assert split_from_grid(grid)[4] == 2


def test_format_calibration_prints_the_tables_and_the_split():
    grid = _grid(lambda n, k: max(0.5 * k, 1.0 * (n - k)))
    split = split_from_grid(grid)
    block = format_calibration(grid, split, row=3, expert_bytes=12 * 2**20 + 2**19, reps=10)
    lines = block.splitlines()
    assert lines[0] == "CPU experts calibration: row 3, expert 12.5 MiB, 10 reps"
    assert lines[1] == "  cpu  ms k=1..8: 0.50 1.00 1.50 2.00 2.50 3.00 3.50 4.00"
    assert lines[2] == "  link ms m=1..8: 1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00"
    assert lines[3].startswith("  layer ms n=1..8 at chosen k: ")
    assert lines[4] == "  split n=0..8: " + " ".join(str(k) for k in split)
```

- [ ] **Step 2: Commit the tests and run them on divix01 to see them fail**

```bash
git add test/registered/unit/kernels/test_cpu_expert_pool.py
git commit -q -m "test(cpu-experts): split_from_grid and format_calibration" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF"
```

First run only, create the worktree:

```bash
git push origin master && ssh divix01 'git -C /data/models/slang/sglang fetch -q origin && \
  git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-cpusplit origin/master && \
  git -C /data/models/slang/nvfp4-work/wt-cpusplit log -1 --oneline'
```

Run on divix01 with `<TESTS>` = `test/registered/unit/kernels/test_cpu_expert_pool.py`.
Expected: collection error, `ImportError: cannot import name 'format_calibration'`, `EXIT=2`.

- [ ] **Step 3: Implement**

Append to `python/sglang/srt/layers/moe/cpu_experts/policy.py` (add `from typing import Sequence` to its imports):

```python
def split_from_grid(grid: Sequence[Sequence[float]], tie: float = 0.02) -> list[int]:
    """``split[n]`` from the startup calibration's grid: per n, the k with the least measured layer time.

    ``grid[1 + n][k]`` is the ms for n lanes with k on the CPU, measured with both paths running together. A k within
    ``tie`` of the best wins over a smaller one: equal layer time, and the link stays free for the GPU's own misses.
    """
    lanes = len(grid) - 2
    split = [0]
    for n in range(1, lanes + 1):
        times = list(grid[1 + n][: n + 1])
        best = min(times)
        split.append(max(k for k, t in enumerate(times) if t <= best * (1 + tie)))
    return split


def format_calibration(
    grid: Sequence[Sequence[float]], split: Sequence[int], *, row: int, expert_bytes: int, reps: int
) -> str:
    """The calibration's report: the CPU and link tables, each n's layer time at its chosen k, and the split."""
    lanes = len(grid) - 2

    def ms(values) -> str:
        return " ".join(f"{v:.2f}" for v in values)

    layer = [grid[1 + n][split[n]] for n in range(1, lanes + 1)]
    return "\n".join(
        [
            f"CPU experts calibration: row {row}, expert {expert_bytes / 2**20:.1f} MiB, {reps} reps",
            f"  cpu  ms k=1..{lanes}: {ms(grid[0][1:])}",
            f"  link ms m=1..{lanes}: {ms(grid[1][1:])}",
            f"  layer ms n=1..{lanes} at chosen k: {ms(layer)}",
            f"  split n=0..{lanes}: " + " ".join(str(k) for k in split),
        ]
    )
```

- [ ] **Step 4: Commit and run on divix01 to see them pass**

```bash
git add python/sglang/srt/layers/moe/cpu_experts/policy.py
git commit -q -m "cpu-experts: split_from_grid and format_calibration for the startup calibration" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF"
```

Run on divix01 with `<TESTS>` = `test/registered/unit/kernels/test_cpu_expert_pool.py`.
Expected: all passed (the existing tests plus 5 new), `EXIT=0`.

---

### Task 2: C++ calibration, FFI export and Python host wrapper

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/split_calibration.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h` (new `CopyEngine::dma_entries`, after `eligible`, ~line 439)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (include; `copy_expert_bytes`, `calibrate_cpu_split` after `cpu_cores()`, ~line 600)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (two exports after `cpu_stats`, ~line 404; two registration lines in `EXPERT_STREAM_HOST_EXPORTS`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (native test forward, one registration line)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (three `ExpertStreamHost` methods after `cpu_stats`, ~line 1146)
- Create: `test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`

**Interfaces:**
- Consumes: `CpuExpertEngine::claim/submit/done/eligible` (`host/cpu_experts.h`), `CopyBackend`, `HostCopyBackend::release`, `CudaCopyBackend(int device, std::string prefix)` (`host/copy_engine.h`), `now_ns()` (`host/reader_base.h:94`), `RamTier::require_owner`, `caller_mutex_`, `tiers_[row].capacity`.
- Produces (Python, on `ExpertStreamHost`):
  - `copy_expert_bytes(row: int) -> int`: bytes the DMA moves per expert of `row`.
  - `calibrate_cpu_split(row: int, *, device: int, reps: int, scratch: torch.Tensor, timeout_s: float = 1.0) -> torch.Tensor`: float64 `[10, 9]` ms; row 0 `cpu[k]`, row 1 `link[m]`, row `1 + n` `both[n][k]` for k ≤ n, other cells 0. `device` -1 copies with the test backend. Raises `RuntimeError` on any failure.
  - `test_forward_address(ns_per_expert: int) -> int` (instr build only): address of a native fake forward.

Why a native test forward: the calibration waits inside one FFI call, which holds the GIL, so the ctypes fake forward of `test_exl3_ram_miss_cpu_experts.py` (which needs the GIL on the engine thread) would deadlock.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`:

```python
"""Startup CPU/DMA split calibration's host half (CPU; spec 2026-10-01-cpu-split-calibration).

The host times CPU jobs through the live CPU expert engine and the DMA through its own copy backend (the test backend
here, device -1) into a scratch buffer, and returns the mean ms of every cell. The forward is the instr build's native
fake: the calibration waits inside one FFI call, which holds the GIL, so a ctypes forward would deadlock.
"""

import os

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROW, ROWS, DST_ROWS, HIDDEN, LANES = 1, 2, 6, 8, 8
FORWARD_NS = 200_000  # 0.2 ms per expert


def _host(tmp_path, *, capacity=12, sm_mask=0, register=True, forward_ns=FORWARD_NS):
    s = ram_miss_setup(tmp_path, capacity=capacity, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    host = attached_host(s, new_page(pin=False), k=3)
    host.enable_copy_engine(-1, spin_us=200)
    dst = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS, sm_mask=sm_mask)
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 2, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))[:2]
    host.enable_cpu_experts(host.test_forward_address(forward_ns), [0] * 9, cores, x_rows, out_rows, threads=2, spin_us=200)
    if register:
        host.set_cpu_layer(ROW, 7)
    row_bytes = [t[0].numel() * t.element_size() for t in s.slabs[ROW].values()]
    return s, host, row_bytes, (dst, x_rows, out_rows)


def _scratch(host, extra=0):
    return torch.zeros(LANES * host.copy_expert_bytes(ROW) + extra, dtype=torch.uint8)


def test_copy_expert_bytes_is_the_dma_entries_sum(tmp_path):
    _, host, row_bytes, _keep = _host(tmp_path)
    assert host.copy_expert_bytes(ROW) == sum(row_bytes)


def test_copy_expert_bytes_leaves_out_sm_entries(tmp_path):
    # Production's DMA skips an SM entry (the copy wait reads it); calibration must time the same bytes.
    _, host, row_bytes, _keep = _host(tmp_path, sm_mask=1)
    assert host.copy_expert_bytes(ROW) == sum(row_bytes[1:])


def test_calibration_times_every_cell_and_runs_each_cpu_job(tmp_path):
    _, host, _, _keep = _host(tmp_path)
    reps = 2
    jobs_before = host.cpu_stats()["jobs"]
    grid = host.calibrate_cpu_split(ROW, device=-1, reps=reps, scratch=_scratch(host))
    assert grid.dtype == torch.float64 and tuple(grid.shape) == (LANES + 2, LANES + 1)
    for k in range(1, LANES + 1):
        assert grid[0, k] >= k * FORWARD_NS / 1e6, (k, grid[0].tolist())
        assert grid[1, k] > 0
    for n in range(1, LANES + 1):
        assert all(grid[1 + n, k] > 0 for k in range(n + 1)), (n, grid[1 + n].tolist())
        assert all(grid[1 + n, k] == 0 for k in range(n + 1, LANES + 1))
        assert grid[1 + n, n] >= n * FORWARD_NS / 1e6
    assert grid[0, 0] == 0 and grid[1, 0] == 0
    # One warm-up plus reps runs per cell: the CPU pass's 8 cells and the concurrent pass's 36 cells with k > 0.
    assert host.cpu_stats()["jobs"] - jobs_before == (8 + 36) * (reps + 1)


def test_calibration_refuses_a_scratch_smaller_than_eight_experts(tmp_path):
    _, host, _, _keep = _host(tmp_path)
    small = torch.zeros(LANES * host.copy_expert_bytes(ROW) - 1, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="scratch"):
        host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=small)


def test_calibration_refuses_a_row_without_a_cpu_layer(tmp_path):
    _, host, _, _keep = _host(tmp_path, register=False)
    with pytest.raises(RuntimeError, match="no registered CPU layer"):
        host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host))


def test_calibration_refuses_a_row_with_fewer_than_eight_slots(tmp_path):
    _, host, _, _keep = _host(tmp_path, capacity=7)
    with pytest.raises(RuntimeError, match="8 RAM slots"):
        host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host))


def test_a_timed_out_calibration_reports_and_a_later_one_completes(tmp_path):
    # 100 ms per expert against a 50 ms timeout: the first CPU run times out with its job still in the engine.
    _, host, _, _keep = _host(tmp_path, forward_ns=100_000_000)
    with pytest.raises(RuntimeError, match="did not finish"):
        host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host), timeout_s=0.05)
    # The engine runs jobs in order; the stale job finishes and a fresh calibration with room for it completes.
    host.test_forward_address(1_000)
    grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host), timeout_s=2.0)
    assert grid[0, 1] > 0


def test_calibration_needs_the_tier_owner(tmp_path):
    _, host, _, _keep = _host(tmp_path)
    host.start_thread(fatal_wait_s=60.0, spin_us=2000)
    try:
        with pytest.raises(RuntimeError, match="needs the service thread paused"):
            host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host))
        host.pause(5.0)
        try:
            grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host))
            assert grid[0, 8] > 0
        finally:
            host.resume()
    finally:
        host.stop()
```

- [ ] **Step 2: Commit the tests and run on divix01 to see them fail**

```bash
git add test/registered/unit/kernels/test_exl3_cpu_split_calibration.py
git commit -q -m "test(expert-stream): the host's startup split calibration" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF"
```

Run on divix01 with `<TESTS>` = `test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`.
Expected: every test fails with `AttributeError: 'ExpertStreamHost' object has no attribute 'test_forward_address'`, `EXIT=1`.

- [ ] **Step 3: Create `host/split_calibration.h`**

```cpp
// Startup calibration of the CPU expert split (spec docs/superpowers/specs/2026-10-01-cpu-split-calibration-design.md):
// mean ms of k CPU lanes, of m DMA'd experts, and of both started together, on one row. Run by the tier's owner while
// the copy engine is not armed, so no device lane competes. Nothing here runs on the decode path.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "copy_engine.h"
#include "cpu_experts.h"
#include "reader_base.h"

namespace sglang::expert_stream {

constexpr int kCalibLanes = static_cast<int>(wire::kLeaseLanes);
// The grid, float64 row-major [kCalibRows][kCalibCols] in ms: row 0 cpu[k], row 1 link[m], row 1 + n both[n][k] for
// k <= n. Column 0 of rows 0 and 1, and cells k > n, stay 0.
constexpr int kCalibCols = kCalibLanes + 1;
constexpr int kCalibRows = kCalibLanes + 2;

struct CalibrationSetup {
  CpuExpertEngine* cpu = nullptr;
  int64_t row = 0;
  std::vector<CopyEntry> entries;  // the entries production's DMA copies
  CopyBackend* backend = nullptr;
  HostCopyBackend* host_backend = nullptr;  // set: release each mark (the test backend completes only released ones)
  uint64_t scratch = 0;                     // kCalibLanes experts: entry e's rows at its own offset
  int reps = 1;
  int64_t timeout_ns = 0;
};

inline int64_t calibration_expert_bytes(std::span<const CopyEntry> entries) {
  int64_t bytes = 0;
  for (const CopyEntry& entry : entries)
    bytes += entry.bytes;
  return bytes;
}

// k CPU lanes on host slots 0..k-1 and the DMA of m experts from slots k..k+m-1, started together. Returns the ns from
// the start until both are observed done. Throws on a failed copy or past the timeout.
inline int64_t calibration_run(const CalibrationSetup& s, int k, int m) {
  const int64_t start = now_ns();
  uint32_t seq = 0;
  if (k > 0) {
    CpuJob job;
    job.row = s.row;
    job.part = 0;
    job.k = k;
    for (int i = 0; i < k; ++i) {
      job.slots[i] = i;
      job.weights[i] = 1.0f;
    }
    job.seq = seq = s.cpu->claim(1);
    if (!s.cpu->submit(job)) throw std::runtime_error("calibration: the CPU expert ring is full");
  }
  int64_t token = -1;
  if (m > 0) {
    uint64_t base = s.scratch;
    for (const CopyEntry& entry : s.entries) {
      const uint64_t bytes = static_cast<uint64_t>(entry.bytes);
      for (int j = 0; j < m; ++j) {
        if (const int r = s.backend->issue(base + j * bytes, entry.src + static_cast<uint64_t>(k + j) * bytes, entry.bytes))
          throw std::runtime_error("calibration: a DMA issue failed (" + std::to_string(r) + ")");
      }
      base += static_cast<uint64_t>(kCalibLanes) * bytes;
    }
    if (const int r = s.backend->mark(&token)) throw std::runtime_error("calibration: a DMA mark failed (" + std::to_string(r) + ")");
    if (s.host_backend != nullptr) s.host_backend->release(-1);
  }
  bool cpu_done = k == 0;
  bool dma_done = m == 0;
  int64_t end = start;
  while (!cpu_done || !dma_done) {
    if (!cpu_done && s.cpu->done(seq)) {
      cpu_done = true;
      end = std::max(end, now_ns());
    }
    if (!dma_done) {
      const int state = s.backend->query(token);
      if (state == CopyBackend::kDone) {
        dma_done = true;
        end = std::max(end, now_ns());
      } else if (state != CopyBackend::kPending) {
        throw std::runtime_error("calibration: a DMA failed (" + std::to_string(state) + ")");
      }
    }
    if (now_ns() - start > s.timeout_ns)
      throw std::runtime_error("calibration: a measurement did not finish within " +
                               std::to_string(s.timeout_ns / 1'000'000) + " ms");
    _mm_pause();
  }
  return end - start;
}

inline double calibration_mean_ms(const CalibrationSetup& s, int k, int m) {
  calibration_run(s, k, m);  // warm-up: page faults, the engine's wake from its futex sleep
  int64_t sum = 0;
  for (int r = 0; r < s.reps; ++r)
    sum += calibration_run(s, k, m);
  return static_cast<double>(sum) / 1e6 / s.reps;
}

inline void calibrate_split(const CalibrationSetup& s, double* out) {
  std::fill(out, out + kCalibRows * kCalibCols, 0.0);
  for (int k = 1; k <= kCalibLanes; ++k)
    out[k] = calibration_mean_ms(s, k, 0);
  for (int m = 1; m <= kCalibLanes; ++m)
    out[kCalibCols + m] = calibration_mean_ms(s, 0, m);
  for (int n = 1; n <= kCalibLanes; ++n)
    for (int k = 0; k <= n; ++k)
      out[(1 + n) * kCalibCols + k] = calibration_mean_ms(s, k, n - k);
}

}  // namespace sglang::expert_stream
```

- [ ] **Step 4: Add `CopyEngine::dma_entries` in `host/copy_engine.h`, right after `eligible(...)`**

```cpp
  // Row `row`'s entries that its DMA copies: an SM entry is the copy wait's, never the DMA's (issue()). For the
  // startup calibration, which must time the bytes decode moves.
  std::vector<CopyEntry> dma_entries(int64_t row) const {
    if (row < 0 || row >= static_cast<int64_t>(tables_.size()) || !tables_[row].ready.load(std::memory_order_acquire))
      throw std::runtime_error("no copy table for row " + std::to_string(row));
    std::vector<CopyEntry> entries;
    for (const CopyEntry& entry : tables_[row].entries)
      if (!entry.sm) entries.push_back(entry);
    return entries;
  }
```

- [ ] **Step 5: Add the RamTier methods in `host/ram_tier.h`**

Add `#include "split_calibration.h"` after `#include "copy_engine.h"`. After `cpu_cores()` add:

```cpp
  // The bytes the DMA moves per expert of `row` (its copy table less SM entries).
  int64_t copy_expert_bytes(int64_t row) const {
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    return calibration_expert_bytes(copy_engine_->dma_entries(row));
  }

  // Startup calibration (split_calibration.h): fills out, float64 [kCalibRows][kCalibCols] ms. The caller owns the
  // tier: it claims CPU job sequences, the owner's. device -1 copies with the test backend; scratch holds
  // kCalibLanes experts on that device.
  void calibrate_cpu_split(int64_t row, int64_t device, int64_t reps, uint64_t scratch, int64_t scratch_bytes,
                           int64_t timeout_ns, double* out) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("calibrate_cpu_split");
    const std::string prefix = error_prefix<Layout>() + "calibration: ";
    if (cpu_ == nullptr) throw std::runtime_error(prefix + "CPU experts are not enabled");
    if (copy_engine_ == nullptr) throw std::runtime_error(prefix + "the copy engine is not enabled");
    if (row < 0 || row >= layers_) throw std::runtime_error(prefix + "row out of range");
    if (!cpu_->eligible(row)) throw std::runtime_error(prefix + "row " + std::to_string(row) + " has no registered CPU layer");
    if (tiers_[row].capacity < kCalibLanes)
      throw std::runtime_error(prefix + "it needs 8 RAM slots in row " + std::to_string(row) + ", the row has " +
                               std::to_string(tiers_[row].capacity));
    if (reps < 1 || timeout_ns <= 0) throw std::runtime_error(prefix + "reps and the timeout must be positive");
    CalibrationSetup s;
    s.cpu = cpu_.get();
    s.row = row;
    s.entries = copy_engine_->dma_entries(row);
    const int64_t need = kCalibLanes * calibration_expert_bytes(s.entries);
    if (s.entries.empty() || need == 0) throw std::runtime_error(prefix + "row " + std::to_string(row) + " copies no bytes");
    if (scratch == 0 || scratch_bytes < need)
      throw std::runtime_error(prefix + "the scratch holds " + std::to_string(scratch_bytes) + " bytes, it needs " +
                               std::to_string(need));
    std::unique_ptr<CopyBackend> backend;
    if (device < 0) {
      auto host = std::make_unique<HostCopyBackend>();
      s.host_backend = host.get();
      backend = std::move(host);
    } else {
      backend = std::make_unique<CudaCopyBackend>(static_cast<int>(device), prefix + "copy: ");
    }
    if (const std::string error = backend->init(); !error.empty()) throw std::runtime_error(prefix + error);
    // Freed only when idle: after a failure a copy may still be in flight into the scratch.
    struct Shutdown {
      CopyBackend* backend;
      bool idle = false;
      ~Shutdown() { backend->shutdown(idle); }
    } shutdown{backend.get()};
    s.backend = backend.get();
    s.scratch = scratch;
    s.reps = static_cast<int>(reps);
    s.timeout_ns = timeout_ns;
    calibrate_split(s, out);
    shutdown.idle = true;
  }
```

- [ ] **Step 6: Add the exports in `host/ffi_exports.h`, after `cpu_stats`**

```cpp
  // CPU experts' calibration: the bytes the DMA moves per expert of `row`.
  static int64_t copy_expert_bytes(int64_t handle, int64_t row) {
    return find(handle)->copy_expert_bytes(row);
  }

  // CPU experts' startup calibration (split_calibration.h): out float64 [10, 9] ms. The caller owns the tier.
  static void calibrate_cpu_split(
      int64_t handle, int64_t row, int64_t device, int64_t reps, int64_t scratch, int64_t scratch_bytes,
      int64_t timeout_ns, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named(
        "out",
        TensorMatcher({expert_stream::kCalibRows, expert_stream::kCalibCols}).with_dtype<double>().with_device<kDLCPU>(cpu),
        out);
    find(handle)->calibrate_cpu_split(
        row, device, reps, static_cast<uint64_t>(scratch), scratch_bytes, timeout_ns, static_cast<double*>(out.data_ptr()));
  }
```

In `#define EXPERT_STREAM_HOST_EXPORTS(Exports)`, after the `expert_stream_cpu_stats` line, add (each line ends with a backslash):

```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_expert_bytes, Exports::copy_expert_bytes);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_calibrate_cpu_split, Exports::calibrate_cpu_split);       \
```

- [ ] **Step 7: Add the native test forward in `host/ffi_test_exports.h`**

Inside `struct HostTestExports<...>`, next to `copy_engine_release`:

```cpp
  // Test only: a native CPU expert forward, for tests whose waits hold the GIL (a ctypes forward needs it on the CPU
  // expert thread). It spins k * the configured ns, then writes out[0] = k (adds it when accumulating).
  static std::atomic<int64_t>& test_forward_ns() {
    static std::atomic<int64_t> ns{0};
    return ns;
  }
  static int test_forward(int64_t, const void*, const int32_t*, const float*, int32_t k, float* out, int32_t,
                          int32_t accumulate) {
    const int64_t until = expert_stream::now_ns() + k * test_forward_ns().load(std::memory_order_relaxed);
    while (expert_stream::now_ns() < until) _mm_pause();
    out[0] = accumulate != 0 ? out[0] + static_cast<float>(k) : static_cast<float>(k);
    return 0;
  }
  static int64_t test_forward_address(int64_t ns_per_expert) {
    if constexpr (!Build::kFaults) {
      test_only("test_forward_address");
      return 0;
    } else {
      test_forward_ns().store(ns_per_expert, std::memory_order_relaxed);
      return static_cast<int64_t>(reinterpret_cast<intptr_t>(&test_forward));
    }
  }
```

In `#define EXPERT_STREAM_HOST_TEST_EXPORTS_OF(Exports)`, add as the first entry (with a trailing backslash):

```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_forward_address, Exports::test_forward_address);      \
```

If `test_only` is declared `[[noreturn]]`, drop the `return 0;` after it.

- [ ] **Step 8: Add the Python wrappers in `expert_stream_transport.py`, after `cpu_stats`**

```python
    def copy_expert_bytes(self, row: int) -> int:
        """CPU experts' calibration: the bytes the DMA moves per expert of ``row`` (its copy table less SM entries)."""
        return int(self._module.expert_stream_copy_expert_bytes(self.handle, int(row)))

    def calibrate_cpu_split(
        self, row: int, *, device: int, reps: int, scratch: torch.Tensor, timeout_s: float = 1.0
    ) -> torch.Tensor:
        """CPU experts' startup calibration on ``row``: float64 ``[10, 9]`` mean ms; row 0 ``cpu[k]``, row 1
        ``link[m]``, row ``1 + n`` ``both[n][k]`` (k <= n). The caller owns the tier (paused, or no thread). ``device``
        -1 copies with the test backend; ``scratch`` holds 8 experts on that device. Raises RuntimeError on failure."""
        out = torch.zeros((10, 9), dtype=torch.float64)
        self._module.expert_stream_calibrate_cpu_split(
            self.handle, int(row), int(device), int(reps), scratch.data_ptr(),
            scratch.numel() * scratch.element_size(), int(timeout_s * 1e9), out,
        )
        return out

    def test_forward_address(self, ns_per_expert: int) -> int:
        """Test only: a native fake CPU expert forward that spins ``ns_per_expert`` per expert and writes out[0] = k.
        Instrumented build only."""
        _refuse_test_only("test_forward_address", self.variant)
        return int(self._module.expert_stream_test_forward_address(int(ns_per_expert)))
```

- [ ] **Step 9: Commit and run on divix01: the new tests plus the host tests the refactored `serve_record` must keep green**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/split_calibration.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py
git commit -q -m "expert_stream: startup CPU/DMA split calibration on the host" -m "RamTier::calibrate_cpu_split times CPU jobs through the live engine and the DMA through its own copy backend into a scratch buffer, one warm-up plus reps runs per cell; the instr build gains a native fake forward for tests.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF"
```

Run on divix01 with `<TESTS>` = `test/registered/unit/kernels/test_exl3_cpu_split_calibration.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_exl3_ram_miss_copy_engine.py test/registered/unit/kernels/test_expert_stream_ownership.py`.
Expected: all passed, `EXIT=0`. This is also the first build and run of the `serve_record` refactor (`490eb737bd`); a failure in the three existing files is that refactor's, and is fixed before continuing.

---

### Task 3: Env vars, `CpuExpertService.calibrate()` and the arm-time hook

**Files:**
- Modify: `python/sglang/srt/environ.py` (CPU experts block, after `SGLANG_DSV41_CPU_EXPERTS_RETUNE_BATCHES`, ~line 1923)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/service.py` (`__init__`, `retune`, new `calibrate`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`_arm_copy_engine`, ~line 997, new `_calibrate_cpu_split`)
- Test: `test/registered/unit/kernels/test_cpu_expert_pool.py`

**Interfaces:**
- Consumes: `split_from_grid`, `format_calibration` (Task 1); `host.copy_expert_bytes(row)`, `host.calibrate_cpu_split(row, *, device, reps, scratch, timeout_s=1.0)` (Task 2).
- Produces: `CpuExpertService.calibrate(device: int) -> Optional[list[int]]` (the caller owns the tier); `CpuExpertService.calibrated: bool`; env vars `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION` (EnvBool, default True) and `SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS` (EnvInt, default 10).

Ruling carried from the spec: the spec's `SGLANG_DSV41_CPU_EXPERTS_CALIBRATE` is renamed `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION`, because env-var-conventions Rule 4 requires a verb category for a feature flag (as `SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES` does). The reps count is a value, not a flag, so it keeps the family's form.

- [ ] **Step 1: Write the failing tests**

Extend `FakeHost` in `test/registered/unit/kernels/test_cpu_expert_pool.py` (add the attributes in `__init__` and the two methods):

```python
class FakeHost:
    def __init__(self):
        self.enabled, self.layers, self.splits = None, {}, []
        self.stats = {"jobs": 0, "lanes": 0, "forward_ns": 0}
        self.grid = None  # calibrate_cpu_split's answer; an Exception instance is raised instead
        self.calibrations = []

    # ... existing methods unchanged ...

    def copy_expert_bytes(self, row):
        return 1024

    def calibrate_cpu_split(self, row, *, device, reps, scratch, timeout_s=1.0):
        self.calibrations.append((row, device, reps, scratch.numel(), str(scratch.device)))
        if isinstance(self.grid, Exception):
            raise self.grid
        self.stats = {"jobs": 999, "lanes": 999, "forward_ns": 999}  # calibration's own jobs show in the stats
        return torch.tensor(self.grid, dtype=torch.float64)
```

Append:

```python
def _calibrating_service(host, capacity=9):
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertService

    trait = FakeServiceTrait()
    slabs = {row: _fake_slabs(capacity) for row in range(2)}
    svc = CpuExpertService(host, trait, slabs, hidden=8, cores=[4, 5, 6], threads=2, split=[0] * 9, pin=False)
    svc.register(0, 10.0)
    svc.register(1, 10.0)
    return svc


def test_calibration_pushes_the_measured_split_once_and_stops_retuning(capsys):
    from sglang.srt.environ import envs

    host = FakeHost()
    host.grid = _grid(lambda n, k: max(0.5 * k, 1.0 * (n - k)))
    svc = _calibrating_service(host)
    with envs.SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS.override(3):
        split = svc.calibrate(-1)
    assert split == split_from_grid(host.grid)
    assert host.splits == [split] and svc.split == split and svc.calibrated
    assert host.calibrations == [(0, -1, 3, 8 * 1024, "cpu")]
    assert "CPU experts calibration: row 0" in capsys.readouterr().out
    # The stats baseline is re-taken after calibration's own jobs, and retune no longer changes the split.
    assert svc._last_stats == host.stats
    host.stats = {"jobs": 2000, "lanes": 2000, "forward_ns": 2000 * 10_000_000}
    assert svc.retune() is None and host.splits == [split]


def test_calibration_is_skipped_when_off_or_when_the_split_is_fixed():
    from sglang.srt.environ import envs

    host = FakeHost()
    host.grid = _grid(lambda n, k: 1.0)
    svc = _calibrating_service(host)
    with envs.SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION.override(False):
        assert svc.calibrate(-1) is None
    with envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.override("0,1,1,2,3,3,4,5,5"):
        assert svc.calibrate(-1) is None
    assert host.calibrations == [] and host.splits == [] and not svc.calibrated


def test_calibration_needs_a_registered_row_with_eight_slots(caplog):
    host = FakeHost()
    host.grid = _grid(lambda n, k: 1.0)
    svc = _calibrating_service(host, capacity=7)
    with caplog.at_level("WARNING", logger="sglang.srt.layers.moe.cpu_experts.service"):
        assert svc.calibrate(-1) is None
    assert "no registered row has 8 RAM slots" in caplog.text
    assert host.calibrations == [] and svc.split == [0] * 9


def test_failed_calibration_warns_and_keeps_the_split(caplog):
    host = FakeHost()
    host.grid = RuntimeError("calibration: a measurement did not finish within 1000 ms")
    svc = _calibrating_service(host)
    with caplog.at_level("WARNING", logger="sglang.srt.layers.moe.cpu_experts.service"):
        assert svc.calibrate(-1) is None
    assert "did not finish" in caplog.text
    assert host.splits == [] and svc.split == [0] * 9 and not svc.calibrated
```

- [ ] **Step 2: Commit the tests and run on divix01 to see them fail**

```bash
git add test/registered/unit/kernels/test_cpu_expert_pool.py
git commit -q -m "test(cpu-experts): the service's startup calibration" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF"
```

Run on divix01 with `<TESTS>` = `test/registered/unit/kernels/test_cpu_expert_pool.py`.
Expected: the 4 new tests fail (`AttributeError: ... SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS` or `... has no attribute 'calibrate'`), the rest pass, `EXIT=1`.

- [ ] **Step 3: Add the env vars in `python/sglang/srt/environ.py`, after `SGLANG_DSV41_CPU_EXPERTS_RETUNE_BATCHES`**

```python
    # Measure the split once when the copy engine arms (spec 2026-10-01-cpu-split-calibration): the CPU, the link and
    # both together on the loaded model, then keep it. Off: the three costs above, and the retune.
    SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION = EnvBool(True)
    # Timed runs per calibration cell, after one discarded warm-up.
    SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS = EnvInt(10)
```

- [ ] **Step 4: Add `calibrate()` to `CpuExpertService` in `service.py`**

Import: `from sglang.srt.layers.moe.cpu_experts.policy import format_calibration, parse_core_list, split_from_grid, split_table`.

In `__init__`, after `self.split = list(split)`: `self.calibrated = False`.

First line of `retune()`'s body, after the docstring:

```python
        if self.calibrated:
            return None
```

New method after `retune()`:

```python
    def calibrate(self, device: int) -> Optional[list[int]]:
        """Measure the split once on the loaded model and keep it (spec 2026-10-01-cpu-split-calibration).

        The caller owns the tier (the RAM thread paused) and has not armed the copy engine. Returns the new split, or
        None when calibration is off, the split is fixed by SGLANG_DSV41_CPU_EXPERTS_SPLIT, or it could not run (a
        warning says why, and the split stays as it was).
        """
        if envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.get() or not envs.SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION.get():
            return None
        row = next((r for r in sorted(self.handles) if self._capacity(self.slabs_by_row[r]) >= LANES), None)
        if row is None:
            logger.warning(
                "CPU experts calibration skipped: no registered row has %d RAM slots; keeping split %s", LANES, self.split
            )
            return None
        reps = envs.SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS.get()
        try:
            expert_bytes = self.host.copy_expert_bytes(row)
            scratch = torch.empty(
                LANES * expert_bytes, dtype=torch.uint8, device="cpu" if device < 0 else torch.device("cuda", device)
            )
            grid = self.host.calibrate_cpu_split(row, device=device, reps=reps, scratch=scratch).tolist()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            logger.warning("CPU experts calibration failed (%s); keeping split %s", error, self.split)
            return None
        split = split_from_grid(grid)
        self.split, self.calibrated = split, True
        self.host.set_cpu_split(split)
        self._last_stats = self.host.cpu_stats()  # calibration's own jobs are not decode's
        report = format_calibration(grid, split, row=row, expert_bytes=expert_bytes, reps=reps)
        print(report, flush=True)
        logger.info("%s", report)
        logger.debug("CPU experts calibration grid, ms (row 1 + n is both[n][k]): %s", grid)
        return split
```

- [ ] **Step 5: Hook it in `exl3_ram_miss.py`**

Replace `_arm_copy_engine` and add `_calibrate_cpu_split` after it:

```python
    def _arm_copy_engine(self) -> None:
        if not self.copy_engine or self._copy_armed or self.device_side is None:
            return
        if self._copy_decodes >= COPY_ENGINE_ARM_DECODES:
            if self.cpu_experts is not None:
                self._calibrate_cpu_split()
            self.host.arm_copy_engine()
            self._copy_armed = True
            logger.info("exl3 RAM miss copy engine armed after %d decode forwards since capture", self._copy_decodes)

    def _calibrate_cpu_split(self) -> None:
        """Before arming, while the device types no CPU or copy-engine lane: the split from measured costs."""
        self.before_host_use()
        try:
            self.cpu_experts.calibrate(torch.cuda.current_device())
        finally:
            self.after_host_use()
```

- [ ] **Step 6: Commit and run on divix01 to see them pass**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/cpu_experts/service.py \
  python/sglang/srt/layers/moe/exl3_ram_miss.py
git commit -q -m "cpu-experts: calibrate the split once when the copy engine arms" -m "CpuExpertService.calibrate measures on the first registered row with 8 slots, chooses split[n] from the concurrent grid, prints it, pushes it and stops retune; failures warn and keep the split. SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION (default on), SGLANG_DSV41_CPU_EXPERTS_CALIBRATION_REPS (default 10).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF"
```

Run on divix01 with `<TESTS>` = `test/registered/unit/kernels/test_cpu_expert_pool.py test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`.
Expected: all passed, `EXIT=0`.

---

### Task 4: GPU calibration test

**Files:**
- Create: `test/manual/dsv41/test_cpu_split_calibration_cuda.py`

**Interfaces:**
- Consumes: Task 2's `ExpertStreamHost.copy_expert_bytes`, `calibrate_cpu_split`, `test_forward_address`.
- Produces: nothing.

- [ ] **Step 1: Write the test**

```python
"""Startup split calibration with the real CUDA copy backend (GPU; spec 2026-10-01-cpu-split-calibration).

The DMA runs on the calibration's own stream from pinned host rows into a VRAM scratch buffer; the CPU side is the
instr build's native fake forward, so this checks the link measurement, not the kernel.
"""

import os

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup

ROW, ROWS, HIDDEN, LANES = 1, 2, 8, 8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_calibration_measures_a_link_that_grows_with_the_experts(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=12, mirror_weights=(1.0, 1.0), hidden=2048, inter=4096)
    host = attached_host(s, new_page(pin=False), k=3)
    host.enable_copy_engine(-1, spin_us=200)
    pinned = {n: t.pin_memory() for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), t.data_ptr(), t[0].numel() * t.element_size()] for t in pinned.values()], dtype=torch.int64
    )
    host.set_copy_table(ROW, table, 1)
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 2, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))[:2]
    host.enable_cpu_experts(host.test_forward_address(100_000), [0] * 9, cores, x_rows, out_rows, threads=2, spin_us=200)
    host.set_cpu_layer(ROW, 7)
    scratch = torch.empty(LANES * host.copy_expert_bytes(ROW), dtype=torch.uint8, device="cuda")
    grid = host.calibrate_cpu_split(ROW, device=torch.cuda.current_device(), reps=5, scratch=scratch)
    link = grid[1, 1:].tolist()
    print("link ms m=1..8:", " ".join(f"{v:.3f}" for v in link))
    assert all(v > 0 for v in link)
    assert link[7] > link[0], link
    for n in range(1, LANES + 1):
        assert all(grid[1 + n, k] > 0 for k in range(n + 1))
```

- [ ] **Step 2: Commit and run on divix01 under the GPU lock**

```bash
git add test/manual/dsv41/test_cpu_split_calibration_cuda.py
git commit -q -m "test(dsv41): split calibration with the CUDA copy backend" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01USx3s4JXhvxa3xN3YWGRNF"
git push origin master && ssh divix01 'W=/data/models/slang/nvfp4-work/wt-cpusplit; \
  git -C /data/models/slang/sglang fetch -q origin && git -C $W checkout -q --detach origin/master && cd $W && \
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest test/manual/dsv41/test_cpu_split_calibration_cuda.py -q -s -p no:randomly \
  2>&1 | tail -15; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `1 passed`, a printed `link ms m=1..8:` line rising with m, `EXIT=0`.

---

### Task 5: End-to-end check on the real model (run by the user)

No code. Starting the production server takes the partition's cores and the GPU, so the user runs it.

- [ ] **Step 1: Pull `master` into a non-production worktree and launch the server with CPU experts on**, with the usual DSV41 EXL3 recipe and its `SGLANG_DSV41_CPU_EXPERTS=1` / `_CORES` settings, and without `SGLANG_DSV41_CPU_EXPERTS_SPLIT`.

- [ ] **Step 2: Read the block on stdout after the 16th decode forward.** Expected shape:

```
CPU experts calibration: row 0, expert <size> MiB, 10 reps
  cpu  ms k=1..8:  ...
  link ms m=1..8:  ...
  layer ms n=1..8 at chosen k: ...
  split n=0..8: 0 ...
```

Check `link[1]` against the expert size over the Gen3 x16 link (about 12-13 GB/s effective) and `cpu[1]` against the 0.52 ms constant.

- [ ] **Step 3: A/B decode ms/token** on the same prompt set: once as launched (calibrated), once with `SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION=0` (the constants' split). Success per the spec: the calibrated split is no slower.
