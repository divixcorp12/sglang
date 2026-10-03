"""Startup CPU/DMA split calibration's host half (CPU; spec 2026-10-01-cpu-split-calibration).

The host times CPU jobs through the live CPU expert engine and the DMA through its own copy backend (the test backend
here, device -1) into a scratch buffer, and returns the mean ms of every cell. The forward is the instr build's native
fake: the calibration waits inside one FFI call, which holds the GIL, so a ctypes forward would deadlock.
"""

import os

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as ram_miss
from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROW, ROWS, DST_ROWS, HIDDEN = 1, 2, 6, 8
LANES = lease.wire_layout(8).lanes
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
    host.enable_cpu_experts(host.test_forward_address(forward_ns), [0] * (LANES + 1), cores, x_rows, out_rows, threads=2, spin_us=200)
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
    sm_mask = 0b110110  # the layout's small tensors (scale vectors); the trellises stay on the DMA
    _, host, row_bytes, _keep = _host(tmp_path, sm_mask=sm_mask)
    assert host.copy_expert_bytes(ROW) == sum(b for i, b in enumerate(row_bytes) if not sm_mask >> i & 1)


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


@pytest.mark.parametrize("lanes", [8, 16])
def test_the_shape_helpers_follow_the_lane_count(lanes):
    # A real 16-lane calibration needs a 16-lane host (ExpertStreamHost(lanes), Task 6); until then this checks the
    # helpers and that the 16-lane host module builds with the lane-scaled capacities.
    width = lease.wire_layout(lanes).lanes
    assert ram_miss.calibration_shape(lanes) == (width + 2, width + 1)
    assert ram_miss.stage_trace_rows(lanes) == 2 * width
    module = ram_miss._host_module("exl3", None, lanes)
    assert hasattr(module, "expert_stream_calibrate_cpu_split")


def test_the_host_calibration_grid_has_the_shape_helper_size(tmp_path):
    _, host, _, _keep = _host(tmp_path)
    grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host))
    assert tuple(grid.shape) == ram_miss.calibration_shape()


@pytest.mark.parametrize(
    "capacities, lanes, row",
    [({0: 12, 1: 20}, 16, 1), ({0: 16, 1: 20}, 16, 0), ({0: 12, 1: 15}, 16, None), ({0: 7}, 8, None), ({0: 8}, 8, 0)],
)
def test_calibration_runs_on_the_first_row_with_a_slot_per_lane(capacities, lanes, row):
    """Calibration needs one RAM slot per lane; a tier with no such row keeps the configured split."""
    from sglang.srt.layers.moe.cpu_experts.service import calibration_row

    assert calibration_row(capacities, lanes) == row
