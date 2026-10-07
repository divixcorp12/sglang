"""Startup CPU/DMA split calibration's host half (CPU; spec 2026-10-01-cpu-split-calibration).

The host times CPU jobs through the live CPU expert engine and the DMA through its own copy backend (the test backend
here, device -1) into a scratch buffer, and returns the mean ms of every cell. The kernel is the instr build's fake
CpuExpertKernel (test_kernel_address): the calibration waits inside one FFI call, so the forward must run natively.
"""

import os

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe import expert_stream_transport as ram_miss
from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, fake_cpu_layer, ram_miss_setup

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROW, ROWS, DST_ROWS, HIDDEN = 1, 2, 6, 8
LANES = lease.wire_layout(8).lanes
FORWARD_NS = 200_000  # 0.2 ms per expert


def _host(tmp_path, *, capacity=12, sm_mask=0, register=True, forward_ns=FORWARD_NS, lanes=8, tokens=1):
    s = ram_miss_setup(tmp_path, capacity=capacity, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    host = attached_host(s, new_page(pin=False, wire=lease.wire_layout(lanes)), k=3, lanes=lanes)
    host.enable_copy_engine(-1)
    dst = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS, sm_mask=sm_mask)
    if tokens == 1:
        x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
        out_rows = torch.zeros((ROWS, 2, HIDDEN), dtype=torch.float32)
    else:  # a verify's rows: the tokens' inputs and the token table (cpu_token_table.h)
        x_rows = torch.zeros((ROWS, lease.cpu_row_bytes(HIDDEN, tokens, host.wire.lanes)), dtype=torch.uint8)
        out_rows = torch.zeros((ROWS, 2, tokens, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))[:2]
    host.enable_cpu_experts(host.test_kernel_address(forward_ns), [0] * (host.wire.lanes + 1), cores, x_rows, out_rows, threads=2)
    if register:
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    row_bytes = [t[0].numel() * t.element_size() for t in s.slabs[ROW].values()]
    return s, host, row_bytes, (dst, x_rows, out_rows)


def _scratch(host, extra=0):
    return torch.zeros(host.wire.lanes * host.copy_expert_bytes(ROW) + extra, dtype=torch.uint8)


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
    host.test_kernel_address(1_000)
    grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host), timeout_s=2.0)
    assert grid[0, 1] > 0


def test_calibration_needs_the_tier_owner(tmp_path):
    _, host, _, _keep = _host(tmp_path)
    host.start_thread(fatal_wait_s=60.0)
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
    width = lease.wire_layout(lanes).lanes
    assert ram_miss.calibration_shape(lanes) == (width + 2, width + 1)
    assert ram_miss.stage_trace_rows(lanes) == 2 * width


def test_a_16_lane_host_calibrates_a_16_lane_grid(tmp_path):
    """The host sizes its calibration output by its own wire, so a 16-lane build is not clipped to 8 lanes (the C++
    grid is (lanes + 2) x (lanes + 1) and the output tensor must hold it)."""
    _, host, _, _keep = _host(tmp_path, capacity=20, lanes=16)
    assert host.wire.lanes == 16
    jobs_before = host.cpu_stats()["jobs"]
    grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host))
    assert tuple(grid.shape) == ram_miss.calibration_shape(16) == (18, 17)
    for k in range(1, 17):
        assert grid[0, k] >= k * FORWARD_NS / 1e6 and grid[1, k] > 0, (k, grid[0].tolist())
    for n in range(1, 17):
        assert all(grid[1 + n, k] > 0 for k in range(n + 1)) and not any(grid[1 + n, n + 1 :])
    assert host.cpu_stats()["jobs"] - jobs_before == (16 + 136) * 2


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


def test_a_capped_calibration_times_only_its_lanes(tmp_path):
    """Spill caps calibration at the victim lanes (the split is only indexed below them): a 16-lane host told 4 lanes
    times the 4-lane cells, leaves every other cell 0, and needs 4 experts of scratch."""
    _, host, _, _keep = _host(tmp_path, capacity=20, lanes=16)
    jobs_before = host.cpu_stats()["jobs"]
    scratch = torch.zeros(4 * host.copy_expert_bytes(ROW), dtype=torch.uint8)
    grid = host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=scratch, lanes=4)
    assert tuple(grid.shape) == (18, 17)
    assert all(grid[0, k] > 0 for k in range(1, 5)) and not any(grid[0, 5:])
    for n in range(1, 17):
        row = grid[1 + n]
        assert all(row[k] > 0 for k in range(n + 1)) if n <= 4 else not any(row), n
    assert host.cpu_stats()["jobs"] - jobs_before == (4 + 10) * 2


def test_the_capped_split_keeps_the_configured_entries_above_the_cap():
    from sglang.srt.layers.moe.cpu_experts.policy import capped_split, split_from_grid

    grid = [[0.0] * 17 for _ in range(18)]
    for n in range(1, 5):
        for k in range(n + 1):
            grid[1 + n][k] = 10.0 - k  # more CPU lanes are faster: split[n] = n
    configured = list(range(100, 117))
    assert capped_split(grid, 4, configured) == [0, 1, 2, 3, 4] + configured[5:]
    assert capped_split(grid, 4, configured)[:5] == split_from_grid([r[:5] for r in grid[:6]])


def _calls_per_job(host, jobs_before, run):
    run()
    jobs = host.cpu_stats()["jobs"] - jobs_before
    return jobs, host.test_kernel_calls()


def test_a_one_token_calibration_runs_one_row_jobs(tmp_path):
    """tokens == 1 (BS1, no spill) is today's calibration: one forward row per job, as before."""
    _, host, _, _keep = _host(tmp_path)
    jobs_before = host.cpu_stats()["jobs"]
    jobs, calls = _calls_per_job(
        host, jobs_before, lambda: host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host), lanes=4)
    )
    assert jobs > 0 and len(calls) == jobs


def test_a_verify_calibration_runs_per_token_jobs_of_the_verifys_tokens(tmp_path):
    """The split decides per_token jobs of the verify's token count, so the calibration's CPU jobs are per_token with
    that many rows, each token routing every lane (a synthetic all-routed mask)."""
    tokens = 6
    _, host, _, _keep = _host(tmp_path, capacity=20, lanes=16, tokens=tokens)
    jobs_before = host.cpu_stats()["jobs"]
    scratch = torch.zeros(4 * host.copy_expert_bytes(ROW), dtype=torch.uint8)
    jobs, calls = _calls_per_job(
        host,
        jobs_before,
        lambda: host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=scratch, lanes=4, tokens=tokens),
    )
    assert jobs == (4 + 10) * 2
    assert len(calls) == jobs * tokens
    for first in range(0, len(calls), tokens):
        job = calls[first : first + tokens]
        k = len(job[0]["slots"])
        assert 1 <= k <= 4
        for call in job:  # every token holds all k lanes: none is -1, weight 1
            assert len(call["slots"]) == k and -1 not in call["slots"] and call["weights"] == [1.0] * k


@pytest.mark.parametrize("held, asked", [(6, 0), (6, 7), (1, 2)])
def test_a_calibration_asks_for_no_more_tokens_than_the_rows_hold(tmp_path, held, asked):
    _, host, _, _keep = _host(tmp_path, tokens=held)
    with pytest.raises(RuntimeError, match="tokens"):
        host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=_scratch(host), tokens=asked)
