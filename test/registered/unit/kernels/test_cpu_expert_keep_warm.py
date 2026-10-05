"""CPU expert keep-warm, the engine half (CPU).

For ``keep_warm_us`` after its last job, an idle CPU expert thread hands its workers to the format's keep-warm
function instead of spinning on ``pause``, so the next job finds the cores at the AVX-512 license rather than paying
the ramp back (about 50 us per call on SKX after a 1 ms gap). The forward and the keep-warm are the instr build's
fake kernel; its keep-warm counts its calls (from 0 at each test_kernel_address) and spins until its word moves or
its deadline passes. Jobs come from the startup calibration, which submits them through the live engine.
"""

import os
import time

import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, fake_cpu_layer, ram_miss_setup

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROW, ROWS, DST_ROWS, HIDDEN, LANES = 1, 2, 6, 8, 8
FORWARD_NS = 200_000  # 0.2 ms per expert


def _host(tmp_path, request, keep_warm_us):
    s = ram_miss_setup(tmp_path, capacity=12, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    host = attached_host(s, new_page(pin=False, wire=wire_layout(8)), k=3)
    host.enable_copy_engine(-1, spin_us=200)
    dst = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS)
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 2, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))[:2]
    host.enable_cpu_experts(
        host.test_kernel_address(FORWARD_NS),
        [0] * (lease.wire_layout(8).lanes + 1),
        cores,
        x_rows,
        out_rows,
        threads=2,
        spin_us=200,
        keep_warm_us=keep_warm_us,
    )
    host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    request.addfinalizer(host.stop)  # the fake's call count is per process: no engine may outlive its test
    return s, host, (dst, x_rows, out_rows)


def _run_jobs(host):
    """Run the calibration's CPU jobs (a warm-up and one timed run per cell); returns its grid in ms."""
    scratch = torch.zeros(LANES * host.copy_expert_bytes(ROW), dtype=torch.uint8)
    return host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=scratch)


def test_keep_warm_waits_for_the_first_job(tmp_path, request):
    _, host, _keep = _host(tmp_path, request, keep_warm_us=500_000)
    time.sleep(0.05)
    assert host.test_keep_warm_calls() == 0


def test_an_idle_engine_keeps_warm_after_a_job(tmp_path, request):
    _, host, _keep = _host(tmp_path, request, keep_warm_us=500_000)
    _run_jobs(host)
    time.sleep(0.02)
    assert host.test_keep_warm_calls() >= 1


def test_a_job_ends_the_keep_warm_at_once(tmp_path, request):
    # A 2 s window: a keep-warm that ran on to its deadline would hold each job for up to 2 s.
    _, host, _keep = _host(tmp_path, request, keep_warm_us=2_000_000)
    _run_jobs(host)
    time.sleep(0.02)
    calls = host.test_keep_warm_calls()
    assert calls >= 1
    grid = _run_jobs(host)
    assert host.test_keep_warm_calls() > calls
    for k in range(1, LANES + 1):
        assert grid[0, k] < k * FORWARD_NS / 1e6 + 5, (k, grid[0].tolist())


def test_keep_warm_stops_after_its_window(tmp_path, request):
    _, host, _keep = _host(tmp_path, request, keep_warm_us=20_000)
    _run_jobs(host)
    time.sleep(0.1)
    calls = host.test_keep_warm_calls()
    assert calls >= 1
    time.sleep(0.1)
    assert host.test_keep_warm_calls() == calls


def test_stop_ends_a_running_keep_warm(tmp_path, request):
    _, host, _keep = _host(tmp_path, request, keep_warm_us=60_000_000)
    _run_jobs(host)
    time.sleep(0.02)
    assert host.test_keep_warm_calls() >= 1
    start = time.monotonic()
    host.stop()
    assert time.monotonic() - start < 5


def test_the_idle_thread_keeps_its_own_cores_warm(tmp_path, request):
    """The keep-warm pins its workers to the group's cores, which must be the ones enable_cpu_experts took. Mutant:
    pass empty cores to keep_warm in CpuExpertEngine::run -- red (core -1)."""
    _, host, _keep = _host(tmp_path, request, keep_warm_us=500_000)
    _run_jobs(host)
    time.sleep(0.02)
    assert host.test_keep_warm_calls() >= 1
    assert host.test_keep_warm_core() == sorted(os.sched_getaffinity(0))[0]
