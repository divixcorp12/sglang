"""The CPU expert engine's team between jobs (CPU).

The engine's team is its own for its lifetime and never sleeps: between jobs the workers run the format's register
work for ``keep_warm_us`` (so the next job finds the cores at the AVX-512 license rather than paying the ramp back,
about 50 us per call on SKX after a 1 ms gap), then PAUSE, until the next job, and the engine thread polls the same
way. No job ever waits for a worker to wake, and no OpenMP runtime policy decides when a worker spins. The forward is
the instr build's fake kernel; jobs come from the startup calibration, which submits them through the live engine.
"""

import os
import time
from pathlib import Path

import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, fake_cpu_layer, ram_miss_setup

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROW, ROWS, DST_ROWS, HIDDEN, LANES, THREADS = 1, 2, 6, 8, 8, 2
FORWARD_NS = 200_000  # 0.2 ms per expert


def _host(tmp_path, request, keep_warm_us):
    s = ram_miss_setup(tmp_path, capacity=12, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    host = attached_host(s, new_page(pin=False, wire=wire_layout(8)), k=3)
    host.enable_copy_engine(-1)
    dst = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS)
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 2, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))[:THREADS]
    host.enable_cpu_experts(
        host.test_kernel_address(FORWARD_NS),
        [0] * (lease.wire_layout(8).lanes + 1),
        cores,
        x_rows,
        out_rows,
        threads=THREADS,
        keep_warm_us=keep_warm_us,
    )
    host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    request.addfinalizer(host.stop)  # the fake's call count is per process: no engine may outlive its test
    return s, host, cores, (dst, x_rows, out_rows)


def _team_threads() -> list[Path]:
    """The engine thread and its workers, which carry its name."""
    return [task for task in Path("/proc/self/task").iterdir() if "-cpu-exp" in (task / "comm").read_text()]


def _cpu_s(task: Path) -> float:
    """A thread's user + system time, from /proc."""
    fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _allowed(task: Path) -> list[int]:
    for line in (task / "status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            spec = line.split(":", 1)[1].strip()
            cores: list[int] = []
            for part in spec.split(","):
                lo, _, hi = part.partition("-")
                cores.extend(range(int(lo), int(hi or lo) + 1))
            return cores
    raise AssertionError(f"no Cpus_allowed_list for {task}")


def _run_jobs(host):
    """Run the calibration's CPU jobs (a warm-up and one timed run per cell); returns its grid in ms."""
    scratch = torch.zeros(LANES * host.copy_expert_bytes(ROW), dtype=torch.uint8)
    return host.calibrate_cpu_split(ROW, device=-1, reps=1, scratch=scratch)


def _jobs_run_at_once(host):
    grid = _run_jobs(host)
    for k in range(1, LANES + 1):
        assert grid[0, k] < k * FORWARD_NS / 1e6 + 5, (k, grid[0].tolist())


def test_the_team_is_made_at_start_and_pinned_to_the_engines_cores(tmp_path, request):
    """The engine thread and threads - 1 workers exist before the first job, each on its own core of the config's.
    Mutant: make the team with empty cores in CpuExpertEngine::run -- red (the workers inherit the process mask)."""
    _, host, cores, _keep = _host(tmp_path, request, keep_warm_us=0)
    time.sleep(0.05)
    team = _team_threads()
    assert len(team) == THREADS
    assert sorted(core for task in team for core in _allowed(task)) == sorted(cores)


def test_the_idle_team_never_sleeps(tmp_path, request):
    """Long past its warm window, every thread of the team is still on its CPU (PAUSE is on-CPU time), and the next
    job still runs at once. Mutant: let the workers' idle wait sleep -- red (their CPU time stalls)."""
    _, host, _cores, _keep = _host(tmp_path, request, keep_warm_us=20_000)
    _run_jobs(host)
    time.sleep(0.1)
    before = {task: _cpu_s(task) for task in _team_threads()}
    time.sleep(0.5)
    for task, cpu in before.items():
        assert _cpu_s(task) - cpu > 0.3, task
    _jobs_run_at_once(host)


def test_a_job_runs_at_once_inside_its_warm_window(tmp_path, request):
    """A 2 s window: register work that ran on to its deadline instead of yielding to the job would hold each job for
    up to 2 s."""
    _, host, _cores, _keep = _host(tmp_path, request, keep_warm_us=2_000_000)
    _jobs_run_at_once(host)
    time.sleep(0.02)
    _jobs_run_at_once(host)


def test_the_team_outlives_its_jobs(tmp_path, request):
    """One team for the engine's lifetime: no thread is made or lost by a job."""
    _, host, _cores, _keep = _host(tmp_path, request, keep_warm_us=0)
    time.sleep(0.02)
    before = sorted(task.name for task in _team_threads())
    _run_jobs(host)
    _run_jobs(host)
    assert sorted(task.name for task in _team_threads()) == before


def test_stop_ends_an_idle_team_at_once(tmp_path, request):
    _, host, _cores, _keep = _host(tmp_path, request, keep_warm_us=60_000_000)
    _run_jobs(host)
    time.sleep(0.02)
    start = time.monotonic()
    host.stop()
    assert time.monotonic() - start < 5
    assert _team_threads() == []
