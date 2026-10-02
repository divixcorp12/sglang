"""CPU expert split policy, the format-free pool, and the EXL3 trait's registration (CPU, fake kernels)."""

import itertools
import os
import threading

import pytest
import torch

from sglang.srt.layers.moe.cpu_experts.exl3 import Exl3CpuQuantTrait
from sglang.srt.layers.moe.cpu_experts.policy import (
    format_calibration,
    k_star,
    parse_core_list,
    split_from_grid,
    split_table,
)
from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CAP, H, INTER = 4, 32, 16


def _cost(k, n, c_cpu, c_link, handoff):
    return max(handoff + k * c_cpu, (n - k) * c_link)


def test_k_star_matches_brute_force_with_ties_to_the_larger_k():
    """Costs are whole microseconds so a tie is exact; the split is the largest k among the minimizers."""
    grid = range(0, 6)
    for c_cpu, c_link, handoff in itertools.product(grid, range(1, 8), grid):
        for n in range(0, 9):
            costs = [_cost(k, n, c_cpu, c_link, handoff) for k in range(n + 1)]
            expected = max(k for k, c in enumerate(costs) if c == min(costs))
            assert k_star(n, c_cpu, c_link, handoff) == expected, (
                n,
                c_cpu,
                c_link,
                handoff,
            )


def test_k_star_zero_and_one_are_all_cpu_when_the_cpu_beats_the_link():
    for c_cpu, handoff, c_link in [
        (0.44, 0.02, 1.0),
        (0.1, 0.5, 0.7),
        (0.3, 0.0, 0.31),
    ]:
        assert c_cpu + handoff < c_link
        assert k_star(0, c_cpu, c_link, handoff) == 0
        assert k_star(1, c_cpu, c_link, handoff) == 1


def test_split_table_is_k_star_per_n_as_int32():
    table = split_table(6, 0.44, 1.0, 0.02)
    assert table.dtype == torch.int32
    # The plan's k*(n) at 12 node-1 threads: 1->1, 2->2, 3->2, 4->3, 6->4.
    assert table.tolist() == [0, 1, 2, 2, 3, 4, 4]


def test_parse_core_list():
    assert parse_core_list("36-38, 40,36") == [36, 37, 38, 40]


class FakeTrait:
    """A format that shares no tensor name with EXL3; the kernel skips ids outside [0, capacity)."""

    name = "fake"
    slab_names = ("alpha", "beta")
    x_dtype = torch.bfloat16
    weights_dtype = torch.float32
    out_dtype = torch.bfloat16

    def __init__(self, fail_on_register=None):
        self.fail_on_register = fail_on_register
        self.registered, self.freed, self.forwards, self.env_checks = [], [], [], 0

    def check_environment(self):
        self.env_checks += 1

    def register_layer(self, slabs, capacity):
        if len(self.registered) == self.fail_on_register:
            raise RuntimeError("register failed")
        self.registered.append((dict(slabs), capacity))
        return len(self.registered) - 1

    def forward(self, handle, x, slots, weights, out, threads):
        capacity = self.registered[handle][1]
        ran = [int(s) for s in slots[0] if 0 <= int(s) < capacity]
        self.forwards.append((handle, ran, threads))

    def free_layer(self, handle):
        self.freed.append(handle)


class RecordingSlabs(dict):
    """Records which names the pool reads."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.read = set()

    def __getitem__(self, name):
        self.read.add(name)
        return super().__getitem__(name)


def _fake_slabs(capacity=CAP):
    return RecordingSlabs(
        alpha=torch.zeros(capacity, 8), beta=torch.zeros(capacity, 2, 4)
    )


@pytest.fixture
def affinity(monkeypatch):
    bound = []
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cores: bound.append(sorted(cores))
    )
    return bound


def _pool(trait, layers, **kw):
    return CpuExpertPool(trait, layers, **{"cores": [4, 5, 6], "threads": 2, **kw})


def _args(slots, dtype_x=torch.bfloat16):
    k = len(slots[0])
    return (
        torch.tensor(slots),
        torch.ones(1, k, dtype=torch.float32),
        torch.zeros(1, H, dtype=dtype_x),
        torch.ones(1, H, dtype=torch.bfloat16),
    )


def test_pool_touches_only_the_traits_slab_names(affinity):
    """Contract: a new format plugs in without the pool knowing its tensor names."""
    trait, slabs = FakeTrait(), _fake_slabs()
    slabs["unrelated"] = torch.zeros(3)  # a name the trait does not list
    slabs.read.clear()
    pool = _pool(trait, {5: slabs})
    assert slabs.read == {"alpha", "beta"}
    assert set(trait.registered[0][0]) == {"alpha", "beta"}
    assert trait.registered[0][1] == CAP and trait.env_checks == 1
    pool.bind_current_thread()
    pool.compute(5, *_args([[2, -1, 0]]))
    assert trait.forwards == [(0, [2, 0], 2)]
    assert affinity == [[4, 5, 6]]


def test_compute_refuses_an_unbound_thread(affinity):
    """The scheduler's thread must never be pinned by accident; only a thread that bound itself may compute."""
    pool = _pool(FakeTrait(), {5: _fake_slabs()})
    slots, w, x, out = _args([[1]])
    with pytest.raises(RuntimeError, match="bind_current_thread"):
        pool.compute(5, slots, w, x, out)
    pool.bind_current_thread()
    errors = []

    def other():
        try:
            pool.compute(5, slots, w, x, out)
        except RuntimeError as e:
            errors.append(e)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert len(errors) == 1


def test_compute_refuses_a_slot_beyond_the_tier_and_more_than_one_row(affinity):
    """The kernel drops an out-of-range id silently, which for a slot past capacity would lose an expert."""
    trait = FakeTrait()
    pool = _pool(trait, {5: _fake_slabs()})
    pool.bind_current_thread()
    with pytest.raises(ValueError, match="outside the tier"):
        pool.compute(5, *_args([[2, CAP, 0]]))
    slots, w, _, out = _args([[1]])
    with pytest.raises(ValueError, match="batch-1"):
        pool.compute(5, slots, w, torch.zeros(2, H, dtype=torch.bfloat16), out)
    assert trait.forwards == []


def test_capacity_zero_layer_zeroes_out_and_refuses_a_real_slot(affinity):
    trait = FakeTrait()
    pool = _pool(trait, {5: _fake_slabs(0)})
    pool.bind_current_thread()
    slots, w, x, out = _args([[-1, -1]])
    pool.compute(5, slots, w, x, out)
    assert not out.any() and trait.forwards == [] and trait.registered == []
    with pytest.raises(ValueError, match="outside the tier"):
        pool.compute(5, *_args([[0]]))
    with pytest.raises(ValueError, match="not in the CPU expert pool"):
        pool.compute(9, slots, w, x, out)


def test_single_core_pool_is_refused_before_touching_the_trait(affinity):
    """Many spinning workers on one core livelocked the box (DSV41_REFERENCE section 28)."""
    trait = FakeTrait()
    with pytest.raises(ValueError, match="at least 2 cores"):
        _pool(trait, {5: _fake_slabs()}, cores=[4, 4], threads=1)
    assert trait.env_checks == 0 and trait.registered == []


def test_failed_registration_frees_the_layers_already_registered(affinity):
    trait = FakeTrait(fail_on_register=1)
    with pytest.raises(RuntimeError, match="register failed"):
        _pool(trait, {1: _fake_slabs(), 2: _fake_slabs()})
    assert trait.freed == [0]


def test_close_frees_each_layer_once(affinity):
    trait = FakeTrait()
    pool = _pool(trait, {1: _fake_slabs(), 2: _fake_slabs()})
    pool.close()
    pool.close()
    assert sorted(trait.freed) == [0, 1]


class FakeExt:
    def __init__(self):
        self.made = []

    def exl3_moe_cpu_make_layer(self, *args):
        self.made.append(args)
        return len(self.made) - 1


def _exl3_slabs():
    i16, f16 = torch.int16, torch.float16
    n = CAP * 2 * (H // 16) * (INTER // 16) * 48
    return {
        "w13_trellis": torch.arange(n, dtype=i16).view(
            CAP, 2, H // 16, INTER // 16, 48
        ),
        "w13_suh": torch.zeros(CAP, 2, H, dtype=f16),
        "w13_svh": torch.zeros(CAP, 2, INTER, dtype=f16),
        "w2_trellis": torch.zeros(CAP, INTER // 16, H // 16, 48, dtype=i16),
        "w2_suh": torch.zeros(CAP, INTER, dtype=f16),
        "w2_svh": torch.zeros(CAP, H, dtype=f16),
    }


@pytest.mark.parametrize("tier_layout", [False, True], ids=["flat_w2", "tier_w2"])
def test_exl3_trait_registers_each_slot_as_the_right_slab_views(tier_layout):
    """The CPU expert id is the host slot: gate and up are w13 parts 0 and 1 of that slot's row, in place. The pinned
    tier's w2 slabs carry a one-part axis ([slot, 1, ...]); the kernel refuses a 4-D trellis or a 2-D sign vector,
    which is what registering real tier slabs gave before that axis was dropped."""
    ext, slabs = FakeExt(), _exl3_slabs()
    if tier_layout:
        slabs = {n: (t.unsqueeze(1) if n.startswith("w2_") else t) for n, t in slabs.items()}
    Exl3CpuQuantTrait(ext, act_limit=10.0).register_layer(slabs, CAP)
    *lists, gb, ub, db, activation, limit, swizzled = ext.made[0]
    assert (gb, ub, db, activation, limit, swizzled) == ([], [], [], 0, 10.0, 0)
    want = [
        lambda s: slabs["w13_trellis"][s, 0],
        lambda s: slabs["w13_suh"][s, 0],
        lambda s: slabs["w13_svh"][s, 0],
        lambda s: slabs["w13_trellis"][s, 1],
        lambda s: slabs["w13_suh"][s, 1],
        lambda s: slabs["w13_svh"][s, 1],
        lambda s: slabs["w2_trellis"][s],
        lambda s: slabs["w2_suh"][s],
        lambda s: slabs["w2_svh"][s],
    ]
    ranks = [3, 1, 1, 3, 1, 1, 3, 1, 1]
    assert len(lists) == 9
    for got, view, rank in zip(lists, want, ranks):
        assert [t.data_ptr() for t in got] == [view(s).data_ptr() for s in range(CAP)]
        assert all(t.is_contiguous() and t.dim() == rank for t in got)


def test_exl3_trait_refuses_a_kernel_that_would_pin_its_own_workers(monkeypatch):
    trait = Exl3CpuQuantTrait(FakeExt(), act_limit=10.0)
    monkeypatch.delenv("EXL3_MOE_CPU_PIN", raising=False)
    with pytest.raises(ValueError, match="EXL3_MOE_CPU_PIN=0"):
        trait.check_environment()
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait.check_environment()


class FakeServiceTrait(FakeTrait):
    """FakeTrait with the RAM-miss service's half: the lazily set activation limit and the native entry points."""

    act_limit = None

    def __init__(self):
        super().__init__()
        self.events = []

    def hidden_size(self, slabs):
        return int(slabs["alpha"].shape[-1])

    def register_layer(self, slabs, capacity):
        self.events.append(("register", self.act_limit))
        return 100 + super().register_layer(slabs, capacity)

    def native_forward(self):
        return 0xF00D

    def native_set_cores(self, cores):
        self.events.append(("cores", list(cores)))


class FakeHost:
    def __init__(self):
        self.enabled, self.layers, self.splits = None, {}, []
        self.stats = {"jobs": 0, "lanes": 0, "forward_ns": 0}
        self.grid = None  # calibrate_cpu_split's answer; an Exception instance is raised instead
        self.calibrations = []

    def copy_expert_bytes(self, row):
        return 1024

    def calibrate_cpu_split(self, row, *, device, reps, scratch, timeout_s=1.0):
        self.calibrations.append((row, device, reps, scratch.numel(), str(scratch.device)))
        self.scratch = scratch
        # Calibration's own jobs show in the stats, also those of a calibration that then fails.
        self.stats = {"jobs": 999, "lanes": 999, "forward_ns": 999}
        if isinstance(self.grid, Exception):
            raise self.grid
        return torch.tensor(self.grid, dtype=torch.float64)

    def enable_cpu_experts(self, forward, split, cores, x_rows, out_rows, *, threads):
        self.enabled = (forward, list(split), list(cores), tuple(x_rows.shape), tuple(out_rows.shape), threads)

    def set_cpu_layer(self, row, handle):
        self.layers[row] = handle

    def set_cpu_split(self, split):
        self.splits.append(list(split))

    def cpu_stats(self):
        return dict(self.stats)


def _service(host, trait, **kw):
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertService

    slabs = {row: _fake_slabs() for row in range(2)}
    return CpuExpertService(
        host, trait, slabs, **{"hidden": 8, "cores": [4, 5, 6], "threads": 2, "split": [0] * 9, "pin": False, **kw}
    )


def test_service_registers_a_row_once_after_the_cores_and_the_activation_limit():
    """The kernel's pool spawns at its first forward, which follows the first registration, so the cores go first; the
    activation limit is only known at the layer's first forward and must be on the trait before register_layer."""
    host, trait = FakeHost(), FakeServiceTrait()
    svc = _service(host, trait)
    # out_rows is two parts per row: the CPU hits' partial sum and the CPU misses'.
    assert host.enabled == (0xF00D, [0] * 9, [4, 5, 6], (2, 16), (2, 2, 8), 2)
    assert host.layers == {}, "a row reached the grant before its registration"
    svc.register(1, 10.0)
    svc.register(1, 10.0)
    svc.register(0, 10.0)
    assert trait.events == [("cores", [4, 5, 6]), ("register", 10.0), ("register", 10.0)]
    assert host.layers == {1: 100, 0: 101}
    with pytest.raises(ValueError, match="activation limits"):
        _service(FakeHost(), trait).register(0, 7.0)


def test_service_retunes_from_the_measured_cost_only_after_enough_lanes():
    from sglang.srt.environ import envs

    host, trait = FakeHost(), FakeServiceTrait()
    svc = _service(host, trait)
    host.stats = {"jobs": 10, "lanes": 63, "forward_ns": 63 * 10_000_000}
    assert svc.retune() is None and host.splits == []
    host.stats = {"jobs": 11, "lanes": 64, "forward_ns": 64 * 100_000}  # 0.1 ms per expert since the start
    expected = split_table(
        8, 0.1, envs.SGLANG_DSV41_CPU_EXPERTS_LINK_MS.get(), envs.SGLANG_DSV41_CPU_EXPERTS_HANDOFF_MS.get()
    ).tolist()
    assert svc.retune() == expected and host.splits == [expected]
    # The next window is measured from here: 64 more lanes at 10 ms each, a CPU slower than any link.
    host.stats = {"jobs": 12, "lanes": 128, "forward_ns": 64 * 100_000 + 64 * 10_000_000}
    assert svc.retune() == [0] * 9


def test_service_logs_the_cumulative_cost_per_lane(caplog):
    host, trait = FakeHost(), FakeServiceTrait()
    svc = _service(host, trait)
    with caplog.at_level("INFO", logger="sglang.srt.layers.moe.cpu_experts.service"):
        assert svc.log_stats() == {"jobs": 0, "lanes": 0, "forward_ns": 0}
        host.stats = {"jobs": 3, "lanes": 8, "forward_ns": 8 * 500_000}
        assert svc.log_stats()["lanes"] == 8
    assert "0 jobs, 0 lanes, 0.000 ms per lane" in caplog.text
    assert "3 jobs, 8 lanes, 0.500 ms per lane" in caplog.text


@pytest.mark.parametrize("spec", ["0,1,1", "0,2,1,1,1,1,1,1,1", "0,-1,0,0,0,0,0,0,0"])
def test_configured_split_refuses_a_table_the_grant_could_not_honour(spec):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts.service import configured_split

    with envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.override(spec):
        with pytest.raises(ValueError, match="0 <= split"):
            configured_split()



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


def test_failed_calibration_keeps_its_scratch_alive_and_rebaselines_the_stats():
    """A timed-out DMA may still be writing into the scratch: freeing it would let the allocator hand the block to
    other work under a late copy."""
    host = FakeHost()
    host.grid = RuntimeError("calibration: a measurement did not finish within 1000 ms")
    svc = _calibrating_service(host)
    assert svc.calibrate(-1) is None
    assert svc._calibration_scratch is host.scratch
    assert svc._last_stats == host.stats


@pytest.mark.parametrize("fails", [False, True])
def test_log_stats_leaves_out_calibrations_jobs(caplog, fails):
    host = FakeHost()
    host.grid = RuntimeError("calibration failed") if fails else _grid(lambda n, k: 1.0)
    svc = _calibrating_service(host)
    svc.calibrate(-1)
    host.stats = {"jobs": 999 + 3, "lanes": 999 + 8, "forward_ns": 999 + 8 * 500_000}  # three decode jobs since
    with caplog.at_level("INFO", logger="sglang.srt.layers.moe.cpu_experts.service"):
        assert svc.log_stats() == {"jobs": 3, "lanes": 8, "forward_ns": 8 * 500_000}
    assert "3 jobs, 8 lanes, 0.500 ms per lane" in caplog.text


class _RowsTrait:
    """A fake EXL3-shaped trait: out[r] = x[r] * sum of row r's weights over valid slots, recorded per call."""

    name = "fake-rows"
    slab_names = ("w13_trellis",)
    act_limit = 10.0
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(self):
        self.calls = []

    def check_environment(self):
        pass

    def register_layer(self, slabs, capacity):
        return capacity

    def forward(self, handle, x, slots, weights, out, threads):
        self.calls.append((handle, slots.clone(), threading.get_native_id()))
        valid = (slots >= 0).to(torch.float32)
        out.copy_(x.float() * (weights.float() * valid).sum(-1, keepdim=True))

    def free_layer(self, handle):
        pass


def _rows_pool(trait, capacity=4):
    cores = sorted(os.sched_getaffinity(0))[:2]
    if len(cores) < 2:
        pytest.skip("needs at least 2 cores in the affinity mask")
    slabs = {7: {"w13_trellis": torch.zeros(capacity, 2, dtype=torch.int16)}}
    return CpuExpertPool(trait, slabs, cores=cores, threads=2)


def _on_bound_thread(pool, fn):
    result = {}

    def run():
        pool.bind_current_thread()
        try:
            result["value"] = fn()
        except BaseException as error:  # re-raised on the caller
            result["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def test_compute_rows_runs_every_row_and_skips_minus_one():
    trait = _RowsTrait()
    pool = _rows_pool(trait)
    x = torch.arange(6 * 4, dtype=torch.float16).reshape(6, 4)
    slots = torch.tensor([[0, 1, -1]] * 6, dtype=torch.int64)
    weights = torch.full((6, 3), 0.5, dtype=torch.float16)
    out = torch.empty(6, 4, dtype=torch.float32)
    _on_bound_thread(pool, lambda: pool.compute_rows(7, slots, weights, x, out))
    assert torch.equal(out, x.float() * 1.0)  # two valid slots of 0.5 per row
    assert trait.calls[0][1].shape == (6, 3)


@pytest.mark.parametrize(
    "slots_shape,weights_shape,out_rows",
    [((6, 3), (6, 2), 6), ((5, 3), (5, 3), 6), ((6, 3), (6, 3), 5)],
)
def test_compute_rows_refuses_mismatched_shapes(slots_shape, weights_shape, out_rows):
    pool = _rows_pool(_RowsTrait())
    x = torch.zeros(6, 4, dtype=torch.float16)
    slots = torch.zeros(slots_shape, dtype=torch.int64)
    weights = torch.zeros(weights_shape, dtype=torch.float16)
    out = torch.empty(out_rows, 4, dtype=torch.float32)
    with pytest.raises(ValueError):
        _on_bound_thread(pool, lambda: pool.compute_rows(7, slots, weights, x, out))


def test_compute_rows_refuses_a_slot_outside_the_layer():
    pool = _rows_pool(_RowsTrait(), capacity=4)
    x = torch.zeros(2, 4, dtype=torch.float16)
    slots = torch.tensor([[0, 4], [1, 2]], dtype=torch.int64)
    weights = torch.zeros(2, 2, dtype=torch.float16)
    out = torch.empty(2, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="host slot 4"):
        _on_bound_thread(pool, lambda: pool.compute_rows(7, slots, weights, x, out))


def test_compute_rows_refuses_an_unbound_thread():
    pool = _rows_pool(_RowsTrait())
    x = torch.zeros(1, 4, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="bind_current_thread"):
        pool.compute_rows(
            7,
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, 1, dtype=torch.float16),
            x,
            torch.empty(1, 4, dtype=torch.float32),
        )
