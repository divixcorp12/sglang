"""CPU expert split policy, the format-free pool, and the EXL3 trait's registration (CPU, fake kernels)."""

import itertools
import os
import threading
import weakref

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuParams, Exl3CpuQuantTrait
from sglang.srt.layers.moe.cpu_experts.policy import (
    format_calibration,
    k_star,
    split_from_grid,
    split_table,
)
from sglang.srt.layers.moe.cpu_experts.pool import (
    CPU_EXPERTS_LAYER_ABI_VERSION,
    CpuExpertPool,
)
from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec
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
        self.freed = []

    def exl3_moe_cpu_free_layer(self, handle):
        self.freed.append(handle)


def _exl3_slabs():
    i16, f16 = torch.int16, torch.float16
    n = CAP * 2 * (H // 16) * (INTER // 16) * 48
    return {
        "w13_trellis": torch.arange(n, dtype=i16).view(CAP, 2, H // 16, INTER // 16, 48),
        "w13_suh": torch.zeros(CAP, 2, H, dtype=f16),
        "w13_svh": torch.zeros(CAP, 2, INTER, dtype=f16),
        "w2_trellis": torch.zeros(CAP, INTER // 16, H // 16, 48, dtype=i16),
        "w2_suh": torch.zeros(CAP, INTER, dtype=f16),
        "w2_svh": torch.zeros(CAP, H, dtype=f16),
    }


class FakeRegisterLayer:
    """Stands in for the kernel's sglang_exl3_cpu_experts_register_layer; records each descriptor's values."""

    def __init__(self, status=0):
        self.status, self.calls = status, []

    def __call__(self, layer, handle):
        d = layer._obj
        assert (d.abi_version, d.slab_count, d.activation) == (CPU_EXPERTS_LAYER_ABI_VERSION, 6, 0)
        params = Exl3CpuParams.from_address(d.params)
        self.calls.append(
            (
                list(d.slabs[:6]),
                list(d.slot_bytes[:6]),
                d.capacity,
                d.hidden,
                d.intermediate,
                params.bits,
                params.swizzled,
                d.act_limit,
            )
        )
        handle._obj.value = 40 + len(self.calls)
        return self.status


def _trait_with(monkeypatch, fake, **kw):
    trait = Exl3CpuQuantTrait(FakeExt(), act_limit=10.0, **kw)
    monkeypatch.setattr(trait, "_native", lambda name: fake if name == "sglang_exl3_cpu_experts_register_layer" else None)
    return trait


@pytest.mark.parametrize("tier_layout", [False, True], ids=["flat_w2", "tier_w2"])
def test_exl3_trait_registers_the_six_slab_bases(monkeypatch, tier_layout):
    """The CPU expert id is the host slot: the kernel addresses slot s at each slab's base plus s rows, gate and up as
    w13 parts 0 and 1. The pinned tier's w2 slabs carry a one-part axis ([slot, 1, ...]), which changes no row."""
    slabs = _exl3_slabs()
    if tier_layout:
        slabs = {n: (t.unsqueeze(1) if n.startswith("w2_") else t) for n, t in slabs.items()}
    fake = FakeRegisterLayer()
    trait = _trait_with(monkeypatch, fake, swizzled=True)
    handle = trait.register_layer(slabs, CAP)
    names = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")
    trellis = H * INTER * 3 // 8  # bytes of one 3-bit [k/16, n/16, 48] trellis
    row_bytes = [2 * trellis, 2 * 2 * H, 2 * 2 * INTER, trellis, 2 * INTER, 2 * H]  # quant.hpp's SlabRowBytes
    assert handle == 41
    assert fake.calls == [([slabs[n].data_ptr() for n in names], row_bytes, CAP, H, INTER, 3, 1, 10.0)]


def test_exl3_trait_reads_a_one_slot_slabs_row_size_not_its_stride():
    """A one-slot slab's stride(0) is arbitrary (PyTorch ignores it for a size-1 dim, and it still counts as
    contiguous), so the registered slot bytes must be the row's size."""
    slabs = {}
    for name, t in _exl3_slabs().items():
        one = t[:1].clone()
        slabs[name] = one.as_strided(one.shape, (1,) + one.stride()[1:])
        assert slabs[name].is_contiguous() and slabs[name].stride(0) == 1
    fake = FakeRegisterLayer()
    trait = Exl3CpuQuantTrait(FakeExt(), act_limit=10.0)
    trait._native = lambda name: fake
    trait.register_layer(slabs, 1)
    trellis = H * INTER * 3 // 8
    assert fake.calls[0][1] == [2 * trellis, 2 * 2 * H, 2 * 2 * INTER, trellis, 2 * INTER, 2 * H]


def test_exl3_trait_keeps_the_slabs_alive_until_free(monkeypatch):
    """The kernel keeps only pointers: the trait holds the tensors until the layer is freed."""
    slabs = _exl3_slabs()
    trait = _trait_with(monkeypatch, FakeRegisterLayer())
    handle = trait.register_layer(slabs, CAP)
    probe = weakref.ref(slabs["w2_svh"])
    del slabs
    assert probe() is not None
    trait.free_layer(handle)
    assert trait.ext.freed == [handle]
    assert probe() is None


@pytest.mark.parametrize(
    "name, bad",
    [
        ("w13_suh", lambda t: t.transpose(1, 2).contiguous().transpose(1, 2)),  # not contiguous
        ("w2_svh", lambda t: t.float()),  # wrong dtype
        ("w13_svh", lambda t: t[:, :, : INTER // 2]),  # wrong row size (and not contiguous)
        ("w2_trellis", lambda t: t[: CAP - 1]),  # fewer rows than the capacity
    ],
    ids=["noncontiguous", "dtype", "row_size", "rows"],
)
def test_exl3_trait_refuses_slabs_the_kernel_would_misaddress(monkeypatch, name, bad):
    slabs = _exl3_slabs()
    slabs[name] = bad(slabs[name])
    fake = FakeRegisterLayer()
    with pytest.raises(ValueError, match=name):
        _trait_with(monkeypatch, fake).register_layer(slabs, CAP)
    assert fake.calls == []


def test_exl3_trait_reports_a_refused_registration(monkeypatch):
    trait = _trait_with(monkeypatch, FakeRegisterLayer(status=2))
    with pytest.raises(RuntimeError, match="status 2"):
        trait.register_layer(_exl3_slabs(), CAP)


def test_exl3_trait_refuses_a_kernel_that_would_pin_its_own_workers(monkeypatch):
    trait = Exl3CpuQuantTrait(FakeExt(), act_limit=10.0)
    monkeypatch.delenv("EXL3_MOE_CPU_PIN", raising=False)
    with pytest.raises(ValueError, match="EXL3_MOE_CPU_PIN=0"):
        trait.check_environment()
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait.check_environment()


@pytest.mark.parametrize("tier_layout", [False, True], ids=["flat_w2", "tier_w2"])
def test_exl3_trait_describes_the_six_slabs_for_make_layer(tier_layout):
    """layer_spec gives the kernel's make_layer what the C ABI's registration took: each slab's base and row size in
    EXL3_STREAMED_NAMES order, the shape, the clamp and SglangExl3CpuParams {bits, swizzled}."""
    import struct

    slabs = _exl3_slabs()
    if tier_layout:
        slabs = {n: (t.unsqueeze(1) if n.startswith("w2_") else t) for n, t in slabs.items()}
    trait = Exl3CpuQuantTrait(FakeExt(), act_limit=10.0, swizzled=True)
    spec = trait.layer_spec(slabs, CAP)
    names = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")
    trellis = H * INTER * 3 // 8  # bytes of one 3-bit [k/16, n/16, 48] trellis
    row_bytes = [2 * trellis, 2 * 2 * H, 2 * 2 * INTER, trellis, 2 * INTER, 2 * H]  # quant.hpp's SlabRowBytes
    assert spec.slabs == tuple(zip([slabs[n].data_ptr() for n in names], row_bytes))
    assert (spec.capacity, spec.hidden, spec.intermediate, spec.act_limit, spec.activation) == (CAP, H, INTER, 10.0, 0)
    assert spec.params == struct.pack("<ii", 3, 1)
    assert all(any(k is slabs[n] for k in spec.keep) for n in names)


class FakeServiceTrait(FakeTrait):
    """FakeTrait with the RAM-miss service's half: the lazily set activation limit, the kernel's address and the layer
    spec."""

    act_limit = None

    def __init__(self):
        super().__init__()
        self.events = []

    def hidden_size(self, slabs):
        return int(slabs["alpha"].shape[-1])

    def kernel_address(self):
        return 0x1234

    def layer_spec(self, slabs, capacity):
        self.events.append(("register", self.act_limit))
        return CpuExpertLayerSpec(capacity, 8, 8, self.act_limit, (), b"")


class FakeHost:
    def __init__(self, lanes=8, nodes=1):
        self.wire = lease.wire_layout(lanes, nodes)
        self.nodes = nodes
        self.enabled, self.layers, self.splits = None, {}, []
        self.layer_calls = []  # every set_cpu_layer call, as (row, spec)
        self.enables, self.group_splits, self.stats_groups = [], [], []
        self.node_ranges = None  # set by a test that wants a multi-group host
        self.stats = {"jobs": 0, "lanes": 0, "forward_ns": 0}
        self.grid = None  # calibrate_cpu_split's answer; an Exception instance is raised instead
        self.calibrations = []

    def copy_expert_bytes(self, row):
        return 1024

    def calibrate_cpu_split(self, row, *, device, reps, scratch, timeout_s=1.0, group=0):
        self.calibrations.append((row, device, reps, scratch.numel(), str(scratch.device), group))
        self.scratch = scratch
        # Calibration's own jobs show in the stats, also those of a calibration that then fails.
        self.stats = {"jobs": 999, "lanes": 999, "forward_ns": 999}
        if isinstance(self.grid, Exception):
            raise self.grid
        return torch.tensor(self.grid, dtype=torch.float64)

    def enable_cpu_experts(
        self, kernel, split, cores, x_rows, out_rows, *, threads, group=0, spin_us=50_000, keep_warm_us=0
    ):
        self.enabled = (kernel, list(split), list(cores), tuple(x_rows.shape), tuple(out_rows.shape), threads, group)
        self.enables.append((group, kernel, list(split), list(cores), tuple(x_rows.shape), tuple(out_rows.shape), threads))
        self.keep_warm_us = keep_warm_us

    def set_cpu_layer(self, row, spec):
        self.layers[row] = spec
        self.layer_calls.append((row, spec))

    def set_cpu_split(self, split, group=0):
        self.splits.append(list(split))
        self.group_splits.append((group, list(split)))

    def cpu_stats(self, group=0):
        self.stats_groups.append(group)
        return dict(self.stats)


def _service(host, trait, **kw):
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertService

    slabs = {row: _fake_slabs() for row in range(2)}
    split = [0] * (host.wire.lanes + 1)
    return CpuExpertService(
        host, trait, slabs, **{"hidden": 8, "cores": [4, 5, 6], "threads": 2, "split": split, "pin": False, **kw}
    )


def test_service_registers_a_row_once_after_the_cores_and_the_activation_limit():
    """The kernel's address and the cores reach the host before the CPU expert thread starts; the activation limit is
    only known at the layer's first forward and must be on the trait before layer_spec."""
    host, trait = FakeHost(), FakeServiceTrait()
    svc = _service(host, trait)
    # out_rows is two parts per row: the CPU hits' partial sum and the CPU misses'.
    assert host.enabled == (0x1234, [0] * (lease.wire_layout(8).lanes + 1), [4, 5, 6], (2, 16), (2, 2, 8), 2, 0)
    assert host.layers == {}, "a row reached the grant before its registration"
    svc.register(1, 10.0)
    svc.register(1, 10.0)
    svc.register(0, 10.0)
    assert trait.events == [("register", 10.0), ("register", 10.0)]
    assert sorted(host.layers) == [0, 1] and [row for row, _ in host.layer_calls] == [1, 0]
    assert all(spec.act_limit == 10.0 for spec in host.layers.values())
    with pytest.raises(ValueError, match="activation limits"):
        _service(FakeHost(), trait).register(0, 7.0)


def test_service_keeps_the_cpu_warm_for_2_ms_by_default_and_not_at_0():
    from sglang.srt.environ import envs

    host = FakeHost()
    _service(host, FakeServiceTrait())
    assert host.enabled[0] == 0x1234 and host.keep_warm_us == 2000
    with envs.SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US.override(0):
        host = FakeHost()
        _service(host, FakeServiceTrait())
    assert host.keep_warm_us == 0


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
    assert svc.retune() == [0] * (lease.wire_layout(8).lanes + 1)


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
            configured_split(8)



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
    assert split_from_grid(far) == [0] * (lease.wire_layout(8).lanes + 1)


def test_split_from_grid_never_exceeds_n_and_ignores_cells_past_n():
    # Cells k > n are unused (0.0 in the C++ grid); a 0.0 there must not win.
    grid = _grid(lambda n, k: 10.0 - k)  # more CPU is always faster
    assert split_from_grid(grid) == list(range(lease.wire_layout(8).lanes + 1))
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
    svc = CpuExpertService(host, trait, slabs, hidden=8, cores=[4, 5, 6], threads=2, split=[0] * (host.wire.lanes + 1), pin=False)
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
    assert host.calibrations == [(0, -1, 3, 8 * 1024, "cpu", 0)]
    assert "CPU experts calibration: row 0" in capsys.readouterr().out
    # The stats baseline is re-taken after calibration's own jobs, and retune no longer changes the split.
    assert svc._last_stats == host.stats
    host.stats = {"jobs": 2000, "lanes": 2000, "forward_ns": 2000 * 10_000_000}
    assert svc.retune() is None and host.splits == [split]


def test_a_16_lane_service_calibrates_a_16_lane_grid_with_16_experts_of_scratch(caplog, capsys):
    """The service takes its lane count from the host's wire: the split table, the scratch and the row gate all
    follow it, so calibrating a 16-lane build is not clipped to the first 8 lanes."""
    host = FakeHost(lanes=16)
    host.grid = _grid(lambda n, k: max(0.5 * k, 1.0 * (n - k)), lanes=16)
    svc = _calibrating_service(host, capacity=15)
    with caplog.at_level("WARNING", logger="sglang.srt.layers.moe.cpu_experts.service"):
        assert svc.calibrate(-1) is None
    assert "no registered row has 16 RAM slots" in caplog.text and host.calibrations == []
    svc = _calibrating_service(host, capacity=16)
    split = svc.calibrate(-1)
    assert len(split) == 17 and split == split_from_grid(host.grid)
    assert host.calibrations[0][3] == 16 * 1024 and host.splits == [split]


@pytest.mark.parametrize("lanes, count, ok", [(16, 17, True), (16, 9, False), (8, 9, True), (8, 17, False)])
def test_the_configured_split_lists_one_count_per_lane_plus_one(lanes, count, ok):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts.service import configured_split

    spec = ",".join("0" for _ in range(count))
    with envs.SGLANG_DSV41_CPU_EXPERTS_SPLIT.override(spec):
        if ok:
            assert configured_split(lanes) == [0] * count
        else:
            with pytest.raises(ValueError, match=f"must list {lanes + 1} counts"):
                configured_split(lanes)


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
    assert host.calibrations == [] and svc.split == [0] * (lease.wire_layout(8).lanes + 1)


def test_failed_calibration_warns_and_keeps_the_split(caplog):
    host = FakeHost()
    host.grid = RuntimeError("calibration: a measurement did not finish within 1000 ms")
    svc = _calibrating_service(host)
    with caplog.at_level("WARNING", logger="sglang.srt.layers.moe.cpu_experts.service"):
        assert svc.calibrate(-1) is None
    assert "did not finish" in caplog.text
    assert host.splits == [] and svc.split == [0] * (lease.wire_layout(8).lanes + 1) and not svc.calibrated


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


def test_cpu_expert_groups_run_one_kernel_per_node_and_register_each_layer_once():
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertGroups
    from sglang.srt.layers.moe.cpu_experts.threading_config import NodePlan

    host, trait = FakeHost(nodes=2), FakeServiceTrait()
    plans = [
        NodePlan(group=0, node=0, ram=17, cpu=(8, 9), sq=None, busy_poll=True),
        NodePlan(group=1, node=1, ram=35, cpu=(18, 19, 20), sq=None, busy_poll=True),
    ]
    split = [0] * (host.wire.lanes + 1)
    groups = CpuExpertGroups(host, trait, {r: _fake_slabs() for r in range(2)}, hidden=8, plans=plans, split=split, pin=False)
    assert [(e[0], e[1], e[3], e[6]) for e in host.enables] == [(0, 0x1234, [8, 9], 2), (1, 0x1234, [18, 19, 20], 3)]
    assert tuple(groups.out_rows.shape) == (2, 4, 8), "two output parts per group"
    assert groups.services[1].out_rows is groups.services[0].out_rows
    groups.register(1, 10.0)
    groups.register(1, 10.0)
    assert [e for e in trait.events if e[0] == "register"] == [("register", 10.0)], "one kernel layer serves both groups"
    assert [row for row, _ in host.layer_calls] == [1] and groups.registered(1), "one set_cpu_layer per row across groups"


def _two_group_service(host, capacity=20):
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertGroups
    from sglang.srt.layers.moe.cpu_experts.threading_config import NodePlan

    plans = [
        NodePlan(group=0, node=0, ram=17, cpu=(8, 9), sq=None, busy_poll=True),
        NodePlan(group=1, node=1, ram=35, cpu=(18, 19), sq=None, busy_poll=True),
    ]
    # Group 0 holds 4 slots of each row (too few to calibrate 8 lanes), group 1 the other 16.
    host.node_ranges = [[(0, 4)] * 2, [(4, capacity)] * 2]
    slabs = {row: _fake_slabs(capacity) for row in range(2)}
    groups = CpuExpertGroups(
        host, FakeServiceTrait(), slabs, hidden=8, plans=plans, split=[0] * (host.wire.lanes + 1), pin=False
    )
    groups.register(0, 10.0)
    groups.register(1, 10.0)
    return groups


def test_each_group_calibrates_over_its_own_slots_and_keeps_its_own_split():
    host = FakeHost(nodes=2)
    host.grid = _grid(lambda n, k: max(0.5 * k, 1.0 * (n - k)))
    groups = _two_group_service(host)
    result = groups.calibrate(-1)
    split = split_from_grid(host.grid)
    assert result == [None, split], "group 0's 4 slots cannot calibrate 8 lanes; group 1's 16 can"
    assert [c[-1] for c in host.calibrations] == [1]
    assert host.group_splits == [(1, split)]
    assert groups.services[1].calibrated and not groups.services[0].calibrated


def test_each_group_retunes_and_logs_its_own_stats(caplog):
    host = FakeHost(nodes=2)
    groups = _two_group_service(host)
    host.stats_groups.clear()
    host.stats = {"jobs": 11, "lanes": 64, "forward_ns": 64 * 100_000}
    retuned = groups.retune()
    assert len(retuned) == 2 and all(r is not None for r in retuned)
    assert [g for g, _ in host.group_splits] == [0, 1]
    assert sorted(set(host.stats_groups)) == [0, 1]
    host.stats_groups.clear()
    with caplog.at_level("INFO", logger="sglang.srt.layers.moe.cpu_experts.service"):
        stats = groups.log_stats()
    assert len(stats) == 2 and host.stats_groups == [0, 1]
    assert "group 0 stats" in caplog.text and "group 1 stats" in caplog.text

