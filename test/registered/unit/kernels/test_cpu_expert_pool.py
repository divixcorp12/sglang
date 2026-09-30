"""CPU expert split policy, the format-free pool, and the EXL3 trait's registration (CPU, fake kernels)."""

import itertools
import os
import threading

import pytest
import torch

from sglang.srt.layers.moe.cpu_experts.exl3 import Exl3CpuQuantTrait
from sglang.srt.layers.moe.cpu_experts.policy import (
    k_star,
    parse_core_list,
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


def test_exl3_trait_registers_each_slot_as_the_right_slab_views():
    """The CPU expert id is the host slot: gate and up are w13 parts 0 and 1 of that slot's row, in place."""
    ext, slabs = FakeExt(), _exl3_slabs()
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
    assert len(lists) == 9
    for got, view in zip(lists, want):
        assert [t.data_ptr() for t in got] == [view(s).data_ptr() for s in range(CAP)]
        assert all(t.is_contiguous() for t in got)


def test_exl3_trait_refuses_a_kernel_that_would_pin_its_own_workers(monkeypatch):
    trait = Exl3CpuQuantTrait(FakeExt(), act_limit=10.0)
    monkeypatch.delenv("EXL3_MOE_CPU_PIN", raising=False)
    with pytest.raises(ValueError, match="EXL3_MOE_CPU_PIN=0"):
        trait.check_environment()
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait.check_environment()
