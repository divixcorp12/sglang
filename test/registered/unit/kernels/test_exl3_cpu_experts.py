"""CPU expert split policy and the pool's registration bookkeeping (CPU, fake kernel)."""

import itertools
import os

import pytest
import torch

from sglang.srt.layers.moe.exl3_cpu_experts import (
    Exl3CpuExpertPool,
    k_star,
    parse_core_list,
    split_table,
)
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


def test_k_star_tie_goes_to_the_larger_k():
    # n=2, cpu 1, link 2: k=1 and k=2 both cost 2; the CPU takes both.
    assert k_star(2, 1, 2, 0) == 2


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


class FakeExt:
    """Records registrations, and runs a forward the way the kernel does: ids outside [0, num_experts) are skipped."""

    def __init__(self):
        self.made, self.freed, self.forwards = [], [], []

    def exl3_moe_cpu_make_layer(self, *args):
        self.made.append(args)
        return len(self.made) - 1

    def exl3_moe_cpu_free_layer(self, handle):
        self.freed.append(handle)

    def exl3_moe_cpu_forward(self, handle, x, selected, weights, out, threads):
        num = len(self.made[handle][0])
        ran = [int(s) for s in selected[0] if 0 <= int(s) < num]
        self.forwards.append((handle, ran, threads))


def _slabs():
    i16, f16 = torch.int16, torch.float16
    return {
        "w13_trellis": torch.arange(
            CAP * 2 * (H // 16) * (INTER // 16) * 48, dtype=i16
        ).view(CAP, 2, H // 16, INTER // 16, 48),
        "w13_suh": torch.zeros(CAP, 2, H, dtype=f16),
        "w13_svh": torch.zeros(CAP, 2, INTER, dtype=f16),
        "w2_trellis": torch.zeros(CAP, INTER // 16, H // 16, 48, dtype=i16),
        "w2_suh": torch.zeros(CAP, INTER, dtype=f16),
        "w2_svh": torch.zeros(CAP, H, dtype=f16),
    }


@pytest.fixture
def pool_env(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    bound = []
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cores: bound.append(sorted(cores))
    )
    return bound


def _pool(ext, slabs, **kw):
    return Exl3CpuExpertPool(
        ext, {7: slabs}, **{"cores": [4, 5, 6], "threads": 2, "act_limit": 10.0, **kw}
    )


def test_each_slot_registers_once_as_the_right_slab_views(pool_env):
    """The CPU expert id is the host slot: gate and up are w13 parts 0 and 1 of that slot's row, in place."""
    ext, slabs = FakeExt(), _slabs()
    _pool(ext, slabs)
    assert len(ext.made) == 1
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
        assert len(got) == CAP
        assert [t.data_ptr() for t in got] == [view(s).data_ptr() for s in range(CAP)]
        assert all(t.is_contiguous() for t in got)


def test_compute_skips_minus_one_and_refuses_a_slot_beyond_the_tier(pool_env):
    """The kernel drops an out-of-range id silently, which for a slot past capacity would lose an expert."""
    ext = FakeExt()
    pool = _pool(ext, _slabs())
    x, w, out = (
        torch.zeros(1, H, dtype=torch.float16),
        torch.ones(1, 3, dtype=torch.float16),
        torch.zeros(1, H),
    )
    pool.compute(7, torch.tensor([[2, -1, 0]]), w, x, out)
    assert ext.forwards == [(0, [2, 0], 2)]
    with pytest.raises(ValueError, match="outside the tier"):
        pool.compute(7, torch.tensor([[2, CAP, 0]]), w, x, out)
    assert len(ext.forwards) == 1
    assert pool_env == [[4, 5, 6]]


def test_compute_refuses_more_than_one_row(pool_env):
    pool = _pool(FakeExt(), _slabs())
    with pytest.raises(ValueError, match="batch-1"):
        pool.compute(
            7,
            torch.zeros(2, 1, dtype=torch.int64),
            torch.zeros(2, 1, dtype=torch.float16),
            torch.zeros(2, H, dtype=torch.float16),
            torch.zeros(2, H),
        )


def test_single_core_pool_is_refused_before_registering_anything(pool_env):
    """Many spinning workers on one core livelocked the box (DSV41_REFERENCE section 28)."""
    ext = FakeExt()
    with pytest.raises(ValueError, match="at least 2 cores"):
        _pool(ext, _slabs(), cores=[4, 4], threads=1)
    assert ext.made == []


def test_pool_refuses_a_kernel_that_would_pin_its_own_workers(pool_env, monkeypatch):
    monkeypatch.delenv("EXL3_MOE_CPU_PIN")
    with pytest.raises(ValueError, match="EXL3_MOE_CPU_PIN=0"):
        _pool(FakeExt(), _slabs())


def test_close_frees_each_layer_once(pool_env):
    ext = FakeExt()
    pool = _pool(ext, _slabs())
    pool.close()
    pool.close()
    assert ext.freed == [0]
