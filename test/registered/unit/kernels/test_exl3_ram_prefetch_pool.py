"""The RAM prefetch's speculative pool (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "The pool"): per row and
group, SPEC_SHARE slots in state kSpec after the staging slots, which the device never maps and no victim, admission
or release path takes (CPU, ChainSim)."""

import pytest
import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup
from sglang.test.dsv41_ram_prefetch_fixtures import HALVES, load, prefetch_rig

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

FREE, READY, STAGING, SPEC = 0, 2, 3, 4


def _states(host, row):
    return [state for state, _, _ in host.slot_info(row)]


def _pool_slots(host, row, group=None):
    return [e["slot"] for e in host.spec_pool(row) if group is None or e["group"] == group]


def test_the_pool_takes_share_free_slots_after_staging_and_publishes_nothing(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        for row in (0, 1):
            assert _states(rig.host, row) == [STAGING] * 3 + [SPEC] * 2 + [FREE] * 2
            assert rig.host.spec_pool(row) == [
                {"group": 0, "slot": 3, "state": "empty", "expert": -1},
                {"group": 0, "slot": 4, "state": "empty", "expert": -1},
            ]
            assert rig.host.mapping(row) == [-1] * 6
            assert rig.sim.delta(row)[0] == 1 and rig.sim.delta(row)[2] == []
        assert rig.host.take_bulk_delta().shape == (0, 3)
    finally:
        rig.host.stop()


def test_each_group_pools_in_its_own_range(tmp_path):
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        for row in (0, 1):
            assert _pool_slots(rig.host, row, 0) == [1] and _pool_slots(rig.host, row, 1) == [5]
            for group, slot in ((0, 1), (1, 5)):
                lo, hi = HALVES[group][row]
                assert lo <= slot < hi
    finally:
        rig.host.stop()


def test_no_demand_victim_and_no_eager_admission_takes_a_pool_slot(tmp_path):
    """Mutant: take_victim_locked taking kSpec as free -- red (the demand's miss lands in a pool slot)."""
    rig = prefetch_rig(tmp_path)
    try:
        load(rig, 1, [0, 1])  # the row's two mappable slots
        load(rig, 1, [2])  # a demand miss: lands in staging and evicts the LRU of 0 and 1, never a pool slot
        mapped = [slot for slot in rig.host.mapping(1) if slot >= 0]
        assert len(mapped) == 2 and not set(mapped) & {3, 4}
        assert _states(rig.host, 1)[3:5] == [SPEC, SPEC]
        assert [e["state"] for e in rig.host.spec_pool(1)] == ["empty", "empty"]
        assert rig.host.victim_census(1, wanted=[0, 1, 2]) == (0, 0)
        with pytest.raises(RuntimeError, match="protected or leased"):
            rig.host.assign(1, 3, protected=[0, 1, 2], protected_fallback=False)
        slots, _ = rig.host.fill_begin(1, [4], protected=[0, 1, 2])
        assert slots == [] and rig.host.fill_end()
        assert _states(rig.host, 1)[3:5] == [SPEC, SPEC]
    finally:
        rig.host.stop()


def test_release_refuses_a_pool_slot(tmp_path):
    """Mutant: drop release()'s kSpec refusal -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="speculative pool slot"):
            rig.host.release(1, 3)
        assert _states(rig.host, 1)[3] == SPEC
    finally:
        rig.host.stop()


def _bare(tmp_path, capacity=7, k=3):
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False, wire=wire_layout(8))
    return s, page, attached_host(s, page, k=k)


def test_spec_pool_without_a_reserved_pool_is_refused(tmp_path):
    s, page, host = _bare(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="no speculative pool"):
            host.spec_pool(0)
    finally:
        host.stop()


@pytest.mark.parametrize("share", [0, 5])
def test_a_share_outside_one_to_four_is_refused(tmp_path, share):
    s, page, host = _bare(tmp_path)
    try:
        with pytest.raises(RuntimeError, match=r"1\.\.4 slots per group"):
            host.reserve_spec_pool(share)
    finally:
        host.stop()


def test_the_pool_is_reserved_once_after_staging_before_any_slot_fills_and_leaves_two_slots(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=7)
    page = new_page(pin=False, wire=wire_layout(8))
    from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost

    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    try:
        with pytest.raises(RuntimeError, match="after reserve_staging"):
            host.reserve_spec_pool(2)
    finally:
        host.stop()
    (tmp_path / "small").mkdir()
    s, page, host = _bare(tmp_path / "small", capacity=6)
    try:
        with pytest.raises(RuntimeError, match="too few slots for a speculative pool of 2"):
            host.reserve_spec_pool(2)
    finally:
        host.stop()
    (tmp_path / "filled").mkdir()
    s, page, host = _bare(tmp_path / "filled")
    try:
        host.assign(0, 1)
        with pytest.raises(RuntimeError, match="before any slot is filled"):
            host.reserve_spec_pool(2)
    finally:
        host.stop()
    (tmp_path / "twice").mkdir()
    s, page, host = _bare(tmp_path / "twice")
    try:
        host.reserve_spec_pool(2)
        with pytest.raises(RuntimeError, match="reserve_spec_pool is once"):
            host.reserve_spec_pool(2)
    finally:
        host.stop()
