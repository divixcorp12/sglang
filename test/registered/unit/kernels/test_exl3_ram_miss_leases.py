"""The lease-aware eviction predicate, and the service's own check of the lease block (CPU); LEASE_PROTOCOL.md,
"Reuse". ``inject_lease`` stands in for a GPU reader's lease. Each test names the mutation of the service it must
fail under.
"""

import faulthandler

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page, page_word
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_lease_sim import LeaseSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, ram_miss_setup, run_host_script

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

FREE, LOADING, READY = 0, 1, 2


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tier(tmp_path, request):
    capacity = getattr(request, "param", 2)
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    yield s, page, host, LeaseSim(host, page, s.slabs)
    host.stop()


def _serve(host, sim, row, lanes):
    """One whole chain, retired: post, pump, Done, pump."""
    req = sim.post(row, lanes)
    assert host.pump() == 1 and sim.wait(req, 1.0).served
    sim.done(req)
    host.pump()
    return req


def _slot_of(host, row, expert):
    return next(slot for slot, (state, e, _) in enumerate(host.slot_info(row)) if state == READY and e == expert)


def test_the_service_refuses_a_block_it_cannot_address(tmp_path, monkeypatch):
    """Mutation: the service drops its own alignment or size check (the Python check is bypassed here)."""
    s = ram_miss_setup(tmp_path, capacity=2)
    monkeypatch.setattr(lease, "check_lease_block", lambda *a, **k: None)
    raw = torch.zeros(lease.BLOCK_BYTES + 2 * lease.BLOCK_ALIGN, dtype=torch.uint8)
    start = (-raw.data_ptr()) % lease.BLOCK_ALIGN
    common = dict(page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="aligned"):
        ExpertStreamHost(s.tables, lease_block=raw[start + 1 : start + 1 + lease.BLOCK_BYTES], **common)
    with pytest.raises(RuntimeError, match="bytes, not"):
        ExpertStreamHost(s.tables, lease_block=raw[start : start + lease.BLOCK_BYTES - lease.BLOCK_ALIGN], **common)


# ---- the eviction predicate ----


def _two_resident(host):
    """Experts 3 then 4 resident (3 the LRU), each in its own slot."""
    host.assign(0, 3, protected=[3])
    host.assign(0, 4, protected=[4])
    return _slot_of(host, 0, 3), _slot_of(host, 0, 4)


def test_assign_never_takes_a_leased_slot_even_when_it_is_the_lru_row(tier):
    """Mutation: the victim choice ignores leases (then the LRU row, expert 3, is evicted)."""
    s, page, host, sim = tier
    old, new = _two_resident(host)
    host.inject_lease(0, old, +1)
    slot, evicted = host.assign(0, 5, protected=[5], protected_fallback=False)
    assert evicted == 4 and slot == new and host.contains(0, 3) and not host.contains(0, 4)


def test_assign_fails_when_every_slot_is_leased_and_evicts_nothing(tier):
    """Mutation: as above, or a lease that only demotes a slot instead of excluding it."""
    s, page, host, sim = tier
    old, new = _two_resident(host)
    for slot in (old, new):
        host.inject_lease(0, slot, +1)
    before = (host.counters()["evictions"], host.slot_info(0))
    with pytest.raises(RuntimeError, match="leased"):
        host.assign(0, 5, protected=[5])
    assert (host.counters()["evictions"], host.slot_info(0)) == before


def test_release_refuses_a_leased_slot_until_the_lease_retires(tier):
    """Mutation: release() ignores leases."""
    s, page, host, sim = tier
    old, _ = _two_resident(host)
    host.inject_lease(0, old, +1)
    with pytest.raises(RuntimeError, match="leased"):
        host.release(0, old)
    assert host.contains(0, 3)
    host.inject_lease(0, old, -1)
    host.release(0, old)
    assert not host.contains(0, 3)


def test_a_lease_that_would_go_below_zero_is_refused(tier):
    """The hook itself: a decrement past zero is an accounting bug and must not wrap."""
    s, page, host, sim = tier
    old, _ = _two_resident(host)
    with pytest.raises(RuntimeError, match="underflow"):
        host.inject_lease(0, old, -1)
    assert host.slot_info(0)[old][2] == 0


def test_the_census_counts_free_evictable_and_leased_slots_without_taking_any(tmp_path):
    """Mutation: a hot or requested slot is counted as a victim, or a leased one as evictable."""
    s = ram_miss_setup(tmp_path, capacity=4)
    page = new_page(pin=False)
    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    try:
        for expert in (0, 1, 2):
            host.assign(0, expert, protected=[expert])
        assert host.victim_census(0) == (1, 3, 0)  # a FREE slot and three evictable rows
        host.inject_lease(0, _slot_of(host, 0, 0), +1)
        host.set_hot(0, [1])
        assert host.victim_census(0, wanted=[2]) == (1, 0, 1), "hot 1 and requested 2 are not victims; leased 0 is blocked"
        assert host.victim_census(0, wanted=[]) == (1, 1, 1)
        assert host.slot_info(0)[_slot_of(host, 0, 2)][0] == READY, "counting takes nothing"
    finally:
        host.stop()


# ---- refusing a request that only leases stand in the way of ----


def test_a_demand_the_tier_could_serve_only_after_a_lease_retires_evicts_nothing(tier):
    """Mutation: no dry run. The take loop then evicts the unleased victim (expert 3) before it runs out on the leased
    one, and a poll that retried the request would evict a row every time. (The demand's deferral itself, and its
    service once the lease retires, are tested in test_exl3_ram_miss_lease_defer.py.)"""
    s, page, host, sim = tier
    _serve(host, sim, 0, [3, 4])
    leased_slot = _slot_of(host, 0, 4)
    host.inject_lease(0, leased_slot, +1)
    assert host.victim_census(0, wanted=[1, 2]) == (0, 1, 1), "precondition: one victim short, one lease away"
    before = (host.counters()["evictions"], host.counters()["version"], host.slot_info(0), host.mapping(0))
    req = sim.post(0, [1, 2])
    assert host.pump() == 0, "deferred: the demand is neither served nor aborted"
    assert page_word(page, "demand_done") != req.seq
    after = (host.counters()["evictions"], host.counters()["version"], host.slot_info(0), host.mapping(0))
    assert after == before
    assert host.counters()["deferred"] == 1


def test_a_demand_no_lease_could_help_is_not_deferred_and_aborts(tmp_path):
    """The dry run defers only when leases are what stands in the way: with every resident row requested and no
    lease held, nothing will ever free a slot, so waiting would hang the device until its deadline."""
    result = run_host_script(
        tmp_path,
        """
        req = sim.post(0, [3, 4])
        assert host.pump() == 1
        sim.done(req)
        host.pump()
        sim.post(0, [1], protect=[1, 3, 4])
        host.pump()
        print("reached", host.counters()["deferred"])
        """,
        capacity=2,
    )
    assert_aborted(result, "no victim slot for a missing row")

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
