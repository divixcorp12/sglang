"""Slot generations, the lease-aware eviction predicate and the lease block's header (CPU); LEASE_PROTOCOL.md steps 1-2.

The service grants no lease of its own yet (step 3), so ``inject_lease`` stands in for a GPU reader's lease. Each test
names the mutation of the service it must fail under; the mutations were applied and observed failing.
"""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post, sim_wait
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

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
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    yield s, page, host
    host.stop()


def _serve(page, host, row, need, protect):
    seq = sim_post(page, row, need=need, protect=protect)
    assert host.pump() == 1
    return sim_wait(page, seq, timeout_s=1.0)


def _slot_of(host, row, expert):
    return next(slot for slot, (state, e, _, _) in enumerate(host.slot_info(row)) if state == READY and e == expert)


# ---- the header and row table ----


def test_the_service_writes_the_header_and_the_row_table_at_open(tier):
    """Mutation: the service computes area D's offset without rounding to a page, or numbers a row's slots wrongly."""
    s, page, host = tier
    layout = host.lease_layout
    header = host.lease_header()
    assert header == {
        "magic": lease.MAGIC,
        "abi_version": lease.ABI_VERSION,
        "ring": lease.RING,
        "lanes": lease.LANES,
        "rows": 2,
        "shutdown": 0,
        "slot_gen_offset": lease.SLOT_GEN,
        "d_offset": layout.d_offset,
        "piece_offset": layout.piece_offset,
    }
    assert host.lease_row_table() == [(0, 2), (2, 2)]


def test_the_service_refuses_a_block_it_cannot_address(tmp_path, monkeypatch):
    """Mutation: the service drops its own alignment or size check (the Python check is bypassed here)."""
    s = ram_miss_setup(tmp_path, capacity=2)
    layout = lease.lease_layout([2, 2])
    monkeypatch.setattr(lease, "check_lease_block", lambda *a, **k: None)
    raw = torch.zeros(layout.total_bytes + 2 * lease.BLOCK_ALIGN, dtype=torch.uint8)
    start = (-raw.data_ptr()) % lease.BLOCK_ALIGN
    common = dict(page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    with pytest.raises(RuntimeError, match="aligned"):
        Exl3RamMissHost(s.tables, lease_block=raw[start + 1 : start + 1 + layout.total_bytes], **common)
    with pytest.raises(RuntimeError, match="layout needs"):
        Exl3RamMissHost(s.tables, lease_block=raw[start : start + layout.total_bytes - lease.BLOCK_ALIGN], **common)


# ---- slot generations ----


def test_a_slot_generation_moves_when_a_slot_is_assigned_to_an_expert_and_not_on_a_hit(tier):
    """Mutation: no bump in assign(), or a bump on a hit. The mapped SlotGen word must track the service's count."""
    s, page, host = tier
    assert [g for *_, g in host.slot_info(0)] == [0, 0] and host.mapped_slot_generations(0) == [0, 0]
    slot, _ = host.assign(0, 3, protected=[3])
    assert [g for *_, g in host.slot_info(0)][slot] == 1 and host.mapped_slot_generations(0)[slot] == 1
    assert _serve(page, host, 0, need=[], protect=[3]) == 1  # a hit: touch only
    assert host.mapped_slot_generations(0)[slot] == 1
    assert host.mapped_slot_generations(1) == [0, 0], "another row's generations are its own words"


def test_each_rows_slot_generations_live_in_its_own_words(tier):
    """Mutation: the service indexes SlotGen without the row's base, so row 1's bump lands in row 0's words. Row 0's
    base is 0, so only a bump in a later row can tell."""
    s, page, host = tier
    slot, _ = host.assign(1, 2, protected=[2])
    layout = host.lease_layout
    assert layout.slot_gen_base == (0, 2)
    assert host.mapped_slot_generations(1)[slot] == 1
    assert host.mapped_slot_generations(0) == [0, 0]
    words = host.lease_block[layout.slot_gen_offset : layout.slot_gen_offset + 16].view(torch.int32).tolist()
    assert words == [0, 0] + [1 if i == slot else 0 for i in range(2)]


def test_a_served_demand_bumps_the_slots_it_loads_and_only_those(tier):
    """Mutation: the demand path never bumps, or bumps the slot of an expert that was already resident."""
    s, page, host = tier
    assert _serve(page, host, 0, need=[1], protect=[1]) == 1
    hit_slot = _slot_of(host, 0, 1)
    assert host.mapped_slot_generations(0)[hit_slot] == 1
    assert _serve(page, host, 0, need=[4], protect=[1, 4]) == 1
    assert host.mapped_slot_generations(0)[hit_slot] == 1, "the resident expert's slot was not reloaded"
    assert host.mapped_slot_generations(0)[_slot_of(host, 0, 4)] == 1


def test_the_generation_is_bumped_before_any_byte_of_the_new_row_is_written(tmp_path):
    """Mutation: the bump moved after the read (or after the publish). While the service sleeps between reserving the
    slot and reading into it, a GPU reader that re-read the generation would otherwise see the old one."""
    s = ram_miss_setup(tmp_path, capacity=2)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    try:
        for slot in range(2):
            for name in EXL3_STREAMED_NAMES:
                s.slabs[0][name][slot].view(torch.uint8).fill_(0xEE)  # a sentinel no row can equal
        host.inject(delay_s=0.5)
        host.start_thread(fatal_wait_s=30.0)
        seq = sim_post(page, 0, need=[1], protect=[1])
        saw_loading = None
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and saw_loading is None:
            for slot, (state, _, _, generation) in enumerate(host.slot_info(0)):
                if state == LOADING:
                    saw_loading = (slot, generation, host.mapped_slot_generations(0)[slot])
            time.sleep(0.002)
        assert saw_loading is not None, "the window between reservation and read was never observed"
        slot, generation, mapped = saw_loading
        assert generation == 1 and mapped == 1, "the bump must precede the load"
        assert all(int(s.slabs[0][n][slot].view(torch.uint8).min()) == 0xEE for n in EXL3_STREAMED_NAMES), (
            "the row's bytes were already written: the window was not the one before the first store"
        )
        assert sim_wait(page, seq, timeout_s=10.0) == 1
    finally:
        host.stop()


# ---- the eviction predicate ----


def _two_resident(host):
    """Experts 3 then 4 resident (3 the LRU), each in its own slot."""
    host.assign(0, 3, protected=[3])
    host.assign(0, 4, protected=[4])
    return _slot_of(host, 0, 3), _slot_of(host, 0, 4)


def test_assign_never_takes_a_leased_slot_even_when_it_is_the_lru_row(tier):
    """Mutation: the victim choice ignores leases (then the LRU row, expert 3, is evicted)."""
    s, page, host = tier
    old, new = _two_resident(host)
    host.inject_lease(0, old, +1)
    slot, evicted = host.assign(0, 5, protected=[5], protected_fallback=False)
    assert evicted == 4 and slot == new and host.contains(0, 3) and not host.contains(0, 4)


def test_assign_fails_when_every_slot_is_leased_and_evicts_nothing(tier):
    """Mutation: as above, or a lease that only demotes a slot instead of excluding it."""
    s, page, host = tier
    old, new = _two_resident(host)
    for slot in (old, new):
        host.inject_lease(0, slot, +1)
    before = (host.counters()["evictions"], host.slot_info(0))
    with pytest.raises(RuntimeError, match="leased"):
        host.assign(0, 5, protected=[5])
    assert (host.counters()["evictions"], host.slot_info(0)) == before


def test_release_refuses_a_leased_slot_until_the_lease_retires(tier):
    """Mutation: release() ignores leases."""
    s, page, host = tier
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
    s, page, host = tier
    old, _ = _two_resident(host)
    with pytest.raises(RuntimeError, match="underflow"):
        host.inject_lease(0, old, -1)
    assert host.slot_info(0)[old][2] == 0


def test_the_census_counts_free_evictable_and_leased_slots_without_taking_any(tmp_path):
    """Mutation: a hot or requested slot is counted as a victim, or a leased one as evictable."""
    s = ram_miss_setup(tmp_path, capacity=4)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
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
    """Mutation: no dry run. The take loop then evicts the unleased victim (expert 3) before failing on the leased
    one, and a poll that retried the request would evict a row every time. (The demand's deferral itself, and its
    service once the lease retires, are tested in test_exl3_ram_miss_lease_defer.py.)"""
    s, page, host = tier
    assert _serve(page, host, 0, need=[3, 4], protect=[3, 4]) == 1
    _, leased_slot = _slot_of(host, 0, 3), _slot_of(host, 0, 4)
    host.inject_lease(0, leased_slot, +1)
    assert host.victim_census(0, wanted=[1, 2]) == (0, 1, 1), "precondition: one victim short, one lease away"
    before = (host.counters()["evictions"], host.counters()["version"], host.slot_info(0), host.mapping(0))
    seq = sim_post(page, 0, need=[1, 2], protect=[1, 2])
    assert host.pump() == 0, "deferred: the demand is neither served nor failed"
    assert page_word(page, "demand_done") != seq
    after = (host.counters()["evictions"], host.counters()["version"], host.slot_info(0), host.mapping(0))
    assert after == before
    assert host.counters()["deferred"] == 1 and host.counters()["no_victim"] == 0


def test_a_demand_no_lease_could_help_takes_the_old_path_and_is_not_counted_as_deferred(tier):
    """The dry run must not change today's behaviour when leases are not what stands in the way."""
    s, page, host = tier
    assert _serve(page, host, 0, need=[3, 4], protect=[3, 4]) == 1
    seq = sim_post(page, 0, need=[1], protect=[1, 3, 4])  # every resident row is requested: nothing to evict
    assert host.pump() == 1
    assert sim_wait(page, seq, 1.0) == 2
    assert host.counters()["no_victim"] == 1 and host.counters()["deferred"] == 0


def test_an_advisory_blocked_only_by_leases_gives_up_without_evicting_or_counting_a_deferral(tier):
    """Mutation: an advisory is counted as deferred, or is allowed to evict past a lease."""
    s, page, host = tier
    assert _serve(page, host, 0, need=[3, 4], protect=[3, 4]) == 1
    for expert in (3, 4):
        host.inject_lease(0, _slot_of(host, 0, expert), +1)
    before = (host.counters()["evictions"], host.slot_info(0))
    sim_post(page, 0, need=[1], protect=[1], advisory=True, after=int(page[:4].view(torch.int32)[0]) + 10)
    assert host.pump() == 2
    assert (host.counters()["evictions"], host.slot_info(0)) == before
    assert host.counters()["deferred"] == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
