"""The prefill share of the C++ RAM tier (CPU): a prefill owns at most K rows per layer, so its admissions stop
evicting decode's rows once it holds K (SGLANG_DSV41_ENABLE_PREFILL_SHARE, plan 2026-09-25-dsv41-prefill-eviction).

Prefill admissions go through ``assign`` (ExpertPinnedHostCache.ensure_rows, a chunk protected at a time); decode
rows through served demands. Each test names the mutation it must fail under.
"""

import faulthandler

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, sim_post, sim_wait
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

EXPERTS = 10


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tier(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4, experts=EXPERTS)
    page = new_page(pin=False)
    host = Exl3RamMissHost(
        s.tables, page=page, slot_map=torch.full((2, EXPERTS), -1, dtype=torch.int32), direct=False
    )
    yield page, host
    host.stop()


def _serve(page, host, need, protect):
    seq = sim_post(page, 0, need=need, protect=protect)
    assert host.pump() == 1
    assert sim_wait(page, seq, timeout_s=1.0) == 1


def _decode_rows(page, host):
    """Decode reads 0, 1, 2, 3 in that order: the tier is full and 0 is the LRU row."""
    for expert in range(4):
        _serve(page, host, need=[expert], protect=[expert])


def _prefill(host, chunks):
    """gather_rows' admissions: each chunk's misses assigned with the chunk protected."""
    for chunk in chunks:
        for expert in chunk:
            host.assign(0, expert, protected=chunk)


def _resident(host):
    return sorted(e for e in host.slot_to_expert(0) if e >= 0)


@pytest.mark.parametrize("share, resident", [(0, [6, 7, 8, 9]), (2, [2, 3, 8, 9])])
def test_decode_rows_survive_a_prefill_that_holds_its_share(tier, share, resident):
    """Mutation: the share is ignored (then the prefill evicts all four decode rows, as with share 0)."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(share)
    _prefill(host, [[4, 5], [6, 7], [8, 9]])
    assert _resident(host) == resident


def test_without_a_share_the_victim_order_is_unchanged(tier):
    """Mutation: share 0 still marks or prefers owned rows."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(0)
    _, evicted = host.assign(0, 4, protected=[4])
    assert evicted == 0
    host.set_prefill_share(0)
    _, evicted = host.assign(0, 5, protected=[5])
    assert evicted == 1


def test_a_decode_hit_ends_a_rows_prefill_ownership(tier):
    """Mutation: a served demand's hit does not clear ownership (then 5's admission evicts 4, not decode's 1)."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(1)
    _, evicted = host.assign(0, 4, protected=[4])
    assert evicted == 0
    host.set_prefill_share(0)
    _serve(page, host, need=[], protect=[4])  # decode routes 4: it is decode's row now
    host.set_prefill_share(1)
    _, evicted = host.assign(0, 5, protected=[5])
    assert evicted == 1 and host.contains(0, 4)
    _, evicted = host.assign(0, 6, protected=[6])
    assert evicted == 5, "5 is owned and the share is full"


def test_an_unarmed_touch_ends_ownership_but_a_prefill_touch_does_not(tier):
    """Mutations: the unarmed record keeps ownership; a Python touch under a share clears it."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(1)
    host.assign(0, 4, protected=[4])
    host.touch(0, 4)  # the prefill's own lookup hit
    _, evicted = host.assign(0, 5, protected=[5])
    assert evicted == 4
    host.set_prefill_share(0)
    sim_post(page, 0, need=[], protect=[5], armed=False)
    assert host.pump() == 1
    host.set_prefill_share(1)
    _, evicted = host.assign(0, 6, protected=[6])
    assert evicted == 1 and host.contains(0, 5)


def test_a_python_touch_without_a_share_ends_ownership(tier):
    """Mutation: touch() never clears ownership (then 5 evicts 4)."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(1)
    host.assign(0, 4, protected=[4])
    host.set_prefill_share(0)
    host.touch(0, 4)
    host.set_prefill_share(1)
    _, evicted = host.assign(0, 5, protected=[5])
    assert evicted == 1 and host.contains(0, 4)


@pytest.mark.parametrize("exclude", ["protected", "hot", "leased"])
def test_an_owned_row_that_is_protected_hot_or_leased_is_never_the_victim(tier, exclude):
    """Mutation: the owned-row branch skips an exclusion of take_slot_locked (then 4 is evicted)."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(1)
    slot4, evicted = host.assign(0, 4, protected=[4])
    assert evicted == 0
    protected = [5]
    if exclude == "protected":
        protected = [4, 5]
    elif exclude == "hot":
        host.set_hot(0, [4])
    else:
        host.inject_lease(0, slot4, +1)
    _, evicted = host.assign(0, 5, protected=protected)
    assert evicted == 1 and host.contains(0, 4)
    if exclude == "leased":
        host.inject_lease(0, slot4, -1)


def test_a_freed_owned_slot_is_no_longer_counted(tier):
    """Mutation: release() leaves the slot owned (then the stale count sends 6 to evict the owned 5, not a decode row)."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(1)
    slot4, _ = host.assign(0, 4, protected=[4])
    host.release(0, slot4)
    host.assign(0, 5, protected=[5])  # takes the free slot, owned: count 1
    host.set_prefill_share(2)
    _, evicted = host.assign(0, 6, protected=[6])
    assert evicted == 1, "one owned row, below the share of 2: a decode row goes"
    _, evicted = host.assign(0, 7, protected=[7])
    assert evicted == 5


def test_a_prefill_takes_free_slots_before_its_own_rows(tier):
    """Mutation: the owned branch runs while a slot is free (then 5 evicts 4 on a cold tier)."""
    page, host = tier
    host.set_prefill_share(1)
    for expert in (4, 5, 6):
        _, evicted = host.assign(0, expert, protected=[expert])
        assert evicted is None
    assert _resident(host) == [4, 5, 6]


def test_a_decode_eviction_of_an_owned_row_ends_its_ownership(tier):
    """Mutation: the eviction in take_slot_locked leaves the slot owned (then 8 evicts decode's 6, whose slot
    still reads as prefill-owned, instead of the LRU decode row 2)."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(2)
    _prefill(host, [[4, 5]])  # evicts 0 and 1; 4 and 5 owned
    host.set_prefill_share(0)
    _serve(page, host, need=[2, 3], protect=[2, 3])  # 4 and 5 are now the LRU rows
    _serve(page, host, need=[6, 7], protect=[6, 7])  # evicts the owned 4 and 5
    assert _resident(host) == [2, 3, 6, 7]
    host.set_prefill_share(2)
    _, evicted = host.assign(0, 8, protected=[8])
    assert evicted == 2


def test_a_row_decode_makes_hot_ends_its_ownership(tier):
    """Mutation: set_hot leaves the row owned (then the share reads full and 6 evicts the owned 5, not decode's 1)."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(2)
    _prefill(host, [[4, 5]])  # evicts 0 and 1
    host.set_hot(0, [4])
    _, evicted = host.assign(0, 6, protected=[6])
    assert evicted == 2 and host.contains(0, 5)


def test_a_negative_share_is_refused(tier):
    page, host = tier
    with pytest.raises(RuntimeError, match="negative"):
        host.set_prefill_share(-1)


# ---- with SGLANG_DSV41_ENABLE_PREFILL_FILLS: the native fill claims through the same rule ----


def test_a_prefetch_fill_stops_at_the_share_and_an_ensure_fill_evicts_the_prefills_own_rows(tier):
    """A layer's prefetch (fill_begin without fallback) claims until the share is full and no owned row can go; the
    chunk admissions after it (with fallback) evict the prefill's gathered rows, not decode's.
    Mutations: fill_begin claims through take_slot_locked (then the prefetch claims 3 and evicts 0, 1, 2); the stop
    applies to ensure fills too (then 6 is not claimed)."""
    page, host = tier
    _decode_rows(page, host)
    host.set_prefill_share(2)
    slots, evictions = host.fill_begin(0, [4, 5, 6], protected=[4, 5, 6])
    assert host.fill_end()
    assert len(slots) == 2 and evictions == 2 and _resident(host) == [2, 3, 4, 5]
    slots, evictions = host.fill_begin(0, [6], protected=[6], fallback=True)
    assert host.fill_end()
    assert len(slots) == 1 and evictions == 1 and _resident(host) == [2, 3, 5, 6]


def test_without_a_share_a_prefetch_fill_claims_as_before(tier):
    page, host = tier
    _decode_rows(page, host)
    slots, evictions = host.fill_begin(0, [4, 5, 6], protected=[4, 5, 6])
    assert host.fill_end()
    assert len(slots) == 3 and evictions == 3 and _resident(host) == [3, 4, 5, 6]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
