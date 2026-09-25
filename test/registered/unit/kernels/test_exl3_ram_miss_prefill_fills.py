"""Prefill fills (SGLANG_DSV41_ENABLE_PREFILL_FILLS, plan 2026-09-25-dsv41-prefill-fills): the C++ tier claims pinned
slots for an eager caller and reads them through the service's reader on a helper thread (CPU)."""

import faulthandler
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# One row's pack takes this long under the pack_delay fault, so a fill of a few rows is still running when checked.
SLOW_PACK_NS = 300_000_000


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _host(s, **kwargs):
    return Exl3RamMissHost(
        s.tables,
        page=new_page(pin=False),
        slot_map=torch.full(tuple(s.tables.starts.shape), -1, dtype=torch.int32),
        **kwargs,
    )


@pytest.fixture(params=["shards", "row_images"])
def tier(tmp_path, request):
    s = ram_miss_setup(tmp_path, capacity=4, row_images=request.param == "row_images")
    host = _host(s, direct=False)
    yield s, host
    host.fill_end()
    host.stop()


@pytest.fixture
def slow_tier(tmp_path):
    """Shard tables, whose rows pack on the owner, so the pack_delay fault can hold a fill open."""
    s = ram_miss_setup(tmp_path, capacity=4)
    host = _host(s, direct=False)
    yield s, host
    host.fill_end()
    host.stop()


def _landed(s, host, layer, experts):
    reference = s.reference(layer, experts)
    mapping = host.mapping(layer)
    return all(
        same_bytes(s.slabs[layer][name][mapping[expert]], reference[name][i])
        for i, expert in enumerate(experts)
        for name in EXL3_STREAMED_NAMES
    )


def test_a_fill_claims_in_order_and_lands_every_row_byte_for_byte(tier):
    s, host = tier
    slots, evictions = host.fill_begin(1, [4, 1, 3], protected=[4, 1, 3])
    assert len(slots) == 3 and evictions == 0
    host.fill_wait(3, 10.0)
    assert host.fill_end()
    assert [host.mapping(1)[e] for e in (4, 1, 3)] == slots
    assert _landed(s, host, 1, [4, 1, 3])
    # Filled rows are ordinary residents afterwards: releasable and evictable.
    host.release(1, slots[0])
    assert not host.contains(1, 4)


def test_claiming_stops_at_the_first_expert_with_no_unprotected_victim(tier):
    s, host = tier
    for expert in (0, 1, 2, 3):
        host.assign(0, expert)
    # Two victims (2 and 3) are unprotected; the third expert finds none and ends the claim.
    slots, evictions = host.fill_begin(0, [4, 5], protected=[0, 1, 4, 5])
    assert len(slots) == 2 and evictions == 2
    assert host.fill_end()
    assert host.contains(0, 0) and host.contains(0, 1) and not host.contains(0, 2) and not host.contains(0, 3)
    host.release(0, host.mapping(0)[4])
    host.release(0, host.mapping(0)[5])
    host.assign(0, 2)
    slots, evictions = host.fill_begin(0, [3, 4], protected=[0, 1, 2, 3, 4])
    assert len(slots) == 1 and evictions == 0
    assert host.fill_end()
    assert _landed(s, host, 0, [3])


def test_fallback_lets_a_fill_evict_a_protected_row(tier):
    s, host = tier
    for expert in (0, 1, 2, 3):
        host.assign(0, expert)
    slots, evictions = host.fill_begin(0, [4], protected=[0, 1, 2, 3, 4], fallback=True)
    assert len(slots) == 1 and evictions == 1
    assert host.fill_end()


def test_a_slot_being_filled_is_never_a_victim_and_cannot_be_released(slow_tier):
    s, host = slow_tier
    host.inject_fault(pack_delay_ns=SLOW_PACK_NS)
    slots, _ = host.fill_begin(0, [0, 1, 2, 3])
    assert len(slots) == 4
    try:
        # Every slot is filling: no admission can take one, even with fallback.
        with pytest.raises(RuntimeError, match="protected or leased"):
            host.assign(0, 5, protected_fallback=True)
        with pytest.raises(RuntimeError, match="fill"):
            host.release(0, slots[3])
    finally:
        assert host.fill_end()
    host.inject_fault()
    assert host.assign(0, 5)[1] is not None  # the fill ended, so its rows are victims again
    assert _landed(s, host, 0, [1, 2, 3])


def test_fill_wait_returns_as_a_prefix_lands(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4)
    host = _host(s, direct=False)
    try:
        host.inject_fault(pack_delay_ns=SLOW_PACK_NS)
        started = time.perf_counter()
        host.fill_begin(0, [0, 1, 2, 3])
        host.fill_wait(1, 10.0)
        first = time.perf_counter() - started
        host.fill_wait(4, 10.0)
        whole = time.perf_counter() - started
        assert first < whole - SLOW_PACK_NS / 1e9
        assert host.fill_end()
    finally:
        host.fill_end()
        host.stop()


def test_waiting_for_more_rows_than_were_claimed_raises(tier):
    s, host = tier
    host.fill_begin(0, [0, 1])
    with pytest.raises(RuntimeError, match="failed before"):
        host.fill_wait(3, 10.0)
    assert host.fill_end()


def test_a_failed_fill_releases_the_rows_that_did_not_land(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4)
    host = _host(s, direct=False)
    try:
        path = s.tables.paths[int(s.tables.extents[1, 2, 0, 0])]
        with open(path, "r+b") as f:
            f.truncate(int(s.tables.extents[1, 2, 0, 1]) + 100)
        host.fill_begin(1, [2])
        with pytest.raises(RuntimeError, match="failed"):
            host.fill_wait(1, 10.0)
        assert not host.fill_end()
        assert not host.contains(1, 2)
        assert host.counters()["read_errors"] == 1
    finally:
        host.stop()


def test_a_fill_of_a_resident_expert_is_refused(tier):
    s, host = tier
    host.assign(0, 1)
    with pytest.raises(RuntimeError, match="holds a slot"):
        host.fill_begin(0, [1])


def test_a_threaded_service_refuses_a_fill_until_paused_and_resume_joins_it(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=4)
    host = _host(s, direct=False)
    try:
        host.start_thread(fatal_wait_s=30.0)
        with pytest.raises(RuntimeError, match="paused"):
            host.fill_begin(0, [0])
        host.pause(5.0)
        host.inject_fault(pack_delay_ns=SLOW_PACK_NS)
        host.fill_begin(0, [0, 1])
        started = time.perf_counter()
        host.resume()  # joins the running fill before the service thread may read again
        assert time.perf_counter() - started > SLOW_PACK_NS / 1e9
        assert host.fill_end()
        host.inject_fault()
        assert _landed(s, host, 0, [0, 1])
    finally:
        host.stop()


def test_a_second_fill_while_one_runs_is_refused(slow_tier):
    s, host = slow_tier
    host.inject_fault(pack_delay_ns=SLOW_PACK_NS)
    host.fill_begin(0, [0, 1])
    with pytest.raises(RuntimeError, match="already running"):
        host.fill_begin(1, [0])
    assert host.fill_end()
    host.inject_fault()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
