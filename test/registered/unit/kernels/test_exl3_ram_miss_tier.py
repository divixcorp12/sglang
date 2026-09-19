"""The C++ slot LRU and request service, pumped by hand against a host-simulated device (CPU)."""

import faulthandler

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import (
    DEMAND_RECORDS,
    Exl3RamMissHost,
    new_page,
    page_word,
    sim_post,
    sim_wait,
)
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    # pump() runs io_uring reads in C++ and may hold the GIL: a broken drain hangs, so
    # dump every stack and exit instead (pytest-timeout could not interrupt it).
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def tier(tmp_path, request):
    capacity = getattr(request, "param", 3)
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=slot_map, direct=False)
    yield s, page, slot_map, host
    host.stop()


def _serve(page, host, row, need, protect):
    seq = sim_post(page, row, need=need, protect=protect)
    assert host.pump() == 1
    return sim_wait(page, seq, timeout_s=1.0)


def test_a_demand_is_read_split_and_published(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 1, need=[2, 5], protect=[2, 5]) == 1
    reference = s.reference(1, [2, 5])
    for i, expert in enumerate((2, 5)):
        slot = int(slot_map[1, expert])
        assert slot >= 0 and host.mapping(1)[expert] == slot
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(s.slabs[1][name][slot], reference[name][i]), (name, expert)
    assert slot_map[0].tolist() == [-1] * 6
    assert host.layer_rows() == [0, 2] and host.counters()["served"] == 1


def test_protected_experts_missing_from_ram_are_read_too(tier):
    s, page, slot_map, host = tier
    # D12: the thread recomputes the missing set from protect, not only from need.
    assert _serve(page, host, 0, need=[1], protect=[1, 4]) == 1
    assert host.contains(0, 1) and host.contains(0, 4) and host.layer_rows() == [2, 0]


def test_eviction_spares_protected_and_hot_experts_and_unmaps_the_victim_first(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 0, need=[0, 1, 2], protect=[0, 1, 2]) == 1
    host.set_hot(0, [0])
    assert _serve(page, host, 0, need=[], protect=[1]) == 1  # touch-only: 2 becomes the LRU non-hot row
    assert _serve(page, host, 0, need=[4], protect=[4, 1]) == 1
    assert slot_map[0, 2].item() == -1
    assert all(slot_map[0, e].item() >= 0 for e in (0, 1, 4))
    slot = int(slot_map[0, 4])
    assert all(same_bytes(s.slabs[0][n][slot], s.reference(0, [4])[n][0]) for n in EXL3_STREAMED_NAMES)
    assert host.lru_order(0)[-1] == 4 and host.counters()["evictions"] == 1
    assert host.counters()["touch_only"] == 1


@pytest.mark.parametrize("tier", [2], indirect=True)
def test_no_evictable_slot_fails_the_request_and_raises_fatal(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 0, need=[0, 1], protect=[0, 1]) == 1
    seq = sim_post(page, 0, need=[3], protect=[3, 0, 1])
    assert host.pump() == 1
    assert sim_wait(page, seq, 1.0) == 2
    assert host.fatal_seq() == seq and host.counters()["no_victim"] == 1
    sim_post(page, 0, need=[], protect=[0])
    assert host.pump() == 1
    assert sim_wait(page, page_word(page, "demand_head"), 1.0) == 3  # sticky


def test_a_failed_read_frees_its_slots_and_reports_failed(tier):
    s, page, slot_map, host = tier
    host.inject(fail_reads=True)
    assert _serve(page, host, 0, need=[1], protect=[1]) == 2
    assert slot_map[0, 1].item() == -1 and not host.contains(0, 1)
    assert host.counters()["read_errors"] == 1


def test_a_record_whose_seq_does_not_match_is_an_overrun(tier):
    s, page, slot_map, host = tier
    seq = sim_post(page, 0, need=[1], protect=[1])
    record = 64 + ((seq - 1) % DEMAND_RECORDS) * 128
    page[record : record + 4].view(torch.int32)[0] = seq + DEMAND_RECORDS  # a lapped ring slot
    assert host.pump() == 1
    assert host.counters()["overruns"] == 1 and not host.contains(0, 1)
    assert sim_wait(page, seq, 1.0) == 2  # never served: the waiting layer fails stop


def test_advisory_rows_are_counted_apart_from_demand_rows(tier):
    s, page, slot_map, host = tier
    sim_post(page, 1, need=[3], protect=[3], advisory=True, after=page_word(page, "demand_head") + 5)
    assert host.pump() == 2
    assert host.contains(1, 3)
    assert host.layer_advisory_rows() == [0, 1] and host.layer_rows() == [0, 0]
    assert host.counters()["advisory_rows"] == 1


def test_python_assign_release(tier):
    s, page, slot_map, host = tier
    slot, evicted = host.assign(1, 3, protected=[3])
    assert evicted is None and host.contains(1, 3) and slot_map[1, 3].item() == slot
    host.assign(1, 4, protected=[4])
    host.assign(1, 0, protected=[0])
    slot5, evicted = host.assign(1, 5, protected=[5])
    assert evicted == 3 and slot_map[1, 3].item() == -1
    host.release(1, slot5)
    assert slot_map[1, 5].item() == -1 and not host.contains(1, 5)
    host.assign(1, 5, protected=[5])
    with pytest.raises(RuntimeError, match="protected"):
        host.assign(1, 2, protected=[0, 4, 5], protected_fallback=False)
    with pytest.raises(ValueError, match="already"):
        host.assign(1, 4)


def test_injected_delay_starts_after_n_demands_that_read(tier):
    import time

    s, page, slot_map, host = tier
    host.inject(delay_s=0.3, delay_after_demands=1)
    started = time.perf_counter()
    assert _serve(page, host, 0, need=[1], protect=[1]) == 1
    assert time.perf_counter() - started < 0.2
    started = time.perf_counter()
    assert _serve(page, host, 0, need=[2], protect=[2]) == 1
    assert time.perf_counter() - started >= 0.3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
