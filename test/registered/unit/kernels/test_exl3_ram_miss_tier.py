"""The C++ slot LRU and request service, pumped by hand against a host-simulated device (CPU)."""

import errno
import faulthandler
import subprocess
import sys
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe import exl3_ram_miss
from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.kernels.ops.moe.exl3_ram_miss import (
    DEMAND_RECORDS,
    PAGE_BYTES,
    Exl3RamMissHost,
    new_page,
    new_hot_page,
    hot_record_bytes,
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


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _serve(page, host, row, need, protect):
    seq = sim_post(page, row, need=need, protect=protect)
    assert host.pump() == 1
    return sim_wait(page, seq, timeout_s=1.0)


def _post_gpu_hot(page, host, hot_page, *, row=0, need=(), protect=(), lanes=(), hot=(), hot_seq=None):
    """CPU simulator for the post kernel's sidecar and lease request publication."""
    seq = sim_post(page, row, need=list(need), protect=list(protect), lanes=len(lanes), armed=True)
    stride = hot_record_bytes(host.experts)
    record = hot_page[(seq - 1) % DEMAND_RECORDS * stride : (seq - 1) % DEMAND_RECORDS * stride + stride]
    record[:4].view(torch.int32)[0] = 0
    record[4:8].view(torch.int32)[0] = host.experts
    record[8 : 8 + (host.experts + 7) // 8].zero_()
    for expert in hot:
        offset = 8 + expert // 8
        record[offset] = int(record[offset]) | (1 << (expert % 8))
    record[:4].view(torch.int32)[0] = seq if hot_seq is None else hot_seq
    start = host.lease_layout.d_offset + lease.LANE_REQUEST + (seq - 1) % lease.RING * lease.LANE_REQUEST_BYTES
    request = host.lease_block[start : start + lease.LANE_REQUEST_BYTES]
    request[:8].view(torch.int64)[0] = 0
    request[8:12].view(torch.int32)[0] = len(lanes)
    request[12:16].view(torch.int32)[0] = row
    request[16:48].view(torch.int32).fill_(-1)
    for lane, expert in enumerate(lanes):
        request[16 + lane * 4 : 20 + lane * 4].view(torch.int32)[0] = expert
    request[:8].view(torch.int64)[0] = lease.tagged(lease.DEMAND_TAG, seq)
    return seq


@pytest.mark.parametrize("two_phase", [False, True], ids=["single_phase", "two_phase"])
def test_gpu_hot_sidecar_arms_no_read_lease_and_protects_a_victim(tmp_path, two_phase):
    """EXL3 DIRECT runs with either lease chain; two-phase grants hit lanes before read(), after the hot set applies."""
    s = ram_miss_setup(tmp_path, capacity=3)
    page, hot_page = new_page(pin=False), new_hot_page(6, pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=slot_map, direct=False, hot_page=hot_page)
    try:
        host.enable_lease_mode()
        if two_phase:
            host.enable_two_phase()
        host.enable_gpu_hot()
        for expert in (0, 1, 2):
            host.assign(0, expert)
        seq = _post_gpu_hot(page, host, hot_page, protect=[0], lanes=[0], hot=[0])
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1
        # A resident lane is a touch-only lease: it needs no host row read.
        assert host.counters()["leases_granted"] == 1
        assert host.counters()["touch_only"] == 1
        assert host.counters()["rows_read"] == 0
        assert sum(info[2] for info in host.slot_info(0)) == 1
        # The next demand sees the posted bitmap before its victim census and serve.
        # Expert 0 is the oldest resident, but must survive the read of expert 3.
        slot = host.mapping(0)[0]
        generation = host.slot_info(0)[slot][3]
        ack = host.lease_layout.d_offset + lease.LANE_ACK + (seq - 1) % lease.RING * lease.LANES * lease.LANE_ACK_BYTES
        host.lease_block[ack : ack + 8].view(torch.int64)[0] = lease.tagged(lease.CONSUMED, seq)
        seq = _post_gpu_hot(page, host, hot_page, need=[3], protect=[3], hot=[0])
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1
        assert host.counters()["leases_acked"] == 1
        assert host.contains(0, 0) and host.contains(0, 3)
        assert host.slot_info(0)[slot][3] == generation
    finally:
        host.stop()


@pytest.mark.parametrize("fault", ["stale", "malformed"])
def test_gpu_hot_sidecar_fails_closed_on_stale_bitmap_and_wraps(tmp_path, fault):
    s = ram_miss_setup(tmp_path, capacity=3)
    page, hot_page = new_page(pin=False), new_hot_page(6, pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=slot_map, direct=False, hot_page=hot_page)
    try:
        host.enable_lease_mode()
        host.enable_gpu_hot()
        for _ in range(DEMAND_RECORDS + 1):
            seq = _post_gpu_hot(page, host, hot_page, hot=[0])
            assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 1
        seq = _post_gpu_hot(page, host, hot_page, need=[3], protect=[3],
                            hot_seq=1 if fault == "stale" else None)
        if fault == "malformed":
            stride = hot_record_bytes(host.experts)
            start = (seq - 1) % DEMAND_RECORDS * stride
            hot_page[start + 4 : start + 8].view(torch.int32)[0] = host.experts + 1
        assert host.pump() == 1 and sim_wait(page, seq, 1.0) == 2
        assert not host.contains(0, 3)
    finally:
        host.stop()


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


def test_eviction_takes_the_lru_row_and_spares_hot_experts(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 0, need=[0, 1, 2], protect=[0, 1, 2]) == 1  # stamps 0 < 1 < 2
    host.set_hot(0, [0])  # 0 is the oldest row but hot: never a victim
    assert _serve(page, host, 0, need=[], protect=[1]) == 1  # touch-only: 2 becomes the LRU non-hot row
    assert host.counters()["touch_only"] == 1
    assert _serve(page, host, 0, need=[4], protect=[4]) == 1
    # The victim is 2, the LRU row; it would be 1 if the touch-only record stamped nothing.
    assert slot_map[0, 2].item() == -1 and all(slot_map[0, e].item() >= 0 for e in (0, 1, 4))
    slot = int(slot_map[0, 4])
    assert all(same_bytes(s.slabs[0][n][slot], s.reference(0, [4])[n][0]) for n in EXL3_STREAMED_NAMES)
    # 1 (touched before 4 was read) is now the LRU non-hot row.
    assert _serve(page, host, 0, need=[5], protect=[5]) == 1
    assert slot_map[0, 1].item() == -1 and all(slot_map[0, e].item() >= 0 for e in (0, 4, 5))
    assert host.lru_order(0) == [0, 4, 5] and host.counters()["evictions"] == 2
    # serve() stamps a request's resident ids before it picks a victim, so the shared
    # victim rule's protect exclusion shows through assign(): 4 is the LRU row but protected.
    slot, evicted = host.assign(0, 2, protected=[4], protected_fallback=False)
    assert evicted == 5 and host.contains(0, 4) and slot_map[0, 2].item() == slot


@pytest.mark.parametrize("tier", [2], indirect=True)
def test_no_evictable_slot_fails_the_request_and_raises_fatal(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 0, need=[0, 1], protect=[0, 1]) == 1
    seq = sim_post(page, 0, need=[3], protect=[3, 0, 1])
    assert host.pump() == 1
    assert sim_wait(page, seq, 1.0) == 2
    assert host.fatal_seq() == seq and host.counters()["no_victim"] == 1
    assert host.counters()["late_after_fatal"] == 0
    sim_post(page, 0, need=[], protect=[0])
    assert host.pump() == 1
    assert host.counters()["late_after_fatal"] == 1  # the pump saw the raised fatal word
    assert sim_wait(page, page_word(page, "demand_head"), 1.0) == 3  # sticky


def test_a_failed_read_frees_its_slots_and_reports_failed(tier):
    s, page, slot_map, host = tier
    host.inject(fail_reads=True)
    assert _serve(page, host, 0, need=[1], protect=[1]) == 2
    assert slot_map[0, 1].item() == -1 and not host.contains(0, 1)
    assert host.counters()["read_errors"] == 1


def _row_states(s, layer, experts):
    """Per slot of ``layer``: 'whole' (a byte-exact row of one of ``experts``), 'untouched' (still the
    0xAB sentinel) or 'torn'. Read from the slabs, so it sees a row packed but never published."""
    reference = s.reference(layer, experts)
    states = []
    for slot in range(int(s.tables.capacity[layer])):
        if all(bool((s.slabs[layer][n][slot].view(torch.uint8) == 0xAB).all()) for n in EXL3_STREAMED_NAMES):
            states.append("untouched")
        elif any(
            all(same_bytes(s.slabs[layer][n][slot], reference[n][i]) for n in EXL3_STREAMED_NAMES)
            for i in range(len(experts))
        ):
            states.append("whole")
        else:
            states.append("torn")
    return states


@pytest.fixture
def mirrored_tier(tmp_path):
    """A tier whose rows each read as two extents (two mirror parts), so a fault can hit one part of one row."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    for slot in range(6):
        for name in EXL3_STREAMED_NAMES:
            s.slabs[1][name][slot].view(torch.uint8).fill_(0xAB)
    page = new_page(pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=slot_map, direct=False)
    yield s, page, slot_map, host
    host.stop()


def test_a_fault_injected_at_the_tier_fails_a_row_after_others_packed_and_publishes_none(mirrored_tier):
    """The case fail_reads cannot reach: that flag returns before the reader runs, so nothing has packed and
    'publishes none of the rows it packed' holds vacuously. inject_fault reaches the reader, so rows 0 and 1
    pack into their slots and row 2's part then fails; the tier must publish none of them."""
    s, page, slot_map, host = mirrored_tier
    # hold_ordinal withholds row 2's completions until the other rows have packed, so rows 0 and 1 are
    # packed when row 2 fails, whatever order the kernel completes the reads in.
    host.inject_fault(part=1, part_error=errno.EIO, ordinal=2, hold_ordinal=2)
    assert _serve(page, host, 1, need=[0, 1, 2], protect=[0, 1, 2]) == 2
    states = _row_states(s, 1, [0, 1, 2])
    # The reader packed two rows whole and never touched the failed one (nor any slot it was not given)...
    assert sorted(states) == ["untouched"] * 4 + ["whole"] * 2, states
    # ...and the tier, which had those two rows in hand, published neither and freed every slot.
    assert host.mapping(1) == [-1] * 6 and slot_map[1].tolist() == [-1] * 6
    assert host.slot_to_expert(1) == [-1] * 6
    assert not any(host.contains(1, e) for e in (0, 1, 2))
    assert host.counters()["read_errors"] == 1 and host.counters()["rows_read"] == 0


def test_fail_reads_never_reaches_the_reader_so_no_row_packs(mirrored_tier):
    """The contrast that makes the test above mean something: with fail_reads no row is packed at all."""
    s, page, slot_map, host = mirrored_tier
    host.inject(fail_reads=True)
    assert _serve(page, host, 1, need=[0, 1, 2], protect=[0, 1, 2]) == 2
    assert _row_states(s, 1, [0, 1, 2]) == ["untouched"] * 6


def test_a_pack_delay_injected_at_the_tier_slows_every_row_it_packs(mirrored_tier):
    s, page, slot_map, host = mirrored_tier
    delay_s = 0.05
    host.inject_fault(pack_delay_ns=int(delay_s * 1e9))
    started = time.perf_counter()
    assert _serve(page, host, 1, need=[0, 1, 2], protect=[0, 1, 2]) == 1
    elapsed = time.perf_counter() - started
    assert elapsed >= 3 * delay_s * 0.9, elapsed  # the reader sleeps inside each of the three rows' packing
    assert all(host.contains(1, e) for e in (0, 1, 2))


def test_a_file_cut_short_after_open_fails_the_read(tier):
    s, page, slot_map, host = tier
    # Open checked the size; a shard truncated afterwards reads short of the bytes the table
    # expects, and that fails the demand (not a clamped, silently short row).
    path = s.tables.paths[int(s.tables.extents[0, 0, 0, 0])]
    with open(path, "r+b") as f:
        f.truncate(int(s.tables.extents[0, 0, 0, 1]) + 100)
    assert _serve(page, host, 0, need=[0], protect=[0]) == 2
    assert not host.contains(0, 0) and host.counters()["read_errors"] == 1


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


def test_a_lapped_demand_ring_counts_every_skipped_record(tier):
    s, page, slot_map, host = tier
    for _ in range(20):
        sim_post(page, 0, need=[], protect=[])
    assert host.pump() == 1
    # The catch-up serves from head - 14 (seq 6): seqs 1-5 are skipped and each is counted.
    assert host.counters()["overruns"] == 5 and page_word(page, "demand_done") == 6


def test_the_seqlock_reader_never_accepts_a_torn_record():
    accepted, torn = exl3_ram_miss.seqlock_stress(seconds=1.0)
    assert accepted > 100 and torn == 0, (accepted, torn)


def test_an_empty_need_that_reads_protected_rows_counts_as_served(tier):
    s, page, slot_map, host = tier
    # D12's race: the posted need is empty but a protected row is missing and is read.
    assert _serve(page, host, 0, need=[], protect=[3]) == 1
    assert host.counters()["served"] == 1 and host.counters()["touch_only"] == 0
    assert host.layer_rows() == [1, 0]


def test_an_unarmed_record_only_touches_and_never_evicts_or_reads(tier):
    """Minor 2: nobody waits on an unarmed (touch-only) record, so the GPU may already be
    gathering the next token's rows. Serving it must not evict or read, even when a
    protected id is missing from RAM; it only refreshes the recency of assigned rows."""
    s, page, slot_map, host = tier
    for expert in (0, 1, 2):  # full (capacity 3); 0 is the LRU-oldest row
        assert _serve(page, host, 0, need=[expert], protect=[expert]) == 1
    before_map, before_counters = slot_map.clone(), host.counters()
    sim_post(page, 0, need=[], protect=[1, 5], armed=False)  # 5 is missing
    assert host.pump() == 1
    assert torch.equal(slot_map, before_map) and host.layer_rows() == [3, 0]
    counters = host.counters()
    assert counters["evictions"] == before_counters["evictions"]
    assert counters["touch_only"] == before_counters["touch_only"] + 1
    # The touch still counts: 1 is now the most recent, so 0 goes first, then 2.
    assert _serve(page, host, 0, need=[3], protect=[3]) == 1
    assert _serve(page, host, 0, need=[4], protect=[4]) == 1
    assert [host.contains(0, e) for e in (0, 1, 2)] == [False, True, False]


def test_a_request_that_evicts_and_then_fails_bumps_the_version(tier):
    """The version is how Python learns the map moved (the device map refresh, the
    cached expert_to_slot). A request whose first slot evicted a row and whose second
    found no victim frees its slots, but the eviction stays: that is a map change."""
    s, page, slot_map, host = tier
    for expert in (0, 1, 2):  # full (capacity 3); 0 is the LRU-oldest row
        assert _serve(page, host, 0, need=[expert], protect=[expert]) == 1
    version = host.version()
    # 3 takes 0's slot; 4 finds only protected rows (1, 2) and fails the request.
    sim_post(page, 0, need=[3, 4], protect=[1, 2, 3, 4])
    assert host.pump() == 1
    assert not host.contains(0, 0) and not host.contains(0, 3)
    assert host.version() > version


def test_a_repeated_protect_id_takes_one_slot(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 0, need=[1], protect=[1, 1, 2]) == 1
    assert host.slot_to_expert(0).count(1) == 1 and host.layer_rows() == [2, 0]


@pytest.mark.parametrize(
    "page_fn, map_fn",
    [
        (lambda: new_page(pin=False), lambda: torch.full((6, 2), -1, dtype=torch.int32).t()),
        (lambda: new_page(pin=False), lambda: torch.zeros((2, 6), dtype=torch.int32)),
        (lambda: torch.zeros(2 * PAGE_BYTES, dtype=torch.uint8)[::2], lambda: torch.full((2, 6), -1, dtype=torch.int32)),
    ],
    ids=["transposed_map", "map_not_empty", "strided_page"],
)
def test_the_host_refuses_a_page_or_slot_map_it_cannot_index(tmp_path, page_fn, map_fn):
    s = ram_miss_setup(tmp_path)
    with pytest.raises(ValueError):
        Exl3RamMissHost(s.tables, page=page_fn(), slot_map=map_fn(), direct=False)


def test_release_refuses_a_slot_that_is_still_loading(tier):
    s, page, slot_map, host = tier
    host.inject(delay_s=0.5)
    seq = sim_post(page, 0, need=[1], protect=[1])
    pumper = threading.Thread(target=host.pump)
    pumper.start()
    try:
        assert _until(lambda: 1 in host.slot_to_expert(0))
        slot = host.slot_to_expert(0).index(1)
        assert slot_map[0, 1].item() == -1  # LOADING: not published yet
        with pytest.raises(RuntimeError, match="loading"):
            host.release(0, slot)
    finally:
        pumper.join(timeout=10)
    assert sim_wait(page, seq, 1.0) == 1 and slot_map[0, 1].item() == slot


_CLOSE_DURING_PUMP = """
import pathlib, sys, threading, time
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, sim_post, sim_wait
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
import torch
s = ram_miss_setup(pathlib.Path(sys.argv[1]))
page = new_page(pin=False)
host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
host.inject(delay_s=0.5)
seq = sim_post(page, 0, need=[1], protect=[1])
results = []
pumper = threading.Thread(target=lambda: results.append(host.pump()))
pumper.start()
deadline = time.perf_counter() + 5.0
while 1 not in host.slot_to_expert(0) and time.perf_counter() < deadline:
    time.sleep(0.005)
host.stop()  # closes the handle while the pump is inside a read
pumper.join(timeout=10)
assert results == [1] and sim_wait(page, seq, 1.0) == 1, results
print("ok")
"""


def test_closing_while_a_pump_is_in_flight_keeps_the_service_alive_until_it_returns(tmp_path):
    # In a subprocess: a regression is a use-after-free, which would kill this process.
    result = subprocess.run(
        [sys.executable, "-c", _CLOSE_DURING_PUMP, str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0 and "ok" in result.stdout, (result.returncode, result.stderr[-2000:])


def test_stop_live_closes_every_host_even_when_one_fails(tmp_path, monkeypatch, capsys):
    hosts = []
    for i in range(2):
        (tmp_path / str(i)).mkdir()
        s = ram_miss_setup(tmp_path / str(i))
        hosts.append(
            Exl3RamMissHost(
                s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False
            )
        )

    def broken():
        raise RuntimeError("counters broke")

    monkeypatch.setattr(hosts[0], "counters", broken)
    monkeypatch.setattr(hosts[1], "counters", broken)
    exl3_ram_miss._stop_live()
    assert not hosts[0]._close.alive and not hosts[1]._close.alive
    assert "counters broke" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
