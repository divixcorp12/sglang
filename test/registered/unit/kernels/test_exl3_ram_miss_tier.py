"""The C++ slot LRU and request service, pumped by hand against a host-simulated device (CPU)."""

import dataclasses
import faulthandler
import re
import subprocess
import sys
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport
from sglang.kernels.ops.moe.expert_stream_transport import (
    DEMAND_RECORDS,
    DEMAND_RING,
    PAGE_BYTES,
    RECORD_BYTES,
    ExpertStreamHost,
    new_hot_page,
    new_page,
    page_word,
)
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, ram_miss_setup, run_host_script, same_bytes

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
    """(capacity, staging slots per row): 4 and 1 by default, so three rows are mappable."""
    capacity, k = getattr(request, "param", (4, 1))
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = ExpertStreamHost(s.tables, page=page, slot_map=slot_map)
    host.reserve_staging(k)
    yield s, page, slot_map, host, ChainSim(host, page, s.slabs)
    host.stop()


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _serve(t, row, lanes, protect=None):
    """One chain: post, pump (serves). Returns whether every miss's pieces landed, as S judges it."""
    s, page, slot_map, host, sim = t
    req = sim.post(row, lanes, protect=protect)
    assert host.pump() == 1
    return sim.served(req)


def test_gpu_hot_sidecar_protects_a_victim_and_a_resident_lane_reads_nothing(tmp_path):
    """The hot set applies before the census and serve: expert 0 is the oldest resident row, but hot, so the read of
    expert 3 takes another slot. Mutation: apply_gpu_hot after serve, or not at all."""
    s = ram_miss_setup(tmp_path, capacity=4)
    page, hot_page = new_page(pin=False), new_hot_page(6, pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = ExpertStreamHost(s.tables, page=page, slot_map=slot_map, hot_page=hot_page)
    host.reserve_staging(1)
    sim = ChainSim(host, page, s.slabs)
    try:
        host.enable_gpu_hot()
        for expert in (0, 1, 2):
            host.assign(0, expert)
        sim.sync_bulk()
        req = sim.post(0, [0], hot=[0])
        assert host.pump() == 1 and req.kinds == [LaneKind.HIT_SM]
        # A RAM-hit lane needs no host row read.
        assert host.counters()["touch_only"] == 1
        assert host.counters()["rows_read"] == 0
        req = sim.post(0, [3], hot=[0])
        assert host.pump() == 1 and sim.served(req)
        assert host.contains(0, 0) and host.contains(0, 3) and not host.contains(0, 1)
    finally:
        host.stop()


@pytest.mark.parametrize("fault", ["stale", "malformed"])
def test_gpu_hot_sidecar_aborts_on_a_stale_or_malformed_record_after_a_wrap(tmp_path, fault):
    """A demand whose hot set cannot be read is not served without one: the victim choice would ignore the GPU's hot
    experts. It fails stop, after a full lap of good records (the record index wraps)."""
    result = run_host_script(
        tmp_path,
        f"""
        from sglang.kernels.ops.moe.expert_stream_transport import DEMAND_RECORDS, hot_record_bytes
        host.enable_gpu_hot()
        for _ in range(DEMAND_RECORDS + 1):
            req = sim.post(0, [], hot=[0])
            assert host.pump() == 1
        fault = "{fault}"
        seq = page_word(page, "demand_head") + 1
        if fault == "malformed":
            start = (seq - 1) % DEMAND_RECORDS * hot_record_bytes(host.experts)
            orig = sim._write_hot
            def write_hot(*args, **kwargs):
                orig(*args, **kwargs)
                hot_page[start + 8] = int(hot_page[start + 8]) | 0x80  # a bit past the last expert
            sim._write_hot = write_hot
        sim.post(0, [3], hot=[0], hot_seq=1 if fault == "stale" else None)
        host.pump()
        print("reached")
        """,
        host_args=", hot_page=hot_page",
    )
    assert_aborted(result, "no hot set for it")


@pytest.mark.parametrize("tier", [(4, 2)], indirect=True)
def test_a_demand_is_read_split_and_published(tier):
    s, page, slot_map, host, sim = tier
    assert _serve(tier, 1, [2, 5])
    reference = s.reference(1, [2, 5])
    for i, expert in enumerate((2, 5)):
        slot = int(slot_map[1, expert])
        assert slot >= 0 and host.mapping(1)[expert] == slot
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(s.slabs[1][name][slot], reference[name][i]), (name, expert)
    assert slot_map[0].tolist() == [-1] * 6
    assert host.layer_rows() == [0, 2] and host.counters()["served"] == 1


def test_protected_experts_missing_from_ram_are_not_read(tier):
    s, page, slot_map, host, sim = tier
    # Only the record's miss lanes are read: a routed expert that is VRAM-hot (protect only) is not.
    assert _serve(tier, 0, [1], protect=[1, 4])
    assert host.contains(0, 1) and not host.contains(0, 4) and host.layer_rows() == [1, 0]


@pytest.mark.parametrize("tier", [(6, 3)], indirect=True)
def test_eviction_takes_the_lru_row_and_spares_hot_experts(tier):
    s, page, slot_map, host, sim = tier
    assert _serve(tier, 0, [0, 1, 2])  # stamps 0 < 1 < 2
    host.set_hot(0, [0])  # 0 is the oldest row but hot: never a victim
    assert _serve(tier, 0, [], protect=[1])  # touch-only: 2 becomes the LRU non-hot row
    assert host.counters()["touch_only"] == 1
    assert _serve(tier, 0, [4])
    # The victim is 2, the LRU row; it would be 1 if the touch-only record stamped nothing.
    assert slot_map[0, 2].item() == -1 and all(slot_map[0, e].item() >= 0 for e in (0, 1, 4))
    slot = int(slot_map[0, 4])
    assert all(same_bytes(s.slabs[0][n][slot], s.reference(0, [4])[n][0]) for n in EXL3_STREAMED_NAMES)
    # 1 (touched before 4 was read) is now the LRU non-hot row.
    assert _serve(tier, 0, [5])
    assert slot_map[0, 1].item() == -1 and all(slot_map[0, e].item() >= 0 for e in (0, 4, 5))
    assert host.lru_order(0) == [0, 4, 5] and host.counters()["evictions"] == 2
    # serve_record stamps a request's resident ids before it picks a victim, so the
    # protect exclusion shows through assign(): 4 is the LRU row but protected.
    slot, evicted = host.assign(0, 2, protected=[4], protected_fallback=False)
    assert evicted == 5 and host.contains(0, 4) and slot_map[0, 2].item() == slot


@pytest.fixture
def mirrored_tier(tmp_path):
    """A tier whose rows each read as two extents (two mirror parts), so a fault can hit one part of one row."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    for slot in range(6):
        for name in EXL3_STREAMED_NAMES:
            s.slabs[1][name][slot].view(torch.uint8).fill_(0xAB)
    page = new_page(pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = ExpertStreamHost(s.tables, page=page, slot_map=slot_map)
    host.reserve_staging(3)
    yield s, page, slot_map, host, ChainSim(host, page, s.slabs)
    host.stop()


def test_a_pack_delay_injected_at_the_tier_slows_every_row_it_packs(mirrored_tier):
    s, page, slot_map, host, sim = mirrored_tier
    delay_s = 0.05
    host.inject_fault(pack_delay_ns=int(delay_s * 1e9))
    started = time.perf_counter()
    assert _serve(mirrored_tier, 1, [0, 1, 2])
    elapsed = time.perf_counter() - started
    assert elapsed >= 3 * delay_s * 0.9, elapsed  # the reader sleeps inside each of the three rows' packing
    assert all(host.contains(1, e) for e in (0, 1, 2))


def test_a_file_cut_short_after_open_aborts_the_read(tmp_path):
    """Open checked the size; a file truncated afterwards reads short of the bytes the table expects, and that fails
    stop (not a clamped, silently short row). Mutation: a short read is accepted as landed."""
    result = run_host_script(
        tmp_path,
        """
        path = s.tables.paths[int(s.tables.extents[0, 0, 0, 0])]
        with open(path, "r+b") as f:
            f.truncate(int(s.tables.extents[0, 0, 0, 1]) + 100)
        sim.post(0, [0])
        host.pump()
        print("reached")
        """,
    )
    assert_aborted(result, "the read failed")


def test_a_slab_table_narrower_than_the_layout_is_refused(tmp_path):
    """tables_from indexes slabs[row][name] for every layout name; a 5-wide table would read past each row.

    check_table_tensors now runs before tables_from and refuses a 5-wide slabs table on shape alone, naming the
    tensor (Ruling 3's verify_named prefix), so tables_from's own "6 names" message is never reached."""
    s = ram_miss_setup(tmp_path)
    narrow = dataclasses.replace(s.tables, slabs=s.tables.slabs[:, :5].contiguous(), row_bytes=s.tables.row_bytes[:5])
    with pytest.raises(RuntimeError, match="^slabs: "):
        expert_stream_transport.read_rows_once(narrow, row=0, experts=[0], slots=[0])


def test_open_refuses_an_extent_table_of_the_wrong_dtype(tmp_path):
    """open() (via read_rows_once's C++ entry) reads extents as int64 [L, E, parts, 4] through a raw pointer; int32
    would be read as packed pairs and name files and offsets that were never written. check_table_tensors catches
    this at the FFI boundary before tables_from ever dereferences the tensor."""
    s = ram_miss_setup(tmp_path)
    bad = dataclasses.replace(s.tables, extents=s.tables.extents.to(torch.int32))
    with pytest.raises(Exception, match="^extents: "):
        expert_stream_transport.read_rows_once(bad, row=0, experts=[0], slots=[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device to show experts is refused off it")
def test_read_rows_refuses_a_cuda_experts_tensor(tmp_path):
    """experts is read on the host through ids_of's raw int64 cast; a CUDA tensor would have this host-only FFI
    entry read GPU memory as host memory. read_rows_once always rebuilds experts as a fresh CPU tensor
    (``_ids``), so this device refusal is only observable by calling the raw C++ export directly."""
    s = ram_miss_setup(tmp_path)
    module = expert_stream_transport._host_module()
    args = expert_stream_transport._table_args(s.tables, True)
    experts = torch.tensor([0], dtype=torch.int64, device="cuda")
    slots = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(Exception, match="^experts: "):
        module.expert_stream_read_rows(*args, 0, experts, slots, expert_stream_transport.BOUNCE_ROWS)


def _out_extent(host, entry):
    """The int64 words each introspection entry writes for row 0: its handle's real extent, not the caller's."""
    capacity = int(host.tables.capacity[0])
    return {
        "slot_info": 3 * capacity,
        "mapping": host.experts,
        "slot_to_expert": capacity,
        "lru_order": capacity,
        "layer_rows": host.layers,
    }[entry]


@pytest.mark.parametrize("delta", [-1, 1], ids=["undersized", "oversized"])
@pytest.mark.parametrize("entry", ["slot_info", "mapping", "slot_to_expert", "lru_order", "layer_rows"])
def test_an_out_buffer_of_the_wrong_size_is_refused_before_any_write(tier, entry, delta):
    """Each entry writes a handle-dependent count of int64s through a raw pointer with no bound of its own, so a
    buffer one word short is written past its end. The out tensor is a view into a larger sentinel-filled backing:
    a missing refusal shows as the call succeeding, and any write, in bounds or past the view's end, as a changed
    sentinel. The exact size is required, so one word too many is refused as well."""
    s, page, slot_map, host, sim = tier
    assert _serve(tier, 0, [0]) and _serve(tier, 0, [1])  # every entry has something to write
    expected = _out_extent(host, entry)
    sentinel = -7
    backing = torch.full((expected + 2,), sentinel, dtype=torch.int64)
    out = backing[: expected + delta]
    call = getattr(host._module, f"expert_stream_{entry}")
    with pytest.raises(RuntimeError, match=rf"^out: (?s:.*)expected {expected} but got {expected + delta}"):
        call(host.handle, out) if entry == "layer_rows" else call(host.handle, 0, out)
    assert backing.eq(sentinel).all(), "the refusal came after a write"


@pytest.mark.parametrize("delta", [-1, 1], ids=["undersized", "oversized"])
def test_fill_begins_out_buffer_of_the_wrong_size_is_refused_before_any_write(tier, delta):
    """fill_begin writes the evictions word at out[len(experts)] first, then a slot per claimed expert: its extent
    comes from the input, len(experts) + 1. Mutation: out checked for dtype, rank and device only."""
    s, page, slot_map, host, sim = tier
    experts = torch.tensor([0, 1], dtype=torch.int64)
    expected = experts.numel() + 1
    sentinel = -7
    backing = torch.full((expected + 2,), sentinel, dtype=torch.int64)
    out = backing[: expected + delta]
    protect = torch.zeros(0, dtype=torch.int64)
    try:
        host._module.expert_stream_fill_begin(host.handle, 0, experts, protect, 0, out)
    except RuntimeError as refused:
        message = str(refused)
    else:
        host._module.expert_stream_fill_end(host.handle)  # a fill started: join it before teardown
        message = "no refusal"
    assert re.match(rf"^out: (?s:.*)expected {expected} but got {expected + delta}", message), message
    assert backing.eq(sentinel).all(), "the refusal came after a write"


def _test_only_entry(case):
    """(tensor name, exact extent, call) of a test-only entry that reads or writes a fixed-extent caller buffer.
    `call(buffer)` passes `buffer` as the named tensor and correct tensors everywhere else."""
    module = expert_stream_transport._host_module()
    return {
        "publish_piece:word": ("word", 1, lambda b: module.expert_stream_publish_piece(b, 0, 1)),
        "seqlock_stress:out": ("out", 2, lambda b: module.expert_stream_seqlock_stress(1_000_000, b)),
    }[case]


@pytest.mark.parametrize("delta", [-1, 1], ids=["undersized", "oversized"])
@pytest.mark.parametrize(
    "case",
    [
        "publish_piece:word",
        "seqlock_stress:out",
    ],
)
def test_a_test_only_entrys_fixed_extent_buffer_of_the_wrong_size_is_refused_before_any_write(case, delta):
    """These entries dereference a caller tensor of a fixed extent through a raw pointer with no check at all.
    Mutation: the tensor is not checked."""
    name, expected, call = _test_only_entry(case)
    sentinel = 0  # for publish_piece a word of generation 0 with bit 1 clear, so an unchecked call would set it
    backing = torch.full((expected + 2,), sentinel, dtype=torch.int64)
    with pytest.raises(RuntimeError, match=rf"^{name}: (?s:.*)expected {expected} but got {expected + delta}"):
        call(backing[: expected + delta])
    assert backing.eq(sentinel).all(), "the refusal came after a write"


def test_the_seqlock_reader_never_accepts_a_torn_record():
    """read_record against a writer thread rewriting one record in the post kernel's seqlock order: a record read
    while its payload changes must be refused. Mutant: drop read_record's second seq load -- torn records accepted."""
    accepted, torn = expert_stream_transport.seqlock_stress(seconds=1.0)
    assert accepted > 100 and torn == 0, (accepted, torn)


_MAPPING_ROW = """
import pathlib, sys, torch
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
s = ram_miss_setup(pathlib.Path(sys.argv[1]))
host = ExpertStreamHost(s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32))
try:
    host._module.expert_stream_mapping(host.handle, host.layers, torch.empty(host.experts, dtype=torch.int64))
    print("no refusal")
except RuntimeError as refused:
    print(refused)
host.stop()
"""


def test_mapping_refuses_a_row_out_of_range(tmp_path):
    """mapping indexes tiers_[row] with no check; its three row-taking siblings refuse through row_capacity.
    In a subprocess: before the check, the row indexes past tiers_, which may kill the process."""
    result = subprocess.run(
        [sys.executable, "-c", _MAPPING_ROW, str(tmp_path)], capture_output=True, text=True, timeout=120
    )
    assert "streamed row 2 is out of range" in result.stdout, (result.returncode, result.stdout, result.stderr[-2000:])


def test_a_record_whose_seq_does_not_match_is_an_overrun(tier):
    s, page, slot_map, host, sim = tier
    req = sim.post(0, [1], hit_copy="sm", kinds=[LaneKind.HIT_SM], slots=[0])
    record = DEMAND_RING + ((req.seq - 1) % DEMAND_RECORDS) * RECORD_BYTES
    page[record : record + 4].view(torch.int32)[0] = req.seq + DEMAND_RECORDS  # a lapped ring slot
    assert host.pump() == 1
    assert host.counters()["overruns"] == 1 and host.counters()["touch_only"] == 0


def test_python_assign_release(tier):
    s, page, slot_map, host, sim = tier
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

    s, page, slot_map, host, sim = tier
    host.inject(delay_s=0.3, delay_after_demands=1)
    started = time.perf_counter()
    assert _serve(tier, 0, [1])
    assert time.perf_counter() - started < 0.2
    started = time.perf_counter()
    assert _serve(tier, 0, [2])
    assert time.perf_counter() - started >= 0.3


def test_a_lapped_demand_ring_counts_every_skipped_record(tier):
    s, page, slot_map, host, sim = tier
    for _ in range(20):
        sim.post(0, [], protect=[])
    assert host.pump() == 1
    # The catch-up serves from head - 14 (seq 6): seqs 1-5 are skipped and each is counted.
    assert host.counters()["overruns"] == 5 and host.handled_through() == 6


def test_an_unarmed_record_only_touches_and_never_evicts_or_reads(tier):
    """Minor 2: nobody waits on an unarmed (touch-only) record, so the GPU may already be
    gathering the next token's rows. Serving it must not evict or read, even when a
    protected id is missing from RAM; it only refreshes the recency of assigned rows."""
    s, page, slot_map, host, sim = tier
    for expert in (0, 1, 2):  # full (three mappable rows); 0 is the LRU-oldest row
        assert _serve(tier, 0, [expert])
    before_map, before_counters = slot_map.clone(), host.counters()
    sim.post(0, [], protect=[1, 5])  # 5 is missing
    assert host.pump() == 1
    assert torch.equal(slot_map, before_map) and host.layer_rows() == [3, 0]
    counters = host.counters()
    assert counters["evictions"] == before_counters["evictions"]
    assert counters["touch_only"] == before_counters["touch_only"] + 1
    # The touch still counts: 1 is now the most recent, so 0 goes first, then 2.
    assert _serve(tier, 0, [3])
    assert _serve(tier, 0, [4])
    assert [host.contains(0, e) for e in (0, 1, 2)] == [False, True, False]


def test_a_repeated_protect_id_takes_one_slot(tier):
    s, page, slot_map, host, sim = tier
    assert _serve(tier, 0, [1], protect=[1, 1, 2])
    assert host.slot_to_expert(0).count(1) == 1 and host.layer_rows() == [1, 0]


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
        ExpertStreamHost(s.tables, page=page_fn(), slot_map=map_fn())


def test_an_owned_call_waits_for_a_pump_on_another_thread(tier):
    """B3 fix round (review Minor 3): the caller of pump() owns the tier for its whole request, and pump() takes
    caller_mutex_, so another Python thread's owned call is serialized behind that request instead of racing it: it
    never sees the claimed slot still LOADING. (This test used to release from a second thread mid-read, expecting the
    "loading" refusal: that second thread is the second owner the single-owner rule forbids. release()'s refusal stays
    as a backstop no legal caller reaches.)"""
    s, page, slot_map, host, sim = tier
    host.inject(delay_s=0.5)
    req = sim.post(0, [1])
    pumper = threading.Thread(target=host.pump)
    pumper.start()
    try:
        assert _until(lambda: host.busy_episode() != 0)  # a lock-free word: the pump is in the request
        start = time.perf_counter()
        assert host.contains(0, 1)  # an owned call: it waits for the pump's request to end
        waited = time.perf_counter() - start
        slot = slot_map[0, 1].item()
        assert slot >= 0, "the owned call ran while the pump's slot was still LOADING"
        assert waited > 0.2, f"the owned call returned after {waited:.3f} s, inside the 0.5 s read"
    finally:
        pumper.join(timeout=10)
    assert sim.served(req) and slot_map[0, 1].item() == slot


def test_closing_while_a_pump_is_in_flight_keeps_the_service_alive_until_it_returns(tmp_path):
    # In a subprocess: a regression is a use-after-free, which would kill this process.
    result = run_host_script(
        tmp_path,
        """
        import threading
        host.inject(delay_s=0.5)
        req = sim.post(0, [1])
        results = []
        pumper = threading.Thread(target=lambda: results.append(host.pump()))
        pumper.start()
        deadline = time.perf_counter() + 5.0
        # busy_episode, a lock-free word: a snapshot would now wait for the pump (it holds caller_mutex_), and the
        # close below must land while the pump is inside the read.
        while host.busy_episode() == 0 and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert host.busy_episode() != 0
        host.stop()  # closes the handle while the pump is inside a read
        pumper.join(timeout=10)
        assert results == [1] and sim.served(req), results
        print("ok")
        """,
    )
    assert result.returncode == 0 and "ok" in result.stdout, (result.returncode, result.stderr[-2000:])


def test_stop_live_closes_every_host_even_when_one_fails(tmp_path, monkeypatch, capsys):
    hosts = []
    for i in range(2):
        (tmp_path / str(i)).mkdir()
        s = ram_miss_setup(tmp_path / str(i))
        hosts.append(
            ExpertStreamHost(
                s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32)
            )
        )

    def broken():
        raise RuntimeError("counters broke")

    monkeypatch.setattr(hosts[0], "counters", broken)
    monkeypatch.setattr(hosts[1], "counters", broken)
    expert_stream_transport._stop_live()
    assert not hosts[0]._close.alive and not hosts[1]._close.alive
    assert "counters broke" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
