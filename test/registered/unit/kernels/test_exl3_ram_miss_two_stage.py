"""Two-stage CPU misses (SGLANG_DSV41_CPU_TWO_STAGE; CPU).

A CPU-computed miss's row image is w13 then w2 (EXL3_STREAMED_NAMES order). With the flag on the reader cuts such a row
in two spans, w13 over every mirror root and then w2 over every root, queued in that order, and reports when the first
span's pieces have all landed. The tier then sends the CPU job at once, as a staged job: its forward runs the gate/up
GEMVs and waits on worker 0 for the tier to open its second stage, once the rows are whole, before it reads any w2
byte. Rows for the GPU copy path, speculative reads and fills keep the one-span cut.

The reader tests run the C++ reader directly (read_rows_sqes, read_rows_two_span, piece_geometry); the tier tests
run the instr build's fake CPU kernel over a layer of the tier's own slabs, which records the byte sums of its slots'
w13 at its start and of their w2 after its stage-two wait: with the poison fault, a forward that read either before it
landed shows the poison.
"""

import dataclasses
import hashlib
import json
import os
import time

import pytest
import torch

import test_exl3_ram_miss_split as split
from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import (
    new_page,
    piece_geometry,
    read_rows_sqes,
    read_rows_traced,
    read_rows_two_span,
)
from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, attached_host, ram_miss_setup, run_host_script

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

PAGE = 4096
FIRST, SECOND = EXL3_STREAMED_NAMES[:3], EXL3_STREAMED_NAMES[3:]
SHAPES = [(1.0,), (1.0, 1.0), (1.0, 1.0, 1.0), (16.0, 10.0, 16.0), (0.0, 1.0), (1.0, 0.0, 1.0)]
SHAPE_IDS = ["one_root", "halves", "thirds", "drive_speed", "zero_first", "zero_middle"]


def _segments(tables):
    """(name, dst, src, bytes) per segment."""
    return [tuple(int(v) for v in line) for line in tables.segments.tolist()]


def _split_at(tables):
    """Where w2 begins in a row image, and the page the first span ends on."""
    start = min(src for name, _, src, _ in _segments(tables) if EXL3_STREAMED_NAMES[name] in SECOND)
    assert all(src + size <= start for name, _, src, size in _segments(tables) if EXL3_STREAMED_NAMES[name] in FIRST)
    return start, -(-start // PAGE) * PAGE


def _need_end(tables):
    return max(src + size for _, _, src, size in _segments(tables))


# ---- Flag off: today's cut, pinned ----

# piece_geometry and the SQE stream of a 4-row piece-streamed read, over the three-root shapes production runs, as the
# reader produced them at 289d28fa01 (before two spans existed). Regenerate only at a commit whose cut is known good.
THREE_ROOT = {"thirds": (1.0, 1.0, 1.0), "drive_speed": (16.0, 10.0, 16.0), "zero_middle": (1.0, 0.0, 1.0)}
GOLDEN = {
    "thirds": "5fb0d0309c7cb5482f20712a26fa674becdc4969181a344605dc317932d63b7d",
    "drive_speed": "0419131587efb7f29919b710c8dc7a0aea703fa8f21a041f25015cb0935b7870",
    "zero_middle": "30fcc7d14fbe036e0a0b6cce491ef3fcf26852cc9d775c73d8c5491a9045fc49",
}


def _today_digest(tmp_path, weights):
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
    h = hashlib.sha256()
    layers, experts = s.tables.extents.shape[:2]
    for row in range(layers):
        for expert in range(experts):
            h.update(json.dumps(piece_geometry(s.tables, row, expert), sort_keys=True).encode())
    result, log, info, record = read_rows_sqes(s.tables, 1, [4, 1, 2, 0], [0, 1, 2, 3], piece_stream=True)
    assert result == 1
    h.update(json.dumps(log).encode())
    return h.hexdigest()


@pytest.mark.parametrize("name", sorted(THREE_ROOT))
def test_with_no_two_span_row_the_cut_and_the_sqes_are_todays(tmp_path, name):
    assert _today_digest(tmp_path, THREE_ROOT[name]) == GOLDEN[name]


# ---- The two-span cut ----


@pytest.mark.parametrize("weights", SHAPES, ids=SHAPE_IDS)
def test_a_two_span_row_reads_its_image_once_w13_over_every_root_first(tmp_path, weights):
    """Every row: the sub-reads tile [0, need_end) once, in dest order; the first span's (k 0) end on the page after
    w13, before every second-span (k 1) one; each reads its own root's file at the row's base; each root's share of
    each span is its weight's, to a page or two; and the first span's pieces hold every w13 byte."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
    w13_end, at = _split_at(s.tables)
    need_end = _need_end(s.tables)
    segments = _segments(s.tables)
    total = sum(weights)
    layers, experts = s.tables.extents.shape[:2]
    for row in range(layers):
        for expert in range(experts):
            subs, pieces = piece_geometry(s.tables, row, expert, two_span=True)
            extents = s.tables.extents[row, expert].tolist()
            spans = [sub["k"] for sub in subs]
            assert spans == sorted(spans) and set(spans) == {0, 1}, (row, expert, spans)
            cursor = 0
            for sub in subs:
                assert sub["dest"] == cursor and sub["length"] > 0
                cursor += sub["length"]
                file, offset, length, dest = extents[sub["part"]]
                assert length > 0 and sub["file"] == file and sub["offset"] - sub["dest"] == offset - dest
            assert cursor == need_end
            first = [sub for sub in subs if sub["k"] == 0]
            assert first[-1]["dest"] + first[-1]["length"] == at
            for span, (lo, hi) in enumerate([(0, at), (at, need_end)]):
                for part, weight in enumerate(weights):
                    got = sum(sub["length"] for sub in subs if sub["k"] == span and sub["part"] == part)
                    assert abs(got - (hi - lo) * weight / total) <= 2 * PAGE, (row, expert, span, part, got)
            # Every byte of every w13 segment is in a piece of the first span, none in the second's.
            for i, (name, dst, src, size) in enumerate(segments):
                if EXL3_STREAMED_NAMES[name] not in FIRST:
                    continue
                held = sum(hi - lo for piece in pieces[: len(first)] for lo, hi in [piece["runs"][i]])
                assert held == size, (row, expert, EXL3_STREAMED_NAMES[name])
            assert piece_geometry(s.tables, row, expert, two_span=False) == piece_geometry(s.tables, row, expert)
    assert w13_end <= at < need_end


def _rows_sqes(log, tables, slots):
    """The SQEs of each row ordinal (bounce slot = ordinal for one batch), as (index, file, dest, length)."""
    rows = {o: [] for o in range(len(slots))}
    for index, (file, offset, length, bounce) in enumerate(log):
        rows[bounce // tables.slot_bytes].append((index, file, bounce % tables.slot_bytes, length))
    return rows


@pytest.mark.parametrize("credit", [0, 1])
@pytest.mark.parametrize("weights", [(1.0, 1.0, 1.0), (16.0, 10.0, 16.0)], ids=["thirds", "drive_speed"])
def test_every_root_reads_a_two_span_rows_w13_before_any_of_its_w2(tmp_path, weights, credit):
    """Rows 0, 1 and 3 of four in two spans, row 2 not: every SQE of a two-span row's first span (dest below the
    split) is prepared before any of its second span's, on every root, also with one read in flight at a time; the
    one-span row reads today's sub-reads in today's order; every row lands exact. Mutant: queue the second span first
    -- red."""
    s = ram_miss_setup(tmp_path, capacity=4, mirror_weights=weights, hidden=256, inter=512)
    _, at = _split_at(s.tables)
    experts, slots = [4, 1, 2, 0], [0, 1, 2, 3]
    for slot in slots:
        split._sentinel(s, 1, slot)
    result, log, info, record = read_rows_sqes(
        s.tables, 1, experts, slots, piece_stream=True, two_span_rows=0b1011, max_outstanding=credit
    )
    assert result == 1
    rows = _rows_sqes(log, s.tables, slots)
    for o in (0, 1, 3):
        first = [index for index, _, dest, _ in rows[o] if dest < at]
        second = [index for index, _, dest, _ in rows[o] if dest >= at]
        assert first and second and max(first) < min(second), (o, rows[o])
        assert all(dest + length <= at for _, _, dest, length in rows[o] if dest < at)
        for file in {f for _, f, _, _ in rows[o]}:
            mine = [(index, dest) for index, f, dest, _ in rows[o] if f == file]
            assert [dest < at for _, dest in mine] == sorted((dest < at for _, dest in mine), reverse=True)
    today = [(sub["file"], sub["dest"]) for sub in piece_geometry(s.tables, 1, experts[2])[0]]
    assert [(file, dest) for _, file, dest, _ in rows[2]] == today
    split._assert_rows(s, 1, experts, slots)


# ---- The first span's signal ----


def _reference(s, experts, ref_slots):
    assert read_rows_traced(s.tables, 1, experts, ref_slots)[0] == 1
    return s.tables.slabs


@pytest.mark.parametrize("part", [0, 2])
def test_a_rows_w13_is_reported_only_once_every_roots_first_span_landed(tmp_path, part):
    """Root `part`'s first span of row 0 is held until everything else has landed, the slots poisoned. Each two-span
    row is reported once, and when it is its w13 bytes are the reference's on every root. Mutant: report the first
    span at its first piece -- red (root 2's w13 is still poison)."""
    s = ram_miss_setup(tmp_path, capacity=8, mirror_weights=(1.0, 1.0, 1.0), hidden=256, inter=512)
    experts, slots, ref_slots = [4, 1, 2], [0, 1, 2], [5, 6, 7]
    reference = _reference(s, experts, ref_slots)
    out = read_rows_two_span(
        s.tables, 1, experts, slots, reference=reference, ref_slots=ref_slots, two_span_rows=0b111, poison=True,
        hold_ordinal=0, part=part, sub=0,
    )
    assert out["result"] == 1 and out["reported"] == 3 and out["differed"] == 0, out
    assert out["in_flight"] == 0 and out["split"] == _split_at(s.tables)[0]
    split._assert_rows(s, 1, experts, slots)


def test_a_rows_w13_is_reported_while_its_w2_is_still_landing(tmp_path):
    """The second spans' completions are held 50 ms: rows 0 and 2 (two spans) are reported before they are whole, with
    their w13 in place; row 1 (one span) is never reported."""
    s = ram_miss_setup(tmp_path, capacity=8, mirror_weights=(1.0, 1.0, 1.0), hidden=256, inter=512)
    experts, slots, ref_slots = [4, 1, 2], [0, 1, 2], [5, 6, 7]
    reference = _reference(s, experts, ref_slots)
    out = read_rows_two_span(
        s.tables, 1, experts, slots, reference=reference, ref_slots=ref_slots, two_span_rows=0b101, poison=True,
        suffix_delay_ns=50_000_000,
    )
    assert out["result"] == 1 and out["reported"] == 2 and out["early"] == 2 and out["differed"] == 0, out
    split._assert_rows(s, 1, experts, slots)


def test_a_failed_w2_read_after_w13_landed_fails_the_read_and_leaves_no_read_counted(tmp_path):
    """Root 1's second span of row 0 fails (EIO) after the row's w13 was reported: the read fails like any failed row,
    and no sub-read stays counted in flight, in the reader or its drive load."""
    s = ram_miss_setup(tmp_path, capacity=8, mirror_weights=(1.0, 1.0, 1.0), hidden=256, inter=512)
    experts, slots, ref_slots = [4, 1, 2], [0, 1, 2], [5, 6, 7]
    reference = _reference(s, experts, ref_slots)
    out = read_rows_two_span(
        s.tables, 1, experts, slots, reference=reference, ref_slots=ref_slots, two_span_rows=0b111,
        suffix_delay_ns=20_000_000, part=1, sub=1, ordinal=0, part_error=5,
    )
    assert out["result"] == 0 and out["early"] >= 1 and out["differed"] == 0, out
    assert out["in_flight"] == 0
    for root in out["drive_load"]["roots"]:
        assert root["demand_reads"] == 0 and root["demand_inflight_bytes"] == 0, out["drive_load"]


# ---- The tier: staged CPU jobs ----

ROW = 1
ROWS = 2
HIDDEN = 8
NO_SPLIT = [0] * (lease.wire_layout(8).lanes + 1)
SLOW_W2_NS = 200_000_000


def _wait(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _cores() -> list[int]:
    return sorted(os.sched_getaffinity(0))[:2]


def _split(**entries):
    out = list(NO_SPLIT)
    for n, k in entries.items():
        out[int(n[1:])] = k
    return out


def slab_layer(s, row: int) -> CpuExpertLayerSpec:
    """A fake-kernel layer over the tier's own slabs of `row`, in EXL3_STREAMED_NAMES order: the fake then sums the
    bytes its slots hold (FakeKernel)."""
    slabs = tuple(
        (s.slabs[row][name].data_ptr(), s.slabs[row][name][0].numel() * s.slabs[row][name].element_size())
        for name in EXL3_STREAMED_NAMES
    )
    capacity = int(s.slabs[row][EXL3_STREAMED_NAMES[0]].shape[0])
    return CpuExpertLayerSpec(capacity=capacity, hidden=HIDDEN, intermediate=0, act_limit=0.0, slabs=slabs,
                              params=b"", keep=tuple(s.slabs[row].values()))


def _sums(s, expert):
    reference = s.reference(ROW, [expert])
    total = lambda names: float(sum(int(reference[n][0].view(torch.uint8).sum()) for n in names))  # noqa: E731
    return total(FIRST), total(SECOND)


def _host(tmp_path, *, two_stage, weights=(1.0, 1.0, 1.0)):
    s = ram_miss_setup(tmp_path, capacity=7, mirror_weights=weights, hidden=256, inter=512)
    page = new_page(pin=False, wire=wire_layout(8))
    host = attached_host(s, page, k=3)
    host.enable_copy_engine(-1)
    dst = {name: torch.zeros((6,) + tuple(slab.shape[1:]), dtype=slab.dtype) for name, slab in s.slabs[ROW].items()}
    table = torch.tensor(
        [[slab.data_ptr(), dst[name].data_ptr(), slab[0].numel() * slab.element_size()] for name, slab in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, 6)
    host.arm_copy_engine()
    x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 2, HIDDEN), dtype=torch.float32)
    host.enable_cpu_experts(host.test_kernel_address(), _split(n2=2), _cores(), x_rows, out_rows, threads=2,
                            two_stage=two_stage)
    host.set_cpu_layer(ROW, slab_layer(s, ROW))
    return s, host, ChainSim(host, page, s.slabs), out_rows


def _serve_two_cpu_misses(s, host, sim, **fault):
    if fault:
        host.inject_fault(**fault)
    req = sim.post(ROW, [5, 4], dst=[0, 1], captured=True, cpu_on=True, cpu_misses=True, weights=[0.5, 0.25])
    assert req.kinds == [LaneKind.MISS_CPU, LaneKind.MISS_CPU]
    assert host.pump() == 1
    assert _wait(lambda: sim.copy_done(req) == req.gen, 10.0)
    return req


def _assert_sums_and_part_one(s, host, req, out_rows):
    """Each forward saw its slots' reference w13 at its start and reference w2 after its stage-two wait, and part 1
    holds both misses once."""
    expert_of = dict(zip(req.slots, [5, 4]))
    calls = host.test_kernel_calls()
    assert sorted(slot for c in calls for slot in c["slots"]) == sorted(req.slots)
    for call in calls:
        want = [_sums(s, expert_of[slot]) for slot in call["slots"]]
        assert call["w13"] == sum(w for w, _ in want), call
        assert call["w2"] == sum(w for _, w in want), call
    expected = torch.arange(HIDDEN, dtype=torch.float32) + sum(w * (slot + 1) for slot, w in zip(req.slots, [0.5, 0.25]))
    assert torch.equal(out_rows[ROW, 1], expected)
    return calls


def test_a_two_stage_cpu_miss_starts_on_w13_and_reads_w2_only_after_it_landed(tmp_path):
    """Two CPU misses, their w2 held 200 ms after their w13 landed, every slot poisoned until read. The CPU jobs are
    staged and start before w2 lands (they wait for it, the counters show the wait), yet each forward sees the
    reference w13 at its start and the reference w2 after its wait, and part 1 sums both. Mutants: the stage-two wait
    returns at once -- red on the w2 sums and the wait; the tier never sends a job before its row is whole -- red on
    staged and the wait."""
    s, host, sim, out_rows = _host(tmp_path, two_stage=True)
    try:
        req = _serve_two_cpu_misses(s, host, sim, poison=True, suffix_delay_ns=SLOW_W2_NS)
        calls = _assert_sums_and_part_one(s, host, req, out_rows)
        assert all(c["staged"] for c in calls)
        stats = host.cpu_stats()
        assert stats["staged_jobs"] == stats["jobs"] == len({tuple(c["slots"]) for c in calls}) >= 1, stats
        assert stats["stage_two_waits"] >= 1 and stats["stage_two_wait_ns"] >= SLOW_W2_NS // 2, stats
        assert host.copy_engine_idle(5.0)
        load = host.drive_load()
        assert all(r["demand_reads"] == 0 and r["demand_inflight_bytes"] == 0 for r in load["roots"]), load
    finally:
        host.inject_fault()
        host.stop()


def test_with_the_flag_off_a_cpu_miss_runs_once_its_row_is_whole(tmp_path):
    """The same record with SGLANG_DSV41_CPU_TWO_STAGE off: no job is staged, none waits, and the sums are the
    reference's."""
    s, host, sim, out_rows = _host(tmp_path, two_stage=False)
    try:
        req = _serve_two_cpu_misses(s, host, sim, poison=True, suffix_delay_ns=SLOW_W2_NS)
        calls = _assert_sums_and_part_one(s, host, req, out_rows)
        assert not any(c["staged"] for c in calls)
        stats = host.cpu_stats()
        assert stats["staged_jobs"] == stats["stage_two_waits"] == stats["stage_two_wait_ns"] == 0, stats
    finally:
        host.inject_fault()
        host.stop()


def test_cpu_hits_are_never_staged(tmp_path):
    """RAM hits have nothing landing: a record's CPU-hit job runs in one stage with the flag on."""
    s, host, sim, out_rows = _host(tmp_path, two_stage=True)
    try:
        load = sim.post(ROW, [3, 2])
        assert host.pump() == 1 and sim.wait_served(load)
        req = sim.post(ROW, [3, 2], dst=[0, 1], captured=True, cpu_on=True, weights=[0.5, 0.25])
        assert req.kinds == [LaneKind.HIT_CPU, LaneKind.HIT_CPU]
        assert host.pump() == 1
        assert _wait(lambda: sim.copy_done(req) == req.gen)
        calls = host.test_kernel_calls()
        assert calls and not any(c["staged"] for c in calls)
        assert host.cpu_stats()["staged_jobs"] == 0
    finally:
        host.stop()


_SCRIPT = """
        sys.path.insert(0, {here!r})
        from test_exl3_ram_miss_two_stage import HIDDEN, ROW, ROWS, _cores, _split, slab_layer
        from sglang.srt.layers.moe.ram_slot_map import LaneKind
        host.enable_copy_engine(-1)
        dst = {{n: torch.zeros((6,) + tuple(v.shape[1:]), dtype=v.dtype) for n, v in s.slabs[ROW].items()}}
        table = torch.tensor(
            [[v.data_ptr(), dst[n].data_ptr(), v[0].numel() * v.element_size()] for n, v in s.slabs[ROW].items()],
            dtype=torch.int64,
        )
        host.set_copy_table(ROW, table, 6)
        host.arm_copy_engine()
        x_rows = torch.zeros((ROWS, 2 * HIDDEN), dtype=torch.uint8)
        out_rows = torch.zeros((ROWS, 2, HIDDEN), dtype=torch.float32)
        host.enable_cpu_experts(host.test_kernel_address(), _split(n1=1, n2=2), _cores(), x_rows, out_rows, threads=2,
                                two_stage=True)
        host.set_cpu_layer(ROW, slab_layer(s, ROW))
"""


def _script(tmp_path, body, **kw):
    here = os.path.dirname(os.path.abspath(__file__))
    return run_host_script(tmp_path, _SCRIPT.format(here=here) + body, capacity=4, staging=2, **kw)


def test_a_failed_w2_read_of_a_staged_cpu_miss_aborts_as_any_failed_read(tmp_path):
    """The CPU job was sent on w13 and waits for w2, whose read fails: the process aborts with the failed read's FATAL,
    as a one-stage CPU miss's does, and never reaches CopyDone."""
    result = _script(tmp_path, """
        host.inject_fault(suffix_delay_ns=50_000_000, part=0, sub=1, part_error=5)
        req = sim.post(ROW, [5], dst=[1], captured=True, cpu_on=True, cpu_misses=True, weights=[1.0])
        assert req.kinds == [LaneKind.MISS_CPU]
        host.pump()
        print("reached", sim.copy_done(req) == req.gen)
    """)
    assert_aborted(result, "the read failed")


def test_the_watchdog_still_aborts_a_hung_w2_read_while_its_job_waits(tmp_path):
    """The w2 read of a staged CPU miss hangs (held 30 s) on a threaded service: the service watchdog aborts at its
    fatal wait, as for any hung read, while the CPU job spins in its stage-two wait."""
    result = _script(tmp_path, """
        host.start_thread(fatal_wait_s=0.3)
        host.inject_fault(suffix_delay_ns=30_000_000_000)
        sim.post(ROW, [5], dst=[1], captured=True, cpu_on=True, cpu_misses=True, weights=[1.0])
        time.sleep(3.0)
        print("reached")
    """, timeout_s=25)
    assert_aborted(result, "stayed in service")
