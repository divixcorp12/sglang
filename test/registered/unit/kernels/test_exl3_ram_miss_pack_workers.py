"""The packing-worker variant of the C++ row reader (CPU): everything the inline reader promises must
hold with the copy on other threads.

Every test of the split and thread suites that goes through the faulted/traced reader entry points
or through a host runs again here with workers packing (the fixture below sets the mode), so a
promise those suites make about the reader is checked for the worker path without a second copy of
the test. The tests written out below are the ones only workers can break.
"""

import errno
import gc
import inspect
import os
import resource
import time
import types

import pytest
import torch

import test_exl3_ram_miss_split as split
import test_exl3_ram_miss_thread as thread
from sglang.kernels.ops.moe import exl3_ram_miss as ops
from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, read_rows_traced
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=240, suite="base-a-test-cpu")

from test_exl3_ram_miss_thread import hang_guard  # noqa: E402,F401  (autouse fixture of the reused thread tests)

# Each fake-checkpoint test leaves 6-8 descriptors open until exit (already true of the split suite, which
# stays under the default limit of 1024); this file runs those tests once per mode, so it lifts the soft limit.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft != resource.RLIM_INFINITY and _soft < 16384:
    resource.setrlimit(resource.RLIMIT_NOFILE, (16384 if _hard == resource.RLIM_INFINITY else min(16384, _hard), _hard))

EIO = errno.EIO
PAGE = split.PAGE

# (workers, chunks per row, piece streaming): pure offload, rows in parallel, and every row cut across the workers;
# then two of them again with SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM's reader (sub-reads, per-piece vetting).
MODES = [(1, 1, False), (3, 1, False), (3, 3, False), (1, 1, True), (3, 3, True)]


def _mode_id(mode):
    return f"w{mode[0]}c{mode[1]}" + ("ps" if mode[2] else "")


@pytest.fixture(params=MODES, ids=_mode_id)
def packing(request, monkeypatch):
    """Every reader a reused test builds packs on workers, unless the test names its own ``pack_workers``, and
    streams pieces in the piece-stream modes whenever it has workers (the flag needs them)."""
    yield from _packing(request.param, monkeypatch)


@pytest.fixture(params=[mode for mode in MODES if not mode[2]], ids=_mode_id)
def packing_without_pieces(request, monkeypatch):
    """``packing``'s modes without piece streaming, for the tests in ONE_READ_PER_PART."""
    yield from _packing(request.param, monkeypatch)


def _packing(mode, monkeypatch):
    workers, chunks, piece_stream = mode
    fault_tensor, host_init = ops._fault_tensor, Exl3RamMissHost.__init__
    monkeypatch.setattr(split, "PIECE_STREAM", piece_stream)

    def with_workers(**faults):
        faults.setdefault("pack_workers", workers)
        faults.setdefault("pack_split", chunks)
        if piece_stream and faults["pack_workers"] > 0:
            faults.setdefault("piece_stream", True)
        return fault_tensor(**faults)

    def init(self, *args, pack_workers=None, **kwargs):
        host_init(self, *args, pack_workers=workers if pack_workers is None else pack_workers, **kwargs)
        if piece_stream and (workers if pack_workers is None else pack_workers) > 0:
            self.enable_piece_stream()

    monkeypatch.setattr(ops, "_fault_tensor", with_workers)
    monkeypatch.setattr(Exl3RamMissHost, "__init__", init)
    yield mode
    gc.collect()  # a host the test dropped closes its files and joins its workers now, not at exit


def _with_packing(test, fixture="packing"):
    """A copy of ``test`` that runs under the ``packing`` fixture (or ``fixture``); the original, still collected
    from its own module, is left as it is."""
    clone = types.FunctionType(test.__code__, test.__globals__, test.__name__, test.__defaults__, test.__closure__)
    clone.__kwdefaults__ = test.__kwdefaults__
    clone.__dict__.update({k: list(v) if k == "pytestmark" else v for k, v in test.__dict__.items()})
    return pytest.mark.usefixtures(fixture)(clone)


# Tests whose assertions state something only the inline reader guarantees, each replaced below.
NOT_REUSED = {
    # pack_ns is the sum of the rows' spans; with workers the spans overlap, so the sum can exceed the
    # first start to last end. The sum itself is asserted in test_pack_ns_is_the_sum_of_the_rows_spans.
    "test_a_read_records_its_stages_and_bytes",
    "test_a_read_over_several_batches_sums_them",
}


# Tests that fault or count one part's single read: a short read (part_short) or an interrupted one of a part. With
# piece streaming a part is up to four sub-reads, and on these tests' rows each is one page, so part_short=PAGE
# never fires. They run in the modes without it; test_exl3_ram_miss_piece_stream has their sub-read counterparts
# (test_a_short_sub_read_*, test_an_interrupted_sub_read_*).
ONE_READ_PER_PART = {
    "test_a_short_read_resubmits_its_own_extent_under_reversed_completions",
    "test_a_short_read_resubmits_only_its_own_extent",
    "test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful",
    "test_a_short_read_in_either_bank_resubmits_only_its_own_extent",
    "test_an_interrupted_read_is_resubmitted_whole_and_counted_as_retried",
}


def _reuse(module, wanted, pieces=True):
    for name, test in vars(module).items():
        if name.startswith("test_") and name not in NOT_REUSED and inspect.isfunction(test):
            if wanted(inspect.getsource(test)):
                fixture = "packing_without_pieces" if name in ONE_READ_PER_PART or not pieces else "packing"
                globals()[name] = _with_packing(test, fixture)


_reuse(split, lambda source: "read_rows_traced(" in source or "read_rows_with_fault(" in source)
# The thread suite's tiers run without lease mode, which a piece-streaming tier refuses per request: they run in the
# modes without it. test_exl3_ram_miss_piece_stream reruns the two-phase suite, whose tiers have leases, with the flag.
_reuse(thread, lambda source: ("_host(" in source or "_tier(" in source) and "_run_script" not in source, pieces=False)


# ---- The flag: off means no thread, and the workers stay off the reserved cores ----


def _pack_threads():
    count = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                count += f.read().strip() == "exl3-pack"
        except OSError:
            pass  # the thread ended between the listing and the read
    return count


def test_the_fixture_puts_a_reused_test_on_the_workers_it_names(tmp_path, packing):
    """The reused tests name no worker count of their own: this is what shows the fixture reached the reader."""
    s = ram_miss_setup(tmp_path, capacity=6)
    stats = {}
    assert ops.read_rows_with_fault(s.tables, 1, [0, 1], [0, 1], [2], [2], direct=False, stats=stats) == (1, 1)
    assert stats["pack_workers"] == packing[0]


def test_the_flag_defaults_to_off():
    assert envs.SGLANG_DSV41_RAM_MISS_PACK_WORKERS.get() == 0


def test_no_worker_thread_exists_unless_asked_for_and_close_joins_them(tmp_path):
    def host(workers):
        (tmp_path / f"w{workers}").mkdir()
        s = ram_miss_setup(tmp_path / f"w{workers}", capacity=3)
        return Exl3RamMissHost(
            s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32),
            direct=False, pack_workers=workers,
        )

    before = _pack_threads()
    off = host(0)
    assert _pack_threads() == before
    on = host(3)
    assert _pack_threads() == before + 3
    on._close()
    assert _pack_threads() == before
    off._close()


def _words(cores):
    words = [0, 0]
    for core in cores:
        words[core // 64] |= 1 << (core % 64)
    return torch.tensor([w - (1 << 64) if w >= 1 << 63 else w for w in words], dtype=torch.int64)


def _cores(words):
    return {c for c in range(128) if (int(words[c // 64]) >> (c % 64)) & 1}


def test_a_worker_may_not_use_cores_64_to_71():
    module = ops._host_module()
    out = torch.zeros(2, dtype=torch.int64)
    module.exl3_ram_miss_pack_worker_cpus(_words(range(0, 72)), out)
    assert _cores(out) == set(range(0, 64))
    module.exl3_ram_miss_pack_worker_cpus(_words([3, 64, 71, 72, 100]), out)
    assert _cores(out) == {3, 72, 100}


def test_the_pool_pins_each_worker_to_its_own_allowed_core_and_refuses_when_too_few_are_left():
    module = ops._host_module()
    mine = sorted(os.sched_getaffinity(0) - set(range(64, 72)))[:4]
    out = torch.zeros(4, dtype=torch.int64)
    module.exl3_ram_miss_pack_pool_affinity(_words(mine), 2, out)
    pinned = [_cores(out[0:2]), _cores(out[2:4])]
    # One CPU each, distinct, allowed: two workers on one CPU would take turns at the same piece.
    assert [len(p) for p in pinned] == [1, 1] and len(pinned[0] | pinned[1]) == 2, pinned
    assert pinned[0] | pinned[1] <= set(mine)
    # Only reserved cores: refused before a thread starts.
    with pytest.raises(RuntimeError, match="no core is left"):
        module.exl3_ram_miss_pack_pool_affinity(_words(range(64, 72)), 2, out)
    # Fewer allowed cores than workers: refused rather than doubled up.
    with pytest.raises(RuntimeError, match="need a core each"):
        module.exl3_ram_miss_pack_pool_affinity(_words(mine[:1]), 2, out)


def _physical_core(cpu):
    base = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
    try:
        with open(base + "physical_package_id") as package, open(base + "core_id") as core:
            return int(package.read()), int(core.read())
    except OSError:
        return None


def test_workers_take_separate_physical_cores_before_hyperthread_siblings():
    """Two workers copying on one core's two hyperthreads share its load and store units, so while the allowed set
    has enough cores the pool gives each worker a whole one."""
    by_core = {}
    for cpu in sorted(os.sched_getaffinity(0) - set(range(64, 72))):
        by_core.setdefault(_physical_core(cpu), []).append(cpu)
    siblings = next((cpus[:2] for core, cpus in by_core.items() if core is not None and len(cpus) >= 2), None)
    others = [cpus[0] for core, cpus in by_core.items() if core is not None and siblings and siblings[0] not in cpus]
    other = others[0] if others else None
    if siblings is None or other is None:
        pytest.skip("needs a core with two allowed hyperthreads, and another core")
    out = torch.zeros(4, dtype=torch.int64)
    ops._host_module().exl3_ram_miss_pack_pool_affinity(_words(siblings + [other]), 2, out)
    pinned = _cores(out[0:2]) | _cores(out[2:4])
    assert len({_physical_core(cpu) for cpu in pinned}) == 2, (siblings, other, pinned)


def _cpu_list(text):
    cpus = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def _allowed_cpus_of(name):
    """The allowed CPUs of each thread of this process named ``name``."""
    found = []
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as comm:
                if comm.read().strip() != name:
                    continue
            with open(f"/proc/self/task/{tid}/status") as status:
                line = next(line for line in status if line.startswith("Cpus_allowed_list:"))
        except OSError:
            continue  # the thread ended between the listing and the read
        found.append(_cpu_list(line.split(":", 1)[1].strip()))
    return found


def test_the_unpinned_service_thread_keeps_off_the_packing_workers_cpus(tmp_path):
    """The workers copy on their CPUs while a read is in service, and the service thread is the one that posts their
    jobs and publishes the pieces: on a worker's CPU it would wait behind that worker's copy."""
    if len(os.sched_getaffinity(0) - set(range(64, 72))) < 3:
        pytest.skip("needs three allowed cores")
    s = ram_miss_setup(tmp_path)
    host = Exl3RamMissHost(
        s.tables, page=new_page(pin=False), slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False,
        pack_workers=2,
    )
    try:
        host.start_thread()
        workers, service = _allowed_cpus_of("exl3-pack"), _allowed_cpus_of("exl3-ram-miss")
        assert len(workers) == 2 and len(service) == 1, (workers, service)
        assert service[0] and not service[0] & (workers[0] | workers[1]), (workers, service)
    finally:
        host.stop()


# ---- Workers copy concurrently, and they copy what the owner vetted ----


def _elapsed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


DELAY_NS = 150_000_000


def test_workers_copy_rows_concurrently_and_a_row_is_cut_across_them(tmp_path):
    """Three rows that are all ready together, each copy made 150 ms slow. Inline packing takes three
    of them in a row; with three workers the copies overlap (a row per worker, or a chunk of every row
    per worker), so the whole read takes about one. Without this a 'worker' that merely runs the same
    serial copy on another thread would pass every byte-exactness test."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    experts, slots = [3, 0, 5], [0, 1, 2]

    def read(workers, chunks):
        for slot in slots:
            split._sentinel(s, 1, slot)  # a chunk nobody copied must show, not be hidden by an earlier read
        return read_rows_traced(
            s.tables, 1, experts, slots, direct=False, pack_delay_ns=DELAY_NS, pack_workers=workers, pack_split=chunks
        )

    (result, _), inline = _elapsed(lambda: read(0, 0))
    assert result == 1 and inline > 2.8 * DELAY_NS / 1e9
    for workers, chunks in [(3, 1), (3, 3), (1, 3)]:
        (result, record), took = _elapsed(lambda: read(workers, chunks))
        assert result == 1
        split._assert_rows(s, 1, experts, slots)
        if workers == 1 and chunks == 3:
            # One worker runs every chunk one after another: no faster than inline.
            assert took > 2.8 * DELAY_NS / 1e9
        else:
            assert took < 1.9 * DELAY_NS / 1e9, (workers, chunks, took)


@pytest.mark.parametrize("workers, chunks", [(2, 64), (1, 100_000), (4, 7)])
def test_a_row_cut_into_many_more_chunks_than_it_has_lines_is_byte_exact(tmp_path, workers, chunks):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    experts, slots = [3, 0, 5, 1], [0, 1, 2, 3]
    result, _ = read_rows_traced(
        s.tables, 1, experts, slots, direct=False, pack_workers=workers, pack_split=chunks, poison=True
    )
    assert result == 1
    split._assert_rows(s, 1, experts, slots)


def test_a_failure_returns_only_when_no_copy_is_still_running(tmp_path, packing):
    """A hard error arrives while workers are still copying already-vetted rows. The caller releases
    every slot when the read returns and the next read reuses the bounce, so a copy still running then
    would write into a released slot and read a bank the next read is filling."""
    s = ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0, 1.0))
    experts, slots = [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]
    for slot in slots:
        split._sentinel(s, 1, slot)
    stats = {}
    # A credit of 2 spreads the completions over several reaps, so the error at the 8th completion (the last read
    # of row 3; with piece streaming, its last sub-read) comes after rows have been handed to workers whose copies
    # take 150 ms.
    cqe_call = split._n(8, s, 1, experts[:4])
    (results, took) = _elapsed(
        lambda: ops.read_rows_with_fault(
            s.tables, 1, experts, slots, [], [], direct=False, max_outstanding=2,
            cqe_error=EIO, cqe_call=cqe_call, pack_delay_ns=DELAY_NS, stats=stats,
        )
    )
    assert results == (0, 0)  # the failed read, and no second one
    assert stats["unfinished_jobs"] == 0
    assert took >= 0.9 * DELAY_NS / 1e9  # it waited for the copies it had started
    split._exact_or_untouched(s, 1, experts, slots)


def test_a_row_short_of_its_segments_is_never_handed_to_a_worker(tmp_path, packing):
    """The coverage check runs on the owner before dispatch. A row it refuses must not be copied at all,
    not copied and then reported failed: the slot stays as it was."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    need_end = int((s.tables.segments[:, 2] + s.tables.segments[:, 3]).max())
    assert int(s.tables.starts[1, 1]) + need_end > 0
    s.tables.extents[1, 1, 1, 2] -= PAGE
    for slot in range(3):
        split._sentinel(s, 1, slot)
    assert ops.read_rows_with_fault(
        s.tables, 1, [0, 1, 2], [0, 1, 2], [5, 4], [3, 4], direct=False, poison=True, pack_delay_ns=20_000_000
    ) == (0, 1)
    assert split._untouched(s, 1, 1)


def test_no_extent_reuses_a_bank_before_the_copies_out_of_it_are_done(tmp_path, packing):
    """12 rows in batches of 4, row 0 held back so its bank cannot free early. Each copy is slow, so a
    bank released when its rows were HANDED to the workers, rather than when their copies were done,
    would let the next batch's reads land in the bank while the copy is still reading it. Every extent
    of the batch that reuses the bank must be submitted after row 0's copy has ended, and the bytes
    must be exact under a poisoned bounce. (With instant copies the two orders cannot be told apart.)"""
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(11, -1, -1)), list(range(12))
    result, record = read_rows_traced(
        s.tables, 1, experts, slots, direct=False, step=4, hold_ordinal=0, pack_delay_ns=20_000_000, poison=True
    )
    assert result == 1 and record["bank_stalls"] >= 1
    _assert_reads_start_after_the_copy_ended(record, reusing_rows=(8, 9, 10, 11), copied_row=0)
    split._assert_rows(s, 1, experts, slots)


def _assert_reads_start_after_the_copy_ended(record, *, reusing_rows, copied_row):
    packs = split._row_packs(record)
    for extent in record["extent_cqe"]:
        if extent["row"] in reusing_rows:
            assert extent["submit"] > packs[copied_row]["end"], (extent, packs[copied_row])


def test_the_ordering_assertion_fires_on_a_record_that_violates_it():
    """The release-at-dispatch mutant cannot reach the assertion above: it deadlocks the reader (or trips one
    of the two guards) before any record exists. So the assertion itself is checked here, on crafted records."""
    row = {"row": 0, "admit": 1, "start": 10, "end": 50}
    ordered = {"row_pack": [row], "extent_cqe": [{"row": 8, "submit": 60}, {"row": 9, "submit": 51}]}
    _assert_reads_start_after_the_copy_ended(ordered, reusing_rows=(8, 9), copied_row=0)
    for early in (49, 50):  # before the copy ended, and at the same instant
        violated = {"row_pack": [row], "extent_cqe": [{"row": 8, "submit": 60}, {"row": 9, "submit": early}]}
        with pytest.raises(AssertionError):
            _assert_reads_start_after_the_copy_ended(violated, reusing_rows=(8, 9), copied_row=0)
    # Rows outside the reusing batch are not held to it.
    other = {"row_pack": [row], "extent_cqe": [{"row": 3, "submit": 5}]}
    _assert_reads_start_after_the_copy_ended(other, reusing_rows=(8, 9), copied_row=0)


def test_held_rows_are_released_together_and_read_byte_exact(tmp_path, packing):
    """hold_rest is the knob the benchmark uses to hand the packer several rows at once."""
    s = ram_miss_setup(tmp_path, capacity=8, experts=8, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(5)), list(range(5))
    result, record = read_rows_traced(s.tables, 1, experts, slots, direct=False, hold_ordinal=1, hold_rest=True)
    assert result == 1
    held = {e["cqe"] for e in record["extent_cqe"] if e["row"] >= 1}
    assert len(held) == 1  # one release: the four held rows completed at the same instant
    assert all(e["cqe"] < min(held) for e in record["extent_cqe"] if e["row"] == 0)
    split._assert_rows(s, 1, experts, slots)


def test_pack_ns_is_the_sum_of_the_rows_spans(tmp_path, packing):
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    result, record = read_rows_traced(s.tables, 1, [5, 0, 2, 3], [0, 1, 2, 3], direct=False, pack_delay_ns=2_000_000)
    assert result == 1
    spans = [row["end"] - row["start"] for row in record["row_pack"]]
    assert all(span > 0 for span in spans) and record["pack_ns"] == sum(spans)
    split._assert_stages_ordered(record)


def test_credit_and_rows_in_flight_do_not_depend_on_where_the_copy_runs(tmp_path, packing):
    """The reader's admission and credit accounting are the owner's alone: the high-water marks match
    the inline reader's for the same read. Piece streaming has no inline reader, so there the reference is
    one worker copying each row whole."""
    s = ram_miss_setup(tmp_path, capacity=16, experts=16, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(16)), list(range(16))
    kwargs = dict(direct=False, max_outstanding=5)
    reference = dict(pack_workers=1, pack_split=1) if packing[2] else dict(pack_workers=0)
    _, inline = read_rows_traced(s.tables, 1, experts, slots, **reference, **kwargs)
    _, workers = read_rows_traced(s.tables, 1, experts, slots, **kwargs)
    assert inline["piece_stream"] == workers["piece_stream"] == int(packing[2])
    for field in ("rows_reading_max", "pending_max", "batches", "extents", "submitted_bytes", "useful_bytes", "bytes"):
        assert workers[field] == inline[field], field


# ---- The record says which mode wrote it (schema 5) ----


@pytest.mark.parametrize("workers, chunks, split", [(0, 0, 0), (1, 0, 1), (3, 0, 3), (3, 1, 1), (3, 3, 3), (2, 5, 5)])
def test_a_stage_record_carries_the_packing_mode_that_produced_it(tmp_path, workers, chunks, split):
    """Inline is pack_workers 0. Without the two fields a worker-mode record is indistinguishable from an inline
    one, and the analysis reads the workers' wake-up delay as a busy packer. ``split`` is what the reader
    keeps: one chunk per worker unless told otherwise."""
    s = ram_miss_setup(tmp_path, capacity=6, mirror_weights=(1.0, 1.0))
    result, record = read_rows_traced(
        s.tables, 1, [3, 0, 5], [0, 1, 2], direct=False, pack_workers=workers, pack_split=chunks
    )
    assert result == 1
    assert (record["pack_workers"], record["pack_split"]) == (workers, split)


def _served_records(tmp_path, workers, rows=(2,)):
    """A demand that reads ``rows``, one that finds them resident (no_read) and an unarmed one (touch), through a
    host built with ``workers``: the three shapes of record a request can end in."""
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    host = Exl3RamMissHost(
        s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False, pack_workers=workers
    )
    host.enable_trace()
    try:
        for _ in range(2):
            seq = ops.sim_post(page, 1, need=list(rows), protect=list(rows))
            assert host.pump() == 1 and ops.sim_wait(page, seq, timeout_s=1.0) == 1
        ops.sim_post(page, 1, need=[], protect=list(rows), armed=False)
        assert host.pump() == 1
        return host.drain_trace()
    finally:
        host.stop()


@pytest.mark.parametrize("workers", [0, 2])
def test_every_record_a_host_pushes_carries_its_mode_including_the_ones_that_read_nothing(tmp_path, workers):
    """The mode is the reader's, not the request's: a no_read or touch record has no packing of its own and
    still says how the run was configured, so a file's mode is the same whichever record is looked at."""
    records = _served_records(tmp_path, workers)
    assert [r["status"] for r in records] == ["served", "no_read", "touch"]
    assert [r["pack_workers"] for r in records] == [workers] * 3
    assert [r["pack_split"] for r in records] == [workers] * 3  # split defaults to one chunk per worker


def test_the_mode_reaches_the_jsonl_a_trace_writes(tmp_path):
    """The analysis reads the file, not the record: the fields must survive the export."""
    import json

    from sglang.srt.layers.moe.exl3_stream_trace import Exl3StreamTrace

    lines = {}
    for workers in (0, 2):
        (tmp_path / f"w{workers}").mkdir()
        records = _served_records(tmp_path / f"w{workers}", workers)
        path = tmp_path / f"trace-w{workers}.jsonl"
        trace = Exl3StreamTrace(str(path))
        try:
            trace.record_ram_miss_requests(records, [10, 11])
        finally:
            trace.close()
        lines[workers] = [json.loads(line) for line in path.read_text().splitlines()]
    for workers, written in lines.items():
        assert len(written) == 3
        assert {(l["request"]["pack_workers"], l["request"]["pack_split"]) for l in written} == {(workers, workers)}


def test_the_analysis_refuses_the_metrics_of_a_trace_a_worker_host_wrote_and_keeps_those_of_an_inline_one(tmp_path):
    """The whole path: a real host's records, through the exporter, into overlap_timeline. Two rows read per
    demand, so the request is multi-row and would otherwise be judged."""
    import sys
    from pathlib import Path

    from sglang.srt.layers.moe.exl3_stream_trace import Exl3StreamTrace

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "analysis" / "dsv41-drive"))
    import overlap_timeline as ot

    results = {}
    for workers in (0, 2):
        (tmp_path / f"w{workers}").mkdir()
        records = _served_records(tmp_path / f"w{workers}", workers, rows=(2, 5, 4))
        path = tmp_path / f"trace-w{workers}.jsonl"
        trace = Exl3StreamTrace(str(path))
        try:
            trace.record_ram_miss_requests(records, [10, 11])
        finally:
            trace.close()
        results[workers] = ot.analyse_file(str(path))
    assert results[0]["judged_requests"] == results[2]["judged_requests"] == 1
    assert results[0]["pack_mode"] == {"inline": 3} and "refused" not in results[0]
    assert "rows_queued_behind_the_packer" in results[0] and "hidden_fraction_of_pack" in results[0]
    assert results[2]["pack_mode"] == {"workers": 3} and "rows_queued_behind_the_packer" not in results[2]
    assert results[2]["refused"]["metrics"] == list(ot.WORKER_MODE_REFUSED)
    assert results[2]["coverage_of_window_by_pack"]["n"] == 1 and results[2]["exposed_tail_us"]["n"] == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
