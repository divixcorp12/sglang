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

# (workers, chunks per row): pure offload, rows in parallel, and every row cut across the workers.
MODES = [(1, 1), (3, 1), (3, 3)]


@pytest.fixture(params=MODES, ids=lambda mode: f"w{mode[0]}c{mode[1]}")
def packing(request, monkeypatch):
    """Every reader a reused test builds packs on workers, unless the test names its own ``pack_workers``."""
    workers, chunks = request.param
    fault_tensor, host_init = ops._fault_tensor, Exl3RamMissHost.__init__

    def with_workers(**faults):
        faults.setdefault("pack_workers", workers)
        faults.setdefault("pack_split", chunks)
        return fault_tensor(**faults)

    def init(self, *args, pack_workers=None, **kwargs):
        host_init(self, *args, pack_workers=workers if pack_workers is None else pack_workers, **kwargs)

    monkeypatch.setattr(ops, "_fault_tensor", with_workers)
    monkeypatch.setattr(Exl3RamMissHost, "__init__", init)
    yield request.param
    gc.collect()  # a host the test dropped closes its files and joins its workers now, not at exit


def _with_packing(test):
    """A copy of ``test`` that runs under the ``packing`` fixture; the original, still collected from its own
    module, is left as it is."""
    clone = types.FunctionType(test.__code__, test.__globals__, test.__name__, test.__defaults__, test.__closure__)
    clone.__kwdefaults__ = test.__kwdefaults__
    clone.__dict__.update({k: list(v) if k == "pytestmark" else v for k, v in test.__dict__.items()})
    return pytest.mark.usefixtures("packing")(clone)


# Tests whose assertions state something only the inline reader guarantees, each replaced below.
NOT_REUSED = {
    # pack_ns is the sum of the rows' spans; with workers the spans overlap, so the sum can exceed the
    # first start to last end. The sum itself is asserted in test_pack_ns_is_the_sum_of_the_rows_spans.
    "test_a_read_records_its_stages_and_bytes",
    "test_a_read_over_several_batches_sums_them",
}


def _reuse(module, wanted):
    for name, test in vars(module).items():
        if name.startswith("test_") and name not in NOT_REUSED and inspect.isfunction(test):
            if wanted(inspect.getsource(test)):
                globals()[name] = _with_packing(test)


_reuse(split, lambda source: "read_rows_traced(" in source or "read_rows_with_fault(" in source)
_reuse(thread, lambda source: ("_host(" in source or "_tier(" in source) and "_run_script" not in source)


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


def test_the_pool_pins_its_workers_to_the_allowed_cores_and_refuses_when_none_is_left():
    module = ops._host_module()
    mine = sorted(os.sched_getaffinity(0) - set(range(64, 72)))[:4]
    out = torch.zeros(4, dtype=torch.int64)
    module.exl3_ram_miss_pack_pool_affinity(_words(mine), 2, out)
    assert _cores(out[0:2]) == set(mine) and _cores(out[2:4]) == set(mine)
    # Only reserved cores: refused before a thread starts.
    with pytest.raises(RuntimeError, match="no core is left"):
        module.exl3_ram_miss_pack_pool_affinity(_words(range(64, 72)), 2, out)


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
    # A credit of 2 spreads the completions over several reaps, so the error at the 8th completion comes
    # after rows have been handed to workers whose copies take 150 ms.
    (results, took) = _elapsed(
        lambda: ops.read_rows_with_fault(
            s.tables, 1, experts, slots, [], [], direct=False, max_outstanding=2,
            cqe_error=EIO, cqe_call=8, pack_delay_ns=DELAY_NS, stats=stats,
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
    the inline reader's for the same read."""
    s = ram_miss_setup(tmp_path, capacity=16, experts=16, mirror_weights=(1.0, 1.0))
    experts, slots = list(range(16)), list(range(16))
    kwargs = dict(direct=False, max_outstanding=5)
    _, inline = read_rows_traced(s.tables, 1, experts, slots, pack_workers=0, **kwargs)
    _, workers = read_rows_traced(s.tables, 1, experts, slots, **kwargs)
    for field in ("rows_reading_max", "pending_max", "batches", "extents", "submitted_bytes", "useful_bytes", "bytes"):
        assert workers[field] == inline[field], field


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
