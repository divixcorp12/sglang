"""Registered buffers and files in the expert-stream reader, with parallel fan-out (plan
2026-09-28-reader-crtp-uring-registration Task 7). A test-only cap (fault word fixed_chunk_cap) makes the fixture's
small slabs register as many row-aligned chunks; fault word `leg` aims the existing short/error/hold faults at one
leg of a fanned-out read. The real >1 GiB slab is the manual test of Task 8."""

import errno
import os
import subprocess
import sys

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes, read_rows_with_fault
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PREFIX = "SGLANG_EXPERT_STREAM_URING_"
EXPERTS = [10, 3, 7, 0, 11, 5, 1, 8, 2, 9, 4]
SLOTS = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]
CAP = 64 * 1024
# The bounce fixture's slots are 159744 B, past CAP, so the bounce is registered under a cap that still cuts it into
# one-slot chunks (a slot is the bounce region's row): the bounce is chunked as well, and a bounce read stays one leg.
BOUNCE_CAP = 4 * CAP


def _cap(images):
    return CAP if images else BOUNCE_CAP


@pytest.fixture
def uring_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(PREFIX):
            monkeypatch.delenv(key)

    def set_(**values):
        for key, value in values.items():
            monkeypatch.setenv(PREFIX + key, str(value))

    return set_


def _setup(tmp_path, images, weights=None):
    root = tmp_path / "ckpt"
    root.mkdir()
    dims = {} if images else dict(hidden=256, inter=512)
    return ram_miss_setup(root, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)


def _snapshot(slabs):
    if isinstance(slabs, dict):
        return {k: _snapshot(v) for k, v in slabs.items()}
    return slabs.clone()


def _equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return same_bytes(a, b)


def _read(s, **faults):
    result, log, info, record = read_rows_sqes(s.tables, 1, EXPERTS, SLOTS, direct=False, **faults)
    return result, log, info, record, _snapshot(s.slabs)


def _supported(fn):
    try:
        return fn()
    except RuntimeError as e:
        if "unsupported by the running kernel" in str(e) or "requires liburing 2.10" in str(e):
            pytest.skip(str(e))
        raise


def _pieces(images, on):
    return ({"piece_stream": True} | ({} if images else {"pack_workers": 2})) if on else {}


def test_regions_are_one_per_slab_with_its_row_bytes(tmp_path):
    s = _setup(tmp_path, True)
    regions = ops._table_buffer_regions(s.tables).tolist()
    assert regions and all(nbytes % row == 0 and row > 0 for _, nbytes, row in regions)
    assert len({base for base, _, _ in regions}) == len(regions)


@pytest.mark.parametrize("images", [False, True], ids=["bounce", "images"])
@pytest.mark.parametrize("read_mode", ["fixed", "readv_fixed"])
@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_fanned_out_reads_are_byte_identical(tmp_path, uring_env, images, read_mode, pieces):
    s = _setup(tmp_path, images)
    base_result, base_log, _, base_rec, base_bytes = _read(s, **_pieces(images, pieces))
    uring_env(READ_MODE=read_mode)
    result, log, info, rec, fixed_bytes = _supported(
        lambda: _read(s, fixed_chunk_cap=_cap(images), **_pieces(images, pieces)))
    assert base_result == result == 1 and _equal(fixed_bytes, base_bytes)
    assert rec["retried_bytes"] == 0 and rec["submitted_bytes"] == base_rec["submitted_bytes"]
    assert rec["bytes"] == base_rec["bytes"] and rec["useful_bytes"] == base_rec["useful_bytes"]
    assert sum(entry[2] for entry in log) == rec["submitted_bytes"]
    if images:  # a multi-slab image read fans out: more SQEs than logical reads, the same byte ranges in the file
        assert info["fixed_cuts"] > 0 and info["fanout_sqes"] > info["fixed_cuts"]
        assert len(log) == len(base_log) - info["fixed_cuts"] + info["fanout_sqes"]
    else:       # one bounce slot is one row: never fanned out
        assert info["fixed_cuts"] == info["fanout_sqes"] == 0 and sorted(log) == sorted(base_log)


def test_normal_mode_never_fans_out(tmp_path, uring_env):
    s = _setup(tmp_path, True)
    result, _, info, _, _ = _read(s, fixed_chunk_cap=CAP)
    assert result == 1 and info["fixed_cuts"] == info["fanout_sqes"] == 0


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_legs_completing_out_of_order(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s, **_pieces(True, pieces))
    uring_env(READ_MODE="readv_fixed")
    # reverse_cqes: every reaped batch (waiting for all in flight) is processed back to front, so each read's legs
    # land last-first.
    result, _, info, _, fixed_bytes = _supported(
        lambda: _read(s, fixed_chunk_cap=CAP, reverse_cqes=True, **_pieces(True, pieces)))
    assert result == 1 and info["fixed_cuts"] > 0 and _equal(fixed_bytes, base_bytes)


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_a_held_leg_keeps_its_read_unretired_and_its_pieces_unpublished(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s, **_pieces(True, pieces))
    uring_env(READ_MODE="readv_fixed")
    # Leg 1 of every read of row 0 is withheld until nothing else is pending: the row may neither retire nor (piece
    # streaming) publish a piece that depends on it before it lands. A reader that retired on the first completion
    # would vet row 0 short and fail the read.
    result, _, _, rec, fixed_bytes = _supported(lambda: _read(
        s, fixed_chunk_cap=CAP, hold_ordinal=0, leg=1, **_pieces(True, pieces)))
    assert result == 1 and _equal(fixed_bytes, base_bytes)
    if pieces:
        assert rec["pieces_published"] == len(EXPERTS) * 8 and rec["piece_publish_refused"] == 0


def test_one_short_leg_resubmits_only_that_leg(tmp_path, uring_env):
    s = _setup(tmp_path, True)
    uring_env(READ_MODE="readv_fixed")
    _, clean_log, clean_info, _, base_bytes = _supported(lambda: _read(s, fixed_chunk_cap=CAP))
    result, log, info, rec, fixed_bytes = _read(s, fixed_chunk_cap=CAP, part=0, part_short=512, leg=1)
    assert result == 1 and _equal(fixed_bytes, base_bytes)
    assert len(log) == len(clean_log) + 1                    # exactly one extra SQE: the short leg's remainder
    extra = sorted(set(log) - set(clean_log))
    assert len(extra) == 1 and rec["retried_bytes"] == extra[0][2]
    assert info["fanout_sqes"] == clean_info["fanout_sqes"]  # first attempts only


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_one_failing_leg_fails_the_read_once_after_every_leg_is_reaped(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    uring_env(READ_MODE="readv_fixed")
    stats, cqes = {}, []
    # Leg 1 of row 0 fails with EIO while its other legs succeed. The read fails once, and the same reader's next read
    # is clean. (unfinished_jobs counts packing jobs, not SQEs, so it cannot see the drain; the drain guard is
    # test_ring_reset_mid_fan_out.)
    first, then = _supported(lambda: read_rows_with_fault(
        s.tables, 1, EXPERTS[:4], SLOTS[:4], EXPERTS[4:], SLOTS[4:], direct=False, part=0, part_error=errno.EIO,
        ordinal=0, leg=1, fixed_chunk_cap=CAP, stats=stats, cqes=cqes, **_pieces(True, pieces)))
    assert (first, then) == (0, 1)
    assert stats["unfinished_jobs"] == 0 and stats["fixed_cuts"] > 0


@pytest.mark.parametrize("submit_first", [False, True], ids=["unconsumed", "in_flight"])
@pytest.mark.parametrize("read_mode", ["fixed", "readv_fixed"])
def test_ring_reset_mid_fan_out(tmp_path, uring_env, capfd, read_mode, submit_first):
    s = _setup(tmp_path, True)
    uring_env(READ_MODE=read_mode, FIXED_FILES=1, DIAGNOSTICS=1)
    # The first submit fails with a fanned-out read's legs prepared: either none reached the kernel (drain rewrites
    # them as NOPs on the same ring) or all did (drain waits for every leg). Either way the clean second read on the
    # same reader succeeds, and the tier was registered exactly once: the diagnostics line (register_ms=) prints once
    # per registration, and a ring reset would print it again (plan 2026-09-29-ring-reset-nop-drain).
    first, then = _supported(lambda: read_rows_with_fault(
        s.tables, 1, EXPERTS[:4], SLOTS[:4], EXPERTS[4:], SLOTS[4:], direct=False,
        submit_error=errno.EIO, submit_call=1, submit_first=submit_first, fixed_chunk_cap=CAP))
    assert (first, then) == (0, 1)
    assert capfd.readouterr().err.count("register_ms=") == 1


@pytest.mark.parametrize("images", [False, True], ids=["bounce", "images"])
@pytest.mark.parametrize("weights", [(1.0, 1.0, 1.0), (1.0, 0.0, 1.0)], ids=["three", "zero_mid"])
def test_fixed_files_with_three_mirror_roots(tmp_path, uring_env, images, weights):
    s = _setup(tmp_path, images, weights)
    base_result, base_log, _, _, base_bytes = _read(s)
    uring_env(FIXED_FILES=1)
    result, log, _, _, fixed_bytes = _read(s)
    assert base_result == result == 1 and sorted(log) == sorted(base_log) and _equal(fixed_bytes, base_bytes)
    roots_read = {str(r) for f, *_ in log for r in s.roots if s.tables.paths[f].startswith(str(r))}
    assert len(roots_read) == sum(1 for w in weights if w > 0)
    uring_env(READ_MODE="readv_fixed")
    result, _, _, _, both_bytes = _supported(lambda: _read(s, fixed_chunk_cap=_cap(images)))
    assert result == 1 and _equal(both_bytes, base_bytes)


def test_registration_refusal_is_a_clear_error_not_a_fallback(tmp_path, uring_env):
    s = _setup(tmp_path, True)
    uring_env(READ_MODE="readv_fixed")
    with pytest.raises(RuntimeError, match="does not fit one registered buffer"):
        _read(s, fixed_chunk_cap=4096)  # the fixture's 49152 B rows exceed a 4 KiB cap
    uring_env(QUEUE_DEPTH=2)
    with pytest.raises(RuntimeError, match="queue depth"):
        _read(s, fixed_chunk_cap=CAP)  # a fanned-out read of up to 6 legs cannot be reserved in a depth-2 ring
    uring_env(QUEUE_DEPTH=0)
    if os.geteuid() == 0:
        pytest.skip("root has CAP_IPC_LOCK: the memlock limit does not bind")
    child = subprocess.run(
        [sys.executable, "-c", _MEMLOCK_CHILD, str(tmp_path / "child")],
        env=dict(os.environ, **{PREFIX + "READ_MODE": "readv_fixed"}), capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "REFUSED" in child.stdout and "RLIMIT_MEMLOCK" in child.stdout, child.stdout


_MEMLOCK_CHILD = r'''
import pathlib, resource, sys
from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
root = pathlib.Path(sys.argv[1]); root.mkdir(parents=True)
s = ram_miss_setup(root, capacity=12, experts=12, row_images=True)
resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))
try:
    read_rows_sqes(s.tables, 1, [0, 1], [0, 1], direct=False)
    print("READ WITHOUT REGISTRATION")  # a silent fallback: the missing REFUSED fails the test
except RuntimeError as e:
    print("REFUSED", e)
'''


@pytest.mark.parametrize("read_mode", ["normal", "readv_fixed"])
def test_a_failed_ring_reset_raises_its_reason_instead_of_aborting(tmp_path, uring_env, read_mode):
    # Final review Important 2. The first submit fails with every SQE still unconsumed, so the failure-path drain
    # resets the ring, and fault word 30 (ring_reset_fail) makes that reset fail. read() must rethrow "io_uring ring
    # reset failed" to its caller; before the fix Quiesce's second drain saw a stale pending count and called
    # std::terminate, losing the reason. The default mode is here too: the reset is not a fixed-mode feature. The
    # read runs in a child, because an abort is a signal, not an exception. The reader is then destroyed (close())
    # on the way out of the call, and the child must go on to exit cleanly.
    if read_mode != "normal":
        uring_env(READ_MODE=read_mode, FIXED_FILES=1)
    child = subprocess.run(
        [sys.executable, "-c", _RESET_CHILD, str(tmp_path / "child"), str(CAP if read_mode != "normal" else 0)],
        capture_output=True, text=True, timeout=120)
    if "unsupported by the running kernel" in child.stdout or "requires liburing 2.10" in child.stdout:
        pytest.skip(child.stdout)
    assert child.returncode == 0, (child.returncode, child.stdout, child.stderr)
    assert "RAISED" in child.stdout and "io_uring ring reset failed" in child.stdout, child.stdout
    assert child.stdout.rstrip().endswith("CLOSED"), child.stdout


_RESET_CHILD = r'''
import errno, pathlib, sys
from sglang.kernels.ops.moe.expert_stream_transport import read_rows_with_fault
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
root = pathlib.Path(sys.argv[1]); root.mkdir(parents=True)
s = ram_miss_setup(root, capacity=12, experts=12, row_images=True)
try:
    result = read_rows_with_fault(
        s.tables, 1, [10, 3, 7, 0], [7, 0, 11, 3], [11, 5], [9, 1], direct=False, submit_error=errno.EIO,
        submit_call=1, ring_reset_fail=True, fixed_chunk_cap=int(sys.argv[2]))
    print("NO ERROR", result)  # the reset did not fail, or its failure was swallowed: the assertion names it
except RuntimeError as e:
    print("RAISED", e)
print("CLOSED")
'''


def test_a_region_with_another_row_size_is_refused_at_open(tmp_path, uring_env, monkeypatch):
    # Final review Minor 2: a slab registered under a region whose row size is not its own used to fail its first read
    # mid-serve ("lies in no registered buffer"); a fixed mode now refuses at open, naming the slab. The normal mode
    # never uses the regions, so it still reads.
    s = _setup(tmp_path, True)
    regions = ops._table_buffer_regions
    halved = lambda tables: regions(tables) * torch.tensor([1, 1, 1]) // torch.tensor([1, 1, 2])  # noqa: E731
    monkeypatch.setattr(ops, "_table_buffer_regions", halved)
    assert _read(s)[0] == 1
    uring_env(READ_MODE="readv_fixed")
    with pytest.raises(RuntimeError, match=r"fixed reads: slab \w+ of layer 0 \(row_bytes \d+\) lies in a registered "
                                           r"buffer region with row_bytes \d+"):
        _supported(lambda: _read(s, fixed_chunk_cap=CAP))
    monkeypatch.setattr(ops, "_table_buffer_regions", lambda tables: regions(tables)[1:])  # one slab unregistered
    with pytest.raises(RuntimeError, match=r"fixed reads: slab \w+ of layer \d+ lies in no registered buffer region"):
        _supported(lambda: _read(s, fixed_chunk_cap=CAP))
