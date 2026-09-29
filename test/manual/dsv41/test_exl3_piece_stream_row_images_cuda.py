"""Piece streaming's device side over the reader's direct mode (row images, plan 2026-09-24-dsv41-row-images Part B.6).

Every test of test_exl3_piece_stream_cuda is collected here too, reading row images through O_DIRECT readv straight
into the (cudaHostRegister'd) pinned slabs, so S, the stages and the fused consumer see pieces the drive wrote and the
reader published without a copy. Since the packed path was deleted (plan 2026-09-29-hotpath-zero-overhead D4) that
suite's harness builds row images itself, so this file no longer patches anything: it keeps the GPU run's entry point
(and its test IDs) and adds the two tests below.

Run on divix01 holding cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import inspect
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_exl3_piece_stream_cuda as cuda_suite  # noqa: E402
from test_exl3_piece_stream_cuda import service  # noqa: E402,F401  (fixture of the reused tests)

pytestmark = cuda_suite.pytestmark


def test_the_harness_reads_row_images_with_o_direct(tmp_path):
    """The harness's tables are row images (the reader's only tables), and serving from them starts no pool."""
    s = cuda_suite.StreamService(tmp_path)
    try:
        assert s.tables.row_images and all(path.endswith(".rows") for path in s.tables.paths)
        s.plan([4, 7])
        s.step()
        assert s.keep.item() == 1.0, s.counters()
        assert s.delivered([4, 7])
        assert not any(t.name == "exl3-pack" for t in _threads()), "the direct mode started a packing pool"
    finally:
        s.close()


def _threads():
    out = []
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                out.append(types.SimpleNamespace(name=f.read().strip()))
        except OSError:
            pass
    return out


# test_g1_a_second_layer_streams_its_own_rows_equal_to_the_flag_off_arm was never collected here and is now deleted:
# its precondition, two streamed rows cut differently, cannot hold for row images (identity segments cut every
# (row, expert) alike). What it also covered, a second layer's image file and slabs reaching S, is
# test_a_second_layer_streams_its_own_image_rows below.
NOT_REUSED = set()


def test_a_second_layer_streams_its_own_image_rows(tmp_path):
    """Streamed row 1 of 2 in the direct mode: the reader must read layer 1's image file into layer 1's slabs and S
    copy them; reading layer 0's file (the same experts, other bytes) or copying layer 0's slabs shows in the bytes."""
    experts = [3, 5, 9, 12]
    s = cuda_suite.StreamService(tmp_path, layers=2, row=1)
    try:
        assert s.tables.paths[1].endswith("layer-001.rows")
        s.host.inject_fault(pack_delay_ns=cuda_suite.PIECE_DELAY_NS, poison=True)
        s.plan(experts)
        s.step()
        assert s.keep.item() == 1.0, (s.counters(), s.stats())
        assert int(s.dev.go_2.item()) == len(experts) and s.stats()["stream_pieces"] > 0
        assert s.delivered(experts)
    finally:
        s.quiet()
        s.close()

for _name, _test in list(vars(cuda_suite).items()):
    if _name.startswith("test_") and inspect.isfunction(_test) and _name not in NOT_REUSED:
        _clone = types.FunctionType(_test.__code__, _test.__globals__, _name, _test.__defaults__, _test.__closure__)
        _clone.__kwdefaults__ = _test.__kwdefaults__
        _clone.__dict__.update({k: list(v) if k == "pytestmark" else v for k, v in _test.__dict__.items()})
        globals()[_name] = _clone


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
