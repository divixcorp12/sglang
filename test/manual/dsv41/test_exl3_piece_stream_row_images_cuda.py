"""Piece streaming's device side over the reader's direct mode (row images, plan 2026-09-24-dsv41-row-images Part B.6).

Every test of test_exl3_piece_stream_cuda runs again here with the service's tables reading row images through
O_DIRECT readv straight into the (cudaHostRegister'd) pinned slabs, so S, the stages and the fused consumer see
pieces the drive wrote and the reader published without a copy. The images are built by the fixtures' reference
builder beside the fake checkpoint; the harness is otherwise unchanged (its ``pack_workers`` are ignored by the direct
mode, which starts no pool).

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

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost  # noqa: E402
from sglang.srt.layers.moe import exl3_ram_miss as ram_miss  # noqa: E402

pytestmark = cuda_suite.pytestmark


@pytest.fixture(autouse=True)
def row_images(monkeypatch):
    from sglang.srt.layers.moe.exl3_row_image import open_row_images
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
    from sglang.test.dsv41_ram_miss_fixtures import write_row_images

    tables_of, host_init = ram_miss.exl3_ram_miss_tables, Exl3RamMissHost.__init__

    def image_tables(layout, segments, slabs, **mirrors):
        assert not mirrors, "the CUDA suite builds its tables without mirror roots"
        source = os.path.dirname(next(iter(layout.records.values())).path)
        root = source.rstrip("/") + "_images"
        write_row_images(layout, segments, source, [root], sorted(slabs))
        images = open_row_images([root], layout, segments, source, sorted(slabs))
        return tables_of(
            layout, segments, slabs, roots=[root], policy=StaticSplitPolicy((1.0,)), source_root=source,
            row_images=images,
        )

    def direct_host(self, tables, *args, direct, **kwargs):
        host_init(self, tables, *args, direct=direct or tables.row_images, **kwargs)

    monkeypatch.setattr(ram_miss, "exl3_ram_miss_tables", image_tables)
    monkeypatch.setattr(Exl3RamMissHost, "__init__", direct_host)
    yield


def test_the_harness_reads_row_images_with_o_direct(tmp_path):
    """The fixture is what this file adds: without it every test below is a plain rerun of the bounce path."""
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


for _name, _test in list(vars(cuda_suite).items()):
    if _name.startswith("test_") and inspect.isfunction(_test):
        _clone = types.FunctionType(_test.__code__, _test.__globals__, _name, _test.__defaults__, _test.__closure__)
        _clone.__kwdefaults__ = _test.__kwdefaults__
        _clone.__dict__.update({k: list(v) if k == "pytestmark" else v for k, v in _test.__dict__.items()})
        globals()[_name] = _clone


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
