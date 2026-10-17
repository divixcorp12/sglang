"""The page-packed Engram table (engram_packed.py, scripts/dsv41/engram_relayout.py) serves the same rows bit for bit
as the split shard it was converted from, through every reader, and a native-store miss reads one page (CPU)."""

import importlib.util
import json
import os
import shutil
import struct
import time

import numpy as np
import pytest
import torch

from sglang.kernels.ops.embeddings import engram_ring
from sglang.srt.layers import engram_packed
from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.srt.layers.engram_host_node import native_engram_host_node
from sglang.srt.layers.engram_row_cache import EngramRowCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# The real geometry (264 B rows, 15 per page); 1000 rows end in a partial page of 10.
ROWS, DIM, BLOCK = 1000, 256, 32
ROW_BYTES = DIM + DIM // BLOCK
WEIGHT_KEY, SCALE_KEY = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41", "engram_relayout.py")
_spec = importlib.util.spec_from_file_location("engram_relayout", _PATH)
relayout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relayout)


def _write(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, tensor) in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for raw in blobs:
            f.write(raw)


def _store(budget_rows):
    try:
        return native_engram_host_node().create_store(budget_rows * ROW_BYTES, ROW_BYTES)
    except RuntimeError as exc:
        if "io_uring_queue_init failed" in str(exc):
            pytest.skip(f"io_uring is unavailable: {exc}")
        raise


def _register(store, t):
    store.register_table(
        t.path, t.weight_offset, t.scale_offset, t.num_embeddings, t.dim, t.dim // t.block, t._tag, t.rows_per_page
    )


@pytest.fixture(scope="module")
def tables(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("engram-packed")
    torch.manual_seed(0)
    weight = (torch.randn(ROWS, DIM) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(110, 130, (ROWS, DIM // BLOCK), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    split_dir, packed_dir = root / "split", root / "packed"
    split_dir.mkdir()
    packed_dir.mkdir()
    split = split_dir / "model-00047-of-00048.safetensors"
    # The padding puts the weights off a page and off a row boundary, as in the real shard (data at byte 664).
    _write(split, {
        "layers.1.engram.test_padding": ("U8", torch.zeros(9, dtype=torch.uint8)),
        WEIGHT_KEY: ("F8_E4M3", weight),
        SCALE_KEY: ("F8_E8M0", scale),
    })
    packed = packed_dir / "engram-layer1-packed.safetensors"
    monkeypatch_module.setattr(relayout, "capture_running", lambda: [])
    # Small chunks, so the conversion and the verify each run several chunks and a partial last one.
    rc = relayout.main([
        "--source", str(split), "--layer", "1", "--out", str(packed), "--chunk-pages", "4",
        "--same-drive", "--lock", "", "--hash-threads", "2",
    ])
    assert rc == 0
    expected = torch.cat([weight.view(torch.uint8), scale.view(torch.uint8)], dim=1).numpy()
    old = EngramFileTable(str(split), WEIGHT_KEY, SCALE_KEY, ROWS, DIM, cache_tag=1)
    new = EngramFileTable(str(packed), WEIGHT_KEY, SCALE_KEY, ROWS, DIM, cache_tag=1)
    return old, new, expected, packed


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_no_row_crosses_a_page_and_the_geometry_is_15_rows_per_page():
    g = engram_packed.PackedGeometry(384006168, DIM, DIM // BLOCK)
    assert (g.row_bytes, g.rows_per_page, g.pages) == (264, 15, 25600412)
    assert engram_packed.PAGE_BYTES - g.rows_per_page * g.row_bytes == 136
    for row in list(range(64)) + [g.rows - 1]:
        assert g.row_offset(row) % 4096 + g.row_bytes <= 4096


def test_the_converted_file_is_page_aligned_padded_and_verified(tables):
    old, new, expected, packed = tables
    g = engram_packed.PackedGeometry(ROWS, DIM, DIM // BLOCK)
    assert os.path.getsize(packed) == engram_packed.HEADER_BYTES + g.pages * 4096
    data_start, header = engram_packed.read_header(str(packed))
    assert data_start == 4096 and new.weight_offset == 4096 and new.scale_offset == 4096 + DIM
    assert new.rows_per_page == 15 and old.rows_per_page == 0
    pages = np.fromfile(packed, np.uint8, offset=4096).reshape(g.pages, 4096)
    assert not pages[:, 15 * ROW_BYTES :].any()
    assert not pages[-1, (ROWS % 15) * ROW_BYTES :].any()
    manifest = json.load(open(str(packed) + ".manifest.json"))
    assert manifest["verify"]["rows_differing"] == 0 and manifest["verify"]["checksum_matches"]
    assert header["__metadata__"]["data_sha256"] == manifest["convert"]["data_sha256"]
    assert not os.path.exists(str(packed) + ".partial")


def test_memmap_rows_and_dequantized_lookups_are_bit_identical(tables):
    old, new, expected, _ = tables
    ids = torch.from_numpy(np.random.default_rng(0).integers(0, ROWS, size=(3, 97)))
    flat = ids.reshape(-1).numpy()
    for t in (old, new):
        w, s = t._rows(flat)
        assert np.array_equal(np.concatenate([w, s], axis=1), expected[flat])
    assert torch.equal(old.lookup(ids), new.lookup(ids))


def test_the_python_row_cache_reads_packed_pages(tables):
    old, new, expected, _ = tables
    t = EngramFileTable(
        new.path, WEIGHT_KEY, SCALE_KEY, ROWS, DIM,
        cache=EngramRowCache(capacity_rows=64, row_bytes=ROW_BYTES), cache_tag=1, direct=False,
    )
    rng = np.random.default_rng(1)
    for _ in range(5):
        flat = rng.integers(0, ROWS, size=200)
        w, s = t._rows(flat)
        assert np.array_equal(np.concatenate([w, s], axis=1), expected[flat])


@pytest.mark.parametrize("budget_rows", [0, 64])
def test_the_native_store_serves_identical_rows_with_one_page_per_miss(tables, budget_rows):
    old, new, expected, _ = tables
    split_store, packed_store = _store(budget_rows), _store(budget_rows)
    _register(split_store, old)
    _register(packed_store, new)
    rng = np.random.default_rng(2)
    for _ in range(20):
        ids = rng.integers(-1, ROWS, size=rng.integers(1, 120)).astype(np.int64)  # repeats, and -1 dead rows
        got_old, got_new = split_store.lookup(ids, old._tag), packed_store.lookup(ids, new._tag)
        assert np.array_equal(got_old, got_new)
        want = expected[ids.clip(0)].copy()
        want[ids < 0] = 0
        assert np.array_equal(got_new, want)
    # Rows on distinct pages: one 4 KiB read each, against at least two for the split shard.
    store, split = _store(0), _store(0)
    _register(store, new)
    _register(split, old)
    ids = (np.arange(0, ROWS // 15, 3) * 15 + rng.integers(0, 15, size=len(range(0, ROWS // 15, 3)))).astype(np.int64)
    store.lookup(ids, new._tag)
    split.lookup(ids, old._tag)
    s, o = dict(store.stats()), dict(split.stats())
    assert s["unique_misses"] == len(ids)
    assert s["submitted_sqes"] == len(ids) and s["direct_read_bytes"] == 4096 * len(ids)
    assert o["direct_read_bytes"] > s["direct_read_bytes"]


def test_lookup_contexts_take_the_layout_and_refuse_a_mismatch(tables):
    old, new, expected, _ = tables
    native = native_engram_host_node()
    store = _store(0)
    _register(store, new)
    n = 7
    ids = np.array([0, 14, 15, 999, -1, 500, 14], dtype=np.int64)
    rows = np.full((n, ROW_BYTES), 0xA5, dtype=np.uint8)
    status = np.full((1,), 9, dtype=np.int32)
    args = (store, new.path, new.weight_offset, new.scale_offset, ROWS, DIM, DIM // BLOCK, new._tag, n,
            ids.ctypes.data, rows.ctypes.data, status.ctypes.data, 0, 0)
    context = native.EngramHostLookup(*args, new.rows_per_page)
    context.run()
    want = expected[ids.clip(0)].copy()
    want[ids < 0] = 0
    assert status[0] == 0 and np.array_equal(rows, want)
    # Without the layout the context would register the file as split and read wrong rows; it is refused.
    with pytest.raises(ValueError, match="another layout"):
        native.EngramHostLookup(*args)

    ring_ids = torch.zeros(n, dtype=torch.int64)
    ring_rows = torch.zeros(n, ROW_BYTES, dtype=torch.uint8)
    control = torch.zeros(engram_ring.CONTROL_WORDS, dtype=torch.int32)
    ring = native.RingLookup(
        store, new.path, new.weight_offset, new.scale_offset, ROWS, DIM, DIM // BLOCK, new._tag, n,
        ring_ids.data_ptr(), ring_rows.data_ptr(), control.data_ptr(), new.rows_per_page,
    )
    try:
        ring_ids.copy_(torch.from_numpy(ids))
        control[engram_ring.POST_SEQ] = 1
        deadline = time.monotonic() + 10
        while int(control[engram_ring.DONE_SEQ]) != 1:
            assert time.monotonic() < deadline
            time.sleep(0.0002)
        assert int(control[engram_ring.STATUS]) == 0
        assert np.array_equal(ring_rows.numpy(), want)
    finally:
        ring.close()


def test_an_unfinished_or_mismatched_file_is_refused(tables, tmp_path):
    _, new, _, packed = tables
    with pytest.raises(ValueError, match="geometry"):
        EngramFileTable(str(packed), WEIGHT_KEY, SCALE_KEY, ROWS + 1, DIM)
    unfinished = tmp_path / "unfinished.safetensors"
    shutil.copy(packed, unfinished)
    geometry = engram_packed.PackedGeometry(ROWS, DIM, DIM // BLOCK)
    with open(unfinished, "r+b") as f:
        f.write(engram_packed.packed_header(1, geometry, {"data_sha256": engram_packed.UNSET_SHA256}))
    with pytest.raises(ValueError, match="did not finish"):
        EngramFileTable(str(unfinished), WEIGHT_KEY, SCALE_KEY, ROWS, DIM)
    store = _store(0)
    with pytest.raises(ValueError, match="rows per page"):
        store.register_table(new.path, 4096, 4096 + DIM, ROWS, DIM, DIM // BLOCK, 1 << 40, 16)


def test_open_finds_a_packed_table_in_the_table_dir(tables):
    _, new, expected, _ = tables
    t = EngramFileTable.open(os.path.dirname(new.path), layer_id=1, num_embeddings=ROWS, dim=DIM)
    assert t.rows_per_page == 15
    w, s = t._rows(np.arange(ROWS))
    assert np.array_equal(np.concatenate([w, s], axis=1), expected)


def test_the_converter_refuses_to_overwrite(tables):
    old, _, _, packed = tables
    with pytest.raises(SystemExit, match="exists"):
        relayout.main(["--source", old.path, "--layer", "1", "--out", str(packed), "--same-drive", "--lock", ""])
