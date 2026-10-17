"""SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB: the native Engram store at 0 (no cache, every row read through io_uring)
returns the same rows bit for bit as at the default 5 GiB, and its counters give a hit rate (CPU).
SGLANG_DSV41_ENGRAM_SCALES_RAM_NODE: with the scale rows in RAM the store serves the same rows with one read a miss
instead of two (CPU)."""

import ast
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

import sglang.srt.layers.engram_file_table as eft
from sglang.srt.environ import envs
from sglang.srt.layers.engram_file_table import EngramFileTable, native_cache_bytes, native_stats_line
from sglang.srt.layers.engram_host_node import native_engram_host_node
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROWS, DIM, BLOCK = 300, 64, 32
ROW_BYTES = DIM + DIM // BLOCK


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


@pytest.fixture
def table(tmp_path):
    torch.manual_seed(0)
    weight = (torch.randn(ROWS, DIM) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 130, (ROWS, DIM // BLOCK), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    path = tmp_path / "model-00047-of-00048.safetensors"
    _write(path, {
        "layers.1.engram.test_padding": ("U8", torch.zeros(7, dtype=torch.uint8)),
        "layers.1.engram.embed.weight": ("F8_E4M3", weight),
        "layers.1.engram.embed.scale": ("F8_E8M0", scale),
    })
    t = EngramFileTable(str(path), "layers.1.engram.embed.weight", "layers.1.engram.embed.scale", ROWS, DIM)
    packed = torch.cat([weight.view(torch.uint8), scale.view(torch.uint8)], dim=1).numpy()
    return t, packed


def _store(t, budget):
    try:
        store = native_engram_host_node().create_store(budget, ROW_BYTES)
    except RuntimeError as exc:
        if "io_uring_queue_init failed" in str(exc):
            pytest.skip(f"io_uring is unavailable: {exc}")
        raise
    store.register_table(t.path, t.weight_offset, t.scale_offset, ROWS, DIM, DIM // BLOCK, t._tag)
    return store


def test_a_zero_budget_returns_the_same_rows_as_the_default(table):
    t, packed = table
    off, on = _store(t, 0), _store(t, native_cache_bytes())
    rng = np.random.default_rng(1)
    for _ in range(20):
        ids = rng.integers(0, ROWS, size=rng.integers(1, 200)).astype(np.int64)  # repeats within and across calls
        got_off, got_on = off.lookup(ids, t._tag), on.lookup(ids, t._tag)
        assert np.array_equal(got_off, got_on)
        assert np.array_equal(got_on, packed[ids])
    s_off, s_on = dict(off.stats()), dict(on.stats())
    assert s_off["capacity_rows"] == 0 and s_off["cache_bytes"] == 0 and s_off["hits"] == 0
    assert s_off["filled_rows"] == 0 and s_off["evictions"] == 0
    assert s_on["hits"] > 0 and s_on["accesses"] == s_off["accesses"]
    assert s_off["unique_misses"] > s_on["unique_misses"]


@pytest.mark.parametrize("budget", [0, ROWS * ROW_BYTES])
def test_a_dead_row_id_is_a_zero_row_that_counts_and_reads_nothing(table, budget):
    """-1 (engram.DEAD_ROW_ID, a cut verify row) is served zeros without an access, a read or an insert; the live
    ids of the same request are served as before. Any other negative id still fails the request."""
    t, packed = table
    store = _store(t, budget)
    keys = ("accesses", "hits", "unique_misses", "submitted_sqes", "filled_rows", "evictions")
    before = {k: store.stats()[k] for k in keys}
    got = store.lookup(np.array([-1, -1], dtype=np.int64), t._tag)
    assert got.shape == (2, ROW_BYTES) and not got.any()
    assert {k: store.stats()[k] for k in keys} == before
    ids = np.array([5, -1, 7, -1, 5], dtype=np.int64)
    got = store.lookup(ids, t._tag)
    live = ids >= 0
    assert np.array_equal(got[live], packed[ids[live]]) and not got[~live].any()
    after = dict(store.stats())
    assert after["accesses"] - before["accesses"] == 3 and after["unique_misses"] - before["unique_misses"] == 2
    with pytest.raises(RuntimeError, match="lookup failed"):
        store.lookup(np.array([5, -2], dtype=np.int64), t._tag)


def test_a_budget_below_one_row_is_still_refused():
    with pytest.raises(ValueError):
        native_engram_host_node().create_store(ROW_BYTES - 1, ROW_BYTES)


def test_the_budget_defaults_to_5_gib_and_takes_0():
    assert envs.SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB.get() == 5.0 and native_cache_bytes() == 5 << 30
    with envs.SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB.override(0):
        assert native_cache_bytes() == 0
    with envs.SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB.override(-1):
        with pytest.raises(ValueError):
            native_cache_bytes()


def test_open_sizes_the_shared_store_from_the_env():
    tree = ast.parse(Path(eft.__file__).read_text())
    open_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "open")
    text = ast.unparse(open_fn)
    assert "get_shared_store(native_cache_bytes()" in text and "5 << 30" not in text
    assert "log_native_stats(table._native_store)" in text


def test_the_stats_line_has_the_window_and_the_total():
    before = {"accesses": 100, "hits": 40, "unique_misses": 30, "submitted_sqes": 50, "cache_bytes": 0,
              "capacity_rows": 0, "scales_ram_bytes": 0}
    now = {"accesses": 300, "hits": 140, "unique_misses": 80, "submitted_sqes": 110, "cache_bytes": 0,
           "capacity_rows": 0, "scales_ram_bytes": 0}
    line = native_stats_line(now, before)
    assert "window 200 accesses, 100 hits, 50 unique misses, hit rate 0.500, 60 reads" in line
    assert "total 300 accesses, 140 hits, 80 unique misses, hit rate 0.467, 110 reads" in line
    assert "scales" not in line
    assert "scales 2.86 GiB in RAM" in native_stats_line({**now, "scales_ram_bytes": 3072049344}, before)


# SGLANG_DSV41_ENGRAM_SCALES_RAM_NODE: the scale rows held in RAM on a NUMA node. Node 0 exists on every Linux box;
# node 1 runs where there is one (divix01).
NODES = [0, 1] if os.path.isdir("/sys/devices/system/node/node1") else [0]
# A table whose every row is one whole 4 KiB weight page and a 128 B scale row, so misses spaced 64 rows apart touch
# pages no other miss touches and that never coalesce: the file path costs exactly two reads a miss, RAM scales one.
PAGED_ROWS, PAGED_DIM = 1024, 4096
PAGED_ROW_BYTES = PAGED_DIM + PAGED_DIM // BLOCK
SPACED = np.arange(0, PAGED_ROWS, 64, dtype=np.int64)


@pytest.fixture
def paged_table(tmp_path):
    torch.manual_seed(2)
    weight = (torch.randn(PAGED_ROWS, PAGED_DIM) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 130, (PAGED_ROWS, PAGED_DIM // BLOCK), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    path = tmp_path / "model-00047-of-00048.safetensors"
    raw_w, raw_s = (x.view(torch.uint8).numpy().tobytes() for x in (weight, scale))
    header = {
        "layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [PAGED_ROWS, PAGED_DIM],
                                         "data_offsets": [0, len(raw_w)]},
        "layers.1.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [PAGED_ROWS, PAGED_DIM // BLOCK],
                                        "data_offsets": [len(raw_w), len(raw_w) + len(raw_s)]},
    }
    blob = json.dumps(header).encode()
    blob += b" " * (-(8 + len(blob)) % 4096)  # the data section, so the weight, starts on a page
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw_w + raw_s)
    t = EngramFileTable(str(path), "layers.1.engram.embed.weight", "layers.1.engram.embed.scale", PAGED_ROWS, PAGED_DIM)
    assert t.weight_offset % 4096 == 0
    packed = torch.cat([weight.view(torch.uint8), scale.view(torch.uint8)], dim=1).numpy()
    return t, packed


def _paged_store(t, budget):
    try:
        store = native_engram_host_node().create_store(budget, PAGED_ROW_BYTES)
    except RuntimeError as exc:
        if "io_uring_queue_init failed" in str(exc):
            pytest.skip(f"io_uring is unavailable: {exc}")
        raise
    store.register_table(t.path, t.weight_offset, t.scale_offset, PAGED_ROWS, PAGED_DIM, PAGED_DIM // BLOCK, t._tag)
    return store


@pytest.mark.parametrize("node", NODES)
def test_scales_in_ram_serve_the_file_rows_bit_for_bit_with_one_read_per_miss(paged_table, node):
    t, packed = paged_table
    file_store, ram_store = _paged_store(t, 0), _paged_store(t, 0)
    loaded = dict(ram_store.load_scales(t._tag, node))
    assert loaded["off_node_pages"] == 0 and loaded["bytes"] == PAGED_ROWS * PAGED_DIM // BLOCK
    assert ram_store.stats()["scales_ram_bytes"] == loaded["bytes"] and file_store.stats()["scales_ram_bytes"] == 0
    before = {s: s.stats()["submitted_sqes"] for s in (file_store, ram_store)}
    got_file, got_ram = file_store.lookup(SPACED, t._tag), ram_store.lookup(SPACED, t._tag)
    assert np.array_equal(got_file, packed[SPACED]) and np.array_equal(got_ram, packed[SPACED])
    assert file_store.stats()["submitted_sqes"] - before[file_store] == 2 * len(SPACED)
    assert ram_store.stats()["submitted_sqes"] - before[ram_store] == len(SPACED)
    rng = np.random.default_rng(3)
    for _ in range(20):
        ids = rng.integers(0, PAGED_ROWS, size=rng.integers(1, 64)).astype(np.int64)
        assert np.array_equal(ram_store.lookup(ids, t._tag), file_store.lookup(ids, t._tag))


def test_scales_in_ram_match_the_file_on_an_unaligned_table(table):
    """The fixture's 7-byte padding puts the scale tensor mid-page, so the RAM copy starts inside its first page."""
    t, packed = table
    assert t.scale_offset % 4096 != 0
    store = _store(t, 0)
    store.load_scales(t._tag, 0)
    ids = np.arange(ROWS, dtype=np.int64)
    assert np.array_equal(store.lookup(ids, t._tag), packed)


def test_the_cache_keeps_its_entries_and_counts_with_scales_in_ram(paged_table):
    """Same lookups, same accesses, hits, misses, fills and evictions as the file path; a hit reads nothing."""
    t, packed = paged_table
    budget = 64 * PAGED_ROW_BYTES
    file_store, ram_store = _paged_store(t, budget), _paged_store(t, budget)
    ram_store.load_scales(t._tag, 0)
    rng = np.random.default_rng(4)
    for _ in range(30):
        ids = rng.integers(0, 200, size=rng.integers(1, 40)).astype(np.int64)
        got = ram_store.lookup(ids, t._tag)
        assert np.array_equal(got, file_store.lookup(ids, t._tag)) and np.array_equal(got, packed[ids])
    keys = ("accesses", "hits", "unique_misses", "filled_rows", "evictions")
    assert {k: ram_store.stats()[k] for k in keys} == {k: file_store.stats()[k] for k in keys}
    ids = np.arange(100, 108, dtype=np.int64)
    ram_store.lookup(ids, t._tag)
    before = dict(ram_store.stats())
    assert np.array_equal(ram_store.lookup(ids, t._tag), packed[ids])
    after = dict(ram_store.stats())
    assert after["hits"] - before["hits"] == len(ids) and after["submitted_sqes"] == before["submitted_sqes"]


@pytest.mark.parametrize("budget", [0, PAGED_ROWS * PAGED_ROW_BYTES])
def test_a_dead_row_is_still_a_zero_row_that_counts_and_reads_nothing_with_scales_in_ram(paged_table, budget):
    t, packed = paged_table
    store = _paged_store(t, budget)
    store.load_scales(t._tag, 0)
    keys = ("accesses", "hits", "unique_misses", "submitted_sqes", "filled_rows", "evictions")
    before = {k: store.stats()[k] for k in keys}
    got = store.lookup(np.array([-1, -1], dtype=np.int64), t._tag)
    assert got.shape == (2, PAGED_ROW_BYTES) and not got.any()
    assert {k: store.stats()[k] for k in keys} == before
    ids = np.array([64, -1, 128, -1, 64], dtype=np.int64)
    got = store.lookup(ids, t._tag)
    live = ids >= 0
    assert np.array_equal(got[live], packed[ids[live]]) and not got[~live].any()
    after = dict(store.stats())
    assert after["accesses"] - before["accesses"] == 3 and after["unique_misses"] - before["unique_misses"] == 2
    assert after["submitted_sqes"] - before["submitted_sqes"] == 2


def test_a_bad_node_is_refused_and_keeps_nothing(paged_table):
    """A node the box does not have fails the bind; nothing is kept, so misses still read both pages."""
    t, packed = paged_table
    store = _paged_store(t, 0)
    with pytest.raises(RuntimeError, match="NUMA node 1000 .*mbind failed"):
        store.load_scales(t._tag, 1000)
    with pytest.raises(ValueError, match="out of range"):
        store.load_scales(t._tag, 4096)
    with pytest.raises(ValueError, match="no table is registered"):
        store.load_scales(t._tag + 1, 0)
    assert store.stats()["scales_ram_bytes"] == 0
    before = store.stats()["submitted_sqes"]
    assert np.array_equal(store.lookup(SPACED, t._tag), packed[SPACED])
    assert store.stats()["submitted_sqes"] - before == 2 * len(SPACED)
    store.load_scales(t._tag, 0)
    with pytest.raises(RuntimeError, match="already held in RAM"):
        store.load_scales(t._tag, 0)


def test_another_table_under_the_tag_drops_the_ram_scales(paged_table, table):
    t, _ = paged_table
    store = _paged_store(t, 0)
    store.load_scales(t._tag, 0)
    store.register_table(t.path, t.weight_offset, t.scale_offset, PAGED_ROWS, PAGED_DIM, PAGED_DIM // BLOCK, t._tag)
    assert store.stats()["scales_ram_bytes"] == PAGED_ROWS * PAGED_DIM // BLOCK  # the same table: kept
    other, _ = table
    store.register_table(other.path, other.weight_offset, other.scale_offset, ROWS, DIM, DIM // BLOCK, t._tag)
    assert store.stats()["scales_ram_bytes"] == 0


def test_the_switch_is_off_by_default_and_refuses_a_negative_node():
    assert envs.SGLANG_DSV41_ENGRAM_SCALES_RAM_NODE.get() is None and eft.engram_scales_ram_node() is None
    with envs.SGLANG_DSV41_ENGRAM_SCALES_RAM_NODE.override(1):
        assert eft.engram_scales_ram_node() == 1
    with envs.SGLANG_DSV41_ENGRAM_SCALES_RAM_NODE.override(-1):
        with pytest.raises(ValueError, match="must be a NUMA node"):
            eft.engram_scales_ram_node()


def test_hold_scales_in_ram_logs_one_line_and_loads_them(paged_table, caplog):
    t, packed = paged_table
    store = _paged_store(t, 0)
    with caplog.at_level("INFO"):
        result = eft.hold_scales_in_ram(store, t, 1, 0)
    assert result["off_node_pages"] == 0
    line = next(r.getMessage() for r in caplog.records if "Engram scales in RAM" in r.getMessage())
    assert "layer 1, 0.000 GiB on NUMA node 0" in line and "0 of 32 pages not on node 0" in line
    assert np.array_equal(store.lookup(SPACED, t._tag), packed[SPACED])


def test_hold_scales_in_ram_refuses_a_missing_node_and_a_short_one(paged_table, monkeypatch):
    t, _ = paged_table
    store = _paged_store(t, 0)
    with pytest.raises(ValueError, match="NUMA node 1000 does not exist"):
        eft.hold_scales_in_ram(store, t, 1, 1000)
    import sglang.srt.layers.moe.host_numa as host_numa

    monkeypatch.setattr(host_numa, "node_memory", lambda node: {"free": 1 << 20, "reclaimable": 0})
    with pytest.raises(ValueError, match="scale rows need"):
        eft.hold_scales_in_ram(store, t, 1, 0)
    assert store.stats()["scales_ram_bytes"] == 0


def test_hold_scales_in_ram_logs_then_refuses_pages_off_the_node(paged_table, caplog):
    t, _ = paged_table

    class OffNode:
        def load_scales(self, tag, node):
            return {"bytes": 131072.0, "node": float(node), "seconds": 0.1, "pages": 32.0, "off_node_pages": 3.0}

    with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="3 of layer 1's scale pages landed off node 0"):
        eft.hold_scales_in_ram(OffNode(), t, 1, 0)
    assert "3 of 32 pages not on node 0" in caplog.text
