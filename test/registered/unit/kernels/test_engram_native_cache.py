"""SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB: the native Engram store at 0 (no cache, every row read through io_uring)
returns the same rows bit for bit as at the default 5 GiB, and its counters give a hit rate (CPU)."""

import ast
import json
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
    before = {"accesses": 100, "hits": 40, "unique_misses": 30, "cache_bytes": 0, "capacity_rows": 0}
    now = {"accesses": 300, "hits": 140, "unique_misses": 80, "cache_bytes": 0, "capacity_rows": 0}
    line = native_stats_line(now, before)
    assert "window 200 accesses, 100 hits, 50 unique misses, hit rate 0.500" in line
    assert "total 300 accesses, 140 hits, 80 unique misses, hit rate 0.467" in line
