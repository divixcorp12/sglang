"""Engram RAM row cache, PagedRowSource base offsets, and EngramFileTable on top."""

import json
import struct

import numpy as np
import pytest
import torch

from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.srt.environ import envs
from sglang.srt.layers import engram_row_cache as row_cache_module
from sglang.srt.layers.engram_row_cache import EngramRowCache, shared_engram_row_cache
from sglang.srt.model_loader.file_row_reader import PagedRowSource, shared_uring_file_reader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fetcher(table, calls):
    def fetch(keys):
        calls.append(list(keys))
        return table[keys]

    return fetch


def test_hits_after_first_lookup_and_order_is_kept():
    table = np.arange(100 * 4, dtype=np.uint8).reshape(100, 4)
    cache = EngramRowCache(capacity_rows=16, row_bytes=4)
    calls = []
    got = cache.lookup(np.array([5, 7, 5]), _fetcher(table, calls))
    assert np.array_equal(got, table[[5, 7, 5]])
    assert calls == [[5, 7]]
    got = cache.lookup(np.array([7, 5]), _fetcher(table, calls))
    assert np.array_equal(got, table[[7, 5]])
    assert calls == [[5, 7]]
    assert (cache.accesses, cache.hits) == (5, 2)


def test_full_set_evicts_its_least_recently_used_way():
    table = np.arange(64 * 2, dtype=np.uint8).reshape(64, 2)
    cache = EngramRowCache(capacity_rows=2, row_bytes=2, ways=2)  # one set, two ways
    calls = []
    cache.lookup(np.array([1]), _fetcher(table, calls))
    cache.lookup(np.array([2]), _fetcher(table, calls))
    cache.lookup(np.array([1]), _fetcher(table, calls))  # 2 is now the older way
    cache.lookup(np.array([3]), _fetcher(table, calls))
    calls.clear()
    cache.lookup(np.array([1, 3]), _fetcher(table, calls))
    assert calls == []
    cache.lookup(np.array([2]), _fetcher(table, calls))
    assert calls == [[2]]


def test_paged_row_source_honours_base_offset(tmp_path):
    rows = np.random.default_rng(0).integers(0, 256, (50, 8), dtype=np.uint8)
    path = tmp_path / "t.bin"
    path.write_bytes(b"\xee" * 1000 + rows.tobytes())
    source = PagedRowSource(shared_uring_file_reader(), path, 8, 50, direct=False, base_offset=1000)
    out = torch.empty((3, 8), dtype=torch.uint8)
    source.read_rows(torch.tensor([49, 0, 17]), out)
    assert np.array_equal(out.numpy(), rows[[49, 0, 17]])


def _engram_shard(path, n, dim, block=32):
    rng = np.random.default_rng(1)
    weight = rng.integers(0, 0x70, (n, dim), dtype=np.uint8)  # finite fp8 values
    scale = np.full((n, dim // block), 127, dtype=np.uint8)  # e8m0 2**0
    header = {
        "layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [n, dim], "data_offsets": [0, n * dim]},
        "layers.1.engram.embed.scale": {
            "dtype": "F8_E8M0", "shape": [n, dim // block], "data_offsets": [n * dim, n * dim + n * dim // block],
        },
    }
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(weight.tobytes())
        f.write(scale.tobytes())


def test_file_table_with_cache_matches_memmap(tmp_path):
    n, dim = 300, 64
    path = str(tmp_path / "model-00047-of-00048.safetensors")
    _engram_shard(path, n, dim)
    plain = EngramFileTable(path, "layers.1.engram.embed.weight", "layers.1.engram.embed.scale", n, dim)
    cache = EngramRowCache(capacity_rows=64, row_bytes=dim + dim // 32)
    cached = EngramFileTable(
        path, "layers.1.engram.embed.weight", "layers.1.engram.embed.scale", n, dim,
        cache=cache, cache_tag=1, direct=False,
    )
    ids = torch.tensor([[3, 299, 3], [150, 0, 42]])
    assert torch.equal(cached.lookup(ids), plain.lookup(ids))
    assert torch.equal(cached.lookup(ids), plain.lookup(ids))
    assert cache.hits == 6


@pytest.fixture
def fresh_shared_cache(monkeypatch):
    """The shared cache is process-global; give each test its own."""
    monkeypatch.setattr(row_cache_module, "_SHARED", None)


def test_empty_lookup_returns_no_rows_and_never_fetches():
    cache = EngramRowCache(capacity_rows=16, row_bytes=4)
    calls = []
    got = cache.lookup(np.array([], dtype=np.int64), _fetcher(np.zeros((4, 4), np.uint8), calls))
    assert got.shape == (0, 4) and got.dtype == np.uint8
    assert calls == [] and cache.accesses == 0


def test_shared_cache_is_off_at_zero_gib(fresh_shared_cache):
    with envs.SGLANG_DSV41_ENGRAM_RAM_GIB.override(0.0):
        assert shared_engram_row_cache(66) is None


def test_shared_cache_is_one_object_and_checks_row_bytes(fresh_shared_cache):
    with envs.SGLANG_DSV41_ENGRAM_RAM_GIB.override(0.001):
        first = shared_engram_row_cache(66)
        assert first is not None and first.row_bytes == 66
        assert shared_engram_row_cache(66) is first
        with pytest.raises(ValueError, match="disagree on row bytes"):
            shared_engram_row_cache(130)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
