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


def test_cached_lookup_reads_into_cpu_rows_under_a_non_cpu_default_device(tmp_path):
    # The reference oracle runs under torch.set_default_device("cuda"); "meta" is the
    # CPU-testable stand-in. The fetch buffers must not follow the default device.
    n, dim = 300, 64
    path = str(tmp_path / "model-00047-of-00048.safetensors")
    _engram_shard(path, n, dim)
    plain = EngramFileTable(path, "layers.1.engram.embed.weight", "layers.1.engram.embed.scale", n, dim)
    cached = EngramFileTable(
        path, "layers.1.engram.embed.weight", "layers.1.engram.embed.scale", n, dim,
        cache=EngramRowCache(capacity_rows=64, row_bytes=dim + dim // 32), cache_tag=1, direct=False,
    )
    ids = torch.tensor([[3, 299, 3], [150, 0, 42]])
    expected = plain.lookup(ids)
    previous = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        got = cached.lookup(ids)
    finally:
        torch.set_default_device(previous)
    assert torch.equal(got, expected)


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


def test_lookup_into_fills_caller_owned_rows_in_duplicate_order():
    table = np.arange(32 * 4, dtype=np.uint8).reshape(32, 4)
    cache = EngramRowCache(capacity_rows=16, row_bytes=4)
    destination = np.zeros((4, 4), dtype=np.uint8)
    cache.lookup_into(np.array([7, 3, 7, 1]), _fetcher(table, []), destination)
    assert np.array_equal(destination, table[[7, 3, 7, 1]])
    with pytest.raises(ValueError, match="destination must be uint8"):
        cache.lookup_into(np.array([1]), _fetcher(table, []), np.zeros((1, 4), dtype=np.int8))


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


def test_stats_report_the_hit_rate():
    table = np.arange(100 * 4, dtype=np.uint8).reshape(100, 4)
    cache = EngramRowCache(capacity_rows=16, row_bytes=4)
    empty = {
        "lookups": 0, "accesses": 0, "hits": 0, "hit_rate": 0.0,
        "misses": 0, "evictions": 0, "filled_rows": 0, "capacity_rows": 16,
    }
    assert cache.stats() == empty
    cache.lookup(np.array([5, 7, 5]), _fetcher(table, []))
    cache.lookup(np.array([7, 5]), _fetcher(table, []))
    assert cache.stats() == {**empty, "lookups": 2, "accesses": 5, "hits": 2, "hit_rate": 0.4, "misses": 2, "filled_rows": 2}


def test_evictions_and_fill_are_counted_and_change_no_result():
    table = np.arange(64 * 2, dtype=np.uint8).reshape(64, 2)
    cache = EngramRowCache(capacity_rows=2, row_bytes=2, ways=2)  # one set, two ways
    for key in (1, 2, 1, 3, 4, 1):
        got = cache.lookup(np.array([key]), _fetcher(table, []))
        assert np.array_equal(got, table[[key]])
    stats = cache.stats()
    # Fetched: 1, 2, 3 (evicts 2), 4 (evicts 1), 1 (evicts 3). Key 1's second lookup hit.
    assert (stats["misses"], stats["evictions"], stats["filled_rows"]) == (5, 3, 2)
    assert stats["capacity_rows"] == 2
    assert stats["misses"] + stats["hits"] == stats["accesses"]


def test_stats_sink_records_cumulative_engram_counters(tmp_path):
    trace = str(tmp_path / "trace.jsonl")
    table = np.arange(100 * 4, dtype=np.uint8).reshape(100, 4)
    saved = (row_cache_module._SINK, row_cache_module._SINK_PATH)
    row_cache_module._SINK, row_cache_module._SINK_PATH = None, ""
    try:
        with envs.SGLANG_DSV41_EXPERT_TRACE_PATH.override(trace):
            cache = EngramRowCache(capacity_rows=16, row_bytes=4)
            cache._sink._interval_s = 0.0
            cache.lookup(np.array([1, 2]), _fetcher(table, []))
            cache.lookup(np.array([1, 3]), _fetcher(table, []))
    finally:
        row_cache_module._SINK, row_cache_module._SINK_PATH = saved
    with open(trace + ".cache-stats") as f:
        lines = [json.loads(line) for line in f]
    assert [line["kind"] for line in lines] == ["engram", "engram"]
    assert (lines[-1]["hits"], lines[-1]["misses"], lines[-1]["filled_rows"]) == (1, 3, 3)
    assert lines[0]["t"] <= lines[1]["t"]


def _log_lines(caplog):
    return [
        json.loads(r.getMessage().split("engram row cache: ", 1)[1])
        for r in caplog.records
        if "engram row cache: " in r.getMessage()
    ]


def test_logs_its_stats_every_log_every_lookups(caplog):
    table = np.arange(100 * 4, dtype=np.uint8).reshape(100, 4)
    cache = EngramRowCache(capacity_rows=16, row_bytes=4, log_every=2)
    with caplog.at_level("INFO", logger="sglang.srt.layers.engram_row_cache"):
        cache.lookup(np.array([1, 2]), _fetcher(table, []))
        assert _log_lines(caplog) == []
        cache.lookup(np.array([1, 2]), _fetcher(table, []))
        assert _log_lines(caplog) == [cache.stats()]
        cache.lookup(np.array([3]), _fetcher(table, []))
        cache.lookup(np.array([3]), _fetcher(table, []))
    lines = _log_lines(caplog)
    assert [line["lookups"] for line in lines] == [2, 4]
    assert lines[-1] == cache.stats()
    assert (lines[-1]["accesses"], lines[-1]["hits"], lines[-1]["hit_rate"]) == (6, 3, 0.5)


def test_log_is_silent_before_any_lookup(caplog):
    cache = EngramRowCache(capacity_rows=16, row_bytes=4)
    with caplog.at_level("INFO", logger="sglang.srt.layers.engram_row_cache"):
        cache.log()
    assert _log_lines(caplog) == []


def test_shared_cache_logs_at_exit(fresh_shared_cache, monkeypatch):
    registered = []
    monkeypatch.setattr(row_cache_module.atexit, "register", lambda fn, *a: registered.append(fn))
    with envs.SGLANG_DSV41_ENGRAM_RAM_GIB.override(0.001):
        cache = shared_engram_row_cache(66)
        assert shared_engram_row_cache(66) is cache
    assert registered == [cache.log]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
