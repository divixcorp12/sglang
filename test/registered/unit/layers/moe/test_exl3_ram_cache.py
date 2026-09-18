"""Bounded LRU host-RAM tier of raw EXL3 rows."""

import pytest

from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_ram_cache import Exl3RamExpertCache
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class CountingReader(Exl3RowReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def read(self, keys, destinations):
        self.calls.append(list(keys))
        return super().read(keys, destinations)


def _setup(tmp_path, capacity, **kwargs):
    rows = write_fake_exl3(str(tmp_path), num_layers=2, num_experts=4)
    reader = CountingReader(build_exl3_expert_layout(str(tmp_path)), direct=False)
    return rows, reader, Exl3RamExpertCache(reader, capacity, **kwargs)


def test_miss_then_hit(tmp_path):
    rows, reader, cache = _setup(tmp_path, capacity=4)
    got = cache.ensure([(0, 1), (1, 2), (0, 1)])
    assert [bytes(v.numpy()) for v in got] == [rows[(0, 1)], rows[(1, 2)], rows[(0, 1)]]
    assert reader.calls == [[(0, 1), (1, 2)]]
    assert (cache.hits, cache.misses) == (0, 2)

    got = cache.ensure([(1, 2)])
    assert bytes(got[0].numpy()) == rows[(1, 2)]
    assert reader.calls == [[(0, 1), (1, 2)]]
    assert (cache.hits, cache.misses) == (1, 2)


def test_evicts_least_recently_used(tmp_path):
    rows, reader, cache = _setup(tmp_path, capacity=2)
    cache.ensure([(0, 0)])
    cache.ensure([(0, 1)])
    cache.ensure([(0, 0)])  # (0, 1) is now the oldest
    cache.ensure([(0, 2)])
    assert cache.contains((0, 0)) and cache.contains((0, 2))
    assert not cache.contains((0, 1))
    assert bytes(cache.ensure([(0, 2)])[0].numpy()) == rows[(0, 2)]


def test_never_evicts_a_pinned_row(tmp_path):
    pinned = {(0, 0)}
    _rows, _reader, cache = _setup(tmp_path, capacity=2, is_pinned=pinned.__contains__)
    cache.ensure([(0, 0)])
    cache.ensure([(0, 1)])
    cache.ensure([(0, 2)])  # (0, 0) is oldest but pinned, so (0, 1) goes
    assert cache.contains((0, 0)) and cache.contains((0, 2))
    assert not cache.contains((0, 1))


def test_batch_larger_than_capacity_is_rejected(tmp_path):
    _rows, _reader, cache = _setup(tmp_path, capacity=2)
    with pytest.raises(ValueError, match="capacity"):
        cache.ensure([(0, 0), (0, 1), (0, 2)])


def test_failed_read_returns_its_slots(tmp_path):
    _rows, reader, cache = _setup(tmp_path, capacity=2)

    def boom(keys, destinations):
        raise RuntimeError("disk")

    reader.read = boom
    with pytest.raises(RuntimeError, match="disk"):
        cache.ensure([(0, 0), (0, 1)])
    assert not cache.contains((0, 0))
    del reader.read
    cache.ensure([(0, 0), (0, 1)])  # both slots are free again
    assert cache.contains((0, 0)) and cache.contains((0, 1))


def test_slots_are_page_aligned(tmp_path):
    _rows, reader, cache = _setup(tmp_path, capacity=3)
    assert cache.slot_bytes % 4096 == 0
    assert Exl3RamExpertCache.slot_bytes_for(reader) == cache.slot_bytes
    assert all(cache.buffer[i].data_ptr() % 4096 == 0 for i in range(3))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
