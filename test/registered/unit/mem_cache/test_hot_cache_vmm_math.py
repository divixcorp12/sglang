"""hot_cache_vmm.tail_release: the granules a hot cache tensor can free when only its first keep_rows rows stay.

Only granules wholly past the kept bytes go: the granule holding the last kept byte stays mapped, so every kept row
remains backed. Covers rows smaller and larger than a granule, row and granule boundaries that meet, keeping all
rows (nothing freed) and none (everything freed).

On a fake reservation and driver (no device): ``bytes_to_back`` prices a grow in whole granules, and a grow whose
mapping fails part-way unmaps and releases every granule it mapped, leaving the backing as it was.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.hot_cache_vmm import (
    HotCacheMapOutOfMemory,
    VmmHotCacheBacking,
    granule_count,
    tail_release,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

G = 2 << 20


@pytest.mark.parametrize("row_bytes", [36864, 1 << 20, G, 5_898_240, 8_847_360])
@pytest.mark.parametrize("rows", [1, 7, 26, 31])
def test_released_granules_hold_no_kept_byte_and_every_byte_past_them_is_released(row_bytes, rows):
    for keep in range(rows + 1):
        t = tail_release(row_bytes=row_bytes, rows=rows, keep_rows=keep, granularity=G)
        assert t.end == granule_count(rows * row_bytes, G)
        assert t.first * G >= keep * row_bytes  # no kept byte in a released granule
        assert (t.first - 1) * G < keep * row_bytes or t.first == 0  # the first released granule is the earliest one
        assert t.nbytes == (t.end - t.first) * G >= 0


def test_keeping_every_row_frees_nothing_and_keeping_none_frees_all():
    full = tail_release(row_bytes=8_847_360, rows=26, keep_rows=26, granularity=G)
    assert full.nbytes == 0
    none = tail_release(row_bytes=8_847_360, rows=26, keep_rows=0, granularity=G)
    assert none.first == 0 and none.nbytes == granule_count(26 * 8_847_360, G) * G


def test_small_rows_free_only_whole_granules():
    # 36 KiB rows: retiring 3 of 26 rows frees nothing (they share the last granule with kept rows).
    assert tail_release(row_bytes=36864, rows=26, keep_rows=23, granularity=G).nbytes == 0


def test_keep_rows_outside_the_tensor_is_refused():
    with pytest.raises(ValueError):
        tail_release(row_bytes=G, rows=4, keep_rows=5, granularity=G)


class _FakeReservation:
    """Maps succeed until ``fail_after`` of them have; the next raises check_drv's out-of-memory error."""

    def __init__(self, base, budget):
        self.base, self._mappings, self.budget = base, [], budget

    def map(self, offset, size, *, retain_handle):
        if self.budget["left"] == 0:
            raise RuntimeError("cuMemCreate(local): <CUresult.CUDA_ERROR_OUT_OF_MEMORY: 2>")
        self.budget["left"] -= 1
        handle = object()
        self._mappings.append((self.base + offset, size, handle))
        return handle


def _backing(monkeypatch, budget, backed=2):
    """A VmmHotCacheBacking without a device: two tensors of 1.5-granule rows over 8 rows, ``backed`` rows mapped."""
    from sglang.srt.utils import cuda_vmm_utils

    released = []
    drv = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        cuMemUnmap=lambda address, size: (0,),
        cuMemRelease=lambda handle: (released.append(handle), 0)[1:],
    )
    monkeypatch.setattr(cuda_vmm_utils, "_get_cuda_driver", lambda: drv)
    b = VmmHotCacheBacking.__new__(VmmHotCacheBacking)
    b.device_id, b.granularity, b.backed_rows = -1, G, None
    b.row_bytes, b.rows = {"w13": 3 * G // 2, "w2": 3 * G // 2}, {"w13": 8, "w2": 8}
    b.upper_rows = 8
    b._reservations = {n: _FakeReservation(i << 40, budget) for i, n in enumerate(("w13", "w2"))}
    b._handles = {n: [None] * granule_count(8 * 3 * G // 2, G) for n in b._reservations}
    budget["left"] = 10**6
    b.set_backed_rows(backed)
    return b, released


def test_bytes_to_back_counts_the_whole_granules_a_grow_maps(monkeypatch):
    b, _ = _backing(monkeypatch, {})
    assert b.mapped_bytes == 2 * 3 * G  # 2 rows of 1.5 granules: 3 granules a tensor
    assert b.bytes_to_back(2) == 0 and b.bytes_to_back(1) == 0
    assert b.bytes_to_back(3) == 2 * 2 * G  # 4.5 granules round up to 5: two more a tensor
    assert b.bytes_to_back(4) == 2 * 3 * G


def test_a_grow_that_runs_out_of_memory_part_way_unmaps_what_it_mapped(monkeypatch):
    budget = {}
    b, released = _backing(monkeypatch, budget)
    before = ({n: list(h) for n, h in b._handles.items()}, {n: list(r._mappings) for n, r in b._reservations.items()})
    budget["left"] = 3  # w13 gets its 3 new granules (2 -> 4 rows), w2's first map fails
    with pytest.raises(HotCacheMapOutOfMemory, match="3 granules mapped and rolled back"):
        b.set_backed_rows(4)
    assert len(released) == 3  # every granule this call mapped was released again
    assert ({n: list(h) for n, h in b._handles.items()}, {n: list(r._mappings) for n, r in b._reservations.items()}) == before
    assert b.backed_rows == 2 and b.mapped_bytes == 2 * 3 * G
    budget["left"] = 10**6  # memory back: the same grow now succeeds
    assert b.set_backed_rows(4) == 2 * 3 * G and b.backed_rows == 4


def test_a_failure_that_is_not_out_of_memory_is_rolled_back_and_re_raised(monkeypatch):
    budget = {}
    b, released = _backing(monkeypatch, budget)

    def broken(offset, size, *, retain_handle):
        raise RuntimeError("cuMemMap(local): <CUresult.CUDA_ERROR_INVALID_VALUE: 1>")

    b._reservations["w2"].map = broken
    with pytest.raises(RuntimeError, match="INVALID_VALUE") as info:
        b.set_backed_rows(4)
    assert not isinstance(info.value, HotCacheMapOutOfMemory) and len(released) == 3 and b.backed_rows == 2
