"""hot_cache_vmm.tail_release: the granules a hot cache tensor can free when only its first keep_rows rows stay.

Only granules wholly past the kept bytes go: the granule holding the last kept byte stays mapped, so every kept row
remains backed. Covers rows smaller and larger than a granule, row and granule boundaries that meet, keeping all
rows (nothing freed) and none (everything freed).
"""

import pytest

from sglang.srt.layers.moe.hot_cache_vmm import granule_count, tail_release
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
