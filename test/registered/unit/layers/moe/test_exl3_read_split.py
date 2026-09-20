"""Split-math tests for dividing an EXL3 row's bytes across drives."""

import pytest

from sglang.srt.layers.moe.exl3_read_split import ReadSplit
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_parts_are_page_aligned_and_sum_to_the_row():
    g = ReadSplit(row_bytes=13271040, weights=(3218.0, 3091.0))
    assert sum(g.part_bytes) == 13271040
    assert all(n % 4096 == 0 for n in g.part_bytes[:-1])
    assert g.starts == (0, g.part_bytes[0])


def test_last_part_absorbs_the_remainder():
    g = ReadSplit(row_bytes=4096 * 10 + 7, weights=(1.0, 1.0))
    assert sum(g.part_bytes) == 4096 * 10 + 7
    assert g.part_bytes[0] % 4096 == 0
    # The last part is generally unaligned (it absorbs the remainder).
    assert g.part_bytes[-1] % 4096 != 0


@pytest.mark.parametrize("row_bytes", [4096 * 10, 4096 * 10 + 7, 4096 * 10 + 4095])
@pytest.mark.parametrize("weights", [(1.0, 1.0), (3218.0, 3091.0), (1.0,)])
def test_part_bytes_still_sum_to_the_row(row_bytes, weights):
    g = ReadSplit(row_bytes=row_bytes, weights=weights)
    assert sum(g.part_bytes) == row_bytes


def test_weights_shape_the_split():
    g = ReadSplit(row_bytes=4096 * 100, weights=(3.0, 1.0))
    assert g.part_bytes[0] > 2 * g.part_bytes[1]


def test_single_part_is_the_whole_row():
    g = ReadSplit(row_bytes=4096 * 5, weights=(1.0,))
    assert g.part_bytes == (4096 * 5,)


@pytest.mark.parametrize("weights", [(), (1.0, 0.0), (1.0, -1.0)])
def test_rejects_degenerate_weights(weights):
    with pytest.raises(ValueError):
        ReadSplit(row_bytes=4096, weights=weights)


def test_too_many_parts_for_the_row_raises():
    """A row too small to hold K page-aligned leading parts must fail loud,
    not silently emit a zero or negative last part."""
    with pytest.raises(ValueError):
        ReadSplit(row_bytes=4096, weights=(1.0, 1.0, 1.0))
