"""Split-math tests for dividing an already page-aligned read length across
drives (see `Exl3ExpertRecord.aligned_read`)."""

import pytest

from sglang.srt.layers.moe.exl3_read_split import ReadSplit, StaticSplitPolicy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("length", [4096, 4096 * 10, 13271040])
@pytest.mark.parametrize("weights", [(1.0, 1.0), (3218.0, 3091.0), (1.0,), (2.0, 1.0, 1.0)])
def test_parts_are_page_aligned_and_sum_to_length(length, weights):
    s = ReadSplit(length=length, weights=weights)
    assert sum(s.part_bytes) == length
    assert all(n % 4096 == 0 for n in s.part_bytes)
    assert s.starts[0] == 0
    assert all(s.starts[i] + s.part_bytes[i] == s.starts[i + 1] for i in range(len(weights) - 1))


def test_a_zero_weight_drops_that_root():
    s = ReadSplit(length=4096 * 8, weights=(1.0, 0.0))
    assert s.part_bytes == (4096 * 8, 0)


def test_every_weight_zero_is_rejected():
    with pytest.raises(ValueError):
        ReadSplit(length=4096, weights=(0.0, 0.0))


def test_a_length_that_is_not_page_aligned_is_rejected():
    with pytest.raises(ValueError):
        ReadSplit(length=4097, weights=(1.0,))


def test_more_roots_than_pages_gives_empty_parts_not_an_error():
    s = ReadSplit(length=4096, weights=(1.0, 1.0, 1.0))
    assert sum(s.part_bytes) == 4096 and s.part_bytes.count(0) == 2


def test_static_policy_plans_the_same_split_every_time():
    p = StaticSplitPolicy(weights=(1.0, 1.0))
    assert p.plan(4096 * 4).part_bytes == p.plan(4096 * 4).part_bytes == (8192, 8192)


# Further cases beyond the brief.


def test_a_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        ReadSplit(length=4096, weights=(1.0, -1.0))


def test_no_weights_is_rejected():
    with pytest.raises(ValueError):
        ReadSplit(length=4096, weights=())


def test_zero_length_is_legal_and_gives_all_zero_parts():
    s = ReadSplit(length=0, weights=(1.0, 1.0))
    assert s.part_bytes == (0, 0)
    assert s.starts == (0, 0)


def test_a_negative_length_is_rejected():
    with pytest.raises(ValueError):
        ReadSplit(length=-4096, weights=(1.0,))


def test_the_largest_weight_root_absorbs_the_remainder_pages():
    # 3 pages to split 2-1 by page count between equal weights: the tie
    # break must be deterministic, and remainder pages must all land on
    # one root rather than spreading unevenly.
    s = ReadSplit(length=4096 * 3, weights=(1.0, 1.0))
    assert sorted(s.part_bytes) == [4096, 4096 * 2]


def test_a_lopsided_weight_still_sums_exactly():
    s = ReadSplit(length=4096 * 7, weights=(1000.0, 1.0))
    assert sum(s.part_bytes) == 4096 * 7
    assert s.part_bytes[1] in (0, 4096)


def test_single_root_takes_everything():
    s = ReadSplit(length=4096 * 5, weights=(1.0,))
    assert s.part_bytes == (4096 * 5,)
    assert s.starts == (0,)


def test_read_split_is_immutable():
    s = ReadSplit(length=4096, weights=(1.0,))
    with pytest.raises(Exception):
        s.part_bytes = (0,)


def test_static_split_policy_plan_rejects_all_zero_weights():
    p = StaticSplitPolicy(weights=(0.0, 0.0))
    with pytest.raises(ValueError):
        p.plan(4096)
