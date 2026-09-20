"""Split-math tests for dividing an already page-aligned read length across
drives (see `Exl3ExpertRecord.aligned_read`)."""

import dataclasses
import random

import pytest

from sglang.srt.layers.moe.exl3_read_split import (
    PAGE_BYTES,
    ReadSplit,
    StaticSplitPolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("length", [4096, 4096 * 10, 13271040])
@pytest.mark.parametrize(
    "weights", [(1.0, 1.0), (3218.0, 3091.0), (1.0,), (2.0, 1.0, 1.0)]
)
def test_parts_are_page_aligned_and_sum_to_length(length, weights):
    s = ReadSplit(length=length, weights=weights)
    assert sum(s.part_bytes) == length
    assert all(n % 4096 == 0 for n in s.part_bytes)
    assert s.starts[0] == 0
    assert all(
        s.starts[i] + s.part_bytes[i] == s.starts[i + 1]
        for i in range(len(weights) - 1)
    )


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


def test_a_non_int_length_is_rejected():
    with pytest.raises(ValueError):
        ReadSplit(length=4096.0, weights=(1.0,))


def test_a_bool_length_is_rejected():
    # bool is an int subclass in Python; it must not silently pass as a length.
    with pytest.raises(ValueError):
        ReadSplit(length=True, weights=(1.0,))


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_weight_is_rejected(weight):
    with pytest.raises(ValueError):
        ReadSplit(length=4096, weights=(1.0, weight))


def test_the_largest_weight_root_absorbs_the_remainder_pages():
    # 3 pages to split 2-1 by page count between equal weights: the tie
    # break must be deterministic (first-largest wins), so the exact tuple
    # is pinned, not just its multiset.
    s = ReadSplit(length=4096 * 3, weights=(1.0, 1.0))
    assert s.part_bytes == (8192, 4096)


def test_the_remainder_lands_on_the_largest_weight_when_weights_differ():
    # 4 pages split 1-3 by weight (1:2): floors give (1, 2) with 1 page left
    # over, which must go to the actual largest weight (index 1), not index 0.
    s = ReadSplit(length=4096 * 4, weights=(1.0, 2.0))
    assert s.part_bytes == (4096, 4096 * 3)


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
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.part_bytes = (0,)


def test_static_split_policy_plan_rejects_all_zero_weights():
    p = StaticSplitPolicy(weights=(0.0, 0.0))
    with pytest.raises(ValueError):
        p.plan(4096)


def test_a_zero_weight_in_the_first_position_with_three_roots():
    s = ReadSplit(length=4096 * 4, weights=(0.0, 1.0, 1.0))
    assert s.part_bytes == (0, 4096 * 2, 4096 * 2)
    assert s.starts == (0, 0, 4096 * 2)


def test_a_zero_weight_in_a_middle_position_with_three_roots():
    s = ReadSplit(length=4096 * 4, weights=(1.0, 0.0, 1.0))
    assert s.part_bytes == (4096 * 2, 0, 4096 * 2)


def test_a_remainder_of_two_or_more_pages():
    # 6 pages over 4 equal-weight roots floors to (1, 1, 1, 1) with 2 pages
    # left over, all going to the single (first) largest-weight root.
    s = ReadSplit(length=4096 * 6, weights=(1.0, 1.0, 1.0, 1.0))
    assert s.part_bytes == (4096 * 3, 4096, 4096, 4096)
    assert sum(s.part_bytes) == 4096 * 6


def test_static_split_policy_with_three_roots():
    p = StaticSplitPolicy(weights=(1.0, 1.0, 1.0))
    s = p.plan(4096 * 3)
    assert s.part_bytes == (4096, 4096, 4096)


def test_read_split_property_random_lengths_and_weights():
    rng = random.Random(0)
    for _ in range(4000):
        num_roots = rng.randint(1, 6)
        length = rng.randint(0, 5000) * PAGE_BYTES
        weight_kinds = [
            lambda: float(rng.randint(0, 10)),
            lambda: rng.random() * rng.choice([1, 100, 1e6]),
            lambda: 0.0,
        ]
        weights = tuple(rng.choice(weight_kinds)() for _ in range(num_roots))
        if sum(weights) <= 0:
            continue
        s = ReadSplit(length=length, weights=weights)
        assert len(s.part_bytes) == num_roots
        for part in s.part_bytes:
            assert isinstance(part, int)
            assert part % PAGE_BYTES == 0
            assert 0 <= part <= length
        assert sum(s.part_bytes) == length
