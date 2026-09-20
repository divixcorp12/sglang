"""Geometry and manifest round-trip for splitting an EXL3 row across drives."""

import pytest

from sglang.srt.layers.moe.exl3_stripe_layout import (
    StripeGeometry,
    StripeInfo,
    StripeManifest,
    validate_set,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_fragments_are_page_aligned_and_sum_to_the_row():
    g = StripeGeometry(row_bytes=13271040, weights=(3218.0, 3091.0))
    assert sum(g.fragment_bytes) == 13271040
    assert all(n % 4096 == 0 for n in g.fragment_bytes[:-1])
    assert g.starts == (0, g.fragment_bytes[0])


def test_last_fragment_absorbs_the_remainder():
    g = StripeGeometry(row_bytes=4096 * 10 + 7, weights=(1.0, 1.0))
    assert sum(g.fragment_bytes) == 4096 * 10 + 7
    assert g.fragment_bytes[0] % 4096 == 0
    # The last fragment's *payload* is generally unaligned (it absorbs the
    # remainder), but its *slot stride* must still be page-aligned so every
    # expert past index 0 on the last drive lands on an O_DIRECT boundary.
    assert g.fragment_bytes[-1] % 4096 != 0
    assert g.strides[-1] % 4096 == 0
    assert g.slot_offset(1, 3) % 4096 == 0


@pytest.mark.parametrize("row_bytes", [4096 * 10, 4096 * 10 + 7, 4096 * 10 + 4095])
@pytest.mark.parametrize("weights", [(1.0, 1.0), (3218.0, 3091.0), (1.0,)])
def test_every_slot_offset_is_page_aligned(row_bytes, weights):
    g = StripeGeometry(row_bytes=row_bytes, weights=weights)
    for stripe in range(len(weights)):
        for expert in range(5):
            assert g.slot_offset(stripe, expert) % 4096 == 0


@pytest.mark.parametrize("row_bytes", [4096 * 10, 4096 * 10 + 7, 4096 * 10 + 4095])
@pytest.mark.parametrize("weights", [(1.0, 1.0), (3218.0, 3091.0), (1.0,)])
def test_fragment_bytes_still_sum_to_the_row(row_bytes, weights):
    g = StripeGeometry(row_bytes=row_bytes, weights=weights)
    assert sum(g.fragment_bytes) == row_bytes


@pytest.mark.parametrize("row_bytes", [4096 * 10, 4096 * 10 + 7, 4096 * 10 + 4095])
@pytest.mark.parametrize("weights", [(1.0, 1.0), (3218.0, 3091.0), (1.0,)])
def test_stride_pads_the_payload_by_less_than_a_page(row_bytes, weights):
    g = StripeGeometry(row_bytes=row_bytes, weights=weights)
    for stride, payload in zip(g.strides, g.fragment_bytes):
        assert stride >= payload
        assert stride - payload < 4096


def test_weights_shape_the_split():
    g = StripeGeometry(row_bytes=4096 * 100, weights=(3.0, 1.0))
    assert g.fragment_bytes[0] > 2 * g.fragment_bytes[1]


def test_single_stripe_is_the_whole_row():
    g = StripeGeometry(row_bytes=4096 * 5, weights=(1.0,))
    assert g.fragment_bytes == (4096 * 5,) and g.slot_offset(0, 3) == 3 * 4096 * 5


def test_slot_offset_is_stride_times_expert():
    g = StripeGeometry(row_bytes=4096 * 8, weights=(1.0, 1.0))
    assert g.slot_offset(1, 5) == 5 * g.fragment_bytes[1]


@pytest.mark.parametrize("weights", [(), (1.0, 0.0), (1.0, -1.0)])
def test_rejects_degenerate_weights(weights):
    with pytest.raises(ValueError):
        StripeGeometry(row_bytes=4096, weights=weights)


def test_too_many_stripes_for_the_row_raises():
    """A row too small to hold K page-aligned leading fragments must fail loud,
    not silently emit a zero or negative last fragment."""
    with pytest.raises(ValueError):
        StripeGeometry(row_bytes=4096, weights=(1.0, 1.0, 1.0))


def _manifest(index: int) -> StripeManifest:
    stripes = (
        StripeInfo(index=0, weight=3218.0, fragment_bytes=6639616, dir_hint="/mnt/nvme0/s0"),
        StripeInfo(index=1, weight=3091.0, fragment_bytes=6631424, dir_hint="/mnt/nvme1/s1"),
    )
    return StripeManifest(
        version=1,
        source="/data/models/DeepSeek-V4.1-Flash-EXL3-3.0bpw",
        num_layers=40,
        num_experts=384,
        row_bytes=13271040,
        tensor_order=("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"),
        stripes=stripes,
        row_sha256_sample={"0:0": "abc", "19:200": "def"},
        index=index,
    )


def test_manifest_round_trips_through_json():
    manifest = _manifest(0)
    assert StripeManifest.from_json(manifest.to_json()) == manifest


def test_validate_set_accepts_manifests_differing_only_in_index():
    validate_set([_manifest(0), _manifest(1)])


def test_validate_set_rejects_row_bytes_mismatch():
    other = _manifest(1)
    other = StripeManifest(**{**other.__dict__, "row_bytes": other.row_bytes + 1})
    with pytest.raises(ValueError):
        validate_set([_manifest(0), other])


def test_validate_set_rejects_num_experts_mismatch():
    other = _manifest(1)
    other = StripeManifest(**{**other.__dict__, "num_experts": other.num_experts + 1})
    with pytest.raises(ValueError):
        validate_set([_manifest(0), other])


def test_validate_set_rejects_num_layers_mismatch():
    other = _manifest(1)
    other = StripeManifest(**{**other.__dict__, "num_layers": other.num_layers + 1})
    with pytest.raises(ValueError):
        validate_set([_manifest(0), other])


def test_validate_set_rejects_tensor_order_mismatch():
    other = _manifest(1)
    other = StripeManifest(**{**other.__dict__, "tensor_order": ("w2_trellis",) + other.tensor_order[1:]})
    with pytest.raises(ValueError):
        validate_set([_manifest(0), other])


def test_validate_set_rejects_stripes_list_mismatch():
    other = _manifest(1)
    changed = other.stripes[0].__class__(
        index=0, weight=999.0, fragment_bytes=other.stripes[0].fragment_bytes, dir_hint="/mnt/nvme9/s0"
    )
    other = StripeManifest(**{**other.__dict__, "stripes": (changed, other.stripes[1])})
    with pytest.raises(ValueError):
        validate_set([_manifest(0), other])


def test_validate_set_rejects_indices_not_exactly_0_to_k_minus_1():
    with pytest.raises(ValueError):
        validate_set([_manifest(0), _manifest(2)])


def test_validate_set_rejects_duplicate_indices():
    with pytest.raises(ValueError):
        validate_set([_manifest(0), _manifest(0)])


def test_validate_set_rejects_a_missing_manifest():
    with pytest.raises(ValueError):
        validate_set([_manifest(0)])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
