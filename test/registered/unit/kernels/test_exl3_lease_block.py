"""The lease block's layout, its allocator and its publication words (CPU); LEASE_PROTOCOL.md section 4."""

import pytest
import torch

from sglang.kernels.ops.moe import exl3_lease_block as lease
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_the_areas_are_where_the_protocol_puts_them():
    assert lease.ROW_TABLE == 0x80 and lease.ROW_RESULT == 0x1000
    assert lease.SLOT_GEN == 0x2000, "RowResult[16][8] at 32 bytes each fills 4096 bytes from 0x1000"
    assert (lease.LANE_REQUEST, lease.LANE_ACK, lease.TERMINAL) == (0, 0x400, 0x800)
    assert lease.AREA_D_BYTES == 0x900


def test_service_written_and_device_written_words_never_share_a_128_byte_line():
    layout = lease.lease_layout([5, 7])
    assert layout.slot_gen_offset + 4 * sum(layout.capacities) <= layout.d_offset
    assert layout.d_offset % lease.BLOCK_ALIGN == 0 and layout.d_offset % 128 == 0
    assert layout.total_bytes % lease.BLOCK_ALIGN == 0 and layout.d_offset + lease.AREA_D_BYTES <= layout.piece_offset


def test_area_p_is_one_word_per_128_byte_line_after_area_d():
    """PieceMask[kLeaseRing][kLeaseLanes] (piece-streaming plan, LEASE_PROTOCOL.md E1 amendment): each of the
    RING*LANES words gets its own cache line rather than packing densely, so 128 lines, 16 KiB total."""
    layout = lease.lease_layout([5, 7])
    assert layout.piece_offset % lease.BLOCK_ALIGN == 0
    assert layout.d_offset + lease.AREA_D_BYTES <= layout.piece_offset
    assert lease.AREA_PIECE_MASK_BYTES == lease.RING * lease.LANES * lease.PIECE_MASK_LINE_BYTES == 16 * 1024
    assert layout.piece_offset + lease.AREA_PIECE_MASK_BYTES <= layout.total_bytes


def test_area_p_header_offset_is_a_new_header_word_beside_d_offset():
    assert lease.HEADER["piece_offset"] == lease.HEADER["d_offset"] + 4
    assert lease.HEADER["piece_offset"] < lease.HEADER_BYTES


def test_a_row_table_entry_fits_before_the_row_results_and_rows_are_bounded():
    assert lease.MAX_ROWS * lease.ROW_TABLE_ENTRY_BYTES + lease.ROW_TABLE <= lease.ROW_RESULT
    with pytest.raises(ValueError, match="row table"):
        lease.lease_layout([1] * (lease.MAX_ROWS + 1))
    with pytest.raises(ValueError, match="row table"):
        lease.lease_layout([])
    with pytest.raises(ValueError, match="slot"):
        lease.lease_layout([3, 0])


def test_slot_generation_words_are_dense_per_row():
    layout = lease.lease_layout([3, 5, 2])
    assert layout.slot_gen_base == (0, 3, 8)


def test_a_slot_generation_area_that_crosses_a_page_pushes_area_d_up():
    small = lease.lease_layout([10])
    big = lease.lease_layout([2000])
    assert small.d_offset == lease.SLOT_GEN + lease.BLOCK_ALIGN
    assert big.d_offset > small.d_offset
    assert big.d_offset == (lease.SLOT_GEN + 4 * 2000 + 4095) // 4096 * 4096


def test_the_block_is_zeroed_aligned_and_exactly_as_long_as_its_layout():
    layout = lease.lease_layout([4, 4])
    block = lease.new_lease_block(layout, pin=False)
    assert block.numel() == layout.total_bytes and block.data_ptr() % lease.BLOCK_ALIGN == 0
    assert not block.any()


@pytest.mark.parametrize("bad", ["dtype", "size", "align", "shape"])
def test_a_block_the_kernels_cannot_address_is_refused(bad):
    layout = lease.lease_layout([4])
    raw = torch.zeros(layout.total_bytes + 2 * lease.BLOCK_ALIGN, dtype=torch.uint8)
    good = raw[(-raw.data_ptr()) % lease.BLOCK_ALIGN :][: layout.total_bytes]
    blocks = {
        "dtype": good.view(torch.int8),
        "size": good[:-1],
        "align": raw[((-raw.data_ptr()) % lease.BLOCK_ALIGN) + 1 :][: layout.total_bytes],
        "shape": good.view(1, -1),
    }
    lease.check_lease_block(good, layout, need_pinned=False)
    with pytest.raises(ValueError):
        lease.check_lease_block(blocks[bad], layout, need_pinned=False)


def test_a_block_that_is_not_pinned_is_refused_for_a_cuda_device():
    layout = lease.lease_layout([4])
    with pytest.raises(ValueError, match="pinned"):
        lease.check_lease_block(lease.new_lease_block(layout, pin=False), layout, need_pinned=True)


def test_a_publication_word_carries_the_tag_over_a_56_bit_generation():
    generation = (3 << 32) | 0xFFFFFFFF
    word = lease.tagged(lease.READY, generation)
    assert lease.untag(word) == (lease.READY, generation) and word >> 56 == lease.READY
    assert lease.untag(lease.tagged(lease.CONSUMED, 1)) == (lease.CONSUMED, 1)


def test_row_result_tag_2_is_loading_not_ready():
    """RowResult.ready tag 2 (piece-streaming plan; LEASE_PROTOCOL.md E1 amendment): a lane granted at
    reservation, still loading. It is not READY, so a reader that only checks tag == READY must reject it."""
    assert (lease.READY, lease.LOADING) == (1, 2)
    word = lease.tagged(lease.LOADING, 7)
    assert lease.untag(word) == (lease.LOADING, 7) and word >> 56 == lease.LOADING != lease.READY


def test_generation_zero_and_generations_past_56_bits_are_refused():
    with pytest.raises(ValueError):
        lease.tagged(lease.READY, 0)
    with pytest.raises(ValueError):
        lease.tagged(lease.READY, 1 << 56)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
