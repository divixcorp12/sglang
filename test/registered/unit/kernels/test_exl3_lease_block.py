"""The lease block's layout, its allocator and its words (CPU); analysis/dsv41-drive/LEASE_PROTOCOL.md."""

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_the_areas_are_where_the_protocol_puts_them():
    assert lease.ROW_RESULT == 0 and lease.ROW_RESULT + lease.RING * lease.LANES * lease.ROW_RESULT_BYTES <= lease.PIECE_MASK
    assert lease.PIECE_MASK == 0x1000 and lease.COPY_DONE == 0x5000, "PieceMask[16][8] at a 128-byte line each"
    assert lease.COPY_GATE == lease.COPY_DONE + 128 and lease.COPY_GATE + 4 <= lease.LANE_REQUEST
    assert lease.LANE_REQUEST == 0x6000 and lease.DONE == lease.LANE_REQUEST + lease.RING * lease.LANE_REQUEST_BYTES
    assert lease.DONE + lease.RING * lease.DONE_BYTES <= lease.BLOCK_BYTES == 7 * lease.BLOCK_ALIGN


def test_host_written_and_device_written_areas_never_share_a_page():
    host_areas_end = lease.COPY_GATE + 4
    assert host_areas_end <= lease.LANE_REQUEST and lease.LANE_REQUEST % lease.BLOCK_ALIGN == 0


def test_the_lane_request_payload_fits_its_record_in_the_order_the_kernels_write_it():
    f = lease.LANE_REQUEST_FIELDS
    assert f["gen"] + 8 == f["count"] and f["count"] + 4 == f["flags"] and f["flags"] + 4 == f["expert"]
    assert f["expert"] + 4 * lease.LANES == f["dst_slot"] and f["dst_slot"] + 4 * lease.LANES == f["weight"]
    assert f["weight"] + 4 * lease.LANES <= lease.LANE_REQUEST_BYTES


def test_the_gate_word_names_its_request_and_only_open_words_pass_the_cyclic_geq():
    """cuStreamWaitValue32(gate, GATE["open"], GEQ) passes iff (int32)(gate - 1) >= 0: every open word of every seq
    passes and every closed word blocks, so the wait never wraps; the seq field tells two requests' words apart."""

    def passes(word):
        diff = (word - lease.GATE["open"]) & 0xFFFFFFFF
        return diff < (1 << 31)

    for seq in (1, 2, 15, 16, lease.GATE_SEQ_MASK, lease.GATE_SEQ_MASK + 1, 0xFFFFFFFF):
        assert not passes(lease.gate_word(seq, "closed"))
        assert passes(lease.gate_word(seq, "open"))
        assert lease.gate_word(seq, "closed") & ~0x80000000 == lease.gate_word(seq, "open")
    assert lease.gate_word(1, "open") != lease.gate_word(2, "open")


def test_the_block_is_zeroed_aligned_and_exactly_block_bytes_long():
    block = lease.new_lease_block(pin=False)
    assert block.numel() == lease.BLOCK_BYTES and block.data_ptr() % lease.BLOCK_ALIGN == 0
    assert not block.any()


@pytest.mark.parametrize("bad", ["dtype", "size", "align", "shape"])
def test_a_block_the_kernels_cannot_address_is_refused(bad):
    raw = torch.zeros(lease.BLOCK_BYTES + 2 * lease.BLOCK_ALIGN, dtype=torch.uint8)
    good = raw[(-raw.data_ptr()) % lease.BLOCK_ALIGN :][: lease.BLOCK_BYTES]
    blocks = {
        "dtype": good.view(torch.int8),
        "size": good[:-1],
        "align": raw[((-raw.data_ptr()) % lease.BLOCK_ALIGN) + 1 :][: lease.BLOCK_BYTES],
        "shape": good.view(1, -1),
    }
    lease.check_lease_block(good, need_pinned=False)
    with pytest.raises(ValueError):
        lease.check_lease_block(blocks[bad], need_pinned=False)


def test_a_block_that_is_not_pinned_is_refused_for_a_cuda_device():
    with pytest.raises(ValueError, match="pinned"):
        lease.check_lease_block(lease.new_lease_block(pin=False), need_pinned=True)


def test_a_ready_word_carries_the_tag_over_a_56_bit_generation():
    generation = (3 << 32) | 0xFFFFFFFF
    word = lease.tagged(lease.READY, generation)
    assert lease.untag(word) == (lease.READY, generation) and word >> 56 == lease.READY
    assert (lease.READY, lease.LOADING, lease.COPYING, lease.CPU) == (1, 2, 3, 4)


def test_generation_zero_and_generations_past_56_bits_are_refused():
    with pytest.raises(ValueError):
        lease.tagged(lease.READY, 0)
    with pytest.raises(ValueError):
        lease.tagged(lease.READY, 1 << 56)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
