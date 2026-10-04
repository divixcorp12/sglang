"""The completion and delta blocks' layout, their allocator and the gate word (CPU); analysis/dsv41-drive/LEASE_PROTOCOL.md."""

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


W = lease.wire_layout(8)


def test_the_areas_are_where_the_protocol_puts_them():
    assert W.piece_mask == 0 and W.copy_done == 0x4000, "PieceMask[16][8] at a 128-byte line each"
    assert W.copy_gate == W.copy_done + 128 and W.copy_armed == W.copy_gate + 128
    assert W.split == W.copy_armed + 128 and W.split + 4 * (W.lanes + 1) <= W.lease_block_bytes
    assert W.lease_block_bytes == 5 * W.block_align


def test_a_delta_record_fits_its_stride_and_each_starts_a_new_line():
    f = W.delta_fields
    assert f["tag"] + 8 == f["count"] and f["staging"] >= f["count"] + 4
    assert f["staging"] % 16 == 0 and f["entries"] % 16 == 0, "the post reads both with 16-byte loads"
    assert f["entries"] == f["staging"] + 2 * W.lanes, "i16 staging slots"
    assert f["entries"] + 4 * W.delta_max_entries <= W.delta_stride and W.delta_stride % 128 == 0
    assert W.delta_max_entries == 2 * W.lanes, "an insert and an eviction per lane"


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


@pytest.mark.parametrize("rows", [1, 16, 17, 61])
def test_the_block_is_zeroed_aligned_and_sized_for_its_rows(rows):
    block = lease.new_lease_block(rows, pin=False, wire=W)
    assert block.numel() == lease.lease_block_bytes(rows, W) and block.data_ptr() % W.block_align == 0
    assert block.numel() >= W.lease_block_bytes + rows * W.delta_stride and block.numel() % W.block_align == 0
    assert not block.any()


@pytest.mark.parametrize("bad", ["dtype", "size", "align", "shape"])
def test_a_block_the_kernels_cannot_address_is_refused(bad):
    size = lease.lease_block_bytes(2, W)
    raw = torch.zeros(size + 2 * W.block_align, dtype=torch.uint8)
    good = raw[(-raw.data_ptr()) % W.block_align :][:size]
    blocks = {
        "dtype": good.view(torch.int8),
        "size": good[:-1],
        "align": raw[((-raw.data_ptr()) % W.block_align) + 1 :][:size],
        "shape": good.view(1, -1),
    }
    lease.check_lease_block(good, 2, need_pinned=False, wire=W)
    with pytest.raises(ValueError):
        lease.check_lease_block(blocks[bad], 2, need_pinned=False, wire=W)


def test_a_block_sized_for_other_rows_is_refused():
    with pytest.raises(ValueError, match="rows"):
        lease.check_lease_block(lease.new_lease_block(1, pin=False, wire=W), 17, need_pinned=False, wire=W)


def test_a_block_that_is_not_pinned_is_refused_for_a_cuda_device():
    with pytest.raises(ValueError, match="pinned"):
        lease.check_lease_block(lease.new_lease_block(1, pin=False, wire=W), 1, need_pinned=True, wire=W)


@pytest.mark.parametrize("lanes", [16, 32])
def test_a_wider_wire_sizes_its_block_by_its_own_layout(lanes):
    w = lease.wire_layout(lanes)
    assert lease.lease_block_bytes(1, w) == w.lease_block_bytes + 4096
    block = lease.new_lease_block(1, pin=False, wire=w)
    assert block.numel() == w.lease_block_bytes + 4096
    lease.check_lease_block(block, 1, need_pinned=False, wire=w)
    with pytest.raises(ValueError, match="bytes"):
        lease.check_lease_block(block, 1, need_pinned=False, wire=W)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
