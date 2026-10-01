"""The completion and delta blocks' layout, their allocator and the gate word (CPU); analysis/dsv41-drive/LEASE_PROTOCOL.md."""

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_the_areas_are_where_the_protocol_puts_them():
    assert lease.PIECE_MASK == 0 and lease.COPY_DONE == 0x4000, "PieceMask[16][8] at a 128-byte line each"
    assert lease.COPY_GATE == lease.COPY_DONE + 128 and lease.COPY_ARMED == lease.COPY_GATE + 128
    assert lease.SPLIT == lease.COPY_ARMED + 128 and lease.SPLIT + 4 * (lease.LANES + 1) <= lease.BLOCK_BYTES
    assert lease.BLOCK_BYTES == 5 * lease.BLOCK_ALIGN and lease.DELTA_BASE == lease.BLOCK_BYTES


def test_a_delta_record_fits_its_stride_and_each_starts_a_new_line():
    f = lease.DELTA_FIELDS
    assert f["tag"] + 8 == f["count"] and f["staging"] >= f["count"] + 4
    assert f["entries"] == f["staging"] + 4 * lease.LANES
    assert f["entries"] + 8 * lease.DELTA_MAX_ENTRIES <= lease.DELTA_STRIDE and lease.DELTA_STRIDE % 128 == 0
    assert lease.DELTA_MAX_ENTRIES == 2 * lease.LANES, "an insert and an eviction per lane"


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
    block = lease.new_lease_block(rows, pin=False)
    assert block.numel() == lease.lease_block_bytes(rows) and block.data_ptr() % lease.BLOCK_ALIGN == 0
    assert block.numel() >= lease.DELTA_BASE + rows * lease.DELTA_STRIDE and block.numel() % lease.BLOCK_ALIGN == 0
    assert not block.any()


@pytest.mark.parametrize("bad", ["dtype", "size", "align", "shape"])
def test_a_block_the_kernels_cannot_address_is_refused(bad):
    size = lease.lease_block_bytes(2)
    raw = torch.zeros(size + 2 * lease.BLOCK_ALIGN, dtype=torch.uint8)
    good = raw[(-raw.data_ptr()) % lease.BLOCK_ALIGN :][:size]
    blocks = {
        "dtype": good.view(torch.int8),
        "size": good[:-1],
        "align": raw[((-raw.data_ptr()) % lease.BLOCK_ALIGN) + 1 :][:size],
        "shape": good.view(1, -1),
    }
    lease.check_lease_block(good, 2, need_pinned=False)
    with pytest.raises(ValueError):
        lease.check_lease_block(blocks[bad], 2, need_pinned=False)


def test_a_block_sized_for_other_rows_is_refused():
    with pytest.raises(ValueError, match="rows"):
        lease.check_lease_block(lease.new_lease_block(1, pin=False), 17, need_pinned=False)


def test_a_block_that_is_not_pinned_is_refused_for_a_cuda_device():
    with pytest.raises(ValueError, match="pinned"):
        lease.check_lease_block(lease.new_lease_block(1, pin=False), 1, need_pinned=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
