"""DraftResidentMoe's route remap: resident ids to their slot, everything else to the sink (CPU)."""

import pytest
import torch

from sglang.srt.layers.quantization.exl3.draft_moe import DraftResidentMoe, draft_slot_map
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_resident_ids_map_to_their_slot_and_the_rest_to_the_sink():
    table = draft_slot_map([3, 0, 7], n_experts=8, device="cpu")
    ids = torch.tensor([[0, 3, 5], [7, -1, 8], [1, 0, 2]])
    remap = DraftResidentMoe.remap_with(table, ids, n_experts=8)
    sink = 3
    assert remap.tolist() == [0, 1, sink, 2, sink, sink, sink, 0, sink]
    assert remap.dtype == torch.int64


def test_slots_follow_the_sorted_resident_ids():
    table = draft_slot_map([5, 2], n_experts=6, device="cpu")
    assert table[:6].tolist() == [2, 2, 0, 2, 2, 1]
    assert table[6].item() == 2  # the bad-id entry is the sink too


def test_a_resident_id_outside_the_experts_is_refused():
    with pytest.raises(ValueError, match="outside"):
        draft_slot_map([1, 6], n_experts=6, device="cpu")
