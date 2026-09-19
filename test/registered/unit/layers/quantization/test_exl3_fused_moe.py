"""Pointer tables and route tables of the fused EXL3 MoE over hot-cache slots (CPU)."""

import pytest
import torch

from sglang.srt.layers.quantization.exl3_fused_moe import route_tables, slot_pointer_tables
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


def test_pointer_tables_address_each_slots_part():
    tensors = {name: torch.zeros((5, 2 if name.startswith("w13") else 1, 16), dtype=torch.int16) for name in NAMES}
    tables = slot_pointer_tables(tensors, 5)
    assert sorted(tables) == sorted(f"{p}_{k}" for p in ("gate", "up", "down") for k in ("trellis", "suh", "svh"))
    for slot in range(5):
        assert tables["gate_trellis"][slot].item() == tensors["w13_trellis"][slot, 0].data_ptr()
        assert tables["up_svh"][slot].item() == tensors["w13_svh"][slot, 1].data_ptr()
        assert tables["down_suh"][slot].item() == tensors["w2_suh"][slot, 0].data_ptr()
    assert all(t.dtype == torch.int64 and t.shape == (5,) for t in tables.values())


def test_route_tables_sort_routes_by_slot_and_scale_by_keep():
    remap = torch.tensor([4, 1, 3])
    count = torch.zeros(6, dtype=torch.long)
    weights = torch.tensor([0.5, 0.25, 0.125])
    inv_order, weight_sorted, det = route_tables(remap, count, torch.ones(3, dtype=torch.long), weights, torch.tensor([1.0]))
    assert count.tolist() == [0, 1, 0, 1, 1, 0]
    order = torch.argsort(remap)
    assert torch.equal(inv_order[order], torch.arange(3))
    assert weight_sorted.dtype == torch.float16 and weight_sorted.tolist() == [0.25, 0.125, 0.5]
    assert det[0].tolist() == [0, 0, 1, 1, 2, 3] and det[2].tolist() == [0, 1, 0, 1, 1, 0]
    _, dropped, _ = route_tables(remap, count, torch.ones(3, dtype=torch.long), weights, torch.tensor([0.0]))
    assert dropped.tolist() == [0.0, 0.0, 0.0]
    assert count.tolist() == [0] * 6  # a dropped layer runs no expert at all


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
