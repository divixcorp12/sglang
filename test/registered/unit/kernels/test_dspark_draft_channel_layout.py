"""The draft channel's offsets and limits: Python's DraftWire mirrors draft_channel.h (CPU)."""

from sglang.kernels.ops.moe import expert_lease_block
from sglang.kernels.ops.moe.dspark_draft_cpu import DraftWire
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def test_the_python_mirror_is_the_header():
    probe = expert_lease_block.channel_probe(8, 1)
    w = DraftWire()
    assert (probe["draft_head"], probe["draft_ring"], probe["draft_records"], probe["draft_record_bytes"]) == (
        w.head,
        w.ring,
        w.records,
        w.record_bytes,
    )
    assert (probe["draft_done"], probe["draft_gate"], probe["draft_channel_bytes"]) == (w.done, w.gate, w.channel_bytes)
    assert (probe["draft_max_rows"], probe["draft_max_k"]) == (w.max_rows, w.max_k)
    assert (probe["draft_rec_stage"], probe["draft_rec_rows"], probe["draft_rec_k"], probe["draft_rec_epoch"]) == (
        w.rec_stage,
        w.rec_rows,
        w.rec_k,
        w.rec_epoch,
    )


def test_the_areas_start_with_an_open_gate():
    """An untouched gate word of 0 is below kGateOpen, so the first wait would block forever: it starts at open(0)."""
    from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuAreas

    areas = DraftCpuAreas(stages=2, hidden=64, pin=False)
    w = DraftWire()
    gate = int.from_bytes(areas.channel[w.gate : w.gate + 4].numpy().tobytes(), "little")
    assert gate == expert_lease_block.gate_word(0, "open") == 1
    assert areas.channel.shape == (w.channel_bytes,)
    assert tuple(areas.x.shape) == (2, w.max_rows, 64) and tuple(areas.slots.shape) == (2, w.max_rows, w.max_k)
    assert tuple(areas.weights.shape) == (2, w.max_rows, w.max_k) and tuple(areas.out.shape) == (2, w.max_rows, 64)
