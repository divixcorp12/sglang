"""The target's lease channel (lease_channel_layout.h) names exactly the wire's own offsets: Task 1's port is
byte-identical."""

import pytest

from sglang.kernels.ops.moe import expert_lease_block
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


@pytest.mark.parametrize("lanes,nodes", [(8, 1), (8, 2), (24, 1), (32, 2)])
def test_the_target_channel_is_the_wire(lanes, nodes):
    probe = expert_lease_block.channel_probe(lanes, nodes)
    wire = expert_lease_block.wire_layout(lanes, nodes)
    assert probe["chan_head"] == wire.cpp_constants()["kDemandHead"] == 0
    assert probe["chan_ring"] == wire.demand_ring
    assert probe["chan_records"] == wire.demand_records
    assert probe["chan_record_bytes"] == wire.record_bytes
    assert probe["chan_done"] == wire.copy_done
    assert probe["chan_gate"] == wire.copy_gate


def test_the_gate_encoding_is_shared():
    probe = expert_lease_block.channel_probe(8, 1)
    for seq in (1, 2, 0x1FFFFFFF, 0x20000001):
        assert probe[f"gate_open_{seq}"] == expert_lease_block.gate_word(seq, "open")
        assert probe[f"gate_closed_{seq}"] == expert_lease_block.gate_word(seq, "closed")
