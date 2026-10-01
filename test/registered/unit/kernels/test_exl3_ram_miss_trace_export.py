"""The RAM-miss stage records as the stream trace writes them to JSONL (CPU)."""

import hashlib
import json

import torch

import sglang.kernels.ops.moe.expert_stream_transport as ops
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.srt.layers.moe.exl3_stream_trace import RAM_MISS_TRACE_SCHEMA, Exl3StreamTrace
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Changing the native record layout changes this digest. When it fails: bump RAM_MISS_TRACE_SCHEMA,
# then paste the new pair. A JSONL file has no field check of its own, so the schema integer is the
# only way a consumer learns which fields a file has.
LAYOUT_PIN = (7, "71405562602a9b5f")


def _layout_digest():
    return hashlib.sha256("\n".join(ops.STAGE_FIELDS).encode()).hexdigest()[:16]


def test_the_schema_moves_with_the_record_layout():
    assert (RAM_MISS_TRACE_SCHEMA, _layout_digest()) == LAYOUT_PIN


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_request_line_carries_the_causal_stamps_and_the_drop_position(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=6)
    page = new_page(pin=False)
    host = attached_host(s, page)
    host.enable_trace(capacity=2)
    trace_path = tmp_path / "trace.jsonl"
    trace = Exl3StreamTrace(str(trace_path))
    sim = ChainSim(host, page, s.slabs)

    def serve(expert):
        req = sim.post(1, [expert])
        assert host.pump() == 1 and sim.wait_served(req, timeout_s=1.0)

    try:
        for expert in (2, 5, 4, 3):  # four requests through a two-slot ring: two are lost
            serve(expert)
        trace.record_ram_miss_requests(host.drain_trace(), [10, 11])
        serve(0)
        trace.record_ram_miss_requests(host.drain_trace(), [10, 11])
    finally:
        trace.close()
        host.stop()
    first, second, after_loss = _lines(trace_path)
    assert {line["schema"] for line in (first, second, after_loss)} == {RAM_MISS_TRACE_SCHEMA}
    assert [line["dropped_before"] for line in (first, second, after_loss)] == [0, 0, 2]
    assert first["layer"] == 11
    assert first["request"]["lanes"] == 1  # the layer and the lanes travel together
    (row,) = first["row_pack_ns"]
    assert set(row) == {"row", "admit", "start", "end"} and 0 < row["admit"] <= row["start"] <= row["end"]
    extents = first["extent_cqe_ns"]
    assert extents and all(set(e) == {"row", "part", "sub", "submit", "attempts", "cqe"} for e in extents)
    assert first["pieces"] and first["request"]["piece_stream"] == 1  # the service always streams pieces
    assert all(row["admit"] <= e["submit"] <= e["cqe"] for e in extents)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
