"""G/f counters and the per-call trace of streamed EXL3 MoE calls (CPU)."""

import json
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.exl3_stream_trace import Exl3StreamTrace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _stats(miss, read, read_ns=0, split_ns=0):
    return SimpleNamespace(miss_rows=miss, host_read_rows=read, host_read_ns=read_ns, host_split_ns=split_ns)


def test_decode_counters_give_G_and_f():
    trace = Exl3StreamTrace()
    # forward 1 (decode): layer 0 misses 2 in VRAM, both read from disk; layer 1 misses 1, a RAM hit
    trace.record(0, torch.tensor([[0, 1]]), _stats(2, 2), 0)
    trace.record(1, torch.tensor([[2, 3]]), _stats(1, 0), 0)
    # forward 2 (decode): layer 0 hits in VRAM; layer 1 misses 1, read from disk
    trace.record(0, torch.tensor([[1, 0]]), _stats(0, 0), 0)
    trace.record(1, torch.tensor([[0, 2]]), _stats(1, 1), 0)
    stats = trace.stats()
    assert (stats["forwards"], stats["decode_tokens"]) == (2, 2)
    assert (stats["decode_vram_misses"], stats["decode_ram_misses"]) == (4, 3)
    assert stats["G"] == pytest.approx(2.0)
    assert stats["f"] == pytest.approx(0.75)


def test_prefill_calls_are_counted_but_not_as_decode():
    trace = Exl3StreamTrace()
    trace.record(0, torch.tensor([[0, 1], [2, 3]]), _stats(4, 4), 0)
    trace.record(0, torch.tensor([[0, 1]]), _stats(1, 0), 0)
    stats = trace.stats()
    assert (stats["forwards"], stats["decode_tokens"]) == (2, 1)
    assert (stats["vram_misses"], stats["ram_misses"]) == (5, 4)
    assert (stats["decode_vram_misses"], stats["decode_ram_misses"]) == (1, 0)
    assert (stats["G"], stats["f"]) == (1.0, 0.0)


def test_a_call_without_a_gather_counts_no_misses():
    trace = Exl3StreamTrace()
    trace.record(3, torch.tensor([[1, 2]]), None, 0)
    stats = trace.stats()
    assert (stats["forwards"], stats["decode_tokens"], stats["vram_misses"]) == (1, 1, 0)


def test_trace_lines_carry_route_counts_and_background_reads(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = Exl3StreamTrace(str(path))
    trace.record(0, torch.tensor([[3, 1], [1, 1]]), _stats(2, 1, read_ns=2_000_000, split_ns=500_000), 5)
    trace.record(1, torch.tensor([[2, 2]]), _stats(1, 0), 0)
    trace.record(0, torch.tensor([[1, 0]]), _stats(0, 0), 7)
    trace.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    stamps = [line.pop("t") for line in lines]
    assert stamps == sorted(stamps)
    assert lines == [
        {"forward": 1, "layer": 0, "tokens": 2, "experts": [1, 3], "counts": [3, 1],
         "vram_miss": 2, "ram_miss": 1, "read_ms": 2.0, "split_ms": 0.5, "background_rows": 5},
        {"forward": 1, "layer": 1, "tokens": 1, "experts": [2], "counts": [2],
         "vram_miss": 1, "ram_miss": 0, "read_ms": 0.0, "split_ms": 0.0, "background_rows": 0},
        {"forward": 2, "layer": 0, "tokens": 1, "experts": [0, 1], "counts": [1, 1],
         "vram_miss": 0, "ram_miss": 0, "read_ms": 0.0, "split_ms": 0.0, "background_rows": 2},
    ]
    assert trace.stats()["background_read_rows"] == 7


def test_logs_every_n_forwards_and_at_close(caplog):
    trace = Exl3StreamTrace(log_every=2)
    with caplog.at_level("INFO", logger="sglang.srt.layers.moe.exl3_stream_trace"):
        for layer in (0, 0, 0):
            trace.record(layer, torch.tensor([[0]]), _stats(1, 1), 0)
        trace.close()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("exl3 expert stream:")]
    assert len(lines) == 2
    assert json.loads(lines[-1].split(": ", 1)[1])["decode_tokens"] == 3


def test_capture_forwards_are_not_recorded(monkeypatch):
    from sglang.srt.layers.moe import exl3_stream_trace as module
    from sglang.srt.model_executor.runner_utils import capture_mode

    trace = module.Exl3StreamTrace()
    monkeypatch.setattr(capture_mode, "is_capture_mode", True)
    assert module.capturing_graphs()
    trace.record(0, torch.zeros((1, 6), dtype=torch.long), None, 0)
    assert trace.forwards == 0 and trace.decode_tokens == 0
    monkeypatch.setattr(capture_mode, "is_capture_mode", False)
    assert not module.capturing_graphs()
    trace.record(0, torch.zeros((1, 6), dtype=torch.long), None, 0)
    assert trace.forwards == 1 and trace.decode_tokens == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


def _stage_record(**over):
    record = dict(
        seq=7, kind="demand", row=1, ok=1, rows=2, batches=1, backlog=0, prev_done=90,
        observed=100, reserved=110, submit=120, first_cqe=200, last_cqe=210, pack_start=220,
        pack_end=260, mapped=270, done=280, submit_to_first_cqe_ns=80, first_to_last_cqe_ns=10,
        pack_ns=40, bytes=45380, extents=2, drives=[{"dev": 49, "bytes": 45380, "extents": 2}],
        status="served", rows_asked=2, missing_stages=[], useful_bytes=45000, submitted_bytes=45380,
        retried_bytes=0, cancelled_bytes=0, rows_untraced=0, extents_untraced=0, dropped_before=0, lanes=6,
        row_pack=[
            {"row": 0, "admit": 112, "start": 220, "end": 240},
            {"row": 1, "admit": 112, "start": 240, "end": 260},
        ],
        extent_cqe=[
            {"row": 0, "part": 0, "submit": 115, "attempts": 0, "cqe": 200},
            {"row": 1, "part": 0, "submit": 115, "attempts": 0, "cqe": 210},
        ],
    )
    record.update(over)
    return record


def test_ram_miss_requests_are_traced_and_skipped_by_tier_sim(tmp_path):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "scripts", "dsv41"))
    import tier_sim

    path = tmp_path / "trace.jsonl"
    trace = Exl3StreamTrace(str(path))
    trace.record_graph_step(layer_rows_delta=[1, 0], routed_rows=6, routed_misses=2)
    trace.record_ram_miss_requests([_stage_record(), _stage_record(seq=8, row=0)], layer_ids=[3, 5])
    trace.record_ram_miss_requests([], layer_ids=[3, 5])  # nothing drained: no line
    trace.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["graph_step", "ram_miss_request", "ram_miss_request"]
    first = lines[1]
    assert (first["layer"], first["forward"], lines[2]["layer"]) == (5, 1, 3)
    assert first["request"] == {"seq": 7, "type": "demand", "ok": 1, "rows": 2, "batches": 1, "backlog": 0, "lanes": 6}
    assert list(first["stages_ns"]) == [
        "observed", "reserved", "submit", "first_cqe", "last_cqe", "pack_start", "pack_end", "mapped", "done"
    ]
    assert first["prev_done_ns"] == 90 and first["spans_ns"]["pack"] == 40
    assert first["drives"] == [{"dev": 49, "bytes": 45380, "extents": 2}] and "tokens" not in first
    assert (first["status"], first["missing_stages"], first["rows_asked"]) == ("served", [], 2)
    assert first["byte_split"] == {
        "useful": 45000, "submitted": 45380, "completed": 45380, "retried": 0, "cancelled": 0
    } and first["bytes"] == 45380
    assert first["row_pack_ns"][1] == {"row": 1, "admit": 112, "start": 240, "end": 260}
    assert first["extent_cqe_ns"][0] == {"row": 0, "part": 0, "submit": 115, "attempts": 0, "cqe": 200}
    assert first["dropped_before"] == 0
    assert first["untraced"] == {"rows": 0, "extents": 0}
    # tier_sim reads the same file: the stage lines are not forward calls, so G and f are unchanged.
    calls = tier_sim.load_trace(str(path))
    assert [call["kind"] for call in calls] == ["graph_step"]
    assert tier_sim.live_summary(calls, warmup=0)["decode_tokens"] == 1


def test_ram_miss_requests_cost_nothing_without_a_trace_file():
    trace = Exl3StreamTrace()
    trace.record_ram_miss_requests([_stage_record()], layer_ids=[0, 1])  # no file: no work, no error


def _tier_sim():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "scripts", "dsv41"))
    import tier_sim

    return tier_sim


def _forward_routes(step, layers, width=6):
    """Distinct, step-dependent routes for every layer: no two steps or layers log the same list."""
    return [[(7 * step + 11 * row + k) % 97 for k in range(width)] for row in range(layers)]


def _log_forward(log, step, layers, count_of=lambda step, row: (step + row) % 4):
    for row, routes in enumerate(_forward_routes(step, layers)):
        log.record(row, torch.tensor(routes, dtype=torch.int64), torch.tensor([count_of(step, row)], dtype=torch.int32))


def test_graph_route_log_writes_every_serving_forward_once_and_no_warmup_forward(tmp_path, monkeypatch):
    """Warmup forwards execute the logged copies, so they sit in the ring below ``warmup``; a replay that
    read them would start from the capture's dummy routes. More forwards than the ring holds, read after
    each one, must all arrive: the ring wraps without losing or repeating a forward."""
    from sglang.srt.layers.moe import exl3_stream_trace as module

    path = tmp_path / "trace.jsonl"
    trace = module.Exl3StreamTrace(str(path))
    log = module.GraphRouteLog(layers=3, width=8, device="cpu", depth=8, margin=2)
    for row, layer in enumerate((4, 5, 9)):
        log.bind(row, layer, 28 + row)
    monkeypatch.setattr(module, "capturing_graphs", lambda: True)
    for step in range(2):
        _log_forward(log, 100 + step, 3)  # warmup routes: never written
    monkeypatch.setattr(module, "capturing_graphs", lambda: False)
    for step in range(20):
        _log_forward(log, step, 3)
        log.poll(trace)
    trace.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    header, steps = lines[0], lines[1:]
    assert header["kind"] == "graph_routes_header" and header["warmup_forwards"] == 2
    assert header["layer_ids"] == [4, 5, 9] and header["hot_capacity"] == [28, 29, 30]
    assert [line["seq"] for line in steps] == list(range(2, 22))
    for step, line in enumerate(steps):
        assert line["routes"] == _forward_routes(step, 3)
        assert line["misses"] == [(step + row) % 4 for row in range(3)]
        assert "dropped_before" not in line


def test_graph_route_log_says_where_a_lagging_reader_lost_forwards(tmp_path):
    from sglang.srt.layers.moe import exl3_stream_trace as module

    path = tmp_path / "trace.jsonl"
    trace = module.Exl3StreamTrace(str(path))
    log = module.GraphRouteLog(layers=1, width=8, device="cpu", depth=4, margin=1)
    _log_forward(log, 0, 1)
    log.poll(trace)
    for step in range(1, 7):  # six forwards between reads: the ring holds four
        _log_forward(log, step, 1)
    log.poll(trace)
    trace.close()
    steps = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert [line["seq"] for line in steps] == [0, 3, 4, 5, 6]
    assert steps[1]["dropped_before"] == 2 and steps[1]["routes"] == _forward_routes(3, 1)
    assert _tier_sim().load_forwards(str(path))["dropped"] == 2


def test_eager_forwards_carry_the_graph_seq_and_tier_sim_interleaves_them(tmp_path):
    """A route line is written a batch late, after the prefill that ran behind it; only the stamps put
    the replay back in execution order: graph forwards 0-1, prefill, graph forward 2."""
    from sglang.srt.layers.moe import exl3_stream_trace as module

    path = tmp_path / "trace.jsonl"
    trace = module.Exl3StreamTrace(str(path))
    log = module.GraphRouteLog(layers=2, width=8, device="cpu", depth=8, margin=2)
    log.bind(0, 0, 12)
    log.bind(1, 1, 12)
    trace.graph_seq_source = log.read_seq
    for step in range(2):
        _log_forward(log, step, 2)
    trace.record(0, torch.tensor([[1, 2], [2, 3]]), _stats(2, 0), 0)
    trace.record(1, torch.tensor([[4, 5], [5, 5]]), _stats(1, 0), 0)
    _log_forward(log, 2, 2)
    log.poll(trace)
    trace.close()
    loaded = _tier_sim().load_forwards(str(path))
    assert [(f["kind"], f.get("seq")) for f in loaded["forwards"]] == [
        ("graph", 0), ("graph", 1), ("eager", None), ("graph", 2)
    ]
    prefill = loaded["forwards"][2]
    assert prefill["tokens"] == 2 and prefill["counts"][1] == ([4, 5], [1, 3])
    assert loaded["forwards"][3]["routes"] == {0: _forward_routes(2, 2)[0], 1: _forward_routes(2, 2)[1]}
    assert loaded["hot_capacity"] == {0: 12, 1: 12}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
def test_graph_route_log_records_each_replay_of_a_captured_graph(tmp_path, monkeypatch):
    """The copies are captured once and must log each replay's own routes, read back without waiting,
    one batch behind, on the host's stream while the graph replays on another."""
    from sglang.srt.layers.moe import exl3_stream_trace as module

    layers, depth = 3, 8
    path = tmp_path / "trace.jsonl"
    trace = module.Exl3StreamTrace(str(path))
    log = module.GraphRouteLog(layers=layers, width=8, device="cuda", depth=depth, margin=2)
    for row in range(layers):
        log.bind(row, row, 12)
    routes = [torch.full((6,), -1, dtype=torch.int64, device="cuda") for _ in range(layers)]
    counts = [torch.zeros(1, dtype=torch.int32, device="cuda") for _ in range(layers)]

    def forward():
        for row in range(layers):
            log.record(row, routes[row], counts[row])

    side = torch.cuda.Stream()
    monkeypatch.setattr(module, "capturing_graphs", lambda: True)
    with torch.cuda.stream(side):
        forward()  # warmup, executed
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            forward()
    monkeypatch.setattr(module, "capturing_graphs", lambda: False)
    torch.cuda.synchronize()
    assert log.warmup == 1
    steps = 3 * depth
    for step in range(steps):
        with torch.cuda.stream(side):
            for row, values in enumerate(_forward_routes(step, layers)):
                routes[row].copy_(torch.tensor(values, device="cuda"), non_blocking=True)
                counts[row].fill_((step + row) % 4)
            graph.replay()
        log.poll(trace)  # the current stream is not the replay stream
    torch.cuda.synchronize()
    log.poll(trace, final=True)
    trace.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()][1:]
    assert [line["seq"] for line in lines] == list(range(1, steps + 1))
    for step, line in enumerate(lines):
        assert line["routes"] == _forward_routes(step, layers)
        assert line["misses"] == [(step + row) % 4 for row in range(layers)]
