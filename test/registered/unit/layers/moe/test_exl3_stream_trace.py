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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
