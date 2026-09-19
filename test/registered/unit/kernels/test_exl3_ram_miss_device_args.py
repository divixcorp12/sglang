"""The device wrapper refuses pages and slot maps the kernels cannot address (CPU)."""

import ast
import operator
import re
from pathlib import Path

import pytest
import torch

import sglang.kernels.ops.moe.exl3_ram_miss as ram_miss
from sglang.kernels.ops.moe.exl3_ram_miss import PAGE_BYTES, STATE_WORDS, Exl3RamMissDevice
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CSRC = Path(ram_miss.__file__).resolve().parents[2] / "jit" / "csrc" / "moe"


def test_state_words_are_distinct_and_dense():
    assert sorted(STATE_WORDS.values()) == list(range(len(STATE_WORDS)))


def test_a_page_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="page"):
        Exl3RamMissDevice(torch.zeros(10, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=10, advise=False)


def test_a_slot_map_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="slot_map"):
        Exl3RamMissDevice(torch.zeros(PAGE_BYTES, dtype=torch.uint8), torch.zeros((3, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=10, advise=False)


def test_the_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout"):
        Exl3RamMissDevice(torch.zeros(PAGE_BYTES, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=0, advise=False)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def _evaluate(node, known):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return known[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand, known)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left, known), _evaluate(node.right, known))
    raise ValueError(f"unsupported constant expression {ast.dump(node)}")


def _constants(path: Path) -> dict[str, int]:
    """Every ``constexpr <type> kName = <integer expression>;`` of a C++ source."""
    known: dict[str, int] = {}
    pattern = r"^\s*(?:static\s+)?constexpr\s+[\w:]+\s+(k\w+)\s*=\s*([^;]+);"
    for name, expression in re.findall(pattern, path.read_text(), re.MULTILINE):
        expression = re.sub(r"(?<=\d)[uU][lL]*\b", "", expression.strip())
        known[name] = _evaluate(ast.parse(expression, mode="eval").body, known)
    return known


def test_the_device_kernels_speak_the_host_page_layout():
    """The page protocol is written three times (Python, host C++, device CUDA): one layout."""
    device = _constants(CSRC / "exl3_ram_miss.cuh")
    host = _constants(CSRC / "exl3_ram_miss_host.cpp")
    shared = sorted(set(device) & set(host) - {"kBlock"})
    assert {"kDemandRing", "kRecordBytes", "kRecStatus", "kRecSeq", "kServed", "kFatal"} <= set(shared)
    assert {name: device[name] for name in shared} == {name: host[name] for name in shared}
    python = {
        "kDemandHead": ram_miss.WORDS["demand_head"],
        "kDemandDone": ram_miss.WORDS["demand_done"],
        "kFatal": ram_miss.WORDS["fatal"],
        "kAdviseHead": ram_miss.WORDS["advise_head"],
        "kRecordBytes": ram_miss.RECORD_BYTES,
        "kDemandRing": ram_miss.DEMAND_RING,
        "kDemandRecords": ram_miss.DEMAND_RECORDS,
        "kAdviseRing": ram_miss.ADVISE_RING,
        "kAdviseRecords": ram_miss.ADVISE_RECORDS,
        "kMaxIds": ram_miss.MAX_IDS,
        "kServed": ram_miss.STATUS["served"],
    }
    assert {name: device[name] for name in python} == python
    assert device["kAdviseRing"] + device["kAdviseRecords"] * device["kRecordBytes"] <= PAGE_BYTES
    state = {
        "kPosted": "posted",
        "kPending": "pending",
        "kTimeouts": "timeouts",
        "kFailures": "failures",
        "kWaits": "waits",
        "kPolls": "polls",
        "kSticky": "sticky",
        "kAdvised": "advised",
        "kUnservedMisses": "unserved_misses",
    }
    assert {word: device[name] for name, word in state.items()} == STATE_WORDS


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
