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


def _device(layers=2, experts=4, page=None):
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8) if page is None else page
    slot_map = torch.full((layers, experts), -1, dtype=torch.int32)
    return Exl3RamMissDevice(page, slot_map, device="cpu", layers=layers, timeout_ms=10, advise=False)


def _args(lanes=6):
    return dict(
        planned=torch.zeros(lanes, dtype=torch.int64),
        count=torch.zeros(1, dtype=torch.int32),
        routes=torch.full((lanes,), -1, dtype=torch.int64),
        host_rows=torch.zeros(lanes, dtype=torch.int64),
        keep=torch.ones(1, dtype=torch.float32),
        ram_miss=torch.zeros(1, dtype=torch.int64),
    )


def _post(dev, a, row=0, next_row=-1):
    dev.post(row, a["planned"], a["count"], a["routes"], next_row)


def _wait(dev, a, row=0):
    dev.wait(row, a["planned"], a["count"], a["host_rows"], a["keep"], a["ram_miss"])


def test_an_unpinned_page_or_slot_map_is_refused_for_a_cuda_device():
    # Checked before any CUDA call: the kernels read both through UVA.
    with pytest.raises(ValueError, match="pinned"):
        Exl3RamMissDevice(torch.zeros(PAGE_BYTES, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.int32), device="cuda", layers=2, timeout_ms=10, advise=False)


def test_a_row_outside_the_streamed_layers_is_refused():
    dev, a = _device(layers=2), _args()
    for row in (-1, 2):
        with pytest.raises(ValueError, match="row"):
            _post(dev, a, row=row)
        with pytest.raises(ValueError, match="row"):
            _wait(dev, a, row=row)
    for next_row in (-2, 2):
        with pytest.raises(ValueError, match="next_row"):
            _post(dev, a, next_row=next_row)


def test_buffers_of_the_wrong_dtype_are_refused():
    dev = _device()
    for name, dtype in (("planned", torch.int32), ("count", torch.int64), ("routes", torch.int32)):
        a = _args()
        a[name] = a[name].to(dtype)
        with pytest.raises(ValueError, match=name):
            _post(dev, a)
    for name, dtype in (("planned", torch.int32), ("count", torch.int64), ("host_rows", torch.int32), ("keep", torch.float16), ("ram_miss", torch.int32)):
        a = _args()
        a[name] = a[name].to(dtype)
        with pytest.raises(ValueError, match=name):
            _wait(dev, a)


def test_planned_shorter_than_host_rows_is_refused():
    dev, a = _device(), _args()
    a["planned"] = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError, match="planned"):
        _wait(dev, a)


def test_the_device_sequences_continue_from_the_page_heads():
    """A device built over a used page posts the next sequence, not 1, which the thread would never serve."""
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    page[ram_miss.WORDS["demand_head"] : ram_miss.WORDS["demand_head"] + 4].view(torch.int32)[0] = 7
    page[ram_miss.WORDS["advise_head"] : ram_miss.WORDS["advise_head"] + 4].view(torch.int32)[0] = -5  # 2**32 - 5
    dev = _device(page=page)
    assert int(dev.state[STATE_WORDS["posted"]]) == 7
    assert int(dev.state[STATE_WORDS["advised"]]) & 0xFFFFFFFF == 2**32 - 5


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
    """Every ``constexpr <type> kName = <integer expression>;`` of a C++ source; a name defined twice fails."""
    known: dict[str, int] = {}
    pattern = r"^\s*(?:static\s+)?constexpr\s+[\w:]+\s+(k\w+)\s*=\s*([^;]+);"
    for name, expression in re.findall(pattern, path.read_text(), re.MULTILINE):
        assert name not in known, f"{path.name} defines {name} twice: the layout check cannot tell which one applies"
        expression = re.sub(r"(?<=\d)[uU][lL]*\b", "", expression.strip())
        known[name] = _evaluate(ast.parse(expression, mode="eval").body, known)
    return known


# The page protocol (plan D10): every name here must be defined, with one value, in both C++ files.
PAGE_PROTOCOL = (
    "kDemandHead", "kDemandDone", "kFatal", "kAdviseHead", "kRecordBytes", "kDemandRing", "kDemandRecords",
    "kAdviseRing", "kAdviseRecords", "kMaxIds", "kRecSeq", "kRecRow", "kRecNeedCount", "kRecProtectCount",
    "kRecStatus", "kRecAfter", "kRecNeed", "kRecProtect", "kRecArmed", "kServed",
)


def test_the_device_kernels_speak_the_host_page_layout():
    """The page protocol is written three times (Python, host C++, device CUDA): one layout."""
    files = {name: _constants(CSRC / name) for name in ("exl3_ram_miss.cuh", "exl3_ram_miss_host.cpp")}
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
    reference = files["exl3_ram_miss_host.cpp"]
    for file, constants in files.items():
        for name in PAGE_PROTOCOL:
            assert name in constants, (file, name)
            assert constants[name] == reference[name], (file, name, constants[name], reference[name])
        for name, value in python.items():
            assert constants[name] == value, (file, name, constants[name], value)
    device = files["exl3_ram_miss.cuh"]
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
