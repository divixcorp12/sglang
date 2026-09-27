"""The expert_stream device headers use CUDA intrinsics and libcu++ for synchronization, not hand-written PTX (CPU).

Each rule here pins a convention whose machine-code equivalence was proven by a SASS gate when it was adopted
(docs/superpowers/plans/2026-09-27-expert-stream-native-sync.md), so a later edit cannot quietly reintroduce the
old form.
"""

import re

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import device_sources

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

HEADERS = {path.name: path for path in device_sources() if path.parent.name == "expert_stream"}


def code_lines(name: str) -> list[tuple[int, str]]:
    """(line number, code) for every line of header `name`, with `//` comments removed."""
    lines = []
    for number, line in enumerate(HEADERS[name].read_text().splitlines(), 1):
        code = line.split("//", 1)[0]
        if code.strip():
            lines.append((number, code))
    return lines


def matches(pattern: str) -> list[tuple[str, int, str]]:
    return [
        (name, number, code.strip())
        for name in sorted(HEADERS)
        for number, code in code_lines(name)
        if re.search(pattern, code)
    ]


def test_the_three_device_headers_are_found():
    assert sorted(HEADERS) == ["lease_device.cuh", "lease_kernels.cuh", "row_copy_kernels.cuh"]


def test_cache_hinted_copies_and_the_timer_use_intrinsics_not_ptx():
    assert matches(r"ld\.global\.cv|st\.global\.cg|%globaltimer") == []


def test_no_inline_ptx_is_left():
    assert matches(r"\basm\b") == []


def test_volatile_is_left_only_on_the_hot_bitmap_bytes():
    # atomic_ref<uint8_t>::store compiles to a system-scope CAS loop, one PCIe read-modify-write per byte.
    assert [(name, code) for name, _, code in matches(r"\bvolatile\b")] == [
        ("lease_kernels.cuh", "volatile uint8_t* bits = reinterpret_cast<volatile uint8_t*>(hot + kHotHeaderBytes);"),
    ]
