"""The expert_stream device headers use CUDA intrinsics and libcu++ for synchronization, not hand-written PTX (CPU).

Each rule here pins a convention whose machine-code equivalence was proven by a SASS gate when it was adopted
(docs/superpowers/plans/2026-09-27-expert-stream-native-sync.md), so a later edit cannot quietly reintroduce the
old form.
"""

import re

import pytest

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


def kernel_body(name: str, kernel: str) -> str:
    """The code of `kernel` in header `name`, comments and blank lines removed, up to its closing brace at column 0."""
    text = "\n".join(code for _, code in code_lines(name))
    start = text.index(f"void {kernel}(")
    return text[start : text.index("\n}", start)]


def test_the_three_device_headers_are_found():
    assert sorted(HEADERS) == ["lease_device.cuh", "lease_kernels.cuh", "row_copy_kernels.cuh"]


def test_cache_hinted_copies_and_the_timer_use_intrinsics_not_ptx():
    assert matches(r"ld\.global\.cv|st\.global\.cg|%globaltimer") == []


def test_inline_ptx_lives_only_in_the_lease_device_helpers():
    assert sorted({name for name, _, _ in matches(r"\basm\b")}) == ["lease_device.cuh"]


def test_volatile_lives_only_in_the_relaxed_helpers():
    # Every concurrent access goes through ld/st_relaxed_sys, ld_acquire_sys{,64}, st_release_sys{,64} or
    # ld_relaxed_gpu; a raw volatile cast at a call site hides which ordering it relies on.
    assert [(name, code) for name, _, code in matches(r"\bvolatile\b") if not code.startswith("asm")] == [
        ("lease_device.cuh", "return *reinterpret_cast<const volatile T*>(word);"),
        ("lease_device.cuh", "*reinterpret_cast<volatile T*>(word) = value;"),
    ]


def test_only_the_post_staging_and_cws_done_and_gate_close_keep_a_seq_cst_system_fence():
    # The post's CPU-input staging: every thread's stores of x to the host row, ordered through __syncthreads before
    # thread 0 publishes the request. CW's Done: its block's SM reads, ordered through __syncthreads, performed at
    # system scope before the host can see Done and rewrite a slot. CW's gate close orders its store before its
    # CopyDone load (a Dekker pair with the copy thread's seq_cst fence, LEASE_PROTOCOL.md "Copy engine"):
    # store->load needs seq_cst.
    assert [(name, code) for name, _, code in matches(r"__threadfence_system\(\)")] == [
        ("lease_kernels.cuh", "__threadfence_system();"),
        ("row_copy_kernels.cuh", "__threadfence_system();"),
        ("row_copy_kernels.cuh", "__threadfence_system();"),
    ]


def test_the_seqlock_writers_fences_are_three_releases():
    # write_record, the hot bitmap record and the LaneRequest: seq (or gen) 0, a release fence, the payload, then the
    # word with a release store. The device reads no seqlock, so it has no acquire fence.
    assert [(name, code) for name, _, code in matches(r"atomic_thread_fence")] == [
        ("lease_device.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
        ("lease_kernels.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
        ("lease_kernels.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
    ]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
