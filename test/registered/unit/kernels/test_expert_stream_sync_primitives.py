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


def test_only_the_smack_publish_and_the_copy_wait_arm_keep_a_seq_cst_system_fence():
    # The five seqlock fences are acquire/release (Boehm's shape, paired with the host's); SmAck orders other
    # threads' loads through __syncthreads before thread 0's release, a different argument, so it stays seq_cst.
    # The copy wait's arm orders its CopyArm store before its loads of the fatal and shutdown words (a Dekker pair
    # with the host's seq_cst fence after raising either, LEASE_PROTOCOL.md 7.6): store->load needs seq_cst.
    # The post's CPU-input staging is SmAck's shape: every thread's stores of x to the host row, ordered through
    # __syncthreads before thread 0 releases the LaneRequest the CPU expert thread acquires.
    assert [(name, code) for name, _, code in matches(r"__threadfence_system\(\)")] == [
        ("lease_kernels.cuh", "__threadfence_system();"),
        ("row_copy_kernels.cuh", "__threadfence_system();"),
        ("row_copy_kernels.cuh", "__threadfence_system();"),
    ]


def test_the_seqlock_fences_are_two_acquires_and_three_releases():
    assert [(name, code) for name, _, code in matches(r"atomic_thread_fence")] == [
        ("lease_device.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
        ("lease_device.cuh", "cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);"),
        ("lease_device.cuh", "cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);"),
        ("lease_kernels.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
        ("lease_kernels.cuh", "cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);"),
    ]


@pytest.mark.parametrize("kernel", ["exl3_ram_miss_lease_stage_ack_kernel", "exl3_ram_miss_lease_ack_kernel"])
def test_the_ack_kernels_combine_lane_verdicts_with_syncthreads_or_not_a_shared_flag(kernel):
    # Several lanes storing to one __shared__ int is a data race; __syncthreads_or combines the verdicts, and it is
    # reached by every thread at the kernel's top level, never inside the lane branch.
    body = kernel_body("lease_kernels.cuh", kernel)
    assert "__shared__" not in body
    assert re.findall(r"__syncthreads\w*\(", body) == ["__syncthreads_or("]
    assert re.search(r"^  const int any = __syncthreads_or\(bad\);$", body, re.MULTILINE)


def test_rest_wait_bounds_its_claimed_loop_before_the_count_check():
    # `claimed` is kLeaseLanes wide and the plan's count is not clamped (lease_layout.h), so the loop that counts
    # the unclaimed lanes runs before the count is refused and must carry its own bound.
    body = kernel_body("lease_kernels.cuh", "exl3_ram_miss_lease_rest_wait_kernel")
    loop = re.search(r"for \(int64_t i = 0; i < (\w+); \+\+i\)\s*if \(claimed\[i\] == 0\) \+\+unclaimed;", body)
    assert loop, "the unclaimed-lane loop is missing"
    bound = re.search(
        rf"const int64_t {loop.group(1)} = min\(planned_count, min\(static_cast<int64_t>\(kLeaseLanes\), lanes\)\);", body
    )
    assert bound and bound.start() < loop.start() < body.index("reason = kLeaseReasonCount")
