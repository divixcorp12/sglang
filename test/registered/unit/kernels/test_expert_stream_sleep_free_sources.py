"""The long I/O waits must not retain a resident GPU polling loop.

Runtime eager/replay/timeout coverage lives in test_expert_stream_sleep_free_cuda.py.
This static gate also runs on builders without a GPU.
"""

from pathlib import Path


def test_io_completion_kernels_do_not_sleep_or_spin():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh"
    ).read_text()
    for kernel in (
        "exl3_ram_miss_wait_kernel",
        "exl3_ram_miss_lease_wait_kernel",
        "exl3_ram_miss_lease_rest_wait_kernel",
    ):
        start = source.index("void " + kernel + "(")
        body = source[start:source.index("\n}", start)]
        assert "__nanosleep" not in body, f"{kernel} still sleeps on the GPU"
        assert "while (" not in body, f"{kernel} still polls on the GPU"
