"""Post and wait kernels of the Engram device-wait lookup (SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT).

The control block is int32 words of pinned host memory shared with the service thread in
``sglang/srt/layers/engram_host_node.cpp``; the offsets below mirror ``ring::`` there and
``engram_ring_device::`` in ``jit/csrc/embeddings/engram_ring.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

POST_SEQ = 0
DONE_SEQ = 16
STATUS = 17
FATAL_SEQ = 32
FATAL_STATUS = 33
CONTROL_WORDS = 48

REFUSED_AFTER_FATAL = 4
DEVICE_TIMEOUT = 5
DEVICE_SAW_FATAL = 6


@cache_once
def _module() -> Module:
    return load_jit(
        "engram_ring",
        cuda_files=["embeddings/engram_ring.cuh"],
        cuda_wrappers=[("engram_ring_post", "engram_ring_post"), ("engram_ring_wait", "engram_ring_wait")],
    )


def engram_ring_post(
    ids_dev: torch.Tensor, ids_host: torch.Tensor, control: torch.Tensor, counter: torch.Tensor
) -> None:
    """Post ``ids_dev`` (int64 [n], CUDA) into ``ids_host`` (pinned) and release the next sequence."""
    _module().engram_ring_post(ids_dev, ids_host, control, counter)


def engram_ring_wait(
    control: torch.Tensor,
    counter: torch.Tensor,
    rows_host: torch.Tensor,
    rows_dev: torch.Tensor,
    status_dev: torch.Tensor,
    timeout_ns: int,
) -> None:
    """Spin until the service serves the posted sequence, then copy the rows; failures land in ``status_dev``."""
    _module().engram_ring_wait(control, counter, rows_host, rows_dev, status_dev, timeout_ns)
