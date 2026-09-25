"""The DSV4 MoE layer's side stream (SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM).

Work that neither reads the routed experts' rows nor feeds the routed MoE kernel is forked onto one process-wide
side stream, and the MoE layer joins it before it adds the shared expert. Every fork in a layer is joined in that
same layer, so the next layer's RAM-miss chain never runs beside a fork that reads the chain's shared outputs.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, TypeVar

import torch

from sglang.srt.environ import envs

T = TypeVar("T")

_stream: Optional[torch.cuda.Stream] = None
_forked = False


def enable_if_requested() -> None:
    """Create the side stream once, at model construction: creating one during graph capture is refused."""
    global _stream
    if _stream is None and envs.SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM.get() and torch.cuda.is_available():
        _stream = torch.cuda.Stream()


def active() -> bool:
    return _stream is not None


def fork(fn: Callable[[], T], *, inputs: Sequence[torch.Tensor] = ()) -> T:
    """Run ``fn`` on the side stream after everything already issued on the current stream.

    ``inputs`` are tensors ``fn`` reads that the current stream allocated and may free before the join; recording
    them on the side stream keeps the caching allocator from handing their blocks back to the current stream early.
    """
    global _forked
    assert _stream is not None
    _stream.wait_stream(torch.cuda.current_stream())
    for tensor in inputs:
        tensor.record_stream(_stream)
    _forked = True
    with torch.cuda.stream(_stream):
        return fn()


def join(outputs: Sequence[torch.Tensor] = ()) -> None:
    """Order the current stream after every fork since the last join; ``outputs`` are the forks' results it reads."""
    global _forked
    if not _forked:
        return
    current = torch.cuda.current_stream()
    current.wait_stream(_stream)
    for tensor in outputs:
        tensor.record_stream(current)
    _forked = False
