"""Host memory ranges registered with ``cudaHostRegister`` by this process."""

from __future__ import annotations

import bisect
import threading

import torch

_LOCK = threading.Lock()
_BASES: list[int] = []
_SIZES: dict[int, int] = {}


def record_cuda_host_registration(base: int, size: int) -> None:
    """Record one successful ``cudaHostRegister(base, size)`` call."""
    with _LOCK:
        if base not in _SIZES:
            bisect.insort(_BASES, base)
        _SIZES[base] = size


def forget_cuda_host_registration(base: int) -> None:
    """Forget a range after its successful ``cudaHostUnregister(base)``."""
    with _LOCK:
        if _SIZES.pop(base, None) is not None:
            del _BASES[bisect.bisect_left(_BASES, base)]


def is_cuda_host_registered(tensor: torch.Tensor) -> bool:
    """Whether all of ``tensor``'s bytes lie inside back-to-back registrations.

    ``_cuda_host_register`` splits large buffers into adjacent chunks, so one
    tensor can span several recorded ranges.
    """
    if tensor.device.type != "cpu":
        return False
    start = tensor.data_ptr()
    end = start + tensor.numel() * tensor.element_size()
    with _LOCK:
        position = bisect.bisect_right(_BASES, start) - 1
        if position < 0:
            return False
        covered = _BASES[position] + _SIZES[_BASES[position]]
        if covered <= start:
            return False
        while covered < end:
            position += 1
            if position == len(_BASES) or _BASES[position] != covered:
                return False
            covered += _SIZES[covered]
        return True


def cuda_host_registration_end(address: int) -> int | None:
    """End address of the recorded registration holding ``address``, or None."""
    with _LOCK:
        position = bisect.bisect_right(_BASES, address) - 1
        if position < 0:
            return None
        end = _BASES[position] + _SIZES[_BASES[position]]
        return end if address < end else None


def is_gpu_readable_host_tensor(tensor: torch.Tensor) -> bool:
    """Whether CUDA kernels and non-blocking copies can read ``tensor`` in place.

    ``Tensor.is_pinned`` recognizes only PyTorch's own pinned allocations: on
    torch 2.13 it returns False for memory registered with ``cudaHostRegister``,
    although CUDA reports that memory as host-registered and kernels read it.
    """
    return tensor.device.type == "cpu" and (
        tensor.is_pinned() or is_cuda_host_registered(tensor)
    )
