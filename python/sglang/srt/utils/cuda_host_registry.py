"""Host memory ranges registered with ``cudaHostRegister`` by this process.

Unregistration often runs from a garbage-collection finalizer (a tier's slabs
released by ``weakref.finalize``), and a collection can start inside any locked
lookup here. ``forget_cuda_host_registration`` called by the thread that already
holds the lock therefore queues the forget instead of waiting for itself, and
every locked function applies the queued forgets first, so a lookup in progress
never sees the tables change under it.
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import threading
from collections.abc import Iterator

import torch

_LOCK = threading.Lock()
_OWNER: int | None = None  # the thread holding _LOCK; only that thread reads its own id
_BASES: list[int] = []
_SIZES: dict[int, int] = {}
_DEFERRED_FORGETS: collections.deque[int] = collections.deque()


@contextlib.contextmanager
def _locked() -> Iterator[None]:
    """Holds the lock, with every queued forget applied."""
    global _OWNER
    with _LOCK:
        _OWNER = threading.get_ident()
        try:
            while _DEFERRED_FORGETS:
                _forget_locked(_DEFERRED_FORGETS.popleft())
            yield
        finally:
            _OWNER = None


def _forget_locked(base: int) -> None:
    if _SIZES.pop(base, None) is not None:
        del _BASES[bisect.bisect_left(_BASES, base)]


def record_cuda_host_registration(base: int, size: int) -> None:
    """Record one successful ``cudaHostRegister(base, size)`` call."""
    with _locked():
        if base not in _SIZES:
            bisect.insort(_BASES, base)
        _SIZES[base] = size


def forget_cuda_host_registration(base: int) -> None:
    """Forget a range after its successful ``cudaHostUnregister(base)``.

    Called from inside a locked lookup on the same thread (a finalizer run by a
    collection), it is queued and applied by the next locked call.
    """
    if _OWNER == threading.get_ident():
        _DEFERRED_FORGETS.append(base)
        return
    with _locked():
        _forget_locked(base)


def is_cuda_host_registered(tensor: torch.Tensor) -> bool:
    """Whether all of ``tensor``'s bytes lie inside back-to-back registrations.

    ``_cuda_host_register`` splits large buffers into adjacent chunks, so one
    tensor can span several recorded ranges.
    """
    if tensor.device.type != "cpu":
        return False
    start = tensor.data_ptr()
    end = start + tensor.numel() * tensor.element_size()
    with _locked():
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
    with _locked():
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
