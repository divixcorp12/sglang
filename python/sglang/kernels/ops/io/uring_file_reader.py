"""Batched positional file reads through io_uring into host memory."""

from __future__ import annotations

import os
import threading
import weakref
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

PAGE_BYTES = 4096

_SHARED_READER: Optional[UringFileReader] = None
_SHARED_READER_LOCK = threading.Lock()


@cache_once
def _uring_file_reader_type():
    import tvm_ffi

    module = load_jit(
        "uring_file_reader",
        cpp_files=["io/uring_file_reader.cpp"],
        extra_ldflags=["-luring"],
        header_only=False,
    )
    module.register_once()

    @tvm_ffi.register_object("sgl.UringFileReader")
    class UringFileReaderFFI(tvm_ffi.Object):
        __slots__ = ("__dict__",)

        def __init__(self, queue_depth: int) -> None:
            self.__ffi_init__(queue_depth)

    return UringFileReaderFFI


def _unregister_quietly(native, address: int, nbytes: int) -> None:
    try:
        native.unregister_buffer(address, nbytes)
    except Exception:
        pass


def _extent_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.device.type != "cpu":
        raise ValueError(f"io_uring read {name} must be a CPU tensor")
    if value.ndim != 1:
        raise ValueError(f"io_uring read {name} must be one-dimensional")
    return value.to(torch.int64).contiguous()


class UringFileReader:
    """Read many ``(file, offset, destination, length)`` extents in one batch.

    Destinations are raw host addresses: callers keep the destination tensors
    alive for the duration of :meth:`read`. A registered buffer is unregistered
    when its tensor object is garbage collected, before its memory can be
    reused: the ring holds its own page references, so a stale registration
    over a reused address range would send reads to the old pages.
    """

    def __init__(self, queue_depth: int = 128) -> None:
        self._native = _uring_file_reader_type()(int(queue_depth))
        self._files: dict[tuple[str, bool], int] = {}
        self._finalizers: dict[tuple[int, int], tuple[object, weakref.finalize]] = {}
        self._lock = threading.Lock()

    def open(self, path: str | os.PathLike[str], *, direct: bool) -> int:
        """Return a file id for ``path``, opening it once per ``direct`` flavour."""
        key = (os.path.realpath(os.fspath(path)), bool(direct))
        with self._lock:
            file_id = self._files.get(key)
            if file_id is None:
                file_id = int(self._native.open_file(key[0], int(key[1])))
                self._files[key] = file_id
            return file_id

    def file_size(self, file_id: int) -> int:
        return int(self._native.file_size(int(file_id)))

    @property
    def registered_buffers_supported(self) -> bool:
        return bool(self._native.registered_buffers_supported())

    def register_buffer(self, tensor: torch.Tensor) -> bool:
        """Register a long-lived host tensor as a fixed read buffer."""
        if tensor.device.type != "cpu":
            raise ValueError("io_uring registered buffers must be CPU tensors")
        address = tensor.data_ptr()
        nbytes = tensor.numel() * tensor.element_size()
        if int(self._native.register_buffer(address, nbytes)) == 0:
            return False
        key = (address, nbytes)
        token = object()
        finalizer = weakref.finalize(tensor, self._release_registration, key, token)
        self._finalizers[key] = (token, finalizer)
        return True

    def unregister_buffer(self, tensor: torch.Tensor) -> int:
        address = tensor.data_ptr()
        nbytes = tensor.numel() * tensor.element_size()
        covered = [
            key
            for key in self._finalizers
            if address <= key[0] and key[0] + key[1] <= address + nbytes
        ]
        for key in covered:
            self._finalizers.pop(key)[1].detach()
        return int(self._native.unregister_buffer(address, nbytes))

    def _release_registration(self, key: tuple[int, int], token: object) -> None:
        entry = self._finalizers.get(key)
        if entry is not None and entry[0] is token:
            del self._finalizers[key]
        _unregister_quietly(self._native, *key)

    def read(
        self,
        file_ids: torch.Tensor,
        offsets: torch.Tensor,
        destinations: torch.Tensor,
        lengths: torch.Tensor,
    ) -> int:
        """Read every extent completely and return the bytes read.

        The result is below ``lengths.sum()`` only where an extent runs past
        the end of its file.
        """
        extents = (
            _extent_tensor("file_ids", file_ids),
            _extent_tensor("offsets", offsets),
            _extent_tensor("destinations", destinations),
            _extent_tensor("lengths", lengths),
        )
        if len({extent.numel() for extent in extents}) != 1:
            raise ValueError("io_uring read extents must have matching lengths")
        if extents[0].numel() == 0:
            return 0
        return int(self._native.read(*extents))

    def close(self) -> None:
        with self._lock:
            self._files.clear()
            self._native.close()


def get_shared_uring_file_reader(queue_depth: int = 128) -> UringFileReader:
    """Return the process-wide reader, creating it with ``queue_depth`` once."""
    global _SHARED_READER
    with _SHARED_READER_LOCK:
        if _SHARED_READER is None:
            _SHARED_READER = UringFileReader(queue_depth)
        return _SHARED_READER
