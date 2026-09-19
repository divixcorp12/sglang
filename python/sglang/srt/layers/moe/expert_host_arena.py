"""Registered host memory holding every streamed NVFP4 expert row."""

from __future__ import annotations

import atexit
import json
import logging
import os

import torch

from sglang.srt.layers.moe.expert_format import iter_expert_streamers

from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_register,
    _cuda_host_unregister,
)

logger = logging.getLogger(__name__)

PAGE_BYTES = 4096


def _page_aligned_like(source: torch.Tensor) -> torch.Tensor:
    nbytes = source.numel() * source.element_size()
    storage = torch.empty(nbytes + PAGE_BYTES, dtype=torch.uint8, device="cpu")
    start = (-storage.data_ptr()) % PAGE_BYTES
    return storage[start : start + nbytes].view(source.dtype).view(source.shape)


def _drop_page_cache(path: str) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class ExpertHostArena:
    """Move every host expert tensor into page-aligned, CUDA-registered memory.

    A CUDA graph can only read host rows that stay at one GPU-reachable address
    for the whole run. File mappings fault on the host and the pinned LRU admits
    rows with host code, so neither can serve a captured gather. The arena holds
    all rows of each host expert tensor: page alignment lets the io_uring reader
    fill it with ``O_DIRECT``, and registration lets GPU kernels read it in
    place. Streamers bound to it drop their file reader and file attribution,
    which also keeps the pinned LRU tier out of their gathers.
    """

    def __init__(self) -> None:
        self._buffers: list[torch.Tensor] = []
        self.nbytes = 0
        self.layers = 0

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> ExpertHostArena | None:
        """Bind every streamed expert layer of ``model``; None when there are none."""
        streamers = list(iter_expert_streamers(model))
        if not streamers:
            return None
        arena = cls()
        try:
            for streamer in streamers:
                arena.bind(streamer)
        except BaseException:
            arena.close()
            raise
        atexit.register(arena.close)
        logger.info(
            "Expert host arena startup %s",
            json.dumps(
                {
                    "layers": arena.layers,
                    "registered_bytes": arena.nbytes,
                    "tensors": len(arena._buffers),
                },
                sort_keys=True,
            ),
        )
        return arena

    def bind(self, streamer) -> None:
        """Copy one streamer's host tensors into the arena and rebind its layer."""
        layer = streamer.layer
        sources = {}
        for name in streamer.tensor_names:
            parameter = getattr(layer, name)
            source = (
                parameter.data
                if isinstance(parameter, torch.nn.Parameter)
                else parameter
            )
            if source.device.type == "cpu":
                sources[name] = (_page_aligned_like(source), source)
        if not sources:
            return
        reader = streamer.file_row_reader
        from_files = {
            name: arena_tensor
            for name, (arena_tensor, _) in sources.items()
            if reader is not None and reader.covers(name)
        }
        if from_files:
            reader.read(torch.arange(streamer.num_experts, device="cpu"), from_files)
        for name, (arena_tensor, source) in sources.items():
            if name not in from_files:
                arena_tensor.copy_(source)
        file_paths = set()
        for name, (arena_tensor, _) in sources.items():
            row_bytes = arena_tensor[0].numel() * arena_tensor.element_size()
            _cuda_host_register(arena_tensor, registration_granularity_bytes=row_bytes)
            self._buffers.append(arena_tensor)
            self.nbytes += arena_tensor.numel() * arena_tensor.element_size()
            parameter = getattr(layer, name)
            group = getattr(parameter, "_sglang_file_cache_group", None)
            tag = getattr(parameter, "_sglang_file_cache_tag", None)
            if group is not None and tag is not None:
                file_paths.add(group.paths[tag])
            if isinstance(parameter, torch.nn.Parameter):
                parameter.data = arena_tensor
            else:
                setattr(layer, name, arena_tensor)
        streamer.file_row_reader = None
        layer.__dict__.pop("_nvfp4_file_source_bytes_per_expert", None)
        for path in file_paths:
            _drop_page_cache(path)
        self.layers += 1

    def close(self) -> None:
        """Unregister every arena buffer; the layers must not be used afterwards."""
        buffers, self._buffers = self._buffers, []
        for buffer in buffers:
            _cuda_host_unregister(buffer)
