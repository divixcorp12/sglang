"""io_uring reads of NVFP4 expert rows from their verified expert files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_loader.file_row_reader import (
    AlignedRowSource,
    read_plans,
    shared_uring_file_reader,
    validate_file_reader_mode,
)

if TYPE_CHECKING:
    from sglang.kernels.ops.io.uring_file_reader import UringFileReader

logger = logging.getLogger(__name__)
_LOGGED_MODES: set[str] = set()


class ExpertFileRowReader:
    """Read host expert rows from the files behind their mapped tensors.

    Only tensors bound by the NVFP4 expert file cache are covered. Their
    parameters carry the verified cache group, and row ``e`` of a tensor lives
    at byte ``e * row_bytes`` of its member file because the runtime view is
    contiguous from storage offset zero, which :meth:`from_layer` checks.
    """

    def __init__(
        self,
        reader: UringFileReader,
        sources: Mapping[str, AlignedRowSource],
        mode: str,
    ) -> None:
        self._reader = reader
        self._sources = dict(sources)
        self.mode = mode
        self.registered_bytes = 0

    @classmethod
    def from_layer(
        cls,
        layer: torch.nn.Module,
        tensor_names: Iterable[str],
        mode: Optional[str] = None,
    ) -> Optional[ExpertFileRowReader]:
        """Build a reader for the CPU expert tensors of ``layer``, or None for ``mmap``."""
        mode = validate_file_reader_mode(
            envs.SGLANG_MOE_EXPERT_FILE_READER.get() if mode is None else mode
        )
        if mode == "mmap":
            return None
        reader: Optional[UringFileReader] = None
        sources: dict[str, AlignedRowSource] = {}
        for name in tensor_names:
            parameter = getattr(layer, name)
            data = (
                parameter.data
                if isinstance(parameter, torch.nn.Parameter)
                else parameter
            )
            if data.device.type != "cpu":
                continue
            group = getattr(parameter, "_sglang_file_cache_group", None)
            tag = getattr(parameter, "_sglang_file_cache_tag", None)
            if group is None or tag is None:
                raise ValueError(
                    f"SGLANG_MOE_EXPERT_FILE_READER={mode} needs file-backed expert "
                    f"tensors, but {name!r} has no expert file; set "
                    "SGLANG_MOE_EXPERT_FILE_DIR or use the mmap reader"
                )
            if (
                data.shape[0] == 0
                or not data.is_contiguous()
                or data.storage_offset() != 0
                or data.data_ptr() != group.tensors[tag].data_ptr()
            ):
                raise ValueError(
                    f"expert tensor {name!r} does not occupy its expert file from "
                    "offset zero"
                )
            if reader is None:
                reader = shared_uring_file_reader()
            sources[name] = AlignedRowSource(
                reader,
                group.paths[tag],
                data[0].numel() * data.element_size(),
                data.shape[0],
                direct=mode == "uring_direct",
            )
        if reader is None:
            return None
        if mode not in _LOGGED_MODES:
            _LOGGED_MODES.add(mode)
            logger.info(
                "MoE expert file reads use io_uring: mode=%s tensors=%s "
                "registered_buffers=%s",
                mode,
                ",".join(sources),
                reader.registered_buffers_supported,
            )
        return cls(reader, sources, mode)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._sources)

    def covers(self, name: str) -> bool:
        return name in self._sources

    def register_destinations(self, tensors: Iterable[torch.Tensor]) -> int:
        """Register long-lived pinned row tensors; returns the bytes registered."""
        registered = 0
        for tensor in tensors:
            if tensor.numel() and self._reader.register_buffer(tensor):
                registered += tensor.numel() * tensor.element_size()
        self.registered_bytes += registered
        return registered

    def read(
        self,
        rows: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        destination_rows: Optional[torch.Tensor] = None,
    ) -> None:
        """Read expert ``rows`` of every named tensor in one io_uring batch."""
        missing = [name for name in destinations if name not in self._sources]
        if missing:
            raise ValueError(f"expert file reader does not cover {missing}")
        read_plans(
            self._reader,
            [
                self._sources[name].plan(rows, destination, destination_rows)
                for name, destination in destinations.items()
            ],
        )
