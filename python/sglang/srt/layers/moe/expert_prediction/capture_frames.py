"""Pinned host frames that receive one forward's captured rows without blocking the GPU."""

from __future__ import annotations

import queue
from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.capture_schema import CAPTURE_FEATURES
from sglang.srt.layers.moe.expert_prediction.contracts import (
    MoeLayerSpec,
    RouteFeature,
    feature_dtype,
    feature_width,
)


def copy_rows(destination: torch.Tensor, source: torch.Tensor) -> None:
    # Cast and compact on the source device so the host copy stays non-blocking.
    destination[: source.shape[0]].copy_(
        source.to(destination.dtype).contiguous(), non_blocking=True
    )


class CaptureFrame:
    def __init__(
        self,
        *,
        specs: Sequence[MoeLayerSpec],
        capacity: int,
        hidden_dtype: torch.dtype,
        pin_memory: bool,
    ) -> None:
        self.capacity = capacity
        self.features = {
            (spec.layer_id, feature): torch.empty(
                (capacity, feature_width(feature, spec)),
                dtype=feature_dtype(feature, hidden_dtype),
                pin_memory=pin_memory,
            )
            for spec in specs
            for feature in CAPTURE_FEATURES
        }
        self.positions = torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory)
        self.token_ids = torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory)
        self.residency = torch.full(
            (len(specs), max(spec.num_experts for spec in specs)),
            -1,
            dtype=torch.int64,
            pin_memory=pin_memory,
        )
        self.event: torch.cuda.Event | None = None

    @property
    def nbytes(self) -> int:
        tensors = [*self.features.values(), self.positions, self.token_ids, self.residency]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def stage(self, *, layer_id: int, feature: RouteFeature, rows: torch.Tensor) -> None:
        copy_rows(self.features[(layer_id, feature)], rows)


class FramePool:
    """Fixed frames with an explicit nonblocking acquisition path for serving."""

    def __init__(self, frames: Sequence[CaptureFrame]) -> None:
        if not frames:
            raise ValueError("SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES must be positive")
        self._frames = tuple(frames)
        self._free: queue.Queue[CaptureFrame] = queue.Queue(maxsize=len(frames))
        for frame in frames:
            self._free.put(frame)

    @property
    def capacity(self) -> int:
        return self._frames[0].capacity

    @property
    def frames(self) -> int:
        return len(self._frames)

    @property
    def nbytes(self) -> int:
        return sum(frame.nbytes for frame in self._frames)

    def acquire(self, timeout: float | None = None) -> CaptureFrame:
        return self._free.get(timeout=timeout)

    def try_acquire(self) -> CaptureFrame | None:
        """Return an owned frame now, or ``None`` without delaying inference."""
        try:
            return self._free.get_nowait()
        except queue.Empty:
            return None

    def release(self, frame: CaptureFrame) -> None:
        frame.event = None
        self._free.put(frame)
