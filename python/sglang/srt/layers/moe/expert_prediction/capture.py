"""Stage each prefill and decode forward's taps, identity, and residency for the writer."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.capture_frames import (
    CaptureFrame,
    FramePool,
    copy_rows,
)
from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    CAPTURE_FEATURES,
    CaptureKind,
    ForwardRecord,
    rows_per_request,
)
from sglang.srt.layers.moe.expert_prediction.capture_writer import PendingForward, ShardWriter
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_residency_clock import ForwardKind, classify_forward

logger = logging.getLogger(__name__)

_CAPTURE_KINDS = {
    ForwardKind.PREFILL: CaptureKind.PREFILL,
    ForwardKind.DECODE: CaptureKind.DECODE,
}


class CaptureSettings(msgspec.Struct, frozen=True):
    directory: Path
    capacity: int
    frames: int
    shard_rows: int
    max_bytes: int


class RouteCapture:
    """Record every prefill and decode row; a forward it cannot record stops capture."""

    def __init__(
        self,
        *,
        specs: Sequence[MoeLayerSpec],
        store: FeatureStore,
        pool: FramePool,
        writer: ShardWriter,
        hot_caches: Mapping[int, Any],
        device: torch.device,
    ) -> None:
        self._specs = tuple(specs)
        self._store = store
        self._pool = pool
        self._writer = writer
        self._hot_caches = dict(hot_caches)
        self._use_events = device.type == "cuda"
        self._frame: CaptureFrame | None = None
        self.forwards = 0
        store.spill = self._spill

    @classmethod
    def build(
        cls,
        *,
        specs: Sequence[MoeLayerSpec],
        store: FeatureStore,
        device: torch.device,
        hidden_dtype: torch.dtype,
        settings: CaptureSettings,
        hot_caches: Mapping[int, Any],
    ) -> "RouteCapture":
        if any(spec.num_experts > torch.iinfo(torch.int16).max for spec in specs):
            raise ValueError("expert capture stores expert ids as int16")
        pool = FramePool(
            [
                CaptureFrame(
                    specs=specs,
                    capacity=settings.capacity,
                    hidden_dtype=hidden_dtype,
                    pin_memory=device.type == "cuda",
                )
                for _ in range(settings.frames)
            ]
        )
        writer = ShardWriter(
            directory=settings.directory,
            specs=specs,
            hidden_dtype=hidden_dtype,
            pool=pool,
            shard_rows=settings.shard_rows,
            max_bytes=settings.max_bytes,
        )
        capture = cls(
            specs=specs,
            store=store,
            pool=pool,
            writer=writer,
            hot_caches=hot_caches,
            device=device,
        )
        atexit.register(capture.close)
        logger.info(
            "MoE expert capture: directory=%s capacity_rows=%d frames=%d "
            "pinned_bytes=%d max_bytes=%d",
            settings.directory,
            settings.capacity,
            settings.frames,
            pool.nbytes,
            settings.max_bytes,
        )
        return capture

    @property
    def stopped(self) -> bool:
        return self._writer.stopped.is_set()

    def close(self) -> None:
        self._writer.close()

    def on_forward_end(self, forward_batch: Any, *, taps_supported: bool) -> None:
        frame, self._frame = self._frame, None
        kind = _CAPTURE_KINDS.get(classify_forward(forward_batch)[0])
        if kind is None or self.stopped:
            self._release(frame)
            return
        rows = forward_batch.input_ids.shape[0]
        per_request = rows_per_request(
            is_extend=kind is CaptureKind.PREFILL,
            batch_size=forward_batch.batch_size,
            extend_seq_lens=forward_batch.extend_seq_lens_cpu,
        )
        reason = _unrecordable_reason(
            rows=rows,
            per_request=per_request,
            positions=forward_batch.positions,
            taps_supported=taps_supported,
            spilled=frame is not None,
            max_rows=self._store.max_rows,
            capacity=self._pool.capacity,
        )
        if reason is not None:
            self._release(frame)
            self._writer.stop(reason)
            return
        if frame is None:
            frame = self._stage_store_rows(rows)
        self._stage_identity(frame=frame, forward_batch=forward_batch, rows=rows)
        if self._use_events:
            frame.event = torch.cuda.Event()
            frame.event.record()
        self._writer.submit(
            PendingForward(
                record=ForwardRecord(
                    forward_index=self.forwards,
                    kind=kind,
                    rids=tuple(forward_batch.rids),
                    rows_per_request=per_request,
                ),
                frame=frame,
            )
        )
        self.forwards += 1

    def _spill(self, layer_id: int, feature: RouteFeature, rows: torch.Tensor) -> None:
        if self.stopped:
            return
        if rows.shape[0] > self._pool.capacity:
            self._writer.stop(
                f"forward with {rows.shape[0]} rows exceeds capture capacity "
                f"{self._pool.capacity}"
            )
            return
        if self._frame is None:
            self._frame = self._pool.acquire()
        self._frame.stage(layer_id=layer_id, feature=feature, rows=rows)

    def _stage_store_rows(self, rows: int) -> CaptureFrame:
        frame = self._pool.acquire()
        for spec in self._specs:
            for feature in CAPTURE_FEATURES:
                frame.stage(
                    layer_id=spec.layer_id,
                    feature=feature,
                    rows=self._store.view(spec.layer_id, feature, rows),
                )
        return frame

    def _stage_identity(self, *, frame: CaptureFrame, forward_batch: Any, rows: int) -> None:
        copy_rows(frame.positions, forward_batch.positions[:rows])
        copy_rows(frame.token_ids, forward_batch.input_ids[:rows])
        width = frame.residency.shape[1]
        for index, spec in enumerate(self._specs):
            frame.residency[index].fill_(-1)
            cache = self._hot_caches.get(spec.layer_id)
            if cache is not None:
                copy_rows(frame.residency[index], cache.expert_to_slot[:width])

    def _release(self, frame: CaptureFrame | None) -> None:
        if frame is not None:
            self._pool.release(frame)


def _unrecordable_reason(
    *,
    rows: int,
    per_request: tuple[int, ...],
    positions: torch.Tensor,
    taps_supported: bool,
    spilled: bool,
    max_rows: int,
    capacity: int,
) -> str | None:
    if not taps_supported:
        return "some MoE layers lack standard top-k outputs"
    if rows == 0 or sum(per_request) != rows:
        return f"forward rows {rows} do not match per-request rows {per_request}"
    if rows > capacity:
        return f"forward with {rows} rows exceeds capture capacity {capacity}"
    if rows > max_rows and not spilled:
        return f"forward with {rows} rows reached no tap"
    if positions.dim() != 1:
        return "expert capture needs 1-D positions"
    return None
