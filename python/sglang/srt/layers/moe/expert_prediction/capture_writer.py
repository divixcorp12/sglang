"""Background thread that turns captured frames into safetensors shards on disk."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import msgspec
import torch
from safetensors.torch import save_file

from sglang.srt.layers.moe.expert_prediction.capture_frames import CaptureFrame, FramePool
from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    CAPTURE_FEATURES,
    FORWARD_INDEX,
    FORWARD_KIND,
    FORWARD_RESIDENCY,
    FORWARD_ROWS,
    ROW_FORWARD,
    ROW_POSITION,
    ROW_PREFIX_HASH,
    ROW_REQUEST,
    ROW_TOKEN,
    SCHEMA_VERSION,
    CaptureKind,
    ForwardRecord,
    disk_dtype,
    feature_key,
)
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.prefix_hash import PrefixHasher, SeenPrefixes

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"
HEADER_NAME = "capture.json"
STOPPED_NAME = "capture-stopped.json"
# Arbitrary; bounds what a killed server loses once traffic pauses.
IDLE_FLUSH_S = 5.0
# Arbitrary; requests whose hash chains are remembered at once.
MAX_HASHED_REQUESTS = 4096
# Arbitrary; how often the writer checks whether a frame's copies finished.
_EVENT_POLL_S = 0.001
# Row keys omitted from a shard that keeps no rows (some safetensors builds
# reject zero-sized tensors); check_capture treats missing row keys as zero rows.
_ROW_KEYS = (ROW_FORWARD, ROW_REQUEST, ROW_POSITION, ROW_TOKEN, ROW_PREFIX_HASH)


class PendingForward(msgspec.Struct, frozen=True):
    record: ForwardRecord
    frame: CaptureFrame


class ShardWriter:
    """Dedupe re-prefilled prefixes and append rows to shards; stops, never thins."""

    def __init__(
        self,
        *,
        directory: Path,
        specs: Sequence[MoeLayerSpec],
        hidden_dtype: torch.dtype,
        pool: FramePool,
        shard_rows: int,
        max_bytes: int,
        idle_flush_s: float = IDLE_FLUSH_S,
    ) -> None:
        if shard_rows < 1:
            raise ValueError("SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_SHARD_ROWS must be positive")
        if max_bytes < 1:
            raise ValueError("SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_MAX_GB must be positive")
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"capture directory {directory} is not empty")
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._specs = tuple(specs)
        self._hidden_dtype = hidden_dtype
        self._pool = pool
        self._shard_rows = shard_rows
        self._max_bytes = max_bytes
        self._idle_flush_s = idle_flush_s
        self._hasher = PrefixHasher(max_requests=MAX_HASHED_REQUESTS)
        self._seen = SeenPrefixes()
        self._shard_index = 0
        self.written_bytes = 0
        self.stopped = threading.Event()
        self._stop_lock = threading.Lock()
        self._stop_reason: str | None = None
        self._stop_recorded = False
        self._reset_shard()
        self._write_header()
        # Frame ownership independently bounds submissions, and this queue
        # makes that bound explicit if a caller bypasses the pool.
        self._queue: queue.Queue[PendingForward | None] = queue.Queue(
            maxsize=pool.frames
        )
        self._thread = threading.Thread(
            target=self._run, name="moe-expert-capture-writer", daemon=True
        )
        self._thread.start()

    def submit(self, pending: PendingForward) -> bool:
        """Accept frame ownership now, or report backpressure without waiting."""
        if self.stopped.is_set():
            return False
        try:
            self._queue.put_nowait(pending)
        except queue.Full:
            return False
        return True

    def stop(self, reason: str) -> None:
        with self._stop_lock:
            if self.stopped.is_set():
                return
            self._stop_reason = reason
            self.stopped.set()

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self._idle_flush_s)
            except queue.Empty:
                self._guarded(self._flush)
                self._guarded(self._record_stop)
                continue
            if item is None:
                self._guarded(self._flush)
                self._guarded(self._record_stop)
                return
            try:
                if not self.stopped.is_set():
                    self._guarded(lambda: self._ingest(item))
            finally:
                self._pool.release(item.frame)
                self._guarded(self._record_stop)

    def _record_stop(self) -> None:
        """Persist a serving-thread stop decision on the background writer."""
        with self._stop_lock:
            if not self.stopped.is_set() or self._stop_recorded:
                return
            reason = self._stop_reason
            self._stop_recorded = True
        try:
            (self._directory / STOPPED_NAME).write_text(
                json.dumps(
                    {
                        "reason": reason,
                        "written_bytes": self.written_bytes,
                        "timestamp_ns": time.time_ns(),
                    }
                )
            )
            logger.warning(
                "MoE expert capture stopped: %s (written_bytes=%d)", reason, self.written_bytes
            )
        except OSError:
            logger.exception("MoE expert capture could not record its stop reason")

    def _guarded(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as error:
            # The writer must keep releasing frames or the serving thread deadlocks.
            logger.exception("MoE expert capture writer failed")
            self.stop(f"writer error: {error!r}")

    def _reset_shard(self) -> None:
        self._columns: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._request_ids: dict[str, int] = {}
        self._forward_count = 0
        self._row_count = 0

    def _write_header(self) -> None:
        header = {
            "schema_version": SCHEMA_VERSION,
            "hidden_dtype": str(self._hidden_dtype).removeprefix("torch."),
            "features": [feature.value for feature in CAPTURE_FEATURES],
            "layers": [msgspec.to_builtins(spec) for spec in self._specs],
            "created_ns": time.time_ns(),
        }
        (self._directory / HEADER_NAME).write_text(json.dumps(header, indent=2))

    def _ingest(self, pending: PendingForward) -> None:
        record, frame = pending.record, pending.frame
        # Polling instead of Event.synchronize keeps sync debug mode quiet in this thread.
        while frame.event is not None and not frame.event.query():
            time.sleep(_EVENT_POLL_S)
        rows = sum(record.rows_per_request)
        positions = frame.positions[:rows].tolist()
        token_ids = frame.token_ids[:rows].tolist()
        hashes: list[int] = []
        request_index: list[int] = []
        start = 0
        for rid, count in zip(record.rids, record.rows_per_request):
            end = start + count
            hashes += self._hasher.hash_rows(
                rid=rid, positions=positions[start:end], token_ids=token_ids[start:end]
            )
            request_index += [self._request_ids.setdefault(rid, len(self._request_ids))] * count
            start = end
        keep = self._seen.keep_mask(
            hashes=hashes, is_prefill=record.kind is CaptureKind.PREFILL
        )
        kept = torch.tensor([row for row, flag in enumerate(keep) if flag], dtype=torch.long)
        columns = self._columns
        if kept.numel():
            columns[ROW_FORWARD].append(
                torch.full((kept.numel(),), record.forward_index, dtype=torch.int64)
            )
            columns[ROW_REQUEST].append(torch.tensor(request_index, dtype=torch.int32)[kept])
            columns[ROW_POSITION].append(frame.positions[:rows][kept])
            columns[ROW_TOKEN].append(frame.token_ids[:rows][kept])
            columns[ROW_PREFIX_HASH].append(torch.tensor(hashes, dtype=torch.int64)[kept])
            for (layer_id, feature), tensor in frame.features.items():
                columns[feature_key(layer_id, feature)].append(
                    tensor[:rows][kept].to(disk_dtype(feature, self._hidden_dtype))
                )
        columns[FORWARD_INDEX].append(torch.tensor([record.forward_index], dtype=torch.int64))
        columns[FORWARD_KIND].append(torch.tensor([int(record.kind)], dtype=torch.uint8))
        columns[FORWARD_ROWS].append(torch.tensor([rows], dtype=torch.int64))
        columns[FORWARD_RESIDENCY].append(frame.residency.to(torch.int16).unsqueeze(0))
        self._forward_count += 1
        self._row_count += kept.numel()
        if self._row_count >= self._shard_rows:
            self._flush()

    def _flush(self) -> None:
        if self._forward_count == 0:
            return
        if self.stopped.is_set():
            self._reset_shard()
            return
        tensors = {key: torch.cat(parts).contiguous() for key, parts in self._columns.items()}
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if self.written_bytes + nbytes > self._max_bytes:
            self._reset_shard()
            self.stop(f"byte cap {self._max_bytes} reached")
            return
        name = f"shard-{self._shard_index:06d}.safetensors"
        path = self._directory / name
        partial = self._directory / f"{name}.partial"
        request_ids = sorted(self._request_ids, key=self._request_ids.__getitem__)
        save_file(
            tensors,
            str(partial),
            metadata={
                "schema_version": str(SCHEMA_VERSION),
                "request_ids": json.dumps(request_ids),
            },
        )
        os.replace(partial, path)
        size = path.stat().st_size
        self.written_bytes += size
        forward_index = tensors[FORWARD_INDEX]
        with open(self._directory / MANIFEST_NAME, "a") as manifest:
            manifest.write(
                json.dumps(
                    {
                        "shard": name,
                        "rows": self._row_count,
                        "forwards": self._forward_count,
                        "first_forward": int(forward_index[0]),
                        "last_forward": int(forward_index[-1]),
                        "bytes": size,
                    }
                )
                + "\n"
            )
        self._shard_index += 1
        self._reset_shard()
