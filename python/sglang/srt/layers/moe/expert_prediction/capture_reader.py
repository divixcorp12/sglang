"""Load expert capture shards and check that no forward or row went missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import msgspec
import torch
from safetensors import safe_open

from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    FORWARD_INDEX,
    FORWARD_KIND,
    FORWARD_ROWS,
    ROW_FORWARD,
    ROW_POSITION,
    ROW_REQUEST,
    CaptureKind,
)
from sglang.srt.layers.moe.expert_prediction.capture_writer import (
    MANIFEST_NAME,
    STOPPED_NAME,
)

_MAX_REPORTED_VIOLATIONS = 50


class CaptureShard(msgspec.Struct, frozen=True):
    name: str
    request_ids: tuple[str, ...]
    tensors: dict[str, torch.Tensor]


class CaptureReport(msgspec.Struct, frozen=True):
    shards: int
    rows: int
    prefill_rows: int
    decode_rows: int
    forwards: int
    ran_rows: int
    requests: int
    bytes: int
    bytes_per_row: float
    stopped_reason: str | None
    violation_count: int
    violations: list[str]


def read_manifest(directory: Path) -> list[dict]:
    path = directory / MANIFEST_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_shard(directory: Path, name: str) -> CaptureShard:
    with safe_open(str(directory / name), framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return CaptureShard(
        name=name, request_ids=tuple(json.loads(metadata["request_ids"])), tensors=tensors
    )


def check_capture(directory: Path) -> CaptureReport:
    entries = read_manifest(directory)
    violations: list[str] = []
    last_position: dict[str, int] = {}
    prefill_rows = decode_rows = ran_rows = 0
    next_forward: int | None = None
    for entry in entries:
        shard = load_shard(directory, entry["shard"])
        tensors = shard.tensors
        forward_index = tensors[FORWARD_INDEX]
        if next_forward is not None and int(forward_index[0]) != next_forward:
            violations.append(f"{shard.name}: forwards jump from {next_forward - 1}")
        if forward_index.numel() > 1 and bool((forward_index.diff() != 1).any()):
            violations.append(f"{shard.name}: forward indices are not consecutive")
        next_forward = int(forward_index[-1]) + 1
        ran_rows += int(tensors[FORWARD_ROWS].sum())
        kinds = dict(zip(forward_index.tolist(), tensors[FORWARD_KIND].tolist()))
        # A shard whose every row was deduplicated omits the row.* keys entirely.
        row_forward = tensors.get(ROW_FORWARD, torch.empty(0, dtype=torch.int64))
        row_request = tensors.get(ROW_REQUEST, torch.empty(0, dtype=torch.int32))
        row_position = tensors.get(ROW_POSITION, torch.empty(0, dtype=torch.int64))
        for forward, request, position in zip(
            row_forward.tolist(), row_request.tolist(), row_position.tolist()
        ):
            rid = shard.request_ids[request]
            if position <= last_position.get(rid, -1):
                violations.append(f"{shard.name}: request {rid} repeats position {position}")
            last_position[rid] = position
            if kinds[forward] == CaptureKind.PREFILL:
                prefill_rows += 1
            else:
                decode_rows += 1
    rows = sum(entry["rows"] for entry in entries)
    total_bytes = sum(entry["bytes"] for entry in entries)
    stopped = directory / STOPPED_NAME
    return CaptureReport(
        shards=len(entries),
        rows=rows,
        prefill_rows=prefill_rows,
        decode_rows=decode_rows,
        forwards=sum(entry["forwards"] for entry in entries),
        ran_rows=ran_rows,
        requests=len(last_position),
        bytes=total_bytes,
        bytes_per_row=total_bytes / rows if rows else 0.0,
        stopped_reason=json.loads(stopped.read_text())["reason"] if stopped.exists() else None,
        violation_count=len(violations),
        violations=violations[:_MAX_REPORTED_VIOLATIONS],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(msgspec.json.encode(check_capture(args.directory)).decode())


if __name__ == "__main__":
    main()
