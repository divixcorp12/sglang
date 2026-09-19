"""Split vs repack: what a RAM miss of one EXL3 expert row costs on the host.

Three ways to land a sample of one layer's expert rows in host memory, each at
several batch sizes (rows per reader submission):

- ``superset_raw``: one page-aligned superset read per row (Exl3RowReader), no split;
- ``superset_split``: the same reads, split into the six per-name rows
  (Exl3ShardRowSource, what the framework runs on a RAM miss);
- ``repacked_per_name``: per-name reads from a repacked copy of the same rows,
  as ExpertFileRowReader would do over repacked files (O_DIRECT for page-multiple
  rows, buffered for the three small names).

The repacked copy is written once, next to nothing else, into ``--work-dir``,
which must sit on the drive being measured; the script deletes it at the end.
Every reader read is bounded up front by ``--max-read-gb``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
from sglang.srt.model_loader.file_row_reader import (
    PAGE_BYTES,
    AlignedRowSource,
    read_plans,
    shared_uring_file_reader,
)


def _page_aligned(nbytes: int) -> torch.Tensor:
    storage = torch.empty(nbytes + PAGE_BYTES, dtype=torch.uint8)
    start = (-storage.data_ptr()) % PAGE_BYTES
    return storage[start : start + nbytes]


def _slabs(specs, rows: int) -> dict[str, torch.Tensor]:
    """Page-aligned per-name ``[rows, *row_shape]`` host slabs."""
    return {
        spec.name: _page_aligned(rows * spec.row_bytes).view(spec.dtype).view((rows,) + spec.row_shape)
        for spec in specs
    }


def _drop_cache(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def planned_read_bytes(
    buffer_bytes: int, streamed_bytes: int, rows: int, batches: list[int]
) -> int:
    """Bytes the run reads from the drive under test.

    The shards give the repack pass plus two superset modes per batch size; the
    repacked copy, which sits on the same drive, gives one per-name pass per
    batch size.
    """
    return rows * (buffer_bytes * (1 + 2 * len(batches)) + streamed_bytes * len(batches))


def run(
    expert_dir: str,
    layer: int,
    rows: int,
    batches: list[int],
    work_dir: str,
    *,
    direct: bool = True,
    seed: int = 0,
    max_read_gb: float = 4.5,
) -> list[dict]:
    layout = build_exl3_expert_layout(expert_dir)
    fmt = Exl3ExpertFormat(layout, layer, direct=direct)
    specs = fmt.tensor_specs(None)
    reader = Exl3RowReader(layout, direct=direct)
    streamed_bytes = sum(spec.row_bytes for spec in specs)
    planned = planned_read_bytes(reader.buffer_bytes, streamed_bytes, rows, batches)
    if planned > max_read_gb * 1e9:
        raise SystemExit(
            f"this run would read {planned / 1e9:.2f} GB, over --max-read-gb {max_read_gb}"
        )
    if rows > layout.num_experts:
        raise SystemExit(f"layer {layer} has only {layout.num_experts} experts")
    experts = sorted(random.Random(seed).sample(range(layout.num_experts), rows))

    created = not os.path.exists(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    paths = {}
    results = []
    try:
        # The repacked copy: this layer's sampled rows, one file per streamed name.
        # Written inside the try, so a failed write still removes what it left.
        truth = _slabs(specs, rows)
        Exl3ShardRowSource(reader, layer, fmt.segment_map()).read(torch.tensor(experts), truth)
        for spec in specs:
            path = os.path.join(work_dir, f"layer{layer}.{spec.name}.bin")
            paths[spec.name] = path
            with open(path, "wb") as f:
                f.write(truth[spec.name].view(torch.uint8).reshape(-1).numpy().tobytes())
                f.flush()
                os.fsync(f.fileno())
            _drop_cache(path)
        uring = shared_uring_file_reader()
        repacked = {
            spec.name: AlignedRowSource(uring, paths[spec.name], spec.row_bytes, rows, direct=direct)
            for spec in specs
        }
        for batch in batches:
            # superset_raw
            bounce = _page_aligned(batch * reader.buffer_bytes).view(batch, reader.buffer_bytes)
            began = time.perf_counter()
            for start in range(0, rows, batch):
                chunk = experts[start : start + batch]
                reader.read(
                    [(layer, expert) for expert in chunk],
                    [bounce[i].data_ptr() for i in range(len(chunk))],
                )
            seconds = time.perf_counter() - began
            results.append(_row("superset_raw", batch, rows, seconds, layout.row_bytes))

            # superset_split
            slabs = _slabs(specs, rows)
            source = Exl3ShardRowSource(reader, layer, fmt.segment_map(), bounce_rows=batch)
            began = time.perf_counter()
            stats = source.read(torch.tensor(experts), slabs)
            seconds = time.perf_counter() - began
            row = _row("superset_split", batch, rows, seconds, streamed_bytes)
            row["read_ms_per_row"] = stats.read_ns / 1e6 / rows
            row["split_ms_per_row"] = stats.split_ns / 1e6 / rows
            row["split_share"] = stats.split_ns / max(stats.read_ns + stats.split_ns, 1)
            row["verified"] = all(torch.equal(slabs[n].view(torch.uint8), truth[n].view(torch.uint8)) for n in slabs)
            results.append(row)

            # repacked_per_name
            for path in paths.values():
                _drop_cache(path)
            slabs = _slabs(specs, rows)
            began = time.perf_counter()
            for start in range(0, rows, batch):
                ids = torch.arange(start, min(start + batch, rows), dtype=torch.int64)
                read_plans(uring, [repacked[n].plan(ids, slabs[n], ids) for n in slabs])
            seconds = time.perf_counter() - began
            row = _row("repacked_per_name", batch, rows, seconds, streamed_bytes)
            row["verified"] = all(torch.equal(slabs[n].view(torch.uint8), truth[n].view(torch.uint8)) for n in slabs)
            results.append(row)
    finally:
        for path in paths.values():
            if os.path.exists(path):
                os.unlink(path)
        # Remove the scratch dir only if this run made it and nothing else is in it.
        if created and os.path.isdir(work_dir) and not os.listdir(work_dir):
            os.rmdir(work_dir)
    return results


def _row(mode: str, batch: int, rows: int, seconds: float, row_bytes: int) -> dict:
    return {
        "mode": mode,
        "batch": batch,
        "rows": rows,
        "seconds": seconds,
        "ms_per_row": seconds * 1e3 / rows,
        "gb_per_s": rows * row_bytes / seconds / 1e9,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert-dir", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, default=12)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--work-dir", required=True, help="scratch dir on the drive under test")
    p.add_argument("--max-read-gb", type=float, default=4.5)
    p.add_argument("--buffered", action="store_true")
    args = p.parse_args()
    for row in run(
        args.expert_dir,
        args.layer,
        args.rows,
        args.batch,
        args.work_dir,
        direct=not args.buffered,
        max_read_gb=args.max_read_gb,
    ):
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
