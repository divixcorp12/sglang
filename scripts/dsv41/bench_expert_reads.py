"""Random EXL3 expert reads through Exl3RowReader: GB/s and ms/expert per batch size.

Nothing but the reads is timed: the shard files are opened, the destination
buffer is faulted in and one row is read before the clock starts. There is no
read budget here (``--reads`` rows of ~13 MB per batch size), so size the run
yourself. ``--buffered`` reads through the page cache; it is for CPU tests,
never for drive numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES


def superset_file_bytes(layout, keys) -> int:
    """Bytes the drive moves for ``keys``: page-aligned supersets, cut at end of file."""
    sizes: dict[str, int] = {}
    total = 0
    for key in keys:
        record = layout.records[key]
        offset, length, _ = record.aligned_read(PAGE_BYTES)
        if record.path not in sizes:
            sizes[record.path] = os.path.getsize(record.path)
        total += min(length, sizes[record.path] - offset)
    return total


def drop_superset_ranges(layout, keys) -> None:
    """Evict the page-cache pages of ``keys``' supersets (matters only for buffered reads)."""
    fds: dict[str, int] = {}
    try:
        for key in keys:
            record = layout.records[key]
            offset, length, _ = record.aligned_read(PAGE_BYTES)
            if record.path not in fds:
                fds[record.path] = os.open(record.path, os.O_RDONLY)
            os.posix_fadvise(fds[record.path], offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        for fd in fds.values():
            os.close(fd)


def run(expert_dir: str, reads: int, batch: int, seed: int = 0, direct: bool = True) -> dict:
    if reads < 1:
        raise SystemExit("reads must be at least 1")
    if batch < 1:
        raise SystemExit("batch must be at least 1")
    layout = build_exl3_expert_layout(expert_dir)
    reader = Exl3RowReader(layout, direct=direct)
    slot = -(-reader.buffer_bytes // PAGE_BYTES) * PAGE_BYTES
    storage = torch.zeros(batch * slot + PAGE_BYTES, dtype=torch.uint8)  # faulted in, untimed
    base = storage.data_ptr() + (-storage.data_ptr()) % PAGE_BYTES
    keys = random.Random(seed).choices(sorted(layout.records), k=reads)
    # Untimed: open (and size) every shard the keys touch, then one warm read.
    for path in {layout.records[key].path for key in keys}:
        reader._file(path)
    reader.read(keys[:1], [base])
    if not direct:
        drop_superset_ranges(layout, keys)
    started = time.perf_counter()
    for start in range(0, reads, batch):
        chunk = keys[start : start + batch]
        reader.read(chunk, [base + i * slot for i in range(len(chunk))])
    seconds = time.perf_counter() - started
    return {
        "batch": batch,
        "reads": reads,
        "direct": bool(direct),
        "queue_depth": int(envs.SGLANG_URING_FILE_READER_QUEUE_DEPTH.get()),
        "layer": "all",
        "seconds": seconds,
        # Payload basis: the 12 tensors of one expert row. file_gb_per_s counts what the
        # drive moved (page-aligned supersets), so the two differ by the alignment waste.
        "bytes_basis": "row_payload",
        "gb_per_s": reads * layout.row_bytes / seconds / 1e9,
        "file_gb_per_s": superset_file_bytes(layout, keys) / seconds / 1e9,
        "ms_per_expert": seconds * 1e3 / reads,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expert-dir", required=True)
    p.add_argument("--reads", type=int, default=512)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--buffered", action="store_true")
    args = p.parse_args()
    for batch in args.batch:
        print(json.dumps(run(args.expert_dir, args.reads, batch, direct=not args.buffered)), flush=True)


if __name__ == "__main__":
    main()
