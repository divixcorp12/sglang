"""Random EXL3 expert reads through Exl3RowReader: GB/s and ms/expert per batch size."""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES


def run(expert_dir: str, reads: int, batch: int, seed: int = 0, direct: bool = True) -> dict:
    layout = build_exl3_expert_layout(expert_dir)
    reader = Exl3RowReader(layout, direct=direct)
    slot = -(-reader.buffer_bytes // PAGE_BYTES) * PAGE_BYTES
    storage = torch.empty(batch * slot + PAGE_BYTES, dtype=torch.uint8)
    base = storage.data_ptr() + (-storage.data_ptr()) % PAGE_BYTES
    keys = random.Random(seed).choices(sorted(layout.records), k=reads)
    started = time.perf_counter()
    for start in range(0, reads, batch):
        chunk = keys[start : start + batch]
        reader.read(chunk, [base + i * slot for i in range(len(chunk))])
    seconds = time.perf_counter() - started
    return {
        "batch": batch,
        "reads": reads,
        "seconds": seconds,
        "gb_per_s": reads * layout.row_bytes / seconds / 1e9,
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
