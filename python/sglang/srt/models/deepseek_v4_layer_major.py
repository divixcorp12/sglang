"""DeepSeek-V4.1 adapter for the layer-major prefill strategy (sglang.srt.layer_major): chunk plan, window-KV ring,
per-chunk metadata, one layer on one chunk, and the late-layer tail. Knows nothing about the expert quant format."""

from __future__ import annotations

import msgspec

DSV4_WINDOW = 128


class ChunkSpan(msgspec.Struct, frozen=True):
    index: int
    start: int
    end: int


def chunk_spans(*, prefix_len: int, seq_len: int, chunk: int) -> list[ChunkSpan]:
    return [
        ChunkSpan(index=i, start=start, end=min(seq_len, start + chunk))
        for i, start in enumerate(range(prefix_len, seq_len, chunk))
    ]


def keep_window_start(*, seq_len: int, window: int, page: int) -> int:
    return max(0, seq_len - window) // page * page


def engram_history(ids: list[int], start: int, n: int) -> list[int]:
    window = list(ids[max(0, start - n) : start])
    return [0] * (n - len(window)) + window
