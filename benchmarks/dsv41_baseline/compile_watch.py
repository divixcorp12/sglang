"""Detect Triton/CUDA JIT compilation that happens after the server reports ready.

Evidence, from `divix01:/data/models/slang/nvfp4-work/cc-dsv41-base/analysis/baseline/
smoke-server.log`: `Triton kernel '_hc_mix_reduce_sinkhorn_kernel' took 34.01 s to
compile after serving started` — twice, once before `/health` first returned 200 and
once again roughly 3 minutes *after* it, mid-decode, for a second request. A passing
health gate does not mean the server is ready to be timed, and a clean warm-up round
does not prove no later request will trigger a fresh compile (the second occurrence
above was for a shape the first warm-up round had already run). So a warm-up
readiness loop and a per-session runtime check are both required — this module backs
both.
"""

from __future__ import annotations

import os
import re

COMPILE_LINE_RE = re.compile(r"took [\d.]+ s to compile after serving started")


def count_compile_events(text: str) -> int:
    return len(COMPILE_LINE_RE.findall(text))


def log_size(path: str) -> int:
    return os.path.getsize(path)


def read_log_range(path: str, *, start_byte: int, end_byte: int | None = None) -> str:
    with open(path, "rb") as f:
        f.seek(start_byte)
        data = f.read() if end_byte is None else f.read(max(0, end_byte - start_byte))
    return data.decode("utf-8", errors="replace")


def compile_events_in_range(path: str, *, start_byte: int, end_byte: int | None = None) -> int:
    return count_compile_events(read_log_range(path, start_byte=start_byte, end_byte=end_byte))
