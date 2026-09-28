#!/usr/bin/env python3
"""The WC NVMe-write check (nvme_wc_check.py JSONL): stale 8-byte words per slab x method x size. Pure Python.

    python3 nvme_report.py <divix01-nvme-wc.jsonl>   # markdown; exit 1 on any wrong word

A word is stale when it still holds the trial's pattern A (written by the CPU before the O_DIRECT read), and wrong
when it differs from the file at all. A zero is evidence, not proof: staleness is timing-dependent.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

METHODS = ("sm_cv16", "weak", "ce")
PAGE = 4096
_MIX, _XOR = 0x9E3779B97F4A7C15, 0xA5A5A5A5DEADBEEF


def pattern_word(trial: int) -> int:
    """Pattern A for a trial, as a signed int64 (torch's int64 view): a per-trial counter mixed and XORed."""
    w = ((trial + 1) * _MIX & 0xFFFFFFFFFFFFFFFF) ^ _XOR
    return w - (1 << 64) if w >= 1 << 63 else w


def regions(n: int, size: int, *, file_bytes: int, files: list[str]) -> list[tuple[str, int]]:
    """n distinct (file, offset) O_DIRECT regions of `size` bytes, 4 KiB aligned, rotating over the files and spread
    through each, so no two trials read the same bytes of one size."""
    if size % PAGE:
        raise ValueError(f"O_DIRECT needs a multiple of {PAGE} B, got {size}")
    slots = (file_bytes - size) // PAGE + 1
    out = []
    for i in range(n):
        path = files[i % len(files)]
        k = i // len(files)
        # a large odd stride spreads successive regions of one file through its whole length
        out.append((path, ((k * 7_919_993 + i * 104_729) % slots) * PAGE))
    if len(set(out)) != n:
        raise ValueError("regions collided; change the stride")
    return out


def summarize(records: list[dict]) -> dict[tuple[str, str, int], dict]:
    s: dict[tuple[str, str, int], dict] = defaultdict(
        lambda: {"trials": 0, "stale_words": 0, "wrong_words": 0, "bad_trials": 0})
    for r in records:
        if r.get("kind") != "trial":
            continue
        for m in METHODS:
            cell = s[(r["slab"], m, r["size"])]
            cell["trials"] += 1
            cell["stale_words"] += r["stale"][m]
            cell["wrong_words"] += r["wrong"][m]
            cell["bad_trials"] += int(r["wrong"][m] > 0 or r["stale"][m] > 0)
    return dict(s)


def main() -> int:
    records = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    s = summarize(records)
    print("| slab | method | size (B) | trials | stale words | wrong words | trials with any |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (slab, method, size), c in sorted(s.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        print(f"| {slab} | {method} | {size} | {c['trials']} | {c['stale_words']} | {c['wrong_words']} | "
              f"{c['bad_trials']} |")
    for r in records:
        if r.get("kind") in ("page_cache", "pattern_in_file"):
            print(json.dumps(r))
    return 1 if any(c["wrong_words"] for c in s.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
