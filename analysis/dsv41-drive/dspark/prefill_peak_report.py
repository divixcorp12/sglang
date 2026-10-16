"""Summary of a server log's SGLANG_DEBUG_PREFILL_PEAK_LOG records (one per DSpark prefill).

Per record: its tokens, allocated at entry, its allocated and reserved peaks, and the reserved growth over the
prefill. Then, for the record with the most tokens (or --tokens N), each call name's largest transient (allocated peak
less allocated at entry) and reserved growth, summed over layers by kind (attn, moe, layer body) and listed for the
largest single calls.

Usage: prefill_peak_report.py SERVER_LOG [--tokens N] [--top 15]
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from sglang.srt.debug_utils.prefill_peak_log import parse_line  # noqa: E402


def _kind(name: str) -> str:
    return re.sub(r"^L\d+", "L*", name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--tokens", type=int)
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()
    records = [r for r in (parse_line(line) for line in open(a.log, errors="replace")) if r]
    print(f"{len(records)} prefill records")
    print("tokens  alloc_in  alloc_peak  reserved_in  reserved_peak  reserved_growth (MiB)")
    for r in records:
        root = r["calls"][-1]
        print(f"{r['tokens']!s:>6}  {root['alloc_before_mib']:8.0f}  {root['alloc_peak_mib']:10.0f}  "
              f"{root['reserved_before_mib']:11.0f}  {root['reserved_peak_mib']:13.0f}  "
              f"{root['reserved_peak_mib'] - root['reserved_before_mib']:15.0f}")
    pick = [r for r in records if a.tokens is None or r["tokens"] == a.tokens]
    if not pick:
        return
    rec = max(pick, key=lambda r: r["tokens"] or 0)
    print(f"\nrecord with {rec['tokens']} tokens: transient = alloc_peak - alloc_before; growth = reserved_peak - "
          "reserved_before")
    by_kind = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for c in rec["calls"]:
        k = by_kind[_kind(c["name"])]
        k[0] = max(k[0], c["alloc_peak_mib"] - c["alloc_before_mib"])
        k[1] += c["reserved_peak_mib"] - c["reserved_before_mib"]
        k[2] += 1
    print("kind              calls  max transient  summed reserved growth (MiB)")
    for kind, (transient, growth, n) in sorted(by_kind.items(), key=lambda kv: -kv[1][0]):
        print(f"{kind:16s}  {n:5d}  {transient:13.0f}  {growth:12.0f}")
    print(f"\nlargest {a.top} calls by transient")
    for c in sorted(rec["calls"], key=lambda c: c["alloc_before_mib"] - c["alloc_peak_mib"])[: a.top]:
        print(f"{c['name']:16s} depth {c['depth']}  in {c['alloc_before_mib']:7.0f}  peak {c['alloc_peak_mib']:7.0f}  "
              f"out {c['alloc_after_mib']:7.0f}  reserved {c['reserved_before_mib']:7.0f} -> {c['reserved_peak_mib']:7.0f}")


if __name__ == "__main__":
    main()
