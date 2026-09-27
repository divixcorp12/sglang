#!/usr/bin/env python3
"""Summaries of a copy-mechanism sweep (mech_bench.py JSONL): link ceilings, knees, bandwidth-delay product.

Pure Python (no torch, no CUDA) so it runs anywhere; tested by mech_report_test.py.

    python3 mech_report.py <results.jsonl>   # markdown summary; exit 1 on an above-ceiling cell, an unsafe method
                                             # or a blind fresh check
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

# Payload GB/s per PCIe lane after 128b/130b line coding, before TLP/DLLP overhead: Gen3 x16 = 15.75.
LANE_GBS = {3: 8.0 * 128 / 130 / 8, 4: 16.0 * 128 / 130 / 8, 5: 32.0 * 128 / 130 / 8}
KNEE_FRACTION = 0.95
CONTROL = "nc_control"  # the fresh check's negative control: it must come out stale


def link_gbs(gen: int, width: int) -> float:
    if gen not in LANE_GBS:
        raise ValueError(f"PCIe gen {gen} has no ceiling here; add it to LANE_GBS")
    return LANE_GBS[gen] * width


def bdp_bytes(latency_ns: float, gbs: float) -> int:
    """Bytes that must be in flight to keep a `gbs` link busy at `latency_ns` per read (ns x GB/s = bytes)."""
    return round(latency_ns * gbs)


def knee(points: list[tuple[int, float]], fraction: float = KNEE_FRACTION) -> int | None:
    """The fewest bytes in flight whose GB/s reaches `fraction` of the best point's."""
    if not points:
        return None
    best = max(gbs for _, gbs in points)
    return min(in_flight for in_flight, gbs in points if gbs >= fraction * best)


def above_ceiling(cells: list[dict], ceiling: float) -> list[dict]:
    """Cells faster than the link allows: L2-resident or mistimed, never a result."""
    return [cell for cell in cells if cell["gbs"] > ceiling]


def summarize(records: list[dict]) -> dict:
    metas = [r for r in records if r.get("meta")]
    if len(metas) != 1:
        raise ValueError(f"expected exactly one meta record, found {len(metas)}")
    meta = metas[0]
    cells = [r for r in records if r.get("kind") == "cell"]
    ceiling = link_gbs(meta["pcie_gen"], meta["pcie_width"])
    by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for cell in cells:
        by_method[cell["method"]].append((cell["in_flight"], cell["gbs"]))
    measured = max((gbs for points in by_method.values() for _, gbs in points), default=0.0)
    latency = next((r["serial_acquire_ns"] for r in records if r.get("kind") == "latency"), None)
    rtt = next((r["rtt_ns_p50"] for r in records if r.get("kind") == "pingpong"), None)
    rtt_min = next((r["rtt_ns_min"] for r in records if r.get("kind") == "pingpong"), None)
    bdp_acquire = bdp_bytes(latency, measured) if latency else None
    fresh = {r["method"]: r["fresh"] for r in records if r.get("kind") == "fresh"}
    return {
        "host": meta["host"],
        "theoretical_gbs": round(ceiling, 2),
        "measured_ceiling_gbs": measured,
        "above_ceiling": above_ceiling(cells, ceiling),
        "serial_acquire_ns": latency,
        "flag_rtt_ns": rtt,
        # Primary: the minimum flag round trip. A copy with a fixed number of bytes in flight is bounded by
        # in_flight / RTT, and the serial acquire understates that latency (divix01 Gen3: 16 KiB cells run at
        # 11.5-11.8 GB/s whatever their grid or width, = 16 KiB / 1.41 us, the 1408 ns ping-pong minimum).
        "bdp_bytes": bdp_bytes(rtt_min, measured) if rtt_min else bdp_acquire,
        "bdp_acquire_bytes": bdp_acquire,
        "flag_rtt_min_ns": rtt_min,
        "unsafe": sorted(m for m, ok in fresh.items() if m != CONTROL and not ok),
        "control_blind": bool(fresh.get(CONTROL, False)),
        "methods": {
            method: {
                "best_gbs": max(gbs for _, gbs in points),
                "knee_bytes": None if method.startswith("ce") else knee(points),
                "share_of_measured": round(max(gbs for _, gbs in points) / measured, 3) if measured else None,
            }
            for method, points in sorted(by_method.items())
        },
        "named": {r["name"]: r["gbs"] for r in cells if r.get("name")},
        "curves": {
            method: sorted({f: max(g for f2, g in points if f2 == f) for f, _ in points}.items())
            for method, points in sorted(by_method.items())
            if not method.startswith("ce")
        },
    }


def main() -> int:
    records = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    s = summarize(records)
    print(f"# {s['host']}: theoretical {s['theoretical_gbs']} GB/s, measured ceiling {s['measured_ceiling_gbs']} GB/s")
    print(f"serial acquire {s['serial_acquire_ns']} ns, flag RTT p50 {s['flag_rtt_ns']} ns, min {s['flag_rtt_min_ns']} ns")
    print(f"BDP (ceiling x RTT min, primary) {s['bdp_bytes']} B; BDP (ceiling x serial acquire) {s['bdp_acquire_bytes']} B\n")
    print("| method | best GB/s | knee (bytes in flight) | share of measured |\n|---|---:|---:|---:|")
    for method, m in s["methods"].items():
        print(f"| {method} | {m['best_gbs']} | {m['knee_bytes']} | {m['share_of_measured']} |")
    print(f"\nnamed cells: {s['named']}")
    sizes = sorted({f for points in s["curves"].values() for f, _ in points})
    print("\nbest GB/s by bytes in flight (K = the method's knee)\n")
    print("| method | " + " | ".join(f"{f // 1024} KiB" for f in sizes) + " |")
    print("|---|" + "---:|" * len(sizes))
    for method, points in s["curves"].items():
        at = dict(points)
        knee_at = s["methods"][method]["knee_bytes"]
        row = [("" if f not in at else f"{at[f]}{' K' if f == knee_at else ''}") for f in sizes]
        print(f"| {method} | " + " | ".join(row) + " |")
    bad = False
    if s["above_ceiling"]:
        print(f"ABOVE CEILING: {s['above_ceiling']}")
        bad = True
    if s["unsafe"]:
        print(f"UNSAFE (failed the fresh check): {s['unsafe']}")
        bad = True
    if s["control_blind"]:
        print("FRESH CHECK BLIND: the .nc control read fresh bytes, so no fresh verdict above means anything")
        bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
