"""Per-session cache-traffic table for the eager arms (eager-cache-arms.sh).

For each arm ``<dir>/<label>-<base|mirror>`` it joins three files on the shared
``time.monotonic()`` clock: the driver's ``.json`` (session start/end, TTFT, decode
tok/s, per-drive bytes), the scheduler's per-forward ``.trace`` and the cache
counters beside it, ``.trace.cache-stats``. A session's counters are the last
snapshot at or before its end minus the last at or before its start; snapshots come
at most every 0.5 s, so a session's edges are blurred by up to that much activity.

Usage: eager_cache_report.py <dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
from collections import defaultdict

GIB = 1 << 30
TIER_KEYS = ("hits", "admissions", "evictions", "lookup_hits", "lookup_misses", "populated_bytes")
ENGRAM_KEYS = ("accesses", "hits", "misses", "evictions")


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass  # a line cut by SIGKILL
    return rows


def at(snapshots, times, t):
    """The last snapshot at or before ``t`` (zeros before the first)."""
    i = bisect.bisect_right(times, t)
    return snapshots[i - 1] if i else {}


def delta(snapshots, times, keys, start, end):
    a, b = at(snapshots, times, start), at(snapshots, times, end)
    return {k: b.get(k, 0) - a.get(k, 0) for k in keys}


def arm_rows(prefix):
    report = json.load(open(prefix + ".json"))
    trace = load_jsonl(prefix + ".trace")
    stats = load_jsonl(prefix + ".trace.cache-stats")
    tier = [s for s in stats if s["kind"] == "pinned_tier"]
    engram = [s for s in stats if s["kind"] == "engram"]
    tier_t, engram_t = [s["t"] for s in tier], [s["t"] for s in engram]
    rows = []
    for s in report["per_session"]:
        start, end = s["start_t"], s["end_t"]
        forwards = [
            r for r in trace if "layer" in r and "vram_miss" in r and start < r["t"] <= end
        ]
        prefill = [r for r in forwards if r["tokens"] > 1]
        decode = [r for r in forwards if r["tokens"] == 1]
        vram = sum(r["vram_miss"] for r in forwards)
        ram = sum(r["ram_miss"] for r in forwards)
        background = sum(r["background_rows"] for r in forwards)
        disk = {k: v / GIB for k, v in s["disk_bytes"].items()}
        disk_total = sum(disk.values())
        d = delta(tier, tier_t, TIER_KEYS, start, end)
        e = delta(engram, engram_t, ENGRAM_KEYS, start, end)
        last = at(tier, tier_t, end)
        last_engram = at(engram, engram_t, end)
        read_rows = ram + background
        rows.append(
            {
                "session": s["session"],
                "ttft_s": s["ttft_s"],
                "decode_tok_s": s["decode_tok_s"],
                "output_sha1": s["output_sha1"],
                "disk_gib": disk,
                "disk_gib_total": disk_total,
                "vram_miss_rows": vram,
                "vram_miss_rows_prefill": sum(r["vram_miss"] for r in prefill),
                "ram_miss_rows": ram,
                "ram_miss_rows_prefill": sum(r["ram_miss"] for r in prefill),
                "background_rows": background,
                "read_time_s": sum(r["read_ms"] for r in forwards) / 1000,
                "forwards": len({r["forward"] for r in forwards}),
                "decode_calls": len(decode),
                "tier": d,
                "tier_occupancy": last.get("occupancy"),
                "tier_capacity": last.get("capacity"),
                "tier_protected_evictions": last.get("protected_evictions"),
                "engram": e,
                "engram_hit_rate": e["hits"] / e["accesses"] if e["accesses"] else None,
                "engram_filled": last_engram.get("filled_rows"),
                "engram_capacity": last_engram.get("capacity_rows"),
                "mib_per_read_row": disk_total * 1024 / read_rows if read_rows else None,
            }
        )
    return report, rows


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--json")
    args = p.parse_args()
    arms = defaultdict(list)  # kind -> [(name, report, rows)]
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        if path.endswith(".trace.json"):
            continue
        name = os.path.basename(path)[:-5]
        kind = name.rsplit("-", 1)[-1]
        if kind not in ("base", "mirror"):
            continue
        report, rows = arm_rows(path[:-5])
        arms[kind].append((name, report, rows))

    everything = {}
    for kind in ("base", "mirror"):
        for name, report, rows in arms[kind]:
            everything[name] = rows
            print(f"\n### {name}   mean decode {report['mean_decode_tok_s']:.4f} tok/s, "
                  f"startup disk {sum(report['startup_disk_bytes'].values()) / GIB:.1f} GiB")
            print("| s | TTFT s | tok/s | disk GiB (n0/n2/n4) | total | VRAM miss | RAM miss | bg rows | "
                  "MiB/read row | tier hits | tier admit | tier evict | route miss | route hit | occ/cap | "
                  "engram hit | engram miss | engram fill |")
            print("|" + "---|" * 18)
            for r in rows:
                t, d = r["tier"], r["disk_gib"]
                print(
                    f"| {r['session']} | {r['ttft_s']:.1f} | {r['decode_tok_s']:.3f} | "
                    f"{d['nvme0']:.1f}/{d['nvme2']:.1f}/{d['nvme4']:.1f} | {r['disk_gib_total']:.1f} | "
                    f"{r['vram_miss_rows']} | {r['ram_miss_rows']} | {r['background_rows']} | "
                    f"{fmt(r['mib_per_read_row'], '.2f')} | {t['hits']} | {t['admissions']} | {t['evictions']} | "
                    f"{t['lookup_misses']} | {t['lookup_hits']} | {fmt(r['tier_occupancy'], 'd')}/"
                    f"{fmt(r['tier_capacity'], 'd')} | {fmt(r['engram_hit_rate'], '.3f')} | "
                    f"{r['engram']['misses']} | {fmt(r['engram_filled'], 'd')}/{fmt(r['engram_capacity'], 'd')} |"
                )

    print("\n### Per-session means over repeats (session index, arm)")
    print("| s | arm | n | TTFT s | tok/s | disk GiB | RAM miss rows | tier admit | tier evict | occupancy |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in range(max((len(r) for v in arms.values() for _, _, r in v), default=0)):
        for kind in ("base", "mirror"):
            rows = [r[s] for _, _, r in arms[kind] if len(r) > s]
            if not rows:
                continue
            m = lambda f: sum(f(r) for r in rows) / len(rows)  # noqa: E731
            print(
                f"| {s} | {kind} | {len(rows)} | {m(lambda r: r['ttft_s']):.1f} | "
                f"{m(lambda r: r['decode_tok_s']):.3f} | {m(lambda r: r['disk_gib_total']):.1f} | "
                f"{m(lambda r: r['ram_miss_rows']):.0f} | {m(lambda r: r['tier']['admissions']):.0f} | "
                f"{m(lambda r: r['tier']['evictions']):.0f} | {m(lambda r: r['tier_occupancy'] or 0):.0f} |"
            )
    print("\n### Arm totals")
    print("| arm | mean tok/s | disk GiB | RAM miss rows | bg rows | tier admit | tier evict |")
    print("|---|---|---|---|---|---|---|")
    for kind in ("base", "mirror"):
        for name, report, rows in arms[kind]:
            print(
                f"| {name} | {report['mean_decode_tok_s']:.4f} | {sum(r['disk_gib_total'] for r in rows):.1f} | "
                f"{sum(r['ram_miss_rows'] for r in rows)} | {sum(r['background_rows'] for r in rows)} | "
                f"{sum(r['tier']['admissions'] for r in rows)} | {sum(r['tier']['evictions'] for r in rows)} |"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(everything, f, indent=1)


if __name__ == "__main__":
    main()
