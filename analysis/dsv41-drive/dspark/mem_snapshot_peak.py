"""What is live at the allocated-memory peak of a torch allocator history (prefill_peak_probe.py --mem-profile).

Replays the snapshot's device trace: each alloc adds a block, each free_completed removes it. Blocks freed in the trace
but never allocated in it were live before recording started; they are counted as a pre-existing total. At the step
where the trace's own live bytes peak, prints those bytes grouped by the innermost sglang frame of their allocation
(and its caller), largest first, plus the pre-existing total and the snapshot's reserved segments.

Usage: mem_snapshot_peak.py SNAPSHOT.pickle [--top 25]
"""

from __future__ import annotations

import argparse
import collections
import pickle


def _site(frames: list[dict]) -> str:
    """The two innermost frames under python/sglang, as file:line name."""
    ours = [f for f in frames if "/sglang/" in f.get("filename", "") and "/torch/" not in f["filename"]]
    picked = ours[:2] or frames[:1]
    return " <- ".join(f"{f['filename'].split('/sglang/', 1)[-1]}:{f['line']} {f['name']}" for f in picked)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshot")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()
    with open(a.snapshot, "rb") as f:
        snap = pickle.load(f)
    trace = max(snap["device_traces"], key=len)
    live: dict[int, tuple[int, str]] = {}
    pre_existing = 0
    cur = peak = 0
    peak_live: dict[int, tuple[int, str]] = {}
    for ev in trace:
        action, addr, size = ev["action"], ev["addr"], ev["size"]
        if action == "alloc":
            live[addr] = (size, _site(ev.get("frames", [])))
            cur += size
            if cur > peak:
                peak, peak_live = cur, dict(live)
        elif action == "free_completed":
            if addr in live:
                cur -= live.pop(addr)[0]
            else:
                pre_existing += size
    by_site = collections.Counter()
    count = collections.Counter()
    for size, site in peak_live.values():
        by_site[site] += size
        count[site] += 1
    reserved = sum(s["total_size"] for s in snap["segments"])
    allocated_end = sum(b["size"] for s in snap["segments"] for b in s["blocks"] if b["state"] == "active_allocated")
    print(f"trace events {len(trace)}; peak of in-trace live bytes {peak / 2**20:.0f} MiB")
    print(f"freed in trace but allocated before it: {pre_existing / 2**20:.0f} MiB")
    print(f"at dump: reserved {reserved / 2**20:.0f} MiB, allocated {allocated_end / 2**20:.0f} MiB")
    for site, size in by_site.most_common(a.top):
        print(f"{size / 2**20:9.1f} MiB  x{count[site]:<5d} {site}")


if __name__ == "__main__":
    main()
