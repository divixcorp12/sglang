"""Prefill GPU idle of the last timed session, split by what the launching thread was doing at the time.

Usage: idle_by_host_state.py MAIN_SQLITE  (a node-mode trace; run where the report lives; bound memory).

Host states on the thread that launches the gathers:
- fill wait: a stretch of at least 1 ms with no CUDA call that ends within 1 ms of a chunk's first gather launch.
  That is gather_rows' _await_fills spinning in RamTier::fill_wait (20 us sleeps) for the chunk's NVMe rows; split
  into a layer's first chunk (after a deepseek_rope_kernel) and its later chunks.
- sync: inside a blocking CUDA call (cudaStreamSynchronize, cudaEventSynchronize, a D2H copy) longer than 50 us.
- other: everything else, the host running Python and launching.
"""

import bisect
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from boundary_timeline import gather_groups, layer_starts  # noqa: E402
from compare_arms import connect, windows  # noqa: E402

FILL_GAP_NS = 1_000_000
NEAR_NS = 1_000_000
SYNC_NS = 50_000


def merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(a, b, spans, starts):
    """Nanoseconds of [a, b) covered by the sorted disjoint spans."""
    total = 0
    i = max(0, bisect.bisect_right(starts, a) - 1)
    while i < len(spans) and spans[i][0] < b:
        total += max(0, min(b, spans[i][1]) - max(a, spans[i][0]))
        i += 1
    return total


def main(path):
    db = connect(path)
    p0, d0 = windows(db)
    names = dict(db.execute("select id, value from StringIds"))
    groups = gather_groups(db, names, p0, d0)
    ropes = layer_starts(db, names, p0, d0)
    launch = {}
    first_corr = {g[2] for g in groups}
    tid = None
    for corr, start, gtid in db.execute(
        "select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?",
        (p0 - 10**9, d0),
    ):
        if corr in first_corr:
            launch[corr] = start
            tid = gtid
    calls = db.execute(
        "select r.start, r.end, s.value, r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join StringIds s on s.id = r.nameId where r.globalTid = ? and r.start >= ? and r.start < ? order by r.start",
        (tid, p0, d0),
    ).fetchall()
    d2h = {
        corr
        for (corr,) in db.execute(
            "select correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 and start >= ? and start < ?",
            (p0, d0),
        )
    }
    syncs = merged(
        [s, e]
        for s, e, name, corr in calls
        if e - s > SYNC_NS and ("ynchronize" in name or corr in d2h)
    )
    # Fill waits: long CUDA-free gaps ending just before a chunk's first gather launch.
    first_of_layer = set()
    for g in groups:
        i = bisect.bisect_left(ropes, g[0])
        if i > 0 and not any(h[0] < g[0] and h[0] > ropes[i - 1] for h in groups if h is not g):
            first_of_layer.add(g[2])
    launches = sorted((launch[g[2]], g[2] in first_of_layer) for g in groups if g[2] in launch)
    launch_times = [t for t, _ in launches]
    fill_first, fill_later = [], []
    last_end = calls[0][1]
    for s, e, _, _ in calls[1:]:
        if s - last_end >= FILL_GAP_NS:
            j = bisect.bisect_left(launch_times, s)
            if j < len(launches) and launch_times[j] - s <= NEAR_NS:
                (fill_first if launches[j][1] else fill_later).append([last_end, s])
        last_end = max(last_end, e)
    busy = merged(
        [s, e]
        for s, e in db.execute(
            "select start, end from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ? "
            "union all select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where end > ? and start < ?",
            (p0, d0, p0, d0),
        )
    )
    idle, cursor = [], p0
    for s, e in busy:
        if s > cursor:
            idle.append([cursor, min(s, d0)])
        cursor = max(cursor, e)
    if cursor < d0:
        idle.append([cursor, d0])
    states = {"fill wait, layer's first chunk": merged(fill_first), "fill wait, later chunks": merged(fill_later),
              "sync": syncs}
    starts = {k: [s for s, _ in v] for k, v in states.items()}
    totals = dict.fromkeys(list(states) + ["other (Python, launches)"], 0)
    for a, b in idle:
        left = b - a
        for k, spans in states.items():
            got = overlap(a, b, spans, starts[k])
            totals[k] += got
            left -= got
        totals["other (Python, launches)"] += max(0, left)
    idle_ns = sum(b - a for a, b in idle)
    print(f"prefill {(d0 - p0) / 1e6:.0f} ms; GPU idle {idle_ns / 1e6:.0f} ms ({idle_ns / (d0 - p0):.0%}); "
          f"{len(groups)} chunks, {len(first_of_layer)} first-of-layer")
    for k, v in totals.items():
        print(f"  {k:32s} {v / 1e6:7.0f} ms")
    for label, spans in (("first chunk", fill_first), ("later chunks", fill_later)):
        if spans:
            lens = sorted((e - s) / 1e6 for s, e in spans)
            print(f"  fill waits, {label}: n={len(lens)}, total {sum(lens):.0f} ms, median {lens[len(lens) // 2]:.1f} ms, "
                  f"max {lens[-1]:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1])
