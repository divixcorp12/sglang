"""Copy-phase metrics of one node-mode decode trace, for the SM small copies A/B.

Per steady decode step (a step is 40 consecutive post kernels; the longest run of sub-second steps is the decode;
steady = from its 15th step on): copy-engine copies, rows (8,847,360 B copies), copies per row, copy busy time,
the time from each layer's last copy to its CW end, and CW kernel time. The copy stream is the stream with the most
host-to-device copies. Usage: trace_compare.py REPORT.sqlite
"""

import bisect
import sqlite3
import statistics
import sys

ROW_HEAD = 8_847_360  # w13_trellis: one per row, the first copy of a row either way
LAYERS = 40

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)


def kern(name):
    return db.execute(
        "select k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
        "where s.value=? order by k.start",
        (name,),
    ).fetchall()


post = kern("exl3_ram_miss_post_kernel")
cw = kern("exl3_ram_miss_lease_copy_wait_kernel")
groups = [post[i : i + LAYERS] for i in range(0, len(post) - LAYERS + 1, LAYERS)]
dur = [(g[-1][0] - g[0][0]) / 1e9 for g in groups]
best, run = (0, 0), None
for i, d in enumerate(dur + [99]):
    if d < 1.0:
        run = i if run is None else run
    else:
        if run is not None and i - run > best[1] - best[0]:
            best = (run, i)
        run = None
first, last = best[0] + 15, best[1] - 1
stream = db.execute(
    "select streamId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 group by streamId order by count(*) desc limit 1"
).fetchone()[0]
lo = groups[first][0][0]
hi = groups[last + 1][0][0] if last + 1 < len(groups) else groups[last][-1][1] + 200_000_000
copies = db.execute(
    "select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=? and start between ? and ? order by start",
    (stream, lo, hi),
).fetchall()
post_starts = [p[0] for p in post]
cw_by_layer = {}
for a, b in cw:
    layer = bisect.bisect_right(post_starts, a) - 1
    cw_by_layer[layer] = (a, b)
by_layer = {}
for c in copies:
    by_layer.setdefault(bisect.bisect_right(post_starts, c[0]) - 1, []).append(c)
steps = last - first + 1
n_copies = len(copies)
rows = sum(c[2] == ROW_HEAD for c in copies)
busy = sum(c[1] - c[0] for c in copies)
tail, cw_time, cw_wait = [], [], 0
for g in range(first, last + 1):
    for layer in range(LAYERS * g, LAYERS * g + LAYERS):
        if layer not in cw_by_layer:
            continue
        a, b = cw_by_layer[layer]
        cw_time.append(b - a)
        if layer in by_layer:
            end = max(c[1] for c in by_layer[layer])
            tail.append(b - end)
            cw_wait += max(0, end - a)
tail.sort()
sizes = {}
for c in copies:
    sizes[c[2]] = sizes.get(c[2], 0) + 1
print(f"trace {sys.argv[1]}")
print(f"copy stream {stream}; decode steps {best[0]}..{best[1] - 1}, steady {first}..{last} ({steps} steps)")
print(f"copies/step {n_copies / steps:.1f}, rows/step {rows / steps:.1f}, copies/row {n_copies / max(rows, 1):.2f}")
print(f"copy sizes per step: " + ", ".join(f"{b}: {n / steps:.1f}" for b, n in sorted(sizes.items(), key=lambda x: -x[0])))
print(f"copy busy {busy / steps / 1e6:.3f} ms/step, {sum(c[2] for c in copies) / steps / 1e6:.1f} MB/step, "
      f"{sum(c[2] for c in copies) / max(busy, 1):.2f} GB/s")
print(f"CW end - last copy end (us): p10 {tail[len(tail) // 10] / 1e3:.1f} p50 {tail[len(tail) // 2] / 1e3:.1f} "
      f"p90 {tail[len(tail) * 9 // 10] / 1e3:.1f}")
print(f"CW kernel: {sum(cw_time) / steps / 1e6:.2f} ms/step, p50 {statistics.median(cw_time) / 1e3:.1f} us/layer; "
      f"CW waiting on copies {cw_wait / steps / 1e6:.2f} ms/step")
print(f"step wall p50 {statistics.median(groups[g + 1][0][0] - groups[g][0][0] for g in range(first, last)) / 1e6:.2f} ms")
