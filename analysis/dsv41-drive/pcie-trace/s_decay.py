"""NVMe wait (S kernel time) and RAM-hit bytes per decode step, by position after the prefill.

Usage: s_decay.py [SQLITE] [--start-ns T]. The defaults are the §27 report and the decode of its second session.
Without --start-ns, decode starts at the first S kernel after the last prefill gather (``_gather_host_rows_kernel``),
i.e. the last session's decode. The RAM-hit copies are the memcpys of the stream that moved the most bytes there.
"""
import argparse
import bisect
import sqlite3

p = argparse.ArgumentParser()
p.add_argument("db", nargs="?", default="/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite")
p.add_argument("--start-ns", type=float, default=None)
args = p.parse_args()
db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)


def kern(n, after):
    return db.execute("select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                      "where s.value=? and k.start > ? order by k.start", (n, after)).fetchall()


start = args.start_ns
if start is None:
    gathers = kern("_gather_host_rows_kernel", 0)
    if not gathers:
        raise SystemExit("no prefill gather in this report (graph mode, or no prefill): pass --start-ns")
    # 1 ns before the first S kernel, so the query's `start >` keeps it.
    start = kern("exl3_ram_miss_lease_stream_kernel", gathers[-1][1])[0][0] - 1
s = kern("exl3_ram_miss_lease_stream_kernel", start)
stream = db.execute("select streamId from CUPTI_ACTIVITY_KIND_MEMCPY where start > ? group by streamId "
                    "order by sum(bytes) desc limit 1", (start,)).fetchone()[0]
cp = db.execute("select start, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=? and start > ? order by start",
                (stream, start)).fetchall()
cs = [a for a, _ in cp]
steps = len(s) // 40
rows = []
for k in range(steps):
    L = s[k * 40:(k + 1) * 40]
    s_ms = sum(b - a for a, b in L) / 1e6
    a0, a1 = L[0][0], (s[(k + 1) * 40][0] if (k + 1) * 40 < len(s) else L[-1][1] + 10_000_000)
    i, j = bisect.bisect_left(cs, a0 - 2_000_000), bisect.bisect_left(cs, a1 - 2_000_000)
    rows.append((s_ms, sum(b for _, b in cp[i:j]) / 1e9, (a1 - a0) / 1e6))
print(f"decode from {start / 1e9:.3f} s: {steps} steps; copy stream {stream}")
for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 70), (70, steps)):
    part = rows[lo:hi]
    if not part:
        continue
    n = len(part)
    print(f"steps {lo:3d}-{hi-1:3d}: S {sum(r[0] for r in part)/n:6.1f} ms/step  RAM-hit copy {sum(r[1] for r in part)/n:5.2f} GB/step"
          f"  step wall {sum(r[2] for r in part[:-1] if r[2] < 1000)/max(1, len([r for r in part[:-1] if r[2] < 1000])):6.1f} ms")
