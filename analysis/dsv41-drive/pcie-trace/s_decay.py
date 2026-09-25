"""NVMe wait (S kernel time) and RAM-hit bytes per decode step, by position after the prefill."""
import bisect
import sqlite3

db = sqlite3.connect("file:/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
def kern(n):
    return db.execute("select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                      "where s.value=? and k.start > 46.5e9 order by k.start", (n,)).fetchall()
s, cw = kern("exl3_ram_miss_lease_stream_kernel"), kern("exl3_ram_miss_lease_copy_wait_kernel")
cp = db.execute("select start, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=141 and start > 46.5e9 order by start").fetchall()
cs = [a for a, _ in cp]
steps = len(s) // 40
rows = []
for k in range(steps):
    L = s[k * 40:(k + 1) * 40]
    s_ms = sum(b - a for a, b in L) / 1e6
    a0, a1 = L[0][0], (s[(k + 1) * 40][0] if (k + 1) * 40 < len(s) else L[-1][1] + 10_000_000)
    i, j = bisect.bisect_left(cs, a0 - 2_000_000), bisect.bisect_left(cs, a1 - 2_000_000)
    rows.append((s_ms, sum(b for _, b in cp[i:j]) / 1e9, (a1 - a0) / 1e6))
print(f"session-2 decode steps: {steps}")
for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 70), (70, steps)):
    part = rows[lo:hi]
    if not part:
        continue
    n = len(part)
    print(f"steps {lo:3d}-{hi-1:3d}: S {sum(r[0] for r in part)/n:6.1f} ms/step  RAM-hit copy {sum(r[1] for r in part)/n:5.2f} GB/step"
          f"  step wall {sum(r[2] for r in part[:-1] if r[2] < 1000)/max(1, len([r for r in part[:-1] if r[2] < 1000])):6.1f} ms")
