"""Decode: S (NVMe wait) by layer; per-step GPU time split into waits, compute and gaps. Prefill: kernel mix."""
import collections
import sqlite3
import statistics as st

db = sqlite3.connect("file:/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
def kern(n):
    return db.execute("select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                      "where s.value=? order by k.start", (n,)).fetchall()
s = kern("exl3_ram_miss_lease_stream_kernel")
steps = len(s) // 40
by_layer = [[(s[k * 40 + l][1] - s[k * 40 + l][0]) / 1e6 for k in range(steps)] for l in range(40)]
tot = [sum(v) / steps for v in by_layer]
print("S ms/step by streamed-layer index (top 8):",
      ", ".join(f"L{l}:{tot[l]:.2f}" for l in sorted(range(40), key=lambda l: -tot[l])[:8]))
print(f"S ms/step total {sum(tot):.1f}; layers with median S > 1 ms: "
      f"{[l for l in range(40) if st.median(by_layer[l]) > 1.0]}")

# Per decode step (graph launches): GPU busy by class and the gaps between kernels.
WAITS = ("exl3_ram_miss_lease_copy_wait_kernel", "exl3_ram_miss_lease_stream_kernel",
         "exl3_ram_miss_lease_stream_hit_wait_kernel", "wait_kernel")
rows = db.execute(
    "select k.start, k.end, s.value, k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
    "where k.graphNodeId is not null and k.start > 47e9 order by k.start").fetchall()
per = collections.defaultdict(list)
for r in rows:
    per[r[3]].append(r)
wait = comp = gap = 0.0
small = 0
n = 0
for c, ks in per.items():
    if len(ks) < 2000:
        continue
    n += 1
    for i, (a, b, name, _) in enumerate(ks):
        if name in WAITS:
            wait += b - a
        else:
            comp += b - a
            small += (b - a) < 3000
        if i:
            g = a - ks[i - 1][1]
            if g > 0:
                gap += g
print(f"decode steps {n}: per step waits {wait/n/1e6:.1f} ms, compute {comp/n/1e6:.1f} ms, "
      f"inter-kernel gaps {gap/n/1e6:.1f} ms, compute kernels <3us: {small/n:.0f}")

# Prefill (session 1): kernel launches by name.
pre = db.execute(
    "select s.value, count(*), sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
    "where k.start between 1.18e9 and 24.56e9 group by s.value order by count(*) desc limit 10").fetchall()
print("prefill kernels by count:")
for name, cnt, t in pre:
    print(f"  {cnt:7d}  {t/1e6:8.1f} ms  {name[:70]}")
