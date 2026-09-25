"""Stream order of the chain kernels in one layer, and what the CW tail after the last copy overlaps."""
import bisect
import sqlite3

db = sqlite3.connect("file:/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
names = ("exl3_ram_miss_post_kernel", "exl3_ram_miss_lease_stream_hit_wait_kernel",
         "exl3_ram_miss_lease_copy_wait_kernel", "exl3_ram_miss_lease_stream_kernel", "exl3_moe_kernel")
K = {n: db.execute("select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                   "where s.value=? order by k.start", (n,)).fetchall() for n in names}
i = 2000  # an arbitrary mid-decode layer
for n in names:
    a, b = K[n][i]
    print(f"{n:45s} start {a/1e9:.6f}  dur {(b-a)/1e3:9.1f} us")
cp = db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=141 order by start").fetchall()
cs = [a for a, _ in cp]
post, cw, s = K[names[0]], K[names[2]], K[names[3]]
cw_tail = cw_tail_while_s = 0
for i in range(len(post) - 1):
    j, k = bisect.bisect_left(cs, post[i][1]), bisect.bisect_left(cs, post[i + 1][0])
    if j >= k:
        continue
    last = max(b for _, b in cp[j:k])
    t = max(0, cw[i][1] - last)
    cw_tail += t
    if s[i][0] <= last and s[i][1] >= cw[i][1]:
        cw_tail_while_s += t
print(f"CW tail {cw_tail/110/1e6:.2f} ms/step, of which during a still-running S kernel {cw_tail_while_s/110/1e6:.2f}")
