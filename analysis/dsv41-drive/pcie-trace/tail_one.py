"""One layer with a long CW tail, printed in full: the chain kernels and every copy between its post and the next."""
import bisect
import sqlite3

db = sqlite3.connect("file:/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
def kern(n):
    return db.execute("select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                      "where s.value=? order by k.start", (n,)).fetchall()
post, cw, s, moe = (kern(n) for n in ("exl3_ram_miss_post_kernel", "exl3_ram_miss_lease_copy_wait_kernel",
                                       "exl3_ram_miss_lease_stream_kernel", "exl3_moe_kernel"))
cp = db.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=141 order by start").fetchall()
cs = [a for a, _, _ in cp]
shown = 0
for i in range(2000, len(post) - 1):
    j, k = bisect.bisect_left(cs, post[i][1]), bisect.bisect_left(cs, post[i + 1][0])
    if j >= k:
        continue
    last = max(b for _, b, _ in cp[j:k])
    if cw[i][1] - last < 1_000_000:
        continue
    t0 = post[i][0]
    r = lambda x: f"{(x - t0)/1e3:9.1f}"
    print(f"layer {i}: post {r(post[i][0])}-{r(post[i][1])}  S {r(s[i][0])}-{r(s[i][1])}  CW {r(cw[i][0])}-{r(cw[i][1])}"
          f"  MoE {r(moe[i][0])}  next post {r(post[i+1][0])}")
    print(f"  {k-j} copies from {r(cp[j][0])} to {r(last)} us; CW ends {(cw[i][1]-last)/1e3:.1f} us after the last copy")
    # copies after the next post (belonging to later layers) that start before this CW ends?
    late = [c for c in cp[k:k + 40] if c[0] < cw[i][1]]
    print(f"  copies after next post starting before CW end: {len(late)}")
    shown += 1
    if shown == 3:
        break
