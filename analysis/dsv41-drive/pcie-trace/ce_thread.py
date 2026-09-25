"""Does the copy thread keep polling during long CW tails, or does it stop running?"""
import bisect
import collections
import sqlite3

db = sqlite3.connect("file:/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
tid = db.execute(
    "select r.globalTid from CUPTI_ACTIVITY_KIND_MEMCPY m join CUPTI_ACTIVITY_KIND_RUNTIME r "
    "on r.correlationId=m.correlationId where m.streamId=141 limit 1").fetchone()[0]
name = db.execute("select s.value from ThreadNames t join StringIds s on s.id=t.nameId where t.globalTid=?", (tid,)).fetchone()
print("copy thread", tid % 1000000, name)
calls = db.execute(
    "select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId "
    "where r.globalTid=? and r.start > 24.6e9 order by r.start", (tid,)).fetchall()
print("API calls by name:", collections.Counter(c[2] for c in calls).most_common(6))
starts = [c[0] for c in calls]

def kern(n):
    return db.execute("select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                      "where s.value=? order by k.start", (n,)).fetchall()
post, cw = kern("exl3_ram_miss_post_kernel"), kern("exl3_ram_miss_lease_copy_wait_kernel")
cp = db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=141 order by start").fetchall()
cs = [a for a, _ in cp]
long_tails = silent = 0
max_gaps = []
for i in range(len(post) - 1):
    j, k = bisect.bisect_left(cs, post[i][1]), bisect.bisect_left(cs, post[i + 1][0])
    if j >= k:
        continue
    last = max(b for _, b in cp[j:k])
    if cw[i][1] - last < 500_000:
        continue
    long_tails += 1
    a, b = bisect.bisect_left(starts, last), bisect.bisect_right(starts, cw[i][1])
    pts = [last] + starts[a:b] + [cw[i][1]]
    gap = max(pts[x + 1] - pts[x] for x in range(len(pts) - 1))
    max_gaps.append(gap / 1e3)
    if gap > 0.8 * (cw[i][1] - last):
        silent += 1
max_gaps.sort()
n = len(max_gaps)
print(f"tails >=0.5ms: {long_tails}; largest silence of the copy thread inside a tail: "
      f"p50 {max_gaps[n//2]:.0f} us, p90 {max_gaps[int(n*.9)]:.0f} us; tails that are >80% one silence: {silent}")
