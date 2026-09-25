"""Copy-thread silences >= 0.5 ms in decode: how many calls lie between them (a CUPTI buffer flush is periodic)."""
import collections
import sqlite3
import statistics as st

db = sqlite3.connect("file:/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
TID = db.execute(
    "select r.globalTid from CUPTI_ACTIVITY_KIND_MEMCPY m join CUPTI_ACTIVITY_KIND_RUNTIME r "
    "on r.correlationId=m.correlationId where m.streamId=141 limit 1").fetchone()[0]
rows = db.execute(
    "select r.start, r.end from CUPTI_ACTIVITY_KIND_RUNTIME r where r.globalTid=? and r.start between 47e9 and 58.6e9 "
    "order by r.start", (TID,)).fetchall()
sil, last_idx = [], 0
between = []
for i in range(1, len(rows)):
    g = rows[i][0] - rows[i - 1][1]
    if g >= 500_000:
        sil.append(g / 1e3)
        between.append(i - last_idx)
        last_idx = i
print(f"calls {len(rows)}; silences >=0.5ms: {len(sil)}, total {sum(sil)/1e3:.1f} ms over {(rows[-1][1]-rows[0][0])/1e9:.1f} s")
print(f"silence length p50 {st.median(sil):.0f} us, max {max(sil):.0f} us")
between = between[1:]
print(f"calls between silences: median {st.median(between):.0f}, p10 {sorted(between)[len(between)//10]}, "
      f"p90 {sorted(between)[len(between)*9//10]}, cv {st.pstdev(between)/st.mean(between):.2f}")
print("most common call counts between silences:", collections.Counter(b // 1000 * 1000 for b in between).most_common(5))
