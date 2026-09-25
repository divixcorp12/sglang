"""Do copy-thread silences coincide with traced cudaGraphLaunch calls on other threads (a CUPTI-lock artefact)?"""
import bisect
import sqlite3

db = sqlite3.connect("file:/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
W = (47e9, 58.6e9)
TID = db.execute(
    "select r.globalTid from CUPTI_ACTIVITY_KIND_MEMCPY m join CUPTI_ACTIVITY_KIND_RUNTIME r "
    "on r.correlationId=m.correlationId where m.streamId=141 limit 1").fetchone()[0]
rows = db.execute("select start, end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start between ? and ? "
                  "order by start", (TID, *W)).fetchall()
sil = [(rows[i - 1][1], rows[i][0]) for i in range(1, len(rows)) if rows[i][0] - rows[i - 1][1] >= 500_000]
other = db.execute(
    "select r.start, r.end, s.value, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId "
    "where r.globalTid != ? and r.start between ? and ? and r.end - r.start > 200000 order by r.start",
    (TID, *W)).fetchall()
print("long (>0.2ms) API calls on other threads:", len(other))
by = {}
for a, b, n, t in other:
    by.setdefault(n, [0, 0.0])
    by[n][0] += 1; by[n][1] += (b - a) / 1e6
for n, (c, ms) in sorted(by.items(), key=lambda kv: -kv[1][1])[:6]:
    print(f"  {n:32s} n={c:6d} total {ms:8.1f} ms")
starts = [o[0] for o in other]
covered = 0
names = {}
for a, b in sil:
    i = bisect.bisect_left(starts, a - 20_000_000)
    best = 0
    for o in other[i:]:
        if o[0] > b:
            break
        ov = min(b, o[1]) - max(a, o[0])
        if ov > best:
            best, nm = ov, o[2]
    covered += best
    if best > 0.5 * (b - a):
        names[nm] = names.get(nm, 0) + 1
tot = sum(b - a for a, b in sil)
print(f"silences {len(sil)} total {tot/1e6:.0f} ms; covered by the single longest overlapping call: {covered/tot*100:.0f}%")
print("silences >50% inside one call, by call:", names)
