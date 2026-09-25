"""Copy-engine service latency per decode layer: post end -> first copy start, last copy end -> CW end."""
import bisect
import sqlite3
import statistics as st

db = sqlite3.connect("file:/home/dimitri/data/divix/nsys-reports/pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
def kern(name):
    return db.execute("select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
                      "where s.value=? order by k.start", (name,)).fetchall()
post, cw, strm, moe = (kern(n) for n in ("exl3_ram_miss_post_kernel", "exl3_ram_miss_lease_copy_wait_kernel",
                                          "exl3_ram_miss_lease_stream_kernel", "exl3_moe_kernel"))
cp = db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=141 order by start").fetchall()
cs = [a for a, _ in cp]
react, tail, cw_after_copy, layer_bytes_time = [], [], [], []
for i, (pa, pb) in enumerate(post):
    nxt = post[i + 1][0] if i + 1 < len(post) else float("inf")
    j = bisect.bisect_left(cs, pb)
    k = bisect.bisect_left(cs, nxt)
    if j >= k:
        continue  # a layer with no RAM-hit copies
    react.append((cp[j][0] - pb) / 1e3)
    last_end = max(b for _, b in cp[j:k])
    c_end = cw[i][1]
    tail.append((c_end - last_end) / 1e3)
    layer_bytes_time.append((last_end - cp[j][0]) / 1e3)
def q(x):
    x = sorted(x); n = len(x)
    return f"p50 {x[n//2]:8.1f}  p90 {x[int(n*.9)]:8.1f}  mean {st.mean(x):8.1f} us  (n={n})"
print("post end -> first copy start :", q(react))
print("last copy end -> CW end      :", q(tail))
print("copy span per layer          :", q(layer_bytes_time))
print(f"per step: react {sum(react)/110/1e3:.2f} ms, CW tail {sum(tail)/110/1e3:.2f} ms")
# S after CW: time the layer spends in S after its copies completed (waiting for NVMe-streamed rows)
s_after = [(strm[i][1] - max(cw[i][1], strm[i][0])) / 1e3 for i in range(len(strm)) if strm[i][1] > cw[i][1]]
print("S beyond CW (NVMe wait)      :", q(s_after), f" -> {sum(s_after)/110/1e3:.2f} ms/step")
order = sum(1 for i in range(len(strm)) if strm[i][0] < cw[i][0])
print(f"S launched before CW in {order}/{len(strm)} layers")
