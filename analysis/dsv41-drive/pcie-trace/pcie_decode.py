"""Attribute PCIe RX over decode steps: calibration against copy-engine intervals, then per-state link use."""
import bisect
import sqlite3

D = "/home/dimitri/data/divix/nsys-reports/"
main = sqlite3.connect(f"file:{D}pcie-node-20260925-170510.sqlite?mode=ro", uri=True)
pcie = sqlite3.connect(f"file:{D}pcie-node-20260925-170510-pcie.sqlite?mode=ro", uri=True)
OFF = 530_792_635  # ns: pcie time = main time + OFF (session start systemClockNs difference)

def kern(name):
    return main.execute(
        "select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
        "where s.value=? order by k.start", (name,)).fetchall()

cw = kern("exl3_ram_miss_lease_copy_wait_kernel")
st = kern("exl3_ram_miss_lease_stream_kernel")
moe = kern("exl3_moe_kernel")
copies = main.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=141 order by start").fetchall()

# Decode steps: 40 MoE launches each; a step runs from its first CW start to the next step's first CW start.
steps = [cw[i][0] for i in range(0, len(cw), 40)]
lo, hi = cw[0][0], moe[-1][1]
rx = pcie.execute(
    "select timestamp, value from GPU_METRICS where metricId=21 and timestamp between ? and ? order by timestamp",
    (lo + OFF, hi + OFF)).fetchall()
ts = [t - OFF for t, _ in rx]
val = [v for _, v in rx]
print(f"decode window {lo/1e9:.2f}-{hi/1e9:.2f}s, {len(steps)} steps, {len(rx)} RX samples")

def mean_in(intervals, min_len=0):
    s = n = 0
    for a, b, *_ in intervals:
        if b - a < min_len:
            continue
        i, j = bisect.bisect_left(ts, a), bisect.bisect_right(ts, b)
        s += sum(val[i:j]); n += j - i
    return (s / n if n else float("nan")), n

# Calibration: merge back-to-back copies into bursts >= 1 ms; the link carries only copy-engine traffic there
bursts, cur = [], None
for a, b, by in copies:
    if cur and a - cur[1] < 10_000:
        cur = [cur[0], b, cur[2] + by]
    else:
        if cur: bursts.append(cur)
        cur = [a, b, by]
bursts.append(cur)
long_b = [x for x in bursts if x[1] - x[0] >= 1_000_000]
gbps = sum(x[2] for x in long_b) / sum(x[1] - x[0] for x in long_b)
m, n = mean_in(long_b)
print(f"calibration: {len(long_b)} copy bursts >=1ms at {gbps:.2f} GB/s show RX {m:.1f}% ({n} samples)"
      f" -> 100% RX ~ {gbps / m * 100:.1f} GB/s")
scale = gbps / m  # GB/s per RX-percent point

# Per-sample state over decode steps (only samples inside step windows).
def mark(intervals):
    flags = [False] * len(ts)
    for a, b, *_ in intervals:
        for i in range(bisect.bisect_left(ts, a), bisect.bisect_right(ts, b)):
            flags[i] = True
    return flags
in_copy, in_cw, in_st = mark(bursts), mark(cw), mark(st)
step_windows = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1) if steps[i + 1] - steps[i] < 1_000_000_000]
in_step = mark(step_windows)
wall = sum(b - a for a, b in step_windows) / 1e6
print(f"steps used: {len(step_windows)}, mean step wall {wall/len(step_windows):.1f} ms")
cats = {}
for i in range(len(ts)):
    if not in_step[i]:
        continue
    k = ("copy" if in_copy[i] else "nocopy") + "+" + ("CW" if in_cw[i] else "S" if in_st[i] else "other")
    c = cats.setdefault(k, [0, 0.0, 0])
    c[0] += 1; c[1] += val[i]; c[2] += val[i] < 5
tot = sum(c[0] for c in cats.values())
print(f"{'state':16s} {'share of step':>13s} {'ms/step':>8s} {'mean RX%':>8s} {'~GB/s':>6s} {'RX<5%':>6s}")
for k, (n, s, idle) in sorted(cats.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:16s} {n/tot*100:12.1f}% {n/tot*wall/len(step_windows):8.1f} {s/n:8.1f} {s/n*scale:6.2f} {idle/n*100:5.1f}%")
allrx = sum(val[i] for i in range(len(ts)) if in_step[i]) / tot
print(f"all step samples: mean RX {allrx:.1f}% ~ {allrx*scale:.2f} GB/s; bytes/step ~ {allrx*scale*wall/len(step_windows)/1e3:.3f} GB")
