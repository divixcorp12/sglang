"""Prefill and decode of the last timed session in two node-mode traced arms, side by side.

Usage: compare_arms.py NAME=MAIN_SQLITE[,PCIE_SQLITE] ...  (run where the reports live; bound memory).

Decode starts at the first S kernel after the last prefill gather (as s_decay.py); a step is 40 S kernels. The
prefill window runs from the first kernel after the largest GPU idle gap since the previous session's last S kernel
to decode start. PCIe RX (metric "PCIe RX Throughput") is aligned by the two sessions' UTC start times.
"""

import bisect
import sqlite3
import statistics
import sys


def connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def kern(db, name, lo=0, hi=1 << 62):
    return db.execute(
        "select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
        "where s.value=? and k.start > ? and k.start < ? order by k.start",
        (name, lo, hi),
    ).fetchall()


def union_ms(intervals):
    total, cur_a, cur_b = 0, None, None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total / 1e6


def windows(db):
    gathers = kern(db, "_gather_host_rows_kernel")
    decode0 = kern(db, "exl3_ram_miss_lease_stream_kernel", gathers[-1][1])[0][0]
    prev_s = db.execute(
        "select max(k.end) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
        "where s.value='exl3_ram_miss_lease_stream_kernel' and k.end < ?",
        (gathers[-1][0],),
    ).fetchone()[0]
    ks = db.execute(
        "select start, end from CUPTI_ACTIVITY_KIND_KERNEL where start > ? and start < ? order by start",
        (prev_s, decode0),
    ).fetchall()
    best, prefill0, last_end = 0, ks[0][0], prev_s
    for a, b in ks:
        if a - last_end > best:
            best, prefill0 = a - last_end, a
        last_end = max(last_end, b)
    return prefill0, decode0


def prefill_report(db, p0, d0):
    ks = db.execute(
        "select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
        "where k.start >= ? and k.start < ?",
        (p0, d0),
    ).fetchall()
    mc = db.execute(
        "select start, end, bytes, copyKind, srcKind, dstKind, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY "
        "where start >= ? and start < ?",
        (p0, d0),
    ).fetchall()
    wall = (d0 - p0) / 1e6
    busy = union_ms([(a, b) for a, b, _ in ks] + [(a, b) for a, b, *_ in mc])
    by_name = {}
    for a, b, n in ks:
        e = by_name.setdefault(n, [0, 0])
        e[0] += 1
        e[1] += b - a
    readbacks = [m for m in mc if m[3] == 2 and m[2] <= 256]
    corr = [m[6] for m in readbacks]
    block = 0.0
    for i in range(0, len(corr), 500):
        chunk = corr[i : i + 500]
        q = ",".join("?" * len(chunk))
        block += db.execute(
            f"select coalesce(sum(end-start),0) from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId in ({q})", chunk
        ).fetchone()[0]
    pageable_h2d = [m for m in mc if m[3] == 1 and m[4] == 0]
    out = {
        "wall ms": wall,
        "GPU busy ms (kernels+copies)": busy,
        "GPU idle %": 100 * (1 - busy / wall),
        "kernels": len(ks),
        "gather kernels / ms": (by_name.get("_gather_host_rows_kernel", [0, 0])[0],
                                by_name.get("_gather_host_rows_kernel", [0, 0])[1] / 1e6),
        "D2H readbacks <=256B / host-blocked ms": (len(readbacks), block / 1e6),
        "pageable H2D copies / MB": (len(pageable_h2d), sum(m[2] for m in pageable_h2d) / 1e6),
        "all H2D MB": sum(m[2] for m in mc if m[3] == 1) / 1e6,
    }
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:8]
    return out, [(n, c, t / 1e6) for n, (c, t) in top]


def decode_report(db, d0):
    s = kern(db, "exl3_ram_miss_lease_stream_kernel", d0 - 1)
    steps = len(s) // 40
    starts = [s[k * 40][0] for k in range(steps)]
    ks = db.execute(
        "select start, end from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? order by start", (starts[0],)
    ).fetchall()
    kstarts = [a for a, _ in ks]
    rows = []
    for k in range(steps - 1):
        a0, a1 = starts[k], starts[k + 1]
        if a1 - a0 > 1_000_000_000:
            continue
        i, j = bisect.bisect_left(kstarts, a0), bisect.bisect_left(kstarts, a1)
        step_k = ks[i:j]
        s_ms = sum(b - a for a, b in s[k * 40 : (k + 1) * 40]) / 1e6
        small = sum(1 for a, b in step_k if b - a < 3000)
        busy = union_ms(step_k)
        rows.append((k, (a1 - a0) / 1e6, s_ms, len(step_k), small, busy))
    out = {}
    for lo, hi in ((0, 1), (1, 5), (5, 15), (15, 10_000)):
        part = [r for r in rows if lo <= r[0] < hi]
        if part:
            out[f"steps {lo}-{min(hi, steps) - 1}"] = (
                len(part),
                statistics.median(r[1] for r in part),
                statistics.median(r[2] for r in part),
            )
    steady = [r for r in rows if r[0] >= 15]
    out["steady kernels/step, sub-3us/step, GPU busy ms/step"] = (
        statistics.median(r[3] for r in steady),
        statistics.median(r[4] for r in steady),
        statistics.median(r[5] for r in steady),
    )
    return out, starts


def pcie_rx(main, pcie, lo, hi):
    t_main = main.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    t_pcie = pcie.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    off = t_main - t_pcie
    mid = pcie.execute(
        "select metricId from TARGET_INFO_GPU_METRICS where metricName like 'PCIe RX Throughput%' limit 1"
    ).fetchone()[0]
    vals = [
        v
        for (v,) in pcie.execute(
            "select value from GPU_METRICS where metricId=? and timestamp between ? and ?", (mid, lo + off, hi + off)
        )
    ]
    return off, (statistics.mean(vals) if vals else float("nan")), len(vals)


for arg in sys.argv[1:]:
    name, paths = arg.split("=", 1)
    main_path, *rest = paths.split(",")
    db = connect(main_path)
    p0, d0 = windows(db)
    print(f"===== {name}: prefill {p0 / 1e9:.3f}-{d0 / 1e9:.3f} s")
    pre, top = prefill_report(db, p0, d0)
    for k, v in pre.items():
        print(f"  prefill {k}: {v if not isinstance(v, float) else round(v, 1)}")
    print("  prefill top kernels (name, count, ms):")
    for n, c, t in top:
        print(f"    {t:9.1f} ms {c:7d}  {n[:70]}")
    dec, starts = decode_report(db, d0)
    for k, v in dec.items():
        print(f"  decode {k}: {v}")
    if rest:
        pc = connect(rest[0])
        off, rx, n = pcie_rx(db, pc, p0, d0)
        print(f"  PCIe RX over prefill: mean {rx:.1f}% ({n} samples; offset {off / 1e6:.1f} ms)")
        off, rx, n = pcie_rx(db, pc, starts[15], starts[-1])
        print(f"  PCIe RX over decode steps 15+: mean {rx:.1f}% ({n} samples)")
