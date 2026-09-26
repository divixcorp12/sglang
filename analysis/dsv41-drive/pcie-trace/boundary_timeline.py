"""The host timeline across a few prefill layer boundaries of a node-mode trace, for finding where the GPU idles.

Usage: boundary_timeline.py MAIN_SQLITE [BOUNDARY ...]  (default: boundaries 5, 20, 35; run where the report lives).

A boundary runs from the GPU end of a layer's last _gather_host_rows_kernel to the GPU start of the next layer's first
one. Printed per boundary, in host time order on the launching thread: runtime calls longer than 0.2 ms (with their
kernel or copy), host gaps longer than 0.3 ms with no runtime call (Python work, or a host wait with no CUDA call such
as an NVMe fill wait), and the GPU idle periods longer than 0.3 ms.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_arms import connect, windows  # noqa: E402

MIN_CALL_NS = 200_000
MIN_GAP_NS = 300_000


def gather_groups(db, names, p0, d0):
    """[(first gather start, last gather end, first gather correlationId)] per chunk, in GPU order."""
    groups, previous = [], None
    for start, end, name_id, corr in db.execute(
        "select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
        "where start >= ? and start < ? order by start",
        (p0, d0),
    ):
        name = names[name_id]
        if name == "_gather_host_rows_kernel":
            if previous == "_gather_host_rows_kernel":
                groups[-1][1] = end
            else:
                groups.append([start, end, corr])
        previous = name
    return groups


def layer_starts(db, names, p0, d0):
    return [
        start
        for (start, name_id) in db.execute(
            "select start, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ? order by start",
            (p0, d0),
        )
        if names[name_id] == "deepseek_rope_kernel"
    ]


def boundaries(db, names, p0, d0):
    """[(last gather group of layer L, first gather group of layer L+1)] where a rope kernel lies between."""
    groups = gather_groups(db, names, p0, d0)
    ropes = layer_starts(db, names, p0, d0)
    out = []
    for a, b in zip(groups, groups[1:]):
        if any(a[1] <= r < b[0] for r in ropes):
            out.append((a, b))
    return out


def main(path, picks):
    db = connect(path)
    p0, d0 = windows(db)
    names = dict(db.execute("select id, value from StringIds"))
    bounds = boundaries(db, names, p0, d0)
    print(f"prefill {(d0 - p0) / 1e6:.0f} ms; {len(bounds)} layer boundaries")
    kernel_of = {}
    for corr, name_id in db.execute(
        "select correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ?", (p0, d0)
    ):
        kernel_of[corr] = names[name_id]
    copy_of = {
        corr: (kind, nbytes)
        for corr, kind, nbytes in db.execute(
            "select correlationId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start >= ? and start < ?",
            (p0, d0),
        )
    }
    launch_host = dict(
        db.execute(
            "select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?",
            (p0 - 10**9, d0),
        )
    )
    for i in picks:
        (a0, a1, a_corr), (b0, b1, b_corr) = bounds[i]
        h0, h1 = launch_host[a_corr], launch_host[b_corr]
        tid = db.execute(
            "select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId = ?", (b_corr,)
        ).fetchone()[0]
        print(f"\n=== boundary {i}: GPU window {(b0 - a1) / 1e6:.1f} ms; host from last-chunk gather launch to next "
              f"layer's first gather launch {(h1 - h0) / 1e6:.1f} ms")
        events = []
        last_end = h0
        for start, end, name_id, corr in db.execute(
            "select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME "
            "where globalTid = ? and start >= ? and start <= ? order by start",
            (tid, h0, h1),
        ):
            if start - last_end > MIN_GAP_NS:
                events.append((last_end, "host", f"gap {(start - last_end) / 1e6:6.2f} ms, then {names[name_id][:40]}"))
            if end - start > MIN_CALL_NS:
                what = kernel_of.get(corr) or (
                    f"copy kind {copy_of[corr][0]} {copy_of[corr][1]} B" if corr in copy_of else ""
                )
                events.append((start, "host", f"call {(end - start) / 1e6:6.2f} ms {names[name_id][:34]} {what[:40]}"))
            last_end = max(last_end, end)
        busy = db.execute(
            "select start, end from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ? "
            "union all select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where end > ? and start < ? order by 1",
            (a1, b0, a1, b0),
        ).fetchall()
        cursor = a1
        for start, end in busy:
            if start - cursor > MIN_GAP_NS:
                events.append((cursor, "GPU ", f"idle {(start - cursor) / 1e6:6.2f} ms"))
            cursor = max(cursor, end)
        if b0 - cursor > MIN_GAP_NS:
            events.append((cursor, "GPU ", f"idle {(b0 - cursor) / 1e6:6.2f} ms"))
        for t, lane, text in sorted(events):
            print(f"  {(t - a1) / 1e6:8.2f} ms  {lane}  {text}")


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or [5, 20, 35])
