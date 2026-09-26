"""Every prefill fill wait of a node-mode trace, by the chunk and copy it precedes, with rows and GPU idle.

Usage: fill_waits.py MAIN_SQLITE  (a node-mode trace; run where the report lives; bound memory).

idle_by_host_state.py finds fill waits only before a chunk's first plain _gather_host_rows_kernel, so with the split
gather it misses the waits before indexed copies (_gather_host_rows_to_kernel). This script takes both kernels:
- A copy: six launches of one gather kernel with one grid (gridX = rows), one per cached tensor; the indexed kernel's
  launches are interleaved with their slot-index kernels.
- A chunk: consecutive copies with nothing between them on the GPU but index and elementwise kernels; it is a layer's
  first chunk when a deepseek_rope_kernel ran since the previous chunk.
- A wait: a >=1 ms CUDA-free stretch on the launching thread ending within 1 ms of a copy's first launch. Its GPU idle
  is the part of it with no kernel or memcpy running.
Waits are grouped by (first/later chunk) x (plain copy / indexed copy that opens its chunk / indexed copy after one
of its chunk's copies).
"""

import bisect
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/pcie-trace")
from compare_arms import connect, windows  # noqa: E402
from idle_by_host_state import merged  # noqa: E402

GATHERS = {"_gather_host_rows_kernel": "plain", "_gather_host_rows_to_kernel": "indexed"}
WITHIN_CHUNK = {"index_elementwise_kernel", "unrolled_elementwise_kernel", "vectorized_elementwise_kernel"}


def main(path):
    db = connect(path)
    p0, d0 = windows(db)
    names = dict(db.execute("select id, value from StringIds"))
    copies, chunk, first, since_copy = [], -1, True, set()
    for corr, sid, gx, ks, ke in db.execute(
        "select correlationId, shortName, gridX, start, end from CUPTI_ACTIVITY_KIND_KERNEL "
        "where start>=? and start<? order by start",
        (p0, d0),
    ):
        name = names[sid]
        if name not in GATHERS:
            since_copy.add(name)
            if name == "deepseek_rope_kernel":
                first = True
            continue
        kind = GATHERS[name]
        last = copies[-1] if copies else None
        if last and not since_copy - WITHIN_CHUNK and last["kind"] == kind and last["rows"] == gx and last["n"] < 6:
            last["n"] += 1
            last["corrs"].append(corr)
            continue
        new_chunk = last is None or bool(since_copy - WITHIN_CHUNK)
        if new_chunk:
            chunk += 1
        copies.append(dict(kind=kind, rows=gx, n=1, corrs=[corr], chunk=chunk, opens=new_chunk,
                           first=first if new_chunk else copies[-1]["first"]))
        if new_chunk:
            first = False
        since_copy = set()
    host = {}
    tid = None
    wanted = {c["corrs"][0]: i for i, c in enumerate(copies)}
    for corr, start, gtid in db.execute(
        "select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?",
        (p0 - 10**9, d0),
    ):
        if corr in wanted:
            host[wanted[corr]] = start
            tid = gtid
    order = sorted(host, key=host.get)
    host_starts = [host[i] for i in order]
    calls = db.execute(
        "select start, end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<? order by start",
        (tid, p0, d0),
    ).fetchall()
    busy = merged(
        [s, e]
        for s, e in db.execute(
            "select start, end from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ? "
            "union all select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where end > ? and start < ?",
            (p0, d0, p0, d0),
        )
    )
    idle, cursor = [], p0
    for s, e in busy:
        if s > cursor:
            idle.append((cursor, min(s, d0)))
        cursor = max(cursor, e)
    idle_starts = [a for a, _ in idle]

    def gpu_idle(a, b):
        total, i = 0, max(0, bisect.bisect_right(idle_starts, a) - 1)
        while i < len(idle) and idle[i][0] < b:
            total += max(0, min(b, idle[i][1]) - max(a, idle[i][0]))
            i += 1
        return total

    waits = []
    last_end = calls[0][1]
    for s, e in calls[1:]:
        if s - last_end >= 1_000_000:
            j = bisect.bisect_left(host_starts, s)
            if j < len(order) and host_starts[j] - s <= 1_000_000:
                c = copies[order[j]]
                role = "plain" if c["kind"] == "plain" else ("indexed, opens chunk" if c["opens"] else "indexed, mid-chunk")
                waits.append(((last_end - p0) / 1e6, (s - last_end) / 1e6, gpu_idle(last_end, s) / 1e6, role, c["rows"],
                              c["first"], c["chunk"]))
        last_end = max(last_end, e)
    idle_ms = sum(b - a for a, b in idle) / 1e6
    print(f"prefill {(d0 - p0) / 1e6:.0f} ms; GPU idle {idle_ms:.0f} ms; {chunk + 1} chunks, {len(copies)} copies "
          f"({sum(c['kind'] == 'plain' for c in copies)} plain), {sum(c['first'] and c['opens'] for c in copies)} "
          f"first-of-layer chunks")
    for label, is_first in (("first chunk", True), ("later chunk", False)):
        for role in ("plain", "indexed, opens chunk", "indexed, mid-chunk"):
            ws = [w for w in waits if w[5] == is_first and w[3] == role]
            if ws:
                print(f"  {label}, before {role:20s}: n={len(ws):3d} ({len({w[6] for w in ws})} chunks) "
                      f"wait {sum(w[1] for w in ws):6.0f} ms, GPU idle {sum(w[2] for w in ws):6.0f} ms, "
                      f"median wait {statistics.median(w[1] for w in ws):5.1f} ms")
        ws = [w for w in waits if w[5] == is_first]
        print(f"  {label}, all waits: GPU idle {sum(w[2] for w in ws):.0f} ms")
    print("\nper wait: t_ms wait_ms gpu_idle_ms role rows_copied_after first_of_layer chunk")
    for w in waits:
        print(f"{w[0]:8.1f} {w[1]:6.1f} {w[2]:6.1f} {w[3]:20s} {w[4]:3d} {w[5]!s:5s} {w[6]}")


if __name__ == "__main__":
    main(sys.argv[1])
