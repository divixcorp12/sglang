"""Per gathered chunk of the last timed session's prefill: host time blocked on the gather, then host time to the
next chunk's gather launch, split by whether that chunk is in the same layer (no deepseek_rope_kernel between).

Usage: chunk_host_time.py MAIN_SQLITE  (run where the report lives; bound memory).
"""

import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_arms import connect, windows  # noqa: E402

BLOCKED_NS = 5_000_000  # a readback longer than this waited for a gather; the rest are sub-millisecond


def main(path):
    db = connect(path)
    p0, d0 = windows(db)
    names = dict(db.execute("select id, value from StringIds"))
    runtime = {
        corr: (start, end)
        for corr, start, end in db.execute(
            "select correlationId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?",
            (p0 - 10**9, d0),
        )
    }
    kernels = db.execute(
        "select shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ? "
        "order by start",
        (p0, d0),
    ).fetchall()
    chunks, previous = [], None
    for name_id, corr in kernels:
        name = names[name_id]
        if name == "_gather_host_rows_kernel" and previous != "_gather_host_rows_kernel":
            chunks.append(runtime[corr][0])
        previous = name
    rope = sorted(runtime[corr][0] for name_id, corr in kernels if names[name_id] == "deepseek_rope_kernel")
    blocks = sorted(
        runtime[corr]
        for (corr,) in db.execute(
            "select correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 and start >= ? and start < ?",
            (p0, d0),
        )
        if runtime[corr][1] - runtime[corr][0] > BLOCKED_NS
    )
    blocked, within, across = [], [], []
    for a, b in zip(chunks, chunks[1:]):
        waits = [w for w in blocks if a <= w[0] < b]
        if not waits:
            continue
        blocked.append(waits[0][1] - waits[0][0])
        after = b - waits[0][1]
        (across if any(waits[0][1] <= r < b for r in rope) else within).append(after)
    print(f"prefill {(d0 - p0) / 1e6:.0f} ms; chunks {len(chunks)}")
    if blocked:
        print(f"blocked-on-gather waits {len(blocked)}: total {sum(blocked) / 1e6:.0f} ms, "
              f"median {statistics.median(blocked) / 1e6:.1f} ms")
    else:
        print("blocked-on-gather waits: none")
    for label, xs in (("next chunk same layer", within), ("next chunk next layer", across)):
        if xs:
            print(f"{label}: n={len(xs)}, host after wait {sum(xs) / 1e6:.0f} ms, "
                  f"median {statistics.median(xs) / 1e6:.1f} ms, p90 {sorted(xs)[int(len(xs) * 0.9)] / 1e6:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1])
