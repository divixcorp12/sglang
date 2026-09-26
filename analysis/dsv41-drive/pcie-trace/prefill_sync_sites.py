"""Attribute the last timed session's prefill host syncs to their call sites, from a node-mode trace.

Usage: prefill_sync_sites.py MAIN_SQLITE  (run where the report lives; bound memory).

Each device-to-host copy is keyed by the two kernels that ran just before it on its stream (a nonzero kernel means
torch.where / boolean indexing, a reduce means .item(), unique's sort kernels mean torch.unique), with the host time
its runtime call blocked. Explicit synchronize calls and the prefill's kernel mix are listed after.
"""

import bisect
import collections
import sqlite3
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_arms import connect, windows  # noqa: E402


def short(name, width=48):
    return name if len(name) <= width else name[: width - 3] + "..."


def main(path):
    db = connect(path)
    p0, d0 = windows(db)
    print(f"prefill {p0 / 1e9:.3f}-{d0 / 1e9:.3f} s ({(d0 - p0) / 1e6:.0f} ms)")
    names = dict(db.execute("select id, value from StringIds"))
    kernels = collections.defaultdict(list)
    kernel_mix = collections.Counter()
    kernel_ms = collections.Counter()
    for start, end, stream, name in db.execute(
        "select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ? "
        "order by start",
        (p0, d0),
    ):
        kernels[stream].append((start, names[name]))
        kernel_mix[names[name]] += 1
        kernel_ms[names[name]] += (end - start) / 1e6
    starts = {stream: [s for s, _ in rows] for stream, rows in kernels.items()}

    blocked = dict(
        db.execute(
            "select correlationId, end - start from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?",
            (p0 - 1_000_000_000, d0),
        )
    )
    sites = collections.defaultdict(lambda: [0, 0, 0])
    sizes = collections.Counter()
    for start, stream, nbytes, corr in db.execute(
        "select start, streamId, bytes, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY "
        "where copyKind = 2 and start >= ? and start < ?",
        (p0, d0),
    ):
        rows = kernels.get(stream, [])
        i = bisect.bisect_left(starts.get(stream, []), start)
        prev = tuple(short(rows[j][1]) for j in (i - 2, i - 1) if j >= 0)
        size = "<=8B" if nbytes <= 8 else "<=256B" if nbytes <= 256 else "<=64KB" if nbytes <= 65536 else ">64KB"
        entry = sites[(prev, size)]
        entry[0] += 1
        entry[1] += blocked.get(corr, 0)
        entry[2] += nbytes
        sizes[size] += 1
    total = sum(e[0] for e in sites.values())
    total_ms = sum(e[1] for e in sites.values()) / 1e6
    print(f"\nD2H copies: {total}, host blocked {total_ms:.0f} ms; by size {dict(sizes)}")
    print(f"{'count':>6} {'host ms':>8} {'size':>7}  preceding kernels (older, newer)")
    for (prev, size), (count, ns, _) in sorted(sites.items(), key=lambda kv: -kv[1][1])[:30]:
        print(f"{count:6d} {ns / 1e6:8.0f} {size:>7}  {' | '.join(prev)}")

    print("\nsynchronize calls:")
    for name, count, ns in db.execute(
        "select s.value, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join StringIds s on s.id = r.nameId where r.start >= ? and r.start < ? and s.value like '%ynchronize%' "
        "group by s.value",
        (p0, d0),
    ):
        print(f"  {count:6d} {ns / 1e6:8.0f} ms  {name}")

    print(f"\nkernels: {sum(kernel_mix.values())}; top 30 by count (count, GPU ms):")
    for name, count in kernel_mix.most_common(30):
        print(f"  {count:7d} {kernel_ms[name]:9.1f}  {short(name, 80)}")


if __name__ == "__main__":
    main(sys.argv[1])
