"""Time the expert-id round trip in `_prepare_promotion` from an existing Nsight Systems SQLite export.

The round trip is `torch.tensor(expert_rows, device=cuda)` followed by `ensure_rows`' `.tolist()`.
In a trace it is one small `cudaMemcpyAsync` host-to-device of 8*n bytes, a `cudaStreamSynchronize`,
a `cudaMemcpyAsync` device-to-host of the same size, and a second `cudaStreamSynchronize`, on the
scheduler thread, in the preparation window before each promotion chunk's six
`copy_expert_rows_gpu_kernel` launches.

Reads the report read-only; runs no GPU work. Run under `taskset -c 0-63` with threads capped:

    OMP_NUM_THREADS=4 taskset -c 0-63 python expert_id_roundtrip_probe.py prof-node.sqlite

A chunk's window is the time between the previous chunk's last copy kernel and this chunk's first
launch. Windows longer than `--max-window-ms` (the first chunk, and the demand-path stretches that
also contain host-to-device and device-to-host pairs of the same size) are skipped, because the
pairing rule cannot tell those pairs from this one.
"""

import argparse
import sqlite3
import statistics

CLUSTER_GAP_NS = 5_000_000  # copy kernels closer than this belong to one chunk
H2D, D2H = 1, 2  # CUPTI copyKind
INDEX_TABLE_BYTES = 3072  # the pageable slot-map refresh, same direction, different payload


def open_report(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def copy_kernel_clusters(conn, names):
    ids = [i for i, v in names.items() if "copy_expert_rows" in v]
    if not ids:
        raise SystemExit("no copy_expert_rows kernel in this report")
    marks = ",".join(map(str, ids))
    kernels = conn.execute(
        "select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
        f"where demangledName in ({marks}) or shortName in ({marks}) order by start"
    ).fetchall()
    clusters = []
    for start, end, _ in kernels:
        if clusters and start - clusters[-1][1] < CLUSTER_GAP_NS:
            clusters[-1][1] = max(clusters[-1][1], end)
            clusters[-1][2] += 1
        else:
            clusters.append([start, end, 1])
    thread = conn.execute(
        "select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=?", (kernels[0][2],)
    ).fetchone()[0]
    return clusters, thread


def round_trips(conn, names, clusters, thread, max_window_ns):
    memcpy = {
        r[0]: (r[1], r[2])
        for r in conn.execute("select correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")
    }
    found, previous_end = [], None
    for index, (start, end, _) in enumerate(clusters):
        window_start = previous_end if previous_end is not None else start - 500_000_000
        previous_end = end
        if start - window_start > max_window_ns:
            continue
        calls = []
        for begin, finish, name_id, correlation in conn.execute(
            "select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME "
            "where globalTid=? and start>=? and end<=? order by start",
            (thread, window_start, start),
        ):
            name = names.get(name_id, "?").split("_v")[0]
            if name.startswith(("cudaMemcpy", "cudaStreamSync")):
                calls.append((begin, finish, name, memcpy.get(correlation)))
        for j, (begin, _, name, meta) in enumerate(calls):
            if name != "cudaMemcpyAsync" or not meta or meta[1] != H2D:
                continue
            nbytes = meta[0]
            if nbytes % 8 or nbytes == INDEX_TABLE_BYTES:
                continue
            for k in range(j + 1, min(j + 4, len(calls))):
                _, finish2, name2, meta2 = calls[k]
                if name2 == "cudaMemcpyAsync" and meta2 and meta2[1] == D2H and meta2[0] == nbytes:
                    end_call = finish2
                    if k + 1 < len(calls) and calls[k + 1][2] == "cudaStreamSynchronize":
                        end_call = calls[k + 1][1]
                    found.append((index, nbytes // 8, (end_call - begin) / 1e3, (start - window_start) / 1e6))
                    break
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", help="an Nsight Systems .sqlite export")
    parser.add_argument("--max-window-ms", type=float, default=200.0)
    args = parser.parse_args()
    conn = open_report(args.report)
    names = {r[0]: r[1] for r in conn.execute("select id, value from StringIds")}
    clusters, thread = copy_kernel_clusters(conn, names)
    pairs = round_trips(conn, names, clusters, thread, int(args.max_window_ms * 1e6))
    if not pairs:
        raise SystemExit("no round-trip pair found in any preparation window")
    micros = sorted(p[2] for p in pairs)
    windows = [p[3] for p in pairs]
    per_window = {}
    for p in pairs:
        per_window[p[0]] = per_window.get(p[0], 0) + 1
    print(f"chunks (copy-kernel clusters): {len(clusters)}; windows kept: {len(per_window)}")
    print(f"pairs per kept window: {sorted(set(per_window.values()))}")
    print(f"ids per chunk: min {min(p[1] for p in pairs)} median {statistics.median(p[1] for p in pairs)} "
          f"max {max(p[1] for p in pairs)}")
    print(f"round trip us (H2D start to D2H sync end): median {statistics.median(micros):.1f} "
          f"mean {statistics.mean(micros):.1f} p90 {micros[int(0.9 * len(micros)) - 1]:.1f} max {micros[-1]:.1f}")
    print(f"total round trip ms: {sum(micros) / 1e3:.3f}")
    print(f"preparation window ms: median {statistics.median(windows):.1f}")


if __name__ == "__main__":
    main()
