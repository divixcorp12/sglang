#!/usr/bin/env python3
"""Task 4 report: did in-graph decode actually read the mirrors?

Reads each arm's session JSON and its expert stream trace. The decisive evidence
is the RAM-miss service's own per-drive accounting, carried on every
``ram_miss_request`` trace line as ``drives: [{dev, bytes, extents}]``. That
attributes expert bytes from inside the service, which aggregate /proc/diskstats
cannot do: diskstats sees every read on the device, whoever caused it.

Device ids are ``st_dev`` of the file's filesystem, so they are resolved here
against the mount points rather than assumed.
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

MOUNTS = {"nvme0": "/mnt/nvme0", "nvme2": "/mnt/nvme2", "nvme4": "/mnt/nvme4"}
GIB = 1073741824


def dev_names():
    out = {}
    for name, path in MOUNTS.items():
        try:
            out[os.stat(path).st_dev] = name
        except OSError:
            pass
    return out


def load_sessions(path):
    return json.load(open(path))["per_session"]


def mean_tps(rows):
    v = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    return sum(v) / len(v) if v else 0.0


def read_requests(path):
    """ram_miss_request lines only; forward-call lines are a different shape."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("kind") == "ram_miss_request":
                out.append(d)
    return out


def summarize(name, path, names):
    reqs = read_requests(path)
    print(f"--- {name}: {len(reqs)} RAM-miss requests")
    if not reqs:
        print("    no ram_miss_request lines; was SGLANG_DSV41_EXPERT_TRACE_PATH set?")
        return None

    per_drive_bytes = defaultdict(int)
    per_drive_extents = defaultdict(int)
    total_bytes = total_extents = 0
    by_type = defaultdict(int)
    failed = 0

    for r in reqs:
        by_type[r["request"]["type"]] += 1
        if not r["request"]["ok"]:
            failed += 1
        total_bytes += r.get("bytes", 0)
        total_extents += r.get("extents", 0)
        for d in r.get("drives", []):
            dev = d.get("dev")
            per_drive_bytes[dev] += d.get("bytes", 0)
            per_drive_extents[dev] += d.get("extents", 0)

    print(f"    request types: {dict(by_type)}   failed: {failed}")
    print(f"    total expert bytes: {total_bytes / GIB:.2f} GiB in {total_extents} extents")
    for dev in sorted(per_drive_bytes, key=lambda d: -per_drive_bytes[d]):
        label = names.get(dev, f"dev={dev}")
        b = per_drive_bytes[dev]
        share = 100.0 * b / total_bytes if total_bytes else 0.0
        print(
            f"      {label:<10} {b / GIB:8.2f} GiB  {share:5.1f}%  "
            f"{per_drive_extents[dev]} extents"
        )

    waits = [r["spans_ns"]["submit_to_first_cqe"] for r in reqs if r["spans_ns"]["submit_to_first_cqe"]]
    spans = [r["spans_ns"]["first_to_last_cqe"] for r in reqs if r["spans_ns"]["first_to_last_cqe"]]
    packs = [r["spans_ns"]["pack"] for r in reqs if r["spans_ns"]["pack"]]
    for label, vals in (("submit->first cqe", waits), ("first->last cqe", spans), ("pack", packs)):
        if vals:
            print(
                f"    {label:<18} n={len(vals):5d} mean={st.mean(vals)/1e6:7.3f} ms "
                f"p50={st.median(vals)/1e6:7.3f} ms"
            )
    return {"bytes": total_bytes, "per_drive": dict(per_drive_bytes), "names": names}


def main():
    b_json, b_tr, m_json, m_tr = sys.argv[1:5]
    names = dev_names()
    print(f"device map (st_dev -> mount): { {k: v for k, v in names.items()} }")
    print()

    base, mirror = load_sessions(b_json), load_sessions(m_json)
    tb, tm = mean_tps(base), mean_tps(mirror)
    print("=" * 66)
    print("DECODE THROUGHPUT (graphs on, GRAPH_GATHER=1)")
    print("=" * 66)
    print(f"base   mean decode tok/s = {tb:.4f}")
    print(f"mirror mean decode tok/s = {tm:.4f}")
    if tb:
        ratio = tm / tb
        print(f"ratio  = {ratio:.4f}x")
        print(f"predicted ceiling was 1.38x (~3.90 tok/s from 2.823); recorded before measuring.")
        if ratio < 1.1:
            print("  -> BELOW 1.1x: the NVMe share of the step is not what 18.2 attributes,")
            print("     or within-row splitting loses at production depth (handoff 3D).")
    print()
    print("=" * 66)
    print("SERVICE-ATTRIBUTED EXPERT BYTES (the gate)")
    print("=" * 66)
    b = summarize("base", b_tr, names)
    print()
    m = summarize("mirror", m_tr, names)

    if b and m:
        print()
        src = next((d for d, n in names.items() if n == "nvme2"), None)
        src_bytes = m["per_drive"].get(src, 0)
        share = 100.0 * src_bytes / m["bytes"] if m["bytes"] else 0.0
        print(f"GATE: nvme2 served {src_bytes / GIB:.2f} GiB of the mirror arm's expert bytes ({share:.2f}%)")
        print("      PASS if approximately zero. The layout is still built from nvme2,")
        print("      so a small residual is expected; a large one means a path still")
        print("      reads the source.")
        if m["bytes"] and b["bytes"]:
            print(
                f"BYTE PARITY: mirror {m['bytes'] / GIB:.2f} GiB vs base {b['bytes'] / GIB:.2f} GiB "
                f"({100.0 * m['bytes'] / b['bytes']:.1f}%) -- want ~100%, as prefill showed in 19."
            )


if __name__ == "__main__":
    main()
