"""Pre-launch step for a served DSV4.1 capture: drop leftover file cache of the draft, row-image and Engram files (no root:
posix_fadvise DONTNEED per file), log per-node FilePages/MemFree, and refuse to start while node 1 cannot hold the run.

Leftover cache from the previous server on the box pushed node 1 below its low watermark in
cores-ab-20261014/splitmodel-stream-mem-decay995 (it started with 23 GiB of file cache there).

Node-1 floor = HiCache host pools on node 1 (--hicache-gib; 0 when they stay on node 0) + the rest of the run's node-1
need + --margin-gib. The rest comes from the arm's previous memsample (--previous, memsample.py's JSONL) where one
exists: node-1 MemFree at its first sample less its minimum, less the pools it bound there (its peak node-1 Shmem
rise). Otherwise it is DEFAULT_REST_GIB, from arm A (splitmodel-stream-mem-hicache1, 2026-10-15: 64.69 GiB free at
start, 1.70 GiB at its lowest, 11.75 GiB of it HiCache). The record says which was used.

Usage: prelaunch_evict.py --log OUT.json [--previous RUN.memsample.jsonl] [--hicache-gib 11.75] [--margin-gib 1.0]
                           [--wait-min 30]
Exit 0: ready; 3: node 1 stayed below the floor for --wait-min minutes (evicting again each minute).
"""
import argparse
import glob
import json
import os
import sys
import time

# Not the target model's shards (dsv41-full40): their ~9 GiB of non-expert weights read cold from the SATA root volume
# at ~55 MB/s, ~150 s a launch (cores-ab-20261014/splitmodel-stream-mem-tiebreak), and the loader drops them after load.
PATHS = (
    "/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-dspark-draft",  # DSpark draft
    "/mnt/nvme0/dsv41_flash/exl3_row_images",  # row images, three mirrors
    "/mnt/nvme4/dsv41_flash/exl3_row_images",
    "/mnt/nvme2/dsv41_flash/exl3_row_images",
    "/mnt/nvme2/DeepSeek-V4.1-Flash",  # Engram tables (SGLANG_DSV41_ENGRAM_TABLE_DIR)
)
DEFAULT_HICACHE_GIB = 11.75
DEFAULT_REST_GIB = 64.69 - 1.70 - DEFAULT_HICACHE_GIB  # 51.24
GIB_KB = 2**20


def node_meminfo() -> dict:
    """Per node MemFree, FilePages and Shmem in GiB."""
    out = {}
    for n in (0, 1):
        d = {}
        for line in open(f"/sys/devices/system/node/node{n}/meminfo"):
            parts = line.split()
            if parts[2].rstrip(":") in ("MemFree", "FilePages", "Shmem"):
                d[parts[2].rstrip(":")] = round(int(parts[3]) / GIB_KB, 2)
        out[str(n)] = d
    return out


def rest_from_samples(lines) -> tuple[float, float]:
    """(node-1 need less its HiCache pools, the pools) in GiB from memsample.py JSONL lines."""
    samples = [json.loads(line) for line in lines if '"nodes"' in line]
    if len(samples) < 2:
        raise ValueError("fewer than two memory samples")
    free = [s["nodes"]["1"]["MemFree"] for s in samples]
    shmem = [s["nodes"]["1"].get("Shmem", 0) for s in samples]
    need = (free[0] - min(free)) / GIB_KB
    pools = max(0, max(shmem) - shmem[0]) / GIB_KB
    return need - pools, pools


def node1_floor(previous: str, hicache_gib: float, margin_gib: float) -> dict:
    """The floor and where its non-HiCache part came from."""
    rest, source = DEFAULT_REST_GIB, "default (arm A, 2026-10-15)"
    if previous and os.path.exists(previous):
        try:
            with open(previous) as f:
                rest, _ = rest_from_samples(f)
            source = previous
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            source = f"default (arm A); {previous} unusable: {error}"
    return {"floor_gib": round(hicache_gib + rest + margin_gib, 2), "rest_gib": round(rest, 2),
            "hicache_gib": hicache_gib, "margin_gib": margin_gib, "rest_source": source}


def evict(paths=PATHS) -> dict:
    files = scanned = failed = 0
    for root in paths:
        for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                failed += 1
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                files += 1
                scanned += os.fstat(fd).st_size
            except OSError:
                failed += 1
            finally:
                os.close(fd)
    return {"files": files, "bytes": scanned, "failed": failed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True)
    p.add_argument("--previous", default="", help="the arm's previous memsample.py JSONL, if any")
    p.add_argument("--hicache-gib", type=float, default=DEFAULT_HICACHE_GIB)
    p.add_argument("--margin-gib", type=float, default=1.0)
    p.add_argument("--wait-min", type=float, default=30.0)
    a = p.parse_args()
    floor = node1_floor(a.previous, a.hicache_gib, a.margin_gib)
    record = {"t": time.time(), **floor, "before": node_meminfo(), "rounds": []}
    print(f"prelaunch: node-1 floor {floor['floor_gib']} GiB = HiCache {a.hicache_gib} + rest {floor['rest_gib']} "
          f"({floor['rest_source']}) + margin {a.margin_gib}", flush=True)
    deadline = time.time() + 60 * a.wait_min
    while True:
        result = evict()
        after = node_meminfo()
        record["rounds"].append({"t": time.time(), "evicted": result, "after": after})
        print(f"prelaunch: evicted {result['files']} files ({result['bytes'] / 2**30:.0f} GiB scanned, "
              f"{result['failed']} failed); node 0 free {after['0']['MemFree']} File {after['0']['FilePages']} GiB, "
              f"node 1 free {after['1']['MemFree']} File {after['1']['FilePages']} GiB", flush=True)
        if after["1"]["MemFree"] >= floor["floor_gib"]:
            record["ready"] = True
            break
        if time.time() > deadline:
            record["ready"] = False
            break
        time.sleep(60)
    with open(a.log, "w") as f:
        json.dump(record, f, indent=1)
    return 0 if record["ready"] else 3


if __name__ == "__main__":
    sys.exit(main())
