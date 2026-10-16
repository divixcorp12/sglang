"""Pre-launch step for a served DSV4.1 capture: drop leftover file cache of the draft, row-image and Engram files (no root:
posix_fadvise DONTNEED per file), log per-node FilePages/MemFree, and refuse to start while node 1 cannot hold the run.

Leftover cache from the previous server on the box pushed node 1 below its low watermark in
cores-ab-20261014/splitmodel-stream-mem-decay995 (it started with 23 GiB of file cache there).

Node-1 floor = HiCache host pools on node 1 (--hicache-gib; 0 when they stay on node 0) + the rest of the run's node-1
need + --margin-gib. The rest comes from the arm's previous memsample (--previous, memsample.py's JSONL) where one
exists: node-1 MemFree at its first sample less its minimum, less the pools it bound there (its peak node-1 Shmem
rise). Otherwise it is DEFAULT_REST_GIB, from arm A (splitmodel-stream-mem-hicache1, 2026-10-15: 64.69 GiB free at
start, 1.70 GiB at its lowest, 11.75 GiB of it HiCache). The record says which was used.

Production (launch_prod.sh) runs it too, with three more options:
- --skip-evict-port PORT: while anything listens on PORT (another of our servers reading the same files), check the
  floor without evicting;
- --node0-warn-gib: warn, never refuse, when node 0 starts below it;
- --holders ROOT: on a refusal, name the directories under each ROOT holding the most page cache (whole box: which
  node a cached page sits on needs root), within --holders-budget-s.
--dry-run evicts nothing and does not wait: it reports the floor, the nodes and the decision it would make.

Usage: prelaunch_evict.py --log OUT.json [--previous RUN.memsample.jsonl] [--hicache-gib 11.75] [--margin-gib 1.0]
                           [--wait-min 30] [--skip-evict-port P] [--node0-warn-gib G] [--holders ROOT ...] [--dry-run]
Exit 0: ready; 3: node 1 stayed below the floor for --wait-min minutes (evicting again each minute).
PRELAUNCH_NODE_SYSFS and PRELAUNCH_EVICT_PATHS (colon-separated) replace the sysfs node directory and the evicted
paths, for tests.
"""
import argparse
import ctypes
import glob
import json
import mmap
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
if os.environ.get("PRELAUNCH_EVICT_PATHS"):
    PATHS = tuple(os.environ["PRELAUNCH_EVICT_PATHS"].split(":"))
NODE_SYSFS = os.environ.get("PRELAUNCH_NODE_SYSFS", "/sys/devices/system/node")
DEFAULT_HICACHE_GIB = 11.75
DEFAULT_REST_GIB = 64.69 - 1.70 - DEFAULT_HICACHE_GIB  # 51.24
GIB_KB = 2**20
WINDOW = 1 << 30  # mincore window: a 1 GiB map at a time keeps the residency vector at 256 KiB


def node_meminfo() -> dict:
    """Per node MemFree, FilePages and Shmem in GiB."""
    out = {}
    for n in (0, 1):
        d = {}
        for line in open(f"{NODE_SYSFS}/node{n}/meminfo"):
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


def listening(port: int, proc_net: str = "/proc/net") -> bool:
    """Whether a TCP socket (v4 or v6) listens on port, from /proc/net/tcp{,6} (state 0A)."""
    for name in ("tcp", "tcp6"):
        try:
            with open(os.path.join(proc_net, name)) as f:
                next(f, None)
                for line in f:
                    fields = line.split()
                    if len(fields) > 3 and fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                        return True
        except OSError:
            continue
    return False


_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
_libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
_libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
_MAP_FAILED = ctypes.c_void_p(-1).value
_RESIDENT_BIT = bytes(b & 1 for b in range(256))  # mincore's vector: bit 0 is residency, the rest are reserved


def cached_bytes(path: str) -> int:
    """Bytes of path in the page cache: mincore over read-only shared maps of it, a window at a time. Mapping reads
    nothing, so the scan itself caches nothing."""
    page = mmap.PAGESIZE
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        total = 0
        for offset in range(0, size, WINDOW):
            length = min(WINDOW, size - offset)
            addr = _libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, offset)
            if addr == _MAP_FAILED:
                raise OSError(ctypes.get_errno(), "mmap")
            try:
                vec = (ctypes.c_ubyte * ((length + page - 1) // page))()
                if _libc.mincore(addr, length, vec) != 0:
                    raise OSError(ctypes.get_errno(), "mincore")
                total += bytes(vec).translate(_RESIDENT_BIT).count(1) * page
            finally:
                _libc.munmap(addr, length)
        return total
    finally:
        os.close(fd)


def page_cache_holders(roots, budget_s: float, top: int = 10) -> dict:
    """The directories one level under each root holding the most page cache, largest first, scanned within budget_s.
    Whole box: without root a cached page's node is not visible."""
    deadline = time.monotonic() + budget_s
    sums: dict[str, int] = {}
    complete = True
    for root in roots:
        try:
            entries = sorted(os.scandir(root), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            total = 0
            walk = [entry.path] if entry.is_dir(follow_symlinks=False) else []
            if entry.is_file(follow_symlinks=False):
                total += _cached_or_zero(entry.path)
            for top_dir in walk:
                for dirpath, _, files in os.walk(top_dir):
                    for name in files:
                        if time.monotonic() > deadline:
                            complete = False
                            break
                        total += _cached_or_zero(os.path.join(dirpath, name))
                    if not complete:
                        break
            if total:
                sums[entry.path] = total
            if not complete:
                break
        if not complete:
            break
    ranked = sorted(sums.items(), key=lambda kv: -kv[1])[:top]
    return {"complete": complete, "holders": [{"path": p, "gib": round(b / 2**30, 2)} for p, b in ranked]}


def _cached_or_zero(path: str) -> int:
    try:
        if os.path.islink(path) or not os.path.isfile(path) or os.path.getsize(path) == 0:
            return 0
        return cached_bytes(path)
    except (OSError, ValueError):
        return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True)
    p.add_argument("--previous", default="", help="the arm's previous memsample.py JSONL, if any")
    p.add_argument("--hicache-gib", type=float, default=DEFAULT_HICACHE_GIB)
    p.add_argument("--margin-gib", type=float, default=1.0)
    p.add_argument("--wait-min", type=float, default=30.0)
    p.add_argument("--skip-evict-port", type=int, action="append", default=[])
    p.add_argument("--node0-warn-gib", type=float, default=0.0)
    p.add_argument("--holders", action="append", default=[], help="a root whose page-cache holders a refusal names")
    p.add_argument("--holders-budget-s", type=float, default=60.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    floor = node1_floor(a.previous, a.hicache_gib, a.margin_gib)
    record = {"t": time.time(), **floor, "before": node_meminfo(), "rounds": [], "dry_run": a.dry_run}
    print(f"prelaunch: node-1 floor {floor['floor_gib']} GiB = HiCache {a.hicache_gib} + rest {floor['rest_gib']} "
          f"({floor['rest_source']}) + margin {a.margin_gib}", flush=True)
    busy = [port for port in a.skip_evict_port if listening(port)]
    record["evict_skipped_for_ports"] = busy
    if busy:
        print(f"prelaunch: port(s) {busy} in use by another server reading the same files; checking without evicting",
              flush=True)
    deadline = time.time() + 60 * a.wait_min
    while True:
        result = {"files": 0, "bytes": 0, "failed": 0} if (a.dry_run or busy) else evict()
        after = node_meminfo()
        record["rounds"].append({"t": time.time(), "evicted": result, "after": after})
        print(f"prelaunch: evicted {result['files']} files ({result['bytes'] / 2**30:.0f} GiB scanned, "
              f"{result['failed']} failed); node 0 free {after['0']['MemFree']} File {after['0']['FilePages']} GiB, "
              f"node 1 free {after['1']['MemFree']} File {after['1']['FilePages']} GiB", flush=True)
        if after["1"]["MemFree"] >= floor["floor_gib"]:
            record["ready"] = True
            break
        if a.dry_run or time.time() > deadline:
            record["ready"] = False
            break
        time.sleep(60)
    after = record["rounds"][-1]["after"]
    if a.node0_warn_gib and after["0"]["MemFree"] < a.node0_warn_gib:
        record["node0_warning"] = True
        print(f"prelaunch: WARNING node 0 has {after['0']['MemFree']} GiB free, below the {a.node0_warn_gib} GiB every "
              f"served arm started with; the load may swap", file=sys.stderr, flush=True)
    if not record["ready"]:
        holders = page_cache_holders(a.holders, a.holders_budget_s) if a.holders else {"complete": True, "holders": []}
        record["page_cache_holders"] = holders
        verb = "would refuse" if a.dry_run else "refusing to start"
        lines = [
            f"prelaunch: {verb}: node 1 has {after['1']['MemFree']} GiB free (file cache {after['1']['FilePages']} GiB),"
            f" below its floor of {floor['floor_gib']} GiB; node 0 has {after['0']['MemFree']} GiB free (file cache "
            f"{after['0']['FilePages']} GiB)."
        ]
        if holders["holders"]:
            lines.append("prelaunch: largest page-cache holders (whole box; per-node attribution needs root)"
                         + ("" if holders["complete"] else ", scan cut short by its time budget") + ":")
            lines += [f"prelaunch:   {h['gib']:8.2f} GiB  {h['path']}" for h in holders["holders"]]
        lines.append("prelaunch: free node 1 (as root: sync; echo 1 > /proc/sys/vm/drop_caches; or stop the co-tenant "
                     "filling it), then start again.")
        print("\n".join(lines), file=sys.stderr, flush=True)
    elif a.dry_run:
        print("prelaunch: would start: node 1 is above its floor", flush=True)
    with open(a.log, "w") as f:
        json.dump(record, f, indent=1)
    return 0 if record["ready"] else 3


if __name__ == "__main__":
    sys.exit(main())
