"""Read-only node-memory sampler for one capture. Every INTERVAL s until the stop file appears, one JSON line: per-node
meminfo, Normal-zone free pages against its watermarks, /proc/vmstat swap/fault/reclaim counters, kswapd CPU time per
node, swap used, and for the scheduler process (the one whose threads include exl3-cpu-exp*) its VmLck and its CPU
expert team threads' fault counts. Once when the server reports ready and once at the end of decode (the verify log
stops growing, else at stop): `fincore` of the model's safetensors shards and the EXL3 row images, per file and in total.
Once in decode: `numastat -p` of the scheduler. Once a minute: the scheduler's /proc/<pid>/numa_maps grouped by mapping
and policy, pages per node. Every sample: /proc/diskstats of the devices under the model, draft and row-image paths.
Reads /proc and /sys and calls fincore/numastat; attaches to nothing. Summarised by memsum.py; prelaunch_evict.py reads
a previous run's samples for its node-1 floor.

Usage: memsample.py RUN_DIR OUT_JSONL STOP_FILE [INTERVAL=5]
"""
import glob
import json
import os
import subprocess
import sys
import time

run, out, stop = sys.argv[1:4]
interval = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
DECODE_LINES = 60
IDLE_S = 60
VMSTAT = ("pswpin", "pswpout", "pgmajfault", "pgscan_kswapd", "pgsteal_kswapd", "pgscan_direct", "pgsteal_direct",
          "allocstall_normal", "compact_stall")
MODEL = "/data/models/slang/nvfp4-work/cc-expert-prediction"
FILES = sorted(glob.glob(f"{MODEL}/dsv41-full40/*.safetensors") + glob.glob(f"{MODEL}/dsv41-dspark-draft/*.safetensors")
               + [p for d in ("nvme0", "nvme4", "nvme2") for p in glob.glob(f"/mnt/{d}/dsv41_flash/exl3_row_images/*.rows")])


def node_meminfo(n):
    d = {}
    for line in open(f"/sys/devices/system/node/node{n}/meminfo"):
        parts = line.split()
        if parts[2].rstrip(":") in ("MemFree", "FilePages", "AnonPages", "Shmem", "Mlocked", "Unevictable",
                                    "PageTables"):
            d[parts[2].rstrip(":")] = int(parts[3])
    return d


def zoneinfo():
    zones, cur = {}, None
    for line in open("/proc/zoneinfo"):
        s = line.split()
        if line.startswith("Node"):
            cur = (int(s[1].rstrip(",")), s[3])
            zones[cur] = {}
        elif cur and cur[1] == "Normal" and len(s) >= 2 and s[0] in ("min", "low", "high") and s[1].isdigit():
            zones[cur][s[0]] = int(s[1])
        elif cur and cur[1] == "Normal" and s[:2] == ["pages", "free"]:
            zones[cur]["free"] = int(s[2])
    return {str(n): z for (n, name), z in zones.items() if name == "Normal"}


def vmstat():
    return {k: int(v) for k, v in (line.split() for line in open("/proc/vmstat")) if k in VMSTAT}


KSWAPD = {}
for _p in glob.glob("/proc/[0-9]*/comm"):
    try:
        name = open(_p).read().strip()
        if name.startswith("kswapd"):
            KSWAPD[name] = _p.replace("comm", "stat")
    except OSError:
        pass


def kswapd():
    d = {}
    for name, path in KSWAPD.items():
        s = open(path).read().rsplit(")", 1)[1].split()
        d[name] = int(s[11]) + int(s[12])  # utime + stime, ticks
    return d


def _mount_devices():
    """The block devices under the model, draft and row-image paths (their mounts' sources, dm- names resolved)."""
    paths = (MODEL, "/mnt/nvme0/dsv41_flash", "/mnt/nvme4/dsv41_flash", "/mnt/nvme2/dsv41_flash")
    names = set()
    for p in paths:
        try:
            dev = os.stat(p).st_dev
            names.add(os.path.basename(os.path.realpath(f"/sys/dev/block/{os.major(dev)}:{os.minor(dev)}")))
        except OSError:
            pass
    return names


DEVICES = _mount_devices()


def diskstats():
    """Per device: reads, sectors read, ms reading, writes, sectors written, ms writing, in flight, ms doing I/O."""
    d = {}
    for line in open("/proc/diskstats"):
        s = line.split()
        if s[2] in DEVICES:
            d[s[2]] = [int(s[i]) for i in (3, 5, 6, 7, 9, 10, 11, 12)]
    return d


def swap_used_kb():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith(("SwapTotal", "SwapFree"))}
    return m["SwapTotal"] - m["SwapFree"]


def find_scheduler():
    for comm in glob.glob("/proc/[0-9]*/task/[0-9]*/comm"):
        try:
            if open(comm).read().startswith("exl3-cpu-exp"):
                return int(comm.split("/")[2])
        except OSError:
            pass
    return None


def scheduler_state(pid):
    try:
        # VmPTE/VmSize alongside VmLck: under mlockall, expandable_segments' CUDA VA reservations show up as locked VMAs.
        vm = {l.split(":")[0]: int(l.split()[1]) for l in open(f"/proc/{pid}/status")
              if l.split(":")[0] in ("VmLck", "VmPTE", "VmSize", "VmRSS")}
        lck = vm.get("VmLck")
        teams = {}
        for task in glob.glob(f"/proc/{pid}/task/*"):
            comm = open(f"{task}/comm").read().strip()
            if comm.startswith("exl3-cpu-exp"):
                s = open(f"{task}/stat").read().rsplit(")", 1)[1].split()
                t = teams.setdefault(comm, {"threads": 0, "minflt": 0, "majflt": 0})
                t["threads"] += 1
                t["minflt"] += int(s[7])
                t["majflt"] += int(s[9])
        return {"pid": pid, "vmlck_kb": lck, "vmpte_kb": vm.get("VmPTE"), "vmsize_kb": vm.get("VmSize"),
                "vmrss_kb": vm.get("VmRSS"), "teams": teams}
    except OSError:
        return None


def fincore(tag, f):
    r = subprocess.run(["fincore", "-b", "-r", "-n", "-o", "RES,SIZE,FILE", *FILES], capture_output=True, text=True)
    files = []
    for line in r.stdout.splitlines():
        res, size, name = line.split(maxsplit=2)
        files.append({"file": name, "cached": int(res), "size": int(size)})
    groups = {}
    for x in files:
        g = "row_images" if x["file"].endswith(".rows") else "draft" if "dspark-draft" in x["file"] else "model"
        groups[g] = groups.get(g, 0) + x["cached"]
    f.write(json.dumps({"t": time.time(), "fincore": tag, "total_cached": sum(x["cached"] for x in files),
                        "by_group": groups, "files": files,
                        "file_pages_kb": {str(n): node_meminfo(n)["FilePages"] for n in (0, 1)}}) + "\n")


def numa_maps(pid, f):
    """The scheduler's mappings grouped by name (file path, where shared anonymous memory shows as /dev/zero or
    SYSV; else heap / stack / anon) and policy, pages per node."""
    groups = {}
    try:
        for line in open(f"/proc/{pid}/numa_maps"):
            parts = line.split()
            name = next((p[5:] for p in parts if p.startswith("file=")), None)
            if name is None:
                name = "heap" if "heap" in parts else "stack" if "stack" in parts else "anon"
            name = f"{name} [{parts[1]}]"
            g = groups.setdefault(name, {})
            for p in parts:
                if p[0] == "N" and "=" in p and p[1:p.index("=")].isdigit():
                    node, pages = p[1:].split("=")
                    g[node] = g.get(node, 0) + int(pages)
    except OSError:
        return
    top = sorted(groups.items(), key=lambda kv: -sum(kv[1].values()))[:25]
    f.write(json.dumps({"t": time.time(), "numa_maps": [{"name": k, "pages": v} for k, v in top]}) + "\n")


def verify_lines():
    paths = glob.glob(os.path.join(run, "events.*.verify.jsonl"))
    return sum(1 for p in paths for _ in open(p)) if paths else 0


def ready():
    try:
        with open(os.path.join(run, "server.log"), errors="replace") as log:
            return any("fired up and ready" in line for line in log)
    except OSError:
        return False


pid = None
NUMA_MAPS_EVERY = 12  # samples: once a minute
tick = 0
done = {"ready": False, "numastat": False, "end": False}
last_lines, last_growth = 0, time.time()
with open(out, "w") as f:
    while not os.path.exists(stop):
        if pid is None or not os.path.exists(f"/proc/{pid}"):
            pid = find_scheduler()
        rec = {"t": time.time(), "nodes": {str(n): node_meminfo(n) for n in (0, 1)}, "zones": zoneinfo(),
               "vmstat": vmstat(), "kswapd": kswapd(), "swap_used_kb": swap_used_kb(), "disk": diskstats(),
               "scheduler": scheduler_state(pid) if pid else None}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        tick += 1
        if pid and tick % NUMA_MAPS_EVERY == 0:
            try:
                numa_maps(pid, f)
            except Exception as error:  # noqa: BLE001 - a diagnostic must not stop the sampler
                f.write(json.dumps({"t": time.time(), "numa_maps_error": repr(error)}) + "\n")
        if not done["ready"] and ready():
            fincore("ready", f)
            done["ready"] = True
        lines = verify_lines()
        if lines > last_lines:
            last_lines, last_growth = lines, time.time()
        if pid and not done["numastat"] and lines > DECODE_LINES:
            f.write(json.dumps({"t": time.time(), "numastat": subprocess.run(
                ["numastat", "-p", str(pid)], capture_output=True, text=True).stdout}) + "\n")
            done["numastat"] = True
        if done["numastat"] and not done["end"] and time.time() - last_growth > IDLE_S:
            fincore("decode_end", f)
            done["end"] = True
        f.flush()
        time.sleep(interval)
    if not done["end"]:
        fincore("stop", f)
