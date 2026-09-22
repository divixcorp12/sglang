#!/usr/bin/env python3
"""Harness for the c measurement (C_MEASUREMENT_PREREG.md). Produces the JSONL that c_analysis.py consumes.

MODES
  --dry-run   no torch, no CUDA: a synthetic device with a known law, to test the plumbing and the schema against
              c_analysis.py. Nothing in a dry run is a measurement.
  (default)   real: needs a CUDA context. **Do not start this without the GPU lock and crypto-c9's window**
              (run it through gpu-run.sh). --check-only does the setup (allocation, registration, the copy check) and exits;
              in real mode that already uses the GPU, so it is the first minute of the window, not a CPU test.

WHAT IT MEASURES (prereg section 2): T(n), the GPU-timeline time of one launch of the production
copy_expert_row_segments_gpu kernel moving n rows (n = 1..6), 13,315,584 B in six segments per row, from distinct rows
of pinned slabs allocated with the service's own allocator (expert_host_tier.allocate_host_slab), to VRAM scratch slots
cycled over 64 slots.

IMPLEMENTATION NOTES THAT THE PREREG DOES NOT SPELL OUT (each is a disclosed choice, not a hidden one)
  * "idle stream": every timed launch is preceded on the same stream by a short spin kernel (torch.cuda._sleep,
    SLEEP_US) so that the host, which needs tens of microseconds to validate and launch, is always ahead of the GPU;
    without it the start event fires on an idle stream and the host launch latency lands inside T. Only the launch
    between the start and end events is timed; the spin, the plan update and the inter-launch gap are outside it.
  * The plan (rows, slots, count) is three fixed device tensors updated by device-to-device copies OUTSIDE the timed
    window from tables built before the cell, for eager and graph alike, so the two differ only in launch mechanism.
  * ABBA: each cell's 200 launches are two visits of 100, the second visit in reverse order within its group.
  * Six segment sizes are derived from the tensor dimensions (hidden 5120, intermediate 2304) and checked to sum to the
    measured 13,315,584 B; they are not read from a loaded layout here.
  * Each cell line carries extra fields (the launching CPU, the worst foreign process, clocks). c_analysis.py ignores them; the
    NUMA page check, free memory and the copy check are in meta.json.
  * STEADY STATE: the user's permanent services are NOT stopped. Foreign CPU is recorded per cell (box_foreign_cores, from /proc/stat) and per pass
    (steady_state_by_pass in meta.json: the top foreign processes and command lines, from a thread scan), and the nvme ratio is refused when the
    foreign CPU differs between the idle and load arms of a pass by more than 1.0 core. The services' data is on /data (LVM root), not on the
    NVMe drives the harness reads, so idle cells must move under 0.02 GB/s on those drives. A steady load cancels in the ratio; a drifting one invalidates it.
  * RETRY: a visit that fails an environmental check (foreign > 10% on our cores, another GPU process, link not Gen3) is re-run in place up to 3 times;
    every failed attempt is in retries.jsonl. The check is never value-based (no T is looked at), so retries do not select on the result.
  * `foreign_max_core_pct` (the field gate 4.1 reads) is FOREIGN CPU USE ON THE CORES WE USE (the harness's and the reader's), the largest per-core sum
    of non-own thread CPU; not the box-wide maximum, so a permanent daemon on a core we do not use does not trip it. Idle-arm cells also record the
    NVMe read/write rates (see STEADY STATE above); in load cells the drives' bytes must match the reader's within 10%.
  * The `hot` arm packs with a torch CPU copy on the launching thread (one thread), which is not the service thread and
    is not pinned to the service's core.
  * `ce` is ExpertDMABackend.copy_rows, the copy engine, labelled as such; it is only the yardstick for the bandwidth gate.
    Its T includes the host time to enqueue 6 x n copies after the start event (the spin covers it only if that fits inside
    SLEEP_US); read it as a lower bound on copy-engine speed, not as its peak.
"""
import argparse, collections, ctypes, json, os, random, statistics as S, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROW_BYTES = 13_315_584
SEGMENTS = (("w13_trellis", 8_847_360), ("w13_suh", 20_480), ("w13_svh", 9_216),
            ("w2_trellis", 4_423_680), ("w2_suh", 4_608), ("w2_svh", 10_240))
assert sum(b for _, b in SEGMENTS) == ROW_BYTES
NS = (1, 2, 3, 4, 5, 6)
LOAD_NS = (1, 3, 6)
ROWS_PER_NODE = 150                 # 150 rows = 2.0 GB per node = 20.9 x the 96 MiB L2
SCRATCH_SLOTS = 64                  # 0.85 GB of destination, cycled
LAUNCHES_PER_CELL = 200
WARMUP_LAUNCHES = 20
PASSES = 5
SLEEP_US = 600
IDLE_DRIVE_MAX_GBS = 0.02          # an idle-arm cell during which the WATCHED NVMe devices (those holding the mirror roots and the source checkpoint) moved more than this (reads + writes) is contaminated: rho's baseline. The user's services live on /data (LVM root), not on these drives; other NVMe devices (nvme1) are recorded, not judged.
MAX_ATTEMPTS = 3                   # a visit that fails an ENVIRONMENTAL check is re-run in place up to this many times; every attempt is logged to retries.jsonl
ENV_FOREIGN_PCT = 10.0             # the frozen per-core gate's threshold, applied to a single visit for the retry decision
LOAD_FOREIGN_IO_MAX = 0.10         # in a load cell, drive bytes beyond the reader's own may not exceed 10% of the reader's bytes
SEED = 20260921

KERNEL_IO_THREADS = ("kworker", "ksoftirqd", "irq/", "nvme", "iou-", "migration", "rcu_", "cpuhp")
DRY_BOX_SHIFT = 0.0                # tests only: the synthetic device reports this much extra foreign CPU (cores) in load cells
BOX_FOREIGN_MAX_DELTA = 1.0        # cores: idle-arm and load-arm cells of a pass may differ by at most this in non-own CPU, or the nvme ratio is invalid

Cell = collections.namedtuple("Cell", "engine state node load launch n")

# ---------------------------------------------------------------------------------------------------- pure helpers

def ring_ids(rows, seed):
    """A seeded permutation ring: consuming it in order re-reads a row only after `rows` other consumptions."""
    order = list(range(rows)); random.Random(seed).shuffle(order); return order

def draw(ring, start, n):
    return [ring[(start + j) % len(ring)] for j in range(n)]

def min_reuse_distance(seq):
    """Smallest number of consumed rows between two reads of one row in `seq`; a large number when nothing repeats."""
    last, best = {}, 10 ** 9
    for i, r in enumerate(seq):
        if r in last: best = min(best, i - last[r])
        last[r] = i
    return best

def cell_list():
    cells = []
    for node in (0, 1):
        for n in NS: cells.append(Cell("sm", "cold", node, "idle", "eager", n))
        for n in NS: cells.append(Cell("sm", "hot", node, "idle", "eager", n))
        for n in NS: cells.append(Cell("ce", "cold", node, "idle", "eager", n))
        cells.append(Cell("sm", "cold", node, "idle", "graph", 3))
    for n in LOAD_NS: cells.append(Cell("sm", "repeat", 0, "idle", "eager", n))
    return cells

def load_cell_list():
    return [Cell("sm", "cold", node, "nvme", "eager", n) for node in (0, 1) for n in LOAD_NS]

def abba(cells, rng):
    """Two visits per cell: a random order, then its reverse. Returns [(cell, visit)]."""
    first = list(cells); rng.shuffle(first)
    return [(c, 0) for c in first] + [(c, 1) for c in reversed(first)]

def _parse_cpus(txt):
    out = []
    for part in [x for x in str(txt).split(",") if x]:
        a, _, b = part.partition("-"); out.extend(range(int(a), int(b or a) + 1))
    return out

def node_cpus(node):
    txt = Path("/sys/devices/system/node/node%d/cpulist" % node).read_text().strip()
    out = []
    for part in txt.split(","):
        a, _, b = part.partition("-"); out.extend(range(int(a), int(b or a) + 1))
    return out

# ------------------------------------------------------------------------------------------------- box conditions

_SIB_CACHE = {}
def thread_siblings(cpu):
    """The logical CPUs sharing a physical core with `cpu` (including itself), from /sys/devices/system/cpu/cpuN/topology/thread_siblings_list.
    SMT is active on this box (2 threads per core): a logical CPU is half a physical core, so 'the cores the run uses' are physical cores."""
    if cpu not in _SIB_CACHE:
        try: _SIB_CACHE[cpu] = tuple(_parse_cpus(Path("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu).read_text().strip()))
        except OSError: _SIB_CACHE[cpu] = (cpu,)
    return _SIB_CACHE[cpu]

class ForeignLoad:
    """Foreign CPU use on the cores WE use, over one window, from /proc/stat (cheap, so it can run around every visit).

    busy(core) = user + nice + system ticks (irq, softirq, steal, iowait and idle are NOT busy, so the interrupt time of our own NVMe and GPU
    traffic is not counted against us). foreign(core) = busy(core) minus the ticks of our own threads (the harness, the reader), each own thread's
    ticks attributed to the core it was on at the end of the window (split with the core it was on at the start if it moved).
    Returns (worst, where): the largest per-core foreign percentage of one core over the cores WE USED in the window (own threads on it for >= USED_FRACTION of the
    window; all of `cores` when we had no presence). RESOLUTION: ticks are 10 ms; a core with fewer than
    MIN_TICKS foreign ticks in the window is reported as at most 9.9%, because one or two ticks in a window of 100-200 ms are not evidence of 10%.
    `last["box_foreign_cores"]` is the same quantity summed over every CPU, in cores. It still contains kernel worker threads that carry our I/O
    (system time), so it is an upper bound on foreign load, which makes the box-drift gate conservative."""
    MIN_TICKS = 3
    USED_FRACTION = 0.10          # a core counts as USED by us if our own threads ran on it for at least this fraction of the window
    def __init__(self, own_pids, cores, siblings=thread_siblings):
        self.own = set(own_pids); self.cores = set(cores); self.hz = os.sysconf("SC_CLK_TCK"); self.siblings = siblings
    def add_own(self, pid): self.own.add(pid)
    last = {}
    def _cpu(self):
        out = {}
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split(); out[int(f[0][3:])] = int(f[1]) + int(f[2]) + int(f[3])
        return out
    def _own(self):
        out = {}
        for pid in self.own:
            try:
                for t in os.listdir("/proc/%d/task" % pid):
                    f = Path("/proc/%d/task/%s/stat" % (pid, t)).read_text(); rest = f[f.rindex(")") + 2:].split()
                    out[(pid, int(t))] = (int(rest[11]) + int(rest[12]), int(rest[36]))
            except (OSError, ValueError, IndexError): pass
        return out
    def start(self):
        self.a = self._cpu(); self.ao = self._own(); self.t = time.monotonic()
    def stop(self):
        b = self._cpu(); bo = self._own(); dt = max(time.monotonic() - self.t, 1e-9)
        own_c = collections.defaultdict(float)
        for key, (j, cpu) in bo.items():
            j0, cpu0 = self.ao.get(key, (j, cpu)); d = max(0, j - j0)
            if cpu0 != cpu: own_c[cpu] += d / 2.0; own_c[cpu0] += d / 2.0
            else: own_c[cpu] += d
        foreign = {c: max(0.0, (b[c] - self.a.get(c, b[c])) - own_c.get(c, 0.0)) for c in b}
        box = sum(foreign.values()) / self.hz / dt
        self.last = {"box_foreign_cores": box, "top": [], "loadavg": Path("/proc/loadavg").read_text().split()[:3], "window_s": dt}
        # the cores the run USES in this window: those where our own threads accrued at least USED_FRACTION of the window; a spare core of the allowed mask on
        # which we did nothing cannot contend with us for CPU. With no own presence at all (a plain survey) every core of the set counts.
        used = {c for c in self.cores if own_c.get(c, 0.0) >= self.USED_FRACTION * self.hz * dt}
        if not used and own_c: used = {c for c in own_c if c in self.cores}
        # SMT: a used logical CPU is half of a physical core, so the foreign CPU on its thread sibling counts against the same gate (summed over the pair)
        def phys(c): return sum(foreign.get(x, 0.0) for x in self.siblings(c) if x in foreign)
        mine = {c: phys(c) for c in (used or self.cores) if c in foreign}
        self.last["used_cores"] = sorted(used); self.last["sibling_foreign_ticks"] = {c: round(phys(c) - foreign.get(c, 0.0), 1) for c in mine}
        if not mine: return (0.0, "")
        c = max(mine, key=mine.get); ticks = mine[c]; pct = 100.0 * ticks / self.hz / dt
        if ticks < self.MIN_TICKS: pct = min(pct, 9.9)
        return (pct, "cpu%d+siblings%s: %.1f foreign ticks in %.2f s" % (c, list(self.siblings(c)), ticks, dt))

class TopScan:
    """Once per pass, a full thread scan (0.25 s at load 30): who the biggest foreign processes are, by process name and CPU, with their command lines.
    Not used by any gate; it is the record of what the steady state was made of."""
    def __init__(self, own_pids): self.own = set(own_pids); self.hz = os.sysconf("SC_CLK_TCK")
    def add_own(self, pid): self.own.add(pid)
    def _snap(self):
        out = {}
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) in self.own: continue
            try: tids = os.listdir("/proc/%s/task" % d)
            except OSError: continue
            for t in tids:
                try:
                    f = Path("/proc/%s/task/%s/stat" % (d, t)).read_text(); name = f[f.index("(") + 1:f.rindex(")")]; rest = f[f.rindex(")") + 2:].split()
                    out[(int(d), int(t))] = (int(rest[11]) + int(rest[12]), name)
                except (OSError, ValueError, IndexError): pass
        return out
    def sample(self, seconds=1.0):
        a = self._snap(); t = time.monotonic(); time.sleep(seconds); b = self._snap(); dt = time.monotonic() - t
        by = collections.defaultdict(float); nm = {}
        for k, (j, name) in b.items():
            if k in a and not name.startswith(KERNEL_IO_THREADS):
                by[k[0]] += 100.0 * (j - a[k][0]) / self.hz / dt
                if k[1] == k[0] or k[0] not in nm: nm[k[0]] = name
        top = sorted(by.items(), key=lambda kv: -kv[1])[:6]
        return [{"pid": p_, "comm": nm[p_], "cpu_pct": round(v, 1), "cmdline": _cmdlines([p_]).get(str(p_), "")[:160]} for p_, v in top if v > 1.0]

DRY_FOREIGN_SEQ = []               # tests only: successive per-visit foreign percentages the dry run reports

class _NoForeign:
    last = {"box_foreign_cores": 0.0, "top": [], "loadavg": ["0", "0", "0"]}
    def start(self): pass
    def add_own(self, pid): pass
    def stop(self): return (DRY_FOREIGN_SEQ.pop(0) if DRY_FOREIGN_SEQ else 0.0, "dry")

class SmiSampler:
    """Streams nvidia-smi so that cell start/end conditions cost nothing inside the timed region."""
    def __init__(self, own_pid):
        self.own = own_pid; self.gen = []; self.apps = []; self._procs = []
    def start(self):
        for args, sink, kind in ((["--query-gpu=pcie.link.gen.current,pstate,clocks.sm", "--format=csv,noheader,nounits", "-lms", "250"], self.gen, "gpu"),
                                 (["--query-compute-apps=pid", "--format=csv,noheader", "-lms", "250"], self.apps, "apps")):
            p = subprocess.Popen(["nvidia-smi"] + args, stdout=subprocess.PIPE, text=True, bufsize=1)
            self._procs.append(p); threading.Thread(target=self._read, args=(p, sink, kind), daemon=True).start()
    def _read(self, p, sink, kind):
        for line in p.stdout:
            t = time.monotonic(); line = line.strip()
            if kind == "gpu":
                try:
                    g, ps, clk = [x.strip() for x in line.split(",")]; sink.append((t, int(g), int(ps.lstrip("P")), int(clk)))
                except ValueError: pass
            else:
                sink.append((t, [int(x) for x in line.split() if x.isdigit() and int(x) != self.own]))
    def window(self, t0, t1):
        g = [x for x in self.gen if t0 - 0.6 <= x[0] <= t1 + 0.6]
        a = [x for x in self.apps if t0 - 0.6 <= x[0] <= t1 + 0.6]
        if not g: return None
        return {"link_gen_start": g[0][1], "link_gen_end": g[-1][1], "pstate_start": g[0][2], "sm_mhz_min": min(x[3] for x in g),
                "sm_mhz_max": max(x[3] for x in g), "other_gpu_procs": max((len(x[1]) for x in a), default=0)}
    def stop(self):
        for p in self._procs: p.terminate()

def nvme_sectors_read():
    """Sectors read so far by every whole NVMe device (nvme<N>n<M>), from /proc/diskstats (field 6)."""
    out = {}
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) > 5 and f[2].startswith("nvme") and "p" not in f[2].split("n", 1)[1]: out[f[2]] = int(f[5])
    return out

def nvme_rw():
    """(bytes read, bytes written) so far by every whole NVMe device, from /proc/diskstats (fields 6 and 10)."""
    out = {}
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) > 9 and f[2].startswith("nvme") and "p" not in f[2].split("n", 1)[1]: out[f[2]] = (int(f[5]) * 512, int(f[9]) * 512)
    return out

DRIVE_PATHS = ("/mnt/nvme0/dsv41_flash", "/mnt/nvme4/dsv41_flash", "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw")   # the two mirror roots the reader reads and the source checkpoint (bench_mirror_rows defaults)

def watched_nvme(paths):
    """Whole NVMe devices (nvme<N>n<M>) that hold `paths`, resolved through st_dev and /sys/dev/block, never by mount label. None entries (a path
    that is not on NVMe or does not exist) are skipped; an empty result means the drives could not be resolved."""
    out = set()
    for pth in paths:
        try:
            st = os.stat(pth).st_dev; real = os.path.realpath("/sys/dev/block/%d:%d" % (os.major(st), os.minor(st)))
        except OSError: continue
        parts = real.split("/")
        for name in reversed(parts):
            if name.startswith("nvme") and "n" in name[4:]:
                out.add(name.split("p")[0] if "p" in name[4:] else name); break
    return out

def _cmdlines(pids):
    out = {}
    for pid in sorted(pids):
        try: out[str(pid)] = Path("/proc/%d/cmdline" % pid).read_bytes().replace(b"\0", b" ").decode(errors="replace")[:300]
        except OSError: out[str(pid)] = "(gone)"
    return out

def drive_gb_per_s(before, after, dt):
    return sum(after[k] - before.get(k, 0) for k in after) * 512 / 1e9 / max(dt, 1e-9)

# -------------------------------------------------------------------------------------------------- NUMA helpers

_libc = None
def _syscalls():
    global _libc
    if _libc is None: _libc = ctypes.CDLL(None, use_errno=True)
    return _libc

def set_mempolicy(node):
    """MPOL_BIND to `node`, or MPOL_DEFAULT when node is None (x86_64 syscall 238)."""
    mask = ctypes.c_ulong(0 if node is None else 1 << node)
    r = _syscalls().syscall(238, 0 if node is None else 2, ctypes.byref(mask), 64)
    if r != 0: raise OSError(ctypes.get_errno(), "set_mempolicy")

def pages_on_node(ptr, nbytes, node, sample=512):
    """Fraction of sampled 4 KiB pages of [ptr, ptr+nbytes) that move_pages (query mode, syscall 279) reports on `node`."""
    page = 4096; n = nbytes // page; step = max(1, n // sample); idx = list(range(0, n, step))[:sample]
    pages = (ctypes.c_void_p * len(idx))(*[ptr + i * page for i in idx]); status = (ctypes.c_int * len(idx))()
    r = _syscalls().syscall(279, 0, len(idx), pages, None, status, 0)
    if r != 0: raise OSError(ctypes.get_errno(), "move_pages")
    return sum(1 for s in status if s == node) / len(idx)

def node_mem_mib(node):
    """MemFree, FilePages (the page cache) and Active(file)+Inactive(file) of a NUMA node, MiB."""
    out = {}
    for line in Path("/sys/devices/system/node/node%d/meminfo" % node).read_text().splitlines():
        f = line.split()
        for key in ("MemFree", "FilePages", "Active(file)", "Inactive(file)"):
            if f[2] == key + ":": out[key] = int(f[3]) // 1024
    return out

def node_free_mib(node):
    for line in Path("/sys/devices/system/node/node%d/meminfo" % node).read_text().splitlines():
        if "MemFree" in line: return int(line.split()[-2]) // 1024
    return -1

# ------------------------------------------------------------------------------------------------------ backends

class DryDevice:
    """A synthetic device with a known law T(n) = f + c*n (node 1 is 4% slower), for plumbing tests. Not a measurement."""
    name = "dry"
    def __init__(self, c=1.08, f=0.006, seed=1): self.c, self.f, self.rng = c, f, random.Random(seed); self.log = []
    def setup(self, args):
        return {"backend": "dry", "c": self.c, "f": self.f}
    def reuse_distance(self, node): return 10 ** 9
    def run_visit(self, cell, launches, args):
        c = self.c * (1.04 if cell.node == 1 else 1.0) * (1.08 if cell.load == "nvme" else 1.0)
        base = (self.f + c * cell.n) * (0.93 if cell.engine == "ce" else 1.0)
        T = [base * (1 + abs(self.rng.gauss(0, 0.004))) for _ in range(launches)]
        return T, {"cpu": 0, "dry_box": DRY_BOX_SHIFT if cell.load == "nvme" else 0.0}
    def close(self): pass

class RealDevice:
    name = "cuda"
    def __init__(self, repo): self.repo = repo
    def setup(self, args):
        sys.path.insert(0, str(Path(self.repo) / "python"))
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        torch.cuda.init(); self.dev = torch.device("cuda", torch.cuda.current_device()); torch.cuda.set_device(self.dev)
        from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu, expert_row_segments
        from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
        from sglang.srt.layers.moe.expert_dma import ExpertDMABackend
        self.copy = copy_expert_row_segments_gpu
        self.stream = torch.cuda.Stream(device=self.dev)
        self.dest = [torch.empty((SCRATCH_SLOTS, b), dtype=torch.uint8, device=self.dev) for _, b in SEGMENTS]
        self.src, self.seg, self.ring, self.consumed, meta = {}, {}, {}, {}, {"nodes": {}}
        self.nodes = tuple(getattr(args, "nodes", (0, 1)))
        self.global_seq = {n_: [] for n_ in self.nodes}
        for node in self.nodes:
            free_before = node_free_mib(node); mem_before = node_mem_mib(node)
            set_mempolicy(node)
            try:
                slabs = [allocate_host_slab(ROWS_PER_NODE, (b,), torch.uint8, register=False) for _, b in SEGMENTS]
                for s in slabs: s.fill_(1)                                      # first touch under MPOL_BIND(node)
            finally:
                set_mempolicy(None)
            fracs = [pages_on_node(s.data_ptr(), s.numel(), node) for s in slabs]
            if min(fracs) < 0.99: raise SystemExit("node %d slabs are only %.2f on the node: refusing (L4)" % (node, min(fracs)))
            from sglang.srt.mem_cache.pool_host.common import _cuda_host_register
            for s, (_, b) in zip(slabs, SEGMENTS): _cuda_host_register(s, registration_granularity_bytes=b)
            self.src[node] = slabs; self.seg[node] = expert_row_segments(list(zip(slabs, self.dest)))
            self.ring[node] = ring_ids(ROWS_PER_NODE, SEED + node); self.consumed[node] = 0
            meta["nodes"][node] = {"page_fraction_on_node": fracs, "free_mib_before": free_before, "free_mib_after": node_free_mib(node), "mem_mib_before": mem_before, "mem_mib_after": node_mem_mib(node)}
        # each row's first 8 bytes carry its id, so a launch can be checked to have moved the intended rows
        for node in self.nodes:
            for s in self.src[node]:
                ids = torch.arange(ROWS_PER_NODE, dtype=torch.int64) + 1000 * (node + 1)
                s[:, :8].copy_(ids.view(torch.uint8).view(ROWS_PER_NODE, 8))
        self.bounce = [torch.full((6, b), 7, dtype=torch.uint8) for _, b in SEGMENTS]
        self.plan_rows = torch.zeros(6, dtype=torch.int64, device=self.dev)
        self.plan_slots = torch.zeros(6, dtype=torch.int32, device=self.dev)
        self.plan_count = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.dma = ExpertDMABackend(); meta["ce_backend"] = self.dma.actual_backend
        self.graph = {}
        # spin calibration: cycles per microsecond of torch.cuda._sleep
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(int(1e6)); torch.cuda.synchronize()
        e0.record(); torch.cuda._sleep(int(2e7)); e1.record(); torch.cuda.synchronize()
        self.cycles_per_us = 2e7 / (e0.elapsed_time(e1) * 1000.0); meta["spin_cycles_per_us"] = self.cycles_per_us
        for node in self.nodes: self._verify(node)
        meta["device"] = torch.cuda.get_device_name(self.dev); meta["torch"] = torch.__version__
        return meta
    def _plan(self, node, n, rows, slots):
        t = self.torch
        r = t.zeros(6, dtype=t.int64); r[:n] = t.tensor(rows); s = t.zeros(6, dtype=t.int32); s[:n] = t.tensor(slots)      # (v5: was bare `torch`, a NameError found on the first real launch)
        return r, s
    def _verify(self, node):
        t = self.torch; rows = [5, 9, 1, 100, 77, 3]; slots = [10, 11, 12, 13, 14, 15]
        r, s = self._plan(node, 6, rows, slots)
        with t.cuda.stream(self.stream):
            self.plan_rows.copy_(r); self.plan_slots.copy_(s); self.plan_count.fill_(6)
            self.copy(self.seg[node], self.plan_rows, self.plan_slots, self.plan_count)
        self.stream.synchronize()
        for lane, (row, slot) in enumerate(zip(rows, slots)):
            for d in self.dest:
                got = int(d[slot, :8].cpu().view(t.int64)[0]); want = row + 1000 * (node + 1)
                if got != want: raise SystemExit("copy check failed: node %d lane %d row %d slot %d got %d want %d" % (node, lane, row, slot, got, want))
    def _tables(self, cell, launches, rows_start):
        t = self.torch; n = cell.n; node = cell.node; rows_tbl = t.zeros((launches, 6), dtype=t.int64); slots_tbl = t.zeros((launches, 6), dtype=t.int32)
        ids, rows_cpu, slots_cpu = [], [], []
        for i in range(launches):
            if cell.state == "repeat": rows = list(range(n))
            else: rows = draw(self.ring[node], rows_start + i * n, n)
            ids.extend(rows); slots = [(i * n + j) % SCRATCH_SLOTS for j in range(n)]
            rows_cpu.append(rows); slots_cpu.append(slots)
            rows_tbl[i, :n] = t.tensor(rows); slots_tbl[i, :n] = t.tensor(slots)
        return rows_tbl.to(self.dev), slots_tbl.to(self.dev), ids, rows_cpu, slots_cpu
    def _graph_for(self, node):
        if node not in self.graph:
            t = self.torch; g = t.cuda.CUDAGraph()
            with t.cuda.stream(self.stream):
                self.plan_count.fill_(3)
            self.stream.synchronize()
            with t.cuda.graph(g, stream=self.stream):
                self.copy(self.seg[node], self.plan_rows, self.plan_slots, self.plan_count)
            self.graph[node] = g
        return self.graph[node]
    def reuse_distance(self, node): return min_reuse_distance(self.global_seq[node])
    def run_visit(self, cell, launches, args):
        t = self.torch; node, n = cell.node, cell.n
        total = launches + WARMUP_LAUNCHES
        rows_tbl, slots_tbl, ids, rows_cpu, slots_cpu = self._tables(cell, total, self.consumed[node])
        launches = total
        counts = t.tensor([n], dtype=t.int32, device=self.dev)
        starts = [t.cuda.Event(enable_timing=True) for _ in range(launches)]; ends = [t.cuda.Event(enable_timing=True) for _ in range(launches)]
        cycles = int(SLEEP_US * self.cycles_per_us)
        graph = self._graph_for(node) if cell.launch == "graph" else None
        hot = cell.state == "hot"
        with t.cuda.stream(self.stream):
            for i in range(launches):
                if hot:                                              # the service packs a row, then the GPU copies it
                    rows = rows_cpu[i]                                   # host copies of the tables: a device .tolist() would sync the stream
                    for j, row in enumerate(rows):
                        for seg_i, src in enumerate(self.src[node]): src[row, 8:].copy_(self.bounce[seg_i][j, 8:])
                t.cuda._sleep(cycles)
                self.plan_rows.copy_(rows_tbl[i]); self.plan_slots.copy_(slots_tbl[i]); self.plan_count.copy_(counts)
                starts[i].record(self.stream)
                if cell.engine == "ce":
                    for src, dst in zip(self.src[node], self.dest): self.dma.copy_rows(src, dst, tuple(rows_cpu[i]), tuple(slots_cpu[i]))
                elif graph is not None: graph.replay()
                else: self.copy(self.seg[node], self.plan_rows, self.plan_slots, self.plan_count)
                ends[i].record(self.stream)
                if hot: ends[i].synchronize()
        self.stream.synchronize()
        T = [s.elapsed_time(e) for s, e in zip(starts, ends)][WARMUP_LAUNCHES:]
        self.consumed[node] += launches * n
        if cell.state != "repeat": self.global_seq[node].extend(ids)     # time-ordered rows read from this node's slabs (warm-up launches included)
        return T, {"cpu": _syscalls().sched_getcpu()}                # (v6: os.sched_getcpu does not exist; found on the first real visit)
    def close(self):
        pass

# ----------------------------------------------------------------------------------------------------- the run

def record(cell, pass_idx, T, reuse, cond, foreign, extra):
    return {"process": pass_idx, "engine": cell.engine, "state": cell.state, "node": cell.node, "load": cell.load, "launch": cell.launch, "n": cell.n,
            "row_bytes": ROW_BYTES, "T_ms": T, "distinct_rows": ROWS_PER_NODE,
            "min_reuse_distance_rows": reuse,
            "link_gen_start": cond["link_gen_start"], "link_gen_end": cond["link_gen_end"], "pstate_start": cond["pstate_start"],
            "other_gpu_procs": cond["other_gpu_procs"], "foreign_max_core_pct": foreign[0], "foreign_max_proc": foreign[1],
            "sm_mhz_min": cond["sm_mhz_min"], "sm_mhz_max": cond["sm_mhz_max"], **extra}

def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dry = args.dry_run
    dev = DryDevice() if dry else RealDevice(args.repo)
    meta = {"harness": str(Path(__file__).name), "dry_run": dry, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "passes": PASSES,
            "launches_per_cell": LAUNCHES_PER_CELL, "rows_per_node": ROWS_PER_NODE, "scratch_slots": SCRATCH_SLOTS, "sleep_us": SLEEP_US}
    import hashlib
    meta["file_sha256"] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("c_harness.py", "nvme_load_reader.py", "c_analysis.py", "quiet_check.py") if (HERE / n).exists()}
    meta["repo"] = str(args.repo)
    try: meta["repo_head"] = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception: meta["repo_head"] = None
    watched = set() if dry else watched_nvme(args.drive_paths)
    if not dry and not watched: raise SystemExit("cannot resolve the NVMe devices of %s" % (args.drive_paths,))
    meta["watched_drives"] = sorted(watched)
    meta.update(dev.setup(args))
    if args.check_only:
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str)); print("check-only ok"); return 0
    rng = random.Random(SEED)
    mycores = set(os.sched_getaffinity(0)) | set(_parse_cpus(args.reader_cpus))
    foreign = _NoForeign() if dry else ForeignLoad([os.getpid()], mycores); smi = None      # a dry run must not depend on the ambient load of the machine it runs on
    topscan = None if dry else TopScan([os.getpid()])
    if not dry:
        smi = SmiSampler(os.getpid()); smi.start(); time.sleep(2.0)
        # untimed load so the link leaves Gen1 (L5), then one look at the state
        for _ in range(3): dev.run_visit(Cell("sm", "cold", 0, "idle", "eager", 6), 40, args)
    reader = None
    if args.with_nvme:
        go, stop, ready, logp = [out / x for x in ("reader.go", "reader.stop", "reader.ready", "reader_log.json")]
        for f in (go, stop, ready): f.unlink(missing_ok=True)
        pin = [] if dry else ["numactl", "--membind=%d" % args.bounce_node, "taskset", "-c", args.reader_cpus]
        cmd = pin + [sys.executable, str(HERE / "nvme_load_reader.py"),
                     "--go", str(go), "--stop", str(stop), "--ready", str(ready), "--out", str(logp)] + (["--dry-run"] if dry else [])
        reader = subprocess.Popen(cmd); foreign.add_own(reader.pid)
        if topscan: topscan.add_own(reader.pid)
        for _ in range(600):
            if ready.exists(): break
            time.sleep(0.5)
        else: reader.kill(); raise SystemExit("reader never became ready")
    lines = (out / "results.jsonl").open("w"); retries_f = (out / "retries.jsonl").open("w")
    skip = set(args.skip_arms)
    cells = [c for c in cell_list() if c.state not in skip and c.engine not in skip and c.launch not in skip]
    load_cells = load_cell_list() if args.with_nvme else []
    meta["skipped_arms"] = sorted(skip); meta["NOT_MEASURED"] = sorted(skip) + ([] if args.with_nvme else ["nvme"])      # a skipped arm is a declared deviation from the registered design and must be named in the report
    order_log = []; load_windows = []; load_drive = []; box_idle = []; box_load = []; box_shift = []
    idle_dirty = []; steady = []; n_visits = [0]; n_retries = [0]
    def env_failure(fp, t0, t1):
        """Environmental (not value-based) reasons to re-run a visit: foreign CPU on our cores, another GPU process, the link not at Gen3."""
        why = []
        if fp[0] > ENV_FOREIGN_PCT: why.append("foreign %.1f%% on our cores (%s)" % (fp[0], fp[1]))
        if smi:
            c = smi.window(t0, t1)
            if c is not None:
                if c["other_gpu_procs"] > 0: why.append("another GPU process")
                if c["link_gen_start"] != 3 or c["link_gen_end"] != 3: why.append("PCIe link gen %s/%s" % (c["link_gen_start"], c["link_gen_end"]))
        return why
    def one_visit(cell):
        la0 = Path("/proc/loadavg").read_text().split()[:3]
        foreign.start(); ds0 = None if dry else nvme_rw(); t0 = time.monotonic()
        T, extra = dev.run_visit(cell, LAUNCHES_PER_CELL // 2, args)
        t1 = time.monotonic(); fp = foreign.stop(); ds1 = None if dry else nvme_rw(); extra = dict(extra)
        extra["foreign"] = dict(foreign.last); extra["loadavg_start"] = la0
        if dry: extra["foreign"]["box_foreign_cores"] = extra.get("dry_box", 0.0)
        extra["drive_read_bytes"] = None if dry else sum(ds1[k][0] - ds0.get(k, (0, 0))[0] for k in ds1 if k in watched)       # only the drives this run reads
        extra["drive_write_bytes"] = None if dry else sum(ds1[k][1] - ds0.get(k, (0, 0))[1] for k in ds1 if k in watched)
        extra["drive_by_dev"] = None if dry else {k: [ds1[k][0] - ds0.get(k, (0, 0))[0], ds1[k][1] - ds0.get(k, (0, 0))[1]] for k in ds1 if ds1[k] != ds0.get(k)}
        extra["drive_bytes"] = extra["drive_read_bytes"]
        return (T, extra, t0, t1, fp)
    def visit(cell, sink, pass_idx):
        for attempt in range(MAX_ATTEMPTS):
            v = one_visit(cell); n_visits[0] += 1
            why = env_failure(v[4], v[2], v[3])
            if not why or attempt == MAX_ATTEMPTS - 1:
                if why: v[1]["env_failure_kept"] = why          # out of attempts: kept, and the frozen gate will judge it
                v[1]["attempts"] = attempt + 1; sink.append(v); return
            n_retries[0] += 1
            retries_f.write(json.dumps({"pass": pass_idx, "cell": list(cell), "attempt": attempt + 1, "why": why, "median_T_ms": S.median(v[0]),
                                        "foreign_where": v[4][1], "loadavg": v[1]["loadavg_start"]}) + "\n"); retries_f.flush()
    for p in range(PASSES):
        if topscan: steady.append({"pass": p, "start": topscan.sample(1.0)})
        acc = collections.defaultdict(list)
        for cell, v in abba(cells, rng):
            visit(cell, acc[cell], p); order_log.append((p, list(cell), v))
        load_acc = collections.defaultdict(list)
        if args.with_nvme:
            go.write_text("go\n"); time.sleep(0.5)
            for cell, v in abba(load_cells, rng):
                visit(cell, load_acc[cell], p); order_log.append((p, list(cell), v)); load_windows.append((list(cell), load_acc[cell][-1][2], load_acc[cell][-1][3])); load_drive.append(load_acc[cell][-1][1]["drive_bytes"])
            go.unlink(); time.sleep(0.5)
        for cell, visits in list(acc.items()) + list(load_acc.items()):
            T = [x for v in visits for x in v[0]]
            t0 = min(v[2] for v in visits); t1 = max(v[3] for v in visits)
            cond = smi.window(t0, t1) if smi else {"link_gen_start": 3, "link_gen_end": 3, "pstate_start": 0, "other_gpu_procs": 0, "sm_mhz_min": 2900, "sm_mhz_max": 2900}
            if cond is None: raise SystemExit("no nvidia-smi samples for cell %s: refusing to write a cell without conditions" % (cell,))
            worst = max((v[4] for v in visits), key=lambda x: x[0])
            dur = sum(v[3] - v[2] for v in visits); db = None if dry else sum(v[1]["drive_bytes"] for v in visits)
            dw = None if dry else sum(v[1]["drive_write_bytes"] for v in visits)
            extra = {"cpu": visits[0][1].get("cpu"), "pass": p, "visits": len(visits), "attempts": [v[1]["attempts"] for v in visits], "drive_bytes": db, "drive_write_bytes": dw,
                     "drive_gb_per_s": None if dry else db / 1e9 / max(dur, 1e-9), "drive_by_dev": [v[1]["drive_by_dev"] for v in visits], "foreign_where": worst[1],
                     "box_foreign_cores": max(v[1]["foreign"]["box_foreign_cores"] for v in visits),
                     "loadavg_start": visits[0][1]["loadavg_start"], "loadavg_end": visits[-1][1]["foreign"]["loadavg"]}
            if cell.load == "idle" and not dry and (db + dw) / 1e9 / max(dur, 1e-9) > IDLE_DRIVE_MAX_GBS: idle_dirty.append((list(cell), p, round((db + dw) / 1e9 / max(dur, 1e-9), 4)))
            (box_load if cell.load == "nvme" else box_idle).append((p, extra["box_foreign_cores"]))
            lines.write(json.dumps(record(cell, p, T, dev.reuse_distance(cell.node) if cell.state != "repeat" else 10 ** 9, cond, worst, extra)) + "\n"); lines.flush()
        if topscan: steady[-1]["end"] = topscan.sample(1.0)
        print("pass %d done" % p, flush=True)
    rc = 0
    if reader:
        (out / "reader.stop").write_text("stop\n"); reader.wait(timeout=60)
        rl = json.loads(logp.read_text())["log"]
        # the load arm must actually have loaded the drives during each of its windows (not gated by c_analysis.py, so gated here)
        gbs = []
        for cell, t0, t1 in load_windows:
            b = sum(x[1] for x in rl if int(t0 * 1e9) <= x[0] <= int(t1 * 1e9))       # time.monotonic() and CLOCK_MONOTONIC share an epoch on Linux
            gbs.append({"cell": cell, "reader_gb_per_s": b / max(t1 - t0, 1e-9) / 1e9})
        for g, (cell, t0, t1), db in zip(gbs, load_windows, load_drive):
            rb = g["reader_gb_per_s"] * (t1 - t0) * 1e9
            g["drive_bytes"] = db; g["foreign_io_fraction"] = None if (dry or rb <= 0) else (db - rb) / rb
        meta["load_windows"] = gbs
        if not dry and any(g["foreign_io_fraction"] is None or abs(g["foreign_io_fraction"]) > LOAD_FOREIGN_IO_MAX for g in gbs):
            (out / "results.INVALID").write_text("drive traffic in a load window differs from the reader's own bytes by more than %.0f%%: another lane used the drives; the nvme arm's rho is not to be quoted\n" % (100 * LOAD_FOREIGN_IO_MAX)); rc = 3
        if any(g["reader_gb_per_s"] < args.min_reader_gbs for g in gbs) and not dry:
            (out / "results.INVALID").write_text("the nvme load arm did not load the drives at >= %.1f GB/s in every window: no load-arm number is to be quoted\n" % args.min_reader_gbs); rc = 3
    for p_ in range(PASSES):
        i_ = [x for q, x in box_idle if q == p_]; l_ = [x for q, x in box_load if q == p_]
        if i_ and l_ and abs(S.mean(l_) - S.mean(i_)) > BOX_FOREIGN_MAX_DELTA: box_shift.append((p_, round(S.mean(i_), 2), round(S.mean(l_), 2)))
    meta["box_foreign_cores_idle_vs_load_by_pass"] = [(p_, round(S.mean([x for q, x in box_idle if q == p_] or [0]), 2), round(S.mean([x for q, x in box_load if q == p_] or [0]), 2)) for p_ in range(PASSES)]
    if box_shift:
        (out / "results.INVALID").write_text("foreign CPU (non-own, from /proc/stat, an upper bound) changed by more than %.1f cores between the idle and load arms in pass(es) %s: the nvme ratio is invalid\n" % (BOX_FOREIGN_MAX_DELTA, box_shift)); rc = 3
    if idle_dirty:
        (out / "results.INVALID").write_text("an idle-arm cell moved NVMe data above %.2f GB/s (rho's baseline is contaminated): %s\n" % (IDLE_DRIVE_MAX_GBS, idle_dirty[:5])); rc = 3
    meta["idle_dirty"] = idle_dirty; meta["steady_state_by_pass"] = steady
    meta["visits"] = n_visits[0]; meta["retried_visits"] = n_retries[0]; meta["retry_fraction"] = n_retries[0] / max(1, n_visits[0])
    if smi: smi.stop()
    meta["cell_order"] = order_log; meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str)); dev.close(); return rc

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True); p.add_argument("--repo", default=str(HERE.parent.parent.parent))
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--check-only", action="store_true")
    p.add_argument("--with-nvme", action="store_true", help="run the nvme load arm (needs the drives idle and the reader's cores)")
    p.add_argument("--bounce-node", type=int, default=1); p.add_argument("--reader-cpus", default="")
    p.add_argument("--min-reader-gbs", type=float, default=1.0)
    p.add_argument("--drive-paths", nargs="+", default=list(DRIVE_PATHS), help="paths whose NVMe devices are the ones the idle-drive and load-match rules watch")
    p.add_argument("--skip-arms", nargs="*", default=[], choices=["hot", "repeat", "ce", "graph"],
                   help="DEVIATION from the registered arm list; recorded in meta.json. The frozen analysis still requires the primary arms.")
    a = p.parse_args(argv)
    if a.with_nvme and not a.reader_cpus: raise SystemExit("--with-nvme needs --reader-cpus (cores of the bounce node, not the harness's own)")
    return run(a)

if __name__ == "__main__":
    sys.exit(main())
