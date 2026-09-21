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
  * STEADY STATE: nimbus_beacon_node, op-reth and aggregate-runner are the user's permanent services and are NOT stopped. Their CPU and the box's
    NVMe traffic are recorded per cell (box_foreign_cores, foreign_top, foreign_proc_io from /proc/<pid>/io where readable, drive_by_dev) and
    the nvme ratio is refused (results.INVALID) when foreign CPU (1.0 core) or foreign NVMe traffic (0.10 GB/s) DIFFERS between the idle and load
    arms of a pass. A steady load cancels in the ratio; a drifting one invalidates it.
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
FOREIGN_DRIVE_MAX_DELTA_GBS = 0.10  # per pass: foreign NVMe traffic (reads+writes, the reader's own bytes removed) may differ by at most this between idle and load cells
WATCH_COMM = ("nimbus_beacon_n", "op-reth", "reth-binary", "aggregate-runner")   # the box's steady-state services (the user's own); their I/O is sampled per cell
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

class ForeignLoad:
    """Foreign CPU use on the cores WE use, over one window.

    Every thread of every process that is not ours is sampled twice (utime + stime and the CPU it last ran on). A thread counts
    if it ran on one of `cores` at either sample. Returns (worst, name): the largest per-core SUM of foreign thread CPU
    (percent of one core) over `cores`, and the biggest contributor on that core. A permanent daemon on a core we do not use
    (nimbus on core 29, say) does not count; the same daemon migrating onto one of our cores does. Own pids, and the reader's,
    are excluded."""
    def __init__(self, own_pids, cores):
        self.own = set(own_pids); self.cores = set(cores); self.hz = os.sysconf("SC_CLK_TCK")
    def add_own(self, pid): self.own.add(pid)
    last = {}
    def _snap(self):
        out = {}
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) in self.own: continue
            try: tids = os.listdir("/proc/%s/task" % d)
            except OSError: continue
            for t in tids:
                try:
                    f = Path("/proc/%s/task/%s/stat" % (d, t)).read_text()
                    name = f[f.index("(") + 1:f.rindex(")")]; rest = f[f.rindex(")") + 2:].split()
                    out[(int(d), int(t))] = (int(rest[11]) + int(rest[12]), int(rest[36]), name)
                except (OSError, ValueError, IndexError): pass
        return out
    def start(self):
        self.a = self._snap(); self.t = time.monotonic()
    def stop(self):
        b = self._snap(); dt = time.monotonic() - self.t; per_core = collections.defaultdict(float); top = {}
        box = 0.0; by_proc = collections.defaultdict(float); names = {}
        for key, (j, cpu, name) in b.items():
            if key not in self.a: continue
            j0, cpu0, _ = self.a[key]; pct = 100.0 * (j - j0) / self.hz / dt
            if pct > 0 and not name.startswith(KERNEL_IO_THREADS):     # kernel threads that carry OUR I/O (softirq, kworker, nvme) are not foreign load
                box += pct; by_proc[key[0]] += pct
                if key[1] == key[0] or key[0] not in names: names[key[0]] = name      # the process name, not whichever thread was seen last
            for c in {cpu, cpu0}:
                if c in self.cores and pct > 0:
                    per_core[c] += pct
                    if pct > top.get(c, (0.0, ""))[0]: top[c] = (pct, "%s[%d/%d]" % (name, key[0], key[1]))
        self.last = {"box_foreign_cores": box / 100.0, "top": [(names[p_], p_, round(v, 1)) for p_, v in sorted(by_proc.items(), key=lambda kv: -kv[1])[:5]],
                     "loadavg": Path("/proc/loadavg").read_text().split()[:3]}
        if not per_core: return (0.0, "")
        c = max(per_core, key=per_core.get); return (per_core[c], "cpu%d: %s" % (c, top[c][1]))

class _NoForeign:
    last = {"box_foreign_cores": 0.0, "top": [], "loadavg": ["0", "0", "0"]}
    def start(self): pass
    def stop(self): return (0.0, "")

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

class ProcIO:
    """Read/write bytes of the watched services (and any foreign process that was among the top CPU users), per visit, from /proc/<pid>/io.
    /proc/<pid>/io is readable only for our own uid; a process it cannot read is recorded as unreadable, never as zero."""
    def __init__(self, own): self.own = set(own); self.pids = {}
    def _refresh(self, extra=()):
        for d in os.listdir("/proc"):
            if d.isdigit() and int(d) not in self.own:
                try:
                    name = Path("/proc/%s/comm" % d).read_text().strip()
                except OSError: continue
                if name in WATCH_COMM or name in extra: self.pids[int(d)] = name
    def _read(self):
        out = {}
        for pid, name in list(self.pids.items()):
            try:
                f = dict(x.split(": ") for x in Path("/proc/%d/io" % pid).read_text().splitlines())
                out[pid] = (name, int(f["read_bytes"]), int(f["write_bytes"]))
            except (OSError, KeyError, ValueError): out[pid] = (name, None, None)
        return out
    def start(self, extra=()): self._refresh(extra); self.a = self._read(); self.t = time.monotonic()
    def stop(self):
        b = self._read(); dt = max(time.monotonic() - self.t, 1e-9); out = {}
        for pid, (name, r, w) in b.items():
            key = "%s[%d]" % (name, pid)
            if r is None or pid not in self.a or self.a[pid][1] is None: out[key] = "unreadable"
            else: out[key] = {"read_MB_s": round((r - self.a[pid][1]) / 1e6 / dt, 2), "write_MB_s": round((w - self.a[pid][2]) / 1e6 / dt, 2)}
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
        self.global_seq = {0: [], 1: []}
        for node in (0, 1):
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
        for node in (0, 1):
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
        self._verify(0); self._verify(1)
        meta["device"] = torch.cuda.get_device_name(self.dev); meta["torch"] = torch.__version__
        return meta
    def _plan(self, node, n, rows, slots):
        t = self.torch
        r = torch.zeros(6, dtype=torch.int64); r[:n] = torch.tensor(rows); s = torch.zeros(6, dtype=torch.int32); s[:n] = torch.tensor(slots)
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
        return T, {"cpu": os.sched_getcpu()}
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
    meta["file_sha256"] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("c_harness.py", "nvme_load_reader.py", "c_analysis.py") if (HERE / n).exists()}
    meta["repo"] = str(args.repo)
    try: meta["repo_head"] = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception: meta["repo_head"] = None
    meta.update(dev.setup(args))
    if args.check_only:
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str)); print("check-only ok"); return 0
    rng = random.Random(SEED)
    mycores = set(os.sched_getaffinity(0)) | set(_parse_cpus(args.reader_cpus))
    foreign = _NoForeign() if dry else ForeignLoad([os.getpid()], mycores); smi = None      # a dry run must not depend on the ambient load of the machine it runs on
    if not dry:
        smi = SmiSampler(os.getpid()); smi.start(); time.sleep(2.0)
        # untimed load so the link leaves Gen1 (L5), then one look at the state
        for _ in range(3): dev.run_visit(Cell("sm", "cold", 0, "idle", "eager", 6), 40, 0, args)
    reader = None
    if args.with_nvme:
        go, stop, ready, logp = [out / x for x in ("reader.go", "reader.stop", "reader.ready", "reader_log.json")]
        for f in (go, stop, ready): f.unlink(missing_ok=True)
        pin = [] if dry else ["numactl", "--membind=%d" % args.bounce_node, "taskset", "-c", args.reader_cpus]
        cmd = pin + [sys.executable, str(HERE / "nvme_load_reader.py"),
                     "--go", str(go), "--stop", str(stop), "--ready", str(ready), "--out", str(logp)] + (["--dry-run"] if dry else [])
        reader = subprocess.Popen(cmd); foreign.add_own(reader.pid) if not dry else None
        for _ in range(600):
            if ready.exists(): break
            time.sleep(0.5)
        else: reader.kill(); raise SystemExit("reader never became ready")
    lines = (out / "results.jsonl").open("w")
    skip = set(args.skip_arms)
    cells = [c for c in cell_list() if c.state not in skip and c.engine not in skip and c.launch not in skip]
    load_cells = load_cell_list() if args.with_nvme else []
    meta["skipped_arms"] = sorted(skip); meta["NOT_MEASURED"] = sorted(skip) + ([] if args.with_nvme else ["nvme"])      # a skipped arm is a declared deviation from the registered design and must be named in the report
    order_log = []; load_windows = []; load_drive = []; box_idle = []; box_load = []; box_shift = []
    drive_rows = []; top_pids = set(); pio = None if dry else ProcIO([os.getpid()]); load_drive_w = []; load_pass = []
    def visit(cell, sink):
        la0 = Path("/proc/loadavg").read_text().split()[:3]
        foreign.start(); ds0 = None if dry else nvme_rw()
        if pio: pio.start([t_[0] for t_ in foreign.last.get("top", [])[:3]] if foreign.last else [])
        t0 = time.monotonic()
        T, extra = dev.run_visit(cell, LAUNCHES_PER_CELL // 2, args)
        t1 = time.monotonic(); fp = foreign.stop(); ds1 = None if dry else nvme_rw(); extra = dict(extra)
        extra["foreign_proc_io"] = pio.stop() if pio else {}
        extra["foreign"] = dict(foreign.last); extra["loadavg_start"] = la0
        for t_ in extra["foreign"]["top"]: top_pids.add(t_[1])
        if dry: extra["foreign"]["box_foreign_cores"] = extra.get("dry_box", 0.0)
        extra["drive_read_bytes"] = None if dry else sum(ds1[k][0] - ds0.get(k, (0, 0))[0] for k in ds1)
        extra["drive_write_bytes"] = None if dry else sum(ds1[k][1] - ds0.get(k, (0, 0))[1] for k in ds1)
        extra["drive_by_dev"] = None if dry else {k: [ds1[k][0] - ds0.get(k, (0, 0))[0], ds1[k][1] - ds0.get(k, (0, 0))[1]] for k in ds1 if ds1[k] != ds0.get(k)}
        extra["drive_bytes"] = extra["drive_read_bytes"]
        sink.append((T, extra, t0, t1, fp))
    for p in range(PASSES):
        acc = collections.defaultdict(list)
        for cell, v in abba(cells, rng):
            visit(cell, acc[cell]); order_log.append((p, list(cell), v))
        load_acc = collections.defaultdict(list)
        if args.with_nvme:
            go.write_text("go\n"); time.sleep(0.5)
            for cell, v in abba(load_cells, rng):
                visit(cell, load_acc[cell]); order_log.append((p, list(cell), v)); load_windows.append((list(cell), load_acc[cell][-1][2], load_acc[cell][-1][3])); load_drive.append(load_acc[cell][-1][1]["drive_bytes"]); load_drive_w.append(load_acc[cell][-1][1]["drive_write_bytes"]); load_pass.append(p)
            go.unlink(); time.sleep(0.5)
        for cell, visits in list(acc.items()) + list(load_acc.items()):
            T = [x for v in visits for x in v[0]]
            t0 = min(v[2] for v in visits); t1 = max(v[3] for v in visits)
            cond = smi.window(t0, t1) if smi else {"link_gen_start": 3, "link_gen_end": 3, "pstate_start": 0, "other_gpu_procs": 0, "sm_mhz_min": 2900, "sm_mhz_max": 2900}
            if cond is None: raise SystemExit("no nvidia-smi samples for cell %s: refusing to write a cell without conditions" % (cell,))
            worst = max((v[4] for v in visits), key=lambda x: x[0])
            dur = sum(v[3] - v[2] for v in visits); db = None if dry else sum(v[1]["drive_bytes"] for v in visits)
            dw = None if dry else sum(v[1]["drive_write_bytes"] for v in visits)
            extra = {"cpu": visits[0][1].get("cpu"), "pass": p, "visits": len(visits), "drive_bytes": db, "drive_write_bytes": dw, "drive_gb_per_s": None if dry else db / 1e9 / max(dur, 1e-9),
                     "drive_by_dev": [v[1]["drive_by_dev"] for v in visits], "foreign_proc_io": [v[1]["foreign_proc_io"] for v in visits], "foreign_where": worst[1],
                     "box_foreign_cores": max(v[1]["foreign"]["box_foreign_cores"] for v in visits), "foreign_top": [v[1]["foreign"]["top"] for v in visits],
                     "loadavg_start": visits[0][1]["loadavg_start"], "loadavg_end": visits[-1][1]["foreign"]["loadavg"]}
            if cell.load == "idle" and not dry: drive_rows.append((p, (db + dw) / 1e9 / max(dur, 1e-9)))
            (box_load if cell.load == "nvme" else box_idle).append((p, extra["box_foreign_cores"]))
            lines.write(json.dumps(record(cell, p, T, dev.reuse_distance(cell.node) if cell.state != "repeat" else 10 ** 9, cond, worst, extra)) + "\n"); lines.flush()
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
        drift = []
        for g, (cell, t0, t1), db, dw, pp in zip(gbs, load_windows, load_drive, load_drive_w, load_pass):
            rb = g["reader_gb_per_s"] * (t1 - t0) * 1e9
            g["drive_bytes"] = db; g["foreign_io_fraction"] = None if (dry or rb <= 0) else (db - rb) / rb; g["pass"] = pp
            g["foreign_drive_gb_per_s"] = None if dry else ((db + dw) - rb) / 1e9 / max(t1 - t0, 1e-9)
        meta["load_windows"] = gbs
        if not dry:                                   # foreign NVMe traffic (reads + writes, the reader's own bytes removed) must not differ between the arms of the ratio
            for pp in range(PASSES):
                i_ = [x for q, x in drive_rows if q == pp]; l_ = [g["foreign_drive_gb_per_s"] for g in gbs if g["pass"] == pp]
                if i_ and l_ and abs(S.mean(l_) - S.mean(i_)) > FOREIGN_DRIVE_MAX_DELTA_GBS: drift.append((pp, round(S.mean(i_), 3), round(S.mean(l_), 3)))
            meta["foreign_drive_gb_per_s_idle_vs_load_by_pass"] = [(pp, round(S.mean([x for q, x in drive_rows if q == pp] or [0]), 3), round(S.mean([g["foreign_drive_gb_per_s"] for g in gbs if g["pass"] == pp] or [0]), 3)) for pp in range(PASSES)]
            if drift: (out / "results.INVALID").write_text("foreign NVMe traffic changed by more than %.2f GB/s between the idle and load arms in pass(es) %s: the nvme ratio is invalid\n" % (FOREIGN_DRIVE_MAX_DELTA_GBS, drift)); rc = 3
        if not dry and any(g["foreign_io_fraction"] is None or abs(g["foreign_io_fraction"]) > LOAD_FOREIGN_IO_MAX for g in gbs):
            (out / "results.INVALID").write_text("drive traffic in a load window differs from the reader's own bytes by more than %.0f%%: another lane used the drives; the nvme arm's rho is not to be quoted\n" % (100 * LOAD_FOREIGN_IO_MAX)); rc = 3
        if any(g["reader_gb_per_s"] < args.min_reader_gbs for g in gbs) and not dry:
            (out / "results.INVALID").write_text("the nvme load arm did not load the drives at >= %.1f GB/s in every window: no load-arm number is to be quoted\n" % args.min_reader_gbs); rc = 3
    for p_ in range(PASSES):
        i_ = [x for q, x in box_idle if q == p_]; l_ = [x for q, x in box_load if q == p_]
        if i_ and l_ and abs(S.mean(l_) - S.mean(i_)) > BOX_FOREIGN_MAX_DELTA: box_shift.append((p_, round(S.mean(i_), 2), round(S.mean(l_), 2)))
    meta["box_foreign_cores_idle_vs_load_by_pass"] = [(p_, round(S.mean([x for q, x in box_idle if q == p_] or [0]), 2), round(S.mean([x for q, x in box_load if q == p_] or [0]), 2)) for p_ in range(PASSES)]
    if box_shift:
        (out / "results.INVALID").write_text("foreign CPU (non-own, kernel I/O threads excluded) changed by more than %.1f cores between the idle and load arms in pass(es) %s: the nvme ratio is invalid\n" % (BOX_FOREIGN_MAX_DELTA, box_shift)); rc = 3
    meta["steady_state"] = _cmdlines(top_pids)
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
    p.add_argument("--skip-arms", nargs="*", default=[], choices=["hot", "repeat", "ce", "graph"],
                   help="DEVIATION from the registered arm list; recorded in meta.json. The frozen analysis still requires the primary arms.")
    a = p.parse_args(argv)
    if a.with_nvme and not a.reader_cpus: raise SystemExit("--with-nvme needs --reader-cpus (cores of the bounce node, not the harness's own)")
    return run(a)

if __name__ == "__main__":
    sys.exit(main())
