"""Lanes priced by RAM tier: why the lane head's +19.5 VRAM hits/verify bought no ms/token, and what a tier-aware key
could buy. Step 1 (this file): the validation gate on the lane-head A/B (lane-head-ab-20261016).

A layer's gather plans its VRAM misses as lanes in key order (expert_route_plan.cuh). The first ``forced_from`` lanes
take VRAM victims and are inserted; the rest are forced CPU lanes (ram_slot_map.type_lanes). An unforced lane whose
row is in the pinned RAM tier is a hit the split model gives to the CPU or the copy engine; an unforced RAM miss is a
MISS_GPU lane, which the stream kernel S copies while the NVMe read lands (device stamps stream_start..stream_end). A
forced RAM miss is read by the host and computed by the CPU after its hit job. The GPU then waits for the CPU lanes
(wait_start..wait_end, which follows stream_end).

Per record (one layer of one verify) this joins, from the job trace:
  tier: ``evict`` (c = the expert a demand miss read in), ``miss_expert`` (a = a MISS_CPU lane's expert), ``pool_point``
    (a = a miss the prefetch pool served);
  copy engine: ``copy_submit`` per NUMA group (seq = the CPU team's first job, a = its last forced-miss job, b = forced
    misses, c = CPU-hit lane mask), ``copy_issue`` (b = DMA lanes);
  CPU: ``cpu_submit`` (a = 0 hit part, 1 forced-miss part), ``cpu_shape`` (a = token rows, b = lanes, c = routes),
    ``cpu_start``/``cpu_end``;
  device stamps (globaltimer, mapped to CLOCK_MONOTONIC by the draft-clock sample).
GPU NVMe lanes = evict incoming - MISS_CPU experts - pool-served experts.

Pricing (fit on one run, applied to another): the stream S of a record is linear in its GPU NVMe and pool lanes (and
its copy lanes, which share the link); each CPU job's time is linear in its shape, by part; a team's finish is its hit
job, then each forced miss after it lands (first byte + per expert, as the split model has it); the record's
stream + wait is the later of S and the CPU finish, plus a fitted offset.

RAM tier (``--tier``): per layer an LRU over ``rows`` slots with the VRAM-hot and the record's routed experts
protected (ram_prefetch_replay.RamTier's rule), touched by every routed expert that is not a VRAM hit, the pool's
served rows entering as the capture shows them. Its prediction of each lane's tier is checked against the trace's.

Usage: lane_tier_replay.py gate OFF_DIR ON_DIR [--bootstrap 200] [--json OUT]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_compare  # noqa: E402


def _one(d: str, pattern: str) -> str:
    (path,) = glob.glob(os.path.join(d, pattern))
    return path


def _events(paths: list[str], kinds: set):
    for path in paths:
        with open(path) as f:
            for line in f:
                if '"event"' not in line:
                    continue
                e = json.loads(line)
                if e.get("event") in kinds:
                    yield path, e


class Record:
    __slots__ = ("seq", "row", "post", "stream", "wait", "evict_in", "miss_cpu", "pool", "dma", "forced_miss",
                 "cpu_hit", "jobs", "verify")

    def __init__(self, seq: int):
        self.seq = seq
        self.row = -1
        self.post = 0
        self.stream = 0.0
        self.wait = 0.0
        self.evict_in: list[int] = []
        self.miss_cpu: list[int] = []
        self.pool: list[int] = []
        self.dma = 0
        self.forced_miss = [0, 0]
        self.cpu_hit = [0, 0]
        self.jobs: list[dict] = []  # per CPU job: team, part, lanes, rows, routes, submit, start, end (ns)
        self.verify = -1

    @property
    def gpu_nvme(self) -> int:
        return len(set(self.evict_in) - set(self.miss_cpu) - set(self.pool))

    @property
    def gpu_pool(self) -> int:
        return len(set(self.pool) - set(self.miss_cpu))


class Run:
    """One served run's records, verifies and serving prompts."""

    def __init__(self, d: str):
        self.dir = d
        self.verifies, _ = serve_compare.load_verifies(d)
        self.prompts = serve_compare.prompts(d, 1.3, 1.2)
        self.clean = [not (p["stall"] or p["cpu_slow"]) for p in self.prompts]
        offset = serve_compare.gpu_offset(d)
        recs: dict[int, Record] = {}
        with open(_one(d, "events.*.device-stamps.jsonl")) as f:
            f.readline()
            for line in f:
                s = json.loads(line)
                if not s["post_start"]:
                    continue
                r = recs.setdefault(s["seq"], Record(s["seq"]))
                r.post = s["post_start"] - offset
                r.stream = (s["stream_end"] - s["stream_start"]) / 1e6 if s["stream_end"] else 0.0
                r.wait = (s["wait_end"] - s["wait_start"]) / 1e6 if s["wait_end"] else 0.0
        tier = glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl"))
        for _, e in _events(tier, {"evict", "miss_expert", "pool_point", "svc_record"}):
            r = recs.get(e["seq"])
            if r is None:
                continue
            r.row = e["row"]
            k = e["event"]
            if k == "evict":
                r.evict_in.append(e["c"])
            elif k == "miss_expert":
                r.miss_cpu.append(e["a"])
            elif k == "pool_point":
                r.pool.append(e["a"])
        ranges = {}  # (team, cpu seq) -> record seq
        copy = glob.glob(os.path.join(d, "events.*.exl3-copy-eng*.jsonl"))
        for _, e in _events(copy, {"copy_submit", "copy_issue"}):
            r = recs.get(e["gen"])
            if r is None:
                continue
            g = e["group"]
            if e["event"] == "copy_issue":
                r.dma += max(e["b"], 0)
                continue
            r.forced_miss[g] += e["b"]
            r.cpu_hit[g] += bin(e["c"] & 0xFFFFFFFF).count("1")
            for s in range(e["seq"], max(e["seq"], e["a"]) + 1):
                ranges[(g, s)] = e["gen"]
        jobs = collections.defaultdict(dict)
        for path, e in _events(glob.glob(os.path.join(d, "events.*.exl3-cpu-exp*.jsonl")),
                               {"cpu_submit", "cpu_shape", "cpu_start", "cpu_end"}):
            team = int(path.split("cpu-exp")[1].split(".")[0])
            jobs[(team, e["seq"])][e["event"]] = e
        self.unjoined_jobs = 0
        for (team, seq), j in jobs.items():
            if len(j) < 4:
                continue
            r = recs.get(ranges.get((team, seq)))
            if r is None:
                self.unjoined_jobs += 1
                continue
            r.jobs.append({"team": team, "part": j["cpu_submit"]["a"], "lanes": j["cpu_shape"]["b"],
                           "rows": j["cpu_shape"]["a"], "routes": j["cpu_shape"]["c"], "submit": j["cpu_submit"]["ns"],
                           "start": j["cpu_start"]["ns"], "end": j["cpu_end"]["ns"]})
        begins = [b for b, _ in self.verifies]
        for r in recs.values():
            k = bisect.bisect_right(begins, r.post) - 1
            if k >= 0 and r.post <= self.verifies[k][1]:
                r.verify = k
        self.records = sorted((r for r in recs.values() if r.verify >= 0 and r.row >= 0), key=lambda r: r.seq)
        # serving prompt of each verify (serve_compare's sessions), and whether it is clean
        session, s = [], 0
        for i, (b, _) in enumerate(self.verifies):
            if i and b - self.verifies[i - 1][1] > 1_000_000_000:
                s += 1
            session.append(s)
        ids = sorted(set(session))[-len(self.prompts):]
        index = {sid: i for i, sid in enumerate(ids)}
        self.prompt_of = [index.get(sid, -1) for sid in session]

    def serving(self, clean_only: bool = True) -> list[Record]:
        return [r for r in self.records if self.prompt_of[r.verify] >= 0
                and (not clean_only or self.clean[self.prompt_of[r.verify]])]

    def n_verifies(self, clean_only: bool = True) -> int:
        return sum(p["verifies"] for p, c in zip(self.prompts, self.clean) if c or not clean_only)


# ---- pricing ---------------------------------------------------------------------------------------------------------


def lstsq(rows: list[list[float]], ys: list[float]) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(a, np.asarray(ys, dtype=np.float64), rcond=None)
    return coef


def cpu_misses(r: Record) -> int:
    return sum(1 for j in r.jobs if j["part"] == 1)


def stream_features(r: Record, gpu_nvme: int, gpu_pool: int) -> list[float]:
    """S's features: whether it runs, its NVMe and pool lanes, the copy lanes sharing the link, and the shared drives:
    the record's forced CPU misses read while S's NVMe lanes are (each GPU NVMe lane times each CPU miss)."""
    m = gpu_nvme + gpu_pool
    on = 1.0 if m else 0.0
    return [on, gpu_nvme, gpu_pool, r.dma * on, gpu_nvme * cpu_misses(r)]


STREAM_NAMES = ("base", "per_nvme_gpu_lane", "per_pool_gpu_lane", "per_copy_lane", "per_nvme_gpu_lane_x_cpu_miss")
LAND_NAMES = ("first_byte", "per_cpu_miss", "per_record_gpu_nvme_lane")


class Prices:
    """Costs fit on one run's serving records, each in ms. The drives are shared: a forced CPU miss lands later for each
    GPU NVMe lane of its record (``miss_land``'s last term), and S runs longer for each CPU miss read beside its NVMe
    lanes (``stream``'s last term). ``spread`` is each coefficient's standard deviation over a bootstrap of verifies."""

    def __init__(self, run: Run, bootstrap: int = 0):
        recs = run.serving(clean_only=False)
        self._fit(recs)
        self.spread = {}
        if bootstrap:
            by_v = collections.defaultdict(list)
            for r in recs:
                by_v[r.verify].append(r)
            keys = sorted(by_v)
            rng = np.random.default_rng(0)
            draws = []
            for _ in range(bootstrap):
                sample = [r for k in rng.choice(keys, size=len(keys)) for r in by_v[k]]
                boot = Prices.__new__(Prices)
                boot._fit(sample)
                draws.append(boot._vector())
            sd = np.std(np.asarray(draws), axis=0)
            self.spread = dict(zip(self._names(), sd.round(3).tolist()))

    def _fit(self, recs: list[Record]) -> None:
        self.stream = lstsq([stream_features(r, r.gpu_nvme, r.gpu_pool) for r in recs], [r.stream for r in recs])
        hit = [(j, r) for r in recs for j in r.jobs if j["part"] == 0]
        miss = [(j, r) for r in recs for j in r.jobs if j["part"] == 1]
        self.hit_job = lstsq([[1.0, j["lanes"], j["routes"]] for j, _ in hit], [(j["end"] - j["start"]) / 1e6 for j, _ in hit])
        self.miss_job = lstsq([[1.0, j["routes"]] for j, _ in miss], [(j["end"] - j["start"]) / 1e6 for j, _ in miss])
        # a forced miss's job starts when its row has landed: by its index on the team and its record's GPU NVMe lanes
        starts = collections.defaultdict(list)
        for j, r in miss:
            starts[(r.seq, j["team"])].append(((j["start"] - r.post) / 1e6, r.gpu_nvme))
        xs, ys = [], []
        for v in starts.values():
            for i, (t, g) in enumerate(sorted(v), 1):
                xs.append([1.0, i, g])
                ys.append(t)
        self.miss_land = lstsq(xs, ys)
        self.hit_start = statistics.median((j["start"] - r.post) / 1e6 for j, r in hit)
        # stream + wait against max(stream, CPU finish): the offset of the wait's end after the CPU's
        res = []
        for r in recs:
            if r.jobs:
                cpu = max((j["end"] - r.post) / 1e6 for j in r.jobs)
                res.append(r.stream + r.wait - max(r.stream, cpu))
        self.offset = statistics.median(res)

    def _names(self) -> list[str]:
        return ([f"stream.{n}" for n in STREAM_NAMES] + ["hit_job.base", "hit_job.per_lane", "hit_job.per_route",
                "miss_job.base", "miss_job.per_route"] + [f"miss_land.{n}" for n in LAND_NAMES]
                + ["hit_start", "wait_offset"])

    def _vector(self) -> list[float]:
        return [*self.stream, *self.hit_job, *self.miss_job, *self.miss_land, self.hit_start, self.offset]

    def team_finish(self, hit_lanes: int, hit_routes: int, miss_routes: list[int], gpu_nvme: int) -> float:
        t = self.hit_start + float(self.hit_job @ [1.0, hit_lanes, hit_routes]) if hit_lanes else 0.0
        for i, routes in enumerate(miss_routes, 1):
            land = float(self.miss_land @ [1.0, i, gpu_nvme])
            t = max(t, land) + float(self.miss_job @ [1.0, routes])
        return t

    def record(self, r: Record, gpu_nvme: int | None = None, gpu_pool: int | None = None) -> tuple[float, float]:
        """(stream, wait) predicted from the record's lane counts; ``gpu_nvme``/``gpu_pool`` override its tiers."""
        nv = r.gpu_nvme if gpu_nvme is None else gpu_nvme
        po = r.gpu_pool if gpu_pool is None else gpu_pool
        stream = float(self.stream @ stream_features(r, nv, po)) if nv + po else 0.0
        cpu = 0.0
        for team in (0, 1):
            hits = [j for j in r.jobs if j["team"] == team and j["part"] == 0]
            misses = [j["routes"] for j in r.jobs if j["team"] == team and j["part"] == 1]
            lanes = sum(j["lanes"] for j in hits)
            routes = sum(j["routes"] for j in hits)
            if hits or misses:
                cpu = max(cpu, self.team_finish(lanes, routes, misses, nv))
        wait = max(0.0, max(stream, cpu) + self.offset - stream) if (cpu or stream) else 0.0
        return stream, wait

    def describe(self) -> dict:
        out = {n: round(float(v), 3) for n, v in zip(self._names(), self._vector())}
        return {"ms": out, "bootstrap_sd": self.spread} if self.spread else {"ms": out}


def per_verify(run: Run, prices: Prices | None = None, tiers: dict | None = None) -> dict:
    """Clean serving verifies: measured or predicted stream, wait and stream layers per verify."""
    recs = run.serving()
    n = run.n_verifies()
    out = collections.Counter()
    for r in recs:
        if prices is None:
            s, w = r.stream, r.wait
            layer = r.stream > serve_compare.STREAM_LAYER_NS / 1e6
        else:
            nv, po = (tiers.get(r.seq) if tiers else None) or (None, None)
            s, w = prices.record(r, nv, po)
            layer = (r.gpu_nvme if nv is None else nv) + (r.gpu_pool if po is None else po) > 0
        out["stream"] += s
        out["wait"] += w
        out["stream_layers"] += layer
    return {k: v / n for k, v in out.items()}


# ---- RAM tier ----------------------------------------------------------------------------------------------------------


class NodeTier:
    """One NUMA group's slots of one row (TIER_RULES 1-4): LRU over its READY slots, VRAM-hot and the record's routed
    experts never victims, a promoted pool row inheriting its victim's recency."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stamp: dict[int, int] = {}

    def has(self, e: int) -> bool:
        return e in self.stamp

    def _victim(self, protected: set) -> int | None:
        cands = [(t, x) for x, t in self.stamp.items() if x not in protected]
        return min(cands)[1] if cands else None

    def insert(self, e: int, tick: int, protected: set) -> None:
        """A demand miss read in: a free slot, else the LRU unprotected victim; none, and it is not cached."""
        if e in self.stamp:
            self.stamp[e] = tick
            return
        if len(self.stamp) >= self.capacity:
            victim = self._victim(protected)
            if victim is None:
                return
            del self.stamp[victim]
        self.stamp[e] = tick

    def promote(self, e: int, protected: set) -> None:
        """A landed pool row mapped by promote_landed: it takes a victim and that victim's stamp."""
        if e in self.stamp or not self.stamp:
            return
        victim = self._victim(protected)
        if victim is None:
            return
        self.stamp[e] = self.stamp.pop(victim)

    def apply_eviction(self, old: int, new: int, tick: int) -> None:
        """An eviction the trace recorded (outside the replayed verifies): ``old`` out (-1 a free slot), ``new`` in."""
        self.stamp.pop(old, None)
        if len(self.stamp) < self.capacity or new in self.stamp:
            self.stamp[new] = tick


# The RAM tier's rules, from the code the A/B ran (lane-head-ab d269f7aa90; paths under
# python/sglang/kernels/jit/csrc/moe/expert_stream/host/ unless given):
#  1. Per NUMA group: a row's slots are split into one contiguous range per node (srt/layers/moe/host_numa.py:300
#     group_ranges, from where the pinned slabs' pages are bound, host_numa.py:267 slot_nodes), and expert e is served by
#     group e % nodes (lease_layout.h:52 home; :148 static_assert "home is expert % nodes"). A victim is taken only in
#     the miss's own group range (ram_tier.h:2114 take_victim_locked, loop over own.lo..own.hi).
#  2. Each group range first gives min(8, range - 1) staging slots (ram_tier.h:1044 reserve_staging, k = the victim
#     lanes, SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES=8) and then `share` speculative pool slots
#     (ram_tier.h:1084 reserve_spec_pool; "RAM prefetch share 2" in the server log); neither ever holds a mapped row.
#  3. Victim: a free slot first, else the least recently stamped READY slot whose expert is not VRAM-hot (the record's
#     hot bitmap, ram_tier.h:594 apply_gpu_hot), not routed by the record (`wanted`) and not filling
#     (ram_tier.h:2120-2128). So a VRAM-resident expert that holds a RAM row keeps it while it is hot; there is no
#     rule that frees it. With no victim the miss is read into staging but not cached (ram_tier.h:2110-2113).
#  4. Recency: every routed expert of a record that the group holds is restamped (ram_tier.h:2100 touch_request over
#     request.protect, the record's routed experts), VRAM hits included; a demand miss restamps its new slot.
#  5. Every demand miss is cached, GPU lane or CPU lane alike: forced CPU misses first, then the other misses, each
#     taking a victim (ram_tier.h:2998 reserve_victims_locked). The lane's kind decides who computes it, not whether
#     the row enters RAM.
#  6. Prefetch: an L+2 read lands in a pool slot (kSpec), not in the map. A demand miss can be served from it: a GPU
#     lane streams from the pool slot (ram_tier.h:3017 point_pooled_locked) and a forced CPU miss takes it over
#     (take_pooled_locked); either way the row is cached as rule 5. An unused landed row may be promoted into the map
#     when its row's delta is claimed (ram_tier.h:2919 promote_landed): it takes an LRU victim (no record protection)
#     and inherits the victim's stamp (ram_tier.h:2959), so it is the next to go unless routed. Promotion is skipped
#     when the row's lock is busy, the read is still in flight, or the delta has no room (spec_promote_* counters).
#  7. Full layers (SGLANG_MOE_PINNED_HOST_FULL_LAYERS) map every expert and never miss.
TIER_RULES = 7


def numa_capacities(d: str) -> tuple[dict[int, int], float, int, list[int]]:
    """From the server log's pinned-tier startup line: rows per layer, node 0's share of the bound bytes, the pool share,
    and the full layers."""
    rows, frac, share, full = {}, 0.5, 0, []
    with open(os.path.join(d, "server.log"), errors="replace") as f:
        for line in f:
            if "Pinned host expert cache startup" in line:
                info = json.loads(line[line.index("{"):])
                rows = {int(k): v for k, v in info["rows_per_layer"].items()}
                b = info["numa"]["bound_bytes"]
                frac = int(b["0"]) / (int(b["0"]) + int(b["1"]))
                full = info.get("full_layers", [])
            elif "RAM prefetch share" in line:
                share = int(line.split("RAM prefetch share")[1].split(",")[0])
    return rows, frac, share, full


def layer_tiers(rows: int, frac: float, share: int, staging: int = 8) -> list[NodeTier]:
    r0 = int(round(rows * frac))
    out = []
    for r in (r0, rows - r0):
        out.append(NodeTier(max(0, r - min(staging, r - 1) - share)))
    return out


def route_log(d: str) -> tuple[dict, list[dict]]:
    header, recs = None, []
    with open(os.path.join(d, "stages.jsonl")) as f:
        for line in f:
            if '"graph_routes' not in line:
                continue
            r = json.loads(line)
            if r.get("kind") == "graph_routes_header":
                header = r
            elif r.get("kind") == "graph_routes" and r.get("phase") == "target_verify" and "hot" in r:
                recs.append(r)
    return header, recs


def align(run: Run, log: list[dict]) -> int:
    """The route-log index of verify window 0: the offset whose per-verify miss totals best match the records' lanes."""
    by_v = collections.Counter()
    for r in run.records:
        by_v[r.verify] += len(set(r.evict_in)) + r.dma + sum(r.cpu_hit) + len(set(r.pool) - set(r.evict_in))
    best, best_score = 0, -1.0
    for off in range(-3, 4):
        pairs = [(sum(log[v + off]["misses"]), by_v[v]) for v in by_v if 0 <= v + off < len(log)]
        if len(pairs) < 10:
            continue
        a, b = np.array(pairs, dtype=float).T
        score = float(np.corrcoef(a, b)[0, 1])
        if score > best_score:
            best, best_score = off, score
    return best


def promote_fraction(d: str) -> float:
    """The share of landed pool rows the run promoted (counters.json: spec_promoted_early / spec_landed)."""
    with open(os.path.join(d, "counters.json")) as f:
        c = json.load(f)
    return c.get("spec_promoted_early", 0) / max(1, c.get("spec_landed", 1))


def _hash01(*xs: int) -> float:
    h = 1469598103934665603
    for x in xs:
        h = ((h ^ (x & 0xFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (h >> 11) / float(1 << 53)


def tier_replay(run: Run, layers: int, rules: bool = True) -> tuple[dict, dict]:
    """Drive the RAM tier through the run (TIER_RULES; ``rules`` False is the first gate's plain LRU of 170 rows) and
    classify each record's GPU lanes (the experts it inserted into VRAM: its unforced lanes) as NVMe-cold, pool-served
    or RAM-warm. Outside the replayed verifies (prefill, warm-up, eager forwards) the trace's own evictions are applied,
    since the lane key does not touch them. Returns per record seq the predicted (gpu_nvme, gpu_pool) and counts."""
    header, log = route_log(run.dir)
    off = align(run, log)
    rows, frac, share, full = numa_capacities(run.dir)
    p_promote = promote_fraction(run.dir)
    nodes = 2
    tiers = [layer_tiers(rows.get(L, 210), frac, share) if rules else [NodeTier(170 // 2), NodeTier(170 - 170 // 2)]
             for L in range(layers)]
    full = set(full)
    by_vr = {(r.verify, r.row): r for r in run.records}
    replayed = {r.seq for r in run.records}
    timeline = []  # (ns, order, kind, payload)
    for v in range(len(run.verifies)):
        i = v + off
        if not 0 <= i < len(log):
            continue
        for L in range(layers):
            r = by_vr.get((v, L))
            ns = r.post if r is not None else run.verifies[v][0] + L * 5_000_000
            timeline.append((ns, 1, "gather", (v, i, L)))
    for _, e in _events(glob.glob(os.path.join(run.dir, "events.*.exl3-tier*.jsonl")), {"evict"}):
        if e["seq"] not in replayed and rules:
            timeline.append((e["ns"], 0, "evict", (e["row"], e["group"], e["a"], e["c"])))
    lands = collections.defaultdict(list)  # (row, verify index after landing) -> experts
    if rules:
        begins = [b for b, _ in run.verifies]
        for _, e in _events(glob.glob(os.path.join(run.dir, "events.*.exl3-spec*.jsonl")), {"spec_land"}):
            k = bisect.bisect_right(begins, e["ns"])
            lands[(e["row"], k)].append((e["a"], e["ns"]))
            lands[(e["row"], k - 1)].append((e["a"], e["ns"]))
    timeline.sort(key=lambda t: (t[0], t[1]))
    pred: dict[int, tuple[int, int]] = {}
    stats = collections.Counter()
    tick = 0
    for ns, _, kind, payload in timeline:
        tick += 1
        if kind == "evict":
            row, g, old, new = payload
            if row < layers and row not in full:
                tiers[row][g].apply_eviction(old, new, tick)
            continue
        v, i, L = payload
        cur = log[i]
        nxt = log[i + 1] if i + 1 < len(log) else None
        routed = {e for e in cur["routes"][L] if e >= 0}
        hot = set(cur["hot"][L])
        misses = routed - hot
        r = by_vr.get((v, L))
        cold_pred = set() if L in full else {e for e in misses if not tiers[L][e % nodes].has(e)}
        if r is not None and nxt is not None:
            inserted = (set(nxt["hot"][L]) - hot) & misses
            pool = set(r.pool)
            nv = len((inserted & cold_pred) - pool)
            po = len(inserted & pool)
            pred[r.seq] = (nv, po)
            if run.prompt_of[v] >= 0 and run.clean[run.prompt_of[v]]:
                obs_cold = set(r.evict_in) - set(r.miss_cpu) - pool
                stats["lane_tp"] += len(obs_cold & ((inserted & cold_pred) - pool))
                stats["records"] += 1
                stats["pred_gpu_nvme"] += nv
                stats["obs_gpu_nvme"] += r.gpu_nvme
                stats["inserted"] += len(inserted)
                stats["obs_gpu_lanes"] += r.gpu_nvme + r.gpu_pool + r.dma
                stats["pred_cold_misses"] += len(cold_pred)
                stats["obs_nvme_reads"] += len(set(r.evict_in))
        if L in full:
            continue
        protected = hot | routed
        for e in routed:  # rule 4: the group's held routed experts are restamped, VRAM hits included
            t = tiers[L][e % nodes]
            if t.has(e):
                t.stamp[e] = tick
        for e in sorted(misses, key=lambda x: x not in cold_pred):  # rule 5: every miss is cached
            tiers[L][e % nodes].insert(e, tick, protected)
        if rules and r is not None:
            for e, land_ns in lands.get((L, v), []):  # rule 6: landed pool rows, before this gather, unused by it
                if land_ns <= ns and e not in routed and _hash01(v, L, e) < p_promote:
                    before = tiers[L][e % nodes].has(e)
                    tiers[L][e % nodes].promote(e, hot)
                    stats["sim_promotions"] += not before and tiers[L][e % nodes].has(e)
    stats["align_offset"] = off
    stats["promote_fraction"] = round(p_promote, 3)
    stats["capacities"] = {L: [t.capacity for t in tiers[L]] for L in (2, 39)}
    return pred, dict(stats)


# ---- the gate ----------------------------------------------------------------------------------------------------------


def within(x: float, target: float, rel: float) -> bool:
    return abs(x - target) <= abs(target) * rel


def gate(off_dir: str, on_dir: str, bootstrap: int) -> dict:
    off, on = Run(off_dir), Run(on_dir)
    # serve_compare's pair verdict: prompts with no flag in either run
    both = [a and b for a, b in zip(off.clean, on.clean)]
    off.clean, on.clean = list(both), list(both)
    layers = max(r.row for r in off.records) + 1
    prices = Prices(off, bootstrap)
    m_off, m_on = per_verify(off), per_verify(on)
    p_off, p_on = per_verify(off, prices), per_verify(on, prices)
    res = {"prices_fit_on_off": prices.describe(),
           "unjoined_cpu_jobs": {"off": off.unjoined_jobs, "on": on.unjoined_jobs},
           "measured": {"off": m_off, "on": m_on, "delta": {k: m_on[k] - m_off[k] for k in m_off}},
           "priced_observed_lanes": {"off": p_off, "on": p_on, "delta": {k: p_on[k] - p_off[k] for k in p_off}}}
    dm, dp = res["measured"]["delta"], res["priced_observed_lanes"]["delta"]
    res["P"] = {
        "stream_delta": [round(dp["stream"], 2), round(dm["stream"], 2), within(dp["stream"], dm["stream"], 0.20)],
        "wait_delta": [round(dp["wait"], 2), round(dm["wait"], 2), within(dp["wait"], dm["wait"], 0.20)],
        "stream_layers_off": [round(p_off["stream_layers"], 2), round(m_off["stream_layers"], 2),
                              abs(p_off["stream_layers"] - m_off["stream_layers"]) <= 1.0],
        "stream_layers_on": [round(p_on["stream_layers"], 2), round(m_on["stream_layers"], 2),
                             abs(p_on["stream_layers"] - m_on["stream_layers"]) <= 1.0],
        "off_in_sample_stream": [round(p_off["stream"], 2), round(m_off["stream"], 2), within(p_off["stream"], m_off["stream"], 0.10)],
        "off_in_sample_wait": [round(p_off["wait"], 2), round(m_off["wait"], 2), within(p_off["wait"], m_off["wait"], 0.10)],
    }
    t_off, s_off = tier_replay(off, layers)
    t_on, s_on = tier_replay(on, layers)
    nv_off = s_off["pred_gpu_nvme"] / off.n_verifies()
    nv_on = s_on["pred_gpu_nvme"] / on.n_verifies()
    ob_off = s_off["obs_gpu_nvme"] / off.n_verifies()
    ob_on = s_on["obs_gpu_nvme"] / on.n_verifies()
    res["tier_stats"] = {"off": s_off, "on": s_on}
    res["tier_error"] = {
        "off_pct": round(100 * (nv_off - ob_off) / ob_off, 1), "on_pct": round(100 * (nv_on - ob_on) / ob_on, 1),
        "delta_pct": round(100 * ((nv_on - nv_off) - (ob_on - ob_off)) / (ob_on - ob_off), 1),
        "lane_precision_off": round(s_off["lane_tp"] / max(1, s_off["pred_gpu_nvme"]), 3),
        "lane_recall_off": round(s_off["lane_tp"] / max(1, s_off["obs_gpu_nvme"]), 3),
        "lane_precision_on": round(s_on["lane_tp"] / max(1, s_on["pred_gpu_nvme"]), 3),
        "lane_recall_on": round(s_on["lane_tp"] / max(1, s_on["obs_gpu_nvme"]), 3),
    }
    _, p_off = tier_replay(off, layers, rules=False)
    _, p_on = tier_replay(on, layers, rules=False)
    res["plain_lru_170_for_reference"] = {"off": round(p_off["pred_gpu_nvme"] / off.n_verifies(), 2),
                                          "on": round(p_on["pred_gpu_nvme"] / on.n_verifies(), 2)}
    res["T"] = {
        "gpu_nvme_off": [round(nv_off, 2), round(ob_off, 2), within(nv_off, ob_off, 0.15)],
        "gpu_nvme_on": [round(nv_on, 2), round(ob_on, 2), within(nv_on, ob_on, 0.15)],
        "gpu_nvme_delta": [round(nv_on - nv_off, 2), round(ob_on - ob_off, 2), within(nv_on - nv_off, ob_on - ob_off, 0.20)],
    }
    e_off, e_on = per_verify(off, prices, t_off), per_verify(on, prices, t_on)
    res["E"] = {
        "stream_delta": [round(e_on["stream"] - e_off["stream"], 2), round(dm["stream"], 2),
                         within(e_on["stream"] - e_off["stream"], dm["stream"], 0.25)],
        "wait_delta": [round(e_on["wait"] - e_off["wait"], 2), round(dm["wait"], 2),
                       within(e_on["wait"] - e_off["wait"], dm["wait"], 0.25)],
    }
    res["passed"] = {part: all(v[-1] for v in res[part].values()) for part in ("P", "T", "E")}
    res["diagnosis"] = diagnose(off, on, prices, layers)
    return res


def diagnose(off: Run, on: Run, prices: Prices, layers: int) -> dict:
    """Where the gate's parts miss: the same prices fit on ON (does a cost move between runs?), the wait error by a
    record's GPU NVMe lanes, and the tier replay's misclassified GPU lanes by the verify's place after its prefill."""
    out = {"prices_fit_on_on": Prices(on).describe()}
    for name, run in (("off", off), ("on", on)):
        by = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
        for r in run.serving():
            s, w = prices.record(r)
            b = by[min(r.gpu_nvme + r.gpu_pool, 4)]
            b[0] += 1
            b[1] += r.wait
            b[2] += w
            b[3] += r.stream
            b[4] += s
        out[f"wait_by_gpu_miss_lanes_{name}"] = {k: {"records": v[0], "wait_ms": round(v[1] / v[0], 2),
                                                     "pred_wait_ms": round(v[2] / v[0], 2),
                                                     "stream_ms": round(v[3] / v[0], 2),
                                                     "pred_stream_ms": round(v[4] / v[0], 2)}
                                                 for k, v in sorted(by.items())}
        # the tier replay against the trace, lane by lane, by verifies since the request's first verify
        header, log = route_log(run.dir)
        off_i = align(run, log)
        pred, _ = tier_replay(run, layers)
        by_vr = {(r.verify, r.row): r for r in run.records}
        first = {}
        for v in range(len(run.verifies)):
            p = run.prompt_of[v]
            first.setdefault(p, v)
        acc = collections.defaultdict(collections.Counter)
        for (v, L), r in by_vr.items():
            if run.prompt_of[v] < 0 or r.seq not in pred:
                continue
            age = min(v - first[run.prompt_of[v]], 8)
            acc[age]["obs"] += r.gpu_nvme
            acc[age]["pred"] += pred[r.seq][0]
            acc[age]["records"] += 1
        out[f"tier_by_verify_after_prefill_{name}"] = {k: dict(v) for k, v in sorted(acc.items())}
    return out


def departures(d: str, layers: int) -> dict:
    """How each cold GPU lane's row had left RAM, from the trace alone. Decode evictions are traced (``evict``); a
    promotion's victim (promote_landed calls take_victim_locked without trace_evict, ram_tier.h:2949) and a prefill
    fill's or eager admission's victim (take_admit_slot_locked / take_slot_locked, ram_tier.h:1982, no event) are not.
    For each verify record's NVMe-cold GPU lane (expert e of row r read at t): ``traced`` when e's last traced event
    before t is an eviction; ``untraced_prefill`` when it is an insertion and a non-verify record (a prefill or eager
    forward) of the row ran between; ``untraced_decode`` when an insertion with no such record between (a promotion's
    victim, or a verify-time path the trace misses); ``never_seen`` when e has no traced event before t (startup or an
    earlier prefill fill put it in, or it never was)."""
    run = Run(d)
    verify_seqs = {r.seq for r in run.records}
    hist = collections.defaultdict(list)  # (row, expert) -> [(ns, +1 in / -1 out)]
    nonverify = collections.defaultdict(list)  # row -> ns of non-verify records
    for _, e in _events(glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl")), {"evict", "svc_record"}):
        if e["event"] == "svc_record":
            if e["seq"] not in verify_seqs:
                nonverify[e["row"]].append(e["ns"])
            continue
        hist[(e["row"], e["c"])].append((e["ns"], 1))
        if e["a"] >= 0:
            hist[(e["row"], e["a"])].append((e["ns"], -1))
    for v in hist.values():
        v.sort()
    for v in nonverify.values():
        v.sort()
    first_in = {}
    out = {"all": collections.Counter(), "clean": collections.Counter()}
    for _, e in _events(glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl")), {"evict"}):
        if e["seq"] not in verify_seqs:
            continue
        first_in.setdefault(e["seq"], []).append((e["ns"], e["row"], e["c"]))
    by_seq = {r.seq: r for r in run.records}
    for seq, reads in first_in.items():
        r = by_seq.get(seq)
        if r is None or run.prompt_of[r.verify] < 0:
            continue
        gpu_cold = set(r.evict_in) - set(r.miss_cpu) - set(r.pool)
        clean = run.clean[run.prompt_of[r.verify]]
        for ns, row, ex in reads:
            if ex not in gpu_cold:
                continue
            h = [x for x in hist[(row, ex)] if x[0] < ns]
            if not h:
                kind = "never_seen"
            elif h[-1][1] < 0:
                kind = "traced"
            else:
                t_in = h[-1][0]
                nv = nonverify.get(row, [])
                k = bisect.bisect_right(nv, t_in)
                kind = "untraced_prefill" if k < len(nv) and nv[k] < ns else "untraced_decode"
            out["all"][kind] += 1
            if clean:
                out["clean"][kind] += 1
    return {k: dict(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("off")
    g.add_argument("on")
    g.add_argument("--bootstrap", type=int, default=200, help="resamples of verifies for the prices' spread")
    g.add_argument("--json")
    dp = sub.add_parser("departures")
    dp.add_argument("runs", nargs="+")
    a = ap.parse_args()
    if a.cmd == "departures":
        print(json.dumps({d: departures(d, 40) for d in a.runs}, indent=1))
        return 0
    res = gate(a.off, a.on, a.bootstrap)
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
