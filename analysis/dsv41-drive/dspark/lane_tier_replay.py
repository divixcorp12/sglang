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

Usage: lane_tier_replay.py gate OFF_DIR ON_DIR [--rows 190] [--json OUT]
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


def stream_features(r: Record) -> list[float]:
    m = r.gpu_nvme + r.gpu_pool
    return [1.0 if m else 0.0, r.gpu_nvme, r.gpu_pool, r.dma if m else 0.0]


class Prices:
    """Costs fit on one run's records (all its serving verifies), each in ms."""

    def __init__(self, run: Run):
        recs = run.serving(clean_only=False)
        self.stream = lstsq([stream_features(r) for r in recs], [r.stream for r in recs])
        hit = [(j, r) for r in recs for j in r.jobs if j["part"] == 0]
        miss = [(j, r) for r in recs for j in r.jobs if j["part"] == 1]
        self.hit_job = lstsq([[1.0, j["lanes"], j["routes"]] for j, _ in hit], [(j["end"] - j["start"]) / 1e6 for j, _ in hit])
        self.miss_job = lstsq([[1.0, j["routes"]] for j, _ in miss], [(j["end"] - j["start"]) / 1e6 for j, _ in miss])
        # a forced miss's job starts when its row has landed: start after the post, by its index on the team
        starts = collections.defaultdict(list)
        for j, r in miss:
            starts[(r.seq, j["team"])].append((j["start"] - r.post) / 1e6)
        xs, ys = [], []
        for v in starts.values():
            for i, t in enumerate(sorted(v), 1):
                xs.append([1.0, i])
                ys.append(t)
        self.miss_land = lstsq(xs, ys)  # first byte, per expert
        # the hit job's start after the post
        self.hit_start = statistics.median((j["start"] - r.post) / 1e6 for j, r in hit)
        # stream + wait against max(stream, CPU finish): the offset of the wait's end after the CPU's
        res = []
        for r in recs:
            if r.jobs:
                cpu = max((j["end"] - r.post) / 1e6 for j in r.jobs)
                res.append(r.stream + r.wait - max(r.stream, cpu))
        self.offset = statistics.median(res)

    def team_finish(self, hit_lanes: int, hit_routes: int, miss_routes: list[int]) -> float:
        t = self.hit_start + float(self.hit_job @ [1.0, hit_lanes, hit_routes]) if hit_lanes else 0.0
        for i, routes in enumerate(miss_routes, 1):
            land = float(self.miss_land @ [1.0, i])
            t = max(t, land) + float(self.miss_job @ [1.0, routes])
        return t

    def record(self, r: Record, gpu_nvme: int | None = None, gpu_pool: int | None = None) -> tuple[float, float]:
        """(stream, wait) predicted from the record's lane counts; ``gpu_nvme``/``gpu_pool`` override its tiers."""
        nv = r.gpu_nvme if gpu_nvme is None else gpu_nvme
        po = r.gpu_pool if gpu_pool is None else gpu_pool
        m = nv + po
        stream = float(self.stream @ [1.0 if m else 0.0, nv, po, r.dma if m else 0.0]) if m else 0.0
        cpu = 0.0
        for team in (0, 1):
            hits = [j for j in r.jobs if j["team"] == team and j["part"] == 0]
            misses = [j["routes"] for j in r.jobs if j["team"] == team and j["part"] == 1]
            lanes = sum(j["lanes"] for j in hits)
            routes = sum(j["routes"] for j in hits)
            if hits or misses:
                cpu = max(cpu, self.team_finish(lanes, routes, misses))
        wait = max(0.0, max(stream, cpu) + self.offset - stream) if (cpu or stream) else 0.0
        return stream, wait

    def describe(self) -> dict:
        return {"stream_ms": dict(zip(("base", "per_nvme_lane", "per_pool_lane", "per_copy_lane"), self.stream.round(3).tolist())),
                "hit_job_ms": dict(zip(("base", "per_lane", "per_route"), self.hit_job.round(3).tolist())),
                "miss_job_ms": dict(zip(("base", "per_route"), self.miss_job.round(3).tolist())),
                "miss_land_ms": dict(zip(("first_byte", "per_expert"), self.miss_land.round(3).tolist())),
                "hit_start_ms": round(self.hit_start, 3), "wait_offset_ms": round(self.offset, 3)}


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


class RamTier:
    """One layer's pinned slots: LRU, protected rows never evicted (ram_prefetch_replay.RamTier, lru)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stamp: dict[int, int] = {}
        self.tick = 0

    def has(self, e: int) -> bool:
        return e in self.stamp

    def touch(self, e: int) -> None:
        self.tick += 1
        self.stamp[e] = self.tick

    def insert(self, e: int, protected: set) -> None:
        if e not in self.stamp and len(self.stamp) >= self.capacity:
            cands = [(t, x) for x, t in self.stamp.items() if x not in protected]
            if cands:
                del self.stamp[min(cands)[1]]
        self.touch(e)


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


def tier_replay(run: Run, rows: int, layers: int) -> tuple[dict, dict]:
    """Drive the tier with the run's own routes and hot sets, and classify each record's GPU lanes (the experts it
    inserted into VRAM: hot at v+1 minus hot at v, its unforced lanes) as RAM-warm or NVMe-cold. Returns per record
    seq the predicted (gpu_nvme, gpu_pool) and the agreement counts."""
    header, log = route_log(run.dir)
    off = align(run, log)
    by_vr = {(r.verify, r.row): r for r in run.records}
    tiers = [RamTier(rows) for _ in range(layers)]
    # full layers (SGLANG_MOE_PINNED_HOST_FULL_LAYERS) hold every expert
    full = set(header.get("pinned_full_layers") or [0, 1])
    pred: dict[int, tuple[int, int]] = {}
    stats = collections.Counter()
    for v in range(len(run.verifies)):
        i = v + off
        if not 0 <= i < len(log):
            continue
        cur = log[i]
        nxt = log[i + 1] if i + 1 < len(log) else None
        for L in range(layers):
            routed = {e for e in cur["routes"][L] if e >= 0}
            hot = set(cur["hot"][L])
            misses = routed - hot
            r = by_vr.get((v, L))
            cold_pred = set() if L in full else {e for e in misses if not tiers[L].has(e)}
            if r is not None and nxt is not None:
                inserted = (set(nxt["hot"][L]) - hot) & misses
                pool = set(r.pool)
                nv = len((inserted & cold_pred) - pool)
                po = len(inserted & pool)
                pred[r.seq] = (nv, po)
                if run.prompt_of[v] >= 0 and run.clean[run.prompt_of[v]]:
                    stats["records"] += 1
                    stats["pred_gpu_nvme"] += nv
                    stats["obs_gpu_nvme"] += r.gpu_nvme
                    stats["inserted"] += len(inserted)
                    stats["obs_gpu_lanes"] += r.gpu_nvme + r.gpu_pool + r.dma
                    stats["pred_cold_misses"] += len(cold_pred)
                    stats["obs_nvme_reads"] += len(set(r.evict_in))
            if L not in full:
                for e in misses:
                    if tiers[L].has(e):
                        tiers[L].touch(e)
                    else:
                        tiers[L].insert(e, hot | routed)
    stats["align_offset"] = off
    return pred, dict(stats)


# ---- the gate ----------------------------------------------------------------------------------------------------------


def within(x: float, target: float, rel: float) -> bool:
    return abs(x - target) <= abs(target) * rel


def gate(off_dir: str, on_dir: str, rows: int) -> dict:
    off, on = Run(off_dir), Run(on_dir)
    # serve_compare's pair verdict: prompts with no flag in either run
    both = [a and b for a, b in zip(off.clean, on.clean)]
    off.clean, on.clean = list(both), list(both)
    layers = max(r.row for r in off.records) + 1
    prices = Prices(off)
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
    if not rows:
        # calibrate the capacity on OFF alone: the replay's NVMe reads per verify against the trace's
        sweep = {}
        for cand in range(120, 330, 10):
            _, s = tier_replay(off, cand, layers)
            sweep[cand] = s["pred_cold_misses"] / max(1, s["obs_nvme_reads"])
        rows = min(sweep, key=lambda c: abs(sweep[c] - 1.0))
        res["tier_calibration"] = {"rows": rows, "pred_over_obs_reads": {k: round(v, 3) for k, v in sweep.items()}}
    t_off, s_off = tier_replay(off, rows, layers)
    t_on, s_on = tier_replay(on, rows, layers)
    nv_off = s_off["pred_gpu_nvme"] / off.n_verifies()
    nv_on = s_on["pred_gpu_nvme"] / on.n_verifies()
    ob_off = s_off["obs_gpu_nvme"] / off.n_verifies()
    ob_on = s_on["obs_gpu_nvme"] / on.n_verifies()
    res["tier_stats"] = {"off": s_off, "on": s_on}
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
    res["diagnosis"] = diagnose(off, on, prices, rows, layers)
    return res


def diagnose(off: Run, on: Run, prices: Prices, rows: int, layers: int) -> dict:
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
        pred, _ = tier_replay(run, rows, layers)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("off")
    g.add_argument("on")
    g.add_argument("--rows", type=int, default=0, help="RAM rows per layer; 0 calibrates on OFF")
    g.add_argument("--json")
    a = ap.parse_args()
    res = gate(a.off, a.on, a.rows)
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
