"""Offline replay of the CPU split's miss cut on a traced DSpark capture: today's flat cut against a graded one.

Today (SGLANG_DSV41_CPU_SPLIT_MISS_CUT=1, _MAX=3; ram_slot_map.type_lanes and lease_device.cuh type_lanes) a node
with 1..3 forced CPU misses takes one fewer of its split's CPU-hit lanes, and that lane goes to the copy engine. The
graded cut sizes the cut per node from what the device knows when it types the lanes: the node's forced misses m,
its split take t, its forced CPU hits f and the record's DMA lanes. It predicts that miss i (1-based) lands at
``first_byte + per_expert * i`` and that the hit job costs ``cpu_lane`` per lane, and cuts the most lanes that still
shorten the node's predicted end, counting ``copy_lane`` per lane on the copy engine's FIFO (``graded``). ``first``
is the literal rule: keep the lanes that end the hit job when the first miss lands, ignoring the copy side.
``graded+`` and ``first+`` never cut less than today's flat cut. ``oracle`` picks each record's best cut pair with
the measured landings, every node included (so it also retunes the split); ``oracle-miss`` varies only nodes with
1..cap misses, and ``oracle-miss-up`` those never below the flat cut. Oracles are bounds, not rules.

What the replay takes from the job trace, per record (row, gen) and NUMA group: ``copy_submit`` (seq..a = the
record's CPU job sequences, b = CPU misses, c = CPU-hit lane mask), ``copy_issue`` (b = DMA lanes),
``copy_dma_observed``, ``gate_open``, the team's ``cpu_submit``/``cpu_start``/``cpu_end`` (a = part: 0 the hit job, 1
a batch of landed CPU misses, so a part-1 submit is a landing), ``svc_record`` (the request seq the device stamps
use) and, when present (InstrBuild on branch expert-breakdown), ``lane_kind`` (b & 255 = kind, | 256 when spilled).
Without lane_kind the forced CPU hits are inferred from the split table (``infer_forced_hits``).

The model: a team runs its record's jobs in order, each at max(its submit, the previous end); the copy engine runs
both groups' DMA lanes as one FIFO in issue order at ``copy_lane`` per lane; the gate opens when the last side ends.
A counterfactual moves lanes between a node's hit job and the FIFO, rescales the hit job's measured length by its
lane count, and shifts the measured gate by the modelled change. The GPU's gate wait shrinks by that shift, down to
zero. Layers are independent: a saving does not move later layers' reads. Times are traced: read the savings as
relative to this capture's own wall time, not as untraced milliseconds.

Usage: misscut_replay.py CAPTURE [--skip 8] [--tokens-per-verify 3.6] [--split 0,1,2,...] [--json OUT]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import glob
import itertools
import json
import os
import statistics
import sys

MS = 1e6
WAIT_NS = 50_000
KIND_DMA, KIND_CPU_HIT, KIND_CPU_MISS = 1, 3, 5
SPILL = 256
ROW_GROUPS = ((0, 1), (2, 9), (10, 19), (20, 29), (30, 39))
SWEPT = ("first", "first+", "graded", "graded+")


@dataclasses.dataclass
class Rule:
    """The device's inputs to a graded cut, in ms (traced)."""

    first_byte: float = 1.1
    per_expert: float = 2.3
    cpu_lane: float = 0.72
    miss_lane: float = 0.60
    copy_lane: float = 1.1
    cap: int = 3


@dataclasses.dataclass
class Group:
    m: int  # CPU misses (forced: no VRAM victim)
    hits: int  # CPU hits in the hit job, forced ones included
    dma: int  # DMA lanes
    forced_hits: int
    take: int  # split take before any cut
    cut: int  # the flat cut the capture applied
    issue: float  # ms
    dma_end: float
    hit: tuple | None  # (start, dur) of the hit job, ms
    misses: list  # [(submit, dur)] of the miss jobs, ms, in sequence order
    free: float  # the team's first start for this record
    cpu_end: float  # measured
    hit_live: int = 0  # the hit job's live (token, expert) pairs (cpu_shape c)
    miss_live: list = dataclasses.field(default_factory=list)  # each miss job's


def split_take(split, n: int) -> int:
    return split[min(n, len(split) - 1)] if n > 0 else 0


def flat_cut(m: int, take: int, cut: int = 1, cut_max: int = 3) -> int:
    return min(take, cut) if cut > 0 and 1 <= m <= cut_max else 0


def infer_forced_hits(hits: int, dma: int, m: int, split) -> int:
    """The fewest forced CPU hits consistent with the split: unforced hits u = hits - f must equal the split's take
    for n = u + dma eligible lanes, less today's flat cut."""
    for f in range(hits + 1):
        u = hits - f
        take = split_take(split, u + dma)
        if u == take - flat_cut(m, take):
            return f
    return hits


def run_team(free: float, hit, misses) -> float:
    """The end of a team's record: the hit job (start, dur) or None, then each miss job (submit, dur) in order."""
    t = free
    if hit is not None:
        t = max(t, hit[0]) + hit[1]
    for submit, dur in misses:
        t = max(t, submit) + dur
    return t


def dma_fifo(issues, lanes, copy_lane: float):
    """Each group's DMA end when the copy engine runs every group's lanes as one FIFO in issue order."""
    ends = [0.0] * len(issues)
    free = float("-inf")
    for i in sorted(range(len(issues)), key=lambda i: issues[i]):
        if lanes[i] <= 0:
            ends[i] = issues[i]
            continue
        free = max(free, issues[i]) + copy_lane * lanes[i]
        ends[i] = free
    return ends


def predict_end(keep: int, m: int, rule: Rule, t0: float = 0.0) -> float:
    """The device's predicted CPU end: a hit job of `keep` lanes, then m misses landing at first_byte + per_expert*i."""
    t = t0 + rule.cpu_lane * keep
    for i in range(1, m + 1):
        t = max(t, t0 + rule.first_byte + rule.per_expert * i) + rule.miss_lane
    return t


def graded_cut(m: int, take: int, forced_hits: int, record_dma: int, rule: Rule, copy_side: bool = True) -> int:
    """The cut minimizing max(predicted CPU end, predicted copy-engine end); the smallest on a tie. record_dma is
    the record's DMA lanes on both nodes before this node's cut. No cut outside 1..cap misses."""
    if not 1 <= m <= rule.cap or take <= 0:
        return 0
    best, best_cost = 0, None
    for c in range(take + 1):
        cost = predict_end(take - c + forced_hits, m, rule)
        if copy_side:
            cost = max(cost, rule.copy_lane * (record_dma + c))
        if best_cost is None or cost < best_cost - 1e-9:
            best, best_cost = c, cost
    return best


def first_landing_cut(m: int, take: int, forced_hits: int, rule: Rule) -> int:
    """The literal rule: keep the lanes whose hit job ends when the first miss lands."""
    if not 1 <= m <= rule.cap or take <= 0:
        return 0
    keep = int((rule.first_byte + rule.per_expert) / rule.cpu_lane) - forced_hits
    return min(take, max(0, take - max(0, keep)))


def replay_record(groups: dict, cuts: dict, copy_lane: float, cpu_lane: dict) -> float:
    """The modelled gate (ms) with node g cut by cuts[g] instead of its captured cut."""
    issues, lanes, cpu = [], [], []
    for gi, g in sorted(groups.items()):
        delta = cuts[gi] - g.cut
        hits = g.hits - delta
        if g.hit is not None and g.hits > 0:
            hit = (g.hit[0], g.hit[1] * hits / g.hits) if hits > 0 else None
        else:
            hit = (g.free, cpu_lane[gi] * hits) if hits > 0 else None
        cpu.append(run_team(g.free, hit, g.misses) if (hit or g.misses) else float("-inf"))
        issues.append(g.issue)
        lanes.append(g.dma + delta)
    dma = dma_fifo(issues, lanes, copy_lane)
    return max(max(c, d if n > 0 else float("-inf")) for c, d, n in zip(cpu, dma, lanes))


def _events(d: str):
    by = collections.defaultdict(list)
    for path in glob.glob(os.path.join(d, "events.*.jsonl")):
        if path.endswith((".device-stamps.jsonl", ".verify.jsonl")):
            continue
        team = int(path.split("cpu-exp")[1].split(".")[0]) if "cpu-exp" in path else None
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if "dropped" in e and int(e["dropped"]):
                    raise ValueError(f"{path}: {e['dropped']} events dropped")
                if "event" in e:
                    if team is not None:
                        e["team"] = team
                    by[e["event"]].append(e)
    return by


def _stamps(d: str) -> dict:
    (clock,) = glob.glob(os.path.join(d, "events.*.draft-clock.json"))
    with open(clock) as f:
        s = min(json.load(f)["samples"], key=lambda s: s["after"] - s["before"])
    offset = s["gpu"] - (s["before"] + s["after"]) // 2
    (path,) = glob.glob(os.path.join(d, "events.*.device-stamps.jsonl"))
    out = {}
    with open(path) as f:
        points = json.loads(f.readline())["points"]
        for line in f:
            r = json.loads(line)
            out[r["seq"]] = {p: (r[p] - offset if r[p] else 0) for p in points}
    return out


def _verifies(d: str) -> list:
    (path,) = glob.glob(os.path.join(d, "events.*.verify.jsonl"))
    begins, ends = [], []
    with open(path) as f:
        f.readline()
        for e in map(json.loads, f):
            (begins if e["event"] == "verify_begin" else ends).append(e["ns"])
    return list(zip(begins, ends))


def load(d: str, split, ev: dict | None = None) -> tuple[list, dict]:
    """Records (dicts with row, verify, wait, gate, groups, lane_list) and the fitted costs. ``ev``: the capture's
    events, when the caller has loaded them already."""
    ev = _events(d) if ev is None else ev
    jobs = {}
    for kind in ("cpu_submit", "cpu_start", "cpu_end"):
        for e in ev[kind]:
            j = jobs.setdefault((e["team"], e["seq"]), {"part": e["a"], "k": e["b"]})
            j[kind] = e["ns"] / MS
    for e in ev["cpu_shape"]:  # c = the job's live (token, expert) pairs
        if (e["team"], e["seq"]) in jobs:
            jobs[(e["team"], e["seq"])]["live"] = e["c"]
    recs = collections.defaultdict(lambda: {"g": collections.defaultdict(dict), "lanes": collections.defaultdict(list),
                                            "lane_list": collections.defaultdict(list)})
    for e in ev["copy_submit"]:
        recs[(e["row"], e["gen"])]["g"][e["group"]].update(first=e["seq"], last=e["a"], m=e["b"], mask=e["c"])
    for e in ev["copy_issue"]:
        recs[(e["row"], e["gen"])]["g"][e["group"]].update(issue=e["ns"] / MS, dma=e["b"])
    for e in ev["copy_dma_observed"]:
        recs[(e["row"], e["gen"])]["g"][e["group"]]["dma_end"] = e["ns"] / MS
    for e in ev["copy_cpu_observed"]:
        recs[(e["row"], e["gen"])]["g"][e["group"]]["cpu_obs"] = e["ns"] / MS
    for e in ev["gate_open"]:
        recs[(e["row"], e["gen"])]["gate"] = e["ns"] / MS
    for e in ev["svc_record"]:
        r = recs[(e["row"], e["gen"])]
        r["seq"] = e["seq"]
        r["start"] = min(r.get("start", float("inf")), (e["ns"] - e["a"]) / MS)
    for e in ev["lane_kind"]:
        recs[(e["row"], e["gen"])]["lanes"][e["group"]].append(e["b"])
        recs[(e["row"], e["gen"])]["lane_list"][e["group"]].append((e["a"], e["b"], e["c"]))
    has_kinds = bool(ev["lane_kind"])

    # Per-lane costs (ms): CPU hit lanes per team from the hit jobs' lengths, misses from k=1 jobs, DMA from the
    # first group in the FIFO, alone on the link.
    cpu_lane, miss_lane = {}, {}
    for team in (0, 1):
        hit = [(j["cpu_end"] - j["cpu_start"]) / j["k"] for (t, _), j in jobs.items()
               if t == team and j["part"] == 0 and j["k"] > 0 and "cpu_end" in j and "cpu_start" in j]
        mis = [j["cpu_end"] - j["cpu_start"] for (t, _), j in jobs.items()
               if t == team and j["part"] == 1 and j["k"] == 1 and "cpu_end" in j and "cpu_start" in j]
        cpu_lane[team] = statistics.median(hit) if hit else 0.72
        miss_lane[team] = statistics.median(mis) if mis else 0.60
    solo = []
    for r in recs.values():
        gs = [g for g in r["g"].values() if g.get("dma") and "dma_end" in g and "issue" in g]
        if gs:
            g = min(gs, key=lambda g: g["issue"])
            solo.append((g["dma_end"] - g["issue"]) / g["dma"])
    copy_lane = statistics.median(solo) if solo else 1.0

    out = []
    for (row, gen), r in recs.items():
        if "seq" not in r or "gate" not in r or len(r["g"]) == 0:
            continue
        groups, ok = {}, True
        for gi, g in r["g"].items():
            if not {"first", "issue"} <= g.keys():
                ok = False
                break
            own = [jobs.get((gi, q)) for q in range(g["first"], g["last"] + 1)]
            own = [j for j in own if j and {"cpu_submit", "cpu_start", "cpu_end"} <= j.keys()]
            hit = next(((j["cpu_start"], j["cpu_end"] - j["cpu_start"]) for j in own if j["part"] == 0), None)
            hit_live = next((j.get("live", j["k"]) for j in own if j["part"] == 0), 0)
            misses = [(j["cpu_submit"], j["cpu_end"] - j["cpu_start"]) for j in own if j["part"] == 1]
            miss_live = [j.get("live", j["k"]) for j in own if j["part"] == 1]
            hits = bin(g["mask"]).count("1")
            if has_kinds:
                kinds = r["lanes"][gi]
                forced = sum(1 for b in kinds if b & 0xFF == KIND_CPU_HIT and b & SPILL)
            else:
                forced = infer_forced_hits(hits, g["dma"], g["m"], split)
            take = split_take(split, hits - forced + g["dma"])
            groups[gi] = Group(
                m=g["m"], hits=hits, dma=g["dma"], forced_hits=forced, take=take, cut=flat_cut(g["m"], take),
                issue=g["issue"], dma_end=g.get("dma_end", g["issue"]), hit=hit, misses=misses,
                free=min((j["cpu_start"] for j in own), default=g["issue"]),
                cpu_end=max((j["cpu_end"] for j in own), default=float("-inf")), hit_live=hit_live,
                miss_live=miss_live)
        if ok:
            out.append({"row": row, "gen": gen, "seq": r["seq"], "start": r.get("start", r["gate"]), "gate": r["gate"],
                        "groups": groups, "kinds": dict(r["lanes"]) if has_kinds else None,
                        "lane_list": dict(r["lane_list"]) if has_kinds else None})
    return out, {"cpu_lane": cpu_lane, "miss_lane": miss_lane, "copy_lane": copy_lane, "has_kinds": has_kinds}


def attach(d: str, recs: list, skip: int) -> list:
    """Each record's verify and GPU gate wait (ms); drops records outside the analysed verifies or unstamped."""
    stamps = _stamps(d)
    verifies = _verifies(d)
    starts = [b / MS for b, _ in verifies]
    out = []
    for r in recs:
        k = bisect.bisect_right(starts, r["start"]) - 1
        s = stamps.get(r["seq"])
        if k < skip or k >= len(verifies) or r["start"] > verifies[k][1] / MS or s is None or not s["wait_start"]:
            continue
        r["verify"] = k
        r["wait_start"] = s["wait_start"] / MS
        r["wait_end"] = s["wait_end"] / MS if s["wait_end"] else r["wait_start"]
        r["wait"] = r["wait_end"] - r["wait_start"]
        r["stream"] = (s["stream_end"] - s["stream_start"]) / MS if s["stream_end"] and s["stream_start"] else 0.0
        out.append(r)
    return out


def model_gate(r: dict, copy_lane: float, cpu_lane: dict, measured_jobs: bool = True, miss_lane=None) -> float:
    """The baseline gate: measured job lengths (structural) or fitted per-lane costs (parametric)."""
    if measured_jobs:
        return replay_record(r["groups"], {gi: g.cut for gi, g in r["groups"].items()}, copy_lane, cpu_lane)
    groups = {}
    for gi, g in r["groups"].items():
        hit = (g.hit[0] if g.hit else g.free, cpu_lane[gi] * g.hits) if g.hits else None
        misses = [(s, miss_lane[gi] * max(1, round(dur / miss_lane[gi]))) for s, dur in g.misses]
        groups[gi] = dataclasses.replace(g, hit=hit, misses=misses)
    return replay_record(groups, {gi: g.cut for gi, g in groups.items()}, copy_lane, cpu_lane)


def wait_deltas(recs: list) -> dict:
    """Per verify, the median of (GPU wait end - host gate_open) over its waited records, on the mapped clock; the
    capture's median where a verify has fewer than three. Per verify, because the mapping drifts."""
    by = collections.defaultdict(list)
    for r in recs:
        if r["wait"] > WAIT_NS / MS:
            by[r["verify"]].append(r["wait_end"] - r["gate"])
    overall = statistics.median(x for xs in by.values() for x in xs)
    return collections.defaultdict(lambda: overall, {k: statistics.median(xs) for k, xs in by.items() if len(xs) >= 3})


def fidelity(recs: list, costs: dict, delta: dict) -> dict:
    """Modelled vs measured gate and GPU gate wait, per layer and summed per verify."""
    out = {}
    for name, measured in (("structural", True), ("parametric", False)):
        gate_err, wait_err, pairs = [], [], []
        per_v = collections.defaultdict(lambda: [0.0, 0.0])
        for r in recs:
            g = model_gate(r, costs["copy_lane"], costs["cpu_lane"], measured, costs["miss_lane"])
            gate_err.append(g - r["gate"])
            w = max(0.0, g + delta[r["verify"]] - r["wait_start"])
            wait_err.append(w - r["wait"])
            pairs.append((w, r["wait"]))
            per_v[r["verify"]][0] += w
            per_v[r["verify"]][1] += r["wait"]
        ve = [a - b for a, b in per_v.values()]
        mw = [b for _, b in per_v.values()]
        out[name] = {
            "gate_err_ms": _dist(gate_err),
            "wait_err_ms": _dist(wait_err),
            "wait_abs_err_ms_mean": round(statistics.mean(abs(x) for x in wait_err), 3),
            "wait_corr": round(_corr(pairs), 3),
            "measured_wait_ms_mean": round(statistics.mean(r["wait"] for r in recs), 3),
            "verify_wait_err_ms": _dist(ve),
            "verify_wait_measured_ms_mean": round(statistics.mean(mw), 2),
        }
    return out


def _dist(xs) -> dict:
    xs = sorted(xs)
    q = lambda p: round(xs[int(p * (len(xs) - 1))], 3)
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3), "p10": q(.1), "p50": q(.5), "p90": q(.9)}


def _corr(pairs) -> float:
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    return statistics.correlation(xs, ys)


def policy_cuts(r: dict, policy: str, rule: Rule, costs: dict) -> dict:
    gs = r["groups"]
    uncut_dma = {gi: g.dma - g.cut for gi, g in gs.items()}
    cuts = {}
    for gi, g in gs.items():
        other = sum(v for k, v in uncut_dma.items() if k != gi)
        if policy == "flat":
            cuts[gi] = g.cut
        elif policy == "none":
            cuts[gi] = 0
        elif policy.startswith("flat"):
            cuts[gi] = flat_cut(g.m, g.take, int(policy[4:]), rule.cap)
        elif policy == "graded":
            cuts[gi] = graded_cut(g.m, g.take, g.forced_hits, uncut_dma[gi] + other, rule)
        elif policy == "graded-nocopy":
            cuts[gi] = graded_cut(g.m, g.take, g.forced_hits, 0, rule, copy_side=False)
        elif policy == "graded+":
            cuts[gi] = max(g.cut, graded_cut(g.m, g.take, g.forced_hits, uncut_dma[gi] + other, rule))
        elif policy == "first":
            cuts[gi] = first_landing_cut(g.m, g.take, g.forced_hits, rule)
        elif policy == "first+":
            cuts[gi] = max(g.cut, first_landing_cut(g.m, g.take, g.forced_hits, rule))
        else:
            raise ValueError(policy)
    return cuts


def oracle_cuts(r: dict, copy_lane: float, cpu_lane: dict, mode: str = "oracle", cap: int = 3, score=None) -> dict:
    """The record's best cut pair on the measured landings. ``oracle`` varies every node (split tuning included);
    ``oracle-miss`` only nodes with 1..cap misses; ``oracle-miss-up`` those, never below the flat cut. ``score``
    (cuts -> value, lower is better) replaces the replayed gate."""
    gs = sorted(r["groups"].items())

    def choices(g):
        if mode == "oracle":
            return range(g.take + 1)
        if not 1 <= g.m <= cap:
            return (g.cut,)
        return range(g.cut if mode == "oracle-miss-up" else 0, g.take + 1)

    best, best_gate = None, None
    for combo in itertools.product(*[choices(g) for _, g in gs]):
        cuts = {gi: c for (gi, _), c in zip(gs, combo)}
        gate = replay_record(r["groups"], cuts, copy_lane, cpu_lane) if score is None else score(cuts)
        if best_gate is None or gate < best_gate - 1e-9:
            best, best_gate = cuts, gate
    return best


STREAM_COUPLED_MS = 0.05
# Off by default: replaying split B on split A's capture (split-ab-20261014 A2, and A on B1) at coupling 0 puts B
# 2-4 ms/verify behind A, near the served pair (verify median +2.6, wall -1.4); 0.5 matches the measured stream
# growth but puts B 7-8 ms behind.
STREAM_COUPLING = 0.0


def layer_effect(r: dict, end: float, shift: float, dlanes: int, copy_lane: float, coupling: float = 1.0):
    """(saving, arrival shift, gate-wait change) of a layer whose host gate moves by `shift` while the copy FIFO
    gains `dlanes` lanes. `end` is the baseline gate as the GPU sees it. The GPU reaches the gate after its stream
    kernel (S), which runs only when the record has GPU NVMe misses and whose pulls share the link with the copy FIFO,
    so with S running its arrival moves by coupling * copy_lane per added lane, never earlier than S's start
    (STREAM_COUPLING has the calibration)."""
    arrive = r["wait_start"]
    if coupling and r.get("stream", 0.0) > STREAM_COUPLED_MS:
        arrive_cf = arrive + max(-r["stream"], coupling * copy_lane * dlanes)
    else:
        arrive_cf = arrive
    wait = max(0.0, end - arrive)
    wait_cf = max(0.0, end + shift - arrive_cf)
    return max(arrive, end) - max(arrive_cf, end + shift), arrive_cf - arrive, wait_cf - wait


def layer_saving(r: dict, end: float, shift: float, dlanes: int, copy_lane: float, coupling: float = 1.0) -> float:
    return layer_effect(r, end, shift, dlanes, copy_lane, coupling)[0]


def evaluate(recs: list, policy: str, rule: Rule, costs: dict, delta: dict, copy_lane: float | None = None,
             coupled: float = STREAM_COUPLING) -> dict:
    """Per-record saving (ms of GPU gate wait) of `policy` against the captured flat cut. delta: per verify, the
    measured wait end less the host's gate_open."""
    copy_lane = rule.copy_lane if copy_lane is None else copy_lane
    cpu_lane = costs["cpu_lane"]
    per_v = collections.Counter()
    by_rows = collections.Counter()
    by_m = collections.Counter()
    moved = collections.Counter()
    for r in recs:
        base = replay_record(r["groups"], {gi: g.cut for gi, g in r["groups"].items()}, copy_lane, cpu_lane)
        end = r["wait_end"] if r["wait"] > WAIT_NS / MS else r["gate"] + delta[r["verify"]]

        def save(cuts, r=r, base=base, end=end):
            shift = replay_record(r["groups"], cuts, copy_lane, cpu_lane) - base
            dlanes = sum(cuts[gi] - g.cut for gi, g in r["groups"].items())
            return layer_saving(r, end, shift, dlanes, copy_lane, coupled)

        if policy.startswith("oracle"):
            cuts = oracle_cuts(r, copy_lane, cpu_lane, policy, rule.cap,
                               score=lambda c: -save(c) + 1e-4 * sum(c.values()))
        else:
            cuts = policy_cuts(r, policy, rule, costs)
        saving = save(cuts)
        per_v[r["verify"]] += saving
        by_rows[_row_group(r["row"])] += saving
        crit = max(r["groups"].values(), key=lambda g: max(g.cpu_end, g.dma_end))
        by_m[min(crit.m, 4)] += saving
        for gi, g in r["groups"].items():
            moved[(min(g.m, 4), cuts[gi] - g.cut)] += 1
    nv = max(1, len(per_v))
    vs = list(per_v.values())
    return {
        "policy": policy,
        "ms_per_verify": round(sum(vs) / nv, 3),
        "verify_saving_p10_p90": [round(sorted(vs)[int(.1 * (len(vs) - 1))], 2), round(sorted(vs)[int(.9 * (len(vs) - 1))], 2)],
        "by_rows_ms_per_verify": {f"{a}-{b}": round(by_rows[(a, b)] / nv, 3) for a, b in ROW_GROUPS},
        "by_critical_node_misses_ms_per_verify": {("4+" if k == 4 else str(k)): round(v / nv, 3) for k, v in sorted(by_m.items())},
        "cut_change_by_misses": {f"m={'4+' if m == 4 else m} d={d:+d}": n for (m, d), n in sorted(moved.items()) if d},
    }


def _row_group(row: int):
    for a, b in ROW_GROUPS:
        if a <= row <= b:
            return (a, b)
    return ROW_GROUPS[-1]


def accounting_check(recs: list, split) -> dict:
    """With lane_kind: how often the inferred forced hits and the split's take agree with the typed lanes."""
    agree = total = take_ok = unforced_miss = 0
    for r in recs:
        if not r["kinds"]:
            continue
        for gi, g in r["groups"].items():
            kinds = r["kinds"].get(gi, [])
            total += 1
            agree += infer_forced_hits(g.hits, g.dma, g.m, split) == g.forced_hits
            take_ok += g.hits - g.forced_hits == g.take - g.cut
            unforced_miss += sum(1 for b in kinds if b & 0xFF == KIND_CPU_MISS and not b & SPILL)
    return {"groups": total, "inferred_forced_hits_agree": agree, "take_minus_cut_matches_unforced_hits": take_ok,
            "unforced_cpu_misses": unforced_miss}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture")
    p.add_argument("--skip", type=int, default=8)
    p.add_argument("--tokens-per-verify", type=float, default=3.6)
    p.add_argument("--split", default="0,1,2,2,3,3,4,5,5,6,7,7,8,9,9,10,11,11,12,13,13,14,15,15,16,17,17,18,19,19,"
                   "20,21,21,22,23,23,24,24,25,26,26", help="SGLANG_DSV41_CPU_EXPERTS_SPLIT of the capture")
    p.add_argument("--infer-forced", action="store_true", help="infer forced CPU hits even when lane_kind is traced")
    p.add_argument("--stream-coupling", type=float, default=STREAM_COUPLING,
                   help="the GPU stream kernel's delay per added copy lane, in copy lanes (layer_effect); 0: none")
    p.add_argument("--json")
    a = p.parse_args()
    split = [int(x) for x in a.split.split(",")]
    coupled = a.stream_coupling
    recs, costs = load(a.capture, split)
    recs = attach(a.capture, recs, a.skip)
    if not recs:
        print("refusing: no stamped records in the analysed verifies", file=sys.stderr)
        return 1
    accounting = accounting_check(recs, split)
    if a.infer_forced:
        for r in recs:
            for g in r["groups"].values():
                g.forced_hits = infer_forced_hits(g.hits, g.dma, g.m, split)
                g.take = split_take(split, g.hits - g.forced_hits + g.dma)
                g.cut = flat_cut(g.m, g.take)
    delta = wait_deltas(recs)
    tpv = a.tokens_per_verify
    nv = len({r["verify"] for r in recs})
    rule0 = Rule(cpu_lane=statistics.mean(costs["cpu_lane"].values()),
                 miss_lane=statistics.mean(costs["miss_lane"].values()), copy_lane=costs["copy_lane"])
    walls = [(e - b) / MS for b, e in _verifies(a.capture)[a.skip:]]
    res = {"capture": os.path.abspath(a.capture), "verifies": nv, "records": len(recs),
           "verify_wall_ms_mean": round(statistics.mean(walls), 1), "tokens_per_verify": tpv, "costs_ms": costs, "infer_forced": a.infer_forced,
           "stream_coupling": coupled,
           "gate_to_wait_end_ms": _dist(list(delta.values())),
           "accounting": accounting, "fidelity": fidelity(recs, costs, delta),
           "rule": dataclasses.asdict(rule0)}
    misses = collections.Counter(min(g.m, 4) for r in recs for g in r["groups"].values())
    res["node_records_by_misses"] = {("4+" if k == 4 else str(k)): v for k, v in sorted(misses.items())}

    policies = []
    for pol in ("none", "flat", "flat2", "flat3", "first", "first+", "graded-nocopy", "graded", "graded+",
                "oracle-miss-up", "oracle-miss", "oracle"):
        e = evaluate(recs, pol, rule0, costs, delta, coupled=coupled)
        e["ms_per_token"] = round(e["ms_per_verify"] / tpv, 3)
        policies.append(e)
    res["policies"] = policies

    sweep = []
    for fb, pe, cl, cap in itertools.product((0.5, 1.1, 2.0), (1.5, 2.3, 3.0), (costs["copy_lane"], 1.1, 1.4),
                                             (1, 2, 3, 4, 6)):
        rule = dataclasses.replace(rule0, first_byte=fb, per_expert=pe, copy_lane=cl, cap=cap)
        row = {"first_byte": fb, "per_expert": pe, "copy_lane": round(cl, 3), "cap": cap}
        for pol in SWEPT:
            row[pol] = evaluate(recs, pol, rule, costs, delta, coupled=coupled)["ms_per_verify"]
        if fb == 1.1 and pe == 2.3:
            row["oracle-miss"] = evaluate(recs, "oracle-miss", rule, costs, delta, coupled=coupled)["ms_per_verify"]
        sweep.append(row)
    res["sweep"] = sweep
    report(res)
    with open(a.json or os.path.join(a.capture, "misscut_replay.json"), "w") as f:
        json.dump(res, f, indent=1)
    return 0


def report(res: dict) -> None:
    print(f"{res['capture']}: {res['verifies']} verifies, {res['records']} records, "
          f"{res['tokens_per_verify']} tokens/verify, verify wall {res['verify_wall_ms_mean']} ms (mean, traced)")
    print("costs (ms, traced):", json.dumps(res["costs_ms"]), "infer_forced", res["infer_forced"])
    print("gate_open -> GPU wait end, per verify (ms, mapped clock):", json.dumps(res["gate_to_wait_end_ms"]))
    print("lane accounting:", json.dumps(res["accounting"]))
    print("node records by CPU misses:", json.dumps(res["node_records_by_misses"]))
    for name, f in res["fidelity"].items():
        print(f"fidelity {name}: {json.dumps(f)}")
    print("policies (ms saved per verify vs the captured flat cut; + is faster):")
    for e in res["policies"]:
        print(f"  {e['policy']:14s} {e['ms_per_verify']:+7.2f} ms/verify {e['ms_per_token']:+6.2f} ms/token "
              f"p10/p90 {e['verify_saving_p10_p90']}")
        print(f"    rows {json.dumps(e['by_rows_ms_per_verify'])}")
        print(f"    critical-node misses {json.dumps(e['by_critical_node_misses_ms_per_verify'])}")
        print(f"    cut changes {json.dumps(e['cut_change_by_misses'])}")
    print(f"sweep (ms/verify): first_byte per_expert copy_lane cap | {' '.join(SWEPT)} [oracle-miss]")
    for s in res["sweep"]:
        extra = f" {s['oracle-miss']:+.2f}" if "oracle-miss" in s else ""
        cells = " ".join(f"{s[p]:+6.2f}" for p in SWEPT)
        print(f"  {s['first_byte']:4} {s['per_expert']:4} {s['copy_lane']:5} {s['cap']} | {cells}{extra}")


if __name__ == "__main__":
    sys.exit(main())
