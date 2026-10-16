"""Offline replay of a model-based CPU split against today's static split table plus the flat miss cut.

Today (lease_device.cuh type_lanes) node n's CPU takes the last split[n][k] of its k eligible unforced hits in plan
order, less the miss cut. The model-based split instead picks each node's CPU hits per record to minimise the
predicted gate, max(cpu_finish[0], cpu_finish[1], copy_finish), from what the device has when it types the lanes:
each node's eligible hits and forced lanes, the tokens each expert is routed by in the record, the forced misses and a
landing estimate (miss i lands at ``first_byte + per_expert * i``), per-team CPU cost constants and the copy engine's
per-lane cost (one FIFO shared by both nodes).

Rules (``--rules``): ``exh`` searches every (take0, take1) with the CPU taking the fewest-token hits; ``greedy``
assigns hits heaviest-token first to whichever side (the hit's node's CPU or the copy FIFO) would finish earliest.
``*-count`` use count-only costs, the CPU taking the last hits in plan order as today. ``oracle`` is ``exh`` scored
on the replayed timeline (measured landings and job lengths): a bound, not a rule.

The replay is misscut_replay's: each team runs its record's jobs in order, the copy engine runs both groups' lanes
as one FIFO in issue order, a counterfactual rescales the hit job by the fitted cost of its new lanes and tokens, and
the GPU's gate wait shrinks by the modelled gate shift. ``--cpu-scale`` scales every CPU job (and the rule's CPU
costs) to model a capture whose CPU ran at another speed, such as untraced. Tokens per expert need the capture's
``stages.jsonl`` graph_routes and InstrBuild ``lane_kind``; without them only the count rules run.

Usage: split_replay.py CAPTURE [--skip 8] [--tokens-per-verify 3.6] [--cpu-scale traced=1,untraced=0.9] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_misses  # noqa: E402
import misscut_replay as mr  # noqa: E402

KIND_DMA, KIND_CPU_HIT, KIND_CPU_MISS = 1, 3, 5


@dataclasses.dataclass
class Cost:
    """A team's CPU job length in ms: a + b * experts + c * (token, expert) pairs."""

    a: float
    b: float
    c: float = 0.0

    def __call__(self, k: int, live: float) -> float:
        return self.a + self.b * k + self.c * live if k > 0 else 0.0


@dataclasses.dataclass
class Node:
    """One node of one record, as the device sees it, plus the replay's measured timeline."""

    team: int
    elig: list  # eligible unforced hits: (lane, tokens), in plan order
    today: frozenset  # the eligible lanes today's split put on the CPU
    forced_tok: list  # forced CPU hits' tokens
    miss_tok: list  # forced CPU misses' tokens
    g: mr.Group


@dataclasses.dataclass
class Params:
    first_byte: float = 0.8
    per_expert: float = 1.6
    copy_lane: float = 1.05
    # The stream-aware rule (greedy-stream): with GPU-streamed misses, the GPU's stream kernel ends at stream_base +
    # first_byte + per_expert * misses + stream_lane * copy lanes, a third side the copy option is charged with.
    stream_lane: float = 0.65
    stream_base: float = 0.0


def fit_costs(jobs: list, tokens: bool) -> dict:
    """Per team and part (0 hit job, 1 miss job): least squares of length on experts (and pairs)."""
    out = {}
    for team in (0, 1):
        for part in (0, 1):
            rows = [(j["k"], j["live"], j["dur"]) for j in jobs if j["team"] == team and j["part"] == part and j["k"]]
            k, live, dur = (np.array(x, dtype=float) for x in zip(*rows))
            if part == 1 or not tokens:
                x = np.stack([np.ones_like(k), live if part == 1 and tokens else k], 1)
            else:
                x = np.stack([np.ones_like(k), k, live], 1)
            coef, *_ = np.linalg.lstsq(x, dur, rcond=None)
            pred = x @ coef
            if part == 1:
                cost = Cost(coef[0], 0.0, coef[1]) if tokens else Cost(coef[0], coef[1])
            else:
                cost = Cost(*coef) if tokens else Cost(coef[0], coef[1])
            out[(team, part)] = {"cost": cost, "mae": float(np.mean(np.abs(pred - dur))), "n": len(dur)}
    return out


def node_finish(n: Node, cpu_lanes, costs: dict, p: Params, scale: float) -> float:
    """The rule's predicted finish of a node whose CPU takes eligible lanes `cpu_lanes` (lane -> tokens)."""
    k = len(cpu_lanes) + len(n.forced_tok)
    live = sum(cpu_lanes.values()) + sum(n.forced_tok)
    t = scale * costs[(n.team, 0)](k, live)
    for i, tok in enumerate(n.miss_tok, 1):
        t = max(t, p.first_byte + p.per_expert * i) + scale * costs[(n.team, 1)](1, tok)
    return t


def order(n: Node, tokens: bool) -> list:
    """The eligible lanes in the order the CPU takes them: fewest tokens first, or today's (last in plan order)."""
    if tokens:
        return sorted(n.elig, key=lambda e: (e[1], -e[0]))
    return sorted(n.elig, key=lambda e: -e[0])


def rule_exhaustive(nodes: list, costs: dict, p: Params, scale: float, tokens: bool, score=None,
                    gpu_miss: int = 0) -> list:
    """Each node's CPU lane set from the best (take0, take1); ties to the smaller sum of side finishes. With GPU
    misses the GPU's own arrival (its stream kernel's pulls, queued behind the copy FIFO) is a third side: one copy
    lane per GPU miss after the FIFO."""
    orders = [order(n, tokens) for n in nodes]
    best, best_cost = None, None
    for takes in itertools.product(*[range(len(o) + 1) for o in orders]):
        sets = [dict(o[:t]) for o, t in zip(orders, takes)]
        if score is not None:
            cost = score(sets)
        else:
            sides = [node_finish(n, s, costs, p, scale) for n, s in zip(nodes, sets)]
            copy = p.copy_lane * (sum(len(o) - t for o, t in zip(orders, takes)) + gpu_miss)
            gate = max(sides + [copy])
            cost = gate + 1e-3 * (sum(sides) + copy)
        if best_cost is None or cost < best_cost - 1e-12:
            best, best_cost = sets, cost
    return best


def rule_greedy(nodes: list, costs: dict, p: Params, scale: float, tokens: bool, gpu_miss: int = 0,
                streamed: int = 0) -> list:
    """Hits heaviest first (count-only: plan order, first lanes first), each to the side that would finish first.
    With `streamed` GPU-streamed misses and p.stream_lane > 0 (greedy-stream), the copy side also ends no earlier than
    the stream kernel would with one more copy lane."""
    sets = [{} for _ in nodes]
    copy = p.copy_lane * gpu_miss
    lanes = 0
    stream0 = p.stream_base + p.first_byte + p.per_expert * streamed
    hits = [(tok, i, lane) for i, n in enumerate(nodes) for lane, tok in n.elig]
    hits.sort(key=lambda h: (-h[0], h[2]) if tokens else (h[2],))
    for tok, i, lane in hits:
        cpu = node_finish(nodes[i], {**sets[i], lane: tok}, costs, p, scale)
        side = copy + p.copy_lane
        if streamed and p.stream_lane > 0:
            side = max(side, stream0 + p.stream_lane * (lanes + 1))
        if cpu <= side:
            sets[i][lane] = tok
        else:
            copy += p.copy_lane
            lanes += 1
    return sets


def residency_tiebreak(nodes: list, sets: list) -> list:
    """The split-model branch's residency tiebreak: per node and token count, keep how many lanes the CPU takes and
    give it the highest plan indexes (lowest residency keys), the copy engine the rest."""
    out = []
    for n, s in zip(nodes, sets):
        groups: dict = {}
        for lane, tok in n.elig:
            groups.setdefault(tok, []).append(lane)
        new = {}
        for tok, lanes in groups.items():
            k = sum(1 for lane in lanes if lane in s)
            for lane in sorted(lanes, reverse=True)[:k]:
                new[lane] = tok
        out.append(new)
    return out


def physics(r: dict, nodes: list, sets: list, costs: dict, copy_lane: float, scale: float) -> float:
    """The replayed gate (ms) when node i's CPU takes eligible lanes sets[i]: the measured hit job plus the fitted cost
    of its change, every CPU job scaled by `scale`, the copy FIFO at `copy_lane` per lane."""
    issues, lanes, cpu = [], [], []
    for n, s in zip(nodes, sets):
        g = n.g
        k0, live0 = len(n.today) + len(n.forced_tok), _live(n, n.today)
        k1, live1 = len(s) + len(n.forced_tok), sum(s.values()) + sum(n.forced_tok)
        f = costs[(n.team, 0)]
        if k1 == 0:
            hit = None
        elif g.hit is not None and k0 > 0:
            hit = (g.hit[0], scale * max(0.05, g.hit[1] + f(k1, live1) - f(k0, live0)))
        else:
            hit = (g.free, scale * f(k1, live1))
        misses = [(sub, scale * dur) for sub, dur in g.misses]
        cpu.append(mr.run_team(g.free, hit, misses) if (hit or misses) else float("-inf"))
        issues.append(g.issue)
        lanes.append(len(n.elig) - len(s))
    dma = mr.dma_fifo(issues, lanes, copy_lane)
    return max(max(c, d if k > 0 else float("-inf")) for c, d, k in zip(cpu, dma, lanes))


def _live(n: Node, lanes) -> float:
    tok = dict(n.elig)
    return sum(tok[x] for x in lanes)


def build_nodes(r: dict, routes: dict | None, live_mean: dict, split) -> list | None:
    """The record's nodes: from lane_kind and the routes when traced, else inferred from the split (count-only)."""
    nodes = []
    counts = routes.get((r["row"], r["gen"])) if routes is not None else None
    for gi, g in sorted(r["groups"].items()):
        if r["lane_list"] is not None and counts is not None:
            lanes = r["lane_list"].get(gi, [])
            elig, today, forced, miss = [], [], [], []
            for expert, b, j in lanes:
                tok = counts.get(expert, 1)
                kind, spill = b & 0xFF, bool(b & mr.SPILL)
                if kind == KIND_CPU_HIT and spill:
                    forced.append(tok)
                elif kind == KIND_CPU_HIT:
                    elig.append((j, tok))
                    today.append(j)
                elif kind == KIND_DMA:
                    elig.append((j, tok))
                elif kind == KIND_CPU_MISS:
                    miss.append(tok)
            nodes.append(Node(gi, elig, frozenset(today), forced, miss, g))
        else:
            # Count-only: lanes are positions; the CPU's today are the last `hits - forced` (type_lanes' order).
            tok = live_mean[gi]
            u = g.hits - g.forced_hits
            n_elig = u + g.dma
            elig = [(j, tok) for j in range(n_elig)]
            today = frozenset(range(n_elig - u, n_elig))
            nodes.append(Node(gi, elig, today, [tok] * g.forced_hits, [tok] * g.m, g))
    return nodes


def route_counts(d: str, recs: list) -> dict:
    """(row, gen) -> {expert: tokens routing it}, joining every verify's records to a graph_routes line."""
    path = os.path.join(d, "stages.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        lines = [x for x in map(json.loads, f) if x.get("kind") == "graph_routes"]
    by_v = collections.defaultdict(list)
    for r in recs:
        by_v[r["verify"]].append(r)
    rows = 1 + max(r["row"] for r in recs)
    forwards = [sorted(rs, key=lambda r: r["row"]) for _, rs in sorted(by_v.items())]
    forwards = [f for f in forwards if [r["row"] for r in f] == list(range(rows))]
    routes = [[set(e for e in layer if e >= 0) for layer in x["routes"]] for x in lines]
    job = [[{e for lanes in r["lane_list"].values() for e, _, _ in lanes} for r in fwd] for fwd in forwards]
    out = {}
    for i, li in layer_misses.match_forwards(job, routes):
        for r, layer in zip(forwards[i], lines[li]["routes"]):
            out[(r["row"], r["gen"])] = collections.Counter(e for e in layer if e >= 0)
    return out


def rule_table(nodes: list, table) -> list:
    """A static split table plus today's flat cut, the CPU taking the last eligible hits in plan order."""
    sets = []
    for n in nodes:
        take = mr.split_take(table, len(n.elig))
        take -= mr.flat_cut(len(n.miss_tok), take)
        sets.append(dict(order(n, False)[:take]))
    return sets


def evaluate(recs: list, nodes_by: dict, rule: str, costs: dict, phys: dict, p: Params, scale: float,
             copy_phys: float, tokens: bool, coupled: float = mr.STREAM_COUPLING, table=None) -> dict:
    """Saving of `rule` (decided with `costs`) against today's lanes, both replayed with `phys` costs."""
    per_v, by_rows, by_class = collections.Counter(), collections.Counter(), collections.Counter()
    differ, stream_v, wait_v = collections.Counter(), collections.Counter(), collections.Counter()
    for r in recs:
        nodes = nodes_by[id(r)]
        today = [{lane: dict(n.elig)[lane] for lane in n.today} for n in nodes]
        base = physics(r, nodes, today, phys, copy_phys, scale)
        # The GPU wait end in this CPU-speed world: the measured one, moved by the scaled baseline's own shift.
        scaled_shift = base - physics(r, nodes, today, phys, copy_phys, 1.0) if scale != 1.0 else 0.0
        end = (r["wait_end"] if r["wait"] > mr.WAIT_NS / mr.MS else r["gate"] + r["delta"]) + scaled_shift

        def effect(sets, r=r, nodes=nodes, today=today, base=base, end=end):
            shift = physics(r, nodes, sets, phys, copy_phys, scale) - base
            dlanes = sum(len(t) - len(s) for s, t in zip(sets, today))
            return mr.layer_effect(r, end, shift, dlanes, copy_phys, coupled)

        if rule == "today":
            sets = today
        elif rule == "table":
            sets = rule_table(nodes, table)
        elif rule == "oracle":
            sets = rule_exhaustive(
                nodes, costs, p, scale, tokens,
                score=lambda s: -effect(s)[0] + 1e-4 * sum(len(n.elig) - len(x) for n, x in zip(nodes, s)))
        elif rule == "exh":
            sets = rule_exhaustive(nodes, costs, p, scale, tokens, gpu_miss=coupled * r["gpu_miss"])
        elif rule == "greedy":
            # The served split model (split-model cffc86e9ed): no stream term.
            sets = rule_greedy(nodes, costs, p, scale, tokens)
        elif rule == "greedy-stream":
            sets = rule_greedy(nodes, costs, p, scale, tokens, streamed=r["gpu_miss"])
        elif rule == "greedy-stream-tb":
            sets = residency_tiebreak(nodes, rule_greedy(nodes, costs, p, scale, tokens, streamed=r["gpu_miss"]))
        else:
            raise ValueError(rule)
        saving, d_arrive, d_wait = effect(sets)
        stream_v[r["verify"]] += d_arrive
        wait_v[r["verify"]] += d_wait
        per_v[r["verify"]] += saving
        by_rows[mr._row_group(r["row"])] += saving
        by_class["misses" if any(n.miss_tok for n in nodes) else "no misses"] += saving
        for n, s in zip(nodes, sets):
            differ["node-records"] += 1
            differ["take differs"] += len(s) != len(n.today)
            differ["set differs"] += set(s) != set(n.today)
            differ["more on copy"] += len(s) < len(n.today)
    nv = max(1, len(per_v))
    vs = sorted(per_v.values())
    return {
        "ms_per_verify": round(sum(vs) / nv, 3),
        "stream_change_ms_per_verify": round(sum(stream_v.values()) / nv, 2),
        "gate_wait_change_ms_per_verify": round(sum(wait_v.values()) / nv, 2),
        "p10_p90": [round(vs[int(.1 * (len(vs) - 1))], 2), round(vs[int(.9 * (len(vs) - 1))], 2)],
        "by_rows": {f"{a}-{b}": round(by_rows[(a, b)] / nv, 2) for a, b in mr.ROW_GROUPS},
        "by_class": {k: round(v / nv, 2) for k, v in by_class.items()},
        "differ": {k: (v if k == "node-records" else round(v / max(1, differ["node-records"]), 3))
                   for k, v in differ.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--tokens-per-verify", type=float, default=3.6)
    ap.add_argument("--split", default="0,1,2,2,3,3,4,5,5,6,7,7,8,9,9,10,11,11,12,13,13,14,15,15,16,17,17,18,19,19,"
                    "20,21,21,22,23,23,24,24,25,26,26")
    ap.add_argument("--cpu-scale", default="traced=1,untraced=0.9",
                    help="name=factor[,...]: every CPU job's length times factor (the rule's costs too)")
    ap.add_argument("--rules", default="today,exh,greedy,exh-count,greedy-count,oracle")
    ap.add_argument("--table", help="also replay this static split table (plus the flat cut) as rule 'table'")
    ap.add_argument("--stream-coupling", type=float, default=mr.STREAM_COUPLING,
                    help="the GPU stream kernel's delay per added copy lane, in copy lanes (misscut_replay.layer_effect)")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    split = [int(x) for x in a.split.split(",")]
    table = [int(x) for x in a.table.split(",")] if a.table else None
    coupled = a.stream_coupling
    ev = mr._events(a.capture)
    recs, mcosts = mr.load(a.capture, split, ev)
    recs = mr.attach(a.capture, recs, 0)
    delta = mr.wait_deltas(recs)
    for r in recs:
        r["delta"] = delta[r["verify"]]
    traced_lanes = all(r["lane_list"] is not None for r in recs)
    routes = route_counts(a.capture, recs) if traced_lanes else {}
    recs = [r for r in recs if r["verify"] >= a.skip]
    joined = [r for r in recs if (r["row"], r["gen"]) in routes]
    tokens = len(joined) > 0.8 * len(recs)
    if tokens:
        recs = joined
    for r in recs:
        # GPU NVMe misses (MISS_GPU lanes) the device typed; untraced kinds: whether the stream kernel ran.
        if r["lane_list"] is not None:
            r["gpu_miss"] = sum(1 for lanes in r["lane_list"].values() for _, b, _ in lanes if b & 0xFF == 4)
        else:
            r["gpu_miss"] = int(r["stream"] > mr.STREAM_COUPLED_MS)

    jobmap = {}
    for kind in ("cpu_start", "cpu_end"):
        for e in ev[kind]:
            jobmap.setdefault((e["team"], e["seq"]), {"team": e["team"], "part": e["a"], "k": e["b"]})[kind] = e["ns"]
    for e in ev["cpu_shape"]:
        if (e["team"], e["seq"]) in jobmap:
            jobmap[(e["team"], e["seq"])]["live"] = e["c"]
    jobs = [dict(j, dur=(j["cpu_end"] - j["cpu_start"]) / mr.MS) for j in jobmap.values()
            if {"cpu_start", "cpu_end", "live"} <= j.keys()]
    fits = {"tokens": fit_costs(jobs, True), "count": fit_costs(jobs, False)}
    live_mean = {t: statistics.mean(j["live"] / j["k"] for j in jobs if j["team"] == t and j["k"]) for t in (0, 1)}

    nodes_tok = {id(r): build_nodes(r, routes if tokens else None, live_mean, split) for r in recs}
    nodes_cnt = {id(r): build_nodes(r, None, live_mean, split) for r in recs}
    join_ok = sum(1 for r in recs for n in nodes_tok[id(r)]
                  if n.g.hit_live == sum(n.forced_tok) + _live(n, n.today)) if tokens else 0

    p0 = Params(copy_lane=round(mcosts["copy_lane"], 3))
    scales = [(k, float(v)) for k, v in (x.split("=") for x in a.cpu_scale.split(","))]
    res = {"capture": os.path.abspath(a.capture), "records": len(recs), "verifies": len({r["verify"] for r in recs}),
           "tokens": tokens, "routes_joined": len(joined), "hit_live_matches_join": join_ok,
           "tokens_per_verify": a.tokens_per_verify, "params": dataclasses.asdict(p0),
           "fits": {kind: {f"team{t}/{'hit' if part == 0 else 'miss'}": {
               "a": round(v["cost"].a, 4), "b": round(v["cost"].b, 4), "c": round(v["cost"].c, 4),
               "mae_ms": round(v["mae"], 3), "n": v["n"]} for (t, part), v in f.items()} for kind, f in fits.items()},
           "live_per_expert": {str(t): round(v, 3) for t, v in live_mean.items()},
           "runs": {}}

    phys = {k: v["cost"] for k, v in fits["tokens" if tokens else "count"].items()}

    def run(rule_name, scale, p, copy_phys):
        count = rule_name.endswith("-count") or not tokens
        rule = rule_name.removesuffix("-count")
        costs = {k: v["cost"] for k, v in fits["count" if count else "tokens"].items()}
        nodes_by = nodes_tok if tokens else nodes_cnt
        e = evaluate(recs, nodes_by, rule, costs, phys, p, scale, copy_phys, not count, coupled, table)
        e["ms_per_token"] = round(e["ms_per_verify"] / a.tokens_per_verify, 3)
        return e

    rules = [x for x in a.rules.split(",") if tokens or not (x in ("exh", "greedy"))] + (["table"] if table else [])
    for sname, scale in scales:
        for rule_name in rules:
            if rule_name == "today" and scale != 1.0:
                continue
            res["runs"][f"{sname}/{rule_name}"] = run(rule_name, scale, p0, p0.copy_lane)
    res["stream_coupling"] = coupled
    head = scales[-1]
    sweep = []
    for cl in () if a.no_sweep else (0.9, 1.05, 1.3):
        for fb, pe in ((0.8, 1.6), (1.1, 2.3), (0.4, 1.2)):
            p = Params(first_byte=fb, per_expert=pe, copy_lane=cl)
            row = {"copy_lane": cl, "first_byte": fb, "per_expert": pe}
            for rule_name in ("exh", "exh-count"):
                if tokens or rule_name == "exh-count":
                    row[rule_name] = run(rule_name, head[1], p, cl)["ms_per_verify"]
                    row[rule_name + "@dev1.05"] = run(rule_name, head[1], dataclasses.replace(p, copy_lane=1.05),
                                                      cl)["ms_per_verify"]
            if fb == 0.8:
                row["oracle"] = run("oracle-count" if not tokens else "oracle", head[1], p, cl)["ms_per_verify"]
            sweep.append(row)
    res["sweep_scale"] = head[0]
    res["sweep"] = sweep
    report(res)
    with open(a.json or os.path.join(a.capture, "split_replay.json"), "w") as f:
        json.dump(res, f, indent=1)
    return 0


def report(res: dict) -> None:
    print(f"{res['capture']}: {res['verifies']} verifies, {res['records']} records, tokens {res['tokens']} "
          f"(joined {res['routes_joined']}, hit-job pairs match the join on {res['hit_live_matches_join']} nodes)")
    print("params:", json.dumps(res["params"]), "pairs per expert:", json.dumps(res["live_per_expert"]),
          "stream coupling:", res["stream_coupling"])
    for kind, f in res["fits"].items():
        print(f"fit {kind}:", json.dumps(f))
    print("runs (ms saved per verify vs today's split + flat cut; + is faster):")
    for name, e in res["runs"].items():
        print(f"  {name:24s} {e['ms_per_verify']:+7.2f} ms/verify {e['ms_per_token']:+6.2f} ms/token p10/p90 "
              f"{e['p10_p90']} class {json.dumps(e['by_class'])} stream {e['stream_change_ms_per_verify']:+.1f} "
              f"gate wait {e['gate_wait_change_ms_per_verify']:+.1f}")
        print(f"      rows {json.dumps(e['by_rows'])} differ {json.dumps(e['differ'])}")
    print(f"sweep at cpu scale {res['sweep_scale']} (ms/verify; @dev1.05: the rule assumes 1.05 ms per copy lane):")
    for s in res["sweep"]:
        print("  " + " ".join(f"{k}={v}" for k, v in s.items()))


if __name__ == "__main__":
    sys.exit(main())
