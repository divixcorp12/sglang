"""Lane-key evaluation on a held-out capture (C2, breakdown base), priced by tier: which misses should take the GPU
lanes. Step 2 of the lane head's next version (PREFILL_CAPTURE_PLAN.md; step 1 is lane_tier_replay.py's gate, which
validated the tier model T, with prefill admissions, and the prices P on the lane-head A/B).

Per arm, one joint replay of the capture's verifies, layer by layer, in time order:
  VRAM: the hot cache under today's victim rule (vram_cache_replay.py's score-k: victims are residents not routed by
    this forward, then those the previous forward did not route, then lowest decayed insert score), inserting at most
    k misses, k = the capture's own GPU-lane count at that (verify, layer); the arm decides which k. At the first verify
    after a prefill the resident set is resynchronized to the capture's (the prefill's own residency update does not
    depend on the lane key).
  RAM: lane_tier_replay's tier (TIER_RULES 1-10) driven by the arm's own hot sets and misses: every miss cached,
    routed rows restamped, the trace's evictions outside the verifies, promotions at the run's rate, and prefill and
    eager admissions from the route log (rule 10).
  Lanes: an inserted miss is a GPU lane: pool-served if the capture served it from the pool, else a copy (DMA) lane if
    its row is in RAM, else an NVMe stream lane. A miss not inserted is a CPU lane of team e % 2: a hit job lane if
    its row is in RAM, else a forced miss.
  Price: P (lane_tier_replay.Prices, fit on the capture itself unless --prices-from) gives each record's stream and
    wait; stream + wait summed over a verify's layers is its MoE critical path.

Arms (the key orders the misses; the first k are inserted):
  actual    the capture's own hot sets and inserts (validation: priced against the measured stream + wait).
  today     the residency key: decayed insert score (vram_cache_replay score-k).
  head      next_verify_head's P(routed next verify) (the served v1 head), ties by score.
  capb      option (b): warm misses (row in RAM) by head P, then cold misses by score, so the GPU never takes a cold
            row the residency order would not have.
  product   the owner's form: the head stays tier-agnostic and the key is the expected saving
            P(recur) x H - G(tier), H = a future VRAM hit's saving, G = this lane's extra cost on the GPU over the CPU
            for the row's live tier (RAM-warm or NVMe-cold). H and G come from P, weighted by how often the stream or
            the CPU path is the critical one (both measurable at startup). capb is its G_cold = infinity case.
  oracle    next1-k: only misses the next verify routes are inserted (by score), victims spare them.
  product-ab  the product key with G_cold raised by P's net error on moving cold lanes to the GPU, measured on the
            lane-head A/B (AB_RESIDUAL_PER_COLD_LANE): the startup-measured tier cost the owner's form would use.
  product-emp  the product key with G_cold the served cost of the move itself (AB_SERVED_PER_COLD_LANE, +0.26 ms per
            NVMe-cold lane put on the GPU), G_warm from P.
  oracle-p  the product key with P = 1 for misses the next verify routes, else 0.
  oracle-emp  product-emp's key with that oracle P: the ceiling of any tier-priced predictor, option (a) included.

Net figures are against `today`. `_bias` prices every arm's NVMe-cold GPU lanes x 15.10/13.79 (T's under-prediction on
the A/B); `_ab` charges AB_RESIDUAL_PER_COLD_LANE per NVMe-cold GPU lane added over `today` (P's net error on that
move, the A/B's only served evidence).

Usage: lane_eval.py CAPTURE --pred PRED.npz [--prices-from RUN] [--head-ms 0.77,0.40,0.32] [--json OUT]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_tier_replay as ltr  # noqa: E402

ARMS = ("actual", "today", "head", "capb", "product", "product-ab", "product-emp", "oracle", "oracle-p",
        "oracle-emp")
HEAD_ARMS = ("head", "capb", "product", "product-ab", "product-emp")
NODES = 2


def load_preds(path: str) -> dict:
    a = np.load(path)
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ri, L, e, p in zip(a["router"].tolist(), a["layer"].tolist(), a["expert"].tolist(), a["p"].tolist()):
        by[ri][L][e] = p
    return by


def price_lanes(P: ltr.Prices, nv: float, po: int, dma: int, teams: dict) -> tuple[float, float]:
    """Prices.record for a lane set that need not exist in the trace: teams[g] = (hit lanes, hit routes, [forced-miss
    routes])."""
    cpu_misses = sum(len(t[2]) for t in teams.values())
    on = 1.0 if nv + po else 0.0
    stream = float(P.stream @ [on, nv, po, dma * on, nv * cpu_misses]) if nv + po else 0.0
    cpu = 0.0
    for lanes, routes, misses in teams.values():
        if lanes or misses:
            cpu = max(cpu, P.team_finish(lanes, routes, misses, nv))
    wait = max(0.0, max(stream, cpu) + P.offset - stream) if (cpu or stream) else 0.0
    return stream, wait


def critical_shares(run: ltr.Run) -> tuple[float, float, float]:
    """Of the serving records with both a stream and CPU jobs, the share whose stream ends after the CPU (w_s) and the
    rest (w_c); and the mean routes of a CPU hit lane."""
    s = c = 0
    lanes = routes = 0
    for r in run.serving(clean_only=False):
        for j in r.jobs:
            if j["part"] == 0:
                lanes += j["lanes"]
                routes += j["routes"]
        if r.stream > 0 and r.jobs:
            cpu = max((j["end"] - r.post) / 1e6 for j in r.jobs)
            s += r.stream >= cpu
            c += r.stream < cpu
    n = max(1, s + c)
    return s / n, c / n, routes / max(1, lanes)


class ProductKey:
    """P(recur) x H - G(tier, routes), from P and the critical shares."""

    def __init__(self, P: ltr.Prices, w_s: float, w_c: float, hit_routes: float, cold_extra: float = 0.0,
                 cold_served: float | None = None):
        self.cold_served = cold_served
        st, hj, mj, ml = P.stream, P.hit_job, P.miss_job, P.miss_land
        self.H = w_c * (hj[1] + hj[2] * hit_routes)
        self.g_cold = (w_s * st[1], w_c * ml[1], w_c * mj[1])  # GPU stream lane; CPU landing + per-route job
        self.g_warm = (w_s * st[3], w_c * hj[1], w_c * hj[2])  # copy lane; CPU hit lane + per route
        self.cold_extra = cold_extra

    def G(self, warm: bool, routes: int) -> float:
        if not warm and self.cold_served is not None:
            return self.cold_served
        a, b, c = self.g_warm if warm else self.g_cold
        return a - b - c * routes + (0.0 if warm else self.cold_extra)

    def describe(self) -> dict:
        return {"H": round(float(self.H), 4), "G_cold": [round(float(x), 4) for x in self.g_cold],
                "G_warm": [round(float(x), 4) for x in self.g_warm], "cold_extra": round(self.cold_extra, 4)}


# P's net error on moving NVMe-cold lanes from the CPU onto the GPU, measured on the lane-head A/B (stage 0): ON - OFF
# stream + wait was +4.76 ms/verify measured (51.68 - 46.92) against -2.45 predicted (E: 50.74 - 53.19), for +18.12
# NVMe-cold GPU lanes/verify. Charged per moved lane, it is the "_ab" correction and the product-ab key's G_cold term.
AB_RESIDUAL_PER_COLD_LANE = (4.76 + 2.45) / 18.12
# The served cost of that move itself: +4.76 ms/verify of stream + wait for +18.12 NVMe-cold GPU lanes (product-emp).
AB_SERVED_PER_COLD_LANE = 4.76 / 18.12


def evaluate(run: ltr.Run, preds: dict, P: ltr.Prices, keys: dict, arm: str, layers: int) -> dict:
    key = keys["product"]
    d = run.dir
    header, log = ltr.route_log(d)
    off = ltr.align(run, log)
    rows, frac, share, full = ltr.numa_capacities(d)
    full = set(full)
    p_promote = ltr.promote_fraction(d)
    tiers = [ltr.layer_tiers(rows.get(L, 210), frac, share) for L in range(layers)]
    cap = header["hot_capacity"]
    by_vr = {(r.verify, r.row): r for r in run.records}
    replayed = {r.seq for r in run.records}
    timeline = []
    for v in range(len(run.verifies)):
        i = v + off
        if not 0 <= i < len(log):
            continue
        for L in range(layers):
            r = by_vr.get((v, L))
            ns = r.post if r is not None else run.verifies[v][0] + L * 5_000_000
            timeline.append((ns, 1, "gather", (v, i, L)))
    for _, e in ltr._events(glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl")), {"evict"}):
        if e["seq"] not in replayed:
            timeline.append((e["ns"], 0, "evict", (e["row"], e["group"], e["a"], e["c"])))
    for _, e in ltr._events(glob.glob(os.path.join(d, "events.*.exl3-spec*.jsonl")), {"spec_land"}):
        if ltr._hash01(e["row"], e["a"], e["ns"] & 0xFFFFFFFF) < p_promote:
            timeline.append((e["ns"], 0, "promote", (e["row"], e["a"])))
    eager = ltr.eager_lines(d)
    for k, line in enumerate(eager):
        timeline.append((int(line["t"] * 1e9) - 1, 0, "eager", k))
    timeline.sort(key=lambda t: (t[0], t[1]))

    resident = [None] * layers
    score = [collections.defaultdict(float) for _ in range(layers)]
    routed_prev = [set() for _ in range(layers)]
    last_hot = [set() for _ in range(layers)]
    resync = [True] * layers
    clean_v = {v for v in range(len(run.verifies)) if run.prompt_of[v] >= 0 and run.clean[run.prompt_of[v]]}
    out = collections.Counter()
    tick = 0
    for ns, _, kind, payload in timeline:
        tick += 1
        if kind == "eager":
            line = eager[payload]
            L = line["layer"]
            if L < layers and L not in full:
                tick, *_ = ltr.eager_admit(tiers[L], line, last_hot[L], tick)
            if line["phase"] == "extend":
                resync = [True] * layers
            continue
        if kind == "evict":
            row, g, old, new = payload
            if row < layers and row not in full:
                tiers[row][g].apply_eviction(old, new, tick)
            continue
        if kind == "promote":
            row, e = payload
            if row < layers and row not in full:
                tiers[row][e % NODES].promote(e, last_hot[row])
            continue
        v, i, L = payload
        cur = log[i]
        nxt = log[i + 1] if i + 1 < len(log) else None
        cnt = collections.Counter(e for e in cur["routes"][L] if e >= 0)
        u = set(cnt)
        hot_act = set(cur["hot"][L])
        if resync[L] or resident[L] is None or arm == "actual":
            resident[L] = set(hot_act)
            resync[L] = False
        res = resident[L]
        hot_plan = set(res)
        hits = u & res
        misses = u - res
        k_cap = len((u - hot_act) & set(nxt["hot"][L])) if nxt is not None else 0
        k = min(k_cap, len(misses))
        tier = tiers[L]
        warm = {e for e in misses if L in full or tier[e % NODES].has(e)}
        sc = score[L]
        d_ = 0.98 ** int(cur.get("forward_tokens") or 1)
        for e in list(sc):
            sc[e] *= d_
        for e, c in cnt.items():
            sc[e] += c
        pred = (preds.get(cur.get("router")) or {}).get(L) or {}
        nxt_u = {e for e in nxt["routes"][L] if e >= 0} if nxt is not None else set()
        if arm == "actual":
            inserted = (u - hot_act) & set(nxt["hot"][L]) if nxt is not None else set()
            plan = []
        elif arm == "today":
            plan = sorted(misses, key=lambda e: (-sc[e], e))
        elif arm == "head":
            plan = sorted(misses, key=lambda e: (-pred.get(e, 0.0), -sc[e], e))
        elif arm == "capb":
            plan = sorted(misses, key=lambda e: (e not in warm, -pred.get(e, 0.0) if e in warm else -sc[e], e))
        elif arm in keys:
            kk = keys[arm]
            plan = sorted(misses, key=lambda e: (-(pred.get(e, 0.0) * kk.H - kk.G(e in warm, cnt[e])), -sc[e], e))
        elif arm == "oracle":
            plan = sorted((e for e in misses if e in nxt_u), key=lambda e: (-sc[e], e))
        elif arm in ("oracle-p", "oracle-emp"):
            kk = key if arm == "oracle-p" else keys["product-emp"]
            plan = sorted(misses, key=lambda e: (-((e in nxt_u) * kk.H - kk.G(e in warm, cnt[e])), -sc[e], e))
        else:
            raise ValueError(arm)
        if arm != "actual":
            inserted = set()
            for e in plan[:k]:
                if len(res) >= cap[L]:
                    cands = [x for x in res if x not in u and not (arm == "oracle" and x in nxt_u)]
                    if not cands:
                        break
                    victim = min(cands, key=lambda x: (x in routed_prev[L], sc[x], -x))
                    res.discard(victim)
                res.add(e)
                inserted.add(e)
        routed_prev[L] = u
        r = by_vr.get((v, L))
        pool = set(r.pool) if r is not None else set()
        nv = sum(1 for e in inserted if e not in warm and e not in pool)
        po = sum(1 for e in inserted if e in pool)
        dma = sum(1 for e in inserted if e in warm and e not in pool)
        teams = {0: [0, 0, []], 1: [0, 0, []]}
        for e in sorted(misses - inserted, key=lambda e: -cnt[e]):
            t = teams[e % NODES]
            if e in warm:
                t[0] += 1
                t[1] += cnt[e]
            else:
                t[2].append(cnt[e])
        if v in clean_v:
            s, w = price_lanes(P, nv, po, dma, teams)
            out["stream"] += s
            out["wait"] += w
            s2, w2 = price_lanes(P, nv * BIAS, po, dma, teams)
            out["stream_bias"] += s2
            out["wait_bias"] += w2
            out["hits"] += len(hits)
            out["misses"] += len(misses)
            out["gpu_lanes"] += len(inserted)
            out["gpu_nvme"] += nv
            out["gpu_pool"] += po
            out["gpu_copy"] += dma
            out["cpu_hit_lanes"] += sum(t[0] for t in teams.values())
            out["cpu_forced"] += sum(len(t[2]) for t in teams.values())
            if r is not None:
                out["measured_stream"] += r.stream
                out["measured_wait"] += r.wait
                ps, pw = P.record(r)
                out["priced_observed"] += ps + pw
                out["obs_gpu_nvme"] += r.gpu_nvme
                out["obs_cpu_forced"] += ltr.cpu_misses(r)
        last_hot[L] = set(res)
        if L in full:
            continue
        for e in u:  # rule 4
            t = tier[e % NODES]
            if t.has(e):
                t.stamp[e] = tick
        for e in sorted(misses, key=lambda x: x in warm):  # rule 5, against the record's hot bitmap (rule 3)
            tier[e % NODES].insert(e, tick, hot_plan | u)
    n = sum(p["verifies"] for p, c in zip(run.prompts, run.clean) if c)
    res = {k: v / n for k, v in out.items()}
    res["ms"] = res["stream"] + res["wait"]
    res["ms_bias"] = res["stream_bias"] + res["wait_bias"]
    res["verifies"] = n
    return res


# T under-predicts NVMe-cold GPU lanes by ~8% on the A/B (stage 0: 13.79 vs 15.10 OFF, 30.61 vs 33.22 ON); the "_bias"
# figures price every arm with its NVMe GPU lanes scaled up by the OFF ratio.
BIAS = 15.10 / 13.79


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--prices-from", default=None, help="fit P on this run instead of the capture")
    ap.add_argument("--head-ms", default="0.77,0.40,0.32")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--json")
    a = ap.parse_args()
    run = ltr.Run(a.capture)
    P = ltr.Prices(ltr.Run(a.prices_from) if a.prices_from else run)
    w_s, w_c, hit_routes = critical_shares(run)
    key = ProductKey(P, w_s, w_c, hit_routes)
    keys = {"product": key, "product-ab": ProductKey(P, w_s, w_c, hit_routes, AB_RESIDUAL_PER_COLD_LANE),
            "product-emp": ProductKey(P, w_s, w_c, hit_routes, cold_served=AB_SERVED_PER_COLD_LANE)}
    layers = max(r.row for r in run.records) + 1
    preds = load_preds(a.pred)
    res = {"capture": a.capture, "prices": P.describe(), "critical_share_stream": round(w_s, 3),
           "hit_routes": round(hit_routes, 3), "keys": {k: v.describe() for k, v in keys.items()}, "arms": {}}
    for arm in a.arms.split(","):
        res["arms"][arm] = evaluate(run, preds, P, keys, arm, layers)
        print(arm, json.dumps({k: round(v, 2) for k, v in res["arms"][arm].items()}), flush=True)
    base = res["arms"].get("today")
    head_ms = [float(x) for x in a.head_ms.split(",")]
    if base:
        net = {}
        for arm, r in res["arms"].items():
            paid = arm in HEAD_ARMS
            net[arm] = {"d_hits": round(r["hits"] - base["hits"], 2),
                        "d_gpu_nvme": round(r["gpu_nvme"] - base["gpu_nvme"], 2),
                        "d_stream": round(r["stream"] - base["stream"], 2),
                        "d_wait": round(r["wait"] - base["wait"], 2),
                        "d_ms_before_head": round(r["ms"] - base["ms"], 2),
                        "net_ms_at_head_cost": {f"{c}": round(r["ms"] - base["ms"] + (c if paid else 0.0), 2)
                                                for c in head_ms},
                        "d_ms_bias_before_head": round(r["ms_bias"] - base["ms_bias"], 2),
                        "d_ms_ab_before_head": round(r["ms"] - base["ms"] + AB_RESIDUAL_PER_COLD_LANE
                                                     * (r["gpu_nvme"] - base["gpu_nvme"]), 2)}
        res["net_vs_today"] = net
        print(json.dumps(net, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
