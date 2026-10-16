"""Offline replay of the GPU drip landing in the VRAM hot cache (DRIP_HOT_DESIGN.md section 3), on a traced capture with
the drip on (breakdown-20261015/drip: lane_kind, drip_* and graph_routes).

Every record (one layer of one verify) is re-typed from its own lanes, in plan order, with the post's rules
(ES/lease_device.cuh type_lanes): lanes from the first one without a VRAM victim on are forced CPU lanes (spill); an
unforced lane whose expert a landed drip names is HIT_SCRATCH; node n's CPU takes the last split[n] of its unforced RAM
hits (one fewer when the node has 1..miss_cut_max forced misses); other hits are HIT_COPY and other misses MISS_GPU.

``gate`` checks that model against the capture: each lane's kind, the scratch lanes the drip timing predicts (a drip is
published for its target record when its drip_land precedes that record's post), and the per-verify lane mix against
expert_breakdown.py's within 5%. ``arms`` then perturbs each record (one verify at a time, from the captured residency):

  today   as captured
  B       landed drips this record routes become VRAM hits before planning; each takes a shortlist victim
          noext: from the lanes' own victim window (forced_from - adoptions); ext: from columns past it
  A2      the first two admitted picks of the source record reserve shortlist-head victims before routing (lost when
          that victim is routed now); landed and routed ones are hits; the lanes keep their window
  C1      one spare row per layer (one slot fewer: the shortlist head is lost when routed); only the first pick, landed
          and routed, is adopted, swapping a victim in as the new spare

and counts per verify: VRAM hits, lanes by kind, forced CPU lanes, inserts, lost hits, adoptions, extra evictions and
their returns within --horizon verifies (rule-1 swaps: priced NVMe-cold or RAM-warm by lane_tier_replay's NodeTier
model, TIER_RULES at a98b5b527c, without prefill admissions).

``victims`` reports the proxy shortlist head's recurrence on any capture with graph_routes (C2 has no drip or lane_kind
events, so only this runs there).

Usage: drip_hot_replay.py gate CAPTURE [--skip 8] [--json OUT]
       drip_hot_replay.py arms CAPTURE [--skip 8] [--horizon 3] [--json OUT]
       drip_hot_replay.py victims CAPTURE [--json OUT]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert_breakdown as eb  # noqa: E402
import lane_tier_replay as ltr  # noqa: E402
import layer_misses  # noqa: E402

HIT_COPY, HIT_SM, HIT_CPU, MISS_GPU, MISS_CPU, HIT_SCRATCH = 1, 2, 3, 4, 5, 6
HITS = (HIT_COPY, HIT_SM, HIT_CPU, HIT_SCRATCH)
SPILL = 256
NODES = 2
VICTIM_LANES = 8
MISS_CUT, MISS_CUT_MAX = 1, 3  # SGLANG_DSV41_CPU_SPLIT_MISS_CUT / _MAX of the -fixedsplit arms
# SGLANG_DSV41_CPU_EXPERTS_SPLIT of breakdown-20261015/run.sh: CPU lanes per n eligible lanes of a node.
SPLIT = [int(x) for x in "0,1,2,2,3,3,4,5,5,6,7,7,8,9,9,10,11,11,12,13,13,14,15,15,16,17,17,18,19,19,20,21,21,22,23,23,"
         "24,24,25,26,26".split(",")]
CATS = ("vram", "drip", "sm", "dma", "pool", "nvme_gpu", "cpu_ram", "cpu_pool", "cpu_nvme")
D2D_MS_PER_ROW = 0.0105  # index copy of one expert row past L2 (MOE/expert_residency_gpu.py class doc)
ADOPT_LAUNCH_MS = 0.004  # one bookkeeping launch per layer
# lane_tier_replay's fit on the lane-head A/B (team-lead message 2026-10-16): an NVMe-cold GPU lane's stream, and the
# delay it puts on a CPU miss's landing.
NVME_GPU_MS, NVME_ON_CPU_MISS_MS = 1.47, 1.09


# ---- the capture ------------------------------------------------------------------------------------------------------


class Capture:
    def __init__(self, d: str, skip: int):
        self.dir = d
        ev = eb.load_events(d)
        self.ev = ev
        stamps = eb.load_stamps(d, eb.gpu_offset(d))
        verifies = eb.load_verifies(d)
        starts = [b for b, _ in verifies]
        recs = eb.build_records(ev)
        order = collections.defaultdict(dict)
        for e in ev["lane_kind"]:
            order[(e["row"], e["gen"])][e["c"]] = (e["a"], e["b"])
        self.records = []
        for key, r in recs.items():
            if r["start"] is None or key not in order:
                continue
            k = bisect.bisect_right(starts, r["start"]) - 1
            if not (skip <= k < len(verifies) and r["start"] <= verifies[k][1]):
                continue
            s = stamps.get(r["seq"])
            r["row"], r["gen"] = key
            r["verify"] = k
            # The post reads the scratch table after its delta wait, so its end, not its start, bounds the read.
            r["post"] = s["post_end"] if s and s["post_end"] else r["start"]
            # An adoption runs at the gather, before the post starts: the arms use this bound.
            r["gather"] = s["post_start"] if s and s["post_start"] else r["start"]
            r["ordered"] = [order[key][j] for j in sorted(order[key])]
            if len(r["ordered"]) != r["lane_count"]:
                continue  # a group's lane_kind events are missing: not a whole record
            self.records.append(r)
        self.records.sort(key=lambda r: r["seq"])
        self.verifies = sorted({r["verify"] for r in self.records})
        self.rows = 1 + max(r["row"] for r in self.records)
        self._drips(ev)
        self._routes()

    def _drips(self, ev: dict) -> None:
        """Per target record: the drips published before its post ends (expert -> land ns), and the admitted picks in
        pick order with their land ns (None: never landed). A drip's target is its row's first record after the scored
        (source) record."""
        by_row = collections.defaultdict(list)
        for r in self.records:
            by_row[r["row"]].append(r)
        seqs = {row: [r["seq"] for r in rs] for row, rs in by_row.items()}

        def target(row: int, source_seq: int):
            rs = by_row.get(row)
            if not rs:
                return None
            i = bisect.bisect_right(seqs[row], source_seq)
            return rs[i] if i < len(rs) else None

        land = {(e["seq"], e["row"], e["a"]): e["ns"] for e in ev["drip_land"]}
        for r in self.records:
            r["published"], r["picks"] = {}, []
        rank = collections.Counter()
        for e in sorted(ev["drip_admit"], key=lambda e: e["ns"]):
            t = target(e["row"], e["seq"])
            k = (e["seq"], e["row"])
            pick = rank[k]
            rank[k] += 1
            if t is None:
                continue
            landed = land.get((e["seq"], e["row"], e["a"]))
            t["picks"].append((pick, e["a"], landed))
            if landed is not None and landed < t["post"]:
                t["published"][e["a"]] = landed

    def _routes(self) -> None:
        """Each record's graph_routes line (routes and the residency before the forward), joined by layer_misses."""
        lines = []
        with open(os.path.join(self.dir, "stages.jsonl")) as f:
            for line in f:
                if '"graph_routes"' in line:
                    x = json.loads(line)
                    if x.get("kind") == "graph_routes":
                        lines.append(x)
        self.lines = lines
        by_v = collections.defaultdict(list)
        for r in self.records:
            by_v[r["verify"]].append(r)
        fwds = [sorted(rs, key=lambda r: r["row"]) for _, rs in sorted(by_v.items())]
        fwds = [f for f in fwds if [r["row"] for r in f] == list(range(self.rows))]
        job = [[{e for e, b in r["ordered"]} for r in f] for f in fwds]
        routes = [[{e for e in layer if e >= 0} for layer in x["routes"]] for x in lines]
        self.line_of_verify = {}
        for i, li in layer_misses.match_forwards(job, routes):
            self.line_of_verify[fwds[i][0]["verify"]] = li


# ---- the post's typing ------------------------------------------------------------------------------------------------


def type_record(lanes: list[tuple[int, bool]], forced_from: int, scratch: set, forced_misses_cut: bool = True) -> list[int]:
    """Kinds for ``lanes`` [(expert, RAM hit)] in plan order (ES/lease_device.cuh:367-451)."""
    n = [0] * NODES
    forced_miss = [0] * NODES
    eligible = []
    for j, (e, hit) in enumerate(lanes):
        node = e % NODES
        forced = j >= forced_from
        el = not forced and hit and e not in scratch
        eligible.append(el)
        n[node] += el
        forced_miss[node] += forced and not hit
    take = [SPLIT[min(n[k], len(SPLIT) - 1)] if n[k] else 0 for k in range(NODES)]
    for k in range(NODES):
        if forced_misses_cut and 1 <= forced_miss[k] <= MISS_CUT_MAX:
            take[k] = max(0, take[k] - MISS_CUT)
    kinds = [0] * len(lanes)
    for j in range(len(lanes) - 1, -1, -1):
        e, hit = lanes[j]
        node = e % NODES
        forced = j >= forced_from
        if not forced and e in scratch:
            kinds[j] = HIT_SCRATCH
        elif forced:
            kinds[j] = HIT_CPU if hit else MISS_CPU
        elif eligible[j] and take[node] > 0:
            take[node] -= 1
            kinds[j] = HIT_CPU
        else:
            kinds[j] = HIT_COPY if hit else MISS_GPU
    return kinds


def record_lanes(r: dict) -> tuple[list[tuple[int, bool]], int]:
    lanes = [(e, (b & 0xFF) in HITS) for e, b in r["ordered"]]
    forced = [j for j, (_, b) in enumerate(r["ordered"]) if b & SPILL]
    return lanes, (forced[0] if forced else len(lanes))


def category(kind: int, e: int, r: dict) -> str:
    c = eb.KIND[kind]
    if c == "nvme_gpu" and e in r["pool"]:
        return "pool"
    if c == "cpu_nvme" and e in r["spec"]:
        return "cpu_pool"
    return c


# ---- gate -------------------------------------------------------------------------------------------------------------


def gate(cap: Capture) -> dict:
    agree = collections.Counter()
    confusion = collections.Counter()
    model = collections.defaultdict(collections.Counter)
    actual = collections.defaultdict(collections.Counter)
    scratch_tp = scratch_pred = scratch_obs = 0
    for r in cap.records:
        lanes, ff = record_lanes(r)
        pred = type_record(lanes, ff, set(r["published"]))
        v = r["verify"]
        model[v]["vram"] += r["routed"] - len(lanes)
        actual[v]["vram"] += r["routed"] - len(lanes)
        model[v]["spilled"] += len(lanes) - ff
        actual[v]["spilled"] += len(lanes) - ff
        for (e, b), k in zip(r["ordered"], pred):
            obs = b & 0xFF
            agree["lanes"] += 1
            agree["same"] += obs == k
            if obs != k:
                confusion[f"{eb.KIND[obs]}->{eb.KIND[k]}"] += 1
            model[v][category(k, e, r)] += 1
            actual[v][category(obs, e, r)] += 1
            scratch_pred += k == HIT_SCRATCH
            scratch_obs += obs == HIT_SCRATCH
            scratch_tp += k == HIT_SCRATCH and obs == HIT_SCRATCH
    nv = len(model)
    keys = CATS + ("spilled",)
    mean = lambda d, c: sum(x[c] for x in d.values()) / nv
    m = {c: round(mean(model, c), 2) for c in keys}
    a = {c: round(mean(actual, c), 2) for c in keys}
    ref_path = os.path.join(cap.dir, "expert_breakdown.json")
    ref = {}
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            j = json.load(f)
        ref = dict(j["per_verify_unique"])
        ref["spilled"] = round(sum(L["spilled"] for L in j["layers"]), 2)
    # the 5% check covers the categories that carry at least 1 lane a verify
    checks = {c: (m[c], a[c], (abs(m[c] - a[c]) / a[c]) if a[c] else 0.0) for c in keys if a[c] >= 1.0}
    passed = all(rel <= 0.05 for _, _, rel in checks.values())
    return {
        "capture": os.path.abspath(cap.dir),
        "verifies": nv,
        "records": len(cap.records),
        "lane_agreement": round(agree["same"] / max(1, agree["lanes"]), 4),
        "confusion_top": dict(confusion.most_common(8)),
        "scratch_lanes": {"predicted": scratch_pred, "observed": scratch_obs, "both": scratch_tp},
        "model_per_verify": m,
        "capture_per_verify": a,
        "expert_breakdown_per_verify": ref,
        "checks": {c: {"model": x, "capture": y, "rel": round(rel, 4)} for c, (x, y, rel) in checks.items()},
        "routes_joined_verifies": len(cap.line_of_verify),
        "passed": passed,
    }


# ---- arms -------------------------------------------------------------------------------------------------------------


def proxy_rankings(lines: list[dict]) -> list[list[list[int]]]:
    """Per route line and layer, the residents before that forward in a proxy of _rank_victims' order: (routed by the
    previous forward, insert score, -expert) ascending, the score decayed 0.98 per token and fed by route counts
    (MOE/expert_residency_gpu.py:650-659, 933-941). Scores start at 0 at the first line: prefill counts are absent."""
    layers = len(lines[0]["routes"])
    score = [collections.defaultdict(float) for _ in range(layers)]
    prev = [set() for _ in range(layers)]
    out = []
    for x in lines:
        tokens = max(1, len(x["routes"][0]) // 6)
        ranked = []
        for L in range(layers):
            hot = [e for e in x["hot"][L] if e >= 0]
            ranked.append(sorted(hot, key=lambda e: (e in prev[L], score[L][e], -e)))
        out.append(ranked)
        decay = 0.98 ** tokens
        for L in range(layers):
            routed = collections.Counter(e for e in x["routes"][L] if e >= 0)
            for e in list(score[L]):
                score[L][e] *= decay
            for e, n in routed.items():
                score[L][e] += n
            prev[L] = set(routed)
    return out


def counts_of(r: dict, lanes: list[tuple[int, bool]], kinds: list[int], forced_from: int) -> collections.Counter:
    c = collections.Counter()
    for j, ((e, _), k) in enumerate(zip(lanes, kinds)):
        c[category(k, e, r)] += 1
        c["forced"] += j >= forced_from
        c["inserts"] += j < forced_from and k in (HIT_COPY, HIT_SM, MISS_GPU, HIT_SCRATCH)
        c["d2d"] += k == HIT_SCRATCH
    return c


def perturb(r: dict, arm: str, ranked: list[int] | None, routed: set | None) -> tuple[collections.Counter, list[int]]:
    """One record under ``arm``: its lane counts and the residents it evicts beyond its inserts' (pre-route losses)."""
    lanes, ff = record_lanes(r)
    count = len(lanes)
    usable = ff if ff < count else 1 << 30  # victims the lanes found (unbounded when nothing spilled)
    if arm == "today":
        kinds = type_record(lanes, ff, set(r["published"]))
        c = counts_of(r, lanes, kinds, ff)
        c["vram"] += r["routed"] - count
        return c, []
    lane_experts = {e for e, _ in lanes}
    ready = {e for _, e, ns in r["picks"] if ns is not None and ns < r["gather"]}
    lost: list[int] = []
    if arm in ("B_noext", "B_ext"):
        adopt = ready & lane_experts
    elif arm == "A2":
        reserved = [(e, ns) for p, e, ns in r["picks"] if p < 2]
        adopt = {e for e, ns in reserved if ns is not None and ns < r["gather"]} & lane_experts
        lost = (ranked or [])[: len(reserved)]
    elif arm == "C1":
        first = [(e, ns) for p, e, ns in r["picks"] if p == 0]
        adopt = {e for e, ns in first if ns is not None and ns < r["gather"]} & lane_experts
        lost = (ranked or [])[:1]
    else:
        raise ValueError(arm)
    lost_routed = [e for e in lost if routed is not None and e in routed and e not in lane_experts]
    # Kept lanes in plan order; a lost resident is a RAM-warm miss of the VRAM cache (its RAM row was pinned while
    # hot, TIER_RULES 3) with the lowest key, so it plans last.
    new = [(e, hit) for e, hit in lanes if e not in adopt] + [(e, True) for e in lost_routed]
    n_adopt_spilled = sum(1 for j, (e, _) in enumerate(lanes) if e in adopt and j >= ff)
    if arm == "B_noext":
        window = usable - len(adopt) if usable < (1 << 30) else 1 << 30
    else:
        window = usable
    ff2 = min(len(new), window)
    kinds = type_record(new, ff2, set())
    c = counts_of(r, new, kinds, ff2)
    c["vram"] += r["routed"] - count + len(adopt) - len(lost_routed)
    c["adopted"] += len(adopt)
    c["adopted_from_spill"] += n_adopt_spilled
    c["d2d"] += len(adopt) if arm.startswith("B") else 0
    c["inserts"] += len(adopt)
    c["lost_hits"] += len(lost_routed)
    c["reserved"] += len(lost) if arm == "A2" else 0
    c["reserved_wasted"] += (len(lost) - len(adopt)) if arm == "A2" else 0
    return c, lost


def arms(cap: Capture, horizon: int) -> dict:
    lines = cap.lines
    ranks = proxy_rankings(lines)
    names = ("today", "B_noext", "B_ext", "A2", "C1")
    per_v = {a: collections.defaultdict(collections.Counter) for a in names}
    released = {a: [] for a in names}  # (route line, layer, expert): residents evicted beyond today's
    for r in cap.records:
        li = cap.line_of_verify.get(r["verify"])
        ranked = routed = None
        if li is not None:
            ranked = ranks[li][r["row"]]
            routed = {e for e in lines[li]["routes"][r["row"]] if e >= 0}
        base, _ = perturb(r, "today", ranked, routed)
        per_v["today"][r["verify"]] += base
        for a in names[1:]:
            c, lost = perturb(r, a, ranked, routed)
            per_v[a][r["verify"]] += c
            if li is None:
                continue
            # Residents this arm evicts that the capture kept: its extra inserts' victims (the next ranked residents
            # the capture did not evict), and A/C's pre-route victims.
            nxt = lines[li + 1]["hot"][r["row"]] if li + 1 < len(lines) else []
            kept = [e for e in ranked if e in set(nxt) and e not in routed]
            # A2's adopted drips land in slots their reservations already emptied; C1's lost resident is its missing
            # slot, recounted each verify, not an eviction.
            if a == "A2":
                extra = max(0, c["inserts"] - c["adopted"] - base["inserts"])
                head = lost
            else:
                extra = max(0, c["inserts"] - base["inserts"])
                head = []
            victims = list(dict.fromkeys(head + [e for e in kept if e not in lost][:extra]))
            c2 = per_v[a][r["verify"]]
            c2["extra_evictions"] += len(victims)
            released[a].extend((li, r["row"], e) for e in victims)
    returns = {a: tier_returns(cap, released[a], horizon) for a in names}
    out = {"verifies": len(per_v["today"]), "horizon": horizon, "arms": {}}
    keys = CATS + ("forced", "inserts", "d2d", "adopted", "adopted_from_spill", "lost_hits", "reserved",
                   "reserved_wasted", "extra_evictions")
    nv = len(per_v["today"])
    mean = {a: {k: sum(v[k] for v in per_v[a].values()) / nv for k in keys} for a in names}
    t = mean["today"]
    cpu = lambda m: m["cpu_ram"] + m["cpu_pool"] + m["cpu_nvme"]
    for a in names:
        m = mean[a]
        ret = returns[a]
        d_cpu = cpu(m) - cpu(t)
        d_nvme = m["nvme_gpu"] - t["nvme_gpu"]
        d_d2d = m["d2d"] - t["d2d"]
        gpu_ms = d_d2d * D2D_MS_PER_ROW + (cap.rows * ADOPT_LAUNCH_MS if a != "today" else 0.0)
        warm = ret.get("warm", 0) / nv
        cold = ret.get("cold", 0) / nv
        net_cpu = d_cpu + warm  # a RAM-warm return comes back as a lane: counted as a CPU lane (the bound)
        priced = {}
        for name, per_lane in (("cpu_0.07", 0.07), ("cpu_0.28", 0.28)):
            priced[name] = round(
                d_cpu * per_lane + warm * per_lane + cold * (NVME_GPU_MS + NVME_ON_CPU_MISS_MS)
                + d_nvme * (NVME_GPU_MS + NVME_ON_CPU_MISS_MS) + gpu_ms, 2)
        out["arms"][a] = {
            "per_verify": {k: round(v, 2) for k, v in m.items()},
            "delta_cpu_lanes": round(d_cpu, 2),
            "delta_nvme_gpu_lanes": round(d_nvme, 2),
            "returns_ram_warm": round(warm, 2),
            "returns_nvme_cold": round(cold, 2),
            "net_cpu_lanes_with_warm_returns": round(net_cpu, 2),
            "added_gpu_ms": round(gpu_ms, 3),
            "ms_per_verify_vs_today": priced,  # negative is a saving
            "build_gate": {
                "cpu_lanes_-30": net_cpu <= -30,
                "no_more_nvme_gpu": d_nvme <= 0 and cold == 0 if a != "today" else None,
                "gpu_ms_<=0.5": gpu_ms <= 0.5,
            },
        }
    out["tier_model"] = returns["today"]["model"]
    return out


def tier_returns(cap: Capture, released: list[tuple[int, int, int]], horizon: int) -> dict:
    """Drive lane_tier_replay's NodeTier (TIER_RULES 1-6, 9; no prefill admissions, rule 8) over the route lines with
    the released residents unpinned from their release until they are routed again or ``horizon`` lines pass, and
    classify each return (routed again inside the horizon, while the capture kept it resident) as RAM-warm or
    NVMe-cold."""
    rows, frac, share, full = ltr.numa_capacities(cap.dir)
    layers = len(cap.lines[0]["routes"])
    tiers = [ltr.layer_tiers(rows.get(L, 210), frac, share) for L in range(layers)]
    p_promote = ltr.promote_fraction(cap.dir)
    lines = cap.lines
    seqs = {r["seq"] for r in cap.records}
    timeline = [(int(x["t"] * 1e9), 1, "line", i) for i, x in enumerate(lines)]
    for e in cap.ev["evict"]:
        if e["seq"] not in seqs:
            timeline.append((e["ns"], 0, "evict", (e["row"], e["group"], e["a"], e["c"])))
    for e in cap.ev["spec_land"]:
        if ltr._hash01(e["row"], e["a"], e["ns"] & 0xFFFFFFFF) < p_promote:
            timeline.append((e["ns"], 0, "promote", (e["row"], e["a"])))
    timeline.sort(key=lambda t: (t[0], t[1]))
    open_ = collections.defaultdict(dict)  # layer -> expert -> release line
    by_line = collections.defaultdict(list)
    for li, L, e in released:
        by_line[li].append((L, e))
    res = collections.Counter()
    last_hot = collections.defaultdict(set)
    tick = 0
    for _, _, kind, payload in timeline:
        tick += 1
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
        li = payload
        x = lines[li]
        for L in range(layers):
            routed = {e for e in x["routes"][L] if e >= 0}
            hot = {e for e in x["hot"][L] if e >= 0}
            for e, start in list(open_[L].items()):
                if li - start > horizon or e not in hot:
                    del open_[L][e]  # past the horizon, or the capture evicted it anyway
                elif e in routed:
                    res["returns"] += 1
                    res["warm" if tiers[L][e % NODES].has(e) else "cold"] += 1
                    del open_[L][e]
            hot_cf = hot - set(open_[L])
            last_hot[L] = hot_cf
            if L in full:
                continue
            protected = hot_cf | routed
            for e in routed:
                t = tiers[L][e % NODES]
                if t.has(e):
                    t.stamp[e] = tick
            for e in routed - hot_cf:
                tiers[L][e % NODES].insert(e, tick, protected)
        for L, e in by_line.get(li, ()):
            open_[L][e] = li
        res["released"] += len(by_line.get(li, ()))
    res["model"] = f"lane_tier_replay NodeTier (TIER_RULES={ltr.TIER_RULES}), no prefill admissions; promote {p_promote:.3f}"
    return dict(res)


def victims(d: str) -> dict:
    """The proxy shortlist's head on any capture with graph_routes: P(routed this forward) and within 3, by rank."""
    lines = []
    with open(os.path.join(d, "stages.jsonl")) as f:
        for line in f:
            if '"graph_routes"' in line:
                x = json.loads(line)
                if x.get("kind") == "graph_routes":
                    lines.append(x)
    ranks = proxy_rankings(lines)
    now, soon, n = collections.Counter(), collections.Counter(), collections.Counter()
    for li in range(8, len(lines)):
        for L, ranked in enumerate(ranks[li]):
            for k, e in enumerate(ranked[:12]):
                n[k] += 1
                now[k] += any(e == x for x in lines[li]["routes"][L])
                soon[k] += any(e in lines[w]["routes"][L] for w in range(li, min(li + 3, len(lines))))
    return {"capture": os.path.abspath(d), "lines": len(lines),
            "by_rank": {k: {"routed_now": round(now[k] / n[k], 3), "routed_within_3": round(soon[k] / n[k], 3),
                            "n": n[k]} for k in sorted(n)}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("gate", "arms", "victims"))
    p.add_argument("capture")
    p.add_argument("--skip", type=int, default=8)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--json")
    a = p.parse_args()
    if a.mode == "victims":
        res = victims(a.capture)
        print(json.dumps(res, indent=1))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
        return 0
    cap = Capture(a.capture, a.skip)
    if a.mode == "gate":
        res = gate(cap)
    else:
        res = arms(cap, a.horizon)
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
