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

Usage: drip_hot_replay.py gate CAPTURE [--skip 8] [--json OUT]
       drip_hot_replay.py arms CAPTURE [--skip 8] [--horizon 3] [--json OUT]
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
            r["post"] = s["post_start"] if s and s["post_start"] else r["start"]
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
        """Per target record: the drips published before its post (expert -> land ns), and the admitted picks in pick
        order. A drip's target is its row's first record after the scored (source) record."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("gate", "arms"))
    p.add_argument("capture")
    p.add_argument("--skip", type=int, default=8)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--json")
    a = p.parse_args()
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
