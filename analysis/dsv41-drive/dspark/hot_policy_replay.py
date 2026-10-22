"""Lane-level replay of the VRAM hot cache's DIRECT insert-on-miss policy on a DSpark capture, for decay and
admission sweeps.

vram_cache_replay.py held each forward's insert count at the capture's own (``actual_k``) and ordered misses by a score
that already held the forward's own routes. Its replay of today's rule came out 22 hits a verify above the capture
(383.6 vs 361.3 on C2), larger than the decay effects it measured. This replay models each mechanism the code runs
instead, so it can be validated lane for lane against an InstrBuild capture's ``lane_kind`` events:

* Scores (GpuResidencyUpdater._apply, expert_residency_gpu.py:661-684): at every boundary, for every expert,
  ``insert_score = insert_score * decay**tokens + route_counts`` in float32, ``tokens`` being the graph rows (and eager
  tokens) counted since the previous boundary (_count, :653). A verify sets the boundary pending; the next forward's
  first gather applies it (on_graph_forward, :525); an eager forward flushes it first (observe_forward, :549), and an
  eager prefill of SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS (256) or more applies its own at once
  (expert_residency_clock.py:95-116). A graph-served extend sets none. Eager gathers never insert.
* The victim shortlist (_rank_victims, :881-922): free slots first, then residents ascending by ``(routed in the
  window just closed, insert score, -expert)`` (residency_rank_keys, expert_residency.py:335; _expert_rank_keys,
  :950); only the first SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES (8, arm_env.py:421) columns are valid.
* The miss lanes (expert_route_plan.py:151, enable_miss_order :930): the forward's distinct misses sorted by the same
  key, highest first. The gather drops shortlist entries whose slot this forward reads (direct_gather.cuh, hazard) and
  gives the survivors to lanes 0.. in order; the lanes past them are forced CPU lanes (lease_kernels.cuh:212-221).
* The split (type_lanes, lease_device.cuh:503-628): ``fixed`` is the captures' table (SGLANG_DSV41_CPU_EXPERTS_SPLIT,
  run_c2.sh) with the miss cut: node ``expert % 2`` takes the last ``split[n]`` of its n unforced RAM hits in lane
  order, one fewer when 1..3 of its forced lanes are NVMe misses; an unforced NVMe miss streams to the GPU. ``model``
  is production's split model (model_cpu_lanes, :380) at production's startup-calibrated costs: unforced RAM hits,
  heaviest-token first, go to their node's CPU while it would finish no later than the copy side would with the lane.
* The commit (_fused_commit_gather, :1141; direct_gather.cuh:202): a live lane the CPU did not take is inserted into
  its victim's slot; a CPU lane leaves its victim resident.

Modes. ``capture`` replays the captured arm (fixed split, NVMe status from the capture's lane kinds; an expert that
was a VRAM hit in the capture is taken as a RAM hit). ``today`` replays production as of 2026-10-22: every expert
pinned in RAM, the split model, and ``--slots`` spread over the layers. The span column in ``today`` mode is the split
model's own predicted expert time per verify (max of each node's CPU finish and the copy side, per layer, summed):
calibrated, untraced, and it prices insert copies at the copy lane's 0.968 ms of copy-engine time.

Admission (``adm``): a lane takes its victim only if its frequency estimate beats the victim's; a rejected miss is a
forced CPU lane and the victim is offered to the next miss. Estimates: ``s`` the insert score at the last boundary,
``s+cur`` plus this forward's routes (the gather has them), ``L<d>`` a second exact per-expert counter decayed by d per
token (both sides read it; [layers, experts] float32 is 61 KB, so a sketch buys nothing).

Usage: hot_policy_replay.py CAPTURE --out DIR [--mode capture|today] [--slots 1010] [--configs NAME,...] [--skip 8]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shlex
import sys
from bisect import bisect_right

import numpy as np

INF = 1 << 60

EXPERTS = 384
VICTIM_LANES = 8
PREFILL_BOUNDARY_TOKENS = 256
# run_c2.sh / breakdown run.sh: SGLANG_DSV41_CPU_EXPERTS_SPLIT for both nodes, n = 0..40 unforced RAM hits.
FIXED_SPLIT = [int(x) for x in "0,1,2,2,3,3,4,5,5,6,7,7,8,9,9,10,11,11,12,13,13,14,15,15,16,17,17,18,19,19,20,21,21,22,"
               "23,23,24,24,25,26,26".split(",")]
MISS_CUT, MISS_CUT_MAX = 1, 3
# Production's split model, calibrated at startup on 2026-10-22 15:31 (nvfp4-work/server.log): per node
# hit = a + b*k + c*pairs ms; copy_lane ms per copy lane.
MODEL_HIT = ((0.045, 0.229, 0.063), (-0.008, 0.220, 0.077))
MODEL_COPY_LANE = 0.968
# lane_kind kinds (lane_cost_fit.KIND): 1 DMA, 2 SM copy, 3 CPU RAM hit, 4 NVMe miss to GPU, 5 NVMe miss to CPU,
# 6 drip scratch hit.
GPU_KINDS = (1, 2, 4, 6)
NVME_KINDS = (4, 5)


# ---- loading -----------------------------------------------------------------------------------------------------


def load_capture(d: str) -> dict:
    """Forwards in forward_pass_id order: graph forwards (routes and the hot snapshot per layer) and eager ones
    (per-layer route counts), with each graph forward's lanes from the tier events when the capture has them."""
    header, graph, eager = None, [], collections.defaultdict(dict)
    with open(os.path.join(d, "stages.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            kind = r.get("kind")
            if kind == "graph_routes_header":
                header = r
            elif kind == "graph_routes" and "hot" in r and r.get("forward_pass_id", -1) >= 0:
                graph.append(r)
            elif (kind is None and "layer" in r and "experts" in r and r.get("forward_pass_id") is not None
                  and r.get("phase") != "target_verify"):
                # A per-layer verify record repeats the graph verify before it (same routes, forward_pass_id one
                # higher): verifies run in the graph. Eager extends are the prefills.
                e = eager[r["forward_pass_id"]]
                e["phase"], e["tokens"] = r["phase"], int(r["tokens"])
                e.setdefault("counts", {})[int(r["layer"])] = dict(zip(r["experts"], r["counts"]))
    if header is None or not graph:
        raise ValueError(f"{d}: no schema-2 graph_routes with hot sets")
    layers = len(header["layer_ids"])
    fwds = []
    for r in graph:
        tokens = int(r.get("forward_tokens") or 1)
        counts = []
        for L in range(layers):
            c = collections.Counter(e for e in r["routes"][L] if e >= 0)
            counts.append(dict(c))
        fwds.append({"graph": True, "fp": r["forward_pass_id"], "phase": r.get("phase"), "tokens": tokens,
                     "rid": (r.get("rids") or ["?"])[0],
                     "counts": counts, "hot": [frozenset(h) for h in r["hot"]]})
    graph_fps = {f["fp"] for f in fwds}
    for fp, e in eager.items():
        if fp in graph_fps:
            continue  # the graph record is the forward; a per-layer record of it would count its routes twice
        fwds.append({"graph": False, "fp": fp, "phase": e["phase"], "tokens": e["tokens"],
                     "counts": [e["counts"].get(L, {}) for L in range(layers)], "hot": None})
    fwds.sort(key=lambda f: f["fp"])
    lanes = attach_lanes(d, [f for f in fwds if f["graph"]], layers)
    return {"dir": os.path.abspath(d), "header": header, "layers": layers, "forwards": fwds, "lanes": lanes}


def attach_lanes(d: str, graph: list, layers: int) -> str:
    """Each graph forward's per-layer lanes ``{expert: (lane, kind)}`` from ``lane_kind``, or its NVMe misses from
    ``miss_expert`` when the capture has no lane kinds. Every graph forward posts one record per layer (gen = first
    gen of the forward + layer), the capture's warm-up and graph-capture forwards first, so the last ``len(graph)``
    forwards of the tier records are the stage records in order. Returns which source was used."""
    lk = collections.defaultdict(dict)
    me = collections.defaultdict(set)
    gens = set()
    for path in glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl")):
        with open(path) as f:
            for line in f:
                if '"lane_kind"' in line:
                    e = json.loads(line)
                    lk[(e["row"], e["gen"])][e["a"]] = (e["c"], e["b"] & 0xFF)
                elif '"miss_expert"' in line:
                    e = json.loads(line)
                    me[(e["row"], e["gen"])].add(e["a"])
                elif '"svc_record"' in line or '"svc_routed"' in line:
                    e = json.loads(line)
                    if e["row"] == 0:
                        gens.add(e["gen"])
    firsts = sorted(gens)
    source = "lane_kind" if lk else ("miss_expert" if me else "none")
    if source == "none" or len(firsts) < len(graph):
        for g in graph:
            g["lanes"] = None
        return "none"
    firsts = firsts[len(firsts) - len(graph):]
    for g, g0 in zip(graph, firsts):
        per = []
        for L in range(layers):
            if source == "lane_kind":
                per.append(lk.get((L, g0 + L), {}))
            else:
                per.append({e: (-1, 4) for e in me.get((L, g0 + L), ())})
        g["lanes"] = per
    return source


# ---- the per-layer policy ----------------------------------------------------------------------------------------


class Config:
    def __init__(self, name: str, decay: float = 0.98, adm: str | None = None, margin: str = "0",
                 long_decay: float = 0.999, prefill_counts: bool = True, prefill_tokens: bool = True):
        self.name, self.decay, self.adm, self.margin, self.long_decay = name, decay, adm, margin, long_decay
        # Code-change candidates, not settings: False drops the eager prefills' route counts (which the code folds into
        # the insert score, exl3.py:796 -> _apply), and their tokens from the decay as well.
        self.prefill_counts = prefill_counts
        self.prefill_tokens = prefill_tokens

    def beats(self, f: float, v: float) -> bool:
        if self.margin.startswith("x"):
            return f > v * float(self.margin[1:])
        return f > v + float(self.margin)


def f32_decay(d: float, tokens: int) -> np.float32:
    return np.float32(1.0 if d >= 1.0 else d ** tokens)


class LayerSim:
    def __init__(self, cap: int, hot0, cfg: Config, mode: str):
        self.cfg, self.mode = cfg, mode
        init = sorted(hot0)[:cap]
        self.slot = init + [-1] * (cap - len(init))
        self.where = {e: s for s, e in enumerate(self.slot) if e >= 0}
        self.score = np.zeros(EXPERTS, dtype=np.float32)
        self.long = np.zeros(EXPERTS, dtype=np.float32)
        self.pend = np.zeros(EXPERTS, dtype=np.float32)
        self.tokens = 0
        self.pending = False
        self.key_routed = np.zeros(EXPERTS, dtype=bool)
        self.key_score = self.score.copy()
        self.key_long = self.long.copy()
        self.short = []
        self.rank()

    def boundary(self) -> None:
        t = self.tokens
        self.score = self.score * f32_decay(self.cfg.decay, t) + self.pend
        if self.cfg.adm and self.cfg.adm.startswith("L"):
            self.long = self.long * f32_decay(self.cfg.long_decay, t) + self.pend
        routed = self.pend > 0
        self.pend = np.zeros(EXPERTS, dtype=np.float32)
        self.tokens = 0
        self.pending = False
        self.key_routed, self.key_score, self.key_long = routed, self.score.copy(), self.long.copy()
        self.rank()

    def rank(self) -> None:
        """The shortlist: free slots by slot id, then residents by (routed, score, -expert); the first VICTIM_LANES."""
        free = [s for s, e in enumerate(self.slot) if e < 0]
        occ = sorted((s for s, e in enumerate(self.slot) if e >= 0),
                     key=lambda s: (self.key_routed[self.slot[s]], self.key_score[self.slot[s]], -self.slot[s]))
        self.short = (free + occ)[:VICTIM_LANES]

    def eager(self, counts: dict, tokens: int, phase: str) -> None:
        if self.pending:
            self.boundary()
        if self.cfg.prefill_tokens or phase == "target_verify":
            self.tokens += tokens
        if self.cfg.prefill_counts or phase == "target_verify":
            for e, c in counts.items():
                self.pend[e] += c
        if phase == "target_verify":
            self.pending = True
        elif tokens >= PREFILL_BOUNDARY_TOKENS:
            self.boundary()

    def freq(self, e: int, cur: int) -> float:
        a = self.cfg.adm
        if a == "s":
            return float(self.key_score[e])
        if a == "s+cur":
            return float(self.key_score[e]) + cur
        if a.startswith("L"):
            return float(self.key_long[e]) + (cur if a.endswith("+cur") else 0)
        raise ValueError(a)

    def victim_freq(self, v: int) -> float:
        return float(self.key_long[v]) if self.cfg.adm.startswith("L") else float(self.key_score[v])

    def graph(self, counts: dict, tokens: int, phase: str, kinds: dict, teacher=None) -> dict:
        """One graph forward. ``teacher`` (the capture's hot set before it) replaces the replay's residency first."""
        if self.pending:
            self.boundary()
        if teacher is not None:
            # the capture's residents, keeping the replay's slot of every expert both hold
            keep = [e if e in teacher else -1 for e in self.slot]
            new = sorted(teacher - set(keep))
            for s in range(len(keep)):
                if keep[s] < 0 and new:
                    keep[s] = new.pop(0)
            self.slot = keep
            self.where = {e: s for s, e in enumerate(self.slot) if e >= 0}
            self.rank()
        self.tokens += tokens
        u = set(counts)
        hits = [e for e in u if e in self.where]
        misses = sorted((e for e in u if e not in self.where),
                        key=lambda e: (-int(self.key_routed[e]), -float(self.key_score[e]), e))
        usable = [s for s in self.short if self.slot[s] < 0 or self.slot[s] not in u]
        if self.cfg.adm is None:
            live_lanes = misses[: len(usable)]
            assigned = list(zip(live_lanes, usable))
            forced = misses[len(usable):]
        else:
            assigned, forced, i = [], [], 0
            for e in misses:
                if i < len(usable):
                    v = self.slot[usable[i]]
                    if v < 0 or self.cfg.beats(self.freq(e, counts[e]), self.victim_freq(v)):
                        assigned.append((e, usable[i]))
                        i += 1
                        continue
                forced.append(e)
        cpu = self.split(assigned, forced, counts, kinds)
        inserted = []
        for e, s in assigned:
            if e in cpu:
                continue
            old = self.slot[s]
            if old >= 0:
                del self.where[old]
            self.slot[s] = e
            self.where[e] = s
            inserted.append(e)
        for e, c in counts.items():
            self.pend[e] += c
        if phase == "target_verify":
            self.pending = True
        out = {"unique": len(u), "hits": len(hits), "pairs": sum(counts.values()),
               "pairs_hit": sum(counts[e] for e in hits), "misses": len(misses), "inserted": len(inserted),
               "live": len(assigned), "forced": len(forced), "order": misses, "ins": inserted, "cpu_set": cpu}
        if self.mode == "today":
            out["span_ms"] = self.span(assigned, forced, cpu, counts)
        return out

    # ---- the split -----------------------------------------------------------------------------------------------

    def split(self, assigned: list, forced: list, counts: dict, kinds: dict) -> set:
        """The live lanes the CPU computes (forced lanes are the CPU's whatever this returns)."""
        if self.mode == "today":
            return self.split_model(assigned, forced, counts)
        return self.split_fixed(assigned, forced, kinds)

    @staticmethod
    def split_fixed(assigned: list, forced: list, kinds: dict) -> set:
        """``kinds``: the capture's lane kind of each expert it had as a lane. An unforced NVMe miss streams to the
        GPU and an unforced drip scratch hit is copied from scratch; neither is eligible for the CPU."""
        n = [0, 0]
        forced_misses = [0, 0]
        gpu_only = {e for e, k in kinds.items() if k in NVME_KINDS or k == 6}
        for e, _ in assigned:
            if e not in gpu_only:
                n[e % 2] += 1
        for e in forced:
            if kinds.get(e) in NVME_KINDS:
                forced_misses[e % 2] += 1
        take = []
        for node in (0, 1):
            t = FIXED_SPLIT[min(n[node], len(FIXED_SPLIT) - 1)] if n[node] else 0
            if MISS_CUT and 1 <= forced_misses[node] <= MISS_CUT_MAX:
                t = max(0, t - MISS_CUT)
            take.append(t)
        cpu = set()
        for e, _ in reversed(assigned):
            if e in gpu_only:
                continue
            if take[e % 2] > 0:
                take[e % 2] -= 1
                cpu.add(e)
        return cpu

    @staticmethod
    def node_finish(node: int, k: int, pairs: int) -> float:
        a, b, c = MODEL_HIT[node]
        return a + b * k + c * pairs if k > 0 else 0.0

    def split_model(self, assigned: list, forced: list, counts: dict) -> set:
        k, pairs = [0, 0], [0, 0]
        for e in forced:
            k[e % 2] += 1
            pairs[e % 2] += counts[e]
        cand = [e for e, _ in assigned]  # plan order; the greedy takes the heaviest, ties to the lowest lane
        order = sorted(range(len(cand)), key=lambda j: (-counts[cand[j]], j))
        copy = 0.0
        cpu = set()
        for j in order:
            e = cand[j]
            node = e % 2
            t = self.node_finish(node, k[node] + 1, pairs[node] + counts[e])
            if t <= copy + MODEL_COPY_LANE:
                cpu.add(e)
                k[node] += 1
                pairs[node] += counts[e]
            else:
                copy += MODEL_COPY_LANE
        return cpu

    def span(self, assigned: list, forced: list, cpu: set, counts: dict) -> float:
        k, pairs = [0, 0], [0, 0]
        for e in forced + [e for e, _ in assigned if e in cpu]:
            k[e % 2] += 1
            pairs[e % 2] += counts[e]
        copies = sum(1 for e, _ in assigned if e not in cpu)
        return max(self.node_finish(0, k[0], pairs[0]), self.node_finish(1, k[1], pairs[1]), copies * MODEL_COPY_LANE)


# ---- runs --------------------------------------------------------------------------------------------------------


def capacities(header: dict, slots: int | None) -> list:
    caps = list(header["hot_capacity"])
    if slots:
        extra = slots - sum(caps)
        i = 0
        while extra != 0:
            caps[i % len(caps)] += 1 if extra > 0 else -1
            extra += -1 if extra > 0 else 1
            i += 1
    return caps


POS_LIMITS = (4, 16, 1 << 30)
POS_BUCKETS = ("verify1-4", "verify5-16", "verify17+")


def replay(cap: dict, cfg: Config, mode: str, slots: int | None, skip: int, teacher: bool = False) -> dict:
    """Run every layer over the capture's forwards; totals over the verifies after ``skip``."""
    fwds = cap["forwards"]
    caps = capacities(cap["header"], slots)
    first = next(f for f in fwds if f["graph"])
    sims = [LayerSim(caps[L], first["hot"][L], cfg, mode) for L in range(cap["layers"])]
    tot = collections.Counter()
    val = collections.Counter()
    verify_i = 0
    graph = [f for f in fwds if f["graph"]]
    next_hot = {id(g): (graph[i + 1]["hot"] if i + 1 < len(graph) else None) for i, g in enumerate(graph)}
    # hits by the verify's position in its request: early verifies are where a long window's history is stalest
    by_pos = collections.defaultdict(collections.Counter)
    seen = collections.Counter()
    for f in fwds:
        if not f["graph"]:
            for L, sim in enumerate(sims):
                sim.eager(f["counts"][L], f["tokens"], f["phase"])
            continue
        is_verify = f["phase"] == "target_verify"
        counted = is_verify and verify_i >= skip
        if is_verify:
            verify_i += 1
            seen[f["rid"]] += 1
        pos = POS_BUCKETS[min(i for i, hi in enumerate(POS_LIMITS) if seen[f["rid"]] <= hi)]
        nh = next_hot[id(f)]
        for L, sim in enumerate(sims):
            lanes = (f.get("lanes") or [None] * cap["layers"])[L]
            kinds = {e: kind for e, (_, kind) in lanes.items()} if (lanes and mode == "capture") else {}
            o = sim.graph(f["counts"][L], f["tokens"], f["phase"], kinds, f["hot"][L] if teacher else None)
            if counted:
                for k in ("unique", "hits", "pairs", "pairs_hit", "misses", "inserted", "live", "forced"):
                    tot[k] += o[k]
                tot["span_ms"] += o.get("span_ms", 0.0)
                by_pos[pos]["hits"] += o["hits"]
                by_pos[pos]["unique"] += o["unique"]
            if teacher and counted and nh is not None:
                validate(val, f["hot"][L], nh[L], o, lanes)
        if counted:
            tot["verifies"] += 1
            by_pos[pos]["verifies"] += 1
    n = max(1, tot["verifies"])
    res = {"config": vars(cfg), "mode": mode, "slots": sum(caps), "verifies": tot["verifies"],
           "unique_per_verify": tot["unique"] / n, "hits_per_verify": tot["hits"] / n,
           "hit_rate": tot["hits"] / max(1, tot["unique"]), "pair_hit_rate": tot["pairs_hit"] / max(1, tot["pairs"]),
           "inserts_per_verify": tot["inserted"] / n, "cpu_lanes_per_verify": (tot["misses"] - tot["inserted"]) / n,
           "forced_per_verify": tot["forced"] / n, "live_per_verify": tot["live"] / n}
    if mode == "today":
        res["model_span_ms_per_verify"] = tot["span_ms"] / n
    res["hits_by_position"] = {p: {"verifies": c["verifies"], "hits_per_verify": c["hits"] / max(1, c["verifies"]),
                                   "hit_rate": c["hits"] / max(1, c["unique"])} for p, c in by_pos.items()}
    if teacher:
        res["validation"] = {k: v for k, v in val.items()}
    return res


def validate(val: collections.Counter, hot: frozenset, nxt: frozenset, o: dict, lanes) -> None:
    """One step from the capture's own residency: the replay's inserts and lane order against the capture's."""
    act_ins = (nxt - hot)
    pred = set(o["ins"])
    val["steps"] += 1
    val["act_inserted"] += len(act_ins)
    val["pred_inserted"] += len(pred)
    val["ins_agree"] += len(act_ins & pred)
    val["act_evicted"] += len(hot - nxt)
    if lanes and all(c >= 0 for c, _ in lanes.values()) and set(lanes) == set(o["order"]):
        val["order_steps"] += 1
        act_order = [e for e, _ in sorted(lanes.items(), key=lambda kv: kv[1][0])]
        val["order_exact"] += act_order == o["order"]
        val["lane_pos_agree"] += sum(a == b for a, b in zip(act_order, o["order"]))
        val["lanes"] += len(act_order)
        act_gpu = {e for e, (_, kind) in lanes.items() if kind in GPU_KINDS}
        pred_gpu = set(o["ins"])
        val["kind_agree"] += sum((e in act_gpu) == (e in pred_gpu) for e in lanes)
        val["act_gpu_lanes"] += len(act_gpu)


# ---- a model-wide pool -------------------------------------------------------------------------------------------
#
# A hypothetical: one pool of N slots shared by every layer. Scores stay per (layer, expert) under the same rule, and
# they are comparable across layers as they stand: every layer routes every token top_k times, so each layer's counts
# sum to the same total, and the decay counts the same tokens for every layer (one device clock, _count :653).
# The shortlist becomes one global ranking at each boundary (free slots, then every resident by (routed, score,
# -expert)); each layer's gather, in layer order, takes up to VICTIM_LANES entries from its head, skipping the slots
# this forward already reassigned and its own hazards (residents of the layer this forward routes). A CPU lane leaves
# its victim for the next layer. The lanes, split and commit are the per-layer rules unchanged.


class GlobalSim:
    def __init__(self, hot0: list, slots: int, cfg: Config, mode: str):
        self.cfg, self.mode, self.layers = cfg, mode, len(hot0)
        init = [(L, e) for L, h in enumerate(hot0) for e in sorted(h)]
        self.slot = init[:slots] + [None] * max(0, slots - len(init))
        self.where = [dict() for _ in range(self.layers)]
        for s, le in enumerate(self.slot):
            if le is not None:
                self.where[le[0]][le[1]] = s
        self.score = np.zeros((self.layers, EXPERTS), dtype=np.float32)
        self.pend = np.zeros_like(self.score)
        self.tokens = 0
        self.pending = False
        self.key_routed = np.zeros((self.layers, EXPERTS), dtype=bool)
        self.key_score = self.score.copy()
        self.rank()

    def boundary(self) -> None:
        self.score = self.score * f32_decay(self.cfg.decay, self.tokens) + self.pend
        self.key_routed, self.key_score = self.pend > 0, self.score.copy()
        self.pend = np.zeros_like(self.score)
        self.tokens = 0
        self.pending = False
        self.rank()

    def rank(self) -> None:
        free = [s for s, le in enumerate(self.slot) if le is None]
        occ = sorted((s for s, le in enumerate(self.slot) if le is not None),
                     key=lambda s: (self.key_routed[self.slot[s]], self.key_score[self.slot[s]], -self.slot[s][1],
                                    -self.slot[s][0]))
        self.ranking = free + occ

    def eager(self, counts: list, tokens: int, phase: str) -> None:
        if self.pending:
            self.boundary()
        if self.cfg.prefill_tokens or phase == "target_verify":
            self.tokens += tokens
        if self.cfg.prefill_counts or phase == "target_verify":
            for L, c in enumerate(counts):
                for e, n in c.items():
                    self.pend[L, e] += n
        if phase == "target_verify":
            self.pending = True
        elif tokens >= PREFILL_BOUNDARY_TOKENS:
            self.boundary()

    def graph(self, f: dict, kinds_of) -> list:
        if self.pending:
            self.boundary()
        self.tokens += f["tokens"]
        moved = set()  # slots reassigned by this forward
        outs = []
        for L in range(self.layers):
            counts = f["counts"][L]
            u = set(counts)
            where = self.where[L]
            hits = [e for e in u if e in where]
            misses = sorted((e for e in u if e not in where),
                            key=lambda e: (-int(self.key_routed[L, e]), -float(self.key_score[L, e]), e))
            usable = []
            for s in self.ranking:
                if len(usable) >= min(VICTIM_LANES, len(misses)):
                    break
                le = self.slot[s]
                if s in moved or (le is not None and le[0] == L and le[1] in u):
                    continue
                usable.append(s)
            if self.cfg.adm is None:
                assigned = list(zip(misses, usable))
                forced = misses[len(usable):]
            else:
                assigned, forced, i = [], [], 0
                for e in misses:
                    if i < len(usable):
                        v = self.slot[usable[i]]
                        if v is None or self.cfg.beats(self.key_score[L, e] + (counts[e] if self.cfg.adm.endswith("+cur") else 0),
                                                       float(self.key_score[v])):
                            assigned.append((e, usable[i]))
                            i += 1
                            continue
                    forced.append(e)
            kinds = kinds_of(L)
            cpu = (LayerSim.split_model(LayerSim, assigned, forced, counts) if self.mode == "today"
                   else LayerSim.split_fixed(assigned, forced, kinds))
            inserted = cross = 0
            for e, s in assigned:
                if e in cpu:
                    continue
                old = self.slot[s]
                if old is not None:
                    del self.where[old[0]][old[1]]
                    cross += old[0] != L
                self.slot[s] = (L, e)
                where[e] = s
                moved.add(s)
                inserted += 1
            for e, n in counts.items():
                self.pend[L, e] += n
            o = {"unique": len(u), "hits": len(hits), "pairs": sum(counts.values()),
                 "pairs_hit": sum(counts[e] for e in hits), "misses": len(misses), "inserted": inserted,
                 "live": len(assigned), "forced": len(forced), "cross": cross}
            if self.mode == "today":
                o["span_ms"] = LayerSim.span(LayerSim, assigned, forced, cpu, counts)
            outs.append(o)
        if f["phase"] == "target_verify":
            self.pending = True
        return outs


def replay_global(cap: dict, cfg: Config, mode: str, slots: int | None, skip: int) -> dict:
    fwds = cap["forwards"]
    total = slots or sum(cap["header"]["hot_capacity"])
    first = next(f for f in fwds if f["graph"])
    sim = GlobalSim(first["hot"], total, cfg, mode)
    tot = collections.Counter()
    occ = [[] for _ in range(cap["layers"])]
    verify_i = 0
    for f in fwds:
        if not f["graph"]:
            sim.eager(f["counts"], f["tokens"], f["phase"])
            continue
        is_verify = f["phase"] == "target_verify"
        counted = is_verify and verify_i >= skip
        verify_i += is_verify
        lanes = f.get("lanes") or [None] * cap["layers"]
        outs = sim.graph(f, lambda L: ({e: k for e, (_, k) in lanes[L].items()} if lanes[L] and mode == "capture"
                                       else {}))
        if counted:
            tot["verifies"] += 1
            for o in outs:
                for k in ("unique", "hits", "pairs", "pairs_hit", "misses", "inserted", "live", "forced", "cross"):
                    tot[k] += o[k]
                tot["span_ms"] += o.get("span_ms", 0.0)
            for L in range(cap["layers"]):
                occ[L].append(len(sim.where[L]))
    n = max(1, tot["verifies"])
    mean_occ = [float(np.mean(o)) for o in occ]
    order = sorted(range(len(mean_occ)), key=lambda L: -mean_occ[L])
    res = {"config": vars(cfg), "mode": mode, "pool": "global", "slots": total, "verifies": tot["verifies"],
           "unique_per_verify": tot["unique"] / n, "hits_per_verify": tot["hits"] / n,
           "hit_rate": tot["hits"] / max(1, tot["unique"]), "pair_hit_rate": tot["pairs_hit"] / max(1, tot["pairs"]),
           "inserts_per_verify": tot["inserted"] / n, "cpu_lanes_per_verify": (tot["misses"] - tot["inserted"]) / n,
           "forced_per_verify": tot["forced"] / n, "live_per_verify": tot["live"] / n,
           "cross_layer_evictions_per_verify": tot["cross"] / n,
           "slots_by_layer_mean": [round(x, 1) for x in mean_occ],
           "slots_by_layer_summary": {"min": min(mean_occ), "median": float(np.median(mean_occ)), "max": max(mean_occ),
                                      "top5": [(L, round(mean_occ[L], 1)) for L in order[:5]],
                                      "bottom5": [(L, round(mean_occ[L], 1)) for L in order[-5:]],
                                      "mean_abs_step_change": float(np.mean([np.mean(np.abs(np.diff(o))) for o in occ]))}}
    if mode == "today":
        res["model_span_ms_per_verify"] = tot["span_ms"] / n
    return res


def belady_bounds(cap: dict, slots: int | None, skip: int) -> dict:
    """Belady with bypass over the graph forwards' distinct (layer, expert) accesses, no lane or insert limits: each
    layer at its own capacity, and one pool of the same total. Hits are counted on the verifies after ``skip``."""
    import heapq

    graph = [f for f in cap["forwards"] if f["graph"]]
    layers = cap["layers"]
    caps = capacities(cap["header"], slots)
    seq = []  # (time, layer, experts, counted)
    vi = 0
    for t, f in enumerate(graph):
        is_verify = f["phase"] == "target_verify"
        counted = is_verify and vi >= skip
        vi += is_verify
        for L in range(layers):
            seq.append((t * layers + L, L, sorted(f["counts"][L]), counted))
    occ = collections.defaultdict(list)
    for g, L, es, _ in seq:
        for e in es:
            occ[(L, e)].append(g)

    def next_use(key, g):
        o = occ[key]
        i = bisect_right(o, g)
        return o[i] if i < len(o) else INF

    def run(pool_of, cap_of):
        resident = {}  # key -> next use
        heaps = collections.defaultdict(list)
        size = collections.Counter()
        hits = unique = 0
        for g, L, es, counted in seq:
            p = pool_of(L)
            for e in es:
                key = (L, e)
                if key in resident:
                    hits += counted
                    resident[key] = next_use(key, g)
                    heapq.heappush(heaps[p], (-resident[key], key))
            for e in es:
                key = (L, e)
                unique += counted
                if key in resident and resident[key] > g:
                    continue
                nu = next_use(key, g)
                if nu == INF:
                    continue
                if size[p] < cap_of(p):
                    resident[key] = nu
                    size[p] += 1
                    heapq.heappush(heaps[p], (-nu, key))
                    continue
                h = heaps[p]
                while h and resident.get(h[0][1]) != -h[0][0]:
                    heapq.heappop(h)
                if not h or -h[0][0] <= nu:
                    continue
                _, victim = heapq.heappop(h)
                del resident[victim]
                resident[key] = nu
                heapq.heappush(h, (-nu, key))
        return hits, unique

    n = sum(1 for _, L, _, c in seq if c and L == 0)
    out = {}
    for name, pool_of, cap_of in (("per_layer", lambda L: L, lambda p: caps[p]),
                                  ("global", lambda L: 0, lambda p: sum(caps))):
        h, u = run(pool_of, cap_of)
        out[name] = {"hits_per_verify": h / n, "hit_rate": h / u}
    return out


CONFIGS = {}


def _add(cfg: Config) -> None:
    CONFIGS[cfg.name] = cfg


for _d in (0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0):
    _add(Config(f"d{_d}", decay=_d))
for _d in (0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0):
    _add(Config(f"d{_d}-noprefill", decay=_d, prefill_counts=False))
for _d in (0.98, 0.995, 0.999, 1.0):
    _add(Config(f"d{_d}-prefillneutral", decay=_d, prefill_counts=False, prefill_tokens=False))
for _base, _d, _pf in (("d0.98", 0.98, True), ("d0.995", 0.995, True), ("d0.999-noprefill", 0.999, False)):
    for _adm in ("s", "s+cur", "L0.999", "L0.999+cur", "L0.9995+cur"):
        for _m in ("0", "1", "x1.25"):
            ld = float(_adm[1:].split("+")[0]) if _adm.startswith("L") else 0.999
            _add(Config(f"{_base}-adm{_adm}-m{_m}", decay=_d, adm=_adm, margin=_m, long_decay=ld, prefill_counts=_pf))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="capture dirs; name=dir to label one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("capture", "today"), default="capture")
    ap.add_argument("--slots", type=int, default=0, help="total slots, spread over the layers (0: the capture's)")
    ap.add_argument("--configs", default="d0.98", help="comma-separated names, 'decays', 'all'")
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--validate", action="store_true", help="also run d0.98 one step at a time from the capture's sets")
    ap.add_argument("--pool", choices=("layer", "global"), default="layer",
                    help="global: one pool of --slots shared by every layer (replay_global)")
    ap.add_argument("--belady", action="store_true", help="also the per-layer and global Belady bounds")
    args = ap.parse_args()
    if args.configs == "all":
        names = list(CONFIGS)
    elif args.configs == "decays":
        names = [n for n in CONFIGS if "adm" not in n]
    else:
        names = args.configs.split(",")
    caps = []
    for spec in args.captures:
        label, _, path = spec.rpartition("=")
        cap = load_capture(path)
        caps.append((label or os.path.basename(path.rstrip("/")), cap))
        print(f"{label or path}: {sum(f['graph'] for f in cap['forwards'])} graph forwards, "
              f"{sum(not f['graph'] for f in cap['forwards'])} eager, lanes from {cap['lanes']}", flush=True)
    cmd = " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)
    tag = ("global-" if args.pool == "global" else "") + f"{args.mode}-{args.slots or 'cap'}"
    run = replay_global if args.pool == "global" else replay
    if args.belady:
        rows = {label: belady_bounds(cap, args.slots or None, args.skip) for label, cap in caps}
        d = os.path.join(args.out, tag, "belady")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "result.json"), "w") as f:
            json.dump(rows, f, indent=1)
        with open(os.path.join(d, "command.txt"), "w") as f:
            f.write(cmd + "\n")
        for label, r in rows.items():
            print(f"BELADY {label}: " + json.dumps(r), flush=True)
    for name in names:
        cfg = CONFIGS[name]
        rows = {}
        for label, cap in caps:
            rows[label] = run(cap, cfg, args.mode, args.slots or None, args.skip)
        pooled = pool(rows)
        rows["pooled"] = pooled
        d = os.path.join(args.out, tag, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "result.json"), "w") as f:
            json.dump(rows, f, indent=1)
        with open(os.path.join(d, "command.txt"), "w") as f:
            f.write(cmd + "\n")
        print(name + "  " + "  ".join(f"{k}: hits {v['hits_per_verify']:.1f} rate {v['hit_rate']:.4f} ins "
                                      f"{v['inserts_per_verify']:.1f} cpu {v['cpu_lanes_per_verify']:.1f}"
                                      + (f" span {v['model_span_ms_per_verify']:.1f}" if "model_span_ms_per_verify" in v else "")
                                      for k, v in rows.items()), flush=True)
    if args.validate:
        for label, cap in caps:
            free = replay(cap, CONFIGS["d0.98"], args.mode, args.slots or None, args.skip)
            step = replay(cap, CONFIGS["d0.98"], args.mode, args.slots or None, args.skip, teacher=True)
            act = actual(cap, args.skip)
            out = {"actual": act, "free_running": free, "teacher_forced": step}
            d = os.path.join(args.out, tag, "validate", label)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "result.json"), "w") as f:
                json.dump(out, f, indent=1)
            with open(os.path.join(d, "command.txt"), "w") as f:
                f.write(cmd + "\n")
            print(f"VALIDATE {label}: actual hits {act['hits_per_verify']:.1f} inserts {act['inserts_per_verify']:.1f}"
                  f" | replay hits {free['hits_per_verify']:.1f} inserts {free['inserts_per_verify']:.1f}"
                  f" | one-step {json.dumps(step['validation'])}", flush=True)
    return 0


def pool(rows: dict) -> dict:
    n = sum(r["verifies"] for r in rows.values())
    out = {"verifies": n}
    for k in rows[next(iter(rows))]:
        if k.endswith("_per_verify"):
            out[k] = sum(r[k] * r["verifies"] for r in rows.values()) / n
    uniq = sum(r["unique_per_verify"] * r["verifies"] for r in rows.values())
    out["hit_rate"] = sum(r["hits_per_verify"] * r["verifies"] for r in rows.values()) / uniq
    return out


def actual(cap: dict, skip: int) -> dict:
    """The capture's own hits and inserts per verify, from its snapshots (hot[t+1] - hot[t] over graph forwards)."""
    graph = [f for f in cap["forwards"] if f["graph"]]
    tot = collections.Counter()
    vi = 0
    for i, f in enumerate(graph):
        if f["phase"] != "target_verify":
            continue
        vi += 1
        if vi <= skip:
            continue
        tot["verifies"] += 1
        for L in range(cap["layers"]):
            u = set(f["counts"][L])
            hot = f["hot"][L]
            tot["unique"] += len(u)
            tot["hits"] += len(u & hot)
            if i + 1 < len(graph):
                ins = graph[i + 1]["hot"][L] - hot
                tot["inserted"] += len(ins)
                tot["misses"] += len(u - hot)
    n = tot["verifies"]
    return {"verifies": n, "hits_per_verify": tot["hits"] / n, "hit_rate": tot["hits"] / tot["unique"],
            "unique_per_verify": tot["unique"] / n, "inserts_per_verify": tot["inserted"] / n,
            "cpu_lanes_per_verify": (tot["misses"] - tot["inserted"]) / n}


if __name__ == "__main__":
    sys.exit(main())
