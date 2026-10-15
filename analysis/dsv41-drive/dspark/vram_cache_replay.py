"""Offline replay of the VRAM hot cache on a DSpark capture's route trace: today's policy against variants and the
offline optimum, per layer and per slot of capacity.

Input: a spec_margin_capture.py dir's ``stages.jsonl`` ``graph_routes`` lines (schema 2). Each verify forward carries
every MoE layer's routes (``routes[layer]``: the verify's tokens x top_k expert ids, token-major, token 0 the bonus
token and tokens 1.. the drafts), ``misses[layer]`` (the gather's miss count) and ``hot[layer]`` (the layer's VRAM
residents when the forward was planned). Two consecutive snapshots tell what the live policy did between them:
``hot[t+1] - hot[t]`` are the experts the gathers of forward t inserted (the lanes the GPU served: HitCopy, SM copy,
streamed misses), and the misses of forward t that are not in ``hot[t+1]`` were served without insertion (the CPU
lanes, expert_residency_gpu._fused_commit_gather leaves them unmapped). ``verify-accept.<pid>.jsonl`` (rid, k,
num_correct_drafts) joined by rid order says which tokens were accepted, and ``draft-top-prob.<pid>.jsonl`` gives the
draft's confidence per drafted token, a gather-time proxy for acceptance.

Policies, each replayed per layer over the verify sequence with capacity C:

* ``actual``: the capture's own hit count, ``|unique(routes) & hot|``; no simulation.
* ``score-k``: today's rule, simulated. Victims are the residents not routed by this forward, those the previous
  forward did not route first, then lowest ``(insert score, -expert)``; the insert score decays by
  ``decay ** forward_tokens`` per forward and adds the forward's route counts (SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY,
  GpuResidencyUpdater._apply). Only the k highest-keyed misses are inserted (the fused plan sorts miss lanes by key,
  highest first, and the CPU takes the tail), k being the number the capture itself inserted at that (layer, forward).
  Variants change the order (``cur``: score plus this forward's token count, known at the gather; ``conf``: score plus
  the draft-confidence-weighted count; ``heavy``/``light``/``random``: by this forward's tokens, as a split that
  ignores residency might send lanes), the bypass (``acc0``: a staged lane whose tokens were all rejected is not
  committed, a lazy insert decided after acceptance; ``conf<x``: a lane whose confidence-weighted count is under x is
  not inserted) and the score's input (``acc``: accepted tokens only; ``conf``: confidence-weighted).
* ``score-all``: today's victims, every miss inserted (the CPU lanes' experts too). ``score-budget-B``: at most B.
* ``lru`` / ``lfu``: insert every miss, evict the least recently / least frequently used unrouted resident.
* ``belady``: the offline optimum with bypass (a miss is inserted only when its next use comes before the farthest
  resident's); ``belady-fill``: Belady without bypass; ``next1``: a one-verify oracle (inserts the misses the next
  verify routes, evicting lowest-scored residents it does not route). Bounds, not rules.

Reports per policy: unique hits per verify (and token-weighted pairs), inserts per verify (each is a 13.3 MB H2D copy
and an eviction), the share of inserts evicted before any reuse, and a capacity sweep. The allocation report spends a
slot budget across layers greedily on each layer's marginal hits, against the uniform split. Reuse distances are in
verifies and as LRU stack distances. The signals table gives, for the capture's own misses, the share routed again
within 1 and 4 verifies by what the gather could know (this forward's tokens, prior score, draft confidence) and by
what only the boundary knows (acceptance).

Usage: vram_cache_replay.py CAPTURE [--skip 8] [--decay 0.98] [--capacities 16,22,...] [--budgets 853,951,...]
                            [--ms-per-hit 0.37] [--json OUT]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import random
import statistics
import sys

INF = 1 << 60


def load(d: str, skip: int):
    header, recs = None, []
    with open(os.path.join(d, "stages.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            kind = r.get("kind")
            if kind == "graph_routes_header":
                header = r
            elif kind == "graph_routes" and r.get("phase") == "target_verify" and "hot" in r:
                recs.append(r)
    if header is None or not recs:
        raise ValueError(f"{d}: no schema-2 graph_routes with hot sets")
    attach_acceptance(d, recs)
    return header, recs[skip:]


def attach_acceptance(d: str, recs: list) -> None:
    """Join verify-accept (and draft-top-prob) lines to the route records by rid, in order of occurrence."""
    accepts = collections.defaultdict(list)
    for path in glob.glob(os.path.join(d, "verify-accept.*.jsonl")):
        with open(path) as f:
            for e in map(json.loads, f):
                accepts[e["rid"]].append(e)
    confs = {}
    for path in glob.glob(os.path.join(d, "draft-top-prob.*.jsonl")):
        with open(path) as f:
            for e in map(json.loads, f):
                confs[(e["rid"], e["k"])] = e["top_prob"]
    seen = collections.Counter()
    for r in recs:
        r["accepted"] = None
        r["conf"] = None
        rids = r.get("rids") or []
        if len(rids) != 1:
            continue
        rid = rids[0]
        k = seen[rid]
        seen[rid] += 1
        if k < len(accepts[rid]):
            e = accepts[rid][k]
            r["accepted"] = int(e["num_correct_drafts"])
            r["conf"] = confs.get((rid, e["k"]))


class LayerTrace:
    """One layer's verify sequence: per forward the routed experts with token counts and the hot snapshot."""

    def __init__(self, layer: int, recs: list):
        self.layer = layer
        self.counts, self.acc_counts, self.conf_counts = [], [], []  # Counter expert -> tokens / accepted / weight
        self.hot = []
        self.tokens = []
        self.reported_misses = []
        self.has_accept = all(r.get("accepted") is not None for r in recs)
        self.has_conf = all(r.get("conf") for r in recs)
        for r in recs:
            flat = r["routes"][layer]
            ntok = int(r.get("forward_tokens") or 1)
            topk = len(flat) // ntok if ntok else len(flat)
            cnt, acc, conf = collections.Counter(), collections.Counter(), collections.Counter()
            accepted = r.get("accepted")
            probs = r.get("conf")
            for i, e in enumerate(flat):
                if e < 0:
                    continue
                tok = i // topk if topk else 0
                cnt[e] += 1
                if accepted is not None and tok <= accepted:
                    acc[e] += 1
                if probs is not None:
                    conf[e] += 1.0 if tok == 0 else (probs[tok - 1] if tok - 1 < len(probs) else 0.0)
            self.counts.append(cnt)
            self.acc_counts.append(acc)
            self.conf_counts.append(conf)
            self.hot.append(frozenset(r["hot"][layer]))
            self.tokens.append(ntok)
            self.reported_misses.append(int(r["misses"][layer]))
        self.n = len(recs)
        last = {}
        nxt = [dict() for _ in range(self.n)]
        for t in range(self.n - 1, -1, -1):
            for e in self.counts[t]:
                nxt[t][e] = last.get(e, INF)
                last[e] = t
        self.next_use = nxt
        self.occurrences = collections.defaultdict(list)
        for t in range(self.n):
            for e in self.counts[t]:
                self.occurrences[e].append(t)

    def _next_after(self, e: int, t: int) -> int:
        """The first forward after t that routes e, INF when none."""
        occ = self.occurrences.get(e)
        if not occ:
            return INF
        i = bisect.bisect_right(occ, t)
        return occ[i] if i < len(occ) else INF

    def actual(self) -> dict:
        """The live policy as the snapshots show it: hits, inserts, each miss's fate, and the signals table."""
        out = collections.Counter()
        reuse = {"inserted": [], "skipped": [], "evicted": []}
        sig = collections.defaultdict(lambda: [0, 0, 0])  # signal -> [n, reused within 1, within 4]
        score = collections.defaultdict(float)
        for t in range(self.n):
            cnt = self.counts[t]
            u = set(cnt)
            hot = self.hot[t]
            hits = u & hot
            misses = u - hot
            out["hits"] += len(hits)
            out["pairs_hit"] += sum(cnt[e] for e in hits)
            out["pairs"] += sum(cnt.values())
            out["unique"] += len(u)
            out["reported_misses"] += self.reported_misses[t]
            out["misses"] += len(misses)
            for e in misses:
                d = self.next_use[t][e] - t
                keys = [f"tokens={min(cnt[e], 3)}", "prior=" + ("seen" if score[e] > 0 else "first")]
                if self.has_accept:
                    a = self.acc_counts[t][e]
                    keys.append("accepted=" + ("0" if a == 0 else "1+"))
                    keys.append(f"acc{min(a, 2)}_tok{min(cnt[e], 3)}")
                if self.has_conf:
                    c = self.conf_counts[t][e]
                    keys.append("conf=" + ("<0.3" if c < 0.3 else "<0.7" if c < 0.7 else "<1.5" if c < 1.5 else "1.5+"))
                for k in keys:
                    s = sig[k]
                    s[0] += 1
                    s[1] += d <= 1
                    s[2] += d <= 4
            if t + 1 < self.n:
                nxt = self.hot[t + 1]
                inserted = misses & nxt
                skipped = misses - nxt
                evicted = hot - nxt
                out["inserted"] += len(inserted)
                out["skipped"] += len(skipped)
                out["evicted"] += len(evicted)
                out["other_in"] += len(nxt - hot - misses)
                for e in inserted:
                    reuse["inserted"].append(self.next_use[t][e] - t)
                    out["inserted_total"] += 1
                    nu = self.next_use[t][e]
                    if nu < self.n and e in self.hot[nu]:
                        out["inserted_reused_while_resident"] += 1
                    s = sig["fate=inserted"]
                    s[0] += 1
                    s[1] += nu - t <= 1
                    s[2] += nu - t <= 4
                for e in skipped:
                    reuse["skipped"].append(self.next_use[t][e] - t)
                    s = sig["fate=cpu"]
                    s[0] += 1
                    s[1] += self.next_use[t][e] - t <= 1
                    s[2] += self.next_use[t][e] - t <= 4
                for e in evicted:
                    reuse["evicted"].append(self._next_after(e, t) - t)
            d = 0.98 ** self.tokens[t]
            for e in list(score):
                score[e] *= d
            for e, c in cnt.items():
                score[e] += c
        return dict(out), reuse, {k: tuple(v) for k, v in sig.items()}

    def reuse_distances(self):
        """(verify distance, LRU stack distance) of every access, INF for a first touch."""
        last_seen = {}
        stack = []
        verify_d, stack_d = [], []
        for t in range(self.n):
            for e in self.counts[t]:
                if e in last_seen:
                    verify_d.append(t - last_seen[e])
                    stack_d.append(len(stack) - 1 - stack.index(e))
                    stack.remove(e)
                else:
                    verify_d.append(INF)
                    stack_d.append(INF)
                stack.append(e)
                last_seen[e] = t
        return verify_d, stack_d

    # ---- simulations -------------------------------------------------------------------------------------------

    def simulate(self, policy: str, capacity: int, *, decay: float = 0.98, margin: float = 0.0, budget: int = 0,
                 actual_k: bool = False, order: str = "score", bypass: str = "none", score_mode: str = "all",
                 seed: int = 1) -> dict:
        resident = set()
        score = collections.defaultdict(float)
        last_used = {}
        freq = collections.Counter()
        routed_prev = set()
        out = collections.Counter()
        insert_t = {}
        wasted = 0
        rng = random.Random(seed)
        for t in range(self.n):
            cnt = self.counts[t]
            u = set(cnt)
            hits = u & resident
            misses = u - resident
            out["hits"] += len(hits)
            out["pairs_hit"] += sum(cnt[e] for e in hits)
            out["pairs"] += sum(cnt.values())
            out["unique"] += len(u)
            for e in hits:
                insert_t.pop(e, None)
            if policy.startswith("belady"):
                nu = self.next_use[t]
                for e in sorted(misses, key=lambda x: nu[x]):
                    if nu[e] == INF and policy == "belady":
                        continue
                    if len(resident) < capacity:
                        resident.add(e)
                        out["inserts"] += 1
                        continue
                    cands = [r for r in resident if r not in u]
                    if not cands:
                        continue
                    victim = max(cands, key=lambda r: self._next_after(r, t))
                    if policy == "belady" and self._next_after(victim, t) <= nu[e]:
                        continue
                    resident.remove(victim)
                    resident.add(e)
                    out["inserts"] += 1
                    out["evictions"] += 1
                continue
            # the score of record: decayed by the forward's tokens, plus this forward's counts
            d = decay ** self.tokens[t]
            for e in list(score):
                score[e] *= d
            src = {"all": cnt, "acc": self.acc_counts[t], "conf": self.conf_counts[t]}[score_mode]
            for e, c in src.items():
                score[e] += c
            for e in u:
                freq[e] += cnt[e]
                last_used[e] = t
            # miss lanes in plan order
            if policy == "next1":
                nxt_u = set(self.counts[t + 1]) if t + 1 < self.n else set()
                cand = [e for e in misses if e in nxt_u]
                cand.sort(key=lambda e: (-score[e], e))
                for e in cand:
                    if len(resident) < capacity:
                        resident.add(e)
                        out["inserts"] += 1
                        continue
                    pool = [r for r in resident if r not in u and r not in nxt_u]
                    if not pool:
                        break
                    victim = min(pool, key=lambda r: (score[r], -r))
                    resident.remove(victim)
                    resident.add(e)
                    out["inserts"] += 1
                    out["evictions"] += 1
                routed_prev = u
                continue
            if order == "score":
                plan = sorted(misses, key=lambda e: (-score[e], e))
            elif order == "cur":
                # the gather knows this forward's routes; the live key does not yet include them
                plan = sorted(misses, key=lambda e: (-(score[e] + (0 if score_mode != "all" else 0) + cnt[e]), e))
            elif order == "conf":
                plan = sorted(misses, key=lambda e: (-(score[e] + self.conf_counts[t][e]), e))
            elif order == "heavy":
                plan = sorted(misses, key=lambda e: (-cnt[e], -score[e], e))
            elif order == "light":
                plan = sorted(misses, key=lambda e: (cnt[e], -score[e], e))
            elif order == "random":
                plan = sorted(misses)
                rng.shuffle(plan)
            else:
                raise ValueError(order)
            k = len(plan)
            if actual_k and t + 1 < self.n:
                k = len(misses & self.hot[t + 1])
            if budget:
                k = min(k, budget)
            for e in plan[:k]:
                if bypass == "acc0" and self.has_accept and self.acc_counts[t][e] == 0:
                    continue
                if bypass.startswith("conf<") and self.has_conf and self.conf_counts[t][e] < float(bypass[5:]):
                    continue
                if len(resident) < capacity:
                    resident.add(e)
                    out["inserts"] += 1
                    insert_t[e] = t
                    continue
                cands = [r for r in resident if r not in u]
                if not cands:
                    break
                if policy == "lru":
                    victim = min(cands, key=lambda r: last_used.get(r, -1))
                elif policy == "lfu":
                    victim = min(cands, key=lambda r: (freq[r], last_used.get(r, -1)))
                else:
                    victim = min(cands, key=lambda r: (r in routed_prev, score[r], -r))
                    if margin and score[e] <= score[victim] + margin:
                        continue
                resident.remove(victim)
                if insert_t.pop(victim, None) is not None:
                    wasted += 1
                resident.add(e)
                out["inserts"] += 1
                out["evictions"] += 1
                insert_t[e] = t
            routed_prev = u
        out["wasted_inserts"] = wasted
        return dict(out)


def quantiles(xs, qs=(0.5, 0.75, 0.9)):
    xs = sorted(x for x in xs if x < INF)
    if not xs:
        return {}
    return {str(q): xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs}


def reuse_summary(ds):
    n = len(ds)
    fin = [d for d in ds if d < INF]
    return {"n": n, "never": (n - len(fin)) / max(1, n),
            **{f"within_{w}": sum(d <= w for d in fin) / max(1, n) for w in (1, 2, 4, 8, 16)},
            "quantiles": quantiles(ds)}


POLICIES = [
    ("score-k", "score", dict(actual_k=True)),
    ("score-k-cur", "score", dict(actual_k=True, order="cur")),
    ("score-k-conf", "score", dict(actual_k=True, order="conf")),
    ("score-k-heavy", "score", dict(actual_k=True, order="heavy")),
    ("score-k-light", "score", dict(actual_k=True, order="light")),
    ("score-k-random", "score", dict(actual_k=True, order="random")),
    ("score-k-acc0lazy", "score", dict(actual_k=True, order="cur", bypass="acc0")),
    ("score-k-conf<0.5", "score", dict(actual_k=True, order="conf", bypass="conf<0.5")),
    ("score-k-accscore", "score", dict(actual_k=True, order="cur", score_mode="acc")),
    ("score-k-acc-lazy+accscore", "score", dict(actual_k=True, order="cur", bypass="acc0", score_mode="acc")),
    ("score-k-conf+confscore", "score", dict(actual_k=True, order="conf", score_mode="conf")),
    ("score-k-d0.90", "score", dict(actual_k=True, decay=0.90)),
    ("score-k-d0.95", "score", dict(actual_k=True, decay=0.95)),
    ("score-k-d0.99", "score", dict(actual_k=True, decay=0.99)),
    ("score-k-d0.995", "score", dict(actual_k=True, decay=0.995)),
    ("score-k-d1.0", "score", dict(actual_k=True, decay=1.0)),
    ("score-all", "score", {}),
    ("score-all-acc0lazy", "score", dict(bypass="acc0")),
    ("score-better-1", "score", dict(margin=1.0)),
    ("score-budget-4", "score", dict(budget=4, order="cur")),
    ("score-budget-6", "score", dict(budget=6, order="cur")),
    ("lru", "lru", {}),
    ("lfu", "lfu", {}),
    ("next1", "next1", {}),
    ("belady", "belady", {}),
    ("belady-fill", "belady-fill", {}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--decay", type=float, default=0.98)
    ap.add_argument("--capacities", default="16,20,22,24,28,32,40,48,64,96")
    ap.add_argument("--budgets", default="853,951,1200,1600")
    ap.add_argument("--ms-per-hit", type=float, default=0.37,
                    help="traced ms a VRAM hit saves (lane_cost_fit.py: the lane-mix-weighted marginal cost)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    header, recs = load(args.capture, args.skip)
    layers = list(range(len(header["layer_ids"])))
    cap_hdr = header["hot_capacity"]
    traces = [LayerTrace(L, recs) for L in layers]
    nv = len(recs)
    res = {"capture": os.path.abspath(args.capture), "verifies": nv, "skipped": args.skip,
           "hot_capacity": cap_hdr, "slots": sum(cap_hdr), "tokens_per_verify": statistics.mean(traces[0].tokens),
           "accept_joined": sum(1 for r in recs if r.get("accepted") is not None),
           "conf_joined": sum(1 for r in recs if r.get("conf")),
           "accepted_mean": statistics.mean(r["accepted"] for r in recs if r.get("accepted") is not None)
           if any(r.get("accepted") is not None for r in recs) else None}

    act_tot = collections.Counter()
    per_layer = []
    reuse_all = {"inserted": [], "skipped": [], "evicted": []}
    sig_tot = collections.defaultdict(lambda: [0, 0, 0])
    for tr in traces:
        a, reuse, sig = tr.actual()
        act_tot.update(a)
        for k in reuse_all:
            reuse_all[k] += reuse[k]
        for k, (n, w1, w4) in sig.items():
            s = sig_tot[k]
            s[0] += n
            s[1] += w1
            s[2] += w4
        per_layer.append({"layer": tr.layer, "cap": cap_hdr[tr.layer], "unique": a["unique"] / nv,
                          "hits": a["hits"] / nv, "hit_rate": a["hits"] / max(1, a["unique"]),
                          "inserted": a["inserted"] / max(1, nv - 1), "skipped": a["skipped"] / max(1, nv - 1)})
    res["actual"] = {
        "unique_per_verify": act_tot["unique"] / nv, "hits_per_verify": act_tot["hits"] / nv,
        "hit_rate": act_tot["hits"] / act_tot["unique"], "pair_hit_rate": act_tot["pairs_hit"] / act_tot["pairs"],
        "inserted_per_verify": act_tot["inserted"] / (nv - 1), "skipped_per_verify": act_tot["skipped"] / (nv - 1),
        "evicted_per_verify": act_tot["evicted"] / (nv - 1),
        "inserted_reused_while_resident": act_tot["inserted_reused_while_resident"] / max(1, act_tot["inserted_total"]),
        "snapshot_check": {"reported_misses": act_tot["reported_misses"], "set_misses": act_tot["misses"],
                           "other_in": act_tot["other_in"]},
        "reuse": {k: reuse_summary(v) for k, v in reuse_all.items()},
        "signals": {k: {"n": n, "within_1": w1 / max(1, n), "within_4": w4 / max(1, n)}
                    for k, (n, w1, w4) in sorted(sig_tot.items())},
        "per_layer": per_layer,
    }

    vd, sd = [], []
    for tr in traces:
        a, b = tr.reuse_distances()
        vd += a
        sd += b
    n = len(vd)
    fin_sd = [d for d in sd if d < INF]
    res["reuse"] = {
        "accesses": n, "first_touch_share": (n - len(fin_sd)) / n,
        "verify_distance_quantiles": quantiles(vd), "stack_distance_quantiles": quantiles(sd),
        "lru_hit_rate_at": {str(c): sum(d < c for d in fin_sd) / n for c in (8, 16, 22, 24, 32, 48, 64, 96, 128, 192)},
    }

    res["policies"] = {}
    base_hits = res["actual"]["hits_per_verify"]
    base_ins = res["actual"]["inserted_per_verify"]
    for name, pol, kw in POLICIES:
        tot = collections.Counter()
        rows = []
        for tr in traces:
            s = tr.simulate(pol, cap_hdr[tr.layer], **({"decay": args.decay} | kw))
            tot.update(s)
            rows.append(round(s["hits"] / max(1, s["unique"]), 3))
        hp = tot["hits"] / nv
        res["policies"][name] = {
            "hit_rate": tot["hits"] / tot["unique"], "hits_per_verify": hp,
            "pair_hit_rate": tot["pairs_hit"] / tot["pairs"], "inserts_per_verify": tot["inserts"] / nv,
            "wasted_insert_share": tot["wasted_inserts"] / max(1, tot["inserts"]),
            "d_hits_vs_actual": hp - base_hits, "d_inserts_vs_actual": tot["inserts"] / nv - base_ins,
            "ms_per_verify_vs_actual": (hp - base_hits) * args.ms_per_hit, "per_layer_hit_rate": rows,
        }

    caps = [int(c) for c in args.capacities.split(",")]
    sweep, curves = {}, {}
    for name, pol, kw in (("score-all", "score", {}), ("score-k-cur", "score", dict(actual_k=True, order="cur")),
                          ("belady", "belady", {}), ("lru", "lru", {})):
        sweep[name], curves[name] = {}, {}
        for c in caps:
            tot = collections.Counter()
            for tr in traces:
                s = tr.simulate(pol, c, decay=args.decay, **kw)
                tot.update(s)
                curves[name].setdefault(tr.layer, {})[c] = s["hits"]
            sweep[name][str(c)] = {"hit_rate": tot["hits"] / tot["unique"], "hits_per_verify": tot["hits"] / nv,
                                  "inserts_per_verify": tot["inserts"] / nv}
    res["capacity_sweep"] = sweep

    alloc = {}
    for name in ("score-all", "score-k-cur", "belady"):
        alloc[name] = {}
        for budget in (int(b) for b in args.budgets.split(",")):
            give = {L: caps[0] for L in layers}
            spent = sum(give.values())
            while spent < budget:
                best, best_gain = None, 0.0
                for L in layers:
                    c = give[L]
                    i = caps.index(c)
                    if i + 1 >= len(caps) or spent + caps[i + 1] - c > budget:
                        continue
                    gain = (curves[name][L][caps[i + 1]] - curves[name][L][c]) / (caps[i + 1] - c)
                    if gain > best_gain:
                        best, best_gain = L, gain
                if best is None:
                    break
                i = caps.index(give[best])
                spent += caps[i + 1] - give[best]
                give[best] = caps[i + 1]
            total_hits = sum(curves[name][L][give[L]] for L in layers)
            per = budget / len(layers)
            lo = max(c for c in caps if c <= per)
            hi = min((c for c in caps if c >= per), default=lo)
            uni = 0.0
            for L in layers:
                a, b = curves[name][L][lo], curves[name][L][hi]
                uni += a if hi == lo else a + (b - a) * (per - lo) / (hi - lo)
            alloc[name][str(budget)] = {"spent": spent, "hits_per_verify": total_hits / nv,
                                        "uniform_hits_per_verify": uni / nv, "slots": [give[L] for L in layers]}
    res["allocation"] = alloc

    report(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


def report(res: dict) -> None:
    a = res["actual"]
    print(f"{res['capture']}: {res['verifies']} verifies after skip {res['skipped']}, {res['slots']} slots, "
          f"{res['tokens_per_verify']:.2f} drafted tokens/verify; acceptance joined {res['accept_joined']}/{res['verifies']} "
          f"(mean correct drafts {res['accepted_mean']}), confidence joined {res['conf_joined']}")
    c = a["snapshot_check"]
    print(f"snapshot check: reported misses {c['reported_misses']} vs set misses {c['set_misses']}, residents arriving "
          f"without a miss here {c['other_in']}")
    print(f"actual: {a['unique_per_verify']:.1f} unique/verify, {a['hits_per_verify']:.1f} VRAM hits ({a['hit_rate']:.3f} "
          f"unique, {a['pair_hit_rate']:.3f} by token pairs); inserted {a['inserted_per_verify']:.1f}, skipped (CPU) "
          f"{a['skipped_per_verify']:.1f} per verify; inserts reused while resident {a['inserted_reused_while_resident']:.3f}")
    for key, r in a["reuse"].items():
        print(f"  reuse of {key:9s}: n={r['n']} within 1/2/4/8/16 verifies="
              f"{r['within_1']:.3f}/{r['within_2']:.3f}/{r['within_4']:.3f}/{r['within_8']:.3f}/{r['within_16']:.3f} "
              f"q={r['quantiles']}")
    print("signals (the capture's misses: share routed again within 1 / 4 verifies):")
    for k, v in a["signals"].items():
        print(f"  {k:18s} n={v['n']:7d} within1={v['within_1']:.3f} within4={v['within_4']:.3f}")
    r = res["reuse"]
    print(f"reuse: {r['accesses']} accesses, first touch {r['first_touch_share']:.3f}, verify distance "
          f"q={r['verify_distance_quantiles']}, stack distance q={r['stack_distance_quantiles']}")
    print("  LRU hit rate at capacity: " + ", ".join(f"{k}:{v:.3f}" for k, v in r["lru_hit_rate_at"].items()))
    print("policies at the live capacity (d = vs actual; ms traced at --ms-per-hit):")
    for name, p in res["policies"].items():
        print(f"  {name:26s} hit {p['hit_rate']:.3f} ({p['hits_per_verify']:6.1f}/verify, d{p['d_hits_vs_actual']:+6.1f} = "
              f"{p['ms_per_verify_vs_actual']:+5.1f} ms/verify) pairs {p['pair_hit_rate']:.3f} inserts "
              f"{p['inserts_per_verify']:6.1f}/verify (d{p['d_inserts_vs_actual']:+6.1f}) wasted {p['wasted_insert_share']:.3f}")
    print("capacity sweep (hit rate | inserts/verify):")
    for name, sw in res["capacity_sweep"].items():
        print(f"  {name:12s} " + " ".join(f"{c}:{v['hit_rate']:.3f}|{v['inserts_per_verify']:.0f}" for c, v in sw.items()))
    print("allocation (greedy per-layer slots vs uniform, hits/verify):")
    for name, al in res["allocation"].items():
        for b, v in al.items():
            print(f"  {name:12s} budget {b}: greedy {v['hits_per_verify']:.1f} vs uniform {v['uniform_hits_per_verify']:.1f}")
    print("per layer (actual): layer cap unique hits rate inserted skipped")
    for L in a["per_layer"]:
        print(f"  {L['layer']:2d} {L['cap']:2d} {L['unique']:5.1f} {L['hits']:5.1f} {L['hit_rate']:.3f} {L['inserted']:4.1f} "
              f"{L['skipped']:4.1f}")


if __name__ == "__main__":
    sys.exit(main())
