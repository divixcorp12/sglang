"""Oracle headroom of a layer-completing prefetch on a DSpark verify capture: how many miss layers the scorer's
candidates could clear, and at what read cost.

The capture's forced misses per (verify step, layer) come from verify_replay's tier with no prefetch (routed, not
hot, not in the layer's RAM tier; the replay's total is 14.5% under the server's counters, §33.13). For each layer
T >= 1 the GPU scorer's candidate list is rebuilt from ``rank.npz`` (verify_gate_rankings.py, h = 1: layer T's gate on
the layer T-1 input): per live token its top-depth experts past hot and RAM-resident rows, merged by each expert's
best margin to its token's 6th score, as ``exl3_ram_prefetch_select_kernel`` merges them (with no per-token cap).

Two rules are scored, both over every layer >= 1 (a rule cannot know which layers miss):
- top-N: read the first N candidates. A miss layer is cleared when all its misses are among them.
- set: S = the candidates with margin >= theta; read all of S when |S| <= B, else none.
A cleared layer is priced at misses x the per-read cost for its miss count (``--read-cost``, ms, from §33.16) and a
partly covered one at half that per covered miss. These prices are rough; read the counts first.

Usage: layer_cover_oracle.py CAPTURE_DIR --out OUT.json [verify_replay options]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_replay  # noqa: E402

READ_COST_MS = (0.43, 0.67, 0.81, 1.14)  # per NVMe read left, at 1 / 2 / 3 / 4+ misses (§33.16)


def candidate_list(order, score, tokens: int, skip) -> list[tuple[int, float]]:
    """(expert, best margin) past ``skip``, by margin descending then id; order/score are [M, depth]."""
    best: dict[int, float] = {}
    for m in range(tokens):
        sixth = float(score[m, 5])
        for e, sc in zip(order[m].tolist(), score[m].tolist()):
            if e < 0 or skip(e):
                continue
            best[e] = max(best.get(e, -float("inf")), sc - sixth)
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


class Recorder(verify_replay.Replay):
    """The no-prefetch replay, recording each layer's misses and candidate list before it is served."""

    def __init__(self, a, layer_ids, ranks) -> None:
        super().__init__(a, layer_ids, ranks)
        self.rows: list[dict] = []

    def layer(self, s, fwd, li, now):
        layer = self.layer_ids[li]
        hot = set(fwd["hot"][layer]) if fwd.get("hot") else set()
        misses = [e for e in dict.fromkeys(fwd["routes"][layer]) if e not in hot and self.where(li, e)[1] is None]
        cand = []
        if li >= 1 and self.ranks["valid"][s, li, 1]:
            cand = candidate_list(self.ranks["order"][s, li, :, 1], self.ranks["score"][s, li, :, 1],
                                  int(self.ranks["tokens"][s]),
                                  lambda e: e in hot or self.where(li, e)[1] is not None)
        self.rows.append({"s": s, "li": li, "misses": misses, "cand": cand})
        return super().layer(s, fwd, li, now)


def price(misses: int, covered: int, cost=READ_COST_MS) -> float:
    c = cost[min(misses, len(cost)) - 1] if misses else 0.0
    return misses * c if covered == misses else 0.5 * covered * c


def score_rule(rows: list[dict], pick) -> dict:
    """``pick(cand) -> experts read`` over layers >= 1; per step: reads, useful, cleared, priced saving."""
    steps = len({r["s"] for r in rows}) or 1
    later = [r for r in rows if r["li"] >= 1]
    miss_layers = sum(bool(r["misses"]) for r in later) or 1
    reads = useful = cleared = 0
    saved = 0.0
    for r in later:
        chosen = set(pick(r["cand"]))
        covered = sum(e in chosen for e in r["misses"])
        reads += len(chosen)
        useful += covered
        cleared += bool(r["misses"]) and covered == len(r["misses"])
        saved += price(len(r["misses"]), covered)
    return {"reads": round(reads / steps, 1), "useful": round(useful / steps, 1),
            "cleared": round(cleared / steps, 1), "cleared_share": round(cleared / miss_layers, 3),
            "saved_ms": round(saved / steps, 1), "ms_per_read": round(saved / reads, 3) if reads else None}


def summarize(rows: list[dict]) -> dict:
    steps = len({r["s"] for r in rows}) or 1
    later = [r for r in rows if r["li"] >= 1]
    ml = [r for r in later if r["misses"]]
    hist = collections.Counter(min(len(r["misses"]), 4) for r in later)
    need = {}
    for i, r in enumerate(ml):
        pos = {e: j for j, (e, _) in enumerate(r["cand"])}
        p = [pos.get(e) for e in r["misses"]]
        need[i] = max(p) + 1 if None not in p else None
    in_list = sum(e in dict(r["cand"]) for r in ml for e in r["misses"])
    n_miss = sum(len(r["misses"]) for r in ml) or 1
    by_misses = {}
    for b in (1, 2, 3, 4):
        idx = [i for i, r in enumerate(ml) if min(len(r["misses"]), 4) == b]
        by_misses["4+" if b == 4 else str(b)] = {
            "layers": len(idx), "coverable_any_n": round(sum(need[i] is not None for i in idx) / max(len(idx), 1), 3),
            **{f"cleared_n{n}": round(sum(need[i] is not None and need[i] <= n for i in idx) / max(len(idx), 1), 3)
               for n in (1, 2, 4, 8)}}
    return {
        "steps": steps, "misses_per_step": round(sum(len(r["misses"]) for r in rows) / steps, 1),
        "miss_hist_rows_1_up": {("4+" if k == 4 else str(k)): hist[k] for k in sorted(hist)},
        "miss_layers_per_step": round(len(ml) / steps, 1),
        "misses_in_any_list": round(in_list / n_miss, 3),
        "layers_coverable_any_n": round(sum(v is not None for v in need.values()) / max(len(ml), 1), 3),
        "by_misses": by_misses,
        "top_n": {str(n): score_rule(rows, lambda c, n=n: [e for e, _ in c[:n]]) for n in (1, 2, 3, 4, 6, 8)},
        "set": {f"theta {th:g}, B {b}": score_rule(rows, lambda c, th=th, b=b: (
                    lambda s: s if len(s) <= b else [])([e for e, mg in c if mg >= th]))
                for th in (0.25, 0.1, 0.0, -0.1) for b in (1, 2, 3, 4, 8)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture", help="a capture_verify.py directory with trace.jsonl and rank.npz")
    p.add_argument("--out", required=True)
    a, extra = p.parse_known_args()
    ra = verify_replay.parse([os.path.join(a.capture, "trace.jsonl"), "--predictor", "none", *extra])
    ranks = dict(np.load(os.path.join(a.capture, "rank.npz")))
    loaded = verify_replay.load_forwards(ra.trace)
    rec = Recorder(ra, loaded["layer_ids"], ranks)
    rec.run(loaded)
    out = summarize(rec.rows)
    out["replay_args"] = vars(ra)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "replay_args"}, indent=1))


if __name__ == "__main__":
    main()
