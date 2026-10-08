"""Replay a DSpark verify capture through the per-group lease-gate chain, with NVMe-to-RAM prefetch.

The step is the verify forward (phase target_verify, M tokens). Per layer, each NUMA group g (expert % 2) serves
its lanes: hits split between the DMA (``--dma-ms``) and the CPU hit job (``--cpu-fixed-ms`` + ``--cpu-lane-ms``
per lane, lanes per the calibration ``--split``); forced misses are demand reads on the shared NVMe queue
(pinned_prefetch_replay.Nvme, ``--nvme-row-ms`` per row, priority queue) whose rows, as they land, each take a
serial CPU job (``--miss-job-ms``) after the hit job; the group's gate opens at the last finisher, the layer ends at
the slower group plus ``--gpu-ms``, and the next layer posts then. A verify adds ``--step-ms`` (the draft graph and
the host between verifies). So the lead a speculative read gets is whatever the chain gives, and hiding one layer's
misses shortens the next layer's lead.

Tier: per layer two Tiers, ``round(rows * --node-share)`` slots for group 0 and the rest for group 1, the RamTier
victim rule (lowest stamp, not hot, not wanted, not filling). Prefill forwards replay their admissions untimed.

Predictors, issued at a layer's post for layer T + h of the same verify (there is no cross-step source):
``oracle`` (the target's true forced misses), ``gate`` (verify_gate_rankings.py: every live token's top-depth at
horizon h, ordered by best score, not hot / in RAM / filling), ``noisy`` (oracle rows right with probability p).
``--budget`` speculative rows per layer, ``--k`` per live token, ``--spec-share`` unused rows a layer's group may hold,
``--admit cold|mru``.

Report: per-step and per-token (``--accept`` tokens per verify) times, the gate per layer and group, RAM misses,
speculative precision, late, harmful evictions, demand delay, NVMe busy, and gate percentiles for calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
from pinned_prefetch_replay import CHUNK, NUM_EXPERTS, Nvme, Tier  # noqa: E402
from tier_sim import load_forwards, ram_rows_per_layer  # noqa: E402

GROUPS = 2


def group_of(expert: int) -> int:
    return expert % GROUPS


def make_tiers(a, layer_ids) -> list[list[Tier]]:
    rows = ram_rows_per_layer(a.ram_rows, len(layer_ids), NUM_EXPERTS)
    out = []
    for n in rows:
        g0 = int(round(n * a.node_share))
        out.append([Tier(g0), Tier(n - g0)])
    return out


class Replay:
    def __init__(self, a, layer_ids, ranks) -> None:
        self.a = a
        self.layer_ids = list(layer_ids)
        self.tiers = make_tiers(a, layer_ids)
        self.nvme = Nvme(a.nvme_row_ms, a.pieces, a.queue, 0.0)
        self.split = [int(s) for s in a.split.split()]
        self.ranks = ranks
        self.rng = random.Random(a.seed)
        self.tick = 0
        self.evicted_by_spec = [[dict() for _ in range(GROUPS)] for _ in layer_ids]
        self.st = dict(demand_misses=0, spec_issued=0, spec_used_target=0, spec_used_any=0, spec_late=0, harmful=0,
                       budget_capped=0, ram_hits=0, prefill_misses=0, steps=0, step_ms=0.0, gate_ms=0.0)
        self.per_layer_exposed = np.zeros(len(layer_ids))
        self.per_layer_misses = np.zeros(len(layer_ids))
        self.per_group_gate = np.zeros(GROUPS)
        self.gates: list[float] = []
        self.per_step_gate: list[float] = []
        self.per_step_ms: list[float] = []

    def _next(self) -> int:
        self.tick += 1
        return self.tick

    # ---- tier

    def admit(self, li: int, expert: int, hot: set, protect: set, now: float, kind: str, record=None):
        tier = self.tiers[li][group_of(expert)]
        slot = gone = -1
        owned = {i for i, r in enumerate(tier.spec) if r is not None and not r["used"]}
        if kind == "spec" and self.a.spec_share and len(owned) >= self.a.spec_share:
            try:
                slot, gone = tier.take(hot, protect, now, only=owned)
            except RuntimeError:
                slot = -1
        if slot < 0:
            slot, gone = tier.take(hot, protect, now)
        if gone >= 0 and kind == "spec":
            self.evicted_by_spec[li][group_of(expert)][gone] = True
        tier.spec[slot] = record
        tier.slot_expert[slot] = expert
        tier.where[expert] = slot
        tier.stamp[slot] = 0 if (kind == "spec" and self.a.admit == "cold") else self._next()
        self.evicted_by_spec[li][group_of(expert)].pop(expert, None)
        job = {"t": now, "kind": kind, "layer": li, "expert": expert, "group": group_of(expert)}
        tier.ready[slot] = float("inf")
        tier.job[slot] = job

        def on_done(j, tier=tier, slot=slot, expert=expert):
            if tier.slot_expert[slot] == expert and tier.job[slot] is j:
                tier.ready[slot] = j["done"]

        job["on_done"] = on_done
        self.nvme.submit(job)
        return job

    def where(self, li: int, expert: int):
        tier = self.tiers[li][group_of(expert)]
        slot = tier.where.get(expert)
        return tier, slot

    # ---- prediction

    def candidates(self, s: int, fwd, target_li: int) -> list[int]:
        a = self.a
        if a.predictor == "gate":
            if not self.ranks["valid"][s, target_li, a.h]:
                return []
            order = self.ranks["order"][s, target_li, :, a.h, : a.depth]  # [M, depth]
            score = self.ranks["score"][s, target_li, :, a.h, : a.depth]
            best: dict[int, float] = {}
            for m in range(int(self.ranks["tokens"][s])):
                for e, sc in zip(order[m].tolist(), score[m].tolist()):
                    if e >= 0 and sc > best.get(e, -1e30):
                        best[e] = sc
            return [e for e, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
        if a.predictor in ("oracle", "noisy"):
            return list(dict.fromkeys(fwd["routes"][self.layer_ids[target_li]]))
        return []

    def issue(self, s: int, fwd, li: int, now: float, budget: list) -> None:
        a = self.a
        target_li = li + a.h
        if target_li >= len(self.layer_ids) or self.layer_ids[target_li] == 0:
            return
        target_layer = self.layer_ids[target_li]
        hot = set(fwd["hot"][target_layer]) if fwd.get("hot") else set()
        want = a.k * max(int(fwd.get("tokens", 1)), 1)
        chosen = []
        for e in self.candidates(s, fwd, target_li):
            if len(chosen) >= want:
                break
            tier, slot = self.where(target_li, e)
            if e in hot or slot is not None or e in chosen:
                continue
            chosen.append(e)
        if a.predictor == "noisy":
            out = []
            for e in chosen:
                if self.rng.random() < a.p:
                    out.append(e)
                    continue
                routed = set(fwd["routes"][target_layer])
                while True:
                    w = self.rng.randrange(NUM_EXPERTS)
                    tier, slot = self.where(target_li, w)
                    if w not in routed and w not in hot and slot is None and w not in out and group_of(w) == group_of(e):
                        break
                out.append(w)
            chosen = out
        target_routes = set(fwd["routes"][target_layer])
        for e in chosen:
            if budget[0] <= 0:
                self.st["budget_capped"] += 1
                return
            budget[0] -= 1
            rec = {"target": (s, target_li), "used": False, "right": e in target_routes and e not in hot}
            self.admit(target_li, e, hot, set(chosen), now, "spec", rec)
            self.st["spec_issued"] += 1

    # ---- the chain

    def layer(self, s: int, fwd, li: int, now: float) -> tuple[float, list[float]]:
        """Serve one layer at ``now``; returns (gate of the slower group, per-group gates), in ms after ``now``."""
        layer = self.layer_ids[li]
        hot = set(fwd["hot"][layer]) if fwd.get("hot") else set()
        routes = list(dict.fromkeys(fwd["routes"][layer]))
        wanted = set(routes)
        hits = [0] * GROUPS
        landing: list[list[dict]] = [[] for _ in range(GROUPS)]
        for expert in routes:
            if expert in hot:
                continue
            g = group_of(expert)
            tier, slot = self.where(li, expert)
            if slot is not None:
                rec = tier.spec[slot]
                if rec is not None and not rec["used"]:
                    rec["used"] = True
                    self.st["spec_used_any"] += 1
                    if rec["target"] == (s, li):
                        self.st["spec_used_target"] += 1
                    tier.spec[slot] = None
                tier.stamp[slot] = self._next()
                if tier.ready[slot] > now:
                    job = tier.job[slot]
                    self.nvme.promote(job)
                    landing[g].append(job)
                    self.st["spec_late"] += 1
                else:
                    hits[g] += 1
                    self.st["ram_hits"] += 1
                continue
            self.st["demand_misses"] += 1
            self.per_layer_misses[li] += 1
            if expert in self.evicted_by_spec[li][g]:
                self.st["harmful"] += 1
            landing[g].append(self.admit(li, expert, hot, wanted, now, "demand"))
        self.nvme.finish([j for group in landing for j in group])
        a = self.a
        gates = []
        for g in range(GROUPS):
            cpu = self.split[min(hits[g], len(self.split) - 1)]
            dma = hits[g] - cpu
            hit_end = max(a.dma_ms if dma else 0.0, a.cpu_fixed_ms + a.cpu_lane_ms * cpu if cpu else 0.0)
            end = hit_end
            for job in sorted(landing[g], key=lambda j: j["done"]):
                end = max(end, job["done"] - now) + a.miss_job_ms
            gates.append(end if (hits[g] or landing[g]) else 0.0)
        return max(gates), gates

    def run(self, loaded) -> dict:
        a = self.a
        now = 0.0
        s = -1
        for fwd in loaded["forwards"]:
            if fwd["kind"] != "graph":
                self.nvme.drain()
                now = max(now, self.nvme.free) + 1000.0
                for layer, (experts, _) in fwd["counts"].items():
                    self.prefill(self.layer_ids.index(layer), list(experts), now)
                continue
            if not fwd.get("verify"):
                continue
            s += 1
            step_start, step_gate = now, 0.0
            budget = [a.budget]
            for li in range(len(self.layer_ids)):
                self.nvme.advance(now)
                if a.predictor != "none":
                    self.issue(s, fwd, li, now, budget)
                gate, gates = self.layer(s, fwd, li, now)
                self.per_layer_exposed[li] += gate
                self.per_group_gate += gates
                self.gates.append(gate)
                step_gate += gate
                now += gate + a.gpu_ms
            now += a.step_ms
            self.st["steps"] += 1
            self.st["step_ms"] += now - step_start
            self.st["gate_ms"] += step_gate
            self.per_step_gate.append(step_gate)
            self.per_step_ms.append(now - step_start)
        self.nvme.drain()
        return self.report()

    def prefill(self, li: int, experts: list[int], now: float) -> None:
        for start in range(0, len(experts), CHUNK):
            chunk = experts[start : start + CHUNK]
            protect = set(chunk)
            for expert in chunk:
                tier, slot = self.where(li, expert)
                if slot is not None:
                    tier.stamp[slot] = self._next()
                    continue
                self.st["prefill_misses"] += 1
                slot, _ = tier.take(set(), protect, float("inf"))
                tier.slot_expert[slot] = expert
                tier.where[expert] = slot
                tier.stamp[slot] = self._next()
                tier.ready[slot] = 0.0
                tier.job[slot] = None
                tier.spec[slot] = None

    def report(self) -> dict:
        st, n = self.st, max(self.st["steps"], 1)
        issued = max(st["spec_issued"], 1)
        gates = np.asarray(self.gates) if self.gates else np.zeros(1)
        return {
            "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(self.a).items()},
            "steps": st["steps"],
            "step_ms_per_step": st["step_ms"] / n,
            "gate_ms_per_step": st["gate_ms"] / n,
            "ms_per_token": st["step_ms"] / n / self.a.accept,
            "ram_misses_per_step": st["demand_misses"] / n,
            "ram_hits_per_step": st["ram_hits"] / n,
            "spec_rows_per_step": st["spec_issued"] / n,
            "precision_target": st["spec_used_target"] / issued,
            "precision_any_use": st["spec_used_any"] / issued,
            "late_per_step": st["spec_late"] / n,
            "wasted_rows_per_step": (st["spec_issued"] - st["spec_used_any"]) / n,
            "harmful_evictions_per_step": st["harmful"] / n,
            "budget_capped_per_step": st["budget_capped"] / n,
            "demand_rows_delayed_per_step": self.nvme.delayed / n,
            "demand_delay_mean_ms": self.nvme.delay_ms / max(self.nvme.delayed, 1),
            "nvme_busy_frac": self.nvme.busy_ms / max(st["step_ms"], 1e-9),
            "gate_p50_ms": float(np.percentile(gates, 50)),
            "gate_p90_ms": float(np.percentile(gates, 90)),
            "gate_p99_ms": float(np.percentile(gates, 99)),
            "per_layer_exposed_ms": (self.per_layer_exposed / n).round(3).tolist(),
            "per_layer_ram_misses": (self.per_layer_misses / n).round(3).tolist(),
            "per_group_gate_ms": (self.per_group_gate / n).round(3).tolist(),
            "per_step_gate_ms": [round(v, 3) for v in self.per_step_gate],
            "per_step_ms": [round(v, 3) for v in self.per_step_ms],
        }


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--ranks", help="verify_gate_rankings.py output (npz)")
    p.add_argument("--predictor", default="none", choices=["none", "oracle", "gate", "noisy"])
    p.add_argument("--h", type=int, default=1)
    p.add_argument("--k", type=int, default=1, help="speculative rows per live token")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--p", type=float, default=0.5)
    p.add_argument("--budget", type=int, default=4, help="speculative rows per layer")
    p.add_argument("--queue", default="prio", choices=["prio", "fifo"])
    p.add_argument("--admit", default="cold", choices=["mru", "cold"])
    p.add_argument("--spec-share", type=int, default=4)
    p.add_argument("--nvme-row-ms", type=float, default=2.2)
    p.add_argument("--pieces", type=int, default=8)
    p.add_argument("--dma-ms", type=float, default=1.2)
    p.add_argument("--cpu-fixed-ms", type=float, default=0.3)
    p.add_argument("--cpu-lane-ms", type=float, default=0.6)
    p.add_argument("--miss-job-ms", type=float, default=1.0)
    p.add_argument("--gpu-ms", type=float, default=1.0)
    p.add_argument("--step-ms", type=float, default=20.0)
    p.add_argument("--ram-rows", type=int, default=8385, help="106496 MiB / 13,315,584 B per row")
    p.add_argument("--node-share", type=float, default=57344 / 106496)
    p.add_argument("--split", default="0 1 2 2 3 3 4 5 5", help="CPU hit lanes by hit count (the calibration)")
    p.add_argument("--accept", type=float, default=4.3, help="accepted tokens per verify, for ms per token")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name")
    p.add_argument("--out")
    return p.parse_args(argv)


def run_one(a, loaded=None, ranks=None) -> dict:
    loaded = loaded or load_forwards(a.trace)
    if ranks is None and a.predictor == "gate":
        ranks = dict(np.load(a.ranks))
    return Replay(a, loaded["layer_ids"], ranks).run(loaded)


def main() -> None:
    a = parse()
    out = run_one(a)
    text = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
