"""Replay NVMe -> pinned-host-tier prefetch over a decode route capture, with a timed NVMe queue.

The destination is the pinned tier, not VRAM: a prefetched row turns a later RAM miss (an NVMe read the layer's S
kernel waits for) into a RAM hit (a copy-engine copy). The GPU link carries the same rows either way.

**Tier.** prefill-evict/ram_replay.py's RamTier replay (validated: 11.33 against 11.26 measured RAM misses per
decode token on varied24), extended with fills: a read claims its slot at issue and the row is READY when its
read completes; a filling slot is never a victim. Prefill forwards are replayed untimed (base policy); the NVMe
queue drains before each one.

**Timing model (first order; constants on the command line).**
- NVMe: one server for both mirrors (each row is split across them), ``--nvme-row-ms`` per 13.3 MB row, served in
  ``--pieces`` pieces. ``fifo``: pieces in issue order. ``prio``: demand pieces before speculative ones; a
  speculative piece already started finishes (non-preemptive at piece granularity).
- Layer T of a decode step starts (posts) at t. Its VRAM misses cross the link one after another,
  ``--link-row-ms`` each (copy-engine rate), RAM hits first. A row from NVMe is pulled as its pieces land, so the
  layer's S+CW ends at max(t + n_rows * link, last NVMe arrival + link / pieces). The excess over the link-bound
  time is the **exposed NVMe wait**, the only thing a pinned-tier prefetch can remove.
- The next layer posts ``--compute-ms`` later; a step adds ``--step-ms`` of tail.

**Predictors** (issued at a layer's post, i.e. once its router input exists):
- ``gate``: layer T+h's gate on layer T's input (gate_rankings.py), first K candidates of its top-6 not already in
  RAM, filling or VRAM-hot. Lead is h layers.
- ``near``: next token, same layer: layer T's gate on this token's layer-T input, ranks 1..12, first K not in RAM
  (ranks 1..6 were just admitted, so these are the near misses). Lead is one step.
- ``freq``: next token, same layer: an online per-layer decayed count of VRAM misses, first K not in RAM.
- ``oracle``: the target's true VRAM-miss rows not in RAM at issue, up to K, at lead ``h`` layers or ``next`` step.
- ``noisy``: oracle-sized issue where each row is right with probability p, else a random wrong expert.
- ``--layers`` restricts targets (e.g. layer 0 only).

Budget: at most ``--budget`` speculative rows per decode step. Admission stamps: ``mru`` (like a demand
admission) or ``cold`` (stamp 0 until first use, so an unused prefetch is the next victim).
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
from tier_sim import load_forwards, ram_rows_per_layer  # noqa: E402

CHUNK = 64
NUM_EXPERTS = 384


# ---------------------------------------------------------------- NVMe queue


class Nvme:
    """One server; jobs are rows split into pieces. Tracks per-row completion times."""

    def __init__(self, row_ms: float, pieces: int, mode: str) -> None:
        self.piece_ms = row_ms / pieces
        self.pieces = pieces
        self.mode = mode
        self.free = 0.0  # time the server finishes its current piece
        self.demand: deque = deque()
        self.spec: deque = deque()
        self.fifo: deque = deque()
        self.busy_ms = 0.0
        self.spec_pieces = 0  # speculative pieces served so far
        self.last_kind = None  # kind of the piece that ends at self.free
        self.delayed = 0  # demand rows that waited on a speculative piece
        self.delay_ms = 0.0

    def submit(self, job: dict) -> None:
        job["left"] = self.pieces
        job["done"] = None
        # Delay a demand row suffers from speculative reads: the speculative piece in service when it arrives
        # (non-preemptive), plus every speculative piece served before it completes (FIFO: those ahead of it).
        job["spec0"] = self.spec_pieces
        job["residual"] = max(0.0, self.free - job["t"]) if self.last_kind == "spec" else 0.0
        if self.mode == "fifo":
            self.fifo.append(job)
        elif job["kind"] == "demand":
            self.demand.append(job)
        else:
            self.spec.append(job)

    def promote(self, job: dict) -> None:
        """A demand arrived for a row still queued as speculative: in prio mode it moves to the demand queue."""
        if self.mode == "prio" and job["kind"] == "spec" and job["done"] is None:
            job["kind"] = "demand"
            job["promoted"] = True
            try:
                self.spec.remove(job)
            except ValueError:
                return
            self.demand.append(job)

    def _next(self):
        for q in ((self.fifo,) if self.mode == "fifo" else (self.demand, self.spec)):
            while q and q[0]["done"] is not None:
                q.popleft()
            if q:
                return q
        return None

    def _serve_one(self, q) -> None:
        job = q[0]
        start = max(self.free, job["t"])
        self.free = start + self.piece_ms
        self.busy_ms += self.piece_ms
        job["left"] -= 1
        self.last_kind = job["kind"]
        if job["kind"] == "spec":
            self.spec_pieces += 1
        if job["left"] == 0:
            job["done"] = self.free
            q.popleft()
            if job["kind"] == "demand" and not job.get("promoted"):
                d = job["residual"] + (self.spec_pieces - job["spec0"]) * self.piece_ms
                if d > 1e-9:
                    self.delayed += 1
                    self.delay_ms += d
            job["on_done"](job)

    def advance(self, t: float) -> None:
        """Serve every piece that starts before t (all queued jobs were issued at or before t)."""
        while True:
            q = self._next()
            if q is None or max(self.free, q[0]["t"]) >= t:
                return
            self._serve_one(q)

    def finish(self, jobs: list) -> None:
        """Serve until every job in ``jobs`` is done (no new job is issued meanwhile)."""
        while any(job["done"] is None for job in jobs):
            q = self._next()
            if q is None:
                raise RuntimeError("waiting on a job that is not queued")
            self._serve_one(q)

    def drain(self) -> None:
        while (q := self._next()) is not None:
            self._serve_one(q)


# ---------------------------------------------------------------- pinned tier with fills


class Tier:
    """RamTier (exl3_ram_miss_host.cpp) as ram_replay.Tier, plus fills: ``ready[slot]`` is when its read lands."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.slot_expert = [-1] * capacity
        self.stamp = [0] * capacity
        self.ready = [0.0] * capacity
        self.job: list = [None] * capacity
        self.spec = [None] * capacity  # for a prefetched row not yet used: its record
        self.where: dict[int, int] = {}

    def take(self, hot: set, protect: set, now: float) -> tuple[int, int]:
        best = -1
        for slot, expert in enumerate(self.slot_expert):
            if expert < 0:
                return slot, -1
            if expert in hot or expert in protect or self.ready[slot] > now:
                continue
            if best < 0 or self.stamp[slot] < self.stamp[best]:
                best = slot
        if best < 0:
            raise RuntimeError("no victim")
        gone = self.slot_expert[best]
        del self.where[gone]
        self.slot_expert[best] = -1
        return best, gone


# ---------------------------------------------------------------- replay


class Replay:
    def __init__(self, a, capacity: list[int], ranks) -> None:
        self.a = a
        self.tiers = [Tier(c) for c in capacity]
        self.nvme = Nvme(a.nvme_row_ms, a.pieces, a.queue)
        self.tick = 0
        self.ranks = ranks
        self.rng = random.Random(a.seed)
        self.freq = np.zeros((len(capacity), NUM_EXPERTS))
        self.evicted_by_spec = [dict() for _ in capacity]
        self.st = dict(
            demand_misses=0, demand_misses_decode=0, spec_issued=0, spec_used_target=0, spec_used_any=0,
            spec_late=0, harmful=0,
            exposed_ms=0.0, link_ms=0.0, step_ms=0.0, decode_steps=0, vram_misses=0, ram_hits=0,
            prefill_misses=0, budget_capped=0, hot_not_in_ram=0,
        )
        self.per_layer_exposed = np.zeros(len(capacity))
        self.per_layer_misses = np.zeros(len(capacity))
        self.steady = dict(steps=0, misses=0, exposed_ms=0.0)

    def _next(self) -> int:
        self.tick += 1
        return self.tick

    # ---- admission

    def admit(self, layer: int, expert: int, hot: set, protect: set, now: float, kind: str, record=None):
        tier = self.tiers[layer]
        slot, gone = tier.take(hot, protect, now)
        if gone >= 0:
            if kind == "spec":
                self.evicted_by_spec[layer][gone] = True
        tier.spec[slot] = record
        tier.slot_expert[slot] = expert
        tier.where[expert] = slot
        tier.stamp[slot] = 0 if (kind == "spec" and self.a.admit == "cold") else self._next()
        self.evicted_by_spec[layer].pop(expert, None)
        job = {"t": now, "kind": kind, "layer": layer, "expert": expert}
        tier.ready[slot] = float("inf")
        tier.job[slot] = job

        def on_done(j, tier=tier, slot=slot, expert=expert):
            if tier.slot_expert[slot] == expert and tier.job[slot] is j:
                tier.ready[slot] = j["done"]

        job["on_done"] = on_done
        self.nvme.submit(job)
        return job

    # ---- prediction

    def candidates(self, step_i: int, fwd_next, target_step: int, layer: int, lead: str):
        """Ranked candidate experts for target (target_step, layer)."""
        a = self.a
        if a.predictor == "gate":
            if not self.ranks["valid"][target_step, layer, a.h]:
                return []
            return self.ranks["order"][target_step, layer, a.h, : a.depth].tolist()
        if a.predictor == "near":
            return self.ranks["order"][step_i, layer, 0, :].tolist()
        if a.predictor == "freq":
            return np.argsort(-self.freq[layer], kind="stable")[: 24].tolist()
        if a.predictor in ("oracle", "noisy"):
            return list(fwd_next["routes"][layer])
        return []

    def issue(self, step_i: int, fwd, target_fwd, target_step: int, layer: int, now: float, budget: list) -> None:
        a = self.a
        if a.layers is not None and layer not in a.layers:
            return
        tier = self.tiers[layer]
        hot = set(target_fwd["hot"][layer]) if a.predictor in ("oracle", "noisy") else set(fwd["hot"][layer])
        cands = self.candidates(step_i, target_fwd, target_step, layer, None)
        chosen = []
        if a.predictor in ("oracle", "noisy"):
            true = [e for e in dict.fromkeys(cands) if e not in hot and e not in tier.where]
            true = true[: a.k]
            if a.predictor == "noisy":
                routed = set(cands)
                out = []
                for e in true:
                    if self.rng.random() < a.p:
                        out.append(e)
                    else:
                        while True:
                            w = self.rng.randrange(NUM_EXPERTS)
                            if w not in routed and w not in hot and w not in tier.where and w not in out:
                                break
                        out.append(w)
                true = out
            chosen = true
        else:
            for e in cands:
                if len(chosen) >= a.k:
                    break
                if e in hot or e in tier.where or e in chosen:
                    continue
                chosen.append(e)
        if not chosen:
            return
        protect = set(chosen)
        target_routes = set(target_fwd["routes"][layer])
        target_hot = set(target_fwd["hot"][layer])
        for e in chosen:
            if budget[0] <= 0:
                self.st["budget_capped"] += 1
                return
            budget[0] -= 1
            rec = {"target": (target_step, layer), "used": False,
                   "right": e in target_routes and e not in target_hot}
            self.admit(layer, e, hot, protect, now, "spec", rec)
            self.st["spec_issued"] += 1

    # ---- decode

    def decode_layer(self, fwd, layer: int, now: float, step_i: int, timed: bool):
        """Demand side of one layer: returns (row arrivals of NVMe/filling rows, n VRAM-miss rows, delayed)."""
        tier = self.tiers[layer]
        routes = fwd["routes"][layer]
        hot = set(fwd["hot"][layer])
        wanted = set(routes)
        arrivals, jobs, n_rows = [], [], 0
        missing = []
        for expert in dict.fromkeys(routes):
            vram_miss = expert not in hot
            n_rows += vram_miss
            slot = tier.where.get(expert)
            if slot is None:
                missing.append(expert)
                continue
            rec = tier.spec[slot]
            if rec is not None and not rec["used"]:
                rec["used"] = True
                self.st["spec_used_any"] += 1
                if rec["target"] == (step_i, layer):
                    self.st["spec_used_target"] += 1
                tier.spec[slot] = None
            tier.stamp[slot] = self._next()
            if tier.ready[slot] > now:  # still filling
                job = tier.job[slot]
                self.nvme.promote(job)
                jobs.append(job)
                self.st["spec_late"] += 1
            elif vram_miss:
                self.st["ram_hits"] += 1
        for expert in missing:
            self.st["demand_misses"] += 1
            if expert in self.evicted_by_spec[layer]:
                self.st["harmful"] += 1
            self.per_layer_misses[layer] += 1
            job = self.admit(layer, expert, hot, wanted, now, "demand")
            if expert in hot:
                self.st["hot_not_in_ram"] += 1  # read, but the gather does not wait for it
            else:
                jobs.append(job)
        return jobs, n_rows, len(missing)

    def run(self, loaded) -> dict:
        a = self.a
        forwards = [f for f in loaded["forwards"] if f["phase"] != "capture"]
        layer_ids = loaded["layer_ids"]
        next_hot: list = [None] * len(forwards)
        upcoming = None
        for i in range(len(forwards) - 1, -1, -1):
            next_hot[i] = upcoming
            if forwards[i]["kind"] == "graph" and forwards[i].get("hot") is not None:
                upcoming = forwards[i]["hot"]
        decode_idx = [i for i, f in enumerate(forwards) if f["kind"] == "graph" and f["phase"] == "decode"]
        step_of = {fi: s for s, fi in enumerate(decode_idx)}
        rid = lambda f: (f["rids"] or [None])[0]  # noqa: E731
        now = 0.0
        since = None
        for i, fwd in enumerate(forwards):
            if fwd["kind"] != "graph":
                # Prefill: drain the queue (a prefill takes seconds), replay admissions untimed at MRU.
                self.nvme.drain()
                now = max(now, self.nvme.free) + 1000.0
                hot_map = next_hot[i] or {}
                for layer, (experts, _) in fwd["counts"].items():
                    self.prefill(layer, list(experts), set(hot_map.get(layer, ())), now)
                if fwd["phase"] != "decode":
                    since = 0
                continue
            s = step_of.get(i)
            decode = s is not None
            # The next decode step of the same request (for next-token and cross-step lookahead targets).
            nxt = None
            if decode and s + 1 < len(decode_idx) and decode_idx[s + 1] == i + 1 and rid(forwards[i + 1]) == rid(fwd):
                nxt = forwards[i + 1]
            budget = [a.budget]
            step_start = now
            step_exposed = 0.0
            step_misses_before = self.st["demand_misses"]
            for layer in layer_ids:
                self.nvme.advance(now)
                jobs, n_rows, _ = self.decode_layer(fwd, layer, now, s if decode else -1, decode)
                if decode and a.predictor != "none":
                    self.issue_at_post(s, fwd, nxt, layer, now, budget, layer_ids)
                self.nvme.finish(jobs)
                link_end = now + n_rows * a.link_row_ms
                end = link_end
                for job in jobs:
                    end = max(end, job["done"] + a.link_row_ms / a.pieces)
                exposed = end - link_end
                if decode:
                    self.per_layer_exposed[layer] += exposed
                    step_exposed += exposed
                    self.st["link_ms"] += n_rows * a.link_row_ms
                    self.st["vram_misses"] += n_rows
                now = end + a.compute_ms
            now += a.step_ms
            if decode:
                self.st["decode_steps"] += 1
                self.st["exposed_ms"] += step_exposed
                self.st["step_ms"] += now - step_start
                self.st["demand_misses_decode"] += self.st["demand_misses"] - step_misses_before
                if since is not None and since >= 15:
                    self.steady["steps"] += 1
                    self.steady["misses"] += self.st["demand_misses"] - step_misses_before
                    self.steady["exposed_ms"] += step_exposed
                if a.predictor == "freq":
                    self.freq *= a.decay
                    for layer in layer_ids:
                        hot = set(fwd["hot"][layer])
                        for e in fwd["routes"][layer]:
                            if e not in hot:
                                self.freq[layer, e] += 1.0
                since = None if since is None else since + 1
        self.nvme.drain()
        return self.report()

    def issue_at_post(self, s, fwd, nxt, layer, now, budget, layer_ids) -> None:
        a = self.a
        L = len(layer_ids)
        if a.lead == "next":
            if nxt is not None:
                self.issue(s, fwd, nxt, s + 1, layer, now, budget)
            return
        target = layer + a.h
        if target < L:
            self.issue(s, fwd, fwd, s, target, now, budget)
        elif nxt is not None:
            self.issue(s, fwd, nxt, s + 1, target - L, now, budget)

    def prefill(self, layer: int, experts: list[int], hot: set, now: float) -> None:
        tier = self.tiers[layer]
        for start in range(0, len(experts), CHUNK):
            chunk = [e for e in experts[start : start + CHUNK] if e not in hot]
            missing = []
            for expert in chunk:
                slot = tier.where.get(expert)
                if slot is None:
                    missing.append(expert)
                else:
                    tier.stamp[slot] = self._next()
            self.st["prefill_misses"] += len(missing)
            protect = set(chunk)
            for expert in missing:
                slot, _ = tier.take(hot, protect, float("inf"))
                tier.slot_expert[slot] = expert
                tier.where[expert] = slot
                tier.stamp[slot] = self._next()
                tier.ready[slot] = 0.0
                tier.job[slot] = None
                tier.spec[slot] = None

    def report(self) -> dict:
        st, n = self.st, max(self.st["decode_steps"], 1)
        issued = max(st["spec_issued"], 1)
        return {
            "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(self.a).items()},
            "decode_steps": st["decode_steps"],
            "ram_misses_per_token": st["demand_misses_decode"] / n,
            "vram_misses_per_token": st["vram_misses"] / n,
            "spec_rows_per_token": st["spec_issued"] / n,
            "spec_gb_per_token": st["spec_issued"] * 13_315_584 / n / 1e9,
            "precision_target": st["spec_used_target"] / issued,
            "precision_any_use": st["spec_used_any"] / issued,
            "late_prefetch_per_token": st["spec_late"] / n,
            "wasted_rows_per_token": (st["spec_issued"] - st["spec_used_any"]) / n,
            "demand_rows_delayed_per_token": self.nvme.delayed / n,
            "demand_delay_mean_ms": self.nvme.delay_ms / max(self.nvme.delayed, 1),
            "hot_not_in_ram_per_token": st["hot_not_in_ram"] / n,
            "harmful_evictions_per_token": st["harmful"] / n,
            "budget_capped_per_token": st["budget_capped"] / n,
            "exposed_ms_per_token": st["exposed_ms"] / n,
            "step_ms_per_token": st["step_ms"] / n,
            "link_ms_per_token": st["link_ms"] / n,
            "nvme_busy_frac": self.nvme.busy_ms / max(st["step_ms"], 1e-9),
            "steady_ram_misses_per_token": self.steady["misses"] / max(self.steady["steps"], 1),
            "steady_exposed_ms_per_token": self.steady["exposed_ms"] / max(self.steady["steps"], 1),
            "per_layer_exposed_ms": (self.per_layer_exposed / n).round(3).tolist(),
            "per_layer_ram_misses": (self.per_layer_misses / n).round(3).tolist(),
        }


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--ranks", help="gate_rankings.py output (npz)")
    p.add_argument("--predictor", default="none", choices=["none", "gate", "near", "freq", "oracle", "noisy"])
    p.add_argument("--lead", default="layers", choices=["layers", "next"], help="h layers ahead, or next token")
    p.add_argument("--h", type=int, default=1)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--depth", type=int, default=6, help="gate: candidates come from the top-depth ranks")
    p.add_argument("--p", type=float, default=0.5, help="noisy: probability each row is right")
    p.add_argument("--decay", type=float, default=0.98)
    p.add_argument("--budget", type=int, default=16, help="speculative rows per decode step")
    p.add_argument("--queue", default="prio", choices=["prio", "fifo"])
    p.add_argument("--admit", default="mru", choices=["mru", "cold"])
    p.add_argument("--layers", type=lambda s: {int(x) for x in s.split(",")}, default=None)
    p.add_argument("--nvme-row-ms", type=float, default=2.5)
    p.add_argument("--pieces", type=int, default=8)
    p.add_argument("--link-row-ms", type=float, default=1.07)
    p.add_argument("--compute-ms", type=float, default=0.35)
    p.add_argument("--step-ms", type=float, default=2.0)
    p.add_argument("--ram-rows", type=int, default=8063)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name")
    p.add_argument("--out")
    return p.parse_args(argv)


def run_one(a, loaded=None, ranks=None) -> dict:
    loaded = loaded or load_forwards(a.trace)
    if ranks is None and a.predictor in ("gate", "near"):
        ranks = dict(np.load(a.ranks))
    capacity = ram_rows_per_layer(a.ram_rows, len(loaded["layer_ids"]), NUM_EXPERTS)
    return Replay(a, capacity, ranks).run(loaded)


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
