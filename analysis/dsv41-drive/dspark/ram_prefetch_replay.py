"""Offline replay of NVMe-to-RAM prefetch a verify ahead: what a next-verify predictor could catch, under a read
budget, with the pinned RAM tier's capacity in the loop.

Inputs: a spec_margin capture with ``stages.jsonl`` (schema-2 ``graph_routes``: routes and VRAM hot sets per verify),
``events.*.verify.jsonl`` (verify windows), the tier/spec job traces (``miss_land`` a = expert read from NVMe on demand,
``spec_use`` / ``pool_point`` a = a demand miss served from the existing L+2 prefetch, ``spec_land`` a = a prefetch
landing), the router capture (``router.*``: gate inputs and top-k per token) and the model dir (gate weights, read by
byte range from the safetensors shards).

The RAM tier is replayed per layer as the host does it (ram_tier.h take_victim_locked): LRU over the slots, VRAM-hot
experts and the current request's experts never evicted, capacity ``--rows-per-layer`` (calibrated: the replay's
RAM misses per verify should match the capture's demand misses plus the misses the existing prefetch covered). The
existing prefetch's landings enter the tier as the capture shows them (``spec_land``), each evicting a victim.

A policy chooses, at the end of verify v, up to ``budget`` rows to read for v+1, in priority order; the drives serve
them serially at ``--ms-per-row`` from the verify's end (the existing prefetch's own reads are not modelled as
contention). Layer L of v+1 needs its rows at ``begin(v+1) + ms_per_layer * L``. A row landing in time turns a miss
into a hit; a late or unrouted row is a wasted read but still enters the tier (evicting a victim), so the capacity
cost shows up as induced misses later.

Policies: ``oracle`` (the next verify's RAM misses, earliest layer first: a bound), ``nearmiss`` (the gate's scores of
this verify's tokens for experts it did not route, highest first, the no-learning analogue of a 384-way next-verify
head), ``recur`` (experts routed at v and not in RAM after v: what a recurrence head such as next_verify_head.py can
offer; structurally near empty, reported to show it), each ``alone`` (the existing prefetch's landings removed, its
covered misses back in the demand) and ``ontop`` (as captured).

Usage: ram_prefetch_replay.py CAPTURE MODEL_DIR [--skip 8] [--rows-per-layer 210] [--budgets 20,40,80]
                              [--ms-per-row 2.3] [--ms-per-layer 5.5] [--json OUT]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import struct
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def bf16_rows(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


def read_tensor(path: str, name: str) -> np.ndarray:
    dtypes = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(n))[name]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = np.frombuffer(f.read(end - start), dtype=dtypes[meta["dtype"]]).reshape(meta["shape"])
    return bf16_rows(raw) if meta["dtype"] == "BF16" else raw.astype(np.float32)


def load_gates(model_path: str, layer_ids: list) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    W = np.stack([read_tensor(os.path.join(model_path, where[f"layers.{i}.ffn.gate.weight"]), f"layers.{i}.ffn.gate.weight") for i in layer_ids])
    b = np.stack([read_tensor(os.path.join(model_path, where[f"layers.{i}.ffn.gate.bias"]), f"layers.{i}.ffn.gate.bias") for i in layer_ids])
    return W, b


class Capture:
    def __init__(self, d: str, skip: int):
        recs = []
        with open(os.path.join(d, "stages.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "graph_routes_header":
                    self.header = r
                elif r.get("kind") == "graph_routes" and r.get("phase") == "target_verify" and "hot" in r:
                    recs.append(r)
        self.recs = recs
        self.layers = len(self.header["layer_ids"])
        (vpath,) = glob.glob(os.path.join(d, "events.*.verify.jsonl"))
        with open(vpath) as f:
            f.readline()
            ev = [json.loads(line) for line in f]
        self.begin = [e["ns"] for e in ev if e["event"] == "verify_begin"]
        self.end = [e["ns"] for e in ev if e["event"] == "verify_end"]
        self.n = min(len(recs), len(self.begin), len(self.end))
        self.skip = skip
        # job-trace events joined to verifies by time
        self.demand = [[set() for _ in range(self.layers)] for _ in range(self.n)]
        self.covered = [[set() for _ in range(self.layers)] for _ in range(self.n)]
        self.landed = [[set() for _ in range(self.layers)] for _ in range(self.n)]
        for path in glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl")) + glob.glob(os.path.join(d, "events.*.exl3-spec*.jsonl")):
            with open(path) as f:
                for line in f:
                    e = json.loads(line)
                    k = e.get("event")
                    if k not in ("miss_land", "spec_use", "pool_point", "spec_land"):
                        continue
                    v = bisect.bisect_right(self.begin, e["ns"]) - 1
                    if not 0 <= v < self.n or e["row"] >= self.layers:
                        continue
                    (self.demand if k == "miss_land" else self.landed if k == "spec_land" else self.covered)[v][e["row"]].add(e["a"])
        with open(os.path.join(d, "router.json")) as f:
            meta = json.load(f)
        self.tokens, self.topk, self.hidden = int(meta["tokens"]), int(meta["topk"]), int(meta["hidden"])
        seq = np.fromfile(os.path.join(d, "router.seq.bin"), dtype=np.int64).reshape(-1, 3)
        self.live = seq[:, 2]
        self.ids = np.fromfile(os.path.join(d, "router.ids.bin"), dtype=np.int32).reshape(seq.shape[0], self.layers, self.tokens, self.topk)
        self.x = np.memmap(os.path.join(d, "router.x.bin"), dtype=np.uint16, mode="r",
                           shape=(seq.shape[0], self.layers, self.tokens, self.hidden))

    def routes(self, v: int, L: int) -> set:
        return set(e for e in self.recs[v]["routes"][L] if e >= 0)

    def hot(self, v: int, L: int) -> set:
        return set(self.recs[v]["hot"][L])

    def gate_scores(self, v: int, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """[layers, experts]: each expert's largest biased gate score over the verify's live tokens."""
        ri = self.recs[v]["router"]
        live = int(self.live[ri]) or self.tokens
        x = bf16_rows(np.asarray(self.x[ri, :, :live, :]))  # [L, live, hidden]
        logits = np.einsum("lth,leh->lte", x, W)
        # DeepSeek-V4.1 routing (router_score.py): scores = sqrt(softplus(logits)), ranked by scores + bias
        s = np.sqrt(np.logaddexp(0.0, logits)) + b[:, None, :]
        self.last_per_token = s
        return s.max(axis=1)


class RamTier:
    """One layer's pinned slots: LRU with VRAM-hot and in-flight protection."""

    def __init__(self, capacity: int, victim: str = "lru"):
        self.capacity = capacity
        self.victim = victim
        self.stamp = {}  # expert -> last use tick
        self.speculative = set()  # rows that arrived by prefetch and were not routed since
        self.tick = 0
        self.evictions = 0
        self.spec_evictions = 0

    def has(self, e: int) -> bool:
        return e in self.stamp

    def touch(self, e: int) -> None:
        self.tick += 1
        self.stamp[e] = self.tick
        self.speculative.discard(e)

    def insert(self, e: int, protected: set, speculative: bool = False) -> bool:
        """Insert e, evicting a row when full: ``lru`` the least recently used unprotected row; ``specfirst`` the
        least recently landed never-routed prefetch row when one exists, else LRU. False when nothing could go."""
        if e in self.stamp:
            self.touch(e)
            return True
        if len(self.stamp) >= self.capacity:
            cands = [(t, r) for r, t in self.stamp.items() if r not in protected]
            if not cands:
                return False
            spec = [(t, r) for t, r in cands if r in self.speculative] if self.victim == "specfirst" else []
            _, victim = min(spec) if spec else min(cands)
            if victim in self.speculative:
                self.spec_evictions += 1
            del self.stamp[victim]
            self.speculative.discard(victim)
            self.evictions += 1
        self.touch(e)
        if speculative:
            self.speculative.add(e)
        return True


def replay(cap: Capture, policy: str, mode: str, budget: int, rows_per_layer: int, ms_per_row: float, ms_per_layer: float,
           scores: list | None, recur_p: dict | None = None, victim: str = "lru") -> dict:
    """Run the tier over the verifies; ``policy`` in (none, oracle, nearmiss, recur); ``mode`` alone|ontop."""
    L = cap.layers
    tiers = [RamTier(rows_per_layer, victim) for _ in range(L)]
    out = collections.Counter()
    pending = []  # (land_ns, layer, expert, for_verify)
    for v in range(cap.n):
        counted = v >= cap.skip
        begin = cap.begin[v]
        # prefetched rows that landed before each layer's gather enter the tier first (in landing order)
        pending.sort()
        still = []
        landed_now = collections.defaultdict(set)
        for land_ns, Lx, e, fv in pending:
            need = begin + int(ms_per_layer * Lx * 1e6)
            if land_ns <= need:
                landed_now[Lx].add(e)
            else:
                still.append((land_ns, Lx, e, fv))
                if fv == v and counted:
                    out["late"] += 1
        pending = still
        for Lx in range(L):
            hot = cap.hot(v, Lx)
            routed = cap.routes(v, Lx)
            for e in landed_now[Lx]:
                tiers[Lx].insert(e, hot | routed, speculative=True)
                if counted:
                    out["reads_landed"] += 1
                    if e in routed:
                        out["useful"] += 1
                    else:
                        out["wasted"] += 1
            if mode == "ontop":
                # the existing prefetch's landings, as captured, enter the tier during this verify
                for e in cap.landed[v][Lx]:
                    tiers[Lx].insert(e, hot | routed, speculative=True)
            # demand: every routed expert that is not a VRAM hit touches RAM
            for e in routed:
                if e in hot:
                    continue
                if tiers[Lx].has(e):
                    tiers[Lx].touch(e)
                    if counted:
                        out["ram_hits"] += 1
                else:
                    if counted:
                        out["ram_misses"] += 1
                        if mode == "ontop" and e in cap.covered[v][Lx]:
                            out["ram_misses_covered_in_capture"] += 1
                    tiers[Lx].insert(e, hot | routed)
        # late rows still land eventually (within this verify) and stay in the tier
        for land_ns, Lx, e, fv in pending:
            if land_ns <= cap.end[v]:
                tiers[Lx].insert(e, cap.hot(v, Lx) | cap.routes(v, Lx), speculative=True)
                if counted:
                    out["reads_landed"] += 1
                    out["wasted"] += 1
        pending = [p for p in pending if p[0] > cap.end[v]]
        # choose the next verify's prefetch at this verify's end
        if policy == "none" or v + 1 >= cap.n or budget <= 0:
            continue
        cands = []  # (priority, layer, expert)
        if policy == "oracle":
            for Lx in range(L):
                nxt_hot = cap.hot(v + 1, Lx)
                for e in cap.routes(v + 1, Lx):
                    if e not in nxt_hot and not tiers[Lx].has(e):
                        cands.append((Lx, Lx, e))
        elif policy == "nearmiss":
            s = scores[v]
            for Lx in range(L):
                routed = cap.routes(v, Lx)
                order = np.argsort(-s[Lx])
                taken = 0
                for e in order[:40].tolist():
                    if e in routed or tiers[Lx].has(e) or e in cap.hot(v, Lx):
                        continue
                    cands.append((-float(s[Lx][e]), Lx, e))
                    taken += 1
                    if taken >= 6:
                        break
        elif policy == "recur":
            for Lx in range(L):
                for e in cap.routes(v, Lx):
                    if not tiers[Lx].has(e) and e not in cap.hot(v, Lx):
                        p = (recur_p or {}).get((v, Lx, e), 0.5)
                        cands.append((-p, Lx, e))
        cands.sort()
        picks = cands[:budget]
        if counted:
            out["reads_issued"] += len(picks)
            out["candidates"] += len(cands)
        t = cap.end[v]
        for i, (_, Lx, e) in enumerate(picks):
            pending.append((t + int(ms_per_row * (i + 1) * 1e6), Lx, e, v + 1))
    nv = cap.n - cap.skip
    res = {k: out[k] / nv for k in out}
    res["evictions_per_verify"] = sum(t.evictions for t in tiers) / nv
    res["speculative_evictions_per_verify"] = sum(t.spec_evictions for t in tiers) / nv
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("model")
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--rows-per-layer", type=int, default=210)
    ap.add_argument("--calibrate", default="170,180,190,200,210")
    ap.add_argument("--budgets", default="20,40,80")
    ap.add_argument("--ms-per-row", type=float, default=2.3)
    ap.add_argument("--ms-per-layer", type=float, default=5.5)
    ap.add_argument("--victims", default="lru,specfirst", help="tier victim rules to replay the policies under")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cap = Capture(args.capture, args.skip)
    nv = cap.n - args.skip
    measured = {
        "demand_misses": sum(len(cap.demand[v][L]) for v in range(args.skip, cap.n) for L in range(cap.layers)) / nv,
        "covered": sum(len(cap.covered[v][L]) for v in range(args.skip, cap.n) for L in range(cap.layers)) / nv,
        "landed": sum(len(cap.landed[v][L]) for v in range(args.skip, cap.n) for L in range(cap.layers)) / nv,
    }
    print(f"{cap.n} verifies; measured per verify: NVMe demand misses {measured['demand_misses']:.1f}, covered by the "
          f"existing prefetch {measured['covered']:.1f}, its landings {measured['landed']:.1f}", flush=True)
    # calibration: the tier replayed with the existing landings should miss as the capture did (demand + covered)
    calib = {}
    for c in (int(x) for x in args.calibrate.split(",")):
        r = replay(cap, "none", "ontop", 0, c, args.ms_per_row, args.ms_per_layer, None)
        calib[c] = {"ram_misses": r["ram_misses"], "of_which_covered_in_capture": r.get("ram_misses_covered_in_capture", 0)}
        print(f"  capacity {c}: replay RAM misses {r['ram_misses']:.1f}/verify (capture: demand {measured['demand_misses']:.1f} "
              f"+ covered {measured['covered']:.1f} = {measured['demand_misses'] + measured['covered']:.1f})", flush=True)
    target = measured["demand_misses"] + measured["covered"]
    best = min(calib, key=lambda c: abs(calib[c]["ram_misses"] - target))
    rows = best if args.rows_per_layer == 0 else args.rows_per_layer
    print(f"using {rows} rows/layer (best calibration {best})", flush=True)
    # the existing prefetch's own churn: tier without its landings, covered misses back in the demand
    alone0 = replay(cap, "none", "alone", 0, rows, args.ms_per_row, args.ms_per_layer, None)
    ontop0 = replay(cap, "none", "ontop", 0, rows, args.ms_per_row, args.ms_per_layer, None)
    print(f"no new policy: RAM misses/verify alone {alone0['ram_misses']:.1f} vs ontop (existing landings in the tier) "
          f"{ontop0['ram_misses']:.1f}; evictions/verify {alone0['evictions_per_verify']:.1f} vs {ontop0['evictions_per_verify']:.1f}", flush=True)
    W, b = load_gates(args.model, cap.header["layer_ids"])
    scores = []
    match = 0
    total = 0
    for v in range(cap.n):
        s = cap.gate_scores(v, W, b)
        scores.append(s)
        if v % 50 == 0:
            ri = cap.recs[v]["router"]
            per_tok = cap.last_per_token  # [L, live, experts]
            for Lx in range(cap.layers):
                for tok in range(per_tok.shape[1]):
                    top = set(np.argsort(-per_tok[Lx, tok])[: cap.topk].tolist())
                    captured = set(cap.ids[ri, Lx, tok].tolist())
                    match += len(top & captured)
                    total += len(captured)
    print(f"gate self-check: {match / max(1, total):.3f} of the captured per-token top-6 reproduced by the recomputed gate",
          flush=True)
    # the tier's victim rule: today's LRU, or never-routed prefetch landings first
    spec0 = replay(cap, "none", "ontop", 0, rows, args.ms_per_row, args.ms_per_layer, None, victim="specfirst")
    print(f"victim rule specfirst, no new policy: RAM misses/verify {spec0['ram_misses']:.1f} (lru {ontop0['ram_misses']:.1f}, "
          f"alone {alone0['ram_misses']:.1f}); speculative evictions {spec0['speculative_evictions_per_verify']:.1f} of "
          f"{spec0['evictions_per_verify']:.1f}", flush=True)
    results = {"measured": measured, "calibration": calib, "rows_per_layer": rows,
               "none": {"alone": alone0, "ontop": ontop0, "ontop_specfirst": spec0}, "policies": {}}
    base = {"alone": alone0["ram_misses"], "ontop": ontop0["ram_misses"]}
    for victim in args.victims.split(","):
        for policy in ("oracle", "nearmiss", "recur"):
            for mode in ("ontop", "alone"):
                for budget in (int(x) for x in args.budgets.split(",")):
                    r = replay(cap, policy, mode, budget, rows, args.ms_per_row, args.ms_per_layer, scores, victim=victim)
                    r["misses_saved"] = base[mode] - r["ram_misses"]
                    r["gib_per_verify"] = r.get("reads_landed", 0) * 13271040 / 2**30
                    results["policies"][f"{policy}-{mode}-{budget}-{victim}"] = r
                    print(f"  {victim:9s} {policy:9s} {mode:6s} budget {budget:3d}: reads {r.get('reads_issued', 0):5.1f}/verify, useful "
                          f"{r.get('useful', 0):5.1f}, wasted {r.get('wasted', 0):5.1f}, late {r.get('late', 0):4.1f}; RAM misses "
                          f"{r['ram_misses']:5.1f} (saved {r['misses_saved']:+5.1f} vs lru-none {base[mode]:.1f}); candidates "
                          f"{r.get('candidates', 0):.0f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
