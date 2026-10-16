"""RAM-prefetch policies priced by tier, with the prefill share on: which speculative NVMe reads are worth their drive
time. Uses lane_tier_replay's validated tier model (rules 1-10, with the share) and P.

Per arm, one replay of the capture's verifies in time order, with the capture's own VRAM hot sets and GPU lanes (the
prefetch does not move the lane key) and the RAM tier driven by them:
  reads: at each source record (verify v, layer L) the arm picks up to n experts of layer T = L + 2 (the horizon), n
    being the number the capture itself issued there (spec_submit: its idle-drive gate and cap 2 are timing, not the
    scorer's), and the i-th pick lands when the capture's i-th read there landed (spec_land; a dropped read drops it);
  use: a landed row the target record (v, T) routes, on a VRAM miss whose row is not in RAM, is used: on a GPU lane it
    is pool-served (no NVMe stream), on a CPU lane it is taken by the forced miss (a RAM-warm CPU lane); either way the
    row is mapped (rule 5);
  waste: a landed row not used is promoted into the map at the run's measured promotion rate (rule 6), else dropped;
  drive time: each read in flight at a record adds one NVMe lane to the record's shared-drive landing term (P's
    miss_land.per_record_gpu_nvme_lane), the cost P measured for a GPU NVMe lane's read beside a CPU miss.
Prefill admissions follow rule 10 with the share at ``--share`` rows.

Arms:
  off      no speculative reads.
  today    the capture's own picks (spec_submit: the served L+2 T3 cap-2 scorer).
  native   layer T's gate on layer L's router input (sqrt(softplus(logits)) + bias, max over live tokens), the best
           scored experts that are neither VRAM-hot at T nor in RAM.
  cpu      the same scores aimed at the CPU's forced misses: among the experts the gate predicts T routes (its top
           ``--predict-routed``), those that are VRAM misses, not in RAM, and not among the k misses the plan would give
           GPU lanes (k = the capture's GPU lanes at (v, T), ranked by the residency key).
Validation (``--validate``): the today arm against the capture's counters: spec_landed, spec_used (spec_use events)
and pool-served rows (pool_point events), each within 10%.

Usage: prefetch_eval.py CAPTURE [--model MODEL_DIR] [--prices-from RUN] [--share 64] [--arms off,today,native,cpu]
                        [--validate] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import struct
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_tier_replay as ltr  # noqa: E402

NODES = 2
HORIZON = 2
MODEL_DIR = "/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40"


def _bf16(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


def _read_tensor(path: str, name: str) -> np.ndarray:
    dtypes = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(n))[name]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = np.frombuffer(f.read(end - start), dtype=dtypes[meta["dtype"]]).reshape(meta["shape"])
    return _bf16(raw) if meta["dtype"] == "BF16" else raw.astype(np.float32)


class GateScorer:
    """Layer T's gate on layer L's router input, from the router capture (ram_prefetch_replay.py's gate_scores)."""

    def __init__(self, d: str, model: str, layer_ids: list[int]):
        with open(os.path.join(model, "model.safetensors.index.json")) as f:
            where = json.load(f)["weight_map"]
        self.W = np.stack([_read_tensor(os.path.join(model, where[f"layers.{i}.ffn.gate.weight"]),
                                        f"layers.{i}.ffn.gate.weight") for i in layer_ids])
        self.b = np.stack([_read_tensor(os.path.join(model, where[f"layers.{i}.ffn.gate.bias"]),
                                        f"layers.{i}.ffn.gate.bias") for i in layer_ids])
        with open(os.path.join(d, "router.json")) as f:
            meta = json.load(f)
        self.tokens, self.hidden, self.layers = int(meta["tokens"]), int(meta["hidden"]), len(layer_ids)
        seq = np.fromfile(os.path.join(d, "router.seq.bin"), dtype=np.int64).reshape(-1, 3)
        self.live = seq[:, 2]
        self.ids = np.fromfile(os.path.join(d, "router.ids.bin"), dtype=np.int32).reshape(
            seq.shape[0], self.layers, self.tokens, int(meta["topk"]))
        self.x = np.memmap(os.path.join(d, "router.x.bin"), dtype=np.uint16, mode="r",
                           shape=(seq.shape[0], self.layers, self.tokens, self.hidden))
        self._cache: dict = {}

    def scores(self, router: int, source: int, target: int) -> np.ndarray:
        key = (router, source, target)
        if key not in self._cache:
            live = int(self.live[router]) or self.tokens
            x = _bf16(np.asarray(self.x[router, source, :live, :]))
            s = np.sqrt(np.logaddexp(0.0, x @ self.W[target].T)) + self.b[target][None, :]
            self._cache = {key: s.max(axis=0)}
        return self._cache[key]

    def self_check(self, router: int, layer: int) -> float:
        """The share of the captured per-token top-k the recomputed gate reproduces at its own layer."""
        live = int(self.live[router]) or self.tokens
        x = _bf16(np.asarray(self.x[router, layer, :live, :]))
        s = np.sqrt(np.logaddexp(0.0, x @ self.W[layer].T)) + self.b[layer][None, :]
        k = self.ids.shape[-1]
        hit = tot = 0
        for t in range(live):
            got = set(np.argsort(-s[t])[:k].tolist())
            want = set(int(e) for e in self.ids[router, layer, t] if e >= 0)
            hit += len(got & want)
            tot += len(want)
        return hit / max(1, tot)


def price(P: ltr.Prices, nv: int, po: int, dma: int, teams: dict, inflight: int) -> tuple[float, float]:
    """lane_eval.price_lanes with ``inflight`` speculative reads sharing the drives: each adds to the forced-miss
    landing term as one more NVMe lane of the record would."""
    cpu_misses = sum(len(t[2]) for t in teams.values())
    on = 1.0 if nv + po else 0.0
    stream = float(P.stream @ [on, nv, po, dma * on, nv * cpu_misses]) if nv + po else 0.0
    cpu = 0.0
    for lanes, routes, misses in teams.values():
        if lanes or misses:
            cpu = max(cpu, P.team_finish(lanes, routes, misses, nv + inflight))
    wait = max(0.0, max(stream, cpu) + P.offset - stream) if (cpu or stream) else 0.0
    return stream, wait


def load_reads(run: ltr.Run, d: str):
    """The capture's speculative reads per source record seq, in pick order: [(target row, expert, submit ns, land ns
    or None)], and its use events per target record seq: {seq: {expert: 'gpu' | 'cpu'}}."""
    subs = collections.defaultdict(list)
    lands = {}
    uses = collections.defaultdict(dict)
    for path, e in ltr._events(glob.glob(os.path.join(d, "events.*.exl3-spec*.jsonl")) +
                               glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl")),
                               {"spec_submit", "spec_land", "spec_use", "pool_point"}):
        k = e["event"]
        if k == "spec_submit":
            subs[e["seq"]].append((e["gen"], e["row"], e["a"], e["ns"]))
        elif k == "spec_land":
            lands[(e["seq"], e["row"], e["a"])] = e["ns"]
        elif k == "spec_use":
            uses[e["seq"]][e["a"]] = "cpu"
        elif k == "pool_point":
            uses[e["seq"]][e["a"]] = "gpu"
    reads = {}
    for seq, xs in subs.items():
        xs.sort()
        reads[seq] = [(row, a, ns, lands.get((seq, row, a))) for _, row, a, ns in xs]
    return reads, uses


def evaluate(run: ltr.Run, P: ltr.Prices, arm: str, layers: int, share: int, scorer: GateScorer | None,
             predict_routed: int, reads, uses) -> dict:
    d = run.dir
    header, log = ltr.route_log(d)
    off = ltr.align(run, log)
    rows, frac, pool_share, full = ltr.numa_capacities(d)
    full = set(full)
    p_promote = ltr.promote_fraction(d)
    tiers = [ltr.layer_tiers(rows.get(L, 210), frac, pool_share) for L in range(layers)]
    by_vr = {(r.verify, r.row): r for r in run.records}
    by_seq = {r.seq: r for r in run.records}
    replayed = {r.seq for r in run.records}
    timeline = []
    for v in range(len(run.verifies)):
        i = v + off
        if not 0 <= i < len(log):
            continue
        for L in range(layers):
            r = by_vr.get((v, L))
            ns = r.post if r is not None else run.verifies[v][0] + L * 5_000_000
            timeline.append((ns, 2, "gather", (v, i, L)))
    for _, e in ltr._events(glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl")), {"evict"}):
        if e["seq"] not in replayed:
            timeline.append((e["ns"], 0, "evict", (e["row"], e["group"], e["a"], e["c"])))
    eager = ltr.eager_lines(d)
    for k, line in enumerate(eager):
        timeline.append((int(line["t"] * 1e9) - 1, 0, "eager", k))
    timeline.sort(key=lambda t: (t[0], t[1]))

    clean_v = {v for v in range(len(run.verifies)) if run.prompt_of[v] >= 0 and run.clean[run.prompt_of[v]]}
    first = {}
    for v in range(len(run.verifies)):
        first.setdefault(run.prompt_of[v], v)
    owned = [set() for _ in range(layers)]
    last_hot = [set() for _ in range(layers)]
    score = [collections.defaultdict(float) for _ in range(layers)]
    pool = collections.defaultdict(dict)  # (v, target row) -> {expert: land ns}
    inflight = []  # (submit ns, land ns) of reads not yet landed
    out = collections.Counter()
    tick = 0
    pending = []  # promotions due: (ns, row, expert)
    for ns, _, kind, payload in timeline:
        tick += 1
        while pending and pending[0][0] <= ns:
            _, row, e = pending.pop(0)
            if row not in full:
                tiers[row][e % NODES].promote(e, last_hot[row])
        if kind == "eager":
            line = eager[payload]
            L = line["layer"]
            if L < layers and L not in full:
                on = line["phase"] == "extend"
                tick, claims, demand, _ = ltr.eager_admit(tiers[L], line, last_hot[L], tick,
                                                          share=share if on else 0, owned=owned[L])
                if on:
                    out["prefill_claims"] += claims
                    out["prefill_demand"] += demand
            continue
        if kind == "evict":
            row, g, old, new = payload
            if row < layers and row not in full:
                tiers[row][g].apply_eviction(old, new, tick)
            continue
        v, i, L = payload
        cur = log[i]
        nxt = log[i + 1] if i + 1 < len(log) else None
        cnt = collections.Counter(e for e in cur["routes"][L] if e >= 0)
        u = set(cnt)
        hot = set(cur["hot"][L])
        misses = u - hot
        inserted = (misses & set(nxt["hot"][L])) if nxt is not None else set()
        tier = tiers[L]
        sc = score[L]
        dk = 0.98 ** int(cur.get("forward_tokens") or 1)
        for e in list(sc):
            sc[e] *= dk
        for e, c in cnt.items():
            sc[e] += c
        r = by_vr.get((v, L))
        # this record's reads, for layer L + HORIZON of the same verify
        T = L + HORIZON
        if arm != "off" and r is not None and r.seq in reads and T < layers:
            obs = [x for x in reads[r.seq] if x[0] == T]
            n = len(obs)
            picks: list[int] = []
            if arm == "today":
                picks = [a for _, a, _, _ in obs]
            elif n:
                hot_t = set(cur["hot"][T])
                s = scorer.scores(cur["router"], L, T)
                order = [int(e) for e in np.argsort(-s)]
                eligible = [e for e in order if e not in hot_t and (T in full or not tiers[T][e % NODES].has(e))
                            and e not in pool[(v, T)]]
                if arm == "native":
                    picks = eligible[:n]
                else:  # cpu: predicted routed misses the plan would not give a GPU lane
                    routed_pred = set(order[:predict_routed])
                    miss_pred = [e for e in order[:predict_routed] if e not in hot_t]
                    k = len((set(e for e in log[i]["routes"][T] if e >= 0) - hot_t) & set(nxt["hot"][T])) \
                        if nxt is not None else 0
                    gpu_pred = set(sorted(miss_pred, key=lambda e: (-score[T][e], e))[:k])
                    picks = [e for e in eligible if e in routed_pred and e not in gpu_pred][:n]
            for (row, a, sub, land), e in zip(obs, picks):
                out["issued"] += 1
                if land is not None:
                    out["landed"] += 1
                    pool[(v, T)][e] = land
                    inflight.append((sub, land))
        # pooled rows for this record, landed before its gather
        pooled = {e for e, land in pool.pop((v, L), {}).items() if land <= ns}
        if L in full:
            warm = set(misses)
        else:
            warm = {e for e in misses if tier[e % NODES].has(e)}
        used = (misses - warm) & pooled
        nv = sum(1 for e in inserted if e not in warm and e not in used)
        po = sum(1 for e in inserted if e in used)
        dma = sum(1 for e in inserted if e in warm)
        teams = {0: [0, 0, []], 1: [0, 0, []]}
        for e in sorted(misses - inserted, key=lambda x: -cnt[x]):
            t = teams[e % NODES]
            if e in warm or e in used:
                t[0] += 1
                t[1] += cnt[e]
            else:
                t[2].append(cnt[e])
        flying = sum(1 for sub, land in inflight if sub <= ns < land)
        inflight = [(sub, land) for sub, land in inflight if land > ns]
        out["used"] += len(used)
        out["pool_served"] += po
        if r is not None and r.seq in uses:
            obs_u = uses[r.seq]
            out["obs_used"] += len(obs_u)
            out["obs_pool_served"] += sum(1 for x in obs_u.values() if x == "gpu")
        for e in pooled - used:
            out["unused_landed"] += 1
            if ltr._hash01(L, e, ns & 0xFFFFFFFF) < p_promote:
                pending.append((ns, L, e))
                out["promoted"] += 1
        if v in clean_v:
            s0, w0 = price(P, nv, po, dma, teams, 0)
            s1, w1 = price(P, nv, po, dma, teams, flying)
            pos = v - first[run.prompt_of[v]]
            b = "0-4" if pos < 5 else "5-14" if pos < 15 else "15+"
            for key, val in (("ms", s1 + w1), ("ms_no_drive_charge", s0 + w0), ("drive_charge_ms", s1 + w1 - s0 - w0),
                             ("stream", s1), ("wait", w1), ("gpu_nvme", nv), ("gpu_pool", po),
                             ("cpu_forced", sum(len(t[2]) for t in teams.values()))):
                out[key] += val
                out[f"{key}@{b}"] += val
            if L == 0:
                out["verifies"] += 1
                out[f"verifies@{b}"] += 1
        last_hot[L] = hot
        if L in full:
            continue
        for e in u:
            t = tier[e % NODES]
            if t.has(e):
                t.stamp[e] = tick
        for e in sorted(misses, key=lambda x: x in warm):
            tier[e % NODES].insert(e, tick, hot | u)
        owned[L] -= u | hot
    nvf = max(1, out["verifies"])
    res = {}
    for key in ("ms", "ms_no_drive_charge", "drive_charge_ms", "stream", "wait", "gpu_nvme", "gpu_pool", "cpu_forced"):
        res[key] = out[key] / nvf
        res[f"{key}_by_position"] = {b: out[f"{key}@{b}"] / max(1, out[f"verifies@{b}"]) for b in ("0-4", "5-14", "15+")}
    prompts = max(1, len(set(x["forward_pass_id"] for x in eager if x["phase"] == "extend")))
    res.update({k: out[k] for k in ("issued", "landed", "used", "pool_served", "unused_landed", "promoted",
                                    "obs_used", "obs_pool_served")})
    res["prefill_claims_per_prefill"] = out["prefill_claims"] / prompts
    res["prefill_demand_per_prefill"] = out["prefill_demand"] / prompts
    res["verifies"] = out["verifies"]
    res["verifies_by_position"] = {b: out[f"verifies@{b}"] for b in ("0-4", "5-14", "15+")}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--prices-from", default=None)
    ap.add_argument("--share", type=int, default=64)
    ap.add_argument("--arms", default="off,today,native,cpu")
    ap.add_argument("--predict-routed", type=int, default=24)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    run = ltr.Run(a.capture)
    P = ltr.Prices(ltr.Run(a.prices_from) if a.prices_from else run)
    layers = max(r.row for r in run.records) + 1
    reads, uses = load_reads(run, a.capture)
    with open(os.path.join(a.capture, "counters.json")) as f:
        counters = json.load(f)
    res = {"capture": a.capture, "share": a.share, "counters": {k: counters.get(k) for k in (
        "spec_issued", "spec_landed", "spec_used", "spec_pool_served", "spec_promoted_early", "spec_promoted")},
        "arms": {}}
    scorer = None
    if any(x in ("native", "cpu") for x in a.arms.split(",")):
        header, log = ltr.route_log(a.capture)
        scorer = GateScorer(a.capture, a.model, header["layer_ids"])
        r0 = log[len(log) // 2]["router"]
        res["gate_self_check"] = round(float(np.mean([scorer.self_check(r0, L) for L in (2, 10, 20, 30)])), 3)
    for arm in a.arms.split(","):
        res["arms"][arm] = evaluate(run, P, arm, layers, a.share, scorer, a.predict_routed, reads, uses)
        print(arm, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["arms"][arm].items()
                               if not isinstance(v, dict)}), flush=True)
    if a.validate and "today" in res["arms"]:
        t = res["arms"]["today"]
        obs_issued = sum(len(x) for x in reads.values())
        obs_landed = sum(1 for x in reads.values() for y in x if y[3] is not None)
        checks = {"landed": (t["landed"], obs_landed), "used": (t["used"], t["obs_used"]),
                  "pool_served": (t["pool_served"], t["obs_pool_served"])}
        res["validation"] = {k: {"model": m, "observed": o, "ok": abs(m - o) <= 0.10 * max(1, o)} for k, (m, o) in
                             checks.items()}
        res["validation"]["trace_vs_counters"] = {"issued": (obs_issued, counters.get("spec_issued")),
                                                  "landed": (obs_landed, counters.get("spec_landed"))}
        print(json.dumps(res["validation"], indent=1))
    base = res["arms"].get("off")
    if base:
        res["delta_vs_off"] = {arm: {"ms": round(r["ms"] - base["ms"], 2),
                                     "ms_no_drive_charge": round(r["ms_no_drive_charge"] - base["ms_no_drive_charge"], 2),
                                     "by_position": {b: round(r["ms_by_position"][b] - base["ms_by_position"][b], 2)
                                                     for b in r["ms_by_position"]}}
                               for arm, r in res["arms"].items()}
        print(json.dumps(res["delta_vs_off"], indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
