"""Gate 2: the h=1 native-gate prefetch replay with the RAM-ready filter.

The router-capture replay treated every predicted row as ready in the pinned tier. At h=1 there is no lead for
an NVMe read, so the runtime plan kernel keeps only rows the pinned tier holds when the prediction is made. This
replays the pinned tier to know which rows those were.

**The pinned-tier replay.** The service's tier (``exl3_ram_miss_host.cpp``, ``take_slot_locked``) is per layer:
``tier_sim.ram_rows_per_layer`` rows, LRU by a stamp every routed expert's request refreshes, inclusive of
the layer's VRAM hot set (a hot expert is never a victim) and protecting the request's own routes. A VRAM miss
that the tier lacks is read from NVMe and admitted. The replay runs beside ``tier_sim.DirectInsertReplay`` (exact
to the capture for VRAM), starting from the VRAM hot set as the tier's content. It is **checked against the
run's own registers**: each graph decode step's ``graph_step.layer_ram_rows`` (the rows the service read per
layer), aligned by ``vram_miss``. Agreement is reported per (step, layer) and per token.

**The arms.** ``native_gate`` d6 K=1 h=1 as router_score.py scores it, and ``native_gate_ram``: the same ranking
with every row the replayed tier lacks at the target's gather removed before the first K are taken (the tier
does not change between the prediction at T-1 and T's gather: it is per layer, and only T's own requests
touch T's rows). Link model and pricing as prefetch_baselines.py, with ``--base-ms`` (default 119.0, the copy
engine's smoke) and 1.0 ms/row.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "router-capture"))
import router_score  # noqa: E402  (puts scripts/dsv41, hot-cache-policy and prefetch-study on sys.path)
import policy_sweep  # noqa: E402
import prefetch_baselines  # noqa: E402
import prefetch_sim  # noqa: E402
import tier_sim  # noqa: E402

NUM_EXPERTS = prefetch_sim.NUM_EXPERTS


class PinnedTier:
    """The service's per-layer LRU tier, inclusive of the VRAM hot set; see the module docstring."""

    def __init__(self, capacity: list[int], initial: list[set[int]]) -> None:
        self.capacity = capacity
        self.tick = 0
        self.rows: list[OrderedDict] = []
        for cap, experts in zip(capacity, initial):
            od = OrderedDict()
            for e in sorted(experts)[:cap]:
                od[e] = None
            self.rows.append(od)
        self.no_victim = 0

    def resident(self, row: int) -> set[int]:
        return set(self.rows[row])

    def request(self, row: int, routed: list[int], hot: np.ndarray) -> int:
        """One request of ``row``: touch the routed experts it holds, then read and admit the routed experts it
        lacks (protecting the routes and the hot set). Returns the rows read."""
        od = self.rows[row]
        routed = list(dict.fromkeys(int(e) for e in routed if e >= 0))
        for e in routed:
            if e in od:
                od.move_to_end(e)
        protect = set(routed)
        reads = 0
        for e in routed:
            if e in od:
                continue
            if len(od) >= self.capacity[row]:
                victim = next((v for v in od if v not in protect and not hot[v]), None)
                if victim is None:
                    self.no_victim += 1
                    continue
                del od[victim]
            od[e] = None
            reads += 1
        return reads


def replay_tier(loaded: dict, ram_rows: int) -> dict:
    """Run DIRECT and the pinned tier over every forward. Returns, per decode step, the tier's residency
    before each layer's gather (``ready[steps, layers, experts]``) and the rows it read (``reads[steps,
    layers]``)."""
    layers = loaded["layer_ids"]
    capacity = loaded["hot_capacity"]
    initial = {layer: list(range(slots)) for layer, slots in capacity.items()}
    direct = tier_sim.DirectInsertReplay(initial, capacity, NUM_EXPERTS)
    ram_caps = tier_sim.ram_rows_per_layer(ram_rows, len(layers), NUM_EXPERTS)
    tier = PinnedTier(ram_caps, [set(initial[layer]) for layer in layers])
    decode = [f for f in loaded["forwards"] if f["phase"] == "decode"]
    ready = np.zeros((len(decode), len(layers), NUM_EXPERTS), dtype=bool)
    reads = np.zeros((len(decode), len(layers)), dtype=np.int64)
    eager_sim, eager_measured = 0, 0
    step = 0
    for forward in loaded["forwards"]:
        if forward["phase"] == "capture":
            continue
        hot = direct.where >= 0  # before this forward's inserts: the hot set each gather's request carries
        if forward["kind"] == "graph":
            if forward["phase"] == "decode":
                for li in range(len(layers)):
                    ready[step, li, list(tier.rows[li])] = True
            for li, layer in enumerate(layers):
                n = tier.request(li, forward["routes"][layer], hot[li])
                if forward["phase"] == "decode":
                    reads[step, li] = n
            direct.graph_forward(forward["routes"], forward["phase"])
            if forward["phase"] == "decode":
                step += 1
        else:
            for li, layer in enumerate(layers):
                if layer in forward["counts"]:
                    eager_sim += tier.request(li, forward["counts"][layer][0], hot[li])
            direct.eager_forward(forward["tokens"], forward["counts"], forward["phase"])
    return {"ready": ready, "reads": reads, "no_victim": tier.no_victim, "eager_reads": eager_sim,
            "capacity": ram_caps}


def measured_reads(trace: str, loaded: dict) -> np.ndarray:
    """``graph_step.layer_ram_rows`` aligned to the decode graph forwards by their ``vram_miss`` totals."""
    steps = []
    with open(trace) as f:
        for text in f:
            if '"graph_step"' not in text:
                continue
            line = json.loads(text)
            if line.get("kind") == "graph_step":
                steps.append((int(line["vram_miss"]), line["layer_ram_rows"]))
    layers = loaded["layer_ids"]
    decode = [f for f in loaded["forwards"] if f["phase"] == "decode"]
    want = [sum(f["misses"][layer] for layer in layers) for f in decode]
    got = [v for v, _ in steps]
    best = None
    for off in range(-4, 5):
        pairs = [(i, i + off) for i in range(len(want)) if 0 <= i + off < len(got)]
        agree = sum(want[i] == got[j] for i, j in pairs)
        if best is None or agree > best[0]:
            best = (agree, off, len(pairs))
    agree, off, n = best
    out = np.full((len(decode), len(layers)), -1, dtype=np.int64)
    for i in range(len(decode)):
        j = i + off
        if 0 <= j < len(steps) and want[i] == got[j]:
            out[i] = steps[j][1]
    print(f"graph_step alignment: offset {off}, vram_miss agrees on {agree} of {n} decode steps", flush=True)
    return out


class NativeGateRam(prefetch_sim.Predictor):
    """router_score.NativeGate with rows the pinned tier lacks removed."""

    def __init__(self, rank: router_score.Rankings, depth: int, ready: np.ndarray) -> None:
        self.rank_, self.depth, self.ready = rank, depth, ready
        self.name = f"native_gate_ram_d{depth}"

    def rank(self, step, target, horizon):
        if not self.rank_.valid[step, target]:
            return ()
        order = self.rank_.order[step, target, : self.depth]
        return [int(e) for e in order if self.ready[step, target, int(e)]]


_READY: dict = {}


def build(arm: prefetch_baselines.Arm) -> prefetch_sim.Predictor:
    if arm.predictor == "native_gate_ram":
        return NativeGateRam(router_score._RANKINGS[arm.horizon], int(arm.decay), _READY["ready"])
    return router_score.build(arm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("router_prefix")
    p.add_argument("model_path")
    p.add_argument("--prompts", type=int, required=True)
    p.add_argument("--ram-rows", type=int, default=8063, help="pinned tier rows (server log: startup 'rows')")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    p.add_argument("--budgets", type=float, nargs="+", default=[0.85, 1.7])
    p.add_argument("--base-ms", type=float, default=119.0)
    p.add_argument("--ms-per-row", type=float, default=1.0)
    p.add_argument("--belady-test", type=float, default=47.23)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    started = time.time()

    loaded = tier_sim.load_forwards(args.trace)
    for forward in loaded["forwards"]:
        forward.pop("hot", None)
    stream = prefetch_sim.decode_stream(loaded)
    train, test, _ = policy_sweep.split_requests(stream.rids, args.prompts)
    steps, layers = len(stream.rids), len(stream.layers)
    report: dict = {"trace": args.trace, "run": loaded["run"], "decode_steps": steps,
                    "test_steps": int(test.sum()), "ram_rows": args.ram_rows}

    tier = replay_tier(loaded, args.ram_rows)
    measured = measured_reads(args.trace, loaded)
    ok = (measured >= 0).all(axis=1)
    sim_r, meas_r = tier["reads"], measured
    cell_eq = (sim_r == meas_r)[ok]
    report["tier_check"] = {
        "capacity_per_layer": [min(tier["capacity"]), max(tier["capacity"])],
        "aligned_steps": int(ok.sum()),
        "cells_equal": float(cell_eq.mean()),
        "sim_reads_per_token": float(sim_r[ok].sum(axis=1).mean()),
        "measured_reads_per_token": float(meas_r[ok].sum(axis=1).mean()),
        "test": {"sim": float(sim_r[ok & test].sum(axis=1).mean()),
                 "measured": float(meas_r[ok & test].sum(axis=1).mean()),
                 "cells_equal": float((sim_r == meas_r)[ok & test].mean())},
        "no_victim": tier["no_victim"],
        "eager_sim_reads": tier["eager_reads"],
    }
    print("tier check", report["tier_check"], flush=True)

    capture = router_score.load_capture(args.router_prefix)
    records = router_score.decode_records(loaded, capture)
    W, bias = router_score.load_gates(args.model_path, stream.layers)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    def x_of(step_index: np.ndarray, layer: int) -> np.ndarray:
        return router_score.bf16_to_f32(np.asarray(capture.x[records[step_index], layer]))

    router_score._RANKINGS[1] = router_score.rank_horizon(stream, x_of, W, bias, 1, 6)
    del W, bias

    # Offline: of the rank-1 non-VRAM-resident candidates, how many the tier lacks, and how that splits by
    # whether the target routes them.
    probe = router_score.Probe(steps, layers)
    base = prefetch_sim.run_arm(loaded, stream, probe, 1, 0)
    resident = probe.resident
    missing = np.zeros_like(resident)
    np.put_along_axis(missing, stream.routes, True, axis=2)
    missing &= ~resident
    rank = router_score._RANKINGS[1]
    order = rank.order[:, :, :6].astype(np.int64)
    here = np.take_along_axis(resident, order, axis=2)
    first = ~here & (np.cumsum(~here, axis=2) == 1) & rank.valid[:, :, None]
    cand = (order * first).sum(axis=2)
    issued = first.any(axis=2)
    cand_ready = np.take_along_axis(tier["ready"], cand[:, :, None], axis=2)[:, :, 0]
    cand_useful = np.take_along_axis(missing, cand[:, :, None], axis=2)[:, :, 0]
    m = issued & test[:, None]
    report["rank1_unfiltered_test"] = {
        "issued_per_token": float(m.sum() / test.sum()),
        "ram_ready_share": float(cand_ready[m].mean()),
        "useful_share": float(cand_useful[m].mean()),
        "ram_ready_share_of_useful": float(cand_ready[m & cand_useful].mean()),
        "ram_ready_share_of_wasted": float(cand_ready[m & ~cand_useful].mean()),
        "misses_ram_ready_share": float((tier["ready"] & missing)[test].sum() / max(missing[test].sum(), 1)),
    }
    print("rank-1 candidates (test)", report["rank1_unfiltered_test"], flush=True)
    for name, ready_mask in (("unfiltered", np.ones_like(tier["ready"])), ("ram_ready", tier["ready"])):
        sub = {}
        for split, mask in (("test", test), ("train", train)):
            filt_order = np.zeros_like(rank.order)
            # the filtered ranking, padded with a resident expert so offline_precision skips it
            for s in np.nonzero(mask)[0]:
                for t in range(layers):
                    kept = [e for e in rank.order[s, t, :6] if ready_mask[s, t, e]]
                    pad = [e for e in rank.order[s, t, :6] if not ready_mask[s, t, e]]
                    filt_order[s, t, :6] = kept + pad
            here2 = resident.copy()
            for s in np.nonzero(mask)[0]:
                for t in range(layers):
                    for e in rank.order[s, t, :6]:
                        if not ready_mask[s, t, e]:
                            here2[s, t, e] = True
            sub[split] = router_score.offline_precision(
                router_score.Rankings(1, filt_order, rank.score, rank.valid), here2, missing, mask, 1, 6)
        report.setdefault("offline_k1", {})[name] = sub
        print("offline", name, sub["test"], flush=True)
    del resident, missing, probe, here, first

    _READY["ready"] = tier["ready"]
    arms = [prefetch_baselines.Arm("none", 0, 1)]
    for k in args.ks:
        arms += [prefetch_baselines.Arm("native_gate", k, 1, 6.0), prefetch_baselines.Arm("native_gate_ram", k, 1, 6.0),
                 prefetch_baselines.Arm("oracle", k, 1)]
    prefetch_baselines.build = build
    prefetch_baselines._STATE.update(loaded=loaded, stream=stream, train=train, test=test,
                                     all=np.ones(steps, dtype=bool), budgets=args.budgets)
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = list(pool.imap_unordered(prefetch_baselines.run, arms))
    none = next(r for r in results if r["arm"]["predictor"] == "none")
    for r in results:
        for name, split in r["splits"].items():
            ref = none["splits"][name]["demand"]
            split["ms"] = {b: args.base_ms + (v - ref) * args.ms_per_row for b, v in split["exposed"].items()}
            if name == "test":
                split["gap_share"] = {b: (ref - v) / (ref - args.belady_test) for b, v in split["exposed"].items()}
                split["precision"] = split["useful"] / split["prefetch"] if split["prefetch"] else float("nan")
        t = r["splits"]["test"]
        a = r["arm"]
        print(f"  {a['predictor']:>16} k={a['k']}  demand {t['demand']:6.2f} prefetch {t['prefetch']:6.2f} useful "
              f"{t['useful']:6.2f} prec {t.get('precision', float('nan')):.2f} exposed "
              + " ".join(f"{b}:{v:6.2f}" for b, v in t["exposed"].items())
              + "  ms " + " ".join(f"{b}:{v:6.1f}" for b, v in t["ms"].items()), flush=True)
    report["link"] = {"budgets": args.budgets, "ms_per_row": args.ms_per_row, "base_ms": args.base_ms}
    report["results"] = sorted(results, key=lambda r: (r["arm"]["predictor"], r["arm"]["k"]))
    report["seconds"] = round(time.time() - started, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
