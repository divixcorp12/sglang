"""Native-gate lookahead on a router capture: layer T's own gate applied to layer T-h's router input.

Input: a stage trace with the graph route log (``stages.jsonl``) and its RouterCapture side files
(``router.*``, exl3_stream_trace.RouterCapture), and the checkpoint's gates (``layers.{i}.ffn.gate.weight``
[384, 5120] and ``.bias`` [384], the score correction). Routing is DeepSeek-V4.1's: scores =
sqrt(softplus(x @ W.T)) in fp32, experts ranked by scores + bias, top 6.

1. **Self-check (h = 0).** Each layer's gate on its own captured input must give the captured routes (as a
   set: the route log keeps router order, not rank order), and the captured weights must be those scores,
   renormalized.
2. **Offline precision.** For each decode step and target layer T, the non-resident rows the predictor would
   fetch: the first K experts of its ranking (cut to the top ``depth``) that are not resident just before T's
   gather, in the DIRECT replay without prefetch (exact to the capture). Precision = those that T routes /
   rows fetched; recall = those / T's misses. The source is prefetch_sim.source_of: the same token's layer
   T-h, or, for T < h, the same request's previous token.
3. **Prefetch replay.** The ranking as a prefetch_sim predictor through PrefetchReplay and link_exposed, next
   to none / oracle / NoisyOracle arms on the same capture, as prefetch_baselines.py scores them.

Fitting (none here beyond the depth) would use even prompts; every number is also reported on the odd
(held-out) prompts, which the tables use.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
sys.path.insert(0, os.path.join(HERE, "..", "hot-cache-policy"))
sys.path.insert(0, os.path.join(HERE, "..", "prefetch-study"))
import policy_sweep  # noqa: E402
import prefetch_baselines  # noqa: E402
import prefetch_sim  # noqa: E402
import tier_sim  # noqa: E402

NUM_EXPERTS = prefetch_sim.NUM_EXPERTS
TOPK = 6


class RouterCapture(msgspec.Struct):
    """The side files, indexed by record: ``x`` bf16 as uint16 ``[records, layers, hidden]`` (memory-mapped),
    ``w`` fp32 ``[records, layers, topk]``, and ``record_of_seq``."""

    header: dict
    x: np.ndarray
    w: np.ndarray
    record_of_seq: dict


def load_capture(prefix: str) -> RouterCapture:
    with open(prefix + ".json") as f:
        header = json.load(f)
    layers, hidden, topk = len(header["layer_ids"]), header["hidden"], header["topk"]
    x = np.memmap(prefix + ".x.bin", dtype=np.uint16, mode="r").reshape(-1, layers, hidden)
    w = np.fromfile(prefix + ".w.bin", dtype=np.float32).reshape(-1, layers, topk)
    keys = np.fromfile(prefix + ".seq.bin", dtype=np.int64).reshape(-1, 2)
    if not (len(x) == len(w) == len(keys)):
        raise ValueError(f"{prefix}: side files disagree on the record count ({len(x)}, {len(w)}, {len(keys)})")
    return RouterCapture(header, x, w, {int(seq): i for i, seq in enumerate(keys[:, 0])})


def bf16_to_f32(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


def load_gates(model_path: str, layer_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """``W`` fp32 ``[layers, 384, hidden]`` and ``bias`` fp32 ``[layers, 384]``, in ``layer_ids`` order."""
    from safetensors import safe_open

    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    weights, biases = [], []
    for layer in layer_ids:
        pair = []
        for name in (f"layers.{layer}.ffn.gate.weight", f"layers.{layer}.ffn.gate.bias"):
            with safe_open(os.path.join(model_path, where[name]), framework="pt") as f:
                pair.append(f.get_tensor(name).float())
        weights.append(pair[0])
        biases.append(pair[1])
    return torch.stack(weights), torch.stack(biases)


def decode_records(loaded: dict, capture: RouterCapture) -> np.ndarray:
    """The capture record of each decode step of prefetch_sim.decode_stream, in its order."""
    records = []
    for forward in loaded["forwards"]:
        if forward["phase"] == "decode":
            records.append(capture.record_of_seq[forward["seq"]])
    return np.asarray(records, dtype=np.int64)


class Rankings(msgspec.Struct):
    """Per horizon: ``order[steps, layers, depth]`` (int16) the target layer's experts by biased score,
    best first, ``score`` the matching biased scores, and ``valid[steps, layers]`` (a source exists)."""

    horizon: int
    order: np.ndarray
    score: np.ndarray
    valid: np.ndarray


def rank_horizon(
    stream: prefetch_sim.DecodeStream,
    x_of: callable,
    W: torch.Tensor,
    bias: torch.Tensor,
    horizon: int,
    depth: int,
) -> Rankings:
    steps, layers = len(stream.rids), len(stream.layers)
    order = np.zeros((steps, layers, depth), dtype=np.int16)
    score = np.zeros((steps, layers, depth), dtype=np.float32)
    valid = np.zeros((steps, layers), dtype=bool)
    for target in range(layers):
        sources = [prefetch_sim.source_of(stream, s, target, horizon) for s in range(steps)]
        ok = np.array([source is not None for source in sources])
        if not ok.any():
            continue
        src_steps = np.array([source[0] for source in sources if source is not None])
        src_layer = {source[1] for source in sources if source is not None}
        if len(src_layer) != 1:
            raise AssertionError("one source layer per target and horizon")
        x = torch.from_numpy(x_of(src_steps, src_layer.pop()))
        logits = x @ W[target].T
        biased = torch.nn.functional.softplus(logits).sqrt() + bias[target]
        top = torch.topk(biased, depth, dim=-1)
        order[ok, target] = top.indices.numpy().astype(np.int16)
        score[ok, target] = top.values.numpy()
        valid[ok, target] = True
    return Rankings(horizon, order, score, valid)


def self_check(stream, x_of, W, bias, capture: RouterCapture, records: np.ndarray) -> dict:
    """h = 0: the offline gate against the captured routes and weights."""
    steps, layers = len(stream.rids), len(stream.layers)
    same_set = np.zeros((steps, layers), dtype=bool)
    weight_err = []
    scale = []
    for target in range(layers):
        x = torch.from_numpy(x_of(np.arange(steps), target))
        logits = x @ W[target].T
        scores = torch.nn.functional.softplus(logits).sqrt()
        ids = torch.topk(scores + bias[target], TOPK, dim=-1).indices.numpy()
        captured = stream.routes[:, target]
        same_set[:, target] = [set(a) == set(b) for a, b in zip(ids, captured)]
        w = capture.w[records, target]  # route order, as captured
        want = np.take_along_axis(scores.numpy(), captured, axis=1)
        want = want / want.sum(axis=1, keepdims=True)
        sums = w.sum(axis=1, keepdims=True)
        scale.append(sums[:, 0])
        weight_err.append(np.abs(w / sums - want).max(axis=1))
    weight_err = np.stack(weight_err, axis=1)
    return {
        "route_sets_equal": float(same_set.mean()),
        "route_sets_differ": int((~same_set).sum()),
        "cells": int(same_set.size),
        "steps_all_layers_equal": float(same_set.all(axis=1).mean()),
        "weight_max_abs_err": float(weight_err.max()),
        "weight_p99_abs_err": float(np.quantile(weight_err, 0.99)),
        "weight_sum_mean": float(np.concatenate(scale).mean()),
    }


# ------------------------------------------------------------------ residency and offline precision


class Probe(prefetch_sim.Predictor):
    """Records, for each decode step and target layer, the residency just before its gather; prefetches nothing."""

    name = "probe"

    def __init__(self, steps: int, layers: int) -> None:
        self.resident = np.zeros((steps, layers, NUM_EXPERTS), dtype=bool)

    def bind(self, sim, index):
        self.sim, self.layer_of = sim, {li: layer for layer, li in index.items()}

    def rank(self, step, target, horizon):
        self.resident[step, target, list(self.sim.resident(self.layer_of[target]))] = True
        return ()


def offline_precision(rank: Rankings, resident: np.ndarray, missing: np.ndarray, mask: np.ndarray, k: int,
                      depth: int) -> dict:
    """The K first non-resident experts of the top ``depth`` against the target's misses, over ``mask`` steps.

    ``missing[step, layer, expert]``: the step routes the expert there and it is not resident."""
    order = rank.order[mask, :, :depth].astype(np.int64)
    here = np.take_along_axis(resident[mask], order, axis=2)
    miss = np.take_along_axis(missing[mask], order, axis=2)
    nth = np.cumsum(~here, axis=2)  # 1-based rank among the non-resident candidates
    pick = ~here & (nth <= k) & rank.valid[mask][:, :, None]
    first = ~here & (nth == 1) & rank.valid[mask][:, :, None]
    issued = first.any(axis=2)
    has_miss = missing[mask].any(axis=2)
    hit1 = (first & miss).any(axis=2)
    fetched, hits, misses = int(pick.sum()), int((pick & miss).sum()), int(missing[mask].sum())
    tokens = max(int(mask.sum()), 1)
    return {
        "rows_per_token": fetched / tokens,
        "precision": hits / fetched if fetched else float("nan"),
        "rank1_precision": float(hit1[issued].mean()) if issued.any() else float("nan"),
        "rank1_precision_given_a_miss": float(hit1[issued & has_miss].mean()) if (issued & has_miss).any() else float("nan"),
        "recall": hits / misses if misses else float("nan"),
        "misses_per_token": misses / tokens,
    }


# ------------------------------------------------------------------ prefetch replay


class NativeGate(prefetch_sim.Predictor):
    """Layer T's gate on layer T-h's router input, cut to its top ``depth`` experts."""

    def __init__(self, rank: Rankings, depth: int) -> None:
        self.rank_, self.depth = rank, depth
        self.name = f"native_gate_d{depth}"

    def rank(self, step, target, horizon):
        if not self.rank_.valid[step, target]:
            return ()
        return self.rank_.order[step, target, : self.depth]


_RANKINGS: dict[int, Rankings] = {}


def build(arm: prefetch_baselines.Arm) -> prefetch_sim.Predictor:
    """prefetch_baselines.build, plus ``native_gate`` (``decay`` carries the ranking depth)."""
    if arm.predictor == "native_gate":
        return NativeGate(_RANKINGS[arm.horizon], int(arm.decay))
    return _BUILD(arm)


_BUILD = prefetch_baselines.build


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("router_prefix")
    p.add_argument("model_path")
    p.add_argument("--prompts", type=int, required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    p.add_argument("--depths", type=int, nargs="+", default=[6, 12, 48])
    p.add_argument("--precisions", type=float, nargs="*", default=[0.25, 0.4, 0.5, 0.6, 0.75])
    p.add_argument("--budgets", type=float, nargs="+", default=[0.85, 1.7])
    p.add_argument("--base-ms", type=float, default=116.0)
    p.add_argument("--ms-per-row", type=float, default=1.0)
    p.add_argument("--capacity", type=int, default=1128, help="total hot slots, for the Belady reference")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    started = time.time()

    loaded = tier_sim.load_forwards(args.trace)
    for forward in loaded["forwards"]:
        forward.pop("hot", None)
    stream = prefetch_sim.decode_stream(loaded)
    train, test, driver = policy_sweep.split_requests(stream.rids, args.prompts)
    capture = load_capture(args.router_prefix)
    if capture.header["layer_ids"] != stream.layers:
        raise ValueError("the router capture and the route log name different layers")
    records = decode_records(loaded, capture)
    steps, layers = len(stream.rids), len(stream.layers)
    print(f"{steps} decode steps ({int(train.sum())} train / {int(test.sum())} test), {len(capture.x)} router "
          f"records", flush=True)
    W, bias = load_gates(args.model_path, stream.layers)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    def x_of(step_index: np.ndarray, layer: int) -> np.ndarray:
        return bf16_to_f32(np.asarray(capture.x[records[step_index], layer]))

    report: dict = {"trace": args.trace, "router_prefix": args.router_prefix, "run": loaded["run"],
                    "decode_steps": steps, "train_steps": int(train.sum()), "test_steps": int(test.sum())}
    report["self_check"] = self_check(stream, x_of, W, bias, capture, records)
    print("self-check", report["self_check"], flush=True)

    depth = max(args.depths)
    for h in [0] + args.horizons:
        _RANKINGS[h] = rank_horizon(stream, x_of, W, bias, h, depth)
    del W, bias

    probe = Probe(steps, layers)
    base = prefetch_sim.run_arm(loaded, stream, probe, 1, 0)
    resident = probe.resident
    report["no_prefetch_misses"] = {"test": float(base.demand[test].sum(axis=1).mean()),
                                    "train": float(base.demand[train].sum(axis=1).mean())}
    belady = policy_sweep.run_opt(policy_sweep.decode_stream(loaded), args.capacity, bypass=True)
    report["belady_test"] = float(belady[test].mean())
    print("no-prefetch", report["no_prefetch_misses"], "belady test", report["belady_test"], flush=True)

    missing = np.zeros_like(resident)
    np.put_along_axis(missing, stream.routes, True, axis=2)
    missing &= ~resident
    if missing.sum() != base.demand.sum():
        raise AssertionError("the probe's residency does not give the replay's misses")
    offline = []
    for h in [0] + args.horizons:
        for d in args.depths:
            for k in (1, 2, 6):
                row = {"horizon": h, "depth": d, "k": k}
                for name, mask in (("test", test), ("train", train)):
                    row[name] = offline_precision(_RANKINGS[h], resident, missing, mask, k, d)
                offline.append(row)
                t = row["test"]
                print(f"  h={h} d={d:2d} k={k}: test rank1 {t['rank1_precision']:.3f} precision "
                      f"{t['precision']:.3f} recall {t['recall']:.3f} rows/tok {t['rows_per_token']:.2f}", flush=True)
    report["offline"] = offline

    arms = [prefetch_baselines.Arm("none", 0, 1)]
    for h in args.horizons:
        for k in args.ks:
            arms.append(prefetch_baselines.Arm("oracle", k, h))
            arms += [prefetch_baselines.Arm("native_gate", k, h, float(d)) for d in args.depths]
    for h in (1, 4):
        arms += [prefetch_baselines.Arm("noisy_oracle", 1, h, p) for p in args.precisions]
    arms += [prefetch_baselines.Arm("noisy_oracle", 2, 4, p) for p in args.precisions]
    arms = list(dict.fromkeys(arm for arm in arms if arm.horizon in args.horizons or arm.predictor == "none"))
    prefetch_baselines.build = build
    prefetch_baselines._STATE.update(loaded=loaded, stream=stream, train=train, test=test,
                                     all=np.ones(steps, dtype=bool), budgets=args.budgets)
    del resident, missing, probe
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = []
        for out in pool.imap_unordered(prefetch_baselines.run, arms):
            a, t = out["arm"], out["splits"]["test"]
            print(f"  {a['predictor']:>12} {a['decay'] or '':>5} k={a['k']} h={a['horizon']}  test demand "
                  f"{t['demand']:6.2f} prefetch {t['prefetch']:6.2f} useful {t['useful']:6.2f} exposed "
                  + " ".join(f"{b}:{v:6.2f}" for b, v in t["exposed"].items()) + f"  ({out['seconds']}s)",
                  flush=True)
            results.append(out)
    none = next(r for r in results if r["arm"]["predictor"] == "none")
    for r in results:
        for name, split in r["splits"].items():
            ref = none["splits"][name]["demand"]
            split["removed"] = ref - split["demand"]
            split["ms"] = {b: args.base_ms + (v - ref) * args.ms_per_row for b, v in split["exposed"].items()}
            if name == "test":
                split["gap_share"] = {b: (ref - v) / (ref - report["belady_test"]) for b, v in split["exposed"].items()}
    report["link"] = {"budgets": args.budgets, "ms_per_row": args.ms_per_row, "base_ms": args.base_ms}
    report["results"] = sorted(results, key=lambda r: (r["arm"]["horizon"], r["arm"]["predictor"], r["arm"]["decay"],
                                                       r["arm"]["k"]))
    report["seconds"] = round(time.time() - started, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
