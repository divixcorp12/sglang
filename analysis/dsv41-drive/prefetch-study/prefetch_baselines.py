"""E1 route-only prefetch baselines on a graph route capture: misses removed, link rows, net ms/token.

Every arm replays the whole run through prefetch_sim.PrefetchReplay (stage DIRECT insert-on-miss, the
running recipe) with one predictor, ``k`` rows per layer and horizon ``h``, and is priced by
prefetch_sim.link_exposed at each idle-link budget. Anything fitted (popularity, cross-layer counts) uses
training steps only (even driver prompts); scores are reported on held-out steps (odd prompts) and train.

Per arm and split: ``demand`` (VRAM misses/token left), ``prefetch`` (prefetched rows/token), ``useful``
(prefetched rows the same forward routed), ``G_total`` = demand + prefetch (link rows/token), and per
budget ``exposed`` (link rows/token on the critical path) and ``ms`` = base_ms + (exposed - the no-prefetch
arm's exposed) * ms_per_row. ``gap_share`` = (no-prefetch misses - exposed) / (no-prefetch misses - the
Belady-with-bypass misses given on the command line).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
sys.path.insert(0, os.path.join(HERE, "..", "hot-cache-policy"))
import prefetch_sim  # noqa: E402
import tier_sim  # noqa: E402
from policy_sweep import split_requests  # noqa: E402


class Arm(msgspec.Struct, frozen=True):
    predictor: str
    k: int
    horizon: int
    decay: float = 0.0  # recency: decay per token; cross_layer: min_prob gate; noisy_oracle: precision


_STATE: dict = {}


def build(arm: Arm) -> prefetch_sim.Predictor:
    stream, train = _STATE["stream"], _STATE["train"]
    if arm.predictor == "none":
        return prefetch_sim.Predictor()
    if arm.predictor == "oracle":
        return prefetch_sim.Oracle(stream)
    if arm.predictor == "prev_token":
        return prefetch_sim.PreviousToken(stream)
    if arm.predictor == "popularity":
        return prefetch_sim.Popularity(stream, train)
    if arm.predictor == "recency":
        return prefetch_sim.Recency(stream, arm.decay)
    if arm.predictor == "cross_layer":
        return prefetch_sim.CrossLayer(stream, train, arm.horizon, min_prob=arm.decay)
    if arm.predictor == "noisy_oracle":
        return prefetch_sim.NoisyOracle(stream, arm.decay, arm.k)
    raise ValueError(arm.predictor)


def run(arm: Arm) -> dict:
    started = time.time()
    loaded, stream = _STATE["loaded"], _STATE["stream"]
    result = prefetch_sim.run_arm(loaded, stream, build(arm), arm.k, arm.horizon)
    out = {"arm": msgspec.to_builtins(arm), "seconds": None, "splits": {}}
    exposed = {
        budget: prefetch_sim.link_exposed(
            result.demand, result.prefetch, stream.continuous, max(arm.horizon, 1), prefetch_sim.LinkModel(budget)
        )
        for budget in _STATE["budgets"]
    }
    for name, mask in (("test", _STATE["test"]), ("train", _STATE["train"]), ("all", _STATE["all"])):
        demand = result.demand[mask].sum(axis=1)
        prefetch = result.prefetch[mask].sum(axis=1)
        useful = result.useful[mask].sum(axis=1)
        split = {
            "demand": float(demand.mean()),
            "prefetch": float(prefetch.mean()),
            "useful": float(useful.mean()),
            "G_total": float((demand + prefetch).mean()),
            "exposed": {str(b): float(exposed[b][mask].mean()) for b in _STATE["budgets"]},
        }
        if name == "test":
            split["demand_per_layer"] = result.demand[mask].mean(axis=0).round(3).tolist()
            split["prefetch_per_layer"] = result.prefetch[mask].mean(axis=0).round(3).tolist()
            split["useful_per_layer"] = result.useful[mask].mean(axis=0).round(3).tolist()
        out["splits"][name] = split
    out["seconds"] = round(time.time() - started, 1)
    return out


def arms(ks, horizons, decays, gates, precisions, only) -> list[Arm]:
    out = [Arm("none", 0, 1)]
    for h in horizons:
        for k in ks:
            out += [Arm("oracle", k, h), Arm("prev_token", k, h), Arm("popularity", k, h), Arm("cross_layer", k, h)]
            out += [Arm("recency", k, h, decay) for decay in decays]
            out += [Arm("cross_layer", k, h, gate) for gate in gates]
            out += [Arm("noisy_oracle", k, h, precision) for precision in precisions]
    if only:
        out = [arm for arm in out if arm.predictor == "none" or arm.predictor in only]
    return list(dict.fromkeys(out))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--prompts", type=int, required=True, help="driver requests at the end of the run")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 6])
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--decays", type=float, nargs="+", default=[0.5, 0.9, 0.98])
    p.add_argument("--gates", type=float, nargs="*", default=[], help="cross_layer min_prob gates")
    p.add_argument("--precisions", type=float, nargs="*", default=[], help="noisy_oracle precisions")
    p.add_argument("--only", nargs="*", default=[], help="run only these predictors (plus the baseline)")
    p.add_argument("--budgets", type=float, nargs="+", default=[0.85, 1.7])
    p.add_argument("--base-ms", type=float, default=116.0, help="measured decode ms/token of the recipe")
    p.add_argument("--ms-per-row", type=float, default=1.0)
    p.add_argument("--belady-test", type=float, required=True, help="Belady-with-bypass held-out misses/token")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    loaded = tier_sim.load_forwards(args.trace)
    for forward in loaded["forwards"]:
        forward.pop("hot", None)  # replay needs no logged hot sets; keeps the forked workers small
    stream = prefetch_sim.decode_stream(loaded)
    train, test, driver = split_requests(stream.rids, args.prompts)
    _STATE.update(loaded=loaded, stream=stream, train=train, test=test, all=np.ones(len(train), dtype=bool),
                  budgets=args.budgets)
    todo = arms(args.ks, args.horizons, args.decays, args.gates, args.precisions, args.only)
    print(f"{len(stream.rids)} decode steps; {int(train.sum())} train / {int(test.sum())} test; {len(todo)} arms",
          flush=True)
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = []
        for out in pool.imap_unordered(run, todo):
            a, t = out["arm"], out["splits"]["test"]
            print(f"  {a['predictor']:>12} {a['decay'] or '':>5} k={a['k']} h={a['horizon']}  test demand "
                  f"{t['demand']:6.2f} prefetch {t['prefetch']:6.2f} useful {t['useful']:6.2f} exposed "
                  + " ".join(f"{b}:{v:6.2f}" for b, v in t["exposed"].items()) + f"  ({out['seconds']}s)",
                  flush=True)
            results.append(out)
    base = next(r for r in results if r["arm"]["predictor"] == "none")
    for r in results:
        for name, split in r["splits"].items():
            ref = base["splits"][name]["demand"]
            split["removed"] = ref - split["demand"]
            split["ms"] = {b: args.base_ms + (v - ref) * args.ms_per_row for b, v in split["exposed"].items()}
            if name == "test":
                split["gap_share"] = {b: (ref - v) / (ref - args.belady_test) for b, v in split["exposed"].items()}
    report = {
        "trace": args.trace,
        "run": loaded["run"],
        "decode_steps": len(stream.rids),
        "train_steps": int(train.sum()),
        "test_steps": int(test.sum()),
        "driver_requests": len(driver),
        "link": {"budgets": args.budgets, "ms_per_row": args.ms_per_row, "base_ms": args.base_ms},
        "belady_test": args.belady_test,
        "results": sorted(results, key=lambda r: (r["arm"]["horizon"], r["arm"]["predictor"], r["arm"]["decay"],
                                                  r["arm"]["k"])),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
