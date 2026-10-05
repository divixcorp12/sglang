#!/usr/bin/env python3
"""DSpark verify in the decode graph, offline: the per-layer expert union of a multi-token verify, and its tok/s.

A verify of ``width`` tokens routes every token through each layer, and the layer's gather moves the union of their
experts. ``window_forwards`` turns a one-token decode trace (``tier_sim.load_forwards``) into verify forwards: each
window is ``width`` consecutive decode tokens of one request, and the next window starts ``stride`` tokens later
(the accept length). The true next tokens' routes stand in for the draft tokens' (teacher forcing). A window never
spans a request or an eager forward. Residency decays per forward, so a verify decays the insert scores as one token
(the simulator's ``graph_forward`` counts one); that is a small bias toward stickier residency.

``project`` replays the windows through ``cpu_expert_sim.replay_nm`` with CPU experts off (v1, DSV41_REFERENCE.md
§33.3), and ``baseline`` is plain decode with CPU hits on the same trace and cost model (§31.2).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cpu_expert_sim import (  # noqa: E402
    CALIBRATED_KS,
    MAX_ROUTES,
    STAGING_MERGED,
    CostModel,
    replay_nm,
    slot_map_costs,
    split_table,
)


def _rid(forward: dict) -> tuple:
    return tuple(forward.get("rids") or ())


def _merge(window: list[dict]) -> dict:
    first = window[0]
    routes = {layer: list(dict.fromkeys(e for f in window for e in f["routes"][layer])) for layer in first["routes"]}
    misses = {layer: sum(f["misses"].get(layer, 0) for f in window) for layer in first["misses"]}
    return {**first, "tokens": len(window), "routes": routes, "misses": misses}


def window_forwards(loaded: dict, width: int, stride: int) -> dict:
    """``loaded`` with every run of one request's decode forwards replaced by its verify windows."""
    if width < 1 or not 1 <= stride <= width:
        raise ValueError(f"want width >= 1 and stride in 1..width, got width {width}, stride {stride}")
    out: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        out.extend(_merge(run[start : start + width]) for start in range(0, len(run), stride))
        run.clear()

    for forward in loaded["forwards"]:
        decode = forward["kind"] == "graph" and forward["phase"] == "decode"
        if decode and run and _rid(forward) == _rid(run[0]):
            run.append(forward)
            continue
        flush()
        if decode:
            run.append(forward)
        else:
            out.append(forward)
    flush()
    return {**loaded, "forwards": out}


def union_stats(loaded: dict) -> dict:
    """Distinct experts per decode forward (a verify, once windowed) and layer."""
    sizes = np.array(
        [
            [len(set(r)) for r in f["routes"].values()]
            for f in loaded["forwards"]
            if f["kind"] == "graph" and f["phase"] == "decode"
        ],
        dtype=np.int64,
    )
    flat = sizes.reshape(-1)
    return {
        "verifies": int(sizes.shape[0]),
        "mean": float(flat.mean()),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": int(flat.max()),
        "per_layer_mean": [float(x) for x in sizes.mean(axis=0)],
    }


def shrink_hot(loaded: dict, slots: int) -> dict:
    """``loaded`` with ``slots`` fewer hot slots per layer: the VRAM a resident draft takes from the target."""
    if slots == 0:
        return loaded
    capacity = {layer: c - slots for layer, c in loaded["hot_capacity"].items()}
    if min(capacity.values()) < 1:
        raise ValueError(f"taking {slots} slots leaves a layer with {min(capacity.values())}")
    return {**loaded, "hot_capacity": capacity}


def verify_ms(n: np.ndarray, m: np.ndarray, *, c_link: float, nvme_ms: float, gpu_ms: float) -> np.ndarray:
    """ms per verify with CPU experts off: every miss crosses the link, an NVMe miss also waits for its read."""
    return ((n + m) * c_link + m * nvme_ms).sum(axis=1) + gpu_ms


def overflow_rate(n: np.ndarray, m: np.ndarray, lanes: int) -> float:
    """Share of (verify, layer) pairs whose misses exceed the record's lanes: the overflow re-verify's rate."""
    return float(((n + m) > lanes).mean())


def project(
    loaded: dict, *, width: int, stride: int, lanes: int, draft_slots: int, ram_rows: int, num_experts: int,
    c_link: float, nvme_ms: float, gpu_ms: float,
) -> dict:
    """One verify arm: windows of ``width`` advancing ``stride``, a ``lanes``-wide DIRECT shortlist, ``draft_slots``
    hot slots per layer given to the draft. CPU experts off, deferred RAM inserts (the slot-map recipe)."""
    windows = window_forwards(shrink_hot(loaded, draft_slots), width, stride)
    nm = replay_nm(windows, ram_rows, num_experts, True, "insert_all", None, ram_insert="deferred", miss_rows=lanes)
    n, m = nm["n"], nm["m"]
    per = verify_ms(n, m, c_link=c_link, nvme_ms=nvme_ms, gpu_ms=gpu_ms)
    ms = float(per.mean())
    union = union_stats(windows)
    return {
        "width": width,
        "stride": stride,
        "lanes": lanes,
        "draft_slots": draft_slots,
        "verifies": int(len(per)),
        "union": {k: union[k] for k in ("mean", "p95", "p99", "max")},
        "lanes_p99": float(np.percentile((n + m).reshape(-1), 99)),
        "overflow": overflow_rate(n, m, lanes),
        "capacity_ok": min(windows["hot_capacity"].values()) >= 2 * lanes,
        "nvme_reads_per_verify": float(m.sum() / max(len(per), 1)),
        "hot_hit_rate": nm["residency"]["hot_hit_rate"],
        "verify_ms": ms,
        "tok_s_no_draft": stride * 1000.0 / ms,
    }


def baseline(
    loaded: dict, *, ram_rows: int, num_experts: int, c_link: float, nvme_ms: float, gpu_ms: float, handoff: float,
    c_cpu: float, split_c_cpu: float,
) -> dict:
    """Plain decode with CPU hits on the same trace: slot_map_results' K = 0, protect-reads, ``hits`` arm."""
    split = split_table(MAX_ROUTES, split_c_cpu, c_link, handoff).tolist()
    nm = replay_nm(loaded, ram_rows, num_experts, True, STAGING_MERGED, split, ram_insert="deferred")
    model = CostModel([c_cpu] * len(CALIBRATED_KS), c_link=c_link, handoff=handoff, nvme_ms=nvme_ms, gpu_ms=gpu_ms)
    ms = slot_map_costs(nm["n"], nm["m"], nm["kh"], nm["km"], model)["pessimistic"]
    return {"ms_per_token": ms, "tok_s": 1000.0 / ms, "split": split}


def gate(
    rows: list[dict], base: dict, *, width: int, stride: int, draft_slots: int, gain: float, draft_ms_floor: float,
    max_overflow: float,
) -> dict:
    """GO when, at the cheapest lane count within the overflow and VRAM limits, a verify leaves at least
    ``draft_ms_floor`` ms for the draft while still beating the baseline by ``gain``."""
    picked = [r for r in rows if (r["width"], r["stride"], r["draft_slots"]) == (width, stride, draft_slots)]
    ok = [r for r in picked if r["overflow"] <= max_overflow and r["capacity_ok"]]
    if not ok:
        return {"go": False, "why": f"no lane count keeps overflow <= {max_overflow} within VRAM capacity"}
    best = min(ok, key=lambda r: r["verify_ms"])
    budget = stride * base["ms_per_token"] / gain - best["verify_ms"]
    return {
        "go": budget >= draft_ms_floor,
        "lanes": best["lanes"],
        "draft_budget_ms": budget,
        "why": f"W={best['lanes']}: {budget:.1f} ms per verify left for the draft at {gain:.2f}x the baseline "
        f"(floor {draft_ms_floor} ms)",
    }


def _pairs(text: str) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in p.split(":")) for p in text.split(",")]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


_LOADED: dict = {}


def _work(job: tuple) -> dict:
    (width, stride), lanes, slots, kw = job
    return project(_LOADED["trace"], width=width, stride=stride, lanes=lanes, draft_slots=slots, **kw)


def _init(path: str) -> None:
    import tier_sim

    _LOADED["trace"] = tier_sim.load_forwards(path)


def main() -> None:
    import argparse
    import json
    from concurrent.futures import ProcessPoolExecutor

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--out", required=True, help="JSON results path; a markdown table is printed")
    p.add_argument("--pairs", type=_pairs, default=_pairs("4:2,4:3,6:2,6:3,6:4"), help="width:stride list")
    p.add_argument("--lanes", type=_ints, default=_ints("8,16,24,32"), help="record lanes W (wire cap 32)")
    p.add_argument("--draft-slots", type=_ints, default=_ints("0,4,13"),
                   help="hot slots per layer the draft takes (hybrid 2.33 GB ~ 4, resident 6.82 GB ~ 13)")
    p.add_argument("--ram-rows", type=int, default=8063)
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--c-link", type=float, default=1.0)
    p.add_argument("--nvme-ms", type=float, default=1.5)
    p.add_argument("--gpu-ms", type=float, default=14.0, help="GPU compute per forward (verify and plain decode)")
    p.add_argument("--handoff", type=float, default=0.02)
    p.add_argument("--measured-c-cpu", type=float, default=0.63)
    p.add_argument("--split-c-cpu", type=float, default=0.52)
    p.add_argument("--gain", type=float, default=1.10)
    p.add_argument("--draft-ms-floor", type=float, default=5.0)
    p.add_argument("--max-overflow", type=float, default=0.02)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()

    _init(args.trace)
    loaded = _LOADED["trace"]
    kw = dict(ram_rows=args.ram_rows, num_experts=args.num_experts, c_link=args.c_link, nvme_ms=args.nvme_ms,
              gpu_ms=args.gpu_ms)
    base = baseline(loaded, handoff=args.handoff, c_cpu=args.measured_c_cpu, split_c_cpu=args.split_c_cpu, **kw)
    curve = {f"{w}:{s}": union_stats(window_forwards(loaded, w, s)) for w, s in args.pairs}
    jobs = [(pair, lanes, slots, kw) for pair in args.pairs for lanes in args.lanes for slots in args.draft_slots]
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.trace,)) as pool:
        rows = list(pool.map(_work, jobs))
    for row in rows:
        row["draft_budget_ms_parity"] = row["stride"] * base["ms_per_token"] - row["verify_ms"]
    verdict = gate(rows, base, width=6, stride=3, draft_slots=4, gain=args.gain, draft_ms_floor=args.draft_ms_floor,
                   max_overflow=args.max_overflow)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "baseline": base, "union_curve": curve, "rows": rows, "gate": verdict}, f,
                  indent=1, default=str)
    print(f"baseline: {base['ms_per_token']:.2f} ms/token = {base['tok_s']:.2f} tok/s, split {base['split']}")
    print("| width:stride | union mean | p95 | p99 | max |\n|---|---|---|---|---|")
    for key, u in curve.items():
        print(f"| {key} | {u['mean']:.2f} | {u['p95']:.0f} | {u['p99']:.0f} | {u['max']} |")
    print("\n| w:s | W | draft slots | overflow | cap ok | NVMe/verify | verify ms | tok/s (no draft) | draft ms at parity |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['width']}:{r['stride']} | {r['lanes']} | {r['draft_slots']} | {r['overflow']:.3f} | "
              f"{r['capacity_ok']} | {r['nvme_reads_per_verify']:.2f} | {r['verify_ms']:.1f} | "
              f"{r['tok_s_no_draft']:.2f} | {r['draft_budget_ms_parity']:.1f} |")
    print(f"\ngate: {'GO' if verdict['go'] else 'NO-GO'} -- {verdict['why']}")


if __name__ == "__main__":
    main()
