#!/usr/bin/env python3
"""Gate for the early post: replay the h=1 K=1 native-gate prefetch on the measured per-layer link timeline.

The native prefetch posts T's prediction after layer T-1's demand chain, and only ~361 us of compute then separates
the post from T's gather (``2026-09-25-dsv41-native-prefetch.md``, "Where the time went"). The proposal is to post at
T-1's router instead and have the service hold the copy until T-1's copy-engine job is done, so that the copy uses the
link while T-1 waits on NVMe. This replays that on the stage records of a prefetch-off smoke arm (A).

**The link timeline, per graph decode step and layer L**, from the arm's ``ram_miss_request`` records (one demand or
touch record per layer per forward):

- ``observed(L)``: the service sees L's post. The commit of a prefetch aimed at L runs just before it, so this is
  L's deadline ``G(L)``: a prefetch not complete by then is waited for.
- the copy-engine job: ``lanes - rows_asked`` hit lanes, whose link work starts at ``reserved(L) + issue`` and costs
  ``--lane-ms`` per lane (fitted below: the step's summed copy latency is ``0.15 ms/job + 1.00 ms/lane``, and a
  no-read layer's observed-to-observed period is ``0.39 ms + 0.98 ms/lane``, so the copy engine alone sets it).
- S's piece copies: each NVMe extent completion (``extent_cqe_ns``) releases its piece, whose link work is its bytes
  at the same rate as a lane.

The link serves this demand work in arrival order at one lane per ``--lane-ms``. A prefetch aimed at T is released at
``max(observed(T-1), CE_end(T-1))`` (early) or at ``observed(T) - --late-window-ms`` (late, B's measured placement,
used to calibrate the replay against B's measured +2.6 ms/token). It needs ``--prefetch-ms`` of link time.

- ``priority``: the prefetch gets only link time no demand work wants (the proposal's rule; optimistic, because a DMA
  in flight is not pre-empted by S's SM copies).
- ``share``: while demand work is queued, the prefetch and demand split the link evenly. The delay this adds to T-1's
  chain end (``max(CE_end, S_end)``) is charged too.

**Charges.** ``exposed(T) = max(0, finish - G(T))``; after ``G(T)`` the link is free (T's post waits for the commit and
T-1's demand is done), so the rest runs at full rate. A used prefetch removes one hit lane from T's copy-engine job:
its saving is ``max(CE_end, S_end) - max(CE_end - lane, S_end)`` on T's own timeline, i.e. a full lane only when the
copy engine, not NVMe, is T's critical path.

**Predictions.** The smoke has no hidden states, so the native gate is replayed by its measured rates: B's counters
give prefetches issued and used per graph step. Issued prefetches are spread uniformly over targets 1..39, and used
ones over the targets with at least one hit lane (a used row is by definition a RAM-ready VRAM miss of T, i.e. a hit
lane). Estimate per token: ``base + issued/39 * sum(exposed + share penalty) - used * mean saving over eligible``.

**Calibration.** The ``late`` arm is B's own placement, measured at +2.6 ms/token. The model's error there is carried
to the early arms as a constant (``calibrated_*``). The gate passes only if both the raw and the calibrated early
estimate clear -8 ms/token.

    window_replay.py A_DIR B_DIR [--base-ms 120.6] [--b-measured-delta 2.6] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import msgspec
import numpy as np

LAYERS = 40
LANE_BYTES = 13315584.0  # one expert row


def load_steps(directory: str) -> tuple[list[list[dict]], dict]:
    """Graph decode steps after the copy engine armed, each the 40 layers' records in post order, plus the
    graph_step lines. Steps are cut by post order (a new step at each layer-0 record), not by the records'
    ``forward`` label, which sometimes names the previous forward on layer 0. Prefill forwards are dropped."""
    records: list[dict] = []
    graph_steps: dict[int, dict] = {}
    extend: set[int] = set()
    for line in open(os.path.join(directory, "stages.jsonl")):
        record = json.loads(line)
        kind = record.get("kind")
        if kind == "ram_miss_request":
            records.append(record)
        elif kind == "graph_step" and record.get("thread"):
            graph_steps[record["forward"]] = record
        elif kind is None and record.get("phase") == "extend":
            extend.add(record["forward"])
    armed = min(g["t"] for g in graph_steps.values() if g["thread"].get("copy_jobs", 0))
    records.sort(key=lambda r: r["stages_ns"]["observed"])
    groups: list[list[dict]] = []
    for record in records:
        if record["layer"] == 0 or not groups:
            groups.append([])
        groups[-1].append(record)
    steps = []
    for group in groups:
        if [r["layer"] for r in group] != list(range(LAYERS)):
            continue
        if group[0]["stages_ns"]["observed"] / 1e9 < armed:
            continue
        if any(r["forward"] in extend for r in group[1:]):
            continue
        steps.append(group)
    return steps, graph_steps


def counters_per_step(directory: str) -> dict:
    _, graph_steps = load_steps(directory)
    live = [graph_steps[f]["thread"] for f in sorted(graph_steps) if graph_steps[f]["thread"].get("copy_jobs", 0)]
    first, last = live[0], live[-1]
    n = len(live) - 1
    delta = lambda k: (last.get(k, 0) - first.get(k, 0)) / n
    return {"steps": n, "issued": delta("prefetch_issued"), "used": delta("prefetch_used"),
            "wasted": delta("prefetch_wasted"), "latency_us": (last["prefetch_latency_ns"] - first["prefetch_latency_ns"])
            / max(1, last["prefetch_issued"] - first["prefetch_issued"]) / 1e3}


class Timeline(msgspec.Struct):
    """One step's demand-only link timeline (ms from layer 0's post)."""

    obs: list[float]
    ce_end: list[float]
    s_end: list[float]
    hits: list[int]
    busy: list[tuple[float, float]]


def timeline(records: list[dict], lane_ms: float, issue_ms: float) -> Timeline:
    """Serve every layer's copy-engine job and S pieces on one link, first come first served, one lane per
    ``lane_ms``."""
    t0 = records[0]["stages_ns"]["observed"]
    obs, hits, items = [], [], []
    for li, record in enumerate(records):
        stages = record["stages_ns"]
        obs.append((stages["observed"] - t0) / 1e6)
        h = max(0, record["request"]["lanes"] - record["rows_asked"])
        hits.append(h)
        if h:
            start = (stages["reserved"] or stages["observed"]) - t0
            items.append((start / 1e6 + issue_ms, h * lane_ms, li, 0))
        extents = record.get("extent_cqe_ns") or []
        if extents:
            piece = record["bytes"] / len(extents) / LANE_BYTES * lane_ms
            items.extend(((e["cqe"] - t0) / 1e6, piece, li, 1) for e in extents)
    items.sort()
    ce_end, s_end = list(obs), list(obs)
    busy: list[list[float]] = []
    t_free = -1e18
    for arrival, work, li, kind in items:
        start = max(arrival, t_free)
        t_free = start + work
        if kind == 0:
            ce_end[li] = max(ce_end[li], t_free)
        else:
            s_end[li] = max(s_end[li], t_free)
        if busy and start <= busy[-1][1] + 1e-12:
            busy[-1][1] = t_free
        else:
            busy.append([start, t_free])
    return Timeline(obs=obs, ce_end=ce_end, s_end=s_end, hits=hits, busy=[(a, b) for a, b in busy])


def segments(busy, start: float, end: float):
    """(t0, t1, is_busy) pieces covering [start, end)."""
    t = start
    for a, b in busy:
        if b <= t:
            continue
        if a >= end:
            break
        if a > t:
            yield t, min(a, end), False
        lo, hi = max(a, t), min(b, end)
        if hi > lo:
            yield lo, hi, True
        t = max(t, hi)
        if t >= end:
            return
    if t < end:
        yield t, end, False


def run_prefetch(tl: Timeline, release: float, deadline: float, work: float, mode: str, chain_end: float):
    """Link time the prefetch gets in [release, deadline). ``priority``: idle link only. ``share``: all of the idle
    link and half of the busy link; the busy share it takes before ``chain_end`` delays T-1's chain by as much.
    Returns (exposed ms at the deadline, chain delay ms)."""
    left, delay = work, 0.0
    for a, b, is_busy in segments(tl.busy, release, deadline):
        if left <= 0:
            break
        if is_busy:
            if mode != "share":
                continue
            take = min(left, (b - a) / 2)
            left -= take
            if a < chain_end:
                delay += min(take, max(0.0, (chain_end - a) / 2))
        else:
            left -= min(left, b - a)
    return max(0.0, left), delay


def idle_between(tl: Timeline, start: float, end: float) -> float:
    return sum(b - a for a, b, is_busy in segments(tl.busy, start, end) if not is_busy)


def replay(timelines, rates, lane_ms, prefetch_ms, variant, mode, late_window_ms):
    exposed_cells, penalty_cells, idle_cells, savings = [], [], [], []
    for tl in timelines:
        for target in range(1, LAYERS):
            prev = target - 1
            deadline = tl.obs[target]
            if variant == "early":
                release = max(tl.obs[prev], tl.ce_end[prev])
            else:
                release = deadline - late_window_ms
            chain_end = max(tl.ce_end[prev], tl.s_end[prev])
            exposed, delay = run_prefetch(tl, release, deadline, prefetch_ms, mode, chain_end)
            exposed_cells.append(exposed)
            penalty_cells.append(delay)
            idle_cells.append(idle_between(tl, tl.obs[prev], deadline))
            if tl.hits[target] >= 1:
                old = max(tl.ce_end[target], tl.s_end[target])
                new = max(tl.ce_end[target] - lane_ms, tl.s_end[target], tl.obs[target])
                savings.append(old - new)
    per_cell_issue = rates["issued"] / (LAYERS - 1)
    cells_per_step = LAYERS - 1
    ec, pc, iw = np.array(exposed_cells), np.array(penalty_cells), np.array(idle_cells)
    exposed_per_token = per_cell_issue * cells_per_step * float(ec.mean())
    penalty_per_token = per_cell_issue * cells_per_step * float(pc.mean())
    mean_saving = float(np.mean(savings))
    saving_per_token = rates["used"] * mean_saving
    return {
        "variant": variant, "mode": mode, "steps": len(timelines),
        "idle_link_ms_from_T-1_post_to_T_gather": {
            "mean": round(float(iw.mean()), 3), "p10": round(float(np.percentile(iw, 10)), 3),
            "p50": round(float(np.median(iw)), 3), "p90": round(float(np.percentile(iw, 90)), 3),
            "share_ge_one_copy": round(float((iw >= prefetch_ms).mean()), 3)},
        "exposed_ms_per_prefetch": round(float(ec.mean()), 3),
        "share_fully_hidden": round(float((ec <= 1e-9).mean()), 3),
        "exposed_ms_per_token": round(exposed_per_token, 2),
        "chain_delay_ms_per_token": round(penalty_per_token, 2),
        "saving_ms_per_used_row": round(mean_saving, 3),
        "saving_ms_per_token": round(saving_per_token, 2),
        "delta_ms_per_token": round(exposed_per_token + penalty_per_token - saving_per_token, 2),
    }


def model_check(timelines, lane_ms):
    """observed(L+1) - chain end(L) on the demand-only timeline: the compute after L's chain (MoE, next attention,
    router). It should not depend on L's lanes or reads if the link model is right."""
    by = collections.defaultdict(list)
    for tl in timelines:
        for li in range(LAYERS - 1):
            end = max(tl.ce_end[li], tl.s_end[li], tl.obs[li])
            by[(min(tl.hits[li], 4), 0 if tl.s_end[li] == tl.obs[li] else 1)].append(tl.obs[li + 1] - end)
    return {f"hits{h}_nvme{r}": {"n": len(v), "p10": round(float(np.percentile(v, 10)), 3),
                                  "p50": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3)}
            for (h, r), v in sorted(by.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a_dir")
    ap.add_argument("b_dir")
    ap.add_argument("--base-ms", type=float, default=120.6)
    ap.add_argument("--lane-ms", type=float, default=0.98)
    ap.add_argument("--issue-ms", type=float, default=0.033)
    ap.add_argument("--prefetch-ms", type=float, default=0.988, help="B's measured read-to-completion per copy")
    ap.add_argument("--late-window-ms", type=float, default=0.361)
    ap.add_argument("--b-measured-delta", type=float, default=2.6, help="B - A measured, ms/token (late post)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    steps, _ = load_steps(a.a_dir)
    if a.max_steps:
        steps = steps[: a.max_steps]
    timelines = [timeline(records, a.lane_ms, a.issue_ms) for records in steps]
    rates = counters_per_step(a.b_dir)
    out = {"a": a.a_dir, "b": a.b_dir, "base_ms": a.base_ms, "lane_ms": a.lane_ms, "prefetch_ms": a.prefetch_ms,
           "rates_from_b": {k: round(v, 3) if isinstance(v, float) else v for k, v in rates.items()},
           "model_check": model_check(timelines, a.lane_ms), "arms": []}
    for variant, mode in (("late", "priority"), ("early", "priority"), ("early", "share")):
        r = replay(timelines, rates, a.lane_ms, a.prefetch_ms, variant, mode, a.late_window_ms)
        r["est_ms_per_token"] = round(a.base_ms + r["delta_ms_per_token"], 2)
        out["arms"].append(r)
    # Calibration: the late arm is B's placement, measured at +2.6 ms/token. The model's error there (mostly the
    # saving per used row, which B implies at ~0.74 ms) is carried to the early arms as a constant correction.
    correction = a.b_measured_delta - out["arms"][0]["delta_ms_per_token"]
    for r in out["arms"]:
        r["calibrated_delta_ms_per_token"] = round(r["delta_ms_per_token"] + correction, 2)
        r["calibrated_est_ms_per_token"] = round(a.base_ms + r["calibrated_delta_ms_per_token"], 2)
        r["clears_8ms"] = r["calibrated_delta_ms_per_token"] <= -8.0 and r["delta_ms_per_token"] <= -8.0
        print(json.dumps(r), flush=True)
    print(json.dumps(out["model_check"]))
    print("rates", out["rates_from_b"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
