#!/usr/bin/env python3
"""Upper bounds on what trimming the RAM-miss frontend can save, from a NODE-mode decode trace (sqlite export).

Per layer (a post kernel starts one): the frontend span post.end -> S.start, W1's duration and CW's. The DMA is
started by the host after post, so a layer's saving from a cut before CW is at most cut - CW's spin past its floor.
S's own NVMe wait can hide more, so these are upper bounds. Node mode inflates small kernels: never read ms/token.

    python3 frontend_bound.py <trace.sqlite> [--skip 20] [--budget-us 100] [--json out.json]
"""

import argparse
import bisect
import collections
import json
import sqlite3
import statistics

import msgspec

CHAIN = {
    "exl3_ram_miss_post_kernel": "post",
    "exl3_ram_miss_lease_stream_hit_wait_kernel": "W1",
    "exl3_ram_miss_lease_stream_reset_kernel": "R",
    "copy_expert_row_segments_gpu_kernel": "C1",
    "exl3_ram_miss_lease_stage_ack_kernel": "A",
    "exl3_ram_miss_lease_stream_kernel": "S",
    "exl3_ram_miss_lease_copy_wait_kernel": "CW",
    "exl3_ram_miss_lease_finalize_kernel": "F",
}


class Layer(msgspec.Struct, frozen=True):
    post_start: int
    post_end: int
    s_start: int
    w1_ns: int  # 0 when the chain has no W1 (the reset chain)
    cw_ns: int
    cw_end: int
    f_end: int


def split_layers(kernels: list[tuple[int, int, str]]) -> list[Layer]:
    layers, cur = [], None
    for start, end, label in kernels:
        if label == "post":
            if cur is not None and {"S", "F"} <= cur.keys():
                layers.append(_layer(cur))
            cur = {"post": (start, end)}
        elif cur is not None and label not in cur:
            cur[label] = (start, end)  # first occurrence: the first A is A1
    if cur is not None and {"S", "F"} <= cur.keys():
        layers.append(_layer(cur))
    return layers


def _layer(k: dict) -> Layer:
    w1 = k.get("W1", (0, 0))
    cw = k.get("CW", (0, 0))
    return Layer(post_start=k["post"][0], post_end=k["post"][1], s_start=k["S"][0], w1_ns=w1[1] - w1[0],
                 cw_ns=cw[1] - cw[0], cw_end=cw[1], f_end=k["F"][1])


def copy_metrics(steps: list[list[Layer]], copies: list[tuple[int, int, int]]) -> dict:
    """A layer's copies are the H2D copies that start between its post and its CW's end: CW waits for all of them."""
    starts = [c[0] for c in copies]
    done_after_post, after_copy = [], []
    total_bytes = busy = 0
    for layer in (l for step in steps for l in step):
        mine = copies[bisect.bisect_left(starts, layer.post_start) : bisect.bisect_right(starts, layer.cw_end)]
        if not mine:
            continue
        last = max(e for _, e, _ in mine)
        total_bytes += sum(b for _, _, b in mine)
        busy += sum(e - s for s, e, _ in mine)
        done_after_post.append(last - layer.post_end)
        after_copy.append(layer.cw_end - last)
    n = len(steps)
    return {
        "copy_bytes_per_step": total_bytes / n,
        "copy_busy_ms_per_step": busy / n / 1e6,
        "copy_done_after_post_p50_us": statistics.median(done_after_post) / 1e3 if done_after_post else 0.0,
        "cw_end_after_copy_p50_us": statistics.median(after_copy) / 1e3 if after_copy else 0.0,
    }


def check_steps(steps: list[list[Layer]]) -> None:
    if not steps:
        raise ValueError("no graph chain kernels: not a node-mode trace, or --skip covers every step")
    counts = {len(step) for step in steps}
    if len(counts) != 1:
        raise ValueError(f"layers per step differ ({sorted(counts)}): a torn chain would understate the bound")


def saving_bound_ns(cut_ns: int, cw_ns: int, cw_floor_ns: int) -> int:
    return max(0, cut_ns - max(0, cw_ns - cw_floor_ns))


def _pct(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(steps: list[list[Layer]], budget_ns: int) -> dict:
    layers = [layer for step in steps for layer in step]
    cw_floor = _pct([l.cw_ns for l in layers], 0.05)
    w1s = [l.w1_ns for l in layers if l.w1_ns > 0]
    w1_floor = _pct(w1s, 0.10) if w1s else 0
    n = len(steps)

    def per_step_ms(f) -> float:
        return sum(f(l) for l in layers) / n / 1e6

    return {
        "steps": n,
        "layers_per_step": len(layers) // n,
        "frontend_ms_per_step": per_step_ms(lambda l: l.s_start - l.post_end),
        "frontend_bound_ms_per_step": per_step_ms(
            lambda l: saving_bound_ns(l.s_start - l.post_end, l.cw_ns, cw_floor)),
        "w1_ms_per_step": per_step_ms(lambda l: l.w1_ns),
        "w1_p50_us": statistics.median(w1s) / 1e3 if w1s else 0.0,
        "w1_p90_us": _pct(w1s, 0.90) / 1e3 if w1s else 0.0,
        "w1_budget_hit_frac": sum(w >= 0.9 * budget_ns for w in w1s) / len(w1s) if w1s else 0.0,
        "hit_wait0_bound_ms_per_step": per_step_ms(
            lambda l: saving_bound_ns(max(0, l.w1_ns - w1_floor), l.cw_ns, cw_floor) if l.w1_ns else 0),
        "cw_floor_us": cw_floor / 1e3,
        "cw_spin_ms_per_step": per_step_ms(lambda l: max(0, l.cw_ns - cw_floor)),
        "post_to_f_p50_us": statistics.median(l.f_end - l.post_end for l in layers) / 1e3,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--skip", type=int, default=20, help="leading graph steps to drop (capture warm-up, unarmed)")
    ap.add_argument("--budget-us", type=int, default=100)
    ap.add_argument("--json")
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    names = dict(c.execute("select id, value from StringIds"))
    by_step = collections.defaultdict(list)
    for start, end, corr, name in c.execute(
        "select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL where graphId != 0 order by start"
    ):
        label = CHAIN.get(names[name])
        if label is not None:
            by_step[corr].append((start, end, label))
    ordered = sorted(by_step.values(), key=lambda k: k[0][0])[a.skip :]
    steps = [split_layers(k) for k in ordered]
    check_steps(steps)
    result = summarize(steps, budget_ns=a.budget_us * 1000)
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        # The copy engine's copies: host to device (copyKind 1) and outside the graph. nsys 2026.3's export has no
        # graphId on memcpy rows, only graphNodeId (NULL outside a graph); see ce_trace.py.
        cols = {r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
        outside = "graphId = 0" if "graphId" in cols else "coalesce(graphNodeId, 0) = 0"
        copies = c.execute(
            f"select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 1 and {outside} order by start"
        ).fetchall()
        result.update(copy_metrics(steps, copies))
    print(json.dumps(result, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
