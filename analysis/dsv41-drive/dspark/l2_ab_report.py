#!/usr/bin/env python3
"""Combine the forward and reversed served L+2 A/B runs of both_cpu_ab.py into one REPORT.md and results.json.

Reads each order's summary.json and every arm's servers/<arm>/<last sorted run>/results.jsonl; stdlib only.
Usage: l2_ab_report.py FORWARD_DIR REVERSED_DIR --out DIR
"""

import argparse
import json
import os
import statistics

L2 = "dspark-both-prefetch-gpu-misscut-idle2-l2"
FLOORS = "dspark-both-prefetch-gpu-misscut-idle2-floors"
IDLE2 = "dspark-both-prefetch-gpu-misscut-idle2"
MISSCUT = "dspark-both-misscut"
ARMS = (MISSCUT, FLOORS, IDLE2, L2)
PAIRS = ((L2, FLOORS), (L2, MISSCUT), (L2, IDLE2))
ORDERS = ("forward", "reversed")

WIN_RATIO = 0.97
LOSE_RATIO = 1.03

IN_FLIGHT_FALL_FACTOR = 0.8  # the in-flight fraction counts as fallen below 80% of L+1's; arbitrary, only words the report


def verdict(ratios_vs_floors: list, paired_ms_median_vs_floors: list, ratios_lose: list) -> str:
    """Per-order lists: the step-time ratio L+2 / idle2-floors, the paired ms/token gain median (% of the floors arm,
    positive favours L+2), and the ratios that decide a loss. Win needs every order at or under WIN_RATIO with the
    paired median favouring L+2; lose needs every order at or over LOSE_RATIO; anything else is a tie."""
    if (
        ratios_vs_floors
        and len(paired_ms_median_vs_floors) == len(ratios_vs_floors)
        and all(r <= WIN_RATIO for r in ratios_vs_floors)
        and all(p > 0 for p in paired_ms_median_vs_floors)
    ):
        return "win"
    if ratios_lose and all(r >= LOSE_RATIO for r in ratios_lose):
        return "lose"
    return "tie"


def _last_run_results(order_dir: str, arm: str) -> list:
    arm_dir = os.path.join(order_dir, "servers", arm)
    runs = sorted(os.listdir(arm_dir))
    with open(os.path.join(arm_dir, runs[-1], "results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def session_figures(rows: list) -> dict:
    """{session_id: {"ms_per_token", "step_ms"}}; the step is per verify, (total - ttft) / spec_verify_ct."""
    figures = {}
    for r in rows:
        if r.get("session_id") is None:
            continue
        entry = {}
        if r.get("decode_tokens_per_sec"):
            entry["ms_per_token"] = 1000.0 / r["decode_tokens_per_sec"]
        verifies = (r.get("spec_tokens_details") or {}).get("spec_verify_ct", 0)
        if verifies and r.get("total") is not None and r.get("ttft") is not None:
            entry["step_ms"] = 1000.0 * (r["total"] - r["ttft"]) / verifies
        figures[r["session_id"]] = entry
    return figures


def _flagged(summary: dict, arm: str) -> set:
    return {f["session_id"] for f in summary[arm].get("drive_flags", [])}


def pair_table(figures: dict, flagged: dict, arm: str, other: str) -> dict:
    """Per-session rows of arm vs other, and the medians over the sessions flagged in neither."""
    rows = []
    excluded = flagged[arm] | flagged[other]
    for sid in sorted(set(figures[arm]) & set(figures[other])):
        a, b = figures[arm][sid], figures[other][sid]
        row = {"session_id": sid, "excluded": sid in excluded}
        for key in ("ms_per_token", "step_ms"):
            row[f"{key}_{arm}"] = a.get(key)
            row[f"{key}_{other}"] = b.get(key)
        if "ms_per_token" in a and "ms_per_token" in b:
            row["paired_gain_pct"] = 100.0 * (b["ms_per_token"] - a["ms_per_token"]) / b["ms_per_token"]
        if "step_ms" in a and "step_ms" in b:
            row["step_ratio"] = a["step_ms"] / b["step_ms"]
        rows.append(row)
    kept = [r for r in rows if not r["excluded"]]
    ratios = [r["step_ratio"] for r in kept if "step_ratio" in r]
    gains = [r["paired_gain_pct"] for r in kept if "paired_gain_pct" in r]
    return {
        "arm": arm,
        "other": other,
        "sessions": rows,
        "sessions_used": len(kept),
        "step_ratio_median": statistics.median(ratios) if ratios else None,
        "paired_gain_pct_median": statistics.median(gains) if gains else None,
        "sessions_faster": sum(1 for g in gains if g > 0),
        "sessions_compared": len(gains),
    }


def arm_counters(summary: dict, arm: str) -> dict:
    entry = summary[arm]
    ram = entry.get("ram") or {}
    landed, used = ram.get("spec_landed"), ram.get("spec_used")
    return {
        "spec_in_flight_at_use": entry.get("spec_in_flight_at_use"),
        "used_over_landed": (used / landed) if landed and used is not None else None,
        "spec_wasted_per_token_lifetime": entry.get("spec_wasted_per_token_lifetime"),
        "nvme_rows_per_token_lifetime": entry.get("nvme_rows_per_token_lifetime"),
        "text_vs_reference": entry.get("text_vs_reference"),
        "drive_flagged_sessions": sorted(_flagged(summary, arm)),
    }


def analyse_order(order_dir: str) -> dict:
    with open(os.path.join(order_dir, "summary.json")) as f:
        summary = json.load(f)
    present = [a for a in ARMS if a in summary]
    figures = {a: session_figures(_last_run_results(order_dir, a)) for a in present}
    flagged = {a: _flagged(summary, a) for a in present}
    return {
        "dir": order_dir,
        "pairs": {f"{a}__vs__{b}": pair_table(figures, flagged, a, b) for a, b in PAIRS if a in present and b in present},
        "counters": {a: arm_counters(summary, a) for a in present},
    }


def combine(orders: dict) -> dict:
    key = f"{L2}__vs__{FLOORS}"
    ratios, gains = [], []
    for name in ORDERS:
        pair = orders[name]["pairs"].get(key)
        ratios.append(pair["step_ratio_median"] if pair else None)
        gains.append(pair["paired_gain_pct_median"] if pair else None)
    if any(r is None for r in ratios) or any(g is None for g in gains):
        outcome = "tie"
        note = "a pair is missing in an order, so no win or loss can be read"
    else:
        outcome = verdict(ratios, gains, ratios)
        note = ""
    return {
        "verdict": outcome,
        "note": note,
        "ratios_vs_floors": dict(zip(ORDERS, ratios)),
        "paired_gain_pct_median_vs_floors": dict(zip(ORDERS, gains)),
        "in_flight": in_flight_reading(orders),
        "win_ratio": WIN_RATIO,
        "lose_ratio": LOSE_RATIO,
    }


def in_flight_reading(orders: dict) -> dict:
    """D7: the used-reads-in-flight fraction of L+2 against the L+1 arms, per order."""
    per_order = {}
    for name in ORDERS:
        counters = orders[name]["counters"]
        l2 = counters.get(L2, {}).get("spec_in_flight_at_use")
        l1 = {a: counters.get(a, {}).get("spec_in_flight_at_use") for a in (FLOORS, IDLE2)}
        known = [v for v in l1.values() if v is not None]
        per_order[name] = {
            "l2": l2,
            "l1": l1,
            "fell": (l2 < IN_FLIGHT_FALL_FACTOR * min(known)) if l2 is not None and known else None,
        }
    fell = [v["fell"] for v in per_order.values()]
    if any(f is None for f in fell):
        text = "the in-flight fraction could not be compared (a counter is missing)"
    elif all(fell):
        text = "the in-flight fraction FELL from L+1 to L+2 in both orders, as the offline replay predicted"
    else:
        text = (
            "the in-flight fraction did NOT fall from L+1 to L+2 in both orders: "
            "the replay's timing model does not hold for serving"
        )
    return {"per_order": per_order, "text": text}


def _fmt(v, spec: str = ".3f") -> str:
    return "n/a" if v is None else format(v, spec)


def _pair_section(name: str, pair: dict) -> list:
    arm, other = pair["arm"], pair["other"]
    lines = [
        f"#### {name}: {arm} vs {other}",
        "",
        f"| session | ms/token {other} | ms/token {arm} | step ms {other} | step ms {arm} | step ratio | paired % | used |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in pair["sessions"]:
        lines.append(
            f"| {r['session_id']} | {_fmt(r.get('ms_per_token_' + other), '.2f')} | {_fmt(r.get('ms_per_token_' + arm), '.2f')} "
            f"| {_fmt(r.get('step_ms_' + other), '.1f')} | {_fmt(r.get('step_ms_' + arm), '.1f')} "
            f"| {_fmt(r.get('step_ratio'))} | {_fmt(r.get('paired_gain_pct'), '+.1f')} | {'no (drive flag)' if r['excluded'] else 'yes'} |"
        )
    lines += [
        "",
        f"Median step ratio {_fmt(pair['step_ratio_median'])} over {pair['sessions_used']} sessions; paired ms/token "
        f"median {_fmt(pair['paired_gain_pct_median'], '+.1f')}%; {pair['sessions_faster']} of "
        f"{pair['sessions_compared']} sessions faster.",
        "",
    ]
    return lines


def _counter_table(order: dict) -> list:
    lines = [
        "| arm | in flight at use | used / landed | wasted rows/token | NVMe rows/token | text vs reference | drive flags |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, c in order["counters"].items():
        lines.append(
            f"| {arm} | {_fmt(c['spec_in_flight_at_use'])} | {_fmt(c['used_over_landed'])} "
            f"| {_fmt(c['spec_wasted_per_token_lifetime'])} | {_fmt(c['nvme_rows_per_token_lifetime'])} "
            f"| {c['text_vs_reference'] if c['text_vs_reference'] is not None else 'n/a'} "
            f"| {', '.join(str(s) for s in c['drive_flagged_sessions']) or 'none'} |"
        )
    return lines + [""]


def render(orders: dict, combined: dict) -> str:
    lines = ["# Served L+2 A/B", "", f"## Verdict: {combined['verdict']}", ""]
    lines += [
        f"Pre-registered (D6): L+2 wins when both orders give a median step-time ratio <= {WIN_RATIO} against "
        f"idle2-floors and the paired ms/token median favours L+2 in both; it loses when both give >= {LOSE_RATIO}; "
        "otherwise tie.",
        "",
        "| order | step ratio L+2 / idle2-floors | paired ms/token median % |",
        "|---|---|---|",
    ]
    for name in ORDERS:
        lines.append(
            f"| {name} | {_fmt(combined['ratios_vs_floors'][name])} "
            f"| {_fmt(combined['paired_gain_pct_median_vs_floors'][name], '+.1f')} |"
        )
    if combined["note"]:
        lines += ["", combined["note"]]
    lines += ["", "## Mechanism (D7)", "", combined["in_flight"]["text"] + ".", ""]
    lines += ["| order | L+2 in flight | idle2-floors | idle2 |", "|---|---|---|---|"]
    for name in ORDERS:
        r = combined["in_flight"]["per_order"][name]
        lines.append(f"| {name} | {_fmt(r['l2'])} | {_fmt(r['l1'][FLOORS])} | {_fmt(r['l1'][IDLE2])} |")
    for name in ORDERS:
        lines += ["", f"## {name} order", "", "### Counters", ""] + _counter_table(orders[name])
        lines += ["### Pairs", ""]
        for pair_name, pair in orders[name]["pairs"].items():
            lines += _pair_section(pair_name, pair)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("forward_dir")
    p.add_argument("reversed_dir")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    orders = {"forward": analyse_order(a.forward_dir), "reversed": analyse_order(a.reversed_dir)}
    combined = combine(orders)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"combined": combined, "orders": orders}, f, indent=2)
    with open(os.path.join(a.out, "REPORT.md"), "w") as f:
        f.write(render(orders, combined))
    print(f"verdict: {combined['verdict']}; {combined['in_flight']['text']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
