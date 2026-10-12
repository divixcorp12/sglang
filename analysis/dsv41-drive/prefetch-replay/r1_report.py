"""Render the R1 run's results.json as REPORT.md, under the exploratory label.

Inputs: results.json with keys label, plan_sha, stage1 ({split: {scorer: {capN: cell}},
histograms: {scorer: {group: {...}}}}), timing ({assumptions, cases: {case: {scorer: cell}}}),
stage2_gate, arms, selection, calibration, cost, l2, continuation_gate, attribution,
stopped_after_stage1. A missing value renders as `pending`.
Usage: python3 r1_report.py RESULTS.json --out REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PENDING = "pending"
LABEL = "exploratory on reused data; independent confirmation pending"
EXPLANATIONS = ("ranking", "abstention", "block_aggregation", "capacity", "training_length", "lead")
CELL_KEYS = ("useful", "wasted", "eligible", "precision", "recall", "reads")
HIST_KEYS = ("0", "1", "2", "3", "4+", "p50", "p90", "p95", "p99", "max")
TIMING_KEYS = ("on_time_useful", "late_useful", "wasted", "recall", "reads")
# Candidate and baseline per explanation; names are arms (results["arms"]) or Stage-1 controls.
EXPLANATION_PAIRS = {
    "ranking": ("F0", "gate_floors"),
    "abstention": ("hybrid", "gate_floors"),
    "block_aggregation": ("T1", "T0"),
    "capacity": ("T2", "T1"),
    "training_length": ("T0", "F0"),
}
RELATIVE_BAR = 0.10  # same relative recall bar as the continuation gate (D10)


def _num(v, *, digits: int = 3) -> str:
    if v is None:
        return PENDING
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header))] + [_row(r) for r in rows] + [""]


def _kv_table(d: dict | None) -> list[str]:
    d = d or {}
    return _table(["item", "value"], [[k, _num(d[k])] for k in sorted(d)])


def _stage1_section(stage1: dict) -> list[str]:
    out = ["## Stage 1 (no training): capped policies at mean reads <= 1 per forward-layer", ""]
    rows = []
    for split in sorted(k for k in stage1 if k != "histograms"):
        for scorer in sorted(stage1[split]):
            for cap in sorted(stage1[split][scorer]):
                cell = stage1[split][scorer][cap]
                rows.append([split, scorer, cap] + [_num(cell.get(k)) for k in CELL_KEYS])
    if not rows:
        return out + [f"Stage 1 tables: {PENDING}", ""]
    return out + _table(["split", "scorer", "cap"] + list(CELL_KEYS), rows)


def _histogram_section(hists: dict) -> list[str]:
    out = ["## Reads per forward-layer before capping", ""]
    if not hists:
        return out + [f"Histograms: {PENDING}", ""]
    rows = [[scorer, group] + [_num(hists[scorer][group].get(k)) for k in HIST_KEYS]
            for scorer in sorted(hists) for group in sorted(hists[scorer])]
    return out + _table(["scorer", "group"] + list(HIST_KEYS), rows)


def _timing_section(timing: dict) -> list[str]:
    out = ["## Timing sensitivity replay", ""]
    assumptions = timing.get("assumptions") or {}
    cases = timing.get("cases") or {}
    if assumptions:
        out += ["Assumptions (bounded sensitivity model; scorer delay included):", ""]
        out += _kv_table({k: v for k, v in assumptions.items() if not isinstance(v, dict)})
        for k in sorted(v for v in assumptions if isinstance(assumptions[v], dict)):
            out += _kv_table({f"{k}.{kk}": vv for kk, vv in assumptions[k].items()})
    if not cases:
        return out + [f"Timing cases: {PENDING}", ""]
    rows = [[case, scorer] + [_num(cases[case][scorer].get(k)) for k in TIMING_KEYS]
            for case in sorted(cases) for scorer in sorted(cases[case])]
    out += ["Only on-time useful rows are priced, not every useful row at the last-miss slope.", ""]
    return out + _table(["case", "scorer"] + list(TIMING_KEYS), rows)


def _arms_section(results: dict) -> list[str]:
    out = ["## Arms", ""]
    arms = results.get("arms") or {}
    if not arms:
        return out + [f"Arms: {PENDING}", ""]
    selection = results.get("selection") or {}
    calibration = results.get("calibration") or {}
    rows = [[a, _num(arms[a].get("parent")), _num(arms[a].get("seed")), _num(arms[a].get("epochs")),
             _num(arms[a].get("checkpoint_sha256")), _num(arms[a].get("parent_sha256")),
             _num(selection.get(a, {}).get("dev_recall_cap2")),
             _num(calibration.get(a, {}).get("threshold")),
             _num(calibration.get(a, {}).get("recall_cap2")),
             _num(calibration.get(a, {}).get("reads"))] for a in sorted(arms)]
    header = ["arm", "parent", "seed", "epochs", "checkpoint sha256", "parent sha256",
              "dev recall cap 2", "calib threshold", "calib recall cap 2", "calib reads"]
    out += _table(header, rows)
    for a in sorted(arms):
        curve = arms[a].get("curve") or []
        if curve:
            out += [f"Curve {a} (dev capped recall per epoch): "
                    + ", ".join(_num(v) for v in curve), ""]
    return out


def _cost_section(cost: dict) -> list[str]:
    out = ["## Complete offline scorer cost (CUDA graph, 6 live tokens)", ""]
    rows = []
    for path in ("bf16", "fp32"):
        for part in ("complete", "head_only"):
            c = (cost.get(path) or {}).get(part) or {}
            rows.append([path, part, _num(c.get("median_ms")), _num(c.get("p90_ms")), _num(c.get("runs"))])
    out += _table(["gate path", "scope", "median ms", "p90 ms", "runs"], rows)
    extra = {k: cost.get(k) for k in ("param_bytes", "buffer_bytes", "read_set_diff_fraction")}
    return out + ["Read-set difference is the fraction of groups whose capped read set differs "
                  "between the bf16 and fp32 head paths (D17).", ""] + _kv_table(extra)


def _gate_section(title: str, outcome: dict | None) -> list[str]:
    outcome = outcome or {}
    passed = outcome.get("passed")
    word = PENDING if passed is None else ("PASS" if passed else "FAIL")
    out = [f"## {title}", "", f"Outcome: {word}", ""]
    for reason in outcome.get("reasons") or []:
        out.append(f"- {reason}")
    if outcome.get("reasons"):
        out.append("")
    return out + _kv_table(_scalars(outcome.get("values")))


def _scalars(d: dict | None) -> dict:
    return {k: v for k, v in (d or {}).items() if not isinstance(v, (dict, list))}


def _l2_section(l2: dict) -> list[str]:
    out = ["## L+2", ""]
    if not l2:
        return out + [f"L+2 not run or {PENDING}", ""]
    rows = [[k, _num(l2[k].get("recall_cap2")), _num(l2[k].get("precision")), _num(l2[k].get("reads"))]
            for k in sorted(l2) if isinstance(l2[k], dict)]
    return out + _table(["arm", "recall cap 2", "precision", "reads"], rows)


def _recall(results: dict, name: str) -> float | None:
    cal = (results.get("calibration") or {}).get(name) or {}
    if cal.get("recall_cap2") is not None:
        return cal["recall_cap2"]
    cell = ((results.get("stage1") or {}).get("calib") or {}).get(name, {}).get("cap2") or {}
    return cell.get("recall")


def _dev_recall(results: dict, name: str) -> float | None:
    return ((results.get("selection") or {}).get(name) or {}).get("dev_recall_cap2")


def _judge(candidate: float | None, baseline: float | None, *, split: str = "calib") -> dict:
    if candidate is None or baseline is None:
        return {"status": "unresolved", "evidence": "missing data"}
    delta = candidate - baseline
    text = f"{split} recall cap 2 {candidate:.3f} vs {baseline:.3f} (delta {delta:+.3f})"
    if delta >= RELATIVE_BAR * baseline and delta > 0:
        return {"status": "supported", "evidence": text}
    if delta <= 0:
        return {"status": "not_supported", "evidence": text}
    return {"status": "unresolved", "evidence": text + "; gain below the relative bar"}


def _lead(results: dict) -> dict:
    l2 = results.get("l2") or {}
    selected = next((v for k, v in l2.items() if isinstance(v, dict) and k != "control"), None)
    control = l2.get("control")
    if selected is None or not isinstance(control, dict):
        return {"status": "unresolved", "evidence": "L+2 not run or incomplete"}
    return _judge(selected.get("recall_cap2"), control.get("recall_cap2"))


def attribution(results: dict) -> dict:
    out = {}
    for name, (cand, base) in EXPLANATION_PAIRS.items():
        cal = (_recall(results, cand), _recall(results, base))
        if None in cal and name in ("block_aggregation", "capacity"):
            # Ladder arms are only calibrated when candidate or T0; fall back to the same-split dev D8 points.
            out[name] = _judge(_dev_recall(results, cand), _dev_recall(results, base), split="dev")
        else:
            out[name] = _judge(*cal)
    out["lead"] = _lead(results)
    return out


def _attribution_section(results: dict) -> list[str]:
    attr = results.get("attribution") or attribution(results)
    rows = [[e, attr.get(e, {}).get("status", "unresolved"), attr.get(e, {}).get("evidence", PENDING)]
            for e in EXPLANATIONS]
    out = ["## What explains any gain", ""] + _table(["explanation", "status", "evidence"], rows)
    open_ = [e for e in EXPLANATIONS if attr.get(e, {}).get("status", "unresolved") == "unresolved"]
    return out + ["Unresolved: " + (", ".join(open_) if open_ else "none"), ""]


def _delay_note(results: dict) -> list[str]:
    out = ["The Stage-2 gate's timing check uses F0's measured scorer delay (timed before the gate); "
           "native gate controls use the gate-only delay.", ""]
    gate = (results.get("stage2_gate") or {}).get("passed")
    with_delay = (results.get("stage2_gate_with_delay") or {}).get("passed")
    if gate is not None and with_delay is not None and gate != with_delay:
        out += [f"WARNING: with the measured scorer delay the Stage-2 gate outcome differs: "
                f"{'PASS' if with_delay else 'FAIL'}.", ""]
    return out


def render(results: dict) -> str:
    label = results.get("label") or LABEL
    out = ["# R1 block-level miss predictor and capped read policy", "", f"Status: {label}", "",
           f"Run plan sha256: {_num(results.get('plan_sha'))}", ""]
    if results.get("stopped_after_stage1"):
        out += ["Stopped after Stage 1: the Stage-2 gate did not pass; no arms were trained.", ""]
    stage1 = results.get("stage1") or {}
    out += _stage1_section(stage1)
    out += _histogram_section(stage1.get("histograms") or {})
    out += _timing_section(results.get("timing") or {})
    out += _gate_section("Stage-2 gate", results.get("stage2_gate"))
    out += _delay_note(results)
    out += _arms_section(results)
    out += _cost_section(results.get("cost") or {})
    out += _l2_section(results.get("l2") or {})
    out += _gate_section("Continuation gate", results.get("continuation_gate"))
    out += _attribution_section(results)
    return "\n".join(out).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.write_text(render(json.loads(args.results.read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
