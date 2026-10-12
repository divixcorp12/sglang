"""Render the R2 run's results.json as REPORT.md, under the exploratory label.

Inputs: results.json with keys label, plan_sha, arms ({arm: {seed: {curve, epochs,
dev_recall_cap2, checkpoint_sha256, parent_sha256}}}), selection ({winner, median_dev_recall:
{arm: v}, t3_median}), comparison ({cap1|cap2: {scorer: {recall, precision, reads}}}), calib
({scorer: {threshold, recall_cap2, precision, reads}}), timing ({assumptions, cases}), cost
({arm: {dtype: {path: {median_ms, p90_ms, p99_ms, runs, param_bytes, buffer_bytes}}}}), l2,
continuation_gate ({passed, reasons, values}), adds_over_t3. A missing value renders as `pending`.
Usage: python3 r2_report.py RESULTS.json --out REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dsv41"))
sys.path.insert(0, str(REPO / "analysis" / "dsv41-drive" / "router-capture"))
sys.path.insert(0, str(HERE))

import r1_plan  # noqa: E402
import r1_report  # noqa: E402
from r1_report import PENDING, _num, _table  # noqa: E402

COMPARE_KEYS = ("recall", "precision", "reads")
CALIB_KEYS = ("threshold", "recall_cap2", "precision", "reads")
COST_PATHS = ("head_only", "complete", "correction")
COST_KEYS = ("median_ms", "p90_ms", "p99_ms", "runs", "param_bytes", "buffer_bytes")
# The increment the continuation gate prices (D9): G0's correction, T5's head.
GATE_PATH = {"G0": "correction", "T5": "head_only"}
D12_SENTENCE = ("Neither T5 nor G0 beats T3: the remaining representation changes add nothing "
                "at this data size, and no further arms are run in this feature.")


def _arms_section(arms: dict) -> list[str]:
    out = ["## Arms: per-seed dev recall at cap 2 and curves", ""]
    if not arms:
        return out + [f"Arms: {PENDING}", ""]
    rows = []
    for arm in sorted(arms):
        for seed in sorted(arms[arm]):
            c = arms[arm][seed] or {}
            rows.append([arm, str(seed), _num(c.get("epochs")), _num(c.get("dev_recall_cap2")),
                         _num(c.get("checkpoint_sha256")), _num(c.get("parent_sha256"))])
    out += _table(["arm", "seed", "epochs", "dev recall cap 2", "checkpoint sha256", "parent sha256"],
                  rows)
    for arm in sorted(arms):
        for seed in sorted(arms[arm]):
            curve = (arms[arm][seed] or {}).get("curve") or []
            if curve:
                out += [f"Curve {arm} seed {seed} (dev recall per epoch): "
                        + ", ".join(_num(v) for v in curve), ""]
    return out


def _selection_section(selection: dict) -> list[str]:
    out = ["## Selection (median dev recall across seeds)", ""]
    medians = selection.get("median_dev_recall") or {}
    rows = [[arm, _num(medians[arm])] for arm in sorted(medians)]
    out += _table(["arm", "median dev recall cap 2"], rows) if rows else [f"Medians: {PENDING}", ""]
    return out + [f"Winner: {_num(selection.get('winner'))}", ""]


def _comparison_section(comparison: dict) -> list[str]:
    out = ["## Dev comparison (identical groups, skip masks and denominators)", ""]
    rows = []
    for cap in sorted(comparison):
        for scorer in sorted(comparison[cap] or {}):
            cell = comparison[cap][scorer] or {}
            rows.append([cap, scorer] + [_num(cell.get(k)) for k in COMPARE_KEYS])
    if not rows:
        return out + [f"Comparison: {PENDING}", ""]
    out += _table(["cap", "scorer"] + list(COMPARE_KEYS), rows)
    return out + ["Cap 2 is the policy; cap 1 is reported but degenerate and not used.", ""]


def _calib_section(calib: dict) -> list[str]:
    out = ["## Calib points", ""]
    if not calib:
        return out + [f"Calib points: {PENDING}", ""]
    rows = [[s] + [_num((calib[s] or {}).get(k)) for k in CALIB_KEYS] for s in sorted(calib)]
    return out + _table(["scorer"] + list(CALIB_KEYS), rows)


def _cost_section(cost: dict) -> list[str]:
    out = ["## Scorer cost (CUDA graph, 6 live tokens)", ""]
    rows = []
    for arm in sorted(cost):
        for dtype in sorted(cost[arm] or {}):
            for path in COST_PATHS:
                c = (cost[arm][dtype] or {}).get(path)
                if c is None:
                    continue
                mark = " (gate figure)" if GATE_PATH.get(arm) == path else ""
                rows.append([arm, dtype, path + mark] + [_num(c.get(k)) for k in COST_KEYS])
    if not rows:
        return out + [f"Cost: {PENDING}", ""]
    out += _table(["arm", "dtype", "path"] + list(COST_KEYS), rows)
    return out + ["The gate prices the serving increment: G0's correction given native scores, "
                  "T5's head. The complete path includes the offline gate.", ""]


def _criteria_rows(values: dict) -> list[list[str]]:
    rows = []
    for arm, crit in sorted((values.get("arms") or {}).items()):
        if not isinstance(crit, dict):
            rows.append([arm, "-", _num(crit)])
            continue
        rows += [[arm, name, _num(crit[name]) if not isinstance(crit[name], (dict, list))
                  else json.dumps(crit[name], sort_keys=True)] for name in sorted(crit)]
    return rows


def _gate_section(gate: dict) -> list[str]:
    passed = gate.get("passed")
    word = PENDING if passed is None else ("PASS" if passed else "FAIL")
    out = ["## R2 continuation gate", "", f"Outcome: {word}", ""]
    out += [f"- {r}" for r in gate.get("reasons") or []]
    if gate.get("reasons"):
        out.append("")
    rows = _criteria_rows(gate.get("values") or {})
    if not rows:
        return out + [f"Criteria: {PENDING}", ""]
    return out + _table(["arm", "criterion", "value / verdict"], rows)


def _reading(results: dict) -> list[str]:
    sel = results.get("selection") or {}
    medians = sel.get("median_dev_recall") or {}
    winner = sel.get("winner")
    t3 = sel.get("t3_median", medians.get("T3"))
    out = ["## Reading", ""]
    if winner is None or medians.get(winner) is None or t3 is None:
        return out + [f"The comparison is {PENDING}; no reading yet. {results.get('label') or r1_plan.LABEL}.", ""]
    beats = medians[winner] > t3
    text = (f"The better arm by median dev recall at cap 2 is {winner} "
            f"({medians[winner]:.3f}) against T3 ({t3:.3f}). ")
    if not beats:
        text += D12_SENTENCE
    elif results.get("adds_over_t3"):
        text += f"{winner} adds over T3 and clears the R2 continuation gate."
    else:
        text += f"{winner} is ahead of T3 on dev but does not clear the R2 continuation gate."
    return out + [text, ""]


def render(results: dict) -> str:
    label = results.get("label") or r1_plan.LABEL
    adds = results.get("adds_over_t3")
    out = ["# R2 supervised projection (T5) and full-score gate correction (G0)", "",
           f"Status: {label}", "", f"R2 plan sha256: {_num(results.get('plan_sha'))}", "",
           f"T5 or G0 adds anything over T3: {PENDING if adds is None else str(bool(adds)).lower()}", ""]
    out += _arms_section(results.get("arms") or {})
    out += _selection_section(results.get("selection") or {})
    out += _comparison_section(results.get("comparison") or {})
    out += _calib_section(results.get("calib") or {})
    out += r1_report._timing_section(results.get("timing") or {})
    out += _cost_section(results.get("cost") or {})
    out += r1_report._l2_section(results.get("l2") or {})
    out += _gate_section(results.get("continuation_gate") or {})
    out += _reading(results)
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
