"""Render the B1 run's results.json as section 33.26 style markdown tables.

Inputs: results.json in the schema of b1_run.run (horizons, learning_curve, bootstrap, cost,
verdict, frozen_sha256, paths). Any missing value renders as `pending`.
Usage: python3 b1_report.py RESULTS.json --out REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PENDING = "pending"
SCORER_ORDER = ("gate", "gate_floors", "prev", "pop", "E2", "E5")
POOLS = ("pooled", "pooled_noflag")
METRICS = ("recall", "precision", "never_recovery")
LEARNING_SIZES = ("15", "30", "60")


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


def _scorer_names(tables: dict) -> list[str]:
    known = [s for s in SCORER_ORDER if s in tables]
    return known + sorted(s for s in tables if s not in SCORER_ORDER)


def _budget_sort_key(budget: str) -> tuple[int, float]:
    kind, _, value = budget.partition("=")
    return (0 if kind == "reads" else 1, float(value))


def _budgets(tables: dict) -> list[str]:
    seen = {b for t in tables.values() for b in t.get("cells", {})}
    return sorted(seen, key=_budget_sort_key)


def _cell(table: dict, *, budget: str, key: str) -> dict | None:
    return table.get("cells", {}).get(budget, {}).get(key)


def _metric_cells(table: dict, *, budget: str, key: str) -> list[str]:
    c = _cell(table, budget=budget, key=key)
    return [_num(None if c is None else c.get(m)) for m in METRICS]


def _pooled_table(tables: dict, *, key: str) -> list[str]:
    header = ["scorer", "budget", "recall", "precision", "never-candidate recovery"]
    rows = [[s, b] + _metric_cells(tables[s], budget=b, key=key)
            for s in _scorer_names(tables) for b in _budgets(tables)]
    return _table(header, rows)


def _keys_with(tables: dict, prefix: str) -> list[str]:
    keys = {k for t in tables.values() for keyed in t.get("cells", {}).values() for k in keyed
            if k.startswith(prefix)}
    return sorted(keys, key=lambda k: (len(k), k))


def _split_table(tables: dict, *, prefix: str, budget: str) -> list[str]:
    scorers = _scorer_names(tables)
    header = [prefix.rstrip(":")] + [f"{s} {m}" for s in scorers for m in ("recall", "precision")]
    rows = []
    for key in _keys_with(tables, prefix):
        row = [key.split(":", 1)[1]]
        for s in scorers:
            c = _cell(tables[s], budget=budget, key=key)
            row += [_num(None if c is None else c.get(m)) for m in ("recall", "precision")]
        rows.append(row)
    return _table(header, rows)


def _horizon_section(h: str, block: dict) -> list[str]:
    tables = block.get("tables", {})
    out = [f"### L+{h}", ""]
    if not tables:
        return out + [f"No tables: {PENDING}", ""]
    out += ["Pooled over all locked-test sessions.", ""] + _pooled_table(tables, key="pooled")
    out += ["Pooled without the drive-flagged sessions.", ""] + _pooled_table(tables, key="pooled_noflag")
    for prefix, title in (("layer:", "Per target layer"), ("stratum:", "Per stratum")):
        if not _keys_with(tables, prefix):
            continue
        for budget in _budgets(tables):
            out += [f"{title} ({budget}).", ""] + _split_table(tables, prefix=prefix, budget=budget)
    return out


def _learning_curve(curve: dict, improving: dict) -> list[str]:
    out = ["## Learning curve (dev, precision at reads=1)", ""]
    variants = sorted(curve) if curve else ["E2", "E5"]
    rows = [[v] + [_num(curve.get(v, {}).get(n)) for n in LEARNING_SIZES] + [_num(improving.get(v))]
            for v in variants]
    header = ["variant"] + [f"{n} sessions" for n in LEARNING_SIZES] + ["still improving"]
    return out + _table(header, rows)


def _second_capture(sc: dict) -> list[str]:
    sc = sc or {}
    rows = [[k, _num(sc.get(k))] for k in ("near_bar", "still_improving", "warranted")]
    return ["Second capture (D5: near the bar and still improving):", ""] + _table(["item", "value"], rows)


def _bootstrap(b: dict) -> list[str]:
    b = b or {}
    line = (f"Session bootstrap of the head-minus-gate-with-floors precision at reads=1: "
            f"median {_num(b.get('median'))}, 95% CI [{_num(b.get('lo'))}, {_num(b.get('hi'))}].")
    return ["## Bootstrap", "", line, ""]


def _cost(c: dict) -> list[str]:
    c = c or {}
    keys = ("bytes_bf16", "median_ms", "p90_ms", "runs", "tokens", "within_0_1ms")
    out = ["## Scorer cost (chosen L+1 head, RTX 5090, 6 tokens, one layer)", ""]
    if c.get("skipped"):
        out += [f"Skipped: {c['skipped']}", ""]
    return out + _table(["metric", "value"], [[k, _num(c.get(k))] for k in keys])


def _verdict(v: dict) -> list[str]:
    v = v or {}
    advance = v.get("advance")
    word = PENDING if advance is None else ("ADVANCE" if advance else "DO NOT ADVANCE")
    out = ["## Verdict (L+1 against the B2 bar)", "", f"Verdict: {word}", ""]
    arms = v.get("arms") or {}
    if arms:
        out += _table(["arm", "value"], [[k, _num(arms[k])] for k in sorted(arms)])
    for reason in v.get("reasons") or []:
        out.append(f"- {reason}")
    if v.get("reasons"):
        out.append("")
    return out


def _provenance(results: dict) -> list[str]:
    paths = results.get("paths") or {}
    rows = [["frozen selection sha256", _num(results.get("frozen_sha256"))]]
    rows += [[k, _num(paths[k])] for k in sorted(paths)]
    return ["## Provenance", ""] + _table(["item", "value"], rows)


def render(results: dict) -> str:
    out = ["# B1 learned predictor vs the native gate", ""]
    horizons = results.get("horizons") or {}
    for h in sorted(horizons, key=int):
        out += _horizon_section(str(h), horizons[h])
    if not horizons:
        out += [f"Horizon tables: {PENDING}", ""]
    out += _learning_curve(results.get("learning_curve") or {}, results.get("still_improving") or {})
    out += _bootstrap(results.get("bootstrap"))
    out += _cost(results.get("cost"))
    out += _verdict(results.get("verdict"))
    out += _second_capture(results.get("second_capture"))
    out += _provenance(results)
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
