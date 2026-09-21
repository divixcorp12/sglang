"""Comparison table over bench_arm results.

    python benchmarks/dsv41_flash/compare.py RUN_DIR_OR_FILES...

Reads only the standard library, so it runs anywhere the .json.gz files are.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

ARMS = ("lease_off", "lease_on")
# (label, extractor over one run, scale, format); None from an extractor means the run lacks it.
ROWS = (
    ("tokens/s (window)", lambda r: r["summary"]["tokens_per_s"], 1, "{:.3f}"),
    (
        "decode tok/s (mean of requests)",
        lambda r: r["summary"]["decode_tok_s_mean"],
        1,
        "{:.3f}",
    ),
    ("e2e p50 (s)", lambda r: r["summary"]["e2e_s"]["p50"], 1, "{:.2f}"),
    ("e2e p95 (s)", lambda r: r["summary"]["e2e_s"]["p95"], 1, "{:.2f}"),
    ("e2e p99 (s)", lambda r: r["summary"]["e2e_s"]["p99"], 1, "{:.2f}"),
    ("TTFT p50 (s)", lambda r: r["summary"]["ttft_s"]["p50"], 1, "{:.2f}"),
    ("decode step p50 (ms)", lambda r: r["summary"]["step_s"]["p50"], 1e3, "{:.2f}"),
    ("decode step p95 (ms)", lambda r: r["summary"]["step_s"]["p95"], 1e3, "{:.2f}"),
    ("decode step p99 (ms)", lambda r: r["summary"]["step_s"]["p99"], 1e3, "{:.2f}"),
    ("rows_read in window", lambda r: r.get("rows_read_window"), 1, "{:.0f}"),
)


def load_runs(paths: list[str]) -> list[dict]:
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json.gz")) if p.is_dir() else [p]
    runs = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            run = json.load(fh)
        run["_file"] = str(f)
        runs.append(run)
    return runs


def _values(runs: list[dict], extract, scale: float) -> list[float]:
    out = []
    for r in runs:
        try:
            v = extract(r)
        except (KeyError, TypeError):
            v = None
        if v is not None:
            out.append(v * scale)
    return out


def _cell(values: list[float], fmt: str) -> str:
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    return (
        f"{fmt.format(mean)} [{fmt.format(min(values))}..{fmt.format(max(values))}]"
        if len(values) > 1
        else fmt.format(mean)
    )


def _delta(off: list[float], on: list[float]) -> str:
    if not off or not on or statistics.fmean(off) == 0:
        return "n/a"
    d = statistics.fmean(on) / statistics.fmean(off) - 1
    noise = (
        max(
            (max(v) - min(v)) / statistics.fmean(v)
            for v in (off, on)
            if len(v) > 1 and statistics.fmean(v) != 0
        )
        if len(off) > 1 or len(on) > 1
        else None
    )
    verdict = (
        "no rep spread to judge"
        if noise is None
        else ("within rep spread" if abs(d) <= noise else "beyond rep spread")
    )
    return f"{d * 100:+.2f}% ({verdict})"


def problems(runs: list[dict]) -> list[str]:
    out = []
    by_arm = {a: [r for r in runs if r["arm"] == a] for a in ARMS}
    for a in ARMS:
        if not by_arm[a]:
            out.append(f"no {a} runs")
    for r in runs:
        tag = f"{r['arm']} rep {r['rep']}"
        if r.get("error"):
            out.append(f"{tag}: run failed: {r['error']}")
        check = r.get("lease_check") or {
            "ok": False,
            "reasons": ["no lease_check in result"],
        }
        if not check["ok"]:
            out.append(f"{tag}: lease path NOT verified: {'; '.join(check['reasons'])}")
    return out


def parity(runs: list[dict]) -> str:
    """Per rep, requests whose greedy output differs between the arms (same prompts, same order)."""
    by = {(r["arm"], r["rep"]): r for r in runs if r.get("requests")}
    total = bad = 0
    for rep in sorted({r["rep"] for r in runs}):
        if ("lease_off", rep) in by and ("lease_on", rep) in by:
            for a, b in zip(
                by[("lease_off", rep)]["requests"], by[("lease_on", rep)]["requests"]
            ):
                total += 1
                bad += a["output_sha1"] != b["output_sha1"]
    return (
        "n/a (no rep has both arms)"
        if total == 0
        else f"{total - bad}/{total} requests identical"
    )


def render(runs: list[dict]) -> str:
    by_arm = {a: [r for r in runs if r["arm"] == a and r.get("summary")] for a in ARMS}
    lines = []
    bad = problems(runs)
    if bad:
        lines += ["**INVALID OR INCOMPLETE COMPARISON**", *(f"- {b}" for b in bad), ""]
    work = next((r["workload"] for r in runs if r.get("workload")), None)
    if work:
        lines.append(
            "workload: "
            + ", ".join(f"{k}={v}" for k, v in work.items() if k != "sessions")
        )
    lines.append(
        f"reps: lease_off={len(by_arm['lease_off'])} lease_on={len(by_arm['lease_on'])}; cells are mean [min..max] over reps"
    )
    lines += ["", "| metric | lease_off | lease_on | on vs off |", "|---|---|---|---|"]
    for label, extract, scale, fmt in ROWS:
        off, on = (_values(by_arm[a], extract, scale) for a in ARMS)
        lines.append(
            f"| {label} | {_cell(off, fmt)} | {_cell(on, fmt)} | {_delta(off, on)} |"
        )
    lease = [
        r["counters_final"]
        for r in runs
        if r["arm"] == "lease_on" and r.get("counters_final")
    ]
    if lease:
        lines += [
            "",
            "lease_on final counters per rep (granted/acked/voided): "
            + "; ".join(
                f"{c['leases_granted']}/{c['leases_acked']}/{c['leases_voided']}"
                for c in lease
            ),
        ]
    lines.append(f"greedy output parity across arms: {parity(runs)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "paths", nargs="+", help="a run directory of *.json.gz, or the files"
    )
    args = p.parse_args(argv)
    runs = load_runs(args.paths)
    if not runs:
        sys.exit("no results found")
    print(render(runs))
    return 1 if problems(runs) else 0


if __name__ == "__main__":
    sys.exit(main())
