"""Comparison table over bench_arm results, and the rule for whether a second pair is needed.

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
# A delta counts as resolved when it is at least this many times the within-arm block spread.
RESOLVE_FACTOR = 2.0
# The window "resolves" a delta of Z standard errors of a difference of two p50s; 1.25 is the p50's SE over the
# sd/sqrt(n) of the mean for a roughly normal sample. Judgment, like RESOLVE_FACTOR.
DETECT_Z = 3.0
P50_SE_FACTOR = 1.25
# Arms whose layer miss fractions differ by more than this saw different work.
MIX_TOLERANCE = 0.05


def detectable_delta_s(*, sd_a: float, n_a: int, sd_b: float, n_b: int) -> float:
    """Smallest difference of two p50s (seconds) this many samples at this sd can resolve."""
    return DETECT_Z * P50_SE_FACTOR * ((sd_a**2 / n_a) + (sd_b**2 / n_b)) ** 0.5


def _get(run: dict, *keys):
    for k in keys:
        if run is None or k not in run:
            return None
        run = run[k]
    return run


# (label, path into a run, scale, format)
ROWS = (
    (
        "tokens/s (whole window, prefill included)",
        ("summary", "tokens_per_s"),
        1,
        "{:.3f}",
    ),
    ("decode step min (ms)", ("summary", "step_s", "min"), 1e3, "{:.2f}"),
    ("decode step p50 (ms)", ("summary", "step_s", "p50"), 1e3, "{:.2f}"),
    ("decode step p95 (ms)", ("summary", "step_s", "p95"), 1e3, "{:.2f}"),
    ("decode step p99 (ms)", ("summary", "step_s", "p99"), 1e3, "{:.2f}"),
    ("per-token samples", ("summary", "step_s", "n"), 1, "{:.0f}"),
    (
        "within-arm spread of block p50 (%)",
        ("summary", "step_blocks", "p50_rel_range"),
        100,
        "{:.2f}",
    ),
    (
        "within-arm spread of block min (%)",
        ("summary", "step_blocks", "min_rel_range"),
        100,
        "{:.2f}",
    ),
    ("e2e per request, mean (s)", ("summary", "e2e_s_mean"), 1, "{:.1f}"),
    ("TTFT per request, mean (s)", ("summary", "ttft_s_mean"), 1, "{:.1f}"),
    (
        "window: layer steps with a RAM miss (fraction)",
        ("mix", "layer_miss_fraction"),
        1,
        "{:.4f}",
    ),
    ("window: served", ("mix", "counters", "served"), 1, "{:.0f}"),
    ("window: touch_only", ("mix", "counters", "touch_only"), 1, "{:.0f}"),
    ("window: rows_read", ("mix", "counters", "rows_read"), 1, "{:.0f}"),
    ("window: evictions", ("mix", "counters", "evictions"), 1, "{:.0f}"),
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


def _values(runs: list[dict], path: tuple, scale: float) -> list[float]:
    out = []
    for r in runs:
        v = _get(r, *path)
        if isinstance(v, (int, float)):
            out.append(v * scale)
    return out


def _cell(values: list[float], fmt: str) -> str:
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return fmt.format(mean)
    return f"{fmt.format(mean)} [{fmt.format(min(values))}..{fmt.format(max(values))}]"


def _pct(off: list[float], on: list[float]) -> str:
    if not off or not on or statistics.fmean(off) == 0:
        return ""
    return f"{(statistics.fmean(on) / statistics.fmean(off) - 1) * 100:+.2f}%"


def problems(runs: list[dict]) -> list[str]:
    out = []
    for a in ARMS:
        if not any(r["arm"] == a for r in runs):
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


def decision(by_arm: dict) -> list[str]:
    """Whether the first pair resolves the effect, from the arms' own numbers.

    Resolved means: the p50 delta is at least RESOLVE_FACTOR times the larger within-arm block spread, and the
    min-to-min delta has the same sign (min is robust to contention; a disagreement means the p50 delta is
    probably box noise). Heuristic, not a test: the thresholds are stated so they can be argued with.
    """
    off, on = (by_arm[a] for a in ARMS)
    get = lambda runs, path: _values(runs, path, 1)  # noqa: E731
    p50 = [
        statistics.fmean(get(r, ("summary", "step_s", "p50")))
        if get(r, ("summary", "step_s", "p50"))
        else None
        for r in (off, on)
    ]
    mn = [
        statistics.fmean(get(r, ("summary", "step_s", "min")))
        if get(r, ("summary", "step_s", "min"))
        else None
        for r in (off, on)
    ]
    spreads = get(off, ("summary", "step_blocks", "p50_rel_range")) + get(
        on, ("summary", "step_blocks", "p50_rel_range")
    )
    if None in p50 or None in mn or not spreads:
        return [
            "decision: not computable (missing per-token statistics or too few samples for blocks)"
        ]
    d_p50, d_min = p50[1] / p50[0] - 1, mn[1] / mn[0] - 1
    spread = max(spreads)
    pooled = [
        (
            statistics.fmean(_values(by_arm[a], ("summary", "step_s", "sd"), 1)),
            sum(_values(by_arm[a], ("summary", "step_s", "n"), 1)),
        )
        for a in ARMS
    ]
    noise = (
        detectable_delta_s(
            sd_a=pooled[0][0], n_a=pooled[0][1], sd_b=pooled[1][0], n_b=pooled[1][1]
        )
        / p50[0]
    )
    bound = max(noise, RESOLVE_FACTOR * spread)
    lines = [
        f"decision inputs: p50 delta {d_p50 * 100:+.2f}%, min-to-min delta {d_min * 100:+.2f}%, "
        f"within-arm spread {spread * 100:.2f}% (x{RESOLVE_FACTOR:g} = {RESOLVE_FACTOR * spread * 100:.2f}%), "
        f"per-token sd noise floor {noise * 100:.2f}% ({DETECT_Z:g} SE)"
    ]
    signs_agree = (d_p50 >= 0) == (d_min >= 0)
    if abs(d_p50) >= bound and signs_agree:
        lines.append(
            f"decision: RESOLVED. |p50 delta| {abs(d_p50) * 100:.2f}% >= the {bound * 100:.2f}% this window resolves, and min-to-min agrees: two loads were enough; do not rerun."
        )
    elif abs(d_p50) >= bound:
        lines.append(
            "decision: NOT RESOLVED. p50 and min-to-min disagree in sign: suspect cross-process noise; run a second pair with the order flipped (--rep-start 1)."
        )
    else:
        lines.append(
            f"decision: UNRESOLVED. Any effect of lease mode on the p50 decode step is below {bound * 100:.2f}% "
            f"({bound * p50[0] * 1e3:.2f} ms of a {p50[0] * 1e3:.1f} ms step); this window could not resolve less. "
            f"Measured delta {d_p50 * 100:+.2f}% is inside that. That is an upper bound, not a null: "
            "a second pair with the order flipped (--rep-start 1) or a longer window only helps if the bound is above the effect you need."
        )
    return lines


def mix_lines(by_arm: dict) -> list[str]:
    labels = {a: [r["mix"]["label"] for r in by_arm[a] if r.get("mix")] for a in ARMS}
    fractions = {a: _values(by_arm[a], ("mix", "layer_miss_fraction"), 1) for a in ARMS}
    lines = [
        "hit/miss mix of the measured window: "
        + "; ".join(f"{a}: {','.join(labels[a]) or 'n/a'}" for a in ARMS)
    ]
    if (
        all(fractions.values())
        and abs(
            statistics.fmean(fractions["lease_off"])
            - statistics.fmean(fractions["lease_on"])
        )
        > MIX_TOLERANCE
    ):
        lines.append(
            "WARNING: the arms saw different miss fractions; their latencies are not comparable"
        )
    if any(label != "mixed" for ls in labels.values() for label in ls):
        lines.append(
            "note: a window that is all-hit or all-miss measures a different regime from steady-state serving (all-hit is where OPEN 11's arming cost falls; leases exist for misses)"
        )
    return lines


def parity(runs: list[dict]) -> str:
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
        f"pairs: lease_off={len(by_arm['lease_off'])} lease_on={len(by_arm['lease_on'])}; with more than one, cells are mean [min..max]"
    )
    positions = {a: [r.get("position") for r in by_arm[a]] for a in ARMS}
    lines.append(f"process order (0 = first of its pair): {positions}")
    lines += ["", "| metric | lease_off | lease_on | on vs off |", "|---|---|---|---|"]
    for label, path, scale, fmt in ROWS:
        off, on = (_values(by_arm[a], path, scale) for a in ARMS)
        lines.append(
            f"| {label} | {_cell(off, fmt)} | {_cell(on, fmt)} | {_pct(off, on)} |"
        )
    lines.append("")
    if all(by_arm.values()):
        lines += decision(by_arm) + mix_lines(by_arm)
    traced = sorted(
        {
            str(r["trace_overhead"]["trace_enabled"])
            for r in runs
            if r.get("trace_overhead")
        }
    )
    lines.append(
        f"stream trace enabled (its per-step overhead and variance cost are NOT measured): {', '.join(traced) or 'n/a'}"
    )
    lines.append(f"greedy output parity across arms: {parity(runs)}")
    lease = [
        r["counters_final"]
        for r in runs
        if r["arm"] == "lease_on" and r.get("counters_final")
    ]
    if lease:
        lines.append(
            "lease_on final counters (granted/acked/voided): "
            + "; ".join(
                f"{c['leases_granted']}/{c['leases_acked']}/{c['leases_voided']}"
                for c in lease
            )
        )
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
