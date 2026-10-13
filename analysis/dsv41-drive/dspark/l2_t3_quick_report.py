#!/usr/bin/env python3
"""Turn the quick T3 L+2 served run plus F108's fwd dir into REPORT.md and results.json; stdlib only.

Reads the quick run's fwd dir (arms L+2 and T3) and F108's fwd dir (misscut and native L+2) read-only.
There is no verdict: one order, 4 sessions, a cross-day baseline.
Usage: l2_t3_quick_report.py QUICK_FWD_DIR --f108 F108_FWD_DIR --out DIR
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import l2_ab_report  # noqa: E402

L2 = l2_ab_report.L2
MISSCUT = l2_ab_report.MISSCUT
T3 = "dspark-both-prefetch-gpu-misscut-idle2-l2-t3"
LABEL = "quick check: one order, 4 sessions, cross-day baseline; not a verdict"
SESSIONS = (0, 1, 2, 3)

# Source labels: the same arm name exists in both directories, so the paired figures are keyed by source.
QUICK_L2 = "quick:l2"
QUICK_T3 = "quick:t3"
F108_MISSCUT = "f108:misscut"
F108_L2 = "f108:l2"

# (name, arm, other, criterion)
PAIRS = (
    ("l2_vs_f108_misscut", QUICK_L2, F108_MISSCUT, "C19"),
    ("t3_vs_f108_misscut", QUICK_T3, F108_MISSCUT, "C19"),
    ("t3_vs_fresh_l2", QUICK_T3, QUICK_L2, "C20"),
    ("fresh_l2_vs_f108_l2_drift", QUICK_L2, F108_L2, "C21"),
)


def _load_summary(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "summary.json")) as f:
        return json.load(f)


def _timed_ids(figures: dict) -> set:
    """The quick run's session ids: results.jsonl keys rows by corpus id, not by the index passed to --sessions."""
    ids = set(figures[QUICK_L2])
    if ids != set(figures[QUICK_T3]) or len(ids) != len(SESSIONS):
        raise SystemExit(f"quick arms timed different or unexpected sessions: {sorted(ids)} vs {sorted(figures[QUICK_T3])}")
    return ids


def _spec_counters(summary: dict, arm: str) -> dict:
    """Server-lifetime speculative read counters; wasted = landed - used."""
    ram = summary[arm].get("ram") or {}
    issued, landed, used = ram.get("spec_issued"), ram.get("spec_landed"), ram.get("spec_used")
    wasted = landed - used if landed is not None and used is not None else None
    counters = {"issued": issued, "landed": landed, "used": used, "wasted": wasted}
    counters.update(l2_ab_report.arm_counters(summary, arm))
    return counters


def _text_status(text) -> str:
    if isinstance(text, dict) and "pass" in text:
        return "pass" if text["pass"] is True else "FAIL"
    return f"n/a ({text})" if text is not None else "n/a (not compared)"


def analyse(quick_dir: str, f108_dir: str) -> dict:
    quick, f108 = _load_summary(quick_dir), _load_summary(f108_dir)
    sources = {
        QUICK_L2: (quick_dir, quick, L2),
        QUICK_T3: (quick_dir, quick, T3),
        F108_MISSCUT: (f108_dir, f108, MISSCUT),
        F108_L2: (f108_dir, f108, L2),
    }
    figures, flagged, counters = {}, {}, {}
    for label, (run_dir, summary, arm) in sources.items():
        figures[label] = l2_ab_report.session_figures(l2_ab_report._last_run_results(run_dir, arm))
        flagged[label] = l2_ab_report._flagged(summary, arm)
        counters[label] = _spec_counters(summary, arm)
        counters[label]["text_status"] = _text_status(counters[label]["text_vs_reference"])
    timed = _timed_ids(figures)
    figures = {label: {sid: fig for sid, fig in figs.items() if sid in timed} for label, figs in figures.items()}
    pairs = {}
    for name, arm, other, criterion in PAIRS:
        pair = l2_ab_report.pair_table(figures, flagged, arm, other)
        pair["criterion"] = criterion
        pairs[name] = pair
    return {
        "label": LABEL,
        "sessions": list(SESSIONS),
        "quick_dir": quick_dir,
        "f108_dir": f108_dir,
        "pairs": pairs,
        "counters": counters,
    }


def _fmt(v, spec: str = ".3f") -> str:
    return "n/a" if v is None else format(v, spec)


def _pair_section(name: str, pair: dict) -> list:
    arm, other = pair["arm"], pair["other"]
    lines = [
        f"### {name} ({pair['criterion']}): {arm} / {other}",
        "",
        f"_{LABEL}_",
        "",
        f"| session | ms/token {other} | ms/token {arm} | ms/token ratio | step ms {other} | step ms {arm} "
        "| step ratio | in median |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in pair["sessions"]:
        a, b = r.get("ms_per_token_" + arm), r.get("ms_per_token_" + other)
        ratio = a / b if a is not None and b else None
        lines.append(
            f"| {r['session_id']} | {_fmt(b, '.2f')} | {_fmt(a, '.2f')} | {_fmt(ratio)} "
            f"| {_fmt(r.get('step_ms_' + other), '.1f')} | {_fmt(r.get('step_ms_' + arm), '.1f')} "
            f"| {_fmt(r.get('step_ratio'))} | {'no (drive flag)' if r['excluded'] else 'yes'} |"
        )
    lines += [
        "",
        f"Median step ratio {_fmt(pair['step_ratio_median'])} over {pair['sessions_used']} sessions; paired ms/token "
        f"median {_fmt(pair['paired_gain_pct_median'], '+.1f')}% (positive favours {arm}).",
        "",
    ]
    return lines


def _counter_table(counters: dict) -> list:
    lines = [
        "## Speculative read counters (server lifetime) and text",
        "",
        f"_{LABEL}_",
        "",
        "| source | issued | landed | used | wasted | used / landed | text vs reference |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, c in counters.items():
        lines.append(
            f"| {label} | {_fmt(c['issued'], 'd')} | {_fmt(c['landed'], 'd')} | {_fmt(c['used'], 'd')} "
            f"| {_fmt(c['wasted'], 'd')} | {_fmt(c['used_over_landed'])} | {c['text_status']} |"
        )
    return lines + [""]


def render(result: dict) -> str:
    lines = [
        "# T3 learned L+2 scorer vs native L+2: served quick check",
        "",
        f"**{LABEL}**",
        "",
        f"Sessions {', '.join(str(s) for s in result['sessions'])}. Quick run: `{result['quick_dir']}`; "
        f"F108 baseline (read only): `{result['f108_dir']}`. Ratios are arm / other; below 1 is faster for the arm.",
        "",
    ]
    lines += _counter_table(result["counters"])
    lines += ["## Per-session ratios", "", f"_{LABEL}_", ""]
    for name, pair in result["pairs"].items():
        lines += _pair_section(name, pair)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("quick_fwd_dir")
    p.add_argument("--f108", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    result = analyse(a.quick_fwd_dir, a.f108)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(a.out, "REPORT.md"), "w") as f:
        f.write(render(result))
    print(f"report written to {a.out}; {LABEL}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
