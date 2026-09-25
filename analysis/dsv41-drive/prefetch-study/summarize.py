"""Markdown table of prefetch_baselines.py reports (held-out steps): one row per arm and budget."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+")
    p.add_argument("--split", default="test")
    p.add_argument("--predictors", nargs="*", default=[])
    args = p.parse_args()
    print("| predictor | k | h | budget | demand/tok | prefetch/tok | useful/tok | precision | G_total | exposed | est ms/tok | gap share |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    seen = set()
    for path in args.reports:
        with open(path) as f:
            report = json.load(f)
        for r in report["results"]:
            a, s = r["arm"], r["splits"][args.split]
            name = a["predictor"] + (f" {a['decay']:g}" if a["decay"] else "")
            if args.predictors and a["predictor"] not in args.predictors:
                continue
            key = (name, a["k"], a["horizon"])
            if key in seen:
                continue
            seen.add(key)
            for budget, exposed in s["exposed"].items():
                precision = s["useful"] / s["prefetch"] if s["prefetch"] else float("nan")
                share = s.get("gap_share", {}).get(budget, float("nan"))
                print(f"| {name} | {a['k']} | {a['horizon']} | {budget} | {s['demand']:.2f} | {s['prefetch']:.2f} | "
                      f"{s['useful']:.2f} | {precision:.2f} | {s['G_total']:.2f} | {exposed:.2f} | "
                      f"{s['ms'][budget]:.1f} | {share:+.0%} |")


if __name__ == "__main__":
    main()
