"""Applies the decision rule of NC_VISIBILITY.md section 4 to a nc_visibility results.jsonl, mechanically.

    python nc_visibility_report.py RESULTS.jsonl [--iters 200000]

The rule is fixed in the pre-registration; this script only evaluates it, so that the verdict is not
decided by whoever reads the numbers. Exit code: 0 keep nc, 10 switch to cv, 11 escalate, 12 inconclusive.
"""

import argparse
import json
import sys

WORDS_PER_ITER = 65536  # 4 rows x 128 KiB / 8 bytes
MODES = ("boundary", "graph", "thrash", "concurrent")
STORES = ("regular", "nt")


def load(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("{"):
                records.append(json.loads(line))
    return records


def decide(records, iters):
    problems = []
    self_check = [r for r in records if r["kind"] == "c_self"]
    if len(self_check) != 1 or not self_check[0]["pass"]:
        problems.append("C-self did not pass (the harness cannot see staleness)")
    cells = {}
    for r in records:
        if r["kind"] == "primary":
            cells[(r["variant"], r["mode"], r["host_store"])] = r
    for variant in ("nc", "cv"):
        for mode in MODES:
            for store in STORES:
                cell = cells.get((variant, mode, store))
                if cell is None:
                    problems.append(f"missing cell {variant}/{mode}/{store}")
                    continue
                if cell["iters"] != iters:
                    problems.append(f"{variant}/{mode}/{store} ran {cell['iters']} iterations, not {iters}")
                if cell["words_checked"] != iters * WORDS_PER_ITER:
                    problems.append(
                        f"{variant}/{mode}/{store} checked {cell['words_checked']} words, not {iters * WORDS_PER_ITER}"
                    )
    if problems:
        return 12, "INCONCLUSIVE: " + "; ".join(problems)
    shows = {v: [k for k, c in cells.items() if k[0] == v and c["stale"] + c["other"] > 0] for v in ("nc", "cv")}
    if shows["cv"]:
        return 11, f"ESCALATE: cv shows staleness in {sorted(shows['cv'])}"
    if shows["nc"]:
        return 10, f"SWITCH TO cv: nc shows staleness in {sorted(shows['nc'])}; cv clean in all eight cells"
    words = iters * WORDS_PER_ITER
    return 0, f"KEEP nc: not observed in {words:.3g} words per cell across the eight nc cells (and the eight cv cells)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--iters", type=int, default=200000)
    args = parser.parse_args()
    code, verdict = decide(load(args.results), args.iters)
    print(verdict)
    sys.exit(code)


if __name__ == "__main__":
    main()
