"""Compare per-test outcomes of the same registered test file run in three class orders (see order_sweep.sh)."""
import collections
import glob
import json
import os
import re
import sys

LINE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)")


def parse(path):
    outcomes, exit_code = {}, None
    for raw in open(path, errors="replace"):
        m = LINE.match(raw)
        if m:
            outcomes[m.group(2)] = m.group(1)
        elif raw.startswith("exit="):
            exit_code = int(raw.split("=")[1])
    return outcomes, exit_code


def main(logs, files):
    report = collections.OrderedDict()
    for f in files:
        base = f.replace("/", "_")
        runs = {}
        for order in ("alpha", "file", "rev"):
            p = os.path.join(logs, f"{base}.{order}")
            if os.path.exists(p):
                runs[order] = parse(p)
        if len(runs) < 3:
            report[f] = {"status": "incomplete", "have": sorted(runs)}
            continue
        if any(code == 124 for _, code in runs.values()):
            report[f] = {"status": "timeout"}
            continue
        ids = set().union(*(set(o) for o, _ in runs.values()))
        if not ids:
            report[f] = {"status": "no results (collection error?)", "exit": {k: v[1] for k, v in runs.items()}}
            continue
        diffs = {}
        for i in sorted(ids):
            res = {order: runs[order][0].get(i, "MISSING") for order in runs}
            if len(set(res.values())) > 1:
                diffs[i] = res
        counts = {order: collections.Counter(runs[order][0].values()) for order in runs}
        bad = {i for i in ids if runs["file"][0].get(i) in ("FAILED", "ERROR")}
        status = "hit" if diffs else ("stable_all_pass" if not bad else "stable_fails_identically")
        report[f] = {"status": status, "tests": len(ids), "diffs": diffs,
                     "failed_any_order": sorted(i for i in ids if any(runs[o][0].get(i) in ("FAILED", "ERROR") for o in runs))[:5],
                     "counts": {o: dict(c) for o, c in counts.items()}}
    return report


if __name__ == "__main__":
    logs, filelist = sys.argv[1], sys.argv[2]
    files = [l.strip() for l in open(filelist) if l.strip()]
    out = main(logs, files)
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    tally = collections.Counter(v["status"] for v in out.values())
    print("files:", len(out), dict(tally))
    dangerous, annoying, other = [], [], []
    for f, v in out.items():
        if v["status"] != "hit":
            continue
        for i, res in v["diffs"].items():
            if res["file"] == "PASSED" and res["alpha"] in ("FAILED", "ERROR"):
                dangerous.append((f, i, res))
            elif res["alpha"] == "PASSED" and res["file"] in ("FAILED", "ERROR"):
                annoying.append((f, i, res))
            else:
                other.append((f, i, res))
    print("DANGEROUS (passes in pytest file order, fails in CI alphabetical order):", len(dangerous))
    for d in dangerous: print("  ", d)
    print("passes in CI order, fails in pytest file order:", len(annoying))
    for d in annoying: print("  ", d)
    print("other differing:", len(other))
    for d in other[:60]: print("  ", d)
