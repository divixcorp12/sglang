"""Per arm: median decode tok/s and TTFT over turns with >= 64 completion tokens, plus shadow budget recall."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="arm=results.jsonl[:prediction-metrics.jsonl]")
    args = parser.parse_args()
    table = {}
    for spec in args.runs:
        arm, paths = spec.split("=", 1)
        results, _, metrics = paths.partition(":")
        turns = [json.loads(line) for line in open(results) if line.strip()]
        good = [t for t in turns if "error" not in t and (t.get("completion_tokens") or 0) >= 64]
        entry = table.setdefault(arm, {"tok_s": [], "ttft": [], "turns": 0, "errors": 0, "budget_recall": None})
        entry["tok_s"] += [t["decode_tokens_per_sec"] for t in good if t["decode_tokens_per_sec"]]
        entry["ttft"] += [t["ttft"] for t in good if t["ttft"] is not None]
        entry["turns"] += len(good)
        entry["errors"] += sum("error" in t for t in turns)
        if metrics and Path(metrics).exists():
            records = [json.loads(line) for line in open(metrics) if '"prefetch"' in line]
            if records:
                entry["budget_recall"] = records[-1]["prefetch"]["budget_recall"]
    summary = {arm: {"median_decode_tok_s": statistics.median(e["tok_s"]) if e["tok_s"] else None,
                     "median_ttft_s": statistics.median(e["ttft"]) if e["ttft"] else None,
                     "turns": e["turns"], "errors": e["errors"], "budget_recall": e["budget_recall"]}
               for arm, e in table.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
