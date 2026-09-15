"""Fixed live A/B subset: the first FinanceBench holdout and ConvFinQA val sessions that fit the time budget."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--val", type=int, default=6)
    parser.add_argument("--max-context-chars", type=int, default=80000)
    args = parser.parse_args()
    picked, counts = [], {"holdout": 0, "val": 0}
    limits = {"holdout": args.holdout, "val": args.val}
    with open(args.sessions) as f:
        for line in f:
            session = json.loads(line)
            split = session["split"]
            if split in limits and counts[split] < limits[split] and session["context_chars"] <= args.max_context_chars:
                picked.append(session)
                counts[split] += 1
    with open(args.out, "w") as f:
        for session in picked:
            f.write(json.dumps(session) + "\n")
    print(json.dumps({"sessions": len(picked), "turns": sum(len(s["turns"]) for s in picked), **counts}))


if __name__ == "__main__":
    main()
