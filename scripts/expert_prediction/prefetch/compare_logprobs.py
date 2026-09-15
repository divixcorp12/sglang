"""Compare two arms' greedy tokens: --exact fails on any flip; otherwise flips must start at a near-tie (E31: top-2 margin <= 0.375 nats)."""

import argparse
import json

NEAR_TIE_NATS = 0.375


def _margin(entry):
    top = entry["top"]
    return top[0][1] - top[1][1] if len(top) > 1 else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()
    base, other = json.load(open(args.base)), json.load(open(args.other))
    failures = []
    for a, b in zip(base, other):
        for index, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
            if x["token"] != y["token"]:
                if args.exact or min(_margin(x), _margin(y)) > NEAR_TIE_NATS:
                    failures.append({"session_id": a["session_id"], "index": index,
                                     "margins": [_margin(x), _margin(y)]})
                break
    print(json.dumps({"compared": len(base), "flips": failures}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
