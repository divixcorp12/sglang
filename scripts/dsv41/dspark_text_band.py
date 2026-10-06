"""The DSpark text bar (DSV41_REFERENCE.md §33.2, §33.9): exact text is the wrong test for a speculative run, because
its verify forward settles near-ties differently from one-token decode. Of two logprob_probe.py outputs of the same
prompts (scripts/expert_prediction/prefetch/logprob_probe.py, --top-logprobs 5), the other run may leave the base run
only where its token is within `band` nats of the base's argmax at the first divergence. 1.4 is §33.2's
verify-vs-decode band; §33.9's largest draft-to-draft divergence was 1.125.

    python scripts/dsv41/dspark_text_band.py BASE.json OTHER.json [--band 1.4]
"""

import argparse
import json

BAND_NATS = 1.4


def compare(base, other, band: float = BAND_NATS) -> dict:
    if [a["session_id"] for a in base] != [b["session_id"] for b in other]:
        raise ValueError("the two probes ran different prompts")
    flips, compared = [], 0
    for a, b in zip(base, other):
        for index, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
            compared += 1
            if x["token"] == y["token"]:
                continue
            top = dict(x["top"])
            gap = round(x["top"][0][1] - top[y["token"]], 6) if y["token"] in top else float("inf")
            flips.append({"session_id": a["session_id"], "index": index, "gap": gap, "within": gap <= band})
            break
    return {
        "prompts": len(base),
        "compared_tokens": compared,
        "flips": flips,
        "max_gap": max((f["gap"] for f in flips), default=0.0),
        "pass": all(f["within"] for f in flips),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--band", type=float, default=BAND_NATS)
    args = parser.parse_args()
    with open(args.base) as f:
        base = json.load(f)
    with open(args.other) as f:
        other = json.load(f)
    report = compare(base, other, args.band)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
