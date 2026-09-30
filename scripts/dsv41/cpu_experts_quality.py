"""CPU experts' quality gate: two logprob_probe.py outputs (CPU off = base, CPU on = other) of the same prompts.

The pre-registered gate is E31 (compare_logprobs.py): every greedy flip must start at a near-tie, a top-2 margin of
at most 0.375 nats in either run. Also reported, for the owner to judge:
- the greedy-match rate (prompts whose tokens are identical over the shorter of the two completions);
- each flip's position;
- the per-token KL(off || on) at every position whose context is still shared (up to and including the first flip),
  over the tokens both runs list in their top-k, each renormalised there -- teacher-forced up to the divergence;
- the absolute change in the off run's chosen token's logprob, where the on run lists it.
"""

import argparse
import json
import math

NEAR_TIE_NATS = 0.375


def _margin(entry):
    top = entry["top"]
    return top[0][1] - top[1][1] if len(top) > 1 else float("inf")


def _kl(p_top, q_top):
    q = dict(q_top)
    shared = [(token, lp) for token, lp in p_top if token in q]
    if not shared:
        return None
    p_norm = math.log(sum(math.exp(lp) for _, lp in shared))
    q_norm = math.log(sum(math.exp(q[token]) for token, _ in shared))
    return sum(math.exp(lp - p_norm) * ((lp - p_norm) - (q[token] - q_norm)) for token, lp in shared)


def compare(base, other):
    if [a["session_id"] for a in base] != [b["session_id"] for b in other]:
        raise ValueError("the two probes ran different prompts")
    prompts, kls, deltas = [], [], []
    for a, b in zip(base, other):
        flip = None
        for index, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
            kl = _kl(x["top"], y["top"])
            if kl is not None:
                kls.append(kl)
            off, on = dict(x["top"]), dict(y["top"])
            if x["token"] in off and x["token"] in on:
                deltas.append(abs(off[x["token"]] - on[x["token"]]))
            if x["token"] != y["token"]:
                margins = [_margin(x), _margin(y)]
                flip = {"index": index, "margins": margins, "near_tie": min(margins) <= NEAR_TIE_NATS}
                break
        prompts.append({"session_id": a["session_id"], "compared": min(len(a["tokens"]), len(b["tokens"])),
                        "flip": flip})
    flips = [p for p in prompts if p["flip"] is not None]
    return {
        "prompts": len(prompts),
        "greedy_match_rate": 1 - len(flips) / len(prompts) if prompts else None,
        "flip_positions": [p["flip"]["index"] for p in flips],
        "e31_pass": all(p["flip"]["near_tie"] for p in flips),
        "kl_positions": len(kls),
        "kl_mean": sum(kls) / len(kls) if kls else None,
        "kl_max": max(kls) if kls else None,
        "chosen_logprob_abs_delta_mean": sum(deltas) / len(deltas) if deltas else None,
        "chosen_logprob_abs_delta_max": max(deltas) if deltas else None,
        "per_prompt": prompts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", help="CPU experts off")
    parser.add_argument("other", help="CPU experts on")
    args = parser.parse_args()
    result = compare(json.load(open(args.base)), json.load(open(args.other)))
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result["e31_pass"] else 1)


if __name__ == "__main__":
    main()
