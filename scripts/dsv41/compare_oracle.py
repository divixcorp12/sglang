"""Run SGLang on the truncated model and compare prompt logprobs with the oracle."""

import argparse
import json

import numpy as np

ACCEPT = {"top1_agree": 0.98, "mean_abs_dlp": 0.05}


def compare_topk(ref_ids, ref_lp, got_ids, got_lp) -> dict:
    agree, overlaps, diffs = [], [], []
    for r_ids, r_lp, g_ids, g_lp in zip(ref_ids, ref_lp, got_ids, got_lp):
        agree.append(int(r_ids[0]) == int(g_ids[0]))
        got = {int(i): float(v) for i, v in zip(g_ids, g_lp)}
        shared = [(float(v), got[int(i)]) for i, v in zip(r_ids, r_lp) if int(i) in got]
        overlaps.append(len(shared) / len(r_ids))
        diffs.extend(abs(a - b) for a, b in shared)
    return {
        "top1_agree": float(np.mean(agree)),
        "mean_overlap": float(np.mean(overlaps)),
        "mean_abs_dlp": float(np.mean(diffs)) if diffs else float("inf"),
        "max_abs_dlp": float(np.max(diffs)) if diffs else float("inf"),
    }


def accepts(metrics: dict) -> bool:
    return metrics["top1_agree"] >= ACCEPT["top1_agree"] and metrics["mean_abs_dlp"] <= ACCEPT["mean_abs_dlp"]


def _sglang_topk(engine, tokens, k):
    out = engine.generate(
        input_ids=tokens,
        sampling_params={"max_new_tokens": 1, "temperature": 0},
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=k,
    )
    # Row 0 is the first prompt token, which has no predictive distribution;
    # row t+1 predicts tokens[t+1] from tokens[:t+1], matching oracle row t.
    rows = out["meta_info"]["input_top_logprobs"][1:]
    ids = np.array([[t[1] for t in r] for r in rows])
    lps = np.array([[t[0] for t in r] for r in rows])
    return ids, lps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=20)
    args = p.parse_args()

    import sglang

    engine = sglang.Engine(
        model_path=args.model,
        tp_size=1,
        disable_cuda_graph=True,
        disable_shared_experts_fusion=True,
        context_length=4096,
        mem_fraction_static=0.8,
    )
    oracle = np.load(args.oracle)
    results = []
    with open(args.prompts) as f:
        for i, line in enumerate(f):
            tokens = json.loads(line)["tokens"]
            got_ids, got_lp = _sglang_topk(engine, tokens, args.k)
            ref_ids = oracle[f"top_ids_{i}"][: len(tokens) - 1]
            ref_lp = oracle[f"top_logprobs_{i}"][: len(tokens) - 1]
            results.append(compare_topk(ref_ids, ref_lp, got_ids, got_lp))
    engine.shutdown()
    summary = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    report = {"per_prompt": results, "mean": summary, "accept": accepts(summary)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["mean"], indent=2), "ACCEPT" if report["accept"] else "REJECT")


if __name__ == "__main__":
    main()
