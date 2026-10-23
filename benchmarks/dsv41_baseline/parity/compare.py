"""Compares two parity dumps: mean per-token KL(base || branch), top-1 agreement, worst prompts.

Usage: compare.py BASE_DUMP_DIR BRANCH_DUMP_DIR [--max-kl 1e-3] [--min-top1 0.995] [--worst 5]
Exit 0 when both thresholds hold, 1 when one fails, 2 when the dumps are not comparable.

The dumps hold only the top-K (K=20) of each distribution, so KL is approximated on the union U of the two top-K
supports plus one lumped "other" bucket, per position:
  - P (base) has top-K probabilities p_i and tail mass t_P = max(1 - sum p_i, 0); Q (branch) likewise.
  - An id in U but outside P's top-K has unknown P-probability, at most P's K-th probability. It is given
    min(p_K, t_P / n_missing), so the assigned mass never exceeds the tail; Q is treated the same way.
  - What is left of each tail (t - assigned) forms the "other" bucket on that side.
  - KL = sum over U and the other bucket of p * log(p / q), with 1e-12 floors against log(0).
The approximation is exact when both supports are equal and the tail is empty. Otherwise it ignores how the tail
splits internally, so it is an estimate, not a bound. Chosen-token logprob deltas (exact, no top-K) are reported next to it as a cross-check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import dump_format

EPS = 1e-12
EXIT_PASS, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2


class NotComparable(Exception):
    pass


def _side_probs(*, own_ids: np.ndarray, own_lp: np.ndarray, union: list[int], n_missing: int) -> tuple[np.ndarray, float]:
    """Probabilities of `union` ids under one side, and its remaining other-bucket mass."""
    valid = own_ids >= 0
    probs = np.exp(own_lp[valid].astype(np.float64))
    known = dict(zip(own_ids[valid].tolist(), probs.tolist()))
    tail = max(1.0 - float(probs.sum()), 0.0)
    floor = float(probs.min()) if probs.size else 0.0
    each = min(floor, tail / n_missing) if n_missing else 0.0
    values = np.array([known.get(i, each) for i in union], dtype=np.float64)
    return values, max(tail - each * n_missing, 0.0)


def position_kl(*, base_ids: np.ndarray, base_lp: np.ndarray, branch_ids: np.ndarray, branch_lp: np.ndarray) -> float:
    base_set = set(base_ids[base_ids >= 0].tolist())
    branch_set = set(branch_ids[branch_ids >= 0].tolist())
    union = sorted(base_set | branch_set)
    p, p_other = _side_probs(own_ids=base_ids, own_lp=base_lp, union=union, n_missing=len(branch_set - base_set))
    q, q_other = _side_probs(own_ids=branch_ids, own_lp=branch_lp, union=union, n_missing=len(base_set - branch_set))
    p = np.append(p, p_other)
    q = np.append(q, q_other)
    live = p > 0
    return float(np.sum(p[live] * (np.log(np.maximum(p[live], EPS)) - np.log(np.maximum(q[live], EPS)))))


def prompt_stats(base: dict[str, np.ndarray], branch: dict[str, np.ndarray]) -> dict:
    count = base["tokens"].shape[0]
    kls = np.array(
        [
            position_kl(
                base_ids=base["top_ids"][j],
                base_lp=base["top_lp"][j],
                branch_ids=branch["top_ids"][j],
                branch_lp=branch["top_lp"][j],
            )
            for j in range(count)
        ]
    )
    agree = base["top_ids"][:, 0] == branch["top_ids"][:, 0]
    chosen_delta = np.abs(base["chosen_lp"].astype(np.float64) - branch["chosen_lp"].astype(np.float64))
    return {
        "tokens": count,
        "kl_sum": float(kls.sum()),
        "kl_max": float(kls.max()) if count else 0.0,
        "top1_agree": int(agree.sum()),
        "chosen_abs_delta_sum": float(chosen_delta.sum()),
    }


def check_comparable(*, base_manifest: dict, branch_manifest: dict, base: dict, branch: dict) -> None:
    key = "prompt_set_sha256"
    if base_manifest.get(key) != branch_manifest.get(key):
        raise NotComparable(f"prompt sets differ: {base_manifest.get(key)} vs {branch_manifest.get(key)}")
    if set(base) != set(branch):
        only = sorted(set(base) ^ set(branch))
        raise NotComparable(f"prompt ids differ: {only[:5]}")
    for prompt_id in sorted(base):
        if not np.array_equal(base[prompt_id]["tokens"], branch[prompt_id]["tokens"]):
            raise NotComparable(f"token ids differ for prompt {prompt_id}")
        if base[prompt_id]["top_ids"].shape[0] != base[prompt_id]["tokens"].shape[0]:
            raise NotComparable(f"base prompt {prompt_id}: top_ids rows do not match tokens")
        if branch[prompt_id]["top_ids"].shape[0] != branch[prompt_id]["tokens"].shape[0]:
            raise NotComparable(f"branch prompt {prompt_id}: top_ids rows do not match tokens")


def compare(*, base_dir: Path, branch_dir: Path) -> dict:
    base_manifest, base = dump_format.load_dump(base_dir)
    branch_manifest, branch = dump_format.load_dump(branch_dir)
    check_comparable(base_manifest=base_manifest, branch_manifest=branch_manifest, base=base, branch=branch)
    per_prompt = {pid: prompt_stats(base[pid], branch[pid]) for pid in sorted(base)}
    total = sum(s["tokens"] for s in per_prompt.values())
    return {
        "prompts": len(per_prompt),
        "tokens": total,
        "mean_kl": sum(s["kl_sum"] for s in per_prompt.values()) / total,
        "top1_agreement": sum(s["top1_agree"] for s in per_prompt.values()) / total,
        "mean_chosen_abs_delta": sum(s["chosen_abs_delta_sum"] for s in per_prompt.values()) / total,
        "per_prompt": per_prompt,
        "same_sglang_file": base_manifest.get("sglang_file") == branch_manifest.get("sglang_file"),
        "base_commit": base_manifest.get("git_commit"),
        "branch_commit": branch_manifest.get("git_commit"),
    }


def render(*, result: dict, max_kl: float, min_top1: float, worst: int) -> str:
    kl_ok = result["mean_kl"] <= max_kl
    top1_ok = result["top1_agreement"] >= min_top1
    lines = [
        f"base {result['base_commit']}  branch {result['branch_commit']}",
        f"{result['prompts']} prompts, {result['tokens']} tokens",
        f"{'metric':<26}{'value':>14}{'threshold':>14}  result",
        f"{'mean KL(base||branch)':<26}{result['mean_kl']:>14.3e}{'<= ' + format(max_kl, '.1e'):>14}  {'PASS' if kl_ok else 'FAIL'}",
        f"{'top-1 agreement':<26}{result['top1_agreement']:>14.4f}{'>= ' + format(min_top1, '.4f'):>14}  {'PASS' if top1_ok else 'FAIL'}",
        f"{'mean |chosen lp delta|':<26}{result['mean_chosen_abs_delta']:>14.3e}",
        "",
        f"worst {worst} prompts by mean KL:",
    ]
    ranked = sorted(result["per_prompt"].items(), key=lambda kv: kv[1]["kl_sum"] / kv[1]["tokens"], reverse=True)
    for prompt_id, s in ranked[:worst]:
        lines.append(
            f"  {prompt_id:<12} mean KL {s['kl_sum'] / s['tokens']:.3e}  max KL {s['kl_max']:.3e}  "
            f"top-1 {s['top1_agree'] / s['tokens']:.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path)
    parser.add_argument("branch", type=Path)
    parser.add_argument("--max-kl", type=float, default=1e-3)
    parser.add_argument("--min-top1", type=float, default=0.995)
    parser.add_argument("--worst", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = compare(base_dir=args.base, branch_dir=args.branch)
    except NotComparable as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED
    if result["same_sglang_file"]:
        print("WARNING: both dumps imported sglang from the same path; they may be one tree", file=sys.stderr)
    print(render(result=result, max_kl=args.max_kl, min_top1=args.min_top1, worst=args.worst))
    passed = result["mean_kl"] <= args.max_kl and result["top1_agreement"] >= args.min_top1
    return EXIT_PASS if passed else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
