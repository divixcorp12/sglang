"""Replay acceptance predictors built from the draft's own probabilities against the confidence head, over an STS
collection that recorded them (StsDataRecorder's draft_q / draft_top1 / draft_entropy).

Each predictor is a logistic model of draft k's acceptance given the drafts before it were accepted, fitted to the
recorded accept_rates (min(1, p/q) for the drawn draft: the expected label, defined at every position), on one
contiguous half of the verifies and replayed on the other, then swapped. Per position ("per-position") or one model
over all positions with a one-hot position ("shared"). Features at position k, all computed by the draft before
the verify:
  head  -- the confidence head's pre-STS logit (reads d_{k-1});
  q     -- logit q_k(d_k), the draft's probability of the token it drew (reads d_k);
  top1  -- logit max q_k (reads d_{k-1});
  ent   -- the entropy of q_k (reads d_{k-1}).
The replay is sts_cut_replay.py's: the stopping rule over the predicted confidences, a cut verify emitting
min(m, w - 1) + 1 tokens for its recorded accept count m, at T(w) = a + b (w - 1) plus --draft-ms. Next to the
predictors: every row, the head with an STS file, the expected oracle (confidence = accept_rates) and the best constant
width. ECE per position is the survival ECE against the accept's outcome (dspark_sts_fit's metric), on the held-out
halves.

Usage: accept_predictor_replay.py --data-glob 'OUT/sts/confidence.*.pt' --sts STS.json --temperature 0.6
                                  [--cost 91.2,21.1 --cost 226.6,16.8] [--draft-ms 5.8] [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json

import torch

from sglang.benchmark.dspark_sts_fit import expected_calibration_error
from sglang.srt.speculative.dspark_components.dspark_confidence_cut import stop_rule_verify_lens
from sglang.srt.speculative.dspark_components.dspark_sts import load_sts_calibration_from_path

EPS = 1e-6
FEATURE_SETS = {
    "head": ("head",),
    "q": ("q",),
    "q+top1+ent": ("q", "top1", "ent"),
    "head+q+top1+ent": ("head", "q", "top1", "ent"),
    "head+top1+ent (no d_k)": ("head", "top1", "ent"),
}


def _logit(p: torch.Tensor) -> torch.Tensor:
    return torch.logit(p.clamp(EPS, 1 - EPS))


def load(data_glob: str, temperature: float) -> dict[str, torch.Tensor]:
    shards = [torch.load(path, map_location="cpu") for path in sorted(glob.glob(data_glob))]
    if not shards or "draft_q" not in shards[0]:
        raise SystemExit(f"{data_glob!r}: no shards with draft features")
    data = {key: torch.cat([s[key] for s in shards]) for key in shards[0]}
    keep = ((data["temperatures"] - temperature).abs() < 1e-3) & torch.isfinite(data["draft_q"]).all(dim=1)
    data = {key: value[keep] for key, value in data.items()}
    data["taken"] = data["sampled_prefix_mask"].sum(dim=1).to(torch.int64)
    return data


def features(data: dict, names: tuple[str, ...]) -> torch.Tensor:
    """[n, gamma, len(names)] float64."""
    columns = {
        "head": data["logits"].double(),
        "q": _logit(data["draft_q"].double()),
        "top1": _logit(data["draft_top1"].double()),
        "ent": data["draft_entropy"].double(),
    }
    return torch.stack([columns[name] for name in names], dim=-1)


def fit_logistic(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3, steps: int = 50) -> torch.Tensor:
    """Newton (IRLS) on the soft-label log loss with an intercept; x [n, f], y [n] in [0, 1]. Returns [f + 1]."""
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
    beta = torch.zeros(x.shape[1], dtype=x.dtype)
    reg = ridge * torch.eye(x.shape[1], dtype=x.dtype)
    reg[-1, -1] = 0.0
    for _ in range(steps):
        p = torch.sigmoid(x @ beta)
        grad = x.T @ (p - y) / x.shape[0] + reg @ beta
        hess = (x.T * (p * (1 - p))) @ x / x.shape[0] + reg
        step = torch.linalg.solve(hess, grad)
        beta = beta - step
        if float(step.abs().max()) < 1e-8:
            break
    return beta


def predict(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x @ beta[:-1] + beta[-1])


def fit_predict(x_train, y_train, x_test, mode: str) -> torch.Tensor:
    """[n_test, gamma] predicted confidences."""
    gamma = x_train.shape[1]
    if mode == "per-position":
        return torch.stack(
            [predict(x_test[:, k], fit_logistic(x_train[:, k], y_train[:, k])) for k in range(gamma)], dim=1
        )
    eye = torch.eye(gamma, dtype=x_train.dtype)

    def with_position(x):
        n = x.shape[0]
        return torch.cat([x, eye.expand(n, gamma, gamma)], dim=-1).reshape(n * gamma, -1)

    beta = fit_logistic(with_position(x_train), y_train.reshape(-1))
    # The intercept column and the one-hot are collinear; the ridge picks the split.
    return predict(with_position(x_test), beta).view(x_test.shape[0], gamma)


def replay(confidence: torch.Tensor, taken: torch.Tensor, cost: torch.Tensor, draft_ms: float) -> dict:
    widths = stop_rule_verify_lens(confidence=confidence.float(), cost_ms=(cost + draft_ms).float())
    return evaluate(widths, taken, cost, draft_ms)


def evaluate(widths: torch.Tensor, taken: torch.Tensor, cost: torch.Tensor, draft_ms: float) -> dict:
    tokens = torch.minimum(taken, widths - 1) + 1
    ms = cost[widths - 1] + draft_ms
    return {
        "ms_per_token": round(float(ms.sum() / tokens.sum()), 2),
        "tokens_per_verify": round(float(tokens.double().mean()), 3),
        "mean_width": round(float(widths.double().mean()), 2),
    }


def survival_ece(confidence: torch.Tensor, prefix_mask: torch.Tensor) -> list[float]:
    survival = torch.cumprod(confidence.double(), dim=1)
    return [
        round(expected_calibration_error(probs=survival[:, k], targets=prefix_mask[:, k], num_bins=15), 4)
        for k in range(confidence.shape[1])
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--sts", required=True, help="the shipped STS file, for the head's row")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--cost", action="append", default=None, help="a,b; repeatable")
    p.add_argument("--draft-ms", type=float, default=0.0)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    data = load(a.data_glob, a.temperature)
    n, gamma = data["logits"].shape
    taken, rates = data["taken"], data["accept_rates"].double()
    prefix_mask = data["sampled_prefix_mask"].double()
    halves = [torch.arange(n) < n // 2, torch.arange(n) >= n // 2]

    # Held-out confidences: each half predicted by the model fitted on the other.
    held_out: dict[str, torch.Tensor] = {}
    for name, names in FEATURE_SETS.items():
        x = features(data, names)
        for mode in ("per-position", "shared"):
            confidence = torch.empty((n, gamma), dtype=torch.float64)
            for test, train in (halves, halves[::-1]):
                confidence[test] = fit_predict(x[train], rates[train], x[test], mode)
            held_out[f"{name} [{mode}]"] = confidence
    temperatures = torch.tensor(load_sts_calibration_from_path(a.sts).temperatures, dtype=torch.float64)
    shipped = torch.sigmoid(data["logits"].double() / temperatures)

    report = {"verifies": n, "draft_ms": a.draft_ms, "ece": {}, "fits": {}}
    report["ece"]["2 head, shipped STS"] = survival_ece(shipped, prefix_mask)
    report["ece"]["3 expected oracle"] = survival_ece(rates, prefix_mask)
    for name, confidence in held_out.items():
        report["ece"][name] = survival_ece(confidence, prefix_mask)
    for spec in a.cost or ["91.2,21.1", "226.6,16.8"]:
        base, per_row = (float(v) for v in spec.split(","))
        cost = torch.tensor([base + per_row * w for w in range(gamma + 1)], dtype=torch.float64)
        rows = {
            "1 every row": evaluate(torch.full((n,), gamma + 1), taken, cost, a.draft_ms),
            "2 head, shipped STS": replay(shipped, taken, cost, a.draft_ms),
            "3 expected oracle": replay(rates, taken, cost, a.draft_ms),
            "4 realized oracle": evaluate(taken + 1, taken, cost, a.draft_ms),
        }
        constant = {w: evaluate(torch.full((n,), w), taken, cost, a.draft_ms) for w in range(1, gamma + 2)}
        best = min(constant, key=lambda w: constant[w]["ms_per_token"])
        rows[f"5 best constant (w={best})"] = constant[best]
        for name, confidence in held_out.items():
            rows[name] = replay(confidence, taken, cost, a.draft_ms)
        report["fits"][spec] = rows
    text = json.dumps(report, indent=1)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
