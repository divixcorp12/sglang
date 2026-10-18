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
import os
from typing import Optional

# CPU only: run as a script, hide the GPU before torch (and sglang's imports) can open a CUDA context outside
# cc-gpu.lock. (Imported by a test, the caller's devices are left alone.)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

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


def _shard_index(path: str) -> int:
    return int(path.rsplit(".", 2)[-2])


def load(data_glob: str, temperature: float, *, need_draft: bool = True) -> dict[str, torch.Tensor]:
    """The shards' verifies at ``temperature``, in serving order (shards sorted by their index, rows as recorded)."""
    paths = sorted(glob.glob(data_glob), key=_shard_index)
    shards = [torch.load(path, map_location="cpu") for path in paths]
    if not shards or (need_draft and "draft_q" not in shards[0]):
        raise SystemExit(f"{data_glob!r}: no shards" + (" with draft features" if need_draft else ""))
    data = {key: torch.cat([s[key] for s in shards]) for key in shards[0]}
    keep = ((data["temperatures"] - temperature).abs() < 1e-3) & torch.isfinite(data["logits"]).all(dim=1)
    if "draft_q" in data:
        keep &= torch.isfinite(data["draft_q"]).all(dim=1)
    data = {key: value[keep] for key, value in data.items()}
    data["taken"] = data["sampled_prefix_mask"].sum(dim=1).to(torch.int64)
    return data


def features(data: dict, names: tuple[str, ...]) -> torch.Tensor:
    """[n, gamma, len(names)] float64."""
    columns = {
        "head": lambda: data["logits"].double(),
        "q": lambda: _logit(data["draft_q"].double()),
        "top1": lambda: _logit(data["draft_top1"].double()),
        "ent": lambda: data["draft_entropy"].double(),
    }
    return torch.stack([columns[name]() for name in names], dim=-1)


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


# ---- target-side history features (--history) ------------------------------------------------------------------------
#
# What the server knows about the target before step t's width is chosen: earlier verifies' outcomes as the cut saw
# them. Under a cut of w rows, positions 1..w ran on live rows, so a step's observed accept count is min(m, w) (the
# capped commit plus cap_trim_lens, up to w), it was "cut short" when every verified position accepted (m >= w), and
# min(1, p/q) is known at the positions whose accept test ran (1..min(m + 1, w)). Every feature of step t reads steps
# before t only (HistoryState takes step t's outcome after its features were read), never step t's target p or its
# accept draws, so it is a constant of step t's verify: c_k still reads only this step's d_1..d_k plus the past, and the
# stopping-rule proof holds unchanged.

HISTORY_DECAYS = (0.5, 0.8, 0.95)
HISTORY_NAMES = (
    "no_history",
    "prev_accept",
    "prev_cut_short",
    "prev_mean_rate",
    "prev_last_accept_rate",
    *(f"ewma_accept_{d}" for d in HISTORY_DECAYS),
    "ewma_mean_rate_0.8",
)


class HistoryState:
    """The history features of the next step, from the steps fed to update() so far."""

    def __init__(self, decays=HISTORY_DECAYS):
        self.decays = decays
        self.steps = 0
        self.prev = (0.0, 0.0, 0.0, 0.0)
        self.ewma_accept = [0.0] * len(decays)
        self.ewma_rate = 0.0
        # With target rows (TARGET_HISTORY_NAMES): the previous bonus row's entropy and top-1, and an EWMA of the entropy.
        self.prev_bonus = (0.0, 0.0)
        self.ewma_bonus_entropy = 0.0

    def features(self, *, with_target: bool = False) -> list[float]:
        base = [float(self.steps == 0), *self.prev, *self.ewma_accept, self.ewma_rate]
        return base + [*self.prev_bonus, self.ewma_bonus_entropy] if with_target else base

    def update(
        self,
        *,
        taken: int,
        width: int,
        rates: list[float],
        target_entropy: Optional[list[float]] = None,
        target_top1: Optional[list[float]] = None,
    ) -> None:
        observed = min(taken, width)
        tested = min(taken + 1, width)
        mean_rate = sum(rates[:tested]) / tested
        last_rate = rates[observed - 1] if observed >= 1 else 0.0
        self.prev = (float(observed), float(taken >= width), mean_rate, last_rate)
        if target_entropy is not None:
            # The bonus was drawn from row min(m, w - 1), a live row.
            row = min(taken, width - 1)
            self.prev_bonus = (target_entropy[row], target_top1[row])
        if self.steps == 0:
            self.ewma_accept = [float(observed)] * len(self.decays)
            self.ewma_rate = mean_rate
            self.ewma_bonus_entropy = self.prev_bonus[0]
        else:
            self.ewma_accept = [d * e + (1 - d) * observed for d, e in zip(self.decays, self.ewma_accept)]
            self.ewma_rate = 0.8 * self.ewma_rate + 0.2 * mean_rate
            self.ewma_bonus_entropy = 0.8 * self.ewma_bonus_entropy + 0.2 * self.prev_bonus[0]
        self.steps += 1


TARGET_HISTORY_NAMES = (*HISTORY_NAMES, "prev_bonus_entropy", "prev_bonus_top1", "ewma_bonus_entropy_0.8")


def _target_rows(data: dict, with_target: bool):
    if not with_target:
        return None, None
    return data["target_entropy"].double().tolist(), data["target_top1"].double().tolist()


def has_target_rows(data: dict) -> bool:
    return "target_entropy" in data and bool(torch.isfinite(data["target_entropy"]).all())


def history_features(
    *,
    taken: torch.Tensor,
    widths: torch.Tensor,
    rates: torch.Tensor,
    target_entropy: Optional[torch.Tensor] = None,
    target_top1: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[n, len(HISTORY_NAMES)] float64 (TARGET_HISTORY_NAMES with the target rows): row t from steps 0..t-1 (their
    widths ``widths``) only."""
    with_target = target_entropy is not None
    state, rows = HistoryState(), []
    rate_lists = rates.double().tolist()
    entropy_rows = target_entropy.double().tolist() if with_target else None
    top1_rows = target_top1.double().tolist() if with_target else None
    for t in range(taken.shape[0]):
        rows.append(state.features(with_target=with_target))
        state.update(
            taken=int(taken[t]), width=int(widths[t]), rates=rate_lists[t],
            target_entropy=entropy_rows[t] if with_target else None,
            target_top1=top1_rows[t] if with_target else None,
        )
    return torch.tensor(rows, dtype=torch.float64)


def _with_history(x: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """[n, gamma, f] step features + [n, h] history -> [n, gamma, f + h]."""
    n, gamma = x.shape[:2]
    return torch.cat([x, history[:, None, :].expand(n, gamma, history.shape[1])], dim=-1)


def fit_per_position(x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    return [fit_logistic(x[:, k], y[:, k]) for k in range(x.shape[1])]


def replay_sequential(
    *,
    x: torch.Tensor,
    betas: list[torch.Tensor],
    data: dict,
    cost: torch.Tensor,
    draft_ms: float,
    with_target: bool = False,
) -> tuple[dict, torch.Tensor]:
    """The cut run step by step with history features from its own earlier widths: what a server would see. Returns
    the replay and the [n, gamma] confidences it read."""
    n, gamma = x.shape[:2]
    taken, rate_lists = data["taken"], data["accept_rates"].double().tolist()
    entropy_rows, top1_rows = _target_rows(data, with_target)
    state = HistoryState()
    widths = torch.empty(n, dtype=torch.int64)
    confidences = torch.empty((n, gamma), dtype=torch.float64)
    objective = (cost + draft_ms).float()
    for t in range(n):
        h = torch.tensor(state.features(with_target=with_target), dtype=torch.float64)
        row = torch.cat([x[t], h[None, :].expand(gamma, h.numel())], dim=-1)
        c = torch.stack([predict(row[k], betas[k]) for k in range(gamma)])
        confidences[t] = c
        widths[t] = int(stop_rule_verify_lens(confidence=c[None].float(), cost_ms=objective)[0])
        state.update(
            taken=int(taken[t]), width=int(widths[t]), rates=rate_lists[t],
            target_entropy=entropy_rows[t] if with_target else None,
            target_top1=top1_rows[t] if with_target else None,
        )
    return evaluate(widths, taken, cost, draft_ms), confidences


def history_report(*, data: dict, other: Optional[dict], shipped_temperatures, cost, draft_ms) -> dict:
    """The --history rows on the two splits: halves of ``data`` (each predicted by the model fitted on the other), and
    across runs (fit on ``other``, replay on ``data``) for the features both collections have."""
    gamma = data["logits"].shape[1]
    objective = (cost + draft_ms).float()

    def shipped_widths(d):
        c = torch.sigmoid(d["logits"].double() / shipped_temperatures)
        return stop_rule_verify_lens(confidence=c.float(), cost_ms=objective), c

    def train(d, names, history_widths, with_target=False):
        x = features(d, names) if names else torch.empty((d["logits"].shape[0], gamma, 0), dtype=torch.float64)
        h = history_features(
            taken=d["taken"], widths=history_widths, rates=d["accept_rates"],
            target_entropy=d["target_entropy"] if with_target else None,
            target_top1=d["target_top1"] if with_target else None,
        )
        return fit_per_position(_with_history(x, h), d["accept_rates"].double())

    def step_x(d, names):
        return features(d, names) if names else torch.empty((d["logits"].shape[0], gamma, 0), dtype=torch.float64)

    # name -> (step features or None for the no-history draft predictor, whether the history has the target rows)
    variants = {
        "history only": ((), False),
        "history + head": (("head",), False),
        "history + head+q+top1+ent": (("head", "q", "top1", "ent"), False),
        "head+q+top1+ent (no history)": (None, False),
    }
    if has_target_rows(data):
        variants.update({
            "target-row history only": ((), True),
            "target-row history + head+q+top1+ent": (("head", "q", "top1", "ent"), True),
        })
    out = {"halves": {}, "across_runs": {}, "ece": {}}
    n = data["logits"].shape[0]
    halves = [torch.arange(n) < n // 2, torch.arange(n) >= n // 2]
    prefix_mask = data["sampled_prefix_mask"].double()

    def sub(d, mask):
        return {k: v[mask] for k, v in d.items()}

    # Halves: train on one half with the shipped cut's widths as its history, replay the other half sequentially.
    for name, (names, with_target) in variants.items():
        tokens = ms = 0.0
        confidence = torch.empty((n, gamma), dtype=torch.float64)
        for test, trn in (halves, halves[::-1]):
            dtest, dtrain = sub(data, test), sub(data, trn)
            if names is None:
                x = features(dtrain, ("head", "q", "top1", "ent"))
                betas = fit_per_position(x, dtrain["accept_rates"].double())
                c = torch.stack(
                    [predict(features(dtest, ("head", "q", "top1", "ent"))[:, k], betas[k]) for k in range(gamma)],
                    dim=1,
                )
                widths = stop_rule_verify_lens(confidence=c.float(), cost_ms=objective)
            else:
                betas = train(dtrain, names, shipped_widths(dtrain)[0], with_target=with_target)
                _, c = replay_sequential(
                    x=step_x(dtest, names), betas=betas, data=dtest, cost=cost, draft_ms=draft_ms,
                    with_target=with_target,
                )
                widths = stop_rule_verify_lens(confidence=c.float(), cost_ms=objective)
            confidence[test] = c
            got = torch.minimum(dtest["taken"], widths - 1) + 1
            tokens += float(got.sum())
            ms += float((cost[widths - 1] + draft_ms).sum())
        out["halves"][name] = {"ms_per_token": round(ms / tokens, 2)}
        out["ece"][name] = survival_ece(confidence, prefix_mask)
    # The optimistic bound: history as if every earlier verify had run every row (not observable under a cut).
    tokens = ms = 0.0
    for test, trn in (halves, halves[::-1]):
        dtest, dtrain = sub(data, test), sub(data, trn)
        full = lambda d: torch.full((d["logits"].shape[0],), gamma + 1)  # noqa: E731
        betas = train(dtrain, ("head", "q", "top1", "ent"), full(dtrain))
        h = history_features(taken=dtest["taken"], widths=full(dtest), rates=dtest["accept_rates"])
        xt = _with_history(features(dtest, ("head", "q", "top1", "ent")), h)
        c = torch.stack([predict(xt[:, k], betas[k]) for k in range(gamma)], dim=1)
        widths = stop_rule_verify_lens(confidence=c.float(), cost_ms=objective)
        tokens += float((torch.minimum(dtest["taken"], widths - 1) + 1).sum())
        ms += float((cost[widths - 1] + draft_ms).sum())
    out["halves"]["history (uncapped, optimistic) + head+q+top1+ent"] = {"ms_per_token": round(ms / tokens, 2)}
    if other is not None:
        for name, names in (("history only", ()), ("history + head", ("head",))):
            betas = train(other, names, shipped_widths(other)[0])
            result, _ = replay_sequential(x=step_x(data, names), betas=betas, data=data, cost=cost, draft_ms=draft_ms)
            out["across_runs"][name] = result
        result = evaluate(shipped_widths(data)[0], data["taken"], cost, draft_ms)
        out["across_runs"]["2 head, shipped STS"] = result
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--sts", required=True, help="the shipped STS file, for the head's row")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--cost", action="append", default=None, help="a,b; repeatable")
    p.add_argument("--draft-ms", type=float, default=0.0)
    p.add_argument("--json", default=None)
    p.add_argument("--history", action="store_true",
                   help="also replay the target-side history features (HistoryState), step by step")
    p.add_argument("--other-glob", default=None,
                   help="with --history: a second collection to fit on for the across-runs split")
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
        if a.history:
            other = load(a.other_glob, a.temperature, need_draft=False) if a.other_glob else None
            report.setdefault("history", {})[spec] = history_report(
                data=data, other=other, shipped_temperatures=temperatures, cost=cost, draft_ms=a.draft_ms
            )
    text = json.dumps(report, indent=1)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
