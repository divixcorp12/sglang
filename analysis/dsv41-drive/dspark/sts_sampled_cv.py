"""Prompt-held-out comparison of STS calibrations for the sampled confidence cut, on a multi-prompt sampled collection.

Rows of an STS collection (``spec_margin_capture.py --prompts-file ... --temperature 0.6``) are mapped to the request
being served through each shard's ``record_ns`` and probe.json's CLOCK_MONOTONIC spans. Whole prompts are held out in
``--folds`` folds, stratified by category, and each candidate is scored by sts_cut_replay's ms/token on the held-out
rows: tokens min(m, w - 1) + 1 for the width w the cut picks, time T(w) = a + b (w - 1) plus the draft.

Candidates: greedy (the production STS file); temp (dspark_sts_fit --label sampled); bias (per-position temperature and
additive logit bias, fitted to minimise the replayed ms/token on the training rows); logreg (per position, over the
head logit and the draft's q(d_k), top-1 probability and entropy; offline only). References: every row, constant
widths, the realized oracle (w = accepted + 1). The extra test set is scored with candidates fitted on all prompts.

Usage: sts_sampled_cv.py --data-glob 'OUT/collect/sts/confidence.*.pt' --probe OUT/collect/probe.json --prompts-dir DIR
       --greedy-sts STS.json --cost 131.9,51.8 [--draft-ms 5.8] [--extra-glob 'SET2/collect/sts/confidence.*.pt'] --json R.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

from sglang.benchmark.dspark_sts_fit import default_temperature_grid, fit_sts_temperatures  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_confidence_cut import stop_rule_verify_lens  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_sts import load_sts_calibration_from_path  # noqa: E402

GAMMA = 5
# prompts_varied.json index -> category; every prompt of prompts_short_answer.json is "short".
VARIED_CATEGORY = {**{i: "code" for i in (0, 6, 11, 18, 22)}, **{i: "chat" for i in (4, 8, 13, 15, 20)},
                   **{i: "lang" for i in (3, 7, 16)},
                   **{i: "reasoning" for i in (1, 2, 5, 9, 10, 12, 14, 17, 19, 21, 23)}}
CATEGORIES = ("code", "chat", "reasoning", "lang", "short")


def category_of(session_id: str) -> str:
    name, index = session_id.rsplit(":", 1)
    return "short" if "short_answer" in name else VARIED_CATEGORY[int(index)]


def load_rows(data_glob: str, temperature: float = 0.6) -> dict:
    shards = [torch.load(p, map_location="cpu") for p in sorted(glob.glob(data_glob))]
    cat = lambda k: torch.cat([s[k] for s in shards])  # noqa: E731
    keep = (cat("temperatures") - temperature).abs() < 1e-3
    rows = {
        "raw": cat("logits")[keep].double(),
        "taken": cat("sampled_prefix_mask")[keep].sum(dim=1).long(),
        "q": cat("draft_q")[keep].double(),
        "top1": cat("draft_top1")[keep].double(),
        "ent": cat("draft_entropy")[keep].double(),
    }
    if all("record_ns" in s for s in shards):
        rows["ns"] = cat("record_ns")[keep]
    ok = ~torch.stack([torch.isnan(rows[k]).any(dim=1) for k in ("raw", "q", "top1", "ent")]).any(dim=0)
    return {k: v[ok] for k, v in rows.items()} | {"dropped_nan": int((~ok).sum())}


def assign_prompts(rows: dict, probe: list[dict]) -> tuple[dict, list[str]]:
    """Keep the rows recorded inside a request's span; ``prompt`` indexes the returned session ids."""
    probe = sorted(probe, key=lambda r: r["sent_ns"])
    sent = torch.tensor([r["sent_ns"] for r in probe])
    done = torch.tensor([r["done_ns"] for r in probe])
    idx = torch.searchsorted(sent, rows["ns"], right=True) - 1
    inside = (idx >= 0) & (rows["ns"] <= done[idx.clamp_min(0)])
    out = {k: v[inside] for k, v in rows.items() if torch.is_tensor(v)}
    out["prompt"] = idx[inside]
    out["outside"] = int((~inside).sum())
    out["dropped_nan"] = rows["dropped_nan"]
    return out, [r["session_id"] for r in probe]


def make_folds(sessions: list[str], folds: int) -> torch.Tensor:
    """[n_sessions] fold id, round-robin within each category."""
    fold = torch.zeros(len(sessions), dtype=torch.int64)
    for c in CATEGORIES:
        for j, i in enumerate(i for i, s in enumerate(sessions) if category_of(s) == c):
            fold[i] = j % folds
    return fold


class Replay:
    def __init__(self, cost: tuple[float, float], draft_ms: float) -> None:
        self.cost = torch.tensor([cost[0] + cost[1] * w for w in range(GAMMA + 1)], dtype=torch.float64)
        self.draft_ms = draft_ms

    def widths(self, conf: torch.Tensor) -> torch.Tensor:
        return stop_rule_verify_lens(confidence=conf.double(), cost_ms=self.cost + self.draft_ms)

    def ms_tokens(self, widths: torch.Tensor, taken: torch.Tensor) -> tuple[float, float]:
        return float((self.cost[widths - 1] + self.draft_ms).sum()), float((torch.minimum(taken, widths - 1) + 1).sum())

    def score(self, conf: torch.Tensor, taken: torch.Tensor) -> float:
        ms, tokens = self.ms_tokens(self.widths(conf), taken)
        return ms / tokens


def fit_temp(rows: dict, mask: torch.Tensor) -> torch.Tensor:
    target = (torch.arange(GAMMA)[None, :] < rows["taken"][mask][:, None]).double()
    r = fit_sts_temperatures(logits=rows["raw"][mask], prefix_mask=target, grid=default_temperature_grid())
    return torch.tensor(r["temperatures"], dtype=torch.float64)


def fit_bias(rows: dict, mask: torch.Tensor, replay: Replay, start_t: torch.Tensor, sweeps: int = 3) -> tuple:
    """Coordinate search over per-position (log T, bias) for the lowest replayed ms/token on the rows."""
    raw, taken = rows["raw"][mask], rows["taken"][mask]
    logt, bias = start_t.log().clone(), torch.zeros(GAMMA, dtype=torch.float64)

    def loss(lt, b):
        return replay.score(torch.sigmoid(raw / lt.exp() + b), taken)

    best = loss(logt, bias)
    for _ in range(sweeps):
        for k in range(GAMMA):
            for var, grid in ((bias, torch.linspace(-3, 3, 25)), (logt, torch.linspace(-1.5, 2.3, 25))):
                keep = var[k].item()
                cand = keep
                for g in grid.tolist():
                    var[k] = g
                    v = loss(logt, bias)
                    if v < best - 1e-9:
                        best, cand = v, g
                var[k] = cand
    return logt.exp(), bias, best


def features(rows: dict, k: int) -> torch.Tensor:
    q = rows["q"][:, k].clamp(1e-6, 1 - 1e-6)
    return torch.stack([rows["raw"][:, k], torch.log(q / (1 - q)), rows["top1"][:, k], rows["ent"][:, k]], dim=1)


def fit_logreg(rows: dict, mask: torch.Tensor) -> list:
    """Per position k: P(draft k accepted | drafts 1..k-1 were), fitted on the masked rows that reached k."""
    models = []
    for k in range(GAMMA):
        sel = mask & (rows["taken"] >= k)
        x, y = features(rows, k)[sel], (rows["taken"][sel] >= k + 1).double()
        mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
        w = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200)

        def closure():
            opt.zero_grad()
            z = ((x - mu) / sd) @ w[:-1] + w[-1]
            l = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + 1e-3 * (w[:-1] ** 2).sum()
            l.backward()
            return l

        opt.step(closure)
        models.append((mu, sd, w.detach()))
    return models


def predict_logreg(models: list, rows: dict) -> torch.Tensor:
    cols = [torch.sigmoid(((features(rows, k) - mu) / sd) @ w[:-1] + w[-1]) for k, (mu, sd, w) in enumerate(models)]
    return torch.stack(cols, dim=1)


def references(replay: Replay, taken: torch.Tensor) -> dict:
    n = taken.numel()
    out = {}
    for w in range(1, GAMMA + 2):
        ms, tok = replay.ms_tokens(torch.full((n,), w), taken)
        out[f"constant w={w}" + (" (every row)" if w == GAMMA + 1 else "")] = ms / tok
    ms, tok = replay.ms_tokens(taken + 1, taken)
    out["realized oracle"] = ms / tok
    return out


def candidates(rows: dict, train: torch.Tensor, greedy_t: torch.Tensor, replay: Replay) -> dict:
    """name -> function(rows) -> [n, gamma] confidence, fitted on the ``train`` mask."""
    temp_t = fit_temp(rows, train)
    bias_t, bias_b, _ = fit_bias(rows, train, replay, greedy_t)
    models = fit_logreg(rows, train)
    return {
        "a greedy STS": (lambda r: torch.sigmoid(r["raw"] / greedy_t), {"T": greedy_t.tolist()}),
        "b sampled temp": (lambda r: torch.sigmoid(r["raw"] / temp_t), {"T": temp_t.tolist()}),
        "c temp+bias": (lambda r: torch.sigmoid(r["raw"] / bias_t + bias_b), {"T": bias_t.tolist(), "b": bias_b.tolist()}),
        "d logreg": (lambda r: predict_logreg(models, r), {}),
    }


def pooled(replay: Replay, per: dict, mask: torch.Tensor, rows: dict) -> float:
    ms, tok = replay.ms_tokens(per["widths"][mask], rows["taken"][mask])
    return ms / tok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--greedy-sts", required=True)
    p.add_argument("--cost", required=True)
    p.add_argument("--draft-ms", type=float, default=5.8)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--extra-glob", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    torch.manual_seed(0)
    replay = Replay(tuple(float(x) for x in a.cost.split(",")), a.draft_ms)
    greedy_t = torch.tensor(load_sts_calibration_from_path(a.greedy_sts).temperatures, dtype=torch.float64)
    with open(a.probe) as f:
        rows, sessions = assign_prompts(load_rows(a.data_glob), json.load(f))
    n = rows["taken"].numel()
    print(f"{n} verifies mapped to {len(sessions)} requests ({rows['outside']} outside any request, "
          f"{rows['dropped_nan']} dropped for NaN features)")
    fold_of_prompt = make_folds(sessions, a.folds)
    fold = fold_of_prompt[rows["prompt"]]
    cat = torch.tensor([CATEGORIES.index(category_of(sessions[i])) for i in rows["prompt"].tolist()])
    report = {"verifies": n, "cost": a.cost, "draft_ms": a.draft_ms,
              "rows_per_category": {c: int((cat == i).sum()) for i, c in enumerate(CATEGORIES)},
              "references": references(replay, rows["taken"]), "folds": {}, "params_all": {}}
    held = {}
    for f_id in range(a.folds):
        train = fold != f_id
        for name, (fn, params) in candidates(rows, train, greedy_t, replay).items():
            held.setdefault(name, {"widths": torch.zeros(n, dtype=torch.int64)})
            test = ~train
            held[name]["widths"][test] = replay.widths(fn({k: v[test] for k, v in rows.items() if torch.is_tensor(v)}))
            report["folds"].setdefault(name, []).append({"fold": f_id, "params": params,
                                                         "ms_per_token": pooled(replay, held[name], test, rows)})
    table = {}
    for name, per in held.items():
        per_fold = [pooled(replay, per, fold == i, rows) for i in range(a.folds)]
        table[name] = {"overall": pooled(replay, per, torch.ones(n, dtype=torch.bool), rows),
                       "fold_min": min(per_fold), "fold_max": max(per_fold), "fold_std": float(torch.tensor(per_fold).std()),
                       "per_category": {c: pooled(replay, per, cat == i, rows) for i, c in enumerate(CATEGORIES)}}
    report["heldout"] = table
    report["category_references"] = {c: references(replay, rows["taken"][cat == i]) for i, c in enumerate(CATEGORIES)}
    allmask = torch.ones(n, dtype=torch.bool)
    full = candidates(rows, allmask, greedy_t, replay)
    report["params_all"] = {name: params for name, (_, params) in full.items()}
    if a.extra_glob:
        extra = load_rows(a.extra_glob)
        report["extra_set"] = {"verifies": int(extra["taken"].numel()), "references": references(replay, extra["taken"]),
                               "candidates": {name: replay.score(fn(extra), extra["taken"]) for name, (fn, _) in full.items()}}
    text = json.dumps(report, indent=1)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
