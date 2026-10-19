"""Offline study: is a cost-aware confidence cut worth building? (idea A: a live cost signal; idea B: per-row miss cost)

Reads costcut_extract.py tables of the sampled-finish captures (prod: every verify at full width; cut, cut-retry: the
sampled cut) and an STS collection of the same prompt for the head's logits and accept outcomes.

1. Cost variance: realized step ms against the startup T(w) fit and against width, and the residual.
2. Signals: what is known before verify i starts, against verify i's cost residual, fitted on the first half of the
   serving verifies and scored on the second.
3. Miss cost: how much of the cost the verify's own misses explain (an upper bound for a perfect predictor).
4. Ceiling: the production greedy STS cut replayed with the static T(w), a corrected static table, an oracle per-verify
   cost, and the best live signal. Truth is Tbar(w) (the mean step ms of the serving verifies at width w) plus the
   verify's residual, additive or proportional. Each pairs a collected verify (head logits, accepts) with a residual from
   the capture in time order: the two are different runs of the same prompt, so the pairing assumes the cost residual is
   independent of the head's confidence.

Widths of the cut arms are not logged per verify. They are inferred from svc_routed (the unique experts the live rows
routed per layer): the 96 calibration verifies (widths 1..6 round robin) train a per-layer naive Bayes classifier, with
the arm's logged width histogram as the prior.

Usage: costcut_study.py TABLE_DIR --sets-glob 'SET2/collect/sts/confidence.*.pt' --json OUT
"""

from __future__ import annotations

import argparse
import glob
import json
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sglang.srt.speculative.dspark_components.dspark_confidence_cut import stop_rule_verify_lens  # noqa: E402

GAMMA = 5
GREEDY_T = [0.8912509083747864, 0.7943282127380371, 1.258925437927246, 0.8912509083747864, 0.6309573650360107]
# From the arms' server.log "T(w) ms = [...]": T(w) plus the draft, the table the rule divides by.
STATIC = {
    "cut": [139.4487, 191.2868, 243.1249, 294.963, 346.8012, 398.6393],
    "cut-retry": [165.6302, 220.4507, 275.2713, 330.0919, 384.9125, 439.7331],
}
# Logged widths 1..6 of the arms' served verifies (report.json / report-cutretry.txt).
LOGGED_HIST = {"cut": [230, 1429, 856, 215, 88, 90], "cut-retry": [185, 1358, 916, 228, 87, 55]}


def load(table_dir: str, arm: str) -> dict:
    z = dict(np.load(os.path.join(table_dir, arm + ".npz")))
    z["routed"] = z["routed"].astype(int)
    return z


def calibration_start(routed: np.ndarray) -> int:
    ones = set(np.where(routed.sum(1) == 240)[0].tolist())
    for k in sorted(ones):
        if all(k + 6 * j in ones for j in range(16)):
            return k
    raise ValueError("no round-robin calibration block")


def fit_nb(routed: np.ndarray, widths: np.ndarray, smooth: float = 0.5) -> np.ndarray:
    p = np.full((6, 50), smooth)
    for w in range(1, 7):
        np.add.at(p[w - 1], routed[widths == w].ravel(), 1)
    return np.log(p / p.sum(1, keepdims=True))


def infer_widths(nb: np.ndarray, routed: np.ndarray, hist: list[int]) -> np.ndarray:
    ll = np.stack([nb[w][routed].sum(1) for w in range(6)], axis=1) + np.log(np.array(hist) / sum(hist))[None, :]
    ll[np.arange(6)[None, :] < np.ceil(routed.max(1) / 6.0)[:, None] - 1] = -np.inf  # fewer rows than routed experts / 6
    return ll.argmax(1) + 1


def serving_mask(z: dict, k0: int | None) -> np.ndarray:
    m = (z["begin_ns"] > z["first_token_ns"]) & (z["end_ns"] < z["end_request_ns"]) & z["complete"]
    if k0 is not None:
        m[k0:k0 + 96] = False
    return m


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def lag_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the n values before each index (the first n use what exists)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    lo = np.maximum(idx - n, 0)
    den = np.maximum(idx - lo, 1)
    return (c[idx] - c[lo]) / den


def ols(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    xb = np.column_stack([np.ones(len(x)), x])
    return np.linalg.solve(xb.T @ xb + ridge * np.eye(xb.shape[1]), xb.T @ y)


def predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coef[0] + x @ coef[1:]


def signal_table(rows: dict, w: np.ndarray, resid: np.ndarray) -> dict:
    """Candidate lagged features of the serving verifies in time order, from what verify i-1.. had finished."""
    wf = w.astype(float)
    feats = {}
    per_row = {"read_ms": rows["read_ms"] / wf, "forced": rows["forced"] / wf, "lanes": rows["lanes"].sum(1) / wf,
               "evict": rows["evict"] / wf, "resid": resid}
    for name, x in per_row.items():
        for n in (1, 4, 16):
            feats[f"{name}_prev{n}"] = lag_mean(x, n)
    return feats


def half_scores(feats: dict, y: np.ndarray) -> dict:
    h = len(y) // 2
    out = {}
    for name, x in feats.items():
        coef = ols(x[:h, None], y[:h])
        p = predict(coef, x[h:, None])
        out[name] = {"r2_test": r2(y[h:], p), "spearman_test": spearman(x[h:], y[h:])}
    names = list(feats)
    x = np.column_stack([feats[n] for n in names])
    mu, sd = x[:h].mean(0), x[:h].std(0) + 1e-9
    coef = ols((x[:h] - mu) / sd, y[:h], ridge=1.0)
    out["ALL (ridge)"] = {"r2_test": r2(y[h:], predict(coef, (x[h:] - mu) / sd)), "spearman_test": None}
    return out


def study_arm(name: str, z: dict, w: np.ndarray, k0: int | None) -> dict:
    m = serving_mask(z, k0)
    cost = z["cost_ms"][m]
    ww = w[m]
    rep = {"serving_verifies": int(m.sum()), "mean_ms_per_verify": float(cost.mean()),
           "width_hist": np.bincount(ww, minlength=7)[1:].tolist()}
    means = {int(k): float(cost[ww == k].mean()) for k in np.unique(ww)}
    rep["mean_cost_by_width"] = means
    rep["std_cost_by_width"] = {int(k): float(cost[ww == k].std()) for k in np.unique(ww)}
    pred_w = np.array([means[k] for k in ww])
    rep["r2_width_onehot"] = r2(cost, pred_w) if len(means) > 1 else 0.0
    if len(means) > 1:
        coef = np.polyfit(ww, cost, 1)
        rep["affine_fit_realized"] = {"a": float(coef[1] + coef[0]), "b": float(coef[0]), "r2": r2(cost, np.polyval(coef, ww))}
    if name in STATIC:
        stat = np.array(STATIC[name])[ww - 1]
        res = cost - stat
        rep["vs_static"] = {"mean_residual": float(res.mean()), "std_residual": float(res.std()),
                            "quantiles_1_10_50_90_99": np.quantile(res, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
                            "r2_static_prediction": r2(cost, stat)}
    resid = cost - pred_w
    rep["residual_std_after_width"] = float(resid.std())
    rep["residual_quantiles_1_10_50_90_99"] = np.quantile(resid, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist()
    rep["residual_autocorr_lag_1_2_4_16"] = [float(np.corrcoef(resid[:-k], resid[k:])[0, 1]) for k in (1, 2, 4, 16)]
    sub = {k: (v[m] if getattr(v, "ndim", 0) >= 1 and len(v) == len(m) else v) for k, v in z.items()}
    h = len(cost) // 2
    train_means = {int(k): float(cost[:h][ww[:h] == k].mean()) for k in np.unique(ww[:h])}
    resid_train = cost - np.array([train_means.get(int(k), means[int(k)]) for k in ww])
    rep["signals_predicting_residual"] = half_scores(signal_table(sub, ww, resid_train), resid_train)
    # Q3: the verify's own misses (known only after it), against its cost.
    own = {"lanes": sub["lanes"].sum(1).astype(float), "forced": sub["forced"].astype(float),
           "evict": sub["evict"].astype(float), "read_ms": sub["read_ms"], "routed": sub["routed"].sum(1).astype(float)}
    rep["own_misses_r2_of_cost"] = {k: r2(cost, predict(ols(v[:, None], cost), v[:, None])) for k, v in own.items()}
    if len(means) > 1:
        onehot = np.stack([(ww == k).astype(float) for k in sorted(means)[1:]], axis=1)
        base = r2(cost, predict(ols(onehot, cost), onehot))
        rep["r2_width_alone"] = base
        for k in ("lanes", "forced", "evict", "read_ms"):
            x = np.column_stack([onehot, own[k]])
            rep[f"r2_width_plus_{k}"] = r2(cost, predict(ols(x, cost), x))
        x = np.column_stack([onehot, own["lanes"], own["forced"], own["evict"]])
        rep["r2_width_plus_lanes_forced_evict"] = r2(cost, predict(ols(x, cost), x))
    else:
        x = np.column_stack([own["lanes"], own["forced"], own["evict"]])
        rep["r2_lanes_forced_evict_at_fixed_width"] = r2(cost, predict(ols(x, cost), x))
    return rep


def tokens_for(widths: torch.Tensor, taken: torch.Tensor) -> torch.Tensor:
    return torch.minimum(taken, widths - 1) + 1


def replay(conf: torch.Tensor, taken: torch.Tensor, decide: torch.Tensor, truth: torch.Tensor) -> dict:
    """decide, truth: [n, 6] cost tables (per verify) the rule divides by / the realized cost at each width."""
    widths = stop_rule_verify_lens(confidence=conf, cost_ms=decide)
    ms = truth.gather(1, (widths - 1)[:, None]).squeeze(1)
    tok = tokens_for(widths, taken)
    return {"ms_per_token": float(ms.sum() / tok.sum()), "tokens_per_verify": float(tok.double().mean()),
            "mean_width": float(widths.double().mean())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table_dir")
    p.add_argument("--sets-glob", required=True)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    arms = {n: load(a.table_dir, n) for n in ("prod", "cut", "cut-retry")}
    ks = {n: calibration_start(arms[n]["routed"]) for n in ("cut", "cut-retry")}
    cal_w = (np.arange(96) % 6) + 1
    nb = fit_nb(np.concatenate([arms[n]["routed"][ks[n]:ks[n] + 96] for n in ks]), np.concatenate([cal_w, cal_w]))
    acc = {}
    for hold in ks:
        other = [n for n in ks if n != hold][0]
        nb1 = fit_nb(arms[other]["routed"][ks[other]:ks[other] + 96], cal_w)
        pred = infer_widths(nb1, arms[hold]["routed"][ks[hold]:ks[hold] + 96], [1] * 6)
        acc[hold] = float((pred == cal_w).mean())
    widths = {"prod": np.full(len(arms["prod"]["cost_ms"]), 6)}
    for n in ks:
        widths[n] = infer_widths(nb, arms[n]["routed"], LOGGED_HIST[n])
        widths[n][ks[n]:ks[n] + 96] = cal_w
    report = {"width_classifier_cross_arm_accuracy_on_calibration": acc, "arms": {}}
    for n in ("prod", "cut", "cut-retry"):
        report["arms"][n] = study_arm(n, arms[n], widths[n], ks.get(n))

    # Q4. Truth table Tbar(w): serving means of the cut arms for w = 1..5, the prod arm for w = 6.
    pool = [(widths[n][serving_mask(arms[n], ks[n])], arms[n]["cost_ms"][serving_mask(arms[n], ks[n])]) for n in ks]
    wcat, ccat = np.concatenate([x[0] for x in pool]), np.concatenate([x[1] for x in pool])
    prod_cost = arms["prod"]["cost_ms"][serving_mask(arms["prod"], None)]
    tbar = np.array([ccat[wcat == k].mean() for k in range(1, 6)] + [prod_cost.mean()])
    report["tbar"] = tbar.tolist()
    shards = [torch.load(p_, map_location="cpu") for p_ in sorted(glob.glob(a.sets_glob))]
    cat = lambda k: torch.cat([s[k] for s in shards])  # noqa: E731
    keep = (cat("temperatures") - 0.6).abs() < 1e-3
    taken_all = cat("sampled_prefix_mask")[keep].sum(1).long()
    conf_all = torch.sigmoid(cat("logits")[keep].double() / torch.tensor(GREEDY_T, dtype=torch.float64))
    report["collection_verifies"] = int(taken_all.numel())
    out = {}
    for arm in ks:
        m = serving_mask(arms[arm], ks[arm])
        cost, w = arms[arm]["cost_ms"][m], widths[arm][m]
        sub = {k: (v[m] if getattr(v, "ndim", 0) >= 1 and len(v) == len(m) else v) for k, v in arms[arm].items()}
        eps = cost - tbar[w - 1]
        rho = cost / tbar[w - 1] - 1
        h = len(cost) // 2
        n = min(len(cost) - h, taken_all.numel() - taken_all.numel() // 2)
        conf, taken = conf_all[taken_all.numel() // 2:][:n], taken_all[taken_all.numel() // 2:][:n]
        eps_t, rho_t = torch.tensor(eps[h:][:n]), torch.tensor(rho[h:][:n])
        tb = torch.tensor(tbar)[None, :].expand(n, 6)
        # the best live predictor: ridge over the lagged signals, fitted on the first half
        feats = signal_table(sub, w, eps)
        names = list(feats)
        x = np.column_stack([feats[k] for k in names])
        mu, sd = x[:h].mean(0), x[:h].std(0) + 1e-9
        coef_e = ols((x[:h] - mu) / sd, eps[:h], ridge=1.0)
        coef_r = ols((x[:h] - mu) / sd, rho[:h], ridge=1.0)
        eps_hat = torch.tensor(predict(coef_e, (x[h:] - mu) / sd)[:n])
        rho_hat = torch.tensor(predict(coef_r, (x[h:] - mu) / sd)[:n])
        # a one-signal predictor: the previous verify's residual
        best1 = "resid_prev1"
        c1 = ols(feats[best1][:h, None], eps[:h])
        eps_hat1 = torch.tensor(predict(c1, feats[best1][h:, None])[:n])
        c1r = ols(feats[best1][:h, None], rho[:h])
        rho_hat1 = torch.tensor(predict(c1r, feats[best1][h:, None])[:n])
        stat = torch.tensor(STATIC[arm])[None, :].expand(n, 6)
        res = {"pairs": int(n)}
        for mode in ("additive", "proportional"):
            truth = tb + eps_t[:, None] if mode == "additive" else tb * (1 + rho_t[:, None])
            def hat(e, r_):
                return tb + e[:, None] if mode == "additive" else tb * (1 + r_[:, None])
            r_ = {
                "i static T(w) (startup fit)": replay(conf, taken, stat, truth),
                "iv corrected static (mean cost per width)": replay(conf, taken, tb, truth),
                "ii oracle per-verify cost": replay(conf, taken, truth, truth),
                "iii live ridge over lagged signals": replay(conf, taken, hat(eps_hat, rho_hat), truth),
                "iii live previous-verify residual": replay(conf, taken, hat(eps_hat1, rho_hat1), truth),
            }
            for wd in range(1, 7):
                r_[f"constant w={wd}"] = {
                    "ms_per_token": float(truth[:, wd - 1].sum() / tokens_for(torch.full((n,), wd), taken).sum())}
            r_["realized oracle width (w = accepts + 1)"] = {
                "ms_per_token": float(truth.gather(1, taken[:, None].clamp(max=5)).sum() / (taken + 1).sum())}
            base = r_["i static T(w) (startup fit)"]["ms_per_token"]
            for v in r_.values():
                v["vs_i_pct"] = 100 * (v["ms_per_token"] / base - 1)
            res[mode] = r_
        out[arm] = res
    report["ceiling"] = out
    text = json.dumps(report, indent=1)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
