"""Offline retrain of the DSpark confidence head on sampled-acceptance labels, replayed as the sampled cut would use it.

The shipped V4.1 head is one linear map w (5,120 hidden + 256 Markov embedding -> 1, no bias, shared by the 5
positions), with per-position STS temperatures after it: c_k = sigmoid(w . x_k / T_k), x_k = [hidden_k, E[prev_k]].
STS collections record x_k's parts (head_hidden fp16, markov_prev_tokens; the embedding table E comes from the draft
checkpoint), the soft label min(1, p/q) per position (accept_rates) and the accept outcome. Models, each giving
logit_k = s_k (w . x_k) + b_k:
  recal     -- w frozen at the shipped weights; s_k, b_k fitted (initialised at 1 / T_k, 0);
  finetune  -- w from the shipped weights, penalised by lam ||w - w_shipped||^2, plus s_k, b_k;
  probe     -- w from zero, penalised by lam ||w||^2, s_k = 1, plus b_k.
Trained full-batch with Adam on soft-label BCE (or --labels realized: BCE on "draft k accepted given 1..k-1 were",
over the verifies that reached k), lam picked from LAMBDAS on the last 20% of the training rows (contiguous), with
early stopping on that validation loss. Evaluated with accept_predictor_replay's replay (the stopping rule over the
predicted confidences) against the shipped head and the expected oracle on the same test rows: ms/token, the share
of the gap recovered, survival ECE per position, and the AUROC of S_4 and S_5 for verifies that accepted 4-5 drafts.

Usage: head_retrain.py --collection 0.6=DIR --collection 0.95=DIR --draft CKPT_DIR --out OUT_DIR
                       [--cost 91.2,21.1] [--draft-ms 5.8] [--labels soft|realized] [--epochs 400]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from typing import Optional

# CPU only: run as a script, hide the GPU before torch (and sglang's imports) can open a CUDA context, which would
# take VRAM outside cc-gpu.lock from a serving run. (Imported by a test, the caller's devices are left alone.)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD_WEIGHT = "mtp.2.confidence_head.proj.weight"
EMBED_WEIGHT = "mtp.2.markov_head.embed.weight"
SHIPPED_STS = (0.8912509083747864, 0.7943282127380371, 1.258925437927246, 0.8912509083747864, 0.6309573650360107)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
RECAL_LR = 0.02
MODELS = ("recal", "finetune", "probe")


def _replay_module():
    spec = importlib.util.spec_from_file_location("accept_predictor_replay", os.path.join(HERE, "accept_predictor_replay.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_draft_weights(draft_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The shipped head's [5376] weights and the Markov embedding table [vocab, 256], float32."""
    from safetensors import safe_open

    with safe_open(os.path.join(draft_dir, "model.safetensors"), "pt") as f:
        return f.get_tensor(HEAD_WEIGHT).float().view(-1), f.get_tensor(EMBED_WEIGHT).float()


def load_collection(directory: str, temperature: float, embed: torch.Tensor) -> dict:
    """The collection's verifies at ``temperature`` with finite head logits and hidden states, in serving order, with
    x [n, gamma, hidden + rank] float32."""
    paths = sorted(glob.glob(os.path.join(directory, "collect", "sts", "confidence.*.pt")),
                   key=lambda p: int(p.rsplit(".", 2)[-2]))
    shards = [s for s in (torch.load(p, map_location="cpu") for p in paths) if "head_hidden" in s]
    if not shards:
        raise SystemExit(f"{directory}: no shards with head_hidden")
    data = {k: torch.cat([s[k] for s in shards]) for k in shards[0]}
    keep = ((data["temperatures"] - temperature).abs() < 1e-3) & torch.isfinite(data["logits"]).all(dim=1)
    keep &= torch.isfinite(data["head_hidden"].float()).flatten(1).all(dim=1)
    data = {k: v[keep] for k, v in data.items()}
    data["x"] = torch.cat([data.pop("head_hidden").float(), embed[data["markov_prev_tokens"]]], dim=-1)
    data["taken"] = data["sampled_prefix_mask"].sum(dim=1).to(torch.int64)
    return data


def sub(data: dict, mask: torch.Tensor) -> dict:
    return {k: v[mask] for k, v in data.items()}


def cat_data(parts: list[dict]) -> dict:
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def labels(data: dict, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[n, gamma] targets and weights: soft min(1, p/q) at every position, or realized accepts over the verifies that
    reached each position."""
    if kind == "soft":
        rates = data["accept_rates"].float()
        return rates, torch.ones_like(rates)
    gamma = data["accept_rates"].shape[1]
    k = torch.arange(1, gamma + 1)[None, :]
    taken = data["taken"][:, None]
    return (taken >= k).float(), (taken >= k - 1).float()


class Head(torch.nn.Module):
    def __init__(self, *, model: str, w_shipped: torch.Tensor, gamma: int):
        super().__init__()
        self.model = model
        self.register_buffer("w0", w_shipped.clone())
        init = torch.zeros_like(w_shipped) if model == "probe" else w_shipped.clone()
        self.w = torch.nn.Parameter(init, requires_grad=model != "recal")
        scale = torch.ones(gamma) if model == "probe" else 1.0 / torch.tensor(SHIPPED_STS[:gamma])
        self.s = torch.nn.Parameter(scale, requires_grad=model != "probe")
        self.b = torch.nn.Parameter(torch.zeros(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.s * (x @ self.w) + self.b

    def penalty(self) -> torch.Tensor:
        if self.model == "recal":
            return torch.zeros(())
        ref = self.w0 if self.model == "finetune" else torch.zeros_like(self.w0)
        return ((self.w - ref) ** 2).sum()


def _loss(head: Head, x, y, weight) -> torch.Tensor:
    logits = head(x)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=weight, reduction="sum")
    return bce / weight.sum()


def train(*, model: str, data: dict, w_shipped: torch.Tensor, label_kind: str, epochs: int, lr: float = 1e-3,
          lambdas=LAMBDAS) -> tuple[Head, dict]:
    """Fit on the first 80% of ``data`` (contiguous), pick lam and the epoch by the last 20%'s loss; refit nothing
    after (the early-stopped weights are kept). Returns the head and its train/validation losses."""
    n, gamma = data["x"].shape[:2]
    cut = int(n * 0.8)
    fit, val = sub(data, torch.arange(n) < cut), sub(data, torch.arange(n) >= cut)
    y_fit, w_fit = labels(fit, label_kind)
    y_val, w_val = labels(val, label_kind)
    best = None
    for lam in lambdas if model != "recal" else (0.0,):
        torch.manual_seed(0)
        head = Head(model=model, w_shipped=w_shipped, gamma=gamma)
        # recal fits 2 x gamma scalars of order 1; the weight models' steps are sized for weights of order |w|.
        opt = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=RECAL_LR if model == "recal" else lr)
        state, best_val = None, float("inf")
        for epoch in range(epochs):
            opt.zero_grad()
            loss = _loss(head, fit["x"], y_fit, w_fit) + lam * head.penalty() / max(cut, 1)
            loss.backward()
            opt.step()
            if epoch % 5 == 4 or epoch == epochs - 1:
                with torch.no_grad():
                    v = float(_loss(head, val["x"], y_val, w_val))
                if v < best_val:
                    best_val, state = v, ({k: t.clone() for k, t in head.state_dict().items()}, epoch + 1)
        head.load_state_dict(state[0])
        with torch.no_grad():
            train_loss = float(_loss(head, fit["x"], y_fit, w_fit))
        if best is None or best_val < best[1]["val_loss"]:
            best = (head, {"lam": lam, "epoch": state[1], "train_loss": round(train_loss, 5),
                           "val_loss": round(best_val, 5)})
    return best


def auroc(score: torch.Tensor, label: torch.Tensor) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = torch.cat([pos, neg]).double().argsort().argsort().double() + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(replay, *, confidence: torch.Tensor, data: dict, cost: torch.Tensor, draft_ms: float,
             shipped: Optional[dict] = None, oracle: Optional[dict] = None) -> dict:
    out = replay.replay(confidence, data["taken"], cost, draft_ms)
    survival = torch.cumprod(confidence.double(), dim=1)
    good = data["taken"] >= 4
    out["ece"] = replay.survival_ece(confidence, data["sampled_prefix_mask"].double())
    out["auroc_s4"], out["auroc_s5"] = round(auroc(survival[:, 3], good), 3), round(auroc(survival[:, 4], good), 3)
    if shipped is not None and oracle is not None:
        gap = shipped["ms_per_token"] - oracle["ms_per_token"]
        out["recovered"] = round((shipped["ms_per_token"] - out["ms_per_token"]) / gap, 3) if gap else None
    return out


def shipped_confidence(data: dict, w_shipped: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid((data["x"] @ w_shipped) / torch.tensor(SHIPPED_STS))


def run_split(replay, *, name: str, train_data: dict, test_data: dict, w_shipped, cost, draft_ms, label_kind, epochs,
              out_dir: Optional[str], lr: float = 1e-3) -> dict:
    shipped = evaluate(replay, confidence=shipped_confidence(test_data, w_shipped), data=test_data, cost=cost,
                       draft_ms=draft_ms)
    oracle = evaluate(replay, confidence=test_data["accept_rates"].double(), data=test_data, cost=cost,
                      draft_ms=draft_ms)
    rows = {"shipped head": shipped, "expected oracle": oracle}
    for model in MODELS:
        head, fit = train(model=model, data=train_data, w_shipped=w_shipped, label_kind=label_kind, epochs=epochs,
                          lr=lr)
        with torch.no_grad():
            c_test = torch.sigmoid(head(test_data["x"])).double()
            y, wgt = labels(test_data, label_kind)
            test_loss = float(_loss(head, test_data["x"], y, wgt))
        rows[model] = evaluate(replay, confidence=c_test, data=test_data, cost=cost, draft_ms=draft_ms,
                               shipped=shipped, oracle=oracle) | {"fit": fit, "test_loss": round(test_loss, 5)}
        if out_dir:
            torch.save({"model": model, "state": head.state_dict(), "fit": fit, "split": name},
                       os.path.join(out_dir, f"{name.replace(' ', '_').replace('/', '-')}.{model}.pt"))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", action="append", required=True, help="TEMPERATURE=DIR")
    p.add_argument("--draft", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cost", default="91.2,21.1")
    p.add_argument("--draft-ms", type=float, default=5.8)
    p.add_argument("--labels", choices=("soft", "realized"), default="soft")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    replay = _replay_module()
    w_shipped, embed = load_draft_weights(a.draft)
    sets = {}
    for spec in a.collection:
        temperature, directory = spec.split("=", 1)
        sets[temperature] = load_collection(directory, float(temperature), embed)
    report = {"labels": a.labels, "verifies": {t: int(d["x"].shape[0]) for t, d in sets.items()}, "splits": {}}
    # The recorded inputs reproduce the shipped head: w . x against the recorded pre-STS logits.
    report["shipped_logit_max_abs_diff"] = {
        t: round(float(((d["x"] @ w_shipped) - d["logits"]).abs().max()), 4) for t, d in sets.items()
    }
    base, per_row = (float(v) for v in a.cost.split(","))
    gamma = next(iter(sets.values()))["x"].shape[1]
    cost = torch.tensor([base + per_row * w for w in range(gamma + 1)], dtype=torch.float64)
    common = dict(w_shipped=w_shipped, cost=cost, draft_ms=a.draft_ms, label_kind=a.labels, epochs=a.epochs,
                  out_dir=a.out, lr=a.lr)

    def halves(d):
        n = d["x"].shape[0]
        first = torch.arange(n) < n // 2
        return sub(d, first), sub(d, ~first)

    for t, d in sets.items():
        h0, h1 = halves(d)
        report["splits"][f"{t} halves A->B"] = run_split(replay, name=f"{t}_A-B", train_data=h0, test_data=h1, **common)
        report["splits"][f"{t} halves B->A"] = run_split(replay, name=f"{t}_B-A", train_data=h1, test_data=h0, **common)
    temps = list(sets)
    if len(temps) == 2:
        for src, dst in (temps, temps[::-1]):
            report["splits"][f"cross {src}->{dst}"] = run_split(
                replay, name=f"cross_{src}-{dst}", train_data=sets[src], test_data=sets[dst], **common)
        split = {t: halves(d) for t, d in sets.items()}
        for i, label in ((0, "A->B"), (1, "B->A")):
            both = cat_data([split[t][i] for t in temps])
            for t in temps:
                report["splits"][f"both {label}, test {t}"] = run_split(
                    replay, name=f"both_{label}_{t}", train_data=both, test_data=split[t][1 - i], **common)
    text = json.dumps(report, indent=1)
    print(text)
    with open(os.path.join(a.out, f"head_retrain_{a.labels}.json"), "w") as f:
        f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
