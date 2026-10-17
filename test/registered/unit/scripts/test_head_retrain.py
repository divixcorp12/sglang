"""head_retrain.py: the confidence head refit on sampled-acceptance labels, and its replay (CPU)."""

import importlib.util
import os

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GAMMA = 5


def _load():
    path = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "head_retrain.py")
    spec = importlib.util.spec_from_file_location("head_retrain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic(n, *, seed):
    """Verifies whose acceptance rate is sigmoid(w_true . x_k); the 'shipped' head has the right direction at the
    wrong scale and offset for half its weights."""
    g = torch.Generator().manual_seed(seed)
    dim = 12
    w_true = torch.randn(dim, generator=g)
    x = torch.randn((n, GAMMA, dim), generator=g)
    rates = torch.sigmoid(x @ w_true + 0.5)
    accepted = torch.rand((n, GAMMA), generator=g) < rates
    prefix = torch.cumprod(accepted.float(), dim=1)
    data = {"x": x, "accept_rates": rates, "sampled_prefix_mask": prefix, "taken": prefix.sum(dim=1).long()}
    w_shipped = w_true.clone()
    w_shipped[: dim // 2] *= -0.5
    return data, w_shipped


def test_the_refits_beat_a_miscalibrated_shipped_head_and_report_train_and_validation_losses():
    retrain = _load()
    replay = retrain._replay_module()
    everything, w_shipped = _synthetic(6000, seed=0)
    train_data = {k: v[:3000] for k, v in everything.items()}
    test_data = {k: v[3000:] for k, v in everything.items()}
    cost = torch.tensor([100.0 + 20 * w for w in range(GAMMA + 1)], dtype=torch.float64)
    rows = retrain.run_split(replay, name="t", train_data=train_data, test_data=test_data, w_shipped=w_shipped,
                             cost=cost, draft_ms=5.0, label_kind="soft", epochs=400, out_dir=None, lr=0.05)
    print({m: rows[m].get("recovered") for m in retrain.MODELS})
    shipped, oracle = rows["shipped head"]["ms_per_token"], rows["expected oracle"]["ms_per_token"]
    assert oracle < shipped
    # The fine-tune from the shipped weights and the fresh probe both recover most of the gap; recalibration alone
    # cannot fix the wrong half of the weights.
    assert rows["finetune"]["recovered"] > 0.6 and rows["probe"]["recovered"] > 0.6
    assert rows["recal"]["recovered"] < rows["finetune"]["recovered"]
    for model in retrain.MODELS:
        fit = rows[model]["fit"]
        assert {"lam", "epoch", "train_loss", "val_loss"} <= set(fit) and rows[model]["test_loss"] > 0
        assert len(rows[model]["ece"]) == GAMMA and 0.0 <= rows[model]["auroc_s4"] <= 1.0


def test_realized_labels_weight_only_the_verifies_that_reached_each_position():
    retrain = _load()
    data = {"accept_rates": torch.zeros((2, GAMMA)), "taken": torch.tensor([0, 3])}
    y, w = retrain.labels(data, "realized")
    assert w.tolist() == [[1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    assert y.tolist() == [[0, 0, 0, 0, 0], [1, 1, 1, 0, 0]]
    soft_y, soft_w = retrain.labels(data, "soft")
    assert torch.equal(soft_y, data["accept_rates"]) and bool((soft_w == 1).all())


def test_a_collection_builds_the_heads_input_and_drops_non_finite_rows(tmp_path):
    retrain = _load()
    shard_dir = tmp_path / "collect" / "sts"
    shard_dir.mkdir(parents=True)
    n, hidden, rank, vocab = 4, 3, 2, 7
    logits = torch.zeros((n, GAMMA))
    logits[1, 0] = float("nan")
    torch.save({
        "logits": logits, "temperatures": torch.tensor([0.6, 0.6, 0.6, 0.0]),
        "head_hidden": torch.ones((n, GAMMA, hidden), dtype=torch.float16),
        "markov_prev_tokens": torch.full((n, GAMMA), 3), "accept_rates": torch.rand((n, GAMMA)),
        "sampled_prefix_mask": torch.zeros((n, GAMMA)),
    }, shard_dir / "confidence.0.pt")
    embed = torch.arange(vocab * rank, dtype=torch.float32).view(vocab, rank)
    data = retrain.load_collection(str(tmp_path), 0.6, embed)
    assert data["x"].shape == (2, GAMMA, hidden + rank)  # the NaN row and the greedy one are dropped
    assert data["x"][0, 0].tolist() == [1.0, 1.0, 1.0, 6.0, 7.0]
