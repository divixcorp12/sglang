"""accept_predictor_replay.py: logistic acceptance predictors from the draft's own probabilities, replayed (CPU)."""

import importlib.util
import json
import os

import torch

from sglang.srt.speculative.dspark_components.dspark_sts import DSparkStsCalibration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _load():
    path = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "accept_predictor_replay.py")
    spec = importlib.util.spec_from_file_location("accept_predictor_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_logistic_fit_recovers_a_known_model_from_soft_labels():
    replay = _load()
    g = torch.Generator().manual_seed(0)
    x = torch.randn((20_000, 2), generator=g, dtype=torch.float64)
    y = torch.sigmoid(1.5 * x[:, 0] - 0.5 * x[:, 1] + 0.3)
    beta = replay.fit_logistic(x, y, ridge=0.0)
    assert torch.allclose(beta, torch.tensor([1.5, -0.5, 0.3], dtype=torch.float64), atol=1e-3)


def test_a_predictor_that_reads_the_drawn_drafts_probability_tracks_the_oracle(tmp_path):
    """Synthetic shards whose acceptance rate is a function of q(d_k): the q predictor lands near the expected oracle
    and well ahead of a head that carries no information."""
    replay = _load()
    g = torch.Generator().manual_seed(1)
    n, gamma = 4_000, 5
    q = torch.rand((n, gamma), generator=g)
    rates = torch.sigmoid(4 * (q - 0.5))
    accepted = torch.rand((n, gamma), generator=g) < rates
    prefix = torch.cumprod(accepted.float(), dim=1)
    torch.save(
        {"logits": torch.randn((n, gamma), generator=g), "prefix_mask": prefix, "sampled_prefix_mask": prefix,
         "accept_rates": rates, "temperatures": torch.full((n,), 0.6), "draft_q": q,
         "draft_top1": torch.full((n, gamma), 0.9), "draft_entropy": torch.ones((n, gamma))},
        tmp_path / "c.0.pt",
    )
    sts = tmp_path / "sts.json"
    sts.write_text(DSparkStsCalibration(temperatures=[1.0] * gamma).to_json())
    out = tmp_path / "r.json"
    assert replay.main(["--data-glob", str(tmp_path / "c.*.pt"), "--sts", str(sts), "--cost", "100,20",
                        "--json", str(out)]) == 0
    rows = json.loads(out.read_text())["fits"]["100,20"]
    oracle = rows["3 expected oracle"]["ms_per_token"]
    assert rows["q [per-position]"]["ms_per_token"] < 1.03 * oracle
    assert rows["q [per-position]"]["ms_per_token"] < rows["head [per-position]"]["ms_per_token"]
