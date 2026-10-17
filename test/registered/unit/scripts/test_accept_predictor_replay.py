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


def test_step_ts_history_features_read_only_steps_before_t():
    """The leakage guard of the target-side features: whatever step t and later did (their target p, their accept
    draws, their widths), step t's history features are unchanged. That is what keeps them constants of step t's
    verify, so the stopping-rule proof holds. Mutant: HistoryState updated before features() is read -- red."""
    replay = _load()
    g = torch.Generator().manual_seed(2)
    n, gamma = 300, 5
    taken = torch.randint(0, gamma + 1, (n,), generator=g)
    widths = torch.randint(1, gamma + 2, (n,), generator=g)
    rates = torch.rand((n, gamma), generator=g)
    base = replay.history_features(taken=taken, widths=widths, rates=rates)
    assert base.shape == (n, len(replay.HISTORY_NAMES))
    for t in (0, 1, 7, 150, n - 1):
        taken2, widths2, rates2 = taken.clone(), widths.clone(), rates.clone()
        taken2[t:] = torch.randint(0, gamma + 1, (n - t,), generator=g)
        widths2[t:] = torch.randint(1, gamma + 2, (n - t,), generator=g)
        rates2[t:] = torch.rand((n - t, gamma), generator=g)
        again = replay.history_features(taken=taken2, widths=widths2, rates=rates2)
        assert torch.equal(again[: t + 1], base[: t + 1]), t
    # And they do read the past: the step after a changed one differs.
    changed = taken.clone()
    changed[10] = (changed[10] + 1) % (gamma + 1)
    assert not torch.equal(replay.history_features(taken=changed, widths=widths, rates=rates)[11], base[11])


def test_the_history_sees_only_what_a_cut_verify_observed():
    replay = _load()
    state = replay.HistoryState()
    assert state.features()[0] == 1.0  # no history yet
    # A 3-row verify whose chain accepted 4 drafts: positions 1..3 ran, all accepted, so the cut was binding.
    state.update(taken=4, width=3, rates=[0.9, 0.8, 0.7, 0.6, 0.5])
    names = dict(zip(replay.HISTORY_NAMES, state.features()))
    assert names["no_history"] == 0.0 and names["prev_accept"] == 3.0 and names["prev_cut_short"] == 1.0
    assert abs(names["prev_mean_rate"] - 0.8) < 1e-9 and names["prev_last_accept_rate"] == 0.7
    # A full verify rejected at draft 2: tests ran at positions 1 and 2.
    state.update(taken=1, width=6, rates=[0.9, 0.2, 0.7, 0.6, 0.5])
    names = dict(zip(replay.HISTORY_NAMES, state.features()))
    assert names["prev_accept"] == 1.0 and names["prev_cut_short"] == 0.0
    assert abs(names["prev_mean_rate"] - 0.55) < 1e-9 and names["prev_last_accept_rate"] == 0.9


def test_a_sequential_replays_step_t_confidence_reads_no_outcome_of_step_t_or_later():
    replay = _load()
    g = torch.Generator().manual_seed(3)
    n, gamma = 200, 5
    data = {"taken": torch.randint(0, gamma + 1, (n,), generator=g), "accept_rates": torch.rand((n, gamma), generator=g)}
    x = torch.randn((n, gamma, 2), generator=g, dtype=torch.float64)
    betas = [torch.randn(2 + len(replay.HISTORY_NAMES) + 1, generator=g, dtype=torch.float64) * 0.3
             for _ in range(gamma)]
    cost = torch.tensor([100.0 + 20 * w for w in range(gamma + 1)], dtype=torch.float64)
    _, base = replay.replay_sequential(x=x, betas=betas, data=data, cost=cost, draft_ms=0.0)
    t = 120
    other = {"taken": data["taken"].clone(), "accept_rates": data["accept_rates"].clone()}
    other["taken"][t:] = torch.randint(0, gamma + 1, (n - t,), generator=g)
    other["accept_rates"][t:] = torch.rand((n - t, gamma), generator=g)
    _, again = replay.replay_sequential(x=x, betas=betas, data=other, cost=cost, draft_ms=0.0)
    assert torch.equal(again[: t + 1], base[: t + 1])


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
                        "--json", str(out), "--history", "--other-glob", str(tmp_path / "c.*.pt")]) == 0
    history = json.loads(out.read_text())["history"]["100,20"]
    assert set(history["halves"]) >= {"history only", "history + head+q+top1+ent"}
    assert set(history["across_runs"]) == {"history only", "history + head", "2 head, shipped STS"}
    rows = json.loads(out.read_text())["fits"]["100,20"]
    oracle = rows["3 expected oracle"]["ms_per_token"]
    assert rows["q [per-position]"]["ms_per_token"] < 1.03 * oracle
    assert rows["q [per-position]"]["ms_per_token"] < rows["head [per-position]"]["ms_per_token"]
