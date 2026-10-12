"""Tests for r1_cost: select_torch against r1_policy, read_set_diff, and the CUDA-graph timing (GPU only)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import b1_head  # noqa: E402
import r1_cost  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402
import r1_policy  # noqa: E402
from r1_groups import GroupBatch, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E = 16
RANK = 8


def _model(*, variant: str = "E5", aggregator: bool = True) -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant=variant, pca_rank=RANK, hidden=16, emb_dim=4, num_experts=E)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.zeros(r1_cost.X_DIM), components=torch.randn((RANK, r1_cost.X_DIM), generator=g),
                   explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=(19, 20), strata=("a",)))
    torch.manual_seed(1)
    return r1_models.R1Model(r1_models.R1Config(head=cfg, position=True, aggregator=aggregator), head=head).eval()


def test_select_torch_matches_policy_with_ties():
    scores = torch.tensor([0.5, 0.9, 0.9, 0.1, 0.9, 0.5, 0.0, 0.7] + [0.2] * 8)
    skip = torch.zeros(E, dtype=torch.bool)
    skip[1] = True
    for cap in (1, 2, 3):
        for thr in (0.0, 0.5, 0.95):
            got = r1_cost.select_torch(scores, skip=skip, threshold=torch.tensor(thr), cap=cap)
            assert got.shape == (cap,)
            c = r1_policy.candidates(scores.numpy(), skip=skip.numpy())
            want = r1_policy.select(c, threshold=thr, cap=cap)
            assert got[got >= 0].tolist() == want.tolist()
            assert (got[len(want):] == -1).all()


def test_select_torch_skipped_never_read():
    scores = torch.full((E,), 0.9)
    skip = torch.ones(E, dtype=torch.bool)
    got = r1_cost.select_torch(scores, skip=skip, threshold=torch.tensor(0.0), cap=2)
    assert got.tolist() == [-1, -1]


def test_selection_mismatches_zero():
    gen = torch.Generator().manual_seed(3)
    scores = torch.rand((5, E), generator=gen).round(decimals=1)
    skips = torch.rand((5, E), generator=gen) < 0.3
    assert r1_cost.selection_mismatches(scores, skips=skips, threshold=0.3, cap=2) == 0


def _inputs(groups: int) -> tuple[ModelInput, torch.Tensor]:
    gen = torch.Generator().manual_seed(5)
    width = 3
    live = np.zeros((groups, 6), dtype=bool)
    live[:, :width] = True
    zero_i = np.zeros(groups, dtype=np.int64)
    batch = GroupBatch(
        live=live, position=np.tile(np.arange(6, dtype=np.int64), (groups, 1)),
        width=np.full(groups, width, dtype=np.float32), source_layer=zero_i, target_layer=zero_i + 1, stratum=zero_i,
        miss=np.zeros((groups, E), np.float32), available=np.ones((groups, E), np.float32),
        route=np.zeros((groups, 6, E), np.float32), token_y=np.zeros((groups, 6, E), np.float32),
        hard_neg=np.zeros((groups, 6, E), np.float32))
    feat = RANK + 4 * E
    u = torch.randn((groups, 6, feat), generator=gen) * torch.from_numpy(live).float().unsqueeze(-1)
    skips = torch.rand((groups, E), generator=gen) < 0.25
    return ModelInput(batch=batch, u=u, x=None, s_native=None), skips


def test_read_set_diff_is_a_fraction_and_zero_at_extremes():
    inp, skips = _inputs(8)
    model = _model()
    frac = r1_cost.read_set_diff(model, inputs=[inp], skips=[skips], threshold=0.0, cap=2)
    assert 0.0 <= frac <= 1.0
    # Threshold 2.0 is above every sigmoid, so both paths read nothing.
    assert r1_cost.read_set_diff(model, inputs=[inp], skips=[skips], threshold=2.0, cap=2) == 0.0


def test_read_set_diff_rejects_empty():
    with pytest.raises(ValueError):
        r1_cost.read_set_diff(_model(), inputs=[], skips=[], threshold=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("aggregator", [False, True])
def test_measure_complete_reports_both_paths(aggregator):
    model = _model(aggregator=aggregator)
    g = torch.Generator().manual_seed(7)
    W = torch.randn((E, r1_cost.X_DIM), generator=g) * 0.01
    bias = torch.zeros(E)
    out = r1_cost.measure_complete(model, W=W, bias=bias, cost=r1_plan.CostSpec(), runs=20, warmup=3)
    for path in ("bf16", "fp32"):
        for part in ("complete", "head_only"):
            row = out[path][part]
            assert 0 < row["median_ms"] <= row["p99_ms"] and row["p90_ms"] >= row["median_ms"]
        assert out[path]["complete"]["median_ms"] >= out[path]["head_only"]["median_ms"] * 0.5
    assert out["gate_ms"] == out["bf16"]["complete"]["median_ms"]
    assert out["param_bytes"] * 2 == out["fp32"]["param_bytes"]
    assert out["buffer_bytes"] * 2 == out["fp32"]["buffer_bytes"]
    assert out["selection_matches_policy"] is True
