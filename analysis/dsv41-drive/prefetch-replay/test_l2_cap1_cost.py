"""Tests for r1_cost at cap 1: select_torch, selection_mismatches, read_set_diff and measure_complete(cap=1)."""

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
from r1_groups import GroupBatch, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E = 16
RANK = 8


def _model() -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=16, emb_dim=4, num_experts=E)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.zeros(r1_cost.X_DIM), components=torch.randn((RANK, r1_cost.X_DIM), generator=g),
                   explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=(19, 20), strata=("a",)))
    torch.manual_seed(1)
    return r1_models.R1Model(r1_models.R1Config(head=cfg, position=True, aggregator=True), head=head).eval()


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


def test_select_torch_cap1_returns_one_slot():
    scores = torch.rand(E, generator=torch.Generator().manual_seed(2))
    skip = torch.zeros(E, dtype=torch.bool)
    got = r1_cost.select_torch(scores, skip=skip, threshold=torch.tensor(0.0), cap=1)
    assert got.shape == (1,)
    assert got.item() == int(scores.argmax())


def test_selection_mismatches_zero_at_cap1():
    gen = torch.Generator().manual_seed(3)
    scores = torch.rand((6, E), generator=gen).round(decimals=1)
    skips = torch.rand((6, E), generator=gen) < 0.3
    assert r1_cost.selection_mismatches(scores, skips=skips, threshold=0.3, cap=1) == 0


def test_read_set_diff_cap1_is_a_fraction():
    inp, skips = _inputs(8)
    frac = r1_cost.read_set_diff(_model(), inputs=[inp], skips=[skips], threshold=0.0, cap=1)
    assert 0.0 <= frac <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_measure_complete_cap1_and_default_has_no_cap_key():
    model = _model()
    g = torch.Generator().manual_seed(7)
    W = torch.randn((E, r1_cost.X_DIM), generator=g) * 0.01
    bias = torch.zeros(E)
    out = r1_cost.measure_complete(model, W=W, bias=bias, cost=r1_plan.CostSpec(), runs=10, warmup=2, cap=1)
    assert out["cap"] == 1
    assert out["bf16"]["complete"]["median_ms"] > 0
    assert out["bf16"]["head_only"]["median_ms"] > 0
    assert out["gate_only_ms"] > 0
    assert out["gate_ms"] == out["bf16"]["complete"]["median_ms"]
    assert out["selection_matches_policy"] is True
    default = r1_cost.measure_complete(model, W=W, bias=bias, cost=r1_plan.CostSpec(), runs=5, warmup=1)
    assert "cap" not in default
