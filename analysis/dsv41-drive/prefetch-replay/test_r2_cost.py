"""Tests for r2_cost: read_set_diff on tiny T5 and G0 models (CPU) and the CUDA-graph path timings (GPU only)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import b1_head  # noqa: E402
import r1_challengers  # noqa: E402
import r1_cost  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402
import r2_cost  # noqa: E402
from r1_groups import GroupBatch, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E = 16
RANK = 8
X_DIM = r1_cost.X_DIM


def _t5() -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=16, emb_dim=4, num_experts=E)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.zeros(X_DIM), components=torch.randn((RANK, X_DIM), generator=g) * 0.02,
                   explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=(19, 20), strata=("a",)))
    torch.manual_seed(1)
    parent = r1_models.R1Model(r1_models.R1Config(head=cfg, position=True, aggregator=True), head=head).eval()
    return r1_challengers.with_projection(parent).eval()


def _g0() -> r1_challengers.G0Model:
    torch.manual_seed(2)
    cfg = r1_challengers.G0Config(rank=8, pos_dim=4, emb_dim=4, num_experts=E)
    model = r1_challengers.G0Model(cfg, n_layers=2, n_strata=1)
    with torch.no_grad():
        model.out_b.normal_(std=0.1)
    return model.eval()


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
    mask = torch.from_numpy(live).float().unsqueeze(-1)
    u = torch.randn((groups, 6, RANK + 4 * E), generator=gen) * mask
    x = torch.randn((groups, 6, X_DIM), generator=gen) * mask
    s_native = torch.randn((groups, 6, E), generator=gen) * mask
    skips = torch.rand((groups, E), generator=gen) < 0.25
    return ModelInput(batch=batch, u=u, x=x, s_native=s_native), skips


@pytest.mark.parametrize("make", [_t5, _g0], ids=["t5", "g0"])
def test_read_set_diff_is_a_fraction_and_zero_at_extremes(make):
    inp, skips = _inputs(8)
    model = make()
    frac = r2_cost.read_set_diff(model, inputs=[inp], skips=[skips], threshold=0.0, cap=2)
    assert 0.0 <= frac <= 1.0
    # Threshold 2.0 is above every sigmoid, so both paths read nothing.
    assert r2_cost.read_set_diff(model, inputs=[inp], skips=[skips], threshold=2.0, cap=2) == 0.0


def test_read_set_diff_rejects_empty():
    with pytest.raises(ValueError):
        r2_cost.read_set_diff(_g0(), inputs=[], skips=[], threshold=0.0)


def _gate() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    return torch.randn((E, X_DIM), generator=g) * 0.01, torch.zeros(E)


def _check_paths(out: dict, paths: tuple[str, ...]) -> None:
    for dtype in ("bf16", "fp32"):
        for path in paths:
            row = out[dtype][path]
            assert 0 < row["median_ms"] <= row["p99_ms"] and row["p90_ms"] >= row["median_ms"]
            assert row["runs"] == 10
        assert out[dtype]["param_bytes"] > 0 and out[dtype]["buffer_bytes"] >= 0
    assert out["bf16"]["param_bytes"] * 2 == out["fp32"]["param_bytes"]
    assert out["tokens"] == 6
    assert out["complete_ms"] == out["bf16"]["complete"]["median_ms"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_measure_t5_reports_both_paths():
    W, bias = _gate()
    out = r2_cost.measure_t5(_t5(), W=W, bias=bias, cost=r1_plan.CostSpec(), runs=10, warmup=2)
    _check_paths(out, ("head_only", "complete"))
    assert out["increment_ms"] == out["bf16"]["head_only"]["median_ms"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_measure_g0_reports_correction_path():
    W, bias = _gate()
    out = r2_cost.measure_g0(_g0(), W=W, bias=bias, cost=r1_plan.CostSpec(), runs=10, warmup=2)
    _check_paths(out, ("correction", "head_only", "complete"))
    assert out["increment_ms"] == out["bf16"]["correction"]["median_ms"]
