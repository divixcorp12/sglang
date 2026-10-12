"""GPU test for b1_cost.measure_cost on a small synthetic head; skips without CUDA."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
import torch  # noqa: E402

import b1_cost  # noqa: E402
import b1_head  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _small_head(variant: str) -> b1_head.B1Head:
    cfg = b1_head.HeadConfig(variant=variant, pca_rank=8, hidden=16, emb_dim=4, num_experts=16)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.zeros(b1_cost.X_DIM), components=torch.randn((8, b1_cost.X_DIM), generator=g),
                   explained_variance=torch.ones(8))
    vocab = b1_head.Vocab(layers=(19, 20), strata=("a",))
    return b1_head.B1Head(cfg, pca=pca, vocab=vocab)


@pytest.mark.parametrize("variant", ["E2", "E5"])
def test_measure_cost_positive(variant):
    model = _small_head(variant)
    out = b1_cost.measure_cost(model, runs=20, warmup=3)
    assert out["bytes_bf16"] == b1_head.head_bytes_bf16(model) > 0
    assert out["median_ms"] > 0
    assert out["p90_ms"] >= out["median_ms"]
    assert out["runs"] == 20 and out["tokens"] == 6
    assert isinstance(out["within_0_1ms"], bool)
