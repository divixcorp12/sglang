"""Tests for l2_fast_profile on tiny synthetic T0-like (noisy-or) and T3-like (aggregator) models."""

import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
import torch  # noqa: E402

import b1_head  # noqa: E402
import l2_fast_profile  # noqa: E402
import r1_cost  # noqa: E402
import r1_models  # noqa: E402
import r1_plan  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E = 16
RANK = 8


def _model(*, aggregator: bool) -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=16, emb_dim=4, num_experts=E)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.randn(r1_cost.X_DIM, generator=g),
                   components=torch.randn((RANK, r1_cost.X_DIM), generator=g), explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=(19, 20), strata=("a",)))
    torch.manual_seed(1)
    cfg_r1 = r1_models.R1Config(head=cfg, position=aggregator, aggregator=aggregator)
    return r1_models.R1Model(cfg_r1, head=head).eval()


def _gate() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    return torch.randn((E, r1_cost.X_DIM), generator=g) * 0.01, torch.rand(E, generator=g)


@pytest.mark.parametrize("aggregator", [False, True], ids=["noisy_or", "aggregator"])
def test_profile_has_every_component_with_nodes(aggregator):
    # Catches a component dropped or captured empty, which would zero a node count and misattribute its cost.
    W, bias = _gate()
    out = l2_fast_profile.profile_model(_model(aggregator=aggregator), W=W, bias=bias, cost=r1_plan.CostSpec(),
                                        runs=5, warmup=1, device="cuda")
    assert set(out) == set(l2_fast_profile.COMPONENTS)
    for name, row in out.items():
        assert row["nodes"] >= 1, name
        assert row["median_ms"] > 0 and row["runs"] == 5, name


@pytest.mark.parametrize("aggregator", [False, True], ids=["noisy_or", "aggregator"])
def test_components_chain_to_r1_cost_complete_selection(aggregator):
    # Catches a split that changes the math: chained components must pick what r1_cost's complete step picks.
    model, (W, bias), cost = _model(aggregator=aggregator), _gate(), r1_plan.CostSpec()
    built = l2_fast_profile.component_steps(model, W=W, bias=bias, cost=cost, device="cuda")
    ref = copy.deepcopy(model).float().cuda().eval()
    low = copy.deepcopy(ref).to(torch.bfloat16).eval()
    W, bias = W.cuda(), bias.cuda()
    st = r1_cost._make_static(tokens=cost.live_tokens, experts=E, device="cuda")
    batch = r1_cost._static_batch(tokens=cost.live_tokens, dtype=torch.bfloat16, device="cuda", experts=E)
    with torch.no_grad():
        u = r1_cost._gate_features(st.x, st.ids, st.weights, ref=ref, W=W, bias=bias)
        inp = r1_cost.ModelInput(batch=batch, u=u.to(torch.bfloat16).unsqueeze(0), x=None, s_native=None)
        want = r1_cost.select_torch(torch.sigmoid(low(inp)[0].float()), skip=st.skip, threshold=st.threshold, cap=1)
        got = built.chain()
        for name in l2_fast_profile.COMPONENTS:
            assert built.steps[name]() is not None
    assert got.shape == (1,)
    assert torch.equal(got, want)
