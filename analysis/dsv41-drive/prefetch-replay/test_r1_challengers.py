"""Tests for r1_challengers: T5 projection parity and trainability, G0 step-0 native margins, checkpoints."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_challengers  # noqa: E402
import r1_models  # noqa: E402
from r1_groups import GroupBatch, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E, RANK, HID, G, W, D = 12, 4, 8, 3, 6, 10
VOCAB = b1_head.Vocab(layers=(1, 2, 3), strata=("a", "b"))
HEAD_CFG = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=HID, emb_dim=2, num_experts=E)
F_DIM = RANK + 4 * E


def _parent() -> r1_models.R1Model:
    torch.manual_seed(0)
    pca = PCAStats(mean=torch.randn(D), components=torch.randn(RANK, D), explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(HEAD_CFG, pca=pca, vocab=VOCAB)
    return r1_models.R1Model(r1_models.R1Config(head=HEAD_CFG), head=head).eval()


def _input(parent: r1_models.R1Model, *, widths=(1, 3, 6), seed=1) -> ModelInput:
    rng = np.random.default_rng(seed)
    live = np.zeros((G, W), dtype=bool)
    for i, w in enumerate(widths):
        live[i, :w] = True
    batch = GroupBatch(
        live=live, position=np.tile(np.arange(W, dtype=np.int64), (G, 1)), width=np.array(widths, dtype=np.float32),
        source_layer=np.array([0, 1, 0], dtype=np.int64), target_layer=np.array([1, 2, 2], dtype=np.int64),
        stratum=np.array([0, 1, 1], dtype=np.int64), miss=np.zeros((G, E), dtype=np.float32),
        available=np.ones((G, E), dtype=np.float32), route=np.zeros((G, W, E), dtype=np.float32),
        token_y=np.zeros((G, W, E), dtype=np.float32), hard_neg=np.zeros((G, W, E), dtype=np.float32))
    mask = torch.from_numpy(live).unsqueeze(-1)
    x = torch.from_numpy(rng.standard_normal((G, W, D)).astype(np.float32)) * mask
    rest = torch.from_numpy(rng.standard_normal((G, W, F_DIM - RANK)).astype(np.float32)) * mask
    pca = parent.head.pca
    u = torch.cat([(x - pca.mean) @ pca.components.T, rest], dim=-1) * mask
    s_native = torch.from_numpy(rng.standard_normal((G, W, E)).astype(np.float32)) * mask
    return ModelInput(batch=batch, u=u, x=x, s_native=s_native)


def test_projection_matches_parent_at_step_zero():
    parent = _parent()
    inp = _input(parent)
    child = r1_challengers.with_projection(parent).eval()
    live = torch.from_numpy(inp.batch.live)
    with torch.no_grad():
        got, want = child.token_logits(inp), parent.token_logits(inp)
    assert torch.allclose(got[live], want[live], atol=1e-5)


def test_projection_is_trainable_and_keeps_state_keys():
    parent = _parent()
    child = r1_challengers.with_projection(parent)
    assert set(child.state_dict()) == set(parent.state_dict())
    names = r1_challengers.projection_params(child)
    params = dict(child.named_parameters())
    assert names <= set(params) and all(params[n].requires_grad for n in names)
    assert "head.pca_components" not in dict(parent.named_parameters())
    inp = _input(parent)
    child.train()
    child(inp).sum().backward()
    assert params["head.pca_components"].grad is not None
    assert params["head.pca_components"].grad.abs().sum() > 0


def test_projection_needs_x():
    parent = _parent()
    inp = _input(parent)
    child = r1_challengers.with_projection(parent)
    with pytest.raises(ValueError):
        child.token_logits(ModelInput(batch=inp.batch, u=inp.u, x=None, s_native=None))


def test_projection_checkpoint_roundtrip(tmp_path):
    parent = _parent()
    child = r1_challengers.with_projection(parent).eval()
    with torch.no_grad():
        child.head.pca_components.add_(0.5)
    path = tmp_path / "t5.pt"
    sha = r1_models.save_r1(path, child, parent_sha="abc", arm="T5")
    loaded, sha2, parent_sha = r1_challengers.load_with_projection(path)
    inp = _input(parent)
    assert sha == sha2 and parent_sha == "abc"
    assert isinstance(loaded.head.pca_components, torch.nn.Parameter)
    with torch.no_grad():
        assert torch.allclose(loaded.token_logits(inp), child.token_logits(inp), atol=1e-6)


def _g0() -> r1_challengers.G0Model:
    torch.manual_seed(0)
    cfg = r1_challengers.G0Config(rank=5, pos_dim=5, emb_dim=2, hidden=D, num_experts=E)
    return r1_challengers.G0Model(cfg, n_layers=len(VOCAB.layers), n_strata=len(VOCAB.strata)).eval()


def test_g0_step_zero_equals_native_margins():
    inp = _input(_parent())
    m = _g0()
    with torch.no_grad():
        z = m.token_logits(inp)
    assert torch.equal(z, inp.s_native)
    assert m(inp).shape == (G, E)


def test_g0_residual_is_trainable():
    inp = _input(_parent())
    m = _g0().train()
    with torch.no_grad():
        m.out_b.normal_(0.0, 0.1)
    m(inp).sum().backward()
    assert m.proj_a.weight.grad is not None and m.proj_a.weight.grad.abs().sum() > 0
    assert m.scale_a.grad is not None


def test_g0_checkpoint_roundtrip(tmp_path):
    inp = _input(_parent())
    m = _g0()
    with torch.no_grad():
        m.out_b.normal_(0.0, 0.1)
    path = tmp_path / "g0.pt"
    sha = r1_challengers.save_g0(path, m, parent_sha="def")
    loaded, sha2, parent_sha = r1_challengers.load_g0(path)
    assert sha == sha2 and parent_sha == "def"
    with torch.no_grad():
        assert torch.equal(loaded(inp), m(inp))
