"""Tests for r1_models: parent parity at step 0, padding invariance, param groups, freezing, checkpoints, scorers."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import r1_models  # noqa: E402
from r1_groups import GroupBatch, ModelInput  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E, RANK, HID, G, W = 12, 4, 8, 3, 6
VOCAB = b1_head.Vocab(layers=(1, 2, 3), strata=("a", "b"))
HEAD_CFG = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=HID, emb_dim=2, num_experts=E)
F_DIM = RANK + 4 * E


def _head() -> b1_head.B1Head:
    torch.manual_seed(0)
    pca = PCAStats(mean=torch.zeros(10), components=torch.randn(RANK, 10), explained_variance=torch.ones(RANK))
    return b1_head.B1Head(HEAD_CFG, pca=pca, vocab=VOCAB).eval()


def _cfg(**kw) -> r1_models.R1Config:
    return r1_models.R1Config(head=HEAD_CFG, **kw)


def _input(*, widths=(1, 3, 6), seed=1) -> ModelInput:
    rng = np.random.default_rng(seed)
    live = np.zeros((G, W), dtype=bool)
    for i, w in enumerate(widths):
        live[i, :w] = True
    batch = GroupBatch(
        live=live, position=np.tile(np.arange(W, dtype=np.int64), (G, 1)), width=np.array(widths, dtype=np.float32),
        source_layer=np.array([0, 1, 0], dtype=np.int64), target_layer=np.array([1, 2, 2], dtype=np.int64),
        stratum=np.array([0, 1, 1], dtype=np.int64), miss=(rng.random((G, E)) < 0.3).astype(np.float32),
        available=np.ones((G, E), dtype=np.float32), route=np.zeros((G, W, E), dtype=np.float32),
        token_y=np.zeros((G, W, E), dtype=np.float32), hard_neg=np.zeros((G, W, E), dtype=np.float32))
    u = torch.from_numpy(rng.standard_normal((G, W, F_DIM)).astype(np.float32))
    u = u * torch.from_numpy(live).unsqueeze(-1)
    return ModelInput(batch=batch, u=u, x=None, s_native=None)


def _parent() -> r1_models.R1Model:
    return r1_models.R1Model(_cfg(), head=_head()).eval()


def test_base_matches_b1_head_per_token():
    m, inp = _parent(), _input()
    flat = inp.u.reshape(-1, F_DIM)
    b = inp.batch
    hb = b1_head.HeadBatch(
        u=flat, source_layer=torch.from_numpy(np.repeat(b.source_layer, W)),
        target_layer=torch.from_numpy(np.repeat(b.target_layer, W)),
        stratum=torch.from_numpy(np.repeat(b.stratum, W)), y=torch.zeros(G * W, E), hard_neg=torch.zeros(G * W, E))
    with torch.no_grad():
        want = m.head(hb).reshape(G, W, E)
        got = m.token_logits(inp)
    assert torch.allclose(got, want, atol=1e-6)


def test_children_match_parent_at_step_zero():
    parent, inp = _parent(), _input()
    child = r1_models.R1Model.from_parent(_cfg(position=True, adapter_rank=3), parent=parent).eval()
    with torch.no_grad():
        assert torch.allclose(child.token_logits(inp), parent.token_logits(inp), atol=1e-6)
    assert not torch.equal(child.adapter_a, torch.zeros_like(child.adapter_a))
    assert torch.count_nonzero(child.adapter_u) == 0 and torch.count_nonzero(child.adapter_b) == 0


def test_from_parent_is_a_deep_copy():
    parent = _parent()
    child = r1_models.R1Model.from_parent(_cfg(aggregator=True), parent=parent)
    with torch.no_grad():
        child.head.predictor.fc_in.weight.add_(1.0)
    assert not torch.equal(child.head.predictor.fc_in.weight, parent.head.predictor.fc_in.weight)


def test_from_parent_carries_shared_parts():
    parent = r1_models.R1Model.from_parent(_cfg(position=True), parent=_parent())
    with torch.no_grad():
        parent.pos_proj.weight.fill_(0.5)
    child = r1_models.R1Model.from_parent(_cfg(position=True, adapter_rank=2), parent=parent)
    assert torch.equal(child.pos_proj.weight, parent.pos_proj.weight)


def test_noisy_or_block_logit_ignores_padding():
    m = _parent()
    inp = _input()
    with torch.no_grad():
        block = m(inp)
        logits = m.token_logits(inp)
    live = torch.from_numpy(inp.batch.live)
    for g in range(G):
        p = torch.sigmoid(logits[g][live[g]])
        want = 1.0 - torch.prod(1.0 - p, dim=0)
        assert torch.allclose(torch.sigmoid(block[g]), want, atol=1e-5)
    # Garbage in padding rows changes nothing.
    noisy = ModelInput(batch=inp.batch, u=inp.u + (~live).unsqueeze(-1) * 7.0, x=None, s_native=None)
    with torch.no_grad():
        assert torch.allclose(m(noisy), block, atol=1e-5)


def test_aggregator_forward_shape_and_padding_invariance():
    m = r1_models.R1Model.from_parent(_cfg(position=True, adapter_rank=2, aggregator=True), parent=_parent()).eval()
    inp = _input()
    live = torch.from_numpy(inp.batch.live)
    with torch.no_grad():
        a = m(inp)
        b = m(ModelInput(batch=inp.batch, u=inp.u + (~live).unsqueeze(-1) * 3.0, x=None, s_native=None))
    assert a.shape == (G, E)
    assert torch.allclose(a, b, atol=1e-5)


def test_param_groups_partition_trainable_parameters():
    m = r1_models.R1Model.from_parent(_cfg(position=True, adapter_rank=2, aggregator=True), parent=_parent())
    groups = r1_models.param_groups(m, new={"aggregator", "pos_proj", "pos_emb", "adapter_u"}, lr_new=1e-3,
                                    lr_inherited=1e-4, weight_decay=1e-2)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == sum(1 for p in m.parameters() if p.requires_grad)
    assert {g["lr"] for g in groups} == {1e-3, 1e-4}
    assert {g["weight_decay"] for g in groups} == {1e-2, 0.0}
    new_ids = {id(p) for g in groups if g["lr"] == 1e-3 for p in g["params"]}
    assert id(m.aggregator.mlp[0].weight) in new_ids
    assert id(m.head.predictor.fc_in.weight) not in new_ids


def test_freeze_encoder_keeps_aggregator_trainable():
    m = r1_models.R1Model.from_parent(_cfg(aggregator=True), parent=_parent())
    r1_models.freeze(m, encoder=True, embeddings=True)
    assert not any(p.requires_grad for p in m.head.parameters())
    assert all(p.requires_grad for p in m.aggregator.parameters())
    groups = r1_models.param_groups(m, new={"aggregator"}, lr_new=1e-3, lr_inherited=1e-4, weight_decay=0.0)
    assert {g["lr"] for g in groups} == {1e-3}


def test_save_load_roundtrip(tmp_path):
    m = r1_models.R1Model.from_parent(_cfg(position=True, adapter_rank=2, aggregator=True), parent=_parent()).eval()
    with torch.no_grad():
        m.adapter_u.normal_()
        m.pos_proj.weight.normal_()
    p = tmp_path / "t4.pt"
    sha = r1_models.save_r1(p, m, parent_sha="abc", arm="T4")
    loaded, sha2, parent_sha = r1_models.load_r1(p)
    assert (sha, parent_sha) == (sha2, "abc")
    inp = _input()
    with torch.no_grad():
        assert torch.allclose(loaded(inp), m(inp), atol=1e-6)


def _fake_group(*, n: int, available: np.ndarray):
    return SimpleNamespace(g=SimpleNamespace(tokens=np.arange(n)), labels=SimpleNamespace(available=available))


def test_scorers_mask_unavailable_and_split_live_rows():
    m = r1_models.R1Model.from_parent(_cfg(aggregator=True), parent=_parent())
    inp = _input()
    avail = np.ones(E, dtype=bool)
    avail[2] = False
    groups = [_fake_group(n=n, available=avail) for n in (1, 3, 6)]
    gs = r1_models.R1GroupScorer(m, collate_fn=lambda gr: inp, name="x")
    scores = gs.score_groups(groups)
    assert len(scores) == G and scores[0].shape == (E,) and scores[0].dtype == np.float32
    assert all(np.isneginf(s[2]) and np.isfinite(np.delete(s, 2)).all() for s in scores)
    tok = r1_models.R1TokenScorer(m, collate_fn=lambda gr: inp, name="x").score_tokens(groups)
    assert [t.shape for t in tok] == [(1, E), (3, E), (6, E)]
