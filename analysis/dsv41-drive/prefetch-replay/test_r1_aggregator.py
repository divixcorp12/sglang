"""Tests for r1_aggregator: padding invariance, feature values and losses (CPU, synthetic)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1_aggregator as agg  # noqa: E402
from r1_groups import GroupBatch  # noqa: E402

E = 384


def _batch(live: np.ndarray, position: np.ndarray) -> GroupBatch:
    g = live.shape[0]
    z = np.zeros((g, 6, E), dtype=np.float32)
    return GroupBatch(live=live, position=position, width=live.sum(1).astype(np.float32),
                      source_layer=np.zeros(g, dtype=np.int64), target_layer=np.ones(g, dtype=np.int64),
                      stratum=np.zeros(g, dtype=np.int64), miss=np.zeros((g, E), dtype=np.float32),
                      available=np.ones((g, E), dtype=np.float32), route=z, token_y=z, hard_neg=z)


def _live(n: int) -> tuple[np.ndarray, np.ndarray]:
    live = np.zeros((1, 6), dtype=bool)
    live[0, :n] = True
    return live, np.arange(6, dtype=np.int64)[None]


def test_features_ignore_padding():
    torch.manual_seed(0)
    live, pos = _live(3)
    a = torch.randn(1, 6, E)
    b = a.clone()
    b[:, 3:] = 1e4
    fa = agg.block_features(a, live=torch.as_tensor(live), position=torch.as_tensor(pos))
    fb = agg.block_features(b, live=torch.as_tensor(live), position=torch.as_tensor(pos))
    assert fa.shape == (1, E, agg.N_BLOCK_FEATURES)
    assert torch.equal(fa, fb)


def test_feature_values_single_token():
    live, pos = _live(1)
    x = torch.randn(1, 6, E)
    f = agg.block_features(x, live=torch.as_tensor(live), position=torch.as_tensor(pos))
    assert torch.allclose(f[0, :, 0], x[0, 0])
    assert torch.all(f[0, :, 1] == 1) and torch.all(f[0, :, 3] == 0)
    assert torch.allclose(f[0, :, 4], x[0, 0]) and torch.allclose(f[0, :, 5], x[0, 0])
    assert torch.allclose(f[0, :, 6], torch.zeros(E))
    assert torch.allclose(f[0, :, 8], torch.sigmoid(x[0, 0]), atol=1e-5)
    assert torch.all(f[0, :, 9] == 1)


def test_aggregator_shape_and_padding_invariance():
    torch.manual_seed(0)
    live, pos = _live(2)
    m = agg.BlockAggregator(n_layers=4, n_strata=3)
    a = torch.randn(1, 6, E)
    b = a.clone()
    b[:, 2:] = -50.0
    batch = _batch(live, pos)
    assert torch.allclose(m(a, batch=batch), m(b, batch=batch))
    assert m(a, batch=batch).shape == (1, E)


def test_block_loss_masks_unavailable():
    logits = torch.randn(2, E)
    miss = (torch.rand(2, E) > 0.9).float()
    avail = torch.ones(2, E)
    avail[0, 10:] = 0
    avail[1] = 0
    got = agg.block_loss(logits, miss=miss, available=avail)
    want = F.binary_cross_entropy_with_logits(logits[0, :10], miss[0, :10])
    assert torch.allclose(got, want)


def test_route_and_b1_loss_ignore_padding():
    logits = torch.randn(1, 6, E)
    y = (torch.rand(1, 6, E) > 0.9).float()
    live = torch.tensor([[True, True, False, False, False, False]])
    hn = (torch.rand(1, 6, E) > 0.8).float()
    r = agg.route_loss(logits, route=y, live=live)
    assert torch.allclose(r, F.binary_cross_entropy_with_logits(logits[:, :2], y[:, :2]))
    b = agg.b1_token_loss(logits, token_y=y, hard_neg=hn, live=live, hard_neg_weight=1.0)
    assert torch.allclose(b, r)
    w = agg.b1_token_loss(logits, token_y=y, hard_neg=hn, live=live, hard_neg_weight=3.0)
    assert w > b
