"""The native-gate lookahead scorer on a synthetic router capture (CPU)."""

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "analysis", "dsv41-drive", "router-capture"))

import router_score  # noqa: E402
from prefetch_sim import DecodeStream  # noqa: E402

LAYERS, HIDDEN, EXPERTS = 3, 32, router_score.NUM_EXPERTS


def _routed_by_gate(x, W, bias):
    scores = torch.nn.functional.softplus(x @ W.T).sqrt()
    ids = torch.topk(scores + bias, 6).indices
    return ids, scores.gather(0, ids)


def _synthetic(tmp_path, steps=10):
    """A capture whose routes are exactly its gates' routes, written through the runtime's RouterCapture."""
    from sglang.srt.layers.moe.exl3_stream_trace import GraphRouteLog, RouterCapture

    gen = torch.Generator().manual_seed(0)
    W = torch.randn((LAYERS, EXPERTS, HIDDEN), generator=gen)
    bias = torch.randn((LAYERS, EXPERTS), generator=gen) * 0.1
    log = GraphRouteLog(LAYERS, 8, "cpu", depth=4, margin=1)
    log.layer_ids = list(range(LAYERS))
    capture = RouterCapture(str(tmp_path / "router"))
    routes = np.zeros((steps, LAYERS, 6), dtype=np.int64)
    for step in range(steps):
        x = torch.randn((LAYERS, HIDDEN), generator=gen).to(torch.bfloat16)
        w = torch.zeros((LAYERS, 6))
        for layer in range(LAYERS):
            ids, weights = _routed_by_gate(x[layer].float(), W[layer], bias[layer])
            routes[step, layer] = ids.numpy()[::-1]  # the route log keeps router order, not rank order
            w[layer] = (weights / weights.sum() * 1.5).flip(0)
        capture.write(log, 100 + step, 500 + step, x, w)
    capture.flush()
    stream = DecodeStream(list(range(LAYERS)), routes, ["r"] * steps, np.array([s > 0 for s in range(steps)]))
    return stream, W, bias


def test_the_self_check_reproduces_routes_made_by_the_gate(tmp_path):
    stream, W, bias = _synthetic(tmp_path)
    capture = router_score.load_capture(str(tmp_path / "router"))
    records = np.arange(len(stream.rids))
    assert capture.record_of_seq[100] == 0 and capture.x.shape == (10, LAYERS, HIDDEN)

    def x_of(steps, layer):
        return router_score.bf16_to_f32(np.asarray(capture.x[records[steps], layer]))

    check = router_score.self_check(stream, x_of, W, bias, capture, records)
    assert check["route_sets_equal"] == 1.0 and check["weight_max_abs_err"] < 1e-5
    assert abs(check["weight_sum_mean"] - 1.5) < 1e-5
    rank0 = router_score.rank_horizon(stream, x_of, W, bias, 0, 8)
    assert all(set(rank0.order[s, t, :6]) == set(stream.routes[s, t]) for s in range(10) for t in range(LAYERS))
    # h = 1 at layer 0 reaches the previous token's last layer; step 0 has no previous token.
    rank1 = router_score.rank_horizon(stream, x_of, W, bias, 1, 8)
    assert not rank1.valid[0, 0] and rank1.valid[1, 0] and rank1.valid[0, 1]
    want, _ = _routed_by_gate(torch.from_numpy(x_of(np.array([3]), 0))[0], W[1], bias[1])
    assert list(rank1.order[3, 1, :6]) == want.tolist()


def test_offline_precision_matches_a_brute_force_count():
    rng = np.random.default_rng(1)
    steps, depth = 20, 12
    order = np.stack([np.stack([rng.permutation(EXPERTS)[:depth] for _ in range(LAYERS)]) for _ in range(steps)])
    rank = router_score.Rankings(1, order.astype(np.int16), np.zeros(order.shape, np.float32),
                                 rng.random((steps, LAYERS)) > 0.1)
    resident = rng.random((steps, LAYERS, EXPERTS)) < 0.9
    routed = np.zeros_like(resident)
    for s in range(steps):
        for t in range(LAYERS):
            routed[s, t, order[s, t, rng.integers(0, depth, 3)]] = True  # some routes fall in the ranking
            routed[s, t, rng.integers(0, EXPERTS, 3)] = True
    missing = routed & ~resident
    mask = rng.random(steps) < 0.6
    for k in (1, 2):
        got = router_score.offline_precision(rank, resident, missing, mask, k, 6)
        fetched = hits = first = first_hits = 0
        for s in np.flatnonzero(mask):
            for t in range(LAYERS):
                if not rank.valid[s, t]:
                    continue
                picks = [e for e in order[s, t, :6] if not resident[s, t, e]][:k]
                fetched += len(picks)
                hits += sum(missing[s, t, e] for e in picks)
                if picks:
                    first += 1
                    first_hits += missing[s, t, picks[0]]
        assert got["precision"] == pytest.approx(hits / fetched)
        assert got["rank1_precision"] == pytest.approx(first_hits / first)
        assert got["recall"] == pytest.approx(hits / missing[mask].sum())
        assert got["rows_per_token"] == pytest.approx(fetched / mask.sum())
