"""The native-gate lookahead scorer on a synthetic router capture (CPU)."""

import json
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
        x = torch.randn((LAYERS, 1, HIDDEN), generator=gen).to(torch.bfloat16)
        w = torch.zeros((LAYERS, 1, 6))
        top = torch.zeros((LAYERS, 1, 6), dtype=torch.int64)
        for layer in range(LAYERS):
            ids, weights = _routed_by_gate(x[layer, 0].float(), W[layer], bias[layer])
            routes[step, layer] = ids.numpy()[::-1]  # the route log keeps router order, not rank order
            top[layer, 0] = ids.flip(0)
            w[layer, 0] = (weights / weights.sum() * 1.5).flip(0)
        capture.write(log, 100 + step, 500 + step, 1, x, top, w)
    capture.flush()
    stream = DecodeStream(list(range(LAYERS)), routes, ["r"] * steps, np.array([s > 0 for s in range(steps)]))
    return stream, W, bias


def test_the_self_check_reproduces_routes_made_by_the_gate(tmp_path):
    stream, W, bias = _synthetic(tmp_path)
    capture = router_score.load_capture(str(tmp_path / "router"))
    records = np.arange(len(stream.rids))
    assert capture.record_of_seq[100] == 0 and capture.x.shape == (10, LAYERS, 1, HIDDEN)

    def x_of(steps, layer):
        return router_score.bf16_to_f32(np.asarray(capture.x[records[steps], layer, 0]))

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


def _write_capture(prefix, *, schema, records=3, layers=2, tokens=2, hidden=4, topk=3):
    header = {"schema": schema, "layer_ids": list(range(layers)), "hidden": hidden, "topk": topk}
    if schema == 2:
        header["tokens"] = tokens
    else:
        tokens = 1
    with open(prefix + ".json", "w") as f:
        json.dump(header, f)
    np.arange(records * layers * tokens * hidden, dtype=np.uint16).tofile(prefix + ".x.bin")
    np.arange(records * layers * tokens * topk, dtype=np.float32).tofile(prefix + ".w.bin")
    if schema == 2:
        np.arange(records * layers * tokens * topk, dtype=np.int32).tofile(prefix + ".ids.bin")
        keys = np.stack([np.arange(records), np.arange(records) + 100, np.array([2, 1, 2])], axis=1)
    else:
        keys = np.stack([np.arange(records), np.arange(records) + 100], axis=1)
    keys.astype(np.int64).tofile(prefix + ".seq.bin")
    return tokens


def test_schema_2_capture_loads_every_token_with_its_live_count(tmp_path):
    prefix = str(tmp_path / "router")
    _write_capture(prefix, schema=2)
    cap = router_score.load_capture(prefix)
    assert cap.x.shape == (3, 2, 2, 4) and cap.ids.shape == cap.w.shape == (3, 2, 2, 3)
    assert cap.tokens.tolist() == [2, 1, 2]
    assert cap.record_of_seq == {0: 0, 1: 1, 2: 2}
    assert cap.ids[1, 0, 1].tolist() == [9, 10, 11]


def test_schema_1_capture_loads_as_one_token(tmp_path):
    prefix = str(tmp_path / "router")
    _write_capture(prefix, schema=1)
    cap = router_score.load_capture(prefix)
    assert cap.x.shape == (3, 2, 1, 4) and cap.w.shape == (3, 2, 1, 3)
    assert cap.tokens.tolist() == [1, 1, 1] and (cap.ids == -1).all()
