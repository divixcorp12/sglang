"""VRAM prefetch replay over DIRECT insert-on-miss, and its link pricing (CPU)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41"))

from prefetch_sim import (  # noqa: E402
    CrossLayer,
    DecodeStream,
    LinkModel,
    NoisyOracle,
    Oracle,
    Predictor,
    PrefetchReplay,
    link_exposed,
    run_arm,
    source_of,
)
from tier_sim import DirectInsertReplay, replay_direct  # noqa: E402


def _one_layer(miss_rows=1):
    # Layer 0 holds experts 0..3. After a forward routing expert 0, the victim order is 3, 2, 1, 0:
    # unrouted rows first, then by score, then the higher expert id.
    sim = PrefetchReplay({0: [0, 1, 2, 3]}, {0: 4}, 8, miss_rows=miss_rows)
    assert sim.graph_forward({0: [0]}) == {0: 0}
    return sim


def test_prefetching_the_right_row_removes_exactly_one_miss():
    base = _one_layer()
    assert base.graph_forward({0: [5]}) == {0: 1}
    sim = _one_layer()
    assert sim.graph_forward({0: [5]}, candidates=lambda layer: [5], k=1) == {0: 0}
    assert sim.last_prefetched == {0: [5]}
    # Link rows are unchanged (one prefetch in place of one miss); the demand miss is gone.
    assert sim.prefetched == 1 and 5 in sim.resident(0)


def test_a_wrong_prefetch_costs_exactly_one_row_and_removes_nothing():
    sim = _one_layer()
    assert sim.graph_forward({0: [5]}, candidates=lambda layer: [6], k=1) == {0: 1}
    assert sim.last_prefetched == {0: [6]} and sim.prefetched == 1
    # The demand miss took the shortlist victim (expert 3); the prefetch took the next one (expert 2).
    assert sim.resident(0) == {0, 1, 5, 6}


def test_a_prefetch_skips_resident_candidates_and_never_takes_the_demand_shortlist():
    sim = _one_layer()
    shortlist_expert = sim.slots[0][sim.shortlist[0][0]]
    assert shortlist_expert == 3
    sim.graph_forward({0: [7]}, candidates=lambda layer: [0, 1, 5], k=1)
    assert sim.last_prefetched == {0: [5]}
    assert sim.resident(0) == {0, 1, 5, 7}  # 3 went to the demand miss, 2 to the prefetch


def test_a_prefetch_does_not_evict_a_row_its_predictor_ranked_higher():
    sim = _one_layer()
    # Expert 2 is the first non-shortlist victim, but the predictor ranked it first: 1 goes instead.
    sim.graph_forward({0: [7]}, candidates=lambda layer: [2, 5], k=1)
    assert sim.resident(0) == {0, 2, 5, 7}


def test_no_candidates_is_the_direct_replay_exactly():
    rng = np.random.default_rng(0)
    capacity = {0: 8, 1: 9, 2: 7}
    initial = {layer: list(range(c)) for layer, c in capacity.items()}
    direct = DirectInsertReplay(initial, capacity, 32)
    prefetch = PrefetchReplay(initial, capacity, 32)
    empty = PrefetchReplay(initial, capacity, 32)
    for _ in range(200):
        routes = {layer: rng.choice(32, 6, replace=False).tolist() for layer in capacity}
        expected = direct.graph_forward(routes)
        assert prefetch.graph_forward(routes) == expected
        assert empty.graph_forward(routes, candidates=lambda layer: (), k=4) == expected
    assert prefetch.slots == direct.slots == empty.slots


def _loaded(steps, layers=2, capacity=14, experts=32, seed=1):
    rng = np.random.default_rng(seed)
    forwards = []
    for s in range(steps):
        routes = {layer: rng.choice(experts, 6, replace=False).tolist() for layer in range(layers)}
        forwards.append({"kind": "graph", "seq": s, "phase": "decode", "tokens": 1, "rids": ["r"],
                         "forward_pass_id": s, "routes": routes, "misses": {layer: 0 for layer in routes},
                         "hot": None})
    return {"layer_ids": list(range(layers)), "hot_capacity": {layer: capacity for layer in range(layers)},
            "forwards": forwards}


def _stream(loaded):
    routes = np.array([[f["routes"][layer] for layer in loaded["layer_ids"]] for f in loaded["forwards"]])
    steps = len(routes)
    return DecodeStream(loaded["layer_ids"], routes, ["r"] * steps, np.arange(steps) > 0)


def test_run_arm_without_a_predictor_matches_replay_direct(monkeypatch):
    import prefetch_sim

    monkeypatch.setattr(prefetch_sim, "NUM_EXPERTS", 32)
    loaded = _loaded(50)
    out = run_arm(loaded, _stream(loaded), Predictor(), 0, 1)
    direct = replay_direct(loaded, num_experts=32)
    assert out.demand.sum(axis=1).tolist() == [sum(sim.values()) for _, sim, _ in direct["per_forward"]]
    assert out.prefetch.sum() == 0


def test_the_oracle_turns_every_later_miss_into_a_useful_prefetch(monkeypatch):
    import prefetch_sim

    monkeypatch.setattr(prefetch_sim, "NUM_EXPERTS", 32)
    loaded = _loaded(50)
    stream = _stream(loaded)
    out = run_arm(loaded, stream, Oracle(stream), 6, 1)
    # Layer 0 of step 0 has no issue point (no previous token); everything else is prefetched in time.
    assert out.demand[1:].sum() == 0 and out.demand[0, 1] == 0
    assert (out.useful == out.prefetch).all()


def test_a_noisy_oracle_spans_the_oracle_and_pure_waste(monkeypatch):
    import prefetch_sim

    monkeypatch.setattr(prefetch_sim, "NUM_EXPERTS", 32)
    loaded = _loaded(50)
    stream = _stream(loaded)
    perfect = run_arm(loaded, stream, NoisyOracle(stream, 1.0, 6), 6, 1)
    oracle = run_arm(loaded, stream, Oracle(stream), 6, 1)
    assert (perfect.demand == oracle.demand).all() and (perfect.useful == perfect.prefetch).all()
    waste = run_arm(loaded, stream, NoisyOracle(stream, 0.0, 1), 1, 1)
    base = run_arm(loaded, stream, Predictor(), 0, 1)
    assert waste.useful.sum() == 0 and waste.prefetch.sum() > 0
    assert waste.demand.sum() >= base.demand.sum()


def test_a_gated_cross_layer_predictor_names_only_confident_rows(monkeypatch):
    import prefetch_sim

    monkeypatch.setattr(prefetch_sim, "NUM_EXPERTS", 32)
    # Layer 1 always routes expert 20 + (layer 0's first expert mod 6), plus five fixed experts.
    rng = np.random.default_rng(2)
    routes = []
    for _ in range(200):
        first = rng.choice(20, 6, replace=False)
        routes.append([first.tolist(), [20 + int(first[0]) % 6, 26, 27, 28, 29, 30]])
    routes = np.array(routes)
    stream = DecodeStream([0, 1], routes, ["r"] * 200, np.arange(200) > 0)
    train = np.arange(200) < 150
    gated = CrossLayer(stream, train, 1, min_prob=0.99)
    for step in range(150, 200):
        ranked = set(int(e) for e in gated.rank(step, 1, 1))
        assert ranked == {26, 27, 28, 29, 30}  # the one certain set; the varying expert is never certain
    assert CrossLayer(stream, train, 1, min_prob=0.5).rank(150, 1, 1)[0] in (26, 27, 28, 29, 30)


def test_source_reaches_the_previous_token_only_within_a_request():
    stream = DecodeStream([0, 1, 2], np.zeros((3, 3, 6), dtype=np.int64), ["a", "a", "b"],
                          np.array([False, True, False]))
    assert source_of(stream, 1, 2, 1) == (1, 1)
    assert source_of(stream, 1, 0, 2) == (0, 1)
    assert source_of(stream, 2, 0, 1) is None


def _price(demand, prefetch, horizon, budget, continuous=None):
    demand, prefetch = np.asarray(demand, dtype=float), np.asarray(prefetch, dtype=float)
    continuous = np.ones(len(demand), dtype=bool) if continuous is None else np.asarray(continuous)
    return link_exposed(demand, prefetch, continuous, horizon, LinkModel(budget))


def test_a_prefetch_the_idle_window_cannot_hold_delays_its_target():
    # One row aimed at layer 1, issued in layer 0's window: 0.85 fits, 0.15 is exposed at layer 1.
    assert _price([[0, 0]], [[0, 1]], 1, 0.85)[0] == pytest.approx(0.15)
    assert _price([[0, 0]], [[0, 1]], 1, 1.7)[0] == pytest.approx(0.0)
    # Demand is always exposed.
    assert _price([[2, 1]], [[0, 0]], 1, 0.85)[0] == pytest.approx(3.0)


def test_a_longer_horizon_spreads_a_prefetch_over_more_windows():
    assert _price([[0, 0, 0]], [[0, 0, 1]], 1, 0.5)[0] == pytest.approx(0.5)
    assert _price([[0, 0, 0]], [[0, 0, 1]], 2, 0.5)[0] == pytest.approx(0.0)


def test_windows_drain_prefetches_first_in_first_out():
    # h=2, budget 0.5: rows for layers 2 and 3 share windows 0-2; layer 2's row takes windows 0 and 1,
    # layer 3's gets window 2 only and exposes its other half.
    assert _price([[0, 0, 0, 0]], [[0, 0, 1, 1]], 2, 0.5)[0] == pytest.approx(0.5)


def test_a_prefetch_can_use_the_previous_tokens_tail_windows():
    # Layer 0 of step 1, h=1: issued in layer 1's window of step 0; half of it is exposed in step 1.
    exposed = _price([[0, 0], [0, 0]], [[0, 0], [1, 0]], 1, 0.5)
    assert exposed.tolist() == pytest.approx([0.0, 0.5])
    # Across a request boundary nothing is issued ahead, so the row is not priced there.
    exposed = _price([[0, 0], [0, 0]], [[0, 0], [1, 0]], 1, 0.5, continuous=[False, False])
    assert exposed.tolist() == pytest.approx([0.0, 0.0])
