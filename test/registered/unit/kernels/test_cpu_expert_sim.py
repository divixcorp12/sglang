"""The CPU-expert offline model: c_cpu interpolation, per-layer cost, policy ordering, n/m counting (CPU only)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41"))

import tier_sim  # noqa: E402
from cpu_expert_sim import (  # noqa: E402
    MAX_ROUTES,
    CostModel,
    _cpu_chooser,
    _queue,
    c_cpu_at,
    histogram,
    load_c_cpu_table,
    predict,
    replay_nm,
    split_costs,
)

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

P0_12 = [0.575, 0.518, 0.475]


def _model(points, nvme_ms=0.0, gpu_ms=0.0):
    return CostModel(points, c_link=1.0, handoff=0.02, nvme_ms=nvme_ms, gpu_ms=gpu_ms)


def test_c_cpu_interpolates_through_the_three_calibrated_points():
    assert [c_cpu_at(k, P0_12) for k in (1, 2, 6)] == pytest.approx(P0_12)
    assert c_cpu_at(4, P0_12) == pytest.approx((0.518 + 0.475) / 2)
    assert c_cpu_at(3, P0_12) == pytest.approx(0.518 + (0.475 - 0.518) / 4)
    # Held flat outside the calibrated range.
    assert c_cpu_at(9, P0_12) == pytest.approx(0.475)
    assert c_cpu_at(0.5, P0_12) == pytest.approx(0.575)


def test_c_cpu_table_accepts_inline_json_and_rejects_wrong_arity():
    assert load_c_cpu_table('{"16": [0.5, 0.4, 0.3]}') == {"16": [0.5, 0.4, 0.3]}
    with pytest.raises(ValueError):
        load_c_cpu_table('{"16": [0.5, 0.4]}')


def test_two_ram_hits_split_gives_the_expected_layer_time():
    model = _model([0.4, 0.4, 0.4])
    tables = model.policy_tables()
    # off: both rows cross the link. k*(2) = 2 (0.02 + 0.8 < 1.0): the CPU takes both, the link none.
    assert tables["off"][2, 0] == pytest.approx(2.0)
    assert tables["k*"][2, 0] == pytest.approx(0.02 + 2 * 0.4)
    assert tables["all-cpu"][2, 0] == pytest.approx(0.82)
    assert tables["k*-cap1"][2, 0] == pytest.approx(1.0)  # k = 1: max(0.42, 1 row * 1.0)
    assert tables["k*-cap2"][2, 0] == pytest.approx(0.82)
    # With no RAM hit there is no handoff at all.
    assert tables["k*"][0, 0] == 0.0


def test_nvme_misses_are_exposed_and_still_cross_the_link_unless_computed_on_cpu():
    tables = _model([0.4, 0.4, 0.4], nvme_ms=1.5).policy_tables()
    assert tables["off"][0, 1] == pytest.approx(1.0 + 1.5)  # one landed row over the link, plus its read
    assert tables["k*"][0, 1] == pytest.approx(1.0 + 1.5)  # k*(0) = 0: nothing to move
    assert tables["k*+nvme"][0, 1] == pytest.approx(0.02 + 0.4 + 1.5)  # the landed row computed on the CPU
    assert tables["k*+nvme"][2, 1] == pytest.approx(0.02 + 3 * 0.4 + 1.5)  # k = 2 hits and the landed row


def test_prediction_averages_layers_and_adds_the_gpu_term():
    model = _model([0.4, 0.4, 0.4], gpu_ms=14.0)
    n = np.array([[2, 0], [0, 0]])
    m = np.zeros_like(n)
    out = predict(n, m, model)
    assert out["off"]["ms_per_token"] == pytest.approx((2.0 + 0.0) / 2 + 14.0)
    assert out["k*"]["ms_per_token"] == pytest.approx((0.82 + 0.0) / 2 + 14.0)
    assert out["k*"]["gain_vs_off_pct"] == pytest.approx(100 * (1.0 - 0.41) / 15.0)


def test_histogram_counts_values_zero_through_six():
    values = np.array([[0, 1, 1], [6, 2, 0]])
    assert histogram(values) == [2, 2, 1, 0, 0, 0, 1]


@pytest.mark.parametrize("points", [[0.4, 0.4, 0.4], P0_12, [0.717, 0.658, 0.644], [1.5, 1.5, 1.5]])
@pytest.mark.parametrize("nvme_ms", [0.0, 1.5])
def test_exact_split_never_loses_to_off_or_all_cpu(points, nvme_ms):
    model = _model(points, nvme_ms=nvme_ms)
    tables = model.policy_tables()
    for n in range(MAX_ROUTES + 1):
        for m in range(MAX_ROUTES + 1 - n):
            best = tables["k*-exact"][n, m]
            assert best <= tables["off"][n, m] + 1e-12
            assert best <= tables["all-cpu"][n, m] + 1e-12
            assert best <= tables["k*-cap1"][n, m] + 1e-12


def test_scalar_split_table_never_loses_to_off_or_all_cpu_when_c_cpu_is_flat():
    tables = _model([0.5, 0.5, 0.5]).policy_tables()
    for n in range(MAX_ROUTES + 1):
        assert tables["k*"][n, 0] <= min(tables["off"][n, 0], tables["all-cpu"][n, 0]) + 1e-12


def _graph(seq, routes, hot):
    return {
        "kind": "graph",
        "seq": seq,
        "phase": "decode",
        "tokens": 1,
        "routes": {0: routes},
        "misses": {0: 0},
        "hot": {0: hot},
    }


def test_replay_counts_ram_hits_and_nvme_misses_per_decode_layer():
    # One layer, one hot slot holding expert 0, a 4-row pinned tier over 8 experts.
    # 1: experts 1 and 2 miss both tiers (m = 2); DIRECT inserts expert 1 into the hot slot.
    # 2: expert 2 misses VRAM and hits the tier (n = 1).
    # 3: expert 0 was never admitted to the tier, so it misses both (m = 1).
    loaded = {
        "layer_ids": [0],
        "hot_capacity": {0: 1},
        "forwards": [_graph(1, [1, 2], [0]), _graph(2, [1, 2], [1]), _graph(3, [0, 1], [1])],
    }
    out = replay_nm(loaded, ram_rows=4, num_experts=8)
    assert out["n"].tolist() == [[0], [1], [0]]
    assert out["m"].tolist() == [[2], [0], [1]]
    assert out["validation"]["decode_tokens"] == 3


def _ram_hit_loaded():
    # One layer, one hot slot, a 4-row pinned tier. 2 and 3 miss both and are inserted in turn, so the tier
    # holds 2 when forward 3 routes it again: a RAM hit (n = 1), which the CPU policies do not insert.
    routes = [[2], [3], [2], [2], [2]]
    return {
        "layer_ids": [0],
        "hot_capacity": {0: 1},
        "forwards": [_graph(seq, r, [0]) for seq, r in enumerate(routes, start=1)],
    }


def test_insert_all_policy_is_the_p1_replay():
    loaded = _ram_hit_loaded()
    base = replay_nm(loaded, ram_rows=4, num_experts=8)
    out = replay_nm(loaded, ram_rows=4, num_experts=8, policy="insert_all", split=[0, 1, 1, 2, 3, 3, 4])
    assert out["n"].tolist() == base["n"].tolist() == [[0], [0], [1], [0], [0]]
    assert out["m"].tolist() == base["m"].tolist() == [[1], [1], [0], [0], [0]]
    assert out["residency"]["hot_hit_rate"] == pytest.approx(2 / 5)
    assert out["residency"]["cpu_lanes_per_token"] == pytest.approx(1 / 5)


def test_a_cpu_lane_is_never_inserted_and_a_deferred_one_lands_a_forward_later():
    loaded = _ram_hit_loaded()
    split = [0, 1, 1, 2, 3, 3, 4]
    order = replay_nm(loaded, ram_rows=4, num_experts=8, policy="cpu_lane_order", split=split)
    assert order["n"].tolist() == [[0], [0], [1], [1], [1]]  # 2 stays a RAM hit: the victim kept 3
    assert order["residency"]["hot_hit_rate"] == 0.0
    late = replay_nm(loaded, ram_rows=4, num_experts=8, policy="cpu_deferred", split=split)
    assert late["n"].tolist() == [[0], [0], [1], [1], [0]]  # inserted at forward 4's commit
    assert late["residency"]["deferred_link_rows_per_token"] == pytest.approx(1 / 5)


def test_cpu_by_score_takes_the_lowest_ranked_ram_hit():
    sim = tier_sim.DirectInsertReplay({0: [0]}, {0: 1}, 8)
    sim.scores[0, 5], sim.scores[0, 6] = 2.0, 1.0
    for policy, want in (("cpu_lane_order", {5}), ("cpu_by_score", {6})):
        chosen = {}
        assert _cpu_chooser(policy, sim, [0, 1, 1], {0: [5, 6]}, chosen)(0, [5, 6]) == want
    sim.scores[0, 6] = 2.0  # a tie goes to the higher expert, as the victim ranking evicts it first
    assert _cpu_chooser("cpu_by_score", sim, [0, 1, 1], {0: [5, 6]}, {})(0, [5, 6]) == {6}


def test_a_cpu_lane_keeps_its_victim_while_the_next_lane_lands_in_its_own_entry():
    """direct_commit_gather_kernel maps no CPU lane and does not compact the others: with shortlist
    [expert 2, 1, 0], lane 0 (expert 5, CPU) leaves expert 2's slot alone and lane 1 (6) still takes 1's."""
    plain = tier_sim.DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=3)
    plain.graph_forward({0: [5, 6]})
    assert plain.resident(0) == {0, 5, 6}

    sim = tier_sim.DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=3)
    assert sim.graph_forward({0: [5, 6]}, cpu_lanes=lambda layer, misses: {misses[0]}) == {0: 2}
    assert sim.resident(0) == {0, 2, 6}


def test_deferred_inserts_take_spare_entries_in_queue_order():
    def forward_one():
        sim = tier_sim.DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=3)
        sim.graph_forward({0: [5, 6]}, cpu_lanes=lambda layer, misses: {misses[0]})
        return sim

    # Shortlist now [expert 2, 0, 6]; the forward reads 0, leaving 2's and 6's slots spare.
    sim = forward_one()
    assert sim.graph_forward({0: [0]}, deferred={0: [5]}) == {0: 0}
    assert sim.resident(0) == {0, 5, 6} and sim.deferred_inserted == 1
    # 6 was routed in the open window, so an unrouted-only spare leaves one entry: the queue's first takes it.
    sim = forward_one()
    sim.graph_forward({0: [0]}, deferred={0: [7, 5]}, deferred_unrouted=True)
    assert sim.resident(0) == {0, 6, 7} and sim.deferred_dropped == 1


def test_direct_replay_without_cpu_lanes_is_unchanged():
    rng = np.random.default_rng(0)
    plain = tier_sim.DirectInsertReplay({0: [0, 1, 2, 3]}, {0: 4}, 16)
    hooked = tier_sim.DirectInsertReplay({0: [0, 1, 2, 3]}, {0: 4}, 16)
    for _ in range(50):
        routes = {0: rng.choice(16, 6, replace=False).tolist()}
        assert plain.graph_forward(routes) == hooked.graph_forward(routes, cpu_lanes=lambda layer, misses: set())
        assert plain.slots == hooked.slots
    assert np.array_equal(plain.scores, hooked.scores)


def test_cpu_by_score_ranks_an_expert_routed_in_the_window_last_whatever_its_score():
    sim = tier_sim.DirectInsertReplay({0: [0]}, {0: 1}, 8)
    sim.scores[0, 5], sim.scores[0, 6] = 0.0, 2.0
    assert _cpu_chooser("cpu_by_score", sim, [0, 1, 1], {0: [5, 6]}, {})(0, [5, 6]) == {5}
    sim.routed[0, 5] = True
    assert _cpu_chooser("cpu_by_score", sim, [0, 1, 1], {0: [5, 6]}, {})(0, [5, 6]) == {6}


def test_chosen_lanes_are_recorded_in_lane_order_and_queued_by_policy():
    sim = tier_sim.DirectInsertReplay({0: [0]}, {0: 1}, 8)
    sim.scores[0, 5], sim.scores[0, 6], sim.scores[0, 7] = 3.0, 1.0, 2.0
    chosen = {}
    assert _cpu_chooser("cpu_by_score", sim, [0, 1, 1, 2], {0: [5, 6, 7]}, chosen)(0, [5, 6, 7]) == {6, 7}
    assert chosen == {0: [6, 7]}
    assert _queue("cpu_deferred", sim, {0: [5, 6, 7]}) == {0: [5, 6, 7]}
    assert _queue("cpu_deferred_scoreq", sim, {0: [5, 6, 7]}) == {0: [5, 7, 6]}  # highest insert score first
    chosen = {}
    assert _cpu_chooser("cpu_numa_local", sim, [0, 1], {0: [5, 6]}, chosen, {0: {6}})(0, [5, 6]) == {6}


def test_a_deferred_row_evicted_from_the_pinned_tier_is_dropped():
    # A 3-row tier: forward 3's CPU lane (expert 2) is queued; the prefill's two admissions evict it.
    loaded = _ram_hit_loaded()
    loaded["forwards"] = loaded["forwards"][:3] + [
        {"kind": "eager", "phase": "extend", "tokens": 20, "counts": {0: ([4, 5], [1, 1])}, "misses": {0: 2}},
        _graph(4, [0], [0]),
    ]
    out = replay_nm(loaded, ram_rows=3, num_experts=8, policy="cpu_deferred", split=[0, 1, 1, 2, 3, 3, 4])
    assert out["residency"]["deferred_evicted_per_token"] == pytest.approx(1 / 4)
    assert out["residency"]["deferred_link_rows_per_token"] == 0.0


def test_background_rows_cost_nothing_in_cpu_slack_and_all_in_the_worst_case():
    model = _model([0.4, 0.4, 0.4])
    split = [0, 1, 1]
    # Layer 0: one RAM hit on the CPU (0.42 ms), no link rows: 0.42 ms of slack. Layer 1 has none.
    n, m = np.array([[1, 0]]), np.zeros((1, 2), dtype=np.int64)
    in_slack = split_costs(n, m, model, split, np.array([[0, 1]]))
    assert in_slack["uncosted"] == pytest.approx(0.42)
    assert in_slack["worst"] == pytest.approx(0.42 + 1.0)
    assert in_slack["amortised"] == pytest.approx(0.42 + 1.0 - 0.42)
    assert split_costs(n, m, model, split)["worst"] == pytest.approx(0.42)


def test_promotion_takes_the_lowest_ranked_victim_only_when_it_leads_by_the_margin():
    """decide_residency_on_device: candidates in (-score, expert) order against victims in _rank_victims'
    order, a swap only past margin, stopping at the first that fails; only pinned-tier rows are candidates."""

    def replay():
        sim = tier_sim.DirectInsertReplay({0: [0, 1]}, {0: 2}, 8)
        for expert, score in ((0, 0.5), (1, 3.0), (4, 10.0), (5, 4.0), (6, 2.0), (7, 3.9)):
            sim.scores[0, expert] = score
        return sim

    sim = replay()  # 4 is not in the pinned tier; 5 beats expert 0 (0.5 + 1); 7 does not beat expert 1 (3 + 1)
    assert sim.promote({0: {5: 0, 6: 1, 7: 2}}, 4, margin=1.0) == {0: 1}
    assert sim.resident(0) == {1, 5} and sim.promoted == 1
    sim = replay()
    assert sim.promote({0: {5: 0, 6: 1, 7: 2}}, 4, margin=0.0) == {0: 2}  # 7 now leads 1 by 0.9
    assert sim.resident(0) == {5, 7}
    sim = replay()
    assert sim.promote({0: {5: 0, 6: 1, 7: 2}}, 1, margin=0.0) == {0: 1}  # at most P per layer
    assert sim.resident(0) == {1, 5}
    sim = replay()
    sim.route_counts[0, 0] = 1.0  # routed in the open window: expert 0 now ranks after 1
    assert sim.promote({0: {5: 0}}, 1, margin=0.0) == {0: 1} and sim.resident(0) == {0, 5}


@pytest.mark.parametrize("interval, boundary", [(2, 4), (3, 6)])
def test_promotion_fires_every_n_decode_forwards(interval, boundary):
    # A CPU lane (expert 2) is never inserted; its score passes 3's by the margin between forwards 4 and 5, so the
    # first boundary at or after decode index 4 promotes it.
    loaded = {
        "layer_ids": [0],
        "hot_capacity": {0: 1},
        "forwards": [_graph(seq, r, [0]) for seq, r in enumerate([[2], [3]] + [[2]] * 6, start=1)],
    }
    out = replay_nm(loaded, 4, 8, policy=f"cpu_by_score_promote_N{interval}_P1", split=[0, 1, 1, 2, 3, 3, 4])
    assert out["b"][:, 0].tolist() == [int(i == boundary) for i in range(8)]
    assert out["n"][:, 0].tolist() == [0, 0] + [1] * (boundary - 2) + [0] * (8 - boundary)
    assert out["residency"]["promoted_rows_per_token"] == pytest.approx(1 / 8)


def test_promoted_rows_amortise_over_their_window():
    model = _model([0.4, 0.4, 0.4])
    n, m = np.array([[1], [1]]), np.zeros((2, 1), dtype=np.int64)  # 0.42 ms of CPU slack per token
    b = np.array([[1], [0]])
    assert split_costs(n, m, model, [0, 1], b)["amortised"] == pytest.approx(0.42 + 0.58 / 2)
    assert split_costs(n, m, model, [0, 1], b, window=2)["amortised"] == pytest.approx(0.42 + 0.16 / 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
