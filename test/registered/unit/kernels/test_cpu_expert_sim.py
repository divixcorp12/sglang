"""The CPU-expert offline model: c_cpu interpolation, per-layer cost, policy ordering, n/m counting (CPU only)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41"))

from cpu_expert_sim import (  # noqa: E402
    MAX_ROUTES,
    CostModel,
    c_cpu_at,
    histogram,
    load_c_cpu_table,
    predict,
    replay_nm,
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
