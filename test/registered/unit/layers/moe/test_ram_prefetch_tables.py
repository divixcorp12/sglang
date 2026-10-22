"""The RAM prefetch's Python side (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1): the router gates registered at
load, the per-row target table and its host copy, and the service's call into the host (CPU)."""

import logging
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import ram_prefetch
from sglang.srt.layers.moe.cpu_experts.threading_config import NodePlan
from sglang.srt.layers.moe.expert_engine import service as module
from sglang.kernels.ops.moe.expert_stream_transport import CAND_PAGE_BYTES, new_candidate_page
from sglang.srt.layers.moe.ram_prefetch import (
    MARGIN_FLOORS,
    GpuScorer,
    RouterGate,
    clear_router_gates,
    load_margin_floors,
    margin_floors_path,
    prefetch_tables,
    prefetch_targets,
    register_moe_gates,
    read_margin_floors,
    register_router_gate,
    registered_gates,
    spec_score_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import dsv41_exl3_pairing

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def no_gates():
    clear_router_gates()
    yield
    clear_router_gates()


def _gate(scale=1.0, experts=4, hidden=8, bias=True, top_k=6):
    w = (torch.arange(experts * hidden, dtype=torch.float32).reshape(experts, hidden) * scale).to(torch.bfloat16)
    return RouterGate(w, torch.arange(experts, dtype=torch.float32) * scale if bias else None, top_k)


def test_each_row_targets_the_next_consecutive_layer_with_a_biased_gate():
    """Row 0's next layer is hash-routed (no bias), row 2's is not consecutive, row 4 is last."""
    gates = {1: _gate(bias=False), 2: _gate(2.0), 4: _gate(4.0), 5: _gate(5.0)}
    t = prefetch_tables([0, 1, 2, 4, 5], gates, hidden=8)
    assert t.targets.tolist() == [[-1, -1], [2, 0], [-1, -1], [4, 1], [-1, -1]]
    assert t.gates.dtype == torch.bfloat16 and t.gates.shape == (2, 4, 8) and t.gates.device.type == "cpu"
    assert torch.equal(t.gates[0], gates[2].weight) and torch.equal(t.gates[1], gates[5].weight)
    assert torch.equal(t.bias[1], gates[5].bias) and t.bias.dtype == torch.float32 and t.top_k == 6


_H2_LAYERS = [0, 1, 2, 3, 5, 6, 7]


def _h2_gates():
    """Layer 3's gate is hash-routed; layer 4 is not streamed, so row 2 has no layer-4 gate."""
    return {2: _gate(2.0), 3: _gate(3.0, bias=False), 5: _gate(5.0), 6: _gate(6.0), 7: _gate(7.0)}


def test_horizon_two_targets_the_layer_two_ahead_over_consecutive_rows():
    """Row 3 (layer 3) would reach layer 5 across the 3->5 gap, row 1's layer-3 gate has no bias, the last two rows
    have no layer two ahead. Mutant: horizon ignored (r + 1) -- red; span check dropped -- row 3 targets row 4, red."""
    from sglang.srt.layers.moe.residency_capture_ref import horizon_targets

    gates = _h2_gates()
    picked = prefetch_targets(_H2_LAYERS, gates, hidden=8, horizon=2)
    assert picked.targets.tolist() == [[2, 0], [-1, -1], [-1, -1], [-1, -1], [6, 1], [-1, -1], [-1, -1]]
    assert picked.gates == (gates[2], gates[7])
    ref, _ = horizon_targets(_H2_LAYERS)
    assert [picked.targets[r, 0].item() for r in (0, 4)] == [ref[0][1], ref[4][1]]


def test_horizon_two_cpu_tables_copy_the_gate_two_layers_ahead():
    """The CPU tables carry row T+2's weights and bias in picks order. Mutant: the table built with horizon 1 -- the
    gate copies are layer T+1's, red."""
    gates = _h2_gates()
    t = prefetch_tables(_H2_LAYERS, gates, hidden=8, horizon=2)
    assert t.targets.tolist() == [[2, 0], [-1, -1], [-1, -1], [-1, -1], [6, 1], [-1, -1], [-1, -1]]
    assert t.gates.shape == (2, 4, 8) and t.bias.dtype == torch.float32
    assert torch.equal(t.gates[0], gates[2].weight) and torch.equal(t.gates[1], gates[7].weight)
    assert torch.equal(t.bias[0], gates[2].bias) and torch.equal(t.bias[1], gates[7].bias)


@pytest.mark.parametrize(
    "gates, why",
    [
        ({}, "no streamed row has a next layer"),
        ({1: _gate(hidden=16)}, "hidden size 16"),
        ({1: _gate(), 2: _gate(top_k=8)}, "top_k"),
        ({1: _gate(), 2: _gate(experts=5)}, "experts"),
    ],
)
def test_a_table_the_host_could_not_score_is_refused(gates, why):
    with pytest.raises(ValueError, match=why):
        prefetch_tables([0, 1, 2], gates, hidden=8)


def test_register_moe_gates_takes_every_moe_layer_and_drops_a_hash_layers_bias():
    gate = SimpleNamespace(weight=torch.zeros(4, 8), e_score_correction_bias=torch.ones(4))
    layers = {
        0: SimpleNamespace(mlp=SimpleNamespace(gate=gate, is_hash=True)),
        1: SimpleNamespace(mlp=SimpleNamespace(gate=gate, is_hash=False)),
        2: SimpleNamespace(mlp=SimpleNamespace()),  # a dense layer
    }
    assert register_moe_gates(layers, 6) == 2
    got = registered_gates()
    assert sorted(got) == [0, 1] and got[0].bias is None and got[1].bias is gate.e_score_correction_bias
    assert got[1].top_k == 6


def test_a_layer_registered_again_with_another_gate_is_refused():
    """Another model's gate for the same layer id (a draft's) must not replace the target's."""
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    register_router_gate(1, gate.weight, gate.bias, 6)
    with pytest.raises(ValueError, match="layer 1 already has a router gate"):
        register_router_gate(1, _gate().weight, gate.bias, 6)
    assert registered_gates()[1].weight is gate.weight


def test_the_service_enables_the_host_with_the_registered_gates_options_and_spare_cores(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_DSV41_RAM_PREFETCH_SCORER", "cpu")
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    calls = []
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
    numa = SimpleNamespace(
        nodes=1, plans=[NodePlan(group=0, node=0, ram=17, cpu=(8, 9), sq=None, busy_poll=True, spec=(0, 1))]
    )
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    with envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER.override(2):
        module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], numa, cpu)
    (targets, gates, bias), kw = calls[0]
    assert targets.tolist() == [[1, 0], [-1, -1]] and torch.equal(gates[0], gate.weight)
    assert kw == dict(
        top_k=6, per_token=1, per_layer=2, cores=[[0, 1]], top_k_only=False, idle_drive=False, idle_deadline_s=0.004,
        idle_spread=False, idle_pieces=1, promote=False,
    )
    with envs.SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY.override(True):
        module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], numa, cpu)
    assert calls[1][1]["top_k_only"] is True
    with envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE.override(True), \
            envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US.override(2500):
        module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], numa, cpu)
    assert (calls[2][1]["idle_drive"], calls[2][1]["idle_deadline_s"]) == (True, 0.0025)
    with envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE.override(True), \
            envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_SPREAD.override(True), \
            envs.SGLANG_DSV41_RAM_PREFETCH_IDLE_PIECES.override(2), caplog.at_level(logging.INFO):
        module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], numa, cpu)
    assert (calls[3][1]["idle_spread"], calls[3][1]["idle_pieces"]) == (True, 2)
    assert "idle-drive reads (own reader per group, deadline 4 ms, spread ties, 2 pieces in flight)" in caplog.text
    with pytest.raises(RuntimeError, match="needs SGLANG_DSV41_CPU_EXPERTS"):
        module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], numa, None)


def _numa():
    return SimpleNamespace(
        nodes=1, plans=[NodePlan(group=0, node=0, ram=17, cpu=(8, 9), sq=None, busy_poll=True, spec=(0, 1))]
    )


def test_the_gpu_scorer_enables_the_host_with_a_candidate_page_and_no_host_gate_copy():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    calls = []
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("gpu"), envs.SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS.override(""):
        scorer = module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)
    (targets, gates, bias), kw = calls[0]
    assert targets.tolist() == [[1, 0], [-1, -1]] and gates is None and bias is None
    page = kw.pop("candidates")
    assert page.dtype == torch.uint8 and page.numel() == CAND_PAGE_BYTES and not page.is_pinned()
    assert kw == dict(
        top_k=6, per_token=1, per_layer=1, cores=[[0, 1]], top_k_only=False, min_margin=None, idle_drive=False,
        idle_deadline_s=0.004, idle_spread=False, idle_pieces=1, promote=False,
    )
    assert scorer.candidates is page and scorer.picked.gates[0].weight is gate.weight
    assert (scorer.per_token, scorer.top_k_only, scorer.picked.top_k) == (1, False, 6)


def test_the_cpu_scorer_returns_no_gpu_scorer_and_an_unknown_one_is_refused():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: None)
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("cpu"):
        assert module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], _numa(), cpu) is None
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("tpu"):
        with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER"):
            module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)


def test_the_service_builds_both_scorers_tables_at_the_configured_horizon(caplog):
    """Horizon 2 on layers [0,1,2]: row 0 targets layer 2's gate for both scorers, the host kwargs match horizon 1's
    and only the log names the horizon. Mutant: horizon not passed to prefetch_targets / prefetch_tables -- row 0
    targets row 1, red."""
    gates = {1: _gate(1.0), 2: _gate(2.0)}
    for layer, gate in gates.items():
        register_router_gate(layer, gate.weight, gate.bias, 6)
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])

    def enable(scorer, horizon):
        calls = []
        host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
        caplog.clear()
        with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override(scorer), \
                envs.SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS.override(""), \
                envs.SGLANG_DSV41_RAM_PREFETCH_HORIZON.override(horizon), caplog.at_level(logging.INFO):
            result = module.ExpertEngineService._enable_ram_prefetch(host, [0, 1, 2], _numa(), cpu)
        (args, kw) = calls[0]
        kw.pop("candidates", None)
        return result, args, kw, caplog.text

    scorer, args, kw_h2, log = enable("gpu", 2)
    assert args[0].tolist() == [[2, 0], [-1, -1], [-1, -1]]
    assert torch.equal(scorer.picked.gates[0].weight, gates[2].weight)
    assert torch.equal(scorer.picked.gates[0].bias, gates[2].bias)
    assert "1 of 3 rows target layer +2" in log
    _, args, kw_h1, log = enable("gpu", 1)
    assert args[0].tolist() == [[1, 0], [2, 1], [-1, -1]]
    assert "2 of 3 rows target the next layer" in log
    assert kw_h1 == kw_h2

    _, args, kw_h2, log = enable("cpu", 2)
    assert args[0].tolist() == [[2, 0], [-1, -1], [-1, -1]]
    assert torch.equal(args[1][0], gates[2].weight) and torch.equal(args[2][0], gates[2].bias)
    assert "1 of 3 rows target layer +2" in log
    _, _, kw_h1, _ = enable("cpu", 1)
    assert kw_h1 == kw_h2


def test_the_gpu_scorers_table_has_a_row_once_its_target_attached():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    picked = prefetch_targets([0, 1, 2], registered_gates(), hidden=8)
    hot = torch.full((3,), -1, dtype=torch.int64)
    assert spec_score_rows(picked, {}) == {}
    rows = spec_score_rows(picked, {1: (hot, 3)})
    assert list(rows) == [0]
    entry = rows[0]
    assert (entry.target, entry.hot_capacity) == (1, 3) and entry.hot_slots is hot
    assert torch.equal(entry.weight, gate.weight) and torch.equal(entry.bias, gate.bias)


@pytest.mark.parametrize(
    "weight_dtype, bias_dtype, why",
    [(torch.float16, torch.float32, "bf16 or fp32 gate"), (torch.bfloat16, torch.bfloat16, "fp32 bias")],
)
def test_a_gate_the_kernels_cannot_read_is_refused_when_the_table_is_built(weight_dtype, bias_dtype, why):
    gate = _gate()
    register_router_gate(1, gate.weight.to(weight_dtype), gate.bias.to(bias_dtype), 6)
    picked = prefetch_targets([0, 1], registered_gates(), hidden=8)
    with pytest.raises(ValueError, match=why):
        spec_score_rows(picked, {1: (torch.full((3,), -1, dtype=torch.int64), 3)})


def test_the_service_binds_each_row_as_its_target_attaches_and_flags_rows_with_a_target():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    svc = module.ExpertEngineService(**dsv41_exl3_pairing())
    picked = prefetch_targets([0, 1], registered_gates(), hidden=8)
    svc._spec_gpu = GpuScorer(picked, new_candidate_page(pin=False), per_token=1, top_k_only=False)
    hot0, hot1 = (torch.full((3,), -1, dtype=torch.int64) for _ in range(2))
    b0 = SimpleNamespace(spec_expected=False, spec_score=None, hot_slots=hot0, hot_capacity=3)
    b1 = SimpleNamespace(spec_expected=False, spec_score=None, hot_slots=hot1, hot_capacity=3)
    svc._note_spec_row(0, b0)
    assert b0.spec_expected and b0.spec_score is None  # row 1, its target, has not attached
    svc._note_spec_row(1, b1)
    assert b0.spec_score.target == 1 and b0.spec_score.hot_slots is hot1
    assert not b1.spec_expected and b1.spec_score is None


def test_the_candidate_page_and_the_score_scratch_are_quarantined(monkeypatch):
    """The select kernel writes the page through UVA: a shutdown that cannot establish the GPU stopped keeps it."""
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    svc = module.ExpertEngineService(**dsv41_exl3_pairing())
    page = new_candidate_page(pin=False)
    svc._spec_gpu = GpuScorer(prefetch_targets([0, 1], registered_gates(), hidden=8), page, 1, False)
    owned = []
    monkeypatch.setattr(module, "quarantine_host_slabs", owned.extend)
    svc._quarantine("test")
    assert any(t is page for t in owned)


def _floors_file(tmp_path, min_margin, **extra):
    import json

    path = tmp_path / "floors.json"
    path.write_text(json.dumps({"min_margin": min_margin, **extra}))
    return str(path)


def test_margin_floors_load_per_target_row_and_a_row_the_file_omits_keeps_every_read(tmp_path):
    floors = load_margin_floors(_floors_file(tmp_path, {"1": 0.25, "3": float("inf")}, floor=0.3), rows=4)
    assert floors.dtype == torch.float32
    assert floors.tolist() == [float("-inf"), 0.25, float("-inf"), float("inf")]


@pytest.mark.parametrize("min_margin", [{"4": 0.1}, {"-1": 0.1}, {"1": float("nan")}, {"x": 0.1}, {"1": "0.1"}])
def test_margin_floors_refuse_a_row_out_of_range_or_a_value_that_is_not_a_number(tmp_path, min_margin):
    with pytest.raises(ValueError, match="margin floors"):
        load_margin_floors(_floors_file(tmp_path, min_margin), rows=4)


def test_the_gpu_scorer_passes_the_margin_floors_file_to_the_host(tmp_path):
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    calls = []
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    path = _floors_file(tmp_path, {"1": 0.5})
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("gpu"), envs.SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS.override(
        path
    ):
        module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)
    assert calls[0][1]["min_margin"].tolist() == [float("-inf"), 0.5]


def test_the_prefetch_alone_runs_the_gpu_scorer_with_the_packaged_margin_floors(caplog):
    """DSV41_REFERENCE.md 33.14-33.18's best arm is the default: SGLANG_DSV41_RAM_PREFETCH=1 needs nothing else."""
    import os

    assert envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.get() == "gpu"
    assert not envs.SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS.is_set()
    assert os.path.dirname(MARGIN_FLOORS) == os.path.dirname(ram_prefetch.__file__) and os.path.isfile(MARGIN_FLOORS)
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    calls = []
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    rows = max(read_margin_floors(MARGIN_FLOORS)) + 1
    with caplog.at_level("INFO", logger=module.logger.name):
        scorer = module.ExpertEngineService._enable_ram_prefetch(host, list(range(rows)), _numa(), cpu)
    assert isinstance(scorer, GpuScorer)
    assert torch.equal(calls[0][1]["min_margin"], load_margin_floors(MARGIN_FLOORS, rows))
    assert "GPU scorer" in caplog.text and f"margin floors {MARGIN_FLOORS}" in caplog.text


def test_empty_margin_floors_read_every_candidate_and_the_cpu_scorer_has_none(caplog):
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    calls = []
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    with envs.SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS.override(""):
        assert margin_floors_path("gpu") is None
        module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)
    assert calls[0][1]["min_margin"] is None
    assert margin_floors_path("gpu") == MARGIN_FLOORS and margin_floors_path("cpu") is None
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("cpu"), caplog.at_level("INFO", logger=module.logger.name):
        module.ExpertEngineService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)
    assert "min_margin" not in calls[1][1]
    assert "CPU scorer" in caplog.text and "margin floors none" in caplog.text
