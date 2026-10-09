"""The RAM prefetch's Python side (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1): the router gates registered at
load, the per-row target table and its host copy, and the service's call into the host (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.cpu_experts.threading_config import NodePlan
from sglang.kernels.ops.moe.expert_stream_transport import CAND_PAGE_BYTES, new_candidate_page
from sglang.srt.layers.moe.ram_prefetch import (
    GpuScorer,
    RouterGate,
    clear_router_gates,
    prefetch_tables,
    prefetch_targets,
    register_moe_gates,
    register_router_gate,
    registered_gates,
    spec_score_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci

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


def test_the_service_enables_the_host_with_the_registered_gates_options_and_spare_cores():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    calls = []
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
    numa = SimpleNamespace(
        nodes=1, plans=[NodePlan(group=0, node=0, ram=17, cpu=(8, 9), sq=None, busy_poll=True, spec=(0, 1))]
    )
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    with envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER.override(2):
        module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], numa, cpu)
    (targets, gates, bias), kw = calls[0]
    assert targets.tolist() == [[1, 0], [-1, -1]] and torch.equal(gates[0], gate.weight)
    assert kw == dict(top_k=6, per_token=1, per_layer=2, cores=[[0, 1]], top_k_only=False)
    with envs.SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY.override(True):
        module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], numa, cpu)
    assert calls[1][1]["top_k_only"] is True
    with pytest.raises(RuntimeError, match="needs SGLANG_DSV41_CPU_EXPERTS"):
        module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], numa, None)


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
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("gpu"):
        scorer = module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)
    (targets, gates, bias), kw = calls[0]
    assert targets.tolist() == [[1, 0], [-1, -1]] and gates is None and bias is None
    page = kw.pop("candidates")
    assert page.dtype == torch.uint8 and page.numel() == CAND_PAGE_BYTES and not page.is_pinned()
    assert kw == dict(top_k=6, per_token=1, per_layer=1, cores=[[0, 1]], top_k_only=False)
    assert scorer.candidates is page and scorer.picked.gates[0].weight is gate.weight
    assert (scorer.per_token, scorer.top_k_only, scorer.picked.top_k) == (1, False, 6)


def test_the_cpu_scorer_returns_no_gpu_scorer_and_an_unknown_one_is_refused():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: None)
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    assert module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], _numa(), cpu) is None
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("tpu"):
        with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER"):
            module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)


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
    svc = module.Exl3RamMissService()
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
    svc = module.Exl3RamMissService()
    page = new_candidate_page(pin=False)
    svc._spec_gpu = GpuScorer(prefetch_targets([0, 1], registered_gates(), hidden=8), page, 1, False)
    owned = []
    monkeypatch.setattr(module, "quarantine_host_slabs", owned.extend)
    svc._quarantine("test")
    assert any(t is page for t in owned)
