"""The DSpark draft's CPU expert runtime: masking, skips, the worker thread, stats and the registry (fake kernel)."""

import os
import threading

import pytest
import torch

from sglang.srt.layers.moe.cpu_experts.draft import (
    DraftCpuExperts,
    DraftCpuExpertsRegistry,
    DraftCpuStats,
    DraftLayer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

E, H = 6, 4  # 5 routed experts plus a fused shared one (id 5); experts 1 and 5 stay on the GPU


class _Trait:
    name = "fake-draft"
    slab_names = ("w13_trellis",)
    act_limit = None
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(self):
        self.seen = []

    def check_environment(self):
        pass

    def register_layer(self, slabs, capacity):
        return capacity

    def forward(self, handle, x, slots, weights, out, threads):
        self.seen.append((slots.clone(), threading.get_native_id()))
        valid = (slots >= 0).to(torch.float32)
        out.copy_(x.float() * (weights.float() * valid).sum(-1, keepdim=True))

    def free_layer(self, handle):
        pass


def _cores():
    cores = sorted(os.sched_getaffinity(0))[:2]
    if len(cores) < 2:
        pytest.skip("needs at least 2 cores in the affinity mask")
    return cores


def _on_cpu():
    on_cpu = torch.ones(E, dtype=torch.bool)
    on_cpu[[1, 5]] = False
    return on_cpu


def _runtime(trait):
    layer = DraftLayer({"w13_trellis": torch.zeros(E, 2, dtype=torch.int16)}, on_cpu=_on_cpu(), act_limit=10.0)
    return DraftCpuExperts(trait, {0: layer}, cores=_cores(), threads=2)


def test_resident_and_fused_shared_ids_never_reach_the_cpu():
    trait = _Trait()
    experts = _runtime(trait)
    try:
        ids = torch.tensor([[0, 1, 5], [2, 5, -1]], dtype=torch.int64)
        out = experts.submit(0, ids, torch.ones(2, H, dtype=torch.bfloat16), torch.full((2, 3), 0.5)).result()
    finally:
        experts.close()
    slots, _ = trait.seen[0]
    assert slots.tolist() == [[0, -1, -1], [2, -1, -1]]
    assert out.dtype == torch.float32 and out.tolist() == [[0.5] * H, [0.5] * H]


def test_a_call_with_only_gpu_routes_skips_the_cpu():
    trait = _Trait()
    experts = _runtime(trait)
    try:
        ids = torch.tensor([[1, 5, -1]], dtype=torch.int64)
        assert experts.submit(0, ids, torch.ones(1, H), torch.ones(1, 3)) is None
    finally:
        experts.close()
    assert trait.seen == [] and experts.stats.skips == 1 and experts.stats.calls == 1


def test_the_kernel_runs_on_the_worker_and_the_caller_keeps_its_affinity():
    trait = _Trait()
    before = os.sched_getaffinity(0)
    experts = _runtime(trait)
    try:
        experts.submit(0, torch.zeros(1, 3, dtype=torch.int64), torch.ones(1, H), torch.ones(1, 3)).result()
    finally:
        experts.close()
    assert trait.seen[0][1] != threading.get_native_id()
    assert os.sched_getaffinity(0) == before


def test_stats_count_union_passes_and_skips():
    stats = DraftCpuStats(log_every=1000)
    # 6 rows: expert 0 routed by all 6 (3 passes), expert 2 by 2 (1 pass), expert 3 by 1 (1 pass)
    slots = torch.tensor([[0, 2, -1]] * 2 + [[0, 3, -1]] + [[0, -1, -1]] * 3, dtype=torch.int64)
    stats.record(slots, 0.010)
    stats.record_skip()
    assert stats.unions == [3] and stats.passes == [5]
    assert stats.calls == 2 and stats.skips == 1
    summary = stats.summary()
    assert "2 stage calls" in summary and "1 without CPU work" in summary and "union 3.0" in summary


def test_the_registry_builds_once_from_the_envs_and_refuses_mixed_limits(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import draft

    built = []
    monkeypatch.setattr(draft, "cpu_trait_for", lambda key: built.append(key) or _Trait())
    registry = DraftCpuExpertsRegistry()
    slabs = {"w13_trellis": torch.zeros(E, 2, dtype=torch.int16)}
    assert registry.register(slabs, _on_cpu(), 10.0, layer_id=0) == 0
    assert registry.register(slabs, _on_cpu(), 10.0, layer_id=1) == 1
    with pytest.raises(ValueError, match="activation limit"):
        registry.register(slabs, _on_cpu(), 7.0, layer_id=2)
    cores = ",".join(str(c) for c in _cores())
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(cores):
        try:
            runtime = registry.runtime()
            assert registry.runtime() is runtime and built == ["exl3"]
            assert runtime.pool.trait.act_limit == 10.0
            assert sorted(runtime.pool.capacity) == [0, 1]
        finally:
            registry.close()


def test_the_registry_refuses_a_resident_file_with_a_stage_the_draft_lacks(monkeypatch, tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import draft
    from sglang.srt.layers.moe.cpu_experts.draft_resident import write_resident_set

    monkeypatch.setattr(draft, "cpu_trait_for", lambda key: _Trait())
    path = tmp_path / "resident.json"
    write_resident_set(str(path), {0: [1], 1: [5]}, n=1, source="")
    registry = DraftCpuExpertsRegistry()
    registry.register({"w13_trellis": torch.zeros(E, 2, dtype=torch.int16)}, _on_cpu(), 10.0, layer_id=0)
    cores = ",".join(str(c) for c in _cores())
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(cores), envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(str(path)):
        try:
            with pytest.raises(ValueError, match=r"stages \[1\]"):
                registry.runtime()
        finally:
            registry.close()
