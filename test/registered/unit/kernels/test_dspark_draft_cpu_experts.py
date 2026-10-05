"""The DSpark draft's CPU expert registry: masking, the runtime's build from the envs, and prepare (fake kernel and
host; the draft channel itself is tested in test_dspark_draft_cpu_thread.py and test_dspark_draft_channel_cuda.py)."""

import os

import pytest
import torch

from sglang.srt.layers.moe.cpu_experts import draft
from sglang.srt.layers.moe.cpu_experts.draft import DraftCpuExpertsRegistry, cpu_slots
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

E, H = 6, 8  # 5 routed experts plus a fused shared one (id 5); experts 1 and 5 stay on the GPU


class _Kernel:
    """A DraftKernel: an address and a layer spec per stage."""

    def __init__(self, act_limit=None):
        self.act_limit = act_limit

    def address(self):
        return 0

    def spec(self, slabs, capacity):
        return ("spec", capacity)


class _Host:
    """A DraftCpuHost: records its layers, starts and stops."""

    built: list = []

    def __init__(self, areas, **kw):
        self.areas, self.kw, self.layers, self.started, self.stopped = areas, kw, {}, 0, 0
        _Host.built.append(self)

    def set_layer(self, stage, kernel, spec):
        self.layers[stage] = spec

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def stats(self):
        return {"jobs": 0, "rows": 0, "forward_ns": 0, "keep_warm_calls": 0}


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    _Host.built = []
    monkeypatch.setattr(draft, "draft_kernel_for", lambda key, act_limit: _Kernel(act_limit))
    monkeypatch.setattr(draft, "_new_host", lambda areas, **kw: _Host(areas, **kw))
    monkeypatch.setattr(draft, "_new_device", lambda areas, on_cpu, device: ("device", on_cpu.clone()))


def _cores():
    cores = sorted(os.sched_getaffinity(0))[:2]
    if len(cores) < 2:
        pytest.skip("needs at least 2 cores in the affinity mask")
    return cores


def _on_cpu():
    on_cpu = torch.ones(E, dtype=torch.bool)
    on_cpu[[1, 5]] = False
    return on_cpu


def _slabs():
    return {"w13_trellis": torch.zeros(E, 2, dtype=torch.int16), "w13_suh": torch.zeros(E, 2, H, dtype=torch.float16)}


def test_resident_fused_shared_and_out_of_range_ids_never_reach_the_cpu():
    ids = torch.tensor([[0, 1, 5], [2, 9, -1]], dtype=torch.int64)
    assert cpu_slots(_on_cpu(), ids).tolist() == [[0, -1, -1], [2, -1, -1]]


def test_prepare_builds_the_runtime_once(monkeypatch):
    from sglang.srt.environ import envs

    registry = DraftCpuExpertsRegistry()
    registry.register(_slabs(), _on_cpu(), 10.0, layer_id=0)
    registry.register(_slabs(), _on_cpu(), 10.0, layer_id=1)
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(",".join(map(str, _cores()))):
        try:
            registry.prepare()
            registry.prepare()
            runtime = registry.runtime()
            (host,) = _Host.built
            assert host.started == 1 and sorted(host.layers) == [0, 1] and host.layers[0] == ("spec", E)
            assert (host.areas.stages, host.areas.hidden) == (2, H)
            assert host.kw["spin_us"] == envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_IDLE_SPIN_US.get()
            assert runtime.stats() == host.stats()
        finally:
            registry.close()
    assert host.stopped == 1


def test_the_registry_builds_once_from_the_envs_and_refuses_mixed_limits(monkeypatch):
    from sglang.srt.environ import envs

    built = []
    monkeypatch.setattr(draft, "draft_kernel_for", lambda key, act_limit: built.append((key, act_limit)) or _Kernel())
    registry = DraftCpuExpertsRegistry()
    assert registry.register(_slabs(), _on_cpu(), 10.0, layer_id=0) == 0
    assert registry.register(_slabs(), _on_cpu(), 10.0, layer_id=1) == 1
    with pytest.raises(ValueError, match="activation limit"):
        registry.register(_slabs(), _on_cpu(), 7.0, layer_id=2)
    cores = ",".join(str(c) for c in _cores())
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(cores):
        try:
            runtime = registry.runtime()
            assert registry.runtime() is runtime and built == [("exl3", 10.0)]
            assert sorted(runtime.capacity) == [0, 1]
        finally:
            registry.close()


def test_without_named_cores_the_registry_runs_on_the_threading_plans_draft_cores(monkeypatch):
    """Unset SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES: the draft runs on the cores ThreadingConfig derived for it, the plan
    the RAM-miss service resolves too, so neither lands on the other. Mutant: parse the empty env list -- red (refused
    for fewer than 2 cores)."""
    from types import SimpleNamespace

    from sglang.srt.layers.moe.cpu_experts.threading_config import ThreadingConfig

    cores = tuple(_cores())
    asked = []
    monkeypatch.setattr(
        ThreadingConfig,
        "from_env",
        classmethod(lambda cls, **kw: asked.append(kw) or SimpleNamespace(draft_cpus=cores)),
    )
    registry = DraftCpuExpertsRegistry()
    registry.register(_slabs(), _on_cpu(), 10.0, layer_id=0)
    try:
        runtime = registry.runtime()
        assert (tuple(runtime.cores), runtime.threads) == (cores, len(cores))
        assert len(asked) == 1
    finally:
        registry.close()


def test_the_registry_refuses_more_threads_than_cores(monkeypatch):
    from sglang.srt.environ import envs

    registry = DraftCpuExpertsRegistry()
    registry.register(_slabs(), _on_cpu(), 10.0, layer_id=0)
    cores = ",".join(str(c) for c in _cores())
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(cores), envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS.override(3):
        try:
            with pytest.raises(ValueError, match="threads"):
                registry.runtime()
        finally:
            registry.close()


def test_the_registry_refuses_a_resident_file_with_a_stage_the_draft_lacks(monkeypatch, tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts.draft_resident import write_resident_set

    path = tmp_path / "resident.json"
    write_resident_set(str(path), {0: [1], 1: [5]}, n=1, source="")
    registry = DraftCpuExpertsRegistry()
    registry.register(_slabs(), _on_cpu(), 10.0, layer_id=0)
    cores = ",".join(str(c) for c in _cores())
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(cores), envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(str(path)):
        try:
            with pytest.raises(ValueError, match=r"stages \[1\]"):
                registry.runtime()
        finally:
            registry.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
