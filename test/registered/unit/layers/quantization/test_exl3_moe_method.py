"""exl3 MoE parameter registration and per-expert loading (no kernels)."""

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CFG = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}
E, HIDDEN, INTER = 4, 256, 128


def _moe():
    layer = nn.Module()
    layer.num_experts = E
    method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=False)
    method.create_weights(layer, E, HIDDEN, INTER, torch.bfloat16)
    return layer, method


def _expert(in_f, out_f, fill):
    return {
        "trellis": torch.full((in_f // 16, out_f // 16, 48), fill, dtype=torch.int16),
        "suh": torch.full((in_f,), float(fill), dtype=torch.float16),
        "svh": torch.full((out_f,), float(fill), dtype=torch.float16),
        "mul1": torch.tensor(1, dtype=torch.int32),
    }


def _load_all(layer):
    for e in range(E):
        for shard, (in_f, out_f), prefix in (
            ("w1", (HIDDEN, INTER), "w13"),
            ("w3", (HIDDEN, INTER), "w13"),
            ("w2", (INTER, HIDDEN), "w2"),
        ):
            fill = 10 * e + {"w1": 1, "w3": 3, "w2": 2}[shard]
            for name, tensor in _expert(in_f, out_f, fill).items():
                param = getattr(layer, f"{prefix}_{name}")
                param.weight_loader(param, tensor, f"experts.{prefix}_{name}", shard_id=shard, expert_id=e)


def test_loads_every_expert_into_its_slot():
    layer, method = _moe()
    _load_all(layer)
    method.process_weights_after_loading(layer)
    gate, up = layer.exl3_w13[2]
    assert int(gate.trellis[0, 0, 0]) == 21 and int(up.trellis[0, 0, 0]) == 23
    assert int(layer.exl3_w2[3].trellis[0, 0, 0]) == 32
    assert (gate.in_features, gate.out_features, gate.bits) == (HIDDEN, INTER, 3)


def test_concurrent_loads_keep_every_expert():
    # deepseek_v4.load_weights runs weight loaders on a thread pool; every thread that finds the
    # param empty must see one shared buffer, or experts loaded into a replaced buffer are lost.
    import threading
    from concurrent.futures import ThreadPoolExecutor

    for _ in range(20):
        layer, method = _moe()
        jobs = []
        for e in range(E):
            for shard, (in_f, out_f), prefix in (
                ("w1", (HIDDEN, INTER), "w13"),
                ("w3", (HIDDEN, INTER), "w13"),
                ("w2", (INTER, HIDDEN), "w2"),
            ):
                fill = 10 * e + {"w1": 1, "w3": 3, "w2": 2}[shard]
                for name, tensor in _expert(in_f, out_f, fill).items():
                    jobs.append((getattr(layer, f"{prefix}_{name}"), tensor, prefix, name, shard, e))
        barrier = threading.Barrier(len(jobs))

        def load(job):
            param, tensor, prefix, name, shard, e = job
            barrier.wait()
            param.weight_loader(param, tensor, f"experts.{prefix}_{name}", shard_id=shard, expert_id=e)

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            list(pool.map(load, jobs))
        for e in range(E):
            assert int(layer.w13_trellis[e, 0, 0, 0, 0]) == 10 * e + 1
            assert int(layer.w13_trellis[e, 1, 0, 0, 0]) == 10 * e + 3
            assert int(layer.w2_trellis[e, 0, 0, 0, 0]) == 10 * e + 2
        method.process_weights_after_loading(layer)


def test_missing_expert_detected():
    layer, method = _moe()
    _load_all(layer)
    layer.exl3_loaded.discard(("w2", "trellis", 1, 0))
    with pytest.raises(RuntimeError, match="expert 1"):
        method.process_weights_after_loading(layer)


def test_sharded_create_weights_rejected():
    layer = nn.Module()
    layer.num_experts = E
    method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=False)
    with pytest.raises(NotImplementedError, match="tensor-parallel size 1"):
        method.create_weights(
            layer, E, HIDDEN, INTER, torch.bfloat16, moe_intermediate_size=INTER * 2
        )


def test_wrong_shape_expert_detected():
    layer, method = _moe()
    # w1 and w3 share one physical w13_* param per slot, so both must be given
    # the same (wrong) in_features to stay self-consistent: `_materialize`'s
    # cross-expert/cross-slot consistency check only compares shapes to each
    # other, not against hidden/inter, so it lets this pass. Only the
    # hidden/inter shape check added to `process_weights_after_loading` should
    # catch it.
    for e in range(E):
        for shard, (in_f, out_f), prefix in (
            ("w1", (HIDDEN + 16, INTER), "w13"),
            ("w3", (HIDDEN + 16, INTER), "w13"),
            ("w2", (INTER, HIDDEN), "w2"),
        ):
            fill = 10 * e + {"w1": 1, "w3": 3, "w2": 2}[shard]
            for name, tensor in _expert(in_f, out_f, fill).items():
                param = getattr(layer, f"{prefix}_{name}")
                param.weight_loader(param, tensor, f"experts.{prefix}_{name}", shard_id=shard, expert_id=e)
    with pytest.raises(RuntimeError, match=r"expert 0 w13\[0\]"):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize("fused", [False, True])
def test_apply_scales_routed_output_unless_fused(monkeypatch, fused):
    # DeepseekV2MoE leaves routed_scaling_factor to the runner on CUDA; the EXL3 method
    # must apply it exactly once (not again when it is fused into topk_weights).
    from types import SimpleNamespace

    from sglang.srt.layers.quantization import exl3 as exl3_mod

    layer, method = _moe()
    layer.should_fuse_routed_scaling_factor_in_topk = fused
    layer.exl3_w13, layer.exl3_w2 = [], []
    method.moe_runner_config = SimpleNamespace(
        routed_scaling_factor=1.5, swiglu_limit=10.0, apply_router_weight_on_input=False
    )
    base = torch.ones(3, HIDDEN)
    monkeypatch.setattr(exl3_mod, "exl3_moe_loop", lambda *a, **k: base.clone())
    topk = SimpleNamespace(topk_weights=torch.ones(3, 2), topk_ids=torch.zeros(3, 2, dtype=torch.int32))
    dispatch = SimpleNamespace(hidden_states=torch.zeros(3, HIDDEN), topk_output=topk)
    out = method.apply(layer, dispatch).hidden_states
    assert torch.equal(out, base if fused else base * 1.5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


from types import SimpleNamespace


def _cpu_draft_moe(monkeypatch, tmp_path, resident, fused_shared=0):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import draft, draft_resident

    registered = []
    registry = SimpleNamespace(register=lambda slabs, on_cpu, limit: registered.append((slabs, on_cpu, limit)) or 0)
    monkeypatch.setattr(draft, "DRAFT_CPU_EXPERTS", registry)
    path = tmp_path / "resident.json"
    draft_resident.write_resident_set(str(path), {0: resident}, n=len(resident), source="")
    draft_resident._cache.clear()
    layer = nn.Module()
    layer.layer_id = 0
    layer.num_experts = E
    layer.num_fused_shared_experts = fused_shared
    layer.moe_runner_config = SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=False, cpu_draft=True)
    method.create_weights(layer, E, HIDDEN, INTER, torch.bfloat16)
    _load_all(layer)
    with envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(str(path)):
        method.process_weights_after_loading(layer)
    return layer, method, registered


def test_a_cpu_draft_layer_loads_into_host_memory_and_keeps_its_resident_set_for_the_gpu(monkeypatch, tmp_path):
    layer, method, registered = _cpu_draft_moe(monkeypatch, tmp_path, resident=[2])
    assert layer.w13_trellis.device.type == "cpu" and layer.w2_svh.device.type == "cpu"
    (slabs, on_cpu, limit), = registered
    assert on_cpu.tolist() == [True, True, False, True] and limit == 10.0
    assert slabs["w13_trellis"].data_ptr() == layer.w13_trellis.data_ptr()
    assert layer.exl3_cpu_draft_key == 0 and layer.exl3_gpu_experts == frozenset({2})
    gate, up = layer.exl3_gpu_w13[2]
    assert int(gate.trellis[0, 0, 0]) == 10 * 2 + 1 and sorted(layer.exl3_gpu_w2) == [2]


def test_a_fused_shared_expert_is_always_on_the_gpu(monkeypatch, tmp_path):
    layer, method, registered = _cpu_draft_moe(monkeypatch, tmp_path, resident=[0], fused_shared=1)
    assert registered[0][1].tolist() == [False, True, True, False]
    assert layer.exl3_gpu_experts == frozenset({0, E - 1})


def test_a_resident_id_outside_the_routed_experts_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="resident"):
        _cpu_draft_moe(monkeypatch, tmp_path, resident=[E - 1], fused_shared=1)


def test_apply_adds_the_gpu_and_cpu_shares(monkeypatch, tmp_path):
    from sglang.srt.layers.moe.cpu_experts import draft
    from sglang.srt.layers.quantization import exl3

    layer, method, _ = _cpu_draft_moe(monkeypatch, tmp_path, resident=[2])
    seen = {}

    class _Future:
        def result(self):
            return torch.full((2, HIDDEN), 1.0)

    def submit(key, ids, x, weights):
        seen["ids"] = ids.tolist()
        return _Future()

    monkeypatch.setattr(draft, "DRAFT_CPU_EXPERTS", SimpleNamespace(runtime=lambda: SimpleNamespace(submit=submit)))

    def accumulate(out, x, w, ids, w13, w2, limit, experts):
        seen["gpu"] = list(experts)
        out += 2.0

    monkeypatch.setattr(exl3, "exl3_moe_accumulate", accumulate)
    x = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    ids = torch.tensor([[2, 0], [1, 3]])
    out = method._apply_cpu_draft(layer, x, torch.ones(2, 2), ids, 10.0)
    assert seen == {"ids": [[2, 0], [1, 3]], "gpu": [2]}
    assert out.dtype == torch.bfloat16 and torch.all(out == 3.0)


def test_without_the_flag_a_draft_layer_stays_on_the_default_device():
    method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=False)
    assert method.cpu_draft is False


def test_cpu_draft_parameters_are_not_staged_to_the_gpu_for_post_load(monkeypatch, tmp_path):
    # The loader stages CPU parameters onto the GPU around process_weights_after_loading; views taken then would
    # keep GPU copies of every draft expert alive.
    layer, _, _ = _cpu_draft_moe(monkeypatch, tmp_path, resident=[2])
    for name, param in layer.named_parameters():
        assert getattr(param, "_sglang_skip_device_loading", False), name
