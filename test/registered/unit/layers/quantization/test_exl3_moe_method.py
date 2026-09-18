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
    method = Exl3MoEMethod(Exl3Config.from_config(CFG))
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
    method = Exl3MoEMethod(Exl3Config.from_config(CFG))
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


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
