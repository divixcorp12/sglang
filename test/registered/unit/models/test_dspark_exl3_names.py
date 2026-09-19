"""The DSpark draft loader's EXL3 name pipeline (fake dequant, no model, no kernels).

Feeds representative DSpark draft checkpoint names through the same two
functions ``DeepseekV4ForCausalLMDSpark.load_weights`` chains under EXL3:
``adapt_exl3_weights`` (merges wo_a's 8 slices, dequantizes the bf16-only
modules, leaves resident EXL3 linears/experts alone) and then
``_remap_dspark_weight_name`` (checkpoint name -> module parameter name).
"""

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.deepseek_v4_exl3_weights import adapt_exl3_weights
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NUM_WO_A_GROUPS = 8

# A fake, unbound `self` for `_remap_dspark_weight_name`, which only reads
# `self.confidence_head` (for the None-vs-enabled branch); no other attribute
# is touched, so a real DeepseekV4ForCausalLMDSpark instance isn't needed.
_FAKE_SELF = type("FakeSelf", (), {"confidence_head": object()})()


def _exl3(prefix, in_f=16, out_f=16, fill=1):
    """One EXL3 quad (suh/svh/mul1/trellis) for `prefix`, matching test_deepseek_v4_exl3_weights.py's helper."""
    return [
        (f"{prefix}.suh", torch.full((in_f,), 1.0, dtype=torch.float16)),
        (f"{prefix}.svh", torch.full((out_f,), 1.0, dtype=torch.float16)),
        (f"{prefix}.mul1", torch.tensor(1, dtype=torch.int32)),
        (f"{prefix}.trellis", torch.full((in_f // 16, out_f // 16, 80), fill, dtype=torch.int16)),
    ]


def fake_dequant(t):
    return torch.zeros(t.in_features, t.out_features, dtype=torch.float16)


def _adapt_and_remap(weights):
    """Run the exact pipeline `load_weights` chains under EXL3, name-only."""
    adapted = adapt_exl3_weights(iter(weights), fake_dequant, _NUM_WO_A_GROUPS)
    out = []
    for name, tensor in adapted:
        mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(_FAKE_SELF, name)
        out.append((name, mapped, tensor))
    return out


def _representative_checkpoint_weights():
    """One instance of every raw checkpoint key pattern the brief documents.

    Stage 0 exercises attention + ffn (dense-ish) fields; stage 2 exercises the
    markov head / confidence head / main-stage-only fields. `mtp.2.confidence_head`
    is skipped here because `_remap_dspark_weight_name` returns None for it only
    when `self.confidence_head is None`, which this test's fake self never is;
    a dedicated case below covers that branch.
    """
    weights = []
    weights += [
        ("mtp.0.attn.attn_sink", torch.zeros(4)),
        ("mtp.0.attn.kv_norm.weight", torch.zeros(4)),
        ("mtp.0.attn.q_norm.weight", torch.zeros(4)),
    ]
    for field in ("wkv", "wo_b", "wq_a", "wq_b"):
        weights += _exl3(f"mtp.0.attn.{field}")
    for slice_idx in range(_NUM_WO_A_GROUPS):
        weights += _exl3(f"mtp.0.attn.wo_a.slice.{slice_idx}")
    weights += [("mtp.0.attn_norm.weight", torch.zeros(4))]
    weights += _exl3("mtp.0.ffn.experts.5.w1")
    weights += _exl3("mtp.0.ffn.experts.5.w2")
    weights += _exl3("mtp.0.ffn.experts.5.w3")
    weights += [
        ("mtp.0.ffn.gate.bias", torch.zeros(4)),
        ("mtp.0.ffn.gate.weight", torch.zeros(4, 4)),
    ]
    weights += _exl3("mtp.0.ffn.shared_experts.w1")
    weights += _exl3("mtp.0.ffn.shared_experts.w2")
    weights += _exl3("mtp.0.ffn.shared_experts.w3")
    weights += [("mtp.0.ffn_norm.weight", torch.zeros(4))]
    for part in ("attn", "ffn"):
        for stat in ("base", "fn", "scale"):
            weights += [(f"mtp.0.hc_{part}_{stat}", torch.zeros(4))]
    weights += [("mtp.0.main_norm.weight", torch.zeros(4))]
    weights += _exl3("mtp.0.main_proj")
    weights += [
        ("mtp.2.markov_head.embed.weight", torch.zeros(8, 4)),
        ("mtp.2.markov_head.head.weight", torch.zeros(4, 8)),
        ("mtp.2.norm.weight", torch.zeros(4)),
    ]
    return weights


def test_wo_a_slices_merge_into_one_self_attn_weight():
    weights = []
    for slice_idx in range(_NUM_WO_A_GROUPS):
        weights += _exl3(f"mtp.0.attn.wo_a.slice.{slice_idx}")
    mapped_names = [mapped for _, mapped, _ in _adapt_and_remap(weights)]
    assert mapped_names == ["stages.0.self_attn.wo_a.weight"]


def test_routed_expert_w1_reaches_the_fused_moe_w13_param():
    mapped = dict((name, mapped) for name, mapped, _ in _adapt_and_remap(_exl3("mtp.0.ffn.experts.5.w1")))
    weight_name = "mtp.0.ffn.experts.5.w1.trellis"
    assert mapped[weight_name] == "stages.0.mlp.experts.5.gate_proj.trellis"

    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=128,
    )
    candidate = mapped[weight_name]
    for param_name, ckpt_weight_name, expert_id, shard_id in expert_params_mapping:
        if ckpt_weight_name not in candidate:
            continue
        candidate = candidate.replace(ckpt_weight_name, param_name)
        assert shard_id == "w1" and expert_id == 5
        break
    else:
        raise AssertionError(f"no expert_params_mapping entry matched {mapped[weight_name]!r}")
    assert candidate == "stages.0.mlp.experts.w13_trellis"


def test_main_proj_keeps_its_own_stage_prefix_and_exl3_suffix():
    mapped = dict((name, mapped) for name, mapped, _ in _adapt_and_remap(_exl3("mtp.0.main_proj")))
    assert mapped["mtp.0.main_proj.suh"] == "stages.0.main_proj.suh"


def test_markov_head_embed_and_head_are_renamed_and_stage_stripped():
    weights = [
        ("mtp.2.markov_head.embed.weight", torch.zeros(8, 4)),
        ("mtp.2.markov_head.head.weight", torch.zeros(4, 8)),
    ]
    mapped = dict((name, mapped) for name, mapped, _ in _adapt_and_remap(weights))
    assert mapped["mtp.2.markov_head.embed.weight"] == "markov_head.markov_w1.weight"
    assert mapped["mtp.2.markov_head.head.weight"] == "markov_head.markov_w2.weight"


def test_confidence_head_is_dropped_when_the_model_has_none():
    fake_self_no_head = type("FakeSelfNoHead", (), {"confidence_head": None})()
    mapped = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
        fake_self_no_head, "mtp.2.confidence_head.proj.weight"
    )
    assert mapped is None


def test_no_representative_name_maps_to_a_weight_scale_inv_param():
    for name, mapped, _tensor in _adapt_and_remap(_representative_checkpoint_weights()):
        assert mapped is None or "weight_scale_inv" not in mapped, (name, mapped)
