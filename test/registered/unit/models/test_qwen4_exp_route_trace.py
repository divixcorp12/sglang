"""Tests for the debug-only Qwen4-Exp MoE route trace writer and hooks."""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.models import qwen4_exp_route_trace as route_trace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC = 2
HIDDEN = 3
EXPERTS = 4
TOP_K = 2
LAYERS = 2


class _Mode:
    def __init__(self, decode):
        self.decode = decode

    def is_decode(self):
        return self.decode


def _batch(token, position, decode=True, request=0):
    return SimpleNamespace(
        forward_mode=_Mode(decode),
        input_ids=torch.tensor([token]),
        positions=torch.tensor([position]),
        seq_lens=torch.tensor([position + 1]),
        req_pool_indices=torch.tensor([request]),
    )


class _HyperConnection(nn.Module):
    config = SimpleNamespace(rms_norm_eps=1e-6)

    def mix(self, hyper_input):
        mixed = hyper_input.view(*hyper_input.shape[:-1], HC, HIDDEN).mean(-2)
        return mixed, hyper_input


class _Gate(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class _TopK(nn.Module):
    topk_config = SimpleNamespace(top_k=TOP_K)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.softmax(-1), TOP_K)
        return SimpleNamespace(
            topk_ids=ids.to(torch.int32),
            topk_weights=weights / weights.sum(-1, keepdim=True),
        )


class _Experts(nn.Module):
    def forward(self, hidden_states, topk_output):
        return hidden_states * 2


class _Moe(nn.Module):
    num_experts = EXPERTS

    def __init__(self):
        super().__init__()
        self.gate = _Gate(HIDDEN, EXPERTS, bias=False)
        self.topk = _TopK()
        self.experts = _Experts()

    def forward(self, hidden_states, forward_batch=None):
        logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states, self.topk(hidden_states, logits))
        output += 1.0
        return output


class _Layer(nn.Module):
    hc_count = HC
    hidden_size = HIDDEN

    def __init__(self):
        super().__init__()
        self.mlp_hyper_connection = _HyperConnection()
        self.mlp = _Moe()

    def forward(self, hidden_states, forward_batch):
        mixed, residual = self.mlp_hyper_connection.mix(hidden_states)
        output = self.mlp(mixed, forward_batch)
        return residual + output.repeat(1, HC)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(16, HC * HIDDEN)
        self.layers = nn.ModuleList([_Layer() for _ in range(LAYERS)])
        self.hyper_connection_mixer = _HyperConnection()

    def forward(self, input_ids=None, positions=None, forward_batch=None):
        hidden_states = self.embed(input_ids)
        self.layer_inputs = []
        for layer in self.layers:
            self.layer_inputs.append(hidden_states)
            hidden_states = layer(hidden_states, forward_batch)
        self.final_hc = hidden_states
        mixed, _ = self.hyper_connection_mixer.mix(hidden_states)
        return mixed


def _run(model, batch):
    with torch.no_grad():
        return model(input_ids=batch.input_ids, forward_batch=batch)


def _load(directory):
    with open(os.path.join(directory, route_trace.MANIFEST_NAME)) as handle:
        manifest = json.load(handle)
    shards = [
        torch.load(os.path.join(directory, shard["file"]))
        for shard in manifest["shards"]
    ]
    return manifest, shards


def test_decode_forward_records_router_tensors_before_in_place_shared_add():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        trace = route_trace.install_moe_route_trace(model, directory, max_tokens=1)
        final_hidden = _run(model, _batch(token=5, position=7))
        manifest, shards = _load(directory)

    assert manifest["complete"] and manifest["tokens"] == 1
    assert manifest["layer_ids"] == [0, 1]
    assert (manifest["hc_count"], manifest["hidden_size"]) == (HC, HIDDEN)
    assert (manifest["num_experts"], manifest["top_k"]) == (EXPERTS, TOP_K)
    shard = shards[0]
    for layer, layer_input in enumerate(model.layer_inputs):
        mixed = layer_input.view(1, HC, HIDDEN).mean(-2)
        logits = model.layers[layer].mlp.gate(mixed)[0]
        weights, ids = torch.topk(logits.softmax(-1), TOP_K)
        expected = {
            "hc_mid": layer_input,
            "router_input": mixed,
            "router_logits": logits,
            "routed_out": mixed * 2,
            "moe_out": mixed * 2 + 1,
        }
        for field, value in expected.items():
            torch.testing.assert_close(
                shard[field][:, layer].float(), value.to(torch.bfloat16).float()
            )
        assert shard["topk_ids"][0, layer].tolist() == ids[0].tolist()
        torch.testing.assert_close(
            shard["topk_weights"][:, layer], weights / weights.sum(-1, keepdim=True)
        )
    torch.testing.assert_close(
        shard["final_hc"].float(), model.final_hc.to(torch.bfloat16).float()
    )
    torch.testing.assert_close(
        shard["final_hidden"].float(), final_hidden.to(torch.bfloat16).float()
    )
    assert shard["input_ids"].tolist() == [5]
    assert shard["positions"].tolist() == [7]
    assert shard["seq_lens"].tolist() == [8]
    assert shard["forward_index"].tolist() == [0]
    assert trace.writer.closed


def test_only_decode_forwards_are_recorded():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        route_trace.install_moe_route_trace(model, directory, max_tokens=2)
        _run(model, _batch(token=1, position=0, decode=False))
        _run(model, _batch(token=2, position=1))
        _run(model, _batch(token=3, position=2, decode=False))
        _run(model, _batch(token=4, position=2))
        manifest, shards = _load(directory)

    assert manifest["tokens"] == 2 and manifest["forwards"] == 2
    assert torch.cat([shard["input_ids"] for shard in shards]).tolist() == [2, 4]


def test_trace_stops_at_max_tokens_and_restores_the_model():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        route_trace.install_moe_route_trace(
            model, directory, max_tokens=3, shard_tokens=2
        )
        for token in range(5):
            _run(model, _batch(token=token, position=token))
        manifest, shards = _load(directory)

    assert manifest["complete"] and manifest["tokens"] == 3
    assert [shard["tokens"] for shard in manifest["shards"]] == [2, 1]
    assert [shard["input_ids"].tolist() for shard in shards] == [[0, 1], [2]]
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
        assert "mix" not in module.__dict__


def test_writer_refuses_a_directory_with_an_existing_trace():
    with tempfile.TemporaryDirectory() as directory:
        route_trace.MoeRouteTraceWriter(directory, 1, [0], {})
        with pytest.raises(FileExistsError):
            route_trace.MoeRouteTraceWriter(directory, 1, [0], {})


def test_writer_refuses_a_non_empty_directory_without_a_manifest():
    with tempfile.TemporaryDirectory() as directory:
        open(os.path.join(directory, "shard_00000.pt"), "wb").close()
        with pytest.raises(FileExistsError, match="not empty"):
            route_trace.MoeRouteTraceWriter(directory, 1, [0], {})


def test_failed_shard_write_stops_the_trace_and_serving_continues():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        trace = route_trace.install_moe_route_trace(
            model, directory, max_tokens=4, shard_tokens=1
        )
        real_save = torch.save

        def fail_after_temp_file(obj, path):
            real_save(obj, path)
            raise OSError(28, "No space left on device")

        with patch.object(route_trace.torch, "save", side_effect=fail_after_temp_file):
            _run(model, _batch(token=1, position=0))
        output = _run(model, _batch(token=2, position=1))
        manifest, _ = _load(directory)
        leftovers = sorted(os.listdir(directory))
        trace.writer.close()

    assert output.shape == (1, HIDDEN)
    assert trace.writer.closed and trace.writer.tokens == 1
    assert not manifest["complete"] and "No space left" in manifest["error"]
    assert manifest["shards"] == []
    assert leftovers == [route_trace.MANIFEST_NAME]
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
        assert "mix" not in module.__dict__


def test_forward_missing_a_layer_field_is_dropped():
    with tempfile.TemporaryDirectory() as directory:
        writer = route_trace.MoeRouteTraceWriter(directory, 1, [0], {})
        writer.begin_forward(_batch(token=1, position=0))
        writer.record_layer("router_input", 0, torch.zeros(1, HIDDEN))
        writer.end_forward()
        assert writer.tokens == 0 and not writer.closed


def test_unset_environment_installs_nothing():
    model = _LanguageModel()
    with patch.dict(os.environ, {"SGLANG_MOE_ROUTE_TRACE_DIR": ""}):
        assert route_trace.maybe_install_moe_route_trace(model) is None
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


def test_bytes_per_token_estimate_for_qwen38_flash():
    estimate = route_trace.estimate_bytes_per_token(48, 2560, 4, 512, 10)
    assert estimate == 48 * (5120 + 20480 + 1024 + 20 + 40 + 5120 + 5120) + (
        20480 + 5120 + 40
    )
