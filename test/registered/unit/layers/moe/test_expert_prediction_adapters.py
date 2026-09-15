import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.moe.expert_prediction import adapters
from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.taps import discover_moe_layers
from sglang.srt.layers.moe.topk import StandardTopKOutput


class FakeTopK(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=top_k, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.softmax(dim=-1), self.topk_config.top_k, dim=-1)
        return StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=router_logits)


class FakeMoE(nn.Module):
    def __init__(self, layer_id, num_experts, hidden_size, num_fused_shared_experts=0):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_fused_shared_experts = num_fused_shared_experts

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(self, layer_id, hidden_size=6):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 8, bias=False)
        self.topk = FakeTopK(2)
        self.experts = FakeMoE(layer_id, 8, hidden_size)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class PlainDecoderLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.input_norm = nn.LayerNorm(6)
        self.mlp = FakeBlock(layer_id)

    def forward(self, positions, hidden_states):
        return self.mlp(self.input_norm(hidden_states))


class PlainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([PlainDecoderLayer(0), PlainDecoderLayer(1)])

    def forward(self, hidden_states):
        positions = torch.arange(hidden_states.shape[0])
        for layer in self.layers:
            layer(positions, hidden_states)


class HyperConnection(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def mix(self, hyper_input):
        return self.proj(hyper_input), hyper_input


class HyperDecoderLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.attn_hyper_connection = HyperConnection()
        self.mlp = FakeBlock(layer_id)

    def forward(self, hyper_states):
        mixed, _ = self.attn_hyper_connection.mix(hyper_states)
        return self.mlp(mixed)


class Qwen4ExpForConditionalGeneration(nn.Module):
    """Named like the real entry class so the registered adapter is selected."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([HyperDecoderLayer(0), HyperDecoderLayer(1)])

    def forward(self, hyper_states):
        for layer in self.model.layers:
            layer(hyper_states)


def _setup(model):
    layers = discover_moe_layers(model, topk_type=FakeTopK, experts_type=FakeMoE)
    store = FeatureStore(
        specs=[layer.spec for layer in layers],
        features=[RouteFeature.PRE_MIXER],
        max_rows=4,
        device=torch.device("cpu"),
        hidden_dtype=torch.float32,
    )
    return layers, store


class TestPreMixerAdapters(unittest.TestCase):
    def test_decoder_layers_map_to_outermost_module_list_elements(self):
        model = PlainModel()
        layers, _ = _setup(model)
        mapping = adapters.decoder_layers_by_moe_layer(model=model, layers=layers)
        self.assertIs(mapping[0], model.layers[0])
        self.assertIs(mapping[1], model.layers[1])

    def test_missing_decoder_layer_raises(self):
        model = FakeBlock(0)
        layers, _ = _setup(model)
        with self.assertRaisesRegex(ValueError, r"MoE layers \[0\]"):
            adapters.decoder_layers_by_moe_layer(model=model, layers=layers)

    def test_default_adapter_taps_decoder_hidden_state_input(self):
        model = PlainModel()
        layers, store = _setup(model)
        removers = adapters.install_pre_mixer_taps(model=model, layers=layers, store=store)
        hidden = torch.randn(3, 6)
        with torch.no_grad():
            model(hidden)
        for layer_id in (0, 1):
            torch.testing.assert_close(store.view(layer_id, RouteFeature.PRE_MIXER, 3), hidden)
        for remove in removers:
            remove()
        with torch.no_grad():
            model(torch.randn(3, 6))
        torch.testing.assert_close(store.view(0, RouteFeature.PRE_MIXER, 3), hidden)

    def test_default_adapter_raises_without_matching_input(self):
        model = PlainModel()
        layers, store = _setup(model)
        adapters.install_pre_mixer_taps(model=model, layers=layers, store=store)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "register a pre-mixer adapter"):
            model.layers[0](torch.arange(3), torch.randn(3))

    def test_qwen4_adapter_taps_mixed_hyper_connection_output(self):
        model = Qwen4ExpForConditionalGeneration()
        layers, store = _setup(model)
        removers = adapters.install_pre_mixer_taps(model=model, layers=layers, store=store)
        hyper = torch.randn(3, 12)
        with torch.no_grad():
            model(hyper)
            for layer_id, layer in zip((0, 1), model.model.layers):
                expected = layer.attn_hyper_connection.proj(hyper)
                torch.testing.assert_close(store.view(layer_id, RouteFeature.PRE_MIXER, 3), expected)
        for remove in removers:
            remove()
        for layer in model.model.layers:
            self.assertNotIn("mix", layer.attn_hyper_connection.__dict__)

    def test_mixer_kinds_unregistered_architecture_reports_unknown(self):
        model = PlainModel()
        layers, _ = _setup(model)
        self.assertEqual(adapters.mixer_kinds(model=model, layers=layers), {0: "unknown", 1: "unknown"})

    def test_mixer_kinds_registered_architecture_classifies_by_decoder_type(self):
        model = Qwen4ExpForConditionalGeneration()
        layers, _ = _setup(model)
        model.model.layers[0].__class__ = type("AttentionDecoderLayer", (HyperDecoderLayer,), {})
        model.model.layers[1].__class__ = type("LinearDecoderLayer", (HyperDecoderLayer,), {})
        self.assertEqual(
            adapters.mixer_kinds(model=model, layers=layers),
            {0: "full_attention", 1: "linear_attention"},
        )

    def test_registered_adapter_overrides_default(self):
        calls = []

        class CustomArch(PlainModel):
            pass

        def installer(*, model, layers, store):
            calls.append(len(layers))
            return []

        adapters.register_pre_mixer_adapter(architecture="CustomArch", installer=installer)
        model = CustomArch()
        layers, store = _setup(model)
        self.assertEqual(adapters.install_pre_mixer_taps(model=model, layers=layers, store=store), [])
        self.assertEqual(calls, [2])
        with self.assertRaisesRegex(ValueError, "CustomArch"):
            adapters.register_pre_mixer_adapter(architecture="CustomArch", installer=installer)


if __name__ == "__main__":
    unittest.main()
