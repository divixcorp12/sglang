"""Explicit CPU-only expert format; generic GPTQ/model serving is deliberately not auto-selected."""

import torch

from sglang.srt.layers.moe.cpu_experts.gptq_int3 import SLAB_NAMES, GptqInt3CpuTrait


class GptqInt3ExpertFormat:
    key = "gptq_int3"
    transport_build = "gptq_int3"
    engine_served = False
    supports_cpu_experts = True
    supports_graph_gather = False
    supports_host_arena = False
    supports_draft = False
    supports_dead_rows = False
    supports_transfer_executor = False
    max_gather_rows = None
    inclusive_pinned_tier = False
    tensor_names = SLAB_NAMES

    @staticmethod
    def source(layer, name):
        tensor = getattr(layer, name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"GPTQ INT3 source {name} must be a tensor")
        return tensor.detach()

    def num_experts(self, layer):
        slabs = {name: self.source(layer, name) for name in SLAB_NAMES}
        capacity = slabs[SLAB_NAMES[0]].shape[0]
        GptqInt3CpuTrait().layer_spec(slabs, capacity)
        return capacity

    def tensor_specs(self, layer):
        # Keep format discovery CPU-only; the streamer's broader transport imports are needed only on demand.
        from sglang.srt.layers.moe.expert_format import ExpertTensorSpec

        self.num_experts(layer)
        return tuple(
            ExpertTensorSpec(
                name,
                tuple(self.source(layer, name).shape[1:]),
                self.source(layer, name).dtype,
                "host",
            )
            for name in SLAB_NAMES
        )

    def default_row_source(self, layer, specs, kind):
        if kind in ("tensor", "auto"):
            return None
        raise ValueError(
            "GPTQ INT3 CPU rows require explicit lossless adaptation into tensor sources"
        )

    def file_source_bytes_per_expert(self, layer, row_source):
        return None

    def pinned_tier_options(self, layer):
        return {}

    def prepare_host_rows(self, layer, name, tensor):
        return tensor

    def build_streamer(self, layer):
        raise NotImplementedError(
            "GPTQ INT3 GPU serving/streamer integration is not implemented"
        )

    def cpu_trait(self, *, source):
        if source != "target":
            raise ValueError("GPTQ INT3 draft CPU experts have not been validated")
        return GptqInt3CpuTrait()

    def layer_dims(self, layer):
        return self.num_experts(layer), GptqInt3CpuTrait().hidden_size(
            {name: self.source(layer, name) for name in SLAB_NAMES}
        )

    def moe_apply(self, *args, **kwargs):
        raise NotImplementedError(
            "GPTQ INT3 has a typed CPU kernel; GPU serving is not implemented"
        )
