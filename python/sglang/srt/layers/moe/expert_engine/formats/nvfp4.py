"""The NVFP4 routed-expert format plugin of the expert engine (DSV4.1 NVFP4; MiMo V2.6 Pro once it is converted).

One expert's row is six tensors, in the C++ layout's copy-table order (``csrc/moe/formats/nvfp4/row_layout.h``, which is
``NVFP4_STREAM_TENSORS``), as ``ModelOptNvFp4FusedMoEMethod`` holds them after ``process_weights_after_loading`` on the
FlashInfer CUTLASS backend with the block scales swizzled on the host (``swizzle_blockscale(target_device="cpu")``):

* ``w13_weight`` ``[2*I, H/2]`` uint8: packed E2M1, ``[up; gate]``.
* ``w2_weight`` ``[H, I/2]`` uint8.
* ``w13_blockscale_swizzled`` ``[pad128(2*I), pad4(H/16)]`` float8_e4m3fn: the GPU's 128x4 swizzle.
* ``w2_blockscale_swizzled`` ``[pad128(H), pad4(I/16)]`` float8_e4m3fn.
* ``g1_alphas`` and ``g2_alphas``: the per-expert fp32 GEMM alphas (input scale x weight global scale). Streamed as
  rows so a hot-cache slot id selects them with the weights. Each is padded to ``SIDE_ROW_ELEMENTS`` fp32 (512 B) with
  the value repeated, because the reader requires every row tensor to be a multiple of 512 B; the kernel (and the
  NVFP4 CPU kernel, which reads the first four bytes of a slot's alpha row) reads element 0.

The GPU apply is slot-indexed: the hot-cache slot buffers stand in for the ``[E, ...]`` weight tensors, ``E`` is the
slot count and the expert ids are slot ids. It is the FlashInfer CUTLASS call ``ModelOptNvFp4FusedMoEMethod.apply``
makes (``_run_flashinfer_cutlass``), so the result equals the dense method's on the same experts. The two scalar
activation scales (``w13_input_scale_quant``, ``w2_input_scale_quant``) are per layer, not per expert, and stay layer
attributes.

Row source: the engine's native reader reads one *row image* file per layer (an expert's six tensors back to back at
512 B offsets, ``e * row_stride`` apart) and requires every part of a row to share one file size and one aligned base.
The NVFP4 expert-file cache (``expert_file_reader.py``, ``file_tensor_cache.py``) holds four per-tensor files per layer
and no alphas, so it cannot serve the native reader as it is: see :meth:`Nvfp4ExpertFormat.default_row_source`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.expert_format import ExpertTensorSpec
from sglang.srt.layers.moe.expert_transfer import NVFP4_TRANSFER_TENSOR_COUNT

if TYPE_CHECKING:
    from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec
    from sglang.srt.layers.moe.expert_row_source import ExpertRowSource

logger = logging.getLogger(__name__)

NVFP4_STREAMED_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_blockscale_swizzled",
    "w2_blockscale_swizzled",
    "g1_alphas",
    "g2_alphas",
)
SCALE_NAMES = ("w13_blockscale_swizzled", "w2_blockscale_swizzled")
SIDE_NAMES = ("g1_alphas", "g2_alphas")
# The names of the NVFP4 CPU kernel's slabs, in the order of NVFP4_STREAMED_NAMES (the kernel's up-alpha slab is not
# streamed: gate and up share one alpha).
CPU_SLAB_NAMES = ("w13", "w2", "sf13", "sf2", "gate_alpha", "down_alpha")
# fp32 elements in an alpha's row: 512 B, the reader's row alignment (row_reader.h kImageAlign).
SIDE_ROW_ELEMENTS = 128
ROW_ALIGNMENT = 512
PAGE_BYTES = 4096
# NVFP4: one E4M3 scale per this many fp4 weights.
GROUP_SIZE = 16
# The block-scale swizzle's tile: rows padded to 128, scale columns to 4.
SCALE_ROW_TILE = 128
SCALE_COL_TILE = 4

# The model profiles whose NVFP4 experts the native engine serves (see ``engine_profile_name_for``).
ENGINE_PROFILES = ("dsv41", "mimo_v26")


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def swizzled_scale_shape(rows: int, hidden_or_inter: int) -> tuple[int, int]:
    """The ``swizzle_blockscale`` output of a ``[rows, K/16]`` scale matrix of an operand with ``K`` columns."""
    return (
        _round_up(rows, SCALE_ROW_TILE),
        _round_up(hidden_or_inter // GROUP_SIZE, SCALE_COL_TILE),
    )


def engine_serves_nvfp4(model_architectures: Optional[Sequence[str]], prefix: str) -> Optional[str]:
    """The profile name that streams the NVFP4 MoE layer at ``prefix`` for a model with these architectures, or None.

    None means the legacy ``ExpertStreamer`` path (or no streaming) handles the layer: the architectures match no
    profile (Qwen), or the layer is not one the profile streams (a draft, a dense layer).
    """
    from sglang.srt.layers.moe.expert_engine.profiles import profile_for_architectures

    try:
        profile = profile_for_architectures(model_architectures)
    except ValueError:
        return None
    if profile.name not in ENGINE_PROFILES or not profile.is_streamed_module(prefix):
        return None
    return profile.name


def engine_profile_name_for(prefix: str) -> Optional[str]:
    """The profile name whose engine streams the NVFP4 MoE layer at ``prefix`` in this process, or None.

    None when the engine's streaming switch is off (``expert_stream_enabled``, the one the models' skip hooks read, so
    the loader and this method agree), or the process's model resolves to no engine profile (every model that is not
    DSV4.1 or MiMo, Qwen included, keeps ``ModelOptNvFp4FusedMoEMethod._attach_expert_streamer`` under the legacy
    ``SGLANG_EXPERT_STREAM_ENABLE``). The model config is read only when streaming is on, so a process that never streams
    never needs one.
    """
    from sglang.srt.layers.moe.expert_engine.formats.mxfp4 import expert_stream_enabled

    if not expert_stream_enabled():
        return None
    from sglang.srt.runtime_context import process_model_config

    return engine_serves_nvfp4(process_model_config().hf_config.architectures, prefix)


def slot_moe_apply(
    *,
    runner_config: Any,
    x: torch.Tensor,
    slot_ids: torch.Tensor,
    weights: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    w13_input_scale_quant: torch.Tensor,
    w2_input_scale_quant: torch.Tensor,
    swiglu_params: tuple = (None, None, None),
    keep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FlashInfer CUTLASS NVFP4 MoE over slot buffers: ``E`` = the buffers' slot count, expert ids = slot ids.

    ``tensors`` holds the six row tensors as ``[slots, *row_shape]`` (the alphas as padded ``[slots, 128]`` rows).
    ``slot_ids`` and ``weights`` are ``[M, top_k]`` (or flat ``M * top_k``, reshaped by ``x``'s rows); ``keep`` is the
    gather's fp32 ``[1]`` drop flag, which scales the route weights (0 empties the routed output). ``swiglu_params`` is
    the ``(alpha, beta, limit)`` triple of ``materialize_swiglu_params_for_cutlass`` (per-expert vectors, cut to the
    slot count here; all None without a clamp). Returns the bf16 ``[M, H]`` output.
    """
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        FlashInferCutlassMoeQuantInfo,
        fused_experts_none_to_flashinfer_cutlass,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    rows = x.shape[0]
    slots = tensors["w13_weight"].shape[0]
    ids = slot_ids.reshape(rows, -1).to(torch.int32)
    route_weights = weights.reshape(rows, -1).to(torch.float32)
    if keep is not None:
        route_weights = route_weights * keep.to(torch.float32)
    alpha, beta, limit = (None if p is None else p[:slots] for p in swiglu_params)
    quant_info = FlashInferCutlassMoeQuantInfo(
        quant_type="fp4",
        w13_weight=tensors["w13_weight"],
        w2_weight=tensors["w2_weight"],
        output_dtype=torch.bfloat16,
        quant_scales=[
            w13_input_scale_quant,
            tensors["w13_blockscale_swizzled"],
            # Element 0 of each slot's padded alpha row: [slots, 128] -> contiguous [slots].
            tensors["g1_alphas"][:, 0].contiguous(),
            w2_input_scale_quant,
            tensors["w2_blockscale_swizzled"],
            tensors["g2_alphas"][:, 0].contiguous(),
        ],
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
        apply_routed_scaling_factor=False,
    )
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(route_weights, ids, None),
    )
    return fused_experts_none_to_flashinfer_cutlass(dispatch, quant_info, runner_config).hidden_states


def _slot_tensors(slot_tensors: Any) -> Mapping[str, torch.Tensor]:
    """The slot buffers by name: a mapping as given, or the hot cache of ``SlotApplyArgs.streamer``."""
    if isinstance(slot_tensors, Mapping):
        return slot_tensors
    return slot_tensors.streamer.hot_cache.tensors


class Nvfp4EngineCpuTrait:
    """The NVFP4 CPU kernel's trait over the engine's tensor names (``CpuExpertQuantTrait``).

    The kernel's own trait (``Nvfp4CpuQuantTrait``) names its slabs ``w13, w2, sf13, sf2, gate_alpha, down_alpha`` and
    takes the layer's shape as constructor arguments; the engine's pinned tier holds them under ``NVFP4_STREAMED_NAMES``
    and the service builds one trait for every layer. This one renames the slabs and reads the shape off them
    (``w13_weight`` is ``[capacity, 2*I, H/2]``).

    A layer's alphas fold in its activation scales, which the CPU kernel's full-precision input must cancel
    (``inv_input_scale13`` / ``inv_input_scale2``, the reciprocals: ``w13_input_scale_quant`` /
    ``w2_input_scale_quant``). They differ between layers, so each layer's pair is registered with
    :meth:`set_layer_scales` (keyed by its ``w13_weight`` slab) before the service registers the layer; a layer
    without one is refused rather than run with neutral scales, which would be silently wrong.
    """

    name = "nvfp4"
    slab_names = NVFP4_STREAMED_NAMES
    x_dtype = torch.float16
    weights_dtype = torch.float32
    out_dtype = torch.float32

    def __init__(self, *, w13_layout: int = 0) -> None:
        self.w13_layout = w13_layout
        self._act_limit: Optional[float] = None
        self._scales: dict[int, tuple[float, float]] = {}
        self._dims: Optional[tuple[int, int]] = None

    @property
    def act_limit(self) -> Optional[float]:
        return self._act_limit

    @act_limit.setter
    def act_limit(self, value: Optional[float]) -> None:
        self._act_limit = value

    def check_environment(self) -> None:
        """Nothing to check: the kernel pins its workers only to the cores a call names."""

    @staticmethod
    def dims_of(slabs: Mapping[str, torch.Tensor]) -> tuple[int, int]:
        """``(hidden, intermediate)`` from the ``w13_weight`` slab's row shape ``[2*I, H/2]``."""
        w13 = slabs["w13_weight"]
        if w13.ndim != 3:
            raise ValueError(f"NVFP4 w13_weight slab must be [rows, 2*I, H/2], got {tuple(w13.shape)}")
        return int(w13.shape[2]) * 2, int(w13.shape[1]) // 2

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        return self.dims_of(slabs)[0]

    def set_layer_scales(self, w13_slab: torch.Tensor, inv_input_scale13: float, inv_input_scale2: float) -> None:
        """Register the activation-scale reciprocals of the layer whose ``w13_weight`` slab is ``w13_slab``."""
        self._scales[w13_slab.data_ptr()] = (float(inv_input_scale13), float(inv_input_scale2))

    def _inner(self, slabs: Mapping[str, torch.Tensor]):
        from sglang.srt.layers.quantization.nvfp4.schemes import Nvfp4CpuQuantTrait

        scales = self._scales.get(slabs["w13_weight"].data_ptr())
        if scales is None:
            raise ValueError(
                "the NVFP4 CPU kernel needs each layer's activation-scale reciprocals before the layer registers "
                "(Nvfp4EngineCpuTrait.set_layer_scales); neutral scales would silently mis-scale the experts"
            )
        hidden, intermediate = self.dims_of(slabs)
        if self._dims is not None and self._dims != (hidden, intermediate):
            raise ValueError(f"NVFP4 CPU experts: layers of shapes {self._dims} and {(hidden, intermediate)}")
        self._dims = (hidden, intermediate)
        return Nvfp4CpuQuantTrait(
            hidden,
            intermediate,
            act_limit=self._act_limit,
            w13_layout=self.w13_layout,
            inv_input_scale13=scales[0],
            inv_input_scale2=scales[1],
        )

    def attach_host(self, host) -> None:
        """Nothing to take: the compiled-in kernel's params need no host query."""

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> "CpuExpertLayerSpec":
        """The layer's slabs under the kernel's names, with this layer's activation scales."""
        renamed = dict(zip(CPU_SLAB_NAMES, (slabs[name] for name in NVFP4_STREAMED_NAMES)))
        return self._inner(slabs).layer_spec(renamed, capacity)


class Nvfp4ExpertFormat:
    """``ExpertFormat`` for one layer of an NVFP4 checkpoint's routed experts.

    Spec-only: no dense ``[experts, ...]`` source exists (the streamed experts are never materialized on the host or
    the device), so every host row comes from the row source. Built from the dims the quant method records on the
    layer (``layer_dims``).
    """

    key = "nvfp4"
    transport_build = "nvfp4"
    engine_served = True
    supports_graph_gather = False
    supports_host_arena = False
    supports_transfer_executor = len(NVFP4_STREAMED_NAMES) == NVFP4_TRANSFER_TENSOR_COUNT
    # The NVFP4 CPU kernel library (csrc/nvfp4/optimized) is the format's CpuKernel in the C++ build.
    supports_cpu_experts = True
    # The speculative draft (DSpark) runs EXL3 CPU experts only; an NVFP4 draft is not written, so the stub refuses it
    # at startup (check_requested_capabilities).
    supports_draft = False
    supports_dead_rows = False
    # NVFP4's FlashInfer call takes a clamp from the runner config, not from the gather: a layer without a SwiGLU limit
    # (MiMo) is valid.
    requires_swiglu_limit = False
    graph_source_kind = "pinned_tier"
    max_gather_rows: Optional[int] = None
    inclusive_pinned_tier = False
    names = NVFP4_STREAMED_NAMES

    def __init__(
        self,
        *,
        layer_id: int,
        num_experts: int,
        hidden: int,
        intermediate: int,
        gated: bool = True,
        runner_config: Any = None,
        source_root: Optional[str] = None,
    ) -> None:
        if hidden % 16 or intermediate % 16 or hidden <= 0 or intermediate <= 0:
            raise ValueError(
                f"nvfp4 needs hidden and intermediate sizes that are positive multiples of 16 "
                f"(got hidden={hidden}, intermediate={intermediate})"
            )
        if num_experts <= 0:
            raise ValueError(f"nvfp4 layer {layer_id} has {num_experts} experts")
        self.layer_id = layer_id
        self.experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.gated = gated
        self.runner_config = runner_config
        self.source_root = source_root
        shards = 2 if gated else 1
        self.w13_rows = shards * intermediate
        self._swiglu_cache: dict = {}
        self._specs = (
            ExpertTensorSpec("w13_weight", (self.w13_rows, hidden // 2), torch.uint8, "host"),
            ExpertTensorSpec("w2_weight", (hidden, intermediate // 2), torch.uint8, "host"),
            ExpertTensorSpec(
                "w13_blockscale_swizzled", swizzled_scale_shape(self.w13_rows, hidden), torch.float8_e4m3fn, "host"
            ),
            ExpertTensorSpec(
                "w2_blockscale_swizzled", swizzled_scale_shape(hidden, intermediate), torch.float8_e4m3fn, "host"
            ),
            ExpertTensorSpec("g1_alphas", (SIDE_ROW_ELEMENTS,), torch.float32, "host"),
            ExpertTensorSpec("g2_alphas", (SIDE_ROW_ELEMENTS,), torch.float32, "host"),
        )
        assert tuple(spec.name for spec in self._specs) == NVFP4_STREAMED_NAMES
        self.check_row_alignment()

    # ---- row geometry ----

    def row_alignment_violations(self) -> list[tuple[str, int]]:
        """``(name, row bytes)`` of every row tensor that is not a multiple of 512 B."""
        return [(s.name, s.row_bytes) for s in self._specs if s.row_bytes % ROW_ALIGNMENT]

    def check_row_alignment(self) -> None:
        bad = self.row_alignment_violations()
        if bad:
            raise ValueError(f"nvfp4 layer {self.layer_id}: row tensors not {ROW_ALIGNMENT} B aligned: {bad}")

    def image_geometry(self) -> tuple[tuple[int, ...], int, int]:
        """``(name_offsets, image_bytes, row_stride)`` of one expert's row image: the six tensors back to back (each
        a multiple of 512 B, so each starts 512-aligned), ``row_stride`` the image rounded up to a page. This is the
        geometry the native reader's tables need of a row-image file; no such file is written for NVFP4 yet."""
        offsets, at = [], 0
        for spec in self._specs:
            offsets.append(at)
            at += spec.row_bytes
        return tuple(offsets), at, _round_up(at, PAGE_BYTES)

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]:
        return self._specs

    def num_experts(self, layer: torch.nn.Module) -> int:
        return self.experts

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        return None

    @staticmethod
    def layer_dims(layer: torch.nn.Module) -> tuple[int, int]:
        return layer.nvfp4_num_experts, layer.nvfp4_hidden

    @staticmethod
    def for_layer(layer: torch.nn.Module) -> "Nvfp4ExpertFormat":
        """The format of a layer whose quant method recorded ``nvfp4_*`` dims (see ``ModelOptNvFp4FusedMoEMethod``)."""
        return Nvfp4ExpertFormat(
            layer_id=layer.layer_id,
            num_experts=layer.nvfp4_num_experts,
            hidden=layer.nvfp4_hidden,
            intermediate=layer.nvfp4_inter,
            gated=layer.nvfp4_gated,
            runner_config=getattr(layer, "moe_runner_config", None),
        )

    @staticmethod
    def build_streamer(layer: torch.nn.Module):
        """An ``ExpertStreamer`` for the layer. Its row source (NVFP4 row images) does not exist yet: see ``default_row_source``."""
        from sglang.srt.layers.moe.expert_stream import ExpertStreamer

        fmt = Nvfp4ExpertFormat.for_layer(layer)
        return ExpertStreamer(layer, fmt.names, layer_id=layer.layer_id, format=fmt)

    # ---- row source ----

    def default_row_source(
        self, layer: torch.nn.Module, specs: Sequence[ExpertTensorSpec], kind: str
    ) -> Optional[ExpertRowSource]:
        raise NotImplementedError(
            "the NVFP4 expert-file cache cannot be the native engine's row source as it is. (1) The engine's reader "
            "(expert_stream/host/row_tables.h, tables_from) reads one row-image file per layer and requires every part "
            "of a row to share one file size and one offset-minus-destination base; the cache is four per-tensor files "
            "per layer (w13_weight, w2_weight, w13_weight_scale, w2_weight_scale: rows at e * row_bytes, different "
            "sizes). (2) g1_alphas and g2_alphas are not in the cache (they are computed in memory at load), and the "
            "engine's segments must tile every slab row. The smallest adapter is an NVFP4 row-image builder: per layer "
            "one file of [experts, row_stride] images laid out by Nvfp4ExpertFormat.image_geometry(), written from "
            "the cache's verified views plus the alphas (padded to 512 B) and read through the engine's existing "
            "tables. That is a new on-disk format (builder, manifest, digests, open_row_images for NVFP4), so it is "
            "not built here."
        )

    def engine_tables(self, streamers: dict):
        """The native reader's flat tables: unavailable until the NVFP4 row images exist (see ``default_row_source``)."""
        return self.default_row_source(None, self._specs, "auto")

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional[ExpertRowSource]
    ) -> Optional[int]:
        return None if row_source is None else row_source.file_bytes_per_expert

    # The pinned-tier and hot-cache hooks are engine behaviour that today lives on the EXL3 format and only reads
    # ``layer_id``; reuse it until the engine moves it out of formats/exl3.py (as formats/mxfp4.py does).
    def pinned_tier_options(self, layer: torch.nn.Module) -> dict:
        from sglang.srt.layers.moe.expert_engine.formats.exl3 import Exl3ExpertFormat

        return Exl3ExpertFormat.pinned_tier_options(self, layer)

    def plan_graph_gather(self, streamer, rows: int, routes: int = 0) -> None:
        from sglang.srt.layers.moe.expert_engine.formats.exl3 import Exl3ExpertFormat

        Exl3ExpertFormat.plan_graph_gather(self, streamer, rows, routes)

    def attach_hot_cache_manager(self, manager, streamer) -> None:
        from sglang.srt.layers.moe.expert_engine.formats.exl3 import Exl3ExpertFormat

        Exl3ExpertFormat.attach_hot_cache_manager(self, manager, streamer)

    # ---- CPU experts ----

    @staticmethod
    def cpu_trait(*, source: str) -> Nvfp4EngineCpuTrait:
        """The NVFP4 CPU kernel's trait over the engine's tensor names (``source`` chooses nothing here)."""
        return Nvfp4EngineCpuTrait()

    # ---- host-side preparation ----

    def prepare_host_rows(self, layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Host-side preprocessing of ``tensor``, rows of the streamed tensor ``name``, as a new tensor where it changes.

        Block scales: a checkpoint-order ``[E, rows, K/16]`` matrix gets ``swizzle_blockscale(target_device="cpu")``, the
        swizzle ``process_weights_after_loading`` applies. Swizzled and checkpoint-order scales have the same shape
        whenever the dims are already tile-aligned (DSV4.1 and MiMo are), so the input is always taken as checkpoint
        order: a source that already holds swizzled rows must not be passed here. Alphas (one fp32
        per expert, ``[E]`` or ``[E, 1]``) are widened to their padded ``[E, 128]`` row. Packed weights are viewed as
        uint8. Source tensors are never mutated.
        """
        if name not in NVFP4_STREAMED_NAMES:
            raise ValueError(f"nvfp4 has no streamed tensor {name!r}; names: {NVFP4_STREAMED_NAMES}")
        if name in SCALE_NAMES:
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name} holds E4M3 scales, got {tensor.dtype}")
            raw_rows = self.w13_rows if name == "w13_blockscale_swizzled" else self.hidden
            cols = (self.hidden if name == "w13_blockscale_swizzled" else self.intermediate) // GROUP_SIZE
            if tuple(tensor.shape[1:]) != (raw_rows, cols):
                raise ValueError(
                    f"{name} rows are {tuple(tensor.shape[1:])}, not the checkpoint's {(raw_rows, cols)}"
                )
            from sglang.srt.layers.quantization.utils import swizzle_blockscale

            return swizzle_blockscale(tensor, target_device="cpu")
        if name in SIDE_NAMES:
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} is fp32 per expert, got {tensor.dtype}")
            return tensor.reshape(-1, 1).expand(-1, SIDE_ROW_ELEMENTS).contiguous()
        return tensor.view(torch.uint8) if tensor.dtype == torch.int8 else tensor

    # ---- GPU apply ----

    def _swiglu_params(self, layer: torch.nn.Module, slots: int, device: torch.device) -> tuple:
        """The CUTLASS SwiGLU ``(alpha, beta, limit)`` vectors for ``slots`` slots, built once per (slots, device)."""
        key = (slots, str(device))
        params = self._swiglu_cache.get(key)
        if params is None:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                materialize_swiglu_params_for_cutlass,
            )

            params = materialize_swiglu_params_for_cutlass(self._runner_config(layer), slots, device)
            self._swiglu_cache[key] = params
        return params

    def _runner_config(self, layer: torch.nn.Module) -> Any:
        config = self.runner_config or getattr(layer, "moe_runner_config", None)
        if config is None:
            raise RuntimeError(f"nvfp4 layer {self.layer_id}: no MoE runner config to apply the experts with")
        return config

    def moe_apply(self, layer, x, slot_ids, weights, slot_tensors) -> torch.Tensor:
        """The FlashInfer CUTLASS NVFP4 MoE over the hot-cache slots the gather filled (bf16 output).

        ``slot_ids`` is the gather's remap of this batch's expert ids to slot ids; ``slot_tensors`` is ``SlotApplyArgs``
        (the hot cache's slot buffers, the gather's ``keep``) or a mapping of the six names to ``[slots, ...]`` tensors.
        The layer carries ``w13_input_scale_quant`` and ``w2_input_scale_quant``.
        """
        tensors = _slot_tensors(slot_tensors)
        missing = [name for name in NVFP4_STREAMED_NAMES if name not in tensors]
        if missing:
            raise ValueError(f"nvfp4 moe_apply: slot buffers lack {missing}")
        scales = [getattr(layer, name, None) for name in ("w13_input_scale_quant", "w2_input_scale_quant")]
        if any(scale is None for scale in scales):
            raise RuntimeError(
                f"nvfp4 layer {self.layer_id}: w13_input_scale_quant / w2_input_scale_quant are unset; the row source "
                "binds a layer's activation scales (they are per layer, not per expert row)"
            )
        slots = tensors["w13_weight"].shape[0]
        return slot_moe_apply(
            runner_config=self._runner_config(layer),
            x=x,
            slot_ids=slot_ids,
            weights=weights,
            tensors=tensors,
            w13_input_scale_quant=scales[0],
            w2_input_scale_quant=scales[1],
            swiglu_params=self._swiglu_params(layer, slots, x.device),
            keep=getattr(slot_tensors, "keep", None),
        )
