"""The MXFP4 routed-expert format plugin of the expert engine (MiMo V2.6 Pro on SM120).

One expert's row is six tensors, in the C++ layout's copy-table order
(``csrc/moe/formats/mxfp4/row_layout.h``), as ``Mxfp4FlashinferCutlassMoEMethod`` holds them after its load-time
processing on SM120:

* ``w13_weight`` ``[2*I, H/2]`` int8: packed E2M1, ``[up; gate]``.
* ``w13_weight_scale`` ``[2*I, H/32]`` uint8: E8M0 block scales, already ``block_scale_interleave``d.
* ``w2_weight`` ``[H, I/2]`` int8 and ``w2_weight_scale`` ``[H, I/32]`` uint8, likewise.
* ``mxfp4_weight_global_scale`` and ``swiglu_limit``: the per-expert fp32 side tensors the runner indexes by expert.
  Streamed as rows so a hot-cache slot id selects them with the weights. Each is padded to ``SIDE_ROW_ELEMENTS``
  fp32 (512 B) with the value repeated, because the reader requires every row tensor to be a multiple of 512 B; the
  kernel reads element 0.

The GPU apply is slot-indexed: the hot-cache slot buffers stand in for the ``[E, ...]`` weight tensors, ``E`` is the
slot count and the expert ids are slot ids. It is the same FlashInfer call the dense method makes (the runner's
``_fused_experts_flashinfer_mxfp4_cutlass``), so the result equals the dense method's on the same experts.

Only SM120 (MXFP8 activations x MXFP4 weights) is streamed. SM90's weight interleave is a different row layout.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import torch

from sglang.srt.layers.moe.expert_format import ExpertTensorSpec
from sglang.srt.layers.moe.expert_transfer import NVFP4_TRANSFER_TENSOR_COUNT

if TYPE_CHECKING:
    from sglang.srt.layers.moe.expert_row_source import ExpertRowSource

logger = logging.getLogger(__name__)

MXFP4_STREAMED_NAMES = (
    "w13_weight",
    "w13_weight_scale",
    "w2_weight",
    "w2_weight_scale",
    "mxfp4_weight_global_scale",
    "swiglu_limit",
)
SCALE_NAMES = ("w13_weight_scale", "w2_weight_scale")
SIDE_NAMES = ("mxfp4_weight_global_scale", "swiglu_limit")
# fp32 elements in a side tensor's row: 512 B, the reader's row alignment (row_reader.h kImageAlign).
SIDE_ROW_ELEMENTS = 128
ROW_ALIGNMENT = 512
# MXFP4: one E8M0 scale per this many fp4 weights.
GROUP_SIZE = 32
# FlashInfer CUTLASS needs hidden and intermediate sizes to be multiples of this.
DIM_ALIGNMENT = 128


def expert_stream_enabled() -> bool:
    """Whether expert streaming is on for this process.

    The one place the MXFP4 model and quant method ask: ``ExpertStreamConfig.enabled``.
    """
    from sglang.srt.layers.moe.expert_engine.config import stream_config

    return stream_config().enabled


def ram_prefetch_enabled() -> bool:
    """Whether the RAM prefetch (which scores the next layer with the registered gates) is on; see ``expert_stream_enabled``."""
    from sglang.srt.layers.moe.expert_engine.config import stream_config

    return stream_config().prefetch.ram.enabled


def streams_experts(quant_config: Any, prefix: str) -> bool:
    """Whether the MoE layer whose experts module is at ``prefix`` streams its MXFP4 experts.

    True when streaming is on, ``quant_config`` is an FP8 config with MXFP4 routed experts that is not DeepSeek V4's
    (its ``is_dsv4_fp4_experts`` is set by the model loader), and the module is a MiMo V2.6 target MoE layer
    (layers 1.., not the MTP draft). The MiMo model and its quant method both ask this, so they cannot disagree.
    """
    if quant_config is None or not expert_stream_enabled():
        return False
    if not getattr(quant_config, "is_fp4_experts", False) or getattr(quant_config, "is_dsv4_fp4_experts", False):
        return False
    from sglang.srt.layers.moe.expert_engine.profiles.mimo_v26 import MimoV26Profile

    return MimoV26Profile().is_streamed_module(prefix)


def interleave_scale_rows(scales: torch.Tensor) -> torch.Tensor:
    """``flashinfer.block_scale_interleave`` of E8M0 scale bytes, on the host, as a new tensor of the same shape.

    ``scales`` is ``[rows, cols]`` for one expert or ``[experts, rows, cols]``; ``rows`` must be a multiple of 128 and
    ``cols`` of 4 (``Mxfp4FlashinferCutlassMoEMethod`` reshapes the interleave back to the input's shape, which needs
    that too). The swizzle is per matrix, so a batch equals its experts interleaved one by one. The source tensor is
    never modified.
    """
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    if scales.dtype != torch.uint8:
        raise TypeError(f"MXFP4 scales are E8M0 bytes (uint8), got {scales.dtype}")
    if scales.ndim not in (2, 3):
        raise ValueError(f"MXFP4 scales must be [rows, cols] or [experts, rows, cols], got {tuple(scales.shape)}")
    rows, cols = scales.shape[-2:]
    if rows % 128 or cols % 4:
        raise ValueError(
            f"MXFP4 scale interleave needs rows % 128 == 0 and cols % 4 == 0, got {rows} x {cols}"
        )
    lead = tuple(scales.shape[:-2])
    tiles = scales.reshape(*lead, rows // 128, 4, 32, cols // 4, 4)
    # [m_tile, r_hi(4), r_lo(32), k_tile, k_in(4)] -> [m_tile, k_tile, r_lo, r_hi, k_in]
    n = len(lead)
    order = (*range(n), n, n + 3, n + 2, n + 1, n + 4)
    return tiles.permute(order).contiguous().reshape(*lead, rows, cols)


def _slot_tensors(slot_tensors: Any) -> Mapping[str, torch.Tensor]:
    """The slot buffers by name: a mapping as given, or the hot cache of ``SlotApplyArgs.streamer``."""
    if isinstance(slot_tensors, Mapping):
        return slot_tensors
    return slot_tensors.streamer.hot_cache.tensors


def slot_moe_apply(
    *,
    runner_config: Any,
    x: torch.Tensor,
    slot_ids: torch.Tensor,
    weights: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    use_swiglu_step: bool,
    use_swiglu_limit: bool = True,
    keep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FlashInfer CUTLASS MXFP4 MoE over slot buffers: ``E`` = the buffers' slot count, expert ids = slot ids.

    ``tensors`` holds the six row tensors as ``[slots, *row_shape]``. ``slot_ids`` and ``weights`` are ``[M, top_k]``
    (or flat ``M * top_k``, reshaped by ``x``'s rows); ``keep`` is the gather's fp32 ``[1]`` drop flag, which scales the
    route weights (0 empties the routed output). Returns the bf16 ``[M, H]`` output.
    """
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        FlashInferCutlassMxfp4MoeQuantInfo,
        _fused_experts_flashinfer_mxfp4_cutlass,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    rows = x.shape[0]
    ids = slot_ids.reshape(rows, -1).to(torch.int32)
    route_weights = weights.reshape(rows, -1).to(torch.float32)
    if keep is not None:
        route_weights = route_weights * keep.to(torch.float32)
    quant_info = FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=tensors["w13_weight"],
        w2_weight=tensors["w2_weight"],
        w13_weight_scale=tensors["w13_weight_scale"],
        w2_weight_scale=tensors["w2_weight_scale"],
        # Element 0 of each slot's padded side row: [slots, 128] -> contiguous [slots].
        mxfp4_weight_global_scale=tensors["mxfp4_weight_global_scale"][:, 0].contiguous(),
        # No configured clamp: the dense method passes None, so the kernel must get None here too.
        swiglu_limit=tensors["swiglu_limit"][:, 0].contiguous() if use_swiglu_limit else None,
        use_swiglu_step=use_swiglu_step,
    )
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(route_weights, ids, None),
    )
    return _fused_experts_flashinfer_mxfp4_cutlass(dispatch, quant_info, runner_config).hidden_states


class Mxfp4ExpertFormat:
    """``ExpertFormat`` for one layer of an MXFP4 checkpoint's routed experts.

    Spec-only: no dense ``[experts, ...]`` source exists (the streamed experts are never materialized), so every host
    row comes from the row source. Built from the dims the quant method records on the layer (``layer_dims``).
    """

    key = "mxfp4"
    transport_build = "mxfp4"
    engine_served = True
    supports_graph_gather = False
    supports_host_arena = False
    supports_transfer_executor = len(MXFP4_STREAMED_NAMES) == NVFP4_TRANSFER_TENSOR_COUNT
    # The CPU kernel and the draft are stubs (UnsupportedCpuKernel in the C++ build): policy never routes work to them.
    supports_cpu_experts = False
    supports_draft = False
    supports_dead_rows = False
    graph_source_kind = "pinned_tier"
    max_gather_rows: Optional[int] = None
    inclusive_pinned_tier = False
    names = MXFP4_STREAMED_NAMES

    def __init__(
        self,
        *,
        layer_id: int,
        num_experts: int,
        hidden: int,
        intermediate: int,
        use_swiglu_step: bool = False,
        swiglu_limit: Optional[float] = None,
        runner_config: Any = None,
        source_root: Optional[str] = None,
    ) -> None:
        if hidden % DIM_ALIGNMENT or intermediate % DIM_ALIGNMENT:
            raise ValueError(
                f"mxfp4 needs hidden and intermediate sizes that are multiples of {DIM_ALIGNMENT} "
                f"(got hidden={hidden}, intermediate={intermediate})"
            )
        if num_experts <= 0:
            raise ValueError(f"mxfp4 layer {layer_id} has {num_experts} experts")
        self.layer_id = layer_id
        self.experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.use_swiglu_step = use_swiglu_step
        # None: the model configures no clamp, the kernel gets no limit and the side row is a placeholder.
        self.swiglu_limit = swiglu_limit
        self.runner_config = runner_config
        self.source_root = source_root
        self._specs = (
            ExpertTensorSpec("w13_weight", (2 * intermediate, hidden // 2), torch.int8, "host"),
            ExpertTensorSpec("w13_weight_scale", (2 * intermediate, hidden // GROUP_SIZE), torch.uint8, "host"),
            ExpertTensorSpec("w2_weight", (hidden, intermediate // 2), torch.int8, "host"),
            ExpertTensorSpec("w2_weight_scale", (hidden, intermediate // GROUP_SIZE), torch.uint8, "host"),
            ExpertTensorSpec("mxfp4_weight_global_scale", (SIDE_ROW_ELEMENTS,), torch.float32, "host"),
            ExpertTensorSpec("swiglu_limit", (SIDE_ROW_ELEMENTS,), torch.float32, "host"),
        )
        assert tuple(spec.name for spec in self._specs) == MXFP4_STREAMED_NAMES
        self.check_row_alignment()

    def row_alignment_violations(self) -> list[tuple[str, int]]:
        """``(name, row bytes)`` of every row tensor that is not a multiple of 512 B (empty for this format)."""
        return [(s.name, s.row_bytes) for s in self._specs if s.row_bytes % ROW_ALIGNMENT]

    def check_row_alignment(self) -> None:
        bad = self.row_alignment_violations()
        if bad:
            raise ValueError(f"mxfp4 layer {self.layer_id}: row tensors not {ROW_ALIGNMENT} B aligned: {bad}")

    def tensor_specs(self, layer: torch.nn.Module) -> tuple[ExpertTensorSpec, ...]:
        return self._specs

    def num_experts(self, layer: torch.nn.Module) -> int:
        return self.experts

    def source(self, layer: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        return None

    @staticmethod
    def layer_dims(layer: torch.nn.Module) -> tuple[int, int]:
        return layer.mxfp4_num_experts, layer.mxfp4_hidden

    @staticmethod
    def for_layer(layer: torch.nn.Module) -> "Mxfp4ExpertFormat":
        """The format of a layer whose quant method recorded ``mxfp4_*`` dims (see ``Mxfp4FlashinferCutlassMoEMethod``)."""
        return Mxfp4ExpertFormat(
            layer_id=layer.layer_id,
            num_experts=layer.mxfp4_num_experts,
            hidden=layer.mxfp4_hidden,
            intermediate=layer.mxfp4_inter,
            use_swiglu_step=layer.mxfp4_use_swiglu_step,
            swiglu_limit=layer.mxfp4_swiglu_limit,
            runner_config=getattr(layer, "moe_runner_config", None),
        )

    @staticmethod
    def build_streamer(layer: torch.nn.Module):
        """An ``ExpertStreamer`` for the layer. The row source (MXFP4 row images) is not built yet: see ``default_row_source``."""
        from sglang.srt.layers.moe.expert_stream import ExpertStreamer

        fmt = Mxfp4ExpertFormat.for_layer(layer)
        return ExpertStreamer(layer, fmt.names, layer_id=layer.layer_id, format=fmt)

    def default_row_source(
        self, layer: torch.nn.Module, specs: Sequence[ExpertTensorSpec], kind: str
    ) -> Optional[ExpertRowSource]:
        raise NotImplementedError(
            "mxfp4 expert row images (checkpoint -> interleaved, 512 B padded rows) have no row source yet; "
            "prepare_host_rows is the host step such a source applies"
        )

    def file_source_bytes_per_expert(
        self, layer: torch.nn.Module, row_source: Optional[ExpertRowSource]
    ) -> Optional[int]:
        return None if row_source is None else row_source.file_bytes_per_expert

    # The pinned-tier and hot-cache hooks are engine behaviour that today lives on the EXL3 format and only reads
    # ``layer_id``; reuse it until the engine moves it out of formats/exl3.py.
    def pinned_tier_options(self, layer: torch.nn.Module) -> dict:
        from sglang.srt.layers.moe.expert_engine.formats.exl3 import Exl3ExpertFormat

        return Exl3ExpertFormat.pinned_tier_options(self, layer)

    def plan_graph_gather(self, streamer, rows: int, routes: int = 0) -> None:
        from sglang.srt.layers.moe.expert_engine.formats.exl3 import Exl3ExpertFormat

        Exl3ExpertFormat.plan_graph_gather(self, streamer, rows, routes)

    def attach_hot_cache_manager(self, manager, streamer) -> None:
        from sglang.srt.layers.moe.expert_engine.formats.exl3 import Exl3ExpertFormat

        Exl3ExpertFormat.attach_hot_cache_manager(self, manager, streamer)

    @staticmethod
    def cpu_trait(*, source: str):
        """None: MXFP4 has no CPU kernel yet (the C++ build's ``UnsupportedCpuKernel`` stub)."""
        return None

    def side_row_value(self, name: str) -> float:
        """The value every element of a side tensor's row holds for this layer: the neutral global scale (1.0, what
        FlashInfer's MXFP4 ABI requires) or the SwiGLU clamp (0.0 when the model has none: the kernel ignores it)."""
        if name == "mxfp4_weight_global_scale":
            return 1.0
        if name == "swiglu_limit":
            return 0.0 if self.swiglu_limit is None else float(self.swiglu_limit)
        raise ValueError(f"{name!r} is not an mxfp4 side tensor; names: {SIDE_NAMES}")

    def prepare_host_rows(self, layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Host-side preprocessing of ``tensor``, rows of the streamed tensor ``name``, as a new tensor.

        Scales get the ``block_scale_interleave`` swizzle; packed weights are viewed as int8; a side tensor (one fp32
        per expert, ``[E]`` or ``[E, 1]``) is widened to its padded ``[E, 128]`` row. Source tensors are never mutated.
        """
        if name in SCALE_NAMES:
            return interleave_scale_rows(tensor)
        if name in SIDE_NAMES:
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} is fp32 per expert, got {tensor.dtype}")
            return tensor.reshape(-1, 1).expand(-1, SIDE_ROW_ELEMENTS).contiguous()
        if name in ("w13_weight", "w2_weight"):
            return tensor.view(torch.int8) if tensor.dtype == torch.uint8 else tensor
        raise ValueError(f"mxfp4 has no streamed tensor {name!r}; names: {MXFP4_STREAMED_NAMES}")

    def moe_apply(self, layer, x, slot_ids, weights, slot_tensors) -> torch.Tensor:
        """The FlashInfer MXFP4 MoE over the hot-cache slots the gather filled (bf16 output).

        ``slot_ids`` is the gather's remap of this batch's expert ids to slot ids; ``slot_tensors`` is ``SlotApplyArgs``
        (the hot cache's slot buffers, the gather's ``keep``) or a mapping of the six names to ``[slots, ...]`` tensors.
        """
        tensors = _slot_tensors(slot_tensors)
        missing = [name for name in MXFP4_STREAMED_NAMES if name not in tensors]
        if missing:
            raise ValueError(f"mxfp4 moe_apply: slot buffers lack {missing}")
        return slot_moe_apply(
            runner_config=self.runner_config or getattr(layer, "moe_runner_config", None),
            x=x,
            slot_ids=slot_ids,
            weights=weights,
            tensors=tensors,
            use_swiglu_step=self.use_swiglu_step,
            use_swiglu_limit=self.swiglu_limit is not None,
            keep=getattr(slot_tensors, "keep", None),
        )


def streamed_apply(layer: torch.nn.Module, method: Any, dispatch_output: Any):
    """The streamed branch of ``Mxfp4FlashinferCutlassMoEMethod.apply``: gather, then ``moe_apply`` over slots.

    This is the small function the quant method calls until ``quant_hook.apply`` serves non-EXL3 formats; replace its
    body with that call then. Returns a ``StandardCombineInput``.
    """
    from sglang.srt.layers.moe.expert_format import expert_streamer_of
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    streamer = expert_streamer_of(layer)
    if streamer is None:
        raise RuntimeError(f"mxfp4 layer {getattr(layer, 'layer_id', '?')} streams but has no expert streamer")
    topk = dispatch_output.topk_output
    if TopKOutputChecker.format_is_bypassed(topk):
        raise RuntimeError("mxfp4 streaming needs STANDARD top-k output (mimo_v2 forces it when streaming)")
    x = dispatch_output.hidden_states
    if x.shape[0] == 0:
        return StandardCombineInput(hidden_states=x)
    remap, _ = streamer.gather(topk.topk_ids)
    out = streamer.format.moe_apply(
        layer, x, remap, topk.topk_weights, _SlotArgs(streamer, streamer.row_backend.keep)
    )
    return StandardCombineInput(hidden_states=out)


class _SlotArgs:
    """``SlotApplyArgs``-shaped (``streamer``, ``keep``) without the EXL3-only ``swiglu_limit`` and ``cpu`` fields."""

    swiglu_limit = None
    cpu = None

    def __init__(self, streamer: Any, keep: Optional[torch.Tensor]) -> None:
        self.streamer = streamer
        self.keep = keep
