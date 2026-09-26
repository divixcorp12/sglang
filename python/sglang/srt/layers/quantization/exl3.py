"""EXL3 (exllamav3 trellis) quantization for the DeepSeek V4.1 EXL3 export.

Each linear is stored as trellis/suh/svh/mul1. A merged linear (e.g. the shared
expert's gate_up) keeps its parts separate: every part carries its own input
sign vector, so parts cannot be concatenated into one trellis.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Mapping, Optional, Sequence

import threading

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_format import (
    EXL3_STREAMED_NAMES,
    build_exl3_expert_streamer,
)
from sglang.srt.layers.moe.exl3_stream_trace import (
    capturing_graphs,
    get_exl3_stream_trace,
)
from sglang.srt.layers.moe.expert_format import STREAMER_ATTRIBUTE, expert_streamer_of
from sglang.srt.layers.quantization.exl3_ops import (
    Exl3RoutePlan,
    Exl3Tensors,
    assert_not_capturing,
    exl3_gemm_bs1,
    exl3_half_input,
    exl3_linear,
    exl3_moe_accumulate,
    exl3_moe_accumulate_planned,
    exl3_moe_loop,
)
from sglang.srt.utils import set_weight_attrs

EXL3_PARAMS = ("trellis", "suh", "svh", "mul1")
UNQUANTIZED_PREFIX_SUFFIXES = (".gate", ".weights_proj")


class Exl3Config(QuantizationConfig):
    def __init__(self, bits: float, head_bits: int, codebook: str, version: str):
        super().__init__()
        if codebook != "mul1":
            raise ValueError(f"exl3: only the mul1 codebook is supported, got {codebook}")
        self.bits, self.head_bits, self.codebook, self.version = bits, head_bits, codebook, version

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Exl3Config":
        return cls(
            bits=config["bits"],
            head_bits=config.get("head_bits", 6),
            codebook=config["codebook"],
            version=config["version"],
        )

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
            UnquantizedLinearMethod,
        )
        from sglang.srt.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        if isinstance(layer, ParallelLMHead):
            return Exl3LinearMethod(self)
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        if isinstance(layer, LinearBase):
            if prefix.endswith(UNQUANTIZED_PREFIX_SUFFIXES):
                return UnquantizedLinearMethod()
            return Exl3LinearMethod(self)
        if isinstance(layer, FusedMoE):
            from sglang.srt.models.deepseek_v4_exl3_weights import (
                is_streamed_expert_module,
            )

            return Exl3MoEMethod(self, streamed=is_streamed_expert_module(prefix))
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


# Weight loaders run on a thread pool (deepseek_v4.load_weights), so several experts of one
# layer can reach an empty param at once; unguarded, each allocated its own buffer and the
# losers' copies landed in a discarded one (zeroed experts, and a transient 2-4x VRAM peak).
_MATERIALIZE_LOCK = threading.Lock()


def _materialize(param: nn.Parameter, lead: tuple[int, ...], loaded: torch.Tensor) -> None:
    shape = lead + tuple(loaded.shape)
    if param.numel() == 0:
        with _MATERIALIZE_LOCK:
            if param.numel() == 0:
                param.data = torch.zeros(shape, dtype=loaded.dtype, device=param.device)
    if tuple(param.shape) != shape or param.dtype != loaded.dtype:
        raise ValueError(f"exl3: expected {shape} {param.dtype}, got {tuple(loaded.shape)} {loaded.dtype}")


def _load_linear_part(layer, name, param, loaded_weight, shard_id=None):
    if shard_id is None:
        part = 0
    elif isinstance(shard_id, int):
        part = shard_id
    else:
        raise NotImplementedError(
            f"exl3: string shard ids ({shard_id!r}) are not supported; "
            "QKVParallelLinear-style fused q/k/v shards cannot be loaded as "
            "separate exl3 parts (each exl3 part needs its own trellis/suh/svh, "
            "one per positional output_partition_sizes entry, not one per named shard)"
        )
    _materialize(param, (layer.exl3_parts,), loaded_weight)
    param.data[part].copy_(loaded_weight)
    layer.exl3_loaded.add((name, part))


class Exl3LinearMethod(LinearMethodBase):
    applies_without_weight = True

    def __init__(self, config: Exl3Config):
        self.config = config
        self.cast_fusion = envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.get()

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # exl3 supports TP=1 only (no sharding of trellis/suh/svh implemented yet).
        # input_size_per_partition != input_size means row-parallel (RowParallelLinear)
        # sharding on the input dim; sum(output_partition_sizes) != output_size means
        # column-parallel (ColumnParallelLinear/QKVParallelLinear/ParallelLMHead)
        # sharding on the output dim. Both are equalities at TP=1 by construction in
        # every LinearBase/VocabParallelEmbedding call site (vocab padding included:
        # ParallelLMHead passes the already-padded output_size, and its per-partition
        # size equals that padded total when tp_size == 1), so this also correctly
        # rejects TP>1 for the LM head without needing get_tensor_model_parallel_world_size().
        if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
            raise NotImplementedError("exl3 supports tensor-parallel size 1 only")
        sizes = list(output_partition_sizes)
        if len(set(sizes)) != 1:
            raise ValueError(f"exl3: merged linear parts must be equal, got {sizes}")
        layer.exl3_parts = len(sizes)
        layer.exl3_in = input_size_per_partition
        layer.exl3_out_part = sizes[0]
        layer.exl3_loaded = set()
        for name in EXL3_PARAMS:
            param = nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False)
            set_weight_attrs(
                param, {"weight_loader": functools.partial(_load_linear_part, layer, name)}
            )
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        for part in range(layer.exl3_parts):
            missing = [n for n in EXL3_PARAMS if (n, part) not in layer.exl3_loaded]
            if missing:
                raise RuntimeError(f"exl3: part {part} is missing {missing}")
        layer.exl3_tensors = [
            Exl3Tensors(
                trellis=layer.trellis[p],
                suh=layer.suh[p],
                svh=layer.svh[p],
                mul1=True,
            )
            for p in range(layer.exl3_parts)
        ]
        for t in layer.exl3_tensors:
            if (t.in_features, t.out_features) != (layer.exl3_in, layer.exl3_out_part):
                raise RuntimeError(
                    f"exl3: loaded {t.in_features}x{t.out_features}, "
                    f"module expects {layer.exl3_in}x{layer.exl3_out_part}"
                )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.cast_fusion and bias is None and x.dtype == torch.bfloat16 and x.numel() == layer.exl3_in:
            y = exl3_gemm_bs1(exl3_half_input(x), layer.exl3_tensors).to(x.dtype)
            return y.reshape(*x.shape[:-1], y.shape[-1])
        outs = [exl3_linear(x, t) for t in layer.exl3_tensors]
        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        return y if bias is None else y + bias


def exl3_cast_fusion_mlp(gate_up: nn.Module, down: nn.Module) -> bool:
    """Whether a gate_up/down MLP runs as ``exl3_swiglu_mlp`` at BS1 (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION)."""
    return all(
        isinstance(linear.quant_method, Exl3LinearMethod) and linear.quant_method.cast_fusion
        for linear in (gate_up, down)
    )


def exl3_swiglu_mlp(x: torch.Tensor, gate_up: nn.Module, down: nn.Module, swiglu_limit: float) -> torch.Tensor:
    """DeepseekV2MLP's gate_up -> silu_and_mul_clamp -> down for one bf16 row, fp16 between the EXL3 gemvs."""
    from sglang.kernels.ops.moe.dsv41_cast_fusion import exl3_silu_mul_clamp_half

    hidden = exl3_silu_mul_clamp_half(exl3_gemm_bs1(exl3_half_input(x), gate_up.exl3_tensors), swiglu_limit)
    y = exl3_gemm_bs1(hidden, down.exl3_tensors).to(x.dtype)
    return y.reshape(*x.shape[:-1], y.shape[-1])


_SLOTS = {"w1": ("w13", 0), "w3": ("w13", 1), "w2": ("w2", 0)}


def _load_expert(layer, prefix, name, param, loaded_weight, weight_name=None, *, shard_id, expert_id):
    want_prefix, slot = _SLOTS[shard_id]
    if want_prefix != prefix:
        raise ValueError(f"exl3: {shard_id} routed to {prefix}_{name}")
    local = layer.exl3_local_expert(expert_id)
    if local < 0:
        return
    _materialize(param, (layer.exl3_num_experts, 2 if prefix == "w13" else 1), loaded_weight)
    param.data[local, slot].copy_(loaded_weight)
    layer.exl3_loaded.add((prefix, name, local, slot))


class Exl3RowViews:
    """``Exl3Tensors`` views of gathered rows, built once per buffer row and reused.

    Gathered rows live in a few stable buffers: the shared eager staging rows
    and each layer's hot-cache slots. Building the dataclasses per call would
    cost about a thousand constructions per prefill layer, so views are cached
    by the buffers' addresses and the row. A cached view keeps its buffer
    alive, so an address is never reused under it. ``max_buffers`` bounds what
    that can pin: 40 layers' hot slots plus the shared staging set fit in 48,
    and a buffer that stops being used (a reallocated staging set holds about
    852 MB of VRAM) is dropped at the next overflow.
    """

    def __init__(self, max_buffers: int = 48) -> None:
        self.max_buffers = max_buffers
        self._buffers: dict[tuple, dict[int, tuple[Exl3Tensors, Exl3Tensors, Exl3Tensors]]] = {}

    def select(
        self,
        rows: Mapping[str, torch.Tensor],
        experts: Sequence[int],
        row_of_source: Sequence[int],
    ) -> tuple[dict[int, tuple[Exl3Tensors, Exl3Tensors]], dict[int, Exl3Tensors]]:
        """``(w13, w2)`` keyed by expert id: expert ``experts[i]`` is row ``row_of_source[i]``."""
        key = tuple((name, rows[name].data_ptr()) for name in EXL3_STREAMED_NAMES)
        cached = self._buffers.get(key)
        if cached is None:
            if len(self._buffers) >= self.max_buffers:
                self._buffers.clear()
            cached = self._buffers[key] = {}
        w13, w2 = {}, {}
        for expert, row in zip(experts, row_of_source):
            views = cached.get(row)
            if views is None:
                views = cached[row] = (
                    self._view(rows, "w13", row, 0),
                    self._view(rows, "w13", row, 1),
                    self._view(rows, "w2", row, 0),
                )
            w13[expert] = (views[0], views[1])
            w2[expert] = views[2]
        return w13, w2

    @staticmethod
    def _view(rows: Mapping[str, torch.Tensor], prefix: str, row: int, part: int) -> Exl3Tensors:
        return Exl3Tensors(
            trellis=rows[f"{prefix}_trellis"][row, part],
            suh=rows[f"{prefix}_suh"][row, part],
            svh=rows[f"{prefix}_svh"][row, part],
            mul1=True,
        )


EXL3_ROW_VIEWS = Exl3RowViews()


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, config: Exl3Config, *, streamed: bool):
        self.config = config
        self.streamed = streamed
        self.moe_runner_config = None
        self.cast_fusion = envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.get()
        self.route_plan = envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.get()

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # exl3 supports TP=1 only (no sharding of trellis/suh/svh implemented yet).
        # FusedMoE.__init__ passes the pre-shard size through as the
        # `moe_intermediate_size` extra kwarg (fused_moe_triton/layer.py:508-520:
        # `intermediate_size_per_partition=self.intermediate_size_per_partition,
        # ..., moe_intermediate_size=intermediate_size`), and computes
        # `self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size`
        # (fused_moe_triton/layer.py:402-403). At TP=1 the two are equal; a caller
        # (e.g. the no-TP-group CPU unit tests) that omits `moe_intermediate_size`
        # entirely is treated as TP=1 by construction.
        full_intermediate_size = extra_weight_attrs.get(
            "moe_intermediate_size", intermediate_size_per_partition
        )
        if full_intermediate_size != intermediate_size_per_partition:
            raise NotImplementedError("exl3 supports tensor-parallel size 1 only")
        layer.exl3_num_experts = num_experts
        layer.exl3_hidden = hidden_size
        layer.exl3_inter = intermediate_size_per_partition
        layer.exl3_loaded = set()
        layer.exl3_streamed = self.streamed and envs.SGLANG_DSV41_EXPERT_STREAM.get()
        if layer.exl3_streamed:
            # Routed experts stay on disk. With no parameters, load_weights skips
            # them (deepseek_v4.load_weights' skip_unmaterialized_expert_param),
            # and process_weights_after_loading attaches an expert streamer.
            return
        # CONTRACT: map a global expert id to this rank's local slot (-1 = not ours),
        # using the same helper FusedMoE.weight_loader uses (identity at TP1/EP1).
        # Evidence: fused_moe_triton/layer.py:993-1000 (_map_global_expert_id_to_local_expert_id)
        # and :1031 (weight_loader calling it before _weight_loader_impl).
        layer.exl3_local_expert = getattr(
            layer, "_map_global_expert_id_to_local_expert_id", lambda expert_id: expert_id
        )
        for prefix in ("w13", "w2"):
            for name in EXL3_PARAMS:
                param = nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False)
                set_weight_attrs(
                    param, {"weight_loader": functools.partial(_load_expert, layer, prefix, name)}
                )
                layer.register_parameter(f"{prefix}_{name}", param)

    def create_moe_runner(self, layer: nn.Module, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "exl3_streamed", False):
            setattr(layer, STREAMER_ATTRIBUTE, build_exl3_expert_streamer(layer))
            return
        for e in range(layer.exl3_num_experts):
            for prefix, slots in (("w13", (0, 1)), ("w2", (0,))):
                for slot in slots:
                    missing = [n for n in EXL3_PARAMS if (prefix, n, e, slot) not in layer.exl3_loaded]
                    if missing:
                        raise RuntimeError(f"exl3: expert {e} {prefix}[{slot}] is missing {missing}")

        def tensors(prefix, e, slot):
            return Exl3Tensors(
                trellis=getattr(layer, f"{prefix}_trellis")[e, slot],
                suh=getattr(layer, f"{prefix}_suh")[e, slot],
                svh=getattr(layer, f"{prefix}_svh")[e, slot],
                mul1=True,
            )

        layer.exl3_w13 = [(tensors("w13", e, 0), tensors("w13", e, 1)) for e in range(layer.exl3_num_experts)]
        layer.exl3_w2 = [tensors("w2", e, 0) for e in range(layer.exl3_num_experts)]

        # Same shape check Exl3LinearMethod does after loading (exl3.py Exl3LinearMethod
        # .process_weights_after_loading above): a part that decoded to the wrong
        # in/out shape is caught here rather than surfacing as a garbled matmul later.
        for e in range(layer.exl3_num_experts):
            for slot, t in enumerate(layer.exl3_w13[e]):
                if (t.in_features, t.out_features) != (layer.exl3_hidden, layer.exl3_inter):
                    raise RuntimeError(
                        f"exl3: expert {e} w13[{slot}] loaded {t.in_features}x{t.out_features}, "
                        f"module expects {layer.exl3_hidden}x{layer.exl3_inter}"
                    )
            t2 = layer.exl3_w2[e]
            if (t2.in_features, t2.out_features) != (layer.exl3_inter, layer.exl3_hidden):
                raise RuntimeError(
                    f"exl3: expert {e} w2[0] loaded {t2.in_features}x{t2.out_features}, "
                    f"module expects {layer.exl3_inter}x{layer.exl3_hidden}"
                )

    def apply(self, layer: nn.Module, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        cfg = self.moe_runner_config
        # CONTRACT: exl3_moe_loop applies the route weight before w2 and does not
        # apply it on the input; reject configs that ask otherwise, matching
        # moe_forward_native (fused_moe_native.py:66) and fused_moe_forward_native
        # (fused_moe_native.py:30), which both raise NotImplementedError() for
        # apply_router_weight_on_input=True.
        if getattr(cfg, "apply_router_weight_on_input", False):
            raise NotImplementedError("exl3 MoE: apply_router_weight_on_input")
        # By name: StandardTopKOutputPacked (moe_fused_gate) carries a 4th field.
        topk = dispatch_output.topk_output
        if TopKOutputChecker.format_is_bypassed(topk):
            # The draft path supplies routing lazily (hidden_states / router_logits); the
            # exl3 loop needs explicit ids and weights, so materialize them once here.
            topk = topk.to_standard(layer_id=layer.layer_id)
        topk_weights, topk_ids = topk.topk_weights, topk.topk_ids
        streamer = expert_streamer_of(layer)
        scale = (
            cfg.routed_scaling_factor
            if cfg.routed_scaling_factor is not None and not layer.should_fuse_routed_scaling_factor_in_topk
            else None
        )
        x = dispatch_output.hidden_states
        if streamer is not None and streamer.serves_graph_gather(topk):
            if self.cast_fusion and scale is not None and x.dtype == torch.bfloat16:
                from sglang.kernels.ops.moe.dsv41_cast_fusion import exl3_scale_to_bf16

                out = self._apply_graph(layer, streamer, x, topk_weights, topk_ids, cfg.swiglu_limit, cast=False)
                return StandardCombineInput(hidden_states=exl3_scale_to_bf16(out, scale))
            out = self._apply_graph(layer, streamer, x, topk_weights, topk_ids, cfg.swiglu_limit)
        elif streamer is not None:
            assert_not_capturing("Exl3MoEMethod.apply")
            out = self._apply_streamed(
                layer,
                streamer,
                dispatch_output.hidden_states,
                topk_weights,
                topk_ids,
                cfg.swiglu_limit,
                route_plan=self.route_plan,
            )
        else:
            assert_not_capturing("Exl3MoEMethod.apply")
            out = exl3_moe_loop(
                dispatch_output.hidden_states,
                topk_weights,
                topk_ids,
                layer.exl3_w13,
                layer.exl3_w2,
                # CONTRACT: MoeRunnerConfig.swiglu_limit (moe_runner/base.py:60) is the
                # DeepSeek V4 swiglu clamp field; deep_gemm._apply_swiglu_limit
                # (moe_runner/deep_gemm.py:1666-1667) clamps up to +-limit and gate to
                # <= limit, matching exl3_moe_loop exactly.
                cfg.swiglu_limit,
            )
        # On CUDA, DeepseekV2MoE never scales the routed output itself (its multiply is
        # under `not _is_cuda`): the runner does, unless the factor is already fused into
        # topk_weights -- as the unquantized triton path does (unquant.py:1075).
        if scale is not None:
            out = out * scale
        return StandardCombineInput(hidden_states=out)

    @staticmethod
    def _apply_graph(layer, streamer, x, topk_weights, topk_ids, swiglu_limit, cast=True):
        """BS1 decode inside a CUDA graph: device-only gather, then the fused MoE over slots.

        Hits are read in place from the hot cache, misses land in scratch rows from
        the pinned tier (PinnedTierRowBackend), and routes are recorded on the device
        by the planner. A row missing from RAM sets the backend's ``keep`` to 0,
        which drops this layer's routed output. Only option C (Task 14) serves such
        a miss and fail-stops on one it cannot serve, so a plain
        ``PinnedTierRowBackend`` is refused unless the layer sets the test-only
        ``_exl3_allow_p3_only``.
        """
        from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissRowBackend
        from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend
        from sglang.srt.layers.quantization.exl3_fused_moe import exl3_fused_moe_for

        if type(streamer.row_backend) is PinnedTierRowBackend and not getattr(layer, "_exl3_allow_p3_only", False):
            # Without option C nothing serves or checks a RAM miss inside a replay.
            raise RuntimeError(
                f"exl3 in-graph MoE, layer {getattr(layer, 'layer_id', '?')}: the graph gather "
                "reads the pinned tier, but option C is not installed (the row backend is a "
                "plain PinnedTierRowBackend, not Exl3RamMissRowBackend), so a RAM miss inside a "
                "replay would be neither served nor fail-stopped. Option C installs when "
                "SGLANG_MOE_EXPERT_GRAPH_GATHER is on at load and the hot cache manager "
                "attaches the EXL3 format."
            )
        if swiglu_limit is None:
            raise NotImplementedError("exl3 in-graph MoE: a swiglu_limit is required (DSV4.1 sets 10.0)")
        routes = getattr(streamer.row_backend, "routes", None)
        if routes is not None:
            # The option C post protects every routed expert of this layer. Captured as a
            # device copy, so every replay refreshes it from the live topk_ids.
            flat = topk_ids.reshape(-1)
            routes[: flat.numel()].copy_(flat)
            routes[flat.numel() :].fill_(-1)
        backend = streamer.row_backend
        prefetch = getattr(backend.device_side, "native_prefetch", None) if isinstance(backend, Exl3RamMissRowBackend) else None
        if prefetch is not None:
            # Before the gather reads residency: wait for the previous layer's prefetch into this one and map it.
            prefetch.commit(layer.layer_id, backend.routes)
        remap, _ = streamer.gather(topk_ids)
        if isinstance(backend, Exl3RamMissRowBackend) and backend.route_log is not None:
            if backend.route_log.router is not None:
                # After the gather: its post ran the layer's route_log.record, which row 0 used to take the slot.
                backend.route_log.record_router(backend.row, x, topk_weights)
        if prefetch is not None:
            # After this layer's demand chain, before its MoE: the next layer's gate on this layer's router input.
            prefetch.predict(layer.layer_id, x)
        fused = exl3_fused_moe_for(layer, streamer)
        out = fused.run(
            x,
            topk_weights.reshape(-1),
            # Layer fusion takes the router's int32 remap and writes the int64 copy inside its one kernel.
            remap.reshape(-1) if fused.layer_fusion else remap.reshape(-1).long(),
            streamer.row_backend.keep,
            swiglu_limit,
        )
        # cast=False hands the fp32 output to exl3_scale_to_bf16, which casts and scales in one kernel.
        return out.to(x.dtype) if cast else out

    @staticmethod
    def _apply_streamed(layer, streamer, x, topk_weights, topk_ids, swiglu_limit, route_plan=False):
        """Routed experts gathered in chunks of distinct experts by the streamer.

        Routes are recorded once for the whole call; every chunk's experts run
        before the next chunk reuses the staging rows. The fp32 accumulation
        order is ascending expert id, the same as ``exl3_moe_loop``.
        """
        flat = topk_ids.reshape(-1)
        routed = flat[flat >= 0]  # record_routes requires ids in [0, E); -1 marks a dropped route
        if not capturing_graphs():
            # Warmup and capture forwards route dummy tokens; they must not move residency.
            streamer.record_routes(routed)
        if route_plan:
            # SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN: one readback before any gather of this layer, then none
            # between a chunk's gather and its compute, so the host queues the compute behind the gather.
            plan = Exl3RoutePlan.from_topk(topk_ids)
            source_ids = plan.source_ids
        else:
            source_ids = torch.unique(routed)
        out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
        gathered = False
        # A no-op unless SGLANG_DSV41_ENABLE_PREFILL_FILLS gave the pinned tier native fills.
        with streamer.prefill_fills(source_ids):
            if route_plan:
                for experts, row_of_source, rows in streamer.iter_gather_experts_host(source_ids, plan.experts):
                    gathered = True
                    copied = torch.cuda.Event() if x.is_cuda else None
                    if copied is not None:
                        copied.record()
                    w13, w2 = EXL3_ROW_VIEWS.select(rows, experts, row_of_source)
                    exl3_moe_accumulate_planned(out, x, topk_weights, plan, w13, w2, swiglu_limit, experts)
                    if copied is not None:
                        # The gather reads pinned slabs the RAM-miss thread may reuse once the host use ends: wait
                        # for it (not for the compute just queued) before the next chunk or the host use can end.
                        copied.synchronize()
            else:
                for chunk, row_of_source, rows in streamer.iter_gather_experts(source_ids):
                    gathered = True
                    experts = chunk.tolist()
                    w13, w2 = EXL3_ROW_VIEWS.select(rows, experts, row_of_source.tolist())
                    exl3_moe_accumulate(out, x, topk_weights, topk_ids, w13, w2, swiglu_limit, experts)
        get_exl3_stream_trace().record(
            layer.layer_id,
            topk_ids,
            streamer.last_gather_stats if gathered else None,
            streamer.background_read_stats.rows,
        )
        return out.to(x.dtype)
