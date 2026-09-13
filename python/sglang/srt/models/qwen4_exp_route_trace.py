"""Debug-only MoE routing trace for Qwen4-Exp decode forwards.

Setting ``SGLANG_MOE_ROUTE_TRACE_DIR`` makes :func:`maybe_install_moe_route_trace`
attach module hooks (and wrap two hyper-connection ``mix`` calls) on the language
model. When the variable is unset it returns before touching the model, so the
serving path carries no hook, wrapper or extra branch. Tracing reads GPU tensors
on the host every decode step and is only valid for eager decode; startup
validation lives in ``arg_groups/memory_hook.py``.

Per decode forward and per MoE layer ``l`` the shards hold, with ``N`` traced
tokens, ``L`` MoE layers, ``H`` hidden size, ``C`` hyper-connection streams,
``E`` routed experts and ``K`` experts per token:

``router_input`` [N, L, H]
    The exact tensor passed to ``mlp.gate``: ``mlp_hyper_connection.mix`` of
    ``hc_mid``.
``hc_mid`` [N, L, C*H]
    The C-stream hyper-connection state after the attention block was injected
    (``attn_hyper_connection.combine``) and before the MoE block; the analogue of
    the post-attention residual ``r_l``.
``router_logits`` [N, L, E]
    ``mlp.gate`` output.
``topk_ids`` [N, L, K] and ``topk_weights`` [N, L, K]
    ``mlp.topk`` output: selected logical expert ids and final routing weights.
``routed_out`` [N, L, H]
    ``mlp.experts`` output, cloned before the shared expert is added in place.
``moe_out`` [N, L, H]
    The full MoE block output (routed plus gated shared expert) that
    ``mlp_hyper_connection.combine`` injects into ``hc_mid``.

Per forward: ``final_hc`` [N, C*H] (the stream state after the last layer, which
the MTP head consumes), ``final_hidden`` [N, H] (``hyper_connection_mixer.mix``
output fed to the LM head), and ``input_ids``, ``positions``, ``seq_lens``,
``req_pool_indices``, ``forward_index`` [N]. A decode forward's ``input_ids``
are the tokens sampled by the previous decode forward of the same request.
"""

import atexit
import contextlib
import json
import logging
import os
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)

TRACE_FORMAT = "sglang-qwen4-exp-moe-route-trace-v1"
MANIFEST_NAME = "manifest.json"
DEFAULT_SHARD_TOKENS = 128

LAYER_FIELDS: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "router_input": torch.bfloat16,
        "hc_mid": torch.bfloat16,
        "router_logits": torch.bfloat16,
        "topk_ids": torch.int16,
        "topk_weights": torch.float32,
        "routed_out": torch.bfloat16,
        "moe_out": torch.bfloat16,
    }
)
FORWARD_FIELDS: Mapping[str, torch.dtype] = MappingProxyType(
    {"final_hc": torch.bfloat16, "final_hidden": torch.bfloat16}
)
TOKEN_FIELDS = ("input_ids", "positions", "seq_lens", "req_pool_indices", "forward_index")


def estimate_bytes_per_token(
    num_layers: int, hidden_size: int, hc_count: int, num_experts: int, top_k: int
) -> int:
    """Shard bytes one traced token costs, excluding container overhead."""
    per_layer = (
        2 * hidden_size
        + 2 * hc_count * hidden_size
        + 2 * num_experts
        + 2 * top_k
        + 4 * top_k
        + 2 * hidden_size
        + 2 * hidden_size
    )
    per_forward = 2 * hc_count * hidden_size + 2 * hidden_size + 8 * len(TOKEN_FIELDS)
    return num_layers * per_layer + per_forward


def _token_vector(value: Optional[torch.Tensor], count: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        return torch.full((count,), -1, dtype=torch.int64)
    return value[:count].detach().to(device="cpu", dtype=torch.int64)


class MoeRouteTraceWriter:
    """Buffers per-layer routing tensors of decode forwards and writes CPU shards.

    Hooks call :meth:`record_layer` / :meth:`record_forward` only while
    :attr:`recording` is true, i.e. between :meth:`begin_forward` of a decode
    forward and its :meth:`end_forward`. After ``max_tokens`` tokens the writer
    flushes, marks the manifest complete and calls ``on_complete``.
    """

    def __init__(
        self,
        directory: str,
        max_tokens: int,
        layer_ids: Sequence[int],
        metadata: Mapping[str, Any],
        shard_tokens: int = DEFAULT_SHARD_TOKENS,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("route trace max_tokens must be positive")
        if shard_tokens < 1:
            raise ValueError("route trace shard_tokens must be positive")
        if not layer_ids:
            raise ValueError("route trace needs at least one MoE layer")
        if os.path.isdir(directory) and os.listdir(directory):
            raise FileExistsError(
                f"{directory} is not empty; choose an empty directory for the route trace"
            )
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.max_tokens = int(max_tokens)
        self.shard_tokens = int(shard_tokens)
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.metadata = dict(metadata)
        self.on_complete = on_complete
        self.recording = False
        self.closed = False
        self.tokens = 0
        self.forwards = 0
        self._buffer: Dict[str, List[torch.Tensor]] = {}
        self._buffered_tokens = 0
        self._shards: List[Dict[str, Any]] = []
        self._token_values: Dict[str, Optional[torch.Tensor]] = {}
        self._layer_values: Dict[str, Dict[int, torch.Tensor]] = {}
        self._forward_values: Dict[str, torch.Tensor] = {}
        self._write_manifest(complete=False)

    def begin_forward(self, forward_batch: Any) -> None:
        """Start recording if ``forward_batch`` is a non-empty decode forward."""
        self.recording = False
        if self.closed or forward_batch is None:
            return
        if not forward_batch.forward_mode.is_decode():
            return
        input_ids = forward_batch.input_ids
        if not isinstance(input_ids, torch.Tensor) or input_ids.numel() == 0:
            return
        self._token_values = {
            "input_ids": input_ids.detach().clone(),
            "positions": getattr(forward_batch, "positions", None),
            "seq_lens": getattr(forward_batch, "seq_lens", None),
            "req_pool_indices": getattr(forward_batch, "req_pool_indices", None),
        }
        self._layer_values = {field: {} for field in LAYER_FIELDS}
        self._forward_values = {}
        self.recording = True

    def record_layer(self, field: str, layer_id: int, tensor: torch.Tensor) -> None:
        self._layer_values[field][layer_id] = tensor.detach().clone()

    def record_forward(self, field: str, tensor: torch.Tensor) -> None:
        self._forward_values[field] = tensor.detach().clone()

    def end_forward(self) -> None:
        """Move the recorded forward to the host buffer; flush or close as due."""
        if not self.recording:
            return
        self.recording = False
        missing = [
            f"{field}@{layer_id}"
            for field, values in self._layer_values.items()
            for layer_id in self.layer_ids
            if layer_id not in values
        ] + [field for field in FORWARD_FIELDS if field not in self._forward_values]
        if missing:
            logger.warning(
                "MoE route trace dropped a decode forward missing %s", missing[:8]
            )
            return

        token_count = int(self._token_values["input_ids"].numel())
        take = min(token_count, self.max_tokens - self.tokens)
        record: Dict[str, torch.Tensor] = {}
        for field, dtype in LAYER_FIELDS.items():
            values = self._layer_values[field]
            stacked = torch.stack([values[layer_id] for layer_id in self.layer_ids], 1)
            record[field] = stacked[:take].to(dtype=dtype).cpu()
        for field, dtype in FORWARD_FIELDS.items():
            record[field] = self._forward_values[field][:take].to(dtype=dtype).cpu()
        for field in ("input_ids", "positions", "seq_lens", "req_pool_indices"):
            record[field] = _token_vector(self._token_values[field], take)
        record["forward_index"] = torch.full((take,), self.forwards, dtype=torch.int64)
        for field, tensor in record.items():
            if tensor.shape[0] != take:
                raise RuntimeError(
                    f"route trace field {field} has {tensor.shape[0]} rows, "
                    f"expected {take}"
                )
            self._buffer.setdefault(field, []).append(tensor)

        self._layer_values = {}
        self._forward_values = {}
        self._token_values = {}
        self.forwards += 1
        self.tokens += take
        self._buffered_tokens += take
        if self._buffered_tokens >= self.shard_tokens:
            try:
                self.flush()
            except OSError as error:
                self._abandon(error)
                return
        if self.tokens >= self.max_tokens:
            self.close()

    def flush(self) -> None:
        """Write buffered tokens as one shard and refresh the manifest."""
        if not self._buffered_tokens:
            return
        name = f"shard_{len(self._shards):05d}.pt"
        shard = {field: torch.cat(parts) for field, parts in self._buffer.items()}
        path = os.path.join(self.directory, name)
        torch.save(shard, path + ".tmp")
        os.replace(path + ".tmp", path)
        self._shards.append({"file": name, "tokens": self._buffered_tokens})
        self._buffer = {}
        self._buffered_tokens = 0
        self._write_manifest(complete=False)

    def close(self) -> None:
        """Flush, mark the trace complete and stop recording for good."""
        if self.closed:
            return
        self.recording = False
        try:
            self.flush()
            self.closed = True
            self._write_manifest(complete=True)
        except OSError as error:
            self._abandon(error)
            return
        logger.warning(
            "MoE route trace complete: %d tokens in %d shards under %s",
            self.tokens,
            len(self._shards),
            self.directory,
        )
        if self.on_complete is not None:
            self.on_complete()

    def _abandon(self, error: OSError) -> None:
        """Stop tracing after a failed write so serving continues untraced."""
        logger.error(
            "MoE route trace stopped after a write failure under %s: %s",
            self.directory,
            error,
        )
        self.recording = False
        self.closed = True
        self._buffer = {}
        self._buffered_tokens = 0
        with contextlib.suppress(OSError):
            for name in os.listdir(self.directory):
                if name.endswith(".tmp"):
                    os.remove(os.path.join(self.directory, name))
        with contextlib.suppress(OSError):
            self._write_manifest(complete=False, error=str(error))
        if self.on_complete is not None:
            self.on_complete()

    def _write_manifest(self, complete: bool, error: Optional[str] = None) -> None:
        manifest = {
            "format": TRACE_FORMAT,
            "complete": complete,
            "error": error,
            "tokens": self.tokens,
            "forwards": self.forwards,
            "max_tokens": self.max_tokens,
            "layer_ids": list(self.layer_ids),
            "shards": list(self._shards),
            "layer_fields": {name: str(dtype) for name, dtype in LAYER_FIELDS.items()},
            "forward_fields": {
                name: str(dtype) for name, dtype in FORWARD_FIELDS.items()
            },
            "token_fields": list(TOKEN_FIELDS),
            **self.metadata,
        }
        path = os.path.join(self.directory, MANIFEST_NAME)
        with open(path + ".tmp", "w") as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(path + ".tmp", path)


class MoeRouteTrace:
    """Owns the hooks installed on one language model and their writer."""

    def __init__(self) -> None:
        self.writer: Optional[MoeRouteTraceWriter] = None
        self._handles: List[Any] = []
        self._wrapped_mix: List[nn.Module] = []

    def remove(self) -> None:
        """Detach every hook and restore the wrapped ``mix`` methods."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        for owner in self._wrapped_mix:
            owner.__dict__.pop("mix", None)
        self._wrapped_mix = []

    def wrap_mix(
        self,
        owner: nn.Module,
        on_input: Callable[[torch.Tensor], None],
        on_output: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> None:
        original = owner.mix
        writer = self.writer

        def mix(hyper_input: torch.Tensor):
            if writer.recording:
                on_input(hyper_input)
            result = original(hyper_input)
            if on_output is not None and writer.recording:
                on_output(result[0])
            return result

        owner.__dict__["mix"] = mix
        self._wrapped_mix.append(owner)


def _moe_layers(language_model: nn.Module) -> List[tuple]:
    layers = []
    for layer_id, layer in enumerate(language_model.layers):
        mlp = getattr(layer, "mlp", None)
        if (
            hasattr(layer, "mlp_hyper_connection")
            and hasattr(mlp, "gate")
            and hasattr(mlp, "topk")
            and hasattr(mlp, "experts")
        ):
            layers.append((layer_id, layer))
    return layers


def install_moe_route_trace(
    language_model: nn.Module,
    directory: str,
    max_tokens: int,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
) -> MoeRouteTrace:
    """Attach the route trace to a Qwen4-Exp language model (``Qwen4ExpModel``)."""
    layers = _moe_layers(language_model)
    if not layers:
        raise ValueError("MoE route trace found no hyper-connection MoE layers")
    first = layers[0][1]
    hidden_size = int(first.hidden_size)
    hc_count = int(first.hc_count)
    num_experts = int(first.mlp.num_experts)
    top_k = int(first.mlp.topk.topk_config.top_k)
    metadata = {
        "hidden_size": hidden_size,
        "hc_count": hc_count,
        "num_experts": num_experts,
        "top_k": top_k,
        "rms_norm_eps": float(first.mlp_hyper_connection.config.rms_norm_eps),
        "bytes_per_token_estimate": estimate_bytes_per_token(
            len(layers), hidden_size, hc_count, num_experts, top_k
        ),
    }

    trace = MoeRouteTrace()
    writer = MoeRouteTraceWriter(
        directory,
        max_tokens,
        [layer_id for layer_id, _ in layers],
        metadata,
        shard_tokens=shard_tokens,
        on_complete=trace.remove,
    )
    trace.writer = writer

    def before_forward(module, args, kwargs):
        forward_batch = kwargs.get("forward_batch")
        if forward_batch is None and len(args) > 2:
            forward_batch = args[2]
        writer.begin_forward(forward_batch)

    def after_forward(module, args, output):
        writer.end_forward()

    trace._handles.append(
        language_model.register_forward_pre_hook(before_forward, with_kwargs=True)
    )
    trace._handles.append(language_model.register_forward_hook(after_forward))

    for layer_id, layer in layers:
        mlp = layer.mlp
        trace._handles.extend(
            [
                mlp.gate.register_forward_hook(_gate_hook(writer, layer_id)),
                mlp.topk.register_forward_hook(_topk_hook(writer, layer_id)),
                mlp.experts.register_forward_hook(
                    _output_hook(writer, "routed_out", layer_id)
                ),
                mlp.register_forward_hook(_output_hook(writer, "moe_out", layer_id)),
            ]
        )
        trace.wrap_mix(
            layer.mlp_hyper_connection,
            lambda hyper_input, layer_id=layer_id: writer.record_layer(
                "hc_mid", layer_id, hyper_input
            ),
        )

    trace.wrap_mix(
        language_model.hyper_connection_mixer,
        lambda hyper_input: writer.record_forward("final_hc", hyper_input),
        lambda mixed: writer.record_forward("final_hidden", mixed),
    )
    atexit.register(writer.close)
    logger.warning(
        "MoE route trace enabled: %d MoE layers, up to %d decode tokens "
        "(~%.2f MiB each) under %s",
        len(layers),
        max_tokens,
        metadata["bytes_per_token_estimate"] / 2**20,
        directory,
    )
    return trace


def _gate_hook(writer: MoeRouteTraceWriter, layer_id: int):
    def hook(module, args, output):
        if not writer.recording:
            return
        logits = output[0] if isinstance(output, tuple) else output
        writer.record_layer("router_input", layer_id, args[0])
        writer.record_layer("router_logits", layer_id, logits)

    return hook


def _topk_hook(writer: MoeRouteTraceWriter, layer_id: int):
    def hook(module, args, output):
        if not writer.recording:
            return
        topk_ids = getattr(output, "topk_ids", None)
        topk_weights = getattr(output, "topk_weights", None)
        if topk_ids is None or topk_weights is None:
            raise RuntimeError(
                "MoE route trace needs a standard TopK output with topk_ids and "
                f"topk_weights, got {type(output).__name__}"
            )
        writer.record_layer("topk_ids", layer_id, topk_ids)
        writer.record_layer("topk_weights", layer_id, topk_weights)

    return hook


def _output_hook(writer: MoeRouteTraceWriter, field: str, layer_id: int):
    def hook(module, args, output):
        if not writer.recording:
            return
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"MoE route trace expected a tensor for {field}, "
                f"got {type(output).__name__}"
            )
        writer.record_layer(field, layer_id, output)

    return hook


def maybe_install_moe_route_trace(language_model: nn.Module) -> Optional[MoeRouteTrace]:
    """Install the route trace when ``SGLANG_MOE_ROUTE_TRACE_DIR`` is set."""
    from sglang.srt.environ import envs

    directory = envs.SGLANG_MOE_ROUTE_TRACE_DIR.get()
    if not directory:
        return None
    return install_moe_route_trace(
        language_model, directory, envs.SGLANG_MOE_ROUTE_TRACE_MAX_TOKENS.get()
    )
