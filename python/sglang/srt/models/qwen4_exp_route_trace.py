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

Speculative mode
----------------
With ``SGLANG_MOE_ROUTE_TRACE_SPECULATIVE`` (NEXTN, eager target and draft) the
target trace records every token of each target verify forward instead, keeping
only ``topk_ids`` (int32), ``topk_weights``, ``final_hidden`` and the token
fields ``input_ids``, ``positions``, ``req_pool_indices``, ``forward_index``,
``sequence``; the manifest carries ``mode: "speculative"``.

The MTP draft model writes a second trace under ``<dir>/mtp/``
(:func:`maybe_install_mtp_hidden_trace`) holding ``hidden`` [N, H], the tensor
its logits processor receives, for EXTEND, DRAFT_EXTEND_V2 and DECODE draft
forwards, plus ``input_ids``, ``positions``, ``req_pool_indices``,
``forward_index``, ``forward_mode`` (``ForwardMode`` value; names in the
manifest's ``forward_modes``), ``sequence``, ``accept_len`` (int32) and
``selected`` (int8). See :func:`_mtp_rows` for which rows are kept and how
``accept_len`` / ``selected`` are derived. The MTP trace closes when the target
trace ends, or after ``4 * SGLANG_MOE_ROUTE_TRACE_MAX_TOKENS`` rows; its manifest
names the cause in ``closed_by``.

``sequence`` is one process-wide counter shared by both traces and taken once
per recorded forward, so sorting by it restores the interleaving of target and
draft forwards. When a trace reaches its cap mid-forward it keeps the first rows
of that forward in every field, which can cut a request's chain short.
"""

import atexit
import contextlib
import itertools
import json
import logging
import os
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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

SPECULATIVE_MODE = "speculative"
SPECULATIVE_LAYER_FIELDS: Mapping[str, torch.dtype] = MappingProxyType(
    {"topk_ids": torch.int32, "topk_weights": torch.float32}
)
SPECULATIVE_FORWARD_FIELDS: Mapping[str, torch.dtype] = MappingProxyType(
    {"final_hidden": torch.bfloat16}
)
SPECULATIVE_TOKEN_FIELDS = (
    "input_ids",
    "positions",
    "req_pool_indices",
    "forward_index",
    "sequence",
)
SPECULATIVE_SETTINGS = (
    "speculative_num_steps",
    "speculative_num_draft_tokens",
    "speculative_eagle_topk",
)

MTP_TRACE_FORMAT = "sglang-qwen4-exp-mtp-hidden-trace-v1"
MTP_MODE = "speculative_mtp"
MTP_SUBDIR = "mtp"
MTP_MAX_TOKENS_FACTOR = 4
MTP_RECORDED_MODES = ("EXTEND", "DRAFT_EXTEND_V2", "DECODE")
MTP_FORWARD_FIELDS: Mapping[str, torch.dtype] = MappingProxyType(
    {"hidden": torch.bfloat16}
)
MTP_TOKEN_FIELDS = (
    "input_ids",
    "positions",
    "req_pool_indices",
    "forward_index",
    "forward_mode",
    "sequence",
    "accept_len",
    "selected",
)

RowSelection = Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]

_SEQUENCE = itertools.count()


def _is_decode(forward_mode: Any) -> bool:
    return forward_mode.is_decode()


def _is_target_verify(forward_mode: Any) -> bool:
    return forward_mode.is_target_verify()


def _is_mtp_recorded(forward_mode: Any) -> bool:
    return getattr(forward_mode, "name", None) in MTP_RECORDED_MODES


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


def _request_token_counts(
    requests: int, token_count: int, extend_seq_lens: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Tokens each request contributes to a forward, or None when unknowable."""
    if isinstance(extend_seq_lens, torch.Tensor) and extend_seq_lens.numel() == requests:
        counts = extend_seq_lens.detach().to(device="cpu", dtype=torch.int64)
        if int(counts.sum()) == token_count:
            return counts
    if token_count % requests == 0:
        return torch.full((requests,), token_count // requests, dtype=torch.int64)
    return None


def _token_vector(
    value: Optional[torch.Tensor],
    count: int,
    token_count: Optional[int] = None,
    extend_seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """First ``count`` per-token values; a per-request vector is expanded per token."""
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        return torch.full((count,), -1, dtype=torch.int64)
    if token_count is not None and 0 < value.numel() < token_count:
        repeats = _request_token_counts(value.numel(), token_count, extend_seq_lens)
        if repeats is None:
            return torch.full((count,), -1, dtype=torch.int64)
        value = torch.repeat_interleave(
            value.detach().to(device="cpu", dtype=torch.int64), repeats
        )
    return value[:count].detach().to(device="cpu", dtype=torch.int64)


def _take_rows(
    tensor: torch.Tensor, take: int, index: Optional[torch.Tensor]
) -> torch.Tensor:
    if index is None:
        return tensor[:take]
    return tensor[index.to(device=tensor.device)]


def _mtp_rows(forward_batch: Any, token_count: int) -> Optional[RowSelection]:
    """Rows of a draft forward the MTP trace keeps, with ``accept_len`` and ``selected``.

    DECODE keeps every row. EXTEND (the draft pass over a prefill) keeps only the
    last row of each request, ``cumsum(extend_seq_lens) - 1``, which seeds draft
    step 0; prompt rows are never written and never count against the cap.
    DRAFT_EXTEND_V2 keeps the whole window and marks with ``selected = 1`` the row
    ``i * window + num_front_tokens + num_accept_tokens[i] - 1`` that seeds the
    next chain, mirroring the worker's ``select_index``; ``accept_len`` is
    ``spec_info.num_accept_tokens`` (bonus token included) per token.

    ``accept_len`` is -1 for DECODE and EXTEND rows, which carry ``selected = 1``.
    A DRAFT_EXTEND_V2 forward without accept counts writes -1 in both fields.
    Returns None when rows cannot be attributed to requests.
    """
    mode = forward_batch.forward_mode.name
    unknown_accept = torch.full((token_count,), -1, dtype=torch.int32)
    if mode == "DECODE":
        return None, {
            "accept_len": unknown_accept,
            "selected": torch.ones(token_count, dtype=torch.int8),
        }
    request_indices = getattr(forward_batch, "req_pool_indices", None)
    requests = (
        request_indices.numel() if isinstance(request_indices, torch.Tensor) else 0
    )
    if requests == 0:
        return None
    if mode == "EXTEND":
        counts = _request_token_counts(
            requests, token_count, getattr(forward_batch, "extend_seq_lens", None)
        )
        if counts is None:
            return None
        return torch.cumsum(counts, 0) - 1, {
            "accept_len": unknown_accept,
            "selected": torch.ones(token_count, dtype=torch.int8),
        }
    if token_count % requests:
        return None
    window = token_count // requests
    spec_info = getattr(forward_batch, "spec_info", None)
    accept = getattr(spec_info, "num_accept_tokens", None)
    if not isinstance(accept, torch.Tensor) or accept.numel() != requests:
        return None, {
            "accept_len": unknown_accept,
            "selected": torch.full((token_count,), -1, dtype=torch.int8),
        }
    accept = accept.detach().to(device="cpu", dtype=torch.int64)
    front = int(getattr(spec_info, "num_front_tokens", 0) or 0)
    selected = torch.zeros(token_count, dtype=torch.int8)
    selected[torch.arange(requests) * window + front + accept - 1] = 1
    return None, {
        "accept_len": accept.repeat_interleave(window).to(torch.int32),
        "selected": selected,
    }


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
        *,
        layer_fields: Mapping[str, torch.dtype] = LAYER_FIELDS,
        forward_fields: Mapping[str, torch.dtype] = FORWARD_FIELDS,
        token_fields: Sequence[str] = TOKEN_FIELDS,
        records: Callable[[Any], bool] = _is_decode,
        select_rows: Optional[Callable[[Any, int], Optional[RowSelection]]] = None,
        trace_format: str = TRACE_FORMAT,
        shared_entries: Sequence[str] = (),
        record_closed_by: bool = False,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("route trace max_tokens must be positive")
        if shard_tokens < 1:
            raise ValueError("route trace shard_tokens must be positive")
        if layer_fields and not layer_ids:
            raise ValueError("route trace needs at least one MoE layer")
        if os.path.isdir(directory) and [
            name for name in os.listdir(directory) if name not in shared_entries
        ]:
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
        self.layer_fields = layer_fields
        self.forward_fields = forward_fields
        self.token_fields = tuple(token_fields)
        self.records = records
        self.select_rows = select_rows
        self.trace_format = trace_format
        self.record_closed_by = record_closed_by
        self.recording = False
        self.closed = False
        self.closed_by: Optional[str] = None
        self.error: Optional[str] = None
        self.tokens = 0
        self.forwards = 0
        self._buffer: Dict[str, List[torch.Tensor]] = {}
        self._buffered_tokens = 0
        self._shards: List[Dict[str, Any]] = []
        self._token_values: Dict[str, Optional[torch.Tensor]] = {}
        self._layer_values: Dict[str, Dict[int, torch.Tensor]] = {}
        self._forward_values: Dict[str, torch.Tensor] = {}
        self._rows: Optional[torch.Tensor] = None
        self._row_values: Dict[str, torch.Tensor] = {}
        self._sequence = -1
        self._forward_mode_value = -1
        self._forward_mode_names: Dict[int, str] = {}
        self._warned_drop = False
        self._write_manifest(complete=False)

    def begin_forward(self, forward_batch: Any) -> None:
        """Start recording if ``forward_batch`` is a non-empty forward of a traced mode."""
        self.recording = False
        if self.closed or forward_batch is None:
            return
        forward_mode = forward_batch.forward_mode
        if not self.records(forward_mode):
            return
        input_ids = forward_batch.input_ids
        if not isinstance(input_ids, torch.Tensor) or input_ids.numel() == 0:
            return
        self._rows, self._row_values = None, {}
        if self.select_rows is not None:
            selection = self.select_rows(forward_batch, int(input_ids.numel()))
            if selection is None:
                self.drop_forward("its rows cannot be attributed to requests")
                return
            self._rows, self._row_values = selection
        self._token_values = {
            "input_ids": input_ids.detach().clone(),
            "positions": getattr(forward_batch, "positions", None),
            "seq_lens": getattr(forward_batch, "seq_lens", None),
            "req_pool_indices": getattr(forward_batch, "req_pool_indices", None),
            "extend_seq_lens": getattr(forward_batch, "extend_seq_lens", None),
        }
        if "sequence" in self.token_fields:
            self._sequence = next(_SEQUENCE)
        self._forward_mode_value = int(getattr(forward_mode, "value", -1))
        self._forward_mode_names[self._forward_mode_value] = str(
            getattr(forward_mode, "name", self._forward_mode_value)
        )
        self._layer_values = {field: {} for field in self.layer_fields}
        self._forward_values = {}
        self.recording = True

    def record_layer(self, field: str, layer_id: int, tensor: torch.Tensor) -> None:
        self._layer_values[field][layer_id] = tensor.detach().clone()

    def record_forward(self, field: str, tensor: torch.Tensor) -> None:
        self._forward_values[field] = tensor.detach().clone()

    def drop_forward(self, reason: str) -> None:
        """Discard the forward being recorded; only the first drop is logged."""
        self.recording = False
        self._layer_values = {}
        self._forward_values = {}
        self._token_values = {}
        self._rows, self._row_values = None, {}
        if not self._warned_drop:
            self._warned_drop = True
            logger.warning(
                "Route trace under %s dropped a forward because %s; later drops "
                "are not logged",
                self.directory,
                reason,
            )

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
        ] + [field for field in self.forward_fields if field not in self._forward_values]
        if missing:
            logger.warning(
                "MoE route trace dropped a decode forward missing %s", missing[:8]
            )
            return

        token_count = int(self._token_values["input_ids"].numel())
        rows = self._rows
        row_count = token_count if rows is None else int(rows.numel())
        take = min(row_count, self.max_tokens - self.tokens)
        index = None if rows is None else rows[:take]
        record: Dict[str, torch.Tensor] = {}
        for field, dtype in self.layer_fields.items():
            values = self._layer_values[field]
            stacked = torch.stack([values[layer_id] for layer_id in self.layer_ids], 1)
            record[field] = _take_rows(stacked, take, index).to(dtype=dtype).cpu()
        for field, dtype in self.forward_fields.items():
            record[field] = (
                _take_rows(self._forward_values[field], take, index)
                .to(dtype=dtype)
                .cpu()
            )
        for field in self.token_fields:
            if field in self._row_values:
                record[field] = _take_rows(self._row_values[field], take, index).cpu()
                continue
            if field == "forward_index":
                value = self.forwards
            elif field == "forward_mode":
                value = self._forward_mode_value
            elif field == "sequence":
                value = self._sequence
            elif index is None:
                record[field] = _token_vector(
                    self._token_values[field],
                    take,
                    token_count,
                    self._token_values["extend_seq_lens"],
                )
                continue
            else:
                record[field] = _token_vector(
                    self._token_values[field],
                    token_count,
                    token_count,
                    self._token_values["extend_seq_lens"],
                )[index]
                continue
            record[field] = torch.full((take,), value, dtype=torch.int64)
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
        self._rows, self._row_values = None, {}
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
            self.close("cap")

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

    def close(self, reason: Optional[str] = None) -> None:
        """Flush, mark the trace complete and stop recording for good.

        ``reason`` lands in the manifest's ``closed_by`` when the writer records it.
        """
        if self.closed:
            return
        self.recording = False
        self.closed_by = reason
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
        self.closed_by = "abandoned"
        self.error = str(error)
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
            "format": self.trace_format,
            "complete": complete,
            "error": error,
            "tokens": self.tokens,
            "forwards": self.forwards,
            "max_tokens": self.max_tokens,
            "layer_ids": list(self.layer_ids),
            "shards": list(self._shards),
            "layer_fields": {
                name: str(dtype) for name, dtype in self.layer_fields.items()
            },
            "forward_fields": {
                name: str(dtype) for name, dtype in self.forward_fields.items()
            },
            "token_fields": list(self.token_fields),
            **self.metadata,
        }
        if "forward_mode" in self.token_fields:
            manifest["forward_modes"] = {
                name: value for value, name in sorted(self._forward_mode_names.items())
            }
        if self.record_closed_by:
            manifest["closed_by"] = self.closed_by
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
        on_input: Optional[Callable[[torch.Tensor], None]],
        on_output: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> None:
        original = owner.mix
        writer = self.writer

        def mix(hyper_input: torch.Tensor):
            if on_input is not None and writer.recording:
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
    speculative: bool = False,
    speculative_settings: Optional[Mapping[str, Optional[int]]] = None,
) -> MoeRouteTrace:
    """Attach the route trace to a Qwen4-Exp language model (``Qwen4ExpModel``).

    ``speculative`` traces target verify forwards with the reduced field set,
    records ``speculative_settings`` (see :data:`SPECULATIVE_SETTINGS`) in the
    manifest, and closes the sibling MTP trace (see
    :func:`install_mtp_hidden_trace`) when done.
    """
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
    if speculative:
        link = _speculative_link(directory)
        metadata["mode"] = SPECULATIVE_MODE
        metadata["bytes_per_token_estimate"] = (
            len(layers) * 8 * top_k
            + 2 * hidden_size
            + 8 * len(SPECULATIVE_TOKEN_FIELDS)
        )
        metadata.update(_settings_metadata(speculative_settings))

        def finish() -> None:
            trace.remove()
            link.finish_target(
                "target_abandoned" if writer.error is not None else "target_complete"
            )

        writer = MoeRouteTraceWriter(
            directory,
            max_tokens,
            [layer_id for layer_id, _ in layers],
            metadata,
            shard_tokens=shard_tokens,
            on_complete=finish,
            layer_fields=SPECULATIVE_LAYER_FIELDS,
            forward_fields=SPECULATIVE_FORWARD_FIELDS,
            token_fields=SPECULATIVE_TOKEN_FIELDS,
            records=_is_target_verify,
            shared_entries=(MTP_SUBDIR,),
        )
    else:
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

    if speculative:
        for layer_id, layer in layers:
            trace._handles.append(
                layer.mlp.topk.register_forward_hook(_topk_hook(writer, layer_id))
            )
        trace.wrap_mix(
            language_model.hyper_connection_mixer,
            None,
            lambda mixed: writer.record_forward("final_hidden", mixed),
        )
        atexit.register(writer.close)
        logger.warning(
            "MoE route trace enabled for speculative verify: %d MoE layers, up to "
            "%d target tokens (~%.1f KiB each) under %s",
            len(layers),
            max_tokens,
            metadata["bytes_per_token_estimate"] / 2**10,
            directory,
        )
        return trace

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
    if envs.SGLANG_MOE_ROUTE_TRACE_SPECULATIVE.get():
        return install_moe_route_trace(
            language_model,
            directory,
            envs.SGLANG_MOE_ROUTE_TRACE_MAX_TOKENS.get(),
            speculative=True,
            speculative_settings=_published_speculative_settings(),
        )
    return install_moe_route_trace(
        language_model, directory, envs.SGLANG_MOE_ROUTE_TRACE_MAX_TOKENS.get()
    )


def _settings_metadata(
    settings: Optional[Mapping[str, Optional[int]]],
) -> Dict[str, Optional[int]]:
    settings = settings or {}
    return {name: settings.get(name) for name in SPECULATIVE_SETTINGS}


def _published_speculative_settings() -> Dict[str, Optional[int]]:
    """Speculative sizes from the published ``spec`` config; None where unpublished."""
    from sglang.srt.runtime_context import get_spec

    try:
        spec = get_spec()
    except ValueError:
        return dict.fromkeys(SPECULATIVE_SETTINGS)
    return {name: getattr(spec, name, None) for name in SPECULATIVE_SETTINGS}


class _SpeculativeTraceLink:
    """Shares completion between a target verify trace and its MTP draft trace.

    The two traces are installed by separately constructed models, so they meet
    through a link keyed by the trace directory and the process. The MTP writers
    close when the target trace ends, whichever model was constructed first; the
    link is then retired so a later install on the same directory starts fresh.
    """

    def __init__(self, key: Tuple[str, int]) -> None:
        self.key = key
        self.mtp_writers: List[MoeRouteTraceWriter] = []

    def finish_target(self, reason: str) -> None:
        if _SPECULATIVE_LINKS.get(self.key) is self:
            del _SPECULATIVE_LINKS[self.key]
        for writer in self.mtp_writers:
            writer.close(reason)


_SPECULATIVE_LINKS: Dict[Tuple[str, int], _SpeculativeTraceLink] = {}


def _speculative_link(directory: str) -> _SpeculativeTraceLink:
    key = (os.path.realpath(directory), os.getpid())
    if key not in _SPECULATIVE_LINKS:
        _SPECULATIVE_LINKS[key] = _SpeculativeTraceLink(key)
    return _SPECULATIVE_LINKS[key]


def install_mtp_hidden_trace(
    mtp_model: nn.Module,
    directory: str,
    max_tokens: int,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    speculative_settings: Optional[Mapping[str, Optional[int]]] = None,
) -> MoeRouteTrace:
    """Record the hidden state MTP draft forwards hand their LM head.

    Hooks ``mtp_model.logits_processor``, whose second positional argument is the
    LM-head input, and writes up to ``MTP_MAX_TOKENS_FACTOR * max_tokens`` rows
    under ``<directory>/mtp``; :func:`_mtp_rows` picks the rows. A forward whose
    hidden state does not have one row per input token is dropped.
    """
    trace = MoeRouteTrace()
    writer = MoeRouteTraceWriter(
        os.path.join(directory, MTP_SUBDIR),
        MTP_MAX_TOKENS_FACTOR * max_tokens,
        (),
        {
            "mode": MTP_MODE,
            "hidden_size": int(getattr(mtp_model, "hidden_size", -1)),
            "target_max_tokens": int(max_tokens),
            **_settings_metadata(speculative_settings),
        },
        shard_tokens=shard_tokens,
        on_complete=trace.remove,
        layer_fields={},
        forward_fields=MTP_FORWARD_FIELDS,
        token_fields=MTP_TOKEN_FIELDS,
        records=_is_mtp_recorded,
        select_rows=_mtp_rows,
        trace_format=MTP_TRACE_FORMAT,
        record_closed_by=True,
    )
    trace.writer = writer

    def before_logits(module, args, kwargs):
        hidden_states = args[1] if len(args) > 1 else kwargs.get("hidden_states")
        forward_batch = args[3] if len(args) > 3 else kwargs.get("logits_metadata")
        writer.begin_forward(forward_batch)
        if not writer.recording:
            return
        rows = int(forward_batch.input_ids.numel())
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.shape[0] != rows:
            writer.drop_forward(
                f"its LM-head input {tuple(getattr(hidden_states, 'shape', ()))} "
                f"does not have {rows} rows"
            )
            return
        writer.record_forward("hidden", hidden_states)
        writer.end_forward()

    trace._handles.append(
        mtp_model.logits_processor.register_forward_pre_hook(
            before_logits, with_kwargs=True
        )
    )
    atexit.register(writer.close, "atexit")
    _speculative_link(directory).mtp_writers.append(writer)
    logger.warning(
        "MTP hidden trace enabled: up to %d draft rows under %s",
        writer.max_tokens,
        writer.directory,
    )
    return trace


def maybe_install_mtp_hidden_trace(mtp_model: nn.Module) -> Optional[MoeRouteTrace]:
    """Install the MTP trace when the route trace runs in speculative mode."""
    from sglang.srt.environ import envs

    directory = envs.SGLANG_MOE_ROUTE_TRACE_DIR.get()
    if not directory or not envs.SGLANG_MOE_ROUTE_TRACE_SPECULATIVE.get():
        return None
    return install_mtp_hidden_trace(
        mtp_model,
        directory,
        envs.SGLANG_MOE_ROUTE_TRACE_MAX_TOKENS.get(),
        speculative_settings=_published_speculative_settings(),
    )
