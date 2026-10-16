"""Per-call CUDA memory of a DSpark prefill (SGLANG_DEBUG_PREFILL_PEAK_LOG).

``install`` wraps, once and only when the flag is on, the DSpark worker's prefill (the root), the target's decoder
layers with their attention and MoE, the target's logits processor and sampler, the draft model and the target-hidden
KV injection. With the flag off nothing is wrapped and the call path is untouched.

Inside a root call every wrapped call records the caching allocator's allocated and reserved bytes before, at their
peak during, and after it. Peaks are read with ``reset_peak_memory_stats`` around each call; an enclosing call folds
its children's peaks into its own, so nesting does not lose them. The root then logs one line,
``prefill peak log {json}``, with the root's token count and every call in order. Calls outside a root, and calls
under CUDA graph capture, pass straight through. All counters are host-side allocator state, so nothing synchronizes.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)

LOG_PREFIX = "prefill peak log "
# A DSV4 attention layer's submodules measured on their own (absent ones are skipped).
ATTENTION_PARTS = ("wqkv_a", "wq_a", "wq_b", "wkv", "compressor", "indexer", "attn_mqa", "wo_a", "wo_b")
MIB = 1 << 20


class _Frame:
    __slots__ = ("alloc_peak", "reserved_peak")

    def __init__(self) -> None:
        self.alloc_peak = 0
        self.reserved_peak = 0


class PrefillPeakLog:
    """The open root's calls and the stack of open frames; one per process."""

    def __init__(self, cuda=torch.cuda) -> None:
        self.cuda = cuda
        self.stack: list[_Frame] = []
        self.entries: list[dict] = []
        self.root_tokens: Optional[int] = None

    def _read(self) -> tuple[int, int, int, int]:
        c = self.cuda
        return c.memory_allocated(), c.max_memory_allocated(), c.memory_reserved(), c.max_memory_reserved()

    def _enter(self) -> tuple[int, int]:
        allocated, alloc_max, reserved, reserved_max = self._read()
        if self.stack:  # fold the parent's peak so far before the reset loses it
            parent = self.stack[-1]
            parent.alloc_peak = max(parent.alloc_peak, alloc_max)
            parent.reserved_peak = max(parent.reserved_peak, reserved_max)
        self.cuda.reset_peak_memory_stats()
        self.stack.append(_Frame())
        return allocated, reserved

    def _exit(self) -> tuple[int, int, int, int]:
        frame = self.stack.pop()
        allocated, alloc_max, reserved, reserved_max = self._read()
        frame.alloc_peak = max(frame.alloc_peak, alloc_max)
        frame.reserved_peak = max(frame.reserved_peak, reserved_max)
        if self.stack:
            parent = self.stack[-1]
            parent.alloc_peak = max(parent.alloc_peak, frame.alloc_peak)
            parent.reserved_peak = max(parent.reserved_peak, frame.reserved_peak)
        return allocated, frame.alloc_peak, reserved, frame.reserved_peak

    def wrap(self, name: str, fn: Callable, *, root: bool = False, tokens: Optional[Callable] = None) -> Callable:
        """``fn`` measured as ``name``; a root opens a record (``tokens(*args)`` gives its token count) and logs it."""

        @functools.wraps(fn)
        def measured(*args: Any, **kwargs: Any):
            if (not root and not self.stack) or self.cuda.is_current_stream_capturing():
                return fn(*args, **kwargs)
            if root and not self.stack:
                self.entries = []
                self.root_tokens = tokens(*args, **kwargs) if tokens is not None else None
            depth = len(self.stack)
            before_alloc, before_reserved = self._enter()
            try:
                return fn(*args, **kwargs)
            finally:
                after_alloc, alloc_peak, after_reserved, reserved_peak = self._exit()
                self.entries.append(
                    {
                        "name": name,
                        "depth": depth,
                        "alloc_before_mib": round(before_alloc / MIB, 1),
                        "alloc_peak_mib": round(alloc_peak / MIB, 1),
                        "alloc_after_mib": round(after_alloc / MIB, 1),
                        "reserved_before_mib": round(before_reserved / MIB, 1),
                        "reserved_peak_mib": round(reserved_peak / MIB, 1),
                        "reserved_after_mib": round(after_reserved / MIB, 1),
                    }
                )
                if not self.stack:
                    logger.info("%s%s", LOG_PREFIX, format_record(self.root_tokens, self.entries))
                    self.entries = []

        measured.__prefill_peak_wrapped__ = True
        return measured


def format_record(tokens: Optional[int], entries: list[dict]) -> str:
    return json.dumps({"tokens": tokens, "calls": entries}, separators=(",", ":"))


def parse_line(line: str) -> Optional[dict]:
    """The record of one logged line, or None for any other line."""
    at = line.find(LOG_PREFIX + "{")  # not the "prefill peak log installed" line
    return json.loads(line[at + len(LOG_PREFIX) :]) if at >= 0 else None


def _wrap_attr(log: PrefillPeakLog, obj: Any, attr: str, name: str, **kw) -> None:
    fn = getattr(obj, attr, None)
    if fn is not None and not getattr(fn, "__prefill_peak_wrapped__", False):
        setattr(obj, attr, log.wrap(name, fn, **kw))


def _batch_tokens(batch, *args, **kwargs) -> Optional[int]:
    tokens = getattr(batch, "extend_num_tokens", None)
    return int(tokens) if tokens is not None else None


def maybe_install(worker) -> Optional[PrefillPeakLog]:
    """``install`` when SGLANG_DEBUG_PREFILL_PEAK_LOG is on, read once here; off, the worker is left untouched."""
    from sglang.srt.environ import envs

    return install(worker) if envs.SGLANG_DEBUG_PREFILL_PEAK_LOG.get() else None


def install(worker, log: Optional[PrefillPeakLog] = None) -> PrefillPeakLog:
    """Wrap a DSpark worker's prefill and the modules it runs. The caller checks the flag; this always wraps."""
    log = log or PrefillPeakLog()
    target = worker.model_runner.model
    _wrap_attr(log, worker, "_forward_prefill", "dspark.prefill", root=True, tokens=_batch_tokens)
    _wrap_attr(log, target, "forward", "target")
    inner = getattr(target, "model", None)
    for i, layer in enumerate(getattr(inner, "layers", ()) or ()):
        if not isinstance(layer, torch.nn.Module) or not hasattr(layer, "self_attn"):
            continue  # PP placeholders
        _wrap_attr(log, layer, "forward", f"L{i}")
        _wrap_attr(log, layer.self_attn, "forward", f"L{i}.attn")
        # Inside attention: the projections, the compressor and indexer, and the FlashMLA core (attn_mqa).
        for name in ATTENTION_PARTS:
            _wrap_attr(log, getattr(layer.self_attn, name, None), "forward", f"L{i}.attn.{name}")
        # The decoder layer's mHC stream mixing, which holds the 4-stream hidden and its residual. The DSV4 layer
        # loop enters a layer through forward_hc_pre_from_prev (not forward), which mixes with _hc_combine and
        # _hc_post_with_combine.
        _wrap_attr(log, layer, "forward_hc_pre_from_prev", f"L{i}")
        for name in ("hc_pre", "hc_post", "_hc_combine", "_hc_post_with_combine"):
            _wrap_attr(log, layer, name, f"L{i}.{name.lstrip('_')}")
        if getattr(layer, "mlp", None) is not None:
            _wrap_attr(log, layer.mlp, "forward", f"L{i}.moe")
    _wrap_attr(log, getattr(target, "logits_processor", None), "forward", "target.logits")
    _wrap_attr(log, getattr(worker.model_runner, "sampler", None), "forward", "target.sampler")
    _wrap_attr(log, getattr(worker, "_kv_injector", None), "inject_target_hidden", "dspark.inject")
    _wrap_attr(log, getattr(worker, "draft_model", None), "forward", "draft")
    logger.info("prefill peak log installed on %s", type(target).__name__)
    return log
