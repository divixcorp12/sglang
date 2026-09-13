"""Target vocabulary weights a speculative draft binds at construction.

A draft that later receives the target's embedding and lm_head through
``set_embed_and_head`` would otherwise allocate its own copies first, and those
copies are still held while the target KV pool is sized.

The offer is module-global, not thread-local: one scheduler per rank builds its
draft on one thread.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from itertools import chain
from typing import Callable, Iterator, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

_shared_embed: Optional[torch.Tensor] = None
_shared_head: Optional[torch.Tensor] = None


@contextmanager
def draft_shares_target_embed_and_head(
    embed: Optional[torch.Tensor], head: Optional[torch.Tensor]
) -> Iterator[None]:
    """Offer the target's embedding and lm_head weights to draft models built inside.

    Only a worker that afterwards calls the draft's ``set_embed_and_head`` with
    these same tensors may open this scope: binding them at construction then
    reaches the state that call produces, without allocating draft copies.
    ``None`` offers nothing for that half.
    """
    global _shared_embed, _shared_head
    previous = _shared_embed, _shared_head
    _shared_embed, _shared_head = embed, head
    try:
        yield
    finally:
        _shared_embed, _shared_head = previous


def shared_target_embed() -> Optional[torch.Tensor]:
    """The target embedding weight offered to the draft being built, if any."""
    return _shared_embed


def shared_target_head() -> Optional[torch.Tensor]:
    """The target lm_head weight offered to the draft being built, if any."""
    return _shared_head


def build_with_target_weight(
    build: Callable[[], nn.Module],
    target_weight: Optional[torch.Tensor],
    name: str,
) -> nn.Module:
    """Build a vocabulary module whose ``weight`` is ``target_weight`` when it fits.

    The module is built on ``meta`` first. If its weight has the target's shape
    and dtype and it registers no other tensor, the target tensor is bound and
    the module is marked ``shares_target_weight``; nothing is allocated.
    Otherwise the declined offer is logged and the module is built normally, as
    it is whenever no target weight is offered.
    """
    if target_weight is None:
        return build()
    with torch.device("meta"):
        module = build()
    weight = module._parameters.get("weight")
    other_tensors = [
        tensor_name
        for tensor_name, _ in chain(module.named_parameters(), module.named_buffers())
        if tensor_name != "weight"
    ]
    if (
        weight is None
        or other_tensors
        or weight.shape != target_weight.shape
        or weight.dtype != target_weight.dtype
    ):
        logger.info(
            "Draft %s declined the target weight and allocates its own: "
            "draft shape=%s dtype=%s, target shape=%s dtype=%s, other tensors=%s",
            name,
            None if weight is None else tuple(weight.shape),
            None if weight is None else weight.dtype,
            tuple(target_weight.shape),
            target_weight.dtype,
            other_tensors,
        )
        return build()
    del module.weight
    module.weight = target_weight
    module.shares_target_weight = True
    return module


def require_vocab_weights_materialized(model: nn.Module) -> None:
    """Raise if a draft ``embed_tokens`` or ``lm_head`` weight is still on ``meta``.

    Only the shared vocabulary weights are checked: other parameters may stay on
    ``meta`` on purpose, as layers offloaded with ``--offload-mode meta`` do.
    """
    for module_name, module in model.named_modules():
        if module_name.rsplit(".", 1)[-1] not in ("embed_tokens", "lm_head"):
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.is_meta:
            raise RuntimeError(
                f"draft {module_name}.weight is still on meta after the target's "
                "embedding and lm_head were shared"
            )
