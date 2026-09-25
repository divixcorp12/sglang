"""JIT wrappers for the native next-layer prefetch kernels (exl3_native_prefetch.cuh).

``plan`` scores one target layer's router logits, filters the top 6 against the target's residency and the pinned
tier's host map, picks a victim slot and posts one request on the prefetch page. ``commit`` waits for that request's
done word at the target layer and moves residency when it was copied. See the .cuh for the protocol and
``sglang.srt.layers.moe.exl3_native_prefetch`` for where they run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# The device counters, in the order of exl3_native_prefetch.cuh's kPosted..kAborted.
NATIVE_PREFETCH_COUNTERS = (
    "posted", "no_candidate", "ram_filtered", "no_victim", "copied", "skipped", "used", "aborted", "window_ns", "wait_ns",
)
# The pending record per target layer: {valid, expert, slot, generation, post %globaltimer ns, unused}.
PENDING_WORDS = 6
TOPK = 6


@cache_once
def _module() -> Module:
    return load_jit(
        "exl3_native_prefetch",
        cuda_files=["moe/exl3_native_prefetch.cuh"],
        cuda_wrappers=[
            ("plan", "exl3_native_prefetch_plan"),
            ("commit", "exl3_native_prefetch_commit"),
        ],
    )


def _check(name: str, tensor: torch.Tensor, dtype: torch.dtype, device: torch.device) -> None:
    if tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"native prefetch: {name} must be a contiguous {dtype} tensor on {device}")


def plan(
    logits: torch.Tensor,
    bias: torch.Tensor,
    mapping: torch.Tensor,
    slot_to_expert: torch.Tensor,
    victims: torch.Tensor,
    victim_valid: torch.Tensor,
    ram_map_row: torch.Tensor,
    page: torch.Tensor,
    prefetch_page: torch.Tensor,
    row: int,
    pending: torch.Tensor,
    generation: torch.Tensor,
    counters: torch.Tensor,
) -> None:
    """Post at most one prefetch for the target layer; ``pending`` (int64 [PENDING_WORDS]) records it as {valid,
    expert, slot, generation, post time}. ``ram_map_row`` is the pinned tier's host map row (int32, pinned, read through UVA); ``page`` the
    RAM-miss request page (its fatal word); ``row`` the target's streamed row, which the service reads."""
    device = logits.device
    _check("logits", logits, torch.float32, device)
    if bias.dtype not in (torch.bfloat16, torch.float32) or bias.device != device or bias.numel() != logits.shape[-1]:
        raise ValueError("native prefetch: bias must be bf16 or fp32 [experts] on the logits' device")
    for name, tensor in (("mapping", mapping), ("slot_to_expert", slot_to_expert), ("victims", victims),
                         ("pending", pending), ("generation", generation), ("counters", counters)):
        _check(name, tensor, torch.int64, device)
    _check("victim_valid", victim_valid, torch.bool, device)
    if victims.numel() != victim_valid.numel():
        raise ValueError("native prefetch: victims and victim_valid differ in length")
    if (mapping.numel() < logits.shape[-1] or pending.numel() < PENDING_WORDS
            or counters.numel() < len(NATIVE_PREFETCH_COUNTERS)):
        raise ValueError("native prefetch: a buffer is too small")
    if ram_map_row.dtype != torch.int32 or ram_map_row.device.type != "cpu" or ram_map_row.numel() < logits.shape[-1]:
        raise ValueError("native prefetch: ram_map_row is the pinned tier's int32 host map row")
    _module().plan(
        logits, bias, mapping, slot_to_expert, victims, victim_valid, int(ram_map_row.data_ptr()),
        int(page.data_ptr()), int(prefetch_page.data_ptr()), int(row), pending, generation, counters,
    )


def commit(
    pending: torch.Tensor,
    prefetch_page: torch.Tensor,
    page: torch.Tensor,
    lease_address: int,
    timeout_ns: int,
    mapping: torch.Tensor,
    slot_to_expert: torch.Tensor,
    slot_state: torch.Tensor,
    slot_generations: torch.Tensor,
    ready_state: int,
    routes: torch.Tensor,
    counters: torch.Tensor,
) -> None:
    """Wait for ``pending``'s request and, if it was copied, move its slot to the new expert; see the module doc."""
    device = pending.device
    for name, tensor in (("pending", pending), ("mapping", mapping), ("slot_to_expert", slot_to_expert),
                         ("slot_generations", slot_generations), ("routes", routes), ("counters", counters)):
        _check(name, tensor, torch.int64, device)
    _check("slot_state", slot_state, torch.uint8, device)
    if timeout_ns <= 0:
        raise ValueError("native prefetch: the commit wait needs a positive timeout")
    _module().commit(
        pending, int(prefetch_page.data_ptr()), int(page.data_ptr()), int(lease_address), int(timeout_ns),
        mapping, slot_to_expert, slot_state, slot_generations, int(ready_state), routes, counters,
    )
