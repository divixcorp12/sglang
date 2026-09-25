"""Native next-layer expert prefetch for DSV4.1 EXL3 decode (SGLANG_DSV41_ENABLE_NATIVE_PREFETCH).

Plan: docs/superpowers/plans/2026-09-25-dsv41-native-prefetch.md. The h=1 native gate at K=1: in layer T-1's
captured forward, after its gather and before its fused MoE, layer T's own router gate scores layer T-1's router
input (``tiny_gemm_bf16`` into fp32 logits), and the plan kernel posts the best top-6 expert of T that is neither in
T's hot cache nor missing from the pinned tier, into the slot DIRECT would evict first after its demand shortlist.
The RAM-miss service copies it with its copy engine (demand jobs go first). At the start of layer T's forward the
commit kernel waits for that copy's done word before T's gather reads residency, and then maps the slot. Target
layers 1..39 only: nothing crosses a token (T=0 is skipped).

Only residency changes, never the math: the MoE reads the same bytes whether a demand copy or a prefetch copy put
them in the slot. Only a captured graph posts or waits (an eager forward never does), and the copy engine arms only
after the kernels here were captured (:meth:`NativePrefetch.check_armable`), so no first module load of them can
happen while an armed step is in flight (LEASE_PROTOCOL.md 7.6 "Module loading").
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.moe import exl3_native_prefetch as kernels

logger = logging.getLogger(__name__)

# Victim candidates past the demand shortlist that the plan considers: up to six may hold one of the predicted
# experts, so seven always leave one that does not.
PREFETCH_VICTIMS = 7

# layer id -> the layer's router gate module (MoEGate: ``weight`` [E, H] bf16, ``e_score_correction_bias`` [E]).
_GATES: dict[int, torch.nn.Module] = {}


def register_gate(layer_id: int, gate: torch.nn.Module) -> None:
    """Called by each DeepseekV2MoE when the flag is on: the gates a predicting layer needs of its next layer."""
    _GATES[int(layer_id)] = gate


def registered_gate(layer_id: int) -> Optional[torch.nn.Module]:
    return _GATES.get(int(layer_id))


class NativePrefetch:
    """Per process: the prefetch page, the device-side pending records and counters, and the two hooks.

    ``page`` is the service's pinned prefetch page, already handed to the host (``enable_native_prefetch``).
    :meth:`bind` attaches the device state once the GPU residency updater and the device side exist.
    """

    def __init__(self, page: torch.Tensor, service) -> None:
        self.page = page
        self.service = service
        self.updater = None
        self.rows: dict[int, int] = {}  # layer id -> residency row
        self.logits: Optional[torch.Tensor] = None
        self.pending: Optional[torch.Tensor] = None
        self.generation: Optional[torch.Tensor] = None
        self.counters: Optional[torch.Tensor] = None
        # Set when a captured graph holds both kernels: the copy engine may arm only then (check_armable).
        self.captured_plan = False
        self.captured_commit = False
        self._snapshot: Optional[torch.Tensor] = None
        self._snapshot_event = None
        self._checks = 0

    def bind(self, updater) -> None:
        if self.updater is not None:
            return
        if getattr(updater, "prefetch_victims", None) is None:
            raise RuntimeError(
                "native prefetch needs DIRECT's extended victim ranking (SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 with "
                "the flag on when the residency updater is built)"
            )
        device = updater.device
        self.updater = updater
        self.rows = {layer_id: row for row, layer_id in enumerate(updater.layer_ids)}
        self.logits = torch.zeros((1, updater.num_experts), dtype=torch.float32, device=device)
        self.pending = torch.zeros((updater.num_layers, 4), dtype=torch.int64, device=device)
        self.generation = torch.zeros(1, dtype=torch.int64, device=device)
        self.counters = torch.zeros(len(kernels.NATIVE_PREFETCH_COUNTERS), dtype=torch.int64, device=device)
        logger.info("exl3 native prefetch bound to %d layers", len(self.rows))

    # ---- the hooks, from Exl3MoEMethod._apply_graph ----

    def commit(self, layer_id: int, routes: torch.Tensor) -> None:
        """At the start of layer ``layer_id``'s graph forward, before its gather: settle its pending prefetch."""
        if self.updater is None or not torch.cuda.is_current_stream_capturing():
            return
        row = self.rows.get(int(layer_id))
        if row is None or row == 0:
            return
        device_side = self.service.device_side
        u = self.updater
        kernels.commit(
            self.pending[row], self.page, self.service.page, device_side._lease_address, device_side.timeout_ns,
            u.mapping[row], u.slot_to_expert[row], u.slot_state[row], u.slot_generations[row], _ready_state(),
            routes, self.counters,
        )
        self.captured_commit = True

    def predict(self, layer_id: int, x: torch.Tensor) -> None:
        """In layer ``layer_id``'s graph forward, after its gather and before its fused MoE: plan the next layer."""
        if self.updater is None or not torch.cuda.is_current_stream_capturing():
            return
        target = int(layer_id) + 1
        row = self.rows.get(target)
        gate = registered_gate(target)
        if row is None or gate is None or row != self.rows.get(int(layer_id), -2) + 1:
            return
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

        # The router's own GEMV (MoEGate.forward: bf16 x bf16 -> fp32, max_m 16), so it is the kernel already loaded.
        tiny_gemm_bf16(x.reshape(1, -1), gate.weight, self.logits, max_m=16)
        u = self.updater
        ram_row = self.service.row_of(target)
        kernels.plan(
            self.logits[0], gate.e_score_correction_bias, u.mapping[row], u.slot_to_expert[row],
            u.prefetch_victims[row], u.prefetch_victim_valid[row], self.service.slot_map[ram_row],
            self.service.page, self.page, ram_row, self.pending[row], self.generation, self.counters,
        )
        self.captured_plan = True

    # ---- safety and reporting ----

    def check_armable(self) -> None:
        """The copy engine may arm only once the decode graph holds both kernels: a kernel first launched (and so
        first loaded) after arming could take the driver lock while a copy wait spins (7.6 "Module loading")."""
        if not (self.captured_plan and self.captured_commit):
            raise RuntimeError(
                "exl3 native prefetch: the copy engine would arm before the prefetch kernels were captured "
                f"(plan {self.captured_plan}, commit {self.captured_commit}); a later first load could deadlock"
            )

    def poll_counters(self, every: int = 512) -> Optional[dict]:
        """Every ``every`` calls, a non-blocking readback of the device counters; logs the previous one when ready."""
        if self.counters is None or not torch.cuda.is_available():
            return None
        self._checks += 1
        out = None
        if self._snapshot_event is not None and self._snapshot_event.query():
            out = dict(zip(kernels.NATIVE_PREFETCH_COUNTERS, self._snapshot.tolist()))
            logger.info("exl3 native prefetch device counters %s", out)
            self._snapshot_event = None
        if self._checks % every == 0 and self._snapshot_event is None and not torch.cuda.is_current_stream_capturing():
            if self._snapshot is None:
                self._snapshot = torch.empty_like(self.counters, device="cpu").pin_memory()
            self._snapshot.copy_(self.counters, non_blocking=True)
            self._snapshot_event = torch.cuda.Event()
            self._snapshot_event.record()
        return out

    def stats(self) -> dict:
        """Device counters, synchronously (tests and shutdown)."""
        if self.counters is None:
            return {}
        return dict(zip(kernels.NATIVE_PREFETCH_COUNTERS, self.counters.cpu().tolist()))


def _ready_state() -> int:
    from sglang.srt.layers.moe.expert_residency_gpu import _READY

    return _READY
