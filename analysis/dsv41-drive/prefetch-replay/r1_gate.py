"""R1 gate recompute: the full 384-score early target gate from source x, native margins and top-12 parity.

Inputs are source x rows (f32, decoded from bf16), the frozen target gate (W, bias) and a b1_rows.Group whose stored
gate_ids/gate_scores came from the same fp32 recompute. The score is sqrt(softplus(x W^T)) + bias; ranking ties break
by ascending expert id.

Usage: s = r1_gate.full_gate_scores(x, W=W, bias=b); p = r1_gate.top12_parity(g, scores=s)
"""

from __future__ import annotations

import os
import sys

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b0_integrity  # noqa: E402
from b1_rows import GATE_DEPTH, TOPK, Group  # noqa: E402

from sglang.srt.layers.moe.residency_capture import ResidencyRow  # noqa: E402


class Parity(msgspec.Struct, frozen=True):
    """Token counts: exact = same top-12 set and scores, tie_only = sets differ only by ties at the 12th score."""

    tokens: int
    exact: int
    tie_only: int
    mismatched: int


def full_gate_scores(x: torch.Tensor, *, W: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Biased fp32 scores [n, 384]."""
    return torch.nn.functional.softplus(x @ W.T).sqrt() + bias


def native_margins(scores: torch.Tensor) -> torch.Tensor:
    """Each score minus its token's 6th-highest score (index 5 of the descending sort)."""
    sixth = torch.topk(scores, TOPK, dim=-1).values[:, TOPK - 1:TOPK]
    return scores - sixth


def load_target_gates(model_path, *, layer_ids: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """(W [384, hidden], bias [384]) fp32 per layer id, through b0_integrity._gates."""
    rows = [ResidencyRow(layer_id=int(layer), stratum="", gate_has_bias=True) for layer in layer_ids]
    W, bias = b0_integrity._gates(os.fspath(model_path), rows)
    return {int(layer): (W[i], bias[i]) for i, layer in enumerate(layer_ids)}


def _ranked(scores: np.ndarray) -> np.ndarray:
    """Expert ids by descending score, ties by ascending id."""
    return np.lexsort((np.arange(len(scores)), -scores))


def _classify(row: np.ndarray, *, stored_ids: np.ndarray, stored_scores: np.ndarray, atol: float) -> str:
    order = _ranked(row)[:GATE_DEPTH]
    got, want = set(order.tolist()), set(stored_ids.tolist())
    if got == want:
        top = np.sort(row[order])[::-1]
        return "exact" if np.allclose(top, np.sort(stored_scores)[::-1], atol=atol, rtol=0.0) else "mismatched"
    edge = float(row[order[-1]])
    swapped = np.array(sorted(got ^ want), dtype=np.int64)
    ties = np.abs(row[swapped] - edge) <= atol
    return "tie_only" if bool(ties.all()) else "mismatched"


def top12_parity(g: Group, *, scores: torch.Tensor, atol: float = 1e-5) -> Parity:
    """Compare the recomputed scores [n, 384] to the group's stored top-12; a group without a biased gate has none."""
    if not g.gate_has_bias:
        return Parity(tokens=0, exact=0, tie_only=0, mismatched=0)
    full = scores.numpy()
    counts = {"exact": 0, "tie_only": 0, "mismatched": 0}
    for i in range(len(full)):
        kind = _classify(full[i], stored_ids=g.gate_ids[i].astype(np.int64), stored_scores=g.gate_scores[i], atol=atol)
        counts[kind] += 1
    return Parity(tokens=len(full), **counts)
