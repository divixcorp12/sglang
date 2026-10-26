"""Lossless GPTQ INT3 -> BesTLA planes for the typed CPU expert kernel.

This adapter accepts canonical, unshuffled GPTQ group128 symmetric tensors only. It never
requantizes, changes scale signs, writes a checkpoint, or enables a GPU serving backend.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec

SLAB_NAMES = ("w13_low2", "w13_high1", "w13_scales", "w2_low2", "w2_high1", "w2_scales")
POLICY = {
    "quant_method": "gptq",
    "bits": 3,
    "group_size": 128,
    "sym": True,
    "desc_act": False,
}


@dataclass(frozen=True)
class Projection:
    low2: torch.Tensor
    high1: torch.Tensor
    scales: torch.Tensor


def _unpack(words: torch.Tensor, count: int, dim: int) -> torch.Tensor:
    """Canonical 32 codes in 3 words, including joins at codes10 and21."""
    w = words.to(torch.int64).bitwise_and(0xFFFFFFFF).movedim(dim, -1)
    pos = torch.arange(count, dtype=torch.int64) * 3
    index, shift = pos // 32, pos % 32
    # A padded zero word makes the final non-crossing read safe.
    padded = torch.cat((w, torch.zeros_like(w[..., :1])), dim=-1)
    code = (padded[..., index] >> shift) | (padded[..., index + 1] << (32 - shift))
    return (code & 7).to(torch.uint8).movedim(-1, dim)


def prepare_projection(
    tensors: Mapping[str, torch.Tensor],
    *,
    bits: int = 3,
    group_size: int = 128,
    symmetric: bool = True,
    desc_act: bool = False,
) -> Projection:
    """Validate policy and exact QDQ, then repack integer bits without rounding scales.

    Tensor names are local projection names, not checkpoint paths. Ownership/hash validation
    remains the checkpoint reader's responsibility; no inference from a filename is made here.
    """
    if (
        type(bits) is not int
        or type(group_size) is not int
        or bits != 3
        or group_size != 128
        or symmetric is not True
        or desc_act is not False
    ):
        raise ValueError("requires canonical symmetric INT3/group128 without act-order")
    if set(tensors) != {"qweight", "qzeros", "scales", "g_idx"}:
        raise ValueError("requires exactly qweight/qzeros/scales/g_idx")
    q, z, s, g = (tensors[n] for n in ("qweight", "qzeros", "scales", "g_idx"))
    for t in (q, z, s, g):
        if t.device.type != "cpu" or not t.is_contiguous():
            raise ValueError("canonical tensors must be contiguous CPU tensors")
    if (
        q.dtype != torch.int32
        or z.dtype != torch.int32
        or g.dtype != torch.int32
        or s.dtype != torch.float16
    ):
        raise ValueError("requires int32 packed tensors/groups and FP16 scales")
    if q.ndim != 2 or s.ndim != 2 or g.ndim != 1 or z.ndim != 2:
        raise ValueError("invalid canonical ranks")
    k, n = g.numel(), q.shape[1]
    if not k or not n or k % 128 or n % 128 or q.shape != (k // 32 * 3, n):
        raise ValueError("invalid projection dimensions")
    if s.shape != (k // 128, n) or z.shape != (k // 128, n // 32 * 3):
        raise ValueError("invalid group scale/zero shapes")
    if not torch.equal(g, torch.arange(k, dtype=torch.int32) // 128):
        raise ValueError("g_idx must be contiguous group128 without permutations")
    if not bool(torch.isfinite(s).all()) or bool((s == 0).any()):
        raise ValueError("scales must be finite nonzero FP16; signed scales are valid")
    if not bool((_unpack(z, n, 1) == 3).all()):
        raise ValueError("symmetric INT3 requires packed zero3 plus GPTQ offset1")
    code = _unpack(q, k, 0).t().contiguous()
    # All codes must have finite FP16 QDQ, even if a particular code is absent from this projection.
    if not bool(
        torch.isfinite((s.float().unsqueeze(-1) * torch.arange(-4, 4)).half()).all()
    ):
        raise ValueError("FP16 QDQ overflows")
    flat = code.flatten().to(torch.int16)
    low = ((flat.reshape(-1, 4) & 3) << torch.arange(0, 8, 2)).sum(-1).to(torch.uint8)
    high = ((flat.reshape(-1, 8) >> 2) << torch.arange(8)).sum(-1).to(torch.uint8)
    return Projection(
        low.reshape(n, k // 4), high.reshape(n, k // 8), s.t().contiguous()
    )


def prepare_expert(
    gate: Projection, up: Projection, down: Projection
) -> dict[str, torch.Tensor]:
    """One expert's six row tensors; concatenate only, with independent storage."""
    i, h4 = gate.low2.shape
    if up.low2.shape != (i, h4) or down.low2.shape != (h4 * 4, i // 4):
        raise ValueError("gate/up/down projection dimensions disagree")
    # Check all planes as callers can construct Projection directly.
    for p, n, k in ((gate, i, h4 * 4), (up, i, h4 * 4), (down, h4 * 4, i)):
        if k % 128 or n % 128:
            raise ValueError("projection dimensions must be multiples of128")
        for t, shape, dtype in (
            (p.low2, (n, k // 4), torch.uint8),
            (p.high1, (n, k // 8), torch.uint8),
            (p.scales, (n, k // 128), torch.float16),
        ):
            if (
                t.shape != shape
                or t.dtype != dtype
                or t.device.type != "cpu"
                or not t.is_contiguous()
            ):
                raise ValueError("invalid projection plane")
        if not bool(torch.isfinite(p.scales).all()) or bool((p.scales == 0).any()):
            raise ValueError("invalid scales")
    return dict(
        zip(
            SLAB_NAMES,
            (
                torch.cat((gate.low2, up.low2)),
                torch.cat((gate.high1, up.high1)),
                torch.cat((gate.scales, up.scales)),
                down.low2.clone(),
                down.high1.clone(),
                down.scales.clone(),
            ),
        )
    )


def load_exported_expert(
    root: str | Path, layer: int, expert: int, *, expected_manifest_sha256: str
) -> dict[str, torch.Tensor]:
    """Read one original portable expert through an explicitly trusted manifest SHA.

    Hash the same bytes that safetensors decodes, bind every key to its requested owner, and
    reject path escapes, incomplete manifests and other quant policies. This does not load
    arbitrary Python from the checkpoint, nor modify it. The caller supplies the SHA from
    its verified export receipt, rather than trusting an adjacent mutable seal alone.
    """
    from safetensors.torch import load

    root = Path(root).resolve()
    if (
        type(layer) is not int
        or type(expert) is not int
        or not 1 <= layer <= 69
        or not 0 <= expert < 288
    ):
        raise ValueError("invalid expert owner")

    def read(relative):
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("export path escapes root")
        return path.read_bytes()

    raw = read("export-manifest.json")
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("export manifest SHA mismatch")
    manifest = json.loads(raw)
    if (
        manifest.get("status") != "completed"
        or manifest.get("format") != "mimo-hybrid-gptq-v1"
        or manifest.get("expert_policy") != POLICY
        or any(
            type(manifest["expert_policy"][k]) is not type(v) for k, v in POLICY.items()
        )
    ):
        raise ValueError("export manifest policy/format/status mismatch")
    relative = f"experts/layer-{layer:02d}/expert-{expert:03d}/weights.safetensors"
    entry = manifest["files"][relative]
    raw = read(relative)
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("expert payload SHA/size mismatch")
    tensors = load(raw)
    prefix = f"model.layers.{layer}.mlp.experts.{expert}."
    names = ("gate_proj", "up_proj", "down_proj")
    fields = ("qweight", "qzeros", "scales", "g_idx")
    if set(tensors) != {prefix + n + "." + f for n in names for f in fields}:
        raise ValueError("expert tensor ownership/key mismatch")
    return prepare_expert(
        *(
            prepare_projection({f: tensors[prefix + n + "." + f] for f in fields})
            for n in names
        )
    )


class GptqInt3CpuTrait:
    name = "gptq_int3"
    slab_names = SLAB_NAMES
    x_dtype = torch.float16
    weights_dtype = out_dtype = torch.float32

    def __init__(self, act_limit: float | None = None):
        self.act_limit = act_limit

    def check_environment(self) -> None:
        import platform

        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("GPTQ INT3 CPU experts require x86-64 Linux")

    def attach_host(self, host) -> None:
        pass

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        tensor = slabs["w2_low2"]
        if tensor.ndim != 3:
            raise ValueError(
                "down slab must have shape [experts, hidden, intermediate/4]"
            )
        return int(tensor.shape[1])

    def layer_spec(
        self, slabs: Mapping[str, torch.Tensor], capacity: int
    ) -> CpuExpertLayerSpec:
        import math

        limit = 0.0 if self.act_limit is None else self.act_limit
        h = self.hidden_size(slabs)
        w13 = slabs["w13_low2"]
        if w13.ndim != 3 or w13.shape[1] % 2:
            raise ValueError("invalid gate/up slab")
        i = w13.shape[1] // 2
        if (
            capacity < 1
            or not 128 <= h <= 65536
            or not 128 <= i <= 65536
            or h % 128
            or i % 128
            or not math.isfinite(limit)
            or limit < 0
        ):
            raise ValueError("invalid capacity, dimensions or clamp")
        shapes = (
            (2 * i, h // 4),
            (2 * i, h // 8),
            (2 * i, h // 128),
            (h, i // 4),
            (h, i // 8),
            (h, i // 128),
        )
        keep = []
        for name, shape in zip(SLAB_NAMES, shapes):
            t = slabs[name]
            dtype = torch.float16 if name.endswith("scales") else torch.uint8
            if (
                t.device.type != "cpu"
                or t.dtype != dtype
                or not t.is_contiguous()
                or t.ndim != 3
                or t.shape[0] < capacity
                or tuple(t.shape[1:]) != shape
            ):
                raise ValueError(f"invalid CPU slab {name}")
            keep.append(t)
        return CpuExpertLayerSpec(
            capacity=capacity,
            hidden=h,
            intermediate=i,
            act_limit=limit,
            slabs=tuple((t.data_ptr(), t[0].numel() * t.element_size()) for t in keep),
            params=(128,),
            keep=tuple(keep),
            plan="bestla-int3-fp32-activation",
        )
