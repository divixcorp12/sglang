"""R2 scorer cost of T5 and G0 under one CUDA graph at 6 live tokens (D9): what serving would pay for each.

T5 (trainable projection) is timed head-only (precomputed gate features and x, then head and sigmoid, no selection,
as r1_cost's head-only) and complete (fp32 gate features from x, then the same plus select). G0 (low-rank correction of the native 384 scores) is timed on
the correction path given precomputed native scores (what serving adds on top of its score kernel), head-only (the
same without selection) and complete (scores computed in-graph in fp32). Each path runs on the plan's low dtype
(model cast; gate and features stay fp32) and on the all-fp32 reference. read_set_diff compares the two read sets.

Inputs: a T5 or G0 model, the target layer's gate (W, bias) and the plan's CostSpec.
Usage: import r2_cost; out = r2_cost.measure_g0(model, W=W, bias=bias, cost=plan.cost, runs=1000, warmup=50)
"""

from __future__ import annotations

import copy
import os
import sys

import msgspec
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_cost  # noqa: E402
import r1_gate  # noqa: E402
import r1_plan  # noqa: E402
from r1_groups import ModelInput  # noqa: E402

_SHORT = r1_cost._SHORT


def _t5_steps(model, *, ref, W: torch.Tensor, bias: torch.Tensor, st: r1_cost._Static, dtype: torch.dtype,
              cap: int, device: str) -> dict:
    batch = r1_cost._static_batch(tokens=st.x.shape[0], dtype=dtype, device=device, experts=ref.head.cfg.num_experts)

    def scores(u: torch.Tensor) -> torch.Tensor:
        inp = ModelInput(batch=batch, u=u.to(dtype).unsqueeze(0), x=st.x.to(dtype).unsqueeze(0), s_native=None)
        return torch.sigmoid(model(inp)[0].float())

    with torch.no_grad():
        u_fixed = r1_cost._gate_features(st.x, st.ids, st.weights, ref=ref, W=W, bias=bias)

    def complete() -> torch.Tensor:
        u = r1_cost._gate_features(st.x, st.ids, st.weights, ref=ref, W=W, bias=bias)
        return r1_cost.select_torch(scores(u), skip=st.skip, threshold=st.threshold, cap=cap)

    def head_only() -> torch.Tensor:
        return scores(u_fixed)

    return {"head_only": head_only, "complete": complete}


def _g0_steps(model, *, ref, W: torch.Tensor, bias: torch.Tensor, st: r1_cost._Static, dtype: torch.dtype,
              cap: int, device: str) -> dict:
    batch = r1_cost._static_batch(tokens=st.x.shape[0], dtype=dtype, device=device, experts=ref.cfg.num_experts)

    def native() -> torch.Tensor:
        return r1_gate.native_margins(r1_gate.full_gate_scores(st.x, W=W, bias=bias))

    def scores(s_native: torch.Tensor) -> torch.Tensor:
        inp = ModelInput(batch=batch, u=None, x=st.x.to(dtype).unsqueeze(0),
                         s_native=s_native.to(dtype).unsqueeze(0))
        return torch.sigmoid(model(inp)[0].float())

    with torch.no_grad():
        s_fixed = native()

    def correction() -> torch.Tensor:
        return r1_cost.select_torch(scores(s_fixed), skip=st.skip, threshold=st.threshold, cap=cap)

    def head_only() -> torch.Tensor:
        return scores(s_fixed)

    def complete() -> torch.Tensor:
        return r1_cost.select_torch(scores(native()), skip=st.skip, threshold=st.threshold, cap=cap)

    return {"correction": correction, "head_only": head_only, "complete": complete}


def _measure(model, *, make_steps, experts: int, W: torch.Tensor, bias: torch.Tensor, cost: r1_plan.CostSpec,
             tokens: int, runs: int, warmup: int, device: str) -> dict:
    r1_cost._check_cost_spec(cost)
    if tokens != cost.live_tokens:
        raise ValueError(f"tokens {tokens} != the plan's live_tokens {cost.live_tokens}")
    ref = copy.deepcopy(model).float().to(device).eval()
    W, bias = W.float().to(device), bias.float().to(device)
    st = r1_cost._make_static(tokens=tokens, experts=experts, device=device)
    out: dict = {"tokens": tokens, "gate_dtype": cost.gate_dtype, "gate_bytes": 4 * (W.numel() + bias.numel())}
    for name in (cost.head_dtype, cost.reference_dtype):
        dtype = r1_cost._dtype(name)
        path_model = ref if dtype == torch.float32 else copy.deepcopy(ref).to(dtype).eval()
        steps = make_steps(path_model, ref=ref, W=W, bias=bias, st=st, dtype=dtype, cap=2, device=device)
        row = {path: r1_cost._time_graph(step, runs=runs, warmup=warmup) for path, step in steps.items()}
        row.update(r1_cost._bytes(path_model, itemsize=torch.finfo(dtype).bits // 8))
        out[_SHORT[name]] = row
    return out


def measure_t5(model, *, W: torch.Tensor, bias: torch.Tensor, cost: r1_plan.CostSpec, tokens: int = 6,
               runs: int = 1000, warmup: int = 50, device: str = "cuda") -> dict:
    """CUDA-graph timings of T5 head_only and complete on bf16 and fp32; increment_ms is the bf16 head-only median."""
    out = _measure(model, make_steps=_t5_steps, experts=model.head.cfg.num_experts, W=W, bias=bias, cost=cost,
                   tokens=tokens, runs=runs, warmup=warmup, device=device)
    low = out[_SHORT[cost.head_dtype]]
    out["increment_ms"] = low["head_only"]["median_ms"]
    out["complete_ms"] = low["complete"]["median_ms"]
    return out


def measure_g0(model, *, W: torch.Tensor, bias: torch.Tensor, cost: r1_plan.CostSpec, tokens: int = 6,
               runs: int = 1000, warmup: int = 50, device: str = "cuda") -> dict:
    """G0 correction, head_only and complete; increment_ms is the bf16 correction median given native scores."""
    out = _measure(model, make_steps=_g0_steps, experts=model.cfg.num_experts, W=W, bias=bias, cost=cost,
                   tokens=tokens, runs=runs, warmup=warmup, device=device)
    low = out[_SHORT[cost.head_dtype]]
    out["increment_ms"] = low["correction"]["median_ms"]
    out["complete_ms"] = low["complete"]["median_ms"]
    return out


def _cast_input(inp: ModelInput, *, dtype: torch.dtype, device) -> ModelInput:
    def cast(t):
        return None if t is None else t.to(device=device, dtype=dtype)

    batch = msgspec.structs.replace(inp.batch, width=torch.as_tensor(inp.batch.width, dtype=dtype, device=device))
    return ModelInput(batch=batch, u=cast(inp.u), x=cast(inp.x), s_native=cast(inp.s_native))


def read_set_diff(model, *, inputs: list[ModelInput], skips: list, threshold: float, cap: int = 2) -> float:
    """Fraction of groups whose capped read set differs between the bf16 and the fp32 copy of a T5 or G0 model."""
    ref = copy.deepcopy(model).float().eval()
    low = copy.deepcopy(ref).to(torch.bfloat16).eval()
    device = ref.device
    thr = torch.tensor(threshold, device=device)
    differ = total = 0
    with torch.no_grad():
        for inp, skip in zip(inputs, skips, strict=True):
            sk = torch.as_tensor(skip, device=device).bool()
            p32 = torch.sigmoid(ref(_cast_input(inp, dtype=torch.float32, device=device)).float())
            p16 = torch.sigmoid(low(_cast_input(inp, dtype=torch.bfloat16, device=device)).float())
            for a, b, s in zip(p32, p16, sk, strict=True):
                ra = r1_cost.select_torch(a, skip=s, threshold=thr, cap=cap)
                rb = r1_cost.select_torch(b, skip=s, threshold=thr, cap=cap)
                differ += int(set(ra[ra >= 0].tolist()) != set(rb[rb >= 0].tolist()))
                total += 1
    if total == 0:
        raise ValueError("no groups to compare")
    return differ / total
