"""P2 gate: exllamav3's fused exl3_moe over slot pointer tables, on real layer rows (GPU).

Reads 12 real experts of one layer through the model's own row source into
hot-cache-shaped slot tensors, then for 8 BS1 top-6 route sets compares:
  * exl3_moe (deterministic: output scratch + exl3_moe_gather) over the slots,
  * exl3_moe_loop over the same slots,
against an fp32 reference built from exl3_linear_reference. Then it captures the
fused call in a CUDA graph and replays it after rewriting slot rows, remap,
routes and input in place; each replay must equal an eager call bitwise.

Bars (plan Design decision D7): rel(fused) <= 1.2e-2 and
rel(fused) <= 2 * rel(loop) + 1e-3 for every route set; replay == eager bitwise.
Both num_active = 6 and num_active = -1 are measured; the report's "num_active"
is 6 when that mode passes, else -1 when it passes, and the verdict is PASS when
either passes.
Env: DSV41_EXL3_DIR (checkpoint), DSV41_PROBE_LAYER (default 3),
DSV41_PROBE_OUT (JSON report path).
"""

import json
import os
import time
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

EXL3_DIR = os.environ.get("DSV41_EXL3_DIR", "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw")
LAYER = int(os.environ.get("DSV41_PROBE_LAYER", "3"))
OUT = os.environ.get("DSV41_PROBE_OUT")
EXPERTS = list(range(0, 384, 32))  # 12 experts -> 12 slots
TOP_K = 6
ROUTE_SETS = 8
REWRITES = 4
R_ROWS = 16  # fused-kernel row tile; one route per slot at BS1
ACT_SILU = 0
ACT_LIMIT = 10.0
REL_BOUND = 1.2e-2


@dataclass
class FusedBuffers:
    expert_count: torch.Tensor
    ones: torch.Tensor
    token_sorted: torch.Tensor
    scratch: torch.Tensor
    out: torch.Tensor
    temp_state_g: torch.Tensor
    temp_state_u: torch.Tensor
    temp_intermediate_g: torch.Tensor
    temp_intermediate_u: torch.Tensor


def make_buffers(ext, slots: int, hidden: int, inter: int, device) -> FusedBuffers:
    concurrency = ext.exl3_moe_max_concurrency(device.index)
    half = dict(dtype=torch.float16, device=device)
    return FusedBuffers(
        expert_count=torch.zeros(slots + 1, dtype=torch.long, device=device),
        ones=torch.ones(TOP_K, dtype=torch.long, device=device),
        token_sorted=torch.zeros(TOP_K, dtype=torch.long, device=device),
        scratch=torch.empty((TOP_K, hidden), dtype=torch.float32, device=device),
        out=torch.empty((1, hidden), dtype=torch.float32, device=device),
        temp_state_g=torch.empty((concurrency, R_ROWS, hidden), **half),
        temp_state_u=torch.empty((concurrency, R_ROWS, hidden), **half),
        temp_intermediate_g=torch.empty((concurrency, R_ROWS, inter), **half),
        temp_intermediate_u=torch.empty((concurrency, R_ROWS, inter), **half),
    )


def pointer_tables(slot_tensors: dict, slots: int, device) -> dict:
    """Nine int64 [slots] tables of raw row addresses: gate = w13 part 0, up = part 1, down = w2."""
    def table(name, part):
        rows = slot_tensors[name]
        return torch.tensor([rows[s, part].data_ptr() for s in range(slots)], dtype=torch.long, device=device)

    return {
        f"{proj}_{kind}": table(f"{prefix}_{kind}", part)
        for proj, prefix, part in (("gate", "w13", 0), ("up", "w13", 1), ("down", "w2", 0))
        for kind in ("trellis", "suh", "svh")
    }


def fused_moe_slots(ext, bufs: FusedBuffers, x16, weights, remap, tables, *, act_limit, bits, num_active):
    """exl3_moe over slots, deterministic accumulation; every op is capture-safe.

    ``remap`` int64 [6] names a slot per route (distinct at BS1), ``weights`` fp32 [6].
    """
    bufs.expert_count.zero_().index_add_(0, remap, bufs.ones)
    order = torch.argsort(remap)
    inv_order = torch.empty_like(order).scatter_(0, order, torch.arange(TOP_K, device=remap.device))
    weight_sorted = weights[order].to(torch.float16)
    expert_start = torch.cumsum(bufs.expert_count, 0) - bufs.expert_count
    det = torch.stack([expert_start, expert_start, (bufs.expert_count > 0).long()])
    bufs.out.zero_()
    ext.exl3_moe(
        x16, bufs.out, bufs.expert_count, bufs.token_sorted, weight_sorted,
        bufs.temp_state_g, bufs.temp_state_u, bufs.temp_intermediate_g, bufs.temp_intermediate_u,
        ACT_SILU, bits["gate"], bits["up"], bits["down"],
        tables["gate_trellis"], tables["gate_suh"], tables["gate_svh"],
        tables["up_trellis"], tables["up_suh"], tables["up_svh"],
        tables["down_trellis"], tables["down_suh"], tables["down_svh"],
        False, True, False, True, False, True,
        act_limit, num_active, bufs.scratch, det[0], 1, R_ROWS, 16,
    )
    slots = bufs.expert_count.shape[0] - 1
    ext.exl3_moe_gather(
        bufs.out, bufs.scratch, remap, inv_order,
        det[1, :slots], det[0, :slots], det[2, :slots], weight_sorted,
    )
    return bufs.out


def _load_slots(device):
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    layout = build_exl3_expert_layout(EXL3_DIR)
    fmt = Exl3ExpertFormat(layout, LAYER, direct=True)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    host = {name: torch.empty((len(EXPERTS),) + spec.row_shape, dtype=spec.dtype) for name, spec in specs.items()}
    source = Exl3ShardRowSource.for_layer(layout, LAYER, fmt.segment_map(), direct=True)
    source.read(torch.tensor(EXPERTS, dtype=torch.long), host)
    return {name: tensor.to(device) for name, tensor in host.items()}


def _views(slot_tensors, slot):
    from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors

    def t(prefix, part):
        return Exl3Tensors(
            trellis=slot_tensors[f"{prefix}_trellis"][slot, part],
            suh=slot_tensors[f"{prefix}_suh"][slot, part],
            svh=slot_tensors[f"{prefix}_svh"][slot, part],
            mul1=True,
        )

    return (t("w13", 0), t("w13", 1)), t("w2", 0)


def _reference(x16, weights, remap, views):
    from sglang.srt.layers.quantization.exl3_ops import exl3_linear_reference

    out = torch.zeros((1, x16.shape[1]), dtype=torch.float32, device=x16.device)
    for k, slot in enumerate(remap.tolist()):
        (gate_t, up_t), down_t = views[slot]
        gate = exl3_linear_reference(x16, gate_t).clamp(max=ACT_LIMIT)
        up = exl3_linear_reference(x16, up_t).clamp(-ACT_LIMIT, ACT_LIMIT)
        h = F.silu(gate) * up * weights[k].float()
        out += exl3_linear_reference(h.to(torch.float16), down_t)
    return out


def _rel(y, ref):
    return float((y.float() - ref).norm() / ref.norm())


def test_exl3_moe_probe():
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext
    from sglang.srt.layers.quantization.exl3_ops import exl3_moe_loop

    ext = exl3_ext()
    missing = [n for n in ("exl3_moe", "exl3_moe_gather", "exl3_moe_max_concurrency") if not hasattr(ext, n)]
    assert not missing, f"extension lacks {missing}"
    device = torch.device("cuda", torch.cuda.current_device())
    slot_tensors = _load_slots(device)
    slots = len(EXPERTS)
    hidden = slot_tensors["w13_suh"].shape[-1]
    inter = slot_tensors["w2_suh"].shape[-1]
    bits = {
        "gate": slot_tensors["w13_trellis"].shape[-1] // 16,
        "up": slot_tensors["w13_trellis"].shape[-1] // 16,
        "down": slot_tensors["w2_trellis"].shape[-1] // 16,
    }
    tables = pointer_tables(slot_tensors, slots, device)
    bufs = make_buffers(ext, slots, hidden, inter, device)
    views = [_views(slot_tensors, s) for s in range(slots)]
    w13 = [v[0] for v in views]
    w2 = [v[1] for v in views]
    gen = torch.Generator(device="cpu").manual_seed(1234)
    report = {"layer": LAYER, "experts": EXPERTS, "bits": bits, "modes": {}}

    def inputs():
        remap = torch.randperm(slots, generator=gen)[:TOP_K].to(device)
        weights = torch.softmax(torch.randn(TOP_K, generator=gen), 0).to(device)
        x16 = (torch.randn((1, hidden), generator=gen) * 0.5).to(device, torch.float16)
        return x16, weights, remap

    route_sets = [inputs() for _ in range(ROUTE_SETS)]
    references = [_reference(x16, w, r, views) for x16, w, r in route_sets]
    loops = [
        exl3_moe_loop(x16, w.view(1, -1), r.view(1, -1), w13, w2, ACT_LIMIT).float()
        for x16, w, r in route_sets
    ]
    # BS1 top-6 slots are always distinct, so num_active = 6 is a static constant a graph
    # can bake in; -1 (all-fused, max concurrency) is the fallback (plan M1).
    for num_active in (TOP_K, -1):
        call = dict(act_limit=ACT_LIMIT, bits=bits, num_active=num_active)
        mode = {"route_sets": [], "replays": []}
        ok_parity = True
        for (x16, weights, remap), ref, loop in zip(route_sets, references, loops):
            fused = fused_moe_slots(ext, bufs, x16, weights, remap, tables, **call).clone()
            rel_fused, rel_loop = _rel(fused, ref), _rel(loop, ref)
            passed = rel_fused <= REL_BOUND and rel_fused <= 2 * rel_loop + 1e-3
            ok_parity &= passed
            mode["route_sets"].append({
                "remap": remap.tolist(),
                "rel_fused": rel_fused,
                "rel_loop": rel_loop,
                "max_abs_fused_vs_loop": float((fused - loop).abs().max()),
                "pass": passed,
            })
        # Capture over static inputs, then rewrite slot rows, remap, routes and input in place.
        x_s, w_s, r_s = inputs()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fused_moe_slots(ext, bufs, x_s, w_s, r_s, tables, **call)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_s = fused_moe_slots(ext, bufs, x_s, w_s, r_s, tables, **call)
        ok_replay = True
        for step in range(REWRITES):
            perm = torch.randperm(slots, generator=gen).to(device)
            for tensor in slot_tensors.values():
                tensor.copy_(tensor[perm].clone())
            x_new, w_new, r_new = inputs()
            x_s.copy_(x_new)
            w_s.copy_(w_new)
            r_s.copy_(r_new)
            graph.replay()
            replayed = out_s.clone()
            eager = fused_moe_slots(ext, bufs, x_s, w_s, r_s, tables, **call).clone()
            same = torch.equal(replayed, eager)
            ok_replay &= same
            mode["replays"].append({"step": step, "bitwise_equal": same, "max_abs": float((replayed - eager).abs().max())})
        # Rewrites permuted the slots: put the original rows back for the next mode.
        slot_tensors.update(_load_slots(device))
        for name, table in pointer_tables(slot_tensors, slots, device).items():
            tables[name].copy_(table)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            fused_moe_slots(ext, bufs, x_s, w_s, r_s, tables, **call)
        torch.cuda.synchronize()
        mode["eager_us"] = (time.perf_counter() - started) * 1e4
        started = time.perf_counter()
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        mode["replay_us"] = (time.perf_counter() - started) * 1e4
        mode["pass"] = ok_parity and ok_replay
        report["modes"][str(num_active)] = mode
        views = [_views(slot_tensors, s) for s in range(slots)]
    passing = [int(m) for m, v in report["modes"].items() if v["pass"]]
    # The fused MoE (Task 9) reads this choice: 6 when it passes, else -1.
    report["num_active"] = TOP_K if TOP_K in passing else (-1 if -1 in passing else None)
    report["verdict"] = "PASS" if report["num_active"] is not None else "FAIL"
    if OUT:
        with open(OUT, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps({"verdict": report["verdict"], "num_active": report["num_active"],
                      **{m: (v["pass"], v["eager_us"], v["replay_us"]) for m, v in report["modes"].items()}}))
    assert report["verdict"] == "PASS", report["modes"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s"]))
