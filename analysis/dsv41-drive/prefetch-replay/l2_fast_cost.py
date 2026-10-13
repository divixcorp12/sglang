"""Cost of the L+2 learned fast path (T0, T3) against the native scorer, under CUDA graphs at live widths 1/2/4/6.

Three configurations are captured as one CUDA graph per layer slot and replayed round-robin over the slots:
native (spec_score + spec_select), fused (spec_score + the spec_head kernels, whose tail selects) and torch
(spec_score + l2_fast_torch.score + select_cap1 on the bf16 fold). Every slot has its own target gate, x, scores and
routes, so a replay never finds the previous slot's gate in L2 (R11); the head weights are shared, as in serving.
incremental_ms is fused minus native. native_bandwidth_gbps is the gate bytes over the native median, an L2 check.

Inputs: an R1 run dir (stages/l2/result.json names the L2_T0 and L2_T3 checkpoints, verified by sha256); optionally
the model path, whose real target gates (bf16) replace the synthetic ones.
Output: --out cost.json {label, scope, widths, slots, runs, gpu, bar_ms_per_layer, models: {L2_T0|L2_T3:
        {checkpoint_sha256, native|fused|torch: {"<w>": {median_ms, p90_ms, p99_ms, runs, nodes}}, incremental_ms}},
        native_bandwidth_gbps: {"<w>": float}}.
Usage: flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 python l2_fast_cost.py --run-dir RUN_DIR
       --out /mnt/nvme1/nvfp4-work/f123-l2-fast-DATE/cost.json [--model-path MODEL] --runs 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable

import msgspec
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import l2_cap1_scorers  # noqa: E402
import l2_fast_common as common  # noqa: E402
import l2_fast_kernels  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_cost  # noqa: E402
import r1_gate  # noqa: E402
import r1_models  # noqa: E402
from b1_rows import TOPK  # noqa: E402

import sglang  # noqa: E402
from sglang.kernels.ops.moe import expert_stream_transport as es  # noqa: E402
from sglang.kernels.ops.moe import spec_head  # noqa: E402

MODELS = ("L2_T0", "L2_T3")
CONFIGS = ("native", "fused", "torch")
EXPERTS = spec_head.HEAD_EXPERTS
HORIZON = 2
L2_BYTES = common.L2_BYTES
# Arbitrary; the kernels' and torch path's cost does not depend on the threshold value.
THRESHOLD = 0.5
NATIVE_HOT = 16  # as l2_fast_interference.native_graphs and bench_exl3_ram_prefetch_score
SCOPE = ("CUDA graph replay on one stream (CUDA events per replay, replays queued in chunks behind a GPU sleep so "
         "host launch latency is excluded), one graph per layer slot replayed round-robin; "
         "each slot has its own bf16 target gate, x, scores and routes (gates rotate past the 96 MiB L2), head weights "
         "shared; median/p90/p99 over --runs replays per width; RTX 5090")
BANDWIDTH_SCOPE = ("gate + bias + x bytes of one spec_score over the native (score + select) median; a lower bound "
                   "on the score kernel's bandwidth, compared against the RTX 5090's ~1.8 TB/s HBM")


class Slot(msgspec.Struct):
    """One layer slot's serving inputs; nothing here is shared with another slot."""

    index: int
    gate: torch.Tensor  # bf16 [E, X_DIM]
    bias: torch.Tensor  # fp32 [E]
    x: torch.Tensor  # bf16 [width, X_DIM]
    scores: torch.Tensor  # fp32 [HEAD_MAX_TOKENS, E], written by spec_score
    ids: torch.Tensor  # int64 [width, TOPK], distinct per token
    weights: torch.Tensor  # fp32 [width, TOPK]
    positions: torch.Tensor  # int64 [width]
    skip: torch.Tensor  # bool [E]
    ram_slot: torch.Tensor  # int32 [1, E]: 0 where skip (mapped), else -1


class NativeSelect(msgspec.Struct):
    """spec_select's shared output and residency buffers (the page is pinned host memory)."""

    page: torch.Tensor
    state: torch.Tensor
    hot: torch.Tensor


def synthetic_gates(*, count: int, device: str, seed: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``count`` distinct (bf16 [E, X_DIM], fp32 [E]) gates, scaled like bench_exl3_ram_prefetch_score's."""
    g = torch.Generator(device=device).manual_seed(seed)
    return [((torch.randn((EXPERTS, r1_cost.X_DIM), generator=g, device=device) * 0.02).to(torch.bfloat16),
             torch.randn(EXPERTS, generator=g, device=device) * 0.1) for _ in range(count)]


def model_gates(model_path: str, *, rows: tuple[l2_fast_torch.FoldRow, ...],
                device: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The real target gates of ``rows``, the weight in bf16 (production's router dtype) and the bias fp32."""
    layers = [r.target_layer for r in rows]
    gates = r1_gate.load_target_gates(model_path, layer_ids=layers)
    return [(gates[t][0].to(device=device, dtype=torch.bfloat16).contiguous(),
             gates[t][1].to(device=device, dtype=torch.float32).contiguous()) for t in layers]


def build_slots(*, gates: list[tuple[torch.Tensor, torch.Tensor]], width: int, slots: int, device: str,
                seed: int = 1) -> list[Slot]:
    """``slots`` slots of ``width`` live tokens; slot i scores gate i % len(gates)."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(slots):
        gate, bias = gates[i % len(gates)]
        skip = torch.rand(EXPERTS, generator=g, device=device) < 0.25
        out.append(Slot(
            index=i, gate=gate, bias=bias,
            x=torch.randn((width, r1_cost.X_DIM), generator=g, device=device).to(torch.bfloat16),
            scores=torch.zeros((spec_head.HEAD_MAX_TOKENS, EXPERTS), dtype=torch.float32, device=device),
            ids=torch.stack([torch.randperm(EXPERTS, generator=g, device=device)[:TOPK] for _ in range(width)]),
            weights=torch.rand((width, TOPK), generator=g, device=device),
            positions=torch.arange(width, dtype=torch.int64, device=device),
            skip=skip, ram_slot=(skip.to(torch.int32) - 1).reshape(1, EXPERTS).contiguous()))
    return out


def new_native_select(*, device: str) -> NativeSelect:
    state = torch.zeros(len(es.STATE_WORDS), dtype=torch.int32, device=device)
    state[es.STATE_WORDS["posted"]] = 1
    return NativeSelect(page=es.new_candidate_page(pin=True), state=state,
                        hot=torch.full((NATIVE_HOT,), -1, dtype=torch.int64, device=device))


def _score(slot: Slot) -> None:
    es.run_spec_score(x=slot.x, w=slot.gate, bias=slot.bias, scores=slot.scores)


def native_step(slot: Slot, *, sel: NativeSelect) -> Callable[[], None]:
    def step() -> None:
        _score(slot)
        es.run_spec_select(slot.scores, slot.x.shape[0], top_k=TOPK, per_token=1, top_k_only=False,
                           hot_slots=sel.hot, hot_capacity=NATIVE_HOT, ram_slot=slot.ram_slot, target=0,
                           state=sel.state, candidates=sel.page)

    return step


def fused_step(slot: Slot, *, head: l2_fast_kernels.KernelHead, rows: int,
               threshold: torch.Tensor) -> Callable[[], None]:
    def step() -> None:
        _score(slot)
        head.launch(scores=slot.scores, x=slot.x, src_ids=slot.ids, src_w=slot.weights, positions=slot.positions,
                    row=slot.index % rows, skip=slot.skip, threshold=threshold)

    return step


def torch_step(slot: Slot, *, folded: l2_fast_torch.FoldedHead,
               threshold: torch.Tensor) -> Callable[[], tuple[torch.Tensor, ...]]:
    def step() -> tuple[torch.Tensor, ...]:
        _score(slot)
        block = l2_fast_torch.score(folded, scores=slot.scores, x=slot.x, src_ids=slot.ids, src_w=slot.weights,
                                    positions=slot.positions, row=slot.index % len(folded.rows))
        return l2_fast_torch.select_cap1(block, skip=slot.skip, threshold=threshold)

    return step


def bf16_fold(folded: l2_fast_torch.FoldedHead, *, device: str) -> l2_fast_torch.FoldedHead:
    """``folded`` on the device with every table stored bf16 (R1's bf16 cost path); score() computes in fp32."""
    parts = {name: getattr(folded, name).to(device=device, dtype=torch.bfloat16).contiguous()
             for name in folded.__struct_fields__ if name != "rows"}
    return msgspec.structs.replace(folded, **parts)


def time_steps(steps: list[Callable[[], object]], *, runs: int, warmup: int) -> dict:
    """{median_ms, p90_ms, p99_ms, runs, nodes}: one graph per step, replayed round-robin; nodes of one graph."""
    graphs = [r1_cost._capture(step) for step in steps]
    stats = common.replay_stats(graphs, runs=runs, warmup=warmup)
    return {**{k: stats[k] for k in ("median_ms", "p90_ms", "p99_ms", "runs")}, "nodes": common.graph_nodes(steps[0])}


def gate_gbps(slot: Slot, *, median_ms: float) -> float:
    moved = sum(t.numel() * t.element_size() for t in (slot.gate, slot.bias, slot.x))
    return moved / (median_ms * 1e-3) / 1e9


def rotated_gate_bytes(slots: list[Slot]) -> int:
    distinct = {s.gate.data_ptr(): s.gate.numel() * s.gate.element_size() for s in slots}
    return sum(distinct.values())


def _sweep_width(slots: list[Slot], *, heads: dict, sel: NativeSelect, threshold: torch.Tensor, runs: int,
                 warmup: int) -> tuple[dict, dict]:
    """(native stats, {model: {fused, torch}}) at one width."""
    native = time_steps([native_step(s, sel=sel) for s in slots], runs=runs, warmup=warmup)
    per_model = {}
    for name, (kernel, folded) in heads.items():
        fused = time_steps([fused_step(s, head=kernel, rows=len(folded.rows), threshold=threshold) for s in slots],
                           runs=runs, warmup=warmup)
        torch_path = time_steps([torch_step(s, folded=folded, threshold=threshold) for s in slots], runs=runs,
                                warmup=warmup)
        per_model[name] = {"fused": fused, "torch": torch_path}
    return native, per_model


def sweep(*, heads: dict[str, tuple[l2_fast_torch.FoldedHead, str]], gates: list[tuple[torch.Tensor, torch.Tensor]],
          widths: tuple[int, ...] = common.WIDTHS, slots: int = common.SLOTS, runs: int = 1000, warmup: int = 50,
          threshold: float = THRESHOLD, device: str = "cuda") -> dict:
    """The cost.json body (K-T8) for ``heads`` {scorer: (folded head, checkpoint sha256)}."""
    built = {name: (l2_fast_kernels.KernelHead(folded, device=device), bf16_fold(folded, device=device))
             for name, (folded, _sha) in heads.items()}
    sel = new_native_select(device=device)
    thr = torch.full((1,), threshold, dtype=torch.float32, device=device)
    models = {name: {"checkpoint_sha256": sha, "native": {}, "fused": {}, "torch": {}, "incremental_ms": {}}
              for name, (_folded, sha) in heads.items()}
    bandwidth, rotated = {}, {}
    with torch.no_grad():
        for w in widths:
            slot_list = build_slots(gates=gates, width=w, slots=slots, device=device, seed=w)
            native, per_model = _sweep_width(slot_list, heads=built, sel=sel, threshold=thr, runs=runs,
                                             warmup=warmup)
            key = str(w)
            bandwidth[key] = gate_gbps(slot_list[0], median_ms=native["median_ms"])
            rotated[key] = rotated_gate_bytes(slot_list)
            for name, got in per_model.items():
                m = models[name]
                m["native"][key], m["fused"][key], m["torch"][key] = native, got["fused"], got["torch"]
                m["incremental_ms"][key] = got["fused"]["median_ms"] - native["median_ms"]
    warnings = measurement_warnings(bandwidth=bandwidth, rotated=rotated)
    for line in warnings:
        print(f"WARNING: {line}", file=sys.stderr)
    return {
        "label": common.OBSERVED, "scope": SCOPE, "widths": list(widths), "slots": slots, "runs": runs,
        "warmup": warmup, "threshold": threshold, "gpu": torch.cuda.get_device_name(),
        "bar_ms_per_layer": common.BAR_MS_PER_LAYER, "models": models, "native_bandwidth_gbps": bandwidth,
        "native_bandwidth_scope": BANDWIDTH_SCOPE, "bandwidth_ceiling_gbps": common.DRAM_GBPS,
        "rotated_gate_bytes": rotated, "l2_bytes": L2_BYTES, "warnings": warnings,
    }


def measurement_warnings(*, bandwidth: dict[str, float], rotated: dict[str, int]) -> list[str]:
    """One line per width whose gates fit in L2 or whose implied bandwidth exceeds the card's DRAM spec."""
    out = []
    for key in rotated:
        if rotated[key] <= L2_BYTES:
            out.append(f"width {key}: rotated gates {rotated[key]} B fit in the {L2_BYTES} B L2; cost is cache-resident")
        if bandwidth[key] > common.DRAM_GBPS:
            out.append(f"width {key}: native implies {bandwidth[key]:.0f} GB/s, above the {common.DRAM_GBPS:.0f} "
                       "GB/s spec; impossible from DRAM")
    return out


def check_rotation(gates: list[tuple[torch.Tensor, torch.Tensor]], *, slots: int) -> None:
    """Refuse a real run whose gates would repeat within the slots or fit in L2 (R11)."""
    if len(gates) < slots:
        raise ValueError(f"{len(gates)} distinct gates for {slots} slots; slots would share a gate (R11)")
    total = sum(w.numel() * w.element_size() for w, _bias in gates[:slots])
    if total <= L2_BYTES:
        raise ValueError(f"rotated gates are {total} B, within the {L2_BYTES} B L2 (R11)")


def cost_rows(model: r1_models.R1Model, *, slots: int) -> tuple[l2_fast_torch.FoldRow, ...]:
    """Horizon-2 rows over the checkpoint's layer vocabulary (last ``slots`` targets, first stratum)."""
    vocab = model.head.vocab
    layers = set(vocab.layers)
    rows = [l2_fast_torch.FoldRow(source_layer=t - HORIZON, target_layer=t, stratum=vocab.strata[0])
            for t in vocab.layers if t - HORIZON in layers]
    return tuple(rows[-slots:])


def load_heads(*, checkpoints: dict[str, tuple[str, str]],
               slots: int) -> dict[str, tuple[l2_fast_torch.FoldedHead, str]]:
    """{scorer: (fp32 folded head, verified checkpoint sha256)} for T0 and T3."""
    out = {}
    for name in MODELS:
        path, sha = checkpoints[name]
        actual = l2_cap1_scorers.verify_ckpt(path, sha=sha)
        model = r1_models.load_r1(path)[0]
        out[name] = (l2_fast_torch.fold(model, rows=cost_rows(model, slots=slots)), actual)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", help="real target gates (bf16) instead of synthetic ones")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--slots", type=int, default=common.SLOTS)
    args = ap.parse_args(argv)
    out = common.guard_out(args.out, run_dir=args.run_dir)
    print(f"sglang.__file__ = {sglang.__file__}")
    with open(os.path.join(args.run_dir, "stages", "l2", "result.json")) as f:
        checkpoints = l2_cap1_scorers.learned_ckpts(json.load(f))
    heads = load_heads(checkpoints=checkpoints, slots=args.slots)
    if args.model_path:
        gates = model_gates(args.model_path, rows=heads[MODELS[0]][0].rows, device="cuda")
    else:
        gates = synthetic_gates(count=args.slots, device="cuda")
    check_rotation(gates, slots=args.slots)
    result = sweep(heads=heads, gates=gates, slots=args.slots, runs=args.runs, warmup=args.warmup)
    result["gate_source"] = args.model_path or "synthetic bf16 N(0, 0.02^2), one per slot"
    result["sglang_file"] = sglang.__file__
    common.write_json(out, result)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
