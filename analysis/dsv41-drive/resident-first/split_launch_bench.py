"""Resident-first step 2: the cost of splitting the in-graph exl3_moe launch in two, at M=1 (BS1 decode).

Plan: docs/superpowers/plans/2026-09-25-dsv41-per-expert-compute.md, section 6 step 2.

Each arm is one CUDA graph of 40 layers, the way the decode graph issues them:

- ``one``: the production call, ``Exl3FusedMoE.run`` with layer fusion (route-tables kernel, exl3_moe, gather).
- ``two_static/<r>+<m>``: the route-tables kernel for the full-route placement, exl3_moe over the resident mask,
  exl3_moe over the missed mask, gather. The masks, compacted weights and slot kind sit in static buffers: this is
  what an extended route-tables kernel (plan 3.4) gives for free.
- ``two_torch/<r>+<m>``: as two_static, with the masks built by the torch ops of the parity test
  (``split_route_tables``), an upper bound on the route-table work.
- ``res_only`` / ``miss_only``: two_static with one of its two launches, to split the cost.
- ``tables_only``: two_static with neither launch (route tables + gather), the floor under every arm.
- ``miss_wide/<r>+<m>``: miss_only with num_active = m (wider expert groups). Not bitwise with the single launch
  (the parity test's ``test_num_active_must_stay_six``); measured only to bound what a non-identical variant saves.

The split is the one ``test_exl3_moe_split_parity_cuda.py`` proves bitwise.

Working set: every layer owns PHYS distinct real rows (copies of real EXL3 expert rows of one layer) behind a
SLOTS-wide pointer table (entry s -> row s % PHYS), and a token touches 40 x 6 rows (~3.2 GB), far past the 96 MiB L2.
The implied HBM bandwidth is printed as the sanity check (CLAUDE.md, GPU microbenchmarks).

Timing: replays alternate between arms round-robin; each replay is timed with CUDA events; medians are reported.
Run on divix01 under gpu-run.sh with PYTHONPATH at the tree under test and SGLANG_EXL3_SRC set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys

import msgspec
import torch

LAYERS = 40
TOP_K = 6
PHYS = 8  # distinct rows per layer
SLOTS = 46  # pointer-table width per layer (production: capacity + scratch rows)
SPLITS = ((6, 0), (5, 1), (4, 2), (3, 3))


def _parity_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "..", "test", "manual", "dsv41", "test_exl3_moe_split_parity_cuda.py")
    spec = importlib.util.spec_from_file_location("split_parity", os.path.normpath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Layer(msgspec.Struct):
    fused: object
    state: object
    rows: dict
    x: torch.Tensor
    weights: torch.Tensor
    remap: torch.Tensor
    keep: torch.Tensor
    hits: dict  # split -> bool [TOP_K] route mask


def build_layers(par, real: dict, device, gen) -> list[Layer]:
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.exl3_fused_moe import Exl3FusedMoE

    n_real = real["w13_trellis"].shape[0]
    hidden = real["w13_suh"].shape[-1]
    inter = real["w2_suh"].shape[-1]
    layers = []
    for layer in range(LAYERS):
        pick = [(layer * PHYS + i) % n_real for i in range(PHYS)]
        rows = {name: t[pick].clone() for name, t in real.items()}
        with envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(True):
            fused = Exl3FusedMoE(rows, PHYS, hidden=hidden, inter=inter, top_k=TOP_K, device=device)
        # Widen to SLOTS table entries: entry s holds row s % PHYS, so a route's slot scan covers production's width.
        for name, table in fused.tables.items():
            fused.tables[name] = table[torch.arange(SLOTS, device=device) % PHYS].contiguous()
        fused.slots = SLOTS
        fused.expert_count = torch.zeros(SLOTS + 1, dtype=torch.int64, device=device)
        fused.det = torch.zeros((3, SLOTS + 1), dtype=torch.int64, device=device)
        # Six routes on distinct physical rows, spread over the table.
        residues = torch.randperm(PHYS, generator=gen)[:TOP_K].tolist()
        remap = torch.tensor(
            [r + PHYS * int(torch.randint(0, (SLOTS - 1 - r) // PHYS + 1, (1,), generator=gen)) for r in residues]
        )
        hits = {}
        for r, m in SPLITS:
            hit = torch.zeros(TOP_K, dtype=torch.bool)
            hit[torch.randperm(TOP_K, generator=gen)[:r]] = True
            hits[(r, m)] = hit.to(device)
        layers.append(
            Layer(
                fused=fused,
                state=par.SplitState.empty(SLOTS, device),
                rows=rows,
                x=(torch.randn((1, hidden), generator=gen) * 0.5).to(device, torch.bfloat16),
                weights=torch.softmax(torch.randn(TOP_K, generator=gen), 0).to(device),
                remap=remap.to(device, torch.int32),
                keep=torch.ones(1, dtype=torch.float32, device=device),
                hits=hits,
            )
        )
    return layers


def one(par, L: Layer):
    L.fused.run(L.x, L.weights, L.remap, L.keep, par.ACT_LIMIT)


def prepare_static(par, L: Layer, split):
    """Fill the split's static buffers (masks, compacted weights, slot kind) outside the graph."""
    hit = L.hits[split]
    remap = L.remap.long()
    par.split_route_tables(remap, L.weights, hit, L.state.resident)
    par.split_route_tables(remap, L.weights, ~hit, L.state.missed)
    L.state.missed.count.mul_((L.keep > 0).long())
    ones = torch.ones(TOP_K, dtype=torch.int64, device=L.x.device)
    count = torch.zeros(SLOTS + 1, dtype=torch.int64, device=L.x.device).index_add_(0, remap, ones)
    L.state.kind.copy_((count > 0).long() * (L.keep > 0).long())


def two(par, L: Layer, split, *, torch_tables: bool, launches=("res", "miss"), miss_active: int = 6):
    f, st = L.fused, L.state
    # Full-route placement: the fused route-tables kernel with keep = 1 (the resident launch precedes F).
    remap64, inv_order, weight_full, det = f._fused_route_tables(L.x, L.weights, L.remap, st.ones_keep)
    if torch_tables:
        hit = L.hits[split]
        par.split_route_tables(remap64, L.weights, hit, st.resident)
        par.split_route_tables(remap64, L.weights, ~hit, st.missed)
    if "res" in launches:
        par.launch(f, f.x16, f.out, st.resident.count, st.resident.weights, det[0])
    if torch_tables:
        kept = (L.keep > 0).long()
        st.missed.count.mul_(kept)
        torch.mul(det[2], kept, out=st.kind)
    if "miss" in launches:
        par.launch(f, f.x16, f.out, st.missed.count, st.missed.weights, det[0], miss_active)
    s = f.slots
    f.ext.exl3_moe_gather(f.out, f.scratch, remap64, inv_order, det[1, :s], det[0, :s], st.kind[:s], weight_full)


def capture(fn, layers):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            for L in layers:
                fn(L)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for L in layers:
            fn(L)
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=300, help="timed replays per arm")
    ap.add_argument("--out", default=None, help="JSON report path")
    args = ap.parse_args()

    par = _parity_module()
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    exl3_ext()
    device = torch.device("cuda", torch.cuda.current_device())
    real = par.load_slot_rows(device)
    row_bytes = sum(t[0].numel() * t.element_size() for t in real.values())
    gen = torch.Generator().manual_seed(2026)
    layers = build_layers(par, real, device, gen)
    del real
    touched = LAYERS * TOP_K * row_bytes
    print(f"row {row_bytes / 1e6:.2f} MB; per token {touched / 1e9:.2f} GB over {LAYERS} layers; "
          f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    arms = {"one": capture(lambda L: one(par, L), layers)}
    for L in layers:
        prepare_static(par, L, SPLITS[1])
    arms["tables_only"] = capture(lambda L: two(par, L, SPLITS[1], torch_tables=False, launches=()), layers)
    for split in SPLITS[1:]:
        tag = f"{split[0]}+{split[1]}"
        for L in layers:
            prepare_static(par, L, split)
        # Static buffers are per layer and shared by every split's graph: capture each split's static graphs
        # only after its buffers are filled, and refill before timing (see the timed loop).
        arms[f"two_static/{tag}"] = capture(lambda L, s=split: two(par, L, s, torch_tables=False), layers)
        arms[f"res_only/{tag}"] = capture(lambda L, s=split: two(par, L, s, torch_tables=False, launches=("res",)), layers)
        arms[f"miss_only/{tag}"] = capture(lambda L, s=split: two(par, L, s, torch_tables=False, launches=("miss",)), layers)
        arms[f"miss_wide/{tag}"] = capture(
            lambda L, s=split: two(par, L, s, torch_tables=False, launches=("miss",), miss_active=s[1]), layers
        )
        arms[f"two_torch/{tag}"] = capture(lambda L, s=split: two(par, L, s, torch_tables=True), layers)
    split_of = {name: tuple(int(v) for v in name.split("/")[1].split("+")) for name in arms if "/" in name}

    times = {name: [] for name in arms}
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    names = list(arms)
    current = None
    for rep in range(args.replays + 10):
        for i in range(len(names)):
            name = names[(i + rep) % len(names)]
            want = split_of.get(name)
            if want and want != current:
                for L in layers:
                    prepare_static(par, L, want)
                current = want
            start.record()
            arms[name].replay()
            stop.record()
            stop.synchronize()
            if rep >= 10:
                times[name].append(start.elapsed_time(stop) * 1e3)  # us per token

    report = {"layers": LAYERS, "slots": SLOTS, "phys_rows_per_layer": PHYS, "row_bytes": row_bytes,
              "replays": args.replays, "arms": {}}
    base = statistics.median(times["one"])
    for name, ts in times.items():
        ts = sorted(ts)
        med = statistics.median(ts)
        report["arms"][name] = {
            "median_us_token": med,
            "p10_us_token": ts[len(ts) // 10],
            "p90_us_token": ts[9 * len(ts) // 10],
            "median_us_layer": med / LAYERS,
            "extra_us_layer_vs_one": (med - base) / LAYERS,
            "extra_us_token_vs_one": med - base,
        }
    report["implied_hbm_gbs_one"] = touched / (base * 1e-6) / 1e9
    for name, r in report["arms"].items():
        print(f"{name:18s} {r['median_us_token']:9.1f} us/token  {r['median_us_layer']:7.2f} us/layer  "
              f"extra {r['extra_us_layer_vs_one']:+7.2f} us/layer {r['extra_us_token_vs_one']:+8.1f} us/token  "
              f"(p10 {r['p10_us_token']:.1f}, p90 {r['p90_us_token']:.1f})")
    print(f"implied bandwidth, one: {report['implied_hbm_gbs_one']:.0f} GB/s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
