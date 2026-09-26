"""Resident-first step 1 gate: one exl3_moe launch over six routes against two masked launches, bitwise (GPU).

Plan: docs/superpowers/plans/2026-09-25-dsv41-per-expert-compute.md, sections 3.2, 3.3 and 6 step 1.

The reference is the production call, ``Exl3FusedMoE.run``, with layer fusion off and on. The split is:

1. the full-route placement: ``route_tables`` with keep = 1, so ``det[0]`` (every route's scratch row), ``inv_order``
   and ``det[2]`` come from all six routes, whatever the masks are;
2. ``exl3_moe`` over the resident mask, then ``exl3_moe`` over the missed mask (its count zeroed when keep is 0,
   because F runs before it), both with the full-route ``det[0]`` and the same ``NUM_ACTIVE`` = 6;
3. one ``exl3_moe_gather`` whose slot kind is ``det[2] * (keep > 0)``.

Two facts from exllamav3's source shape this, and ``test_full_weight_table_misplaces_masked_weights`` pins the first:

- ``exl3_moe`` does not read a start table for its inputs. It walks ``expert_count`` and keeps a running prefix
  (``start = end; end += expert_count[e]``) and reads ``weight_sorted[start + row]``. Only the scratch row comes from
  ``fused_base`` (``det[0]``). So a masked launch needs its own compacted weight table (the j-th masked route's
  weight at index j); the full-route ``weight_sorted`` gives masked routes their neighbours' weights.
- ``exl3_moe`` applies the route weight (``had_d_out`` scales by ``weight``); the gather with slot kind 1 does not.
  keep therefore cannot ride in the resident launch's weights, which are fixed before F. It goes into the gather's
  slot kind instead, which is what a dropped layer's zeroed count does today.

Run on divix01 under gpu-run.sh with PYTHONPATH at the tree under test, SGLANG_EXL3_SRC set and the EXL3 checkpoint at
DSV41_EXL3_DIR.
"""

import os

import msgspec
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC")),
    reason="needs a GPU and SGLANG_EXL3_SRC (an exllamav3 checkout)",
)

EXL3_DIR = os.environ.get("DSV41_EXL3_DIR", "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw")
LAYER = int(os.environ.get("DSV41_PROBE_LAYER", "3"))
EXPERTS = list(range(0, 384, 32))  # 12 real experts -> 12 slots
TOP_K = 6
ACT_LIMIT = 10.0
TRIALS = 4


def load_slot_rows(device, layer: int = LAYER, experts=EXPERTS) -> dict:
    """Real EXL3 expert rows of one layer, in hot-cache slot layout (as the P2 probe reads them)."""
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    layout = build_exl3_expert_layout(EXL3_DIR)
    fmt = Exl3ExpertFormat(layout, layer, direct=True)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    host = {name: torch.empty((len(experts),) + spec.row_shape, dtype=spec.dtype) for name, spec in specs.items()}
    source = Exl3ShardRowSource.for_layer(layout, layer, fmt.segment_map(), direct=True)
    source.read(torch.tensor(experts, dtype=torch.long), host)
    return {name: tensor.to(device) for name, tensor in host.items()}


class SplitBuffers(msgspec.Struct):
    """One launch's work mask: route counts per slot and the masked routes' weights, compacted in slot order."""

    count: torch.Tensor  # int64 [slots + 1]
    weights: torch.Tensor  # fp16 [TOP_K]; index j = the j-th masked route in slot order
    spill: torch.Tensor  # fp16 [TOP_K + 1]; scatter target, the last element absorbs unmasked routes

    @classmethod
    def empty(cls, slots: int, device) -> "SplitBuffers":
        return cls(
            count=torch.zeros(slots + 1, dtype=torch.int64, device=device),
            weights=torch.zeros(TOP_K, dtype=torch.float16, device=device),
            spill=torch.zeros(TOP_K + 1, dtype=torch.float16, device=device),
        )


def split_route_tables(remap, weights, mask, bufs: SplitBuffers) -> None:
    """Fill ``bufs`` for the routes where ``mask`` (bool [TOP_K]) is set; capture-safe torch ops.

    The weights carry no keep: the resident launch runs before F writes it.
    """
    order = torch.argsort(remap)
    masked = mask[order]
    pos = torch.cumsum(masked.long(), 0) - 1
    idx = torch.where(masked, pos, torch.full_like(pos, TOP_K))
    bufs.spill.zero_().scatter_(0, idx, weights[order].float().to(torch.float16))
    bufs.weights.copy_(bufs.spill[:TOP_K])
    bufs.count.zero_().index_add_(0, remap, mask.long())


def launch(fused, x16, out, count, weight_sorted, det0, num_active: int = 6) -> None:
    """One exl3_moe over ``fused``'s slot tables and temps, deterministic path, as Exl3FusedMoE.run issues it."""
    from sglang.srt.layers.quantization.exl3_fused_moe import ACT_SILU, ROW_TILE

    t = fused.tables
    fused.ext.exl3_moe(
        x16, out, count, fused.token_sorted, weight_sorted,
        fused.temp_state_g, fused.temp_state_u, fused.temp_intermediate_g, fused.temp_intermediate_u,
        ACT_SILU, fused.bits["gate"], fused.bits["up"], fused.bits["down"],
        t["gate_trellis"], t["gate_suh"], t["gate_svh"],
        t["up_trellis"], t["up_suh"], t["up_svh"],
        t["down_trellis"], t["down_suh"], t["down_svh"],
        False, True, False, True, False, True,
        ACT_LIMIT, num_active, fused.scratch, det0, 1, ROW_TILE, 16,
    )


class SplitState(msgspec.Struct):
    """The split path's per-layer buffers beside an Exl3FusedMoE (whose x16, out, scratch and det it uses)."""

    full_count: torch.Tensor
    ones_keep: torch.Tensor
    resident: SplitBuffers
    missed: SplitBuffers
    kind: torch.Tensor

    @classmethod
    def empty(cls, slots: int, device) -> "SplitState":
        return cls(
            full_count=torch.zeros(slots + 1, dtype=torch.int64, device=device),
            ones_keep=torch.ones(1, dtype=torch.float32, device=device),
            resident=SplitBuffers.empty(slots, device),
            missed=SplitBuffers.empty(slots, device),
            kind=torch.zeros(slots + 1, dtype=torch.int64, device=device),
        )


def split_run(fused, state: SplitState, x, weights, remap, hit, keep, *, between=None, num_active: int = 6):
    """Resident launch, then missed launch, then one gather; ``between(stage)`` observes the buffers between steps."""
    from sglang.srt.layers.quantization.exl3_fused_moe import route_tables

    fused.x16.copy_(x)
    # Placement from every route. keep = 1 here: the resident launch cannot know keep.
    inv_order, weight_full, det = route_tables(remap, state.full_count, fused.ones, weights, state.ones_keep)
    split_route_tables(remap, weights, hit, state.resident)
    split_route_tables(remap, weights, ~hit, state.missed)
    fused.out.zero_()
    if between:
        between("tables", det=det)
    launch(fused, fused.x16, fused.out, state.resident.count, state.resident.weights, det[0], num_active)
    if between:
        between("resident", det=det)
    # F has run by now: a dropped layer runs no missed expert (its rows may be half written).
    kept = (keep > 0).long()
    state.missed.count.mul_(kept)
    launch(fused, fused.x16, fused.out, state.missed.count, state.missed.weights, det[0], num_active)
    if between:
        between("missed", det=det)
    torch.mul(det[2], kept, out=state.kind)
    s = fused.slots
    fused.ext.exl3_moe_gather(fused.out, fused.scratch, remap, inv_order, det[1, :s], det[0, :s], state.kind[:s], weight_full)
    return fused.out


def _fused(slot_rows, device, layer_fusion: bool):
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.exl3_fused_moe import Exl3FusedMoE

    slots = slot_rows["w13_trellis"].shape[0]
    with envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(layer_fusion):
        return Exl3FusedMoE(
            slot_rows,
            slots,
            hidden=slot_rows["w13_suh"].shape[-1],
            inter=slot_rows["w2_suh"].shape[-1],
            top_k=TOP_K,
            device=device,
        )


@pytest.fixture(scope="module")
def slot_rows():
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    ext = exl3_ext()
    missing = [n for n in ("exl3_moe", "exl3_moe_gather", "exl3_moe_max_concurrency") if not hasattr(ext, n)]
    assert not missing, f"extension lacks {missing}"
    return load_slot_rows(torch.device("cuda", torch.cuda.current_device()))


def _inputs(gen, slots, hidden, device, hits: int):
    remap = torch.randperm(slots, generator=gen)[:TOP_K].to(device)
    weights = torch.softmax(torch.randn(TOP_K, generator=gen), 0).to(device)
    x = (torch.randn((1, hidden), generator=gen) * 0.5).to(device)
    hit = torch.zeros(TOP_K, dtype=torch.bool)
    hit[torch.randperm(TOP_K, generator=gen)[:hits]] = True
    return x, weights, remap, hit.to(device)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32)


def _bench_module():
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "..", "analysis", "dsv41-drive", "resident-first", "split_launch_bench.py")
    spec = importlib.util.spec_from_file_location("split_launch_bench", os.path.normpath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("layer_fusion", [False, True])
def test_split_launch_is_bitwise_the_single_launch(slot_rows, layer_fusion):
    device = slot_rows["w13_trellis"].device
    ref = _fused(slot_rows, device, layer_fusion)
    split = _fused(slot_rows, device, False)
    state = SplitState.empty(split.slots, device)
    hidden = slot_rows["w13_suh"].shape[-1]
    gen = torch.Generator().manual_seed(4242 + layer_fusion)
    nan_bits = torch.full((), float("nan"), dtype=torch.float32).view(torch.int32).item()
    cases = 0
    for keep_value in (1.0, 0.0):
        keep = torch.tensor([keep_value], device=device)
        for hits in range(TOP_K + 1):
            for trial in range(TRIALS):
                x, weights, remap, hit = _inputs(gen, split.slots, hidden, device, hits)
                want = ref.run(x, weights, remap, keep, ACT_LIMIT).clone()
                want_scratch = ref.scratch.clone()
                label = f"keep={keep_value} hits={hits} trial={trial} remap={remap.tolist()} hit={hit.tolist()}"

                # Poison every row the split must write, so a row it skips cannot pass on stale bytes.
                split.scratch.fill_(float("nan"))
                rank = torch.argsort(torch.argsort(remap))  # route -> its scratch row (slot order)
                seen = {}

                def between(stage, det):
                    torch.cuda.synchronize()
                    assert torch.equal(split.out, torch.zeros_like(split.out)), f"{label}: exl3_moe wrote out ({stage})"
                    rows = _bits(split.scratch)
                    for route in range(TOP_K):
                        r = int(rank[route])
                        # ranks are the full-route placement, whatever the mask
                        assert int(det[0, int(remap[route])]) == r, f"{label}: placement moved"
                        written = not bool((rows[r] == nan_bits).all())
                        resident = bool(hit[route])
                        expect = {
                            "tables": False,
                            "resident": resident,
                            "missed": resident or keep_value > 0,
                        }[stage]
                        assert written == expect, f"{label}: route {route} row {r} written={written} after {stage}"
                        if written and keep_value > 0:
                            assert torch.equal(rows[r], _bits(want_scratch)[r]), f"{label}: route {route} row differs"
                    seen[stage] = True

                got = split_run(split, state, x, weights, remap, hit, keep, between=between)
                torch.cuda.synchronize()
                assert seen.keys() == {"tables", "resident", "missed"}
                assert torch.equal(_bits(got), _bits(want)), (
                    f"{label}: max abs {float((got - want).abs().max())}"
                )
                cases += 1
    assert cases == 2 * (TOP_K + 1) * TRIALS
    print(f"layer_fusion={layer_fusion}: {cases} cases bitwise equal")


def test_full_weight_table_misplaces_masked_weights(slot_rows):
    """The hazard the compacted weights avoid: a masked launch indexes weight_sorted by its own running prefix."""
    from sglang.srt.layers.quantization.exl3_fused_moe import route_tables

    device = slot_rows["w13_trellis"].device
    ref = _fused(slot_rows, device, False)
    split = _fused(slot_rows, device, False)
    state = SplitState.empty(split.slots, device)
    hidden = slot_rows["w13_suh"].shape[-1]
    keep = torch.ones(1, device=device)
    gen = torch.Generator().manual_seed(7)
    x, _, remap, _ = _inputs(gen, split.slots, hidden, device, 0)
    weights = torch.tensor([0.30, 0.25, 0.20, 0.12, 0.08, 0.05], device=device)
    rank = torch.argsort(torch.argsort(remap))
    hit = (rank % 2 == 1)  # routes in slot-order positions 1, 3, 5: each has an unmasked route before it
    want = ref.run(x, weights, remap, keep, ACT_LIMIT).clone()

    split.x16.copy_(x)
    inv_order, weight_full, det = route_tables(remap, state.full_count, split.ones, weights, keep)
    split_route_tables(remap, weights, hit, state.resident)
    split_route_tables(remap, weights, ~hit, state.missed)
    split.out.zero_()
    for bufs in (state.resident, state.missed):
        launch(split, split.x16, split.out, bufs.count, weight_full, det[0])  # full table: the wrong one
    s = split.slots
    split.ext.exl3_moe_gather(split.out, split.scratch, remap, inv_order, det[1, :s], det[0, :s], det[2, :s], weight_full)
    torch.cuda.synchronize()
    assert not torch.equal(split.out, want), "masked launches read their weights through the full-route start"


def test_num_active_must_stay_six(slot_rows):
    """Record whether a masked launch sized by its own active count (wider groups) changes the bytes.

    exl3_moe sizes group_size = min(num_sms / num_active, 32) from num_active, and a group's split-K slicing
    (exl3_gemm_kernel_inner: slice_beg = tiles * blockIdx.x / gridDim.x) follows the group width, so the partial sums
    reduce in a different grouping. The split therefore passes NUM_ACTIVE = 6 to both launches; this test only reports
    what the alternative does.
    """
    device = slot_rows["w13_trellis"].device
    ref = _fused(slot_rows, device, False)
    split = _fused(slot_rows, device, False)
    state = SplitState.empty(split.slots, device)
    hidden = slot_rows["w13_suh"].shape[-1]
    keep = torch.ones(1, device=device)
    gen = torch.Generator().manual_seed(11)
    differs = 0
    for _ in range(4):
        x, weights, remap, hit = _inputs(gen, split.slots, hidden, device, 4)
        want = ref.run(x, weights, remap, keep, ACT_LIMIT).clone()
        got = split_run(split, state, x, weights, remap, hit, keep, num_active=4).clone()
        torch.cuda.synchronize()
        differs += not torch.equal(got, want)
    print(f"num_active=4 on a 4+2 split: {differs}/4 route sets differ from the single launch")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s"]))


def test_the_bench_copy_lands_in_the_rows_the_missed_launch_reads(slot_rows):
    """The resident-first bench's copy arms are only meaningful if copy_missed writes the rows the missed launch
    reads: poisoned sources must change the layer output, and the true bytes must restore it bitwise."""
    bench = _bench_module()
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator().manual_seed(7)
    par = bench._parity_module()
    layers = bench.build_layers(par, slot_rows, device, gen)[:1]
    L = layers[0]
    bench.one(par, L)
    want = L.fused.out.clone()
    bench.add_pinned_sources(layers, (4, 2))
    true_bytes = {n: t.clone() for n, t in L.pinned.items()}
    for t in L.pinned.values():
        t.zero_()
    bench.copy_missed(L)
    bench.one(par, L)
    torch.cuda.synchronize()
    assert not torch.equal(_bits(L.fused.out), _bits(want)), "poisoned rows did not reach the launch"
    for n, t in L.pinned.items():
        t.copy_(true_bytes[n])
    bench.copy_missed(L)
    bench.one(par, L)
    torch.cuda.synchronize()
    assert torch.equal(_bits(L.fused.out), _bits(want))
