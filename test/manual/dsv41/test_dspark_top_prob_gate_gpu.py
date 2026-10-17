"""DSpark's top-probability verify gate on the GPU: the in-graph top probability of the sharded greedy proposal, the
static epilogue's capped accept, and a captured 6-token verify whose cut rows route no experts through the EXL3 lease
chain.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_ram_miss_graph_gpu import HIDDEN, INTER, REL_BOUND, _reference, _rel  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

ACT_LIMIT = 10.0
TOP_K, TOKENS, LANES, EXPERTS = 6, 6, 8, 48
SINGLE_RANK = SimpleNamespace(world_size=1)


def test_the_sharded_greedy_step_reduces_the_softmax_top_probability_with_its_argmax():
    from sglang.kernels.ops.speculative.dspark.sharded_greedy import sharded_greedy_step

    generator = torch.Generator(device="cuda").manual_seed(3)
    rows, vocab = 4, 129_280  # the V4.1 vocabulary: 32 blocks, the last one partial
    base = torch.randn((rows, vocab), device="cuda", generator=generator) * 4
    bias = (torch.randn((rows, vocab), device="cuda", generator=generator) * 2).to(torch.bfloat16)
    base[1, 1000] = 60.0  # a near-certain row
    base[2, 7:4000] = -float("inf")  # masked entries, a whole block of them included
    base[2, 4096:8192] = -float("inf")
    want_tokens = torch.argmax(base + bias.float(), dim=-1)
    want_probs = torch.softmax(base + bias.float(), dim=-1).amax(dim=-1)

    plain = sharded_greedy_step(bias, base, group=SINGLE_RANK, vocab_start=0)
    # A column of a [rows, gamma] buffer, as the draft sampler passes it.
    probs = torch.full((rows, 5), -1.0, device="cuda")
    tokens = sharded_greedy_step(bias, base, group=SINGLE_RANK, vocab_start=0, top_prob_out=probs[:, 2])
    assert torch.equal(tokens, plain) and torch.equal(tokens, want_tokens)
    assert torch.allclose(probs[:, 2], want_probs, rtol=1e-4, atol=1e-7)
    assert bool((probs[:, [0, 1, 3, 4]] == -1.0).all())
    assert float(probs[1, 2]) > 0.999

    base[3, 17] = float("nan")
    sharded_greedy_step(bias, base, group=SINGLE_RANK, vocab_start=0, top_prob_out=probs[:, 2])
    assert torch.isnan(probs[3, 2]) and not bool(torch.isnan(probs[:3, 2]).any())


def test_the_static_epilogue_stops_the_accept_at_the_gates_cut():
    from sglang.srt.speculative.dspark_components.dspark_verify import DsparkVerifyEpilogue
    from sglang.srt.speculative.spec_tp_sync import SpecTpSync

    stride, vocab = TOKENS, 4096
    epilogue = DsparkVerifyEpilogue(
        max_bs=1, verify_num_draft_tokens=stride, device="cuda", tp_sync=SpecTpSync(SINGLE_RANK), static_cutoff=True
    )
    drafts = [7, 3, 9, 2, 5]
    predict = drafts + [11]  # the target agrees with every draft
    logits = torch.full((stride, vocab), -10.0, device="cuda")
    logits[torch.arange(stride), torch.tensor(predict)] = 10.0
    batch = SimpleNamespace(
        batch_size=1,
        input_ids=torch.tensor([1] + drafts, device="cuda"),
        seq_lens=torch.tensor([100], device="cuda"),
    )

    def accept():
        epilogue._static_epilogue(SimpleNamespace(next_token_logits=logits), batch)
        out = epilogue.read_accept(1)
        return int(out.correct_len), int(out.bonus), int(out.commit_lens), out.out_tokens[0].tolist()

    # Three rows verified: the anchor and drafts 1-2. Draft 3 matched but was cut, so the bonus after draft 2 is the
    # target's prediction there (draft 3's token), and the commit is drafts 1-2 plus that bonus.
    epilogue.begin_gated_static_step(torch.tensor([3], device="cuda"), armed=False)
    correct_len, bonus, commit_len, out_tokens = accept()
    assert (correct_len, bonus, commit_len, out_tokens[:commit_len]) == (2, 9, 3, [7, 3, 9])
    epilogue.begin_static_step(1, armed=False)  # an ungated step verifies every draft again
    correct_len, bonus, commit_len, out_tokens = accept()
    assert (correct_len, bonus, commit_len, out_tokens) == (5, 11, 6, drafts + [11])


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_the_one_kernel_dead_row_masking_matches_its_torch_definition(id_dtype):
    from sglang.srt.layers.moe.verify_dead_rows import VerifyDeadRows, mask_dead_rows_for_graph

    generator = torch.Generator(device="cpu").manual_seed(5)
    ids = torch.randint(0, 384, (TOKENS, TOP_K), generator=generator).to(id_dtype)
    weights = torch.rand((TOKENS, TOP_K), generator=generator)
    x = torch.randn((TOKENS, 7168), generator=generator).to(torch.bfloat16)
    for live in (1, 3, TOKENS):
        got_routes, want_routes = torch.full((64,), 7, dtype=torch.long, device="cuda"), torch.full((64,), 7)
        dead_cuda, dead_cpu = VerifyDeadRows(max_rows=TOKENS, device="cuda"), VerifyDeadRows(max_rows=TOKENS, device="cpu")
        dead_cuda.set_live_rows(torch.tensor([live], device="cuda"))
        dead_cpu.set_live_rows(torch.tensor([live]))
        got = mask_dead_rows_for_graph(
            topk_ids=ids.cuda(), topk_weights=weights.cuda(), dead=dead_cuda.dead, routes=got_routes, x=x.cuda()
        )
        want = mask_dead_rows_for_graph(
            topk_ids=ids, topk_weights=weights, dead=dead_cpu.dead, routes=want_routes, x=x
        )
        for g, w in zip(got, want):
            assert g.dtype == w.dtype and torch.equal(g.cpu(), w), live
        assert torch.equal(got_routes.cpu(), want_routes), live


@pytest.mark.parametrize("live", [1, 2, TOKENS])
def test_the_dedup_plan_leaves_dead_rows_out_of_the_route_counts_and_nothing_else(live):
    """A verify's masked routes (dead rows repeat row 0's experts) through the dedup plan with the dead mask: the
    residency route counts are the live rows' alone, and the remap, the scratch rows and every graph and outcome
    counter equal the plan without the mask. With no row dead the mask is the identity. Mutant: count a dead route
    -- red for live < 6."""
    from sglang.srt.layers.moe.expert_route_plan import plan_graph_routes_fused
    from sglang.srt.layers.moe.verify_dead_rows import VerifyDeadRows, mask_dead_routes

    generator = torch.Generator(device="cpu").manual_seed(9)
    raw = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(TOKENS)])
    dead_rows = VerifyDeadRows(max_rows=TOKENS, device="cuda")
    dead_rows.set_live_rows(torch.tensor([live], device="cuda"))
    ids, _ = mask_dead_routes(
        topk_ids=raw.cuda(), topk_weights=torch.ones(raw.shape, device="cuda"), dead=dead_rows.dead
    )
    expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.int64, device="cuda")
    expert_to_slot[::3] = torch.arange(EXPERTS // 3, device="cuda")  # a third of the experts hot
    routes = TOKENS * TOP_K

    def plan(dead):
        out = dict(
            source_rows_out=torch.full((routes,), -1, dtype=torch.int64, device="cuda"),
            slots_out=torch.full((routes,), -1, dtype=torch.int32, device="cuda"),
            count_out=torch.zeros(1, dtype=torch.int32, device="cuda"),
            graph_counters=torch.zeros(2, dtype=torch.int64, device="cuda"),
            graph_unique_counters=torch.zeros(2, dtype=torch.int64, device="cuda"),
            route_counts=torch.zeros(EXPERTS, dtype=torch.float32, device="cuda"),
        )
        out["remap"] = plan_graph_routes_fused(
            ids.reshape(-1), expert_to_slot, 100, torch.int64, dedup=True, dead_rows=dead, row_top_k=TOP_K, **out
        )
        return out

    masked, plain = plan(dead_rows.dead), plan(None)
    for name, tensor in plain.items():
        if name != "route_counts":
            assert torch.equal(masked[name], tensor), name
    want = torch.bincount(raw[:live].reshape(-1), minlength=EXPERTS).float()
    assert torch.equal(masked["route_counts"].cpu(), want)
    if live == TOKENS:
        assert torch.equal(masked["route_counts"], plain["route_counts"])


@pytest.mark.parametrize("fused", [False, True], ids=["generic", "layer_fusion"])
def test_the_cut_rows_of_a_captured_verify_route_no_experts(tmp_path, fused):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.srt.layers.moe.verify_dead_rows import VerifyDeadRows
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3
    from sglang.test.dsv41_ram_miss_fixtures import service_row_images

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    service = service_module.Exl3RamMissService.get()
    service.plan_gather_width(LANES)
    try:
        with (
            service_row_images(tmp_path),
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
            envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"),
            envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(fused),
            envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(fused),
        ):
            model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.layer_id, layer.top_k = 0, TOP_K
            fmt = Exl3ExpertFormat(layout, 0, source_root=str(tmp_path))
            fmt.max_gather_rows = LANES
            streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            model.add_module("expert_layer", layer)
            ExpertPinnedHostCache(streamer, 4 * LANES, **fmt.pinned_tier_options(layer))
            manager = ExpertHotCacheManager.from_model(
                model, budget_bytes=2 * LANES * streamer.bytes_per_expert,
                seed_path=None, dynamic=True, update_prefill_tokens=16,
                min_residence_forwards=0, benefit_ratio=0.0,
                graph_gather_batch_size=TOKENS, graph_gather_miss_lanes=LANES, update_decode_forwards=1,
                gpu_residency_update=True, insert_on_miss=2,
            )
            updater = manager.gpu_residency

            dead_rows = VerifyDeadRows(max_rows=TOKENS, device="cuda")
            generator = torch.Generator(device="cpu").manual_seed(41)
            x = (torch.randn((TOKENS, HIDDEN), generator=generator) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.full((TOKENS, TOP_K), 1.0 / TOP_K, device="cuda")
            ids = torch.tensor([list(range(TOP_K))] * TOKENS, device="cuda", dtype=torch.int32)

            def gated_apply():
                # What Exl3MoEMethod.apply runs for a layer that verify_dead_rows reached.
                return Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT, dead=dead_rows.dead)

            gated_apply()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = gated_apply()
            manager.discard_graph_capture_routes()

            def mapping():
                return updater.mapping[0, :EXPERTS].cpu().tolist()

            outsiders = [expert for expert, slot in enumerate(mapping()) if slot < 0]
            live, cut = outsiders[:TOP_K], outsiders[TOP_K : TOP_K + 12]
            # Two live rows share 6 misses; the four cut rows route 12 others, more than the 8 lanes on their own.
            routes = [live, live] + [[cut[(2 * t + k) % 12] for k in range(TOP_K)] for t in range(TOKENS - 2)]
            ids.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))

            updater.overflow_flag.zero_()
            updater.route_counts.zero_()
            dead_rows.set_live_rows(torch.tensor([2], device="cuda"))
            graph.replay()
            torch.cuda.synchronize()
            service.fail_stop_check()
            if fused:
                # The fused plan's residency route counts: the two live rows' routes only, none of the cut rows'
                # (which repeat row 0's experts).
                counts = updater.route_counts[0, :EXPERTS].cpu().tolist()
                assert counts == [2.0 if e in live else 0.0 for e in range(EXPERTS)]
            assert streamer.row_backend.keep.item() == 1.0
            assert int(updater.overflow_flag.item()) == 0, "a cut row took a miss lane"
            now = mapping()
            assert all(now[e] >= 0 for e in live)
            assert all(now[e] < 0 for e in cut), "a cut row's expert was read"
            for t in range(2):
                slots = torch.tensor([now[e] for e in live])
                ref = _reference(x[t : t + 1], weights[t], slots, manager.caches[0].tensors)
                assert _rel(out[t : t + 1], ref) <= REL_BOUND, t
            assert not bool(out[2:].float().any()), "a cut row has routed output"

            # The same graph with every row live: the cut rows' 12 misses are back, past the lanes.
            dead_rows.clear()
            graph.replay()
            torch.cuda.synchronize()
            service.fail_stop_check()
            assert int(updater.overflow_flag.item()) == 1
    finally:
        service.shutdown()
