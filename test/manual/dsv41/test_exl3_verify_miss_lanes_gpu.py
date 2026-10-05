"""D2-2 gate: a captured 6-token verify through the EXL3 lease chain at 8 miss lanes (GPU).

Six tokens of top-6 route 36 ids a layer, past the wire's 32 lanes. The service builds 8 lanes (the miss width). A
verify whose union of misses fits the lanes is served exactly: every routed expert is mapped, its slot holds its
checkpoint bytes, and every token's output meets the probe's bar. A verify with more misses than lanes serves 8,
flags the forward, and leaves residency exact: every mapped expert's slot still holds its own bytes, and the chain
neither traps nor fail-stops.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_ram_miss_graph_gpu import HIDDEN, INTER, REL_BOUND, _reference, _rel, _source_rows  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

ACT_LIMIT = 10.0
TOP_K, TOKENS, LANES, EXPERTS = 6, 6, 8, 48


@pytest.mark.parametrize("fused", [False, True], ids=["generic", "layer_fusion"])
def test_a_verify_gather_serves_its_lanes_and_flags_what_it_cannot(tmp_path, fused):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3
    from sglang.test.dsv41_ram_miss_fixtures import service_row_images

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    source = _source_rows(tmp_path, num_experts=EXPERTS)
    source_cuda = {name: rows.cuda() for name, rows in source.items()}
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
            # The eager staging bound: this test runs no eager gather, and the inclusive tier holds
            # 3 * LANES - max_gather_rows hot slots, so the miss width leaves the 2 * LANES DIRECT needs.
            fmt.max_gather_rows = LANES
            streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            model.add_module("expert_layer", layer)
            ExpertPinnedHostCache(streamer, 3 * LANES, **fmt.pinned_tier_options(layer))
            manager = ExpertHotCacheManager.from_model(
                model, budget_bytes=2 * LANES * streamer.bytes_per_expert,
                seed_path=None, dynamic=True, update_prefill_tokens=16,
                min_residence_forwards=0, benefit_ratio=0.0,
                graph_gather_batch_size=TOKENS, graph_gather_miss_lanes=LANES, update_decode_forwards=1,
                gpu_residency_update=True, insert_on_miss=2,
            )
            updater = manager.gpu_residency
            assert streamer.graph_gather_rows == TOKENS * TOP_K and streamer.graph_miss_width == LANES
            assert service.lanes == LANES and updater.miss_rows == LANES
            assert manager.caches[0].capacity == 2 * LANES and manager.caches[0].scratch_rows == 0
            assert updater.layer_fusion is fused

            generator = torch.Generator(device="cpu").manual_seed(37)
            x = (torch.randn((TOKENS, HIDDEN), generator=generator) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.full((TOKENS, TOP_K), 1.0 / TOP_K, device="cuda")
            ids = torch.tensor([list(range(TOP_K))] * TOKENS, device="cuda", dtype=torch.int32)
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            manager.discard_graph_capture_routes()

            def residency_is_exact():
                mapping = updater.mapping[0, :EXPERTS].cpu()
                for expert in range(EXPERTS):
                    slot = int(mapping[expert])
                    if slot < 0:
                        continue
                    for name, rows in manager.caches[0].tensors.items():
                        assert torch.equal(rows[slot].cpu(), source[name][expert]), (expert, slot, name)

            def replay(routes, overflow, clear=True):
                if clear:
                    updater.overflow_flag.zero_()
                ids.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                service.fail_stop_check()
                assert streamer.row_backend.keep.item() == 1.0
                assert int(updater.overflow_flag.item()) == int(overflow), routes
                assert updater.insertion_truncated[0].item() == 0
                residency_is_exact()
                if overflow:
                    return
                mapping = updater.mapping[0, :EXPERTS].cpu()
                for t, route in enumerate(routes):
                    slots = torch.tensor([int(mapping[e]) for e in route])
                    assert bool((slots >= 0).all()), route
                    ref = _reference(x[t : t + 1], weights[t], slots, manager.caches[0].tensors)
                    assert _rel(out[t : t + 1], ref) <= REL_BOUND, (t, route)

            def outsiders():
                """Experts with no slot right now, so a route to them is a miss."""
                mapping = updater.mapping[0, :EXPERTS].cpu().tolist()
                return [expert for expert, slot in enumerate(mapping) if slot < 0]

            shared = outsiders()[:TOP_K]
            replay([shared] * TOKENS, overflow=False)  # 6 misses shared by every token, copied once
            rows_read = service.host.counters()["rows_read"]
            replay([shared] * TOKENS, overflow=False)  # all hits now
            assert service.host.counters()["rows_read"] == rows_read
            eight = outsiders()[:LANES]
            replay([[eight[(t + k) % 8] for k in range(TOP_K)] for t in range(TOKENS)], overflow=False)  # 8 fill the lanes
            overflowed = updater.gather_overflow[0].item()
            twelve = outsiders()[:12]
            routes = [[twelve[(2 * t + k) % 12] for k in range(TOP_K)] for t in range(TOKENS)]
            replay(routes, overflow=True, clear=False)  # 12 > 8
            assert updater.gather_overflow[0].item() == overflowed + 1
            # The DSpark re-run: read and clear the flag, then the same verify with the graph gather suspended.
            assert manager.take_verify_overflow()
            with manager.suspend_graph_gather():
                assert not streamer.serves_graph_gather(SimpleNamespace(topk_ids=ids))
                eager = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids, ACT_LIMIT)
            torch.cuda.synchronize()
            service.fail_stop_check()
            assert int(updater.overflow_flag.item()) == 0, "the eager re-run flagged"
            assert updater.insertion_truncated[0].item() == 0
            residency_is_exact()
            for t, route in enumerate(routes):
                ref = _reference(x[t : t + 1], weights[t], torch.tensor(route), source_cuda)
                assert _rel(eager[t : t + 1], ref) <= REL_BOUND, (t, route)
            replay([outsiders()[:TOP_K]] * TOKENS, overflow=False)  # served exactly again afterwards
    finally:
        service.shutdown()
