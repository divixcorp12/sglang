"""A captured 6-token verify through the real EXL3 lease chain with CPU experts and spill (GPU, real CPU kernel).

Plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 13. A lane per route (36 on a 40-lane wire), two victim lanes.
  - Before the copy engine arms, forced lanes cannot be the CPU's: the replay overflows (Review Focus 1, the one
    overflow left) and the eager re-run is exact.
  - Armed, 36 distinct RAM-resident experts (Review Focus 5) are served with no overflow: two on the GPU, the rest on
    the CPU, every token within the CPU kernel's bar of the fp32 reference.
  - Armed, 36 distinct cold experts, all NVMe misses on the one node (Review Focus 2) are served with no overflow:
    the live misses stage, the forced ones are read into RAM victims and computed there, and stay cached.
Run on divix01 from the pushed worktree, holding rowimg-disk.lock then cc-gpu.lock, on the recipe's server cores (the
CPU expert team derives node 0's 6-15):
  CUDA_MODULE_LOADING=EAGER SGLANG_DSV41_CPU_EXPERTS=1 EXL3_MOE_CPU_PIN=0 SGLANG_EXL3_SRC=... SGLANG_EXL3_CPU_CXX=...
  taskset -c 0-5,36-41 python -m pytest -q test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_ram_miss_graph_gpu import HIDDEN, INTER, _reference, _rel, _source_rows  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC") and os.environ.get("SGLANG_DSV41_CPU_EXPERTS") == "1"),
    reason="needs a GPU, SGLANG_EXL3_SRC and SGLANG_DSV41_CPU_EXPERTS=1 (the optimized CPU build)",
)

ACT_LIMIT = 10.0
TOP_K, TOKENS, VICTIMS, EXPERTS = 6, 6, 2, 128
LANES = TOKENS * TOP_K  # 36: a lane per route
HOT = 8  # VRAM slots per layer; the DIRECT floor is 2 * VICTIMS
TIER = 64  # pinned rows: >= staging (2) + 36 lanes + HOT (8) = 46, Task 10's room check
CPU_BOUND = 2e-2  # the CPU kernel's own bar (test_exl3_moe_split_parity_cuda.py)


def _distinct(experts):
    """Six tokens of top-6 over 36 distinct experts: token t routes experts[6t .. 6t + 5]."""
    return [experts[TOP_K * t : TOP_K * (t + 1)] for t in range(TOKENS)]


def test_a_verify_spills_every_victimless_lane_to_the_cpu_and_never_overflows_once_armed(tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.srt.layers.moe.ram_slot_map import LaneKind
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3
    from sglang.test.dsv41_ram_miss_fixtures import service_row_images

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    source = _source_rows(tmp_path, num_experts=EXPERTS)
    source_cuda = {name: rows.cuda() for name, rows in source.items()}
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    service = service_module.Exl3RamMissService.get()
    try:
        with (
            service_row_images(tmp_path),
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
            envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"),
            envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(True),
            envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(True),
            envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.override(True),
            envs.SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION.override(False),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(VICTIMS),
        ):
            model, layer = torch.nn.Module(), torch.nn.Module()
            layer.layer_id, layer.top_k = 0, TOP_K
            fmt = Exl3ExpertFormat(layout, 0, source_root=str(tmp_path))
            fmt.max_gather_rows = 8
            streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            model.add_module("expert_layer", layer)
            ExpertPinnedHostCache(streamer, TIER, **fmt.pinned_tier_options(layer))
            manager = ExpertHotCacheManager.from_model(
                model, budget_bytes=HOT * streamer.bytes_per_expert, seed_path=None, dynamic=True,
                update_prefill_tokens=16, min_residence_forwards=0, benefit_ratio=0.0,
                graph_gather_batch_size=TOKENS, graph_gather_miss_lanes=0, update_decode_forwards=1,
                gpu_residency_update=True, insert_on_miss=2,
            )
            updater = manager.gpu_residency
            assert streamer.graph_miss_width == LANES and (updater.miss_rows, updater.victim_lanes) == (LANES, VICTIMS)
            assert service.staging_width() == VICTIMS

            generator = torch.Generator(device="cpu").manual_seed(41)
            x = (torch.randn((TOKENS, HIDDEN), generator=generator) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.softmax(torch.randn((TOKENS, TOP_K), generator=generator), 1).to("cuda")
            ids = torch.tensor([list(range(TOP_K))] * TOKENS, device="cuda", dtype=torch.int32)
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            manager.discard_graph_capture_routes()
            assert service.lanes == 40 and service.cpu_experts is not None

            def outsiders():
                mapping = updater.mapping[0, :EXPERTS].cpu().tolist()
                return [expert for expert, slot in enumerate(mapping) if slot < 0]

            def in_ram(experts):
                """Load experts into the pinned tier through the eager path (the service maps them on the device)."""
                with manager.suspend_graph_gather():
                    for start in range(0, len(experts), TOP_K):
                        chunk = experts[start : start + TOP_K]
                        Exl3MoEMethod._apply_streamed(
                            layer, streamer, x[:1], weights[:1], torch.tensor([chunk], device="cuda", dtype=torch.int32),
                            ACT_LIMIT,
                        )
                torch.cuda.synchronize()

            def replay(routes):
                updater.overflow_flag.zero_()
                ids.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                service.fail_stop_check()
                assert updater.insertion_truncated[0].item() == 0
                return int(updater.overflow_flag.item())

            def exact(result, routes):
                for t, route in enumerate(routes):
                    ref = _reference(x[t : t + 1], weights[t], torch.tensor(route), source_cuda)
                    assert _rel(result[t : t + 1], ref) <= CPU_BOUND, (t, route)

            def served_without_overflow(routes):
                inserted = updater.gather_insertions[0].item()
                assert replay(routes) == 0, "an armed verify overflowed"
                count = int(streamer._graph_miss_count.item())
                kinds = service.device_side.lane_kind[:count].tolist()
                cpu = sum(k in (int(LaneKind.HIT_CPU), int(LaneKind.MISS_CPU)) for k in kinds)
                live = updater.gather_insertions[0].item() - inserted
                assert count == LANES and live <= VICTIMS and cpu >= LANES - VICTIMS, (count, live, kinds)
                exact(out, routes)

            # Review Focus 1: before the copy engine arms, forced lanes cannot be the CPU's; the re-run is exact.
            warm = outsiders()[:LANES]
            in_ram(warm)
            routes = _distinct(warm)
            assert replay(routes) == 1
            assert manager.take_verify_overflow()
            with manager.suspend_graph_gather():
                eager = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids, ACT_LIMIT)
            torch.cuda.synchronize()
            exact(eager, routes)

            service._copy_decodes = service_module.COPY_ENGINE_ARM_DECODES
            service._arm_copy_engine()
            assert service._copy_armed
            before = updater.gather_overflow[0].item()

            # Review Focus 5: 36 distinct RAM-resident experts, a lane each.
            in_ram(warm)
            served_without_overflow(_distinct(warm))

            # Review Focus 2: 36 distinct cold experts, every one an NVMe miss on the one node. The live ones stage, the
            # forced ones are read into RAM victims, and all stay cached in the tier afterwards.
            cold = [e for e in outsiders() if e not in warm][:LANES]
            assert len(cold) == LANES
            served_without_overflow(_distinct(cold))
            resident = {e for e, s in enumerate(service.host.mapping(0).tolist()) if s >= 0}
            assert len(set(cold) - resident) <= VICTIMS, "the forced misses are cached in their RAM victims"
            assert updater.gather_overflow[0].item() == before, "no armed verify overflowed"
    finally:
        service.shutdown()
