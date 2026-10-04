"""Exl3RamMissService with two NUMA groups (spec 2026-10-03-numa-node-distributor-design, Part 3): ThreadingConfig's
plans reach the service threads, the tier's slot ranges reach the host, and a demand or an eager admission lands
each expert in its home group's slots. The topology and the page bindings are stood in for (Tasks 1 and 5 test
them); the thread runs, no device."""

import contextlib
import faulthandler
import os

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.cpu_experts.threading_config import NodePlan, ThreadingConfig
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_fake_exl3 import write_fake_exl3
from sglang.test.dsv41_ram_miss_fixtures import ROW_IMAGE_DIM, paused, service_row_images

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 8, 8


@pytest.fixture
def two_groups(tmp_path, monkeypatch):
    faulthandler.dump_traceback_later(120, exit=True)
    cores = sorted(os.sched_getaffinity(0))
    plans = (
        NodePlan(group=0, node=0, ram=cores[0], cpu=(), sq=None, busy_poll=False),
        NodePlan(group=1, node=1, ram=cores[1], cpu=(), sq=None, busy_poll=False),
    )
    monkeypatch.setattr(
        ThreadingConfig, "from_env", classmethod(lambda cls, **kw: ThreadingConfig(plans, (cores[2],), 0))
    )
    monkeypatch.setattr(module, "slot_nodes", lambda slabs, capacity: [0] * (capacity // 2) + [1] * (capacity // 2))
    write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS, hidden=ROW_IMAGE_DIM, inter=ROW_IMAGE_DIM)
    layout = build_exl3_expert_layout(str(tmp_path))
    module.Exl3RamMissService._instance = None
    caches = {}
    with contextlib.ExitStack() as stack:
        stack.enter_context(service_row_images(tmp_path))
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
            for layer_id in range(LAYERS):
                layer = torch.nn.Module()
                layer.layer_id = layer_id
                fmt = Exl3ExpertFormat(layout, layer_id, source_root=str(tmp_path))
                streamer = ExpertStreamer(layer, fmt.names, layer_id=layer_id, format=fmt)
                layer._nvfp4_expert_streamer = streamer
                caches[layer_id] = ExpertPinnedHostCache(
                    streamer, CAPACITY, device="cpu", **fmt.pinned_tier_options(layer)
                )
        service = module.Exl3RamMissService.get()
        service.plan_gather_width(1)
        yield service, caches, cores
        service.shutdown()
    module.Exl3RamMissService._instance = None
    faulthandler.cancel_dump_traceback_later()


def test_each_group_runs_on_its_plans_core_over_its_half_of_every_row(two_groups, capfd):
    service, caches, cores = two_groups
    caches[1].ensure_rows(torch.tensor([4, 3]))
    host = service.host
    assert host.nodes == 2 and service.numa.nodes == 2
    assert host.node_ranges == [[(0, 4)] * LAYERS, [(4, 8)] * LAYERS]
    assert host.group_counters(0)["spin_cpu"] == cores[0] and host.group_counters(1)["spin_cpu"] == cores[1]
    host.stop()
    assert "exl3 RAM miss group 1 counters" in capfd.readouterr().err


def test_an_eager_admission_lands_each_expert_in_its_home_groups_slots(two_groups):
    service, caches, cores = two_groups
    caches[1].ensure_rows(torch.tensor([4, 3]))
    row = service.row_of(1)
    with paused(service.host):
        mapping = service.host.mapping(row)
    assert 0 <= mapping[4] < 4, "expert 4 is node 0's"
    assert 4 <= mapping[3] < 8, "expert 3 is node 1's"


def test_a_demand_on_each_node_is_served_by_its_group(two_groups):
    service, caches, cores = two_groups
    caches[1].ensure_rows(torch.tensor([0]))  # starts the service
    sim = ChainSim(service.host, service.page, {})
    row = service.row_of(1)
    type(service.host).pause(service.host, 10.0)
    try:
        sim.sync_bulk()
    finally:
        type(service.host).resume(service.host)
    before = [service.host.group_counters(g)["rows_read"] for g in (0, 1)]
    req = sim.post(row, [1, 2])  # one miss per node: expert 2 is node 0's, expert 1 node 1's
    assert sim.wait_served(req, timeout_s=10.0) and sim.wait_handled(req, timeout_s=10.0)
    after = [service.host.group_counters(g)["rows_read"] for g in (0, 1)]
    assert [a - b for a, b in zip(after, before)] == [1, 1]
