"""hot_cache_retire: one function lowers every hot-cache capacity holder, evicts the retired slots, and refuses late.

Two kinds of test:
* the capacity-holder census reads the real classes' source (ExpertHotCache, ExpertResidencyPolicy,
  GpuResidencyUpdater, Exl3RamMissRowBackend, ExpertRowPlanner, SpecScoreRow) and fails if any of them assigns a
  capacity-named attribute that CAPACITY_HOLDERS neither lists nor exempts, so a new holder cannot slip past retire;
* retire itself on CPU fakes with the real attribute names: evictions from the device bank, mapping and host lists,
  every capacity value lowered, the RAM-miss hot sets re-pushed, victims re-ranked, the VMM tail released, and the
  ordering refusals (captured graph, armed copy engine, non-VMM cache, scratch rows, keep below the victim floor).
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import hot_cache_retire as hr
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _assigned_capacity_attrs(cls) -> set:
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    names = set()
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, (ast.AnnAssign, ast.AugAssign)) else []
        for t in targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                names.add(t.attr)
            elif isinstance(t, ast.Name) and isinstance(node, ast.AnnAssign):
                names.add(t.id)  # dataclass fields
    return {n for n in names if "capacity" in n.lower()}


def test_every_capacity_holder_in_the_real_classes_is_listed_or_exempt():
    from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissRowBackend
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_residency import ExpertResidencyPolicy
    from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlanner
    from sglang.srt.layers.moe.ram_prefetch import SpecScoreRow

    listed = {attr.split("[")[0].split(".")[-1] for _, attr in hr.CAPACITY_HOLDERS} | hr.EXEMPT_CAPACITY_ATTRS
    found = {}
    for cls in (ExpertHotCache, ExpertResidencyPolicy, GpuResidencyUpdater, Exl3RamMissRowBackend, ExpertRowPlanner,
                SpecScoreRow):
        for attr in _assigned_capacity_attrs(cls):
            found.setdefault(attr, []).append(cls.__name__)
    unlisted = {a: c for a, c in found.items() if a not in listed}
    assert not unlisted, f"capacity holders retire does not update: {unlisted}"
    assert {"capacity", "capacity_bytes", "hot_capacity", "max_capacity"} <= set(found)


# --- retire on fakes ---------------------------------------------------------------------------------------------

E, L, C = 16, 2, 20  # experts, layers, capacity (all layers)


class FakeBacking:
    def __init__(self):
        self.released = None

    def release_tail(self, k):
        self.released = k
        return (C - k) << 20


def _world(victim_lanes=8):
    slot_to_expert = torch.full((L, C + 1), -1, dtype=torch.long)
    mapping = torch.full((L, E + 1), -1, dtype=torch.long)
    for row in range(L):  # experts 0..15 in slots 4..19 (the tail holds residents)
        for e in range(E):
            slot_to_expert[row, 4 + e] = e
            mapping[row, e] = 4 + e
    slot_ids = torch.arange(C + 1)
    capacity = torch.full((L,), C)
    miss_columns = torch.arange(6)
    ranked = []
    updater = SimpleNamespace(
        enabled=torch.tensor(False), layer_ids=[3, 7], slot_to_expert=slot_to_expert, mapping=mapping,
        slot_state=torch.full((L, C + 1), 3, dtype=torch.uint8), slot_generations=torch.zeros((L, C + 1)),
        capacity=capacity, slot_ids=slot_ids, slot_valid=slot_ids.unsqueeze(0) < capacity.unsqueeze(1),
        miss_columns=miss_columns, scratch_columns=capacity.unsqueeze(1) + miss_columns.unsqueeze(0),
        victim_lanes=victim_lanes, route_counts=torch.zeros((L, E)), max_capacity=C,
        _rank_victims=lambda routed: ranked.append(routed.shape),
    )
    caches, policies = {}, {}
    for row, layer in enumerate(updater.layer_ids):
        cache = SimpleNamespace(
            capacity=C, capacity_bytes=C * 100, allocation_bytes=C * 100, bytes_per_expert=100,
            slot_to_expert=updater.slot_to_expert[row, :C].tolist(), slot_states=[3] * C, _slot_generations=[0] * C,
            slot_state=None, slot_generations=None, vmm_backing=FakeBacking(), scratch_rows=0,
            reserves_prefetch_pull_row=False, streamer=SimpleNamespace(row_planner=SimpleNamespace(scratch_base=C)),
        )
        caches[layer] = cache
        policies[layer] = SimpleNamespace(capacity=C)
    manager = SimpleNamespace(caches=caches, residency_policies=policies, gpu_residency=updater)
    pushed, bound = {}, []
    service = SimpleNamespace(
        _copy_armed=False, row_of=lambda layer: {3: 0, 7: 1}[layer],
        _row_backends={0: SimpleNamespace(hot_capacity=C, spec_score=None),
                       1: SimpleNamespace(hot_capacity=C, spec_score=None)},
        _spec_hot={0: ("slots0", C), 1: ("slots1", C)}, _spec_gpu=object(), _bind_spec_scores=lambda: bound.append(1),
        gpu_hot_enabled=True, _hot_snapshot=torch.empty_like(slot_to_expert), _hot_lists={},
        before_host_use=lambda: None, after_host_use=lambda: None,
        host=SimpleNamespace(set_hot=lambda row, experts: pushed.__setitem__(row, experts)),
    )

    def refresh():
        for row, layer in enumerate(updater.layer_ids):
            cap = int(updater.capacity[row])
            service._hot_lists[layer] = [int(e) for e in service._hot_snapshot[row, :cap].tolist() if e >= 0]

    service._refresh_hot_lists = refresh
    return manager, updater, service, pushed, bound, ranked


def test_retire_evicts_the_tail_and_lowers_every_holder():
    manager, updater, service, pushed, bound, ranked = _world()
    result = hr.retire_hot_slots(manager, {3: 17, 7: 20}, service=service)
    assert result.kept == {3: 17, 7: 20} and result.retired == {3: (17, 20), 7: (20, 20)}
    assert result.evicted == {3: [13, 14, 15]}  # experts in slots 17..19
    c3 = manager.caches[3]
    assert c3.capacity == 17 and c3.capacity_bytes == 1700 and len(c3.slot_to_expert) == 17
    assert len(c3.slot_states) == 17 and len(c3._slot_generations) == 17 and c3.slot_state.numel() == 17
    assert manager.residency_policies[3].capacity == 17 and c3.streamer.row_planner.scratch_base == 17
    assert int(updater.capacity[0]) == 17 and int(updater.slot_valid[0].sum()) == 17
    assert updater.scratch_columns[0].tolist() == list(range(17, 23))
    assert (updater.mapping[0, [13, 14, 15]] == -1).all() and (updater.slot_to_expert[0, 17:20] == -1).all()
    assert (updater.slot_state[0, 17:20] == 0).all()
    assert service._row_backends[0].hot_capacity == 17 and service._spec_hot[0] == ("slots0", 17)
    assert pushed[0] == list(range(0, 13)) and pushed[1] == list(range(16)) and bound and ranked
    assert c3.vmm_backing.released == 17 and manager.caches[7].vmm_backing.released is None
    assert result.freed_bytes == 3 << 20
    # Layer 7 kept everything: untouched.
    assert manager.caches[7].capacity == 20 and int(updater.capacity[1]) == 20


@pytest.mark.parametrize(
    "spoil,match",
    [
        (lambda m, u, s: u.__setattr__("enabled", torch.tensor(True)), "after CUDA graph capture"),
        (lambda m, u, s: s.__setattr__("_copy_armed", True), "copy engine armed"),
        (lambda m, u, s: m.caches[3].__setattr__("vmm_backing", None), "not on VMM"),
        (lambda m, u, s: m.caches[3].__setattr__("scratch_rows", 4), "scratch"),
    ],
)
def test_retire_refuses_once_the_slot_count_is_frozen(spoil, match):
    manager, updater, service, *_ = _world()
    spoil(manager, updater, service)
    with pytest.raises(hr.RetireRefused, match=match):
        hr.retire_hot_slots(manager, {3: 17}, service=service)
    assert manager.caches[3].capacity == C  # nothing changed


def test_retire_refuses_below_the_victim_floor_and_above_capacity():
    manager, updater, service, *_ = _world(victim_lanes=8)
    with pytest.raises(hr.RetireRefused, match="victim lanes"):
        hr.retire_hot_slots(manager, {3: 15}, service=service)
    with pytest.raises(hr.RetireRefused, match="outside"):
        hr.retire_hot_slots(manager, {3: 21}, service=service)


def test_an_unlisted_capacity_attribute_is_refused():
    manager, updater, service, *_ = _world()
    manager.caches[3].streamer.row_planner.shadow_capacity = 20
    with pytest.raises(hr.RetireRefused, match="unlisted capacity holder row_planner.shadow_capacity"):
        hr.retire_hot_slots(manager, {3: 17}, service=service)
    assert manager.caches[3].capacity == C and int(updater.capacity[0]) == C  # refused before any change
