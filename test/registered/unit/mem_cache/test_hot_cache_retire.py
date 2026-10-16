"""hot_cache_retire: one function lowers every hot-cache capacity holder, evicts the retired slots, and refuses late.

Two kinds of test:
* the capacity-holder census reads the real classes' source (ExpertHotCache, ExpertResidencyPolicy,
  GpuResidencyUpdater, Exl3RamMissRowBackend, ExpertRowPlanner, SpecScoreRow) and fails if any of them assigns a
  capacity-named attribute that CAPACITY_HOLDERS neither lists nor exempts, so a new holder cannot slip past retire;
* retire itself on CPU fakes with the real attribute names: evictions from the device bank, mapping and host lists,
  every capacity value lowered, the RAM-miss hot sets re-pushed, victims re-ranked, the VMM tail released, and the
  ordering refusals (captured graph, armed copy engine, non-VMM cache, scratch rows, keep below the victim floor);
* a grow that cannot get its memory (injected: a mapping OOM on the first layer or part-way, or the free-memory guard
  tripping before any mapping) stops short without raising, keeps the layers already grown, leaves every holder
  consistent and the hot sets re-pushed, and the calibration hook still returns, so the launch goes on.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import hot_cache_retire as hr
from sglang.srt.layers.moe.hot_cache_vmm import HotCacheMapOutOfMemory
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
    """1 MiB a row. ``oom`` makes the next grow raise as the real backing does after rolling its mapping back."""

    def __init__(self):
        self.released = None
        self.backed_rows = C
        self.oom = False
        self.grow_calls = 0

    def bytes_to_back(self, rows):
        return max(0, rows - self.backed_rows) << 20

    def set_backed_rows(self, rows):
        if rows > self.backed_rows:
            self.grow_calls += 1
            if self.oom:
                raise HotCacheMapOutOfMemory("injected cuMemCreate OOM (rolled back)")
        delta, self.backed_rows = (rows - self.backed_rows) << 20, rows
        return delta

    def release_tail(self, k):
        self.released = k
        return -self.set_backed_rows(k)


def _world(victim_lanes=8, upper=C):
    """Two layers of C slots; the residency banks and each cache's reservation span ``upper`` slots."""
    slot_to_expert = torch.full((L, upper + 1), -1, dtype=torch.long)
    mapping = torch.full((L, E + 1), -1, dtype=torch.long)
    for row in range(L):  # experts 0..15 in slots 4..19 (the tail holds residents)
        for e in range(E):
            slot_to_expert[row, 4 + e] = e
            mapping[row, e] = 4 + e
    slot_ids = torch.arange(upper + 1)
    capacity = torch.full((L,), C)
    miss_columns = torch.arange(6)
    ranked = []
    slot_state = torch.zeros((L, upper + 1), dtype=torch.uint8)
    slot_state[:, :C] = 3
    updater = SimpleNamespace(
        enabled=torch.tensor(False), layer_ids=[3, 7], slot_to_expert=slot_to_expert, mapping=mapping,
        slot_state=slot_state, slot_generations=torch.zeros((L, upper + 1)),
        capacity=capacity, slot_ids=slot_ids, slot_valid=slot_ids.unsqueeze(0) < capacity.unsqueeze(1),
        miss_columns=miss_columns, scratch_columns=capacity.unsqueeze(1) + miss_columns.unsqueeze(0),
        victim_lanes=victim_lanes, route_counts=torch.zeros((L, E)), max_capacity=upper,
        _rank_victims=lambda routed: ranked.append(routed.shape),
    )
    caches, policies = {}, {}
    for row, layer in enumerate(updater.layer_ids):
        cache = SimpleNamespace(
            capacity=C, capacity_bytes=C * 100, allocation_bytes=C * 100, bytes_per_expert=100,
            slot_to_expert=updater.slot_to_expert[row, :C].tolist(), slot_states=[3] * C, _slot_generations=[0] * C,
            slot_state=None, slot_generations=None, vmm_backing=FakeBacking(), scratch_rows=0, upper_capacity=upper,
            reserves_prefetch_pull_row=False, streamer=SimpleNamespace(row_planner=SimpleNamespace(scratch_base=C)),
            device=torch.device("cpu"),
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


def _holders(manager, updater, service, layer):
    """Every capacity holder of one layer, as plain values."""
    cache, row = manager.caches[layer], updater.layer_ids.index(layer)
    srow = service.row_of(layer)
    return {
        "cache": (cache.capacity, cache.capacity_bytes, cache.allocation_bytes, len(cache.slot_to_expert),
                  len(cache.slot_states), len(cache._slot_generations)),
        "policy": manager.residency_policies[layer].capacity,
        "updater": (int(updater.capacity[row]), int(updater.slot_valid[row].sum()),
                    updater.scratch_columns[row].tolist()),
        "planner": cache.streamer.row_planner.scratch_base,
        "service": (service._row_backends[srow].hot_capacity, service._spec_hot[srow][1]),
        "backed": cache.vmm_backing.backed_rows,
    }


def test_set_hot_capacity_grows_into_the_reservation_with_free_slots():
    manager, updater, service, pushed, bound, ranked = _world(upper=26)
    result = hr.set_hot_capacity(manager, {3: 24}, service=service)
    c3 = manager.caches[3]
    assert c3.vmm_backing.backed_rows == 24 and result.freed_bytes == -(4 << 20)  # mapped first, net negative
    assert _holders(manager, updater, service, 3)["cache"][:4] == (24, 2400, 2400, 24)
    assert c3.slot_to_expert[20:] == [-1] * 4 and c3.slot_states[20:] == [0] * 4 and c3.slot_state.numel() == 24
    assert (updater.slot_to_expert[0, 20:24] == -1).all() and (updater.slot_state[0, 20:24] == 0).all()
    assert int(updater.slot_valid[0].sum()) == 24 and updater.scratch_columns[0].tolist() == list(range(24, 30))
    assert c3.streamer.row_planner.scratch_base == 24 and service._row_backends[0].hot_capacity == 24
    assert pushed[0] == list(range(16))  # residents unchanged: the new slots are empty
    with pytest.raises(hr.RetireRefused, match="outside"):
        hr.set_hot_capacity(manager, {3: 27}, service=service)  # past the reservation


def test_growing_then_shrinking_back_restores_every_holder():
    """The inverse census: each holder set_hot_capacity raises comes back to its starting value when lowered again."""
    manager, updater, service, *_ = _world(upper=26)
    before = _holders(manager, updater, service, 3)
    hr.set_hot_capacity(manager, {3: 26}, service=service)
    grown = _holders(manager, updater, service, 3)
    assert all(grown[k] != before[k] for k in before), (before, grown)  # every holder moved
    result = hr.set_hot_capacity(manager, {3: C}, service=service)
    assert _holders(manager, updater, service, 3) == before
    assert result.evicted.get(3, []) == []  # the added slots were empty


def test_an_unlisted_capacity_attribute_is_refused():
    manager, updater, service, *_ = _world()
    manager.caches[3].streamer.row_planner.shadow_capacity = 20
    with pytest.raises(hr.RetireRefused, match="unlisted capacity holder row_planner.shadow_capacity"):
        hr.retire_hot_slots(manager, {3: 17}, service=service)
    assert manager.caches[3].capacity == C and int(updater.capacity[0]) == C  # refused before any change


def _real_spec_scores(updater, service):
    """The service's real ``_bind_spec_scores`` (and ``spec_score_rows``) on the fake world: row 0 (layer 3) scores
    the prefetch for row 1 (layer 7), as the L+2 scorer does; row 1 has no target."""
    import types

    from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissService

    gate = SimpleNamespace(weight=torch.zeros(E, 4), bias=torch.zeros(E))
    service._spec_gpu = SimpleNamespace(picked=SimpleNamespace(targets=torch.tensor([[1, 0], [-1, 0]]), gates=[gate]))
    service._spec_hot = {0: (updater.slot_to_expert[0], C), 1: (updater.slot_to_expert[1], C)}
    service._bind_spec_scores = types.MethodType(Exl3RamMissService._bind_spec_scores, service)
    service._bind_spec_scores()
    assert service._row_backends[0].spec_score.target == 1 and service._row_backends[1].spec_score is None


def test_a_rows_spec_score_carries_its_target_layers_capacity_through_a_grow():
    """calib5: layer 25 grew to 27 while its L+2 target stayed at 26, and the census compared the score's 26 with
    layer 25's own 27. The score is right; the check must read the target's capacity."""
    manager, updater, service, *_ = _world(upper=26)
    _real_spec_scores(updater, service)
    hr.set_hot_capacity(manager, {3: 24}, service=service)  # the source layer grows, its target does not
    assert service._row_backends[0].spec_score.hot_capacity == C
    hr.set_hot_capacity(manager, {7: 25}, service=service)  # the target grows: the source's score follows
    assert service._row_backends[0].spec_score.hot_capacity == 25


def test_a_spec_score_left_stale_by_a_grow_is_refused_in_strict_mode_and_rolled_back_otherwise():
    manager, updater, service, *_ = _world(upper=26)
    _real_spec_scores(updater, service)
    before = _all_holders(manager, updater, service)
    service._bind_spec_scores = lambda: None  # the rebind is lost: layer 3's score keeps layer 7's old 20
    with pytest.raises(hr.RetireRefused, match="spec_score.hot_capacity is 20, not its target layer 7's 25"):
        hr.set_hot_capacity(manager, {7: 25}, service=service)
    manager, updater, service, *_ = _world(upper=26)
    _real_spec_scores(updater, service)
    service._bind_spec_scores = lambda: None
    result = hr.set_hot_capacity(manager, {3: 24, 7: 25}, service=service, strict=False)
    assert result.kept == {3: C, 7: C} and result.short == {3: 4, 7: 5} and result.short_reason.startswith("rolled")
    assert _all_holders(manager, updater, service) == before
    assert all(manager.caches[layer].vmm_backing.backed_rows == C for layer in (3, 7))


def _grow_all(manager, service, **kwargs):
    return hr.set_hot_capacity(manager, {3: 24, 7: 24}, service=service, **kwargs)


def _all_holders(manager, updater, service):
    return {layer: _holders(manager, updater, service, layer) for layer in (3, 7)}


def test_an_oom_on_the_first_layer_grows_nothing_and_does_not_raise():
    manager, updater, service, pushed, bound, ranked = _world(upper=26)
    before = _all_holders(manager, updater, service)
    manager.caches[3].vmm_backing.oom = True
    result = _grow_all(manager, service)
    assert result.kept == {3: C, 7: C} and result.short == {3: 4, 7: 4}
    assert "injected cuMemCreate OOM" in result.short_reason and result.freed_bytes == 0
    assert manager.caches[7].vmm_backing.grow_calls == 0  # stopped at the first failure
    assert _all_holders(manager, updater, service) == before
    assert pushed[0] == list(range(16)) and pushed[1] == list(range(16)) and bound and ranked  # service refreshed


def test_an_oom_part_way_keeps_the_layers_already_grown():
    manager, updater, service, *_ = _world(upper=26)
    before7 = _holders(manager, updater, service, 7)
    manager.caches[7].vmm_backing.oom = True
    result = _grow_all(manager, service)
    assert result.kept == {3: 24, 7: C} and result.short == {7: 4} and result.freed_bytes == -(4 << 20)
    grown = _holders(manager, updater, service, 3)
    assert grown["cache"][:4] == (24, 2400, 2400, 24) and grown["updater"][:2] == (24, 24)
    assert grown["service"] == (24, 24) and grown["backed"] == 24 and grown["planner"] == 24
    assert _holders(manager, updater, service, 7) == before7  # the failed layer is untouched
    # The tables stay usable: the same authority moves both layers again.
    manager.caches[7].vmm_backing.oom = False
    hr.set_hot_capacity(manager, {3: C, 7: 22}, service=service)
    assert _holders(manager, updater, service, 3)["cache"][0] == C and int(updater.capacity[1]) == 22


def test_the_free_memory_guard_stops_before_any_mapping():
    manager, updater, service, *_ = _world(upper=26)
    before = _all_holders(manager, updater, service)
    result = _grow_all(manager, service, free_memory=lambda: 100 << 20, keep_free=97 << 20)  # layer 3 needs 4 MiB
    assert result.kept == {3: C, 7: C} and result.short == {3: 4, 7: 4}
    assert "layer 3 needs 4.0 MiB" in result.short_reason
    assert all(manager.caches[layer].vmm_backing.grow_calls == 0 for layer in (3, 7))
    assert _all_holders(manager, updater, service) == before


def test_the_guard_reads_free_memory_per_layer_and_stops_at_the_first_short_one():
    manager, updater, service, *_ = _world(upper=26)
    free = iter([10 << 20, 5 << 20])  # layer 3 mapped 4 MiB between the two reads
    result = _grow_all(manager, service, free_memory=lambda: next(free), keep_free=2 << 20)
    assert result.kept == {3: 24, 7: C} and result.short == {7: 4}
    assert manager.caches[7].vmm_backing.grow_calls == 0


def _calibrate_on_fakes(monkeypatch, manager, service, *, strict=True, calibrate=None):
    """maybe_calibrate_hot_cache on the fake world: measured values given, the forwards and the trial stubbed."""
    from sglang.srt import runtime_context
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import hot_cache_calibration as hc
    from sglang.srt.layers.moe import hot_cache_dummy_extend as hde
    from sglang.srt.layers.moe import hot_cache_trial_capture as tct

    M = 1 << 20
    monkeypatch.setattr(hr, "_service", lambda: service)
    run = hc.RunPeak(2500 * M, 2100 * M, 2500 * M, 2480 * M)
    cal = hc.Calibration(4096, 4000 * M, 1000 * M, (run, run), 1 * M, 0, 1000 * M)
    monkeypatch.setattr(hc, "calibrate", calibrate or (lambda *a, **k: cal))
    monkeypatch.setattr(hc, "settle", lambda c, d: hc.replace(c, free_final=2420 * M, reserved_final=1000 * M))
    monkeypatch.setattr(hde, "dummy_extend_forward", lambda runner, tokens: None)
    monkeypatch.setattr(hde, "forget_graph_path_state", lambda model: 0)
    monkeypatch.setattr(hc, "log_grow_bound", lambda *a: None)  # reads real streamers
    monkeypatch.setattr(tct, "trial_capture", lambda *a: None)  # its own tests cover it
    monkeypatch.setattr(tct, "reset_trial_routes", lambda manager: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a: None)
    monkeypatch.setattr(
        runtime_context, "get_schedule", lambda: SimpleNamespace(max_total_tokens=None, chunked_prefill_size=4096)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3000 * M, 32000 * M))
    runner = SimpleNamespace(
        expert_hot_cache_manager=manager, max_total_num_tokens=None, model=None, forward=lambda *a, **k: None,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
    )
    with envs.SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP.override(True):
        return hc.maybe_calibrate_hot_cache(runner, strict=strict)


def _failing_calibrate(*a, **k):
    raise RuntimeError("Inference tensors do not track version counter")  # calib8's failure


def test_a_failed_calibration_is_logged_cleaned_up_and_the_launch_starts_at_the_configured_size(monkeypatch, caplog):
    manager, updater, service, *_ = _world(upper=26)
    before = _all_holders(manager, updater, service)
    with caplog.at_level("ERROR"):
        result = _calibrate_on_fakes(monkeypatch, manager, service, strict=False, calibrate=_failing_calibrate)
    assert result is None and _all_holders(manager, updater, service) == before
    error = [r for r in caplog.records if r.levelname == "ERROR"][-1]
    assert "failed during calibration prefills" in error.getMessage() and error.exc_info is not None


def test_strict_calibration_re_raises(monkeypatch):
    manager, updater, service, *_ = _world(upper=26)
    with pytest.raises(RuntimeError, match="version counter"):
        _calibrate_on_fakes(monkeypatch, manager, service, strict=True, calibrate=_failing_calibrate)


def test_a_failure_inside_the_resize_is_re_raised(monkeypatch):
    """Holders may be half-moved: the launch must not go on."""
    manager, updater, service, *_ = _world(upper=26)

    def broken(*a, **k):
        raise ValueError("unexpected failure part-way through the resize")

    monkeypatch.setattr(hr, "set_hot_capacity", broken)
    with pytest.raises(ValueError, match="part-way"):
        _calibrate_on_fakes(monkeypatch, manager, service, strict=False)


def test_a_failure_that_leaves_the_holders_inconsistent_is_unrecoverable(monkeypatch):
    from sglang.srt.layers.moe import hot_cache_calibration as hc

    manager, updater, service, *_ = _world(upper=26)

    def spoil_then_fail(*a, **k):
        manager.residency_policies[3].capacity = 99  # a holder no longer agrees
        raise RuntimeError("failed after touching state")

    with pytest.raises(hc.CalibrationUnrecoverable, match="policy.capacity is 99"):
        _calibrate_on_fakes(monkeypatch, manager, service, strict=False, calibrate=spoil_then_fail)


def test_calibration_does_not_resize_a_hot_cache_the_real_capture_would_not_expect(monkeypatch, caplog):
    """After the trial capture the updater must still be disabled and every holder agree; otherwise no resize."""
    manager, updater, service, *_ = _world(upper=26)
    updater.enabled.fill_(True)  # e.g. a trial that went through the enabling capture path
    before = _all_holders(manager, updater, service)
    with caplog.at_level("WARNING"):
        result = _calibrate_on_fakes(monkeypatch, manager, service)
    assert result.retire is None and result.keep == {3: C, 7: C} and "residency updater is enabled" in result.extra["unsettled"]
    assert _all_holders(manager, updater, service) == before
    assert any("not resizing" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")


def test_calibration_survives_a_grow_oom_and_reports_the_slots_it_got(monkeypatch):
    """The launch path on fakes: the hook plans a grow, the mapping fails part-way, and it returns."""
    from sglang.srt import runtime_context
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import hot_cache_calibration as hc
    from sglang.srt.layers.moe import hot_cache_dummy_extend as hde

    M = 1 << 20
    manager, updater, service, *_ = _world(upper=26)
    monkeypatch.setattr(hr, "_service", lambda: service)
    manager.caches[7].vmm_backing.oom = True
    run = hc.RunPeak(2500 * M, 2100 * M, 2500 * M, 2480 * M)
    cal = hc.Calibration(4096, 4000 * M, 1000 * M, (run, run), 1 * M, 0, 1000 * M)
    monkeypatch.setattr(hc, "calibrate", lambda *a, **k: cal)
    monkeypatch.setattr(hc, "settle", lambda c, d: hc.replace(c, free_final=2420 * M, reserved_final=1000 * M))
    monkeypatch.setattr(hde, "dummy_extend_forward", lambda runner, tokens: None)
    monkeypatch.setattr(hde, "forget_graph_path_state", lambda model: 0)
    monkeypatch.setattr(hc, "log_grow_bound", lambda *a: None)  # reads real streamers
    from sglang.srt.layers.moe import hot_cache_trial_capture as tct

    monkeypatch.setattr(tct, "trial_capture", lambda *a: None)  # its own tests cover it
    monkeypatch.setattr(
        runtime_context, "get_schedule", lambda: SimpleNamespace(max_total_tokens=None, chunked_prefill_size=4096)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3000 * M, 32000 * M))
    runner = SimpleNamespace(
        expert_hot_cache_manager=manager, max_total_num_tokens=None, model=None, forward=lambda *a, **k: None,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
    )
    with envs.SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP.override(True):
        result = hc.maybe_calibrate_hot_cache(runner, strict=True)
    # room = (2420 - 1500 regrow) - 400 margin = 520 MiB at 1 MiB a slot: both layers planned up to their bound 26.
    assert result.retire.short == {7: 6} and result.keep == {3: 26, 7: C}
    assert int(updater.capacity[0]) == 26 and int(updater.capacity[1]) == C
