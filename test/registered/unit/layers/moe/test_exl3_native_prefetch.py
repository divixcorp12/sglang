"""Native prefetch, the Python side (CPU): the arming guard, the refusals, the hooks' capture-only rule and DIRECT's
extended victim ranking. The kernels themselves are tested on the GPU (test/manual/dsv41/test_exl3_native_prefetch_cuda.py).
"""

import types

import pytest
import torch

from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_native_prefetch as native
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prefetch():
    return native.NativePrefetch(torch.zeros(256, dtype=torch.uint8), service=types.SimpleNamespace())


def test_the_flag_is_off_by_default_and_reaches_the_config():
    assert Dsv41Config.from_envs().enable_native_prefetch is False
    with envs.SGLANG_DSV41_ENABLE_NATIVE_PREFETCH.override(True):
        assert Dsv41Config.from_envs().enable_native_prefetch is True


@pytest.mark.parametrize("plan, commit", [(False, False), (True, False), (False, True)])
def test_the_copy_engine_does_not_arm_before_both_prefetch_kernels_were_captured(plan, commit):
    """Deadlock safety (LEASE_PROTOCOL.md 7.6 "Module loading"): a prefetch kernel first launched after arming would
    load its module while an armed step's copy wait spins."""
    p = _prefetch()
    p.captured_plan, p.captured_commit = plan, commit
    with pytest.raises(RuntimeError, match="before the prefetch kernels were captured"):
        p.check_armable()


def test_the_copy_engine_arms_once_both_were_captured():
    p = _prefetch()
    p.captured_plan = p.captured_commit = True
    p.check_armable()


def test_the_service_refuses_to_arm_while_the_prefetch_kernels_are_not_captured():
    from sglang.srt.layers.moe.exl3_ram_miss import COPY_ENGINE_ARM_DECODES, Exl3RamMissService

    service = Exl3RamMissService()
    armed = []
    service.copy_engine = True
    service.device_side = object()
    service.host = types.SimpleNamespace(arm_copy_engine=lambda: armed.append(1))
    service._copy_decodes = COPY_ENGINE_ARM_DECODES
    service.native_prefetch = _prefetch()
    with pytest.raises(RuntimeError, match="captured"):
        service._arm_copy_engine()
    assert not armed
    service.native_prefetch.captured_plan = service.native_prefetch.captured_commit = True
    service._arm_copy_engine()
    assert armed == [1]


def test_an_unbound_or_eager_hook_launches_nothing():
    """Only a captured graph posts or waits: an eager forward must never wait on the copy engine."""
    p = _prefetch()
    p.commit(3, torch.zeros(6, dtype=torch.int64))
    p.predict(2, torch.zeros(1, 8))
    assert not p.captured_plan and not p.captured_commit
    p.updater = object()  # bound, but not capturing
    p.commit(3, torch.zeros(6, dtype=torch.int64))
    p.predict(2, torch.zeros(1, 8))
    assert not p.captured_plan and not p.captured_commit


def test_bind_refuses_an_updater_without_the_extended_ranking():
    p = _prefetch()
    with pytest.raises(RuntimeError, match="extended victim ranking"):
        p.bind(types.SimpleNamespace(prefetch_victims=None))


def test_the_gate_registry():
    gate = torch.nn.Linear(4, 4)
    native.register_gate(7, gate)
    assert native.registered_gate(7) is gate and native.registered_gate(8) is None


def _updater(extended: bool):
    """A GpuResidencyUpdater with just what _rank_victims reads, on the CPU."""
    from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater

    u = GpuResidencyUpdater.__new__(GpuResidencyUpdater)
    layers, experts, capacity, width = 2, 16, 12, 3
    u.num_layers, u.num_experts, u.max_capacity, u.miss_rows, u.device = layers, experts, capacity, width, "cpu"
    u.victim_columns = capacity + 1
    torch.manual_seed(0)
    u.slot_to_expert = torch.full((layers, capacity + 1), -1, dtype=torch.long)
    u.slot_state = torch.zeros((layers, capacity + 1), dtype=torch.uint8)
    for row in range(layers):
        u.slot_to_expert[row, :10] = torch.randperm(experts)[:10]
        u.slot_state[row, :10] = 3
    u.slot_ids = torch.arange(capacity + 1)
    u.slot_valid = u.slot_ids.unsqueeze(0) < torch.tensor([[capacity], [capacity]])
    u.insert_scores = torch.rand(layers, experts)
    u.victims = torch.zeros((layers, width), dtype=torch.long)
    u.victim_valid = torch.zeros((layers, width), dtype=torch.bool)
    u.prefetch_victims = torch.zeros((layers, 4), dtype=torch.long) if extended else None
    u.prefetch_victim_valid = torch.zeros((layers, 4), dtype=torch.bool) if extended else None
    return u


def test_the_prefetch_victims_continue_directs_own_order_past_the_shortlist():
    """The replay's victim rule (prefetch_sim.PrefetchReplay): the lowest-ranked slots that are not in the demand
    shortlist, in DIRECT's own order. Ranking with the extension on must leave the shortlist exactly as it was."""
    routed = torch.zeros(2, 16, dtype=torch.bool)
    routed[:, :4] = True
    plain, extended = _updater(False), _updater(True)
    plain._rank_victims(routed)
    extended._rank_victims(routed)
    assert torch.equal(plain.victims, extended.victims) and torch.equal(plain.victim_valid, extended.victim_valid)
    for row in range(2):
        # The full order, from a shortlist as wide as every slot.
        full = _updater(False)
        full.miss_rows = full.victim_columns
        full.victims = torch.zeros((2, full.miss_rows), dtype=torch.long)
        full.victim_valid = torch.zeros((2, full.miss_rows), dtype=torch.bool)
        full._rank_victims(routed)
        order = full.victims[row][full.victim_valid[row]].tolist()
        want = order[3:7]
        got = extended.prefetch_victims[row][extended.prefetch_victim_valid[row]].tolist()
        assert got == want and not set(got) & set(extended.victims[row].tolist())
