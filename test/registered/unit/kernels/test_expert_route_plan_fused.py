"""Equivalence tests for the fused BS1, top_k<=32 demand route planner kernel."""

import random
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.layers.moe.expert_route_plan import plan_graph_routes
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused expert route plan tests require a CUDA GPU.",
)

EXPERTS = 32


class _NoHostReads(TorchDispatchMode):
    FORBIDDEN = {
        "aten::_local_scalar_dense",
        "aten::item",
        "aten::nonzero",
        "aten::is_nonzero",
        "aten::equal",
    }

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name in self.FORBIDDEN:
            raise AssertionError(f"host read {func._schema.name}")
        return func(*args, **(kwargs or {}))


def _expert_to_slot(resident: list[int], num_experts: int = EXPERTS) -> torch.Tensor:
    mapping = torch.full((num_experts,), -1, dtype=torch.int64, device="cuda")
    for slot, expert in enumerate(resident):
        mapping[expert] = slot
    return mapping


def run_fused_case(
    ids: torch.Tensor,
    expert_to_slot: torch.Tensor,
    scratch_base: int,
    remap_dtype: torch.dtype = torch.int64,
) -> SimpleNamespace:
    """Allocate outputs/counters, supply zero-prefetch state, and run the kernel."""
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    device = ids.device
    count = ids.numel()
    source_rows_out = torch.full((count,), -1, dtype=torch.int64, device=device)
    slots_out = torch.full((count,), -1, dtype=torch.int32, device=device)
    count_out = torch.full((1,), -1, dtype=torch.int32, device=device)
    remap_out = torch.full((count,), -1, dtype=remap_dtype, device=device)
    graph_counters = torch.zeros(2, dtype=torch.int64, device=device)
    graph_unique_counters = torch.zeros(2, dtype=torch.int64, device=device)
    route_counts = torch.zeros(expert_to_slot.numel(), dtype=torch.float32, device=device)
    prefetch_expert = torch.zeros(1, dtype=torch.int64, device=device)
    prefetch_count = torch.zeros(1, dtype=torch.int32, device=device)

    plan_unique_routes_cuda(
        ids,
        expert_to_slot,
        scratch_base,
        source_rows_out,
        slots_out,
        count_out,
        remap_out,
        graph_counters,
        graph_unique_counters,
        route_counts,
        prefetch_expert,
        prefetch_count,
        0,
    )
    return SimpleNamespace(
        source_rows=source_rows_out,
        slots=slots_out,
        count=count_out,
        remap=remap_out,
        graph_counters=graph_counters,
        graph_unique_counters=graph_unique_counters,
        route_counts=route_counts,
    )


def _expected_slots(ids: torch.Tensor, oracle: SimpleNamespace) -> torch.Tensor:
    """The destination each compacted `oracle.source_rows` position should hold.

    IDs are unique, so each compacted position corresponds to exactly one
    original route; that route's `remap` value is the destination the fused
    kernel's `slots_out` must carry at that position.
    """
    match = ids.unsqueeze(1) == oracle.source_rows.unsqueeze(0)
    assert bool((match.sum(dim=0) == 1).all())
    origin = match.float().argmax(dim=0)
    return oracle.remap[origin]


def assert_matches_reference(
    ids: torch.Tensor, resident: list[int], scratch_base: int
) -> SimpleNamespace:
    expert_to_slot = _expert_to_slot(resident)
    oracle = plan_graph_routes(ids, expert_to_slot, ids.numel(), scratch_base)
    result = run_fused_case(ids, expert_to_slot, scratch_base)

    torch.testing.assert_close(result.source_rows, oracle.source_rows)
    torch.testing.assert_close(result.remap, oracle.remap)
    torch.testing.assert_close(result.slots.to(torch.int64), _expected_slots(ids, oracle))
    assert int(result.count.item()) == int(oracle.miss_plan_rows)
    assert int(result.graph_counters[0].item()) == ids.numel()
    assert int(result.graph_counters[1].item()) == int(oracle.routed_miss_rows)
    assert int(result.graph_unique_counters[0].item()) == int(oracle.unique_hit_rows)
    assert int(result.graph_unique_counters[1].item()) == int(oracle.unique_miss_rows)
    expected_route_counts = torch.zeros_like(result.route_counts)
    expected_route_counts.index_add_(
        0, ids, torch.ones(ids.numel(), dtype=torch.float32, device=ids.device)
    )
    torch.testing.assert_close(result.route_counts, expected_route_counts)
    return result


def test_unique_plan_matches_reference_worked_case():
    ids = torch.tensor([5, 2, 9, 1], device="cuda", dtype=torch.int64)
    expert_to_slot = _expert_to_slot([])
    expert_to_slot[2], expert_to_slot[1] = 7, 3

    oracle = plan_graph_routes(ids, expert_to_slot, ids.numel(), 10)
    result = run_fused_case(ids, expert_to_slot, scratch_base=10)

    torch.testing.assert_close(result.source_rows, ids.new_tensor([5, 9, 2, 1]))
    torch.testing.assert_close(result.remap, ids.new_tensor([10, 7, 11, 3]))
    torch.testing.assert_close(result.slots, ids.new_tensor([10, 11, 7, 3], dtype=torch.int32))
    assert int(result.count.item()) == 2
    torch.testing.assert_close(result.source_rows, oracle.source_rows)
    torch.testing.assert_close(result.remap, oracle.remap)


def test_all_hit_has_no_residual_rows():
    ids = torch.tensor([4, 1, 3, 2], device="cuda", dtype=torch.int64)
    assert_matches_reference(ids, resident=[1, 2, 3, 4], scratch_base=4)


def test_all_miss_uses_every_scratch_row():
    ids = torch.tensor([4, 1, 3, 2], device="cuda", dtype=torch.int64)
    assert_matches_reference(ids, resident=[], scratch_base=0)


@pytest.mark.parametrize("top_k", [1, 10, 32])
def test_supported_top_k_sizes(top_k):
    for seed in range(20):
        rng = random.Random(seed * 1000 + top_k)
        ids = torch.tensor(
            rng.sample(range(EXPERTS), top_k), device="cuda", dtype=torch.int64
        )
        resident = rng.sample(range(EXPERTS), rng.randint(0, EXPERTS // 2))
        assert_matches_reference(ids, resident, scratch_base=len(resident))


@pytest.mark.parametrize("topk_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("remap_dtype", [torch.int32, torch.int64])
def test_routing_dtypes(topk_dtype, remap_dtype):
    ids64 = torch.tensor([5, 2, 9, 1], device="cuda", dtype=torch.int64)
    ids = ids64.to(dtype=topk_dtype)
    expert_to_slot = _expert_to_slot([])
    expert_to_slot[2], expert_to_slot[1] = 7, 3
    oracle = plan_graph_routes(ids64, expert_to_slot, ids64.numel(), 10)

    result = run_fused_case(ids, expert_to_slot, scratch_base=10, remap_dtype=remap_dtype)

    assert result.remap.dtype == remap_dtype
    torch.testing.assert_close(result.remap.to(torch.int64), oracle.remap)
    torch.testing.assert_close(result.source_rows, oracle.source_rows)


@pytest.mark.parametrize("seed", range(200))
def test_random_unique_ids_and_maps(seed):
    rng = random.Random(seed)
    top_k = rng.randint(1, 32)
    ids = torch.tensor(rng.sample(range(EXPERTS), top_k), device="cuda", dtype=torch.int64)
    resident = rng.sample(range(EXPERTS), rng.randint(0, EXPERTS // 2))
    assert_matches_reference(ids, resident, scratch_base=len(resident))


def test_map_change_replay_updates_plan_between_calls():
    ids = torch.tensor([5, 2, 9, 1], device="cuda", dtype=torch.int64)
    expert_to_slot = _expert_to_slot([])
    expert_to_slot[2], expert_to_slot[1] = 7, 3

    first_oracle = plan_graph_routes(ids, expert_to_slot, ids.numel(), 10)
    first = run_fused_case(ids, expert_to_slot, scratch_base=10)
    torch.testing.assert_close(first.source_rows, first_oracle.source_rows)
    torch.testing.assert_close(first.remap, first_oracle.remap)
    assert int(first.count.item()) == 2

    expert_to_slot[5] = 0
    expert_to_slot[1] = -1
    second_oracle = plan_graph_routes(ids, expert_to_slot, ids.numel(), 10)
    second = run_fused_case(ids, expert_to_slot, scratch_base=10)

    torch.testing.assert_close(second.source_rows, second_oracle.source_rows)
    torch.testing.assert_close(second.remap, second_oracle.remap)
    assert int(second.count.item()) == int(second_oracle.miss_plan_rows)


def test_planning_reads_no_device_value_on_the_host():
    ids = torch.tensor([4, 2, 9, 0, 1, 7], device="cuda", dtype=torch.int64)
    expert_to_slot = _expert_to_slot([2, 7])
    with pytest.raises(AssertionError):
        with _NoHostReads():
            torch.tensor([1]).sum().item()

    with _NoHostReads():
        result = run_fused_case(ids, expert_to_slot, scratch_base=2)

    oracle = plan_graph_routes(ids, expert_to_slot, ids.numel(), 2)
    torch.testing.assert_close(result.source_rows, oracle.source_rows)
    torch.testing.assert_close(result.remap, oracle.remap)


def test_route_counts_none_is_accepted():
    """`route_counts=None` is the normal production shape whenever the layer
    has no residency policy; it exercises the tvm_ffi Optional<TensorView>
    binding for an absent tensor, distinct from an absent-but-allocated one."""
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    ids = torch.tensor([5, 2, 9, 1], device="cuda", dtype=torch.int64)
    expert_to_slot = _expert_to_slot([])
    expert_to_slot[2], expert_to_slot[1] = 7, 3
    oracle = plan_graph_routes(ids, expert_to_slot, ids.numel(), 10)

    source_rows_out = torch.full((4,), -1, dtype=torch.int64, device="cuda")
    slots_out = torch.full((4,), -1, dtype=torch.int32, device="cuda")
    count_out = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    remap_out = torch.full((4,), -1, dtype=torch.int64, device="cuda")
    prefetch_expert = torch.zeros(1, dtype=torch.int64, device="cuda")
    prefetch_count = torch.zeros(1, dtype=torch.int32, device="cuda")

    plan_unique_routes_cuda(
        ids,
        expert_to_slot,
        10,
        source_rows_out,
        slots_out,
        count_out,
        remap_out,
        None,
        None,
        None,
        prefetch_expert,
        prefetch_count,
        0,
    )

    torch.testing.assert_close(source_rows_out, oracle.source_rows)
    torch.testing.assert_close(remap_out, oracle.remap)
    assert int(count_out.item()) == int(oracle.miss_plan_rows)


def test_fused_plan_captures_and_replays_with_changed_ids():
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    device = "cuda"
    ids = torch.zeros(4, dtype=torch.int64, device=device)
    ids.copy_(torch.tensor([5, 2, 9, 1], device=device))
    expert_to_slot = _expert_to_slot([])
    expert_to_slot[2], expert_to_slot[1] = 7, 3
    scratch_base = 10

    source_rows_out = torch.full((4,), -1, dtype=torch.int64, device=device)
    slots_out = torch.full((4,), -1, dtype=torch.int32, device=device)
    count_out = torch.full((1,), -1, dtype=torch.int32, device=device)
    remap_out = torch.full((4,), -1, dtype=torch.int64, device=device)
    graph_counters = torch.zeros(2, dtype=torch.int64, device=device)
    graph_unique_counters = torch.zeros(2, dtype=torch.int64, device=device)
    route_counts = torch.zeros(expert_to_slot.numel(), dtype=torch.float32, device=device)
    prefetch_expert = torch.zeros(1, dtype=torch.int64, device=device)
    prefetch_count = torch.zeros(1, dtype=torch.int32, device=device)

    def run():
        plan_unique_routes_cuda(
            ids,
            expert_to_slot,
            scratch_base,
            source_rows_out,
            slots_out,
            count_out,
            remap_out,
            graph_counters,
            graph_unique_counters,
            route_counts,
            prefetch_expert,
            prefetch_count,
            0,
        )

    run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for replay_ids in ([9, 1, 5, 2], [0, 31, 4, 5], [2, 7, 5, 9]):
        ids.copy_(torch.tensor(replay_ids, device=device, dtype=torch.int64))
        graph_counters.zero_()
        graph_unique_counters.zero_()
        route_counts.zero_()
        graph.replay()
        torch.cuda.synchronize()

        oracle = plan_graph_routes(ids, expert_to_slot, ids.numel(), scratch_base)
        torch.testing.assert_close(source_rows_out, oracle.source_rows)
        torch.testing.assert_close(remap_out, oracle.remap)
        assert int(count_out.item()) == int(oracle.miss_plan_rows)
        assert int(graph_counters[1].item()) == int(oracle.routed_miss_rows)
        assert int(graph_unique_counters[1].item()) == int(oracle.unique_miss_rows)


def test_gate_refuses_multi_token_calls_even_when_shape_would_otherwise_qualify():
    """`graph_gather_rows` is sized `tokens * top_k`, so a multi-token call's
    combined route count can still fit `scratch_rows`; only requiring exactly
    one token row keeps a multi-request batch off this path. `single_token`
    is duplicate-free on purpose: a within-row duplicate cannot arise from
    real routing (see `supports_fused_graph_routes`'s docstring), so this
    fixture must not assert anything about duplicate handling."""
    from sglang.srt.layers.moe.expert_route_plan import supports_fused_graph_routes

    expert_to_slot = _expert_to_slot([1, 6])
    single_token = torch.tensor([[1, 2, 3, 6]], device="cuda", dtype=torch.int64)
    multi_token = torch.tensor(
        [[1, 2, 2, 6], [3, 4, 4, 5]], device="cuda", dtype=torch.int64
    )

    assert supports_fused_graph_routes(single_token, expert_to_slot, scratch_rows=8)
    assert not supports_fused_graph_routes(multi_token, expert_to_slot, scratch_rows=8)
