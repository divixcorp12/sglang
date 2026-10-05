"""The layer-fusion launchers refuse bad inputs on their own, without the Python wrappers' help.

Each case calls a JIT module's ``run`` directly -- what a caller that skips the wrapper does -- with one input wrong and
every other input valid, and expects the C++ launcher to raise before it launches. Deleting the check that raises the
case's ``match`` from the launcher turns that case red: the kernel would launch on the bad input instead (a device
read of a host pointer, a read of mistyped bytes, or, for a width past 32, an overrun of the commit kernel's 32-entry
lane arrays).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

CUDA = "cuda"


def _gather_args(width: int = 6, routes: int = 6, experts: int = 16) -> dict:
    """Valid inputs for ``direct_gather_destinations_gpu<int32_t, int32_t, int32_t>``, in FFI parameter order."""
    return {
        "topk_ids": (torch.arange(routes) % experts).to(device=CUDA, dtype=torch.int32),
        "expert_to_slot": torch.full((experts,), -1, dtype=torch.int64, device=CUDA),
        "victims": torch.arange(width, dtype=torch.int64, device=CUDA),
        "victim_valid": torch.ones(width, dtype=torch.bool, device=CUDA),
        "miss_count": torch.zeros(1, dtype=torch.int32, device=CUDA),
        "remap_in": torch.arange(routes, dtype=torch.int32, device=CUDA),
        "scratch_base": 64,
        "destination_slots_out": torch.zeros(width, dtype=torch.int32, device=CUDA),
        "destinations_out": torch.zeros(width, dtype=torch.int64, device=CUDA),
        "live_out": torch.zeros(width, dtype=torch.bool, device=CUDA),
        "remap_out": torch.zeros(routes, dtype=torch.int32, device=CUDA),
    }


def _commit_args(width: int = 6, experts: int = 16, slots: int = 8) -> dict:
    """Valid inputs for ``direct_commit_gather_gpu`` (leased form), in FFI parameter order."""
    return {
        "destinations": torch.arange(width, dtype=torch.int64, device=CUDA) % slots,
        "live": torch.zeros(width, dtype=torch.bool, device=CUDA),
        "new_experts": torch.arange(width, dtype=torch.int64, device=CUDA),
        "num_experts": experts,
        "slot_dump": slots,
        "mapping": torch.full((experts + 1,), -1, dtype=torch.int64, device=CUDA),
        "slot_to_expert": torch.full((slots + 1,), -1, dtype=torch.int64, device=CUDA),
        "slot_state": torch.zeros(slots + 1, dtype=torch.uint8, device=CUDA),
        "slot_generations": torch.zeros(slots + 1, dtype=torch.int64, device=CUDA),
        "gather_insertions": torch.zeros(1, dtype=torch.int64, device=CUDA),
        "gather_evictions": torch.zeros(1, dtype=torch.int64, device=CUDA),
        "insertion_truncated": torch.zeros(1, dtype=torch.int64, device=CUDA),
        "delivered": torch.zeros(1, dtype=torch.int32, device=CUDA),
        "keep": torch.ones(1, dtype=torch.float32, device=CUDA),
        "miss_count": torch.zeros(1, dtype=torch.int32, device=CUDA),
        "cpu_lanes": None,
        "ready": 3,
        "free_state": 0,
    }


def _route_args(routes: int = 6, slots: int = 12, hidden: int = 64, tokens: int = 1) -> dict:
    """Valid inputs for ``exl3_moe_route_tables_gpu<int32_t, bf16_t, bf16_t>``, in FFI parameter order."""
    return {
        "remap": torch.arange(routes, dtype=torch.int32, device=CUDA),
        "weights": torch.ones(routes, dtype=torch.bfloat16, device=CUDA),
        "keep": torch.ones(1, dtype=torch.float32, device=CUDA),
        "x": torch.zeros(tokens, hidden, dtype=torch.bfloat16, device=CUDA),
        "remap64_out": torch.zeros(routes, dtype=torch.int64, device=CUDA),
        "x16_out": torch.zeros(tokens, hidden, dtype=torch.float16, device=CUDA),
        "out_zero": torch.zeros(tokens, hidden, dtype=torch.float32, device=CUDA),
        "expert_count": torch.zeros(slots + 1, dtype=torch.int64, device=CUDA),
        "inv_order": torch.zeros(routes, dtype=torch.int64, device=CUDA),
        "weight_sorted": torch.zeros(routes, dtype=torch.float16, device=CUDA),
        "det": torch.zeros(3, slots + 1, dtype=torch.int64, device=CUDA),
        "token_sorted_out": torch.empty(0, dtype=torch.int64, device=CUDA),
        "cpu_lanes": torch.empty(0, dtype=torch.int32, device=CUDA),
        "dst_slots": torch.empty(0, dtype=torch.int32, device=CUDA),
        "cpu_out": 0,
        "cpu_part_stride": 0,
    }


def _run_gather(args: dict) -> None:
    from sglang.kernels.ops.moe.expert_residency_direct_gather import _gather_module

    _gather_module(torch.int32, torch.int32, torch.int32).run(*args.values())


def _run_commit(args: dict) -> None:
    from sglang.kernels.ops.moe.expert_residency_direct_gather import _commit_module

    _commit_module().run(*args.values())


def _run_route(args: dict) -> None:
    from sglang.kernels.ops.moe.exl3_route_tables import _route_tables_module

    _route_tables_module(torch.int32, torch.bfloat16, torch.bfloat16).run(
        *args.values()
    )


GATHER_REFUSALS = {
    "width_past_32": (
        lambda: _gather_args(width=33),
        "the shortlist and the routes must hold 1-32 entries",
    ),
    "routes_past_32": (
        lambda: _gather_args(routes=33),
        "the shortlist and the routes must hold 1-32 entries",
    ),
    "victims_on_cpu": (
        lambda: {**_gather_args(), "victims": torch.arange(6, dtype=torch.int64)},
        "^victims: ",
    ),
    "topk_ids_int64_into_int32": (
        lambda: {
            **_gather_args(),
            "topk_ids": torch.arange(6, dtype=torch.int64, device=CUDA),
        },
        "^topk_ids: ",
    ),
    "remap_in_shorter_than_routes": (
        lambda: {
            **_gather_args(),
            "remap_in": torch.arange(5, dtype=torch.int32, device=CUDA),
        },
        "^remap_in: ",
    ),
    "victim_valid_not_bool": (
        lambda: {
            **_gather_args(),
            "victim_valid": torch.ones(6, dtype=torch.uint8, device=CUDA),
        },
        "victim_valid: must be a bool tensor",
    ),
    "live_out_not_bool": (
        lambda: {
            **_gather_args(),
            "live_out": torch.zeros(6, dtype=torch.uint8, device=CUDA),
        },
        "live_out: must be a bool tensor",
    ),
}

COMMIT_REFUSALS = {
    "width_past_32": (
        lambda: _commit_args(width=33),
        "the commit must cover 1-32 lanes",
    ),
    "delivered_without_keep": (
        lambda: {**_commit_args(), "keep": None},
        "delivered and keep go together",
    ),
    "slot_state_sized_unlike_slot_to_expert": (
        lambda: {
            **_commit_args(),
            "slot_state": torch.zeros(8, dtype=torch.uint8, device=CUDA),
        },
        "^slot_state: ",
    ),
    "num_experts_not_mapping_minus_dump": (
        lambda: {**_commit_args(), "num_experts": 17},
        "num_experts must be mapping's size minus the dump column",
    ),
    "destinations_on_cpu": (
        lambda: {**_commit_args(), "destinations": torch.arange(6, dtype=torch.int64)},
        "^destinations: ",
    ),
    "keep_not_float32": (
        lambda: {
            **_commit_args(),
            "keep": torch.ones(1, dtype=torch.int32, device=CUDA),
        },
        "^keep: ",
    ),
    "slot_dump_not_last_slot_column": (
        lambda: {**_commit_args(), "slot_dump": 7},
        "slot_dump must be slot_to_expert's last column",
    ),
    "delivered_int64_not_int32": (
        lambda: {
            **_commit_args(),
            "delivered": torch.zeros(1, dtype=torch.int64, device=CUDA),
        },
        "^delivered: ",
    ),
    "cpu_lanes_unleased": (
        lambda: {
            **_commit_args(),
            "delivered": None,
            "keep": None,
            "cpu_lanes": torch.zeros(2, dtype=torch.int32, device=CUDA),
        },
        "cpu_lanes needs the leased delivery count",
    ),
}

ROUTE_REFUSALS = {
    "routes_past_64": (lambda: _route_args(routes=65), "remap must hold 1-64 routes"),
    "routes_not_a_multiple_of_tokens": (lambda: _route_args(routes=7, tokens=2), "multiple of the tokens"),
    "cpu_experts_with_two_tokens": (
        lambda: {
            **_route_args(routes=12, tokens=2),
            "cpu_lanes": torch.zeros(2, dtype=torch.int32, device=CUDA),
            "cpu_out": 16,
        },
        "CPU experts run one token",
    ),
    "token_sorted_out_wrong_size": (
        lambda: {**_route_args(), "token_sorted_out": torch.zeros(5, dtype=torch.int64, device=CUDA)},
        "token_sorted_out: ",
    ),
    "x16_out_wrong_tokens": (
        lambda: {**_route_args(routes=12, tokens=2), "x16_out": torch.zeros(1, 64, dtype=torch.float16, device=CUDA)},
        "^x16_out: ",
    ),
    "x_on_cpu": (
        lambda: {**_route_args(), "x": torch.zeros(1, 64, dtype=torch.bfloat16)},
        "^x: ",
    ),
    "det_flattened": (
        lambda: {
            **_route_args(),
            "det": torch.zeros(3 * 13, dtype=torch.int64, device=CUDA),
        },
        "^det: ",
    ),
    "weights_fp16_into_bf16": (
        lambda: {
            **_route_args(),
            "weights": torch.ones(6, dtype=torch.float16, device=CUDA),
        },
        "^weights: ",
    ),
    "x16_out_wrong_hidden": (
        lambda: {
            **_route_args(),
            "x16_out": torch.zeros(1, 65, dtype=torch.float16, device=CUDA),
        },
        "^x16_out: ",
    ),
    "cpu_lanes_three_words": (
        lambda: {**_route_args(), "cpu_lanes": torch.zeros(3, dtype=torch.int32, device=CUDA)},
        "cpu_lanes: two words",
    ),
    "cpu_out_missing": (
        lambda: {**_route_args(), "cpu_lanes": torch.zeros(2, dtype=torch.int32, device=CUDA)},
        "cpu_out: the CPU partial",
    ),
    "cpu_lanes_on_host": (
        lambda: {**_route_args(), "cpu_lanes": torch.zeros(2, dtype=torch.int32), "cpu_out": 16},
        "^cpu_lanes: ",
    ),
    "cpu_part_stride_unaligned": (
        lambda: {**_route_args(), "cpu_part_stride": 6},
        "cpu_part_stride: floats between parts",
    ),
}


@pytest.mark.parametrize("case", list(GATHER_REFUSALS))
def test_gather_launcher_refuses(case):
    make, match = GATHER_REFUSALS[case]
    with pytest.raises(Exception, match=match):
        _run_gather(make())


@pytest.mark.parametrize("case", list(COMMIT_REFUSALS))
def test_commit_launcher_refuses(case):
    make, match = COMMIT_REFUSALS[case]
    with pytest.raises(Exception, match=match):
        _run_commit(make())


@pytest.mark.parametrize("case", list(ROUTE_REFUSALS))
def test_route_tables_launcher_refuses(case):
    make, match = ROUTE_REFUSALS[case]
    with pytest.raises(Exception, match=match):
        _run_route(make())
