"""Warp-geometry tests for the expert-row GPU transfer kernel.

Every case compares the whole destination against a host reference built by
indexing the sources, so a byte copied wrong or written outside the plan fails.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Expert cache transfer tests require a CUDA GPU.",
)

LAUNCH_THREADS = 8 * 256
WARPS = LAUNCH_THREADS // 32

EXPERT_LIKE_SHAPES = (
    ((64, 20), torch.uint8),
    ((40, 10), torch.uint8),
    ((64, 2), torch.float8_e4m3fn),
    ((40, 1), torch.float8_e4m3fn),
    ((), torch.float32),
    ((), torch.float32),
)


def _sources(rows, shapes, generator):
    sources = []
    for shape, dtype in shapes:
        element = torch.empty((), dtype=dtype).element_size()
        raw = torch.randint(
            0, 256, (rows, *shape, element) if shape else (rows, element),
            dtype=torch.uint8, generator=generator,
        )
        sources.append(raw.view(dtype).reshape(rows, *shape).pin_memory())
    return sources


def _plan(capacity, rows, slots):
    source_rows = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    destination_slots = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
    source_rows[: len(rows)] = torch.as_tensor(rows, dtype=torch.int64).to("cuda")
    destination_slots[: len(slots)] = torch.as_tensor(slots, dtype=torch.int32).to("cuda")
    count = torch.tensor([len(rows)], dtype=torch.int32, device="cuda")
    return source_rows, destination_slots, count


def _row_bytes(tensor):
    return tensor.reshape(tensor.shape[0], -1).view(torch.uint8)


def _assert_matches_reference(sources, destinations, rows, slots):
    """The whole destination equals zeros with exactly the planned source rows copied in."""
    for source, destination in zip(sources, destinations):
        source_bytes = _row_bytes(source)
        expected = torch.zeros((destination.shape[0], source_bytes.shape[1]), dtype=torch.uint8)
        if rows:
            expected[torch.as_tensor(slots, dtype=torch.long)] = source_bytes[torch.as_tensor(rows, dtype=torch.long)]
        assert torch.equal(_row_bytes(destination.cpu()), expected)


def _pick(generator, source_count, slot_count, rows):
    picked_rows = torch.randperm(source_count, generator=generator)[:rows].tolist()
    picked_slots = torch.randperm(slot_count, generator=generator)[:rows].tolist()
    return picked_rows, picked_slots


def _copy_segments(sources, destinations, rows, slots, capacity):
    from sglang.kernels.ops.moe.expert_cache_transfer import (
        copy_expert_row_segments_gpu,
        expert_row_segments,
    )

    segments = expert_row_segments(list(zip(sources, destinations)))
    copy_expert_row_segments_gpu(segments, *_plan(capacity, rows, slots))
    torch.cuda.synchronize()


def test_count_zero_writes_nothing():
    generator = torch.Generator().manual_seed(0)
    sources = _sources(16, EXPERT_LIKE_SHAPES, generator)
    destinations = [torch.zeros((12, *s.shape[1:]), dtype=s.dtype, device="cuda") for s in sources]
    _copy_segments(sources, destinations, [], [], capacity=8)
    _assert_matches_reference(sources, destinations, [], [])


@pytest.mark.parametrize("rows", [1, 2, 31, WARPS - 1, WARPS])
def test_counts_up_to_one_warp_per_row(rows):
    generator = torch.Generator().manual_seed(rows)
    sources = _sources(rows + 7, EXPERT_LIKE_SHAPES, generator)
    destinations = [torch.zeros((rows + 5, *s.shape[1:]), dtype=s.dtype, device="cuda") for s in sources]
    picked_rows, picked_slots = _pick(generator, rows + 7, rows + 5, rows)
    _copy_segments(sources, destinations, picked_rows, picked_slots, capacity=rows)
    _assert_matches_reference(sources, destinations, picked_rows, picked_slots)


@pytest.mark.parametrize("rows", [WARPS + 1, 130, 257, LAUNCH_THREADS + 52])
def test_more_rows_than_warps(rows):
    generator = torch.Generator().manual_seed(rows)
    shapes = (((520,), torch.uint8), ((37,), torch.uint8), ((), torch.float32))
    sources = _sources(rows + 11, shapes, generator)
    destinations = [torch.zeros((rows + 9, *s.shape[1:]), dtype=s.dtype, device="cuda") for s in sources]
    picked_rows, picked_slots = _pick(generator, rows + 11, rows + 9, rows)
    _copy_segments(sources, destinations, picked_rows, picked_slots, capacity=rows)
    _assert_matches_reference(sources, destinations, picked_rows, picked_slots)


@pytest.mark.parametrize("row_bytes", [1, 3, 15, 17, 4099, 65537])
@pytest.mark.parametrize("source_offset,destination_offset", [(0, 0), (1, 0), (0, 3), (5, 11)])
@pytest.mark.parametrize("rows", [1, 9, 70])
def test_misaligned_and_odd_length_segments(row_bytes, source_offset, destination_offset, rows):
    generator = torch.Generator().manual_seed(row_bytes * 131 + source_offset * 7 + destination_offset + rows)
    source_count, slot_count = rows + 4, rows + 2
    host = torch.randint(
        0, 256, (source_count * row_bytes + source_offset,), dtype=torch.uint8, generator=generator
    ).pin_memory()
    source = host[source_offset:].view(source_count, row_bytes)
    device = torch.zeros((slot_count * row_bytes + destination_offset,), dtype=torch.uint8, device="cuda")
    destination = device[destination_offset:].view(slot_count, row_bytes)
    picked_rows, picked_slots = _pick(generator, source_count, slot_count, rows)
    _copy_segments([source], [destination], picked_rows, picked_slots, capacity=rows)
    _assert_matches_reference([source], [destination], picked_rows, picked_slots)
    assert not device[:destination_offset].any()


def test_registered_arena_rows_match_reference():
    from sglang.srt.layers.moe.expert_host_arena import _page_aligned_like
    from sglang.srt.mem_cache.pool_host.common import _cuda_host_register, _cuda_host_unregister

    generator = torch.Generator().manual_seed(7)
    rows = 40
    sources = []
    for shape, dtype in EXPERT_LIKE_SHAPES:
        source = _page_aligned_like(torch.empty((rows, *shape), dtype=dtype, device="meta"))
        source.view(torch.uint8).reshape(-1).copy_(
            torch.randint(0, 256, (source.numel() * source.element_size(),), dtype=torch.uint8, generator=generator)
        )
        _cuda_host_register(source, registration_granularity_bytes=source[0].numel() * source.element_size())
        sources.append(source)
    try:
        destinations = [torch.zeros((rows, *s.shape[1:]), dtype=s.dtype, device="cuda") for s in sources]
        picked_rows, picked_slots = _pick(generator, rows, rows, 17)
        _copy_segments(sources, destinations, picked_rows, picked_slots, capacity=17)
        _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
    finally:
        for source in sources:
            _cuda_host_unregister(source)


def test_graph_replay_follows_changing_count_and_plan():
    from sglang.kernels.ops.moe.expert_cache_transfer import (
        copy_expert_row_segments_gpu,
        expert_row_segments,
    )

    generator = torch.Generator().manual_seed(11)
    capacity, source_count, slot_count = 130, 300, 140
    sources = _sources(source_count, EXPERT_LIKE_SHAPES, generator)
    destinations = [torch.zeros((slot_count, *s.shape[1:]), dtype=s.dtype, device="cuda") for s in sources]
    segments = expert_row_segments(list(zip(sources, destinations)))
    first_rows, first_slots = _pick(generator, source_count, slot_count, 3)
    source_rows, destination_slots, count = _plan(capacity, first_rows, first_slots)

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(2):
            copy_expert_row_segments_gpu(segments, source_rows, destination_slots, count)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_expert_row_segments_gpu(segments, source_rows, destination_slots, count)

    for rows in (0, 1, 10, WARPS, WARPS + 1, capacity, 5):
        picked_rows, picked_slots = _pick(generator, source_count, slot_count, rows)
        for destination in destinations:
            destination.view(torch.uint8).zero_()
        if rows:
            source_rows[:rows].copy_(torch.as_tensor(picked_rows, dtype=torch.int64).to("cuda"))
            destination_slots[:rows].copy_(torch.as_tensor(picked_slots, dtype=torch.int32).to("cuda"))
        count.fill_(rows)
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches_reference(sources, destinations, picked_rows, picked_slots)


@pytest.mark.parametrize("rows", [1, 64, 65, 300])
def test_single_tensor_op_matches_reference(rows):
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_rows_gpu

    generator = torch.Generator().manual_seed(rows + 1000)
    source_count, slot_count = rows + 3, rows + 6
    source = _sources(source_count, (((33, 17), torch.uint8),), generator)[0]
    destination = torch.zeros((slot_count, 33, 17), dtype=torch.uint8, device="cuda")
    picked_rows, picked_slots = _pick(generator, source_count, slot_count, rows)
    copy_expert_rows_gpu(source, destination, *_plan(rows, picked_rows, picked_slots))
    torch.cuda.synchronize()
    _assert_matches_reference([source], [destination], picked_rows, picked_slots)
