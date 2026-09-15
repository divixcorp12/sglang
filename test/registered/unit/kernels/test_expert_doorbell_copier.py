"""Tests for the doorbell expert-row copier.

A GPU kernel posts a copy plan into a pinned request page, a CPU spin thread
copies the planned pinned rows on its own CUDA stream, and a GPU waiter blocks
until the thread publishes completion or falls back to the in-graph copy.

Every byte case compares the whole destination against a host reference built
by indexing the sources, so a byte copied wrong or written outside the plan
fails. ``test_reference_check_fails_on_flipped_byte_and_stray_write`` shows the
reference check itself can fail.
"""

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Expert doorbell tests require a CUDA GPU.",
)

CAPACITY = 200
SPIN_CORE = int(os.environ.get("DOORBELL_SPIN_CORE", "71"))
STALL_TIMEOUT_POLLS = 20_000

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


def _destinations(sources, slots):
    return [torch.zeros((slots, *s.shape[1:]), dtype=s.dtype, device="cuda") for s in sources]


def _plan(capacity, rows, slots):
    source_rows = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    destination_slots = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
    source_rows[: len(rows)] = torch.as_tensor(rows, dtype=torch.int64).to("cuda")
    destination_slots[: len(slots)] = torch.as_tensor(slots, dtype=torch.int32).to("cuda")
    count = torch.tensor([len(rows)], dtype=torch.int32, device="cuda")
    return source_rows, destination_slots, count


def _row_bytes(tensor):
    return tensor.reshape(tensor.shape[0], -1).view(torch.uint8)


def _reference_mismatch(sources, destinations, rows, slots):
    """Name the first segment whose whole destination differs from zeros plus the planned rows."""
    for index, (source, destination) in enumerate(zip(sources, destinations)):
        source_bytes = _row_bytes(source)
        expected = torch.zeros((destination.shape[0], source_bytes.shape[1]), dtype=torch.uint8)
        if rows:
            expected[torch.as_tensor(slots, dtype=torch.long)] = source_bytes[torch.as_tensor(rows, dtype=torch.long)]
        if not torch.equal(_row_bytes(destination.cpu()), expected):
            return f"segment {index}"
    return None


def _assert_matches_reference(sources, destinations, rows, slots):
    assert _reference_mismatch(sources, destinations, rows, slots) is None


def _pick(generator, source_count, slot_count, rows):
    picked_rows = torch.randperm(source_count, generator=generator)[:rows].tolist()
    picked_slots = torch.randperm(slot_count, generator=generator)[:rows].tolist()
    return picked_rows, picked_slots


def _copier(sources, destinations, capacity=CAPACITY, **kwargs):
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

    kwargs.setdefault("cpu_core", SPIN_CORE)
    kwargs.setdefault("stream", torch.cuda.Stream())
    return ExpertDoorbellCopier(expert_row_segments(list(zip(sources, destinations))), capacity, **kwargs)


def _load_plan(plan, rows, slots):
    source_rows, destination_slots, count = plan
    if rows:
        source_rows[: len(rows)].copy_(torch.as_tensor(rows, dtype=torch.int64).to("cuda"))
        destination_slots[: len(slots)].copy_(torch.as_tensor(slots, dtype=torch.int32).to("cuda"))
    count.fill_(len(rows))


def _zero(destinations):
    for destination in destinations:
        destination.view(torch.uint8).zero_()
    torch.cuda.synchronize()


def _capture(body):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    return graph


def _monotonic_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def _copy_and_compute_end_ns(copier, event, seq):
    """Spin until both the GPU event and request ``seq`` complete; return their host times."""
    compute_end = complete = None
    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline and (compute_end is None or complete is None):
        if compute_end is None and event.query():
            compute_end = _monotonic_ns()
        if complete is None:
            trace = copier.trace()
            if trace and trace[-1]["seq"] == seq and trace[-1]["complete_ns"]:
                complete = trace[-1]["complete_ns"]
    return complete, compute_end


def _completed_request(copier, seq, timeout_s=10.0):
    """Spin until the thread reports request ``seq`` handled and, if serviced, its copies complete."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        trace = copier.trace()
        if trace and trace[-1]["seq"] == seq and (trace[-1]["status"] != "serviced" or trace[-1]["complete_ns"]):
            return trace[-1]
    return None


COPY_ARMS = (
    ("batch", "stream", False),
    ("batch", "any", False),
    ("batch", "during_call", False),
    ("batch", "stream", True),
    ("batch", "any", True),
    ("per_segment", "stream", False),
    ("per_segment", "stream", True),
)


@pytest.mark.parametrize("copy_api,src_access_order,torch_stream", COPY_ARMS)
def test_copy_arm_thread_copies_match_reference_before_wait(copy_api, src_access_order, torch_stream):
    """Each copy arm's thread copies alone produce the reference bytes.

    Bytes are checked once the thread reports the request complete and before
    ``wait`` launches, so an in-graph fallback copy cannot hide a wrong thread
    copy. The configured arm is read back from the thread, so an option that
    never reaches it fails here. Access order values are the CUDA
    ``cudaMemcpySrcAccessOrder`` enumerators.
    """
    generator = torch.Generator().manual_seed(67)
    sources = _sources(90, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 80)
    stream = torch.cuda.Stream() if torch_stream else None
    with _copier(sources, destinations, capacity=64, copy_api=copy_api, src_access_order=src_access_order,
                 stream=stream) as copier:
        requests = []
        for rows in (37, 1, 64):
            picked_rows, picked_slots = _pick(generator, 90, 80, rows)
            _zero(destinations)
            copier.post(*_plan(64, picked_rows, picked_slots))
            torch.cuda.synchronize()
            request = _completed_request(copier, copier.stats()["posted"])
            requests.append(request)
            _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
            copier.wait()
            torch.cuda.synchronize()
        stats = copier.stats()
    assert [request and request["status"] for request in requests] == ["serviced"] * 3, requests
    assert stats["copy_errors"] == 0, stats["last_copy_error"]
    assert stats["timeouts"] == 0
    assert stats["copy_api"] == {"batch": 0, "per_segment": 1}[copy_api]
    assert stats["src_access_order"] == {"stream": 1, "during_call": 2, "any": 3}[src_access_order]
    assert stats["external_stream"] == int(torch_stream)


def test_segment_sets_copy_each_tags_rows_into_its_own_destinations():
    """With one segment set per tag, each request copies through its tag's set, and a tag
    without a set is refused as an invalid record instead of copying another set's rows."""
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

    generator = torch.Generator().manual_seed(71)
    sources_a = _sources(30, EXPERT_LIKE_SHAPES, generator)
    destinations_a = _destinations(sources_a, 20)
    sources_b = _sources(25, (((17, 3), torch.uint8), ((5,), torch.uint8)), generator)
    destinations_b = _destinations(sources_b, 12)
    rows_a, slots_a = _pick(generator, 30, 20, 9)
    rows_b, slots_b = _pick(generator, 25, 12, 9)
    sets = [
        expert_row_segments(list(zip(sources_a, destinations_a))),
        expert_row_segments(list(zip(sources_b, destinations_b))),
    ]
    with ExpertDoorbellCopier(sets, 9, max_tags=3, cpu_core=SPIN_CORE) as copier:
        copier.post(*_plan(9, rows_a, slots_a), tag=0)
        copier.post(*_plan(9, rows_b, slots_b), tag=1)
        torch.cuda.synchronize()
        serviced = _completed_request(copier, copier.stats()["posted"])
        _assert_matches_reference(sources_a, destinations_a, rows_a, slots_a)
        _assert_matches_reference(sources_b, destinations_b, rows_b, slots_b)
        copier.wait(tag=0)
        copier.wait(tag=1)
        copier.post(*_plan(9, rows_a, slots_a), tag=2)
        torch.cuda.synchronize()
        invalid = _completed_request(copier, copier.stats()["posted"])
        stats = copier.stats()
    _assert_matches_reference(sources_b, destinations_b, rows_b, slots_b)
    assert serviced and serviced["status"] == "serviced"
    assert invalid and invalid["status"] == "invalid_record"
    assert stats["invalid_records"] == 1
    assert stats["timeouts"] == 0
    assert stats["late_completions"] == 0


def test_reference_check_fails_on_flipped_byte_and_stray_write():
    generator = torch.Generator().manual_seed(1)
    sources = _sources(8, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 6)
    rows, slots = [2, 5], [1, 4]
    for source, destination in zip(sources, destinations):
        _row_bytes(destination)[torch.tensor(slots, device="cuda")] = _row_bytes(source)[torch.tensor(rows)].cuda()
    assert _reference_mismatch(sources, destinations, rows, slots) is None

    planned = _row_bytes(destinations[0])
    planned[4, 3] ^= 0xFF
    assert _reference_mismatch(sources, destinations, rows, slots) == "segment 0"
    planned[4, 3] ^= 0xFF
    assert _reference_mismatch(sources, destinations, rows, slots) is None

    _row_bytes(destinations[4])[0, 0] = 1
    assert _reference_mismatch(sources, destinations, rows, slots) == "segment 4"


@pytest.mark.parametrize("rows", [0, 1, 3, 130, CAPACITY])
def test_posted_rows_match_reference(rows):
    generator = torch.Generator().manual_seed(rows + 100)
    sources = _sources(CAPACITY + 7, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, CAPACITY + 5)
    picked_rows, picked_slots = _pick(generator, CAPACITY + 7, CAPACITY + 5, rows)
    with _copier(sources, destinations) as copier:
        copier.post(*_plan(CAPACITY, picked_rows, picked_slots))
        copier.wait()
        torch.cuda.synchronize()
        stats = copier.stats()
    _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
    assert stats["timeouts"] == 0
    assert stats["serviced"] == 1
    assert stats["rows_copied"] == rows


@pytest.mark.parametrize("poll_mode", ["acquire", "volatile", "noncoherent"])
def test_poll_modes_match_reference(poll_mode):
    generator = torch.Generator().manual_seed(19)
    sources = _sources(90, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 80)
    picked_rows, picked_slots = _pick(generator, 90, 80, 37)
    with _copier(sources, destinations, capacity=64, poll_mode=poll_mode) as copier:
        copier.post(*_plan(64, picked_rows, picked_slots))
        copier.wait()
        torch.cuda.synchronize()
        assert copier.stats()["timeouts"] == 0
    _assert_matches_reference(sources, destinations, picked_rows, picked_slots)


@pytest.mark.parametrize("head_store", ["release", "volatile"])
def test_head_stores_match_reference(head_store):
    generator = torch.Generator().manual_seed(59)
    sources = _sources(90, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 80)
    picked_rows, picked_slots = _pick(generator, 90, 80, 37)
    with _copier(sources, destinations, capacity=64, head_store=head_store) as copier:
        copier.post(*_plan(64, picked_rows, picked_slots))
        copier.wait()
        torch.cuda.synchronize()
        assert copier.stats()["timeouts"] == 0
    _assert_matches_reference(sources, destinations, picked_rows, picked_slots)


def test_quiesce_drains_posted_requests_then_holds_new_ones():
    """``quiesce`` returns only after every posted request's copy completed, and the thread then
    services nothing until ``resume``."""
    generator = torch.Generator().manual_seed(61)
    sources = _sources(60, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 60)
    rows, slots = _pick(generator, 60, 60, 40)
    with _copier(sources, destinations, capacity=10) as copier:
        for tag in range(3):
            copier.post(*_plan(10, rows[10 * tag : 10 * (tag + 1)], slots[10 * tag : 10 * (tag + 1)]), tag=tag)
        copier.quiesce()
        _assert_matches_reference(sources, destinations, rows[:30], slots[:30])
        assert copier.stats()["serviced"] == 3

        copier.post(*_plan(10, rows[30:], slots[30:]), tag=3)
        torch.cuda.synchronize()
        time.sleep(0.05)
        assert copier.stats()["serviced"] == 3
        _assert_matches_reference(sources, destinations, rows[:30], slots[:30])

        copier.resume()
        copier.wait(tag=3)
        torch.cuda.synchronize()
        stats = copier.stats()
    _assert_matches_reference(sources, destinations, rows, slots)
    assert stats["serviced"] == 4
    assert stats["timeouts"] == 0


@pytest.mark.parametrize("prefer_overlap", [True, False])
def test_overlap_preference_matches_reference(prefer_overlap):
    generator = torch.Generator().manual_seed(17)
    sources = _sources(90, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 80)
    picked_rows, picked_slots = _pick(generator, 90, 80, 37)
    with _copier(sources, destinations, capacity=64, prefer_overlap=prefer_overlap) as copier:
        copier.post(*_plan(64, picked_rows, picked_slots))
        copier.wait()
        torch.cuda.synchronize()
        assert copier.stats()["timeouts"] == 0
    _assert_matches_reference(sources, destinations, picked_rows, picked_slots)


@pytest.mark.parametrize("row_bytes", [1, 3, 17, 4099, 65537])
@pytest.mark.parametrize("source_offset,destination_offset", [(0, 0), (1, 0), (5, 11)])
@pytest.mark.parametrize("rows", [1, 9])
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
    with _copier([source], [destination], capacity=rows) as copier:
        copier.post(*_plan(rows, picked_rows, picked_slots))
        copier.wait()
        torch.cuda.synchronize()
        assert copier.stats()["timeouts"] == 0
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
        destinations = _destinations(sources, rows)
        picked_rows, picked_slots = _pick(generator, rows, rows, 17)
        with _copier(sources, destinations, capacity=17) as copier:
            copier.post(*_plan(17, picked_rows, picked_slots))
            copier.wait()
            torch.cuda.synchronize()
            assert copier.stats()["timeouts"] == 0
        _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
    finally:
        for source in sources:
            _cuda_host_unregister(source)


def test_graph_replay_post_compute_wait_follows_changing_plans():
    """Bytes are exact on every replay and the thread receives every replayed request.

    Whether the thread or the in-graph fallback wrote the rows depends on the
    launch hold characterized in ``test_graph_launch_holds_thread_copies_until_launch_ends``.
    """
    generator = torch.Generator().manual_seed(11)
    capacity, source_count, slot_count = 130, 300, 140
    sources = _sources(source_count, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, slot_count)
    compute_in = torch.randn((256, 256), device="cuda")
    compute_out = torch.empty_like(compute_in)
    with _copier(sources, destinations, capacity=capacity, timeout_polls=STALL_TIMEOUT_POLLS) as copier:
        plan = _plan(capacity, *_pick(generator, source_count, slot_count, 3))

        def body():
            copier.post(*plan, tag=3)
            torch.matmul(compute_in, compute_in, out=compute_out)
            copier.wait(tag=3)

        graph = _capture(body)
        before = copier.stats()
        replays = (0, 1, 10, 64, 65, capacity, 5)
        for rows in replays:
            picked_rows, picked_slots = _pick(generator, source_count, slot_count, rows)
            _zero(destinations)
            _load_plan(plan, picked_rows, picked_slots)
            graph.replay()
            torch.cuda.synchronize()
            _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
            torch.testing.assert_close(compute_out, compute_in @ compute_in)
        time.sleep(0.05)
        after = copier.stats()
    assert after["posted"] - before["posted"] == len(replays)
    assert after["waits"] - before["waits"] == len(replays)
    handled = (after["serviced"] - before["serviced"]) + (after["skipped_abandoned"] - before["skipped_abandoned"])
    assert handled == len(replays)
    assert after["record_mismatches"] == 0


def test_capture_refuses_a_thread_created_stream():
    """Copies on a stream the thread creates are held behind CUDA-graph replays (E29, E32), so a
    copier without a caller stream refuses to be captured."""
    generator = torch.Generator().manual_seed(59)
    sources = _sources(8, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 8)
    with _copier(sources, destinations, capacity=4, stream=None) as copier:
        plan = _plan(4, [1, 2], [3, 4])
        with pytest.raises(RuntimeError, match="stream=torch.cuda.Stream"):
            _capture(lambda: copier.post(*plan))


def test_graph_launch_lets_torch_stream_copies_overlap_compute():
    """On a torch-created stream the thread's copies overlap a CUDA graph launch's compute, as
    they do when the same post and compute run eagerly (E32 on RTX 5090, driver 610.57)."""
    generator = torch.Generator().manual_seed(53)
    sources = _sources(128, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 128)
    compute_in = torch.randn((2048, 2048), device="cuda")
    compute_out = torch.empty_like(compute_in)
    with _copier(sources, destinations, capacity=16) as copier:
        plan = _plan(16, *_pick(generator, 128, 128, 16))

        def post_and_compute():
            copier.post(*plan)
            for _ in range(128):
                torch.matmul(compute_in, compute_in, out=compute_out)

        graph = _capture(lambda: post_and_compute())
        copier.wait()
        torch.cuda.synchronize()

        measured = {}
        for name, run in (("graph", graph.replay), ("eager", post_and_compute)):
            _load_plan(plan, *_pick(generator, 128, 128, 16))
            torch.cuda.synchronize()
            seq = copier.stats()["posted"] + 1
            launched = _monotonic_ns()
            run()
            event = torch.cuda.Event()
            event.record()
            complete, compute_end = _copy_and_compute_end_ns(copier, event, seq)
            copier.wait()
            torch.cuda.synchronize()
            measured[name] = ((complete - launched) / 1e6, (compute_end - launched) / 1e6)
    graph_copy_ms, graph_compute_ms = measured["graph"]
    eager_copy_ms, eager_compute_ms = measured["eager"]
    assert graph_compute_ms > 2.0 and eager_compute_ms > 2.0, measured
    assert graph_copy_ms < 0.5 * graph_compute_ms, measured
    assert eager_copy_ms < 0.5 * eager_compute_ms, measured


def test_copy_overlaps_stand_in_compute():
    generator = torch.Generator().manual_seed(23)
    shapes = (((1 << 21,), torch.uint8), ((4099,), torch.uint8))
    rows, source_count, slot_count = 30, 64, 48
    sources = _sources(source_count, shapes, generator)
    destinations = _destinations(sources, slot_count)
    picked_rows, picked_slots = _pick(generator, source_count, slot_count, rows)
    compute_in = torch.randn((4096, 4096), device="cuda")
    compute_out = torch.empty_like(compute_in)
    with _copier(sources, destinations, capacity=rows) as copier:
        plan = _plan(rows, picked_rows, picked_slots)
        torch.matmul(compute_in, compute_in, out=compute_out)
        torch.cuda.synchronize()
        compute_end = torch.cuda.Event()
        posted_ns = _monotonic_ns()
        copier.post(*plan, tag=1)
        for _ in range(40):
            torch.matmul(compute_in, compute_in, out=compute_out)
        compute_end.record()
        copier.wait(tag=1)
        while not compute_end.query():
            pass
        compute_end_ns = _monotonic_ns()
        torch.cuda.synchronize()
        request = copier.trace()[-1]
        assert copier.stats()["timeouts"] == 0
    _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
    compute_ms = (compute_end_ns - posted_ns) / 1e6
    copy_done_ms = (request["complete_ns"] - posted_ns) / 1e6
    assert request["status"] == "serviced"
    assert compute_ms > 50.0
    assert posted_ns < request["seen_ns"] < request["enqueued_ns"] <= request["complete_ns"]
    assert copy_done_ms < 0.5 * compute_ms, (copy_done_ms, compute_ms)


def test_stalled_thread_times_out_and_falls_back_with_correct_bytes():
    generator = torch.Generator().manual_seed(29)
    capacity, source_count, slot_count = 40, 100, 60
    sources = _sources(source_count, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, slot_count)
    with _copier(sources, destinations, capacity=capacity, timeout_polls=STALL_TIMEOUT_POLLS) as copier:
        plan = _plan(capacity, *_pick(generator, source_count, slot_count, 1))

        def body():
            copier.post(*plan, tag=5)
            copier.wait(tag=5)

        graph = _capture(body)
        assert copier.stats()["timeouts"] == 0

        copier.pause()
        stalled = []
        for rows in (7, capacity, 0):
            picked_rows, picked_slots = _pick(generator, source_count, slot_count, rows)
            _zero(destinations)
            _load_plan(plan, picked_rows, picked_slots)
            started = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            stalled.append(time.perf_counter() - started)
            _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
        for rows in (5, 0):
            picked_rows, picked_slots = _pick(generator, source_count, slot_count, rows)
            _zero(destinations)
            _load_plan(plan, picked_rows, picked_slots)
            copier.post(*plan, tag=5)
            copier.wait(tag=5)
            torch.cuda.synchronize()
            _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
        stats = copier.stats()
        assert stats["timeouts"] == 5
        assert stats["serviced"] == 2
        assert stats["record_mismatches"] == 0
        assert max(stalled) < 5.0

        copier.resume()
        time.sleep(0.05)
        for rows in (3, 11, 11):
            picked_rows, picked_slots = _pick(generator, source_count, slot_count, rows)
            _zero(destinations)
            _load_plan(plan, picked_rows, picked_slots)
            copier.post(*plan, tag=5)
            copier.wait(tag=5)
            torch.cuda.synchronize()
            _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
        recovered = copier.stats()
    assert recovered["skipped_abandoned"] == 5
    assert recovered["serviced"] == stats["serviced"] + 3
    assert recovered["timeouts"] == stats["timeouts"]
    assert recovered["degraded"] == 0


def test_timed_out_wait_drains_a_claimed_request_until_its_copies_land():
    """A request the thread claimed before its wait timed out is drained, not copied again: the
    wait returns only after the thread's delayed copies landed, so nothing lands after it.

    The thread copies on a torch-created stream, as serving does: copies queued on a stream the
    thread created are held back until the running drain chunk ends."""
    generator = torch.Generator().manual_seed(37)
    sources = _sources(20, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 20)
    rows, slots = _pick(generator, 20, 20, 7)
    with _copier(
        sources,
        destinations,
        capacity=8,
        timeout_polls=STALL_TIMEOUT_POLLS,
        drain_polls=400_000_000,
        stream=torch.cuda.Stream(),
    ) as copier:
        copier.inject_fault(service_delay_s=1.0)
        copier.post(*_plan(8, rows, slots))
        torch.cuda.synchronize()
        started = time.perf_counter()
        copier.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stats = copier.stats()
        delivered = int(copier.delivered[0].item())
        _assert_matches_reference(sources, destinations, rows, slots)
        _zero(destinations)
        time.sleep(1.5)
        torch.cuda.synchronize()
        landed_after = [int(destination.view(torch.uint8).count_nonzero()) for destination in destinations]
    assert stats["timeouts"] == 1
    assert stats["drains"] == 1
    assert stats["drain_timeouts"] == 0
    assert stats["disabled"] == 0
    assert delivered == 1
    assert stats["done"] >= stats["posted"]
    assert elapsed >= 0.8
    assert landed_after == [0] * len(destinations)


def test_timed_out_wait_on_an_unclaimed_request_falls_back_without_draining():
    """A paused thread never claims the request, so the wait falls back after its timeout
    instead of spending the drain budget."""
    generator = torch.Generator().manual_seed(41)
    sources = _sources(8, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 8)
    with _copier(
        sources, destinations, capacity=4, timeout_polls=STALL_TIMEOUT_POLLS, drain_polls=400_000_000
    ) as copier:
        copier.pause()
        picked_rows, picked_slots = [3, 0], [5, 2]
        copier.post(*_plan(4, picked_rows, picked_slots))
        torch.cuda.synchronize()
        started = time.perf_counter()
        copier.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stats = copier.stats()
        delivered = int(copier.delivered[0].item())
        _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
    assert stats["timeouts"] == 1
    assert stats["drains"] == 0
    assert stats["drain_timeouts"] == 0
    assert delivered == 0
    assert elapsed < 1.0


def test_failed_copy_is_published_unserviced_and_the_wait_falls_back_at_once():
    """A request whose copy failed is published marked unserviced, so its wait copies it in-graph
    without spending the timeout."""
    generator = torch.Generator().manual_seed(43)
    sources = _sources(20, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 20)
    rows, slots = _pick(generator, 20, 20, 6)
    with _copier(sources, destinations, capacity=8, timeout_polls=400_000_000) as copier:
        copier.inject_fault(fail_copies=True)
        copier.post(*_plan(8, rows, slots))
        torch.cuda.synchronize()
        started = time.perf_counter()
        copier.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stats = copier.stats()
        delivered = int(copier.delivered[0].item())
        _assert_matches_reference(sources, destinations, rows, slots)
        trace = copier.trace()
    assert trace[-1]["status"] == "copy_failed"
    assert stats["copy_errors"] == 1
    assert stats["timeouts"] == 0
    assert delivered == 0
    assert elapsed < 1.0


def test_timeout_polls_bound_the_wait_when_thread_is_paused():
    generator = torch.Generator().manual_seed(31)
    sources = _sources(8, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 8)
    with _copier(sources, destinations, capacity=4, timeout_polls=STALL_TIMEOUT_POLLS) as copier:
        copier.pause()
        picked_rows, picked_slots = [1, 6], [0, 7]
        copier.post(*_plan(4, picked_rows, picked_slots))
        torch.cuda.synchronize()
        started = time.perf_counter()
        copier.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stats = copier.stats()
    _assert_matches_reference(sources, destinations, picked_rows, picked_slots)
    assert stats["timeouts"] == 1
    assert stats["last_polls"] == STALL_TIMEOUT_POLLS
    assert elapsed < 5.0


def test_outstanding_requests_for_two_tags_both_copy():
    generator = torch.Generator().manual_seed(37)
    sources = _sources(50, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 40)
    rows, slots = _pick(generator, 50, 40, 20)
    with _copier(sources, destinations, capacity=10) as copier:
        copier.post(*_plan(10, rows[:10], slots[:10]), tag=1)
        copier.post(*_plan(10, rows[10:], slots[10:]), tag=2)
        copier.wait(tag=1)
        copier.wait(tag=2)
        torch.cuda.synchronize()
        stats = copier.stats()
    _assert_matches_reference(sources, destinations, rows, slots)
    assert stats["timeouts"] == 0
    assert stats["serviced"] == 2


def test_resolve_reports_only_its_own_tags_delivery():
    """With two target-layer tags outstanding, each resolve reports its own tag's request, and a
    resolve of a tag with nothing posted reports nothing delivered without waiting."""
    generator = torch.Generator().manual_seed(67)
    sources = _sources(50, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 40)
    rows, slots = _pick(generator, 50, 40, 12)
    with _copier(sources, destinations, capacity=6, max_tags=4, timeout_polls=400_000_000) as copier:
        copier.pause()
        copier.post(*_plan(6, rows[:6], slots[:6]), tag=1)
        copier.post(*_plan(6, rows[6:], slots[6:]), tag=2)
        empty = copier.resolve(tag=3)
        torch.cuda.synchronize()
        empty_flag = int(empty.item())
        idle = copier.stats()
        copier.resume()
        second = copier.resolve(tag=2)
        torch.cuda.synchronize()
        second_flag = int(second.item())
        first_before_resolve = int(copier.delivered[1].item())
        first = copier.resolve(tag=1)
        torch.cuda.synchronize()
        first_flag = int(first.item())
        stats = copier.stats()
    _assert_matches_reference(sources, destinations, rows, slots)
    assert empty_flag == 0
    assert idle["waits"] == 0 and idle["timeouts"] == 0
    assert second_flag == 1
    assert first_before_resolve == 0
    assert first_flag == 1
    assert stats["timeouts"] == 0
    assert stats["serviced"] == 2


def test_resolving_an_earlier_tag_does_not_wait_for_a_later_tags_request():
    """The thread services and publishes requests in order, so resolving tag L returns once L's
    own request lands even while a later tag's request is still being copied."""
    generator = torch.Generator().manual_seed(71)
    sources = _sources(40, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 40)
    rows, slots = _pick(generator, 40, 40, 8)
    with _copier(sources, destinations, capacity=4, max_tags=4, timeout_polls=400_000_000) as copier:
        copier.inject_fault(service_delay_s=0.5)
        copier.post(*_plan(4, rows[:4], slots[:4]), tag=1)
        copier.post(*_plan(4, rows[4:], slots[4:]), tag=2)
        torch.cuda.synchronize()
        started = time.perf_counter()
        copier.resolve(tag=1)
        torch.cuda.synchronize()
        first_s = time.perf_counter() - started
        copier.resolve(tag=2)
        torch.cuda.synchronize()
        both_s = time.perf_counter() - started
        stats = copier.stats()
    _assert_matches_reference(sources, destinations, rows, slots)
    assert stats["serviced"] == 2 and stats["timeouts"] == 0
    assert first_s < 0.8, first_s
    assert both_s >= 0.9, both_s


_FAIL_STOP_CHILD = """
import sys
import torch
from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

core = int(sys.argv[1])
source = torch.randint(0, 256, (8, 64), dtype=torch.uint8).pin_memory()
destination = torch.zeros((8, 64), dtype=torch.uint8, device="cuda")
copier = ExpertDoorbellCopier(
    expert_row_segments([(source, destination)]),
    4,
    cpu_core=core,
    stream=torch.cuda.Stream(),
    timeout_polls=20_000,
    drain_polls=70_000,
    fatal_wait_s=0.5,
)
copier.inject_fault(service_delay_s=6.0)
rows = torch.tensor([1, 2, 0, 0], dtype=torch.int64, device="cuda")
slots = torch.tensor([3, 4, 0, 0], dtype=torch.int32, device="cuda")
count = torch.tensor([2], dtype=torch.int32, device="cuda")
copier.post(rows, slots, count)
print("POSTED", flush=True)
copier.resolve()
torch.cuda.synchronize()
print("RETURNED", copier.stats(), flush=True)
"""


def test_a_disabled_drain_that_outlives_the_fatal_wait_aborts_the_process():
    """Fail-stop: a committed copy stuck past the drain budget disables the copier, and if it
    still has not landed after the fatal wait the watchdog aborts the process with an ERROR
    instead of hanging the resolve or returning before the copy lands."""
    import signal
    import subprocess
    import sys

    started = time.perf_counter()
    child = subprocess.run(
        [sys.executable, "-c", _FAIL_STOP_CHILD, str(SPIN_CORE)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.perf_counter() - started
    assert "POSTED" in child.stdout, child.stderr[-2000:]
    assert "RETURNED" not in child.stdout, child.stdout
    assert child.returncode == -signal.SIGABRT, (child.returncode, child.stderr[-2000:])
    assert "ERROR expert doorbell" in child.stderr, child.stderr[-2000:]
    assert elapsed < 60.0


def test_stop_drains_posted_requests_and_is_idempotent():
    generator = torch.Generator().manual_seed(41)
    sources = _sources(30, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 30)
    rows, slots = _pick(generator, 30, 30, 12)
    for _ in range(4):
        _zero(destinations)
        copier = _copier(sources, destinations, capacity=12)
        assert copier.stats()["running"] == 1
        copier.post(*_plan(12, rows, slots))
        copier.stop()
        assert copier.stats()["running"] == 0
        copier.stop()
        _assert_matches_reference(sources, destinations, rows, slots)
        with pytest.raises(RuntimeError, match="stopped"):
            copier.post(*_plan(12, rows, slots))


def test_a_plan_with_rows_outside_a_segment_is_refused_whole_and_resolves_undelivered():
    """The thread copies none of a plan that names a row or slot outside a segment and publishes
    it unserviced, so its resolve reports nothing delivered instead of a partial copy."""
    generator = torch.Generator().manual_seed(43)
    sources = _sources(10, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 10)
    _zero(destinations)
    with _copier(sources, destinations, capacity=3, timeout_polls=400_000_000) as copier:
        copier.post(*_plan(3, [1, 10, 2], [0, 1, 99]))
        delivered = copier.resolve()
        torch.cuda.synchronize()
        stats = copier.stats()
        delivered_flag = int(delivered.item())
        status = copier.trace()[-1]["status"]
    assert [int(destination.view(torch.uint8).count_nonzero()) for destination in destinations] == [0] * len(
        destinations
    )
    assert delivered_flag == 0
    assert status == "invalid_record"
    assert stats["copy_errors"] == 1
    assert stats["rows_copied"] == 0
    assert stats["timeouts"] == 0


def test_rejects_invalid_construction_and_plans():
    generator = torch.Generator().manual_seed(47)
    sources = _sources(10, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 10)
    with pytest.raises(ValueError, match="capacity"):
        _copier(sources, destinations, capacity=0)
    with pytest.raises(ValueError, match="poll_mode"):
        _copier(sources, destinations, capacity=4, poll_mode="bogus")
    with pytest.raises(ValueError, match="copy_api"):
        _copier(sources, destinations, capacity=4, copy_api="bogus")
    with pytest.raises(ValueError, match="src_access_order"):
        _copier(sources, destinations, capacity=4, src_access_order="bogus")
    with pytest.raises(ValueError, match="only to copy_api='batch'"):
        _copier(sources, destinations, capacity=4, copy_api="per_segment", src_access_order="any")
    with _copier(sources, destinations, capacity=4, max_tags=2) as copier:
        source_rows, destination_slots, count = _plan(4, [1], [1])
        with pytest.raises(ValueError, match="capacity"):
            copier.post(*_plan(5, [1], [1]))
        with pytest.raises(ValueError, match="tag"):
            copier.post(source_rows, destination_slots, count, tag=2)
        with pytest.raises(ValueError, match="tag"):
            copier.wait(tag=-1)
        with pytest.raises(ValueError, match="int32"):
            copier.post(source_rows, destination_slots.to(torch.int64), count)
