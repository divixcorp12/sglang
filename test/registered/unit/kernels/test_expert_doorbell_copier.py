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
PRIME_IN_TESTS = os.environ.get("DOORBELL_TEST_UNPRIMED") != "1"
PRIME_SKIPS = 1 if PRIME_IN_TESTS else 0
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
    """A copier over ``sources`` -> ``destinations``, primed as serving primes it unless
    ``prime_slot`` is passed explicitly or DOORBELL_TEST_UNPRIMED=1. A default prime writes the
    last destination slot, which is zeroed again before the copier is returned."""
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

    kwargs.setdefault("cpu_core", SPIN_CORE)
    kwargs.setdefault("stream", torch.cuda.Stream())
    default_prime = "prime_slot" not in kwargs
    if default_prime:
        kwargs["prime_slot"] = min(d.shape[0] for d in destinations) - 1 if PRIME_IN_TESTS else None
    copier = ExpertDoorbellCopier(expert_row_segments(list(zip(sources, destinations))), capacity, **kwargs)
    if default_prime and PRIME_IN_TESTS:
        _zero(destinations)
    return copier


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
    assert recovered["skipped_abandoned"] == 5 + PRIME_SKIPS
    assert recovered["serviced"] == stats["serviced"] + 3
    assert recovered["timeouts"] == stats["timeouts"]
    assert recovered["degraded"] == 0


def _wait_for_claim(copier, deadline_s=30.0):
    """Block until the thread has written the latest posted sequence into the page's claimed word.

    Waiting on the claim makes "the thread claimed the request before the wait timed out" a
    precondition instead of a race the thread has to win within the wait's poll budget."""
    posted = int(copier.stats()["posted"]) & 0xFFFFFFFF
    claimed_word = copier.page[4:8].view(torch.int32)
    deadline = time.perf_counter() + deadline_s
    while time.perf_counter() < deadline:
        claimed = int(claimed_word.item()) & 0xFFFFFFFF
        if claimed == posted:
            return
        time.sleep(0.0005)
    raise AssertionError(
        f"the doorbell thread did not claim sequence {posted} within {deadline_s} s "
        f"(claimed word {int(claimed_word.item()) & 0xFFFFFFFF}); the drain precondition never held"
    )


def test_timed_out_wait_drains_a_claimed_request_until_its_copies_land():
    """A request the thread claimed before its wait timed out is drained, not copied again: the
    wait returns only after the thread's delayed copies landed, so nothing lands after it.

    The test waits for the thread's claim before calling wait(), so the drain path is exercised
    whatever the CPU scheduling. The thread copies on a torch-created stream, as serving does.

    The copy is held 60 ms after the claim against a 524,288-poll drain (134-142 ms at 256-270 ns
    per poll, the serving drain budget), so the drain must see it land. The copier is primed as
    serving primes it (``_copier``); an unprimed process stalls this overlap until the drain runs
    out (E34), which DOORBELL_TEST_UNPRIMED=1 shows as a failure here.

    This is where the positive recovery property lives, because this is the path that can deliver
    it. Measured on divix01 (E35i): the copy is enqueued 60.002 ms after the wait starts, i.e.
    ~55 ms AFTER the drain kernel went resident (the drain becomes resident 5.12-5.40 ms in, at
    20,000 polls), and it completes in 6 MICROSECONDS with the drain still spinning; nsys (E35j)
    shows two copies executing inside this arm's drain window on the copier's own stream while the
    drain kernel occupies stream 7. The graph-gather path cannot do this at any budget -- see
    test_doorbell_drain_on_this_path_ends_at_its_budget_and_never_recovers_the_copy -- so asserting
    recovery there is a test that cannot pass, and it is asserted here instead.

    Red: raising service_delay_s above the drain budget, or running under DOORBELL_TEST_UNPRIMED=1,
    turns drain_timeouts to 1 and delivered to 0 and fails these assertions."""
    generator = torch.Generator().manual_seed(37)
    sources = _sources(20, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 20)
    rows, slots = _pick(generator, 20, 19, 7)
    with _copier(
        sources,
        destinations,
        capacity=8,
        timeout_polls=STALL_TIMEOUT_POLLS,
        drain_polls=524_288,
        stream=torch.cuda.Stream(),
    ) as copier:
        copier.inject_fault(service_delay_s=0.06)
        copier.post(*_plan(8, rows, slots))
        torch.cuda.synchronize()
        _wait_for_claim(copier)
        started = time.perf_counter()
        copier.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stats = copier.stats()
        delivered = int(copier.delivered[0].item())
        _assert_matches_reference(sources, destinations, rows, slots)
        _zero(destinations)
        time.sleep(0.3)
        torch.cuda.synchronize()
        landed_after = [int(destination.view(torch.uint8).count_nonzero()) for destination in destinations]
    assert stats["timeouts"] == 1
    assert stats["drains"] == 1
    assert stats["drain_timeouts"] == 0, (stats, elapsed)
    assert stats["disabled"] == 0
    assert delivered == 1
    assert stats["done"] >= stats["posted"]
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
import time
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
claimed_word = copier.page[4:8].view(torch.int32)
deadline = time.perf_counter() + 30.0
while int(claimed_word.item()) != 1:
    if time.perf_counter() > deadline:
        raise AssertionError("the doorbell thread never claimed the request; the drain precondition never held")
    time.sleep(0.0005)
copier.resolve()
torch.cuda.synchronize()
print("RETURNED", flush=True)
copier.fail_stop_check()
print("CHECKED", copier.stats(), flush=True)
"""


def test_a_disabled_drain_that_outlives_the_fatal_wait_aborts_the_process():
    """Fail-stop: a committed copy stuck past the drain budget disables the copier and resolves
    undelivered at once, and the host fail-stop check that must run before the next forward holds
    until the copy lands; if it has not landed after the fatal wait the watchdog aborts the
    process with an ERROR instead of letting the next forward run or hanging it."""
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
    assert "RETURNED" in child.stdout, (child.stdout[-2000:], child.stderr[-2000:])
    assert "CHECKED" not in child.stdout, child.stdout
    assert child.returncode == -signal.SIGABRT, (child.returncode, child.stderr[-2000:])
    assert "ERROR expert doorbell" in child.stderr, child.stderr[-2000:]
    assert elapsed < 60.0


_COLD_LAUNCH_DURING_DRAIN_CHILD = """
import json
import sys
import time
import torch
from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

core = int(sys.argv[1])
TIMEOUT_POLLS = 20_000
DRAIN_POLLS = 524_288
CALIBRATION_POLLS = 2_000_000
source = torch.randint(0, 256, (8, 64), dtype=torch.uint8).pin_memory()
destination = torch.zeros((8, 64), dtype=torch.uint8, device="cuda")
rows = torch.tensor([1, 2, 0, 0], dtype=torch.int64, device="cuda")
slots = torch.tensor([3, 4, 0, 0], dtype=torch.int32, device="cuda")
count = torch.tensor([2], dtype=torch.int32, device="cuda")

with ExpertDoorbellCopier(
    expert_row_segments([(source, destination)]), 4, cpu_core=core, stream=torch.cuda.Stream(),
    timeout_polls=CALIBRATION_POLLS, drain_polls=DRAIN_POLLS, fatal_wait_s=30.0,
) as calibration:
    calibration.pause()
    calibration.post(rows, slots, count)
    torch.cuda.synchronize()
    started = time.perf_counter()
    calibration.resolve()
    torch.cuda.synchronize()
    poll_s = (time.perf_counter() - started) / CALIBRATION_POLLS
    calibration.resume()

copier = ExpertDoorbellCopier(
    expert_row_segments([(source, destination)]), 4, cpu_core=core, stream=torch.cuda.Stream(),
    timeout_polls=TIMEOUT_POLLS, drain_polls=DRAIN_POLLS, fatal_wait_s=5.0,
)
for _ in range(2):
    copier.post(rows, slots, count)
    copier.wait()
    torch.cuda.synchronize()
destination.zero_()
torch.cuda.synchronize()
left = torch.empty(5, dtype=torch.int16, device="cuda")
right = torch.empty(5, dtype=torch.int16, device="cuda")
out = torch.empty(5, dtype=torch.int16, device="cuda")
copier.inject_fault(service_delay_s=1.0)
copier.post(rows, slots, count)
torch.cuda.synchronize()
claimed_word = copier.page[4:8].view(torch.int32)
posted = int(copier.stats()["posted"]) & 0xFFFFFFFF
deadline = time.perf_counter() + 30.0
while time.perf_counter() < deadline and (int(claimed_word.item()) & 0xFFFFFFFF) != posted:
    time.sleep(0.0005)
print("CLAIMED", flush=True)
copier.wait()
time.sleep(0.05)
launch_started = time.perf_counter()
torch.bitwise_xor(left, right, out=out)
launch_s = time.perf_counter() - launch_started
torch.cuda.synchronize()
time.sleep(1.5)
torch.cuda.synchronize()
bytes_ok = bool(torch.equal(destination[3:5].cpu(), source[1:3]))
print("RESULT", json.dumps({
    "poll_s": poll_s, "launch_s": launch_s, "timeout_polls": TIMEOUT_POLLS, "drain_polls": DRAIN_POLLS,
    "bytes_ok": bytes_ok, "stats": copier.stats(),
}), flush=True)
copier.stop()
"""


def test_a_cold_launch_during_a_drain_returns_within_the_queued_wait_bound():
    """A never-launched kernel's launch blocks until all queued device work finishes, and the
    thread's copy call blocks with it (E34, E34f). A drain running into such a launch must end
    within its own bounded budget, fall back in-graph with correct bytes, and not abort the
    process when the held copy lands afterwards, instead of waiting for a copy that cannot land
    until the drain and an in-kernel fatal poll give up.

    Bound: the whole queued wait is at most timeout_polls + drain_polls polls, at the poll cost
    the child measures on this GPU; x1.25 for poll jitter, plus 0.5 s, which is 13x the largest
    first-launch cost measured on an idle device (37 ms, E34f)."""
    import json
    import subprocess
    import sys

    child = subprocess.run(
        [sys.executable, "-c", _COLD_LAUNCH_DURING_DRAIN_CHILD, str(SPIN_CORE)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, (child.returncode, child.stdout[-2000:], child.stderr[-2000:])
    result = json.loads(child.stdout.split("RESULT", 1)[1].strip().splitlines()[0])
    bound_s = (result["timeout_polls"] + result["drain_polls"]) * result["poll_s"] * 1.25 + 0.5
    assert result["launch_s"] <= bound_s, (result, bound_s)
    assert result["bytes_ok"], result
    assert result["stats"]["drain_timeouts"] == 1, result
    assert "ERROR expert doorbell: request" not in child.stderr, child.stderr[-2000:]


_LATE_LANDING_CHILD = """
import json
import sys
import time
import torch
from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

core = int(sys.argv[1])
source = torch.randint(0, 256, (8, 64), dtype=torch.uint8).pin_memory()
destination = torch.zeros((8, 64), dtype=torch.uint8, device="cuda")
copier = ExpertDoorbellCopier(
    expert_row_segments([(source, destination)]), 4, cpu_core=core, stream=torch.cuda.Stream(),
    timeout_polls=20_000, drain_polls=70_000, fatal_wait_s=5.0,
)
rows = torch.tensor([1, 2, 0, 0], dtype=torch.int64, device="cuda")
slots = torch.tensor([3, 4, 0, 0], dtype=torch.int32, device="cuda")
count = torch.tensor([2], dtype=torch.int32, device="cuda")
copier.post(rows, slots, count)
copier.wait()
torch.cuda.synchronize()
destination.zero_()
torch.cuda.synchronize()
copier.inject_fault(service_delay_s=1.0)
copier.post(rows, slots, count)
torch.cuda.synchronize()
claimed_word = copier.page[4:8].view(torch.int32)
posted = int(copier.stats()["posted"]) & 0xFFFFFFFF
deadline = time.perf_counter() + 30.0
while time.perf_counter() < deadline and (int(claimed_word.item()) & 0xFFFFFFFF) != posted:
    time.sleep(0.0005)
started = time.perf_counter()
copier.wait()
torch.cuda.synchronize()
resolved_s = time.perf_counter() - started
fallback_ok = bool(torch.equal(destination[3:5].cpu(), source[1:3]))
time.sleep(2.5)
torch.cuda.synchronize()
landed_ok = bool(torch.equal(destination[3:5].cpu(), source[1:3]))
print("RESULT", json.dumps({
    "resolved_s": resolved_s, "fallback_ok": fallback_ok, "landed_ok": landed_ok,
    "delivered": int(copier.delivered[0].item()), "stats": copier.stats(),
}), flush=True)
copier.stop()
"""


def test_a_drain_that_runs_out_resolves_undelivered_at_once_and_a_late_landing_does_not_abort():
    """A claimed copy still in flight when the drain budget runs out: the resolve returns at once
    with the request undelivered and the copier disabled, the in-graph fallback writes the
    planned bytes, and the thread's copy landing later (well inside the fatal wait) clears the
    fail-stop without aborting the process. No resolve waits in the kernel for the fatal wait.

    resolved_s < 0.8: the thread's copy is held for 1.0 s, and the timeout plus drain budgets are
    90,000 polls, 23-93 ms at 256-1,028 ns per poll."""
    import json
    import subprocess
    import sys

    child = subprocess.run(
        [sys.executable, "-c", _LATE_LANDING_CHILD, str(SPIN_CORE)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, (child.returncode, child.stdout[-2000:], child.stderr[-2000:])
    result = json.loads(child.stdout.split("RESULT", 1)[1].strip().splitlines()[0])
    assert result["resolved_s"] < 0.8, result
    assert result["delivered"] == 0, result
    assert result["fallback_ok"] and result["landed_ok"], result
    assert result["stats"]["drain_timeouts"] == 1, result
    assert result["stats"]["disabled"] == 1, result
    assert "ERROR expert doorbell: request" not in child.stderr, child.stderr[-2000:]


_TWO_EXHAUSTED_DRAINS_CHILD = """
import json
import sys
import time
import torch
from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

core = int(sys.argv[1])
second_landing_s = float(sys.argv[2])
fatal_wait_s = float(sys.argv[3])
resolve_order = (0, 1) if sys.argv[4] == "ascending" else (1, 0)
source = torch.randint(0, 256, (8, 64), dtype=torch.uint8).pin_memory()
destination = torch.zeros((8, 64), dtype=torch.uint8, device="cuda")
copier = ExpertDoorbellCopier(
    expert_row_segments([(source, destination)]), 4, max_tags=2, cpu_core=core, stream=torch.cuda.Stream(),
    timeout_polls=20_000, drain_polls=70_000, fatal_wait_s=fatal_wait_s, prime_slot=0,
)
plans = {
    0: (torch.tensor([1, 2, 0, 0], dtype=torch.int64, device="cuda"),
        torch.tensor([3, 4, 0, 0], dtype=torch.int32, device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda")),
    1: (torch.tensor([5, 6, 0, 0], dtype=torch.int64, device="cuda"),
        torch.tensor([6, 7, 0, 0], dtype=torch.int32, device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda")),
}
for tag in (0, 1):
    copier.post(*plans[tag], tag=tag)
    copier.wait(tag=tag)
    torch.cuda.synchronize()
destination.zero_()
torch.cuda.synchronize()
copier.inject_fault(defer_landing_s=(1.0, second_landing_s))
for tag in (0, 1):
    copier.post(*plans[tag], tag=tag)
torch.cuda.synchronize()
claimed_word = copier.page[4:8].view(torch.int32)
posted = int(copier.stats()["posted"]) & 0xFFFFFFFF
deadline = time.perf_counter() + 30.0
while time.perf_counter() < deadline and (int(claimed_word.item()) & 0xFFFFFFFF) != posted:
    time.sleep(0.0005)
claimed_at = time.perf_counter()
# Both resolve orders matter. Descending (tag 1 first) makes the LAST drain to run out carry the
# LOWER sequence, so a fatal word holding the latest sequence would wait only for the first
# landing; ascending makes the FIRST one carry the lower sequence, so a fatal word that keeps the
# sequence it saw first would do the same. Only "highest sequence wins" passes both.
for tag in resolve_order:
    copier.wait(tag=tag)
torch.cuda.synchronize()
resolved_s = time.perf_counter() - claimed_at
delivered = copier.delivered.cpu().tolist()
fatal_word = int(copier.page[12:16].view(torch.int32).item()) & 0xFFFFFFFF
print("RESOLVED", json.dumps({"resolved_s": resolved_s, "delivered": delivered, "fatal_word": fatal_word, "posted": posted}), flush=True)
waited_s = copier.fail_stop_check()
checked_s = time.perf_counter() - claimed_at
completed_seq = copier.completed_seq()
bytes_ok = bool(torch.equal(destination[3:5].cpu(), source[1:3]) and torch.equal(destination[6:8].cpu(), source[5:7]))
fresh = (torch.tensor([7, 0, 0, 0], dtype=torch.int64, device="cuda"),
         torch.tensor([0, 0, 0, 0], dtype=torch.int32, device="cuda"),
         torch.tensor([1], dtype=torch.int32, device="cuda"))
copier.post(*fresh, tag=0)
copier.wait(tag=0)
torch.cuda.synchronize()
fresh_ok = bool(torch.equal(destination[0:1].cpu(), source[7:8]))
print("CHECKED", json.dumps({
    "waited_s": waited_s, "checked_s": checked_s, "completed_seq": completed_seq, "bytes_ok": bytes_ok,
    "fresh_ok": fresh_ok, "stats": copier.stats(),
}), flush=True)
copier.stop()
"""


def _run_two_exhausted_drains(second_landing_s, fatal_wait_s, resolve_order="descending"):
    import subprocess
    import sys

    return subprocess.run(
        [
            sys.executable,
            "-c",
            _TWO_EXHAUSTED_DRAINS_CHILD,
            str(SPIN_CORE),
            str(second_landing_s),
            str(fatal_wait_s),
            resolve_order,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _child_json(stdout, label):
    return __import__("json").loads(stdout.split(label, 1)[1].strip().splitlines()[0])


@pytest.mark.parametrize("resolve_order", ["ascending", "descending"])
def test_two_drains_exhausted_in_one_sequence_hold_the_fail_stop_until_both_late_copies_land(resolve_order):
    """Two tags' requests are both committed (claimed) before either resolve, and each copy lands
    only after its drain runs out. Both drains exhaust in the same eager sequence; the fatal word
    holds the higher sequence, and the host fail-stop check returns only once the thread has
    completed every committed sequence up to it: after the second landing, not the first. Nothing
    aborts, both plans' bytes are correct, and a post on the now disabled copier is served in-graph.

    The landings are deferred 1.0 s and 1.5 s after the claim; the timeout plus drain budget is
    90,000 polls (23-93 ms at 256-1,028 ns per poll), so resolved_s < 0.8 separates a bounded
    drain from one that waits for a landing, and checked_s >= 1.4 shows the check waited for the
    second landing."""
    child = _run_two_exhausted_drains(1.5, 5.0, resolve_order)
    assert child.returncode == 0, (child.returncode, child.stdout[-2000:], child.stderr[-2000:])
    resolved = _child_json(child.stdout, "RESOLVED")
    checked = _child_json(child.stdout, "CHECKED")
    assert resolved["resolved_s"] < 0.8, resolved
    assert resolved["delivered"] == [0, 0], resolved
    assert resolved["fatal_word"] == resolved["posted"], resolved
    assert 1.4 <= checked["checked_s"] < 5.0, checked
    assert checked["waited_s"] > 0.0, checked
    assert checked["completed_seq"] >= resolved["posted"], checked
    assert checked["bytes_ok"] and checked["fresh_ok"], checked
    assert checked["stats"]["drain_timeouts"] == 2, checked
    assert checked["stats"]["disabled"] == 1, checked
    assert checked["stats"]["disabled_posts"] == 1, checked
    assert child.stderr.count("ERROR expert doorbell: drain exhausted") == 1, child.stderr[-2000:]


def test_a_second_exhausted_drain_whose_copy_never_lands_aborts_after_the_fatal_wait():
    """The same two exhausted drains, but the second committed copy lands 30 s after its claim,
    past a 2 s fatal wait. The first landing (1.0 s) does not satisfy the fail-stop: the
    watchdog aborts the process with an ERROR while the host check is still waiting."""
    import signal

    started = time.perf_counter()
    child = _run_two_exhausted_drains(30.0, 2.0)
    elapsed = time.perf_counter() - started
    assert "RESOLVED" in child.stdout, (child.stdout[-2000:], child.stderr[-2000:])
    assert "CHECKED" not in child.stdout, child.stdout[-2000:]
    assert child.returncode == -signal.SIGABRT, (child.returncode, child.stderr[-2000:])
    assert "did not complete" in child.stderr, child.stderr[-2000:]
    assert elapsed < 30.0


def test_the_fail_stop_check_without_an_exhausted_drain_costs_one_stream_synchronize(monkeypatch):
    """A step whose resolves all landed reads host words only: the check must not wait, must never
    synchronize the whole device, and must cost nothing beyond the single current-stream
    synchronize its caller asked for.

    It deliberately does NOT assert that a step without posts skips the synchronize. A replayed
    CUDA graph posts from the device without entering Python, so any "did a post run" flag reads
    False on exactly the steps that posted; a check that skipped the synchronize on that basis
    would read a stale zero fatal word and return at once, which is the failure the check exists
    to catch. ``synchronize=True`` therefore always synchronizes, and the property worth pinning
    is that it is ONE stream synchronize and never a device-wide one.

    Red: restoring the skip (gating the synchronize on a posted-since-check flag) leaves
    ``synchronizes`` empty for the synchronize=True call and fails the stream assertion below;
    widening it to torch.cuda.synchronize() records "device" and fails the device assertion."""
    generator = torch.Generator().manual_seed(97)
    sources = _sources(20, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 20)
    rows, slots = _pick(generator, 20, 19, 5)
    with _copier(sources, destinations, capacity=8) as copier:
        copier.post(*_plan(8, rows, slots))
        copier.wait()
        torch.cuda.synchronize()

        synchronizes = []
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: synchronizes.append("device"))
        monkeypatch.setattr(torch.cuda.Stream, "synchronize", lambda self: synchronizes.append("stream"))
        started = time.perf_counter()
        waited_without_synchronize = copier.fail_stop_check()
        after_caller_synchronized = list(synchronizes)
        waited_with_synchronize = copier.fail_stop_check(synchronize=True)
        elapsed = time.perf_counter() - started
        monkeypatch.undo()
        stats = copier.stats()
    assert "device" not in synchronizes, synchronizes
    assert after_caller_synchronized == [], after_caller_synchronized
    assert synchronizes == ["stream"], synchronizes
    assert waited_without_synchronize == 0.0 and waited_with_synchronize == 0.0
    assert elapsed < 0.01
    assert stats["disabled"] == 0 and stats["drain_timeouts"] == 0


def test_the_constructor_prime_writes_only_its_slot_and_leaves_serving_state_clean():
    """The prime is a paused timed-out wait with a one-row residual copy into ``prime_slot``. It
    must queue no thread copy (the thread, resumed, claims the abandoned prime request and skips
    it), write nothing but that slot, and leave the resolve counters, degraded and disabled clean,
    so a later request is serviced normally. The abandoned word is cleared only after the thread
    consumed the prime request, and the reset keeps the sequence counters."""
    generator = torch.Generator().manual_seed(101)
    sources = _sources(20, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 20)
    with _copier(sources, destinations, capacity=8, prime_slot=19) as copier:
        torch.cuda.synchronize()
        primed_mismatch = _reference_mismatch(sources, destinations, [0], [19])
        primed = copier.stats()
        prime_seq = int(primed["posted"])
        prime_row = copier.trace()[-1]
        abandoned = int(copier.page[16:20].view(torch.int32).item())
        prime_s = copier.prime_s
        _zero(destinations)
        rows, slots = _pick(generator, 20, 19, 6)
        copier.post(*_plan(8, rows, slots))
        copier.wait()
        torch.cuda.synchronize()
        served_mismatch = _reference_mismatch(sources, destinations, rows, slots)
        served = copier.stats()
    assert primed_mismatch is None
    assert primed["serviced"] == 0 and primed["skipped_abandoned"] == 1
    assert prime_row["seq"] == prime_seq and prime_row["status"] == "skipped_abandoned", prime_row
    assert prime_row["bytes"] == 0 or prime_row["enqueued_ns"] != 0, prime_row
    for name in ("timeouts", "waits", "degraded", "disabled", "drains", "drain_timeouts"):
        assert primed[name] == 0, (name, primed)
    assert abandoned == 0
    assert primed["completed_seq"] == prime_seq and prime_seq == 1, primed
    assert 0.0 < prime_s < 5.0
    assert served_mismatch is None
    assert served["serviced"] == 1 and served["timeouts"] == 0


def test_the_prime_request_is_never_serviced_by_the_thread():
    """The prime request must end as skipped_abandoned and never be copied by the thread, however
    long after construction: a thread copy of it could land on its slot at any later time, for
    example during capture. Checked after a pause long enough for a resumed thread to have
    serviced it had its abandoned word been cleared before the thread consumed it."""
    generator = torch.Generator().manual_seed(103)
    sources = _sources(20, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 20)
    with _copier(sources, destinations, capacity=8, prime_slot=19) as copier:
        time.sleep(0.2)
        torch.cuda.synchronize()
        stats = copier.stats()
        mismatch = _reference_mismatch(sources, destinations, [0], [19])
        statuses = [row["status"] for row in copier.trace()]
    assert stats["serviced"] == 0, stats
    assert stats["skipped_abandoned"] == 1, stats
    assert stats["copy_errors"] == 0, stats
    assert statuses == ["skipped_abandoned"], statuses
    assert mismatch is None


_EXIT_WITH_LIVE_COPIER_CHILD = """
import sys
import torch
from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

core = int(sys.argv[1])
source = torch.randint(0, 256, (8, 64), dtype=torch.uint8).pin_memory()
destination = torch.zeros((8, 64), dtype=torch.uint8, device="cuda")
copier = ExpertDoorbellCopier(
    expert_row_segments([(source, destination)]), 4, cpu_core=core, stream=torch.cuda.Stream()
)
rows = torch.tensor([1, 2, 0, 0], dtype=torch.int64, device="cuda")
slots = torch.tensor([3, 4, 0, 0], dtype=torch.int32, device="cuda")
count = torch.tensor([2], dtype=torch.int32, device="cuda")
copier.post(rows, slots, count)
copier.wait()
torch.cuda.synchronize()
assert torch.equal(destination[3:5].cpu(), source[1:3])
print("RESOLVED", copier.stats()["serviced"], flush=True)
"""


def test_a_process_exiting_with_a_live_copier_exits_cleanly():
    """A server process that exits without stopping its copier must not destroy a running thread
    during static teardown (std::terminate, then a scheduler stuck in the driver holding the GPU)."""
    import subprocess
    import sys

    started = time.perf_counter()
    child = subprocess.run(
        [sys.executable, "-c", _EXIT_WITH_LIVE_COPIER_CHILD, str(SPIN_CORE)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.perf_counter() - started
    assert "RESOLVED 1" in child.stdout, (child.stdout, child.stderr[-2000:])
    assert "terminate called" not in child.stderr, child.stderr[-2000:]
    assert child.returncode == 0, (child.returncode, child.stderr[-2000:])
    assert elapsed < 60.0


_SCHEDULER_LIKE_LAUNCHER = """
import multiprocessing as mp
import os
import sys
import time

SERVING_STATE = []


def child(mode, core, err_path, ready):
    fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 2)
    from sglang.srt.utils.common import kill_itself_when_parent_died

    kill_itself_when_parent_died()
    import torch
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.kernels.ops.moe.expert_doorbell import ExpertDoorbellCopier

    source = torch.randint(0, 256, (8, 64), dtype=torch.uint8).pin_memory()
    destination = torch.zeros((8, 64), dtype=torch.uint8, device="cuda")
    options = dict(cpu_core=core, stream=torch.cuda.Stream())
    if mode == "sigterm_fatal_wait":
        options.update(timeout_polls=20_000, drain_polls=70_000, fatal_wait_s=30.0)
    copier = ExpertDoorbellCopier(expert_row_segments([(source, destination)]), 4, **options)
    # The scheduler holds its copier through the hot-cache manager, streamers and model runner
    # until interpreter teardown, not as a local freed when the event loop returns.
    SERVING_STATE.append(copier)
    rows = torch.tensor([1, 2, 0, 0], dtype=torch.int64, device="cuda")
    slots = torch.tensor([3, 4, 0, 0], dtype=torch.int32, device="cuda")
    count = torch.tensor([2], dtype=torch.int32, device="cuda")
    if mode == "sigterm_fatal_wait":
        copier.inject_fault(service_delay_s=120.0)
        copier.post(rows, slots, count)
        claimed_word = copier.page[4:8].view(torch.int32)
        while int(claimed_word.item()) != 1:
            time.sleep(0.0005)
        copier.resolve()
        torch.cuda.synchronize()
        ready.set()
        copier.fail_stop_check()
        return
    copier.post(rows, slots, count)
    copier.wait()
    torch.cuda.synchronize()
    assert torch.equal(destination[3:5].cpu(), source[1:3])
    ready.set()
    if mode == "graceful":
        return
    time.sleep(600)


if __name__ == "__main__":
    mode, core, err_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    mp.set_start_method("spawn", force=True)
    ready = mp.Event()
    process = mp.Process(target=child, args=(mode, core, err_path, ready))
    process.start()
    if not ready.wait(300):
        print("NOT_READY", flush=True)
        process.kill()
        sys.exit(2)
    if mode == "sigterm_fatal_wait":
        time.sleep(3.0)
    started = time.perf_counter()
    if mode != "graceful":
        process.terminate()
    process.join(30)
    elapsed = time.perf_counter() - started
    alive = process.is_alive()
    if alive:
        process.kill()
        process.join(30)
    print(f"RESULT exitcode={process.exitcode} alive_after={int(alive)} elapsed={elapsed:.2f}", flush=True)
"""


@pytest.mark.parametrize("mode", ["graceful", "sigterm_idle", "sigterm_fatal_wait"])
def test_a_scheduler_like_process_with_a_live_copier_exits_cleanly(mode, tmp_path):
    """Mirror the sglang scheduler process: a spawned child that set PDEATHSIG and installs no
    SIGTERM handler runs a live copier, then either returns from its target (the ShutdownReq
    path), or is sent SIGTERM while idle or while its fail-stop check waits for a committed copy a
    disabled drain gave up on. It must
    exit within 10 s with no "terminate called" (a joinable thread destroyed at teardown, which
    left production's scheduler stuck in the driver)."""
    import re
    import signal
    import subprocess
    import sys

    launcher = tmp_path / "launcher.py"
    launcher.write_text(_SCHEDULER_LIKE_LAUNCHER)
    child_stderr = tmp_path / "child.stderr"
    run = subprocess.run(
        [sys.executable, str(launcher), mode, str(SPIN_CORE), str(child_stderr)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    stderr = child_stderr.read_text() if child_stderr.exists() else ""
    result = re.search(r"RESULT exitcode=(-?\d+) alive_after=(\d) elapsed=([0-9.]+)", run.stdout)
    assert result, (run.stdout, run.stderr[-2000:], stderr[-2000:])
    exitcode, alive_after, elapsed = int(result.group(1)), int(result.group(2)), float(result.group(3))
    assert "terminate called" not in stderr, stderr[-2000:]
    assert alive_after == 0, (mode, elapsed, stderr[-2000:])
    assert elapsed < 10.0, (mode, elapsed)
    expected = 0 if mode == "graceful" else -signal.SIGTERM
    assert exitcode == expected, (mode, exitcode, stderr[-2000:])


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


def test_the_exit_hook_stops_every_copier_without_a_device_sync(monkeypatch):
    """The exit hook also runs after a scheduler exception, where the device may be wedged. With
    a synchronize that raises, it must still stop every live copier, and the thread's own drain
    must still land the requests already posted."""
    from sglang.kernels.ops.moe import expert_doorbell

    generator = torch.Generator().manual_seed(43)
    sources = _sources(30, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 30)
    rows, slots = _pick(generator, 30, 30, 12)
    copiers = [_copier(sources, destinations, capacity=12) for _ in range(2)]
    for copier in copiers:
        copier.post(*_plan(12, rows, slots))
    torch.cuda.synchronize()

    def wedged_synchronize(*args, **kwargs):
        raise RuntimeError("wedged device")

    monkeypatch.setattr(torch.cuda, "synchronize", wedged_synchronize)
    expert_doorbell._stop_live_copiers()
    monkeypatch.undo()
    assert [copier.stats()["running"] for copier in copiers] == [0, 0]
    _assert_matches_reference(sources, destinations, rows, slots)


def _scheduler_stub(expert_hot_cache_manager):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    return SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(expert_hot_cache_manager=expert_hot_cache_manager)
        ),
        hisparse_coordinator=None,
        tree_cache=MagicMock(),
        decode_offload_manager=None,
    )


def _release_scheduler_host_resources(stub):
    from unittest.mock import patch

    from sglang.srt.managers import scheduler as scheduler_module

    with (
        patch.object(scheduler_module, "destroy_global_experts_capturer"),
        patch.object(scheduler_module, "destroy_global_indexer_capturer"),
        patch.object(scheduler_module, "rank_consensus_checker"),
    ):
        scheduler_module.Scheduler.release_host_resources(stub)


def test_scheduler_shutdown_stops_the_doorbell_through_the_model_runner():
    """The graceful scheduler shutdown reaches the copier through
    tp_worker.model_runner.expert_hot_cache_manager and the manager's own stop_doorbell."""
    from types import SimpleNamespace

    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

    generator = torch.Generator().manual_seed(47)
    sources = _sources(30, EXPERT_LIKE_SHAPES, generator)
    destinations = _destinations(sources, 30)
    rows, slots = _pick(generator, 30, 30, 12)
    copier = _copier(sources, destinations, capacity=12)
    copier.post(*_plan(12, rows, slots))
    manager = SimpleNamespace(doorbell=copier)
    manager.stop_doorbell = lambda: ExpertHotCacheManager.stop_doorbell(manager)
    stub = _scheduler_stub(manager)

    _release_scheduler_host_resources(stub)

    assert copier.stats()["running"] == 0
    _assert_matches_reference(sources, destinations, rows, slots)
    stub.tree_cache.release_host_resources.assert_called_once()


def test_a_failing_doorbell_stop_still_releases_the_scheduler_host_resources():
    from unittest.mock import MagicMock

    manager = MagicMock()
    manager.stop_doorbell.side_effect = RuntimeError("CUDA error: an illegal memory access")
    stub = _scheduler_stub(manager)

    _release_scheduler_host_resources(stub)

    manager.stop_doorbell.assert_called_once()
    stub.tree_cache.release_host_resources.assert_called_once()


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
