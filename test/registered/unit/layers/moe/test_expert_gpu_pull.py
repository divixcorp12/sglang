"""Focused tests for the captured side-stream one-row expert pull primitive.

Every test here drives synthetic plans through ``ExpertGpuPullPipeline`` only;
no predictor, scoring, or serving path is exercised. Host reads happen only
after the device work they check has been made to complete (or, in the
join-is-load-bearing tests, after only the origin-stream portion has been
made to complete), never as a blind race on wall-clock time.
"""

import random

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Expert GPU pull tests require a CUDA GPU.",
)


def _pinned_source(rows: int, row_bytes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 255, (rows, row_bytes), dtype=torch.uint8, generator=generator
    ).pin_memory()


def _sentinel_destination(slots: int, row_bytes: int, device: torch.device) -> torch.Tensor:
    return torch.full((slots, row_bytes), 255, dtype=torch.uint8, device=device)


def _make_pipeline_and_target(
    *,
    device: torch.device,
    tag: str,
    source: torch.Tensor,
    destination: torch.Tensor,
    slot: int,
    expert_id: int,
    count: int,
):
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.srt.layers.moe.expert_gpu_pull import ExpertGpuPullPipeline
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

    pipeline = ExpertGpuPullPipeline(device)
    plan = ExpertRowPlan(
        expert_ids=torch.tensor([expert_id], dtype=torch.int64, device=device),
        slots=torch.tensor([slot], dtype=torch.int32, device=device),
        count=torch.tensor([count], dtype=torch.int32, device=device),
    )
    segments = expert_row_segments([(source, destination)])
    target = pipeline.create_target(tag, segments, plan, slot=slot)
    return pipeline, target


def _warmup(
    device: torch.device,
    pipeline,
    body,
) -> None:
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        body()
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)


def test_fork_join_byte_exact_with_independent_origin_compute():
    device = torch.device("cuda")
    row_bytes = 4099
    tensor_count = 2
    source_rows = 6
    slot_count = 4
    seed_base = 100

    sources = [
        _pinned_source(source_rows, row_bytes, seed_base + i) for i in range(tensor_count)
    ]
    destinations = [
        _sentinel_destination(slot_count, row_bytes, device) for _ in range(tensor_count)
    ]

    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.srt.layers.moe.expert_gpu_pull import ExpertGpuPullPipeline
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

    pipeline = ExpertGpuPullPipeline(device)
    plan = ExpertRowPlan(
        expert_ids=torch.tensor([3], dtype=torch.int64, device=device),
        slots=torch.tensor([2], dtype=torch.int32, device=device),
        count=torch.tensor([1], dtype=torch.int32, device=device),
    )
    segments = expert_row_segments(list(zip(sources, destinations)))
    target = pipeline.create_target("multi-tensor", segments, plan, slot=2)

    origin_input = torch.randn(2048, 2048, device=device)
    origin_weight = torch.randn(2048, 2048, device=device)

    def independent_origin_compute() -> torch.Tensor:
        return origin_input @ origin_weight

    def body() -> None:
        pipeline.post_target(target)
        independent_origin_compute()
        pipeline.join_target(target)

    _warmup(device, pipeline, body)
    for destination in destinations:
        destination.fill_(255)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()

    graph.replay()
    torch.cuda.synchronize(device)

    for source, destination in zip(sources, destinations):
        torch.testing.assert_close(destination[2].cpu(), source[3])
    del graph


_DELAY_CYCLES = 400_000_000


def _build_delayed_target(tag: str):
    """A target whose side branch spins for a large, device-clock-bound cycle
    count before it copies, so its completion is far later than the origin
    branch's own work in device time regardless of host scheduling noise."""
    device = torch.device("cuda")
    row_bytes = 4096
    source = _pinned_source(4, row_bytes, seed=17)
    destination = _sentinel_destination(4, row_bytes, device)
    pipeline, target = _make_pipeline_and_target(
        device=device,
        tag=tag,
        source=source,
        destination=destination,
        slot=2,
        expert_id=1,
        count=1,
    )
    return device, source, destination, pipeline, target


def _patch_delayed_copy(monkeypatch) -> None:
    """Make ``post_target``'s own side-stream copy call incur a large,
    device-clock-bound delay before it runs, by wrapping the production copy
    function at the exact module attribute ``post_target`` calls. This drives
    the delay through the real ``ExpertGpuPullPipeline.post_target`` body
    instead of a hand-duplicated reimplementation of it, so these tests catch
    a regression in where ``post_target`` records ``done`` relative to the
    copy, not just a regression in a copy of that ordering.
    """
    import sglang.srt.layers.moe.expert_gpu_pull as expert_gpu_pull_module

    original_copy = expert_gpu_pull_module.copy_expert_row_segments_gpu

    def delayed_copy(segments, source_rows, destination_slots, count) -> None:
        torch.cuda._sleep(_DELAY_CYCLES)
        original_copy(segments, source_rows, destination_slots, count)

    monkeypatch.setattr(
        expert_gpu_pull_module, "copy_expert_row_segments_gpu", delayed_copy
    )


def _capture_and_replay_delayed_fork(
    monkeypatch, *, join: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture a delayed fork through the real ``post_target``, joining only
    when ``join`` is true, then replay.

    The result is read after synchronizing only the origin stream's own
    event, never the whole device, so an unjoined fork cannot accidentally
    pass by virtue of the test itself waiting for the side stream.
    """
    device, source, destination, pipeline, target = _build_delayed_target("delayed-capture")
    _patch_delayed_copy(monkeypatch)

    def body() -> None:
        pipeline.post_target(target)
        if join:
            pipeline.join_target(target)

    _warmup(device, pipeline, body)
    destination.fill_(255)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()

    origin = torch.cuda.current_stream(device)
    graph.replay()
    origin_done = torch.cuda.Event(enable_timing=False)
    origin_done.record(origin)
    origin_done.synchronize()

    result = destination.clone()
    torch.cuda.synchronize(device)
    del graph
    return source, result


def test_removing_the_join_makes_graph_capture_itself_refuse_to_end(monkeypatch):
    """The single most important assertion in this stage.

    A fork/join test that passes without the join has verified nothing. This
    proves the reverse, and the actual failure mode is stronger than a bad
    byte compare: CUDA's own stream-capture machinery refuses to end capture
    while the forked side-stream branch is still unjoined, raising
    ``cudaErrorStreamCaptureUnjoined`` out of ``capture_end()``. Removing the
    join does not merely risk a race later; it makes the graph uncapturable
    at all.
    """
    with pytest.raises(RuntimeError, match="unjoined"):
        _capture_and_replay_delayed_fork(monkeypatch, join=False)


def test_join_makes_a_delayed_fork_deterministically_correct(monkeypatch):
    """Regression guard on where ``post_target`` records ``done``.

    Runs the real ``post_target``/``join_target`` pair with the copy delayed
    at its actual call site, then reads the result after synchronizing only
    the origin stream's own event -- never a whole-device sync, which would
    wait out the delay regardless of whether the graph's join edge is
    correct and so could not catch a ``done`` recorded too early.
    """
    source, destination = _capture_and_replay_delayed_fork(monkeypatch, join=True)
    torch.testing.assert_close(destination[2].cpu(), source[1])


def test_forking_without_a_join_leaves_the_destination_unwritten_when_observed_early(
    monkeypatch,
):
    """A second, independent view of the same load-bearing join, in eager mode.

    Outside of capture, CUDA does not refuse an unjoined fork -- it is only
    ``capture_end`` that enforces this. So here, forking without a join and
    reading the destination immediately after only the origin stream's own
    work is confirmed complete (never after a device-wide synchronization)
    shows the actual data-level symptom the capture-time error prevents from
    ever reaching a captured graph: the destination still holds its sentinel
    value, not the pulled row. This also drives the real ``post_target``, not
    a duplicate of its body.
    """
    device, source, destination, pipeline, target = _build_delayed_target("delayed-eager")
    _patch_delayed_copy(monkeypatch)
    origin = torch.cuda.current_stream(device)

    pipeline.post_target(target)

    origin_done = torch.cuda.Event(enable_timing=False)
    origin_done.record(origin)
    origin_done.synchronize()
    observed = destination[2].clone()
    torch.cuda.synchronize(device)

    assert torch.equal(observed.cpu(), torch.full((4096,), 255, dtype=torch.uint8))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(observed.cpu(), source[1])


def test_count_zero_and_one_across_many_replays_and_changing_expert_ids():
    device = torch.device("cuda")
    row_bytes = 512
    source_rows = 128
    slot_count = 4
    source = _pinned_source(source_rows, row_bytes, seed=7)
    destination = _sentinel_destination(slot_count, row_bytes, device)

    pipeline, target = _make_pipeline_and_target(
        device=device,
        tag="replay-sweep",
        source=source,
        destination=destination,
        slot=3,
        expert_id=0,
        count=0,
    )

    def body() -> None:
        pipeline.post_target(target)
        pipeline.join_target(target)

    _warmup(device, pipeline, body)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()

    rng = random.Random(11)
    sentinel_row = torch.full((row_bytes,), 255, dtype=torch.uint8)
    for i in range(120):
        expert_id = rng.randrange(source_rows)
        count = i % 2
        target.plan.expert_ids.fill_(expert_id)
        target.plan.count.fill_(count)
        destination.fill_(255)
        graph.replay()
        torch.cuda.synchronize(device)
        if count == 1:
            torch.testing.assert_close(destination[3].cpu(), source[expert_id])
        else:
            assert torch.equal(destination[3].cpu(), sentinel_row)
    del graph
    torch.cuda.synchronize(device)


def test_two_separately_allocated_graph_states_do_not_interfere():
    device = torch.device("cuda")
    row_bytes = 256
    row_count = 8

    def build(seed: int, tag: str):
        source = _pinned_source(row_count, row_bytes, seed)
        destination = _sentinel_destination(4, row_bytes, device)
        pipeline, target = _make_pipeline_and_target(
            device=device,
            tag=tag,
            source=source,
            destination=destination,
            slot=1,
            expert_id=0,
            count=1,
        )

        def body() -> None:
            pipeline.post_target(target)
            pipeline.join_target(target)

        _warmup(device, pipeline, body)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return pipeline, target, graph, source, destination

    state_a = build(31, "state-a")
    state_b = build(32, "state-b")
    pipeline_a, target_a, graph_a, source_a, destination_a = state_a
    pipeline_b, target_b, graph_b, source_b, destination_b = state_b

    assert pipeline_a.side_stream != pipeline_b.side_stream
    assert target_a.ready != target_b.ready
    assert target_a.done != target_b.done

    for expert_a, expert_b in zip((1, 2, 3), (5, 6, 7)):
        target_a.plan.expert_ids.fill_(expert_a)
        target_b.plan.expert_ids.fill_(expert_b)
        destination_a.fill_(255)
        destination_b.fill_(255)
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(destination_a[1].cpu(), source_a[expert_a])
        torch.testing.assert_close(destination_b[1].cpu(), source_b[expert_b])

    del graph_a
    del graph_b
    torch.cuda.synchronize(device)


def test_target_tag_lookup_and_duplicate_registration():
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.srt.layers.moe.expert_gpu_pull import ExpertGpuPullPipeline
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

    device = torch.device("cuda")
    source = _pinned_source(4, 64, seed=41)
    destination = _sentinel_destination(2, 64, device)
    pipeline = ExpertGpuPullPipeline(device)
    plan = ExpertRowPlan(
        expert_ids=torch.zeros(1, dtype=torch.int64, device=device),
        slots=torch.zeros(1, dtype=torch.int32, device=device),
        count=torch.zeros(1, dtype=torch.int32, device=device),
    )
    segments = expert_row_segments([(source, destination)])
    target = pipeline.create_target("only-tag", segments, plan, slot=0)

    assert pipeline.target("only-tag") is target
    with pytest.raises(ValueError, match="already registered"):
        pipeline.create_target("only-tag", segments, plan, slot=0)


def test_join_all_joins_every_registered_target_before_capture_ends():
    """``join_all`` must itself close out every posted target's fork.

    Two targets share one pipeline; the captured body posts both and calls
    only ``join_all`` (never a per-target ``join_target``), mirroring the
    target-disabled/model-tail case in section 7.4 where a generic tail join
    is responsible for every outstanding fork. Capture must succeed -- an
    unjoined fork would raise at ``capture_end`` per the test above -- and
    both destinations must be byte-exact after replay.
    """
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.srt.layers.moe.expert_gpu_pull import ExpertGpuPullPipeline
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

    device = torch.device("cuda")
    row_bytes = 128
    row_count = 4
    pipeline = ExpertGpuPullPipeline(device)

    def make_target(tag: str, expert_id: int):
        source = _pinned_source(row_count, row_bytes, seed=hash(tag) % 1000)
        destination = _sentinel_destination(2, row_bytes, device)
        plan = ExpertRowPlan(
            expert_ids=torch.tensor([expert_id], dtype=torch.int64, device=device),
            slots=torch.tensor([1], dtype=torch.int32, device=device),
            count=torch.tensor([1], dtype=torch.int32, device=device),
        )
        segments = expert_row_segments([(source, destination)])
        target = pipeline.create_target(tag, segments, plan, slot=1)
        return source, destination, target

    source_a, destination_a, target_a = make_target("join-all-a", 0)
    source_b, destination_b, target_b = make_target("join-all-b", 2)

    def body() -> None:
        pipeline.post_target(target_a)
        pipeline.post_target(target_b)
        pipeline.join_all()

    _warmup(device, pipeline, body)
    destination_a.fill_(255)
    destination_b.fill_(255)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()

    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(destination_a[1].cpu(), source_a[0])
    torch.testing.assert_close(destination_b[1].cpu(), source_b[2])
    del graph


def test_create_target_rejects_a_plan_with_capacity_other_than_one():
    from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
    from sglang.srt.layers.moe.expert_gpu_pull import ExpertGpuPullPipeline
    from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

    device = torch.device("cuda")
    source = _pinned_source(4, 64, seed=42)
    destination = _sentinel_destination(2, 64, device)
    pipeline = ExpertGpuPullPipeline(device)
    plan = ExpertRowPlan(
        expert_ids=torch.zeros(2, dtype=torch.int64, device=device),
        slots=torch.zeros(2, dtype=torch.int32, device=device),
        count=torch.zeros(1, dtype=torch.int32, device=device),
    )
    segments = expert_row_segments([(source, destination)])
    with pytest.raises(ValueError, match="capacity 1"):
        pipeline.create_target("bad-capacity", segments, plan, slot=0)


def test_physical_overlap_of_side_pull_and_origin_compute():
    """Measure real device kernel intervals, not stream naming, for overlap.

    Compares an overlapped schedule (origin compute begins immediately after
    forking the pull) against a serialized schedule (origin joins before
    starting the same compute) using CUDA event timestamps on both branches.
    If the measured overlap is not positive on this hardware, that is a real
    finding and is reported as such rather than weakened into a pass.
    """
    device = torch.device("cuda")
    row_bytes = 1 << 20
    row_count = 32
    slot_count = 4
    source = _pinned_source(row_count, row_bytes, seed=53)
    destination = _sentinel_destination(slot_count, row_bytes, device)

    pipeline, target = _make_pipeline_and_target(
        device=device,
        tag="overlap",
        source=source,
        destination=destination,
        slot=2,
        expert_id=5,
        count=1,
    )

    compute_input = torch.randn(4096, 4096, device=device)
    compute_weight = torch.randn(4096, 4096, device=device)

    def origin_compute() -> torch.Tensor:
        return compute_input @ compute_weight

    def body() -> None:
        pipeline.post_target(target)
        origin_compute()
        pipeline.join_target(target)

    _warmup(device, pipeline, body)
    torch.cuda.synchronize(device)

    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

    def timed_schedule(*, overlapped: bool) -> tuple[float, float, float, float]:
        origin = torch.cuda.current_stream(device)
        t0 = torch.cuda.Event(enable_timing=True)
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)

        t0.record(origin)
        target.ready.record(origin)
        with torch.cuda.stream(pipeline.side_stream):
            pipeline.side_stream.wait_event(target.ready)
            copy_start.record(pipeline.side_stream)
            copy_expert_row_segments_gpu(
                target.segments, target.plan.expert_ids, target.plan.slots, target.plan.count
            )
            copy_end.record(pipeline.side_stream)
            target.done.record(pipeline.side_stream)

        if not overlapped:
            origin.wait_event(target.done)

        compute_start.record(origin)
        origin_compute()
        compute_end.record(origin)
        origin.wait_event(target.done)

        torch.cuda.synchronize(device)
        return (
            t0.elapsed_time(copy_start),
            t0.elapsed_time(copy_end),
            t0.elapsed_time(compute_start),
            t0.elapsed_time(compute_end),
        )

    overlapped_copy_start, overlapped_copy_end, overlapped_compute_start, overlapped_compute_end = (
        timed_schedule(overlapped=True)
    )
    serialized_copy_start, serialized_copy_end, serialized_compute_start, serialized_compute_end = (
        timed_schedule(overlapped=False)
    )

    overlap_ms = min(overlapped_copy_end, overlapped_compute_end) - max(
        overlapped_copy_start, overlapped_compute_start
    )
    serialized_gap_ms = serialized_compute_start - serialized_copy_end

    print(
        "physical_overlap_ms="
        f"{overlap_ms:.4f} serialized_gap_ms={serialized_gap_ms:.4f} "
        f"overlapped_copy=[{overlapped_copy_start:.4f},{overlapped_copy_end:.4f}] "
        f"overlapped_compute=[{overlapped_compute_start:.4f},{overlapped_compute_end:.4f}] "
        f"serialized_copy=[{serialized_copy_start:.4f},{serialized_copy_end:.4f}] "
        f"serialized_compute=[{serialized_compute_start:.4f},{serialized_compute_end:.4f}]"
    )

    assert serialized_gap_ms >= -0.05, (
        "the serialized schedule's own compute started before its copy ended; "
        "the join is not enforcing the ordering it is supposed to enforce."
    )
    assert overlap_ms > 0.0, (
        "no positive kernel-interval overlap was measured between the side pull "
        "and origin compute on this hardware; overlap here is nominal, not physical."
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
