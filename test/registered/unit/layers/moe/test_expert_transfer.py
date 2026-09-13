import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.expert_transfer import (
    submit_expert_row_copies,
    AsyncExpertTransferExecutor,
    FixedRowTransferPlan,
)

from sglang.srt.layers.moe import expert_transfer


class _Event:
    def __init__(self):
        self.complete = False
        self.recorded_on = []
        self.waited_on = []

    def record(self, stream):
        self.recorded_on.append(stream)

    def query(self):
        return self.complete

    def wait(self, stream):
        self.waited_on.append(stream)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCudaExpertRowCopies(unittest.TestCase):
    def test_d2d_copy_uses_fixed_plan_kernel_not_generic_index_select(self):
        plan = FixedRowTransferPlan(max_rows=2, device="cuda")
        plan.set_rows([4, 1], [0, 1], [7, 8])
        source = (
            torch.arange(12, device="cuda", dtype=torch.float32)
            .reshape(6, 2)
            .to(torch.float8_e4m3fn)
        )
        destination = torch.zeros(2, 2, device="cuda", dtype=torch.float8_e4m3fn)
        paths = []

        with patch.object(
            expert_transfer,
            "_copy_rows_by_bytes",
            side_effect=AssertionError("D2D must not materialize indexed rows"),
        ):
            expert_transfer._copy_d2d_rows(
                source, destination, plan.source_rows, plan, paths
            )

        self.assertEqual(paths, ["d2d"])
        self.assertTrue(
            torch.equal(
                destination.view(torch.uint8),
                source[torch.tensor([4, 1], device="cuda")].view(torch.uint8),
            )
        )

    def test_d2d_copy_replays_in_a_cuda_graph(self):
        plan = FixedRowTransferPlan(max_rows=2, device="cuda")
        plan.set_rows([4, 1], [0, 1], [7, 8])
        source = (
            torch.arange(12, device="cuda", dtype=torch.float32)
            .reshape(6, 2)
            .to(torch.float8_e4m3fn)
        )
        destination = torch.zeros(2, 2, device="cuda", dtype=torch.float8_e4m3fn)
        paths = []

        expert_transfer._copy_d2d_rows(
            source, destination, plan.source_rows, plan, paths
        )
        torch.cuda.synchronize()
        destination.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            expert_transfer._copy_d2d_rows(
                source, destination, plan.source_rows, plan, paths
            )
        graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(
                destination.view(torch.uint8),
                source[torch.tensor([4, 1], device="cuda")].view(torch.uint8),
            )
        )


class _Stream:
    pass


class TestFixedRowTransferPlan(unittest.TestCase):
    def test_plan_keeps_fixed_buffers_while_replacing_rows(self):
        plan = FixedRowTransferPlan(max_rows=3, device="cpu")
        pointers = plan.data_ptrs()

        plan.set_rows(
            source_rows=torch.tensor([7, 2]),
            destination_slots=torch.tensor([1, 0]),
            generations=torch.tensor([11, 12]),
        )

        self.assertEqual(plan.count.item(), 2)
        self.assertEqual(plan.source_rows.tolist(), [7, 2, 0])
        self.assertEqual(plan.destination_slots.tolist(), [1, 0, 0])
        self.assertEqual(plan.generations.tolist(), [11, 12, 0])
        self.assertEqual(plan.data_ptrs(), pointers)

    def test_plan_rejects_mismatched_or_oversized_rows(self):
        plan = FixedRowTransferPlan(max_rows=1, device="cpu")

        with self.assertRaisesRegex(ValueError, "same number"):
            plan.set_rows([1], [2, 3], [4])
        with self.assertRaisesRegex(ValueError, "capacity"):
            plan.set_rows([1, 2], [3, 4], [5, 6])


class TestAsyncExpertTransferExecutor(unittest.TestCase):
    def test_ticket_waits_for_all_six_operations_and_reuses_completed_ring_slot(self):
        events = []
        stream = _Stream()
        executor = AsyncExpertTransferExecutor(
            device="cpu",
            max_inflight=1,
            stream=stream,
            event_factory=lambda: events.append(_Event()) or events[-1],
            stream_context=lambda _: nullcontext(),
        )
        plan = FixedRowTransferPlan(max_rows=1, device="cpu")
        plan.set_rows([3], [2], [9])
        calls = []

        ticket = executor.submit(plan, [lambda i=i: calls.append(i) for i in range(6)])

        self.assertEqual(calls, list(range(6)))
        self.assertFalse(executor.is_complete(ticket))
        with self.assertRaisesRegex(RuntimeError, "ring is full"):
            executor.submit(plan, [lambda: None] * 6)
        executor.wait(ticket, consumer_stream="consumer")
        self.assertEqual(events[0].waited_on, ["consumer"])

        events[0].complete = True
        self.assertTrue(executor.is_complete(ticket))
        replacement = executor.submit(plan, [lambda: None] * 6)

        self.assertNotEqual(replacement.sequence, ticket.sequence)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            executor.is_complete(ticket)

    def test_submit_requires_the_atomic_nvfp4_six_tensor_bundle(self):
        executor = AsyncExpertTransferExecutor(
            device="cpu",
            max_inflight=1,
            stream=_Stream(),
            event_factory=_Event,
            stream_context=lambda _: nullcontext(),
        )
        plan = FixedRowTransferPlan(max_rows=1, device="cpu")
        plan.set_rows([3], [2], [9])

        with self.assertRaisesRegex(ValueError, "six"):
            executor.submit(plan, [lambda: None] * 5)

    def test_callback_submission_preserves_legacy_stream_callback_contract(self):
        event = _Event()
        executor = AsyncExpertTransferExecutor(
            device="cpu",
            max_inflight=1,
            stream=_Stream(),
            event_factory=lambda: event,
            stream_context=lambda _: nullcontext(),
        )
        plan = FixedRowTransferPlan(max_rows=1, device="cpu")
        plan.set_rows([3], [2], [9])
        called = []

        ticket = executor.submit_callback(plan, lambda: called.append("copied"))

        self.assertEqual(called, ["copied"])
        executor.wait(ticket, consumer_stream="consumer")
        plan.set_rows([4], [1], [10])


class TestExpertRowCopySubmission(unittest.TestCase):
    def setUp(self):
        self.executor = AsyncExpertTransferExecutor(
            device="cpu",
            max_inflight=1,
            stream=_Stream(),
            event_factory=_Event,
            stream_context=lambda _: nullcontext(),
        )
        self.plan = FixedRowTransferPlan(max_rows=2, device="cpu")
        self.plan.set_rows([4, 1], [0, 1], [7, 8])
        self.pairs = [(torch.zeros(6, 1), torch.zeros(2, 1)) for _ in range(6)]

    def test_gpu_submission_passes_the_fixed_plan_to_every_tensor_copy(self):
        calls = []

        def copy(source, destination, source_rows, destination_slots, count):
            calls.append((source_rows, destination_slots, count))

        with (
            patch("sglang.srt.layers.moe.expert_transfer.copy_expert_rows_gpu", copy),
            patch(
                "sglang.srt.layers.moe.expert_transfer._can_copy_with_gpu",
                return_value=True,
            ),
        ):
            metrics = submit_expert_row_copies(
                self.executor,
                self.plan,
                self.pairs,
                backend="gpu",
                source_rows_cpu=[4, 1],
                destination_slots_cpu=[0, 1],
            )

        self.assertEqual(metrics.requested_backend, "gpu")
        self.assertEqual(metrics.actual_backend, "gpu")
        self.assertEqual(metrics.rows, 2)
        self.assertEqual(metrics.submissions, 1)
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(call[0] is self.plan.source_rows for call in calls))
        self.assertTrue(all(call[1] is self.plan.destination_slots for call in calls))
        self.assertTrue(all(call[2] is self.plan.count for call in calls))

    def test_gpu_submission_uses_fallback_for_nonpinned_sources(self):
        for source, destination in self.pairs:
            source[:, 0].copy_(torch.arange(source.shape[0]))
            destination.fill_(-1)

        metrics = submit_expert_row_copies(
            self.executor,
            self.plan,
            self.pairs,
            backend="gpu",
            source_rows_cpu=[4, 1],
            destination_slots_cpu=[0, 1],
        )
        self.assertEqual(metrics.actual_backend, "fallback")
        self.assertEqual(metrics.fallbacks, 1)
        for source, destination in self.pairs:
            self.assertEqual(destination[:, 0].tolist(), [4, 1])


if __name__ == "__main__":
    unittest.main()
