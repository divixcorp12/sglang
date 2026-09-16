"""CPU-only contract tests for nonblocking telemetry snapshots."""

from __future__ import annotations

import threading
import unittest

from sglang.srt.layers.moe.async_telemetry import AsyncTelemetry


class _Event:
    def __init__(self) -> None:
        self.ready = False
        self.synchronized = False


class _Backend:
    """A deterministic stand-in for CUDA copies and events."""

    def __init__(self) -> None:
        self.events: list[_Event] = []
        self.enqueued: list[dict[str, list[int]]] = []

    def allocate(self, sources):
        return {name: [0] * len(value) for name, value in sources.items()}

    def event(self):
        event = _Event()
        self.events.append(event)
        return event

    def enqueue(self, buffers, sources, event):
        for name, value in sources.items():
            buffers[name][:] = value
        self.enqueued.append({name: list(value) for name, value in sources.items()})

    def ready(self, event):
        return event.ready

    def synchronize(self, event):
        event.synchronized = True
        event.ready = True


class TestAsyncTelemetry(unittest.TestCase):
    def test_completed_slot_hands_the_writer_an_immutable_accepted_snapshot(self):
        backend = _Backend()
        written = []
        telemetry = AsyncTelemetry(
            sources={"counter": [0]},
            backend=backend,
            writer=lambda buffers, metadata: written.append((list(buffers["counter"]), metadata)),
            slots=1,
            writer_jobs=1,
            thread_name="telemetry-test",
        )
        try:
            source = [7]
            self.assertTrue(telemetry.schedule({"counter": source}, {"forward": 3}))
            source[0] = 99
            self.assertEqual(written, [])
            self.assertFalse(backend.events[0].synchronized)

            backend.events[0].ready = True
            telemetry.poll()
            telemetry.close()

            self.assertEqual(written, [([7], {"forward": 3})])
            self.assertEqual(telemetry.stats()["accepted"], 1)
        finally:
            telemetry.close()

    def test_busy_slots_drop_instead_of_waiting_for_a_gpu_copy(self):
        backend = _Backend()
        telemetry = AsyncTelemetry(
            sources={"counter": [0]},
            backend=backend,
            writer=lambda _buffers, _metadata: None,
            slots=1,
            writer_jobs=1,
            thread_name="telemetry-test",
        )
        try:
            self.assertTrue(telemetry.schedule({"counter": [1]}, {}))
            self.assertFalse(telemetry.schedule({"counter": [2]}, {}))
            self.assertFalse(backend.events[0].synchronized)
            self.assertEqual(telemetry.stats()["dropped_pending"], 1)
        finally:
            telemetry.close()

    def test_full_writer_queue_drops_a_completed_sample_without_blocking(self):
        backend = _Backend()
        started = threading.Event()
        unblock = threading.Event()

        def writer(_buffers, _metadata):
            started.set()
            unblock.wait(timeout=5)

        telemetry = AsyncTelemetry(
            sources={"counter": [0]},
            backend=backend,
            writer=writer,
            slots=3,
            writer_jobs=1,
            thread_name="telemetry-test",
        )
        try:
            for value in (1, 2, 3):
                self.assertTrue(telemetry.schedule({"counter": [value]}, {}))
                backend.events[value - 1].ready = True
                telemetry.poll()
            self.assertTrue(started.wait(timeout=1))
            self.assertEqual(telemetry.stats()["dropped_writer"], 1)
        finally:
            unblock.set()
            telemetry.close()

    def test_close_waits_only_at_teardown_and_flushes_pending_slots(self):
        backend = _Backend()
        written = []
        telemetry = AsyncTelemetry(
            sources={"counter": [0]},
            backend=backend,
            writer=lambda buffers, _metadata: written.append(list(buffers["counter"])),
            slots=1,
            writer_jobs=1,
            thread_name="telemetry-test",
        )
        self.assertTrue(telemetry.schedule({"counter": [4]}, {}))
        telemetry.close()
        self.assertTrue(backend.events[0].synchronized)
        self.assertEqual(written, [[4]])

    def test_prepare_close_makes_every_event_visible_before_writer_joins(self):
        backend = _Backend()
        started = threading.Event()
        unblock = threading.Event()

        def writer(_buffers, _metadata):
            started.set()
            unblock.wait(timeout=5)

        telemetry = AsyncTelemetry(
            sources={"counter": [0]},
            backend=backend,
            writer=writer,
            slots=1,
            writer_jobs=1,
            thread_name="telemetry-test",
        )
        try:
            self.assertTrue(telemetry.schedule({"counter": [5]}, {}))
            telemetry.prepare_close()
            self.assertTrue(backend.events[0].synchronized)
            self.assertTrue(started.wait(timeout=1))
        finally:
            unblock.set()
            telemetry.finish_close()


if __name__ == "__main__":
    unittest.main()
