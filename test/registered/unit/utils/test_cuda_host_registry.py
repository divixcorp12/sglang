"""Unit tests for srt/utils/cuda_host_registry.py: bookkeeping only, no CUDA calls."""

import threading
import unittest
import weakref

import torch

from sglang.srt.utils import cuda_host_registry as registry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

_DEADLINE_S = 5.0


class _Owner:
    """Stands in for an object whose finalizer unregisters its host memory."""


def _run_holding_the_lock(body) -> threading.Thread:
    """Runs ``body`` on a daemon thread that holds the registry lock, as a lookup
    does when a garbage collection runs a finalizer inside it."""

    def target():
        with registry._LOCK:
            body()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(_DEADLINE_S)
    return thread


def _lock_is_free() -> bool:
    """False if an earlier test's thread deadlocked and still holds the lock."""
    if not registry._LOCK.acquire(timeout=1.0):
        return False
    registry._LOCK.release()
    return True


class TestCudaHostRegistry(CustomTestCase):
    def setUp(self):
        if not _lock_is_free():
            self.skipTest("the registry lock is held by a deadlocked thread")
        self.tensor = torch.empty(4096, dtype=torch.uint8)
        self.base = self.tensor.data_ptr()
        registry.record_cuda_host_registration(self.base, self.tensor.numel())

    def tearDown(self):
        if _lock_is_free():
            registry.forget_cuda_host_registration(self.base)
            self.assertFalse(registry.is_cuda_host_registered(self.tensor))

    def test_a_finalizer_that_forgets_inside_a_locked_lookup_does_not_deadlock(self):
        owner = _Owner()
        weakref.finalize(owner, registry.forget_cuda_host_registration, self.base)

        def drop_owner():
            nonlocal owner
            owner = None  # the finalizer runs here, with the lock held

        thread = _run_holding_the_lock(drop_owner)
        self.assertFalse(thread.is_alive(), "the finalizer deadlocked on the lock")
        self.assertFalse(registry.is_cuda_host_registered(self.tensor))

    def test_a_forget_deferred_by_a_held_lock_is_seen_by_the_next_lookup(self):
        thread = _run_holding_the_lock(
            lambda: registry.forget_cuda_host_registration(self.base)
        )
        self.assertFalse(thread.is_alive(), "forget deadlocked on the lock")
        self.assertFalse(registry.is_cuda_host_registered(self.tensor))
        self.assertIsNone(registry.cuda_host_registration_end(self.base))

    def test_a_registration_after_a_deferred_forget_of_the_same_base_is_kept(self):
        thread = _run_holding_the_lock(
            lambda: registry.forget_cuda_host_registration(self.base)
        )
        self.assertFalse(thread.is_alive(), "forget deadlocked on the lock")
        registry.record_cuda_host_registration(self.base, self.tensor.numel())
        self.assertTrue(registry.is_cuda_host_registered(self.tensor))

    def test_a_lookup_inside_the_lock_sees_the_state_from_before_a_deferred_forget(
        self,
    ):
        seen = []

        def forget_then_look():
            registry.forget_cuda_host_registration(self.base)
            seen.append(self.base in registry._SIZES)

        thread = _run_holding_the_lock(forget_then_look)
        self.assertFalse(thread.is_alive(), "forget deadlocked on the lock")
        self.assertEqual(seen, [True])  # deferred: the lookup in progress is not torn
        self.assertFalse(registry.is_cuda_host_registered(self.tensor))


if __name__ == "__main__":
    unittest.main()
