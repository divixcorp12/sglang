"""Shutdown of the option C service: stop admission, establish that no GPU reader runs, then free; else quarantine
(CPU, fake device barrier); LEASE_PROTOCOL.md 14.3 and item 9(c) of 18.2.

Written after the code, unlike the service-side lease tests: the sequence is a list of named steps and the tests
read it back. Each test names the mutation it must fail under. A fake CUDA barrier (``_synchronize``) stands in for
``torch.cuda.synchronize``; it is a stand-in for the barrier, so these tests show the WIRING, not that a real device
barrier orders the GPU's work.
"""

import faulthandler
import gc
import sys
import threading
import time
import weakref

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe import expert_host_tier
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 6, 3


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def world(tmp_path, monkeypatch):
    write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS)
    layout = build_exl3_expert_layout(str(tmp_path))
    module.Exl3RamMissService._instance = None
    streamers, caches = {}, {}
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
        for layer_id in range(LAYERS):
            layer = torch.nn.Module()
            layer.layer_id = layer_id
            fmt = Exl3ExpertFormat(layout, layer_id, direct=False, source_root=str(tmp_path))
            streamer = ExpertStreamer(layer, fmt.names, layer_id=layer_id, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            caches[layer_id] = ExpertPinnedHostCache(streamer, CAPACITY, device="cpu", **fmt.pinned_tier_options(layer))
            streamers[layer_id] = streamer
    service = module.Exl3RamMissService.get()
    service.ensure_started()
    order = []
    # A CUDA device is pretended, so the barrier path runs; the barrier itself is a fake the test controls.
    monkeypatch.setattr(module.Exl3RamMissService, "_cuda_active", lambda self: True)
    close_admission, stop = service.host.close_admission, service.host.stop
    monkeypatch.setattr(service.host, "close_admission", lambda: (order.append("close_admission"), close_admission()))
    monkeypatch.setattr(service.host, "stop", lambda: (order.append("stop"), stop()))
    for layer_id, cache in caches.items():
        close, quarantine = cache.close, cache.quarantine
        monkeypatch.setattr(cache, "close", lambda close=close, i=layer_id: (order.append(f"free{i}"), close()))
        monkeypatch.setattr(cache, "quarantine", lambda q=quarantine, i=layer_id: (order.append(f"quarantine{i}"), q()))
    yield service, caches, order
    module.Exl3RamMissService._instance = None


def _barrier(service, monkeypatch, order, behaviour):
    def fake(device=None):
        order.append("synchronize")
        behaviour()

    monkeypatch.setattr(service, "_synchronize", fake)
    monkeypatch.setattr(service, "_barrier_devices", lambda: [torch.device("cuda", 0)])  # no CUDA here; see the F2 tests


def test_shutdown_closes_admission_then_establishes_completion_then_stops_then_frees(world, monkeypatch):
    """Mutation: the barrier is skipped, run before admission closes, or run after the thread stops. The header's
    shutdown word is read from inside the barrier, so 'admission closed first' is observed, not inferred."""
    service, caches, order = world
    seen = {}
    _barrier(service, monkeypatch, order, lambda: seen.setdefault("shutdown", service.host.lease_header()["shutdown"]))
    service.shutdown()
    assert order == ["close_admission", "synchronize", "stop", "free0", "free1"]
    assert seen["shutdown"] == 1, "the header's shutdown word was set before the device barrier ran"
    assert not service._quarantined and not service.host.threaded, "and the service thread is gone"


def test_a_cuda_error_at_the_barrier_quarantines_everything_and_frees_nothing(world, monkeypatch):
    """Mutation: an error at the barrier is treated as success (the slabs are then freed under a possibly running
    kernel), the finalizer that unregisters at exit is left attached, or the request page, slot map and lease block
    are not kept."""
    service, caches, order = world

    def fail():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    _barrier(service, monkeypatch, order, fail)
    page, block = service.host.page, service.host.lease_block
    service.shutdown()
    assert order == ["close_admission", "synchronize", "stop", "quarantine0", "quarantine1"]
    assert service._quarantined and not any(step.startswith("free") for step in order)
    assert all(not cache._release_slabs.alive for cache in caches.values()), "the exit-time unregister is detached"
    kept = expert_host_tier._QUARANTINED
    assert any(t is page for t in kept) and any(t is block for t in kept)


def test_a_barrier_that_does_not_return_in_time_quarantines_and_does_not_hang_shutdown(world, monkeypatch):
    """Mutation: a timeout is treated as success, or shutdown joins the barrier without a deadline. The barrier really
    was started (it is in the sequence) and really blocked (it waits on an event the test releases at the end)."""
    service, caches, order = world
    release = threading.Event()
    monkeypatch.setattr(service, "_completion_deadline_s", lambda: 0.3)
    _barrier(service, monkeypatch, order, lambda: release.wait(30))
    start = time.perf_counter()
    service.shutdown()
    elapsed = time.perf_counter() - start
    release.set()
    assert order == ["close_admission", "synchronize", "stop", "quarantine0", "quarantine1"]
    assert 0.25 < elapsed < 5.0, "it waited for the deadline, and not for the barrier"
    assert service._quarantined


def test_shutdown_at_exit_quarantines_without_attempting_a_barrier(world, monkeypatch):
    """Mutation: the exit path calls the device barrier (in an exit handler CUDA may already be tearing down) or
    frees. The barrier is armed to raise if it is called."""
    service, caches, order = world

    def must_not_run():
        raise AssertionError("the exit path must not attempt a device barrier")

    _barrier(service, monkeypatch, order, must_not_run)
    service.shutdown(at_exit=True)
    assert order == ["close_admission", "stop", "quarantine0", "quarantine1"]
    assert service._quarantined


def test_quarantined_slabs_survive_the_tier_and_the_service_going_away(world, monkeypatch):
    """The point of the whole path: after a failed barrier the slabs outlive every owner. A weak reference to one is
    held; the caches, the service and the module list are all dropped."""
    service, caches, order = world
    _barrier(service, monkeypatch, order, lambda: (_ for _ in ()).throw(RuntimeError("CUDA error")))
    slab = weakref.ref(next(iter(caches[0].tensors.values())))
    service.shutdown()
    caches.clear()
    monkeypatch.undo()
    module.Exl3RamMissService._instance = None
    del service
    gc.collect()
    expert_host_tier._QUARANTINED.clear()
    gc.collect()
    assert slab() is not None


# ---- stopping the service thread, and the barrier's device (independent review F1, F2, F5) ----


def _stop_that(service, monkeypatch, order, error):
    real_stop = service.host.stop

    def stop():
        order.append("stop")
        raise error

    monkeypatch.setattr(service.host, "stop", stop)
    return real_stop  # the test calls it at the end, or the native handle would leak


def test_a_failed_stop_of_the_service_thread_after_a_clean_barrier_quarantines_and_frees_nothing(world, monkeypatch):
    """F1. Mutation: a stop() failure does not change the outcome, so the slabs are freed under a service thread that
    may still write into them (today's exit path never frees, so this would be worse than not wiring)."""
    service, caches, order = world
    _barrier(service, monkeypatch, order, lambda: None)
    real_stop = _stop_that(service, monkeypatch, order, RuntimeError("the join failed"))
    service.shutdown()
    steps = list(order)
    real_stop()  # cleanup: the fixture's wrapper records a step of its own
    assert steps == ["close_admission", "synchronize", "stop", "quarantine0", "quarantine1"]
    assert service._quarantined and service._completed


def test_an_interrupt_during_the_stop_quarantines_first_and_then_goes_on(world, monkeypatch):
    """F1, BaseException. Mutation: the interrupt is swallowed, or it propagates before the quarantine."""
    service, caches, order = world
    _barrier(service, monkeypatch, order, lambda: None)
    real_stop = _stop_that(service, monkeypatch, order, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        service.shutdown()
    steps = list(order)
    real_stop()
    assert steps == ["close_admission", "synchronize", "stop", "quarantine0", "quarantine1"]


def _record_synchronize(monkeypatch, on_thread=None):
    seen = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: seen.append(device))
    return seen


def test_the_barrier_synchronizes_the_tiers_cuda_device_not_whatever_a_new_thread_defaults_to(world, monkeypatch):
    """F2. Mutation: a device-less torch.cuda.synchronize() on the helper thread (device 0 there, whatever the rank
    serves). Here the tiers claim cuda:3; the real _synchronize runs, on the helper thread, against a recorder."""
    service, caches, order = world
    for cache in caches.values():
        monkeypatch.setattr(cache, "device", torch.device("cuda", 3))
    seen = _record_synchronize(monkeypatch)
    service.shutdown()
    assert seen == [torch.device("cuda", 3)]
    assert order[-2:] == ["free0", "free1"]


def test_without_a_cuda_tier_the_barrier_uses_the_device_current_on_the_calling_thread(world, monkeypatch):
    """F2, fallback. The recorder answers 2 on the main thread and 0 anywhere else, like a thread that never called
    set_device. Mutation: the device is looked up inside the helper thread."""
    service, caches, order = world
    seen = _record_synchronize(monkeypatch)
    main = threading.main_thread()
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: 2 if threading.current_thread() is main else 0
    )
    service.shutdown()
    assert seen == [torch.device("cuda", 2)]


def test_a_shutdown_that_did_not_complete_is_finished_as_a_quarantine_by_the_exit_hook(world, monkeypatch):
    """F5. Mutation: the exit hook returns once a shutdown STARTED (the interpreter's own finalizers would then
    unregister the slabs with no barrier), or completion is recorded on entry."""
    service, caches, order = world
    _barrier(service, monkeypatch, order, lambda: None)
    close = caches[0].close

    def broken_close():
        order.append("free0-raised")
        raise RuntimeError("close failed midway")

    monkeypatch.setattr(caches[0], "close", broken_close)
    with pytest.raises(RuntimeError, match="midway"):
        service.shutdown()
    assert not service._completed
    module._quarantine_service_at_exit(weakref.ref(service))
    assert service._completed and service._quarantined
    assert "quarantine0" in order and "quarantine1" in order, "the exit hook quarantined every tier"
    assert all(not cache._release_slabs.alive for cache in caches.values()), "no finalizer is left to unregister"
    monkeypatch.undo()
    close()


# ---- second review pass: interrupts alike (S1), the device branch production takes first (S2) ----


def test_an_interrupt_in_the_barrier_quarantines_first_and_then_goes_on_like_one_during_the_stop(world, monkeypatch):
    """S1. The interrupt is raised in the CALLING thread (the barrier's helper thread cannot deliver one to shutdown);
    `_establish_gpu_completion` is replaced by a function that raises it. Mutation: it is swallowed after quarantining."""
    service, caches, order = world

    def interrupted():
        order.append("synchronize")
        raise KeyboardInterrupt()

    monkeypatch.setattr(service, "_establish_gpu_completion", interrupted)
    with pytest.raises(KeyboardInterrupt):
        service.shutdown()
    assert order == ["close_admission", "synchronize", "stop", "quarantine0", "quarantine1"]
    assert service._quarantined and service._completed


def test_the_barrier_syncs_the_service_device_and_the_tiers_device_each_once_with_explicit_indices(world, monkeypatch):
    """S2. The branch production takes first is `device_side.state.device`; the fixture has none, so a stub on cuda:1 is
    attached while the tiers claim cuda:3 (and one an index-less `cuda`, resolved on the calling thread to the current
    device, 2 there). Mutations: the service's device is ignored; only the first device is synced; an index-less device
    is passed on as it is (a helper thread would resolve it to device 0)."""
    from types import SimpleNamespace

    service, caches, order = world
    monkeypatch.setattr(service, "device_side", SimpleNamespace(state=SimpleNamespace(device=torch.device("cuda", 1))))
    monkeypatch.setattr(caches[0], "device", torch.device("cuda", 3))
    monkeypatch.setattr(caches[1], "device", torch.device("cuda"))
    seen = _record_synchronize(monkeypatch)
    main = threading.main_thread()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2 if threading.current_thread() is main else 0)
    service.shutdown()
    assert seen == [torch.device("cuda", 1), torch.device("cuda", 3), torch.device("cuda", 2)]
    assert not service._quarantined


def test_a_device_shared_by_the_service_and_the_tiers_is_synced_once(world, monkeypatch):
    """S2. Mutation: distinct devices are not de-duplicated (one barrier per tier)."""
    from types import SimpleNamespace

    service, caches, order = world
    monkeypatch.setattr(service, "device_side", SimpleNamespace(state=SimpleNamespace(device=torch.device("cuda", 1))))
    for cache in caches.values():
        monkeypatch.setattr(cache, "device", torch.device("cuda", 1))
    seen = _record_synchronize(monkeypatch)
    service.shutdown()
    assert seen == [torch.device("cuda", 1)]


# ---- the scheduler's graceful shutdown reaches the service (LEASE_PROTOCOL.md 20.2i) ----
# These run the REAL, unbound Scheduler.release_host_resources on a stub, as test_expert_doorbell_copier.py does for
# the doorbell. The barrier is still the fake one above: they show the wiring and its order, not that a real device
# barrier orders GPU work.


def _release_scheduler_host_resources(order, *, manager=True):
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from sglang.srt.managers import scheduler as scheduler_module

    def recorder(name):
        mock = MagicMock()
        getattr(mock, "stop_doorbell" if name == "doorbell" else "release_host_resources" if name != "hisparse" else "destroy").side_effect = (
            lambda: order.append(name)
        )
        return mock

    stub = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(expert_hot_cache_manager=recorder("doorbell") if manager else None)),
        hisparse_coordinator=recorder("hisparse"),
        tree_cache=recorder("tree_cache"),
        decode_offload_manager=recorder("decode_offload"),
    )
    consensus = MagicMock()
    consensus.shutdown.side_effect = lambda: order.append("rank_consensus")
    with (
        patch.object(scheduler_module, "destroy_global_experts_capturer", side_effect=lambda: order.append("experts_capturer")),
        patch.object(scheduler_module, "destroy_global_indexer_capturer", side_effect=lambda: order.append("indexer_capturer")),
        patch.object(scheduler_module, "rank_consensus_checker", consensus),
    ):
        scheduler_module.Scheduler.release_host_resources(stub)


CHEAP = ["doorbell", "hisparse", "tree_cache", "decode_offload", "experts_capturer", "indexer_capturer", "rank_consensus"]


def test_the_graceful_scheduler_shutdown_shuts_the_service_down_after_every_cheaper_release(monkeypatch):
    """Mutations: the block is omitted; it runs before the doorbell stop, before hisparse, before the tree cache or
    before the last cheap release. The whole order is asserted, so the failing line is the `==` below."""
    order = []
    monkeypatch.setattr(module, "shutdown_exl3_ram_miss_service", lambda: order.append("ram_miss"))
    _release_scheduler_host_resources(order)
    assert order == CHEAP + ["ram_miss"]


def test_a_failing_service_shutdown_does_not_escape_the_scheduler_release(monkeypatch):
    """Mutation: the try/except is removed. Nothing but the caller's abort_distributed_environment() follows this
    block, so the requirement is that the method RETURNS: it is asserted directly, not through a call that raises."""
    order = []

    def fail():
        order.append("ram_miss")
        raise RuntimeError("the service could not shut down")

    monkeypatch.setattr(module, "shutdown_exl3_ram_miss_service", fail)
    escaped = None
    try:
        _release_scheduler_host_resources(order)
    except Exception as error:  # noqa: BLE001
        escaped = error
    assert escaped is None, f"the release raised {escaped!r}"
    assert order == CHEAP + ["ram_miss"]


def test_the_service_is_shut_down_even_when_the_scheduler_has_no_expert_hot_cache_manager(world, monkeypatch):
    """Mutation: the block is nested under `if expert_hot_cache_manager is not None` (the natural place to paste it)."""
    service, caches, order = world
    _barrier(service, monkeypatch, order, lambda: None)
    _release_scheduler_host_resources(order, manager=False)
    assert "close_admission" in order and order[-2:] == ["free0", "free1"]
    assert "doorbell" not in order


def test_scheduler_shutdown_drives_a_live_service_through_the_barrier_last(world, monkeypatch):
    """The real service, the real function, the real scheduler method. Mutations: the service is not shut down at all
    (the exit hook would quarantine instead of an orderly free); the delegating function passes at_exit=True."""
    service, caches, order = world
    _barrier(service, monkeypatch, order, lambda: None)
    _release_scheduler_host_resources(order)
    assert order == CHEAP + ["close_admission", "synchronize", "stop", "free0", "free1"]
    assert not service._quarantined


def test_scheduler_shutdown_with_a_failed_barrier_quarantines(world, monkeypatch):
    service, caches, order = world

    def fail():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    _barrier(service, monkeypatch, order, fail)
    _release_scheduler_host_resources(order)
    assert order == CHEAP + ["close_admission", "synchronize", "stop", "quarantine0", "quarantine1"]
    assert service._quarantined


def test_the_exit_hook_after_an_orderly_scheduler_shutdown_does_nothing_more(world, monkeypatch):
    """The two paths cannot both free. Mutations: the exit hook ignores completion; completion is never recorded."""
    service, caches, order = world
    _barrier(service, monkeypatch, order, lambda: None)
    _release_scheduler_host_resources(order)
    before = list(order)
    module._quarantine_service_at_exit(weakref.ref(service))
    assert order == before and not service._quarantined
    assert all(not cache._release_slabs.alive for cache in caches.values())


def test_a_second_orderly_shutdown_does_nothing(world, monkeypatch):
    """Mutation: shutdown() is not idempotent."""
    service, caches, order = world
    _barrier(service, monkeypatch, order, lambda: None)
    service.shutdown()
    before = list(order)
    service.shutdown()
    assert order == before


def test_shutting_down_without_a_service_does_not_construct_one():
    """Mutation: the function calls Exl3RamMissService.get() (a run without EXL3 would create a service at shutdown)."""
    module.Exl3RamMissService._instance = None
    module.shutdown_exl3_ram_miss_service()
    assert module.Exl3RamMissService._instance is None


def test_the_scheduler_release_imports_nothing_when_the_module_was_never_loaded(monkeypatch):
    """F4. Mutation: the block imports the module unconditionally (every non-EXL3 run would load the MoE stack)."""
    key = "sglang.srt.layers.moe.exl3_ram_miss"
    monkeypatch.delitem(sys.modules, key)
    order = []
    _release_scheduler_host_resources(order)
    assert key not in sys.modules, "the scheduler imported the module"
    assert order == CHEAP


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
