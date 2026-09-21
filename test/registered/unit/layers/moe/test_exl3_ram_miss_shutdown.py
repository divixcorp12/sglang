"""Shutdown of the option C service: stop admission, establish that no GPU reader runs, then free; else quarantine
(CPU, fake device barrier); LEASE_PROTOCOL.md 14.3 and item 9(c) of 18.2.

Written after the code, unlike the service-side lease tests: the sequence is a list of named steps and the tests
read it back. Each test names the mutation it must fail under. A fake CUDA barrier (``_synchronize``) stands in for
``torch.cuda.synchronize``; it is a stand-in for the barrier, so these tests show the WIRING, not that a real device
barrier orders the GPU's work.
"""

import faulthandler
import gc
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
    def fake():
        order.append("synchronize")
        behaviour()

    monkeypatch.setattr(service, "_synchronize", fake)


def test_shutdown_closes_admission_then_establishes_completion_then_stops_then_frees(world, monkeypatch):
    """Mutation: the barrier is skipped, run before admission closes, or run after the thread stops. The header's
    shutdown word is read from inside the barrier, so 'admission closed first' is observed, not inferred."""
    service, caches, order = world
    seen = {}
    _barrier(service, monkeypatch, order, lambda: seen.setdefault("shutdown", service.host.lease_header()["shutdown"]))
    service.shutdown()
    assert order == ["close_admission", "synchronize", "stop", "free0", "free1"]
    assert seen["shutdown"] == 1, "the header's shutdown word was set before the device barrier ran"
    assert not service._quarantined


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
