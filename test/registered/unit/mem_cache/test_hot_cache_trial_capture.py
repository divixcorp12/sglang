"""hot_cache_trial_capture: a throwaway capture into a private pool measures what the real one will cost.

On fakes for the capture path and the CUDA allocator:
* the global graph pool is swapped for a private one during the trial and restored afterwards;
* every captured graph backend is cleaned up, the draft runners' graph runners are dropped;
* the cost split: the drop while alive, torch's part, what stayed taken after teardown, the pinned torch residue;
  the estimate credits back module loads but not the residue;
* observability forward hooks configured: the trial is skipped;
* reset_trial_routes runs the capture discard but leaves the residency updater disabled.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import hot_cache_trial_capture as tc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

M = 1 << 20


class _Backend:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def cleanup(self):
        self.log.append(("cleanup", self.name))


@pytest.fixture
def world(monkeypatch):
    """free/reserved move like calib7: target +146 reserved and 26 outside torch, draft +42; after teardown 8 MiB of
    module loads stay and 4 MiB of torch residue stays pinned."""
    from sglang.srt import runtime_context
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
    from sglang.srt.model_executor.runner_utils import pool

    log, mem = [], {"free": 3000 * M, "reserved": 12000 * M}
    pools = {"global": None}
    monkeypatch.setattr(pool, "get_global_graph_memory_pool", lambda: pools["global"])
    monkeypatch.setattr(pool, "set_global_graph_memory_pool", lambda p: log.append(("pool", p)) or pools.update({"global": p}))
    monkeypatch.setattr(runtime_context, "get_observability", lambda: SimpleNamespace(forward_hooks=None))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: "private")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (mem["free"], 32000 * M))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: mem["reserved"])
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [])

    def empty_cache():
        if log and log[-1] == ("pool", None):  # teardown: everything but the residue and module loads comes back
            mem.update(free=(3000 - 8 - 4) * M, reserved=(12000 + 4) * M)

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    def capture(model_runner):
        log.append(("capture target", pools["global"]))
        mem.update(free=mem["free"] - 172 * M, reserved=mem["reserved"] + 146 * M)
        runner = lambda name: SimpleNamespace(backend=_Backend(log, name))
        return SimpleNamespace(decode=SimpleNamespace(runner=runner("target decode")), prefill=SimpleNamespace(runner=None))

    monkeypatch.setattr(cuda_graph_setup, "capture_cuda_graphs", capture)
    draft_runner = SimpleNamespace(
        decode_cuda_graph_runner=None,
        prefill_cuda_graph_runner=None,
        attn_backend=SimpleNamespace(cuda_graph_metadata_of_bucket_and_bs={"draft": {1: "meta"}}),
    )

    def draft_init():
        log.append(("capture draft", pools["global"]))
        mem.update(free=mem["free"] - 42 * M, reserved=mem["reserved"] + 42 * M)
        draft_runner.decode_cuda_graph_runner = SimpleNamespace(backend=_Backend(log, "draft decode"))

    draft = SimpleNamespace(init_cuda_graphs=draft_init, _draft_model_runners=lambda: (draft_runner,))
    return SimpleNamespace(log=log, pools=pools, draft=draft, draft_runner=draft_runner)


def test_the_trial_captures_into_a_private_pool_measures_and_tears_down(world):
    target_backend = SimpleNamespace(cuda_graph_metadata_of_bucket_and_bs={"verify": {1: "meta"}, "decode": {}})
    trial = tc.trial_capture(SimpleNamespace(device="cuda", attn_backend=target_backend), world.draft)
    # The capture metadata that pinned calib8b's 168 MiB is emptied, keeping its buckets.
    assert target_backend.cuda_graph_metadata_of_bucket_and_bs == {"verify": {}, "decode": {}}
    assert world.draft_runner.attn_backend.cuda_graph_metadata_of_bucket_and_bs == {"draft": {}}
    assert ("capture target", "private") in world.log and ("capture draft", "private") in world.log
    assert ("cleanup", "target decode") in world.log and ("cleanup", "draft decode") in world.log
    assert world.pools["global"] is None and world.log[-1] == ("pool", None)  # restored
    assert world.draft_runner.decode_cuda_graph_runner is None  # the real capture assigns it again
    assert trial.free_drop == 214 * M and trial.torch_reserved == 188 * M and trial.nontorch == 26 * M
    assert trial.persistent == 12 * M and trial.residue_torch == 4 * M
    assert trial.estimate == (214 - 8) * M  # module loads credited back, the pinned residue not
    assert trial.summary()["estimate_mib"] == 206


def test_drop_capture_metadata_leaves_other_backends_alone():
    assert tc.drop_capture_metadata(SimpleNamespace()) == 0
    assert tc.drop_capture_metadata(None) == 0
    backend = SimpleNamespace(cuda_graph_metadata_of_bucket_and_bs={"a": {1: 0, 2: 0}, "b": {4: 0}})
    assert tc.drop_capture_metadata(backend) == 3


def test_residue_blocks_are_charged_to_the_holder_whose_tensor_starts_inside_them():
    paths = {1000: "input_buffer_pool[('input_ids', 6)]", 5000: "target.attn_backend._verify_mask", 5100: "target.x.y.z"}
    new_blocks = {900: 512, 4096: 2048, 9000: 64}  # holds 1000; holds 5000 and 5100; holds nothing known
    held = tc.attribute_blocks(new_blocks, paths)
    assert held == {"target.attn_backend._verify_mask": 2048, "input_buffer_pool[('input_ids', 6)]": 512,
                    "unattributed": 64}


def test_holder_of_folds_layer_indices_and_drops_module_plumbing():
    path = "model._modules['model']._modules['layers']._modules['12']._modules['self_attn'].indexer_ws"
    assert tc.holder_of(path) == "model['model']['layers'][*]['self_attn'].indexer_ws"
    assert tc.holder_of("draft0.attn_backend.bufs[3]") == "draft0.attn_backend.bufs[*]"


def _trial(free_drop, persistent, residue, nontorch=0):
    return tc.TrialCapture(free_drop * M, (free_drop - nontorch) * M, persistent * M, residue * M, 1.0)


def test_the_capture_estimate_is_the_warm_trial_never_below_the_cold_one_less_its_residue():
    """calib10's cold trial: 212 drop, 148 MiB residue the real capture reuses (cold less residue 64); a warm trial
    of ~100 then measures the real capture directly."""
    from sglang.srt.layers.moe import hot_cache_calibration as hc

    cold, warm = _trial(212, 148, 148), _trial(100, 2, 2)
    assert hc.capture_estimate_of([cold, warm], 0) == 100 * M
    assert hc.capture_estimate_of([cold, _trial(40, 0, 0)], 0) == 64 * M  # the warm one reads low: keep 64
    assert hc.capture_estimate_of([cold, warm], 150 * M) == 150 * M  # never below the eager estimate
    assert hc.capture_estimate_of([cold], 0) == 212 * M  # one trial: its own estimate
    assert hc.capture_estimate_of([], 30 * M) == 30 * M


def test_the_walk_does_not_call_a_dict_subclass_items():
    """calib11b: transformers' lazy auto-mapping imports models from items(); the walk must not call it."""

    class Lazy(dict):
        def items(self):
            raise ValueError("Could not find VoxtralRealtimeTextModel")

    assert tc.tensor_paths({"root": SimpleNamespace(mapping=Lazy(a=1), plain={"b": 2})}, depth=6) == {}


def test_residue_attribution_never_raises_into_the_trial(monkeypatch, caplog):
    """calib11: a service thread grew a dict mid-walk ('dictionary changed size during iteration') and the exception
    aborted the trial. Any failure in the diagnostic is now logged and skipped."""
    monkeypatch.setattr(tc, "tensor_paths", lambda *a, **k: 1 / 0)
    with caplog.at_level("WARNING"):
        tc.log_residue_holders("cold", {1: 2}, SimpleNamespace(model=None), None, ())
    assert any("attribution failed" in r.getMessage() for r in caplog.records)


def test_live_blocks_reads_only_allocated_blocks(monkeypatch):
    snapshot = [{"blocks": [{"address": 10, "size": 4, "state": "active_allocated"},
                            {"address": 14, "size": 8, "state": "inactive"}]}, {"blocks": []}]
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: snapshot)
    assert tc.live_blocks() == {10: 4}


def test_forward_hooks_skip_the_trial(world, monkeypatch):
    from sglang.srt import runtime_context

    monkeypatch.setattr(runtime_context, "get_observability", lambda: SimpleNamespace(forward_hooks=["x"]))
    assert tc.trial_capture(SimpleNamespace(device="cuda"), world.draft) is None
    assert world.log == []


def test_reset_trial_routes_discards_the_capture_routes_and_keeps_the_updater_disabled():
    calls = []
    updater = SimpleNamespace(enabled=torch.tensor(False))

    def discard():
        calls.append("discard")
        updater.enabled.fill_(True)  # what reset_after_capture does

    tc.reset_trial_routes(SimpleNamespace(discard_graph_capture_routes=discard, gpu_residency=updater))
    assert calls == ["discard"] and not bool(updater.enabled)
