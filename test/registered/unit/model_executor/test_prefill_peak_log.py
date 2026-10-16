"""SGLANG_DEBUG_PREFILL_PEAK_LOG: off leaves the DSpark worker untouched; on, a prefill logs each call's memory.

A fake allocator stands in for torch.cuda (allocated/reserved and their peaks, reset by reset_peak_memory_stats), and
a fake worker holds the modules install() wraps. The nested case checks that a child's reset does not lose the parent's
earlier peak, and that the logged line parses back to the calls in order.
"""

import logging
from types import SimpleNamespace

import torch

from sglang.srt.debug_utils import prefill_peak_log as ppl
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20


class FakeCuda:
    def __init__(self):
        self.allocated = self.reserved = 0
        self.alloc_max = self.reserved_max = 0
        self.capturing = False

    def alloc(self, mib):
        self.allocated += mib * MIB
        self.reserved = max(self.reserved, self.allocated)
        self.alloc_max = max(self.alloc_max, self.allocated)
        self.reserved_max = max(self.reserved_max, self.reserved)

    def free(self, mib):
        self.allocated -= mib * MIB

    def memory_allocated(self):
        return self.allocated

    def max_memory_allocated(self):
        return self.alloc_max

    def memory_reserved(self):
        return self.reserved

    def max_memory_reserved(self):
        return self.reserved_max

    def reset_peak_memory_stats(self):
        self.alloc_max, self.reserved_max = self.allocated, self.reserved

    def is_current_stream_capturing(self):
        return self.capturing


class Module(torch.nn.Module):
    def __init__(self, body=None, **children):
        super().__init__()
        self.body = body
        for k, v in children.items():
            setattr(self, k, v)

    def forward(self, *args):
        return self.body(*args) if self.body else None


def _worker(cuda):
    def attn_body():
        cuda.alloc(300)  # transient
        cuda.free(300)
        cuda.alloc(10)  # kept

    attn = Module(attn_body)
    moe = Module(lambda: (cuda.alloc(50), cuda.free(50)))

    def layer_body():
        cuda.alloc(500)  # the layer's own transient, larger than its children's and before them
        cuda.free(500)
        attn()
        moe()

    layer = Module(layer_body, self_attn=attn, mlp=moe)
    inner = Module(layers=torch.nn.ModuleList([layer]))
    target = Module(lambda: layer(), model=inner, logits_processor=Module())
    runner = SimpleNamespace(model=target, sampler=Module())
    injector = SimpleNamespace(inject_target_hidden=lambda **kw: cuda.alloc(5))
    worker = SimpleNamespace(model_runner=runner, _kv_injector=injector, draft_model=Module())

    def forward_prefill(batch, on_publish):
        target()
        worker._kv_injector.inject_target_hidden(target_hidden=None)
        return "done"

    worker._forward_prefill = forward_prefill
    return worker, (worker._forward_prefill, target.forward, layer.forward, attn.forward, moe.forward)


def test_off_wraps_nothing():
    cuda = FakeCuda()
    worker, originals = _worker(cuda)
    injector_fn = worker._kv_injector.inject_target_hidden
    with envs.SGLANG_DEBUG_PREFILL_PEAK_LOG.override(False):
        assert ppl.maybe_install(worker) is None
    layer = worker.model_runner.model.model.layers[0]
    now = (worker._forward_prefill, worker.model_runner.model.forward, layer.forward, layer.self_attn.forward,
           layer.mlp.forward)
    assert all(a is b for a, b in zip(now, originals))
    assert worker._kv_injector.inject_target_hidden is injector_fn
    assert "forward" not in vars(layer) and "forward" not in vars(worker.model_runner.model)


def test_on_logs_each_call_with_nested_peaks(caplog):
    cuda = FakeCuda()
    worker, _ = _worker(cuda)
    ppl.install(worker, ppl.PrefillPeakLog(cuda=cuda))
    with caplog.at_level(logging.INFO, logger=ppl.__name__):
        assert worker._forward_prefill(SimpleNamespace(extend_num_tokens=4096), None) == "done"
    records = [ppl.parse_line(r.getMessage()) for r in caplog.records]
    records = [r for r in records if r is not None]
    assert len(records) == 1 and records[0]["tokens"] == 4096
    calls = {c["name"]: c for c in records[0]["calls"]}
    assert [c["name"] for c in records[0]["calls"]] == ["L0.attn", "L0.moe", "L0", "target", "dspark.inject",
                                                        "dspark.prefill"]
    assert calls["L0.attn"]["alloc_peak_mib"] == 300 and calls["L0.attn"]["alloc_after_mib"] == 10
    assert calls["L0.moe"]["alloc_before_mib"] == 10 and calls["L0.moe"]["alloc_peak_mib"] == 60
    # The layer's own 500 MiB came before its children reset the peak counter; it survives into the layer and root.
    assert calls["L0"]["alloc_peak_mib"] == 500 and calls["L0"]["depth"] == 2
    assert calls["L0.attn"]["depth"] == 3 and calls["target"]["depth"] == 1
    assert calls["dspark.prefill"]["alloc_peak_mib"] == 500 and calls["dspark.prefill"]["depth"] == 0
    assert calls["dspark.prefill"]["alloc_after_mib"] == 15
    assert calls["dspark.prefill"]["reserved_peak_mib"] == 500 and calls["L0.moe"]["reserved_peak_mib"] == 500
    for c in records[0]["calls"]:
        assert set(c) == {"name", "depth", "alloc_before_mib", "alloc_peak_mib", "alloc_after_mib",
                          "reserved_before_mib", "reserved_peak_mib", "reserved_after_mib"}


def test_calls_outside_a_prefill_or_under_capture_pass_through(caplog):
    cuda = FakeCuda()
    worker, _ = _worker(cuda)
    log = ppl.install(worker, ppl.PrefillPeakLog(cuda=cuda))
    with caplog.at_level(logging.INFO, logger=ppl.__name__):
        worker.model_runner.model()  # a decode forward: no root open
        cuda.capturing = True
        worker._forward_prefill(SimpleNamespace(extend_num_tokens=8), None)
    assert not [r for r in caplog.records if ppl.parse_line(r.getMessage())]
    assert log.stack == [] and log.entries == []


def test_on_installs_once_at_construction():
    worker, originals = _worker(FakeCuda())
    with envs.SGLANG_DEBUG_PREFILL_PEAK_LOG.override(True):
        assert isinstance(ppl.maybe_install(worker), ppl.PrefillPeakLog)
    assert worker._forward_prefill is not originals[0]
    assert getattr(worker.model_runner.model.model.layers[0].mlp.forward, "__prefill_peak_wrapped__", False)


def test_install_is_idempotent():
    cuda = FakeCuda()
    worker, _ = _worker(cuda)
    log = ppl.PrefillPeakLog(cuda=cuda)
    ppl.install(worker, log)
    once = worker.model_runner.model.model.layers[0].self_attn.forward
    ppl.install(worker, log)
    assert worker.model_runner.model.model.layers[0].self_attn.forward is once
