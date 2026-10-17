"""The scheduler's end of the startup warm-up: the EXL3 copy engine arms (its CPU split and split model calibrated)
only after the warm-up signal, with the scheduler idle, and the reply that holds the server's ready waits for it
(CPU)."""

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.copy_engine_quiet import QuietReport
from sglang.srt.managers.io_struct import EndStartupWarmupReqInput
from sglang.srt.managers.scheduler_components import startup_warmup_end as swe
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ARM_DECODES = 16


class FakeService:
    def __init__(self, *, decodes=ARM_DECODES, copy_engine=True):
        self.copy_decodes = decodes
        self.copy_engine = copy_engine
        self.armed = False
        self.calls = []

    @property
    def copy_engine_arm_pending(self):
        return self.copy_engine and not self.armed

    def arm_copy_engine_when_quiet(self, *, deadline, busy_s, warmup_ended_s_ago):
        self.calls.append((deadline, dict(busy_s), warmup_ended_s_ago))
        self.armed = True
        return QuietReport(quiet=True, waited_s=0.0, busy_s=dict(busy_s), still_busy=(), probes=("ram_miss_thread",))


class Harness:
    def __init__(self, service, *, idle=True):
        self.service = service
        self.idle = idle
        self.now = 1000.0
        self.sent = []
        self.end = swe.SchedulerStartupWarmupEnd(
            is_fully_idle=lambda: self.idle,
            send_output=lambda output, req: self.sent.append((output, req)),
            service=lambda: self.service,
            clock=lambda: self.now,
            wall_clock=lambda: 5000.0,
            timeout_s=30.0,
        )


@pytest.fixture(autouse=True)
def ram_miss_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules, swe._RAM_MISS_MODULE, SimpleNamespace(COPY_ENGINE_ARM_DECODES=ARM_DECODES)
    )


def test_nothing_arms_before_the_warm_up_signal_however_many_decodes():
    h = Harness(FakeService(decodes=10 * ARM_DECODES))
    for _ in range(5):
        h.end.check_pending()
    assert h.service.calls == [] and not h.service.armed


def test_an_idle_scheduler_arms_at_the_signal_and_replies_inline():
    h = Harness(FakeService())
    req = EndStartupWarmupReqInput(warmup_ended_at=4998.5)
    output = h.end.handle(req)
    assert h.service.calls == [(1030.0, {}, 1.5)]
    assert output.success and output.message.startswith("copy engine armed; quiet after")
    assert h.sent == []
    h.end.check_pending()
    assert len(h.service.calls) == 1
    again = h.end.handle(EndStartupWarmupReqInput(warmup_ended_at=None))
    assert again.message == "startup warm-up already ended" and len(h.service.calls) == 1


def test_a_busy_scheduler_holds_the_reply_until_idle_and_counts_its_busy_time():
    h = Harness(FakeService(), idle=False)
    req = EndStartupWarmupReqInput(warmup_ended_at=4990.0)
    assert h.end.handle(req) is None
    h.now += 2.0
    h.end.check_pending()
    assert h.service.calls == [] and h.sent == []
    h.now += 1.5
    h.idle = True
    h.end.check_pending()
    deadline, busy_s, ended_s_ago = h.service.calls[0]
    assert deadline == 1030.0 and busy_s == {"scheduler_requests": 3.5} and ended_s_ago == 10.0
    assert len(h.sent) == 1 and h.sent[0][1] is req and h.sent[0][0].message.startswith("copy engine armed")


def test_a_scheduler_never_idle_arms_anyway_at_the_deadline():
    h = Harness(FakeService(), idle=False)
    req = EndStartupWarmupReqInput(warmup_ended_at=4990.0)
    assert h.end.handle(req) is None
    h.now += 29.9
    h.end.check_pending()
    assert h.service.calls == []
    h.now += 0.1
    h.end.check_pending()
    assert h.service.calls[0][1] == {"scheduler_requests": 30.0}
    assert len(h.sent) == 1


def test_too_few_decodes_replies_at_once_and_arms_at_the_first_idle_moment_after_enough():
    service = FakeService(decodes=3)
    h = Harness(service)
    output = h.end.handle(EndStartupWarmupReqInput(warmup_ended_at=4999.0))
    assert output.message == "copy engine arming deferred (3 decodes)" and service.calls == []
    service.copy_decodes = ARM_DECODES
    h.idle = False
    h.end.check_pending()
    assert service.calls == []
    h.now += 4.0
    h.idle = True
    h.end.check_pending()
    assert service.calls == [(1030.0, {"scheduler_requests": 4.0}, 1.0)] and h.sent == []


def test_no_copy_engine_replies_at_once():
    assert Harness(None).end.handle(EndStartupWarmupReqInput()).message == "no copy engine to arm"
    h = Harness(FakeService(copy_engine=False))
    assert h.end.handle(EndStartupWarmupReqInput()).message == "no copy engine to arm"
    h.end.check_pending()
    assert h.service.calls == []
