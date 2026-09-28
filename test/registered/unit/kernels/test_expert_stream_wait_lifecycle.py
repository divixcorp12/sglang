"""Completion monitors keep mapped buffers alive until queued CUDA work drains."""

from unittest.mock import Mock, call

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as transport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_completion_monitor_closes_only_after_device_work_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = Mock()
    monitor = operations.monitor
    owners = (object(), object(), object())
    device = torch.device("cuda:2")
    quarantine: list[tuple[object, ...]] = []
    monkeypatch.setattr(torch.cuda, "synchronize", operations.synchronize)
    monkeypatch.setattr(transport, "_wait_completion_quarantine", quarantine)

    transport._close_wait_completion(monitor, 17, owners, device)

    assert operations.mock_calls == [
        call.monitor.expert_stream_wait_completion_cancel(17),
        call.synchronize(device),
        call.monitor.expert_stream_wait_completion_close(17),
    ]
    assert quarantine == []


def test_failed_device_drain_retains_monitor_and_buffer_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = Mock()
    monitor = operations.monitor
    owners = (object(), object(), object())
    device = torch.device("cuda:2")
    quarantine: list[tuple[object, ...]] = []
    failure = RuntimeError("CUDA synchronization failed")
    operations.synchronize.side_effect = failure
    monkeypatch.setattr(torch.cuda, "synchronize", operations.synchronize)
    monkeypatch.setattr(transport, "_wait_completion_quarantine", quarantine)

    with pytest.raises(RuntimeError) as raised:
        transport._close_wait_completion(monitor, 17, owners, device)

    assert raised.value is failure
    assert operations.mock_calls == [
        call.monitor.expert_stream_wait_completion_cancel(17),
        call.synchronize(device),
    ]
    assert len(quarantine) == 1
    retained_monitor, retained_handle, retained_owners, retained_device = quarantine[0]
    assert retained_monitor is monitor
    assert retained_handle == 17
    assert retained_owners is owners
    assert retained_device is device
