"""Real CUDA regression checks for the host-monitored, sleep-free request wait."""

import pytest
import torch

from test_exl3_ram_miss_cuda import _buffers, _close, _service, _set, _step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "graph"])
def test_delayed_requests_complete_without_device_polling(tmp_path, captured):
    _, _, _, slabs, host, dev = _service(tmp_path, capacity=1)
    try:
        buffers = _buffers()
        _set(buffers, [1], [1])
        _step(dev, buffers)
        torch.cuda.synchronize()
        before = dev.stats()["polls"]
        graph = None
        if captured:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _step(dev, buffers)
        host.inject(delay_s=0.03)
        # Reusing one mailbox must not let a previous replay release the next wait.
        for expert in (3, 5, 7):
            _set(buffers, [expert], [expert])
            if graph is None:
                _step(dev, buffers)
            else:
                graph.replay()
            torch.cuda.synchronize()
            assert buffers["keep"].item() == 1.0
            assert buffers["host_rows"][0].item() == host.mapping(0)[expert] >= 0
            assert buffers["ram_miss"].item() == 0
        assert dev.stats()["polls"] == before, "the stream wait must not run a device polling loop"
        assert dev.stats()["timeouts"] == 0
        assert host.fatal_seq() == 0
    finally:
        host.inject(delay_s=0.0)
        _close(host, slabs)


def test_delayed_request_times_out_without_device_polling(tmp_path):
    _, _, _, slabs, host, dev = _service(tmp_path, timeout_ms=50)
    try:
        buffers = _buffers()
        host.inject(delay_s=0.5)
        _set(buffers, [7], [7])
        _step(dev, buffers)
        torch.cuda.synchronize()
        assert buffers["keep"].item() == 0.0
        assert host.fatal_seq() != 0
        assert dev.stats()["timeouts"] == 1
        assert dev.stats()["polls"] == 0
        # Sticky failure must bypass the gate even though no service reply follows.
        _step(dev, buffers)
        torch.cuda.synchronize()
        assert buffers["keep"].item() == 0.0
        assert dev.stats()["timeouts"] == 1
    finally:
        host.inject(delay_s=0.0)
        _close(host, slabs)


@pytest.mark.parametrize("staged", [False, True], ids=["lease", "rest"])
def test_delayed_lease_copy_completes_without_device_polling(tmp_path, staged):
    from test_exl3_lease_kernels_cuda import Service, _delivered
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

    service = Service(tmp_path)
    try:
        service.plan([3, 5])
        service.host.inject(delay_s=0.03)
        dev = service.dev
        if not staged:
            service.step()
        else:
            dev.post(0, service.planned, service.count, service.routes, -1)
            dev.hit_wait(0, service.planned, service.count, service.dest_slots, 0)
            copy_expert_row_segments_gpu(service.segments, dev.host_rows_1, dev.dst_slots_1, dev.go_1)
            dev.stage_ack(1)
            dev.rest_wait(0, service.planned, service.count, service.dest_slots, service.ram_miss)
            copy_expert_row_segments_gpu(service.segments, dev.host_rows_2, dev.dst_slots_2, dev.go_2)
            dev.stage_ack(2)
            dev.finalize(service.count, service.keep)
        torch.cuda.synchronize()
        assert service.keep.item() == 1.0
        _delivered(service, [3, 5])
        assert dev.stats()["polls"] == 0
        assert service.until(lambda: service.host.counters()["leases_acked"] == 2)
        assert service.host.fatal_seq() == 0
    finally:
        service.host.inject(delay_s=0.0)
        service.close()
