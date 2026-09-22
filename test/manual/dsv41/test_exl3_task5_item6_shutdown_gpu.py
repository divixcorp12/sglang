"""Task 5 item 6, on a real GPU: shutdown establishes completion of every GPU reader before freeing, and quarantines
what a CUDA error leaves uncertain.

``test/registered/unit/layers/moe/test_exl3_ram_miss_shutdown.py`` pins the WIRING with a fake barrier. Its own
docstring says it "show[s] the WIRING, not that a real device barrier orders the GPU's work". This file runs the same
service against the real device:

* ``test_shutdown_does_not_free_a_tier_while_the_gpu_still_has_work_in_flight``: real work is queued on the stream and
  the tier's ``close`` (the free) asserts, when it runs, that the GPU has finished it.
* ``TestARealCudaError``: a child process takes a REAL device-side error, so ``torch.cuda.synchronize`` really raises,
  then runs the real ``shutdown()``. What is checked is what the process does at EXIT, which a fake barrier cannot
  reach: the slabs are never unregistered (the finalizer that would do it at exit is detached), where an orderly
  shutdown does unregister them. The control run proves the observation is able to see an unregister.

Run on divix01 under ``gpu-run.sh`` with PYTHONPATH pointing at the tree under test and SGLANG_EXL3_SRC /
SGLANG_EXL3_BUILD_DIR set.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _tiers(service):
    return [service.tables[i].streamer_of().pinned_host_cache for i in sorted(service.tables)]


@pytest.mark.parametrize("side_stream", [False, True], ids=["default_stream", "side_stream"])
def test_shutdown_does_not_free_a_tier_while_the_gpu_still_has_work_in_flight(tmp_path, side_stream):
    from test_exl3_ram_miss_graph_gpu import HIDDEN, TOP_K, _layers, _step_route

    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path, timeout_ms=2000, lease=True)
    try:
        x = torch.zeros((1, HIDDEN), device="cuda", dtype=torch.bfloat16)
        weights = torch.full((1, TOP_K), 1.0 / TOP_K, device="cuda")
        ids = torch.tensor([_step_route(0, -1)], device="cuda", dtype=torch.int32)
        Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        graph.replay()
        torch.cuda.synchronize()
        # The GPU's reader of the slabs, in flight: a real leased-chain replay, then a long kernel queued behind it.
        # On a side stream the work is invisible to a barrier that waits only for the current stream.
        stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            graph.replay()
            torch.cuda._sleep(int(4e9))  # a second or more of GPU time
            done = torch.cuda.Event()
            done.record()
        assert not done.query(), "precondition: the GPU is still busy when shutdown starts"
        seen = []
        for tier in _tiers(service):
            real_close = tier.close
            tier.close = lambda real_close=real_close: (seen.append(done.query()), real_close())[1]
        start = time.perf_counter()
        service.shutdown()
        elapsed = time.perf_counter() - start
        assert seen and all(seen), f"a tier was freed while the GPU still had work in flight: {seen}"
        assert not service._quarantined
        assert elapsed >= 0.5, "shutdown waited for the GPU"
        assert done.query()
    finally:
        import sglang.srt.layers.moe.exl3_ram_miss as service_module

        service_module.Exl3RamMissService._instance = None


CHILD = """
import json, sys
sys.path.insert(0, {here!r})
import torch
import test_exl3_task5_item6_shutdown_gpu as t
t.child({mode!r}, {log!r}, {tmp!r})
"""


def child(mode, log, tmp):
    """Runs in the child process: build the real service, optionally take a real CUDA error, shut down, report."""
    from pathlib import Path as _P

    import sglang.srt.mem_cache.pool_host.common as common
    from test_exl3_ram_miss_graph_gpu import _layers

    def logged(slab, *args, **kwargs):  # the observation: was a slab unregistered, and when
        with open(log, "a") as f:
            f.write("unregister\n")

    common._cuda_host_unregister = logged
    layer, streamer, service, checks = _layers(_P(tmp), timeout_ms=200, lease=True)
    if mode == "cuda_error":
        bad = torch.tensor([10**9], device="cuda")
        torch.ones(4, device="cuda")[bad]  # an asynchronous device-side assert: the next synchronize raises
    service.shutdown()
    from sglang.srt.layers.moe import expert_host_tier

    print(
        "RESULT "
        + json.dumps(
            {
                "quarantined": service._quarantined,
                "quarantined_slabs": expert_host_tier.quarantined_slab_count(),
                "finalizers_alive": [t._release_slabs.alive for t in _tiers(service)],
            }
        ),
        flush=True,
    )


def _run_child(mode, tmp_path):
    log = tmp_path / f"unregister_{mode}.log"
    log.write_text("")
    work = tmp_path / f"work_{mode}"
    work.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", CHILD.format(here=str(HERE), mode=mode, log=str(log), tmp=str(work))],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=300,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, f"child produced no result (rc {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT "):]), len([line for line in log.read_text().splitlines() if line])


class TestARealCudaError:
    def test_the_control_an_orderly_shutdown_unregisters_the_slabs(self, tmp_path):
        result, unregistered = _run_child("clean", tmp_path)
        assert result["quarantined"] is False and result["quarantined_slabs"] == 0
        assert unregistered > 0, "the observation can see an unregister"

    def test_a_real_cuda_error_quarantines_and_the_process_exit_never_unregisters_a_slab(self, tmp_path):
        result, unregistered = _run_child("cuda_error", tmp_path)
        assert result["quarantined"] is True
        assert result["quarantined_slabs"] > 0
        assert result["finalizers_alive"] and not any(result["finalizers_alive"]), "the exit-time unregister is detached"
        assert unregistered == 0, "not at shutdown, and not at process exit either: uncertain storage is never recycled"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
