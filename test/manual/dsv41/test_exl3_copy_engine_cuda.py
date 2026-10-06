"""The copy engine (LEASE_PROTOCOL.md, "Copy engine") against the real C++ service and the production chain.

Under capture, with the engine armed, the service grants resident lanes COPYING and its copy thread copies them with
cuMemcpyAsync; CW closes the gate, the stream waits on it, and CC checks CopyDone. Every byte check reads a snapshot
enqueued right after the gather, before anything synchronizes: a gate opened before the copies landed shows as stale
bytes, most visibly under the ballast, which delays every copy job's completion by one large extra copy.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from lease_chain_rig import LAYERS, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe.expert_lease_block import wire_layout  # noqa: E402

DEMAND_RECORDS = wire_layout(8).demand_records

BALLAST_BYTES = 256 << 20  # ~20 ms of H2D at the copy engine's ~13.5 GB/s, ahead of every copy job
POOL = 10  # experts the plans draw from: a capacity of 8 keeps most lanes hits and still evicts


def _ballast(c, nbytes=BALLAST_BYTES):
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    src = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    c.host.copy_engine_ballast(dst, src)
    return dst, src


def _capture(c):
    """Warm every kernel eagerly (a first launch while a copy wait holds the stream can stall the copy thread), then
    capture both rows' gathers in one graph."""
    for row in range(LAYERS):
        c.plan([0, 1], row)
        c.gather(row)
        c.snapshot(row)
    torch.cuda.synchronize()
    assert c.handled()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for row in range(LAYERS):
            c.gather(row)
    return graph, stream


def _replay(c, graph, stream, plans):
    with torch.cuda.stream(stream):
        for row in range(LAYERS):
            c.plan(plans[row], row)
        graph.replay()
        snapshots = {row: c.snapshot(row) for row in range(LAYERS)}
    stream.synchronize()
    for row in range(LAYERS):
        c.check(plans[row], snapshots[row], row)


def _plans(rng):
    return {row: rng.sample(range(POOL), rng.randint(1, TOP_K)) for row in range(LAYERS)}


@pytest.mark.parametrize("sm_small_copies", [False, True], ids=["six_copies", "sm_small_copies"])
@pytest.mark.parametrize("ballast", [False, True], ids=["prompt", "ballast"])
def test_captured_hits_are_copied_by_the_engine_byte_exact_armed_and_unarmed(tmp_path, ballast, sm_small_copies):
    """Unarmed replays copy nothing on the engine; armed ones copy their hits there while S streams the misses, three
    times round the ring under eviction, and every destination row holds its expert's bytes when the stream moves on."""
    c = Chain(tmp_path, copy_engine=True, sm_small_copies=sm_small_copies)
    try:
        keep = _ballast(c) if ballast else None
        graph, stream = _capture(c)
        rng = random.Random(11)
        for _ in range(4):
            _replay(c, graph, stream, _plans(rng))
        assert c.handled()
        assert c.host.counters()["copy_lanes"] == 0, "an unarmed replay copied on the engine"
        c.host.arm_copy_engine()
        for _ in range(3 * DEMAND_RECORDS // LAYERS):
            _replay(c, graph, stream, _plans(rng))
        assert c.handled()
        counters = c.host.counters()
        assert counters["copy_lanes"] > 0
        assert counters["overruns"] == 0 and counters["evictions"] > 0
        assert c.host.copy_engine_idle(5.0)
        del keep
    finally:
        c.close()


def test_an_eager_gather_never_copies_on_the_engine(tmp_path):
    """Only a captured post lets the service copy: an eager gather may load a kernel module while a copy wait holds the
    stream, so it is served by C1 even with the engine armed."""
    c = Chain(tmp_path, copy_engine=True)
    try:
        c.host.arm_copy_engine()
        for experts in ([0, 1, 2], [2, 1, 0], [0, 1]):
            c.plan(experts)
            c.gather()
            snapshot = c.snapshot()
            torch.cuda.synchronize()
            c.check(experts, snapshot)
        assert c.handled()
        assert c.host.counters()["copy_lanes"] == 0
    finally:
        c.close()


_HELD_SCRIPT = """
import sys
import torch
sys.path.insert(0, sys.argv[2])
from lease_chain_rig import Chain
from test_exl3_copy_engine_cuda import _ballast, _capture, _replay
c = Chain(sys.argv[1], copy_engine=True, copy_wait_ms=2)
keep = _ballast(c, 1 << 30)
graph, stream = _capture(c)
c.host.arm_copy_engine()
for _ in range(50):
    _replay(c, graph, stream, {0: [0, 1], 1: [0, 1]})
print("reached", flush=True)
"""


def test_a_copy_wait_held_past_its_timeout_aborts_the_process(tmp_path):
    """A 2 ms copy-wait timeout under a ~75 ms ballast: the watchdog, which samples the gate every 20 ms, sees it closed
    too long and aborts; the decode stream is never left waiting on a copy that may not come."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_HELD_SCRIPT), str(tmp_path), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=300,
    )
    assert "reached" not in result.stdout, result.stdout
    assert result.returncode != 0, result.returncode
    assert "FATAL" in result.stderr and "a copy wait held the decode stream" in result.stderr, result.stderr[-2000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
