"""The MoE side stream's ordering contract, on the GPU, eager and inside a captured graph.

A fork runs after everything the current stream issued before it, and the current stream reads a fork's result
only after join. Both directions are made deterministic with a long device sleep, so a missing wait reads stale
data instead of racing.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SLEEP_CYCLES = 200_000_000  # about 70 ms at 3 GHz: far longer than any launch gap


@pytest.fixture
def side(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import moe_side_stream

    monkeypatch.setattr(moe_side_stream, "_stream", None)
    monkeypatch.setattr(moe_side_stream, "_forked", False)
    with envs.SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM.override(True):
        moe_side_stream.enable_if_requested()
    assert moe_side_stream.active()
    return moe_side_stream


def _fork_and_join(side, src: torch.Tensor) -> torch.Tensor:
    # Main writes src late; the fork must see it. The fork writes late; main must see that after join.
    torch.cuda._sleep(SLEEP_CYCLES)
    src.add_(1)

    def work():
        torch.cuda._sleep(SLEEP_CYCLES)
        return src * 2

    out = side.fork(work, inputs=(src,))
    side.join((out,))
    return out.clone()


def test_fork_sees_prior_work_and_join_orders_the_read(side):
    src = torch.zeros(4, device="cuda")
    got = _fork_and_join(side, src)
    torch.cuda.synchronize()
    assert got.tolist() == [2.0] * 4


def test_captured_fork_and_join_replay_in_order(side):
    src = torch.zeros(4, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _fork_and_join(side, src)  # warm the allocator outside capture
        with torch.cuda.graph(graph, stream=stream):
            got = _fork_and_join(side, src)
    torch.cuda.synchronize()
    src.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert got.tolist() == [2.0] * 4
    graph.replay()
    torch.cuda.synchronize()
    assert got.tolist() == [4.0] * 4


def test_join_without_fork_is_a_no_op(side):
    side.join()
    assert not side._forked


def test_flag_off_creates_no_stream(monkeypatch):
    from sglang.srt.layers.moe import moe_side_stream

    monkeypatch.setattr(moe_side_stream, "_stream", None)
    moe_side_stream.enable_if_requested()
    assert not moe_side_stream.active()
