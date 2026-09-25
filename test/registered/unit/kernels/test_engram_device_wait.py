"""The Engram device-wait service thread against a simulated device (CPU).

The simulated device follows engram_ring.cuh: post writes the ids, then the post word; wait
polls the done word, then reads the status and the rows.
"""

import faulthandler
import json
import struct
import time

import pytest
import torch

from sglang.kernels.ops.embeddings import engram_ring
from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.srt.layers.engram_host_node import native_engram_host_node
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROWS, DIM, BLOCK = 100, 64, 32
ROW_BYTES = DIM + DIM // BLOCK
N = 3


@pytest.fixture(autouse=True)
def hang_guard():
    # The service runs in C++: a lost publish or a join that never returns hangs, so dump stacks and exit.
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _write(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, tensor) in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for raw in blobs:
            f.write(raw)


class Ring:
    """One RingLookup over plain CPU buffers, and the simulated device's side of the protocol."""

    def __init__(self, table: EngramFileTable, store):
        self.ids = torch.zeros(N, dtype=torch.int64)
        self.rows = torch.zeros(N, ROW_BYTES, dtype=torch.uint8)
        self.control = torch.zeros(engram_ring.CONTROL_WORDS, dtype=torch.int32)
        self.seq = 0
        self.native = native_engram_host_node().RingLookup(
            store, table.path, table.weight_offset, table.scale_offset, table.num_embeddings,
            table.dim, table.dim // table.block, table._tag, N,
            self.ids.data_ptr(), self.rows.data_ptr(), self.control.data_ptr(),
        )

    def post(self, ids) -> int:
        self.seq += 1
        self.ids.copy_(torch.tensor(ids, dtype=torch.int64))
        self.control[engram_ring.POST_SEQ] = self.seq
        return self.seq

    def done(self) -> int:
        return int(self.control[engram_ring.DONE_SEQ])

    def wait(self, seq: int, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.done() != seq:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.0002)
        return True

    def status(self) -> int:
        return int(self.control[engram_ring.STATUS])


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    weight = (torch.randn(ROWS, DIM) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 130, (ROWS, DIM // BLOCK), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    path = tmp_path / "model-00047-of-00048.safetensors"
    _write(
        path,
        {
            "layers.1.engram.test_padding": ("U8", torch.zeros(7, dtype=torch.uint8)),
            "layers.1.engram.embed.weight": ("F8_E4M3", weight),
            "layers.1.engram.embed.scale": ("F8_E8M0", scale),
        },
    )
    table = EngramFileTable(str(path), "layers.1.engram.embed.weight", "layers.1.engram.embed.scale", ROWS, DIM)
    try:
        store = native_engram_host_node().create_store(64 * ROW_BYTES, ROW_BYTES)
    except RuntimeError as exc:
        if "io_uring_queue_init failed" in str(exc):
            pytest.skip(f"io_uring is unavailable: {exc}")
        raise
    ring = Ring(table, store)
    packed = torch.cat([weight.view(torch.uint8), scale.view(torch.uint8)], dim=1)
    yield ring, store, packed
    ring.native.close()


def test_layout_matches_the_native_service():
    layout = native_engram_host_node().ring_layout()
    assert layout == {
        "post_seq": engram_ring.POST_SEQ,
        "done_seq": engram_ring.DONE_SEQ,
        "status": engram_ring.STATUS,
        "fatal_seq": engram_ring.FATAL_SEQ,
        "fatal_status": engram_ring.FATAL_STATUS,
        "control_words": engram_ring.CONTROL_WORDS,
        "refused_after_fatal": engram_ring.REFUSED_AFTER_FATAL,
        "device_timeout": engram_ring.DEVICE_TIMEOUT,
        "device_saw_fatal": engram_ring.DEVICE_SAW_FATAL,
        "waits": engram_ring.WAITS,
        "spins": engram_ring.SPINS,
        "spin_ns_lo": engram_ring.SPIN_NS_LO,
        "spin_ns_hi": engram_ring.SPIN_NS_HI,
        "spin_us_max": engram_ring.SPIN_US_MAX,
    }


def test_each_posted_sequence_is_served_once_with_its_own_rows(setup):
    ring, _, packed = setup
    for ids in ([3, 0, 49], [3, 0, 49], [7, 7, 1], [99, 2, 50]):
        seq = ring.post(ids)
        assert ring.wait(seq)
        assert ring.status() == 0
        assert torch.equal(ring.rows, packed[ids])
    assert ring.native.served() == 4


def test_done_is_published_only_after_the_rows_and_status(setup):
    """The service delays each request, so a done word published before the rows (an unordered or early publish)
    is observed with the previous request's rows still in the buffer."""
    ring, _, packed = setup
    assert ring.wait(ring.post([1, 2, 3]))
    ring.control[engram_ring.STATUS] = -1
    ring.native.set_test_delay_us(50_000)
    seq = ring.post([4, 5, 6])
    while ring.done() != seq:
        pass
    assert ring.status() == 0
    assert torch.equal(ring.rows, packed[[4, 5, 6]])


def test_a_failed_lookup_publishes_its_status(setup):
    ring, _, _ = setup
    seq = ring.post([1, -1, 2])
    assert ring.wait(seq)
    assert ring.status() == 2
    assert not ring.rows.any()


def test_a_device_timeout_latches_and_the_service_refuses_later_requests(setup):
    """The device gives up first (its timeout is shorter than the service's delay) and latches the fatal word,
    as wait_kernel does; the late publish of that request must not revive the lookup."""
    ring, store, packed = setup
    ring.native.set_test_delay_us(200_000)
    seq = ring.post([8, 9, 10])
    assert not ring.wait(seq, timeout_s=0.02)
    ring.control[engram_ring.FATAL_STATUS] = engram_ring.DEVICE_TIMEOUT
    ring.control[engram_ring.FATAL_SEQ] = seq
    assert ring.wait(seq)
    ring.native.set_test_delay_us(0)
    accesses = store.stats()["accesses"]
    seq = ring.post([11, 12, 13])
    assert ring.wait(seq)
    assert ring.status() == engram_ring.REFUSED_AFTER_FATAL
    assert store.stats()["accesses"] == accesses
    assert not torch.equal(ring.rows, packed[[11, 12, 13]])


def test_a_closed_lookup_is_not_served(setup):
    """close() is what the graph's retained context runs before its pinned buffers are freed."""
    ring, _, _ = setup
    assert ring.wait(ring.post([1, 2, 3]))
    ring.native.close()
    seq = ring.post([4, 5, 6])
    assert not ring.wait(seq, timeout_s=0.1)
    assert ring.native.served() == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
