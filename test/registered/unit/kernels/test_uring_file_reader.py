"""Batched io_uring file reads: buffered, O_DIRECT, fixed buffers, EOF, errors."""

import os
import tempfile

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/include/liburing.h"),
    reason="io_uring file reader tests require liburing headers.",
)

PAGE = 4096


def _reader(queue_depth=8):
    from sglang.kernels.ops.io.uring_file_reader import UringFileReader

    return UringFileReader(queue_depth=queue_depth)


def _aligned(nbytes):
    storage = torch.empty(nbytes + PAGE, dtype=torch.uint8)
    start = (-storage.data_ptr()) % PAGE
    return storage, storage[start : start + nbytes]


@pytest.fixture
def data_file():
    generator = torch.Generator().manual_seed(7)
    payload = torch.randint(
        0, 256, (10 * PAGE + 123,), dtype=torch.uint8, generator=generator
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.bin")
        with open(path, "wb") as stream:
            stream.write(payload.numpy().tobytes())
        yield path, payload


def _extents(file_id, offsets, destination_offsets, lengths, base):
    count = len(offsets)
    return (
        torch.full((count,), file_id, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(destination_offsets, dtype=torch.int64) + base,
        torch.tensor(lengths, dtype=torch.int64),
    )


def test_buffered_reads_scattered_extents_exactly(data_file):
    path, payload = data_file
    reader = _reader()
    file_id = reader.open(path, direct=False)
    self_check = reader.open(path, direct=False)
    assert self_check == file_id
    assert reader.file_size(file_id) == payload.numel()

    offsets = [0, 17, 3 * PAGE + 5, 9 * PAGE, 10 * PAGE]
    lengths = [PAGE, 900, 2 * PAGE + 11, PAGE + 123, 0]
    destination = torch.zeros(sum(lengths), dtype=torch.uint8)
    destination_offsets = [sum(lengths[:index]) for index in range(len(lengths))]

    read = reader.read(
        *_extents(
            file_id, offsets, destination_offsets, lengths, destination.data_ptr()
        )
    )

    assert read == sum(lengths)
    expected = torch.cat(
        [payload[offset : offset + length] for offset, length in zip(offsets, lengths)]
    )
    assert torch.equal(destination, expected)


def test_direct_reads_page_aligned_extents_and_more_extents_than_queue_depth(data_file):
    path, payload = data_file
    reader = _reader(queue_depth=4)
    file_id = reader.open(path, direct=True)
    pages = [index % 10 for index in range(37)]
    _, destination = _aligned(len(pages) * PAGE)

    read = reader.read(
        *_extents(
            file_id,
            [page * PAGE for page in pages],
            [index * PAGE for index in range(len(pages))],
            [PAGE] * len(pages),
            destination.data_ptr(),
        )
    )

    assert read == len(pages) * PAGE
    expected = torch.cat([payload[page * PAGE : (page + 1) * PAGE] for page in pages])
    assert torch.equal(destination, expected)


def test_final_partial_page_returns_only_bytes_before_end_of_file(data_file):
    path, payload = data_file
    reader = _reader()
    file_id = reader.open(path, direct=True)
    _, destination = _aligned(2 * PAGE)

    read = reader.read(
        *_extents(file_id, [9 * PAGE], [0], [2 * PAGE], destination.data_ptr())
    )

    assert read == PAGE + 123
    assert torch.equal(destination[: PAGE + 123], payload[9 * PAGE :])


def test_extent_starting_at_end_of_file_reads_nothing(data_file):
    path, payload = data_file
    reader = _reader()
    file_id = reader.open(path, direct=False)
    destination = torch.zeros(PAGE, dtype=torch.uint8)

    read = reader.read(
        *_extents(file_id, [payload.numel()], [0], [PAGE], destination.data_ptr())
    )

    assert read == 0


def test_reads_into_registered_buffer(data_file):
    path, payload = data_file
    reader = _reader()
    if not reader.registered_buffers_supported:
        pytest.skip("kernel does not support sparse registered buffers")
    file_id = reader.open(path, direct=True)
    _, destination = _aligned(4 * PAGE)
    assert reader.register_buffer(destination)

    read = reader.read(
        *_extents(
            file_id,
            [2 * PAGE, 0],
            [0, 2 * PAGE],
            [2 * PAGE, 2 * PAGE],
            destination.data_ptr(),
        )
    )

    assert read == 4 * PAGE
    assert torch.equal(destination[: 2 * PAGE], payload[2 * PAGE : 4 * PAGE])
    assert torch.equal(destination[2 * PAGE :], payload[: 2 * PAGE])
    assert not reader.register_buffer(destination[PAGE : 3 * PAGE])
    assert reader.unregister_buffer(destination) >= 1


def test_registration_is_dropped_when_its_tensor_is_freed():
    import gc

    reader = _reader()
    if not reader.registered_buffers_supported:
        pytest.skip("kernel does not support sparse registered buffers")
    storage, destination = _aligned(2 * PAGE)
    start = destination.data_ptr() - storage.data_ptr()
    assert reader.register_buffer(destination)
    assert not reader.register_buffer(storage[start : start + PAGE])

    del destination
    gc.collect()

    replacement = storage[start : start + 2 * PAGE]
    assert reader.register_buffer(replacement)
    assert reader.unregister_buffer(replacement) >= 1


def test_one_reader_resets_state_between_reads_of_different_sizes(data_file):
    path, payload = data_file
    reader = _reader(queue_depth=4)
    file_id = reader.open(path, direct=False)

    for count in (37, 3, 50, 1):
        pages = [(index * 7) % 10 for index in range(count)]
        destination = torch.zeros(count * PAGE, dtype=torch.uint8)
        read = reader.read(
            *_extents(
                file_id,
                [page * PAGE for page in pages],
                [index * PAGE for index in range(count)],
                [PAGE] * count,
                destination.data_ptr(),
            )
        )

        assert read == count * PAGE
        expected = torch.cat(
            [payload[page * PAGE : (page + 1) * PAGE] for page in pages]
        )
        assert torch.equal(destination, expected)


def test_calls_from_another_thread_raise_without_touching_the_reader(data_file):
    import threading

    path, payload = data_file
    reader = _reader()
    file_id = reader.open(path, direct=False)
    destination = torch.zeros(PAGE, dtype=torch.uint8)
    errors = []

    def use_from_another_thread():
        for call in (
            lambda: reader.open(path, direct=True),
            lambda: reader.read(
                *_extents(file_id, [0], [0], [PAGE], destination.data_ptr())
            ),
        ):
            try:
                call()
            except Exception as error:
                errors.append(str(error))

    thread = threading.Thread(target=use_from_another_thread)
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert len(errors) == 2
    assert all("belongs to thread" in error for error in errors)
    assert not destination.any()
    read = reader.read(*_extents(file_id, [0], [0], [PAGE], destination.data_ptr()))
    assert read == PAGE
    assert torch.equal(destination, payload[:PAGE])


def test_registration_collected_on_another_thread_is_dropped_by_the_owner():
    import gc
    import threading

    reader = _reader()
    if not reader.registered_buffers_supported:
        pytest.skip("kernel does not support sparse registered buffers")
    storage, destination = _aligned(2 * PAGE)
    start = destination.data_ptr() - storage.data_ptr()
    assert reader.register_buffer(destination)
    holder = [destination]
    del destination

    def collect_on_another_thread():
        holder.clear()
        gc.collect()

    thread = threading.Thread(target=collect_on_another_thread)
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive()
    replacement = storage[start : start + 2 * PAGE]
    assert reader.register_buffer(replacement)
    assert reader.unregister_buffer(replacement) >= 1


def test_failed_read_raises_and_reader_stays_usable(data_file):
    path, payload = data_file
    reader = _reader()
    direct_id = reader.open(path, direct=True)
    _, aligned = _aligned(2 * PAGE)

    with pytest.raises(Exception, match="io_uring read"):
        reader.read(*_extents(direct_id, [0], [1], [PAGE], aligned.data_ptr()))

    read = reader.read(*_extents(direct_id, [PAGE], [0], [PAGE], aligned.data_ptr()))
    assert read == PAGE
    assert torch.equal(aligned[:PAGE], payload[PAGE : 2 * PAGE])


def test_rejects_unknown_file_mismatched_extents_and_closed_reader(data_file):
    path, _ = data_file
    reader = _reader()
    destination = torch.zeros(PAGE, dtype=torch.uint8)

    with pytest.raises(Exception, match="not open"):
        reader.read(*_extents(5, [0], [0], [PAGE], destination.data_ptr()))
    with pytest.raises(ValueError, match="matching lengths"):
        reader.read(
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        )
    reader.close()
    with pytest.raises(Exception, match="closed"):
        reader.open(path, direct=False)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
