"""Row reads of file-backed tensors through io_uring, checked against the file bytes."""

import os
import tempfile

import pytest
import torch

from sglang.srt.model_loader.file_row_reader import (
    AlignedRowSource,
    PagedRowBatch,
    PagedRowSource,
    read_plans,
    validate_file_reader_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/include/liburing.h"),
    reason="io_uring row reader tests require liburing headers.",
)

PAGE = 4096


def _reader():
    from sglang.kernels.ops.io.uring_file_reader import UringFileReader

    return UringFileReader(queue_depth=16)


def _write_rows(directory, name, row_count, row_bytes, seed):
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randint(
        0, 256, (row_count, row_bytes), dtype=torch.uint8, generator=generator
    )
    path = os.path.join(directory, name)
    with open(path, "wb") as stream:
        stream.write(rows.numpy().tobytes())
    return path, rows


def _aligned_rows(row_count, row_bytes):
    storage = torch.empty(row_count * row_bytes + PAGE, dtype=torch.uint8)
    start = (-storage.data_ptr()) % PAGE
    return storage, storage[start : start + row_count * row_bytes].view(
        row_count, row_bytes
    )


def test_mode_names_are_validated():
    assert validate_file_reader_mode("uring_direct") == "uring_direct"
    with pytest.raises(ValueError, match="unknown file reader mode"):
        validate_file_reader_mode("uring-direct")


@pytest.mark.parametrize("direct", [False, True])
def test_paged_rows_match_file_across_page_boundaries_and_partial_last_page(direct):
    with tempfile.TemporaryDirectory() as directory:
        path, table = _write_rows(directory, "ple.bin", 1000, 160, seed=1)
        assert os.path.getsize(path) % PAGE != 0
        source = PagedRowSource(_reader(), path, 160, 1000, direct=direct)
        rows = torch.tensor([25, 999, 0, 25, 26, 700, 998, 1, 25])
        assert (25 * 160) // PAGE != (26 * 160 - 1) // PAGE
        destination = torch.zeros(rows.numel() + 3, 160, dtype=torch.uint8)

        source.read_rows(rows, destination)

        assert torch.equal(destination[: rows.numel()], table[rows])
        assert torch.equal(
            destination[rows.numel() :], torch.zeros(3, 160, dtype=torch.uint8)
        )
        first_bounce = source.bounce_bytes
        source.read_rows(rows[:2], destination)
        assert source.bounce_bytes == first_bounce
        assert torch.equal(destination[:2], table[rows[:2]])


def test_paged_rows_reject_out_of_range_ids_and_small_destinations():
    with tempfile.TemporaryDirectory() as directory:
        path, _ = _write_rows(directory, "ple.bin", 10, 7, seed=2)
        source = PagedRowSource(_reader(), path, 7, 10, direct=False)
        with pytest.raises(IndexError):
            source.read_rows(torch.tensor([10]), torch.zeros(1, 7, dtype=torch.uint8))
        with pytest.raises(ValueError, match="must hold"):
            source.read_rows(torch.tensor([1, 2]), torch.zeros(1, 7, dtype=torch.uint8))
        with pytest.raises(ValueError, match="fewer than"):
            PagedRowSource(_reader(), path, 7, 11, direct=False)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("aligned", [False, True])
def test_aligned_rows_scatter_into_destination_rows(direct, aligned):
    with tempfile.TemporaryDirectory() as directory:
        reader = _reader()
        wide_path, wide = _write_rows(directory, "wide.bin", 6, 2 * PAGE, seed=3)
        narrow_path, narrow = _write_rows(directory, "narrow.bin", 6, PAGE, seed=4)
        wide_source = AlignedRowSource(reader, wide_path, 2 * PAGE, 6, direct=direct)
        narrow_source = AlignedRowSource(reader, narrow_path, PAGE, 6, direct=direct)
        if aligned:
            _, wide_destination = _aligned_rows(4, 2 * PAGE)
            _, narrow_destination = _aligned_rows(4, PAGE)
        else:
            wide_destination = torch.zeros(4, 2 * PAGE + 1, dtype=torch.uint8)[:, 1:]
            wide_destination = wide_destination.contiguous()
            narrow_storage = torch.zeros(4 * PAGE + 1, dtype=torch.uint8)
            narrow_destination = narrow_storage[1:].view(4, PAGE)
        rows = torch.tensor([5, 0, 3])
        slots = torch.tensor([2, 0, 3])

        read_plans(
            reader,
            [
                wide_source.plan(rows, wide_destination, slots),
                narrow_source.plan(rows, narrow_destination, slots),
            ],
        )

        assert torch.equal(wide_destination[slots], wide[rows])
        assert torch.equal(narrow_destination[slots], narrow[rows])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("direct", [False, True])
def test_row_sources_stay_on_cpu_under_a_cuda_default_device(direct):
    """Model construction and forwards run under ``with torch.device("cuda")``."""
    with tempfile.TemporaryDirectory() as directory:
        reader = _reader()
        ple_path, ple = _write_rows(directory, "ple.bin", 1000, 160, seed=6)
        wide_path, wide = _write_rows(directory, "wide.bin", 4, PAGE, seed=7)
        rows = torch.tensor([999, 25, 26])
        paged_destination = torch.zeros(3, 160, dtype=torch.uint8)
        _, aligned_destination = _aligned_rows(3, PAGE)

        with torch.device("cuda"):
            paged = PagedRowSource(reader, ple_path, 160, 1000, direct=direct)
            aligned = AlignedRowSource(reader, wide_path, PAGE, 4, direct=direct)
            paged.read_rows(rows, paged_destination)
            read_plans(
                reader,
                [
                    aligned.plan(
                        torch.tensor([3, 0, 1], device="cpu"), aligned_destination
                    )
                ],
            )

        assert torch.equal(paged_destination, ple[rows])
        assert torch.equal(aligned_destination, wide[[3, 0, 1]])


@pytest.mark.parametrize("direct", [False, True])
def test_paged_row_batch_matches_per_source_reads_across_files_and_windows(direct):
    with tempfile.TemporaryDirectory() as directory:
        reader = _reader()
        wide_path, wide = _write_rows(directory, "wide.bin", 1000, 160, seed=8)
        narrow_path, narrow = _write_rows(directory, "narrow.bin", 700, 7, seed=9)
        sources = [
            PagedRowSource(reader, wide_path, 160, 1000, direct=direct),
            PagedRowSource(reader, narrow_path, 7, 700, direct=direct),
            PagedRowSource(reader, wide_path, 160, 1000, direct=direct),
        ]
        tables = [wide, narrow, wide]
        windows = [(0, 1000), (50, 750), (1000, 1600)]
        batch = PagedRowBatch(sources, windows)
        generator = torch.Generator().manual_seed(10)

        def expected(table, ids, window):
            start, end = window
            inside = (ids >= start) & (ids < end)
            rows = table[torch.where(inside, ids - start, 0)].clone()
            rows[~inside] = 0
            return rows

        destinations = {}
        for counts in ([9, 12, 7], [9, 12, 7], [0, 3, 1], [2, 0, 0]):
            per_source = []
            for count, (start, end) in zip(counts, windows):
                ids = torch.randint(start - 40, end + 40, (count,), generator=generator)
                if count >= 4:
                    ids[:4] = torch.tensor([start + 25, start + 26, start + 25, end - 1])
                    ids[-1] = -1
                per_source.append(ids)
            ids = torch.cat(per_source)
            total = sum(c * rb for c, rb in zip(counts, batch.row_bytes))
            destination = destinations.setdefault(
                total, torch.full((total + 5,), 0xAB, dtype=torch.uint8)
            )

            batch.read_rows(ids, counts, destination)

            offset = 0
            for source_ids, source, table, window, row_bytes in zip(
                per_source, sources, tables, windows, batch.row_bytes
            ):
                end = offset + source_ids.numel() * row_bytes
                reference = expected(table, source_ids, window)
                assert torch.equal(destination[offset:end].view(-1, row_bytes), reference)
                inside = (source_ids >= window[0]) & (source_ids < window[1])
                paged = torch.zeros(source_ids.numel(), row_bytes, dtype=torch.uint8)
                source.read_rows(
                    torch.where(inside, source_ids - window[0], 0), paged
                )
                paged[~inside] = 0
                assert torch.equal(destination[offset:end].view(-1, row_bytes), paged)
                offset = end
            assert torch.equal(destination[total:], torch.full((5,), 0xAB, dtype=torch.uint8))

        with pytest.raises(Exception, match="file row id outside"):
            PagedRowBatch([sources[1]], [(0, 701)]).read_rows(
                torch.tensor([700]), [1], torch.zeros(7, dtype=torch.uint8)
            )
        with pytest.raises(ValueError, match="expected 3 ids"):
            batch.read_rows(torch.tensor([1, 2]), [1, 1, 1], torch.zeros(400, dtype=torch.uint8))
        with pytest.raises(ValueError, match="must hold"):
            batch.read_rows(torch.tensor([1]), [1, 0, 0], torch.zeros(100, dtype=torch.uint8))


def test_paged_row_batch_rejects_a_source_with_a_base_offset():
    with tempfile.TemporaryDirectory() as directory:
        path, _ = _write_rows(directory, "rows.bin", 50, 8, seed=6)
        source = PagedRowSource(_reader(), path, 8, 40, direct=False, base_offset=64)
        with pytest.raises(ValueError, match="byte 0"):
            PagedRowBatch([source], [(0, 40)])


def test_aligned_rows_reject_wrong_width_and_mismatched_slots():
    with tempfile.TemporaryDirectory() as directory:
        path, _ = _write_rows(directory, "rows.bin", 3, PAGE, seed=5)
        source = AlignedRowSource(_reader(), path, PAGE, 3, direct=False)
        with pytest.raises(ValueError, match="expected 4096"):
            source.plan(torch.tensor([0]), torch.zeros(1, 8, dtype=torch.uint8))
        with pytest.raises(ValueError, match="must match in length"):
            source.plan(
                torch.tensor([0, 1]),
                torch.zeros(2, PAGE, dtype=torch.uint8),
                torch.tensor([0]),
            )
