"""CPU tests for expert host row sources and how the streamer routes host reads."""

import os
import tempfile
import unittest

import torch

from sglang.srt.layers.moe.expert_row_source import (
    CompletedReadTicket,
    ExpertRowSource,
    HostSlotLayout,
    ReadTicket,
    RowReadStats,
    TensorRowSource,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import CountingRowSource

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4096
HAS_URING = os.path.exists("/usr/include/liburing.h")


class TestRowReadStats(unittest.TestCase):
    def test_stats_add_field_by_field(self):
        total = RowReadStats(1, 2, 3, 4, 5) + RowReadStats(10, 20, 30, 40, 50)
        self.assertEqual(total, RowReadStats(11, 22, 33, 44, 55))
        self.assertEqual(RowReadStats() + RowReadStats(rows=2), RowReadStats(rows=2))


class TestCompletedReadTicket(unittest.TestCase):
    def test_synchronous_submit_returns_a_done_ticket(self):
        rows = torch.arange(12, dtype=torch.uint8).reshape(4, 3)
        source = CountingRowSource({"rows": rows})
        destination = torch.zeros(2, 3, dtype=torch.uint8)
        ticket = source.submit(torch.tensor([3, 1]), {"rows": destination})
        self.assertIsInstance(ticket, ReadTicket)
        self.assertTrue(ticket.done())
        self.assertEqual(ticket.wait().rows, 2)
        self.assertTrue(torch.equal(destination, rows[[3, 1]]))
        self.assertIsNone(source.calls[0].destination_rows)

    def test_a_failed_read_raises_from_wait(self):
        source = CountingRowSource({"rows": torch.zeros(4, 3)})
        ticket = source.submit(torch.tensor([0]), {"missing": torch.zeros(1, 3)})
        self.assertTrue(ticket.done())
        with self.assertRaisesRegex(ValueError, "does not cover"):
            ticket.wait()

    def test_a_ticket_holds_stats_or_an_error(self):
        with self.assertRaises(ValueError):
            CompletedReadTicket()
        with self.assertRaises(ValueError):
            CompletedReadTicket(RowReadStats(), RuntimeError("both"))


class TestTensorRowSource(unittest.TestCase):
    def _source(self):
        tensors = {
            "a": torch.arange(24, dtype=torch.int16).reshape(6, 4),
            "b": torch.arange(6, dtype=torch.float32),
        }
        return tensors, TensorRowSource(tensors.get, ("a", "b"), 6)

    def test_reads_rows_into_destination_slots(self):
        tensors, source = self._source()
        destinations = {
            "a": torch.zeros(4, 4, dtype=torch.int16),
            "b": torch.zeros(4, dtype=torch.float32),
        }
        stats = source.read(torch.tensor([5, 0, 3]), destinations, torch.tensor([2, 0, 3]))
        for name, tensor in tensors.items():
            self.assertTrue(
                torch.equal(destinations[name][[2, 0, 3]], tensor[[5, 0, 3]]), name
            )
            self.assertEqual(destinations[name][1].abs().sum().item(), 0)
        self.assertEqual(stats.rows, 3)
        self.assertEqual(stats.file_bytes, 0)
        self.assertGreaterEqual(stats.read_ns, 0)

    def test_reads_rows_into_leading_rows_without_slots(self):
        tensors, source = self._source()
        destination = torch.zeros(5, 4, dtype=torch.int16)
        source.read(torch.tensor([4, 1]), {"a": destination})
        self.assertTrue(torch.equal(destination[:2], tensors["a"][[4, 1]]))
        self.assertEqual(destination[2:].abs().sum().item(), 0)

    def test_covers_only_names_with_a_dense_source(self):
        tensor = torch.zeros(6, 2)
        source = TensorRowSource({"a": tensor, "b": None}.get, ("a", "b", "c"), 6)
        self.assertTrue(source.covers("a"))
        self.assertFalse(source.covers("b"))
        self.assertFalse(source.covers("c"))
        self.assertFalse(source.covers("d"))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            source.read(torch.tensor([0]), {"b": torch.zeros(1, 2)})

    def test_conforms_to_the_row_source_protocol(self):
        _, source = self._source()
        self.assertIsInstance(source, ExpertRowSource)
        self.assertEqual(source.host_layouts, frozenset({HostSlotLayout.PER_NAME}))
        self.assertEqual(source.register_destinations([torch.zeros(2)]), 0)
        self.assertEqual(source.num_experts, 6)
        self.assertEqual(source.file_bytes_per_expert, 0)
        self.assertIsInstance(source.submit(torch.tensor([1]), {"b": torch.zeros(1)}), CompletedReadTicket)


def _file_group(directory):
    from sglang.srt.model_loader.file_tensor_cache import (
        FileTensorCacheGroup,
        FileTensorSpec,
    )

    specs = (
        FileTensorSpec("w13_trellis", (6, 2, PAGE), (2 * PAGE, PAGE, 1), torch.uint8),
        FileTensorSpec("w2_svh", (6, 40), (40, 1), torch.float16),
    )
    group = FileTensorCacheGroup.open(directory, "row_source_test", {"k": 1}, specs)
    generator = torch.Generator().manual_seed(17)
    for spec in specs:
        target = group.tensors[spec.tag].view(torch.uint8)
        target.copy_(
            torch.randint(0, 256, target.shape, dtype=torch.uint8, generator=generator)
        )
    return group


@unittest.skipUnless(HAS_URING, "io_uring reads need liburing")
class TestExpertFileRowReaderRowSource(unittest.TestCase):
    def test_from_group_reads_named_members_into_slots(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        for mode in ("uring", "uring_direct"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                group = _file_group(directory)
                try:
                    reader = ExpertFileRowReader.from_group(group, mode=mode)
                    self.assertIsInstance(reader, ExpertRowSource)
                    self.assertEqual(reader.names, ("w13_trellis", "w2_svh"))
                    self.assertEqual(reader.num_experts, 6)
                    self.assertEqual(reader.file_bytes_per_expert, 2 * PAGE + 80)
                    destinations = {
                        name: torch.zeros(
                            (4,) + tuple(group.tensors[name].shape[1:]),
                            dtype=group.tensors[name].dtype,
                        )
                        for name in reader.names
                    }
                    rows = torch.tensor([5, 0, 3])
                    slots = torch.tensor([1, 3, 0])
                    stats = reader.read(rows, destinations, slots)
                    for name in reader.names:
                        self.assertTrue(
                            torch.equal(
                                destinations[name][slots].view(torch.uint8),
                                group.tensors[name][rows].view(torch.uint8),
                            ),
                            name,
                        )
                    self.assertEqual(stats.rows, 3)
                    self.assertEqual(stats.file_bytes, 3 * (2 * PAGE + 80))
                    self.assertGreater(stats.read_ns, 0)
                    reader.close()
                finally:
                    group.close()

    def test_from_group_can_cover_a_subset_of_members(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        with tempfile.TemporaryDirectory() as directory:
            group = _file_group(directory)
            try:
                reader = ExpertFileRowReader.from_group(group, ("w2_svh",), mode="uring")
                self.assertEqual(reader.names, ("w2_svh",))
                self.assertFalse(reader.covers("w13_trellis"))
                self.assertEqual(reader.file_bytes_per_expert, 80)
            finally:
                group.close()

    def test_from_group_refuses_mmap_and_unknown_members(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        with tempfile.TemporaryDirectory() as directory:
            group = _file_group(directory)
            try:
                with self.assertRaisesRegex(ValueError, "TensorRowSource"):
                    ExpertFileRowReader.from_group(group, mode="mmap")
                with self.assertRaisesRegex(ValueError, "no member 'w2_suh'"):
                    ExpertFileRowReader.from_group(group, ("w2_suh",), mode="uring")
            finally:
                group.close()

    def test_from_layer_reader_reports_its_file_bytes(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        with tempfile.TemporaryDirectory() as directory:
            group = _file_group(directory)
            try:
                layer = torch.nn.Module()
                for name, tensor in group.tensors.items():
                    parameter = torch.nn.Parameter(tensor, requires_grad=False)
                    parameter._sglang_file_cache_group = group
                    parameter._sglang_file_cache_tag = name
                    setattr(layer, name, parameter)
                reader = ExpertFileRowReader.from_layer(
                    layer, ("w13_trellis", "w2_svh"), mode="uring"
                )
                self.assertEqual(reader.file_bytes_per_expert, 2 * PAGE + 80)
                self.assertEqual(reader.num_experts, 6)
                stats = reader.read(
                    torch.tensor([2]),
                    {"w2_svh": torch.zeros(1, 40, dtype=torch.float16)},
                )
                self.assertEqual((stats.rows, stats.file_bytes), (1, 80))
            finally:
                group.close()


if __name__ == "__main__":
    unittest.main()
