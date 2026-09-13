import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers.moe.expert_dma import ExpertDMABackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestExpertDMABackend(unittest.TestCase):
    def setUp(self):
        self.source = torch.arange(6 * 3 * 4, dtype=torch.float32).reshape(6, 3, 4)
        self.source = self.source.pin_memory()
        self.destination = torch.full_like(self.source, -1, device="cuda")

    def test_copies_selected_pinned_rows_into_cuda_slots(self):
        backend = ExpertDMABackend()

        actual_backend = backend.copy_rows(
            self.source, self.destination, source_rows=[5, 1], destination_slots=[2, 4]
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(self.destination[2], self.source[5].cuda())
        torch.testing.assert_close(self.destination[4], self.source[1].cuda())
        self.assertTrue(torch.all(self.destination[0] == -1))
        self.assertEqual(actual_backend, backend.actual_backend)
        self.assertIn(actual_backend, {"dma", "fallback"})

    def test_uses_row_copy_fallback_when_aot_primitive_is_unavailable(self):
        backend = ExpertDMABackend()

        with patch(
            "sglang.srt.layers.moe.expert_dma.transfer_embedding_ranges_direct",
            None,
        ):
            actual_backend = backend.copy_rows(
                self.source,
                self.destination,
                source_rows=[3, 0],
                destination_slots=[1, 5],
            )
        torch.cuda.synchronize()

        self.assertEqual(actual_backend, "fallback")
        self.assertEqual(backend.actual_backend, "fallback")
        torch.testing.assert_close(self.destination[1], self.source[3].cuda())
        torch.testing.assert_close(self.destination[5], self.source[0].cuda())

    def test_rejects_unpinned_source_and_mismatched_row_sequences(self):
        backend = ExpertDMABackend()

        with self.assertRaisesRegex(ValueError, "pinned"):
            backend.copy_rows(
                self.source.clone(),
                self.destination,
                source_rows=[0],
                destination_slots=[0],
            )
        with self.assertRaisesRegex(ValueError, "same number"):
            backend.copy_rows(
                self.source,
                self.destination,
                source_rows=[0, 1],
                destination_slots=[0],
            )


if __name__ == "__main__":
    unittest.main()
