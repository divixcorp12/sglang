import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.layers.moe.expert_prediction.capture_frames import CaptureFrame, FramePool
from sglang.srt.layers.moe.expert_prediction.capture_reader import (
    check_capture,
    load_shard,
    read_manifest,
)
from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    CaptureKind,
    ForwardRecord,
)
from sglang.srt.layers.moe.expert_prediction.capture_writer import (
    STOPPED_NAME,
    PendingForward,
    ShardWriter,
)
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore

SPECS = [
    MoeLayerSpec(layer_id=0, num_experts=8, top_k=2, hidden_size=4),
    MoeLayerSpec(layer_id=1, num_experts=8, top_k=2, hidden_size=4),
]


def _pool():
    return FramePool(
        [
            CaptureFrame(specs=SPECS, capacity=8, hidden_dtype=torch.float32, pin_memory=False)
            for _ in range(2)
        ]
    )


def _writer(directory, pool, **overrides):
    settings = dict(shard_rows=4, max_bytes=1 << 30, idle_flush_s=30.0)
    settings.update(overrides)
    return ShardWriter(
        directory=directory, specs=SPECS, hidden_dtype=torch.float32, pool=pool, **settings
    )


def _submit(writer, pool, *, index, kind, rid, positions, token_ids):
    frame = pool.acquire(timeout=5)
    rows = len(positions)
    frame.positions[:rows] = torch.tensor(positions)
    frame.token_ids[:rows] = torch.tensor(token_ids)
    for (layer_id, feature), tensor in frame.features.items():
        if feature is RouteFeature.TOPK_IDS:
            tensor[:rows] = torch.tensor(positions).unsqueeze(1) % 8
        else:
            tensor[:rows] = torch.tensor(positions, dtype=tensor.dtype).unsqueeze(1)
    writer.submit(
        PendingForward(
            record=ForwardRecord(
                forward_index=index, kind=kind, rids=(rid,), rows_per_request=(rows,)
            ),
            frame=frame,
        )
    )


class TestShardWriter(unittest.TestCase):
    def test_frame_pool_exhaustion_is_observable_without_waiting(self):
        pool = _pool()
        first = pool.try_acquire()
        second = pool.try_acquire()
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNone(pool.try_acquire())
        pool.release(first)
        pool.release(second)

    def test_stop_defers_stop_artifact_io_to_the_writer_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            writer = _writer(directory, _pool())
            with mock.patch.object(Path, "write_text") as write_text:
                writer.stop("test backpressure")
                self.assertEqual(write_text.call_count, 0)
            writer.close()
            self.assertIn("test backpressure", (directory / STOPPED_NAME).read_text())

    def test_round_trip_through_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool)
            _submit(writer, pool, index=0, kind=CaptureKind.PREFILL, rid="a",
                    positions=[0, 1, 2], token_ids=[1, 2, 3])
            _submit(writer, pool, index=1, kind=CaptureKind.DECODE, rid="a",
                    positions=[3], token_ids=[4])
            writer.close()
            report = check_capture(directory)
            self.assertEqual((report.rows, report.prefill_rows, report.decode_rows), (4, 3, 1))
            self.assertEqual(report.violations, [])
            shard = load_shard(directory, read_manifest(directory)[0]["shard"])
            self.assertEqual(shard.request_ids, ("a",))
            self.assertEqual(
                shard.tensors["layer.0.router_input"][:, 0].tolist(), [0.0, 1.0, 2.0, 3.0]
            )
            self.assertEqual(shard.tensors["layer.1.topk_ids"].dtype, torch.int16)
            self.assertEqual(tuple(shard.tensors["forward.expert_to_slot"].shape), (2, 2, 8))

    def test_reprefilled_prefix_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool, shard_rows=100)
            _submit(writer, pool, index=0, kind=CaptureKind.PREFILL, rid="a",
                    positions=[0, 1, 2], token_ids=[1, 2, 3])
            _submit(writer, pool, index=1, kind=CaptureKind.DECODE, rid="a",
                    positions=[3], token_ids=[4])
            _submit(writer, pool, index=2, kind=CaptureKind.PREFILL, rid="b",
                    positions=[0, 1, 2, 3, 4], token_ids=[1, 2, 3, 4, 5])
            writer.close()
            report = check_capture(directory)
            self.assertEqual((report.rows, report.ran_rows, report.forwards), (5, 9, 3))
            shard = load_shard(directory, read_manifest(directory)[0]["shard"])
            self.assertEqual(shard.tensors["row.position"].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(shard.tensors["row.request_index"].tolist(), [0, 0, 0, 0, 1])

    def test_shard_with_every_row_deduplicated_still_records_its_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool, shard_rows=1)
            _submit(writer, pool, index=0, kind=CaptureKind.PREFILL, rid="a",
                    positions=[0, 1], token_ids=[1, 2])
            _submit(writer, pool, index=1, kind=CaptureKind.PREFILL, rid="b",
                    positions=[0, 1], token_ids=[1, 2])
            writer.close()
            report = check_capture(directory)
            self.assertEqual(
                (report.shards, report.rows, report.forwards, report.ran_rows), (2, 2, 2, 4)
            )
            self.assertEqual(report.violations, [])

    def test_byte_cap_stops_instead_of_thinning(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool, shard_rows=1, max_bytes=1)
            for index in range(3):
                _submit(writer, pool, index=index, kind=CaptureKind.DECODE, rid="a",
                        positions=[index], token_ids=[index])
            writer.close()
            self.assertTrue(writer.stopped.is_set())
            self.assertEqual(read_manifest(directory), [])
            self.assertIn("byte cap", check_capture(directory).stopped_reason)
            self.assertTrue((directory / STOPPED_NAME).exists())
            pool.acquire(timeout=1)
            pool.acquire(timeout=1)

    def test_idle_flush_writes_a_partial_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool, shard_rows=100, idle_flush_s=0.05)
            _submit(writer, pool, index=0, kind=CaptureKind.DECODE, rid="a",
                    positions=[0], token_ids=[7])
            deadline = time.monotonic() + 5
            while not read_manifest(directory) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(read_manifest(directory)[0]["rows"], 1)
            writer.close()

    def test_refuses_non_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "leftover").write_text("x")
            with self.assertRaisesRegex(ValueError, "not empty"):
                _writer(Path(tmp), _pool())


class TestFeatureStoreSpill(unittest.TestCase):
    def test_oversized_batches_go_to_spill_with_routed_columns_only(self):
        store = FeatureStore(
            specs=SPECS, features=[RouteFeature.TOPK_IDS], max_rows=2,
            device=torch.device("cpu"), hidden_dtype=torch.float32,
        )
        calls = []
        store.spill = lambda layer_id, feature, rows: calls.append((layer_id, feature, rows))
        store.write(0, RouteFeature.TOPK_IDS, torch.arange(9).reshape(3, 3))
        store.write(0, RouteFeature.TOPK_IDS, torch.arange(6).reshape(2, 3))
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0][2].shape), (3, 2))
        self.assertEqual(store.view(0, RouteFeature.TOPK_IDS, 2).tolist(), [[0, 1], [3, 4]])


if __name__ == "__main__":
    unittest.main()
