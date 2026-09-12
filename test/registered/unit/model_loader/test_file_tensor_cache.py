import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.model_loader import file_tensor_cache as file_tensor_cache_module
from sglang.srt.model_loader.file_tensor_cache import (
    FileTensorCacheGroup,
    FileTensorSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFileTensorSpec(unittest.TestCase):
    def test_nbytes_covers_the_requested_strided_storage(self):
        spec = FileTensorSpec("weight", (2, 3), (4, 1), torch.uint8)
        self.assertEqual(spec.nbytes, 7)


class TestFileTensorCacheGroup(unittest.TestCase):
    def _open(self, directory, specs=None, identity=None):
        return FileTensorCacheGroup.open(
            directory,
            "expert",
            identity or {"commit": "abc"},
            specs or [FileTensorSpec("weight", (4, 8), (8, 1), torch.uint8)],
        )

    def test_completed_group_reopens_as_hit_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = [FileTensorSpec("weight", (4, 8), (8, 1), torch.uint8)]
            cold = self._open(directory, specs)
            cold.tensors["weight"].fill_(17)
            cold.complete()

            warm = self._open(directory, specs)
            try:
                self.assertTrue(warm.cache_hit)
                self.assertTrue(
                    torch.equal(
                        warm.tensors["weight"],
                        torch.full((4, 8), 17, dtype=torch.uint8),
                    )
                )
            finally:
                warm.close()

    def test_logs_one_group_lifecycle_with_verified_warm_hit(self):
        specs = [
            FileTensorSpec("weight", (4, 8), (8, 1), torch.uint8),
            FileTensorSpec("scale", (4, 2), (2, 1), torch.float8_e4m3fn),
        ]
        identity = {"checkpoint": "/private/checkpoints/customer-model"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertLogs(
                file_tensor_cache_module.__name__, level="INFO"
            ) as logs:
                cold = self._open(directory, specs, identity)
                self.assertFalse(cold.cache_hit)
                self.assertFalse(os.path.exists(cold.manifest_path))
                cold.tensors["weight"].fill_(17)
                cold.complete()
                cold.complete()
                self.assertFalse(
                    any("outcome=verified_hit" in entry for entry in logs.output)
                )
                warm = self._open(directory, specs, identity)
                self.assertTrue(warm.cache_hit)
                self.assertEqual(warm.tensors["weight"].sum().item(), 544)
                warm.close()
            messages = [record.getMessage() for record in logs.records]
            self.assertEqual(len(messages), 3)
            for message, outcome in zip(
                messages, ("building_miss", "published_completed_cache", "verified_hit")
            ):
                self.assertIn(f"outcome={outcome}", message)
                self.assertIn("namespace=expert", message)
                self.assertIn("members=2 bytes=40", message)
                self.assertRegex(message, r"key=[0-9a-f]{12}\b")
                self.assertNotIn(directory, message)
                self.assertNotIn(identity["checkpoint"], message)
            self.assertEqual(
                len({message.split("key=")[1].split()[0] for message in messages}), 1
            )

    def test_abort_logs_invalidation_once_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertLogs(
                file_tensor_cache_module.__name__, level="INFO"
            ) as logs:
                cache = self._open(directory)
                cache.abort()
                cache.abort()
            self.assertEqual(len(logs.records), 2)
            self.assertIn("outcome=aborted_invalidated", logs.output[-1])
            self.assertIn("members=1 bytes=32", logs.output[-1])
            self.assertFalse(os.path.exists(cache.manifest_path))
            self.assertIsNone(cache._lock)

    def test_invalid_cache_logs_reason_without_a_false_hit(self):
        cases = (
            "manifest_unreadable",
            "manifest_mismatch",
            "member_size_mismatch",
            "member_unavailable",
        )
        for reason in cases:
            with (
                self.subTest(reason=reason),
                tempfile.TemporaryDirectory() as directory,
            ):
                cache = self._open(directory)
                cache.complete()
                path = Path(cache.manifest_path)
                if reason == "manifest_unreadable":
                    path.write_text("{not-json", encoding="utf-8")
                elif reason == "manifest_mismatch":
                    manifest = json.loads(path.read_text())
                    manifest["cache_identity"] = {"private_path": "/secret/model"}
                    path.write_text(json.dumps(manifest))
                elif reason == "member_size_mismatch":
                    os.truncate(cache.paths["weight"], 1)
                else:
                    os.unlink(cache.paths["weight"])
                with self.assertLogs(
                    file_tensor_cache_module.__name__, level="DEBUG"
                ) as logs:
                    rebuilt = self._open(directory)
                    self.assertFalse(rebuilt.cache_hit)
                    rebuilt.abort()
                self.assertTrue(
                    any(f"reason={reason}" in message for message in logs.output)
                )
                self.assertFalse(
                    any("outcome=verified_hit" in message for message in logs.output)
                )
                info = [
                    record.getMessage()
                    for record in logs.records
                    if record.levelno == logging.INFO
                ]
                self.assertEqual(len(info), 2)
                self.assertIn("outcome=building_miss", info[0])
                self.assertIn("outcome=aborted_invalidated", info[1])
                self.assertFalse(
                    any(
                        "/secret/model" in message or directory in message
                        for message in logs.output
                    )
                )

    def test_failed_publication_never_logs_completed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertLogs(
                file_tensor_cache_module.__name__, level="INFO"
            ) as logs:
                cache = self._open(directory)
                with mock.patch.object(
                    file_tensor_cache_module,
                    "_fsync_directory",
                    side_effect=OSError("private path"),
                ):
                    with self.assertRaises(OSError):
                        cache.complete()
            self.assertFalse(
                any(
                    "outcome=published_completed_cache" in message
                    for message in logs.output
                )
            )
            self.assertTrue(
                any("outcome=publication_failed" in message for message in logs.output)
            )
            self.assertIsNone(cache._lock)
            self.assertFalse(any("private path" in message for message in logs.output))

    def test_mapping_failure_does_not_log_a_verified_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self._open(directory)
            cache.complete()
            with self.assertLogs(
                file_tensor_cache_module.__name__, level="INFO"
            ) as logs:
                with mock.patch.object(
                    file_tensor_cache_module,
                    "_map_tensor",
                    side_effect=OSError("private path"),
                ):
                    with self.assertRaises(OSError):
                        self._open(directory)
            self.assertFalse(
                any("outcome=verified_hit" in message for message in logs.output)
            )
            self.assertTrue(
                any("outcome=open_failed" in message for message in logs.output)
            )
            reopened = self._open(directory)
            self.assertTrue(reopened.cache_hit)
            reopened.close()

    def test_failed_abort_logs_failure_without_claiming_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            cold = self._open(directory)
            cold.complete()
            cache = self._open(directory)
            with self.assertLogs(
                file_tensor_cache_module.__name__, level="WARNING"
            ) as logs:
                with mock.patch.object(
                    file_tensor_cache_module,
                    "_invalidate_manifest",
                    side_effect=OSError("private path"),
                ):
                    with self.assertRaises(OSError):
                        cache.abort()
            self.assertIsNone(cache._lock)
            self.assertTrue(os.path.exists(cache.manifest_path))
            self.assertEqual(len(logs.records), 1)
            self.assertIn("outcome=invalidation_failed", logs.output[0])
            self.assertNotIn("outcome=aborted_invalidated", logs.output[0])
            self.assertNotIn("private path", logs.output[0])

    def test_completed_hit_flushes_and_republishes_after_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            cold = self._open(directory)
            cold.tensors["weight"].fill_(17)
            cold.complete()

            warm = self._open(directory)
            self.assertTrue(warm.cache_hit)
            original_manifest_inode = os.stat(warm.manifest_path).st_ino
            warm.tensors["weight"].fill_(29)
            with mock.patch.object(
                file_tensor_cache_module,
                "_fsync_file",
                wraps=file_tensor_cache_module._fsync_file,
            ) as fsync_file:
                warm.complete()
            fsync_file.assert_called_once_with(warm.paths["weight"])
            self.assertTrue(os.path.isfile(warm.manifest_path))
            self.assertNotEqual(
                os.stat(warm.manifest_path).st_ino, original_manifest_inode
            )

            reopened = self._open(directory)
            try:
                self.assertTrue(reopened.cache_hit)
                self.assertTrue(
                    torch.equal(
                        reopened.tensors["weight"],
                        torch.full((4, 8), 29, dtype=torch.uint8),
                    )
                )
            finally:
                reopened.close()

    def test_strided_tensor_reopens_with_shape_stride_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = FileTensorSpec("weight", (2, 3), (4, 1), torch.uint8)
            cold = self._open(directory, [spec])
            cold.tensors["weight"].copy_(torch.tensor([[1, 2, 3], [4, 5, 6]]))
            cold.complete()

            warm = self._open(directory, [spec])
            try:
                self.assertTrue(warm.cache_hit)
                self.assertEqual(tuple(warm.tensors["weight"].shape), (2, 3))
                self.assertEqual(tuple(warm.tensors["weight"].stride()), (4, 1))
                self.assertTrue(
                    torch.equal(
                        warm.tensors["weight"],
                        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8),
                    )
                )
            finally:
                warm.close()

    def test_group_is_all_miss_when_one_member_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = [
                FileTensorSpec("weight", (4, 8), (8, 1), torch.uint8),
                FileTensorSpec("scale", (4, 2), (2, 1), torch.float8_e4m3fn),
            ]
            cold = self._open(directory, specs)
            cold.tensors["weight"].fill_(17)
            cold.complete()
            os.truncate(cold.paths["scale"], 1)

            reopened = self._open(directory, specs)
            try:
                self.assertFalse(reopened.cache_hit)
                self.assertTrue(torch.count_nonzero(reopened.tensors["weight"]) == 0)
                self.assertEqual(os.path.getsize(reopened.paths["scale"]), 8)
            finally:
                reopened.abort()

    def test_same_sized_files_without_manifest_are_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            cold = self._open(directory)
            cold.tensors["weight"].fill_(9)
            cold.close()

            reopened = self._open(directory)
            try:
                self.assertFalse(reopened.cache_hit)
                self.assertTrue(torch.count_nonzero(reopened.tensors["weight"]) == 0)
            finally:
                reopened.abort()

    def test_malformed_manifest_is_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            cold = self._open(directory)
            cold.complete()
            Path(cold.manifest_path).write_text("{not-json", encoding="utf-8")

            reopened = self._open(directory)
            try:
                self.assertFalse(reopened.cache_hit)
            finally:
                reopened.abort()

    def test_every_manifest_contract_field_must_match_exactly(self):
        mutations = {
            "format version": lambda manifest: manifest.__setitem__(
                "format_version", manifest["format_version"] + 1
            ),
            "identity": lambda manifest: manifest.__setitem__(
                "cache_identity", {"commit": "different"}
            ),
            "shape": lambda manifest: manifest["tensors"][0].__setitem__(
                "shape", [2, 16]
            ),
            "stride": lambda manifest: manifest["tensors"][0].__setitem__(
                "stride", [1, 4]
            ),
            "dtype": lambda manifest: manifest["tensors"][0].__setitem__(
                "dtype", "torch.int8"
            ),
            "nbytes": lambda manifest: manifest["tensors"][0].__setitem__("nbytes", 31),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                cold = self._open(directory)
                cold.complete()
                manifest_path = Path(cold.manifest_path)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                reopened = self._open(directory)
                try:
                    self.assertFalse(reopened.cache_hit)
                finally:
                    reopened.abort()

    def test_identity_change_uses_a_distinct_cold_group(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._open(directory, identity={"commit": "abc"})
            first.complete()
            second = self._open(directory, identity={"commit": "def"})
            try:
                self.assertFalse(second.cache_hit)
                self.assertNotEqual(first.manifest_path, second.manifest_path)
                self.assertNotEqual(first.paths["weight"], second.paths["weight"])
            finally:
                second.abort()

    def test_close_is_idempotent_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self._open(directory)
            cache.close()
            cache.close()
            self.assertFalse(os.path.exists(cache.manifest_path))

    def test_abort_removes_manifest_and_rebuilds_next_open(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self._open(directory)
            cache.tensors["weight"].fill_(23)
            cache.abort()
            self.assertFalse(os.path.exists(cache.manifest_path))

            reopened = self._open(directory)
            try:
                self.assertFalse(reopened.cache_hit)
                self.assertTrue(torch.count_nonzero(reopened.tensors["weight"]) == 0)
            finally:
                reopened.abort()

    def test_atomic_completion_leaves_only_the_final_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self._open(directory)
            cache.complete()
            manifest_path = Path(cache.manifest_path)
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(
                list(manifest_path.parent.glob(f"{manifest_path.name}.tmp*")), []
            )


if __name__ == "__main__":
    unittest.main()
