"""CPU tests for the verify-only open of a file tensor cache group."""

import os
import tempfile
import unittest

import torch

from sglang.srt.model_loader.file_tensor_cache import (
    FileTensorCacheGroup,
    FileTensorSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NAMESPACE = "expert_rows_test"
IDENTITY = {"layout": "rows_v1", "layer": 3}
SPECS = (
    FileTensorSpec("w13_trellis", (4, 2, 4096), (8192, 4096, 1), torch.uint8),
    FileTensorSpec("w2_svh", (4, 40), (40, 1), torch.float16),
)


def _stats(paths):
    return {path: (os.stat(path).st_ino, os.stat(path).st_mtime_ns) for path in paths}


def _completed_group(directory):
    group = FileTensorCacheGroup.open(directory, NAMESPACE, IDENTITY, SPECS)
    for index, tensor in enumerate(group.tensors.values()):
        tensor.view(torch.uint8).fill_(index + 7)
    manifest_path, paths = group.manifest_path, dict(group.paths)
    group.complete()
    return manifest_path, paths


class TestOpenVerified(unittest.TestCase):
    def test_a_completed_group_opens_without_touching_its_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, paths = _completed_group(directory)
            before = _stats(list(paths.values()) + [manifest_path])
            group = FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            try:
                self.assertTrue(group.cache_hit)
                self.assertEqual(group.paths, paths)
                self.assertTrue(
                    torch.all(group.tensors["w13_trellis"].view(torch.uint8) == 7)
                )
                self.assertTrue(torch.all(group.tensors["w2_svh"].view(torch.uint8) == 8))
            finally:
                group.close()
            self.assertEqual(_stats(list(paths.values()) + [manifest_path]), before)

    def test_an_unpublished_group_is_refused_and_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            group = FileTensorCacheGroup.open(directory, NAMESPACE, IDENTITY, SPECS)
            manifest_path, paths = group.manifest_path, dict(group.paths)
            group.close()
            before = _stats(paths.values())
            with self.assertRaisesRegex(ValueError, "no verified file tensor cache group"):
                FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            self.assertEqual(_stats(paths.values()), before)
            self.assertFalse(os.path.exists(manifest_path))

    def test_a_corrupt_manifest_is_refused_and_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, paths = _completed_group(directory)
            with open(manifest_path, "w", encoding="utf-8") as stream:
                stream.write("not json")
            before = _stats(list(paths.values()) + [manifest_path])
            with self.assertRaisesRegex(ValueError, "no verified file tensor cache group"):
                FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            self.assertEqual(_stats(list(paths.values()) + [manifest_path]), before)
            with open(manifest_path, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "not json")

    def test_a_different_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            _completed_group(directory)
            with self.assertRaisesRegex(ValueError, "no verified file tensor cache group"):
                FileTensorCacheGroup.open_verified(
                    directory, NAMESPACE, {**IDENTITY, "layer": 4}, SPECS
                )

    def test_a_missing_directory_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "absent")
            with self.assertRaises(FileNotFoundError):
                FileTensorCacheGroup.open_verified(missing, NAMESPACE, IDENTITY, SPECS)
            self.assertFalse(os.path.exists(missing))

    def test_a_verified_group_maps_read_only_files_privately(self):
        with tempfile.TemporaryDirectory() as directory:
            _, paths = _completed_group(directory)
            for path in paths.values():
                os.chmod(path, 0o444)
            try:
                group = FileTensorCacheGroup.open_verified(
                    directory, NAMESPACE, IDENTITY, SPECS
                )
                try:
                    group.tensors["w2_svh"].view(torch.uint8).fill_(1)
                finally:
                    group.close()
                with open(paths["w2_svh"], "rb") as stream:
                    self.assertEqual(set(stream.read()), {8})
            finally:
                for path in paths.values():
                    os.chmod(path, 0o644)

    def test_a_verified_group_cannot_be_published_or_invalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, _ = _completed_group(directory)
            group = FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            try:
                with self.assertRaisesRegex(RuntimeError, "verify-only"):
                    group.complete()
                with self.assertRaisesRegex(RuntimeError, "verify-only"):
                    group.abort()
            finally:
                group.close()
            self.assertTrue(os.path.exists(manifest_path))
            reopened = FileTensorCacheGroup.open_verified(
                directory, NAMESPACE, IDENTITY, SPECS
            )
            reopened.close()

    @unittest.skipUnless(
        os.path.exists("/usr/include/liburing.h"), "io_uring reads need liburing"
    )
    def test_the_file_row_reader_reads_a_verified_group(self):
        from sglang.srt.layers.moe.expert_file_reader import ExpertFileRowReader

        with tempfile.TemporaryDirectory() as directory:
            _completed_group(directory)
            group = FileTensorCacheGroup.open_verified(directory, NAMESPACE, IDENTITY, SPECS)
            try:
                reader = ExpertFileRowReader.from_group(group, mode="uring")
                destination = torch.zeros(2, 40, dtype=torch.float16)
                reader.read(torch.tensor([3, 1]), {"w2_svh": destination})
                self.assertTrue(torch.all(destination.view(torch.uint8) == 8))
            finally:
                group.close()


if __name__ == "__main__":
    unittest.main()
