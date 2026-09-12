import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

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
