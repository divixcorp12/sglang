"""--weight-loader-drop-cache-after-load: a shard the consumer still maps when the iterator drops it stays cached, so
DefaultModelLoader.load_model drops every loaded shard again once the model is built (CPU; the DSpark draft's one
7.2 GB shard stayed 4-5.6 GiB cached at ready, cores-ab-20261014/splitmodel-stream-mem-hicache1)."""

import ast
import inspect
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import sglang.srt.model_loader.loader as loader
from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    drop_files_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _cached(path: Path) -> int:
    out = subprocess.run(["fincore", "-b", "-n", "-r", "-o", "RES", str(path)], capture_output=True, text=True)
    return int(out.stdout.split()[0])


@pytest.fixture
def shard(tmp_path):
    if shutil.which("fincore") is None:
        pytest.skip("needs fincore (util-linux)")
    path = tmp_path / "model.safetensors"
    save_file({f"w{i}": torch.randn(1024, 1024) for i in range(8)}, str(path))  # 32 MiB
    # Written back first: POSIX_FADV_DONTNEED leaves dirty pages cached, so an unsynced shard can survive the drop
    # (it did under load on divix01, and always on the xfs root's /tmp).
    with open(path, "rb+") as f:
        os.fsync(f.fileno())
    path.read_bytes()  # into the page cache
    if _cached(path) == 0:
        pytest.skip("this filesystem keeps no page cache to drop (tmpfs?)")
    return path


def test_a_shard_still_mapped_at_the_iterators_drop_stays_cached_until_dropped_again(shard):
    held = None
    for _, tensor in buffered_multi_thread_safetensors_weights_iterator([str(shard)], 1, drop_cache_after_load=True):
        held = tensor  # a consumer's loop variable: it outlives the iterator's own drop
    assert _cached(shard) > (16 << 20), "the iterator's drop freed pages a live tensor still maps"
    del held, tensor
    drop_files_cache([str(shard)])
    assert _cached(shard) == 0


def test_load_model_drops_every_loaded_shard_after_postprocessing():
    tree = ast.parse(inspect.getsource(loader.DefaultModelLoader))
    fns = {n.name: ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    load = fns["load_model"]
    assert load.index("self.load_weights_and_postprocess(") < load.index(
        "drop_files_cache(self.__dict__.pop('_drop_cache_files', []))")
    assert "self._drop_cache_files = [*getattr(self, '_drop_cache_files', []), *hf_weights_files]" in \
        fns["_get_weights_iterator"]
