"""SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL: empty_cache before each prefill larger than any before it, else nothing.

A fake allocator counts empty_cache calls. Off, the worker's _forward_prefill is the original function; on, the
warm-up's sizes release before each new high-water mark only, later prompts at or below it never release, the batch
and arguments reach the original unchanged, and installing twice wraps once.
"""

from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache import prefill_cache_release as pcr
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WARMUP = (1, 5, 8, 16, 33, 64, 257, 400, 2048, 2049, 4096)


class FakeCuda:
    def __init__(self):
        self.reserved = 1000 << 20
        self.empties = 0

    def memory_reserved(self):
        return self.reserved

    def empty_cache(self):
        self.empties += 1
        self.reserved -= 1 << 20


def _worker():
    seen = []
    worker = SimpleNamespace()
    worker._forward_prefill = lambda batch, on_publish: seen.append((batch.extend_num_tokens, on_publish)) or "out"
    return worker, seen


def test_off_wraps_nothing():
    worker, _ = _worker()
    fn = worker._forward_prefill
    with envs.SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL.override(False):
        assert pcr.maybe_install(worker) is None
    assert worker._forward_prefill is fn


def test_on_releases_before_each_new_largest_prefill_only():
    worker, seen = _worker()
    cuda = FakeCuda()
    release = pcr.install(worker, pcr.LargerPrefillRelease(cuda=cuda))
    for n in WARMUP:
        assert worker._forward_prefill(SimpleNamespace(extend_num_tokens=n), "pub") == "out"
    assert cuda.empties == len(WARMUP) and release.largest == 4096
    for n in (4096, 16, 6, 1, 4096, 3000):  # serving after warm-up: never above the largest chunk
        worker._forward_prefill(SimpleNamespace(extend_num_tokens=n), None)
    assert cuda.empties == len(WARMUP)
    assert [s[0] for s in seen] == [*WARMUP, 4096, 16, 6, 1, 4096, 3000] and seen[0][1] == "pub"


def test_a_batch_without_a_token_count_passes_through():
    worker, seen = _worker()
    cuda = FakeCuda()
    pcr.install(worker, pcr.LargerPrefillRelease(cuda=cuda))
    worker._forward_prefill(SimpleNamespace(extend_num_tokens=None), None)
    assert cuda.empties == 0 and len(seen) == 1


def test_on_installs_once():
    worker, _ = _worker()
    with envs.SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL.override(True):
        first = pcr.maybe_install(worker)
        wrapped = worker._forward_prefill
        pcr.install(worker, first)
    assert isinstance(first, pcr.LargerPrefillRelease)
    assert worker._forward_prefill is wrapped and wrapped.__larger_prefill_release__
