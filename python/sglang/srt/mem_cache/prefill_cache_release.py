"""Release the CUDA caching allocator's unused segments before a DSpark prefill larger than any before it
(SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL).

The startup warm-up prefills growing shapes (dsv41_prefill_shapes: 1 .. 2048, 2049, 4096 tokens). Each larger one
cannot reuse the cached blocks sized for the smaller ones, so it maps new segments, and the stranded ones stay
reserved: the -mergechunk arm's reserve peaked 1,879 MiB above its live allocated peak. ``torch.cuda.empty_cache``
before each new high-water prefill returns those blocks first. Once the largest chunk has run it never fires again.
It frees only blocks no tensor holds, so outputs do not change.

``maybe_install`` reads the flag once, at DSpark worker construction; off, nothing is wrapped.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

MIB = 1 << 20


class LargerPrefillRelease:
    """Wraps a prefill callable: empty_cache before a call whose token count exceeds every earlier one."""

    def __init__(self, cuda=torch.cuda) -> None:
        self.cuda = cuda
        self.largest = 0
        self.releases = 0

    def wrap(self, fn):
        @functools.wraps(fn)
        def release_then_call(batch, *args: Any, **kwargs: Any):
            tokens = getattr(batch, "extend_num_tokens", None)
            if tokens is not None and tokens > self.largest:
                before = self.cuda.memory_reserved()
                self.cuda.empty_cache()
                self.releases += 1
                logger.info(
                    "empty_cache before a %d-token prefill (largest so far %d): reserved %.0f -> %.0f MiB",
                    tokens,
                    self.largest,
                    before / MIB,
                    self.cuda.memory_reserved() / MIB,
                )
                self.largest = tokens
            return fn(batch, *args, **kwargs)

        release_then_call.__larger_prefill_release__ = True
        return release_then_call


def maybe_install(worker) -> Optional[LargerPrefillRelease]:
    from sglang.srt.environ import envs

    return install(worker) if envs.SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL.get() else None


def install(worker, release: Optional[LargerPrefillRelease] = None) -> LargerPrefillRelease:
    """Wrap a DSpark worker's ``_forward_prefill`` (an instance attribute, so once). The caller checks the flag."""
    release = release or LargerPrefillRelease()
    fn = worker._forward_prefill
    if not getattr(fn, "__larger_prefill_release__", False):
        worker._forward_prefill = release.wrap(fn)
    return release
